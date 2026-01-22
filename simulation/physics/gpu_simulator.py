"""
GPU CT Simulation Strategy

Handles GPU-accelerated physical CT simulation logic:
- GPU memory detection and management
- Dynamic batch sizing
- Slice-by-slice simulation kernel
"""

import logging
from typing import Optional, Callable, Tuple
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from ..backends.radon_kernels import GPURadonTransform

class GPUSimulator:
    """
    GPU implementation of Physical CT Simulation.
    
    Handles memory management and batch processing for GPU acceleration.
    """
    
    def __init__(self, theta: np.ndarray):
        """
        Initialize GPU simulator.
        
        Args:
            theta: Projection angles in degrees
        """
        self.theta = theta
        self._gpu_total_mem = 0
        self._gpu_free_mem = 0
        self._radon_transform = None
        
        if HAS_CUPY:
            self._detect_memory()
            
    def _detect_memory(self) -> None:
        """Detect GPU memory and calculate available VRAM."""
        try:
            mempool = cp.get_default_memory_pool()
            device = cp.cuda.Device()
            self._gpu_total_mem = device.mem_info[1]
            self._gpu_free_mem = device.mem_info[0]
            
            total_gb = self._gpu_total_mem / (1024**3)
            free_gb = self._gpu_free_mem / (1024**3)
            logging.info(f"  GPU Simulator: {free_gb:.1f}/{total_gb:.1f} GB VRAM available")
        except Exception as e:
            logging.warning(f"Could not detect GPU memory: {e}")
            self._gpu_total_mem = 8 * (1024**3)
            self._gpu_free_mem = 6 * (1024**3)

    def _calculate_batch_sizes(self, image_size: int) -> Tuple[int, int, int]:
        """
        Calculate optimal batch sizes based on available GPU memory.
        
        Returns:
            (slice_batch, radon_batch, iradon_batch)
        """
        n_det = int(np.ceil(np.sqrt(2) * image_size)) + 32
        
        # Use 60% of free memory as safety margin
        available = self._gpu_free_mem * 0.6 if self._gpu_free_mem > 0 else 4 * (1024**3)
        
        # Memory per operation (empirical estimates)
        mem_per_radon_angle = n_det * n_det * 6 * 4
        radon_batch = max(5, min(60, int(available / (mem_per_radon_angle * 4))))
        
        mem_per_iradon_angle = image_size * image_size * 4 * 4
        iradon_batch = max(10, min(120, int(available / (mem_per_iradon_angle * 2))))
        
        mem_per_slice = image_size * image_size * 4 * 10
        slice_batch = max(1, min(16, int(available / (mem_per_slice * 8))))
        
        logging.info(f"  Auto batch sizes: slices={slice_batch}, radon={radon_batch}, iradon={iradon_batch}")
        
        return slice_batch, radon_batch, iradon_batch

    def simulate_volume(
        self,
        volume_data: np.ndarray,
        mu_object: np.ndarray,
        mu_background: np.ndarray,
        bin_weights: np.ndarray,
        bin_centers: np.ndarray,
        photon_count: float,
        mu_water_effective: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> np.ndarray:
        """
        Run simulation for entire volume using GPU batch processing.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not available for GPU simulation")

        num_slices = volume_data.shape[2]
        h, w = volume_data.shape[:2]
        output_size = max(h, w)
        
        # setup batch sizes
        slice_batch, radon_batch, iradon_batch = self._calculate_batch_sizes(output_size)
        
        # Initialize kernels
        self._radon_transform = GPURadonTransform(radon_batch=radon_batch, iradon_batch=iradon_batch)
        
        # Pre-compute physics arrays on GPU
        mu_obj_gpu = cp.asarray(mu_object, dtype=cp.float32)
        mu_bg_gpu = cp.asarray(mu_background, dtype=cp.float32)
        weights_gpu = cp.asarray(bin_weights, dtype=cp.float32)
        theta_gpu = cp.asarray(self.theta, dtype=cp.float32)
        
        # Initialize output
        reconstructed = np.zeros((output_size, output_size, num_slices), dtype=np.float32)
        
        BATCH_SIZE = slice_batch
        logging.info(f"  GPU batch processing {num_slices} slices (batch_size={BATCH_SIZE})")
        
        # Streams
        compute_stream = cp.cuda.Stream(non_blocking=True)
        transfer_stream = cp.cuda.Stream(non_blocking=True)
        
        completed = 0
        prev_batch_stack = None
        prev_batch_slices = 0
        prev_batch_start = 0
        
        for batch_start in range(0, num_slices, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_slices)
            current_batch_size = batch_end - batch_start
            
            # Transfer
            with transfer_stream:
                batch_data = cp.asarray(volume_data[:, :, batch_start:batch_end], dtype=cp.float32)
            
            # Sync
            compute_stream.wait_event(transfer_stream.record())
            
            # Compute
            with compute_stream:
                batch_results = []
                for i in range(current_batch_size):
                    slice_2d = batch_data[:, :, i]
                    recon_gpu = self._simulate_slice_kernel(
                        slice_2d, mu_obj_gpu, mu_bg_gpu, weights_gpu, 
                        theta_gpu, photon_count, output_size, mu_water_effective
                    )
                    batch_results.append(recon_gpu)
                
                batch_stack = cp.stack(batch_results, axis=2)
            
            # Handle previous results
            if prev_batch_stack is not None:
                with transfer_stream:
                    transfer_stream.wait_event(compute_stream.record())
                    # Prev batch end index
                    p_end = prev_batch_start + prev_batch_slices
                    reconstructed[:, :, prev_batch_start:p_end] = cp.asnumpy(prev_batch_stack)
                    del prev_batch_stack
            
            prev_batch_stack = batch_stack
            prev_batch_slices = current_batch_size
            prev_batch_start = batch_start
            
            del batch_data, batch_results
            
            completed += current_batch_size
            if progress_callback:
                progress_callback(completed / num_slices)
                
        # Final batch
        if prev_batch_stack is not None:
            compute_stream.synchronize()
            p_end = prev_batch_start + prev_batch_slices
            reconstructed[:, :, prev_batch_start:p_end] = cp.asnumpy(prev_batch_stack)
            del prev_batch_stack
            
        # Cleanup
        cp.get_default_memory_pool().free_all_blocks()
        
        return reconstructed

    def _simulate_slice_kernel(
        self,
        slice_gpu: "cp.ndarray",
        mu_obj_gpu: "cp.ndarray",
        mu_bg_gpu: "cp.ndarray",
        weights_gpu: "cp.ndarray",
        theta_gpu: "cp.ndarray",
        photon_count: float,
        output_size: int,
        mu_water_effective: float
    ) -> "cp.ndarray":
        """
        Core simulation kernel for a single slice on GPU.
        """
        original_shape = slice_gpu.shape
        
        # Pad to diagonal for Radon
        diag_len = int(cp.ceil(cp.sqrt(output_size**2 + output_size**2))) + 32
        
        # Pad to square
        pad_h = output_size - original_shape[0]
        pad_w = output_size - original_shape[1]
        
        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            left = pad_w // 2
            square_slice = cp.pad(
                slice_gpu,
                ((top, pad_h - top), (left, pad_w - left)),
                'constant'
            )
        else:
            square_slice = slice_gpu
            
        # Pad to diagonal
        diag_pad = diag_len - output_size
        diag_pad_half = diag_pad // 2
        
        if diag_pad > 0:
            padded_slice = cp.pad(
                square_slice,
                ((diag_pad_half, diag_pad - diag_pad_half),
                 (diag_pad_half, diag_pad - diag_pad_half)),
                'constant'
            )
        else:
            padded_slice = square_slice
            diag_pad_half = 0
            
        # Radon
        path_lengths = self._radon_transform.radon(padded_slice, theta_gpu)
        max_path = float(path_lengths.max()) * 1.1 if path_lengths.max() > 0 else 100.0
        
        # Polychromatic Attenuation
        I_transmitted = cp.zeros_like(path_lengths)
        n_bins = len(weights_gpu)
        
        for i in range(n_bins):
            w = weights_gpu[i]
            if w <= 0: continue
            
            L_obj = path_lengths
            L_bg = cp.maximum(max_path - L_obj, 0)
            
            attenuation = cp.exp(-mu_obj_gpu[i] * L_obj - mu_bg_gpu[i] * L_bg)
            I_transmitted += w * attenuation
            
        # Noise
        I_counts = cp.maximum(I_transmitted * photon_count, 1.0)
        I_noisy = cp.random.poisson(I_counts.astype(cp.int64)).astype(cp.float32)
        I_noisy = cp.maximum(I_noisy, 1.0)
        
        # Log transform
        sinogram = -cp.log(I_noisy / photon_count)
        
        # Inverse Radon
        reconstructed = self._radon_transform.iradon(sinogram, theta_gpu, diag_len)
        
        # Crop
        if diag_pad > 0:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half+output_size,
                diag_pad_half:diag_pad_half+output_size
            ]
            
        # Convert to HU
        hu = 1000 * (reconstructed - mu_water_effective) / mu_water_effective
        return hu.astype(cp.float32)
