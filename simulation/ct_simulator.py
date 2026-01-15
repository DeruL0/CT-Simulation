"""
CT Simulator

Core CT simulation engine using Radon transform for forward projection
and filtered back projection for reconstruction.

Uses backend abstraction (CPU/GPU) for Radon/IRadon transforms.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Callable
import logging
import time
import numpy as np
from scipy import ndimage
import concurrent.futures
import threading

from .voxelizer import VoxelGrid
from .materials import MaterialType, MaterialDatabase
from .backends import get_backend, HAS_CUPY

try:
    import cupy as cp
except ImportError:
    cp = None


@dataclass
class CTVolume:
    """
    Reconstructed CT volume with metadata.
    
    Attributes:
        data: 3D numpy array of Hounsfield Units (Z, Y, X)
        voxel_size: Voxel edge length in mm
        origin: World coordinates of volume origin
    """
    data: np.ndarray  # Shape: (slices, height, width), dtype: float32/int16
    voxel_size: float  # mm per voxel
    origin: np.ndarray  # (3,) world position
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def num_slices(self) -> int:
        return self.data.shape[0]
    
    def get_slice(self, index: int, axis: int = 0) -> np.ndarray:
        """Get a 2D slice along specified axis (0=axial, 1=coronal, 2=sagittal)."""
        if axis == 0:
            return self.data[index, :, :]
        elif axis == 1:
            return self.data[:, index, :]
        else:
            return self.data[:, :, index]
    
    def apply_window(
        self, 
        window_center: float, 
        window_width: float
    ) -> np.ndarray:
        """
        Apply windowing to convert HU values to display range [0, 255].
        
        Args:
            window_center: Center of the window in HU
            window_width: Width of the window in HU
            
        Returns:
            uint8 array suitable for display
        """
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        
        windowed = np.clip(self.data, lower, upper)
        normalized = (windowed - lower) / (upper - lower)
        return (normalized * 255).astype(np.uint8)


class CTSimulator:
    """
    CT scanning simulator.
    
    Simulates the CT imaging process:
    1. Forward projection: Radon transform to generate sinograms
    2. Reconstruction: Filtered back projection to create CT images
    
    The simulation applies material-specific Hounsfield Unit values
    and can optionally add realistic noise.
    
    Uses backend abstraction for CPU/GPU dispatch.
    """
    
    def __init__(
        self,
        num_projections: int = 360,
        detector_pixels: Optional[int] = None,
        add_noise: bool = True,
        noise_level: float = 0.02,
        use_gpu: bool = False
    ):
        """
        Initialize CT simulator.
        
        Args:
            num_projections: Number of projection angles (default: 360)
            detector_pixels: Number of detector pixels (default: auto)
            add_noise: Whether to add Poisson-like noise
            noise_level: Noise standard deviation as fraction of signal
            use_gpu: Whether to use GPU acceleration (requires cupy)
        """
        self.num_projections = num_projections
        self.detector_pixels = detector_pixels
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.use_gpu = use_gpu and HAS_CUPY
        
        # Projection angles in degrees
        self.theta = np.linspace(0, 180, num_projections, endpoint=False)
        if self.use_gpu:
            self.theta_gpu = cp.asarray(self.theta)
        
        # Material database
        self.materials = MaterialDatabase()
        
        # Get appropriate backend
        self._backend = get_backend(use_gpu=self.use_gpu)
        logging.info(f"CTSimulator using backend: {self._backend.name}")
        
        # Timing info (populated after simulation)
        self._last_timing = None
    
    def simulate(
        self,
        voxel_grid: VoxelGrid,
        material: MaterialType = MaterialType.BONE_CORTICAL,
        background: MaterialType = MaterialType.AIR,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> CTVolume:
        """
        Simulate CT scanning of a voxel grid.
        
        Args:
            voxel_grid: Input VoxelGrid from voxelization
            material: Material type for the object
            background: Material type for empty space
            progress_callback: Optional callback(progress: 0.0-1.0)
            
        Returns:
            CTVolume containing reconstructed Hounsfield Unit values
        """
        # Get HU values for materials
        object_hu = self.materials.get_hu(material)
        background_hu = self.materials.get_hu(background)
        
        # Convert binary voxel grid to HU values
        hu_volume = np.where(
            voxel_grid.data > 0.5,
            object_hu,
            background_hu
        ).astype(np.float32)
        
        # Dispatch to appropriate implementation
        if self.use_gpu:
            try:
                return self._simulate_gpu(
                    voxel_grid, hu_volume, 
                    object_hu, background_hu, 
                    progress_callback
                )
            except Exception as e:
                logging.warning(f"GPU simulation failed: {e}. Falling back to CPU.")
                self._backend = get_backend(use_gpu=False)
        
        return self._simulate_cpu(hu_volume, background_hu, progress_callback, voxel_grid)
    
    def _simulate_cpu(
        self,
        hu_volume: np.ndarray,
        background_hu: float,
        progress_callback: Optional[Callable[[float], None]],
        voxel_grid: VoxelGrid
    ) -> CTVolume:
        """CPU simulation using ThreadPoolExecutor."""
        total_start = time.perf_counter()
        
        num_slices = hu_volume.shape[2]
        reconstructed = np.zeros_like(hu_volume)
        
        progress_lock = threading.Lock()
        completed_slices = 0
        backend = self._backend
        
        def process_slice(i):
            slice_2d = hu_volume[:, :, i]
            slice_shifted = slice_2d - background_hu
            
            # Use backend for processing
            reconstructed_slice = backend.process_slice(
                slice_shifted,
                self.theta,
                add_noise=self.add_noise,
                noise_level=self.noise_level
            )
            
            return i, reconstructed_slice + background_hu
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_slice = {
                executor.submit(process_slice, i): i 
                for i in range(num_slices)
            }
            
            for future in concurrent.futures.as_completed(future_to_slice):
                i, result_slice = future.result()
                reconstructed[:, :, i] = result_slice
                
                with progress_lock:
                    completed_slices += 1
                    if progress_callback is not None:
                        progress_callback(completed_slices / num_slices)
        
        ct_data = np.transpose(reconstructed, (2, 1, 0))
        
        self._last_timing = {
            'total': time.perf_counter() - total_start,
            'slices': num_slices,
            'backend': 'CPU'
        }
        
        return CTVolume(
            data=ct_data,
            voxel_size=voxel_grid.voxel_size,
            origin=voxel_grid.origin
        )
    
    def _simulate_gpu(
        self,
        voxel_grid: VoxelGrid,
        hu_volume: np.ndarray,
        object_hu: float,
        background_hu: float,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> CTVolume:
        """GPU simulation with optimized data transfers."""
        from .backends.gpu_backend import GPUBackend
        
        total_start = time.perf_counter()
        
        logging.info("=" * 60)
        logging.info("GPU SIMULATION STARTED")
        logging.info("=" * 60)
        
        cpu_data = voxel_grid.data
        slices = cpu_data.shape[2]
        h, w = cpu_data.shape[0], cpu_data.shape[1]
        
        logging.info(f"Volume shape: {cpu_data.shape} ({h}x{w}x{slices})")
        logging.info(f"Number of projections: {self.num_projections}")
        
        reconstructed_cpu = np.zeros(cpu_data.shape, dtype=np.float32)
        
        # Timing accumulators
        time_transfer_to_gpu = 0.0
        time_radon = 0.0
        time_iradon = 0.0
        time_transfer_to_cpu = 0.0
        
        theta_gpu = self.theta_gpu
        gpu_backend = self._backend
        
        for i in range(slices):
            t0 = time.perf_counter()
            
            # Move slice to GPU and convert to HU
            slice_cpu = cpu_data[:, :, i]
            slice_gpu = cp.asarray(slice_cpu, dtype=cp.float32)
            slice_gpu = cp.where(slice_gpu > 0.5, object_hu, background_hu)
            slice_gpu -= background_hu
            
            cp.cuda.Stream.null.synchronize()
            time_transfer_to_gpu += time.perf_counter() - t0
            
            # Process using GPU backend
            t1 = time.perf_counter()
            recon_gpu = gpu_backend.process_slice_gpu(
                slice_gpu, 
                theta_gpu,
                add_noise=self.add_noise,
                noise_level=self.noise_level
            )
            cp.cuda.Stream.null.synchronize()
            time_radon += time.perf_counter() - t1  # Combined radon+iradon
            
            # Shift back and transfer
            recon_gpu += background_hu
            
            t3 = time.perf_counter()
            reconstructed_cpu[:, :, i] = cp.asnumpy(recon_gpu)
            time_transfer_to_cpu += time.perf_counter() - t3
            
            if progress_callback:
                progress_callback((i + 1) / slices)
        
        ct_data = np.transpose(reconstructed_cpu, (2, 1, 0))
        total_time = time.perf_counter() - total_start
        
        # Log timing summary
        logging.info("-" * 60)
        logging.info("GPU SIMULATION TIMING SUMMARY")
        logging.info("-" * 60)
        logging.info(f"Total slices: {slices}")
        logging.info(f"Transfer to GPU: {time_transfer_to_gpu:.3f}s ({time_transfer_to_gpu/total_time*100:.1f}%)")
        logging.info(f"Radon+IRadon:    {time_radon:.3f}s ({time_radon/total_time*100:.1f}%)")
        logging.info(f"Transfer to CPU: {time_transfer_to_cpu:.3f}s ({time_transfer_to_cpu/total_time*100:.1f}%)")
        logging.info(f"TOTAL TIME:      {total_time:.3f}s")
        logging.info(f"Per-slice:       {total_time/slices*1000:.1f}ms")
        logging.info("=" * 60)
        
        self._last_timing = {
            'total': total_time,
            'transfer_to_gpu': time_transfer_to_gpu,
            'radon': time_radon,
            'iradon': 0,  # Combined in radon
            'transfer_to_cpu': time_transfer_to_cpu,
            'slices': slices,
            'backend': 'GPU'
        }
        # For compatibility with old API
        self._last_gpu_timing = self._last_timing
        
        return CTVolume(
            data=ct_data,
            voxel_size=voxel_grid.voxel_size,
            origin=voxel_grid.origin
        )
    
    def simulate_fast(
        self,
        voxel_grid: VoxelGrid,
        material: MaterialType = MaterialType.BONE_CORTICAL,
        background: MaterialType = MaterialType.AIR,
    ) -> CTVolume:
        """
        Fast CT simulation without full projection/reconstruction.
        
        Uses direct voxel-to-HU conversion with optional smoothing
        and noise addition. Suitable for quick preview.
        """
        object_hu = self.materials.get_hu(material)
        background_hu = self.materials.get_hu(background)
        
        hu_volume = np.where(
            voxel_grid.data > 0.5,
            object_hu,
            background_hu
        ).astype(np.float32)
        
        # Apply Gaussian smoothing
        sigma = 0.5
        hu_volume = ndimage.gaussian_filter(hu_volume, sigma=sigma)
        
        # Add noise
        if self.add_noise and self.noise_level > 0:
            noise = np.random.normal(0, self.noise_level * 1000, hu_volume.shape)
            hu_volume = hu_volume + noise
        
        ct_data = np.transpose(hu_volume, (2, 1, 0))
        
        return CTVolume(
            data=ct_data,
            voxel_size=voxel_grid.voxel_size,
            origin=voxel_grid.origin
        )
    
    @staticmethod
    def apply_beam_hardening(
        sinogram: np.ndarray,
        factor: float = 0.1
    ) -> np.ndarray:
        """
        Apply simple beam hardening artifact simulation.
        """
        normalized = sinogram / (sinogram.max() + 1e-10)
        correction = factor * (normalized ** 2)
        return sinogram * (1 - correction)
