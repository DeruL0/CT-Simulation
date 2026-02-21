"""
GPU Backend for CT Simulation

Fully vectorized CuPy-based Radon/IRadon transforms.
"""

import logging
import numpy as np

try:
    import cupy as cp
    import cupyx.scipy.ndimage
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from .base import SimulationBackend
from .radon_kernels import GPURadonTransform


class GPUBackend(SimulationBackend):
    """
    GPU-accelerated simulation using CuPy.
    
    Uses fully vectorized implementations of Radon/IRadon transforms
    with manual bilinear interpolation for maximum parallelism.
    """
    
    def __init__(self, radon_batch: int = 20, iradon_batch: int = 60):
        if not HAS_CUPY:
            raise ImportError(
                "CuPy is required for GPU simulation. "
                "Install with: pip install cupy-cuda11x (or appropriate version)"
            )
        self._radon_transform = GPURadonTransform(radon_batch=radon_batch, iradon_batch=iradon_batch)
        logging.info("GPU Backend initialized (CuPy)")
    
    @property
    def name(self) -> str:
        return "GPU (CuPy)"
    
    def radon(
        self, 
        image: np.ndarray, 
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Fully vectorized GPU Radon transform with diagonal-based detector size.
        
        All angles are processed using batch processing to manage GPU memory.
        """
        try:
            # Move to GPU
            image_gpu = cp.asarray(image, dtype=cp.float32)
            theta_gpu = cp.asarray(theta, dtype=cp.float32)
            
            # Use GPURadonTransform for actual computation
            sinogram_gpu = self._radon_transform.radon(image_gpu, theta_gpu)
            
            return cp.asnumpy(sinogram_gpu)
        except cp.cuda.memory.OutOfMemoryError:
            logging.error("GPU out of memory in radon transform")
            raise
    
    def iradon(
        self, 
        sinogram: np.ndarray, 
        theta: np.ndarray,
        output_size: int
    ) -> np.ndarray:
        """
        Vectorized GPU Inverse Radon (Filtered Back Projection).
        
        Uses batch processing to manage GPU memory efficiently.
        """
        try:
            # Move to GPU
            sinogram_gpu = cp.asarray(sinogram, dtype=cp.float32)
            theta_gpu = cp.asarray(theta, dtype=cp.float32)
            
            # Use GPURadonTransform for actual computation
            recon_gpu = self._radon_transform.iradon(sinogram_gpu, theta_gpu, output_size)
            
            return cp.asnumpy(recon_gpu)
        except cp.cuda.memory.OutOfMemoryError:
            logging.error("GPU out of memory in iradon transform")
            raise
    
    def process_slice_gpu(
        self,
        slice_gpu,
        theta_gpu,
        add_noise: bool = False,
        noise_level: float = 0.02
    ):
        """
        Process slice entirely on GPU without CPU transfers.
        
        This is an optimized version that keeps data on GPU.
        Output is always a square based on the longer edge.
        Returns a CuPy array.
        """
        original_shape = slice_gpu.shape
        
        # Output will be a square based on the longer edge
        output_size = max(original_shape[0], original_shape[1])
        
        # Pad to diagonal size for proper radon/iradon transform
        diag_len = int(np.ceil(np.sqrt(output_size**2 + output_size**2))) + 32
        
        # First pad to square (output_size x output_size)
        pad_to_square_h = output_size - original_shape[0]
        pad_to_square_w = output_size - original_shape[1]
        sq_pad_top = pad_to_square_h // 2
        sq_pad_left = pad_to_square_w // 2
        
        if pad_to_square_h > 0 or pad_to_square_w > 0:
            square_slice = cp.pad(
                slice_gpu,
                ((sq_pad_top, pad_to_square_h - sq_pad_top), 
                 (sq_pad_left, pad_to_square_w - sq_pad_left)),
                mode='constant',
                constant_values=0.0
            )
        else:
            square_slice = slice_gpu
        
        # Then pad to diagonal for radon transform
        diag_pad = diag_len - output_size
        diag_pad_half = diag_pad // 2
        
        if diag_pad > 0:
            slice_padded = cp.pad(
                square_slice,
                ((diag_pad_half, diag_pad - diag_pad_half), 
                 (diag_pad_half, diag_pad - diag_pad_half)),
                mode='constant',
                constant_values=0.0
            )
        else:
            slice_padded = square_slice
            diag_pad_half = 0
        
        # Forward projection (on GPU)
        sinogram = self._radon_transform.radon(slice_padded, theta_gpu)
        
        # Add noise
        if add_noise and noise_level > 0:
            noise = cp.random.normal(
                0,
                noise_level * cp.abs(sinogram).mean(),
                sinogram.shape,
                dtype=cp.float32
            )
            sinogram += noise
        
        # Backward projection (on GPU)
        reconstructed = self._radon_transform.iradon(sinogram, theta_gpu, diag_len)
        
        # Crop back to output_size x output_size (square)
        if diag_pad > 0:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half + output_size,
                diag_pad_half:diag_pad_half + output_size
            ]
        
        return reconstructed
