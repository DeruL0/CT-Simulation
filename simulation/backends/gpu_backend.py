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


class GPUBackend(SimulationBackend):
    """
    GPU-accelerated simulation using CuPy.
    
    Uses fully vectorized implementations of Radon/IRadon transforms
    with manual bilinear interpolation for maximum parallelism.
    """
    
    def __init__(self):
        if not HAS_CUPY:
            raise ImportError(
                "CuPy is required for GPU simulation. "
                "Install with: pip install cupy-cuda11x (or appropriate version)"
            )
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
        
        All angles are processed in a single GPU operation using
        manual bilinear interpolation (no Python loops).
        """
        # Move to GPU
        image_gpu = cp.asarray(image, dtype=cp.float32)
        theta_deg = cp.asarray(theta, dtype=cp.float32)
        
        H, W = image_gpu.shape
        
        # Compute detector count based on image diagonal (matching skimage behavior)
        n_det = int(np.ceil(np.sqrt(H**2 + W**2)))
        n_angles = len(theta_deg)
        
        # Convert angles to radians
        theta_rad = theta_deg * (cp.pi / 180.0)
        
        # Image center
        img_center_x = (W - 1) / 2.0
        img_center_y = (H - 1) / 2.0
        
        # Detector coordinates centered at 0
        det_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        
        # Ray path coordinates - use n_det for full diagonal coverage
        s_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        
        # Reshape for broadcasting
        theta = theta_rad.reshape(-1, 1, 1)
        t = det_coords.reshape(1, -1, 1)
        s = s_coords.reshape(1, 1, -1)
        
        # Compute sampling coordinates
        cos_t = cp.cos(theta)
        sin_t = cp.sin(theta)
        
        x_coords = t * cos_t - s * sin_t + img_center_x
        y_coords = t * sin_t + s * cos_t + img_center_y
        
        # Bilinear interpolation
        x0 = cp.floor(x_coords).astype(cp.int32)
        y0 = cp.floor(y_coords).astype(cp.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        wx = x_coords - x0.astype(cp.float32)
        wy = y_coords - y0.astype(cp.float32)
        
        x0_clamped = cp.clip(x0, 0, W - 1)
        x1_clamped = cp.clip(x1, 0, W - 1)
        y0_clamped = cp.clip(y0, 0, H - 1)
        y1_clamped = cp.clip(y1, 0, H - 1)
        
        valid_mask = ((x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)).astype(cp.float32)
        
        v00 = image_gpu[y0_clamped, x0_clamped]
        v01 = image_gpu[y0_clamped, x1_clamped]
        v10 = image_gpu[y1_clamped, x0_clamped]
        v11 = image_gpu[y1_clamped, x1_clamped]
        
        samples = (
            v00 * (1 - wx) * (1 - wy) +
            v01 * wx * (1 - wy) +
            v10 * (1 - wx) * wy +
            v11 * wx * wy
        ) * valid_mask
        
        sino = samples.sum(axis=2).T
        
        return cp.asnumpy(sino.astype(cp.float32))
    
    def iradon(
        self, 
        sinogram: np.ndarray, 
        theta: np.ndarray,
        output_size: int
    ) -> np.ndarray:
        """
        Vectorized GPU Inverse Radon (Filtered Back Projection).
        
        1. Apply ramp filter in frequency domain
        2. Back-project all angles simultaneously
        """
        # Move to GPU
        sinogram_gpu = cp.asarray(sinogram, dtype=cp.float32)
        theta_deg = cp.asarray(theta, dtype=cp.float32)
        
        n_angles = len(theta_deg)
        n_det = sinogram_gpu.shape[0]
        
        # Step 1: Ramp filtering
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
        pad_width = ((0, projection_size_padded - n_det), (0, 0))
        padded_sino = cp.pad(sinogram_gpu, pad_width, mode='constant')
        
        f = cp.fft.fftfreq(projection_size_padded).reshape(-1, 1)
        ramp = cp.abs(f)
        
        f_sino = cp.fft.fft(padded_sino, axis=0)
        filtered_sino = cp.fft.ifft(f_sino * ramp, axis=0).real
        filtered_sino = filtered_sino[:n_det, :].astype(cp.float32)
        
        # Step 2: Vectorized backprojection
        center = (output_size - 1) / 2.0
        grid_1d = cp.arange(output_size, dtype=cp.float32) - center
        y_grid, x_grid = cp.meshgrid(grid_1d, grid_1d, indexing='ij')
        
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        
        theta_rad = theta_deg * (cp.pi / 180.0)
        cos_t = cp.cos(theta_rad).reshape(-1, 1)
        sin_t = cp.sin(theta_rad).reshape(-1, 1)
        
        t_all = x_flat * cos_t + y_flat * sin_t
        det_center = (n_det - 1) / 2.0
        t_idx = t_all + det_center
        
        t_idx_floor = cp.floor(t_idx).astype(cp.int32)
        t_idx_ceil = t_idx_floor + 1
        t_frac = t_idx - t_idx_floor.astype(cp.float32)
        
        t_idx_floor = cp.clip(t_idx_floor, 0, n_det - 1)
        t_idx_ceil = cp.clip(t_idx_ceil, 0, n_det - 1)
        
        angle_idx = cp.arange(n_angles, dtype=cp.int32).reshape(-1, 1)
        angle_idx = cp.broadcast_to(angle_idx, t_idx.shape)
        
        val_floor = filtered_sino[t_idx_floor, angle_idx]
        val_ceil = filtered_sino[t_idx_ceil, angle_idx]
        
        sampled = val_floor * (1 - t_frac) + val_ceil * t_frac
        recon_flat = sampled.sum(axis=0)
        recon = recon_flat.reshape(output_size, output_size)
        recon = recon * (cp.pi / (2 * n_angles))
        
        return cp.asnumpy(recon.astype(cp.float32))
    
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
        sinogram = self._radon_gpu(slice_padded, theta_gpu)
        
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
        reconstructed = self._iradon_gpu(sinogram, theta_gpu, diag_len)
        
        # Crop back to output_size x output_size (square)
        if diag_pad > 0:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half + output_size,
                diag_pad_half:diag_pad_half + output_size
            ]
        
        return reconstructed
    
    def _radon_gpu(self, image_gpu, theta_deg):
        """Internal GPU Radon with diagonal-based detector size."""
        H, W = image_gpu.shape
        
        # Compute detector count based on image diagonal (matching skimage behavior)
        n_det = int(np.ceil(np.sqrt(H**2 + W**2)))
        n_angles = len(theta_deg)
        
        theta_rad = theta_deg * (cp.pi / 180.0)
        
        # Image center
        img_center_x = (W - 1) / 2.0
        img_center_y = (H - 1) / 2.0
        
        # Detector coordinates centered at 0
        det_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        
        # Ray path coordinates (s along the ray direction)
        # Must use n_det (diagonal length) to ensure rays fully traverse the image at all angles
        s_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        
        # Reshape for broadcasting: (n_angles, n_det, n_s)
        theta = theta_rad.reshape(-1, 1, 1)
        t = det_coords.reshape(1, -1, 1)
        s = s_coords.reshape(1, 1, -1)
        
        cos_t = cp.cos(theta)
        sin_t = cp.sin(theta)
        
        # Compute sampling coordinates in image space
        # Standard Radon transform coordinate mapping:
        # x = t * cos(theta) - s * sin(theta)
        # y = t * sin(theta) + s * cos(theta)
        x_coords = t * cos_t - s * sin_t + img_center_x
        y_coords = t * sin_t + s * cos_t + img_center_y
        
        x0 = cp.floor(x_coords).astype(cp.int32)
        y0 = cp.floor(y_coords).astype(cp.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        wx = x_coords - x0.astype(cp.float32)
        wy = y_coords - y0.astype(cp.float32)
        
        x0_clamped = cp.clip(x0, 0, W - 1)
        x1_clamped = cp.clip(x1, 0, W - 1)
        y0_clamped = cp.clip(y0, 0, H - 1)
        y1_clamped = cp.clip(y1, 0, H - 1)
        
        valid_mask = ((x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)).astype(cp.float32)
        
        v00 = image_gpu[y0_clamped, x0_clamped]
        v01 = image_gpu[y0_clamped, x1_clamped]
        v10 = image_gpu[y1_clamped, x0_clamped]
        v11 = image_gpu[y1_clamped, x1_clamped]
        
        samples = (
            v00 * (1 - wx) * (1 - wy) +
            v01 * wx * (1 - wy) +
            v10 * (1 - wx) * wy +
            v11 * wx * wy
        ) * valid_mask
        
        # Sum along ray path and transpose to (n_det, n_angles)
        return samples.sum(axis=2).T.astype(cp.float32)
    
    def _iradon_gpu(self, sinogram_gpu, theta_deg, output_size):
        """Internal GPU IRadon without CPU transfers."""
        n_angles = len(theta_deg)
        n_det = sinogram_gpu.shape[0]
        
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
        pad_width = ((0, projection_size_padded - n_det), (0, 0))
        padded_sino = cp.pad(sinogram_gpu, pad_width, mode='constant')
        
        f = cp.fft.fftfreq(projection_size_padded).reshape(-1, 1)
        ramp = cp.abs(f)
        
        f_sino = cp.fft.fft(padded_sino, axis=0)
        filtered_sino = cp.fft.ifft(f_sino * ramp, axis=0).real
        filtered_sino = filtered_sino[:n_det, :].astype(cp.float32)
        
        center = (output_size - 1) / 2.0
        grid_1d = cp.arange(output_size, dtype=cp.float32) - center
        y_grid, x_grid = cp.meshgrid(grid_1d, grid_1d, indexing='ij')
        
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        
        theta_rad = theta_deg * (cp.pi / 180.0)
        cos_t = cp.cos(theta_rad).reshape(-1, 1)
        sin_t = cp.sin(theta_rad).reshape(-1, 1)
        
        t_all = x_flat * cos_t + y_flat * sin_t
        det_center = (n_det - 1) / 2.0
        t_idx = t_all + det_center
        
        t_idx_floor = cp.floor(t_idx).astype(cp.int32)
        t_idx_ceil = t_idx_floor + 1
        t_frac = t_idx - t_idx_floor.astype(cp.float32)
        
        t_idx_floor = cp.clip(t_idx_floor, 0, n_det - 1)
        t_idx_ceil = cp.clip(t_idx_ceil, 0, n_det - 1)
        
        angle_idx = cp.arange(n_angles, dtype=cp.int32).reshape(-1, 1)
        angle_idx = cp.broadcast_to(angle_idx, t_idx.shape)
        
        val_floor = filtered_sino[t_idx_floor, angle_idx]
        val_ceil = filtered_sino[t_idx_ceil, angle_idx]
        
        sampled = val_floor * (1 - t_frac) + val_ceil * t_frac
        recon_flat = sampled.sum(axis=0)
        recon = recon_flat.reshape(output_size, output_size)
        
        return recon * (cp.pi / (2 * n_angles))
