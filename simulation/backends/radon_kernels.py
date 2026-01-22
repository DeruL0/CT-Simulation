"""
GPU Radon Transform Kernels

Extracted from physical_simulator.py for better modularity.
Provides GPU-accelerated forward (Radon) and inverse (iRadon/FBP) transforms.
"""

import numpy as np
import logging

# Optional CuPy import
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


class GPURadonTransform:
    """
    GPU-accelerated Radon and inverse Radon (FBP) transforms.
    
    Designed for CT simulation with configurable batch sizes
    based on available GPU memory.
    """
    
    def __init__(self, radon_batch: int = 20, iradon_batch: int = 60):
        """
        Initialize GPU Radon transform.
        
        Args:
            radon_batch: Batch size for forward projection angles
            iradon_batch: Batch size for backprojection angles
        """
        self.radon_batch = radon_batch
        self.iradon_batch = iradon_batch
    
    def radon(self, image_gpu: "cp.ndarray", theta_deg: "cp.ndarray") -> "cp.ndarray":
        """
        Vectorized GPU Radon transform with diagonal-based detector size.
        
        Args:
            image_gpu: 2D CuPy array (H, W)
            theta_deg: 1D CuPy array of projection angles in degrees
            
        Returns:
            Sinogram as CuPy array (n_det, n_angles)
        """
        H, W = image_gpu.shape
        
        # Compute detector count based on image diagonal
        n_det = int(cp.ceil(cp.sqrt(H**2 + W**2)))
        n_angles = len(theta_deg)
        
        theta_rad_all = theta_deg * (cp.pi / 180.0)
        
        # Image center
        img_center_x = (W - 1) / 2.0
        img_center_y = (H - 1) / 2.0
        
        # Detector coordinates centered at 0
        det_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        s_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        
        # Output sinogram
        sinogram = cp.zeros((n_det, n_angles), dtype=cp.float32)
        
        t = det_coords.reshape(1, -1, 1)
        s = s_coords.reshape(1, 1, -1)
        
        BATCH_SIZE = self.radon_batch
        
        for i in range(0, n_angles, BATCH_SIZE):
            theta_chunk = theta_rad_all[i:i+BATCH_SIZE]
            current_batch_size = len(theta_chunk)
            
            theta = theta_chunk.reshape(-1, 1, 1)
            
            cos_t = cp.cos(theta)
            sin_t = cp.sin(theta)
            
            # Compute sampling coordinates
            x_coords = t * cos_t - s * sin_t + img_center_x
            y_coords = t * sin_t + s * cos_t + img_center_y
            
            # Bilinear interpolation
            x0 = cp.floor(x_coords).astype(cp.int32)
            y0 = cp.floor(y_coords).astype(cp.int32)
            x1 = x0 + 1
            y1 = y0 + 1
            
            wx = x_coords - x0.astype(cp.float32)
            wy = y_coords - y0.astype(cp.float32)
            
            del x_coords, y_coords
            
            x0_clamped = cp.clip(x0, 0, W - 1)
            x1_clamped = cp.clip(x1, 0, W - 1)
            y0_clamped = cp.clip(y0, 0, H - 1)
            y1_clamped = cp.clip(y1, 0, H - 1)
            
            valid_mask = ((x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)).astype(cp.float32)
            
            del x0, y0, x1, y1
            
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
            
            sinogram[:, i:i+current_batch_size] = samples.sum(axis=2).T
            
            del samples, valid_mask, wx, wy, v00, v01, v10, v11
            del x0_clamped, x1_clamped, y0_clamped, y1_clamped
        
        # Clean up memory pool once after all batches (avoids sync overhead)
        cp.get_default_memory_pool().free_all_blocks()
        
        return sinogram
    
    def iradon(self, sinogram_gpu: "cp.ndarray", theta_deg: "cp.ndarray", output_size: int) -> "cp.ndarray":
        """
        Vectorized GPU inverse Radon (filtered back projection).
        
        Args:
            sinogram_gpu: 2D CuPy array (n_det, n_angles)
            theta_deg: 1D CuPy array of projection angles in degrees
            output_size: Size of output image (output_size x output_size)
            
        Returns:
            Reconstructed image as CuPy array (output_size, output_size)
        """
        n_angles = len(theta_deg)
        n_det = sinogram_gpu.shape[0]
        
        # Ramp filter
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
        pad_width = ((0, projection_size_padded - n_det), (0, 0))
        padded_sino = cp.pad(sinogram_gpu, pad_width, mode='constant')
        
        f = cp.fft.fftfreq(projection_size_padded).reshape(-1, 1)
        ramp = cp.abs(f)
        
        f_sino = cp.fft.fft(padded_sino, axis=0)
        filtered_sino = cp.fft.ifft(f_sino * ramp, axis=0).real
        filtered_sino = filtered_sino[:n_det, :].astype(cp.float32)
        
        del padded_sino, f_sino, f, ramp
        cp.get_default_memory_pool().free_all_blocks()
        
        # Backprojection
        center = (output_size - 1) / 2.0
        grid_1d = cp.arange(output_size, dtype=cp.float32) - center
        y_grid, x_grid = cp.meshgrid(grid_1d, grid_1d, indexing='ij')
        
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        
        reconstructed = cp.zeros((output_size * output_size), dtype=cp.float32)
        
        theta_rad_all = theta_deg * (cp.pi / 180.0)
        det_center = (n_det - 1) / 2.0
        
        BATCH_SIZE = self.iradon_batch
        
        for i in range(0, n_angles, BATCH_SIZE):
            theta_chunk = theta_rad_all[i:i+BATCH_SIZE]
            current_batch_size = len(theta_chunk)
            
            cos_t = cp.cos(theta_chunk).reshape(-1, 1)
            sin_t = cp.sin(theta_chunk).reshape(-1, 1)
            
            t_all = x_flat.reshape(1, -1) * cos_t + y_flat.reshape(1, -1) * sin_t
            t_idx = t_all + det_center
            
            t_idx_floor = cp.floor(t_idx).astype(cp.int32)
            t_idx_ceil = t_idx_floor + 1
            t_frac = t_idx - t_idx_floor.astype(cp.float32)
            
            t_idx_floor = cp.clip(t_idx_floor, 0, n_det - 1)
            t_idx_ceil = cp.clip(t_idx_ceil, 0, n_det - 1)
            
            angle_indices = cp.arange(i, i+current_batch_size, dtype=cp.int32).reshape(-1, 1)
            
            val_floor = filtered_sino[t_idx_floor, angle_indices]
            val_ceil = filtered_sino[t_idx_ceil, angle_indices]
            
            sampled = val_floor * (1 - t_frac) + val_ceil * t_frac
            
            reconstructed += sampled.sum(axis=0)
            
            del t_all, t_idx, t_idx_floor, t_idx_ceil, t_frac, val_floor, val_ceil, sampled
        
        # Clean up memory pool once after all batches (avoids sync overhead)
        cp.get_default_memory_pool().free_all_blocks()
        
        return reconstructed.reshape(output_size, output_size) * (cp.pi / (2 * n_angles))
