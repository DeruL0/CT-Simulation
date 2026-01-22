"""
Elasticity Solver

GPU-accelerated Finite Difference Method solver for linear elasticity.
Simulates physical compression with material-dependent deformation.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import numpy as np

# GPU support detection
try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None
    cp_ndimage = None

# CPU fallback
from scipy import ndimage as sp_ndimage


@dataclass
class ElasticityConfig:
    """Configuration for elasticity simulation."""
    compression_ratio: float = 0.1  # 0.0 - 0.5 (10% - 50%)
    poisson_ratio: float = 0.3  # 0.0 - 0.5
    solid_stiffness: float = 1.0
    void_stiffness: float = 1e-6
    solver_iterations: int = 500
    solver_tolerance: float = 1e-4
    downsample_factor: int = 4  # Physics grid resolution reduction


class ElasticitySolver:
    """
    GPU-accelerated linear elasticity solver using Finite Differences.
    
    Solves for displacement field under uniaxial compression boundary conditions.
    Uses Jacobi iteration for computational efficiency on GPU.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize elasticity solver.
        
        Args:
            use_gpu: Whether to use GPU acceleration (requires CuPy)
        """
        self.use_gpu = use_gpu and HAS_GPU
        self.xp = cp if self.use_gpu else np
        self.ndimage = cp_ndimage if self.use_gpu else sp_ndimage
        
        if self.use_gpu:
            logging.info("ElasticitySolver: GPU mode (CuPy)")
        else:
            logging.info("ElasticitySolver: CPU mode (NumPy/SciPy)")
    
    def solve(
        self,
        volume: np.ndarray,
        config: ElasticityConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Solve for displacement field under compression.
        
        Args:
            volume: 3D density volume (Z, Y, X)
            config: Elasticity configuration
            progress_callback: Optional callback(progress: 0-1, status: str)
            
        Returns:
            Tuple of (u_z, u_y, u_x) displacement fields at original resolution
        """
        xp = self.xp
        original_shape = volume.shape
        
        # Step 1: Downsample for physics
        if progress_callback:
            progress_callback(0.05, "Downsampling volume for physics...")
        
        physics_shape = tuple(max(1, s // config.downsample_factor) for s in original_shape)
        zoom_factors = tuple(ps / os for ps, os in zip(physics_shape, original_shape))
        
        if self.use_gpu:
            volume_gpu = xp.asarray(volume, dtype=xp.float32)
            physics_volume = cp_ndimage.zoom(volume_gpu, zoom_factors, order=1)
            del volume_gpu
        else:
            physics_volume = sp_ndimage.zoom(volume.astype(np.float32), zoom_factors, order=1)
        
        logging.info(f"Physics grid: {physics_shape} (downsampled from {original_shape})")
        
        # Step 2: Create stiffness map
        if progress_callback:
            progress_callback(0.10, "Creating stiffness map...")
        
        # Solid = high stiffness, Void = near-zero stiffness
        threshold = 0.5
        stiffness = xp.where(
            physics_volume > threshold,
            config.solid_stiffness,
            config.void_stiffness
        ).astype(xp.float32)
        
        # Step 3: Initialize displacement field
        nz, ny, nx = physics_shape
        u_z = xp.zeros(physics_shape, dtype=xp.float32)
        u_y = xp.zeros(physics_shape, dtype=xp.float32)
        u_x = xp.zeros(physics_shape, dtype=xp.float32)
        
        # Boundary conditions: Linear compression gradient as initial guess
        # Top (z=nz-1) compressed, Bottom (z=0) fixed
        total_displacement = config.compression_ratio * nz
        z_coords = xp.arange(nz, dtype=xp.float32).reshape(-1, 1, 1)
        u_z = -total_displacement * (z_coords / (nz - 1))
        u_z = xp.broadcast_to(u_z, physics_shape).copy()
        
        # Poisson effect: Lateral expansion proportional to compression
        # u_x = nu * epsilon * x, u_y = nu * epsilon * y (from center)
        y_coords = xp.arange(ny, dtype=xp.float32) - (ny - 1) / 2.0
        x_coords = xp.arange(nx, dtype=xp.float32) - (nx - 1) / 2.0
        
        _, Y, X = xp.meshgrid(
            xp.arange(nz, dtype=xp.float32), 
            y_coords, 
            x_coords, 
            indexing='ij'
        )
        
        # Average strain
        epsilon_z = config.compression_ratio
        u_x = config.poisson_ratio * epsilon_z * X
        u_y = config.poisson_ratio * epsilon_z * Y
        
        # Step 4: Iterative solver (Jacobi relaxation)
        if progress_callback:
            progress_callback(0.15, "Solving elasticity equations...")
        
        u_z, u_y, u_x = self._jacobi_solve(
            u_z, u_y, u_x, stiffness, config, progress_callback
        )
        
        # Step 5: Upsample displacement fields to original resolution
        if progress_callback:
            progress_callback(0.90, "Upsampling displacement field...")
        
        upsample_factors = tuple(os / ps for os, ps in zip(original_shape, physics_shape))
        
        if self.use_gpu:
            u_z_full = cp_ndimage.zoom(u_z, upsample_factors, order=1)
            u_y_full = cp_ndimage.zoom(u_y, upsample_factors, order=1)
            u_x_full = cp_ndimage.zoom(u_x, upsample_factors, order=1)
            
            # Scale displacements to match high-res grid
            u_z_full *= config.downsample_factor
            u_y_full *= config.downsample_factor
            u_x_full *= config.downsample_factor
            
            # Transfer to CPU
            u_z_np = cp.asnumpy(u_z_full)
            u_y_np = cp.asnumpy(u_y_full)
            u_x_np = cp.asnumpy(u_x_full)
            
            # Free GPU memory
            del u_z, u_y, u_x, u_z_full, u_y_full, u_x_full, stiffness
            cp.get_default_memory_pool().free_all_blocks()
        else:
            u_z_full = sp_ndimage.zoom(u_z, upsample_factors, order=1)
            u_y_full = sp_ndimage.zoom(u_y, upsample_factors, order=1)
            u_x_full = sp_ndimage.zoom(u_x, upsample_factors, order=1)
            
            u_z_full *= config.downsample_factor
            u_y_full *= config.downsample_factor
            u_x_full *= config.downsample_factor
            
            u_z_np, u_y_np, u_x_np = u_z_full, u_y_full, u_x_full
        
        if progress_callback:
            progress_callback(1.0, "Displacement field computed")
        
        return u_z_np, u_y_np, u_x_np
    
    def _jacobi_solve(
        self,
        u_z, u_y, u_x,
        stiffness,
        config: ElasticityConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """
        Jacobi iterative solver for equilibrium.
        
        Simplified version using Laplacian smoothing weighted by stiffness.
        This is an approximation suitable for visualization purposes.
        """
        xp = self.xp
        nz, ny, nx = u_z.shape
        
        # 3D Laplacian kernel (6-point stencil weights)
        kernel = xp.array([
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
        ], dtype=xp.float32) / 6.0
        
        # Relaxation factor
        omega = 0.6
        
        for iteration in range(config.solver_iterations):
            # Store old values for convergence check
            if iteration % 50 == 0:
                u_z_old = u_z.copy()
            
            # Smooth displacement with stiffness weighting
            u_z_smooth = self.ndimage.convolve(u_z * stiffness, kernel, mode='nearest')
            u_y_smooth = self.ndimage.convolve(u_y * stiffness, kernel, mode='nearest')
            u_x_smooth = self.ndimage.convolve(u_x * stiffness, kernel, mode='nearest')
            
            stiffness_smooth = self.ndimage.convolve(stiffness, kernel, mode='nearest')
            stiffness_smooth = xp.maximum(stiffness_smooth, 1e-10)
            
            u_z_new = u_z_smooth / stiffness_smooth
            u_y_new = u_y_smooth / stiffness_smooth
            u_x_new = u_x_smooth / stiffness_smooth
            
            # Apply boundary conditions
            # Bottom fixed
            u_z_new[0, :, :] = 0
            u_y_new[0, :, :] = 0
            u_x_new[0, :, :] = 0
            
            # Top: prescribed displacement
            total_disp = config.compression_ratio * nz
            u_z_new[-1, :, :] = -total_disp
            
            # Relaxation update
            u_z = (1 - omega) * u_z + omega * u_z_new
            u_y = (1 - omega) * u_y + omega * u_y_new
            u_x = (1 - omega) * u_x + omega * u_x_new
            
            # Progress and convergence check
            if iteration % 50 == 0:
                diff = float(xp.abs(u_z - u_z_old).max())
                if progress_callback:
                    progress = 0.15 + 0.75 * (iteration / config.solver_iterations)
                    progress_callback(progress, f"Solving: iter {iteration}, diff={diff:.2e}")
                
                if diff < config.solver_tolerance:
                    logging.info(f"Converged at iteration {iteration}")
                    break
        
        return u_z, u_y, u_x


def apply_displacement_field(
    volume: np.ndarray,
    u_z: np.ndarray,
    u_y: np.ndarray,
    u_x: np.ndarray,
    use_gpu: bool = True,
    chunk_size: int = 64,
    progress_callback: Optional[Callable[[float, str], None]] = None
) -> np.ndarray:
    """
    Apply displacement field to warp a volume using tiled processing.
    
    Uses chunking to avoid GPU OOM on large volumes.
    
    Args:
        volume: Original 3D volume
        u_z, u_y, u_x: Displacement fields
        use_gpu: Use GPU for warping
        chunk_size: Size of chunks for tiled processing
        progress_callback: Optional progress callback
        
    Returns:
        Warped volume
    """
    use_gpu = use_gpu and HAS_GPU
    xp = cp if use_gpu else np
    ndimage = cp_ndimage if use_gpu else sp_ndimage
    
    nz, ny, nx = volume.shape
    result = np.zeros_like(volume)
    
    # Create coordinate grids
    z_base = np.arange(nz, dtype=np.float32).reshape(-1, 1, 1)
    y_base = np.arange(ny, dtype=np.float32).reshape(1, -1, 1)
    x_base = np.arange(nx, dtype=np.float32).reshape(1, 1, -1)
    
    # Deformed coordinates = original coordinates - displacement
    # (We sample the original volume at the "source" position)
    z_coords = np.broadcast_to(z_base, volume.shape) - u_z
    y_coords = np.broadcast_to(y_base, volume.shape) - u_y
    x_coords = np.broadcast_to(x_base, volume.shape) - u_x
    
    if use_gpu:
        # Tiled GPU processing
        num_chunks = (nz + chunk_size - 1) // chunk_size
        
        for i, z_start in enumerate(range(0, nz, chunk_size)):
            z_end = min(z_start + chunk_size, nz)
            
            if progress_callback:
                progress_callback((i + 1) / num_chunks, f"Warping chunk {i+1}/{num_chunks}")
            
            # Upload chunk data
            vol_chunk = cp.asarray(volume, dtype=cp.float32)  # Need full volume for sampling
            coords_chunk = cp.array([
                z_coords[z_start:z_end],
                y_coords[z_start:z_end],
                x_coords[z_start:z_end]
            ], dtype=cp.float32)
            
            # Warp using map_coordinates
            warped_chunk = cp_ndimage.map_coordinates(
                vol_chunk, coords_chunk, order=1, mode='constant', cval=0.0
            )
            
            # Download result
            result[z_start:z_end] = cp.asnumpy(warped_chunk)
            
            # Free memory
            del vol_chunk, coords_chunk, warped_chunk
            cp.get_default_memory_pool().free_all_blocks()
    else:
        # CPU processing
        coords = np.array([z_coords, y_coords, x_coords])
        result = sp_ndimage.map_coordinates(
            volume.astype(np.float32), coords, order=1, mode='constant', cval=0.0
        )
    
    return result.astype(volume.dtype)
