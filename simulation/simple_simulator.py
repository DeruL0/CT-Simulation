"""
Simple CT Simulator

Fast, non-physical CT simulator using Radon/IRadon transforms directly.
Designed for quick previews without complex physics calculations.
"""

from typing import Optional, Callable
import numpy as np

from .voxelizer import VoxelGrid
from .volume import CTVolume
from .backends import get_backend
from .materials import MaterialType, material_type_to_physical


class SimpleCTSimulator:
    """
    Fast, non-physical CT simulator using Radon/IRadon transforms directly.
    Bypasses complex physics for speed (Fast Mode).
    
    Use this for quick previews. For accurate simulation, use PhysicalCTSimulator.
    """
    
    def __init__(self, num_projections: int = 360, use_gpu: bool = False):
        """
        Initialize simple CT simulator.
        
        Args:
            num_projections: Number of projection angles (default 360)
            use_gpu: Whether to use GPU acceleration
        """
        self.num_projections = num_projections
        self.use_gpu = use_gpu
        self.backend = get_backend(use_gpu)
        
    def simulate(
        self, 
        voxel_grid: VoxelGrid, 
        material=None, 
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> CTVolume:
        """
        Run fast CT simulation.
        
        Args:
            voxel_grid: Binary voxel grid from voxelization
            material: Material type for absolute attenuation scaling (optional)
            progress_callback: Progress callback (0.0 to 1.0)
            
        Returns:
            CTVolume with reconstructed absolute linear attenuation values
        """
        current_volume = voxel_grid.data.astype(np.float32)
        
        # Projection angles
        theta = np.linspace(0., 180., self.num_projections, endpoint=False)
        
        # Process slice by slice
        num_slices = current_volume.shape[0]
        reconstructed = np.zeros_like(current_volume)
        
        for i in range(num_slices):
            slice_img = current_volume[i]
            recon_slice = self._process_slice(slice_img, theta)
            reconstructed[i] = recon_slice
            
            if progress_callback:
                progress_callback((i + 1) / num_slices)
        
        # Convert reconstructed values to absolute linear attenuation scale (cm^-1)
        mu_volume = self._scale_to_absolute_mu(reconstructed, material)

        return CTVolume(mu_volume, voxel_grid.voxel_size, voxel_grid.origin)
    
    def _process_slice(self, slice_img: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Process a single 2D slice through Radon/iRadon."""
        orig_shape = slice_img.shape
        h, w = orig_shape
        max_dim = max(h, w)
        
        # Handle non-square slices: pad to square
        if h != w:
            padded = np.zeros((max_dim, max_dim), dtype=np.float32)
            pad_h = (max_dim - h) // 2
            pad_w = (max_dim - w) // 2
            padded[pad_h:pad_h+h, pad_w:pad_w+w] = slice_img
            slice_to_process = padded
        else:
            slice_to_process = slice_img
        
        # Project (Radon)
        sinogram = self.backend.radon(slice_to_process, theta)
        
        # Reconstruct (FBP)
        recon_slice = self.backend.iradon(
            sinogram, 
            theta, 
            output_size=max_dim
        )
        
        # Crop back to original size if needed
        if h != w:
            pad_h = (max_dim - h) // 2
            pad_w = (max_dim - w) // 2
            recon_slice = recon_slice[pad_h:pad_h+h, pad_w:pad_w+w]
        
        return recon_slice
    
    def _resolve_reference_mu(self, material) -> float:
        """Resolve material reference linear attenuation (cm^-1)."""
        default_mu = 0.171  # Approximate water attenuation near 100 keV.

        if material is None:
            return default_mu

        if hasattr(material, "linear_attenuation"):
            try:
                return float(material.linear_attenuation)
            except Exception:
                return default_mu

        if isinstance(material, MaterialType):
            phys = material_type_to_physical(material.value)
            if phys is not None:
                return float(phys.linear_attenuation)

        if hasattr(material, "value"):
            phys = material_type_to_physical(str(material.value))
            if phys is not None:
                return float(phys.linear_attenuation)

        if hasattr(material, "name"):
            mat_name = str(material.name).lower()
            phys = material_type_to_physical(mat_name)
            if phys is not None:
                return float(phys.linear_attenuation)

        phys = material_type_to_physical(str(material).lower())
        if phys is not None:
            return float(phys.linear_attenuation)

        return default_mu

    def _scale_to_absolute_mu(self, reconstructed: np.ndarray, material) -> np.ndarray:
        """Convert normalized reconstruction values to absolute attenuation (cm^-1)."""
        material_mu = self._resolve_reference_mu(material)

        # Normalize reconstructed values to 0-1 range.
        rmin, rmax = reconstructed.min(), reconstructed.max()
        if rmax > rmin:
            normalized = (reconstructed - rmin) / (rmax - rmin)
        else:
            normalized = np.zeros_like(reconstructed, dtype=np.float32)

        # Scale to [0, material_mu].
        mu_volume = np.clip(normalized, 0.0, 1.0) * material_mu

        return mu_volume.astype(np.float32)
