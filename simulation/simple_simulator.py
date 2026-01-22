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


class SimpleCTSimulator:
    """
    Fast, non-physical CT simulator using Radon/IRadon transforms directly.
    Bypasses complex physics for speed (Fast Mode).
    
    Use this for quick previews. For accurate simulation, use PhysicalCTSimulator.
    """
    
    # Material to approximate HU mapping
    MATERIAL_HU_MAP = {
        'WATER': 0,
        'SOFT_TISSUE': 50,
        'MUSCLE': 40,
        'FAT': -100,
        'BONE_CORTICAL': 1500,
        'BONE_TRABECULAR': 400,
        'CALCIUM': 1800,
        'CALCIUM_ITE': 1700,
        'CALCIUM_ITE_HIGH': 1900,
        'CALCIUM_URIC': 500,
        'CALCIUM_STRUVITE': 700,
    }
    
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
            material: Material type for HU mapping (optional)
            progress_callback: Progress callback (0.0 to 1.0)
            
        Returns:
            CTVolume with reconstructed HU values
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
        
        # Convert to HU values
        hu_volume = self._convert_to_hu(reconstructed, material)
        
        return CTVolume(hu_volume, voxel_grid.voxel_size, voxel_grid.origin)
    
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
    
    def _convert_to_hu(self, reconstructed: np.ndarray, material) -> np.ndarray:
        """Convert normalized values to HU scale."""
        # Get approximate HU value for material
        material_hu = 1000  # Default: bone-like
        if material is not None:
            mat_name = material.name if hasattr(material, 'name') else str(material)
            material_hu = self.MATERIAL_HU_MAP.get(mat_name, 1000)
        
        # Normalize reconstructed values to 0-1 range
        rmin, rmax = reconstructed.min(), reconstructed.max()
        if rmax > rmin:
            normalized = (reconstructed - rmin) / (rmax - rmin)
        else:
            normalized = reconstructed
        
        # Map to HU: 0 -> -1000 (air), 1 -> material_hu
        hu_volume = normalized * (material_hu + 1000) - 1000
        
        return hu_volume.astype(np.float32)
