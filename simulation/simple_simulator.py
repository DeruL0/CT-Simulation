"""
Simple CT Simulator

Fast, non-physical CT simulator using Radon/IRadon transforms directly.
Designed for quick previews without complex physics calculations.
"""

from dataclasses import dataclass
from typing import Any, Optional, Callable
import numpy as np

from core import BaseAnalyzer, ScientificData

from .voxelizer import VoxelGrid
from .volume import CTVolume
from .backends import get_backend
from .materials import MaterialType, require_physical_material


@dataclass(frozen=True)
class SimpleProcessConfig:
    """
    Strongly-typed processing parameters for `SimpleCTSimulator.process`.
    """

    material: Optional[MaterialType] = None
    progress_callback: Optional[Callable[[float], None]] = None


class SimpleCTSimulator(
    BaseAnalyzer[
        VoxelGrid,
        dict[str, Any],
        CTVolume,
        dict[str, Any],
        SimpleProcessConfig,
    ]
):
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

    def process(
        self,
        data: ScientificData[VoxelGrid, dict[str, Any]],
        process_config: SimpleProcessConfig,
    ) -> ScientificData[CTVolume, dict[str, Any]]:
        """
        BaseAnalyzer contract entrypoint.

        Expects `data.primary_data` (or `data.secondary_data["voxel_grid"]`) to be a VoxelGrid.
        """
        if not isinstance(process_config, SimpleProcessConfig):
            raise TypeError(
                "SimpleCTSimulator.process expects SimpleProcessConfig as process_config."
            )

        voxel_grid = data.primary_data
        if not isinstance(voxel_grid, VoxelGrid):
            secondary = data.secondary_data if isinstance(data.secondary_data, dict) else {}
            voxel_grid = secondary.get("voxel_grid")

        if not isinstance(voxel_grid, VoxelGrid):
            raise TypeError(
                "SimpleCTSimulator.process expects ScientificData.primary_data "
                "to be a VoxelGrid."
            )

        if process_config.material is not None and not isinstance(
            process_config.material, MaterialType
        ):
            raise TypeError(
                "SimpleProcessConfig.material must be MaterialType or None."
            )

        ct_volume = self.simulate(
            voxel_grid=voxel_grid,
            material=process_config.material,
            progress_callback=process_config.progress_callback,
        )

        output_metadata = dict(data.metadata)
        output_metadata.update(
            {
                "analyzer": "SimpleCTSimulator",
                "num_projections": self.num_projections,
                "use_gpu": self.use_gpu,
            }
        )

        return ScientificData(
            primary_data=ct_volume,
            secondary_data={"input_voxel_grid": voxel_grid},
            spatial_info={
                "voxel_size_mm": ct_volume.voxel_size,
                "origin": ct_volume.origin,
            },
            metadata=output_metadata,
        )
        
    def simulate(
        self, 
        voxel_grid: VoxelGrid, 
        material: Optional[MaterialType] = None,
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
        voxel_grid.validate()

        # VoxelGrid is treated as (X, Y, Z) in geometry/structure modules.
        # Convert once to CT-standard (Z, Y, X) so CTVolume contract stays consistent
        # with PhysicalCTSimulator and downstream viewers/exporters.
        current_volume = np.transpose(
            voxel_grid.data.astype(np.float32),
            (2, 1, 0),
        )
        
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
    
    def _resolve_reference_mu(self, material: Optional[MaterialType]) -> float:
        """Resolve material reference linear attenuation (cm^-1)."""
        default_mu = 0.171  # Approximate water attenuation near 100 keV.

        if material is None:
            return default_mu

        try:
            return float(require_physical_material(material).linear_attenuation)
        except (TypeError, ValueError, KeyError):
            return default_mu

    def _scale_to_absolute_mu(
        self,
        reconstructed: np.ndarray,
        material: Optional[MaterialType],
    ) -> np.ndarray:
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
