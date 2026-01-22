"""
Compression Manager

Orchestrates compression simulation workflow including multi-step time series
and DICOM export.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List
import numpy as np

from .elasticity import ElasticitySolver, ElasticityConfig, apply_displacement_field

# GPU support detection
try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None


@dataclass
class CompressionConfig:
    """Configuration for compression simulation."""
    # Simulation parameters
    total_compression: float = 0.2  # 20% total compression
    num_steps: int = 5  # Number of time steps
    poisson_ratio: float = 0.3  # Material Poisson ratio
    axis: str = 'Z'  # Compression axis: 'Z' (Top-Bottom), 'Y' (Front-Back), 'X' (Left-Right)
    
    # Physics solver settings
    use_physics: bool = True  # Use elasticity solver vs simple affine
    downsample_factor: int = 4  # Physics grid downsample
    solver_iterations: int = 300
    
    # GPU settings
    use_gpu: bool = True
    warp_chunk_size: int = 64
    
    # Output settings
    output_dir: Optional[Path] = None
    export_dicom: bool = True


@dataclass
class CompressionResult:
    """Result of a compression simulation step."""
    step_index: int
    compression_ratio: float
    _volume: Optional[np.ndarray] = field(default=None, repr=False)
    voxel_size: float = 0.5
    dicom_path: Optional[Path] = None
    cache_path: Optional[str] = None
    
    def __post_init__(self):
        # Backward compatibility for positional args if needed, though dataclasses handle init
        pass

    @property
    def volume(self) -> np.ndarray:
        """Get volume data, loading from cache if necessary."""
        if self._volume is not None:
            return self._volume
        if self.cache_path and Path(self.cache_path).exists():
            # Load and cache for subsequent access
            self._volume = np.load(self.cache_path)
            return self._volume
        raise ValueError(f"Volume data not available for step {self.step_index}")
    
    @volume.setter
    def volume(self, data: np.ndarray):
        self._volume = data
        
    def offload_to_disk(self, directory: Path):
        """Move volume data to disk cache."""
        if self._volume is None:
            return
            
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        filename = directory / f"step_{self.step_index:03d}.npy"
        np.save(filename, self._volume)
        self.cache_path = str(filename)
        self._volume = None  # Free memory
    
    def cleanup(self):
        """Remove cached file from disk."""
        if self.cache_path and Path(self.cache_path).exists():
            try:
                Path(self.cache_path).unlink()
            except OSError:
                pass
            self.cache_path = None


class CompressionManager:
    """
    Manages compression simulation workflow.
    
    Provides:
    - Multi-step time series compression
    - Physical (elasticity) or geometric (affine) simulation modes
    - DICOM export for each time step
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize compression manager.
        
        Args:
            use_gpu: Whether to use GPU acceleration
        """
        self.use_gpu = use_gpu and HAS_GPU
        self._solver: Optional[ElasticitySolver] = None
        
        if self.use_gpu:
            logging.info("CompressionManager: GPU mode enabled")
        else:
            logging.info("CompressionManager: CPU mode")
    
    def run_simulation(
        self,
        volume: np.ndarray,
        voxel_size: float,
        config: CompressionConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> List[CompressionResult]:
        """
        Run multi-step compression simulation.
        
        Args:
            volume: Initial 3D volume data
            voxel_size: Voxel size in mm
            config: Compression configuration
            progress_callback: Optional callback(progress: 0-1, status: str)
            
        Returns:
            List of CompressionResult for each time step
        """
        results = []
        current_volume = volume.copy()
        
        # Rotate volume to standardize compression axis (always compress on Z)
        axis_rotated = False
        original_axis = config.axis.upper()
        if original_axis == 'X':
            # Rotate X axis to Z: (Z,Y,X) -> (X,Y,Z) = transpose (2,1,0) then we compress on axis 0
            # Actually transpose axes so X becomes Z (axis 0)
            current_volume = np.moveaxis(current_volume, 2, 0)  # Move X to position 0
            axis_rotated = True
            logging.info(f"Rotated volume for X-axis compression: {volume.shape} -> {current_volume.shape}")
        elif original_axis == 'Y':
            # Rotate Y axis to Z: Move Y (axis 1) to Z (axis 0)
            current_volume = np.moveaxis(current_volume, 1, 0)
            axis_rotated = True
            logging.info(f"Rotated volume for Y-axis compression: {volume.shape} -> {current_volume.shape}")
        # Z axis: no rotation needed
        
        # Add initial state (step 0)
        initial_volume = current_volume.copy()
        if axis_rotated:
            # Un-rotate for storage
            if original_axis == 'X':
                initial_volume = np.moveaxis(initial_volume, 0, 2)
            elif original_axis == 'Y':
                initial_volume = np.moveaxis(initial_volume, 0, 1)
        
        results.append(CompressionResult(
            step_index=0,
            compression_ratio=0.0,
            _volume=initial_volume,
            voxel_size=voxel_size
        ))
        
        for step in range(1, config.num_steps + 1):
            step_ratio = config.total_compression * step / config.num_steps
            
            if progress_callback:
                overall_progress = (step - 1) / config.num_steps
                progress_callback(overall_progress, f"Step {step}/{config.num_steps}")
            
            # Calculate incremental compression for this step
            prev_ratio = config.total_compression * (step - 1) / config.num_steps
            incremental_ratio = step_ratio - prev_ratio
            
            # Apply compression (always on Z axis now)
            if config.use_physics:
                deformed = self._apply_physical_compression(
                    current_volume, incremental_ratio, config, 
                    lambda p, s: progress_callback(
                        ((step - 1) + p) / config.num_steps, s
                    ) if progress_callback else None
                )
            else:
                deformed = self._apply_affine_compression(
                    current_volume, incremental_ratio, config.poisson_ratio, voxel_size
                )
            
            current_volume = deformed
            
            # Un-rotate for storage
            store_volume = current_volume.copy()
            if axis_rotated:
                if original_axis == 'X':
                    store_volume = np.moveaxis(store_volume, 0, 2)
                elif original_axis == 'Y':
                    store_volume = np.moveaxis(store_volume, 0, 1)
            
            # Store result
            result = CompressionResult(
                step_index=step,
                compression_ratio=step_ratio,
                _volume=store_volume,
                voxel_size=voxel_size
            )
            results.append(result)
            
            logging.info(f"Step {step}/{config.num_steps}: compression={step_ratio*100:.1f}%")
        
        # Export DICOM if requested
        if config.export_dicom and config.output_dir:
            self._export_all_steps(results, config, progress_callback)
        
        if progress_callback:
            progress_callback(1.0, "Simulation complete")
        
        return results
    
    def _apply_physical_compression(
        self,
        volume: np.ndarray,
        compression_ratio: float,
        config: CompressionConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """Apply physical compression using elasticity solver."""
        
        if self._solver is None:
            self._solver = ElasticitySolver(use_gpu=self.use_gpu)
        
        # Configure elasticity
        elasticity_config = ElasticityConfig(
            compression_ratio=compression_ratio,
            poisson_ratio=config.poisson_ratio,
            downsample_factor=config.downsample_factor,
            solver_iterations=config.solver_iterations
        )
        
        # Solve for displacement field
        u_z, u_y, u_x = self._solver.solve(
            volume, elasticity_config, progress_callback
        )
        
        # Apply displacement to warp volume
        warped = apply_displacement_field(
            volume, u_z, u_y, u_x,
            use_gpu=self.use_gpu,
            chunk_size=config.warp_chunk_size,
            progress_callback=progress_callback
        )
        
        return warped
    
    def _apply_affine_compression(
        self,
        volume: np.ndarray,
        compression_ratio: float,
        poisson_ratio: float,
        voxel_size: float
    ) -> np.ndarray:
        """
        Apply simple affine compression (geometric scaling).
        
        Fast fallback without physics simulation.
        """
        from scipy import ndimage as sp_ndimage
        
        # Scale factors
        scale_z = 1.0 - compression_ratio
        scale_xy = 1.0 + poisson_ratio * compression_ratio
        
        # Zoom with adjusted factors
        # Note: zoom > 1 means expansion, zoom < 1 means compression
        zoom_factors = (scale_z, scale_xy, scale_xy)
        
        # Apply zoom (this changes the array size)
        zoomed = sp_ndimage.zoom(volume.astype(np.float32), zoom_factors, order=1)
        
        # Pad or crop to original size (keep centered)
        result = np.zeros_like(volume)
        
        for axis in range(3):
            orig_size = volume.shape[axis]
            new_size = zoomed.shape[axis]
            
            if new_size > orig_size:
                # Crop center
                start = (new_size - orig_size) // 2
                slices_src = [slice(None)] * 3
                slices_src[axis] = slice(start, start + orig_size)
                zoomed = zoomed[tuple(slices_src)]
            elif new_size < orig_size:
                # This path is taken for z-axis compression, need to handle padding
                pass
        
        # Final placement
        result_shape = result.shape
        zoomed_shape = zoomed.shape
        
        # Center the zoomed volume in the result
        offsets = tuple((r - z) // 2 for r, z in zip(result_shape, zoomed_shape))
        slices_dst = tuple(slice(o, o + z) for o, z in zip(offsets, zoomed_shape))
        slices_src = tuple(slice(max(0, -o), z - max(0, o + z - r)) 
                          for o, z, r in zip(offsets, zoomed_shape, result_shape))
        
        # Handle out-of-bounds
        valid_dst = tuple(slice(max(0, s.start), min(r, s.stop)) 
                         for s, r in zip(slices_dst, result_shape))
        valid_src = tuple(slice(s.start + max(0, -o), s.stop + max(0, -o) - s.start + (vd.stop - vd.start)) 
                         for s, o, vd in zip(slices_src, offsets, valid_dst))
        
        try:
            result[valid_dst] = zoomed[valid_src]
        except (ValueError, IndexError):
            # Fallback: simple copy with clipping
            min_shape = tuple(min(r, z) for r, z in zip(result_shape, zoomed_shape))
            result[:min_shape[0], :min_shape[1], :min_shape[2]] = \
                zoomed[:min_shape[0], :min_shape[1], :min_shape[2]]
        
        return result.astype(volume.dtype)
    
    def _export_all_steps(
        self,
        results: List[CompressionResult],
        config: CompressionConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ):
        """Export all steps as DICOM series."""
        try:
            from exporters.dicom import DICOMExporter
            from simulation.volume import CTVolume
        except ImportError as e:
            logging.warning(f"Cannot export DICOM: {e}")
            return
        
        output_dir = Path(config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for i, result in enumerate(results):
            step_dir = output_dir / f"Step_{result.step_index:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            
            # Create CTVolume
            ct_volume = CTVolume(
                data=result.volume,
                voxel_size=result.voxel_size
            )
            
            # Export
            exporter = DICOMExporter(
                series_description=f"Compression Step {result.step_index} ({result.compression_ratio*100:.0f}%)"
            )
            
            files = exporter.export(ct_volume, step_dir)
            result.dicom_path = step_dir
            
            logging.info(f"Exported step {result.step_index} to {step_dir}: {len(files)} files")
            
            if progress_callback:
                progress_callback(0.9 + 0.1 * (i + 1) / len(results), f"Exported step {result.step_index}")
