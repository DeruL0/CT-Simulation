"""
Compression Manager

Orchestrates compression simulation workflow for multi-step time series.
"""

import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Tuple
import numpy as np

from .elasticity import ElasticitySolver, ElasticityConfig, apply_displacement_field
from ..structures.annotations import AnnotationSet

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
    warp_chunk_size: int = 64


@dataclass
class CompressionResult:
    """Result of a compression simulation step."""
    step_index: int
    compression_ratio: float
    _volume: Optional[np.ndarray] = field(default=None, repr=False)
    voxel_size: float = 0.5
    cache_path: Optional[str] = None
    annotations: Optional[AnnotationSet] = None
    density_scale: float = 1.0

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
            except OSError as exc:
                logging.warning("Failed to remove cached compression step %s: %s", self.cache_path, exc)
            self.cache_path = None


class CompressionManager:
    """
    Manages compression simulation workflow.
    
    Provides:
    - Multi-step time series compression
    - Physical (elasticity) or geometric (affine) simulation modes
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
        progress_callback: Optional[Callable[[float, str], None]] = None,
        initial_annotations: Optional[AnnotationSet] = None,
    ) -> List[CompressionResult]:
        """
        Run multi-step compression simulation.
        
        Args:
            volume: Initial 3D volume data
            voxel_size: Voxel size in mm
            config: Compression configuration
            progress_callback: Optional callback(progress: 0-1, status: str)
            initial_annotations: Optional initial void annotations to track
            
        Returns:
            List of CompressionResult for each time step
        """
        results = []
        current_volume = volume.copy()
        
        # Track cumulative compression for annotation transformation and density scaling.
        cumulative_scale = np.array([1.0, 1.0, 1.0])  # (z, y, x)
        
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
            voxel_size=voxel_size,
            density_scale=1.0,
            annotations=initial_annotations,
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
                deformed, disp_fields = self._apply_physical_compression(
                    current_volume, incremental_ratio, config, 
                    lambda p, s: progress_callback(
                        ((step - 1) + p) / config.num_steps, s
                    ) if progress_callback else None
                )
            else:
                deformed = self._apply_affine_compression(
                    current_volume, incremental_ratio, config.poisson_ratio
                )
                disp_fields = None
            
            current_volume = deformed

            # Compute cumulative affine deformation scale for this time step.
            inc_scale_z = 1.0 - incremental_ratio
            inc_scale_xy = 1.0 + config.poisson_ratio * incremental_ratio
            cumulative_scale *= np.array([inc_scale_z, inc_scale_xy, inc_scale_xy])
            
            # --- Transform annotations for this step ---
            step_annotations = None
            if initial_annotations is not None:
                step_annotations = self._transform_annotations(
                    initial_annotations, step, step_ratio,
                    cumulative_scale, voxel_size,
                    current_volume.shape, original_axis,
                    disp_fields,
                )

            density_scale = self._mass_conserving_density_scale(cumulative_scale)
            
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
                voxel_size=voxel_size,
                density_scale=density_scale,
                annotations=step_annotations,
            )
            results.append(result)
            
            logging.info(f"Step {step}/{config.num_steps}: compression={step_ratio*100:.1f}%")

        if progress_callback:
            progress_callback(1.0, "Simulation complete")
        
        return results

    @staticmethod
    def _mass_conserving_density_scale(scale_factors: np.ndarray) -> float:
        """Compute density multiplier from affine volume scaling (rho' = rho / det(S))."""
        volume_scale = float(np.prod(np.abs(np.asarray(scale_factors, dtype=np.float64))))
        return 1.0 / max(volume_scale, 1e-12)

    def _transform_annotations(
        self,
        initial_annotations: AnnotationSet,
        step_index: int,
        compression_ratio: float,
        cumulative_scale: np.ndarray,
        voxel_size: float,
        volume_shape: Tuple[int, int, int],
        axis: str,
        disp_fields: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> AnnotationSet:
        """Transform annotations for a compression step.

        Uses displacement fields (physical mode) or affine scaling (geometric mode)
        to update void center positions and radii.
        """
        new_voids = []
        origin = initial_annotations.origin.copy()

        # Map axis name to scale index ordering
        # The internal compression always happens on Z axis (index 0)
        # cumulative_scale is (z, y, x) in the *rotated* frame
        # Annotations are in world coords, so remap scale based on original axis
        if axis == 'X':
            scale_world = np.array([cumulative_scale[1], cumulative_scale[2], cumulative_scale[0]])  # z,y,x -> z,y,x_world
        elif axis == 'Y':
            scale_world = np.array([cumulative_scale[1], cumulative_scale[0], cumulative_scale[2]])
        else:  # Z
            scale_world = cumulative_scale.copy()

        # Volume center for scaling reference
        vol_center = origin + np.array(volume_shape) * voxel_size * 0.5

        for v in initial_annotations.voids:
            # Scale center relative to volume center
            rel = v.center_mm - vol_center
            new_center = vol_center + rel * scale_world

            new_void = v.transformed(new_center, scale_world)
            new_voids.append(new_void)

        ann_set = AnnotationSet(
            step_index=step_index,
            compression_ratio=compression_ratio,
            voxel_size=voxel_size,
            volume_shape=volume_shape,
            origin=origin,
            voids=new_voids,
        )
        ann_set.recompute_bboxes()
        logging.info(f"  Annotations transformed for step {step_index}: {len(new_voids)} voids")
        return ann_set
    
    def _apply_physical_compression(
        self,
        volume: np.ndarray,
        compression_ratio: float,
        config: CompressionConfig,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Apply physical compression using elasticity solver.

        Returns:
            Tuple of (warped volume, (u_z, u_y, u_x) displacement fields)
        """
        
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
        
        return warped, (u_z, u_y, u_x)
    
    def _apply_affine_compression(
        self,
        volume: np.ndarray,
        compression_ratio: float,
        poisson_ratio: float,
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
        
        # Centered crop/pad back to original shape.
        result = np.zeros_like(volume)
        src_slices = []
        dst_slices = []

        for orig_size, new_size in zip(volume.shape, zoomed.shape):
            if new_size >= orig_size:
                src_start = (new_size - orig_size) // 2
                src_end = src_start + orig_size
                dst_start = 0
                dst_end = orig_size
            else:
                src_start = 0
                src_end = new_size
                dst_start = (orig_size - new_size) // 2
                dst_end = dst_start + new_size

            src_slices.append(slice(src_start, src_end))
            dst_slices.append(slice(dst_start, dst_end))

        result[tuple(dst_slices)] = zoomed[tuple(src_slices)]
        return result.astype(volume.dtype)
