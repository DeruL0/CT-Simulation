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
    total_compression: float = 0.2  # 20% total compression (ratio mode)
    num_steps: int = 5  # Number of time steps
    poisson_ratio: float = 0.3  # Material Poisson ratio
    axis: str = 'Z'  # Compression axis: 'Z' (Top-Bottom), 'Y' (Front-Back), 'X' (Left-Right)
    drive_mode: str = "ratio"  # 'ratio' or 'force'
    compression_force_n: float = 0.0  # Applied force in newtons (force mode)
    youngs_modulus_pa: float = 1.0e9  # Equivalent Young's modulus (force mode)
    max_force_compression: float = 0.5  # Safety cap for force-driven strain

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

        total_compression = self._resolve_total_compression(
            current_volume,
            voxel_size,
            config,
        )
        logging.info(
            "Requested compression resolved to %.3f%% using %s mode",
            total_compression * 100.0,
            config.drive_mode,
        )
        
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
            step_ratio = total_compression * step / config.num_steps
            
            if progress_callback:
                overall_progress = (step - 1) / config.num_steps
                progress_callback(overall_progress, f"Step {step}/{config.num_steps}")
            
            # Calculate incremental compression for this step
            prev_ratio = total_compression * (step - 1) / config.num_steps
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

    def _resolve_total_compression(
        self,
        volume: np.ndarray,
        voxel_size: float,
        config: CompressionConfig,
    ) -> float:
        """Resolve the total compression ratio from the selected drive mode."""
        drive_mode = (config.drive_mode or "ratio").strip().lower()
        if drive_mode == "force":
            return self._force_to_compression_ratio(volume, voxel_size, config)
        return float(np.clip(config.total_compression, 0.0, config.max_force_compression))

    def _force_to_compression_ratio(
        self,
        volume: np.ndarray,
        voxel_size: float,
        config: CompressionConfig,
    ) -> float:
        """
        Convert a compressive force into an equivalent uniaxial strain.

        Uses sigma = F / A and epsilon = sigma / E with an area estimate derived
        from the occupied voxel footprint orthogonal to the compression axis.
        """
        force_n = max(float(config.compression_force_n), 0.0)
        youngs_modulus_pa = float(config.youngs_modulus_pa)
        if force_n == 0.0:
            logging.info("Force-driven compression disabled because force is 0 N")
            return 0.0
        if not np.isfinite(youngs_modulus_pa) or youngs_modulus_pa <= 0.0:
            raise ValueError("Young's modulus must be a positive finite value in force mode.")

        contact_area_m2 = self._estimate_contact_area_m2(volume, voxel_size)
        stress_pa = force_n / contact_area_m2
        strain = stress_pa / youngs_modulus_pa
        compression_ratio = float(np.clip(strain, 0.0, config.max_force_compression))

        logging.info(
            "Force-driven compression: F=%.3f N, A=%.6e m^2, sigma=%.6e Pa, E=%.6e Pa, epsilon=%.6f, capped=%.6f",
            force_n,
            contact_area_m2,
            stress_pa,
            youngs_modulus_pa,
            strain,
            compression_ratio,
        )
        return compression_ratio

    @staticmethod
    def _estimate_contact_area_m2(volume: np.ndarray, voxel_size: float) -> float:
        """Estimate compression contact area from the occupied footprint on the YX plane."""
        solid_mask = np.asarray(volume) > 0.5
        if not np.any(solid_mask):
            raise ValueError("Cannot estimate compression area from an empty volume.")

        footprint = np.any(solid_mask, axis=0)
        voxel_area_m2 = (float(voxel_size) * 1e-3) ** 2
        contact_area_m2 = float(np.count_nonzero(footprint)) * voxel_area_m2
        return max(contact_area_m2, voxel_area_m2)

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
        volume_shape_world = tuple(int(v) for v in self._rotated_to_world_vector(
            np.asarray(volume_shape, dtype=np.float64),
            axis,
        ))

        # The internal compression always happens on axis 0 of the rotated frame.
        scale_world = self._rotated_to_world_vector(
            np.asarray(cumulative_scale, dtype=np.float64),
            axis,
        )

        # Volume center for scaling reference
        vol_center = origin + np.array(volume_shape_world, dtype=np.float64) * voxel_size * 0.5

        for v in initial_annotations.voids:
            if disp_fields is not None:
                center_voxel_world = (v.center_mm - origin) / voxel_size
                center_voxel_rot = self._world_to_rotated_vector(center_voxel_world, axis)
                disp_center_rot = self._sample_displacement_vector(disp_fields, center_voxel_rot)
                displaced_center_rot = center_voxel_rot + disp_center_rot
                displaced_center_world = self._rotated_to_world_vector(displaced_center_rot, axis)
                displaced_center_world = np.clip(
                    displaced_center_world,
                    0.0,
                    np.maximum(np.asarray(volume_shape_world, dtype=np.float64) - 1.0, 0.0),
                )
                local_scale_rot = self._estimate_local_scale(disp_fields, center_voxel_rot)
                local_scale_world = self._rotated_to_world_vector(local_scale_rot, axis)
                local_scale_world = np.clip(local_scale_world, 1e-3, 10.0)
                new_center = origin + displaced_center_world * voxel_size
                new_void = v.transformed(new_center, local_scale_world)
            else:
                # Scale center relative to volume center
                rel = v.center_mm - vol_center
                new_center = vol_center + rel * scale_world
                new_void = v.transformed(new_center, scale_world)
            new_voids.append(new_void)

        ann_set = AnnotationSet(
            step_index=step_index,
            compression_ratio=compression_ratio,
            voxel_size=voxel_size,
            volume_shape=volume_shape_world,
            origin=origin,
            voids=new_voids,
        )
        ann_set.recompute_bboxes()
        logging.info(f"  Annotations transformed for step {step_index}: {len(new_voids)} voids")
        return ann_set

    @staticmethod
    def _world_to_rotated_vector(vector: np.ndarray, axis: str) -> np.ndarray:
        """Map a vector from world/original volume order into the rotated solver order."""
        v = np.asarray(vector, dtype=np.float64)
        if axis == 'X':
            return v[[2, 0, 1]]
        if axis == 'Y':
            return v[[1, 0, 2]]
        return v.copy()

    @staticmethod
    def _rotated_to_world_vector(vector: np.ndarray, axis: str) -> np.ndarray:
        """Map a vector from rotated solver order back into world/original volume order."""
        v = np.asarray(vector, dtype=np.float64)
        if axis == 'X':
            return v[[1, 2, 0]]
        if axis == 'Y':
            return v[[1, 0, 2]]
        return v.copy()

    @staticmethod
    def _sample_displacement_vector(
        disp_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
        point: np.ndarray,
    ) -> np.ndarray:
        """Sample the local displacement vector at a fractional voxel coordinate."""
        from scipy import ndimage as sp_ndimage

        coords = np.asarray(point, dtype=np.float64).reshape(3, 1)
        sampled = [
            float(sp_ndimage.map_coordinates(field, coords, order=1, mode='nearest')[0])
            for field in disp_fields
        ]
        return np.asarray(sampled, dtype=np.float64)

    def _estimate_local_scale(
        self,
        disp_fields: Tuple[np.ndarray, np.ndarray, np.ndarray],
        point: np.ndarray,
    ) -> np.ndarray:
        """Estimate local axis scale from diagonal displacement gradients."""
        point = np.asarray(point, dtype=np.float64)
        local_scale = np.ones(3, dtype=np.float64)

        for axis_index in range(3):
            step = np.zeros(3, dtype=np.float64)
            step[axis_index] = 1.0
            disp_plus = self._sample_displacement_vector(disp_fields, point + step)
            disp_minus = self._sample_displacement_vector(disp_fields, point - step)
            grad_ii = 0.5 * (disp_plus[axis_index] - disp_minus[axis_index])
            local_scale[axis_index] = 1.0 + grad_ii

        return local_scale
    
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
