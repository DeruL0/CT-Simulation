"""
Structure Modifier

Main StructureModifier class for generating lattice structures and defects.
GPU acceleration available via CuPy.
"""

from typing import Optional, Tuple, Callable, List
import logging
import numpy as np

from ..voxelizer import VoxelGrid
from .types import LatticeType, DefectShape, LatticeConfig, DefectConfig
from .annotations import VoidAnnotation, AnnotationSet
from .geometry_ops import (
    add_sphere_void as ops_add_sphere_void,
    add_cylinder_void as ops_add_cylinder_void,
    add_ellipsoid_void as ops_add_ellipsoid_void,
    apply_grid_pattern as ops_apply_grid_pattern,
)
from .lattice_ops import compute_tpms_field, annotate_lattice_voids

# GPU support detection
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    HAS_GPU = True
except ImportError:
    HAS_GPU = False
    cp = None


class StructureModifier:
    """
    Industrial structure modifier for VoxelGrid.
    
    Provides methods to:
    - Generate TPMS lattice structures with density control
    - Generate random defects (voids) with density and size distribution
    - Manually add geometric primitives (spheres, cylinders)
    """
    
    def __init__(self, voxel_grid: VoxelGrid):
        """
        Initialize with a VoxelGrid to modify.
        
        Args:
            voxel_grid: The VoxelGrid to modify (will be modified in place)
        """
        self.grid = voxel_grid
        self._original_data = voxel_grid.data.copy()
    
    def reset(self) -> None:
        """Reset grid to original state."""
        self.grid.data = self._original_data.copy()
    
    # =========================================================================
    # TPMS Lattice Generation
    # =========================================================================
    
    def generate_lattice(
        self,
        config: LatticeConfig,
        mask_to_solid: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
        use_gpu: bool = True
    ) -> Tuple[VoxelGrid, AnnotationSet]:
        """
        Generate TPMS lattice structure with target density.
        GPU-accelerated with optional shell preservation.

        Returns:
            Tuple of (modified VoxelGrid, AnnotationSet with lattice void annotations)
        """
        logging.info(f"Generating {config.lattice_type.value} lattice: "
                     f"density={config.density_percent}%, cell={config.cell_size_mm}mm"
                     f"{' (shell preserved)' if config.preserve_shell else ''}")
        
        if progress_callback: progress_callback(0.05)
        
        shape = self.grid.shape
        voxel_size = self.grid.voxel_size
        
        # Determine backend
        xp = cp if (use_gpu and HAS_GPU) else np
        is_gpu = xp is cp
        
        if is_gpu:
            logging.info("Using GPU for lattice generation")
        
        # Shell preservation: compute inner mask via erosion
        if config.preserve_shell:
            inner_mask = self._compute_inner_mask(config.shell_thickness_mm, use_gpu)
        else:
            inner_mask = None
        
        if progress_callback: progress_callback(0.15)
        
        # Create coordinate arrays (1D for broadcasting efficiency)
        x = xp.arange(shape[0], dtype=xp.float32) * voxel_size
        y = xp.arange(shape[1], dtype=xp.float32) * voxel_size
        z = xp.arange(shape[2], dtype=xp.float32) * voxel_size
        
        L = config.cell_size_mm
        freq = 2 * xp.pi / L
        
        # Compute TPMS using broadcasting (memory efficient)
        Xs = x[:, None, None] * freq
        Ys = y[None, :, None] * freq
        Zs = z[None, None, :] * freq
        
        if progress_callback: progress_callback(0.3)
        
        # Compute TPMS field
        tpms_field = compute_tpms_field(config.lattice_type, Xs, Ys, Zs, xp)
        
        if progress_callback: progress_callback(0.5)
        
        # Find iso-value for target density
        solid_mask_orig = self.grid.data > 0.5
        if is_gpu:
            solid_mask_orig_xp = xp.asarray(solid_mask_orig)
            tpms_flat = tpms_field[solid_mask_orig_xp] if mask_to_solid else tpms_field.ravel()
            target_density = config.density_percent / 100.0
            iso_value = float(xp.percentile(tpms_flat, (1 - target_density) * 100))
        else:
            tpms_flat = tpms_field[solid_mask_orig] if mask_to_solid else tpms_field.ravel()
            target_density = config.density_percent / 100.0
            iso_value = float(np.percentile(tpms_flat, (1 - target_density) * 100))
        
        if progress_callback: progress_callback(0.7)
        
        # Apply threshold
        lattice_mask = tpms_field > iso_value
        
        if progress_callback: progress_callback(0.85)
        
        # Combine with solid mask and shell
        # GPU path: keep operations on GPU, transfer only final result
        if is_gpu:
            # solid_mask_orig_xp is already on GPU from iso_value calculation
            if mask_to_solid:
                modified_gpu = solid_mask_orig_xp & lattice_mask
            else:
                modified_gpu = lattice_mask
            
            if config.preserve_shell and inner_mask is not None:
                inner_mask_gpu = xp.asarray(inner_mask)
                shell_mask_gpu = solid_mask_orig_xp & ~inner_mask_gpu
                final_gpu = xp.where(shell_mask_gpu | (inner_mask_gpu & modified_gpu), 1.0, 0.0)
                self.grid.data = xp.asnumpy(final_gpu.astype(xp.float32))
            else:
                self.grid.data = xp.asnumpy(modified_gpu.astype(xp.float32))
        else:
            # CPU path (unchanged)
            if mask_to_solid:
                modified = solid_mask_orig & lattice_mask
            else:
                modified = lattice_mask
            
            if config.preserve_shell and inner_mask is not None:
                shell_mask = solid_mask_orig & ~inner_mask
                self.grid.data = np.where(shell_mask | (inner_mask & modified), 1.0, 0.0).astype(np.float32)
            else:
                self.grid.data = modified.astype(np.float32)
        
        actual_density = np.mean(self.grid.data > 0.5) * 100
        logging.info(f"Lattice generated: actual density = {actual_density:.1f}%")
        
        if progress_callback: progress_callback(0.9)
        
        # --- Annotate lattice voids via connected component labeling ---
        annotations = annotate_lattice_voids(self.grid, self._original_data)
        logging.info(f"Lattice annotation: {len(annotations.voids)} void regions detected")
        
        if progress_callback: progress_callback(1.0)
        
        return self.grid, annotations

    def _compute_inner_mask(self, shell_thickness_mm: float, use_gpu: bool = True) -> np.ndarray:
        """Compute inner mask by eroding the solid."""
        voxel_size = self.grid.voxel_size
        iterations = max(1, int(shell_thickness_mm / voxel_size))
        
        solid_mask = self._original_data > 0.5
        
        if use_gpu and HAS_GPU:
            solid_gpu = cp.asarray(solid_mask)
            inner_gpu = cp_ndimage.binary_erosion(solid_gpu, iterations=iterations, brute_force=True)
            return cp.asnumpy(inner_gpu)
        else:
            from scipy import ndimage
            return ndimage.binary_erosion(solid_mask, iterations=iterations)

    # =========================================================================
    # Random Defects Generation
    # =========================================================================

    def generate_random_voids(
        self,
        config: DefectConfig,
        mask_to_solid: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
        use_gpu: bool = True
    ) -> Tuple[VoxelGrid, AnnotationSet]:
        """Generate random voids (defects) with target porosity.

        Returns:
            Tuple of (modified VoxelGrid, AnnotationSet with per-void annotations)
        """
        logging.info(f"Generating random {config.shape.value} voids: "
                     f"porosity={config.density_percent}%, "
                     f"size={config.size_mean_mm}±{config.size_std_mm}mm"
                     f"{' (shell preserved)' if config.preserve_shell else ''}")
        
        rng = np.random.default_rng(config.seed)
        
        shape = self.grid.shape
        voxel_size = self.grid.voxel_size
        
        if config.preserve_shell:
            inner_mask = self._compute_inner_mask(config.shell_thickness_mm, use_gpu)
            work_mask = inner_mask & (self._original_data > 0.5)
        else:
            work_mask = self._original_data > 0.5 if mask_to_solid else np.ones(shape, dtype=bool)
        
        if progress_callback: progress_callback(0.05)
        
        solid_indices = np.argwhere(work_mask)
        if len(solid_indices) == 0:
            logging.warning("No modifiable voxels found, skipping defect generation")
            empty_ann = AnnotationSet(
                step_index=0, compression_ratio=0.0,
                voxel_size=voxel_size, volume_shape=shape,
                origin=self.grid.origin.copy(),
            )
            return self.grid, empty_ann
        
        min_idx = solid_indices.min(axis=0)
        max_idx = solid_indices.max(axis=0)
        
        min_pos = self.grid.origin + min_idx * voxel_size
        max_pos = self.grid.origin + max_idx * voxel_size
        
        if progress_callback: progress_callback(0.1)
        
        total_volume = np.prod(max_pos - min_pos)
        target_void_volume = total_volume * (config.density_percent / 100.0)
        
        current_void_volume = 0.0
        defects_placed = 0
        max_attempts = 10000
        
        max_candidates = max_attempts
        all_radii = np.maximum(0.5 * voxel_size, 
                               rng.normal(config.size_mean_mm, config.size_std_mm, size=max_candidates))
        all_centers = rng.uniform(min_pos, max_pos, size=(max_candidates, 3))
        
        for i in range(max_candidates):
            r = all_radii[i]
            all_centers[i] = np.clip(all_centers[i], min_pos + r, max_pos - r)
        
        if progress_callback: progress_callback(0.15)
        
        placed_centers = []
        placed_radii = []
        void_annotations: List[VoidAnnotation] = []
        candidate_idx = 0
        
        while current_void_volume < target_void_volume and candidate_idx < max_candidates:
            center = all_centers[candidate_idx]
            radius = all_radii[candidate_idx]
            candidate_idx += 1
            
            if len(placed_centers) > 0:
                centers_arr = np.array(placed_centers)
                radii_arr = np.array(placed_radii)
                distances = np.linalg.norm(centers_arr - center, axis=1)
                if np.any(distances < radii_arr + radius):
                    continue
            
            # --- Per-shape void placement + annotation ---
            void_id = defects_placed + 1
            ann_kwargs: dict = {}

            if config.shape == DefectShape.SPHERE:
                self._add_sphere_void_masked(center, radius, work_mask if config.preserve_shell else None)
                void_volume = (4/3) * np.pi * radius**3
                ann_kwargs = dict(
                    shape="sphere",
                    radius_mm=float(radius),
                    volume_mm3=float(void_volume),
                )
            elif config.shape == DefectShape.CYLINDER:
                axis = rng.normal(size=3)
                axis = axis / np.linalg.norm(axis)
                length = radius * 3
                start = center - axis * length / 2
                end = center + axis * length / 2
                self._add_cylinder_void_masked(start, end, radius / 2, work_mask if config.preserve_shell else None)
                void_volume = np.pi * (radius/2)**2 * length
                ann_kwargs = dict(
                    shape="cylinder",
                    radius_mm=float(radius),
                    volume_mm3=float(void_volume),
                    axis_direction=axis.copy(),
                    length_mm=float(length),
                )
            elif config.shape == DefectShape.ELLIPSOID:
                radii = rng.uniform(0.5, 1.5, size=3) * radius
                self._add_ellipsoid_void_masked(center, radii, work_mask if config.preserve_shell else None)
                void_volume = (4/3) * np.pi * np.prod(radii)
                ann_kwargs = dict(
                    shape="ellipsoid",
                    radius_mm=float(radius),
                    volume_mm3=float(void_volume),
                    radii_mm=radii.copy(),
                )

            # Build annotation
            void_annotations.append(VoidAnnotation(
                id=void_id,
                center_mm=center.copy(),
                **ann_kwargs,
            ))
            
            placed_centers.append(center)
            placed_radii.append(radius)
            current_void_volume += void_volume
            defects_placed += 1
            
            if progress_callback and defects_placed % 20 == 0:
                vol_progress = min(1.0, current_void_volume / (target_void_volume + 1e-9))
                progress_callback(0.15 + vol_progress * 0.8)
        
        actual_porosity = (1 - np.mean(self.grid.data > 0.5)) * 100
        logging.info(f"Generated {defects_placed} defects, actual porosity = {actual_porosity:.1f}%")
        
        # Build AnnotationSet and compute bounding boxes
        ann_set = AnnotationSet(
            step_index=0,
            compression_ratio=0.0,
            voxel_size=voxel_size,
            volume_shape=shape,
            origin=self.grid.origin.copy(),
            voids=void_annotations,
        )
        ann_set.recompute_bboxes()
        logging.info(f"Annotation: {len(void_annotations)} voids annotated")
        
        if progress_callback: progress_callback(1.0)
        
        return self.grid, ann_set

    # =========================================================================
    # Manual Geometric Modifiers
    # =========================================================================

    def add_sphere_void(self, center: Tuple[float, float, float], radius: float) -> VoxelGrid:
        """Add a spherical void (optimized with bounding box)."""
        return ops_add_sphere_void(self.grid, center, radius, work_mask=None)
    
    def add_cylinder_void(self, start: Tuple[float, float, float], end: Tuple[float, float, float], radius: float) -> VoxelGrid:
        """Add a cylindrical void (optimized with bounding box)."""
        return ops_add_cylinder_void(self.grid, start, end, radius, work_mask=None)
    
    def add_ellipsoid_void(self, center: Tuple[float, float, float], radii: Tuple[float, float, float]) -> VoxelGrid:
        """Add an ellipsoidal void (optimized)."""
        return ops_add_ellipsoid_void(self.grid, center, radii, work_mask=None)

    # =========================================================================
    # Masked Void Methods (for Shell Preservation)
    # =========================================================================
    
    def _add_sphere_void_masked(self, center: Tuple[float, float, float], radius: float, work_mask: Optional[np.ndarray] = None) -> VoxelGrid:
        """Add spherical void, optionally respecting work_mask."""
        return ops_add_sphere_void(self.grid, center, radius, work_mask=work_mask)
    
    def _add_cylinder_void_masked(self, start: Tuple[float, float, float], end: Tuple[float, float, float], radius: float, work_mask: Optional[np.ndarray] = None) -> VoxelGrid:
        """Add cylindrical void, optionally respecting work_mask."""
        return ops_add_cylinder_void(self.grid, start, end, radius, work_mask=work_mask)
    
    def _add_ellipsoid_void_masked(self, center: Tuple[float, float, float], radii: Tuple[float, float, float], work_mask: Optional[np.ndarray] = None) -> VoxelGrid:
        """Add ellipsoidal void, optionally respecting work_mask."""
        return ops_add_ellipsoid_void(self.grid, center, radii, work_mask=work_mask)

    # =========================================================================
    # Grid Pattern
    # =========================================================================
    
    def apply_grid_pattern(self, spacing_mm: float, thickness_mm: float) -> VoxelGrid:
        """Apply a rectilinear grid pattern (logical AND with existing solid)."""
        return ops_apply_grid_pattern(self.grid, spacing_mm, thickness_mm)
