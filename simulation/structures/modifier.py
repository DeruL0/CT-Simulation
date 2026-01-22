"""
Structure Modifier

Main StructureModifier class for generating lattice structures and defects.
GPU acceleration available via CuPy.
"""

from typing import Optional, Tuple, Callable
import logging
import numpy as np

from ..voxelizer import VoxelGrid
from .types import LatticeType, DefectShape, LatticeConfig, DefectConfig

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
    ) -> VoxelGrid:
        """
        Generate TPMS lattice structure with target density.
        GPU-accelerated with optional shell preservation.
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
        tpms_field = self._compute_tpms_gpu(config.lattice_type, Xs, Ys, Zs, xp)
        
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
        
        if progress_callback: progress_callback(1.0)
        
        return self.grid
    
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
    
    def _compute_tpms_gpu(self, lattice_type: LatticeType, Xs, Ys, Zs, xp):
        """Compute TPMS field using vectorized operations (works with np or cp)."""
        if lattice_type == LatticeType.GYROID:
            return xp.sin(Xs) * xp.cos(Ys) + xp.sin(Ys) * xp.cos(Zs) + xp.sin(Zs) * xp.cos(Xs)
        elif lattice_type == LatticeType.SCHWARZ_PRIMITIVE:
            return xp.cos(Xs) + xp.cos(Ys) + xp.cos(Zs)
        elif lattice_type == LatticeType.SCHWARZ_DIAMOND:
            return (xp.sin(Xs) * xp.sin(Ys) * xp.sin(Zs) + 
                    xp.sin(Xs) * xp.cos(Ys) * xp.cos(Zs) +
                    xp.cos(Xs) * xp.sin(Ys) * xp.cos(Zs) +
                    xp.cos(Xs) * xp.cos(Ys) * xp.sin(Zs))
        elif lattice_type == LatticeType.LIDINOID:
            return (0.5 * (xp.sin(2*Xs) * xp.cos(Ys) * xp.sin(Zs) +
                          xp.sin(2*Ys) * xp.cos(Zs) * xp.sin(Xs) +
                          xp.sin(2*Zs) * xp.cos(Xs) * xp.sin(Ys)) -
                    0.5 * (xp.cos(2*Xs) * xp.cos(2*Ys) +
                           xp.cos(2*Ys) * xp.cos(2*Zs) +
                           xp.cos(2*Zs) * xp.cos(2*Xs)))
        elif lattice_type == LatticeType.SPLIT_P:
            return (1.1 * (xp.sin(2*Xs) * xp.sin(Zs) * xp.cos(Ys) +
                          xp.sin(2*Ys) * xp.sin(Xs) * xp.cos(Zs) +
                          xp.sin(2*Zs) * xp.sin(Ys) * xp.cos(Xs)) -
                    0.2 * (xp.cos(2*Xs) * xp.cos(2*Ys) +
                           xp.cos(2*Ys) * xp.cos(2*Zs) +
                           xp.cos(2*Zs) * xp.cos(2*Xs)) -
                    0.4 * (xp.cos(2*Xs) + xp.cos(2*Ys) + xp.cos(2*Zs)))
        else:
            return xp.sin(Xs) * xp.cos(Ys) + xp.sin(Ys) * xp.cos(Zs) + xp.sin(Zs) * xp.cos(Xs)

    # =========================================================================
    # Random Defects Generation
    # =========================================================================

    def generate_random_voids(
        self,
        config: DefectConfig,
        mask_to_solid: bool = True,
        progress_callback: Optional[Callable[[float], None]] = None,
        use_gpu: bool = True
    ) -> VoxelGrid:
        """Generate random voids (defects) with target porosity."""
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
            return self.grid
        
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
            
            if config.shape == DefectShape.SPHERE:
                self._add_sphere_void_masked(center, radius, work_mask if config.preserve_shell else None)
                void_volume = (4/3) * np.pi * radius**3
            elif config.shape == DefectShape.CYLINDER:
                axis = rng.normal(size=3)
                axis = axis / np.linalg.norm(axis)
                length = radius * 3
                start = center - axis * length / 2
                end = center + axis * length / 2
                self._add_cylinder_void_masked(start, end, radius / 2, work_mask if config.preserve_shell else None)
                void_volume = np.pi * (radius/2)**2 * length
            elif config.shape == DefectShape.ELLIPSOID:
                radii = rng.uniform(0.5, 1.5, size=3) * radius
                self._add_ellipsoid_void_masked(center, radii, work_mask if config.preserve_shell else None)
                void_volume = (4/3) * np.pi * np.prod(radii)
            
            placed_centers.append(center)
            placed_radii.append(radius)
            current_void_volume += void_volume
            defects_placed += 1
            
            if progress_callback and defects_placed % 20 == 0:
                vol_progress = min(1.0, current_void_volume / (target_void_volume + 1e-9))
                progress_callback(0.15 + vol_progress * 0.8)
        
        actual_porosity = (1 - np.mean(self.grid.data > 0.5)) * 100
        logging.info(f"Generated {defects_placed} defects, actual porosity = {actual_porosity:.1f}%")
        
        if progress_callback: progress_callback(1.0)
        
        return self.grid

    # =========================================================================
    # Manual Geometric Modifiers
    # =========================================================================
    
    def add_sphere_void(self, center: Tuple[float, float, float], radius: float) -> VoxelGrid:
        """Add a spherical void (optimized with bounding box)."""
        center = np.asarray(center)
        voxel_size = self.grid.voxel_size
        
        center_v = (center - self.grid.origin) / voxel_size
        radius_v = radius / voxel_size
        
        min_v = np.floor(center_v - radius_v).astype(int)
        max_v = np.ceil(center_v + radius_v).astype(int) + 1
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, self.grid.shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        sl_x, sl_y, sl_z = slice(min_v[0], max_v[0]), slice(min_v[1], max_v[1]), slice(min_v[2], max_v[2])
        
        x = np.arange(min_v[0], max_v[0])
        y = np.arange(min_v[1], max_v[1])
        z = np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        dist_sq = (X - center_v[0])**2 + (Y - center_v[1])**2 + (Z - center_v[2])**2
        mask = dist_sq <= radius_v**2
        self.grid.data[sl_x, sl_y, sl_z][mask] = 0.0
        
        return self.grid
    
    def add_cylinder_void(self, start: Tuple[float, float, float], end: Tuple[float, float, float], radius: float) -> VoxelGrid:
        """Add a cylindrical void (optimized with bounding box)."""
        start = np.asarray(start)
        end = np.asarray(end)
        voxel_size = self.grid.voxel_size
        
        start_v = (start - self.grid.origin) / voxel_size
        end_v = (end - self.grid.origin) / voxel_size
        radius_v = radius / voxel_size
        
        axis = end_v - start_v
        length_sq = np.dot(axis, axis)
        if length_sq < 1e-9: return self.grid
        length = np.sqrt(length_sq)
        axis_u = axis / length
        
        min_p = np.minimum(start_v, end_v) - radius_v
        max_p = np.maximum(start_v, end_v) + radius_v
        min_v = np.floor(min_p).astype(int)
        max_v = np.ceil(max_p).astype(int) + 1
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, self.grid.shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        sl_x, sl_y, sl_z = slice(min_v[0], max_v[0]), slice(min_v[1], max_v[1]), slice(min_v[2], max_v[2])
        
        x, y, z = np.arange(min_v[0], max_v[0]), np.arange(min_v[1], max_v[1]), np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        vx, vy, vz = X - start_v[0], Y - start_v[1], Z - start_v[2]
        proj = vx * axis_u[0] + vy * axis_u[1] + vz * axis_u[2]
        v_sq = vx**2 + vy**2 + vz**2
        perp_dist_sq = v_sq - proj**2
        
        inside = (perp_dist_sq <= radius_v**2) & (proj >= 0) & (proj <= length)
        self.grid.data[sl_x, sl_y, sl_z][inside] = 0.0
        
        return self.grid
    
    def add_ellipsoid_void(self, center: Tuple[float, float, float], radii: Tuple[float, float, float]) -> VoxelGrid:
        """Add an ellipsoidal void (optimized)."""
        center = np.asarray(center)
        radii = np.asarray(radii)
        
        center_v = (center - self.grid.origin) / self.grid.voxel_size
        radii_v = radii / self.grid.voxel_size
        
        min_v = np.floor(center_v - radii_v).astype(int)
        max_v = np.ceil(center_v + radii_v).astype(int) + 1
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, self.grid.shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        sl_x, sl_y, sl_z = slice(min_v[0], max_v[0]), slice(min_v[1], max_v[1]), slice(min_v[2], max_v[2])
        
        x, y, z = np.arange(min_v[0], max_v[0]), np.arange(min_v[1], max_v[1]), np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        norm_dist_sq = (
            ((X - center_v[0]) / radii_v[0])**2 +
            ((Y - center_v[1]) / radii_v[1])**2 +
            ((Z - center_v[2]) / radii_v[2])**2
        )
        
        self.grid.data[sl_x, sl_y, sl_z][norm_dist_sq <= 1.0] = 0.0
        
        return self.grid

    # =========================================================================
    # Masked Void Methods (for Shell Preservation)
    # =========================================================================
    
    def _add_sphere_void_masked(self, center: Tuple[float, float, float], radius: float, work_mask: Optional[np.ndarray] = None) -> VoxelGrid:
        """Add spherical void, optionally respecting work_mask."""
        if work_mask is None:
            return self.add_sphere_void(center, radius)
        
        center = np.asarray(center)
        voxel_size = self.grid.voxel_size
        
        center_v = (center - self.grid.origin) / voxel_size
        radius_v = radius / voxel_size
        
        min_v = np.floor(center_v - radius_v).astype(int)
        max_v = np.ceil(center_v + radius_v).astype(int) + 1
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, self.grid.shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        sl_x, sl_y, sl_z = slice(min_v[0], max_v[0]), slice(min_v[1], max_v[1]), slice(min_v[2], max_v[2])
        
        x = np.arange(min_v[0], max_v[0])
        y = np.arange(min_v[1], max_v[1])
        z = np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        dist_sq = (X - center_v[0])**2 + (Y - center_v[1])**2 + (Z - center_v[2])**2
        void_mask = (dist_sq <= radius_v**2) & work_mask[sl_x, sl_y, sl_z]
        
        self.grid.data[sl_x, sl_y, sl_z][void_mask] = 0.0
        return self.grid
    
    def _add_cylinder_void_masked(self, start: Tuple[float, float, float], end: Tuple[float, float, float], radius: float, work_mask: Optional[np.ndarray] = None) -> VoxelGrid:
        """Add cylindrical void, optionally respecting work_mask."""
        if work_mask is None:
            return self.add_cylinder_void(start, end, radius)
        
        start = np.asarray(start)
        end = np.asarray(end)
        voxel_size = self.grid.voxel_size
        
        start_v = (start - self.grid.origin) / voxel_size
        end_v = (end - self.grid.origin) / voxel_size
        radius_v = radius / voxel_size
        
        axis = end_v - start_v
        length_sq = np.dot(axis, axis)
        if length_sq < 1e-9: return self.grid
        length = np.sqrt(length_sq)
        axis_u = axis / length
        
        min_p = np.minimum(start_v, end_v) - radius_v
        max_p = np.maximum(start_v, end_v) + radius_v
        min_v = np.floor(min_p).astype(int)
        max_v = np.ceil(max_p).astype(int) + 1
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, self.grid.shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        sl_x, sl_y, sl_z = slice(min_v[0], max_v[0]), slice(min_v[1], max_v[1]), slice(min_v[2], max_v[2])
        
        x, y, z = np.arange(min_v[0], max_v[0]), np.arange(min_v[1], max_v[1]), np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        vx, vy, vz = X - start_v[0], Y - start_v[1], Z - start_v[2]
        proj = vx * axis_u[0] + vy * axis_u[1] + vz * axis_u[2]
        v_sq = vx**2 + vy**2 + vz**2
        perp_dist_sq = v_sq - proj**2
        
        inside = (perp_dist_sq <= radius_v**2) & (proj >= 0) & (proj <= length)
        void_mask = inside & work_mask[sl_x, sl_y, sl_z]
        
        self.grid.data[sl_x, sl_y, sl_z][void_mask] = 0.0
        return self.grid
    
    def _add_ellipsoid_void_masked(self, center: Tuple[float, float, float], radii: Tuple[float, float, float], work_mask: Optional[np.ndarray] = None) -> VoxelGrid:
        """Add ellipsoidal void, optionally respecting work_mask."""
        if work_mask is None:
            return self.add_ellipsoid_void(center, radii)
        
        center = np.asarray(center)
        radii = np.asarray(radii)
        
        center_v = (center - self.grid.origin) / self.grid.voxel_size
        radii_v = radii / self.grid.voxel_size
        
        min_v = np.floor(center_v - radii_v).astype(int)
        max_v = np.ceil(center_v + radii_v).astype(int) + 1
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, self.grid.shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        sl_x, sl_y, sl_z = slice(min_v[0], max_v[0]), slice(min_v[1], max_v[1]), slice(min_v[2], max_v[2])
        
        x, y, z = np.arange(min_v[0], max_v[0]), np.arange(min_v[1], max_v[1]), np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        norm_dist_sq = (
            ((X - center_v[0]) / radii_v[0])**2 +
            ((Y - center_v[1]) / radii_v[1])**2 +
            ((Z - center_v[2]) / radii_v[2])**2
        )
        
        void_mask = (norm_dist_sq <= 1.0) & work_mask[sl_x, sl_y, sl_z]
        self.grid.data[sl_x, sl_y, sl_z][void_mask] = 0.0
        return self.grid

    # =========================================================================
    # Grid Pattern
    # =========================================================================
    
    def apply_grid_pattern(self, spacing_mm: float, thickness_mm: float) -> VoxelGrid:
        """Apply a rectilinear grid pattern (logical AND with existing solid)."""
        shape = self.grid.shape
        voxel_size = self.grid.voxel_size
        
        x = np.arange(shape[0]) * voxel_size
        y = np.arange(shape[1]) * voxel_size
        z = np.arange(shape[2]) * voxel_size
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        half_thick = thickness_mm / 2
        
        x_dist = np.abs((X % spacing_mm) - spacing_mm / 2)
        y_dist = np.abs((Y % spacing_mm) - spacing_mm / 2)
        z_dist = np.abs((Z % spacing_mm) - spacing_mm / 2)
        
        grid_mask = (
            (x_dist > spacing_mm / 2 - half_thick) |
            (y_dist > spacing_mm / 2 - half_thick) |
            (z_dist > spacing_mm / 2 - half_thick)
        )
        
        solid_mask = self.grid.data > 0.5
        self.grid.data = np.where(solid_mask & grid_mask, 1.0, 0.0).astype(np.float32)
        
        return self.grid
