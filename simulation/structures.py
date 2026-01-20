"""
Structure Modifier

Industrial-grade structure generation for CT simulation.
Supports lattice structures (TPMS), random defect generation, and manual modifiers.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List
import logging
import numpy as np

from .voxelizer import VoxelGrid


class LatticeType(Enum):
    """Available TPMS lattice types."""
    GYROID = "gyroid"
    SCHWARZ_PRIMITIVE = "primitive"
    SCHWARZ_DIAMOND = "diamond"
    LIDINOID = "lidinoid"
    SPLIT_P = "split_p"


class DefectShape(Enum):
    """Available defect shapes for random generation."""
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    ELLIPSOID = "ellipsoid"


@dataclass
class DefectConfig:
    """Configuration for random defect generation."""
    shape: DefectShape = DefectShape.SPHERE
    density_percent: float = 5.0  # Target porosity (0-100%)
    size_mean_mm: float = 2.0  # Mean defect size in mm
    size_std_mm: float = 0.5  # Size standard deviation
    seed: Optional[int] = None  # Random seed for reproducibility


@dataclass
class LatticeConfig:
    """Configuration for lattice generation."""
    lattice_type: LatticeType = LatticeType.GYROID
    density_percent: float = 30.0  # Target solid volume fraction (0-100%)
    cell_size_mm: float = 5.0  # Unit cell size in mm


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
        progress_callback=None
    ) -> VoxelGrid:
        """
        Generate TPMS lattice structure with target density.
        
        Args:
            config: LatticeConfig with type, density, and cell size
            mask_to_solid: If True, only apply lattice where original solid exists
            progress_callback: Optional callback(float) for progress reporting
            
        Returns:
            Modified VoxelGrid
        """
        logging.info(f"Generating {config.lattice_type.value} lattice: "
                     f"density={config.density_percent}%, cell={config.cell_size_mm}mm")
        
        if progress_callback: progress_callback(0.1)
        
        # Create coordinate grids
        shape = self.grid.shape
        voxel_size = self.grid.voxel_size
        
        # Physical coordinates (in mm)
        x = np.arange(shape[0]) * voxel_size
        y = np.arange(shape[1]) * voxel_size
        z = np.arange(shape[2]) * voxel_size
        
        # Memory optimization: Compute TPMS in chunks if needed?
        # For now, just keep it simple but maybe explicit broadcast if heavy
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        if progress_callback: progress_callback(0.3)
        
        # Scale to unit cell
        L = config.cell_size_mm
        freq = 2 * np.pi / L
        Xs, Ys, Zs = X * freq, Y * freq, Z * freq
        
        # Compute TPMS implicit function
        tpms_field = self._compute_tpms(config.lattice_type, Xs, Ys, Zs)
        
        if progress_callback: progress_callback(0.6)
        
        # Find iso-value for target density
        iso_value = self._find_iso_for_density(
            tpms_field, 
            config.density_percent / 100.0,
            self.grid.data if mask_to_solid else None
        )
        
        if progress_callback: progress_callback(0.8)
        
        # Apply threshold
        lattice_mask = tpms_field > iso_value
        
        if mask_to_solid:
            # Only keep lattice where original solid exists
            solid_mask = self.grid.data > 0.5
            self.grid.data = np.where(solid_mask & lattice_mask, 1.0, 0.0).astype(np.float32)
        else:
            self.grid.data = lattice_mask.astype(np.float32)
        
        actual_density = np.mean(self.grid.data > 0.5) * 100
        logging.info(f"Lattice generated: actual density = {actual_density:.1f}%")
        
        if progress_callback: progress_callback(1.0)
        
        return self.grid
    
    # ... (skipping _compute_tpms and _find_iso_for_density which are largely math) ...

    def generate_random_voids(
        self,
        config: DefectConfig,
        mask_to_solid: bool = True,
        progress_callback=None
    ) -> VoxelGrid:
        """
        Generate random voids (defects) with target porosity.
        
        Args:
            config: DefectConfig with shape, density, and size distribution
            mask_to_solid: If True, only place defects within original solid
            progress_callback: Optional callback(float) for progress reporting
            
        Returns:
            Modified VoxelGrid
        """
        logging.info(f"Generating random {config.shape.value} voids: "
                     f"porosity={config.density_percent}%, "
                     f"size={config.size_mean_mm}±{config.size_std_mm}mm")
        
        rng = np.random.default_rng(config.seed)
        
        shape = self.grid.shape
        voxel_size = self.grid.voxel_size
        
        # Get solid region bounds
        if mask_to_solid:
            solid_mask = self._original_data > 0.5
            solid_indices = np.argwhere(solid_mask)
            if len(solid_indices) == 0:
                logging.warning("No solid voxels found, skipping defect generation")
                return self.grid
            
            min_idx = solid_indices.min(axis=0)
            max_idx = solid_indices.max(axis=0)
        else:
            min_idx = np.array([0, 0, 0])
            max_idx = np.array(shape) - 1
        
        # Convert to physical coordinates
        min_pos = self.grid.origin + min_idx * voxel_size
        max_pos = self.grid.origin + max_idx * voxel_size
        
        # Target void volume
        # Use approximate total volume of bounding box for target calculation logic
        # Ideally we'd use exact solid volume, but this is consistent with previous logic
        total_volume = np.prod(max_pos - min_pos)
        target_void_volume = total_volume * (config.density_percent / 100.0)
        
        current_void_volume = 0.0
        defects_placed = 0
        max_attempts = 10000
        attempts = 0
        
        # Keep track of placed defects
        placed_centers = []
        placed_radii = []
        
        while current_void_volume < target_void_volume and attempts < max_attempts:
            attempts += 1
            
            # Progress reporting (every 50 attempts or so)
            if progress_callback and attempts % 50 == 0:
                 # Heuristic progress: mix of volume achieved and attempts used
                 vol_progress = min(1.0, current_void_volume / (target_void_volume + 1e-9))
                 progress_callback(vol_progress * 0.9)
            
            # Generate random size
            radius = max(0.5 * voxel_size, 
                        rng.normal(config.size_mean_mm, config.size_std_mm))
            
            # Generate random position
            center = rng.uniform(min_pos + radius, max_pos - radius, size=3)
            
            # Check for overlap with existing defects (simplified)
            overlap = False
            for pc, pr in zip(placed_centers, placed_radii):
                dist = np.linalg.norm(center - pc)
                if dist < radius + pr:
                    overlap = True
                    break
            
            if overlap:
                continue
            
            # Place the defect
            if config.shape == DefectShape.SPHERE:
                self.add_sphere_void(center, radius)
                void_volume = (4/3) * np.pi * radius**3
            elif config.shape == DefectShape.CYLINDER:
                # Random orientation
                axis = rng.normal(size=3)
                axis = axis / np.linalg.norm(axis)
                length = radius * 3  # Aspect ratio 3:1
                start = center - axis * length / 2
                end = center + axis * length / 2
                self.add_cylinder_void(start, end, radius / 2) # cylinder radius is distinct from length-defining radius
                void_volume = np.pi * (radius/2)**2 * length
            elif config.shape == DefectShape.ELLIPSOID:
                # Random radii
                radii = rng.uniform(0.5, 1.5, size=3) * radius
                self.add_ellipsoid_void(center, radii)
                void_volume = (4/3) * np.pi * np.prod(radii)
            
            placed_centers.append(center)
            placed_radii.append(radius)
            current_void_volume += void_volume
            defects_placed += 1
        
        actual_porosity = (1 - np.mean(self.grid.data > 0.5)) * 100
        logging.info(f"Generated {defects_placed} defects, "
                     f"actual porosity = {actual_porosity:.1f}%")
        
        if progress_callback: progress_callback(1.0)
        
        return self.grid

    # =========================================================================
    # Manual Geometric Modifiers
    # =========================================================================
    
    def add_sphere_void(
        self,
        center: Tuple[float, float, float],
        radius: float
    ) -> VoxelGrid:
        """Add a spherical void (optimized with bounding box)."""
        center = np.asarray(center)
        shape = self.grid.shape
        voxel_size = self.grid.voxel_size
        
        # Convert to voxel coordinates
        center_v = (center - self.grid.origin) / voxel_size
        radius_v = radius / voxel_size
        
        # Define bounding box (clamped to grid)
        min_v = np.floor(center_v - radius_v).astype(int)
        max_v = np.ceil(center_v + radius_v).astype(int) + 1
        
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        # Extract slices
        sl_x = slice(min_v[0], max_v[0])
        sl_y = slice(min_v[1], max_v[1])
        sl_z = slice(min_v[2], max_v[2])
        
        # Local coordinate grids
        x = np.arange(min_v[0], max_v[0])
        y = np.arange(min_v[1], max_v[1])
        z = np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Compute local distance
        dist_sq = (X - center_v[0])**2 + (Y - center_v[1])**2 + (Z - center_v[2])**2
        
        # Update grid locally
        mask = dist_sq <= radius_v**2
        self.grid.data[sl_x, sl_y, sl_z][mask] = 0.0
        
        return self.grid
    
    def add_cylinder_void(
        self,
        start: Tuple[float, float, float],
        end: Tuple[float, float, float],
        radius: float
    ) -> VoxelGrid:
        """Add a cylindrical void (optimized with bounding box)."""
        start = np.asarray(start)
        end = np.asarray(end)
        shape = self.grid.shape
        voxel_size = self.grid.voxel_size
        
        # Voxel coordinates
        start_v = (start - self.grid.origin) / voxel_size
        end_v = (end - self.grid.origin) / voxel_size
        radius_v = radius / voxel_size
        
        # Cylinder axis
        axis = end_v - start_v
        length_sq = np.dot(axis, axis)
        if length_sq < 1e-9: return self.grid
        length = np.sqrt(length_sq)
        axis_u = axis / length
        
        # Bounding Box Estimation
        # A cylinder from A to B with radius R is contained in [min(A,B)-R, max(A,B)+R]
        min_p = np.minimum(start_v, end_v) - radius_v
        max_p = np.maximum(start_v, end_v) + radius_v
        
        min_v = np.floor(min_p).astype(int)
        max_v = np.ceil(max_p).astype(int) + 1
        
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        # Extract slices
        sl_x = slice(min_v[0], max_v[0])
        sl_y = slice(min_v[1], max_v[1])
        sl_z = slice(min_v[2], max_v[2])
        
        # Local grids
        x = np.arange(min_v[0], max_v[0])
        y = np.arange(min_v[1], max_v[1])
        z = np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Vector from start to point
        vx = X - start_v[0]
        vy = Y - start_v[1]
        vz = Z - start_v[2]
        
        # Project onto axis
        proj = vx * axis_u[0] + vy * axis_u[1] + vz * axis_u[2]
        
        # Perpendicular distance squared
        # dist_sq = |v|^2 - proj^2
        v_sq = vx**2 + vy**2 + vz**2
        perp_dist_sq = v_sq - proj**2
        
        # Inside check
        # Must be within radius AND within [0, length] along axis
        inside = (perp_dist_sq <= radius_v**2) & (proj >= 0) & (proj <= length)
        
        self.grid.data[sl_x, sl_y, sl_z][inside] = 0.0
        
        return self.grid
    
    def add_ellipsoid_void(
        self,
        center: Tuple[float, float, float],
        radii: Tuple[float, float, float]
    ) -> VoxelGrid:
        """Add an ellipsoidal void (optimized)."""
        center = np.asarray(center)
        radii = np.asarray(radii)
        
        # Voxel coords
        center_v = (center - self.grid.origin) / self.grid.voxel_size
        radii_v = radii / self.grid.voxel_size
        
        # Bounding box
        min_v = np.floor(center_v - radii_v).astype(int)
        max_v = np.ceil(center_v + radii_v).astype(int) + 1
        
        min_v = np.maximum(min_v, 0)
        max_v = np.minimum(max_v, self.grid.shape)
        
        if np.any(min_v >= max_v): return self.grid
        
        # Slices
        sl_x = slice(min_v[0], max_v[0])
        sl_y = slice(min_v[1], max_v[1])
        sl_z = slice(min_v[2], max_v[2])
        
        x = np.arange(min_v[0], max_v[0])
        y = np.arange(min_v[1], max_v[1])
        z = np.arange(min_v[2], max_v[2])
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Normalized distance squared
        norm_dist_sq = (
            ((X - center_v[0]) / radii_v[0])**2 +
            ((Y - center_v[1]) / radii_v[1])**2 +
            ((Z - center_v[2]) / radii_v[2])**2
        )
        
        self.grid.data[sl_x, sl_y, sl_z][norm_dist_sq <= 1.0] = 0.0
        
        return self.grid
    
    def apply_grid_pattern(
        self,
        spacing_mm: float,
        thickness_mm: float
    ) -> VoxelGrid:
        """
        Apply a rectilinear grid pattern (logical AND with existing solid).
        
        Args:
            spacing_mm: Distance between grid lines in mm
            thickness_mm: Thickness of grid lines in mm
            
        Returns:
            Modified VoxelGrid
        """
        shape = self.grid.shape
        voxel_size = self.grid.voxel_size
        
        # Create coordinate grids
        x = np.arange(shape[0]) * voxel_size
        y = np.arange(shape[1]) * voxel_size
        z = np.arange(shape[2]) * voxel_size
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Grid pattern: solid where close to grid lines
        half_thick = thickness_mm / 2
        
        # Distance to nearest grid line for each axis
        x_dist = np.abs((X % spacing_mm) - spacing_mm / 2)
        y_dist = np.abs((Y % spacing_mm) - spacing_mm / 2)
        z_dist = np.abs((Z % spacing_mm) - spacing_mm / 2)
        
        # Inside grid if close to any grid line
        grid_mask = (
            (x_dist > spacing_mm / 2 - half_thick) |
            (y_dist > spacing_mm / 2 - half_thick) |
            (z_dist > spacing_mm / 2 - half_thick)
        )
        
        # Apply to solid
        solid_mask = self.grid.data > 0.5
        self.grid.data = np.where(solid_mask & grid_mask, 1.0, 0.0).astype(np.float32)
        
        return self.grid
