"""
Low-level voxel geometry operations for structure modifiers.
"""

from typing import Optional, Tuple

import numpy as np

from ..voxelizer import VoxelGrid


def world_to_voxel(grid: VoxelGrid, points: np.ndarray) -> np.ndarray:
    """Convert world-space points (mm) to voxel-space coordinates."""
    return (np.asarray(points, dtype=np.float64) - grid.origin) / float(grid.voxel_size)


def prepare_local_grid(
    grid_shape: Tuple[int, int, int],
    min_corner_v: np.ndarray,
    max_corner_v: np.ndarray,
) -> Optional[Tuple[Tuple[slice, slice, slice], Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Build a clipped local voxel grid for an axis-aligned bounding box in voxel space.

    Returns:
        ((sl_x, sl_y, sl_z), (X, Y, Z)) or None if the box is empty/outside volume.
    """
    eps = 1e-9
    shape = np.asarray(grid_shape, dtype=np.int64)
    min_v = np.floor(np.asarray(min_corner_v, dtype=np.float64) - eps).astype(np.int64)
    max_v = np.ceil(np.asarray(max_corner_v, dtype=np.float64) + eps).astype(np.int64) + 1

    min_v = np.clip(min_v, 0, shape)
    max_v = np.clip(max_v, 0, shape)

    if np.any(min_v >= max_v):
        return None

    sl_x = slice(int(min_v[0]), int(max_v[0]))
    sl_y = slice(int(min_v[1]), int(max_v[1]))
    sl_z = slice(int(min_v[2]), int(max_v[2]))

    x = np.arange(min_v[0], max_v[0], dtype=np.float64)
    y = np.arange(min_v[1], max_v[1], dtype=np.float64)
    z = np.arange(min_v[2], max_v[2], dtype=np.float64)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    return (sl_x, sl_y, sl_z), (X, Y, Z)


def apply_void_mask(
    grid_data: np.ndarray,
    slices_xyz: Tuple[slice, slice, slice],
    inside_mask: np.ndarray,
    work_mask: Optional[np.ndarray] = None,
) -> None:
    """Apply boolean inside-mask to carve voids, optionally restricted by work mask."""
    if work_mask is not None:
        inside_mask = inside_mask & work_mask[slices_xyz]
    grid_data[slices_xyz][inside_mask] = 0.0


def add_sphere_void(
    grid: VoxelGrid,
    center: Tuple[float, float, float],
    radius: float,
    work_mask: Optional[np.ndarray] = None,
) -> VoxelGrid:
    center_v = world_to_voxel(grid, center)
    radius_v = float(radius) / float(grid.voxel_size)
    if radius_v <= 0.0:
        return grid

    local = prepare_local_grid(grid.shape, center_v - radius_v, center_v + radius_v)
    if local is None:
        return grid
    slices_xyz, (X, Y, Z) = local

    dist_sq = (X - center_v[0]) ** 2 + (Y - center_v[1]) ** 2 + (Z - center_v[2]) ** 2
    inside = dist_sq <= radius_v ** 2
    apply_void_mask(grid.data, slices_xyz, inside, work_mask)
    return grid


def add_cylinder_void(
    grid: VoxelGrid,
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    radius: float,
    work_mask: Optional[np.ndarray] = None,
) -> VoxelGrid:
    start_v = world_to_voxel(grid, start)
    end_v = world_to_voxel(grid, end)
    radius_v = float(radius) / float(grid.voxel_size)
    if radius_v <= 0.0:
        return grid

    axis = end_v - start_v
    length_sq = float(np.dot(axis, axis))
    if length_sq < 1e-9:
        return grid
    length = np.sqrt(length_sq)
    axis_u = axis / length

    min_corner = np.minimum(start_v, end_v) - radius_v
    max_corner = np.maximum(start_v, end_v) + radius_v
    local = prepare_local_grid(grid.shape, min_corner, max_corner)
    if local is None:
        return grid
    slices_xyz, (X, Y, Z) = local

    vx, vy, vz = X - start_v[0], Y - start_v[1], Z - start_v[2]
    proj = vx * axis_u[0] + vy * axis_u[1] + vz * axis_u[2]
    v_sq = vx ** 2 + vy ** 2 + vz ** 2
    perp_dist_sq = v_sq - proj ** 2

    inside = (perp_dist_sq <= radius_v ** 2) & (proj >= 0.0) & (proj <= length)
    apply_void_mask(grid.data, slices_xyz, inside, work_mask)
    return grid


def add_ellipsoid_void(
    grid: VoxelGrid,
    center: Tuple[float, float, float],
    radii: Tuple[float, float, float],
    work_mask: Optional[np.ndarray] = None,
) -> VoxelGrid:
    center_v = world_to_voxel(grid, center)
    radii_v = np.asarray(radii, dtype=np.float64) / float(grid.voxel_size)
    if np.any(radii_v <= 0.0):
        return grid

    local = prepare_local_grid(grid.shape, center_v - radii_v, center_v + radii_v)
    if local is None:
        return grid
    slices_xyz, (X, Y, Z) = local

    norm_dist_sq = (
        ((X - center_v[0]) / radii_v[0]) ** 2
        + ((Y - center_v[1]) / radii_v[1]) ** 2
        + ((Z - center_v[2]) / radii_v[2]) ** 2
    )
    inside = norm_dist_sq <= 1.0
    apply_void_mask(grid.data, slices_xyz, inside, work_mask)
    return grid


def apply_grid_pattern(grid: VoxelGrid, spacing_mm: float, thickness_mm: float) -> VoxelGrid:
    """Apply a rectilinear grid pattern (logical AND with existing solid)."""
    shape = grid.shape
    voxel_size = grid.voxel_size

    x = np.arange(shape[0]) * voxel_size
    y = np.arange(shape[1]) * voxel_size
    z = np.arange(shape[2]) * voxel_size
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    half_thick = thickness_mm / 2

    x_dist = np.abs((X % spacing_mm) - spacing_mm / 2)
    y_dist = np.abs((Y % spacing_mm) - spacing_mm / 2)
    z_dist = np.abs((Z % spacing_mm) - spacing_mm / 2)

    grid_mask = (
        (x_dist > spacing_mm / 2 - half_thick)
        | (y_dist > spacing_mm / 2 - half_thick)
        | (z_dist > spacing_mm / 2 - half_thick)
    )

    solid_mask = grid.data > 0.5
    grid.data = np.where(solid_mask & grid_mask, 1.0, 0.0).astype(np.float32)
    return grid
