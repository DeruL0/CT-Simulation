"""
Lattice-specific TPMS field and annotation helpers.
"""

from typing import List
import logging

import numpy as np
from scipy import ndimage as sp_ndimage

from .types import LatticeType
from .annotations import VoidAnnotation, AnnotationSet


def compute_tpms_field(lattice_type: LatticeType, Xs, Ys, Zs, xp):
    """Compute TPMS field using vectorized operations (works with np or cp)."""
    if lattice_type == LatticeType.GYROID:
        return xp.sin(Xs) * xp.cos(Ys) + xp.sin(Ys) * xp.cos(Zs) + xp.sin(Zs) * xp.cos(Xs)
    if lattice_type == LatticeType.SCHWARZ_PRIMITIVE:
        return xp.cos(Xs) + xp.cos(Ys) + xp.cos(Zs)
    if lattice_type == LatticeType.SCHWARZ_DIAMOND:
        return (
            xp.sin(Xs) * xp.sin(Ys) * xp.sin(Zs)
            + xp.sin(Xs) * xp.cos(Ys) * xp.cos(Zs)
            + xp.cos(Xs) * xp.sin(Ys) * xp.cos(Zs)
            + xp.cos(Xs) * xp.cos(Ys) * xp.sin(Zs)
        )
    if lattice_type == LatticeType.LIDINOID:
        return (
            0.5
            * (
                xp.sin(2 * Xs) * xp.cos(Ys) * xp.sin(Zs)
                + xp.sin(2 * Ys) * xp.cos(Zs) * xp.sin(Xs)
                + xp.sin(2 * Zs) * xp.cos(Xs) * xp.sin(Ys)
            )
            - 0.5
            * (
                xp.cos(2 * Xs) * xp.cos(2 * Ys)
                + xp.cos(2 * Ys) * xp.cos(2 * Zs)
                + xp.cos(2 * Zs) * xp.cos(2 * Xs)
            )
        )
    if lattice_type == LatticeType.SPLIT_P:
        return (
            1.1
            * (
                xp.sin(2 * Xs) * xp.sin(Zs) * xp.cos(Ys)
                + xp.sin(2 * Ys) * xp.sin(Xs) * xp.cos(Zs)
                + xp.sin(2 * Zs) * xp.sin(Ys) * xp.cos(Xs)
            )
            - 0.2
            * (
                xp.cos(2 * Xs) * xp.cos(2 * Ys)
                + xp.cos(2 * Ys) * xp.cos(2 * Zs)
                + xp.cos(2 * Zs) * xp.cos(2 * Xs)
            )
            - 0.4 * (xp.cos(2 * Xs) + xp.cos(2 * Ys) + xp.cos(2 * Zs))
        )
    return xp.sin(Xs) * xp.cos(Ys) + xp.sin(Ys) * xp.cos(Zs) + xp.sin(Zs) * xp.cos(Xs)


def annotate_lattice_voids(grid, original_data: np.ndarray) -> AnnotationSet:
    """
    Detect and annotate individual void regions in the current grid using
    connected component labeling.
    """
    voxel_size = grid.voxel_size
    solid_mask = original_data > 0.5
    void_mask = (grid.data <= 0.5) & solid_mask

    labeled, num_features = sp_ndimage.label(void_mask)
    logging.info("Connected component labeling: %s void regions", num_features)
    components = sp_ndimage.find_objects(labeled)

    void_annotations: List[VoidAnnotation] = []
    for region_id, component_slices in enumerate(components, start=1):
        if component_slices is None:
            continue

        component = labeled[component_slices] == region_id
        voxel_count = int(component.sum())
        if voxel_count < 8:
            continue

        indices = np.argwhere(component)
        if len(indices) == 0:
            continue

        starts = np.array(
            [component_slices[0].start, component_slices[1].start, component_slices[2].start],
            dtype=np.int64,
        )
        centroid_voxel = starts + indices.mean(axis=0)
        center_mm = grid.origin + centroid_voxel * voxel_size
        bbox_min = starts + indices.min(axis=0)
        bbox_max = starts + indices.max(axis=0)
        extent = (bbox_max - bbox_min + 1) * voxel_size
        equiv_radius = (np.prod(extent) * 3 / (4 * np.pi)) ** (1 / 3)
        vol_mm3 = float(voxel_count) * (voxel_size ** 3)

        void_annotations.append(
            VoidAnnotation(
                id=region_id,
                shape="lattice_void",
                center_mm=center_mm,
                radius_mm=float(equiv_radius),
                volume_mm3=vol_mm3,
                bbox_voxel_min=bbox_min,
                bbox_voxel_max=bbox_max,
            )
        )

    return AnnotationSet(
        step_index=0,
        compression_ratio=0.0,
        voxel_size=voxel_size,
        volume_shape=grid.shape,
        origin=grid.origin.copy(),
        voids=void_annotations,
    )
