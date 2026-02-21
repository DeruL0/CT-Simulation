"""
Void Annotations for AI Training

Data structures for annotating pore/void defects in simulated CT volumes.
Supports per-void metadata, time-series tracking, and export in multiple
formats (JSON, COCO, label volumes).
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from .types import DefectShape


@dataclass
class VoidAnnotation:
    """Annotation for a single void/pore defect."""

    id: int                                        # Unique identifier
    shape: str                                     # "sphere", "cylinder", "ellipsoid", "lattice_void"
    center_mm: np.ndarray                          # World coordinates (x, y, z) in mm
    radius_mm: float                               # Primary radius in mm
    volume_mm3: float                              # Volume in mm³
    radii_mm: Optional[np.ndarray] = None          # Ellipsoid tri-axial radii
    axis_direction: Optional[np.ndarray] = None    # Cylinder axis unit vector
    length_mm: Optional[float] = None              # Cylinder length in mm
    bbox_voxel_min: Optional[np.ndarray] = None    # Bounding box min corner (voxel indices)
    bbox_voxel_max: Optional[np.ndarray] = None    # Bounding box max corner (voxel indices)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        d: Dict[str, Any] = {
            "id": int(self.id),
            "shape": self.shape,
            "center_mm": self.center_mm.tolist(),
            "radius_mm": float(self.radius_mm),
            "volume_mm3": float(self.volume_mm3),
        }
        if self.radii_mm is not None:
            d["radii_mm"] = self.radii_mm.tolist()
        if self.axis_direction is not None:
            d["axis_direction"] = self.axis_direction.tolist()
        if self.length_mm is not None:
            d["length_mm"] = float(self.length_mm)
        if self.bbox_voxel_min is not None:
            d["bbox_voxel_min"] = self.bbox_voxel_min.tolist()
        if self.bbox_voxel_max is not None:
            d["bbox_voxel_max"] = self.bbox_voxel_max.tolist()
        # Derived: voxel-space center (for convenience)
        if self.bbox_voxel_min is not None and self.bbox_voxel_max is not None:
            center_voxel = ((self.bbox_voxel_min + self.bbox_voxel_max) / 2.0).tolist()
            d["center_voxel"] = center_voxel
        return d

    def transformed(
        self,
        new_center: np.ndarray,
        scale_factors: np.ndarray,
    ) -> "VoidAnnotation":
        """Create a transformed copy (for compression tracking).

        Args:
            new_center: New center in world coords (mm).
            scale_factors: Scale factors (sx, sy, sz) to apply to radii.
        """
        scale_factors = np.asarray(scale_factors, dtype=np.float64)
        abs_scale = np.abs(scale_factors)
        if abs_scale.shape != (3,):
            raise ValueError(f"scale_factors must be shape (3,), got {abs_scale.shape}")

        volume_scale = float(np.prod(abs_scale))
        isotropic_tol = 1e-3

        new_shape = self.shape
        new_axis = None
        new_length = self.length_mm
        new_radii = None
        new_radius = float(self.radius_mm)
        new_vol = float(self.volume_mm3 * volume_scale)

        if self.shape == DefectShape.SPHERE.value:
            if np.max(abs_scale) - np.min(abs_scale) <= isotropic_tol:
                new_radius = float(self.radius_mm * abs_scale.mean())
                new_vol = float((4.0 / 3.0) * np.pi * new_radius ** 3)
            else:
                # A sphere under anisotropic scaling becomes an ellipsoid.
                ellipsoid_radii = np.maximum(self.radius_mm * abs_scale, 0.0)
                new_shape = DefectShape.ELLIPSOID.value
                new_radii = ellipsoid_radii
                new_vol = float((4.0 / 3.0) * np.pi * np.prod(ellipsoid_radii))
                new_radius = float(np.cbrt(np.prod(ellipsoid_radii)))

        elif self.shape == DefectShape.ELLIPSOID.value:
            base_radii = self.radii_mm if self.radii_mm is not None else np.full(3, self.radius_mm, dtype=np.float64)
            ellipsoid_radii = np.maximum(np.asarray(base_radii, dtype=np.float64) * abs_scale, 0.0)
            new_radii = ellipsoid_radii
            new_vol = float((4.0 / 3.0) * np.pi * np.prod(ellipsoid_radii))
            new_radius = float(np.cbrt(np.prod(ellipsoid_radii)))

        elif self.shape == DefectShape.CYLINDER.value:
            axis = np.asarray(
                self.axis_direction if self.axis_direction is not None else np.array([1.0, 0.0, 0.0], dtype=np.float64),
                dtype=np.float64,
            )
            axis_norm = np.linalg.norm(axis)
            if axis_norm > 0:
                axis = axis / axis_norm
            else:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)

            scaled_axis_vec = axis * scale_factors
            length_scale = float(np.linalg.norm(scaled_axis_vec))
            if length_scale > 0:
                new_axis = scaled_axis_vec / length_scale
            else:
                new_axis = axis.copy()
                length_scale = 1.0

            # Legacy representation stores cylinder diameter in radius_mm.
            cyl_radius = float(self.radius_mm) / 2.0
            area_scale = volume_scale / max(length_scale, 1e-12)
            cyl_radius_new = cyl_radius * float(np.sqrt(max(area_scale, 0.0)))
            new_length = float((self.length_mm or 0.0) * length_scale)
            new_radius = float(cyl_radius_new * 2.0)
            new_vol = float(np.pi * cyl_radius_new ** 2 * new_length)

        elif self.axis_direction is not None:
            new_axis = np.asarray(self.axis_direction, dtype=np.float64) * scale_factors
            norm = np.linalg.norm(new_axis)
            if norm > 0:
                new_axis = new_axis / norm
            else:
                new_axis = None

        return VoidAnnotation(
            id=self.id,
            shape=new_shape,
            center_mm=new_center.copy(),
            radius_mm=float(new_radius),
            volume_mm3=float(new_vol),
            radii_mm=new_radii.copy() if new_radii is not None else None,
            axis_direction=new_axis.copy() if new_axis is not None else None,
            length_mm=new_length,
            # bbox will be recomputed by AnnotationSet.recompute_bboxes()
            bbox_voxel_min=None,
            bbox_voxel_max=None,
        )


@dataclass
class AnnotationSet:
    """Complete set of void annotations for one time step."""

    step_index: int
    compression_ratio: float
    voxel_size: float
    volume_shape: Tuple[int, int, int]
    origin: np.ndarray                          # Volume origin in world coords
    voids: List[VoidAnnotation] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Bounding box computation
    # ------------------------------------------------------------------

    def _continuous_bounds_to_inclusive_bbox(
        self,
        min_corner_voxel: np.ndarray,
        max_corner_voxel: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert continuous voxel-space bounds to inclusive integer bbox indices.

        Uses a tiny epsilon to suppress float-jitter boundary expansion, e.g.
        15.000000001 should map to 15 instead of 16.
        """
        eps = 1e-6  # voxel units
        vol_max = np.asarray(self.volume_shape, dtype=np.int64) - 1

        min_idx = np.floor(np.asarray(min_corner_voxel, dtype=np.float64) + eps).astype(np.int64)
        max_idx = np.ceil(np.asarray(max_corner_voxel, dtype=np.float64) - eps).astype(np.int64)

        min_idx = np.clip(min_idx, 0, vol_max)
        max_idx = np.clip(max_idx, 0, vol_max)
        max_idx = np.maximum(max_idx, min_idx)
        return min_idx, max_idx

    def recompute_bboxes(self) -> None:
        """Recompute voxel-space bounding boxes for all voids."""
        for v in self.voids:
            center_voxel = (v.center_mm - self.origin) / self.voxel_size
            if v.shape == "sphere":
                half = v.radius_mm / self.voxel_size
                v.bbox_voxel_min, v.bbox_voxel_max = self._continuous_bounds_to_inclusive_bbox(
                    center_voxel - half,
                    center_voxel + half,
                )
            elif v.shape == "ellipsoid" and v.radii_mm is not None:
                half = v.radii_mm / self.voxel_size
                v.bbox_voxel_min, v.bbox_voxel_max = self._continuous_bounds_to_inclusive_bbox(
                    center_voxel - half,
                    center_voxel + half,
                )
            elif v.shape == "cylinder":
                # Conservative bbox using max extent
                max_extent = max(
                    v.radius_mm, (v.length_mm or 0) / 2
                ) / self.voxel_size
                v.bbox_voxel_min, v.bbox_voxel_max = self._continuous_bounds_to_inclusive_bbox(
                    center_voxel - max_extent,
                    center_voxel + max_extent,
                )
            else:
                # Lattice voids or unknown: use radius as half-extent
                half = v.radius_mm / self.voxel_size
                v.bbox_voxel_min, v.bbox_voxel_max = self._continuous_bounds_to_inclusive_bbox(
                    center_voxel - half,
                    center_voxel + half,
                )

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def _compute_void_local_mask(
        self,
        v: VoidAnnotation,
    ) -> Optional[Tuple[int, int, int, np.ndarray]]:
        """
        Compute a per-void local 3D boolean mask within its bbox extent.

        Returns:
            (z0, y0, x0, mask) where mask has shape (z, y, x), or None.
        """
        if v.bbox_voxel_min is None or v.bbox_voxel_max is None:
            return None

        nz, ny, nx = self.volume_shape
        vs = self.voxel_size
        center_voxel = (v.center_mm - self.origin) / vs

        z0, y0, x0 = v.bbox_voxel_min.astype(int)
        z1, y1, x1 = v.bbox_voxel_max.astype(int) + 1  # inclusive -> half-open

        # Clamp to volume bounds
        z0, y0, x0 = max(z0, 0), max(y0, 0), max(x0, 0)
        z1, y1, x1 = min(z1, nz), min(y1, ny), min(x1, nx)
        if z1 <= z0 or y1 <= y0 or x1 <= x0:
            return None

        zz, yy, xx = np.mgrid[z0:z1, y0:y1, x0:x1]

        if v.shape == "sphere":
            dist2 = (
                (zz - center_voxel[0]) ** 2
                + (yy - center_voxel[1]) ** 2
                + (xx - center_voxel[2]) ** 2
            )
            r_vox = v.radius_mm / vs
            mask = dist2 <= r_vox ** 2

        elif v.shape == "ellipsoid" and v.radii_mm is not None:
            r_vox = v.radii_mm / vs
            mask = (
                ((zz - center_voxel[0]) / r_vox[0]) ** 2
                + ((yy - center_voxel[1]) / r_vox[1]) ** 2
                + ((xx - center_voxel[2]) / r_vox[2]) ** 2
            ) <= 1.0

        elif v.shape == "cylinder" and v.axis_direction is not None:
            axis = v.axis_direction
            cyl_r = (v.radius_mm / 2) / vs
            cyl_half_len = ((v.length_mm or 0) / 2) / vs
            dz = zz - center_voxel[0]
            dy = yy - center_voxel[1]
            dx = xx - center_voxel[2]
            proj = dz * axis[0] + dy * axis[1] + dx * axis[2]
            perp2 = dz ** 2 + dy ** 2 + dx ** 2 - proj ** 2
            mask = (perp2 <= cyl_r ** 2) & (np.abs(proj) <= cyl_half_len)

        else:
            # Lattice voids or fallback: sphere approximation from radius.
            dist2 = (
                (zz - center_voxel[0]) ** 2
                + (yy - center_voxel[1]) ** 2
                + (xx - center_voxel[2]) ** 2
            )
            r_vox = v.radius_mm / vs
            mask = dist2 <= r_vox ** 2

        return z0, y0, x0, mask

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to JSON-friendly dict."""
        return {
            "step_index": self.step_index,
            "compression_ratio": float(self.compression_ratio),
            "voxel_size": float(self.voxel_size),
            "volume_shape": list(self.volume_shape),
            "origin": self.origin.tolist(),
            "num_voids": len(self.voids),
            "voids": [v.to_dict() for v in self.voids],
        }

    def to_coco(self) -> Dict[str, Any]:
        """Convert to COCO object-detection format (per-slice 2D bboxes).

        Produces annotations for axial slices (axis=0) where each void
        intersects the slice plane, using the true per-slice intersection mask.
        """
        images = []
        annotations = []
        ann_id = 1
        nz, ny, nx = self.volume_shape

        for z in range(nz):
            image_id = z + 1
            images.append({
                "id": image_id,
                "file_name": f"slice_{z:04d}.png",
                "width": int(nx),
                "height": int(ny),
            })

        for v in self.voids:
            local = self._compute_void_local_mask(v)
            if local is None:
                continue
            z0, y0, x0, mask = local

            for local_z in range(mask.shape[0]):
                slice_mask = mask[local_z]
                if not np.any(slice_mask):
                    continue

                ys, xs = np.where(slice_mask)
                y_min = int(y0 + ys.min())
                y_max = int(y0 + ys.max())
                x_min = int(x0 + xs.min())
                x_max = int(x0 + xs.max())
                z = int(z0 + local_z)

                # Inclusive discrete bounds -> +1
                w = int(x_max - x_min + 1)
                h = int(y_max - y_min + 1)
                if w <= 0 or h <= 0:
                    continue

                annotations.append({
                    "id": ann_id,
                    "image_id": z + 1,
                    "category_id": 1,
                    "bbox": [x_min, y_min, w, h],
                    "area": w * h,
                    "iscrowd": 0,
                    "void_id": int(v.id),
                    "void_shape": v.shape,
                })
                ann_id += 1

        return {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "void", "supercategory": "defect"}],
        }

    def save_json(self, path: Path) -> None:
        """Save annotations as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logging.info(f"Saved annotations ({len(self.voids)} voids) → {path}")

    def save_coco_json(self, path: Path) -> None:
        """Save COCO-format annotations as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_coco(), f, indent=2, ensure_ascii=False)
        logging.info(f"Saved COCO annotations → {path}")

    def generate_label_volume(self) -> np.ndarray:
        """Generate a 3D label volume where each void has a unique integer ID.

        Returns:
            int16 array of shape ``volume_shape`` with 0 = background.
        """
        labels = np.zeros(self.volume_shape, dtype=np.int16)

        for v in self.voids:
            local = self._compute_void_local_mask(v)
            if local is None:
                continue
            z0, y0, x0, mask = local

            z1 = z0 + mask.shape[0]
            y1 = y0 + mask.shape[1]
            x1 = x0 + mask.shape[2]
            labels[z0:z1, y0:y1, x0:x1][mask] = v.id

        return labels

    def save_label_volume(self, path: Path) -> None:
        """Save instance-segmentation label volume as .npy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        labels = self.generate_label_volume()
        np.save(str(path), labels)
        logging.info(f"Saved label volume {labels.shape} → {path}")


@dataclass
class TimeSeriesAnnotations:
    """Annotations across all compression time steps."""

    steps: List[AnnotationSet] = field(default_factory=list)
    config_summary: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: Path) -> None:
        """Save all time-step annotations to *output_dir*."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for ann_set in self.steps:
            step_dir = output_dir / f"Step_{ann_set.step_index:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            ann_set.save_json(step_dir / "annotations.json")
            ann_set.save_coco_json(step_dir / "coco_annotations.json")
            ann_set.save_label_volume(step_dir / "labels.npy")

        # Summary file
        summary = {
            "num_steps": len(self.steps),
            "config": self.config_summary,
            "steps": [
                {
                    "step_index": s.step_index,
                    "compression_ratio": float(s.compression_ratio),
                    "num_voids": len(s.voids),
                }
                for s in self.steps
            ],
        }
        summary_path = output_dir / "annotations_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved time-series annotation summary → {summary_path}")
