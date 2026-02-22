"""
Export pipeline helpers for background workers.
"""

import logging
from pathlib import Path
from typing import List, Callable

import numpy as np

from exporters.dicom import DICOMExporter
from simulation.volume import CTVolume


def export_volume_or_series(
    volume_or_list,
    output_dir: str,
    window_center: float,
    window_width: float,
    progress_callback: Callable[[float], None],
    initial_annotations=None,
) -> List[str]:
    """
    Export a single CT volume or a list of compression/time-series volumes.

    Returns:
        Flat list of produced DICOM file paths.
    """
    exporter = DICOMExporter()
    all_files: List[str] = []

    if isinstance(volume_or_list, list):
        items = volume_or_list
        num_items = len(items)

        for i, item in enumerate(items):
            if hasattr(item, "volume"):
                vol_data = item.volume
                voxel_size = item.voxel_size
                origin = getattr(item, "origin", np.zeros(3))
                step_annotations = getattr(item, "annotations", None)
            elif hasattr(item, "data"):
                vol_data = item.data
                voxel_size = item.voxel_size
                origin = item.origin
                step_annotations = None
            else:
                vol_data = item
                voxel_size = 0.5
                origin = np.zeros(3)
                step_annotations = None

            step_dir = Path(output_dir) / f"Step_{i:02d}"
            step_dir.mkdir(parents=True, exist_ok=True)

            temp_vol = CTVolume(vol_data, voxel_size, origin)

            def step_progress(p, idx=i):
                progress_callback((idx + p) / num_items)

            files = exporter.export(
                temp_vol,
                str(step_dir),
                window_center=window_center,
                window_width=window_width,
                progress_callback=step_progress,
            )
            all_files.extend(files)

            if step_annotations is not None:
                try:
                    step_annotations.save_json(step_dir / "annotations.json")
                    step_annotations.save_coco_json(step_dir / "coco_annotations.json")
                    step_annotations.save_label_volume(step_dir / "labels.npy")
                except (OSError, ValueError, RuntimeError) as exc:
                    logging.warning("Failed to save annotations for step %s: %s", i, exc)
    else:
        files = exporter.export(
            volume_or_list,
            output_dir,
            window_center=window_center,
            window_width=window_width,
            progress_callback=progress_callback,
        )
        all_files.extend(files)

        if initial_annotations is not None:
            out_dir = Path(output_dir)
            initial_annotations.save_json(out_dir / "annotations.json")
            initial_annotations.save_coco_json(out_dir / "coco_annotations.json")
            initial_annotations.save_label_volume(out_dir / "labels.npy")

    return all_files
