"""
Compression workflow helpers for background workers.
"""

import logging
import time
from typing import Optional, Tuple, List

import numpy as np

from simulation.mechanics import CompressionManager
from simulation.mechanics.manager import CompressionConfig, CompressionResult
from simulation.voxelizer import VoxelGrid


def run_compression_workflow(
    voxel_grid: VoxelGrid,
    ct_volume,
    simulator,
    physics_mode: bool,
    material,
    use_gpu: bool,
    compression_config: dict,
    tracker,
    initial_annotations=None,
) -> Tuple[List[CompressionResult], float]:
    """
    Run compression physics and per-step CT simulation workflow.

    Returns:
        Tuple of (compression_results, elapsed_seconds).
    """
    comp_start = time.perf_counter()
    cfg = compression_config
    manager = CompressionManager(use_gpu=use_gpu)

    comp_config = CompressionConfig(
        total_compression=cfg["total_compression"],
        num_steps=cfg["num_steps"],
        poisson_ratio=cfg["poisson_ratio"],
        axis=cfg.get("axis", "Z"),
        use_physics=cfg["mode"] == "physical",
        downsample_factor=cfg.get("downsample_factor", 4),
        solver_iterations=cfg.get("solver_iterations", 300),
    )

    voxel_data = voxel_grid.data.astype(np.float32)
    phys_cb = tracker.sub_progress(3)

    def phys_progress(progress_value, _status):
        phys_cb(progress_value)

    deformed_grids = manager.run_simulation(
        voxel_data,
        voxel_grid.voxel_size,
        comp_config,
        progress_callback=phys_progress,
        initial_annotations=initial_annotations,
    )
    tracker.end_phase()  # End physics phase

    logging.info("  Generated %s compressed voxel grids", len(deformed_grids))

    tracker.start_phase(4)  # Batch CT phase
    num_steps = len(deformed_grids)
    compression_results: List[CompressionResult] = []

    for i, deform_result in enumerate(deformed_grids):
        step_cb = tracker.sub_range(i / num_steps, (i + 1) / num_steps, 4)

        if deform_result.compression_ratio == 0.0 and ct_volume is not None:
            logging.info("  Skipping simulation for Step %s (0%% compression) - reusing initial CT", i)
            step_ct_data = ct_volume.data
            step_ct_voxel_size = ct_volume.voxel_size
            step_cb(1.0)
        else:
            logging.info("  CT simulation for step %s/%s...", i, num_steps - 1)

            deformed_grid = VoxelGrid(
                data=deform_result.volume,
                voxel_size=deform_result.voxel_size,
                origin=voxel_grid.origin,
            )

            if physics_mode:
                step_ct = simulator.simulate(
                    deformed_grid,
                    material=material,
                    density_scale=getattr(deform_result, "density_scale", 1.0),
                    progress_callback=step_cb,
                )
            else:
                step_ct = simulator.simulate(
                    deformed_grid,
                    material=material,
                    progress_callback=step_cb,
                )
            step_ct_data = step_ct.data
            step_ct_voxel_size = step_ct.voxel_size

        compression_results.append(
            CompressionResult(
                step_index=deform_result.step_index,
                compression_ratio=deform_result.compression_ratio,
                _volume=step_ct_data,
                voxel_size=step_ct_voxel_size,
                density_scale=getattr(deform_result, "density_scale", 1.0),
                annotations=deform_result.annotations,
            )
        )

    elapsed = time.perf_counter() - comp_start
    return compression_results, elapsed
