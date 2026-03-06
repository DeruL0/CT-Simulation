"""
Simulation and export action handlers for MainWindow.
"""

from PySide6.QtWidgets import QMessageBox, QFileDialog

from core import ScientificData, SimulationTimingResult
from simulation.volume import CTVolume

from .workers import SimulationWorker, ExportWorker


def start_simulation(window) -> None:
    """Kick off simulation worker for the active mesh/configuration."""
    mesh_data = window._data_manager.mesh_data
    if mesh_data is None or mesh_data.primary_data is None:
        return

    sim_config = window._simulation_config_builder.build()

    if window._worker is not None and window._worker.isRunning():
        QMessageBox.warning(window, "Task Running", "Please wait for the current task to complete.")
        return

    window._simulate_btn.setEnabled(False)
    window._export_btn.setEnabled(False)
    window._status_bar.showMessage("Running simulation...")
    window._progress_dialog = window._create_progress_dialog("Running CT Simulation...")

    window._worker = SimulationWorker(
        mesh=mesh_data.primary_data,
        mesh_data=mesh_data,
        voxel_size=sim_config.voxel_size,
        fill_interior=sim_config.fill_interior,
        num_projections=sim_config.num_projections,
        add_noise=sim_config.add_noise,
        noise_level=sim_config.noise_level,
        material=sim_config.material,
        fast_mode=sim_config.fast_mode,
        memory_limit_gb=sim_config.memory_limit_gb,
        use_gpu=sim_config.use_gpu,
        physics_mode=sim_config.physics_mode,
        physics_kvp=sim_config.physics_kvp,
        physics_tube_current=sim_config.physics_tube_current,
        physics_filtration=sim_config.physics_filtration,
        physics_energy_bins=sim_config.physics_energy_bins,
        physics_exposure_multiplier=sim_config.physics_exposure_multiplier,
        voxel_data=window._data_manager.voxel_data,
        structure_config=sim_config.structure_config,
        compression_config=sim_config.compression_config,
    )

    window._worker.progress.connect(window._on_sim_progress)
    window._worker.finished.connect(window._on_sim_finished)
    window._worker.error.connect(window._on_sim_error)
    window._worker.start()


def handle_simulation_finished(
    window,
    ct_result: object,
    timing_info: SimulationTimingResult,
    compression_results: list,
    annotations=None,
) -> None:
    """Apply completed simulation outputs to UI/data model and show summary dialog."""
    if isinstance(ct_result, ScientificData):
        ct_data = ct_result
        ct_volume = ct_result.primary_data
    elif isinstance(ct_result, CTVolume):
        ct_volume = ct_result
        ct_data = ScientificData(
            primary_data=ct_volume,
            secondary_data={},
            spatial_info={
                "voxel_size_mm": ct_volume.voxel_size,
                "origin": ct_volume.origin.copy(),
            },
            metadata={"stage": "simulated"},
        )
    else:
        raise TypeError(f"Unsupported simulation result type: {type(ct_result).__name__}")

    if not isinstance(ct_volume, CTVolume):
        raise TypeError("Simulation result ScientificData.primary_data must be CTVolume.")

    window._compression_results = compression_results
    window._initial_annotations = annotations

    window._close_progress_dialog()
    window._simulate_btn.setEnabled(True)
    window._export_btn.setEnabled(True)

    if compression_results and len(compression_results) > 1:
        final_result = compression_results[-1]
        threshold = window._auto_isosurface_threshold(final_result.volume)
        window._series_threshold = float(threshold)

        volumes = [r.volume for r in compression_results]
        window._viewer_panel.set_volume_series(volumes)
        window._compression_panel.set_results(compression_results)

        final_ct = CTVolume(
            data=final_result.volume,
            voxel_size=final_result.voxel_size,
            origin=ct_volume.origin.copy(),
        )
        window._data_manager.set_ct_data(
            ScientificData(
                primary_data=final_ct,
                secondary_data={
                    "compression_step": final_result.step_index,
                    "compression_ratio": final_result.compression_ratio,
                },
                spatial_info={
                    "voxel_size_mm": final_ct.voxel_size,
                    "origin": final_ct.origin.copy(),
                },
                metadata=dict(ct_data.metadata, stage="compression_final"),
            )
        )
        window._viewer_3d_panel.set_ct_volume(
            final_result.volume,
            final_result.voxel_size,
            threshold=threshold,
            preserve_threshold=False,
        )
        window._status_bar.showMessage(f"Simulation complete with {len(compression_results)} compression steps")
    else:
        window._data_manager.set_ct_data(ct_data)
        window._viewer_panel.clear_volume_series()
        window._viewer_panel.set_volume(ct_volume.data)
        window._compression_panel.clear_results()

        threshold = window._auto_isosurface_threshold(ct_volume.data)
        window._series_threshold = float(threshold)
        window._viewer_3d_panel.set_ct_volume(
            ct_volume.data,
            ct_volume.voxel_size,
            threshold=threshold,
            preserve_threshold=False,
        )
        window._status_bar.showMessage(
            f"Simulation complete: {ct_volume.num_slices} slices, {ct_volume.voxel_size:.2f} mm/voxel"
        )

    if timing_info.physics_mode:
        mode_str = "Physics Mode (Polychromatic)"
    elif timing_info.use_gpu:
        mode_str = "GPU"
    else:
        mode_str = "CPU"
    if timing_info.fast_mode:
        mode_str += " (Fast Mode)"

    timing_msg = (
        "Successfully generated CT volume.\n\n"
        f"Dimensions: {ct_volume.shape}\n"
        f"Voxel Size: {ct_volume.voxel_size:.2f} mm\n\n"
        f"--- Timing ({mode_str}) ---\n"
        f"Voxelization: {timing_info.voxelization_time:.2f}s\n"
        f"Simulation: {timing_info.simulation_time:.2f}s\n"
    )

    if timing_info.compression_time is not None:
        timing_msg += f"Compression: {timing_info.compression_time:.2f}s\n"
    timing_msg += f"Total: {timing_info.total_time:.2f}s\n"

    if compression_results and len(compression_results) > 1:
        timing_msg += "\n--- Compression ---\n"
        timing_msg += f"Steps: {len(compression_results)}\n"
        timing_msg += "Use slider in Compression Panel to view steps.\n"

    if timing_info.gpu_timing is not None:
        gt = timing_info.gpu_timing
        total = gt.total if gt.total > 0 else 1.0
        slices = gt.slices if gt.slices > 0 else 1
        timing_msg += (
            "\n--- GPU Details ---\n"
            f"Transfer to GPU: {gt.transfer_to_gpu:.2f}s ({gt.transfer_to_gpu/total*100:.1f}%)\n"
            f"Radon: {gt.radon:.2f}s ({gt.radon/total*100:.1f}%)\n"
            f"IRadon: {gt.iradon:.2f}s ({gt.iradon/total*100:.1f}%)\n"
            f"Transfer to CPU: {gt.transfer_to_cpu:.2f}s ({gt.transfer_to_cpu/total*100:.1f}%)\n"
            f"Per-slice: {gt.total/slices*1000:.1f}ms"
        )

    QMessageBox.information(window, "Simulation Complete", timing_msg)


def start_export(window) -> None:
    """Kick off export worker for current CT volume or compression series."""
    has_series = bool(window._compression_results and len(window._compression_results) > 1)
    if not has_series and not window._data_manager.has_ct_volume:
        QMessageBox.warning(window, "No Data", "Please run a simulation first before exporting.")
        return

    if window._worker is not None and window._worker.isRunning():
        QMessageBox.warning(window, "Task Running", "Please wait for the current task to complete.")
        return

    output_dir = QFileDialog.getExistingDirectory(window, "Select Output Directory")
    if not output_dir:
        return

    window._simulate_btn.setEnabled(False)
    window._export_btn.setEnabled(False)
    window._status_bar.showMessage("Exporting DICOM...")
    window._progress_dialog = window._create_progress_dialog("Exporting DICOM Series...")

    if window._compression_results and len(window._compression_results) > 1:
        data_to_export = window._compression_results
    else:
        data_to_export = window._data_manager.ct_volume

    window._worker = ExportWorker(
        volume_or_list=data_to_export,
        output_dir=output_dir,
        window_center=window._viewer_panel.window_center,
        window_width=window._viewer_panel.window_width,
        initial_annotations=window._initial_annotations,
    )

    window._worker.progress.connect(window._on_export_progress)
    window._worker.finished.connect(window._on_export_finished)
    window._worker.error.connect(window._on_export_error)
    window._worker.start()
