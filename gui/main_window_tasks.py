"""
Simulation and export action handlers for MainWindow.
"""

from PySide6.QtWidgets import QMessageBox, QFileDialog

from simulation.volume import CTVolume

from .workers import SimulationWorker, ExportWorker


def start_simulation(window) -> None:
    """Kick off simulation worker for the active mesh/configuration."""
    stl_loader = window._data_manager.stl_loader
    if stl_loader is None or stl_loader.mesh is None:
        return

    if window._worker is not None and window._worker.isRunning():
        QMessageBox.warning(window, "Task Running", "Please wait for the current task to complete.")
        return

    window._simulate_btn.setEnabled(False)
    window._export_btn.setEnabled(False)
    window._status_bar.showMessage("Running simulation...")
    window._progress_dialog = window._create_progress_dialog("Running CT Simulation...")

    window._worker = SimulationWorker(
        mesh=stl_loader.mesh,
        voxel_size=window._params_panel.voxel_size,
        fill_interior=window._params_panel.fill_interior,
        num_projections=window._params_panel.num_projections,
        add_noise=window._params_panel.add_noise,
        noise_level=window._params_panel.noise_level,
        material=window._loader_panel.selected_material,
        fast_mode=window._params_panel.fast_mode,
        memory_limit_gb=window._params_panel.memory_limit_gb,
        use_gpu=window._params_panel.use_gpu,
        physics_mode=window._params_panel.physics_mode,
        physics_kvp=window._params_panel.physics_kvp,
        physics_tube_current=window._params_panel.physics_tube_current,
        physics_filtration=window._params_panel.physics_filtration,
        physics_energy_bins=window._params_panel.physics_energy_bins,
        physics_exposure_multiplier=window._params_panel.physics_exposure_multiplier,
        voxel_grid=window._data_manager.voxel_grid,
        structure_config=window._structure_panel.get_active_config(),
        compression_config=window._compression_panel.get_config() if window._compression_panel.is_enabled() else None,
    )

    window._worker.progress.connect(window._on_sim_progress)
    window._worker.finished.connect(window._on_sim_finished)
    window._worker.error.connect(window._on_sim_error)
    window._worker.start()


def handle_simulation_finished(window, ct_volume: CTVolume, timing_info: dict, compression_results: list, annotations=None) -> None:
    """Apply completed simulation outputs to UI/data model and show summary dialog."""
    window._compression_results = compression_results
    window._initial_annotations = annotations

    window._close_progress_dialog()
    window._simulate_btn.setEnabled(True)
    window._export_btn.setEnabled(True)

    if compression_results and len(compression_results) > 1:
        volumes = [r.volume for r in compression_results]
        window._viewer_panel.set_volume_series(volumes)
        window._compression_panel.set_results(compression_results)

        final_result = compression_results[-1]
        window._data_manager.set_ct_volume(
            CTVolume(
                data=final_result.volume,
                voxel_size=final_result.voxel_size,
                origin=ct_volume.origin.copy(),
            )
        )
        threshold = window._auto_isosurface_threshold(final_result.volume)
        window._viewer_3d_panel.set_ct_volume(
            final_result.volume,
            final_result.voxel_size,
            threshold=threshold,
        )
        window._status_bar.showMessage(f"Simulation complete with {len(compression_results)} compression steps")
    else:
        window._data_manager.set_ct_volume(ct_volume)
        window._viewer_panel.set_volume(ct_volume.data)
        window._compression_panel.clear_results()

        threshold = window._auto_isosurface_threshold(ct_volume.data)
        window._viewer_3d_panel.set_ct_volume(
            ct_volume.data,
            ct_volume.voxel_size,
            threshold=threshold,
        )
        window._status_bar.showMessage(
            f"Simulation complete: {ct_volume.num_slices} slices, {ct_volume.voxel_size:.2f} mm/voxel"
        )

    if timing_info.get("physics_mode"):
        mode_str = "Physics Mode (Polychromatic)"
    elif timing_info["use_gpu"]:
        mode_str = "GPU"
    else:
        mode_str = "CPU"
    if timing_info["fast_mode"]:
        mode_str += " (Fast Mode)"

    timing_msg = (
        "Successfully generated CT volume.\n\n"
        f"Dimensions: {ct_volume.shape}\n"
        f"Voxel Size: {ct_volume.voxel_size:.2f} mm\n\n"
        f"--- Timing ({mode_str}) ---\n"
        f"Voxelization: {timing_info['voxelization_time']:.2f}s\n"
        f"Simulation: {timing_info['simulation_time']:.2f}s\n"
    )

    if timing_info.get("compression_time"):
        timing_msg += f"Compression: {timing_info['compression_time']:.2f}s\n"
    timing_msg += f"Total: {timing_info['total_time']:.2f}s\n"

    if compression_results and len(compression_results) > 1:
        timing_msg += "\n--- Compression ---\n"
        timing_msg += f"Steps: {len(compression_results)}\n"
        timing_msg += "Use slider in Compression Panel to view steps.\n"

    if timing_info.get("gpu_timing"):
        gt = timing_info["gpu_timing"]
        timing_msg += (
            "\n--- GPU Details ---\n"
            f"Transfer to GPU: {gt['transfer_to_gpu']:.2f}s ({gt['transfer_to_gpu']/gt['total']*100:.1f}%)\n"
            f"Radon: {gt['radon']:.2f}s ({gt['radon']/gt['total']*100:.1f}%)\n"
            f"IRadon: {gt['iradon']:.2f}s ({gt['iradon']/gt['total']*100:.1f}%)\n"
            f"Transfer to CPU: {gt['transfer_to_cpu']:.2f}s ({gt['transfer_to_cpu']/gt['total']*100:.1f}%)\n"
            f"Per-slice: {gt['total']/gt['slices']*1000:.1f}ms"
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
