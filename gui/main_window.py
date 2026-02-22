"""
Main Window

The main application window for CT Simulation Software.
"""

from typing import Optional
import numpy as np

from PySide6.QtWidgets import (
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QProgressDialog,
)
from PySide6.QtCore import Qt, QThread, Slot
from PySide6.QtGui import QAction

from .main_window_ui import setup_main_window_ui
from .main_window_tasks import (
    start_simulation,
    handle_simulation_finished,
    start_export,
)

from loaders import MeshLoader as STLLoader
from simulation.volume import CTVolume


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Data management
        from .data_manager import DataManager
        self._data_manager = DataManager(self)
        
        self._compression_results: Optional[list] = None  # Store list of CompressionResult
        self._initial_annotations = None  # Store initial AnnotationSet for export
        self._worker: Optional[QThread] = None
        self._progress_dialog: Optional[QProgressDialog] = None
        
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
    
    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        setup_main_window_ui(self)
    
    def _setup_menu(self) -> None:
        """Set up the menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open STL...", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._on_open_stl)
        file_menu.addAction(open_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("Export DICOM...", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self._on_export)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        clear_cache_action = QAction("Clear Mesh Cache", self)
        clear_cache_action.triggered.connect(self._on_clear_cache)
        file_menu.addAction(clear_cache_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
    
    def _connect_signals(self) -> None:
        """Connect widget signals."""
        self._loader_panel.stl_loaded.connect(self._on_stl_loaded)
        self._loader_panel.mesh_scaled.connect(self._on_mesh_scaled)
        self._params_panel.params_changed.connect(self._on_params_changed)
        
        # Compression panel signals
        self._compression_panel.step_changed.connect(self._on_compression_step_changed)
    
    # ========== Helper Methods ==========
    
    def _create_progress_dialog(self, title: str) -> QProgressDialog:
        """Create and configure a modal progress dialog."""
        dialog = QProgressDialog(title, "Cancel", 0, 100, self)
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setAutoClose(False)
        dialog.setAutoReset(False)
        dialog.setCancelButton(None)  # Disable cancel (workers don't support interruption)
        dialog.show()
        return dialog
    
    def _close_progress_dialog(self) -> None:
        """Close and clean up the progress dialog."""
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
    
    def _show_error(self, title: str, message: str) -> None:
        """Display an error message and restore UI state."""
        self._close_progress_dialog()
        self._simulate_btn.setEnabled(self._data_manager.has_stl)
        self._export_btn.setEnabled(self._data_manager.has_ct_volume)
        self._status_bar.showMessage(f"Error: {message}")
        QMessageBox.critical(self, title, f"An error occurred:\n\n{message}")

    @staticmethod
    def _auto_isosurface_threshold(volume_data: np.ndarray) -> float:
        """Compute a robust default isosurface threshold from attenuation data."""
        finite = volume_data[np.isfinite(volume_data)]
        if finite.size == 0:
            return 0.0

        if finite.size > 2_000_000:
            step = max(1, finite.size // 2_000_000)
            finite = finite[::step]

        vmin = float(np.min(finite))
        vmax = float(np.max(finite))
        if vmax <= vmin:
            return vmin

        q10 = float(np.percentile(finite, 10.0))
        q90 = float(np.percentile(finite, 90.0))
        threshold = q10 + 0.25 * (q90 - q10)
        return max(threshold, vmin + 1e-4)
    
    def _on_clear_cache(self) -> None:
        """Handle File > Clear Mesh Cache."""
        count = STLLoader.clear_cache()
        QMessageBox.information(
            self,
            "Cache Cleared",
            f"Deleted {count} cached mesh file(s).\n\n"
            "Please reload your STL file to apply the latest normalization settings."
        )
        self._status_bar.showMessage(f"Cleared {count} cached files")
    
    def _on_open_stl(self) -> None:
        """Handle File > Open STL."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select STL File",
            "",
            "STL Files (*.stl);;All Files (*.*)"
        )
        
        if file_path:
            self._loader_panel.load_stl(file_path)
    
    def _on_stl_loaded(self, loader: STLLoader) -> None:
        """Handle STL file loaded."""
        self._data_manager.set_stl_loader(loader)
        self._simulate_btn.setEnabled(True)
        self._reset_stl_btn.setEnabled(True)
        self._compression_results = None
        self._initial_annotations = None
        
        # Display mesh in 3D viewer
        if loader.mesh is not None:
            self._viewer_3d_panel.set_mesh(loader.mesh)
        
        # Update memory estimation
        if loader.info is not None:
            self._params_panel.update_memory_estimate(tuple(loader.info.dimensions), loader.info.num_faces)
        
        self._status_bar.showMessage(
            f"Loaded: {loader.filepath.name} "
            f"({loader.info.num_faces:,} faces)"
        )
    
    def _on_mesh_scaled(self, scale_factor: float) -> None:
        """Handle mesh scaling."""
        # Reset data when scaled to avoid stale caches
        self._data_manager.clear_ct_volume()
        self._compression_results = None
        self._initial_annotations = None
        self._data_manager.clear_voxel_grid()
        self._compression_panel.clear_results()
        if hasattr(self, '_structure_panel'):
            self._structure_panel.reset_structure()
        
        # Reset viewers
        stl_loader = self._data_manager.stl_loader
        if stl_loader and stl_loader.mesh:
            self._viewer_3d_panel.set_mesh(stl_loader.mesh)
        
        # Update memory estimate with new dimensions
        if stl_loader is not None and stl_loader.info:
             self._params_panel.update_memory_estimate(
                 tuple(stl_loader.info.dimensions),
                 stl_loader.info.num_faces
             )
        
        self._status_bar.showMessage(f"Model scaled by {scale_factor:.2f}×. Data reset.")
        self._export_btn.setEnabled(False)
    
    def _on_params_changed(self) -> None:
        """Handle parameter changes - update memory estimate."""
        stl_info = self._data_manager.stl_info
        if stl_info is not None:
            self._params_panel.update_memory_estimate(tuple(stl_info.dimensions), stl_info.num_faces)
    
    def _on_simulate(self) -> None:
        """Run CT simulation."""
        start_simulation(self)
    
    @Slot(float)
    def _on_sim_progress(self, progress: float) -> None:
        """Handle simulation progress update."""
        if self._progress_dialog:
            self._progress_dialog.setValue(int(progress * 100))
    
    @Slot(object, dict, list, object)
    def _on_sim_finished(self, ct_volume: CTVolume, timing_info: dict, compression_results: list, annotations=None) -> None:
        """Handle simulation completed."""
        handle_simulation_finished(self, ct_volume, timing_info, compression_results, annotations)
    
    @Slot(str)
    def _on_sim_error(self, error_msg: str) -> None:
        """Handle simulation error."""
        self._show_error("Simulation Error", error_msg)
    
    def _on_export(self) -> None:
        """Export CT volume to DICOM."""
        start_export(self)
    
    @Slot(float)
    def _on_export_progress(self, progress: float) -> None:
        """Handle export progress update."""
        if self._progress_dialog:
            self._progress_dialog.setValue(int(progress * 100))
    
    @Slot(list)
    def _on_export_finished(self, files: list) -> None:
        """Handle export completed."""
        self._close_progress_dialog()
        
        self._simulate_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        
        self._status_bar.showMessage(
            f"Exported {len(files)} DICOM files"
        )
        
        QMessageBox.information(
            self,
            "Export Complete",
            f"Successfully exported {len(files)} DICOM files."
        )
    
    @Slot(str)
    def _on_export_error(self, error_msg: str) -> None:
        """Handle export error."""
        self._show_error("Export Error", error_msg)

    def _on_reset_stl(self) -> None:
        """Reset 3D view to show STL mesh and reset structure."""
        stl_loader = self._data_manager.stl_loader
        if stl_loader is not None and stl_loader.mesh is not None:
            self._viewer_3d_panel.set_mesh(stl_loader.mesh)
            
            # Reset structure generation panel as well
            if hasattr(self, '_structure_panel'):
                self._structure_panel.reset_structure()
                
            self._status_bar.showMessage("Reset view and structure to original STL")
    
    def _on_about(self) -> None:
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About CT Simulation Software",
            "<h3>CT Simulation Software</h3>"
            "<p>Version 1.0</p>"
            "<p>A scientific tool for generating simulated CT images "
            "from 3D STL models.</p>"
            "<p>Features:</p>"
            "<ul>"
            "<li>STL model import</li>"
            "<li>Voxelization with configurable resolution</li>"
            "<li>CT projection simulation using Radon transform</li>"
            "<li>Filtered back projection reconstruction</li>"
            "<li>DICOM export with proper metadata</li>"
            "</ul>"
        )
    
    def _on_compression_step_changed(self, step_index: int, volume_data) -> None:
        """Handle compression step slider change - update 3D and 2D viewers."""
        if volume_data is None:
            return
        
        # Update 3D viewer with the selected step's volume
        voxel_size = self._data_manager.voxel_grid.voxel_size if self._data_manager.has_voxel_grid else 0.5
        current_ct = self._data_manager.ct_volume
        origin = current_ct.origin.copy() if current_ct is not None else np.zeros(3, dtype=np.float32)
        self._data_manager.set_ct_volume(
            CTVolume(
                data=volume_data,
                voxel_size=voxel_size,
                origin=origin,
            )
        )
        threshold = self._auto_isosurface_threshold(volume_data)
        
        self._viewer_3d_panel.set_ct_volume(
            volume_data,
            voxel_size,
            threshold=threshold
        )
        
        # Update 2D viewer with the selected step
        self._viewer_panel.set_current_step(step_index)
        
        self._status_bar.showMessage(f"Viewing compression step {step_index}")


