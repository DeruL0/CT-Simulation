"""
Main Window

The main application window for CT Simulation Software.
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QGroupBox, QPushButton, QLabel,
    QFileDialog, QProgressBar, QStatusBar, QMenuBar,
    QMenu, QMessageBox, QApplication, QTabWidget,
    QScrollArea, QFrame, QDockWidget, QProgressDialog
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QAction

from .style import ScientificStyle
from .panels import STLPanel, ParamsPanel, ViewerPanel
from .panels.structure_panel import StructurePanel
from .workers import SimulationWorker, ExportWorker

from loaders.stl_loader import STLLoader
from simulation.ct_simulator import CTVolume
from visualization import MeshViewer


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        super().__init__()
        
        # Data management
        from core.data_manager import DataManager
        self._data_manager = DataManager(self)
        
        self._stl_loader: Optional[STLLoader] = None
        self._ct_volume: Optional[CTVolume] = None
        self._worker: Optional[QThread] = None
        self._progress_dialog: Optional[QProgressDialog] = None
        
        self._setup_ui()
        self._setup_menu()
        self._connect_signals()
    
    def _setup_ui(self) -> None:
        """Set up the main window UI."""
        self.setWindowTitle("CT Simulation Software")
        self.setMinimumSize(1200, 800)
        self.resize(1400, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)
        
        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Horizontal)
        
        # Left panel (controls)
        # Left panel controls container
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(4, 4, 4, 4)  # Add small margin inside scroll area
        
        # STL Panel
        self._stl_panel = STLPanel()
        controls_layout.addWidget(self._stl_panel)
        
        # Parameters Panel
        self._params_panel = ParamsPanel()
        controls_layout.addWidget(self._params_panel)
        
        # Structure Panel (Industrial/Manual Modifiers)
        self._structure_panel = StructurePanel(self._data_manager)
        controls_layout.addWidget(self._structure_panel)
        
        # Action buttons
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)
        
        self._simulate_btn = QPushButton("Run Simulation")
        self._simulate_btn.setEnabled(False)
        self._simulate_btn.clicked.connect(self._on_simulate)
        actions_layout.addWidget(self._simulate_btn)
        
        self._export_btn = QPushButton("Export DICOM")
        self._export_btn.setObjectName("secondaryButton")
        self._export_btn.setEnabled(False)
        self._export_btn.clicked.connect(self._on_export)
        actions_layout.addWidget(self._export_btn)

        self._reset_stl_btn = QPushButton("Reset to STL")
        self._reset_stl_btn.setObjectName("secondaryButton")
        self._reset_stl_btn.setEnabled(False)
        self._reset_stl_btn.clicked.connect(self._on_reset_stl)
        actions_layout.addWidget(self._reset_stl_btn)
        
        controls_layout.addWidget(actions_group)
        controls_layout.addStretch()
        
        # Scroll Area for left panel
        scroll_area = QScrollArea()
        scroll_area.setWidget(controls_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setMinimumWidth(320)  # Slightly wider for structure controls
        scroll_area.setMaximumWidth(450)
        
        splitter.addWidget(scroll_area)
        
        # Right panel (viewers in tabs)
        viewer_tabs = QTabWidget()
        
        # 3D Viewer tab
        self._viewer_3d_panel = MeshViewer()
        viewer_tabs.addTab(self._viewer_3d_panel, "3D View")
        
        # 2D CT Viewer tab
        self._viewer_panel = ViewerPanel()
        viewer_tabs.addTab(self._viewer_panel, "2D Slices")
        
        splitter.addWidget(viewer_tabs)
        
        # Prevent collapsing
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setStretchFactor(1, 1)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 900])
        
        main_layout.addWidget(splitter)
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        
        self._progress_bar = QProgressBar()
        self._progress_bar.setMaximumWidth(200)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)
        
        self._status_bar.showMessage("Ready")
    
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
        self._stl_panel.stl_loaded.connect(self._on_stl_loaded)
        self._params_panel.params_changed.connect(self._on_params_changed)
    
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
        self._simulate_btn.setEnabled(self._stl_loader is not None)
        self._export_btn.setEnabled(self._ct_volume is not None)
        self._status_bar.showMessage(f"Error: {message}")
        QMessageBox.critical(self, title, f"An error occurred:\n\n{message}")
    
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
            self._stl_panel.load_stl(file_path)
    
    def _on_stl_loaded(self, loader: STLLoader) -> None:
        """Handle STL file loaded."""
        self._stl_loader = loader
        self._simulate_btn.setEnabled(True)
        self._reset_stl_btn.setEnabled(True)
        
        # Update DataManager for StructurePanel
        self._data_manager._stl_loader = loader
        self._data_manager.stl_loaded.emit(loader)
        
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
    
    def _on_params_changed(self) -> None:
        """Handle parameter changes - update memory estimate."""
        if self._stl_loader is not None and self._stl_loader.info is not None:
            self._params_panel.update_memory_estimate(tuple(self._stl_loader.info.dimensions), self._stl_loader.info.num_faces)
    
    def _on_simulate(self) -> None:
        """Run CT simulation."""
        if self._stl_loader is None or self._stl_loader.mesh is None:
            return
        
        # Check if a worker is already running
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(
                self,
                "Task Running",
                "Please wait for the current task to complete."
            )
            return
        
        # Disable controls during simulation
        self._simulate_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._status_bar.showMessage("Running simulation...")
        
        # Create progress dialog
        self._progress_dialog = self._create_progress_dialog("Running CT Simulation...")
        
        # Create worker
        self._worker = SimulationWorker(
            mesh=self._stl_loader.mesh,
            voxel_size=self._params_panel.voxel_size,
            fill_interior=self._params_panel.fill_interior,
            num_projections=self._params_panel.num_projections,
            add_noise=self._params_panel.add_noise,
            noise_level=self._params_panel.noise_level,
            material=self._stl_panel.selected_material,
            fast_mode=self._params_panel.fast_mode,
            memory_limit_gb=self._params_panel.memory_limit_gb,
            use_gpu=self._params_panel.use_gpu,
            physics_mode=self._params_panel.physics_mode,
            physics_kvp=self._params_panel.physics_kvp,
            physics_tube_current=self._params_panel.physics_tube_current,
            physics_filtration=self._params_panel.physics_filtration,
            physics_energy_bins=self._params_panel.physics_energy_bins,
            voxel_grid=self._data_manager.voxel_grid,  # Use pre-computed if available
            structure_config=self._structure_panel.get_active_config() # Pass active structure config
        )
        
        self._worker.progress.connect(self._on_sim_progress)
        self._worker.finished.connect(self._on_sim_finished)
        self._worker.error.connect(self._on_sim_error)
        self._worker.start()
    
    @Slot(float)
    def _on_sim_progress(self, progress: float) -> None:
        """Handle simulation progress update."""
        if self._progress_dialog:
            self._progress_dialog.setValue(int(progress * 100))
    
    @Slot(object, dict)
    def _on_sim_finished(self, ct_volume: CTVolume, timing_info: dict) -> None:
        """Handle simulation completed."""
        self._ct_volume = ct_volume
        
        self._close_progress_dialog()
        
        self._simulate_btn.setEnabled(True)
        self._export_btn.setEnabled(True)
        
        # Display in viewers
        self._viewer_panel.set_volume(ct_volume.data)
        
        # Update 3D view with CT isosurface
        threshold = 0.0  # Use water level as default threshold
        self._viewer_3d_panel.set_ct_volume(
            ct_volume.data, 
            ct_volume.voxel_size, 
            threshold=threshold
        )
        
        self._status_bar.showMessage(
            f"Simulation complete: {ct_volume.num_slices} slices, "
            f"{ct_volume.voxel_size:.2f} mm/voxel"
        )
        
        # Build detailed timing message
        if timing_info.get('physics_mode'):
            mode_str = "Physics Mode (Polychromatic)"
        elif timing_info['use_gpu']:
            mode_str = "GPU"
        else:
            mode_str = "CPU"
        if timing_info['fast_mode']:
            mode_str += " (Fast Mode)"
        
        timing_msg = (
            f"Successfully generated CT volume.\n\n"
            f"Dimensions: {ct_volume.shape}\n"
            f"Voxel Size: {ct_volume.voxel_size:.2f} mm\n\n"
            f"--- Timing ({mode_str}) ---\n"
            f"Voxelization: {timing_info['voxelization_time']:.2f}s\n"
            f"Simulation: {timing_info['simulation_time']:.2f}s\n"
            f"Total: {timing_info['total_time']:.2f}s\n"
        )
        
        # Add GPU-specific timing breakdown if available
        if timing_info.get('gpu_timing'):
            gt = timing_info['gpu_timing']
            timing_msg += (
                f"\n--- GPU Details ---\n"
                f"Transfer to GPU: {gt['transfer_to_gpu']:.2f}s ({gt['transfer_to_gpu']/gt['total']*100:.1f}%)\n"
                f"Radon: {gt['radon']:.2f}s ({gt['radon']/gt['total']*100:.1f}%)\n"
                f"IRadon: {gt['iradon']:.2f}s ({gt['iradon']/gt['total']*100:.1f}%)\n"
                f"Transfer to CPU: {gt['transfer_to_cpu']:.2f}s ({gt['transfer_to_cpu']/gt['total']*100:.1f}%)\n"
                f"Per-slice: {gt['total']/gt['slices']*1000:.1f}ms"
            )
        
        QMessageBox.information(
            self,
            "Simulation Complete",
            timing_msg
        )
    
    @Slot(str)
    def _on_sim_error(self, error_msg: str) -> None:
        """Handle simulation error."""
        self._show_error("Simulation Error", error_msg)
    
    def _on_export(self) -> None:
        """Export CT volume to DICOM."""
        if self._ct_volume is None:
            QMessageBox.warning(
                self,
                "No Data",
                "Please run a simulation first before exporting."
            )
            return
        
        # Check if a worker is already running
        if self._worker is not None and self._worker.isRunning():
            QMessageBox.warning(
                self,
                "Task Running",
                "Please wait for the current task to complete."
            )
            return
        
        # Select output directory
        output_dir = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory"
        )
        
        if not output_dir:
            return
        
        # Disable controls during export
        self._simulate_btn.setEnabled(False)
        self._export_btn.setEnabled(False)
        self._status_bar.showMessage("Exporting DICOM...")
        
        # Create progress dialog
        self._progress_dialog = self._create_progress_dialog("Exporting DICOM Series...")
        
        # Create worker
        self._worker = ExportWorker(
            ct_volume=self._ct_volume,
            output_dir=output_dir,
            window_center=self._viewer_panel.window_center,
            window_width=self._viewer_panel.window_width
        )
        
        self._worker.progress.connect(self._on_export_progress)
        self._worker.finished.connect(self._on_export_finished)
        self._worker.error.connect(self._on_export_error)
        self._worker.start()
    
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
        if self._stl_loader is not None and self._stl_loader.mesh is not None:
            self._viewer_3d_panel.set_mesh(self._stl_loader.mesh)
            
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
