"""
Structure Panel

Industrial structure generation panel for CT simulation.
Provides automated lattice and defect generation with density/complexity controls.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QPushButton, QSlider,
    QTabWidget, QFormLayout, QMessageBox, QProgressDialog,
    QStackedWidget, QButtonGroup, QSizePolicy
)
from PySide6.QtCore import Qt, Signal
import logging

from simulation.voxelizer import Voxelizer, VoxelGrid
from simulation.structures import (
    StructureModifier, LatticeType, LatticeConfig,
    DefectShape, DefectConfig
)


class StructurePanel(QWidget):
    """
    Panel for generating industrial structures in the voxel grid.
    
    Provides tabs for:
    - Industrial Lattice (TPMS with density/complexity control)
    - Industrial Defects (random voids with porosity/size control)
    - Manual Modifiers (sphere/cylinder voids)
    """
    
    # Signals
    structure_applied = Signal(object)  # Emits modified VoxelGrid
    
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self._data_manager = data_manager
        self._modifier: StructureModifier = None
        self._worker: Optional[QThread] = None
        self._progress_dialog: Optional[QProgressDialog] = None
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main container group for everything
        struct_group = QGroupBox("Structure Generation")
        struct_group.setCheckable(True)
        struct_group.setChecked(False) # Default to off as per user request
        struct_layout = QVBoxLayout(struct_group)
        struct_layout.setSpacing(10)
        
        # --- 1. Global Settings (Top) ---
        # Removed voxel size spinner as it is now synced with simulation
        
        # Divider line
        from PySide6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        struct_layout.addWidget(line)

        # ... (rest of UI setup) ...




        
        # --- 2. Navigation Buttons ---
        nav_layout = QHBoxLayout()
        self._btn_lattice = QPushButton("Lattice")
        self._btn_defects = QPushButton("Random Defects")
        self._btn_manual = QPushButton("Manual Modifiers")
        
        self._btn_lattice.setCheckable(True)
        self._btn_defects.setCheckable(True)
        self._btn_manual.setCheckable(True)
        
        # Styling for cleaner look
        btn_style = """
            QPushButton {
                padding: 6px;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
            }
            QPushButton:checked {
                background-color: #0078d7;
                color: white;
                border: 1px solid #005a9e;
            }
        """
        self._btn_lattice.setStyleSheet(btn_style)
        self._btn_defects.setStyleSheet(btn_style)
        self._btn_manual.setStyleSheet(btn_style)
        
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_group.addButton(self._btn_lattice, 0)
        self._nav_group.addButton(self._btn_defects, 1)
        self._nav_group.addButton(self._btn_manual, 2)
        
        nav_layout.addWidget(self._btn_lattice)
        nav_layout.addWidget(self._btn_defects)
        nav_layout.addWidget(self._btn_manual)
        struct_layout.addLayout(nav_layout)
        
        # --- 3. Stacked Pages ---
        self._stack = QStackedWidget()
        
        self._lattice_page = self._create_lattice_page()
        self._defects_page = self._create_defects_page()
        self._manual_page = self._create_manual_page()
        
        self._stack.addWidget(self._lattice_page)
        self._stack.addWidget(self._defects_page)
        self._stack.addWidget(self._manual_page)
        
        struct_layout.addWidget(self._stack)
        
        # Add the main group to the widget layout
        layout.addWidget(struct_group)
        
        # Status Label (Outside hierarchy, or at very bottom)
        self._status_label = QLabel("Structure generation will be applied during simulation.")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        
        # IMPORTANT: Adaptive Vertical sizing
        # Add stretch at the end so the groupbox only takes necessary space 
        # and pushes up against top of the panel container
        layout.addStretch()
        
        # Connect buttons to stack switching
        self._nav_group.idClicked.connect(self._stack.setCurrentIndex)
        
        # Handle adaptive sizing of stack
        self._stack.currentChanged.connect(self._update_stack_sizing)
        
        # Default Reset
        self._btn_lattice.setChecked(True)
        self._update_stack_sizing(0)  # Initial size set
        
        # Store group for enabling/disabling
        self._struct_group = struct_group
        
        # Enable by default since we don't need to wait for STL load to configure
        self._set_controls_enabled(True)

    def get_active_config(self):
        """
        Get the configuration for the currently active structure generation mode.
        Returns:
            (method_name, config_object) or None if Disabled/Manual/None selected.
        """
        if not self._struct_group.isChecked():
            return None
        
        idx = self._stack.currentIndex()
        
        if idx == 0:  # Lattice
            config = LatticeConfig(
                lattice_type=self._lattice_type_combo.currentData(),
                density_percent=self._lattice_density_spin.value(),
                cell_size_mm=self._cell_size_spin.value(),
                preserve_shell=self._lattice_preserve_shell_check.isChecked(),
                shell_thickness_mm=self._lattice_shell_thickness_spin.value()
            )
            return ("generate_lattice", config)
            
        elif idx == 1:  # Defects
            config = DefectConfig(
                shape=self._defect_shape_combo.currentData(),
                density_percent=self._porosity_spin.value(),
                size_mean_mm=self._pore_size_spin.value(),
                size_std_mm=self._pore_std_spin.value(),
                preserve_shell=self._defects_preserve_shell_check.isChecked(),
                shell_thickness_mm=self._defects_shell_thickness_spin.value()
            )
            return ("generate_random_voids", config)
            
        return None  # Manual mode not handled via auto-generation pipeline

    def _update_stack_sizing(self, index):
        """Update stack widget size policy to fit current page."""
        for i in range(self._stack.count()):
            widget = self._stack.widget(i)
            if i == index:
                widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                widget.updateGeometry()
            else:
                widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        # Trigger layout update
        self._stack.adjustSize()

    def _create_lattice_page(self) -> QWidget:
        """Create the Industrial Lattice page."""
        widget = QWidget()
        # Main layout for the page with stretch at bottom
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Form layout for controls
        form_layout = QFormLayout()
        
        # Lattice type
        self._lattice_type_combo = QComboBox()
        for lt in LatticeType:
            self._lattice_type_combo.addItem(lt.value.replace("_", " ").title(), lt)
        form_layout.addRow("Type:", self._lattice_type_combo)
        
        # Cell size (complexity)
        self._cell_size_spin = QDoubleSpinBox()
        self._cell_size_spin.setRange(1.0, 50.0)
        self._cell_size_spin.setValue(5.0)
        self._cell_size_spin.setSuffix(" mm")
        form_layout.addRow("Cell Size:", self._cell_size_spin)
        
        # Density
        density_widget = QWidget()
        density_layout = QHBoxLayout(density_widget)
        density_layout.setContentsMargins(0, 0, 0, 0)
        
        self._lattice_density_slider = QSlider(Qt.Horizontal)
        self._lattice_density_slider.setRange(5, 95)
        self._lattice_density_slider.setValue(30)
        self._lattice_density_slider.valueChanged.connect(
            lambda v: self._lattice_density_spin.setValue(v)
        )
        
        self._lattice_density_spin = QSpinBox()
        self._lattice_density_spin.setRange(5, 95)
        self._lattice_density_spin.setValue(30)
        self._lattice_density_spin.setSuffix(" %")
        self._lattice_density_spin.valueChanged.connect(
            lambda v: self._lattice_density_slider.setValue(v)
        )
        
        density_layout.addWidget(self._lattice_density_slider)
        density_layout.addWidget(self._lattice_density_spin)
        form_layout.addRow("Density:", density_widget)
        
        # Shell Preservation
        from PySide6.QtWidgets import QCheckBox
        self._lattice_preserve_shell_check = QCheckBox("Preserve Outer Shell")
        form_layout.addRow("", self._lattice_preserve_shell_check)
        
        self._lattice_shell_thickness_spin = QDoubleSpinBox()
        self._lattice_shell_thickness_spin.setRange(0.1, 20.0)
        self._lattice_shell_thickness_spin.setValue(1.0)
        self._lattice_shell_thickness_spin.setSuffix(" mm")
        self._lattice_shell_thickness_spin.setEnabled(False)
        form_layout.addRow("Shell Thickness:", self._lattice_shell_thickness_spin)
        
        self._lattice_preserve_shell_check.toggled.connect(
            self._lattice_shell_thickness_spin.setEnabled
        )
        
        main_layout.addLayout(form_layout)
        
        # Removed Generate button
        
        main_layout.addStretch()
        
        return widget
    
    def _create_defects_page(self) -> QWidget:
        """Create the Industrial Defects page."""
        widget = QWidget()
        # Main layout for the page with stretch at bottom
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Form layout for controls
        form_layout = QFormLayout()
        
        # Defect shape
        self._defect_shape_combo = QComboBox()
        for ds in DefectShape:
            self._defect_shape_combo.addItem(ds.value.title(), ds)
        form_layout.addRow("Shape:", self._defect_shape_combo)
        
        # Target porosity (density)
        porosity_widget = QWidget()
        porosity_layout = QHBoxLayout(porosity_widget)
        porosity_layout.setContentsMargins(0, 0, 0, 0)
        
        self._porosity_slider = QSlider(Qt.Horizontal)
        self._porosity_slider.setRange(1, 50)
        self._porosity_slider.setValue(5)
        self._porosity_slider.valueChanged.connect(
            lambda v: self._porosity_spin.setValue(v)
        )
        
        self._porosity_spin = QSpinBox()
        self._porosity_spin.setRange(1, 50)
        self._porosity_spin.setValue(5)
        self._porosity_spin.setSuffix(" %")
        self._porosity_spin.valueChanged.connect(
            lambda v: self._porosity_slider.setValue(v)
        )
        
        porosity_layout.addWidget(self._porosity_slider)
        porosity_layout.addWidget(self._porosity_spin)
        form_layout.addRow("Porosity:", porosity_widget)
        
        # Pore size (complexity)
        self._pore_size_spin = QDoubleSpinBox()
        self._pore_size_spin.setRange(0.5, 20.0)
        self._pore_size_spin.setValue(2.0)
        self._pore_size_spin.setSuffix(" mm")
        form_layout.addRow("Mean Size:", self._pore_size_spin)
        
        self._pore_std_spin = QDoubleSpinBox()
        self._pore_std_spin.setRange(0.0, 10.0)
        self._pore_std_spin.setValue(0.5)
        self._pore_std_spin.setSuffix(" mm")
        form_layout.addRow("Size Std:", self._pore_std_spin)
        
        # Shell Preservation
        from PySide6.QtWidgets import QCheckBox
        self._defects_preserve_shell_check = QCheckBox("Preserve Outer Shell")
        form_layout.addRow("", self._defects_preserve_shell_check)
        
        self._defects_shell_thickness_spin = QDoubleSpinBox()
        self._defects_shell_thickness_spin.setRange(0.1, 20.0)
        self._defects_shell_thickness_spin.setValue(1.0)
        self._defects_shell_thickness_spin.setSuffix(" mm")
        self._defects_shell_thickness_spin.setEnabled(False)
        form_layout.addRow("Shell Thickness:", self._defects_shell_thickness_spin)
        
        self._defects_preserve_shell_check.toggled.connect(
            self._defects_shell_thickness_spin.setEnabled
        )
        
        main_layout.addLayout(form_layout)
        
        # Removed Generate button
        
        main_layout.addStretch()
        
        return widget
    
    def _create_manual_page(self) -> QWidget:
        """Create the Manual Modifiers page."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Sphere void
        sphere_group = QGroupBox("Add Sphere")
        sphere_layout = QFormLayout(sphere_group)
        
        self._sphere_x = QDoubleSpinBox()
        self._sphere_x.setRange(-1000, 1000)
        self._sphere_x.setSuffix(" mm")
        sphere_layout.addRow("X:", self._sphere_x)
        
        self._sphere_y = QDoubleSpinBox()
        self._sphere_y.setRange(-1000, 1000)
        self._sphere_y.setSuffix(" mm")
        sphere_layout.addRow("Y:", self._sphere_y)
        
        self._sphere_z = QDoubleSpinBox()
        self._sphere_z.setRange(-1000, 1000)
        self._sphere_z.setSuffix(" mm")
        sphere_layout.addRow("Z:", self._sphere_z)
        
        self._sphere_r = QDoubleSpinBox()
        self._sphere_r.setRange(0.1, 100)
        self._sphere_r.setValue(5.0)
        self._sphere_r.setSuffix(" mm")
        sphere_layout.addRow("R:", self._sphere_r)
        
        self._add_sphere_btn = QPushButton("Add Sphere")
        self._add_sphere_btn.clicked.connect(self._on_add_sphere)
        sphere_layout.addRow(self._add_sphere_btn)
        
        layout.addWidget(sphere_group)
        
        # Cylinder void
        cyl_group = QGroupBox("Add Cylinder")
        cyl_layout = QFormLayout(cyl_group)
        
        self._cyl_x1 = QDoubleSpinBox()
        self._cyl_x1.setRange(-1000, 1000)
        self._cyl_x1.setSuffix(" mm")
        cyl_layout.addRow("Start X:", self._cyl_x1)
        
        self._cyl_y1 = QDoubleSpinBox()
        self._cyl_y1.setRange(-1000, 1000)
        self._cyl_y1.setSuffix(" mm")
        cyl_layout.addRow("Start Y:", self._cyl_y1)
        
        self._cyl_z1 = QDoubleSpinBox()
        self._cyl_z1.setRange(-1000, 1000)
        self._cyl_z1.setSuffix(" mm")
        cyl_layout.addRow("Start Z:", self._cyl_z1)
        
        self._cyl_x2 = QDoubleSpinBox()
        self._cyl_x2.setRange(-1000, 1000)
        self._cyl_x2.setValue(10)
        self._cyl_x2.setSuffix(" mm")
        cyl_layout.addRow("End X:", self._cyl_x2)
        
        self._cyl_y2 = QDoubleSpinBox()
        self._cyl_y2.setRange(-1000, 1000)
        self._cyl_y2.setSuffix(" mm")
        cyl_layout.addRow("End Y:", self._cyl_y2)
        
        self._cyl_z2 = QDoubleSpinBox()
        self._cyl_z2.setRange(-1000, 1000)
        self._cyl_z2.setSuffix(" mm")
        cyl_layout.addRow("End Z:", self._cyl_z2)
        
        self._cyl_r = QDoubleSpinBox()
        self._cyl_r.setRange(0.1, 50)
        self._cyl_r.setValue(2.0)
        self._cyl_r.setSuffix(" mm")
        cyl_layout.addRow("Radius:", self._cyl_r)
        
        self._add_cyl_btn = QPushButton("Add Cylinder")
        self._add_cyl_btn.clicked.connect(self._on_add_cylinder)
        cyl_layout.addRow(self._add_cyl_btn)
        
        layout.addWidget(cyl_group)
        layout.addStretch()
        
        return widget

    def _connect_signals(self):
        self._data_manager.stl_loaded.connect(self._on_stl_loaded)
        self._data_manager.voxel_grid_changed.connect(self._on_voxel_grid_changed)
    
    def _set_controls_enabled(self, enabled: bool):
        self._struct_group.setEnabled(enabled)
    
    def _on_stl_loaded(self, loader):
        """Handle new STL loaded - reset voxel grid."""
        self._modifier = None
        self._set_controls_enabled(True) # Enable controls immediately for auto-init
        self._status_label.setText("STL loaded. Ready to generate.")
    
    def _on_voxel_grid_changed(self, voxel_grid):
        """Handle voxel grid changes."""
        if voxel_grid is not None:
            self._modifier = StructureModifier(voxel_grid)
            self._set_controls_enabled(True)
            shape = voxel_grid.shape
            self._status_label.setText(
                f"Grid: {shape[0]}×{shape[1]}×{shape[2]} "
                f"({voxel_grid.voxel_size:.2f}mm/voxel)"
            )
        else:
            self._modifier = None
            # Do NOT disable controls if STL is loaded, as we want to allow re-initialization
            if self._data_manager.has_stl:
                 self._set_controls_enabled(True)
                 self._status_label.setText("STL loaded. Ready to generate.")
            else:
                 self._set_controls_enabled(False)
                 self._status_label.setText("No voxel grid initialized")
    
    def _run_worker(self, method_name: str, config: object, message: str):
        """Run structure generation in background."""
        if not self._data_manager.has_stl:
             QMessageBox.warning(self, "No STL", "Please load an STL file first.")
             return

        from gui.workers import StructureWorker
        
        self._progress_dialog = QProgressDialog(message, "Cancel", 0, 100, self)
        self._progress_dialog.setWindowModality(Qt.WindowModal)
        self._progress_dialog.setMinimumDuration(0)
        self._progress_dialog.setValue(0)
        
        # Pass mesh and voxel size for auto-initialization if needed
        mesh = self._data_manager.mesh
        voxel_size = self._voxel_size_spin.value()
        
        self._worker = StructureWorker(
            modifier=self._modifier, 
            method_name=method_name, 
            config=config,
            mesh=mesh,
            voxel_size=voxel_size
        )
        self._worker.progress.connect(
            lambda p: self._progress_dialog.setValue(int(p * 100))
        )
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        
        self._worker.start()

    def _on_worker_finished(self, voxel_grid):
        """Handle worker completion."""
        # Update modifier with the (possibly newly created) grid
        if self._modifier is None:
             self._modifier = StructureModifier(voxel_grid)
        else:
             self._modifier.grid = voxel_grid
             
        self._data_manager.set_voxel_grid(voxel_grid)
        self.structure_applied.emit(voxel_grid)
        self._cleanup_worker()
        
        shape = voxel_grid.shape
        self._status_label.setText(
            f"Grid: {shape[0]}×{shape[1]}×{shape[2]} "
            f"({voxel_grid.voxel_size:.2f}mm/voxel)"
        )
        
        QMessageBox.information(self, "Success", "Structure generation completed.")
        
    def _on_worker_error(self, error_msg):
        """Handle worker error."""
        self._cleanup_worker()
        QMessageBox.critical(self, "Error", f"Generation failed: {error_msg}")
        
    def _cleanup_worker(self):
        """Clean up worker and progress dialog."""
        if self._progress_dialog:
            self._progress_dialog.close()
            self._progress_dialog = None
        self._worker = None
    


    # ... (skipping other methods) ...

    def _on_add_sphere(self):
        """Add a manual sphere void (sync)."""
        self._ensure_modifier_sync()
        if self._modifier is None: return
        
        center = (self._sphere_x.value(), self._sphere_y.value(), self._sphere_z.value())
        radius = self._sphere_r.value()
        
        try:
            self._modifier.add_sphere_void(center, radius)
            self._data_manager.set_voxel_grid(self._modifier.grid)
            self.structure_applied.emit(self._modifier.grid)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add sphere: {e}")
    
    def _on_add_cylinder(self):
        """Add a manual cylinder void (sync)."""
        self._ensure_modifier_sync()
        if self._modifier is None: return

        start = (self._cyl_x1.value(), self._cyl_y1.value(), self._cyl_z1.value())
        end = (self._cyl_x2.value(), self._cyl_y2.value(), self._cyl_z2.value())
        radius = self._cyl_r.value()
        
        try:
            self._modifier.add_cylinder_void(start, end, radius)
            self._data_manager.set_voxel_grid(self._modifier.grid)
            self.structure_applied.emit(self._modifier.grid)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add cylinder: {e}")

    def _ensure_modifier_sync(self):
        """Ensure modifier exists for synchronous operations."""
        if self._modifier is None:
            if not self._data_manager.has_stl:
                QMessageBox.warning(self, "No STL", "Please load an STL file first.")
                return
            
            # Sync initialization if needed for manual tools (less ideal, but simple)
            reply = QMessageBox.question(
                self, "Initialize Grid?", 
                "Voxel grid not initialized. Initialize now with current settings?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self._perform_initialization_sync()

    def _perform_initialization_sync(self):
         # ... existing initialization logic reused for sync helper ...
        mesh = self._data_manager.mesh
        # Fix: voxel size is now determined by simulation, but for manual tools we need a default/dialog?
        # Since _voxel_size_spin was removed, we default to 0.5 or ask user?
        # For simplicity, default to 0.5mm if not available
        voxel_size = 0.5 
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            voxelizer = Voxelizer(voxel_size=voxel_size, fill_interior=True)
            voxel_grid = voxelizer.voxelize(mesh)
            self._modifier = StructureModifier(voxel_grid)
            self._data_manager.set_voxel_grid(voxel_grid)
            logging.info(f"Voxel grid initialized: {voxel_grid.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Voxelization failed: {e}")
        finally:
            QApplication.restoreOverrideCursor()
    
    def reset_structure(self):
        """
        Reset to original voxel grid.
        Public method to be called from MainWindow.
        """
        if self._modifier is not None:
            self._modifier.reset()
            self._data_manager.set_voxel_grid(self._modifier.grid)
            self.structure_applied.emit(self._modifier.grid)
