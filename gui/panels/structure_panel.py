"""
Structure Panel

Industrial structure generation panel for CT simulation.
Provides automated lattice and defect generation with density/complexity controls.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QStackedWidget, QButtonGroup, QMessageBox,
    QProgressDialog, QSizePolicy
)
from PySide6.QtCore import Qt, Signal, QThread
import logging
import numpy as np

from simulation.voxelizer import Voxelizer, VoxelGrid
from simulation.structures import StructureModifier
from .pages.lattice_page import LatticePage
from .pages.defects_page import DefectsPage
from .pages.manual_page import ManualPage


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
        self._worker: QThread = None
        self._progress_dialog: QProgressDialog = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main container group
        self._struct_group = QGroupBox("Structure Generation")
        self._struct_group.setCheckable(True)
        self._struct_group.setChecked(False)  # Off by default
        struct_layout = QVBoxLayout(self._struct_group)
        struct_layout.setSpacing(10)
        
        # Divider
        from PySide6.QtWidgets import QFrame
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        struct_layout.addWidget(line)
        
        # Navigation Buttons
        nav_layout = QHBoxLayout()
        self._btn_lattice = QPushButton("Lattice")
        self._btn_defects = QPushButton("Defects")
        self._btn_manual = QPushButton("Manual")
        
        self._btn_lattice.setCheckable(True)
        self._btn_defects.setCheckable(True)
        self._btn_manual.setCheckable(True)
        
        # Styling
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
        
        # Stacked Pages
        self._stack = QStackedWidget()
        
        self._lattice_page = LatticePage()
        self._defects_page = DefectsPage()
        self._manual_page = ManualPage()
        
        self._stack.addWidget(self._lattice_page)
        self._stack.addWidget(self._defects_page)
        self._stack.addWidget(self._manual_page)
        
        struct_layout.addWidget(self._stack)
        
        layout.addWidget(self._struct_group)
        
        # Status Label
        self._status_label = QLabel("Structure generation will be applied during simulation.")
        self._status_label.setStyleSheet("color: #888; font-size: 11px;")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)
        
        layout.addStretch()
        
        # Connect navigation
        self._nav_group.idClicked.connect(self._stack.setCurrentIndex)
        self._stack.currentChanged.connect(self._update_stack_sizing)
        
        # Default state
        self._btn_lattice.setChecked(True)
        self._update_stack_sizing(0)
        self._set_controls_enabled(True)

    def _connect_signals(self):
        # Manual page signals
        self._manual_page.add_sphere.connect(self._on_add_sphere)
        self._manual_page.add_cylinder.connect(self._on_add_cylinder)
        
        # Data manager signals
        self._data_manager.stl_loaded.connect(self._on_stl_loaded)
        
        # Structure group toggle
        self._struct_group.toggled.connect(self._on_group_toggled)

    def _on_group_toggled(self, checked):
        pass

    def _on_stl_loaded(self):
        self._set_controls_enabled(True)
        self._modifier = None  # Reset modifier on new STL

    def _set_controls_enabled(self, enabled: bool):
        self._struct_group.setEnabled(enabled)

    def _update_stack_sizing(self, index):
        """Update stack widget size policy to fit current page."""
        for i in range(self._stack.count()):
            widget = self._stack.widget(i)
            if i == index:
                widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
                widget.updateGeometry()
            else:
                widget.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self._stack.adjustSize()

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
            return ("generate_lattice", self._lattice_page.get_config())
            
        elif idx == 1:  # Defects
            return ("generate_random_voids", self._defects_page.get_config())
            
        return None  # Manual mode not handled via auto-generation pipeline

    def reset_structure(self):
        """Reset structure modifier and voxel grid state."""
        self._modifier = None
        logging.info("Structure panel reset")

    # =========================================================================
    # Manual Modification Handlers
    # =========================================================================

    def _ensure_modifier_sync(self):
        """Ensure modifier exists for synchronous operations."""
        if self._modifier is None:
            if not self._data_manager.has_stl:
                QMessageBox.warning(self, "No STL", "Please load an STL file first.")
                return False
            
            reply = QMessageBox.question(
                self, "Initialize Grid?", 
                "Voxel grid not initialized. Initialize now with default settings (0.5mm)?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                return self._perform_initialization_sync()
            return False
        return True

    def _perform_initialization_sync(self):
        from PySide6.QtWidgets import QApplication
        mesh = self._data_manager.mesh
        voxel_size = 0.5 
        try:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            voxelizer = Voxelizer(voxel_size=voxel_size, fill_interior=True)
            voxel_grid = voxelizer.voxelize(mesh)
            self._modifier = StructureModifier(voxel_grid)
            self._data_manager.set_voxel_grid(voxel_grid)
            logging.info(f"Voxel grid initialized: {voxel_grid.shape}")
            QApplication.restoreOverrideCursor()
            return True
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Error", f"Voxelization failed: {e}")
            return False

    def _on_add_sphere(self, center, radius):
        if not self._ensure_modifier_sync(): return
        
        try:
            self._modifier.add_sphere_void(center, radius)
            self._data_manager.set_voxel_grid(self._modifier.grid)
            self.structure_applied.emit(self._modifier.grid)
            QMessageBox.information(self, "Success", "Sphere added.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add sphere: {e}")

    def _on_add_cylinder(self, start, end, radius):
        if not self._ensure_modifier_sync(): return
        
        try:
            self._modifier.add_cylinder_void(start, end, radius)
            self._data_manager.set_voxel_grid(self._modifier.grid)
            self.structure_applied.emit(self._modifier.grid)
            QMessageBox.information(self, "Success", "Cylinder added.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to add cylinder: {e}")
