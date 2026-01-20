"""
Lattice Generation Page

Page for configuring industrial lattice structure generation parameters.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox,
    QSizePolicy
)
from PySide6.QtCore import Qt

from simulation.structures import LatticeType, LatticeConfig


class LatticePage(QWidget):
    """
    UI for configuring TPMS lattice structures.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
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
        self._cell_size_spin.setToolTip("Size of the TPMS unit cell. Smaller values mean more complex structures.")
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
        self._preserve_shell_check = QCheckBox("Preserve Outer Shell")
        self._preserve_shell_check.setChecked(False)
        self._preserve_shell_check.stateChanged.connect(self._on_shell_toggled)
        form_layout.addRow(self._preserve_shell_check)
        
        self._shell_thickness_spin = QDoubleSpinBox()
        self._shell_thickness_spin.setRange(0.1, 100.0)
        self._shell_thickness_spin.setValue(2.0)
        self._shell_thickness_spin.setSuffix(" mm")
        self._shell_thickness_spin.setEnabled(False)
        form_layout.addRow("Shell Thickness:", self._shell_thickness_spin)
        
        main_layout.addLayout(form_layout)
        
        # Add stretch to push everything up
        main_layout.addStretch()
        
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def _on_shell_toggled(self, state):
        self._shell_thickness_spin.setEnabled(state == Qt.Checked.value)

    def get_config(self) -> LatticeConfig:
        """Get the current configuration."""
        return LatticeConfig(
            lattice_type=self._lattice_type_combo.currentData(),
            density_percent=self._lattice_density_spin.value(),
            cell_size_mm=self._cell_size_spin.value(),
            preserve_shell=self._preserve_shell_check.isChecked(),
            shell_thickness_mm=self._shell_thickness_spin.value()
        )
