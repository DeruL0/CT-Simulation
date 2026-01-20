"""
Defects Generation Page

Page for configuring random defect generation parameters.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, 
    QComboBox, QDoubleSpinBox, QSpinBox, QSlider, QCheckBox,
    QSizePolicy
)
from PySide6.QtCore import Qt

from simulation.structures import DefectShape, DefectConfig


class DefectsPage(QWidget):
    """
    UI for configuring random defects (voids).
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
        
        # Defect shape
        self._shape_combo = QComboBox()
        for ds in DefectShape:
            self._shape_combo.addItem(ds.value.title(), ds)
        form_layout.addRow("Shape:", self._shape_combo)
        
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
        self._size_mean_spin = QDoubleSpinBox()
        self._size_mean_spin.setRange(0.5, 20.0)
        self._size_mean_spin.setValue(2.0)
        self._size_mean_spin.setSuffix(" mm")
        form_layout.addRow("Mean Size:", self._size_mean_spin)
        
        self._size_std_spin = QDoubleSpinBox()
        self._size_std_spin.setRange(0.0, 10.0)
        self._size_std_spin.setValue(0.5)
        self._size_std_spin.setSuffix(" mm")
        form_layout.addRow("Size Std:", self._size_std_spin)
        
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

    def get_config(self) -> DefectConfig:
        """Get the current configuration."""
        return DefectConfig(
            shape=self._shape_combo.currentData(),
            density_percent=self._porosity_spin.value(),
            size_mean_mm=self._size_mean_spin.value(),
            size_std_mm=self._size_std_spin.value(),
            preserve_shell=self._preserve_shell_check.isChecked(),
            shell_thickness_mm=self._shell_thickness_spin.value()
        )
