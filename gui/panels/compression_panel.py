"""
Compression Panel

Configuration panel for compression simulation with time-step viewer.
Compression is integrated into the main Run Simulation workflow.
"""

from pathlib import Path
from typing import Optional, List

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QCheckBox, QProgressBar, QFileDialog,
    QLineEdit, QGroupBox, QSlider, QFrame
)
from PySide6.QtCore import Qt, Signal
import logging
import numpy as np


class CompressionPanel(QWidget):
    """
    Configuration panel for compression simulation.
    
    Features:
    - Simulation parameters (mode, axis, compression %, steps, Poisson ratio)
    - Time-step slider to view different compression stages (after simulation)
    
    Note: Compression is run as part of "Run Simulation", not standalone.
    """
    
    # Signals
    step_changed = Signal(int, object)  # Emits (step_index, volume_data)
    config_changed = Signal()  # Emits when any config changes
    
    def __init__(self, data_manager, parent=None):
        super().__init__(parent)
        self._data_manager = data_manager
        self._results: List = []  # Store simulation results
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(8)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Main container group
        self._main_group = QGroupBox("Compression Simulation")
        self._main_group.setCheckable(True)
        self._main_group.setChecked(False)
        main_layout = QVBoxLayout(self._main_group)
        main_layout.setSpacing(10)
        
        # Description
        desc = QLabel(
            "Configure compression parameters.\n"
            "Compression will be applied after CT simulation."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #666; font-size: 11px;")
        main_layout.addWidget(desc)
        
        # === Parameters Section ===
        params_group = QGroupBox("Parameters")
        params_layout = QFormLayout(params_group)
        params_layout.setSpacing(6)
        
        # Mode
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Physical (Elasticity)", "physical")
        self._mode_combo.addItem("Geometric (Affine)", "geometric")
        params_layout.addRow("Mode:", self._mode_combo)
        
        # Axis
        self._axis_combo = QComboBox()
        self._axis_combo.addItem("Z-axis (Top-Bottom)", "Z")
        self._axis_combo.addItem("Y-axis (Front-Back)", "Y")
        self._axis_combo.addItem("X-axis (Left-Right)", "X")
        self._axis_combo.setToolTip("Direction of compression force")
        params_layout.addRow("Axis:", self._axis_combo)
        
        # Total Compression
        self._compression_spin = QDoubleSpinBox()
        self._compression_spin.setRange(1.0, 50.0)
        self._compression_spin.setValue(20.0)
        self._compression_spin.setSuffix(" %")
        self._compression_spin.setDecimals(1)
        params_layout.addRow("Compression:", self._compression_spin)
        
        # Steps
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(2, 20)
        self._steps_spin.setValue(5)
        params_layout.addRow("Time Steps:", self._steps_spin)
        
        # Poisson
        self._poisson_spin = QDoubleSpinBox()
        self._poisson_spin.setRange(0.0, 0.49)
        self._poisson_spin.setValue(0.3)
        self._poisson_spin.setDecimals(2)
        params_layout.addRow("Poisson Ratio:", self._poisson_spin)
        
        # Physics settings
        self._downsample_spin = QSpinBox()
        self._downsample_spin.setRange(2, 16)
        self._downsample_spin.setValue(4)
        params_layout.addRow("Downsample:", self._downsample_spin)
        
        self._iterations_spin = QSpinBox()
        self._iterations_spin.setRange(50, 1000)
        self._iterations_spin.setValue(300)
        self._iterations_spin.setSingleStep(50)
        params_layout.addRow("Iterations:", self._iterations_spin)
        
        self._gpu_check = QCheckBox("Use GPU")
        self._gpu_check.setChecked(True)
        params_layout.addRow("", self._gpu_check)
        
        main_layout.addWidget(params_group)
        
        # === Results Viewer Section (hidden until results) ===
        self._results_group = QGroupBox("Time-Step Viewer")
        self._results_group.setVisible(False)
        results_layout = QVBoxLayout(self._results_group)
        
        # Slider
        slider_layout = QHBoxLayout()
        self._step_label = QLabel("Step 0 / 0")
        self._step_label.setMinimumWidth(80)
        
        self._step_slider = QSlider(Qt.Horizontal)
        self._step_slider.setMinimum(0)
        self._step_slider.setMaximum(0)
        self._step_slider.setValue(0)
        self._step_slider.setTickPosition(QSlider.TicksBelow)
        self._step_slider.setTickInterval(1)
        
        slider_layout.addWidget(self._step_label)
        slider_layout.addWidget(self._step_slider, stretch=1)
        results_layout.addLayout(slider_layout)
        
        # Info label
        self._info_label = QLabel("Compression: 0%")
        self._info_label.setStyleSheet("color: #555;")
        results_layout.addWidget(self._info_label)
        
        main_layout.addWidget(self._results_group)
        
        layout.addWidget(self._main_group)
        layout.addStretch()
        
        # Mode change handler
        self._mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed()
    
    def _connect_signals(self):
        self._step_slider.valueChanged.connect(self._on_step_changed)
        
        # Emit config_changed on any parameter change
        self._mode_combo.currentIndexChanged.connect(lambda: self.config_changed.emit())
        self._axis_combo.currentIndexChanged.connect(lambda: self.config_changed.emit())
        self._compression_spin.valueChanged.connect(lambda: self.config_changed.emit())
        self._steps_spin.valueChanged.connect(lambda: self.config_changed.emit())
        self._poisson_spin.valueChanged.connect(lambda: self.config_changed.emit())
        self._main_group.toggled.connect(lambda: self.config_changed.emit())
    
    def _on_mode_changed(self):
        is_physical = self._mode_combo.currentData() == "physical"
        self._downsample_spin.setEnabled(is_physical)
        self._iterations_spin.setEnabled(is_physical)
    
    def is_enabled(self) -> bool:
        """Check if compression is enabled."""
        return self._main_group.isChecked()
    
    def get_config(self) -> dict:
        """Get current compression configuration."""
        return {
            'enabled': self._main_group.isChecked(),
            'mode': self._mode_combo.currentData(),
            'axis': self._axis_combo.currentData(),
            'total_compression': self._compression_spin.value() / 100.0,
            'num_steps': self._steps_spin.value(),
            'poisson_ratio': self._poisson_spin.value(),
            'downsample_factor': self._downsample_spin.value(),
            'solver_iterations': self._iterations_spin.value(),
            'use_gpu': self._gpu_check.isChecked(),
        }
    
    def set_results(self, results: list):
        """
        Set compression results for time-step viewing.
        
        Called by MainWindow after simulation completes.
        """
        self._results = results
        
        if results and len(results) > 1:
            self._step_slider.setMaximum(len(results) - 1)
            self._step_slider.setValue(len(results) - 1)  # Show final step
            self._results_group.setVisible(True)
            self._on_step_changed(len(results) - 1)
        else:
            self._results_group.setVisible(False)
    
    def clear_results(self):
        """Clear previous results."""
        self._results = []
        self._results_group.setVisible(False)
    
    def _on_step_changed(self, step_index: int):
        """Handle slider change - update viewer."""
        if not self._results or step_index >= len(self._results):
            return
        
        result = self._results[step_index]
        total_steps = len(self._results)
        
        # Update labels
        self._step_label.setText(f"Step {step_index} / {total_steps - 1}")
        self._info_label.setText(f"Compression: {result.compression_ratio * 100:.1f}%")
        
        # Emit signal with volume data for viewer update
        self.step_changed.emit(step_index, result.volume)
