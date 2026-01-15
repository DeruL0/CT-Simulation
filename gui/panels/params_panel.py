"""
Simulation Parameters Panel

Provides UI controls for configuring CT simulation parameters.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSlider, QSpinBox, QDoubleSpinBox,
    QCheckBox, QFormLayout, QComboBox, QStackedWidget
)
from PySide6.QtCore import Qt, Signal
import math


class ParamsPanel(QWidget):
    """Panel for CT simulation parameters."""
    
    # Signal emitted when parameters change
    params_changed = Signal()
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._mesh_dims = None  # Store mesh dimensions for calculations
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Voxelization settings
        voxel_group = QGroupBox("Voxelization / Resolution")
        voxel_layout = QVBoxLayout(voxel_group)
        
        # Strategy selection
        strat_layout = QHBoxLayout()
        strat_layout.addWidget(QLabel("Mode:"))
        self._res_strategy_combo = QComboBox()
        self._res_strategy_combo.addItems([
            "Voxel Size (Manual)",
            "Octree Depth",
            "Target Resolution (Slice Size)",
            "Target Slice Count"
        ])
        self._res_strategy_combo.currentIndexChanged.connect(self._on_strategy_changed)
        strat_layout.addWidget(self._res_strategy_combo)
        voxel_layout.addLayout(strat_layout)
        
        # Stacked widget for different inputs
        self._input_stack = QStackedWidget()
        
        # Page 0: Manual Voxel Size
        page_manual = QWidget()
        layout_manual = QFormLayout(page_manual)
        layout_manual.setContentsMargins(0, 0, 0, 0)
        self._voxel_size_spin = QDoubleSpinBox()
        self._voxel_size_spin.setRange(0.001, 100.0)
        self._voxel_size_spin.setValue(0.5)
        self._voxel_size_spin.setSingleStep(0.01)
        self._voxel_size_spin.setDecimals(4)
        self._voxel_size_spin.setSuffix(" mm")
        self._voxel_size_spin.valueChanged.connect(self._on_param_changed)
        layout_manual.addRow("Voxel Size:", self._voxel_size_spin)
        self._input_stack.addWidget(page_manual)
        
        # Page 1: Octree Depth
        page_octree = QWidget()
        layout_octree = QFormLayout(page_octree)
        layout_octree.setContentsMargins(0, 0, 0, 0)
        self._octree_depth_spin = QSpinBox()
        self._octree_depth_spin.setRange(4, 14)  # 2^4=16 to 2^14=16384
        self._octree_depth_spin.setValue(8)      # 2^8=256
        self._octree_depth_spin.valueChanged.connect(self._on_octree_depth_changed)
        layout_octree.addRow("Octree Depth:", self._octree_depth_spin)
        self._octree_info_label = QLabel("Grid: -")
        self._octree_info_label.setObjectName("secondaryLabel")
        layout_octree.addRow("", self._octree_info_label)
        self._input_stack.addWidget(page_octree)
        
        # Page 2: Target Resolution (X/Y)
        page_res = QWidget()
        layout_res = QFormLayout(page_res)
        layout_res.setContentsMargins(0, 0, 0, 0)
        self._target_res_spin = QSpinBox()
        self._target_res_spin.setRange(32, 4096)
        self._target_res_spin.setValue(512)
        self._target_res_spin.setSingleStep(32)
        self._target_res_spin.setSuffix(" px")
        self._target_res_spin.valueChanged.connect(self._on_target_res_changed)
        layout_res.addRow("Max Resolution:", self._target_res_spin)
        self._input_stack.addWidget(page_res)
        
        # Page 3: Target Slice Count (Z)
        page_slices = QWidget()
        layout_slices = QFormLayout(page_slices)
        layout_slices.setContentsMargins(0, 0, 0, 0)
        self._target_slices_spin = QSpinBox()
        self._target_slices_spin.setRange(32, 4096)
        self._target_slices_spin.setValue(256)
        self._target_slices_spin.setSingleStep(16)
        self._target_slices_spin.valueChanged.connect(self._on_target_slices_changed)
        layout_slices.addRow("Num Slices:", self._target_slices_spin)
        self._input_stack.addWidget(page_slices)
        
        voxel_layout.addWidget(self._input_stack)
        
        # Fill interior checkbox (common)
        self._fill_interior_check = QCheckBox("Fill interior")
        self._fill_interior_check.setChecked(True)
        self._fill_interior_check.stateChanged.connect(self._on_param_changed)
        voxel_layout.addWidget(self._fill_interior_check)
        
        layout.addWidget(voxel_group)
        
        # CT simulation settings
        ct_group = QGroupBox("CT Simulation")
        ct_layout = QFormLayout(ct_group)
        
        # Number of projections
        proj_row = QHBoxLayout()
        self._projections_spin = QSpinBox()
        self._projections_spin.setRange(90, 720)
        self._projections_spin.setValue(360)
        self._projections_spin.setSingleStep(30)
        self._projections_spin.valueChanged.connect(self._on_param_changed)
        proj_row.addWidget(self._projections_spin)
        ct_layout.addRow("Projections:", proj_row)
        
        # Tube voltage (kVp)
        kvp_row = QHBoxLayout()
        self._kvp_spin = QSpinBox()
        self._kvp_spin.setRange(80, 140)
        self._kvp_spin.setValue(120)
        self._kvp_spin.setSingleStep(10)
        self._kvp_spin.setSuffix(" kVp")
        self._kvp_spin.valueChanged.connect(self._on_param_changed)
        kvp_row.addWidget(self._kvp_spin)
        ct_layout.addRow("Tube Voltage:", kvp_row)
        
        layout.addWidget(ct_group)
        
        # Noise settings
        noise_group = QGroupBox("Noise")
        noise_layout = QVBoxLayout(noise_group)
        
        self._add_noise_check = QCheckBox("Add realistic noise")
        self._add_noise_check.setChecked(True)
        self._add_noise_check.stateChanged.connect(self._on_noise_toggle)
        noise_layout.addWidget(self._add_noise_check)
        
        # Noise level slider
        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Level:"))
        
        self._noise_slider = QSlider(Qt.Horizontal)
        self._noise_slider.setRange(0, 100)
        self._noise_slider.setValue(20)  # 2% default
        self._noise_slider.valueChanged.connect(self._on_noise_level_changed)
        noise_row.addWidget(self._noise_slider, stretch=1)
        
        self._noise_value_label = QLabel("2.0%")
        self._noise_value_label.setMinimumWidth(50)
        noise_row.addWidget(self._noise_value_label)
        
        noise_layout.addLayout(noise_row)
        layout.addWidget(noise_group)
        
        # Use fast simulation option
        speed_group = QGroupBox("Performance")
        speed_layout = QVBoxLayout(speed_group)
        
        self._fast_mode_check = QCheckBox("Fast mode (approximate, no reconstruction)")
        self._fast_mode_check.setChecked(False)
        self._fast_mode_check.stateChanged.connect(self._on_param_changed)
        speed_layout.addWidget(self._fast_mode_check)
        
        self._use_gpu_check = QCheckBox("Use GPU Acceleration (requires CuPy)")
        self._use_gpu_check.setChecked(False)
        self._use_gpu_check.setToolTip("Requires NVIDIA GPU and 'cupy' library installed.")
        self._use_gpu_check.stateChanged.connect(self._on_param_changed)
        speed_layout.addWidget(self._use_gpu_check)
        
        # Memory limit setting
        mem_row = QHBoxLayout()
        mem_row.addWidget(QLabel("Memory Limit:"))
        self._memory_limit_spin = QDoubleSpinBox()
        self._memory_limit_spin.setRange(0.5, 16.0)
        self._memory_limit_spin.setValue(2.0)
        self._memory_limit_spin.setSingleStep(0.5)
        self._memory_limit_spin.setDecimals(1)
        self._memory_limit_spin.setSuffix(" GB")
        self._memory_limit_spin.valueChanged.connect(self._on_param_changed)
        mem_row.addWidget(self._memory_limit_spin)
        speed_layout.addLayout(mem_row)
        
        # Memory estimation display
        self._memory_estimate_label = QLabel("Est. Memory: -")
        self._memory_estimate_label.setObjectName("secondaryLabel")
        speed_layout.addWidget(self._memory_estimate_label)
        
        layout.addWidget(speed_group)
        
        # Stretch at bottom
        layout.addStretch()
    
    def _on_param_changed(self) -> None:
        """Handle parameter change."""
        self.params_changed.emit()
    
    def _on_strategy_changed(self, index: int) -> None:
        """Handle strategy change."""
        self._input_stack.setCurrentIndex(index)
        self._recalc_voxel_size()
        self.params_changed.emit()
        
    def _on_octree_depth_changed(self) -> None:
        """Handle octree depth change."""
        depth = self._octree_depth_spin.value()
        grid_size = 2 ** depth
        self._octree_info_label.setText(f"Grid: {grid_size}³")
        self._recalc_voxel_size()
        self.params_changed.emit()
        
    def _on_target_res_changed(self) -> None:
        """Handle target resolution change."""
        self._recalc_voxel_size()
        self.params_changed.emit()
        
    def _on_target_slices_changed(self) -> None:
        """Handle target slices change."""
        self._recalc_voxel_size()
        self.params_changed.emit()
        
    def _recalc_voxel_size(self):
        """Recalculate and update voxel size based on strategy."""
        if self._mesh_dims is None:
            return
            
        strategy = self._res_strategy_combo.currentIndex()
        dims = self._mesh_dims
        max_dim = max(dims)
        
        new_voxel_size = self._voxel_size_spin.value()
        
        if strategy == 1:  # Octree Depth
            depth = self._octree_depth_spin.value()
            new_voxel_size = max_dim / (2 ** depth)
            
        elif strategy == 2:  # Target Resolution
            res = self._target_res_spin.value()
            # Assuming res applies to max dimension
            new_voxel_size = max_dim / res
            
        elif strategy == 3:  # Target Slices
            slices = self._target_slices_spin.value()
            # Slices applies to Z dimension (index 2)
            # Add small padding factor or just straight division
            new_voxel_size = dims[2] / slices
            
        # Update spinbox without triggering signal loop if possible
        # But we want to keep it consistent.
        # Actually, if we are in Manual mode, we don't update from calculation.
        # If we are in other modes, we update the manual spinbox but also store a "calculated" value?
        # Simpler: The `voxel_size` property will return the CALCULATED value if not in manual mode.
        # And we update the manual spinbox for visual reference (blocking signal).
        
        if strategy != 0:
            self._voxel_size_spin.blockSignals(True)
            self._voxel_size_spin.setValue(new_voxel_size)
            self._voxel_size_spin.blockSignals(False)
    
    def _on_noise_toggle(self, state: int) -> None:
        """Handle noise checkbox toggle."""
        enabled = state == Qt.Checked
        self._noise_slider.setEnabled(enabled)
        self._noise_value_label.setEnabled(enabled)
        self.params_changed.emit()
    
    def _on_noise_level_changed(self, value: int) -> None:
        """Handle noise level slider change."""
        percent = value / 10.0
        self._noise_value_label.setText(f"{percent:.1f}%")
        self.params_changed.emit()
    
    # Properties for accessing parameter values
    
    @property
    def voxel_size(self) -> float:
        """Get voxel size in mm."""
        return self._voxel_size_spin.value()
    
    @property
    def fill_interior(self) -> bool:
        """Get fill interior setting."""
        return self._fill_interior_check.isChecked()
    
    @property
    def num_projections(self) -> int:
        """Get number of projections."""
        return self._projections_spin.value()
    
    @property
    def kvp(self) -> int:
        """Get tube voltage in kVp."""
        return self._kvp_spin.value()
    
    @property
    def add_noise(self) -> bool:
        """Get noise enable setting."""
        return self._add_noise_check.isChecked()
    
    @property
    def noise_level(self) -> float:
        """Get noise level as fraction (0.0-1.0)."""
        return self._noise_slider.value() / 1000.0
    
    @property
    def fast_mode(self) -> bool:
        """Get fast mode setting."""
        return self._fast_mode_check.isChecked()
    
    @property
    def use_gpu(self) -> bool:
        """Get GPU acceleration setting."""
        return self._use_gpu_check.isChecked()

    @property
    def memory_limit_gb(self) -> float:
        """Get memory limit in GB."""
        return self._memory_limit_spin.value()
    
    def update_memory_estimate(self, mesh_dimensions: tuple) -> None:
        """
        Update the memory estimation display based on mesh dimensions.
        
        Args:
            mesh_dimensions: (width, height, depth) in mm
        """
        self._mesh_dims = mesh_dimensions  # Store for calculations
        self._recalc_voxel_size()  # Recalculate voxel size with new dims if needed
        if mesh_dimensions is None:
            self._memory_estimate_label.setText("Est. Memory: -")
            return
        
        voxel_size = self.voxel_size
        dims = mesh_dimensions
        
        # Calculate grid dimensions
        grid_shape = [int(d / voxel_size) + 4 for d in dims]  # +4 for padding
        total_voxels = grid_shape[0] * grid_shape[1] * grid_shape[2]
        
        # Memory estimate (float32 = 4 bytes)
        memory_bytes = total_voxels * 4
        memory_gb = memory_bytes / (1024 ** 3)
        
        # Check if exceeds limit
        limit_gb = self.memory_limit_gb
        if memory_gb > limit_gb:
            self._memory_estimate_label.setText(
                f"Est. Memory: {memory_gb:.2f} GB ⚠️ (exceeds {limit_gb:.1f} GB limit)"
            )
            self._memory_estimate_label.setStyleSheet("color: #E53935;")  # Red warning
        else:
            self._memory_estimate_label.setText(f"Est. Memory: {memory_gb:.2f} GB")
            self._memory_estimate_label.setStyleSheet("color: #666666;")  # Normal
