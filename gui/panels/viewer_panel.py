"""
Viewer Panel

Provides 2D slice viewer for CT volumes with window/level controls.
"""

from typing import Optional, List
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSlider, QSpinBox, QComboBox, QSplitter
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


# Window presets for CT viewing
WINDOW_PRESETS = {
    "Bone": {"center": 500, "width": 2000},
    "Soft Tissue": {"center": 40, "width": 400},
    "Lung": {"center": -600, "width": 1500},
    "Brain": {"center": 40, "width": 80},
    "Liver": {"center": 60, "width": 160},
    "Custom": {"center": 0, "width": 1000},
}


class ViewerPanel(QWidget):
    """Panel for viewing CT slices with window/level controls."""
    
    slice_changed = Signal(int)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._volume: Optional[np.ndarray] = None
        self._volumes: List[np.ndarray] = []  # Time-series volumes
        self._current_step: int = 0
        self._current_slice = 0
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Set up the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Image display area
        view_group = QGroupBox("CT Viewer")
        view_layout = QVBoxLayout(view_group)
        
        # Use pyqtgraph ImageView if available, otherwise QLabel
        if HAS_PYQTGRAPH:
            self._image_view = pg.ImageView()
            self._image_view.ui.roiBtn.hide()
            self._image_view.ui.menuBtn.hide()
            # Use grayscale colormap
            cmap = pg.colormap.getFromMatplotlib('gray')
            self._image_view.setColorMap(cmap)
            view_layout.addWidget(self._image_view)
        else:
            self._image_label = QLabel("No image loaded")
            self._image_label.setAlignment(Qt.AlignCenter)
            self._image_label.setMinimumSize(400, 400)
            self._image_label.setStyleSheet("background-color: #000000;")
            view_layout.addWidget(self._image_label)
        
        layout.addWidget(view_group, stretch=1)
        
        # Slice navigation
        nav_group = QGroupBox("Navigation")
        nav_layout = QHBoxLayout(nav_group)
        
        nav_layout.addWidget(QLabel("Slice:"))
        
        self._slice_slider = QSlider(Qt.Horizontal)
        self._slice_slider.setRange(0, 0)
        self._slice_slider.setValue(0)
        self._slice_slider.valueChanged.connect(self._on_slice_changed)
        nav_layout.addWidget(self._slice_slider, stretch=1)
        
        self._slice_spin = QSpinBox()
        self._slice_spin.setRange(0, 0)
        self._slice_spin.setValue(0)
        self._slice_spin.valueChanged.connect(self._on_slice_spin_changed)
        nav_layout.addWidget(self._slice_spin)
        
        self._total_slices_label = QLabel("/ 0")
        nav_layout.addWidget(self._total_slices_label)
        
        layout.addWidget(nav_group)
        
        # Window/Level controls
        wl_group = QGroupBox("Window/Level")
        wl_layout = QVBoxLayout(wl_group)
        
        # Preset selector
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        
        self._preset_combo = QComboBox()
        for preset_name in WINDOW_PRESETS.keys():
            self._preset_combo.addItem(preset_name)
        self._preset_combo.currentTextChanged.connect(self._on_preset_changed)
        preset_row.addWidget(self._preset_combo, stretch=1)
        wl_layout.addLayout(preset_row)
        
        # Window center slider
        wc_row = QHBoxLayout()
        wc_row.addWidget(QLabel("Center:"))
        
        self._wc_slider = QSlider(Qt.Horizontal)
        self._wc_slider.setRange(-1000, 3000)
        self._wc_slider.setValue(40)
        self._wc_slider.valueChanged.connect(self._on_window_changed)
        wc_row.addWidget(self._wc_slider, stretch=1)
        
        self._wc_label = QLabel("40")
        self._wc_label.setMinimumWidth(50)
        wc_row.addWidget(self._wc_label)
        wl_layout.addLayout(wc_row)
        
        # Window width slider
        ww_row = QHBoxLayout()
        ww_row.addWidget(QLabel("Width:"))
        
        self._ww_slider = QSlider(Qt.Horizontal)
        self._ww_slider.setRange(1, 4000)
        self._ww_slider.setValue(400)
        self._ww_slider.valueChanged.connect(self._on_window_changed)
        ww_row.addWidget(self._ww_slider, stretch=1)
        
        self._ww_label = QLabel("400")
        self._ww_label.setMinimumWidth(50)
        ww_row.addWidget(self._ww_label)
        wl_layout.addLayout(ww_row)
        
        layout.addWidget(wl_group)
    
    def set_volume(self, volume: np.ndarray) -> None:
        """
        Set the CT volume to display.
        
        Args:
            volume: 3D numpy array (slices, height, width) in Hounsfield Units
        """
        self._volume = volume
        num_slices = volume.shape[0]
        
        # Update slice controls
        self._slice_slider.setRange(0, num_slices - 1)
        self._slice_spin.setRange(0, num_slices - 1)
        self._total_slices_label.setText(f"/ {num_slices}")
        
        # Reset to middle slice
        middle = num_slices // 2
        self._slice_slider.setValue(middle)
        self._current_slice = middle
        
        self._update_display()
    
    def _on_slice_changed(self, value: int) -> None:
        """Handle slice slider change."""
        self._current_slice = value
        self._slice_spin.blockSignals(True)
        self._slice_spin.setValue(value)
        self._slice_spin.blockSignals(False)
        self._update_display()
        self.slice_changed.emit(value)
    
    def _on_slice_spin_changed(self, value: int) -> None:
        """Handle slice spin box change."""
        self._current_slice = value
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(value)
        self._slice_slider.blockSignals(False)
        self._update_display()
        self.slice_changed.emit(value)
    
    def _on_preset_changed(self, preset_name: str) -> None:
        """Handle preset combo change."""
        if preset_name in WINDOW_PRESETS:
            preset = WINDOW_PRESETS[preset_name]
            self._wc_slider.blockSignals(True)
            self._ww_slider.blockSignals(True)
            self._wc_slider.setValue(preset["center"])
            self._ww_slider.setValue(preset["width"])
            self._wc_label.setText(str(preset["center"]))
            self._ww_label.setText(str(preset["width"]))
            self._wc_slider.blockSignals(False)
            self._ww_slider.blockSignals(False)
            self._update_display()
    
    def _on_window_changed(self) -> None:
        """Handle window/level slider change."""
        wc = self._wc_slider.value()
        ww = self._ww_slider.value()
        self._wc_label.setText(str(wc))
        self._ww_label.setText(str(ww))
        
        # Set combo to Custom if values don't match any preset
        for name, preset in WINDOW_PRESETS.items():
            if preset["center"] == wc and preset["width"] == ww:
                self._preset_combo.blockSignals(True)
                self._preset_combo.setCurrentText(name)
                self._preset_combo.blockSignals(False)
                break
        else:
            self._preset_combo.blockSignals(True)
            self._preset_combo.setCurrentText("Custom")
            self._preset_combo.blockSignals(False)
        
        self._update_display()
    
    def _update_display(self) -> None:
        """Update the image display."""
        if self._volume is None:
            return
        
        # Get current slice
        slice_data = self._volume[self._current_slice, :, :]
        
        # Apply window/level
        wc = self._wc_slider.value()
        ww = self._ww_slider.value()
        
        lower = wc - ww / 2
        upper = wc + ww / 2
        
        windowed = np.clip(slice_data, lower, upper)
        normalized = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        
        if HAS_PYQTGRAPH:
            self._image_view.setImage(normalized.T, autoLevels=False, 
                                       levels=(0, 255))
        else:
            # Convert to QPixmap for QLabel
            height, width = normalized.shape
            bytes_per_line = width
            q_image = QImage(normalized.data, width, height, 
                           bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(self._image_label.size(), 
                                  Qt.KeepAspectRatio, 
                                  Qt.SmoothTransformation)
            self._image_label.setPixmap(scaled)
    
    @property
    def window_center(self) -> int:
        """Get current window center."""
        return self._wc_slider.value()
    
    @property
    def window_width(self) -> int:
        """Get current window width."""
        return self._ww_slider.value()
    
    def set_volume_series(self, volumes: List[np.ndarray]) -> None:
        """
        Set a time-series of volumes for step-based viewing.
        
        Args:
            volumes: List of 3D volumes for each time step
        """
        self._volumes = volumes
        self._current_step = 0
        
        if volumes:
            self.set_volume(volumes[0])
    
    def set_current_step(self, step: int) -> None:
        """
        Switch 2D view to specified time step.
        
        Args:
            step: Time step index
        """
        if not self._volumes or step < 0 or step >= len(self._volumes):
            return
        
        self._current_step = step
        
        # Preserve current slice position
        current_slice = self._current_slice
        
        # Set new volume but try to keep same slice
        volume = self._volumes[step]
        self._volume = volume
        
        # Adjust slice if out of range
        max_slice = volume.shape[0] - 1
        if current_slice > max_slice:
            current_slice = max_slice
        
        # Update display without resetting slice position
        self._slice_slider.setRange(0, volume.shape[0] - 1)
        self._slice_spin.setRange(0, volume.shape[0] - 1)
        self._total_slices_label.setText(f"/ {volume.shape[0]}")
        
        self._current_slice = current_slice
        self._slice_slider.setValue(current_slice)
        self._update_display()
    
    def clear_volume_series(self) -> None:
        """Clear time-series volumes."""
        self._volumes = []
        self._current_step = 0

