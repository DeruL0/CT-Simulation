"""
Viewer Panel

Provides 2D slice viewer for CT volumes with window/level controls.
"""

from typing import Optional, List
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QLabel, QSlider, QSpinBox, QComboBox
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap

from core.windowing import window_to_uint
from visualization.slice_viewer import SliceViewer, WINDOW_PRESETS

try:
    import pyqtgraph as pg
    HAS_PYQTGRAPH = True
except ImportError:
    HAS_PYQTGRAPH = False


class ViewerPanel(QWidget):
    """Panel for viewing reconstructed slices with window/level controls."""
    
    slice_changed = Signal(int)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self._slice_viewer = SliceViewer()
        self._volumes: List[np.ndarray] = []  # Time-series volumes
        self._current_step: int = 0
        self._series_range: Optional[tuple[float, float]] = None
        self._display_buffer: Optional[np.ndarray] = None

        # High-resolution percent mapping for smooth window/level control.
        self._slider_resolution = 100000
        self._center_min = -1.0
        self._center_max = 1.0
        self._width_min = 1e-4
        self._width_max = 2.0
        
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
        wl_group = QGroupBox("Window/Level (mu, cm^-1)")
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
        self._wc_slider.setRange(0, self._slider_resolution)
        self._wc_slider.setValue(int(0.5 * self._slider_resolution))
        self._wc_slider.valueChanged.connect(self._on_window_changed)
        wc_row.addWidget(self._wc_slider, stretch=1)
        
        self._wc_label = QLabel("0.2000 (50.0%)")
        self._wc_label.setMinimumWidth(120)
        wc_row.addWidget(self._wc_label)
        wl_layout.addLayout(wc_row)
        
        # Window width slider
        ww_row = QHBoxLayout()
        ww_row.addWidget(QLabel("Width:"))
        
        self._ww_slider = QSlider(Qt.Horizontal)
        self._ww_slider.setRange(1, self._slider_resolution)
        self._ww_slider.setValue(int(0.2 * self._slider_resolution))
        self._ww_slider.valueChanged.connect(self._on_window_changed)
        ww_row.addWidget(self._ww_slider, stretch=1)
        
        self._ww_label = QLabel("0.4000 (20.0%)")
        self._ww_label.setMinimumWidth(120)
        ww_row.addWidget(self._ww_label)
        wl_layout.addLayout(ww_row)
        
        layout.addWidget(wl_group)

        # Default preset; manual adjustment remains freely available.
        self._preset_combo.setCurrentText("Water / Soft")

    @staticmethod
    def _sample_finite_values(volume: np.ndarray, max_samples: int = 500_000) -> Optional[np.ndarray]:
        """Return finite values from a volume, downsampled for fast robust stats."""
        finite = volume[np.isfinite(volume)]
        if finite.size == 0:
            return None
        if finite.size > max_samples:
            step = max(1, finite.size // max_samples)
            finite = finite[::step]
        return finite

    def _estimate_volume_range(self, volume: np.ndarray) -> Optional[tuple[float, float]]:
        """Estimate value range for one volume."""
        finite = self._sample_finite_values(volume)
        if finite is None:
            return None
        return float(np.min(finite)), float(np.max(finite))

    def _estimate_series_range(self, volumes: List[np.ndarray]) -> Optional[tuple[float, float]]:
        """Estimate a shared value range across all time steps in a series."""
        mins: list[float] = []
        maxs: list[float] = []
        for vol in volumes:
            est = self._estimate_volume_range(vol)
            if est is None:
                continue
            vmin, vmax = est
            mins.append(vmin)
            maxs.append(vmax)
        if not mins:
            return None
        return min(mins), max(maxs)

    
    def set_volume(self, volume: np.ndarray) -> None:
        """
        Set the CT volume to display.
        
        Args:
            volume: 3D numpy array (slices, height, width) in linear attenuation values (cm^-1)
        """
        self._slice_viewer.set_volume(volume)
        num_slices = self._slice_viewer.num_slices
        
        # Update slice controls
        self._slice_slider.setRange(0, num_slices - 1)
        self._slice_spin.setRange(0, num_slices - 1)
        self._total_slices_label.setText(f"/ {num_slices}")
        
        # Reset to middle slice
        middle = self._slice_viewer.current_slice
        self._slice_slider.blockSignals(True)
        self._slice_spin.blockSignals(True)
        self._slice_slider.setValue(middle)
        self._slice_spin.setValue(middle)
        self._slice_slider.blockSignals(False)
        self._slice_spin.blockSignals(False)

        range_hint = self._estimate_volume_range(volume)
        if range_hint is not None:
            self._configure_window_controls(*range_hint)
        
        self._update_display()

    def _configure_window_controls(self, vol_min: float, vol_max: float) -> None:
        """Adapt window sliders to the value range of the current volume."""
        if vol_max <= vol_min:
            vol_max = vol_min + 0.05

        value_range = vol_max - vol_min
        margin = max(0.05, value_range * 0.15)
        self._center_min = vol_min - margin
        self._center_max = vol_max + margin
        if self._center_max <= self._center_min:
            self._center_max = self._center_min + 0.05

        self._width_min = max(1e-4, value_range / self._slider_resolution)
        self._width_max = max(0.10, value_range * 2.0)
        if self._width_max <= self._width_min:
            self._width_max = self._width_min + 0.05

        current_preset = self._preset_combo.currentText()
        if current_preset == "Custom":
            auto_center = 0.5 * (vol_min + vol_max)
            auto_width = max(value_range, 0.05)
            self._set_window_values(auto_center, auto_width)
            return

        if current_preset not in WINDOW_PRESETS:
            current_preset = self._choose_auto_preset(vol_max)

        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentText(current_preset)
        self._preset_combo.blockSignals(False)
        self._apply_preset(current_preset)

    def _set_window_values(self, center: float, width: float) -> None:
        """Set window center/width while respecting current slider bounds."""
        center_slider = self._center_to_slider(center)
        width_slider = self._width_to_slider(width)

        self._wc_slider.blockSignals(True)
        self._ww_slider.blockSignals(True)
        self._wc_slider.setValue(center_slider)
        self._ww_slider.setValue(width_slider)
        self._wc_slider.blockSignals(False)
        self._ww_slider.blockSignals(False)

        self._slice_viewer.set_window(
            self._slider_to_center(center_slider),
            self._slider_to_width(width_slider),
        )
        self._update_window_labels()

    def _center_to_slider(self, center: float) -> int:
        """Convert center value (cm^-1) to high-resolution slider coordinate."""
        rng = self._center_max - self._center_min
        if rng <= 0:
            return 0
        frac = (center - self._center_min) / rng
        return int(np.clip(round(frac * self._slider_resolution), 0, self._slider_resolution))

    def _width_to_slider(self, width: float) -> int:
        """Convert width value (cm^-1) to high-resolution slider coordinate."""
        rng = self._width_max - self._width_min
        if rng <= 0:
            return 1
        frac = (width - self._width_min) / rng
        return int(np.clip(round(frac * self._slider_resolution), 1, self._slider_resolution))

    def _slider_to_center(self, slider_value: int) -> float:
        """Convert center slider coordinate to value (cm^-1)."""
        frac = float(slider_value) / float(self._slider_resolution)
        return self._center_min + frac * (self._center_max - self._center_min)

    def _slider_to_width(self, slider_value: int) -> float:
        """Convert width slider coordinate to value (cm^-1)."""
        frac = float(slider_value) / float(self._slider_resolution)
        return self._width_min + frac * (self._width_max - self._width_min)

    def _update_window_labels(self) -> None:
        """Update WC/WW labels with value + percent."""
        wc = self._slider_to_center(self._wc_slider.value())
        ww = self._slider_to_width(self._ww_slider.value())
        wc_pct = 100.0 * self._wc_slider.value() / self._slider_resolution
        ww_pct = 100.0 * self._ww_slider.value() / self._slider_resolution
        self._wc_label.setText(f"{wc:.4f} ({wc_pct:.1f}%)")
        self._ww_label.setText(f"{ww:.4f} ({ww_pct:.1f}%)")

    def _choose_auto_preset(self, vol_max: float) -> str:
        """Select a reasonable non-custom preset from attenuation range."""
        if vol_max < 0.12:
            return "Low Density"
        if vol_max < 0.45:
            return "Water / Soft"
        if vol_max < 2.0:
            return "Bone / Light Metal"
        if vol_max < 15.0:
            return "Dense Metal"
        return "High-Z Metal"

    def _apply_preset(self, preset_name: str) -> None:
        """Apply a non-custom preset to sliders."""
        preset = WINDOW_PRESETS[preset_name]
        self._set_window_values(preset["center"], preset["width"])
    
    def _on_slice_changed(self, value: int) -> None:
        """Handle slice slider change."""
        self._slice_viewer.set_slice(value)
        current_slice = self._slice_viewer.current_slice
        self._slice_spin.blockSignals(True)
        self._slice_spin.setValue(current_slice)
        self._slice_spin.blockSignals(False)
        self._update_display()
        self.slice_changed.emit(current_slice)
    
    def _on_slice_spin_changed(self, value: int) -> None:
        """Handle slice spin box change."""
        self._slice_viewer.set_slice(value)
        current_slice = self._slice_viewer.current_slice
        self._slice_slider.blockSignals(True)
        self._slice_slider.setValue(current_slice)
        self._slice_slider.blockSignals(False)
        self._update_display()
        self.slice_changed.emit(current_slice)
    
    def _on_preset_changed(self, preset_name: str) -> None:
        """Handle preset combo change."""
        if preset_name not in WINDOW_PRESETS:
            return

        if preset_name != "Custom":
            self._apply_preset(preset_name)

        self._update_display()
    
    def _on_window_changed(self) -> None:
        """Handle window/level slider change."""
        wc = self._slider_to_center(self._wc_slider.value())
        ww = self._slider_to_width(self._ww_slider.value())
        self._slice_viewer.set_window(wc, ww)
        self._update_window_labels()
        
        # Set combo to Custom if values don't match any preset
        for name, preset in WINDOW_PRESETS.items():
            if abs(preset["center"] - wc) < 5e-3 and abs(preset["width"] - ww) < 5e-3:
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
        normalized = self._slice_viewer.get_windowed_slice()
        if normalized is None:
            return

        # Keep a contiguous display buffer alive for Qt image backends that may
        # reference raw memory beyond the scope of this function call.
        self._display_buffer = np.ascontiguousarray(normalized)
        
        if HAS_PYQTGRAPH:
            self._image_view.setImage(self._display_buffer.T, autoLevels=False, 
                                       levels=(0, 255))
        else:
            # Convert to QPixmap for QLabel
            height, width = self._display_buffer.shape
            bytes_per_line = width
            q_image = QImage(self._display_buffer.data, width, height, 
                           bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            scaled = pixmap.scaled(self._image_label.size(), 
                                  Qt.KeepAspectRatio, 
                                  Qt.SmoothTransformation)
            self._image_label.setPixmap(scaled)
    
    @property
    def window_center(self) -> float:
        """Get current window center."""
        return self._slice_viewer.window_center
    
    @property
    def window_width(self) -> float:
        """Get current window width."""
        return self._slice_viewer.window_width
    
    def set_volume_series(self, volumes: List[np.ndarray]) -> None:
        """
        Set a time-series of volumes for step-based viewing.
        
        Args:
            volumes: List of 3D volumes for each time step
        """
        self._volumes = volumes
        self._current_step = 0
        self._series_range = self._estimate_series_range(volumes)
        
        if volumes:
            self.set_volume(volumes[0])
            if self._series_range is not None:
                self._configure_window_controls(*self._series_range)
    
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
        current_slice = self._slice_viewer.current_slice
        
        # Set new volume but try to keep same slice
        volume = self._volumes[step]
        self._slice_viewer.set_volume(volume)
        
        # Adjust slice if out of range
        max_slice = volume.shape[0] - 1
        if current_slice > max_slice:
            current_slice = max_slice
        
        # Update display without resetting slice position
        self._slice_slider.setRange(0, volume.shape[0] - 1)
        self._slice_spin.setRange(0, volume.shape[0] - 1)
        self._total_slices_label.setText(f"/ {volume.shape[0]}")
        
        self._slice_viewer.set_slice(current_slice)
        self._slice_slider.blockSignals(True)
        self._slice_spin.blockSignals(True)
        self._slice_slider.setValue(self._slice_viewer.current_slice)
        self._slice_spin.setValue(self._slice_viewer.current_slice)
        self._slice_slider.blockSignals(False)
        self._slice_spin.blockSignals(False)

        self._update_display()
    
    def clear_volume_series(self) -> None:
        """Clear time-series volumes."""
        self._volumes = []
        self._current_step = 0
        self._series_range = None

    def reset_window_selection(self) -> None:
        """
        Reset clipping/selection controls to default preset state.

        For time-series data, keep one shared control range across all steps.
        """
        if self._series_range is not None:
            self._configure_window_controls(*self._series_range)
        elif self._slice_viewer.num_slices > 0 and self._slice_viewer.volume is not None:
            range_hint = self._estimate_volume_range(self._slice_viewer.volume)
            if range_hint is not None:
                self._configure_window_controls(*range_hint)
        else:
            self._center_min = -1.0
            self._center_max = 1.0
            self._width_min = 1e-4
            self._width_max = 2.0
            self._set_window_values(0.20, 0.40)

        self._preset_combo.blockSignals(True)
        self._preset_combo.setCurrentText("Water / Soft")
        self._preset_combo.blockSignals(False)
        self._apply_preset("Water / Soft")
        self._update_display()
