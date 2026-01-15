"""
Slice Viewer

Framework-agnostic 2D slice visualization for volumetric data.
Extracted from gui/viewer_panel.py for separation of concerns.
"""

from typing import Optional
import numpy as np


# Window presets for CT viewing
WINDOW_PRESETS = {
    "Bone": {"center": 500, "width": 2000},
    "Soft Tissue": {"center": 40, "width": 400},
    "Lung": {"center": -600, "width": 1500},
    "Brain": {"center": 40, "width": 80},
    "Liver": {"center": 60, "width": 160},
    "Custom": {"center": 0, "width": 1000},
}


class SliceViewer:
    """
    Framework-agnostic slice viewer for 3D volumetric data.
    
    This class handles the data logic for slice viewing,
    independent of any GUI framework.
    """
    
    def __init__(self):
        self._volume: Optional[np.ndarray] = None
        self._current_slice: int = 0
        self._window_center: float = 40.0
        self._window_width: float = 400.0
    
    def set_volume(self, volume: np.ndarray) -> None:
        """
        Set the 3D volume to view.
        
        Args:
            volume: 3D numpy array (slices, height, width)
        """
        self._volume = volume
        self._current_slice = volume.shape[0] // 2
    
    def set_slice(self, index: int) -> None:
        """Set the current slice index."""
        if self._volume is None:
            return
        self._current_slice = max(0, min(index, self._volume.shape[0] - 1))
    
    def set_window(self, center: float, width: float) -> None:
        """Set window center and width for display."""
        self._window_center = center
        self._window_width = max(1.0, width)
    
    def get_windowed_slice(self) -> Optional[np.ndarray]:
        """
        Get the current slice with windowing applied.
        
        Returns:
            2D uint8 array ready for display, or None if no volume loaded
        """
        if self._volume is None:
            return None
        
        slice_data = self._volume[self._current_slice, :, :]
        
        lower = self._window_center - self._window_width / 2
        upper = self._window_center + self._window_width / 2
        
        windowed = np.clip(slice_data, lower, upper)
        normalized = ((windowed - lower) / (upper - lower) * 255).astype(np.uint8)
        
        return normalized
    
    @property
    def num_slices(self) -> int:
        """Get total number of slices."""
        return self._volume.shape[0] if self._volume is not None else 0
    
    @property
    def current_slice(self) -> int:
        """Get current slice index."""
        return self._current_slice
    
    @property
    def window_center(self) -> float:
        """Get current window center."""
        return self._window_center
    
    @property
    def window_width(self) -> float:
        """Get current window width."""
        return self._window_width
