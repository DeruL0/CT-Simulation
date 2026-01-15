"""
Volume Viewer

Framework-agnostic 3D volume visualization logic.
Extracted from gui/viewer_3d_panel.py for separation of concerns.
"""

from typing import Optional, Dict, Any
import numpy as np


class VolumeViewer:
    """
    Framework-agnostic 3D volume viewer.
    
    This class handles the data logic for 3D visualization,
    independent of any specific rendering framework (PyVista, VTK, etc.).
    """
    
    # View presets mapping names to camera positions
    VIEW_PRESETS = {
        "Isometric": {"azimuth": 45, "elevation": 30},
        "Front": {"azimuth": 0, "elevation": 0},
        "Back": {"azimuth": 180, "elevation": 0},
        "Left": {"azimuth": -90, "elevation": 0},
        "Right": {"azimuth": 90, "elevation": 0},
        "Top": {"azimuth": 0, "elevation": 90},
        "Bottom": {"azimuth": 0, "elevation": -90},
    }
    
    def __init__(self):
        self._mesh = None
        self._volume_data: Optional[np.ndarray] = None
        self._voxel_size: float = 1.0
        self._threshold: float = 0.0
        self._show_edges: bool = False
        self._current_view: str = "Isometric"
    
    def set_mesh(self, mesh: Any) -> None:
        """
        Set a mesh object for visualization.
        
        Args:
            mesh: Mesh object (trimesh.Trimesh or similar)
        """
        self._mesh = mesh
        self._volume_data = None
    
    def set_volume(
        self, 
        data: np.ndarray, 
        voxel_size: float = 1.0,
        threshold: float = 0.0
    ) -> None:
        """
        Set volumetric data for isosurface visualization.
        
        Args:
            data: 3D numpy array (Z, Y, X)
            voxel_size: Size of each voxel in mm
            threshold: Isosurface threshold value
        """
        self._volume_data = data
        self._voxel_size = voxel_size
        self._threshold = threshold
        self._mesh = None
    
    def set_view(self, view_name: str) -> Dict[str, float]:
        """
        Set the camera view to a preset.
        
        Args:
            view_name: Name of the view preset
            
        Returns:
            Dictionary with azimuth and elevation values
        """
        if view_name in self.VIEW_PRESETS:
            self._current_view = view_name
            return self.VIEW_PRESETS[view_name]
        return self.VIEW_PRESETS["Isometric"]
    
    def set_edges_visible(self, visible: bool) -> None:
        """Set whether mesh edges should be visible."""
        self._show_edges = visible
    
    @property
    def has_data(self) -> bool:
        """Check if any data is loaded for visualization."""
        return self._mesh is not None or self._volume_data is not None
    
    @property
    def mesh(self) -> Any:
        """Get the current mesh."""
        return self._mesh
    
    @property
    def volume_data(self) -> Optional[np.ndarray]:
        """Get the current volume data."""
        return self._volume_data
    
    @property
    def voxel_size(self) -> float:
        """Get the current voxel size."""
        return self._voxel_size
    
    @property
    def threshold(self) -> float:
        """Get the current isosurface threshold."""
        return self._threshold
    
    @property
    def show_edges(self) -> bool:
        """Get whether edges should be shown."""
        return self._show_edges
