"""
Data Manager

Centralized data state management for the application.
"""

import logging
from typing import Optional

from PySide6.QtCore import QObject, Signal

from loaders import MeshLoader as STLLoader
from simulation.volume import CTVolume
from simulation.voxelizer import VoxelGrid


class DataManager(QObject):
    """
    Manages application data state.
    
    Provides a centralized location for:
    - STL mesh data
    - Voxel grid (modified for structures)
    - CT volume data
    - State change notifications via signals
    """
    
    # Signals
    stl_loaded = Signal(object)  # Emits STLLoader
    voxel_grid_changed = Signal(object)  # Emits VoxelGrid or None
    ct_volume_changed = Signal(object)  # Emits CTVolume or None
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._stl_loader: Optional[STLLoader] = None
        self._voxel_grid: Optional[VoxelGrid] = None
        self._ct_volume: Optional[CTVolume] = None
    
    @property
    def stl_loader(self) -> Optional[STLLoader]:
        """Current STL loader."""
        return self._stl_loader
    
    @property
    def mesh(self):
        """Current mesh from STL loader."""
        if self._stl_loader is not None:
            return self._stl_loader.mesh
        return None
    
    @property
    def stl_info(self):
        """Current STL info."""
        if self._stl_loader is not None:
            return self._stl_loader.info
        return None
    
    @property
    def voxel_grid(self) -> Optional[VoxelGrid]:
        """Current voxel grid (may be modified with structures)."""
        return self._voxel_grid
    
    @property
    def ct_volume(self) -> Optional[CTVolume]:
        """Current CT volume."""
        return self._ct_volume
    
    @property
    def has_stl(self) -> bool:
        """Whether STL data is loaded."""
        return self._stl_loader is not None and self._stl_loader.mesh is not None
    
    @property
    def has_voxel_grid(self) -> bool:
        """Whether voxel grid is available."""
        return self._voxel_grid is not None
    
    @property
    def has_ct_volume(self) -> bool:
        """Whether CT volume is available."""
        return self._ct_volume is not None
    
    def load_stl(self, filepath: str) -> bool:
        """
        Load STL file.
        
        Args:
            filepath: Path to STL file
            
        Returns:
            True if loaded successfully
        """
        try:
            loader = STLLoader()
            loader.load(filepath)
            self.set_stl_loader(loader)
            logging.info(f"Loaded STL: {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to load STL: {e}")
            return False

    def set_stl_loader(self, stl_loader: Optional[STLLoader]) -> None:
        """
        Set the current STL loader and reset derived data.

        Args:
            stl_loader: STLLoader instance or None to clear
        """
        self._stl_loader = stl_loader
        self._voxel_grid = None
        self._ct_volume = None

        self.stl_loaded.emit(stl_loader)
        self.voxel_grid_changed.emit(None)
        self.ct_volume_changed.emit(None)

        if stl_loader is not None:
            logging.info("STL loader set")
    
    def set_voxel_grid(self, voxel_grid: Optional[VoxelGrid]) -> None:
        """
        Set the current voxel grid.
        
        Args:
            voxel_grid: VoxelGrid instance or None to clear
        """
        self._voxel_grid = voxel_grid
        self.voxel_grid_changed.emit(voxel_grid)
        if voxel_grid is not None:
            logging.info(f"Voxel grid set: {voxel_grid.shape}")

    def clear_voxel_grid(self) -> None:
        """Clear the current voxel grid."""
        self.set_voxel_grid(None)
    
    def set_ct_volume(self, ct_volume: Optional[CTVolume]) -> None:
        """
        Set the current CT volume.
        
        Args:
            ct_volume: CTVolume instance or None to clear
        """
        self._ct_volume = ct_volume
        self.ct_volume_changed.emit(ct_volume)
        if ct_volume is not None:
            logging.info(f"CT volume set: {ct_volume.shape}")

    def clear_ct_volume(self) -> None:
        """Clear the current CT volume."""
        self.set_ct_volume(None)
    
    def clear(self) -> None:
        """Clear all data."""
        self._stl_loader = None
        self._voxel_grid = None
        self._ct_volume = None
        self.voxel_grid_changed.emit(None)
        self.ct_volume_changed.emit(None)
        logging.info("Data cleared")

