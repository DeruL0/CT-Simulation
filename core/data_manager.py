"""
Data Manager

Centralized data state management for the application.
"""

import logging
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal

from loaders.stl_loader import STLLoader
from simulation.ct_simulator import CTVolume


class DataManager(QObject):
    """
    Manages application data state.
    
    Provides a centralized location for:
    - STL mesh data
    - CT volume data
    - State change notifications via signals
    """
    
    # Signals
    stl_loaded = Signal(object)  # Emits STLLoader
    ct_volume_changed = Signal(object)  # Emits CTVolume or None
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._stl_loader: Optional[STLLoader] = None
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
    def ct_volume(self) -> Optional[CTVolume]:
        """Current CT volume."""
        return self._ct_volume
    
    @property
    def has_stl(self) -> bool:
        """Whether STL data is loaded."""
        return self._stl_loader is not None and self._stl_loader.mesh is not None
    
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
            self._stl_loader = loader
            self.stl_loaded.emit(loader)
            logging.info(f"Loaded STL: {filepath}")
            return True
        except Exception as e:
            logging.error(f"Failed to load STL: {e}")
            return False
    
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
    
    def clear(self) -> None:
        """Clear all data."""
        self._stl_loader = None
        self._ct_volume = None
        self.ct_volume_changed.emit(None)
        logging.info("Data cleared")
