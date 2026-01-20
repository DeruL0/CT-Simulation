"""
CT Volume Data Structure

Defines the structure for reconstructed CT volumes.
"""

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass
class CTVolume:
    """
    Reconstructed CT volume with metadata.
    
    Attributes:
        data: 3D numpy array of Hounsfield Units (Z, Y, X)
        voxel_size: Voxel edge length in mm
        origin: World coordinates of volume origin
    """
    data: np.ndarray  # Shape: (slices, height, width), dtype: float32/int16
    voxel_size: float  # mm per voxel
    origin: np.ndarray  # (3,) world position
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        return self.data.shape
    
    @property
    def num_slices(self) -> int:
        return self.data.shape[0]
    
    def get_slice(self, index: int, axis: int = 0) -> np.ndarray:
        """Get a 2D slice along specified axis (0=axial, 1=coronal, 2=sagittal)."""
        if axis == 0:
            return self.data[index, :, :]
        elif axis == 1:
            return self.data[:, index, :]
        else:
            return self.data[:, :, index]
    
    def apply_window(
        self, 
        window_center: float, 
        window_width: float
    ) -> np.ndarray:
        """
        Apply windowing to convert HU values to display range [0, 255].
        
        Args:
            window_center: Center of the window in HU
            window_width: Width of the window in HU
            
        Returns:
            uint8 array suitable for display
        """
        lower = window_center - window_width / 2
        upper = window_center + window_width / 2
        
        windowed = np.clip(self.data, lower, upper)
        normalized = (windowed - lower) / (upper - lower)
        return (normalized * 255).astype(np.uint8)
