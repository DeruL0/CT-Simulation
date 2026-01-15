"""
Base Simulation Backend

Abstract interface for Radon/IRadon implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class SimulationBackend(ABC):
    """Abstract base class for simulation backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name for logging."""
        pass
    
    @abstractmethod
    def radon(
        self, 
        image: np.ndarray, 
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute Radon transform (forward projection).
        
        Args:
            image: 2D input image (H, W)
            theta: Projection angles in degrees
            
        Returns:
            Sinogram (n_det, n_angles)
        """
        pass
    
    @abstractmethod
    def iradon(
        self, 
        sinogram: np.ndarray, 
        theta: np.ndarray,
        output_size: int
    ) -> np.ndarray:
        """
        Compute inverse Radon transform (filtered backprojection).
        
        Args:
            sinogram: Input sinogram (n_det, n_angles)
            theta: Projection angles in degrees
            output_size: Output image size
            
        Returns:
            Reconstructed image (output_size, output_size)
        """
        pass
    
    def process_slice(
        self,
        slice_2d: np.ndarray,
        theta: np.ndarray,
        add_noise: bool = False,
        noise_level: float = 0.02
    ) -> np.ndarray:
        """
        Process a single slice through forward/backward projection.
        
        This is the main entry point for slice-by-slice simulation.
        Subclasses may override for optimized implementations.
        
        Args:
            slice_2d: Input 2D slice (already shifted, background=0)
            theta: Projection angles in degrees
            add_noise: Whether to add noise to sinogram
            noise_level: Noise level as fraction of signal
            
        Returns:
            Reconstructed slice
        """
        # Pad to diagonal
        original_shape = slice_2d.shape
        diag_len = int(np.ceil(np.sqrt(original_shape[0]**2 + original_shape[1]**2)))
        pad_h = diag_len - original_shape[0]
        pad_w = diag_len - original_shape[1]
        
        if pad_h > 0 or pad_w > 0:
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            slice_padded = np.pad(
                slice_2d,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0.0
            )
        else:
            slice_padded = slice_2d
            pad_top = 0
            pad_left = 0
        
        # Forward projection
        sinogram = self.radon(slice_padded, theta)
        
        # Add noise
        if add_noise and noise_level > 0:
            noise = np.random.normal(
                0,
                noise_level * np.abs(sinogram).mean(),
                sinogram.shape
            )
            sinogram = sinogram + noise
        
        # Backward projection
        reconstructed = self.iradon(sinogram, theta, diag_len)
        
        # Crop back
        if pad_h > 0 or pad_w > 0:
            reconstructed = reconstructed[
                pad_top:pad_top+original_shape[0],
                pad_left:pad_left+original_shape[1]
            ]
        
        return reconstructed
