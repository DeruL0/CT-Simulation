"""
Base Simulation Backend

Abstract interface for Radon/IRadon implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np

from .geometry import crop_from_radon_canvas, pad_slice_to_radon_canvas


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
    ) -> tuple:
        """
        Process a single slice through forward/backward projection.
        
        This is the main entry point for slice-by-slice simulation.
        Subclasses may override for optimized implementations.
        
        Output is always a square based on the longer edge.
        
        Args:
            slice_2d: Input 2D slice (already shifted, background=0)
            theta: Projection angles in degrees
            add_noise: Whether to add noise to sinogram
            noise_level: Noise level as fraction of signal
            
        Returns:
            Tuple of (reconstructed_slice, output_size) where slice is square
        """
        slice_padded, geometry = pad_slice_to_radon_canvas(
            slice_2d,
            xp=np,
            constant_value=0.0,
        )
        
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
        reconstructed = self.iradon(sinogram, theta, geometry.diagonal_size)
        reconstructed = crop_from_radon_canvas(reconstructed, geometry)
        
        return reconstructed, geometry.output_size
