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
        original_shape = slice_2d.shape
        
        # Output will be a square based on the longer edge
        output_size = max(original_shape[0], original_shape[1])
        
        # Pad to diagonal size for proper radon/iradon transform
        diag_len = int(np.ceil(np.sqrt(output_size**2 + output_size**2))) + 32
        
        # First pad to square (output_size x output_size)
        pad_to_square_h = output_size - original_shape[0]
        pad_to_square_w = output_size - original_shape[1]
        sq_pad_top = pad_to_square_h // 2
        sq_pad_left = pad_to_square_w // 2
        
        if pad_to_square_h > 0 or pad_to_square_w > 0:
            square_slice = np.pad(
                slice_2d,
                ((sq_pad_top, pad_to_square_h - sq_pad_top), 
                 (sq_pad_left, pad_to_square_w - sq_pad_left)),
                mode='constant',
                constant_values=0.0
            )
        else:
            square_slice = slice_2d
        
        # Then pad to diagonal for radon transform
        diag_pad = diag_len - output_size
        diag_pad_half = diag_pad // 2
        
        if diag_pad > 0:
            slice_padded = np.pad(
                square_slice,
                ((diag_pad_half, diag_pad - diag_pad_half), 
                 (diag_pad_half, diag_pad - diag_pad_half)),
                mode='constant',
                constant_values=0.0
            )
        else:
            slice_padded = square_slice
            diag_pad_half = 0
        
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
        
        # Crop back to output_size x output_size (square)
        if diag_pad > 0:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half + output_size,
                diag_pad_half:diag_pad_half + output_size
            ]
        
        return reconstructed, output_size
