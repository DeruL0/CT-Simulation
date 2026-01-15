"""
CPU Backend for CT Simulation

Uses scikit-image radon/iradon for CPU-based transforms.
"""

import numpy as np

try:
    from skimage.transform import radon, iradon
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

from .base import SimulationBackend


class CPUBackend(SimulationBackend):
    """CPU-based simulation using scikit-image."""
    
    def __init__(self):
        if not HAS_SKIMAGE:
            raise ImportError(
                "scikit-image is required for CPU simulation. "
                "Install with: pip install scikit-image"
            )
    
    @property
    def name(self) -> str:
        return "CPU (scikit-image)"
    
    def radon(
        self, 
        image: np.ndarray, 
        theta: np.ndarray
    ) -> np.ndarray:
        """
        Compute Radon transform using scikit-image.
        
        Args:
            image: 2D input image (H, W)
            theta: Projection angles in degrees
            
        Returns:
            Sinogram (n_det, n_angles)
        """
        return radon(image, theta=theta, circle=True)
    
    def iradon(
        self, 
        sinogram: np.ndarray, 
        theta: np.ndarray,
        output_size: int
    ) -> np.ndarray:
        """
        Compute inverse Radon transform using scikit-image.
        
        Args:
            sinogram: Input sinogram (n_det, n_angles)
            theta: Projection angles in degrees
            output_size: Output image size
            
        Returns:
            Reconstructed image (output_size, output_size)
        """
        return iradon(
            sinogram,
            theta=theta,
            circle=True,
            filter_name='ramp',
            output_size=output_size
        )
