"""
Scatter Physics Models

Implements scatter estimation for CT simulation.
Scatter causes characteristic "cupping" artifacts and reduces contrast.
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

# Optional CuPy import for GPU acceleration
try:
    import cupy as cp
    from cupyx.scipy import ndimage as cp_ndimage
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None
    cp_ndimage = None


class ScatterModel(ABC):
    """Abstract base class for scatter simulation models."""
    
    @abstractmethod
    def compute_scatter(
        self, 
        primary_intensity: np.ndarray,
        path_lengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute scatter contribution from primary intensity.
        
        Args:
            primary_intensity: Primary (unscattered) transmitted intensity.
                Shape: (n_det, n_angles) for sinogram or (H, W) for 2D.
            path_lengths: Optional path lengths through material (for thickness modulation).
            
        Returns:
            Scatter intensity with same shape as primary_intensity.
        """
        pass


class ConvolutionScatter(ScatterModel):
    """
    Convolution-based scatter estimation.
    
    Models scatter as a blurred version of the primary signal, scaled by
    a scatter-to-primary ratio (SPR). This is a fast approximation suitable
    for generating training data with realistic scatter artifacts.
    
    Physics basis:
    - Scatter spreads over a broad area (approximated by Gaussian kernel)
    - Scatter fraction depends on object size/thickness
    - Low-frequency "haze" reduces image contrast
    
    Reference: Siewerdsen, Jaffray, "Cone-beam computed tomography with a 
    flat-panel imager: Magnitude and effects of x-ray scatter" (2001)
    """
    
    def __init__(
        self,
        scatter_fraction: float = 0.15,
        kernel_sigma: float = 30.0,
        use_gpu: bool = False
    ):
        """
        Initialize convolution scatter model.
        
        Args:
            scatter_fraction: Scatter-to-Primary Ratio (SPR), typically 0.05-0.3.
            kernel_sigma: Standard deviation of Gaussian scatter kernel in pixels.
                Larger values = broader scatter spread.
            use_gpu: Whether to use GPU acceleration.
        """
        self.scatter_fraction = scatter_fraction
        self.kernel_sigma = kernel_sigma
        self.use_gpu = use_gpu and HAS_CUPY
        
    def compute_scatter(
        self,
        primary_intensity: np.ndarray,
        path_lengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute scatter using Gaussian convolution.
        
        The scatter is computed as:
            I_scatter = SPR * GaussianBlur(I_primary, sigma)
            
        Optionally modulated by path length (thicker objects scatter more).
        """
        if self.use_gpu and HAS_CUPY:
            return self._compute_scatter_gpu(primary_intensity, path_lengths)
        else:
            return self._compute_scatter_cpu(primary_intensity, path_lengths)

    def _sigma_for_shape(self, ndim: int):
        """
        Build Gaussian sigma per axis.

        For 3D sinograms (n_det, n_angles, n_slices), blur over detector and
        slice axes while keeping angular bins independent.
        """
        if ndim == 2:
            return (self.kernel_sigma, self.kernel_sigma)
        if ndim == 3:
            return (self.kernel_sigma, 0.0, self.kernel_sigma)
        return tuple(self.kernel_sigma for _ in range(ndim))
    
    def _compute_scatter_cpu(
        self,
        primary_intensity: np.ndarray,
        path_lengths: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """CPU implementation using scipy."""
        from scipy import ndimage

        scattered = ndimage.gaussian_filter(
            primary_intensity,
            sigma=self._sigma_for_shape(primary_intensity.ndim),
            mode='constant'
        )
        
        # Scale by scatter fraction
        scatter_contribution = self.scatter_fraction * scattered
        
        # Optional: modulate by path length (more material = more scatter)
        if path_lengths is not None:
            # Normalize path lengths to [0, 1] range
            max_path = path_lengths.max() if path_lengths.max() > 0 else 1.0
            thickness_factor = np.clip(path_lengths / max_path, 0.1, 1.0)
            scatter_contribution = scatter_contribution * thickness_factor
            
        return scatter_contribution.astype(primary_intensity.dtype)
    
    def _compute_scatter_gpu(
        self,
        primary_intensity: "cp.ndarray",
        path_lengths: Optional["cp.ndarray"] = None
    ) -> "cp.ndarray":
        """GPU implementation using CuPy."""
        scattered = cp_ndimage.gaussian_filter(
            primary_intensity,
            sigma=self._sigma_for_shape(primary_intensity.ndim),
            mode='constant'
        )
        
        # Scale by scatter fraction
        scatter_contribution = self.scatter_fraction * scattered
        
        # Optional thickness modulation
        if path_lengths is not None:
            max_path = float(path_lengths.max()) if float(path_lengths.max()) > 0 else 1.0
            thickness_factor = cp.clip(path_lengths / max_path, 0.1, 1.0)
            scatter_contribution = scatter_contribution * thickness_factor
            
        return scatter_contribution.astype(primary_intensity.dtype)


class MotionBlur:
    """
    Motion blur simulation for CT gantry rotation.
    
    During X-ray exposure, the gantry rotates continuously. This causes
    motion blur in the projection data, especially visible at sharp edges.
    
    Implemented as 1D uniform filter along the angular axis of the sinogram.
    """
    
    def __init__(
        self,
        blur_angle_deg: float = 1.0,
        use_gpu: bool = False
    ):
        """
        Initialize motion blur model.
        
        Args:
            blur_angle_deg: Integration angle in degrees (typically 0.5-2.0).
                Represents the angular range over which signal is integrated.
            use_gpu: Whether to use GPU acceleration.
        """
        self.blur_angle_deg = blur_angle_deg
        self.use_gpu = use_gpu and HAS_CUPY
        
    def apply(
        self,
        intensity: np.ndarray,
        angular_step_deg: float
    ) -> np.ndarray:
        """
        Apply motion blur to intensity sinogram.
        
        Args:
            intensity: Transmitted intensity sinogram (n_det, n_angles).
            angular_step_deg: Angular step between projections in degrees.
            
        Returns:
            Motion-blurred intensity sinogram.
        """
        # Calculate kernel size (number of angles to blur over)
        kernel_size = max(1, int(round(self.blur_angle_deg / angular_step_deg)))
        
        if kernel_size <= 1:
            return intensity  # No blur needed
            
        if self.use_gpu and HAS_CUPY:
            return self._apply_gpu(intensity, kernel_size)
        else:
            return self._apply_cpu(intensity, kernel_size)
    
    def _apply_cpu(self, intensity: np.ndarray, kernel_size: int) -> np.ndarray:
        """CPU implementation using scipy."""
        from scipy import ndimage
        
        # Uniform filter along angular axis (axis 1 for sinogram)
        blurred = ndimage.uniform_filter1d(
            intensity,
            size=kernel_size,
            axis=1,  # Angular axis
            mode='wrap'  # Wrap around for continuous rotation
        )
        return blurred.astype(intensity.dtype)
    
    def _apply_gpu(self, intensity: "cp.ndarray", kernel_size: int) -> "cp.ndarray":
        """GPU implementation using CuPy."""
        # Uniform filter along angular axis
        blurred = cp_ndimage.uniform_filter1d(
            intensity,
            size=kernel_size,
            axis=1,
            mode='wrap'
        )
        return blurred.astype(intensity.dtype)
