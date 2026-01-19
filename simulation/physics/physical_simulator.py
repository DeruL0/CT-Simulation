"""
Physical CT Simulator

Realistic CT simulation with polychromatic X-ray physics including:
- Energy-dependent attenuation
- Beam hardening effects
- Realistic photon noise (Poisson statistics)
"""

from dataclasses import dataclass
from typing import Optional, Callable, Tuple
import logging
import time
import numpy as np

from ..voxelizer import VoxelGrid
from ..materials import MaterialType
from .spectrum import SpectrumGenerator, XRaySpectrum
from .attenuation import get_attenuation_database
from .physical_material import (
    PhysicalMaterial, 
    PHYSICAL_MATERIALS,
    material_type_to_physical
)


@dataclass
class PhysicsConfig:
    """
    Configuration for physical CT simulation.
    
    Attributes:
        kvp: Tube voltage in kV (80, 100, 120, 140)
        filtration_mm_al: Aluminum equivalent filtration in mm
        photon_count: Base photon count per detector element (affects noise)
        energy_bins: Number of energy bins for spectral integration
        enable_scatter: Whether to simulate scatter (not implemented)
    """
    kvp: int = 120
    filtration_mm_al: float = 2.5
    photon_count: float = 1e5  # Photons per detector element
    energy_bins: int = 10      # Coarse binning for speed
    enable_scatter: bool = False  # Future feature


class PhysicalCTSimulator:
    """
    Physical CT simulator with polychromatic X-ray physics.
    
    This simulator accounts for:
    1. Polychromatic X-ray spectrum (Bremsstrahlung + characteristic)
    2. Energy-dependent attenuation (from NIST XCOM data)
    3. Beer-Lambert law for transmission
    4. Poisson noise on photon counts
    5. Beam hardening artifacts
    
    The simulation process:
    1. Generate X-ray spectrum based on kVp and filtration
    2. For each projection angle:
       a. Calculate path lengths through material (Radon transform)
       b. Apply polychromatic attenuation
       c. Sum transmitted photons across all energies
       d. Apply Poisson noise
       e. Log-transform to get projection value
    3. Reconstruct using filtered back projection
    """
    
    def __init__(
        self,
        config: Optional[PhysicsConfig] = None,
        num_projections: int = 360
    ):
        """
        Initialize physical CT simulator.
        
        Args:
            config: Physics configuration (uses defaults if None)
            num_projections: Number of projection angles
        """
        self.config = config or PhysicsConfig()
        self.num_projections = num_projections
        
        # Generate spectrum
        self._spectrum_gen = SpectrumGenerator()
        self._spectrum = self._spectrum_gen.generate(
            kvp=self.config.kvp,
            filtration_mm_al=self.config.filtration_mm_al
        )
        
        # Attenuation database
        self._attenuation_db = get_attenuation_database()
        
        # Projection angles
        self.theta = np.linspace(0, 180, num_projections, endpoint=False)
        
        # Water attenuation for HU conversion
        self._mu_water_effective = self._calculate_effective_mu("water")
        
        logging.info(
            f"PhysicalCTSimulator initialized: {self.config.kvp} kVp, "
            f"{self.config.filtration_mm_al} mm Al, "
            f"mean energy: {self._spectrum.mean_energy:.1f} keV"
        )
    
    def _calculate_effective_mu(self, material: str) -> float:
        """
        Calculate effective linear attenuation coefficient for a material.
        
        Uses the spectrum-weighted average.
        """
        table = self._attenuation_db.get_table(material)
        if table is None:
            return 0.171  # Default water at 100 keV
        
        # Weight by spectrum
        total_weight = 0.0
        weighted_mu = 0.0
        
        for E, phi in zip(self._spectrum.energies, self._spectrum.photons):
            if phi > 0:
                mu = table.get_mu(E)
                weighted_mu += phi * mu
                total_weight += phi
        
        if total_weight > 0:
            return weighted_mu / total_weight
        return 0.171
    
    def simulate(
        self,
        voxel_grid: VoxelGrid,
        material: MaterialType = MaterialType.BONE_CORTICAL,
        background: MaterialType = MaterialType.AIR,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> "CTVolume":
        """
        Simulate CT scan with physical X-ray model.
        
        Args:
            voxel_grid: Binary voxel grid from voxelization
            material: Material type for the object
            background: Material type for empty space
            progress_callback: Optional progress callback (0.0-1.0)
            
        Returns:
            CTVolume with reconstructed HU values
        """
        from ..ct_simulator import CTVolume
        
        start_time = time.perf_counter()
        
        # Get physical materials
        phys_material = material_type_to_physical(material.value)
        phys_background = material_type_to_physical(background.value)
        
        if phys_material is None:
            logging.warning(f"No physics data for {material.value}, using bone_cortical")
            phys_material = PHYSICAL_MATERIALS["bone_cortical"]
        if phys_background is None:
            phys_background = PHYSICAL_MATERIALS["air"]
        
        # Prepare energy bins (coarse for speed)
        energy_edges = np.linspace(
            self._spectrum.energies[0],
            self._spectrum.energies[-1],
            self.config.energy_bins + 1
        )
        bin_centers = (energy_edges[:-1] + energy_edges[1:]) / 2
        
        # Pre-calculate attenuation coefficients per energy bin
        mu_object = phys_material.get_mu_array(bin_centers)
        mu_background = phys_background.get_mu_array(bin_centers)
        
        # Bin the spectrum
        bin_weights = np.zeros(self.config.energy_bins)
        for i in range(self.config.energy_bins):
            mask = (self._spectrum.energies >= energy_edges[i]) & \
                   (self._spectrum.energies < energy_edges[i+1])
            bin_weights[i] = np.sum(self._spectrum.photons[mask])
        bin_weights /= np.sum(bin_weights)  # Normalize
        
        logging.info(f"Using {self.config.energy_bins} energy bins")
        
        # Process each slice
        num_slices = voxel_grid.data.shape[2]
        reconstructed = np.zeros_like(voxel_grid.data, dtype=np.float32)
        
        for slice_idx in range(num_slices):
            slice_2d = voxel_grid.data[:, :, slice_idx]
            
            # Reconstruct this slice
            recon_slice = self._simulate_slice(
                slice_2d,
                mu_object, mu_background,
                bin_weights, bin_centers
            )
            
            reconstructed[:, :, slice_idx] = recon_slice
            
            if progress_callback:
                progress_callback((slice_idx + 1) / num_slices)
        
        # Convert to standard CT orientation (Z, Y, X)
        ct_data = np.transpose(reconstructed, (2, 1, 0))
        
        elapsed = time.perf_counter() - start_time
        logging.info(f"Physical simulation completed in {elapsed:.2f}s")
        
        return CTVolume(
            data=ct_data,
            voxel_size=voxel_grid.voxel_size,
            origin=voxel_grid.origin
        )
    
    def _simulate_slice(
        self,
        binary_slice: np.ndarray,
        mu_object: np.ndarray,
        mu_background: np.ndarray,
        bin_weights: np.ndarray,
        bin_centers: np.ndarray
    ) -> np.ndarray:
        """
        Simulate CT for a single 2D slice with polychromatic physics.
        
        Returns reconstructed HU values.
        """
        from skimage.transform import radon, iradon
        
        original_shape = binary_slice.shape
        
        # Pad to square for proper radon/iradon transform
        max_dim = max(original_shape)
        pad_h = max_dim - original_shape[0]
        pad_w = max_dim - original_shape[1]
        
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        if pad_h > 0 or pad_w > 0:
            padded_slice = np.pad(
                binary_slice,
                ((pad_top, pad_bottom), (pad_left, pad_right)),
                mode='constant',
                constant_values=0
            )
        else:
            padded_slice = binary_slice
        
        # Calculate path length through material using Radon transform
        path_lengths = radon(padded_slice.astype(np.float64), theta=self.theta)
        
        # Polychromatic projection
        max_path = path_lengths.max() * 1.1 if path_lengths.max() > 0 else 100
        
        # Initialize transmitted intensity
        I_transmitted = np.zeros_like(path_lengths)
        
        # Sum over energy bins
        for i, (weight, E) in enumerate(zip(bin_weights, bin_centers)):
            if weight <= 0:
                continue
                
            mu_obj = mu_object[i]
            mu_bg = mu_background[i]
            
            L_object = path_lengths
            L_total = max_path
            L_bg = np.maximum(L_total - L_object, 0)
            
            # Beer-Lambert: I = I0 * exp(-μ*L)
            attenuation = np.exp(-mu_obj * L_object - mu_bg * L_bg)
            I_transmitted += weight * attenuation
        
        # Photon noise (Poisson statistics)
        I_counts = I_transmitted * self.config.photon_count
        I_counts = np.maximum(I_counts, 1)
        
        # Apply Poisson noise
        I_noisy = np.random.poisson(I_counts.astype(int)).astype(np.float64)
        I_noisy = np.maximum(I_noisy, 1)
        
        # Log transform to get projection values
        I_ratio = I_noisy / self.config.photon_count
        sinogram = -np.log(np.maximum(I_ratio, 1e-10))
        
        # Filtered back projection - output_size must match padded input
        reconstructed = iradon(sinogram, theta=self.theta, filter_name='ramp', output_size=max_dim)
        
        # Crop back to original shape
        if pad_h > 0 or pad_w > 0:
            reconstructed = reconstructed[
                pad_top:pad_top + original_shape[0],
                pad_left:pad_left + original_shape[1]
            ]
        
        # Convert to Hounsfield Units
        hu_slice = 1000 * (reconstructed - self._mu_water_effective) / self._mu_water_effective
        
        return hu_slice.astype(np.float32)
    
    @property
    def spectrum(self) -> XRaySpectrum:
        """Get the current X-ray spectrum."""
        return self._spectrum
    
    def set_kvp(self, kvp: int) -> None:
        """Update tube voltage and regenerate spectrum."""
        self.config.kvp = kvp
        self._spectrum = self._spectrum_gen.generate(
            kvp=kvp,
            filtration_mm_al=self.config.filtration_mm_al
        )
        self._mu_water_effective = self._calculate_effective_mu("water")
        logging.info(f"kVp updated to {kvp}, mean energy: {self._spectrum.mean_energy:.1f} keV")
