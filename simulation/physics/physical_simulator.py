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
import threading
import concurrent.futures
import numpy as np

# Optional CuPy import for GPU acceleration
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


from ..voxelizer import VoxelGrid
from ..materials import MaterialType
from .spectrum import SpectrumGenerator, XRaySpectrum
from .attenuation import get_attenuation_database
from .physical_material import (
    PhysicalMaterial, 
    PHYSICAL_MATERIALS,
    PHYSICAL_MATERIALS,
    material_type_to_physical
)
from ..backends.radon_kernels import GPURadonTransform, HAS_CUPY as RADON_HAS_CUPY
from ..backends import get_backend


@dataclass
class PhysicsConfig:
    """
    Configuration for physical CT simulation.
    
    Attributes:
        kvp: Tube voltage in kV (60-200)
        tube_current_ma: Tube current in mA (affects photon count/noise)
        filtration_mm_al: Aluminum equivalent filtration in mm
        photon_count_base: Base photon count at 200mA per detector element
        energy_bins: Number of energy bins for spectral integration
        use_gpu: Whether to use GPU acceleration
        enable_scatter: Whether to simulate scatter (not implemented)
    """
    kvp: int = 120
    tube_current_ma: int = 200  # mA
    filtration_mm_al: float = 2.5
    photon_count_base: float = 1e5  # Photons per detector element at 200mA
    energy_bins: int = 10
    use_gpu: bool = False
    enable_scatter: bool = False
    
    @property
    def photon_count(self) -> float:
        """Effective photon count scaled by tube current."""
        return self.photon_count_base * (self.tube_current_ma / 200.0)


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
        
        # Initialize backend (defaults to CPU if not using GPU path)
        self._backend = get_backend(use_gpu=False)
    
    
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
        from ..volume import CTVolume
        
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
        
        # Decide GPU vs CPU
        use_gpu = self.config.use_gpu and HAS_CUPY
        if self.config.use_gpu and not HAS_CUPY:
            logging.warning("GPU requested but CuPy not available, falling back to CPU")
        
        # Process slices - output will be square based on max(H, W)
        num_slices = voxel_grid.data.shape[2]
        h, w = voxel_grid.data.shape[:2]
        output_size = max(h, w)
        
        if use_gpu:
            # GPU path: delegate to GPUSimulator
            logging.info("Using GPU for physics simulation")
            from .gpu_simulator import GPUSimulator
            
            gpu_sim = GPUSimulator(self.theta)
            reconstructed = gpu_sim.simulate_volume(
                volume_data=voxel_grid.data,
                mu_object=mu_object,
                mu_background=mu_background,
                bin_weights=bin_weights,
                bin_centers=bin_centers,
                photon_count=self.config.photon_count,
                mu_water_effective=self._mu_water_effective,
                progress_callback=progress_callback
            )

        else:
            # CPU path: parallel threads
            reconstructed = np.zeros((output_size, output_size, num_slices), dtype=np.float32)
            progress_lock = threading.Lock()
            completed_slices = 0
            
            def process_slice(slice_idx: int):
                nonlocal completed_slices
                slice_2d = voxel_grid.data[:, :, slice_idx]
                recon_slice, _ = self._simulate_slice(
                    slice_2d,
                    mu_object, mu_background,
                    bin_weights, bin_centers
                )
                return slice_idx, recon_slice
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_idx = {
                    executor.submit(process_slice, i): i
                    for i in range(num_slices)
                }
                
                for future in concurrent.futures.as_completed(future_to_idx):
                    slice_idx, recon_slice = future.result()
                    reconstructed[:, :, slice_idx] = recon_slice
                    
                    with progress_lock:
                        completed_slices += 1
                        if progress_callback:
                            progress_callback(completed_slices / num_slices)
        
        # Convert to standard CT orientation (Z, Y, X)
        ct_data = np.transpose(reconstructed, (2, 1, 0))
        
        elapsed = time.perf_counter() - start_time
        mode = "GPU" if use_gpu else "CPU"
        logging.info(f"Physical simulation ({mode}) completed in {elapsed:.2f}s")
        
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
        
        Output is always square based on the longer edge for consistent CT geometry.
        Returns reconstructed HU values.
        """
        from skimage.transform import radon, iradon
        
        original_shape = binary_slice.shape
        
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
                binary_slice,
                ((sq_pad_top, pad_to_square_h - sq_pad_top), 
                 (sq_pad_left, pad_to_square_w - sq_pad_left)),
                mode='constant',
                constant_values=0
            )
        else:
            square_slice = binary_slice
        
        # Then pad to diagonal for radon transform
        diag_pad = diag_len - output_size
        diag_pad_half = diag_pad // 2
        
        if diag_pad > 0:
            padded_slice = np.pad(
                square_slice,
                ((diag_pad_half, diag_pad - diag_pad_half), 
                 (diag_pad_half, diag_pad - diag_pad_half)),
                mode='constant',
                constant_values=0
            )
        else:
            padded_slice = square_slice
            diag_pad_half = 0
        
        # Calculate path length through material using Radon transform
        # Use CPU backend
        path_lengths = self._backend.radon(padded_slice.astype(np.float64), theta=self.theta)
        
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
        
        # Filtered back projection
        reconstructed = self._backend.iradon(sinogram, theta=self.theta, output_size=diag_len)
        
        # Crop back to output_size x output_size (square)
        if diag_pad > 0:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half + output_size,
                diag_pad_half:diag_pad_half + output_size
            ]
        
        # Convert to Hounsfield Units
        hu_slice = 1000 * (reconstructed - self._mu_water_effective) / self._mu_water_effective
        
        return hu_slice.astype(np.float32), output_size
    
    
    
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

