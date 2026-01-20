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
    material_type_to_physical
)


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
        
        # Decide GPU vs CPU
        use_gpu = self.config.use_gpu and HAS_CUPY
        if self.config.use_gpu and not HAS_CUPY:
            logging.warning("GPU requested but CuPy not available, falling back to CPU")
        
        # Process slices - output will be square based on max(H, W)
        num_slices = voxel_grid.data.shape[2]
        h, w = voxel_grid.data.shape[:2]
        output_size = max(h, w)
        
        if use_gpu:
            # GPU path: batch process on GPU
            logging.info("Using GPU for physics simulation")
            reconstructed = self._simulate_gpu(
                voxel_grid.data, mu_object, mu_background,
                bin_weights, bin_centers, progress_callback
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
        path_lengths = radon(padded_slice.astype(np.float64), theta=self.theta, circle=False)
        
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
        reconstructed = iradon(sinogram, theta=self.theta, filter_name='ramp', output_size=diag_len, circle=False)
        
        # Crop back to output_size x output_size (square)
        if diag_pad > 0:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half + output_size,
                diag_pad_half:diag_pad_half + output_size
            ]
        
        # Convert to Hounsfield Units
        hu_slice = 1000 * (reconstructed - self._mu_water_effective) / self._mu_water_effective
        
        return hu_slice.astype(np.float32), output_size
    
    def _simulate_gpu(
        self,
        volume_data: np.ndarray,
        mu_object: np.ndarray,
        mu_background: np.ndarray,
        bin_weights: np.ndarray,
        bin_centers: np.ndarray,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> np.ndarray:
        """
        GPU-accelerated simulation for all slices.
        
        Uses CuPy for vectorized Radon/iRadon transforms.
        Output is always square (based on max of H, W).
        """
        num_slices = volume_data.shape[2]
        h, w = volume_data.shape[:2]
        
        # Output will be square based on longer edge
        output_size = max(h, w)
        
        # Pre-compute on GPU
        mu_obj_gpu = cp.asarray(mu_object, dtype=cp.float32)
        mu_bg_gpu = cp.asarray(mu_background, dtype=cp.float32)
        weights_gpu = cp.asarray(bin_weights, dtype=cp.float32)
        theta_gpu = cp.asarray(self.theta, dtype=cp.float32)
        photon_count = self.config.photon_count
        
        # Output array (square slices)
        reconstructed = np.zeros((output_size, output_size, num_slices), dtype=np.float32)
        
        # Process slices (could batch for even more speed)
        for slice_idx in range(num_slices):
            slice_2d = cp.asarray(volume_data[:, :, slice_idx], dtype=cp.float32)
            
            # GPU slice simulation (returns square slice)
            recon_gpu, _ = self._simulate_slice_gpu(
                slice_2d, mu_obj_gpu, mu_bg_gpu,
                weights_gpu, theta_gpu, photon_count
            )
            
            # Transfer result back
            reconstructed[:, :, slice_idx] = cp.asnumpy(recon_gpu)
            
            if progress_callback:
                progress_callback((slice_idx + 1) / num_slices)
        
        return reconstructed
    
    def _simulate_slice_gpu(
        self,
        slice_gpu: "cp.ndarray",
        mu_object: "cp.ndarray",
        mu_background: "cp.ndarray",
        bin_weights: "cp.ndarray",
        theta_gpu: "cp.ndarray",
        photon_count: float
    ) -> "cp.ndarray":
        """
        Simulate CT for a single slice on GPU with polychromatic physics.
        
        Output is always a square based on the longer edge for consistent CT geometry.
        """
        original_shape = slice_gpu.shape
        
        # Output will be a square based on the longer edge
        output_size = max(original_shape[0], original_shape[1])
        
        # Pad to diagonal size for proper radon/iradon transform
        # Add safety margin to prevent edge artifacts
        diag_len = int(cp.ceil(cp.sqrt(output_size**2 + output_size**2))) + 32
        
        # First pad to square (output_size x output_size)
        pad_to_square_h = output_size - original_shape[0]
        pad_to_square_w = output_size - original_shape[1]
        sq_pad_top = pad_to_square_h // 2
        sq_pad_left = pad_to_square_w // 2
        
        if pad_to_square_h > 0 or pad_to_square_w > 0:
            square_slice = cp.pad(
                slice_gpu,
                ((sq_pad_top, pad_to_square_h - sq_pad_top), 
                 (sq_pad_left, pad_to_square_w - sq_pad_left)),
                mode='constant',
                constant_values=0.0
            )
        else:
            square_slice = slice_gpu
        
        # Then pad to diagonal for radon transform
        diag_pad = diag_len - output_size
        diag_pad_half = diag_pad // 2
        
        if diag_pad > 0:
            padded_slice = cp.pad(
                square_slice,
                ((diag_pad_half, diag_pad - diag_pad_half), 
                 (diag_pad_half, diag_pad - diag_pad_half)),
                mode='constant',
                constant_values=0.0
            )
        else:
            padded_slice = square_slice
            diag_pad_half = 0
        
        # Forward projection (Radon) - vectorized on GPU
        path_lengths = self._radon_gpu(padded_slice, theta_gpu)
        
        max_path = float(path_lengths.max()) * 1.1 if path_lengths.max() > 0 else 100.0
        
        # Polychromatic Beer-Lambert (vectorized over energy bins)
        I_transmitted = cp.zeros_like(path_lengths)
        n_bins = len(bin_weights)
        
        for i in range(n_bins):
            weight = bin_weights[i]
            if weight <= 0:
                continue
            
            mu_obj = mu_object[i]
            mu_bg = mu_background[i]
            
            L_object = path_lengths
            L_bg = cp.maximum(max_path - L_object, 0)
            
            attenuation = cp.exp(-mu_obj * L_object - mu_bg * L_bg)
            I_transmitted += weight * attenuation
        
        # Poisson noise
        I_counts = I_transmitted * photon_count
        I_counts = cp.maximum(I_counts, 1.0)
        
        # CuPy Poisson requires integer lambda
        I_noisy = cp.random.poisson(I_counts.astype(cp.int64)).astype(cp.float32)
        I_noisy = cp.maximum(I_noisy, 1.0)
        
        # Log transform
        I_ratio = I_noisy / photon_count
        sinogram = -cp.log(cp.maximum(I_ratio, 1e-10))
        
        # Inverse Radon (FBP) on GPU
        reconstructed = self._iradon_gpu(sinogram, theta_gpu, diag_len)
        
        # Crop back to output_size x output_size (square)
        if diag_pad > 0:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half + output_size,
                diag_pad_half:diag_pad_half + output_size
            ]
        
        # Convert to HU
        hu_slice = 1000 * (reconstructed - self._mu_water_effective) / self._mu_water_effective
        
        return hu_slice.astype(cp.float32), output_size
    
    def _radon_gpu(self, image_gpu: "cp.ndarray", theta_deg: "cp.ndarray") -> "cp.ndarray":
        """Vectorized GPU Radon transform with diagonal-based detector size."""
        H, W = image_gpu.shape
        
        # Compute detector count based on image diagonal (matching skimage behavior)
        n_det = int(cp.ceil(cp.sqrt(H**2 + W**2)))
        n_angles = len(theta_deg)
        
        theta_rad = theta_deg * (cp.pi / 180.0)
        
        # Image center
        img_center_x = (W - 1) / 2.0
        img_center_y = (H - 1) / 2.0
        
        # Detector coordinates centered at 0
        det_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        
        # Ray path coordinates (s along the ray direction)
        # Must use n_det (diagonal length) to ensure rays fully traverse the image at all angles
        s_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        
        # Reshape for broadcasting: (n_angles, n_det, n_s)
        theta = theta_rad.reshape(-1, 1, 1)
        t = det_coords.reshape(1, -1, 1)
        s = s_coords.reshape(1, 1, -1)
        
        cos_t = cp.cos(theta)
        sin_t = cp.sin(theta)
        
        # Compute sampling coordinates in image space
        # Standard Radon transform coordinate mapping:
        # x = t * cos(theta) - s * sin(theta)
        # y = t * sin(theta) + s * cos(theta)
        # s is the integration variable (along the ray)
        # t is the detector position (perpendicular to ray)
        x_coords = t * cos_t - s * sin_t + img_center_x
        y_coords = t * sin_t + s * cos_t + img_center_y
        
        # Bilinear interpolation
        x0 = cp.floor(x_coords).astype(cp.int32)
        y0 = cp.floor(y_coords).astype(cp.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        
        wx = x_coords - x0.astype(cp.float32)
        wy = y_coords - y0.astype(cp.float32)
        
        x0_clamped = cp.clip(x0, 0, W - 1)
        x1_clamped = cp.clip(x1, 0, W - 1)
        y0_clamped = cp.clip(y0, 0, H - 1)
        y1_clamped = cp.clip(y1, 0, H - 1)
        
        valid_mask = ((x0 >= 0) & (x1 < W) & (y0 >= 0) & (y1 < H)).astype(cp.float32)
        
        v00 = image_gpu[y0_clamped, x0_clamped]
        v01 = image_gpu[y0_clamped, x1_clamped]
        v10 = image_gpu[y1_clamped, x0_clamped]
        v11 = image_gpu[y1_clamped, x1_clamped]
        
        samples = (
            v00 * (1 - wx) * (1 - wy) +
            v01 * wx * (1 - wy) +
            v10 * (1 - wx) * wy +
            v11 * wx * wy
        ) * valid_mask
        
        # Sum along ray path and transpose to (n_det, n_angles)
        return samples.sum(axis=2).T.astype(cp.float32)
    
    def _iradon_gpu(self, sinogram_gpu: "cp.ndarray", theta_deg: "cp.ndarray", output_size: int) -> "cp.ndarray":
        """Vectorized GPU inverse Radon (filtered back projection)."""
        n_angles = len(theta_deg)
        n_det = sinogram_gpu.shape[0]
        
        # Ramp filter
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
        pad_width = ((0, projection_size_padded - n_det), (0, 0))
        padded_sino = cp.pad(sinogram_gpu, pad_width, mode='constant')
        
        f = cp.fft.fftfreq(projection_size_padded).reshape(-1, 1)
        ramp = cp.abs(f)
        
        f_sino = cp.fft.fft(padded_sino, axis=0)
        filtered_sino = cp.fft.ifft(f_sino * ramp, axis=0).real
        filtered_sino = filtered_sino[:n_det, :].astype(cp.float32)
        
        # Backprojection
        center = (output_size - 1) / 2.0
        grid_1d = cp.arange(output_size, dtype=cp.float32) - center
        y_grid, x_grid = cp.meshgrid(grid_1d, grid_1d, indexing='ij')
        
        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        
        theta_rad = theta_deg * (cp.pi / 180.0)
        cos_t = cp.cos(theta_rad).reshape(-1, 1)
        sin_t = cp.sin(theta_rad).reshape(-1, 1)
        
        t_all = x_flat * cos_t + y_flat * sin_t
        det_center = (n_det - 1) / 2.0
        t_idx = t_all + det_center
        
        t_idx_floor = cp.floor(t_idx).astype(cp.int32)
        t_idx_ceil = t_idx_floor + 1
        t_frac = t_idx - t_idx_floor.astype(cp.float32)
        
        t_idx_floor = cp.clip(t_idx_floor, 0, n_det - 1)
        t_idx_ceil = cp.clip(t_idx_ceil, 0, n_det - 1)
        
        angle_idx = cp.arange(n_angles, dtype=cp.int32).reshape(-1, 1)
        angle_idx = cp.broadcast_to(angle_idx, t_idx.shape)
        
        val_floor = filtered_sino[t_idx_floor, angle_idx]
        val_ceil = filtered_sino[t_idx_ceil, angle_idx]
        
        sampled = val_floor * (1 - t_frac) + val_ceil * t_frac
        recon_flat = sampled.sum(axis=0)
        recon = recon_flat.reshape(output_size, output_size)
        
        return recon * (cp.pi / (2 * n_angles))
    
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

