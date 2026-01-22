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
        
        # GPU memory detection for dynamic batch sizing
        self._gpu_total_mem = 0
        self._gpu_free_mem = 0
        if self.config.use_gpu and HAS_CUPY:
            self._detect_gpu_memory()
        
        logging.info(
            f"PhysicalCTSimulator initialized: {self.config.kvp} kVp, "
            f"{self.config.filtration_mm_al} mm Al, "
            f"mean energy: {self._spectrum.mean_energy:.1f} keV"
        )
        
        # Initialize backend (defaults to CPU if not using GPU path)
        self._backend = get_backend(use_gpu=False)
    
    def _detect_gpu_memory(self) -> None:
        """Detect GPU memory and log available VRAM."""
        try:
            mempool = cp.get_default_memory_pool()
            # Get device memory info
            device = cp.cuda.Device()
            self._gpu_total_mem = device.mem_info[1]  # Total memory in bytes
            self._gpu_free_mem = device.mem_info[0]   # Free memory in bytes
            
            total_gb = self._gpu_total_mem / (1024**3)
            free_gb = self._gpu_free_mem / (1024**3)
            logging.info(f"  GPU Memory: {free_gb:.1f}/{total_gb:.1f} GB available")
        except Exception as e:
            logging.warning(f"Could not detect GPU memory: {e}")
            self._gpu_total_mem = 8 * (1024**3)  # Default 8GB
            self._gpu_free_mem = 6 * (1024**3)   # Conservative estimate
    
    def _calculate_batch_sizes(self, image_size: int) -> Tuple[int, int, int]:
        """
        Calculate optimal batch sizes based on available GPU memory and image size.
        
        Returns:
            (slice_batch, radon_batch, iradon_batch)
        """
        # Memory estimation per operation (empirical)
        # Radon: ~6 arrays of (batch_angles, n_det, n_det) float32 = batch * n_det^2 * 24 bytes
        # iRadon: ~4 arrays of (batch_angles, output_size^2) float32 = batch * size^2 * 16 bytes
        
        n_det = int(np.ceil(np.sqrt(2) * image_size)) + 32  # Diagonal + padding
        
        # Use 60% of free memory as safety margin
        available = self._gpu_free_mem * 0.6 if self._gpu_free_mem > 0 else 4 * (1024**3)
        
        # Calculate Radon batch size
        # Each angle needs ~6 arrays of (n_det, n_det) floats
        mem_per_radon_angle = n_det * n_det * 6 * 4  # bytes
        radon_batch = max(5, min(60, int(available / (mem_per_radon_angle * 4))))
        
        # Calculate iRadon batch size  
        # Each angle needs ~4 arrays of (output_size^2) floats
        mem_per_iradon_angle = image_size * image_size * 4 * 4  # bytes
        iradon_batch = max(10, min(120, int(available / (mem_per_iradon_angle * 2))))
        
        # Calculate slice batch size
        # Each slice needs ~2x the slice data + intermediate buffers
        mem_per_slice = image_size * image_size * 4 * 10  # Conservative estimate
        slice_batch = max(1, min(16, int(available / (mem_per_slice * 8))))
        
        logging.info(f"  Auto batch sizes: slices={slice_batch}, radon={radon_batch}, iradon={iradon_batch}")
        
        return slice_batch, radon_batch, iradon_batch
    
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
        GPU-accelerated simulation for all slices with batch processing.
        
        Uses CuPy for vectorized Radon/iRadon transforms.
        Processes slices in batches to reduce kernel launch overhead.
        Output is always square (based on max of H, W).
        """
        num_slices = volume_data.shape[2]
        h, w = volume_data.shape[:2]
        
        # Output will be square based on longer edge
        output_size = max(h, w)
        
        # Calculate dynamic batch sizes based on GPU memory
        slice_batch, radon_batch, iradon_batch = self._calculate_batch_sizes(output_size)
        
        # Initialize GPU Radon transform with calculated batch sizes
        self._radon_transform = GPURadonTransform(radon_batch=radon_batch, iradon_batch=iradon_batch)
        
        # Pre-compute on GPU (cached for all slices)
        mu_obj_gpu = cp.asarray(mu_object, dtype=cp.float32)
        mu_bg_gpu = cp.asarray(mu_background, dtype=cp.float32)
        weights_gpu = cp.asarray(bin_weights, dtype=cp.float32)
        theta_gpu = cp.asarray(self.theta, dtype=cp.float32)
        photon_count = self.config.photon_count
        
        # Pre-compute geometry arrays (OPTIMIZATION: only once per simulation)
        theta_rad = theta_gpu * (cp.pi / 180.0)
        cos_theta = cp.cos(theta_rad)
        sin_theta = cp.sin(theta_rad)
        
        # Cache geometry in instance for batch processing
        self._cached_cos_theta = cos_theta
        self._cached_sin_theta = sin_theta
        
        # Output array (square slices)
        reconstructed = np.zeros((output_size, output_size, num_slices), dtype=np.float32)
        
        # Dynamic batch processing based on detected GPU memory
        BATCH_SIZE = slice_batch
        
        logging.info(f"  GPU batch processing {num_slices} slices (batch_size={BATCH_SIZE})")
        
        # Create streams for overlapped transfer
        compute_stream = cp.cuda.Stream(non_blocking=True)
        transfer_stream = cp.cuda.Stream(non_blocking=True)
        
        completed = 0
        prev_batch_stack = None
        prev_batch_start = 0
        prev_batch_end = 0
        
        for batch_start in range(0, num_slices, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, num_slices)
            batch_slices = batch_end - batch_start
            
            # Transfer batch to GPU (can overlap with previous D2H)
            with transfer_stream:
                batch_data = cp.asarray(volume_data[:, :, batch_start:batch_end], dtype=cp.float32)
            
            # Wait for transfer to complete before compute
            compute_stream.wait_event(transfer_stream.record())
            
            # Process batch on compute stream
            with compute_stream:
                batch_results = []
                for i in range(batch_slices):
                    slice_2d = batch_data[:, :, i]
                    recon_gpu, _ = self._simulate_slice_gpu(
                        slice_2d, mu_obj_gpu, mu_bg_gpu,
                        weights_gpu, theta_gpu, photon_count
                    )
                    batch_results.append(recon_gpu)
                
                batch_stack = cp.stack(batch_results, axis=2)
            
            # Transfer previous batch back to CPU while current batch computes
            if prev_batch_stack is not None:
                with transfer_stream:
                    # Wait for previous compute to finish
                    transfer_stream.wait_event(compute_stream.record())
                    reconstructed[:, :, prev_batch_start:prev_batch_end] = cp.asnumpy(prev_batch_stack)
                    del prev_batch_stack
            
            prev_batch_stack = batch_stack
            prev_batch_start = batch_start
            prev_batch_end = batch_end
            
            # Clean up current batch input
            del batch_data, batch_results
            
            completed += batch_slices
            if progress_callback:
                progress_callback(completed / num_slices)
        
        # Transfer final batch
        if prev_batch_stack is not None:
            compute_stream.synchronize()
            reconstructed[:, :, prev_batch_start:prev_batch_end] = cp.asnumpy(prev_batch_stack)
            del prev_batch_stack
        
        # Final memory cleanup
        cp.get_default_memory_pool().free_all_blocks()
        
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
        
        # Forward projection (Radon) - using extracted kernel
        path_lengths = self._radon_transform.radon(padded_slice, theta_gpu)
        
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
        
        # Inverse Radon (FBP) using extracted kernel
        reconstructed = self._radon_transform.iradon(sinogram, theta_gpu, diag_len)
        
        # Crop back to output_size x output_size (square)
        if diag_pad > 0:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half + output_size,
                diag_pad_half:diag_pad_half + output_size
            ]
        
        # Convert to HU
        hu_slice = 1000 * (reconstructed - self._mu_water_effective) / self._mu_water_effective
        
        return hu_slice.astype(cp.float32), output_size
    
    # NOTE: _radon_gpu and _iradon_gpu methods have been extracted to
    # simulation/backends/radon_kernels.py (GPURadonTransform class)
    
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

