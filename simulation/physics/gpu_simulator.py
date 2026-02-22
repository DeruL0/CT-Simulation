"""
GPU CT Simulation Strategy

Handles GPU-accelerated physical CT simulation logic:
- GPU memory detection and management
- Dynamic batch sizing
- Volume-domain scatter integration
"""

import logging
from typing import Optional, Callable, Tuple, List
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from ..backends.radon_kernels import GPURadonTransform
from .scatter import ConvolutionScatter, MotionBlur


class GPUSimulator:
    """
    GPU implementation of Physical CT Simulation.

    Handles memory management and batch processing for GPU acceleration.
    """

    def __init__(
        self,
        theta: np.ndarray,
        enable_scatter: bool = False,
        scatter_fraction: float = 0.15,
        scatter_kernel_sigma: float = 30.0,
        enable_motion_blur: bool = False,
        motion_blur_angle: float = 1.0,
    ):
        """
        Initialize GPU simulator.

        Args:
            theta: Projection angles in degrees
            enable_scatter: Whether to simulate X-ray scatter
            scatter_fraction: Scatter-to-Primary Ratio (SPR)
            scatter_kernel_sigma: Scatter kernel width in pixels
            enable_motion_blur: Whether to simulate gantry motion blur
            motion_blur_angle: Integration angle in degrees
        """
        self.theta = theta
        self._gpu_total_mem = 0
        self._gpu_free_mem = 0
        self._radon_transform = None

        self.enable_scatter = enable_scatter
        self.enable_motion_blur = enable_motion_blur

        if enable_scatter and HAS_CUPY:
            self._scatter_model = ConvolutionScatter(
                scatter_fraction=scatter_fraction,
                kernel_sigma=scatter_kernel_sigma,
                use_gpu=True,
            )
            logging.info("  Scatter enabled: SPR=%s, sigma=%s", scatter_fraction, scatter_kernel_sigma)
        else:
            self._scatter_model = None

        if enable_motion_blur and HAS_CUPY:
            self._motion_blur = MotionBlur(
                blur_angle_deg=motion_blur_angle,
                use_gpu=True,
            )
            self._angular_step = 180.0 / len(theta) if len(theta) > 0 else 1.0
            logging.info("  Motion blur enabled: angle=%s deg", motion_blur_angle)
        else:
            self._motion_blur = None
            self._angular_step = 1.0

        if HAS_CUPY:
            self._detect_memory()

    def _detect_memory(self) -> None:
        """Detect GPU memory and calculate available VRAM."""
        try:
            device = cp.cuda.Device()
            self._gpu_total_mem = device.mem_info[1]
            self._gpu_free_mem = device.mem_info[0]

            total_gb = self._gpu_total_mem / (1024 ** 3)
            free_gb = self._gpu_free_mem / (1024 ** 3)
            logging.info("  GPU Simulator: %.1f/%.1f GB VRAM available", free_gb, total_gb)
        except Exception as e:
            logging.warning("Could not detect GPU memory: %s", e)
            self._gpu_total_mem = 8 * (1024 ** 3)
            self._gpu_free_mem = 6 * (1024 ** 3)

    def _batch_ranges(self, count: int, batch_size: int) -> List[Tuple[int, int]]:
        """Build [start, end) ranges for a given batch size."""
        return [
            (batch_start, min(batch_start + batch_size, count))
            for batch_start in range(0, count, batch_size)
        ]

    @staticmethod
    def _halve_batch_sizes(
        slice_batch: int,
        radon_batch: int,
        iradon_batch: int,
    ) -> Tuple[int, int, int]:
        """Conservative OOM fallback: halve all batch dimensions."""
        return (
            max(1, slice_batch // 2),
            max(2, radon_batch // 2),
            max(4, iradon_batch // 2),
        )

    def _probe_batch_configuration(
        self,
        image_size: int,
        slice_batch: int,
        radon_batch: int,
        iradon_batch: int,
    ) -> bool:
        """
        Probe whether a batch configuration can be allocated.

        This probe intentionally allocates representative temporary buffers.
        If it fails with OOM, callers should reduce batches by half.
        """
        n_det = int(np.ceil(np.sqrt(2.0) * image_size)) + 32
        probe_buffers = []
        try:
            probe_buffers.append(cp.empty((image_size, image_size, slice_batch), dtype=cp.float32))
            probe_buffers.append(cp.empty((radon_batch, n_det, n_det), dtype=cp.float32))
            probe_buffers.append(cp.empty((iradon_batch, image_size, image_size), dtype=cp.float32))
            return True
        except cp.cuda.memory.OutOfMemoryError:
            return False
        finally:
            del probe_buffers

    def _to_pinned_float32(
        self,
        volume_data: np.ndarray,
    ) -> Tuple[np.ndarray, Optional["cp.cuda.PinnedMemory"]]:
        """
        Create a pinned float32 host copy for H2D transfers.
        """
        host_f32 = np.ascontiguousarray(volume_data, dtype=np.float32)
        try:
            pinned_mem = cp.cuda.alloc_pinned_memory(host_f32.nbytes)
            pinned_array = np.frombuffer(
                pinned_mem,
                dtype=np.float32,
                count=host_f32.size,
            ).reshape(host_f32.shape)
            np.copyto(pinned_array, host_f32)
            return pinned_array, pinned_mem
        except Exception as e:
            logging.warning("Pinned memory allocation failed, using pageable host memory: %s", e)
            return host_f32, None

    def _calculate_batch_sizes(self, image_size: int) -> Tuple[int, int, int]:
        """
        Calculate optimal batch sizes based on available GPU memory.

        Returns:
            (slice_batch, radon_batch, iradon_batch)
        """
        self._detect_memory()

        n_det = int(np.ceil(np.sqrt(2) * image_size)) + 32
        available = self._gpu_free_mem * 0.7 if self._gpu_free_mem > 0 else 4 * (1024 ** 3)

        mem_per_radon_angle = n_det * n_det * 6 * 4
        radon_batch = max(5, min(60, int(available / (mem_per_radon_angle * 4))))

        mem_per_iradon_angle = image_size * image_size * 4 * 4
        iradon_batch = max(10, min(120, int(available / (mem_per_iradon_angle * 2))))

        mem_per_slice = image_size * image_size * 4 * 10
        slice_batch = max(1, min(16, int(available / (mem_per_slice * 8))))

        while True:
            try:
                if self._probe_batch_configuration(image_size, slice_batch, radon_batch, iradon_batch):
                    break

                new_slice_batch, new_radon_batch, new_iradon_batch = self._halve_batch_sizes(
                    slice_batch, radon_batch, iradon_batch
                )
                if (new_slice_batch, new_radon_batch, new_iradon_batch) == (
                    slice_batch, radon_batch, iradon_batch
                ):
                    break

                logging.warning(
                    "  Batch probe OOM. Halving batches: slices %d->%d, radon %d->%d, iradon %d->%d",
                    slice_batch,
                    new_slice_batch,
                    radon_batch,
                    new_radon_batch,
                    iradon_batch,
                    new_iradon_batch,
                )
                slice_batch, radon_batch, iradon_batch = (
                    new_slice_batch,
                    new_radon_batch,
                    new_iradon_batch,
                )
            except cp.cuda.memory.OutOfMemoryError:
                new_slice_batch, new_radon_batch, new_iradon_batch = self._halve_batch_sizes(
                    slice_batch, radon_batch, iradon_batch
                )
                if (new_slice_batch, new_radon_batch, new_iradon_batch) == (
                    slice_batch, radon_batch, iradon_batch
                ):
                    break
                slice_batch, radon_batch, iradon_batch = (
                    new_slice_batch,
                    new_radon_batch,
                    new_iradon_batch,
                )

        logging.info(
            "  Auto batch sizes: slices=%d, radon=%d, iradon=%d",
            slice_batch,
            radon_batch,
            iradon_batch,
        )
        return slice_batch, radon_batch, iradon_batch

    @staticmethod
    def _projection_geometry(output_size: int) -> Tuple[int, int, int]:
        """Compute projection geometry constants for all slices in the volume."""
        diag_len = int(np.ceil(np.sqrt(output_size ** 2 + output_size ** 2))) + 32
        diag_pad_half = (diag_len - output_size) // 2
        n_det = int(np.ceil(np.sqrt(diag_len ** 2 + diag_len ** 2)))
        return diag_len, diag_pad_half, n_det

    @staticmethod
    def _pad_slice_for_projection(
        slice_gpu: "cp.ndarray",
        output_size: int,
        diag_len: int,
    ) -> "cp.ndarray":
        """Pad slice to square, then to diagonal extent for Radon transform."""
        original_shape = slice_gpu.shape

        pad_h = output_size - original_shape[0]
        pad_w = output_size - original_shape[1]

        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            left = pad_w // 2
            square_slice = cp.pad(
                slice_gpu,
                ((top, pad_h - top), (left, pad_w - left)),
                mode="constant",
            )
        else:
            square_slice = slice_gpu

        diag_pad = diag_len - output_size
        if diag_pad > 0:
            diag_pad_half = diag_pad // 2
            return cp.pad(
                square_slice,
                ((diag_pad_half, diag_pad - diag_pad_half), (diag_pad_half, diag_pad - diag_pad_half)),
                mode="constant",
            )
        return square_slice

    def _simulate_primary_slice(
        self,
        slice_gpu: "cp.ndarray",
        mu_obj_gpu: "cp.ndarray",
        mu_bg_gpu: "cp.ndarray",
        weights_gpu: "cp.ndarray",
        theta_gpu: "cp.ndarray",
        output_size: int,
        diag_len: int,
        voxel_size_cm: float,
    ) -> Tuple["cp.ndarray", "cp.ndarray"]:
        """
        Compute per-slice primary intensity and path lengths (before scatter/noise/log).
        """
        padded_slice = self._pad_slice_for_projection(slice_gpu, output_size, diag_len)

        path_lengths_px = self._radon_transform.radon(padded_slice, theta_gpu)
        path_lengths = path_lengths_px * voxel_size_cm
        max_path = float(path_lengths.max()) * 1.1 if float(path_lengths.max()) > 0 else 10.0

        i_primary = cp.zeros_like(path_lengths, dtype=cp.float32)
        l_bg = cp.maximum(max_path - path_lengths, 0.0)

        for i in range(int(weights_gpu.size)):
            if float(weights_gpu[i]) <= 0.0:
                continue
            attenuation = cp.exp(-mu_obj_gpu[i] * path_lengths - mu_bg_gpu[i] * l_bg)
            i_primary += weights_gpu[i] * attenuation

        if self._motion_blur is not None:
            i_primary = self._motion_blur.apply(i_primary, self._angular_step)

        return i_primary.astype(cp.float32), path_lengths.astype(cp.float32)

    @staticmethod
    def _apply_noise_and_log(intensity: "cp.ndarray", photon_count: float) -> "cp.ndarray":
        """Apply Poisson/Gaussian photon noise and safe logarithmic transform."""
        i_counts = cp.maximum(intensity * photon_count, 1.0)

        max_count = float(i_counts.max())
        if max_count > 1e7:
            i_noisy = cp.random.normal(i_counts, cp.sqrt(i_counts)).astype(cp.float32)
        else:
            i_noisy = cp.random.poisson(i_counts.astype(cp.int64)).astype(cp.float32)
        i_noisy = cp.maximum(i_noisy, 1.0)

        i_ratio = cp.clip(i_noisy / photon_count, 1e-10, 1.0)
        return -cp.log(i_ratio)

    def _reconstruct_slice(
        self,
        sinogram_gpu: "cp.ndarray",
        theta_gpu: "cp.ndarray",
        diag_len: int,
        diag_pad_half: int,
        output_size: int,
    ) -> "cp.ndarray":
        """Reconstruct one slice from sinogram to absolute linear attenuation (cm^-1)."""
        reconstructed = self._radon_transform.iradon(sinogram_gpu, theta_gpu, diag_len)

        if diag_len > output_size:
            reconstructed = reconstructed[
                diag_pad_half:diag_pad_half + output_size,
                diag_pad_half:diag_pad_half + output_size,
            ]

        return reconstructed.astype(cp.float32)

    def simulate_volume(
        self,
        volume_data: np.ndarray,
        mu_object: np.ndarray,
        mu_background: np.ndarray,
        bin_weights: np.ndarray,
        photon_count: float,
        voxel_size_mm: float,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> np.ndarray:
        """
        Run simulation for entire volume using GPU batch processing.

        The workflow is:
        1. Compute full primary intensity tensor I_primary (n_det, n_angles, n_slices)
        2. Apply scatter in detector-slice projection planes before log transform
        3. Apply noise/log and reconstruct all slices

        Raises:
            RuntimeError: If CuPy is not available
            cp.cuda.memory.OutOfMemoryError: If GPU runs out of memory
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not available for GPU simulation")

        num_slices = volume_data.shape[2]
        h, w = volume_data.shape[:2]
        output_size = max(h, w)
        if num_slices == 0:
            return np.zeros((output_size, output_size, 0), dtype=np.float32)
        voxel_size_cm = max(float(voxel_size_mm), 1e-9) / 10.0

        slice_batch, radon_batch, iradon_batch = self._calculate_batch_sizes(output_size)
        volume_data_host, _pinned_handle = self._to_pinned_float32(np.transpose(volume_data, (2, 0, 1)))

        try:
            mu_obj_gpu = cp.asarray(mu_object, dtype=cp.float32)
            mu_bg_gpu = cp.asarray(mu_background, dtype=cp.float32)
            weights_gpu = cp.asarray(bin_weights, dtype=cp.float32)
            theta_gpu = cp.asarray(self.theta, dtype=cp.float32)
        except cp.cuda.memory.OutOfMemoryError as e:
            logging.error("GPU OOM during initialization: %s", e)
            raise

        diag_len, diag_pad_half, n_det = self._projection_geometry(output_size)
        n_angles = int(theta_gpu.size)

        reconstructed = np.zeros((output_size, output_size, num_slices), dtype=np.float32)

        current_slice_batch = slice_batch
        current_radon_batch = radon_batch
        current_iradon_batch = iradon_batch

        max_oom_retries = 6
        oom_retries = 0

        while True:
            self._radon_transform = GPURadonTransform(
                radon_batch=current_radon_batch,
                iradon_batch=current_iradon_batch,
            )

            logging.info(
                "  GPU volume simulation (slices=%d, slice_batch=%d, radon_batch=%d, iradon_batch=%d)",
                num_slices,
                current_slice_batch,
                current_radon_batch,
                current_iradon_batch,
            )

            try:
                # Stage 1: pure primary intensity accumulation over full slice stack.
                i_primary_host = np.zeros((n_det, n_angles, num_slices), dtype=np.float32)
                path_host = np.zeros_like(i_primary_host) if self._scatter_model is not None else None

                slice_ranges = self._batch_ranges(num_slices, current_slice_batch)
                for batch_start, batch_end in slice_ranges:
                    current_batch_size = batch_end - batch_start
                    batch_gpu = cp.asarray(volume_data_host[batch_start:batch_end, :, :], dtype=cp.float32)

                    for local_idx in range(current_batch_size):
                        slice_idx = batch_start + local_idx
                        i_primary, path_lengths = self._simulate_primary_slice(
                            batch_gpu[local_idx],
                            mu_obj_gpu,
                            mu_bg_gpu,
                            weights_gpu,
                            theta_gpu,
                            output_size,
                            diag_len,
                            voxel_size_cm,
                        )
                        i_primary_host[:, :, slice_idx] = cp.asnumpy(i_primary)
                        if path_host is not None:
                            path_host[:, :, slice_idx] = cp.asnumpy(path_lengths)

                    del batch_gpu

                    if progress_callback:
                        progress_callback(0.45 * (batch_end / num_slices))

                # Stage 2: scatter + noise + log in angle chunks.
                sinogram_host = np.zeros_like(i_primary_host, dtype=np.float32)
                angle_chunk = max(1, min(n_angles, current_radon_batch))
                angle_ranges = self._batch_ranges(n_angles, angle_chunk)

                for angle_start, angle_end in angle_ranges:
                    i_primary_chunk = cp.asarray(
                        i_primary_host[:, angle_start:angle_end, :],
                        dtype=cp.float32,
                    )

                    if self._scatter_model is not None:
                        path_chunk = cp.asarray(
                            path_host[:, angle_start:angle_end, :],
                            dtype=cp.float32,
                        ) if path_host is not None else None
                        i_scatter_chunk = self._scatter_model.compute_scatter(i_primary_chunk, path_chunk)
                        i_total_chunk = i_primary_chunk + i_scatter_chunk
                    else:
                        path_chunk = None
                        i_total_chunk = i_primary_chunk

                    sinogram_chunk = self._apply_noise_and_log(i_total_chunk, photon_count)
                    sinogram_host[:, angle_start:angle_end, :] = cp.asnumpy(sinogram_chunk.astype(cp.float32))

                    del i_primary_chunk, i_total_chunk, sinogram_chunk, path_chunk
                    if self._scatter_model is not None:
                        del i_scatter_chunk

                    if progress_callback and n_angles > 0:
                        progress_callback(0.45 + 0.25 * (angle_end / n_angles))

                # Stage 3: slice reconstruction from final sinograms.
                for batch_start, batch_end in slice_ranges:
                    current_batch_size = batch_end - batch_start
                    sinogram_batch_gpu = cp.asarray(
                        np.transpose(sinogram_host[:, :, batch_start:batch_end], (2, 0, 1)),
                        dtype=cp.float32,
                    )

                    for local_idx in range(current_batch_size):
                        slice_idx = batch_start + local_idx
                        recon_gpu = self._reconstruct_slice(
                            sinogram_batch_gpu[local_idx],
                            theta_gpu,
                            diag_len,
                            diag_pad_half,
                            output_size,
                        )
                        reconstructed[:, :, slice_idx] = cp.asnumpy(recon_gpu)

                    del sinogram_batch_gpu

                    if progress_callback:
                        progress_callback(0.70 + 0.30 * (batch_end / num_slices))

                break
            except cp.cuda.memory.OutOfMemoryError as e:
                oom_retries += 1
                if oom_retries > max_oom_retries:
                    logging.error(
                        "GPU OOM persisted after %d retries (slice=%d, radon=%d, iradon=%d): %s",
                        max_oom_retries,
                        current_slice_batch,
                        current_radon_batch,
                        current_iradon_batch,
                        e,
                    )
                    raise

                new_slice_batch, new_radon_batch, new_iradon_batch = self._halve_batch_sizes(
                    current_slice_batch,
                    current_radon_batch,
                    current_iradon_batch,
                )
                if (new_slice_batch, new_radon_batch, new_iradon_batch) == (
                    current_slice_batch,
                    current_radon_batch,
                    current_iradon_batch,
                ):
                    logging.error(
                        "GPU OOM at minimum batch sizes (slice=%d, radon=%d, iradon=%d)",
                        current_slice_batch,
                        current_radon_batch,
                        current_iradon_batch,
                    )
                    raise

                logging.warning(
                    "GPU OOM. Retrying with halved batches: slices %d->%d, radon %d->%d, iradon %d->%d",
                    current_slice_batch,
                    new_slice_batch,
                    current_radon_batch,
                    new_radon_batch,
                    current_iradon_batch,
                    new_iradon_batch,
                )

                current_slice_batch = new_slice_batch
                current_radon_batch = new_radon_batch
                current_iradon_batch = new_iradon_batch
                continue
            finally:
                cp.get_default_memory_pool().free_all_blocks()

        return reconstructed
