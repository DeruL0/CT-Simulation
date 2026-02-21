"""
GPU Radon Transform Kernels

Provides GPU-accelerated forward (Radon) and inverse (iRadon/FBP) transforms.
The preferred path uses custom CuPy RawKernel implementations, with an automatic
fallback to vectorized CuPy code when kernel compilation/execution is unavailable.
"""

import logging
import numpy as np

# Optional CuPy import
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


class GPURadonTransform:
    """
    GPU-accelerated Radon and inverse Radon (FBP) transforms.

    Uses fused CUDA kernels when possible; otherwise falls back to safer
    vectorized CuPy implementations with reduced batch sizes.
    """

    _CUDA_SOURCE = r"""
    extern "C" {
    __global__ void radon_forward_kernel(
        const float* image,
        int height,
        int width,
        const float* theta_rad,
        int n_angles,
        int n_det,
        float det_center,
        float* sinogram
    ) {
        const int det_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int angle_idx = blockIdx.y * blockDim.y + threadIdx.y;
        if (det_idx >= n_det || angle_idx >= n_angles) {
            return;
        }

        const float img_cx = 0.5f * (float)(width - 1);
        const float img_cy = 0.5f * (float)(height - 1);
        const float det = (float)det_idx - det_center;
        const float c = cosf(theta_rad[angle_idx]);
        const float s = sinf(theta_rad[angle_idx]);

        float accum = 0.0f;
        for (int sample_idx = 0; sample_idx < n_det; ++sample_idx) {
            const float sample = (float)sample_idx - det_center;
            const float x = det * c - sample * s + img_cx;
            const float y = det * s + sample * c + img_cy;

            if (x >= 0.0f && x < (float)(width - 1) &&
                y >= 0.0f && y < (float)(height - 1)) {
                const int x0 = (int)floorf(x);
                const int y0 = (int)floorf(y);
                const int x1 = x0 + 1;
                const int y1 = y0 + 1;

                const float wx = x - (float)x0;
                const float wy = y - (float)y0;

                const float v00 = image[y0 * width + x0];
                const float v01 = image[y0 * width + x1];
                const float v10 = image[y1 * width + x0];
                const float v11 = image[y1 * width + x1];

                const float interp =
                    v00 * (1.0f - wx) * (1.0f - wy) +
                    v01 * wx * (1.0f - wy) +
                    v10 * (1.0f - wx) * wy +
                    v11 * wx * wy;
                accum += interp;
            }
        }

        sinogram[det_idx * n_angles + angle_idx] = accum;
    }

    __global__ void iradon_backproject_kernel(
        const float* filtered_sino,
        const float* theta_rad,
        int n_angles,
        int n_det,
        int output_size,
        float det_center,
        float scale,
        float* reconstructed
    ) {
        const int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
        const int y_idx = blockIdx.y * blockDim.y + threadIdx.y;
        if (x_idx >= output_size || y_idx >= output_size) {
            return;
        }

        const float center = 0.5f * (float)(output_size - 1);
        const float x = (float)x_idx - center;
        const float y = (float)y_idx - center;

        float accum = 0.0f;
        for (int angle_idx = 0; angle_idx < n_angles; ++angle_idx) {
            const float c = cosf(theta_rad[angle_idx]);
            const float s = sinf(theta_rad[angle_idx]);
            const float t = x * c + y * s + det_center;

            if (t >= 0.0f && t <= (float)(n_det - 1)) {
                const int t0 = (int)floorf(t);
                const int t1 = min(t0 + 1, n_det - 1);
                const float wt = t - (float)t0;

                const float v0 = filtered_sino[t0 * n_angles + angle_idx];
                const float v1 = filtered_sino[t1 * n_angles + angle_idx];
                accum += v0 * (1.0f - wt) + v1 * wt;
            }
        }

        reconstructed[y_idx * output_size + x_idx] = accum * scale;
    }
    }
    """

    def __init__(self, radon_batch: int = 20, iradon_batch: int = 60):
        """
        Initialize GPU Radon transform.

        Args:
            radon_batch: Batch size for fallback forward projection angles
            iradon_batch: Batch size for fallback backprojection angles
        """
        self.radon_batch = radon_batch
        self.iradon_batch = iradon_batch

        self._rawkernel_state_checked = False
        self._rawkernel_available = False
        self._radon_kernel = None
        self._iradon_kernel = None

    def _degrade_fallback_batches(self) -> None:
        """Conservative fallback settings to avoid OOM on vectorized path."""
        self.radon_batch = max(4, min(self.radon_batch, 8))
        self.iradon_batch = max(8, min(self.iradon_batch, 16))

    def _disable_rawkernels(self, reason: str) -> None:
        """Disable RawKernel path and switch to conservative fallback mode."""
        if self._rawkernel_available:
            logging.warning(reason)
        self._rawkernel_available = False
        self._degrade_fallback_batches()

    def _ensure_rawkernels(self) -> bool:
        """Lazily initialize RawKernel handles."""
        if not HAS_CUPY:
            return False
        if self._rawkernel_state_checked:
            return self._rawkernel_available

        self._rawkernel_state_checked = True
        try:
            self._radon_kernel = cp.RawKernel(
                self._CUDA_SOURCE,
                "radon_forward_kernel",
                options=("--std=c++11",),
            )
            self._iradon_kernel = cp.RawKernel(
                self._CUDA_SOURCE,
                "iradon_backproject_kernel",
                options=("--std=c++11",),
            )
            self._rawkernel_available = True
        except Exception as e:
            logging.warning(
                "RawKernel initialization failed; falling back to vectorized CuPy implementation: %s",
                e
            )
            self._rawkernel_available = False
            self._degrade_fallback_batches()

        return self._rawkernel_available

    def _apply_ramp_filter(self, sinogram_gpu: "cp.ndarray") -> "cp.ndarray":
        """Apply ramp filter in Fourier space."""
        n_det = sinogram_gpu.shape[0]
        projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * n_det))))
        pad_width = ((0, projection_size_padded - n_det), (0, 0))
        padded_sino = cp.pad(sinogram_gpu, pad_width, mode="constant")

        f = cp.fft.fftfreq(projection_size_padded).reshape(-1, 1)
        ramp = cp.abs(f)

        f_sino = cp.fft.fft(padded_sino, axis=0)
        filtered_sino = cp.fft.ifft(f_sino * ramp, axis=0).real
        return filtered_sino[:n_det, :].astype(cp.float32)

    def _radon_vectorized(self, image_gpu: "cp.ndarray", theta_deg: "cp.ndarray") -> "cp.ndarray":
        """Fallback vectorized Radon transform."""
        h, w = image_gpu.shape
        n_det = int(np.ceil(np.sqrt(float(h * h + w * w))))
        n_angles = int(theta_deg.size)
        theta_rad_all = theta_deg * np.float32(np.pi / 180.0)

        img_center_x = (w - 1) / 2.0
        img_center_y = (h - 1) / 2.0
        det_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0
        sample_coords = cp.arange(n_det, dtype=cp.float32) - (n_det - 1) / 2.0

        sinogram = cp.zeros((n_det, n_angles), dtype=cp.float32)
        t = det_coords.reshape(1, -1, 1)
        s = sample_coords.reshape(1, 1, -1)

        batch_size = max(1, self.radon_batch)
        for i in range(0, n_angles, batch_size):
            theta_chunk = theta_rad_all[i:i + batch_size]
            current_batch_size = int(theta_chunk.size)

            theta = theta_chunk.reshape(-1, 1, 1)
            cos_t = cp.cos(theta)
            sin_t = cp.sin(theta)

            x_coords = t * cos_t - s * sin_t + img_center_x
            y_coords = t * sin_t + s * cos_t + img_center_y

            x0 = cp.floor(x_coords).astype(cp.int32)
            y0 = cp.floor(y_coords).astype(cp.int32)
            x1 = x0 + 1
            y1 = y0 + 1

            wx = x_coords - x0.astype(cp.float32)
            wy = y_coords - y0.astype(cp.float32)

            x0_clamped = cp.clip(x0, 0, w - 1)
            x1_clamped = cp.clip(x1, 0, w - 1)
            y0_clamped = cp.clip(y0, 0, h - 1)
            y1_clamped = cp.clip(y1, 0, h - 1)

            valid_mask = ((x0 >= 0) & (x1 < w) & (y0 >= 0) & (y1 < h)).astype(cp.float32)

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

            sinogram[:, i:i + current_batch_size] = samples.sum(axis=2).T

        return sinogram

    def _iradon_backprojection_vectorized(
        self,
        filtered_sino: "cp.ndarray",
        theta_deg: "cp.ndarray",
        output_size: int
    ) -> "cp.ndarray":
        """Fallback vectorized backprojection."""
        n_det, n_angles = filtered_sino.shape
        center = (output_size - 1) / 2.0
        grid_1d = cp.arange(output_size, dtype=cp.float32) - center
        y_grid, x_grid = cp.meshgrid(grid_1d, grid_1d, indexing="ij")

        x_flat = x_grid.ravel()
        y_flat = y_grid.ravel()
        reconstructed = cp.zeros((output_size * output_size), dtype=cp.float32)

        theta_rad_all = theta_deg * np.float32(np.pi / 180.0)
        det_center = (n_det - 1) / 2.0
        batch_size = max(1, self.iradon_batch)

        for i in range(0, n_angles, batch_size):
            theta_chunk = theta_rad_all[i:i + batch_size]
            current_batch_size = int(theta_chunk.size)

            cos_t = cp.cos(theta_chunk).reshape(-1, 1)
            sin_t = cp.sin(theta_chunk).reshape(-1, 1)

            t_all = x_flat.reshape(1, -1) * cos_t + y_flat.reshape(1, -1) * sin_t
            t_idx = t_all + det_center

            t_idx_floor = cp.floor(t_idx).astype(cp.int32)
            t_idx_ceil = t_idx_floor + 1
            t_frac = t_idx - t_idx_floor.astype(cp.float32)

            t_idx_floor = cp.clip(t_idx_floor, 0, n_det - 1)
            t_idx_ceil = cp.clip(t_idx_ceil, 0, n_det - 1)

            angle_indices = cp.arange(i, i + current_batch_size, dtype=cp.int32).reshape(-1, 1)
            val_floor = filtered_sino[t_idx_floor, angle_indices]
            val_ceil = filtered_sino[t_idx_ceil, angle_indices]

            sampled = val_floor * (1 - t_frac) + val_ceil * t_frac
            reconstructed += sampled.sum(axis=0)

        scale = np.float32(np.pi / (2.0 * n_angles)) if n_angles > 0 else np.float32(0.0)
        return reconstructed.reshape(output_size, output_size) * scale

    def radon(self, image_gpu: "cp.ndarray", theta_deg: "cp.ndarray") -> "cp.ndarray":
        """
        GPU Radon transform.

        RawKernel path fuses coordinate transform, clipping and interpolation.
        Fallback path uses vectorized CuPy in smaller angle batches.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not available for GPURadonTransform")

        image_gpu = cp.ascontiguousarray(image_gpu.astype(cp.float32, copy=False))
        theta_deg = cp.asarray(theta_deg, dtype=cp.float32)

        h, w = image_gpu.shape
        n_det = int(np.ceil(np.sqrt(float(h * h + w * w))))
        n_angles = int(theta_deg.size)

        if n_angles == 0:
            return cp.zeros((n_det, 0), dtype=cp.float32)

        if self._ensure_rawkernels():
            try:
                theta_rad = theta_deg * np.float32(np.pi / 180.0)
                sinogram = cp.zeros((n_det, n_angles), dtype=cp.float32)

                threads = (16, 16, 1)
                blocks = (
                    (n_det + threads[0] - 1) // threads[0],
                    (n_angles + threads[1] - 1) // threads[1],
                    1
                )

                self._radon_kernel(
                    blocks,
                    threads,
                    (
                        image_gpu,
                        np.int32(h),
                        np.int32(w),
                        theta_rad,
                        np.int32(n_angles),
                        np.int32(n_det),
                        np.float32((n_det - 1) / 2.0),
                        sinogram,
                    )
                )
                return sinogram
            except Exception as e:
                self._disable_rawkernels(f"RawKernel radon execution failed, switching to fallback: {e}")

        return self._radon_vectorized(image_gpu, theta_deg)

    def iradon(self, sinogram_gpu: "cp.ndarray", theta_deg: "cp.ndarray", output_size: int) -> "cp.ndarray":
        """
        GPU inverse Radon transform (filtered back projection).

        Ramp filtering remains on CuPy FFT, then backprojection is executed by
        RawKernel if available, otherwise by vectorized fallback batches.
        """
        if not HAS_CUPY:
            raise RuntimeError("CuPy is not available for GPURadonTransform")

        sinogram_gpu = cp.ascontiguousarray(sinogram_gpu.astype(cp.float32, copy=False))
        theta_deg = cp.asarray(theta_deg, dtype=cp.float32)

        n_det, n_angles = sinogram_gpu.shape
        if n_angles == 0:
            return cp.zeros((output_size, output_size), dtype=cp.float32)

        filtered_sino = self._apply_ramp_filter(sinogram_gpu)
        filtered_sino = cp.ascontiguousarray(filtered_sino)

        if self._ensure_rawkernels():
            try:
                theta_rad = theta_deg * np.float32(np.pi / 180.0)
                reconstructed = cp.zeros((output_size, output_size), dtype=cp.float32)

                threads = (16, 16, 1)
                blocks = (
                    (output_size + threads[0] - 1) // threads[0],
                    (output_size + threads[1] - 1) // threads[1],
                    1
                )

                self._iradon_kernel(
                    blocks,
                    threads,
                    (
                        filtered_sino,
                        theta_rad,
                        np.int32(n_angles),
                        np.int32(n_det),
                        np.int32(output_size),
                        np.float32((n_det - 1) / 2.0),
                        np.float32(np.pi / (2.0 * n_angles)),
                        reconstructed,
                    )
                )
                return reconstructed
            except Exception as e:
                self._disable_rawkernels(f"RawKernel iradon execution failed, switching to fallback: {e}")

        return self._iradon_backprojection_vectorized(filtered_sino, theta_deg, output_size)
