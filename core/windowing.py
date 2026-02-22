"""
Shared window/level and linear grayscale mapping helpers.
"""

from typing import Any

import numpy as np
import numpy.typing as npt


def compute_window_bounds(
    center: float,
    width: float,
    *,
    min_width: float = 1e-6,
) -> tuple[float, float]:
    """
    Compute lower/upper window bounds from center/width.
    """
    c = float(center)
    w = max(float(width), float(min_width))
    lower = c - w / 2.0
    upper = lower + w
    return lower, upper


def normalize_linear(
    values: Any,
    *,
    lower: float,
    upper: float,
    clip: bool = True,
) -> npt.NDArray[np.float64]:
    """
    Normalize values linearly to [0, 1] from a given source range.
    """
    data = np.asarray(values, dtype=np.float64)
    low = float(lower)
    high = float(upper)
    span = high - low
    if not np.isfinite(span) or span <= 0:
        return np.zeros_like(data, dtype=np.float64)

    normalized = (data - low) / span
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def linear_to_uint(
    values: Any,
    *,
    lower: float,
    upper: float,
    output_dtype: npt.DTypeLike,
) -> np.ndarray:
    """
    Linearly map values from [lower, upper] to unsigned integer range.
    """
    dtype = np.dtype(output_dtype)
    if dtype.kind != "u":
        raise TypeError(f"output_dtype must be an unsigned integer dtype, got {dtype}.")

    max_value = float(np.iinfo(dtype).max)
    normalized = normalize_linear(values, lower=lower, upper=upper, clip=True)
    return (normalized * max_value).astype(dtype)


def window_to_uint(
    values: Any,
    *,
    center: float,
    width: float,
    output_dtype: npt.DTypeLike,
    min_width: float = 1e-6,
) -> np.ndarray:
    """
    Apply window/level and map to unsigned integer range.
    """
    lower, upper = compute_window_bounds(center, width, min_width=min_width)
    return linear_to_uint(
        values,
        lower=lower,
        upper=upper,
        output_dtype=output_dtype,
    )


def map_window_to_uint_range(
    *,
    center: float,
    width: float,
    source_min: float,
    source_max: float,
    output_dtype: npt.DTypeLike,
    min_output_width: float = 1.0,
) -> tuple[float, float]:
    """
    Convert window center/width from source domain into target uint domain.
    """
    dtype = np.dtype(output_dtype)
    if dtype.kind != "u":
        raise TypeError(f"output_dtype must be an unsigned integer dtype, got {dtype}.")

    out_max = float(np.iinfo(dtype).max)
    src_min = float(source_min)
    src_max = float(source_max)
    if not np.isfinite(src_min) or not np.isfinite(src_max) or src_max <= src_min:
        return out_max / 2.0, out_max

    scale = out_max / (src_max - src_min)
    mapped_center = (float(center) - src_min) * scale
    mapped_width = max(float(min_output_width), float(width) * scale)
    mapped_center = float(np.clip(mapped_center, 0.0, out_max))
    mapped_width = float(np.clip(mapped_width, float(min_output_width), out_max))
    return mapped_center, mapped_width
