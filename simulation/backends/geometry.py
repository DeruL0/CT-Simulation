"""
Shared geometric helpers for Radon/IRadon slice preparation.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np


RADON_DIAGONAL_MARGIN = 32


@dataclass(frozen=True)
class RadonCanvasGeometry:
    """
    Geometry metadata for square+diagonal Radon padding/cropping.
    """

    output_size: int
    diagonal_size: int
    crop_offset: int


def _split_padding(total: int) -> tuple[int, int]:
    """Split total padding into centered (before, after)."""
    before = total // 2
    return before, total - before


def compute_radon_canvas_geometry(
    shape: tuple[int, int],
    *,
    diagonal_margin: int = RADON_DIAGONAL_MARGIN,
) -> RadonCanvasGeometry:
    """
    Compute output/canvas sizes for Radon-safe rotation geometry.
    """
    if len(shape) != 2:
        raise ValueError(f"Expected 2D shape, got {shape}.")
    h, w = int(shape[0]), int(shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"Slice dimensions must be positive, got {shape}.")

    output_size = max(h, w)
    diagonal_size = int(np.ceil(np.sqrt(float(output_size * output_size * 2))))
    diagonal_size += int(diagonal_margin)

    diag_pad = max(0, diagonal_size - output_size)
    crop_offset, _ = _split_padding(diag_pad)
    return RadonCanvasGeometry(
        output_size=output_size,
        diagonal_size=diagonal_size,
        crop_offset=crop_offset,
    )


def pad_slice_to_radon_canvas(
    slice_2d: Any,
    *,
    xp: Any,
    diagonal_margin: int = RADON_DIAGONAL_MARGIN,
    constant_value: float = 0.0,
) -> tuple[Any, RadonCanvasGeometry]:
    """
    Center-pad 2D slice to Radon-safe diagonal canvas.

    Works with both NumPy and CuPy arrays via the provided `xp` module.
    """
    geometry = compute_radon_canvas_geometry(
        tuple(slice_2d.shape),
        diagonal_margin=diagonal_margin,
    )

    h, w = int(slice_2d.shape[0]), int(slice_2d.shape[1])
    square_pad_h = max(0, geometry.output_size - h)
    square_pad_w = max(0, geometry.output_size - w)
    sq_top, sq_bottom = _split_padding(square_pad_h)
    sq_left, sq_right = _split_padding(square_pad_w)

    padded = slice_2d
    if square_pad_h > 0 or square_pad_w > 0:
        padded = xp.pad(
            padded,
            ((sq_top, sq_bottom), (sq_left, sq_right)),
            mode="constant",
            constant_values=constant_value,
        )

    diag_pad = max(0, geometry.diagonal_size - geometry.output_size)
    diag_top, diag_bottom = _split_padding(diag_pad)
    if diag_pad > 0:
        padded = xp.pad(
            padded,
            ((diag_top, diag_bottom), (diag_top, diag_bottom)),
            mode="constant",
            constant_values=constant_value,
        )

    return padded, geometry


def crop_from_radon_canvas(image_2d: Any, geometry: RadonCanvasGeometry) -> Any:
    """
    Crop diagonal-canvas reconstruction back to square output size.
    """
    if geometry.crop_offset <= 0:
        return image_2d
    start = geometry.crop_offset
    stop = start + geometry.output_size
    return image_2d[start:stop, start:stop]
