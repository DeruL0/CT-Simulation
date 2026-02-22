"""
Shared validation helpers for ndarray- and geometry-centric data contracts.
"""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt


def require_ndarray(
    value: Any,
    *,
    name: str,
    ndim: Optional[int] = None,
    dtype: Optional[npt.DTypeLike] = None,
    numeric: bool = False,
    require_c_contiguous: bool = False,
    coerce_c_contiguous: bool = False,
) -> npt.NDArray[Any]:
    """
    Validate ndarray shape/type/layout contract and return validated array.
    """
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be numpy.ndarray, got {type(value).__name__}.")

    array = value

    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got ndim={array.ndim}.")

    if dtype is not None:
        expected = np.dtype(dtype)
        if array.dtype != expected:
            raise TypeError(f"{name} must have dtype={expected}, got dtype={array.dtype}.")

    if numeric and not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} dtype must be numeric, got dtype={array.dtype}.")

    if require_c_contiguous and not array.flags.c_contiguous:
        if coerce_c_contiguous:
            array = np.ascontiguousarray(array)
        else:
            raise ValueError(f"{name} must be C-contiguous.")

    return array


def require_positive_finite_scalar(value: Any, *, name: str) -> float:
    """Validate scalar is finite and > 0."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real number.") from exc

    if not np.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be positive and finite (got {value}).")

    return number


def require_finite_vector(
    value: Any,
    *,
    name: str,
    length: int,
) -> npt.NDArray[np.float64]:
    """Validate a finite 1D vector with fixed length."""
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},), got {vector.shape}.")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values.")
    return vector
