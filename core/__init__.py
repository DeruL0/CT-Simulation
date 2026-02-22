"""
Core Package

Contains base data structures and abstract interfaces for the scientific application.
"""

from .base import (
    ScientificData,
    GPUSimulationTiming,
    SimulationTimingResult,
    BaseLoader,
    BaseAnalyzer,
    BaseVisualizer,
)
from .validation import (
    require_ndarray,
    require_positive_finite_scalar,
    require_finite_vector,
)
from .windowing import (
    compute_window_bounds,
    normalize_linear,
    linear_to_uint,
    window_to_uint,
    map_window_to_uint_range,
)

__all__ = [
    'ScientificData',
    'GPUSimulationTiming',
    'SimulationTimingResult',
    'BaseLoader', 
    'BaseAnalyzer',
    'BaseVisualizer',
    'require_ndarray',
    'require_positive_finite_scalar',
    'require_finite_vector',
    'compute_window_bounds',
    'normalize_linear',
    'linear_to_uint',
    'window_to_uint',
    'map_window_to_uint_range',
]

