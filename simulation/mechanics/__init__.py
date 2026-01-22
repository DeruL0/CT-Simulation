"""
Mechanics Simulation Package

Provides physical compression simulation using GPU-accelerated elasticity solver.
"""

from .elasticity import ElasticitySolver
from .manager import CompressionManager

__all__ = [
    'ElasticitySolver',
    'CompressionManager',
]
