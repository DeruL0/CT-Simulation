"""
Core Package

Contains base data structures and abstract interfaces for the scientific application.
"""

from .base import (
    ScientificData,
    BaseLoader,
    BaseAnalyzer,
    BaseVisualizer,
)

__all__ = [
    'ScientificData',
    'BaseLoader', 
    'BaseAnalyzer',
    'BaseVisualizer',
]

