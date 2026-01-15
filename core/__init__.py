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
from .data_manager import DataManager

__all__ = [
    'ScientificData',
    'BaseLoader', 
    'BaseAnalyzer',
    'BaseVisualizer',
    'DataManager',
]

