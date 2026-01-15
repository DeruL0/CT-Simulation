"""
Loaders Package

Contains data loading strategies for different file formats.
"""

from .stl_loader import STLLoader, MeshInfo

__all__ = [
    'STLLoader',
    'MeshInfo',
]
