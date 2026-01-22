"""
Loaders Package

Contains data loading strategies for different file formats.
"""

from .mesh_loader import (
    MeshLoader,
    MeshInfo,
    STLLoader,  # Backwards compatibility alias
    SUPPORTED_EXTENSIONS,
)

__all__ = [
    'MeshLoader',
    'MeshInfo',
    'STLLoader',  # Backwards compatibility alias
    'SUPPORTED_EXTENSIONS',
]
