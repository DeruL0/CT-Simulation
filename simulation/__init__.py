"""
Simulation package for CT image generation.

Contains core simulation algorithms: voxelization, CT simulation, materials.
STL loading has been moved to loaders.stl_loader.
DICOM export has been moved to exporters.dicom.
"""

from .voxelizer import Voxelizer, VoxelGrid
from .materials import MaterialDatabase, MaterialType, Material
from .ct_simulator import CTSimulator, CTVolume

# Re-export from loaders for backwards compatibility
from loaders.stl_loader import STLLoader, MeshInfo

__all__ = [
    "STLLoader", 
    "MeshInfo",
    "Voxelizer", 
    "VoxelGrid",
    "MaterialDatabase",
    "MaterialType",
    "Material", 
    "CTSimulator",
    "CTVolume",
]
