"""
Simulation package for CT image generation.

Contains core simulation algorithms: voxelization, CT simulation, materials.
STL loading has been moved to loaders.stl_loader.
DICOM export has been moved to exporters.dicom.
"""

from .voxelizer import Voxelizer, VoxelGrid
from .materials import MaterialDatabase, MaterialType, Material
from .volume import CTVolume

# Physics simulation
from .physics import (
    PhysicalCTSimulator, 
    PhysicsConfig,
    SpectrumGenerator,
    XRaySpectrum,
)

# Re-export from loaders for backwards compatibility
from loaders import MeshLoader as STLLoader, MeshInfo

__all__ = [
    "STLLoader", 
    "MeshInfo",
    "Voxelizer", 
    "VoxelGrid",
    "MaterialDatabase",
    "MaterialType",
    "Material", 
    "CTVolume",
    # Physics
    "PhysicalCTSimulator",
    "PhysicsConfig",
    "SpectrumGenerator",
    "XRaySpectrum",
]
