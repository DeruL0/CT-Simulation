"""
Simulation package for CT image generation.

Contains core simulation algorithms: voxelization, CT simulation, materials.
STL loading has been moved to loaders.stl_loader.
DICOM export has been moved to exporters.dicom.
"""

from .voxelizer import Voxelizer, VoxelGrid
from .materials import (
    MaterialDatabase,
    MaterialType,
    Material,
    require_physical_material,
)
from .volume import CTVolume

# Physics simulation
from .physics import (
    PhysicalCTSimulator, 
    PhysicsConfig,
    PhysicalProcessConfig,
    SpectrumGenerator,
    XRaySpectrum,
)
from .simple_simulator import SimpleCTSimulator, SimpleProcessConfig

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
    "require_physical_material",
    "CTVolume",
    # Simulators
    "SimpleCTSimulator",
    "SimpleProcessConfig",
    "PhysicalCTSimulator",
    "PhysicsConfig",
    "PhysicalProcessConfig",
    "SpectrumGenerator",
    "XRaySpectrum",
]

