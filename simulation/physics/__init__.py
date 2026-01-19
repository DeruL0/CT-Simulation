"""
X-ray Physics Package

Provides realistic X-ray physics simulation including:
- Polychromatic spectrum generation
- Energy-dependent attenuation
- Beam hardening effects
"""

from .spectrum import SpectrumGenerator, XRaySpectrum
from .attenuation import AttenuationDatabase, get_attenuation
from .physical_material import PhysicalMaterial, PHYSICAL_MATERIALS
from .physical_simulator import PhysicalCTSimulator, PhysicsConfig

__all__ = [
    "SpectrumGenerator",
    "XRaySpectrum", 
    "AttenuationDatabase",
    "get_attenuation",
    "PhysicalMaterial",
    "PHYSICAL_MATERIALS",
    "PhysicalCTSimulator",
    "PhysicsConfig",
]
