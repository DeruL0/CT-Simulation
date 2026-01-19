"""
Physical Material Definitions

NOTE: This module is deprecated. All functionality has been merged into
simulation.materials. This file re-exports for backward compatibility.
"""

from ..materials import (
    Material as PhysicalMaterial,
    PHYSICAL_MATERIALS,
    get_physical_material,
    material_type_to_physical,
)

__all__ = [
    "PhysicalMaterial",
    "PHYSICAL_MATERIALS", 
    "get_physical_material",
    "material_type_to_physical",
]

