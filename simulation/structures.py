"""
Structure Modifier - Backwards compatibility module

This module re-exports from simulation.structures package.
All actual implementation is in simulation/structures/*.py
"""

# Re-export everything from the new package for backwards compatibility
from .structures import (
    LatticeType,
    DefectShape,
    LatticeConfig,
    DefectConfig,
    StructureModifier,
)

__all__ = [
    'LatticeType',
    'DefectShape',
    'LatticeConfig',
    'DefectConfig',
    'StructureModifier',
]
