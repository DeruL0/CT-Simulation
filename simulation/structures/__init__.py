"""
Structure Modifier Package

Industrial-grade structure generation for CT simulation.
Supports lattice structures (TPMS), random defect generation, and manual modifiers.
GPU acceleration available via CuPy.
"""

from .types import LatticeType, DefectShape, LatticeConfig, DefectConfig
from .modifier import StructureModifier

__all__ = [
    'LatticeType',
    'DefectShape', 
    'LatticeConfig',
    'DefectConfig',
    'StructureModifier',
]
