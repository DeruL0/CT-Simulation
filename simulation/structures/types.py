"""
Structure Types and Configurations

Enums and dataclasses for structure generation configuration.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class LatticeType(Enum):
    """Available TPMS lattice types."""
    GYROID = "gyroid"
    SCHWARZ_PRIMITIVE = "primitive"
    SCHWARZ_DIAMOND = "diamond"
    LIDINOID = "lidinoid"
    SPLIT_P = "split_p"


class DefectShape(Enum):
    """Available defect shapes for random generation."""
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    ELLIPSOID = "ellipsoid"


@dataclass
class DefectConfig:
    """Configuration for random defect generation."""
    shape: DefectShape = DefectShape.SPHERE
    density_percent: float = 5.0  # Target porosity (0-100%)
    size_mean_mm: float = 2.0  # Mean defect size in mm
    size_std_mm: float = 0.5  # Size standard deviation
    seed: Optional[int] = None  # Random seed for reproducibility
    preserve_shell: bool = False  # Preserve outer shell
    shell_thickness_mm: float = 1.0  # Shell thickness in mm


@dataclass
class LatticeConfig:
    """Configuration for lattice generation."""
    lattice_type: LatticeType = LatticeType.GYROID
    density_percent: float = 30.0  # Target solid volume fraction (0-100%)
    cell_size_mm: float = 5.0  # Unit cell size in mm
    preserve_shell: bool = False  # Preserve outer shell
    shell_thickness_mm: float = 1.0  # Shell thickness in mm
