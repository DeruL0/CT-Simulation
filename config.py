"""
CT Simulation Software Configuration

Contains constants and default settings for the CT simulation engine.
"""

from dataclasses import dataclass, field
from typing import Tuple

# Import MaterialType from the canonical location
from simulation.materials import MaterialType


# Hounsfield Unit (HU) values for different materials
# Reference: https://radiopaedia.org/articles/hounsfield-unit
# Note: Full material database is in simulation.materials.MaterialDatabase
MATERIAL_HU: dict[MaterialType, float] = {
    MaterialType.AIR: -1000.0,
    MaterialType.WATER: 0.0,
    MaterialType.FAT: -100.0,
    MaterialType.MUSCLE: 40.0,
    MaterialType.BONE_CANCELLOUS: 400.0,
    MaterialType.BONE_CORTICAL: 1000.0,
    MaterialType.TITANIUM: 3000.0,
}



@dataclass
class VoxelizationConfig:
    """Configuration for mesh voxelization."""
    voxel_size_mm: float = 0.5  # Voxel edge length in millimeters
    fill_interior: bool = True  # Fill interior of closed meshes
    
    
@dataclass
class CTSimulationConfig:
    """Configuration for CT simulation parameters."""
    num_projections: int = 360  # Number of projection angles (0-360°)
    detector_pixels: int = 512  # Number of detector pixels
    kvp: float = 120.0  # Tube voltage in kVp (80-140 typical)
    add_noise: bool = True  # Add Poisson noise for realism
    noise_level: float = 0.02  # Noise standard deviation as fraction of signal
    
    
@dataclass
class DICOMConfig:
    """Configuration for DICOM export."""
    patient_name: str = "Anonymous^Patient"
    patient_id: str = "SIMULATION001"
    study_description: str = "CT Simulation"
    series_description: str = "Simulated CT Series"
    manufacturer: str = "CT Simulation Software"
    institution_name: str = "Research Institution"
    slice_thickness_mm: float = 1.0
    
    # Window/Level presets
    window_presets: dict = field(default_factory=lambda: {
        "bone": {"center": 500, "width": 2000},
        "soft_tissue": {"center": 40, "width": 400},
        "lung": {"center": -600, "width": 1500},
    })
    

@dataclass
class GUIConfig:
    """Configuration for GUI appearance."""
    window_title: str = "CT Simulation Software"
    window_size: Tuple[int, int] = (1400, 900)
    min_size: Tuple[int, int] = (1000, 700)
    
    # Scientific white theme colors
    background_color: str = "#FFFFFF"
    text_color: str = "#333333"
    accent_color: str = "#2962FF"
    secondary_color: str = "#F5F5F5"
    border_color: str = "#E0E0E0"
    
    # Typography
    font_family: str = "Segoe UI"
    font_size: int = 10
    header_font_size: int = 12


# Default configurations
DEFAULT_VOXELIZATION = VoxelizationConfig()
DEFAULT_CT_SIMULATION = CTSimulationConfig()
DEFAULT_DICOM = DICOMConfig()
DEFAULT_GUI = GUIConfig()
