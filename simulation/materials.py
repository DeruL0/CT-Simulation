"""
Material Properties Database

Provides X-ray attenuation properties and Hounsfield Unit values
for different materials used in CT simulation.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import numpy as np


class MaterialType(Enum):
    """Standard material types for CT simulation."""
    AIR = "air"
    WATER = "water"
    FAT = "fat"
    SOFT_TISSUE = "soft_tissue"
    MUSCLE = "muscle"
    BLOOD = "blood"
    LIVER = "liver"
    BONE_CANCELLOUS = "bone_cancellous"
    BONE_CORTICAL = "bone_cortical"
    CALCIUM = "calcium"
    TITANIUM = "titanium"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    CUSTOM = "custom"


@dataclass
class Material:
    """
    Material properties for CT simulation.
    
    Attributes:
        name: Human-readable name
        hounsfield_unit: CT number in Hounsfield Units (HU)
        density: Physical density in g/cm³
        color: RGB color for visualization (0-255)
    """
    name: str
    hounsfield_unit: float
    density: float  # g/cm³
    color: tuple  # RGB (0-255)
    
    @property
    def linear_attenuation(self) -> float:
        """
        Approximate linear attenuation coefficient at 100 keV.
        
        Uses simplified relationship: μ ≈ μ_water * (HU/1000 + 1)
        where μ_water ≈ 0.171 cm⁻¹ at 100 keV
        """
        mu_water = 0.171  # cm⁻¹ at 100 keV
        return mu_water * (self.hounsfield_unit / 1000.0 + 1.0)


class MaterialDatabase:
    """
    Database of material properties for CT simulation.
    
    Provides Hounsfield Unit values based on clinical CT standards.
    Reference: Radiological standards and NIST XCOM database.
    """
    
    # Standard material definitions
    # HU values from: https://radiopaedia.org/articles/hounsfield-unit
    _MATERIALS: Dict[MaterialType, Material] = {
        MaterialType.AIR: Material(
            name="Air",
            hounsfield_unit=-1000.0,
            density=0.0012,
            color=(20, 20, 40)
        ),
        MaterialType.WATER: Material(
            name="Water",
            hounsfield_unit=0.0,
            density=1.0,
            color=(0, 100, 200)
        ),
        MaterialType.FAT: Material(
            name="Fat",
            hounsfield_unit=-100.0,
            density=0.92,
            color=(255, 220, 150)
        ),
        MaterialType.SOFT_TISSUE: Material(
            name="Soft Tissue",
            hounsfield_unit=40.0,
            density=1.06,
            color=(255, 180, 180)
        ),
        MaterialType.MUSCLE: Material(
            name="Muscle",
            hounsfield_unit=40.0,
            density=1.06,
            color=(180, 80, 80)
        ),
        MaterialType.BLOOD: Material(
            name="Blood",
            hounsfield_unit=55.0,
            density=1.06,
            color=(200, 0, 0)
        ),
        MaterialType.LIVER: Material(
            name="Liver",
            hounsfield_unit=60.0,
            density=1.05,
            color=(160, 82, 45)
        ),
        MaterialType.BONE_CANCELLOUS: Material(
            name="Cancellous Bone",
            hounsfield_unit=400.0,
            density=1.18,
            color=(200, 200, 180)
        ),
        MaterialType.BONE_CORTICAL: Material(
            name="Cortical Bone",
            hounsfield_unit=1000.0,
            density=1.85,
            color=(255, 255, 240)
        ),
        MaterialType.CALCIUM: Material(
            name="Calcium",
            hounsfield_unit=1500.0,
            density=1.55,
            color=(255, 255, 255)
        ),
        MaterialType.TITANIUM: Material(
            name="Titanium",
            hounsfield_unit=3000.0,
            density=4.5,
            color=(180, 180, 200)
        ),
        MaterialType.STEEL: Material(
            name="Stainless Steel",
            hounsfield_unit=8000.0,  # Causes significant artifacts
            density=7.8,
            color=(150, 150, 160)
        ),
        MaterialType.ALUMINUM: Material(
            name="Aluminum",
            hounsfield_unit=1500.0,
            density=2.7,
            color=(200, 200, 210)
        ),
    }
    
    def __init__(self):
        """Initialize material database with default materials."""
        self._custom_materials: Dict[str, Material] = {}
    
    def get_material(self, material_type: MaterialType) -> Material:
        """Get material properties by type."""
        if material_type == MaterialType.CUSTOM:
            raise ValueError("Use get_custom_material() for custom materials")
        return self._MATERIALS[material_type]
    
    def get_hu(self, material_type: MaterialType) -> float:
        """Get Hounsfield Unit value for a material type."""
        return self.get_material(material_type).hounsfield_unit
    
    def add_custom_material(
        self,
        name: str,
        hounsfield_unit: float,
        density: float = 1.0,
        color: tuple = (128, 128, 128)
    ) -> Material:
        """
        Add a custom material to the database.
        
        Args:
            name: Unique material name
            hounsfield_unit: CT number in HU
            density: Physical density in g/cm³
            color: RGB color for visualization
            
        Returns:
            The created Material object
        """
        material = Material(
            name=name,
            hounsfield_unit=hounsfield_unit,
            density=density,
            color=color
        )
        self._custom_materials[name] = material
        return material
    
    def get_custom_material(self, name: str) -> Material:
        """Get a custom material by name."""
        if name not in self._custom_materials:
            raise KeyError(f"Custom material '{name}' not found")
        return self._custom_materials[name]
    
    def list_materials(self) -> list:
        """List all available material types."""
        return list(MaterialType)
    
    def list_custom_materials(self) -> list:
        """List all custom material names."""
        return list(self._custom_materials.keys())
    
    @staticmethod
    def hu_to_normalized(hu_value: float) -> float:
        """
        Convert Hounsfield Units to normalized [0, 1] range.
        
        Uses standard CT window: [-1000, 3000] HU
        """
        hu_min, hu_max = -1000.0, 3000.0
        return np.clip((hu_value - hu_min) / (hu_max - hu_min), 0.0, 1.0)
    
    @staticmethod
    def normalized_to_hu(normalized: float) -> float:
        """
        Convert normalized [0, 1] value to Hounsfield Units.
        
        Uses standard CT window: [-1000, 3000] HU
        """
        hu_min, hu_max = -1000.0, 3000.0
        return normalized * (hu_max - hu_min) + hu_min
