"""
Material Properties Database

Provides X-ray attenuation properties and Hounsfield Unit values
for different materials used in CT simulation.

This unified module supports both simplified (HU-based) and
physics-based (energy-dependent attenuation) simulation modes.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from .physics.attenuation import AttenuationDatabase


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
    IODINE_CONTRAST = "iodine_contrast"
    # Industrial/Common materials
    PLASTIC_PVC = "plastic_pvc"
    PLASTIC_PE = "plastic_pe"
    FOAM_SPONGE = "foam_sponge"
    RUBBER = "rubber"
    # Food materials
    BREAD = "bread"
    CHOCOLATE = "chocolate"
    CHEESE = "cheese"
    FRUIT = "fruit"
    CUSTOM = "custom"


@dataclass
class Material:
    """
    Unified material properties for CT simulation.
    
    Supports both simplified (HU-based) and physics-based simulation.
    
    Attributes:
        name: Human-readable name
        hounsfield_unit: CT number in Hounsfield Units (HU)
        density: Physical density in g/cm³
        color: RGB color for visualization (0-255)
        attenuation_key: Key in AttenuationDatabase (for physics mode)
    """
    name: str
    hounsfield_unit: float
    density: float  # g/cm³
    color: tuple = (128, 128, 128)  # RGB (0-255)
    attenuation_key: Optional[str] = None  # For physics mode
    
    @property
    def linear_attenuation(self) -> float:
        """
        Approximate linear attenuation coefficient at 100 keV.
        
        Uses simplified relationship: μ ≈ μ_water * (HU/1000 + 1)
        where μ_water ≈ 0.171 cm⁻¹ at 100 keV
        """
        mu_water = 0.171  # cm⁻¹ at 100 keV
        return mu_water * (self.hounsfield_unit / 1000.0 + 1.0)
    
    def get_mu(self, energy: float) -> float:
        """
        Get linear attenuation coefficient at given energy (cm⁻¹).
        
        Uses NIST data if attenuation_key is set, otherwise uses
        simplified HU-based approximation.
        """
        if self.attenuation_key:
            from .physics.attenuation import get_attenuation_database
            db = get_attenuation_database()
            table = db.get_table(self.attenuation_key)
            if table is not None:
                mu_rho = table.get_mu_rho(energy)
                return mu_rho * self.density
        # Fallback to simplified model
        return self.linear_attenuation
    
    def get_mu_array(self, energies: np.ndarray) -> np.ndarray:
        """Get linear attenuation coefficients for array of energies."""
        if self.attenuation_key:
            from .physics.attenuation import get_attenuation_database
            db = get_attenuation_database()
            table = db.get_table(self.attenuation_key)
            if table is not None:
                mu_rho = np.interp(energies, table.energies, table.mu_rho)
                return mu_rho * self.density
        # Fallback: constant across energies
        return np.full_like(energies, self.linear_attenuation)


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
            density=0.001205,
            color=(20, 20, 40),
            attenuation_key="air"
        ),
        MaterialType.WATER: Material(
            name="Water",
            hounsfield_unit=0.0,
            density=1.0,
            color=(0, 100, 200),
            attenuation_key="water"
        ),
        MaterialType.FAT: Material(
            name="Fat",
            hounsfield_unit=-100.0,
            density=0.92,
            color=(255, 220, 150),
            attenuation_key="adipose"
        ),
        MaterialType.SOFT_TISSUE: Material(
            name="Soft Tissue",
            hounsfield_unit=40.0,
            density=1.06,
            color=(255, 180, 180),
            attenuation_key="soft_tissue"
        ),
        MaterialType.MUSCLE: Material(
            name="Muscle",
            hounsfield_unit=40.0,
            density=1.05,
            color=(180, 80, 80),
            attenuation_key="muscle"
        ),
        MaterialType.BLOOD: Material(
            name="Blood",
            hounsfield_unit=55.0,
            density=1.06,
            color=(200, 0, 0),
            attenuation_key="soft_tissue"  # Approximate
        ),
        MaterialType.LIVER: Material(
            name="Liver",
            hounsfield_unit=60.0,
            density=1.05,
            color=(160, 82, 45),
            attenuation_key="soft_tissue"  # Approximate
        ),
        MaterialType.BONE_CANCELLOUS: Material(
            name="Cancellous Bone",
            hounsfield_unit=400.0,
            density=1.18,
            color=(200, 200, 180),
            attenuation_key="bone_cancellous"
        ),
        MaterialType.BONE_CORTICAL: Material(
            name="Cortical Bone",
            hounsfield_unit=1000.0,
            density=1.85,
            color=(255, 255, 240),
            attenuation_key="bone_cortical"
        ),
        MaterialType.CALCIUM: Material(
            name="Calcium",
            hounsfield_unit=1500.0,
            density=1.55,
            color=(255, 255, 255),
            attenuation_key="calcium"
        ),
        MaterialType.TITANIUM: Material(
            name="Titanium",
            hounsfield_unit=3000.0,
            density=4.5,
            color=(180, 180, 200),
            attenuation_key="titanium"
        ),
        MaterialType.STEEL: Material(
            name="Stainless Steel",
            hounsfield_unit=8000.0,
            density=7.87,
            color=(150, 150, 160),
            attenuation_key="iron"
        ),
        MaterialType.ALUMINUM: Material(
            name="Aluminum",
            hounsfield_unit=1500.0,
            density=2.7,
            color=(200, 200, 210),
            attenuation_key="aluminum"
        ),
        MaterialType.IODINE_CONTRAST: Material(
            name="Iodine Contrast",
            hounsfield_unit=300.0,
            density=1.03,
            color=(255, 200, 0),
            attenuation_key="iodine"
        ),
        # Industrial/Common materials
        MaterialType.PLASTIC_PVC: Material(
            name="PVC Plastic",
            hounsfield_unit=100.0,  # ~50-150 HU typical
            density=1.4,
            color=(200, 200, 220),
            attenuation_key=None  # Use HU approximation
        ),
        MaterialType.PLASTIC_PE: Material(
            name="Polyethylene",
            hounsfield_unit=-70.0,  # Low density plastic
            density=0.95,
            color=(240, 240, 250),
            attenuation_key=None
        ),
        MaterialType.FOAM_SPONGE: Material(
            name="Foam/Sponge",
            hounsfield_unit=-700.0,  # Very low density
            density=0.03,
            color=(255, 255, 200),
            attenuation_key=None
        ),
        MaterialType.RUBBER: Material(
            name="Rubber",
            hounsfield_unit=50.0,
            density=1.1,
            color=(60, 60, 60),
            attenuation_key=None
        ),
        # Food materials (based on literature values)
        MaterialType.BREAD: Material(
            name="Bread",
            hounsfield_unit=-500.0,  # Porous, air-filled structure
            density=0.25,
            color=(210, 180, 140),
            attenuation_key=None
        ),
        MaterialType.CHOCOLATE: Material(
            name="Chocolate",
            hounsfield_unit=150.0,  # Dense, ~100-200 HU
            density=1.25,
            color=(80, 45, 20),
            attenuation_key=None
        ),
        MaterialType.CHEESE: Material(
            name="Cheese",
            hounsfield_unit=80.0,  # Similar to soft tissue
            density=1.05,
            color=(255, 220, 100),
            attenuation_key=None
        ),
        MaterialType.FRUIT: Material(
            name="Fruit (Apple)",
            hounsfield_unit=30.0,  # High water content
            density=0.85,
            color=(200, 50, 50),
            attenuation_key=None
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
        color: tuple = (128, 128, 128),
        attenuation_key: Optional[str] = None
    ) -> Material:
        """
        Add a custom material to the database.
        
        Args:
            name: Unique material name
            hounsfield_unit: CT number in HU
            density: Physical density in g/cm³
            color: RGB color for visualization
            attenuation_key: Key in AttenuationDatabase for physics mode
            
        Returns:
            The created Material object
        """
        material = Material(
            name=name,
            hounsfield_unit=hounsfield_unit,
            density=density,
            color=color,
            attenuation_key=attenuation_key
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


# ============== Backward Compatibility ==============
# Aliases for physics module compatibility

# Alias for PhysicalMaterial (now just Material)
PhysicalMaterial = Material

# Pre-built dictionary for physics module compatibility
PHYSICAL_MATERIALS: Dict[str, Material] = {
    mat_type.value: MaterialDatabase._MATERIALS[mat_type]
    for mat_type in MaterialType
    if mat_type != MaterialType.CUSTOM
}


def get_physical_material(name: str) -> Optional[Material]:
    """Get physical material by name (case-insensitive)."""
    return PHYSICAL_MATERIALS.get(name.lower())


def material_type_to_physical(material_type_value: str) -> Optional[Material]:
    """
    Convert MaterialType enum value to Material with physics support.
    
    Handles mapping between simple material names and physical database.
    """
    return PHYSICAL_MATERIALS.get(material_type_value.lower())
