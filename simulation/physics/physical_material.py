"""
Physical Material Definitions

Extended material definitions with physics properties for
energy-dependent CT simulation.
"""

from dataclasses import dataclass
from typing import Dict, Optional
from enum import Enum
import numpy as np

from .attenuation import get_attenuation_database, AttenuationTable


@dataclass
class PhysicalMaterial:
    """
    Physical material for realistic CT simulation.
    
    Attributes:
        name: Human-readable name
        attenuation_key: Key in AttenuationDatabase
        density: Physical density in g/cm³
        reference_hu: Expected HU at ~70 keV (for validation)
    """
    name: str
    attenuation_key: str  # Key in AttenuationDatabase
    density: float        # g/cm³
    reference_hu: float   # Expected HU at reference energy
    
    def get_mu(self, energy: float) -> float:
        """Get linear attenuation coefficient at given energy (cm⁻¹)."""
        db = get_attenuation_database()
        table = db.get_table(self.attenuation_key)
        if table is None:
            raise KeyError(f"Material '{self.attenuation_key}' not found")
        # Use actual density if different from table
        mu_rho = table.get_mu_rho(energy)
        return mu_rho * self.density
    
    def get_mu_array(self, energies: np.ndarray) -> np.ndarray:
        """Get linear attenuation coefficients for array of energies."""
        db = get_attenuation_database()
        table = db.get_table(self.attenuation_key)
        if table is None:
            raise KeyError(f"Material '{self.attenuation_key}' not found")
        mu_rho = np.interp(energies, table.energies, table.mu_rho)
        return mu_rho * self.density


# Pre-defined physical materials matching MaterialType enum
PHYSICAL_MATERIALS: Dict[str, PhysicalMaterial] = {
    "air": PhysicalMaterial(
        name="Air",
        attenuation_key="air",
        density=0.001205,
        reference_hu=-1000.0
    ),
    "water": PhysicalMaterial(
        name="Water",
        attenuation_key="water",
        density=1.0,
        reference_hu=0.0
    ),
    "fat": PhysicalMaterial(
        name="Fat/Adipose",
        attenuation_key="adipose",
        density=0.92,
        reference_hu=-100.0
    ),
    "soft_tissue": PhysicalMaterial(
        name="Soft Tissue",
        attenuation_key="soft_tissue",
        density=1.06,
        reference_hu=40.0
    ),
    "muscle": PhysicalMaterial(
        name="Muscle",
        attenuation_key="muscle",
        density=1.05,
        reference_hu=40.0
    ),
    "bone_cancellous": PhysicalMaterial(
        name="Cancellous Bone",
        attenuation_key="bone_cancellous",
        density=1.18,
        reference_hu=400.0
    ),
    "bone_cortical": PhysicalMaterial(
        name="Cortical Bone",
        attenuation_key="bone_cortical",
        density=1.85,
        reference_hu=1000.0
    ),
    "calcium": PhysicalMaterial(
        name="Calcium",
        attenuation_key="calcium",
        density=1.55,
        reference_hu=1500.0
    ),
    "titanium": PhysicalMaterial(
        name="Titanium",
        attenuation_key="titanium",
        density=4.50,
        reference_hu=3000.0
    ),
    "aluminum": PhysicalMaterial(
        name="Aluminum",
        attenuation_key="aluminum",
        density=2.70,
        reference_hu=1500.0
    ),
    "iron": PhysicalMaterial(
        name="Iron/Steel",
        attenuation_key="iron",
        density=7.87,
        reference_hu=8000.0
    ),
    "iodine_contrast": PhysicalMaterial(
        name="Iodine Contrast",
        attenuation_key="iodine",
        density=1.03,  # Dilute solution
        reference_hu=300.0
    ),
}


def get_physical_material(name: str) -> Optional[PhysicalMaterial]:
    """Get physical material by name (case-insensitive)."""
    return PHYSICAL_MATERIALS.get(name.lower())


def material_type_to_physical(material_type_value: str) -> Optional[PhysicalMaterial]:
    """
    Convert MaterialType enum value to PhysicalMaterial.
    
    Handles mapping between simple material names and physical database.
    """
    # Direct mapping where names match
    mapping = {
        "air": "air",
        "water": "water",
        "fat": "fat",
        "soft_tissue": "soft_tissue",
        "muscle": "muscle",
        "blood": "soft_tissue",  # Approximate as soft tissue
        "liver": "soft_tissue",  # Approximate as soft tissue
        "bone_cancellous": "bone_cancellous",
        "bone_cortical": "bone_cortical",
        "calcium": "calcium",
        "titanium": "titanium",
        "steel": "iron",
        "aluminum": "aluminum",
    }
    
    physical_key = mapping.get(material_type_value.lower())
    if physical_key:
        return PHYSICAL_MATERIALS.get(physical_key)
    return None
