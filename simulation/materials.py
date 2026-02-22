"""
Material Properties Database

Provides absolute linear attenuation properties for materials used in CT simulation.

The core quantity is the linear attenuation coefficient:
    mu(E) = density * (mu/rho)(E)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Mapping, Iterator, Any
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

    # Generic metallic aliases
    TITANIUM = "titanium"
    STEEL = "steel"
    ALUMINUM = "aluminum"
    COPPER = "copper"
    GOLD = "gold"

    # Industrial-specific material definitions
    ALUMINUM_6061 = "aluminum_6061"
    TITANIUM_ALLOY = "titanium_alloy"
    CARBON_STEEL = "carbon_steel"
    STAINLESS_STEEL_304 = "stainless_steel_304"
    STAINLESS_STEEL_316 = "stainless_steel_316"

    # High-Z / contrast-like materials
    IODINE_CONTRAST = "iodine_contrast"
    BARIUM_CONTRAST = "barium_contrast"
    GADOLINIUM_CONTRAST = "gadolinium_contrast"

    # Industrial/Common non-metallics
    PLASTIC_PVC = "plastic_pvc"
    PLASTIC_PE = "plastic_pe"
    FOAM_SPONGE = "foam_sponge"
    RUBBER = "rubber"

    # Food materials (kept for legacy/demo workflows)
    BREAD = "bread"
    CHOCOLATE = "chocolate"
    CHEESE = "cheese"
    FRUIT = "fruit"

    CUSTOM = "custom"

    @classmethod
    def parse(
        cls,
        value: "MaterialType | str",
        *,
        allow_custom: bool = False,
    ) -> "MaterialType":
        """
        Parse external material representation into `MaterialType`.

        This is the domain-level deserialization entrypoint and should be used at
        I/O boundaries (UI text, config text, persisted payloads), not inside
        simulation algorithm classes.
        """
        if isinstance(value, cls):
            parsed = value
        elif isinstance(value, str):
            text = value.strip().lower()
            if not text:
                raise ValueError("Material value must be a non-empty string.")
            try:
                parsed = cls(text)
            except ValueError as exc:
                raise ValueError(f"Unknown material type: {value!r}") from exc
        else:
            raise TypeError(
                f"Material value must be MaterialType or str, got {type(value).__name__}."
            )

        if parsed == cls.CUSTOM and not allow_custom:
            raise ValueError("MaterialType.CUSTOM is not valid in this context.")
        return parsed


@dataclass
class Material:
    """
    Material properties for CT simulation.

    Attributes:
        name: Human-readable name
        density: Physical density in g/cm^3
        color: RGB color for visualization (0-255)
        attenuation_key: Key in AttenuationDatabase (for energy-dependent physics)
        reference_energy_keV: Reference energy used for `linear_attenuation`
    """

    name: str
    density: float  # g/cm^3
    color: tuple = (128, 128, 128)  # RGB (0-255)
    attenuation_key: Optional[str] = None
    reference_energy_keV: float = 100.0

    @property
    def linear_attenuation(self) -> float:
        """
        Absolute linear attenuation coefficient at reference energy (cm^-1).

        Preferred path uses attenuation table lookup and density scaling.
        Fallback uses density-scaled water-like approximation.
        """
        if self.attenuation_key:
            from .physics.attenuation import get_attenuation_database

            db = get_attenuation_database()
            table = db.get_table(self.attenuation_key)
            if table is not None:
                return table.get_mu_rho(self.reference_energy_keV) * self.density

        # Fallback for materials without explicit attenuation data.
        return 0.171 * self.density

    def get_mu(self, energy: float) -> float:
        """Get linear attenuation coefficient at energy (cm^-1)."""
        if self.attenuation_key:
            from .physics.attenuation import get_attenuation_database

            db = get_attenuation_database()
            table = db.get_table(self.attenuation_key)
            if table is not None:
                return table.get_mu_rho(energy) * self.density
        return self.linear_attenuation

    def get_mu_array(self, energies: np.ndarray) -> np.ndarray:
        """Get linear attenuation coefficients for an array of energies."""
        if self.attenuation_key:
            from .physics.attenuation import get_attenuation_database

            db = get_attenuation_database()
            table = db.get_table(self.attenuation_key)
            if table is not None:
                mu_rho = np.interp(energies, table.energies, table.mu_rho)
                return mu_rho * self.density
        return np.full_like(energies, self.linear_attenuation)


class MaterialDatabase:
    """
    Database of material properties for CT simulation.

    Uses absolute density and energy-dependent attenuation keys.
    """

    _MATERIALS: Dict[MaterialType, Material] = {
        MaterialType.AIR: Material(
            name="Dry Air",
            density=0.001205,
            color=(20, 20, 40),
            attenuation_key="air",
        ),
        MaterialType.WATER: Material(
            name="Liquid Water",
            density=1.000,
            color=(0, 100, 200),
            attenuation_key="water",
        ),
        MaterialType.FAT: Material(
            name="Fat",
            density=0.920,
            color=(255, 220, 150),
            attenuation_key="adipose",
        ),
        MaterialType.SOFT_TISSUE: Material(
            name="Soft Tissue",
            density=1.060,
            color=(255, 180, 180),
            attenuation_key="soft_tissue",
        ),
        MaterialType.MUSCLE: Material(
            name="Muscle",
            density=1.050,
            color=(180, 80, 80),
            attenuation_key="muscle",
        ),
        MaterialType.BLOOD: Material(
            name="Blood",
            density=1.060,
            color=(200, 0, 0),
            attenuation_key="soft_tissue",
        ),
        MaterialType.LIVER: Material(
            name="Liver",
            density=1.050,
            color=(160, 82, 45),
            attenuation_key="soft_tissue",
        ),
        MaterialType.BONE_CANCELLOUS: Material(
            name="Cancellous Bone",
            density=1.180,
            color=(200, 200, 180),
            attenuation_key="bone_cancellous",
        ),
        MaterialType.BONE_CORTICAL: Material(
            name="Cortical Bone",
            density=1.850,
            color=(255, 255, 240),
            attenuation_key="bone_cortical",
        ),
        MaterialType.CALCIUM: Material(
            name="Calcium",
            density=1.550,
            color=(255, 255, 255),
            attenuation_key="calcium",
        ),
        MaterialType.ALUMINUM: Material(
            name="Aluminum",
            density=2.700,
            color=(200, 200, 210),
            attenuation_key="aluminum",
        ),
        MaterialType.ALUMINUM_6061: Material(
            name="Aluminum 6061",
            density=2.700,
            color=(205, 205, 215),
            attenuation_key="aluminum_6061",
        ),
        MaterialType.TITANIUM: Material(
            name="Pure Titanium",
            density=4.506,
            color=(180, 180, 200),
            attenuation_key="titanium",
        ),
        MaterialType.TITANIUM_ALLOY: Material(
            name="Ti-6Al-4V",
            density=4.430,
            color=(175, 175, 195),
            attenuation_key="titanium_alloy",
        ),
        MaterialType.STEEL: Material(
            name="Stainless Steel",
            density=7.930,
            color=(150, 150, 160),
            attenuation_key="stainless_steel_304",
        ),
        MaterialType.CARBON_STEEL: Material(
            name="Carbon Steel",
            density=7.860,
            color=(135, 135, 145),
            attenuation_key="carbon_steel",
        ),
        MaterialType.STAINLESS_STEEL_304: Material(
            name="Stainless Steel 304",
            density=7.930,
            color=(148, 148, 160),
            attenuation_key="stainless_steel_304",
        ),
        MaterialType.STAINLESS_STEEL_316: Material(
            name="Stainless Steel 316",
            density=8.000,
            color=(145, 145, 158),
            attenuation_key="stainless_steel_316",
        ),
        MaterialType.COPPER: Material(
            name="Copper",
            density=8.960,
            color=(170, 120, 90),
            attenuation_key="copper",
        ),
        MaterialType.GOLD: Material(
            name="Pure Gold (Z=79)",
            density=19.320,
            color=(230, 180, 30),
            attenuation_key="gold",
        ),
        MaterialType.IODINE_CONTRAST: Material(
            name="Iodine",
            density=4.930,
            color=(255, 200, 0),
            attenuation_key="iodine",
        ),
        MaterialType.BARIUM_CONTRAST: Material(
            name="Barium",
            density=3.620,
            color=(250, 245, 180),
            attenuation_key="barium",
        ),
        MaterialType.GADOLINIUM_CONTRAST: Material(
            name="Gadolinium",
            density=7.900,
            color=(220, 220, 240),
            attenuation_key="gadolinium",
        ),
        MaterialType.PLASTIC_PVC: Material(
            name="PVC",
            density=1.400,
            color=(200, 200, 220),
            attenuation_key="pvc",
        ),
        MaterialType.PLASTIC_PE: Material(
            name="Polyethylene (PE)",
            density=0.950,
            color=(240, 240, 250),
            attenuation_key="polyethylene",
        ),
        MaterialType.FOAM_SPONGE: Material(
            name="PU Foam / Sponge",
            density=0.030,
            color=(255, 255, 200),
            attenuation_key="polyurethane_foam",
        ),
        MaterialType.RUBBER: Material(
            name="Rubber",
            density=1.100,
            color=(60, 60, 60),
            attenuation_key="rubber",
        ),
        MaterialType.BREAD: Material(
            name="Bread",
            density=0.250,
            color=(210, 180, 140),
            attenuation_key="bread",
        ),
        MaterialType.CHOCOLATE: Material(
            name="Chocolate",
            density=1.250,
            color=(80, 45, 20),
            attenuation_key="chocolate",
        ),
        MaterialType.CHEESE: Material(
            name="Cheese",
            density=1.050,
            color=(255, 220, 100),
            attenuation_key="cheese",
        ),
        MaterialType.FRUIT: Material(
            name="Fruit / Apple",
            density=0.850,
            color=(200, 50, 50),
            attenuation_key="fruit",
        ),
    }

    def __init__(self):
        self._custom_materials: Dict[str, Material] = {}

    def get_material(self, material_type: MaterialType) -> Material:
        """Get material properties by type."""
        if material_type == MaterialType.CUSTOM:
            raise ValueError("Use get_custom_material() for custom materials")
        return self._MATERIALS[material_type]

    def add_custom_material(
        self,
        name: str,
        density: float,
        color: tuple = (128, 128, 128),
        attenuation_key: Optional[str] = None,
        reference_energy_keV: float = 100.0,
    ) -> Material:
        """Add a custom material to the database."""
        material = Material(
            name=name,
            density=density,
            color=color,
            attenuation_key=attenuation_key,
            reference_energy_keV=reference_energy_keV,
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


# ============== Backward Compatibility ==============

PhysicalMaterial = Material

def _material_type_from_key(material_key: Any) -> Optional[MaterialType]:
    """Resolve a material key to MaterialType, returning None for invalid/custom keys."""
    if material_key is None:
        return None
    try:
        return MaterialType.parse(material_key, allow_custom=False)
    except (TypeError, ValueError):
        return None


class _PhysicalMaterialRegistry(Mapping[str, Material]):
    """
    Read-only compatibility view over MaterialDatabase.

    This avoids maintaining a copied/stale physical-material dictionary.
    """

    def __getitem__(self, key: str) -> Material:
        mat_type = _material_type_from_key(key)
        if mat_type is None:
            raise KeyError(key)
        return MaterialDatabase._MATERIALS[mat_type]

    def __iter__(self) -> Iterator[str]:
        return (mat_type.value for mat_type in MaterialType if mat_type != MaterialType.CUSTOM)

    def __len__(self) -> int:
        return sum(1 for mat_type in MaterialType if mat_type != MaterialType.CUSTOM)

    def get(self, key: str, default: Optional[Material] = None) -> Optional[Material]:
        mat_type = _material_type_from_key(key)
        if mat_type is None:
            return default
        return MaterialDatabase._MATERIALS.get(mat_type, default)


PHYSICAL_MATERIALS: Mapping[str, Material] = _PhysicalMaterialRegistry()


def get_physical_material(name: str) -> Optional[Material]:
    """Get physical material by name or key (case-insensitive)."""
    return material_type_to_physical(name)


def material_type_to_physical(material_type_value: Any) -> Optional[Material]:
    """Resolve a material key or MaterialType to physical Material."""
    mat_type = _material_type_from_key(material_type_value)
    if mat_type is None:
        return None
    return MaterialDatabase._MATERIALS.get(mat_type)


def require_physical_material(material_type: MaterialType) -> Material:
    """
    Resolve a strict `MaterialType` to `Material` or raise.
    """
    if not isinstance(material_type, MaterialType):
        raise TypeError(
            f"material_type must be MaterialType, got {type(material_type).__name__}."
        )
    if material_type == MaterialType.CUSTOM:
        raise ValueError("MaterialType.CUSTOM is not supported by CT simulators.")
    return MaterialDatabase._MATERIALS[material_type]
