"""
X-ray Attenuation Database

Provides energy-dependent mass attenuation coefficients (mu/rho) for
materials used in CT simulation.

Data format supports per-material non-uniform energy grids so K-absorption
edge discontinuities can be represented explicitly.
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np


@dataclass
class AttenuationTable:
    """
    Energy-dependent attenuation data for a material.

    Attributes:
        name: Material name
        energies: Energy values in keV (strictly increasing)
        mu_rho: Mass attenuation coefficients in cm^2/g
        density: Physical density in g/cm^3
    """

    name: str
    energies: np.ndarray  # keV
    mu_rho: np.ndarray  # cm^2/g
    density: float  # g/cm^3

    def get_mu_rho(self, energy: float) -> float:
        """Get mass attenuation coefficient at given energy (interpolated)."""
        return float(np.interp(energy, self.energies, self.mu_rho))

    def get_mu(self, energy: float) -> float:
        """Get linear attenuation coefficient at given energy (cm^-1)."""
        return self.get_mu_rho(energy) * self.density

    def get_mu_array(self, energies: np.ndarray) -> np.ndarray:
        """Get linear attenuation coefficients for array of energies."""
        mu_rho = np.interp(energies, self.energies, self.mu_rho)
        return mu_rho * self.density


class AttenuationDatabase:
    """
    Database of mass attenuation coefficients for CT materials.

    Reference source: NIST XCOM / X-ray mass attenuation tables.
    """

    _COMMON_E = [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0]

    # Format: {material_name: (density, [energies_keV], [mu_rho_cm2_per_g])}
    _DATA: Dict[str, tuple] = {
        "air": (
            0.001205,
            _COMMON_E,
            [0.335, 0.241, 0.207, 0.193, 0.1875, 0.1760, 0.165, 0.157, 0.150, 0.146, 0.1420, 0.138, 0.134, 0.130],
        ),
        "water": (
            1.000,
            _COMMON_E,
            [0.810, 0.375, 0.268, 0.227, 0.2059, 0.1930, 0.184, 0.177, 0.171, 0.163, 0.1550, 0.148, 0.143, 0.138],
        ),
        "soft_tissue": (
            1.060,
            _COMMON_E,
            [0.816, 0.379, 0.270, 0.228, 0.2060, 0.1935, 0.184, 0.178, 0.172, 0.165, 0.1580, 0.151, 0.146, 0.141],
        ),
        "adipose": (
            0.920,
            _COMMON_E,
            [0.547, 0.290, 0.223, 0.197, 0.183, 0.174, 0.168, 0.163, 0.159, 0.155, 0.153, 0.150, 0.148, 0.146],
        ),
        # Low-Z industrial/food materials (surrogate NIST-like curves).
        "polyethylene": (
            0.950,
            _COMMON_E,
            [0.560, 0.295, 0.225, 0.198, 0.184, 0.175, 0.169, 0.164, 0.160, 0.156, 0.152, 0.149, 0.147, 0.145],
        ),
        "polyurethane_foam": (
            0.030,
            _COMMON_E,
            [0.575, 0.304, 0.231, 0.201, 0.185, 0.176, 0.169, 0.164, 0.160, 0.156, 0.152, 0.149, 0.147, 0.145],
        ),
        "rubber": (
            1.100,
            _COMMON_E,
            [0.640, 0.325, 0.240, 0.208, 0.190, 0.179, 0.171, 0.165, 0.160, 0.156, 0.152, 0.149, 0.146, 0.144],
        ),
        "pvc": (
            1.400,
            _COMMON_E,
            [1.950, 0.780, 0.430, 0.300, 0.240, 0.205, 0.185, 0.173, 0.165, 0.158, 0.151, 0.145, 0.140, 0.136],
        ),
        "bread": (
            0.250,
            _COMMON_E,
            [0.790, 0.365, 0.262, 0.222, 0.202, 0.190, 0.182, 0.176, 0.170, 0.164, 0.158, 0.152, 0.147, 0.142],
        ),
        "fruit": (
            0.850,
            _COMMON_E,
            [0.805, 0.372, 0.266, 0.225, 0.204, 0.191, 0.183, 0.177, 0.171, 0.165, 0.159, 0.153, 0.148, 0.143],
        ),
        "cheese": (
            1.050,
            _COMMON_E,
            [0.760, 0.350, 0.252, 0.216, 0.198, 0.188, 0.181, 0.175, 0.170, 0.165, 0.160, 0.155, 0.151, 0.147],
        ),
        "chocolate": (
            1.250,
            _COMMON_E,
            [0.700, 0.330, 0.240, 0.208, 0.193, 0.184, 0.177, 0.172, 0.168, 0.164, 0.160, 0.156, 0.152, 0.149],
        ),
        "muscle": (
            1.050,
            _COMMON_E,
            [0.820, 0.385, 0.271, 0.228, 0.206, 0.193, 0.184, 0.177, 0.171, 0.167, 0.163, 0.160, 0.157, 0.155],
        ),
        "bone_cortical": (
            1.850,
            _COMMON_E,
            [2.867, 0.990, 0.511, 0.370, 0.3148, 0.2480, 0.210, 0.182, 0.166, 0.157, 0.1500, 0.144, 0.138, 0.132],
        ),
        "bone_cancellous": (
            1.180,
            _COMMON_E,
            [1.510, 0.590, 0.340, 0.250, 0.210, 0.186, 0.170, 0.159, 0.151, 0.145, 0.140, 0.136, 0.133, 0.130],
        ),
        "calcium": (
            1.550,
            _COMMON_E,
            [6.040, 2.230, 1.050, 0.600, 0.430, 0.333, 0.270, 0.228, 0.198, 0.178, 0.165, 0.152, 0.140, 0.132],
        ),
        "aluminum": (
            2.700,
            _COMMON_E,
            [3.441, 1.128, 0.520, 0.350, 0.2770, 0.2090, 0.171, 0.147, 0.141, 0.138, 0.1350, 0.128, 0.122, 0.118],
        ),
        "aluminum_6061": (
            2.700,
            _COMMON_E,
            [3.441, 1.128, 0.520, 0.350, 0.2770, 0.2090, 0.171, 0.147, 0.141, 0.138, 0.1350, 0.128, 0.122, 0.118],
        ),
        "titanium": (
            4.506,
            _COMMON_E,
            [7.092, 2.426, 1.115, 0.820, 0.6340, 0.4160, 0.300, 0.234, 0.193, 0.176, 0.1660, 0.154, 0.145, 0.136],
        ),
        "titanium_alloy": (
            4.430,
            _COMMON_E,
            [7.092, 2.426, 1.115, 0.820, 0.6340, 0.4160, 0.300, 0.234, 0.193, 0.176, 0.1660, 0.154, 0.145, 0.136],
        ),
        "iron": (
            7.870,
            _COMMON_E,
            [10.90, 3.770, 1.700, 0.940, 1.1300, 0.7060, 0.490, 0.350, 0.270, 0.235, 0.2170, 0.198, 0.184, 0.172],
        ),
        "carbon_steel": (
            7.860,
            _COMMON_E,
            [10.90, 3.770, 1.700, 0.940, 1.1300, 0.7060, 0.490, 0.350, 0.270, 0.235, 0.2170, 0.198, 0.184, 0.172],
        ),
        "stainless_steel_304": (
            7.930,
            _COMMON_E,
            [10.90, 3.770, 1.700, 0.940, 1.1300, 0.7060, 0.490, 0.350, 0.270, 0.235, 0.2170, 0.198, 0.184, 0.172],
        ),
        "stainless_steel_316": (
            8.000,
            _COMMON_E,
            [10.90, 3.770, 1.700, 0.940, 1.1300, 0.7060, 0.490, 0.350, 0.270, 0.235, 0.2170, 0.198, 0.184, 0.172],
        ),
        "copper": (
            8.960,
            _COMMON_E,
            [13.70, 4.720, 2.090, 1.130, 0.706, 0.487, 0.364, 0.289, 0.241, 0.208, 0.184, 0.166, 0.152, 0.141],
        ),
        "tungsten": (
            19.300,
            [20.0, 30.0, 40.0, 50.0, 60.0, 69.499, 69.501, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0],
            [65.73, 22.73, 10.67, 5.949, 3.713, 3.650, 11.20, 11.03, 7.810, 5.795, 4.438, 3.482, 2.790, 2.276, 1.885, 1.581],
        ),
        # K-edge at 33.169 keV represented via 33.168 / 33.170 keV pair.
        "iodine": (
            4.930,
            [20.0, 30.0, 33.168, 33.170, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0],
            [15.0, 7.5, 6.270, 39.19, 25.0, 18.0, 11.75, 7.123, 5.57, 3.11, 2.074, 1.10],
        ),
        # K-edge at 37.441 keV represented via 37.440 / 37.442 keV pair.
        "barium": (
            3.620,
            [20.0, 30.0, 37.440, 37.442, 40.0, 50.0, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0],
            [18.0, 8.8, 4.768, 29.19, 22.0, 16.0, 13.79, 8.543, 5.80, 3.20, 2.308, 1.20],
        ),
        # K-edge at 50.239 keV represented via 50.238 / 50.240 keV pair.
        "gadolinium": (
            7.900,
            [20.0, 30.0, 40.0, 50.238, 50.240, 60.0, 70.0, 80.0, 100.0, 120.0, 150.0],
            [20.0, 9.8, 5.2, 3.812, 18.64, 11.5, 7.2, 4.9, 2.9, 1.9, 1.0],
        ),
        # K-edge at 80.725 keV represented via 80.724 / 80.726 keV pair.
        "gold": (
            19.320,
            [20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.724, 80.726, 90.0, 100.0, 110.0, 120.0, 130.0, 150.0],
            [120.0, 45.0, 22.0, 12.0, 7.0, 4.2, 2.137, 8.904, 6.8, 5.0, 4.0, 3.2, 2.7, 1.9],
        ),
    }

    def __init__(self):
        """Initialize the attenuation database."""
        self._tables: Dict[str, AttenuationTable] = {}
        self._build_tables()

    def _build_tables(self) -> None:
        """Build interpolation tables from raw data."""
        for name, (density, energies, mu_rho_list) in self._DATA.items():
            energies_np = np.array(energies, dtype=np.float64)
            mu_rho_np = np.array(mu_rho_list, dtype=np.float64)
            if energies_np.shape != mu_rho_np.shape:
                raise ValueError(f"Energy and mu/rho array length mismatch for material '{name}'")
            if np.any(np.diff(energies_np) <= 0):
                raise ValueError(f"Energy grid must be strictly increasing for material '{name}'")

            self._tables[name] = AttenuationTable(
                name=name,
                energies=energies_np,
                mu_rho=mu_rho_np,
                density=float(density),
            )

    def get_table(self, material: str) -> Optional[AttenuationTable]:
        """Get attenuation table for a material."""
        return self._tables.get(material.lower())

    def get_mu(self, material: str, energy: float) -> float:
        """Get linear attenuation coefficient (cm^-1) for material at energy."""
        table = self.get_table(material)
        if table is None:
            raise KeyError(f"Material '{material}' not in database")
        return table.get_mu(energy)

    def get_mu_array(self, material: str, energies: np.ndarray) -> np.ndarray:
        """Get linear attenuation coefficients for array of energies."""
        table = self.get_table(material)
        if table is None:
            raise KeyError(f"Material '{material}' not in database")
        return table.get_mu_array(energies)

    def list_materials(self) -> list:
        """List available materials."""
        return list(self._tables.keys())

    def get_density(self, material: str) -> float:
        """Get material density in g/cm^3."""
        table = self.get_table(material)
        if table is None:
            raise KeyError(f"Material '{material}' not in database")
        return table.density


_database: Optional[AttenuationDatabase] = None


def get_attenuation_database() -> AttenuationDatabase:
    """Get the global attenuation database instance."""
    global _database
    if _database is None:
        _database = AttenuationDatabase()
    return _database


def get_attenuation(material: str, energy: float) -> float:
    """Convenience function to get linear attenuation coefficient (cm^-1)."""
    return get_attenuation_database().get_mu(material, energy)
