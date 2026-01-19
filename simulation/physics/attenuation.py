"""
X-ray Attenuation Database

Provides energy-dependent mass attenuation coefficients (μ/ρ) for
common materials used in CT simulation. Data based on NIST XCOM.
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
        energies: Energy values in keV
        mu_rho: Mass attenuation coefficients in cm²/g
        density: Physical density in g/cm³
    """
    name: str
    energies: np.ndarray  # keV
    mu_rho: np.ndarray    # cm²/g (mass attenuation coefficient)
    density: float        # g/cm³
    
    def get_mu_rho(self, energy: float) -> float:
        """Get mass attenuation coefficient at given energy (interpolated)."""
        return np.interp(energy, self.energies, self.mu_rho)
    
    def get_mu(self, energy: float) -> float:
        """Get linear attenuation coefficient at given energy (cm⁻¹)."""
        return self.get_mu_rho(energy) * self.density
    
    def get_mu_array(self, energies: np.ndarray) -> np.ndarray:
        """Get linear attenuation coefficients for array of energies."""
        mu_rho = np.interp(energies, self.energies, self.mu_rho)
        return mu_rho * self.density


class AttenuationDatabase:
    """
    Database of mass attenuation coefficients for CT materials.
    
    Data sourced from NIST XCOM database, simplified for common
    CT energy range (20-150 keV).
    
    Reference: https://physics.nist.gov/PhysRefData/Xcom/html/xcom1.html
    """
    
    # Standard energy points (keV)
    _ENERGIES = np.array([
        20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150
    ], dtype=np.float64)
    
    # Mass attenuation coefficients (cm²/g) from NIST XCOM
    # Format: {material_name: (density, [mu/rho values at each energy])}
    _DATA: Dict[str, tuple] = {
        "air": (0.001205, [
            0.778, 0.387, 0.241, 0.177, 0.145, 0.127, 0.116, 0.109,
            0.103, 0.099, 0.096, 0.093, 0.091, 0.089
        ]),
        "water": (1.0, [
            0.804, 0.376, 0.268, 0.227, 0.206, 0.193, 0.184, 0.177,
            0.171, 0.166, 0.163, 0.160, 0.157, 0.155
        ]),
        "soft_tissue": (1.06, [
            0.810, 0.380, 0.269, 0.226, 0.205, 0.192, 0.183, 0.176,
            0.171, 0.166, 0.162, 0.159, 0.156, 0.154
        ]),
        "adipose": (0.92, [
            0.547, 0.290, 0.223, 0.197, 0.183, 0.174, 0.168, 0.163,
            0.159, 0.155, 0.153, 0.150, 0.148, 0.146
        ]),
        "muscle": (1.05, [
            0.820, 0.385, 0.271, 0.228, 0.206, 0.193, 0.184, 0.177,
            0.171, 0.167, 0.163, 0.160, 0.157, 0.155
        ]),
        "bone_cortical": (1.85, [
            2.867, 0.990, 0.511, 0.330, 0.248, 0.204, 0.178, 0.161,
            0.150, 0.142, 0.136, 0.131, 0.127, 0.124
        ]),
        "bone_cancellous": (1.18, [
            1.510, 0.590, 0.340, 0.250, 0.210, 0.186, 0.170, 0.159,
            0.151, 0.145, 0.140, 0.136, 0.133, 0.130
        ]),
        "calcium": (1.55, [
            6.040, 2.230, 1.050, 0.600, 0.400, 0.297, 0.238, 0.202,
            0.177, 0.160, 0.147, 0.138, 0.130, 0.124
        ]),
        "aluminum": (2.70, [
            3.441, 1.128, 0.520, 0.305, 0.209, 0.161, 0.135, 0.119,
            0.109, 0.101, 0.096, 0.092, 0.088, 0.086
        ]),
        "titanium": (4.50, [
            7.092, 2.426, 1.115, 0.634, 0.416, 0.301, 0.234, 0.193,
            0.166, 0.148, 0.134, 0.124, 0.117, 0.111
        ]),
        "iron": (7.87, [
            10.90, 3.770, 1.700, 0.940, 0.598, 0.420, 0.318, 0.257,
            0.217, 0.189, 0.169, 0.154, 0.142, 0.133
        ]),
        "copper": (8.96, [
            13.70, 4.720, 2.090, 1.130, 0.706, 0.487, 0.364, 0.289,
            0.241, 0.208, 0.184, 0.166, 0.152, 0.141
        ]),
        "iodine": (4.93, [
            30.60, 9.930, 4.280, 2.230, 1.310, 0.856, 0.604, 0.454,
            0.360, 0.297, 0.254, 0.223, 0.199, 0.181
        ]),
    }
    
    def __init__(self):
        """Initialize the attenuation database."""
        self._tables: Dict[str, AttenuationTable] = {}
        self._build_tables()
    
    def _build_tables(self) -> None:
        """Build interpolation tables from raw data."""
        for name, (density, mu_rho_list) in self._DATA.items():
            self._tables[name] = AttenuationTable(
                name=name,
                energies=self._ENERGIES.copy(),
                mu_rho=np.array(mu_rho_list, dtype=np.float64),
                density=density
            )
    
    def get_table(self, material: str) -> Optional[AttenuationTable]:
        """Get attenuation table for a material."""
        return self._tables.get(material.lower())
    
    def get_mu(self, material: str, energy: float) -> float:
        """
        Get linear attenuation coefficient (cm⁻¹) for material at energy.
        
        Args:
            material: Material name (e.g., "water", "bone_cortical")
            energy: Photon energy in keV
            
        Returns:
            Linear attenuation coefficient in cm⁻¹
        """
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
        """Get material density in g/cm³."""
        table = self.get_table(material)
        if table is None:
            raise KeyError(f"Material '{material}' not in database")
        return table.density


# Global database instance
_database: Optional[AttenuationDatabase] = None


def get_attenuation_database() -> AttenuationDatabase:
    """Get the global attenuation database instance."""
    global _database
    if _database is None:
        _database = AttenuationDatabase()
    return _database


def get_attenuation(material: str, energy: float) -> float:
    """
    Convenience function to get linear attenuation coefficient.
    
    Args:
        material: Material name
        energy: Photon energy in keV
        
    Returns:
        Linear attenuation coefficient in cm⁻¹
    """
    return get_attenuation_database().get_mu(material, energy)
