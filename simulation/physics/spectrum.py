"""
X-ray Spectrum Generation

Generates realistic X-ray spectra based on tube voltage (kVp) and filtration.
Uses Kramers' law approximation with tungsten target self-absorption and
characteristic radiation peaks.
"""

from dataclasses import dataclass
import numpy as np

from .attenuation import get_attenuation_database


@dataclass
class XRaySpectrum:
    """
    Represents an X-ray spectrum.

    Attributes:
        energies: Energy values in keV (typically 1 to kVp)
        photons: Relative photon fluence at each energy
        kvp: Tube voltage in kV
        filtration_mm_al: Aluminum equivalent filtration in mm
    """

    energies: np.ndarray  # keV
    photons: np.ndarray   # Relative fluence (normalized)
    kvp: int
    filtration_mm_al: float

    @property
    def mean_energy(self) -> float:
        """Calculate the mean energy of the spectrum."""
        return np.average(self.energies, weights=self.photons)

    @property
    def effective_energy(self) -> float:
        """
        Approximate effective energy (energy of monochromatic beam
        with equivalent attenuation in water).
        Roughly 1/3 to 1/2 of peak kVp after filtration.
        """
        return self.mean_energy

    def normalize(self) -> "XRaySpectrum":
        """Return a normalized spectrum (total fluence = 1)."""
        total = np.sum(self.photons)
        if total > 0:
            return XRaySpectrum(
                energies=self.energies.copy(),
                photons=self.photons / total,
                kvp=self.kvp,
                filtration_mm_al=self.filtration_mm_al,
            )
        return self


class SpectrumGenerator:
    """
    Generates X-ray spectra for CT simulation.

    Uses Kramers' law for bremsstrahlung and overvoltage-scaled tungsten
    characteristic radiation.
    """

    # Tungsten K-edge and characteristic lines
    W_K_EDGE = 69.50    # keV (K-shell binding energy)
    W_K_ALPHA1 = 59.32  # keV
    W_K_ALPHA2 = 57.98  # keV
    W_K_BETA = 67.24    # keV

    # Characteristic line coefficients for:
    # I_k = C_k * (kVp - E_edge)^1.67
    W_CHARACTERISTIC_COEFFS = {
        W_K_ALPHA1: 1.0e-3,
        W_K_ALPHA2: 5.6e-4,
        W_K_BETA: 2.2e-4,
    }

    # Typical tungsten target geometry for self-absorption approximation.
    W_TARGET_ANGLE_DEG = 12.0
    W_EFFECTIVE_DEPTH_UM = 12.0

    # Aluminum mass attenuation coefficients (simplified, cm^2/g)
    _AL_MU_RHO = {
        20: 3.44,
        30: 1.13,
        40: 0.52,
        50: 0.30,
        60: 0.21,
        70: 0.16,
        80: 0.14,
        90: 0.12,
        100: 0.11,
        120: 0.09,
        140: 0.08,
    }
    AL_DENSITY = 2.7  # g/cm^3

    def __init__(self):
        """Initialize spectrum generator."""
        attenuation_db = get_attenuation_database()
        self._w_table = attenuation_db.get_table("tungsten")

    def generate(
        self,
        kvp: int = 120,
        filtration_mm_al: float = 2.5,
        add_characteristic: bool = True,
    ) -> XRaySpectrum:
        """
        Generate an X-ray spectrum.

        Args:
            kvp: Tube voltage in kV (typically 80-140)
            filtration_mm_al: Aluminum equivalent filtration in mm
            add_characteristic: Whether to add characteristic radiation peaks

        Returns:
            XRaySpectrum object
        """
        # Energy range: 1 keV to kVp.
        energies = np.arange(1, kvp + 1, dtype=np.float64)

        # Kramers continuous spectrum (scaled form): I(E) ~ (Emax - E) / E.
        bremsstrahlung = np.zeros_like(energies)
        valid = energies < kvp
        bremsstrahlung[valid] = (kvp - energies[valid]) / energies[valid]
        bremsstrahlung = np.maximum(bremsstrahlung, 0.0)

        # Tungsten target self-absorption suppresses unphysical low-energy photons.
        target_transmission = self._target_self_absorption_transmission(energies)
        bremsstrahlung *= target_transmission

        # Add characteristic lines with overvoltage scaling.
        if add_characteristic and kvp > self.W_K_EDGE:
            self._add_characteristic_lines(
                energies=energies,
                photons=bremsstrahlung,
                kvp=kvp,
                target_transmission=target_transmission,
            )

        # Apply inherent + added aluminum filtration.
        photons = self._apply_filtration(energies, bremsstrahlung, filtration_mm_al)

        return XRaySpectrum(
            energies=energies,
            photons=photons,
            kvp=kvp,
            filtration_mm_al=filtration_mm_al,
        ).normalize()

    def _target_self_absorption_transmission(self, energies: np.ndarray) -> np.ndarray:
        """
        Compute tungsten target self-absorption transmission.

        Effective tungsten thickness:
            t_eff = depth / sin(target_angle)
        """
        if self._w_table is None:
            return np.ones_like(energies, dtype=np.float64)

        depth_cm = self.W_EFFECTIVE_DEPTH_UM * 1e-4  # 1 um = 1e-4 cm
        angle_rad = np.deg2rad(max(self.W_TARGET_ANGLE_DEG, 1e-3))
        effective_thickness_cm = depth_cm / np.sin(angle_rad)

        mu_w = self._w_table.get_mu_array(energies)
        return np.exp(-mu_w * effective_thickness_cm)

    def _add_characteristic_lines(
        self,
        energies: np.ndarray,
        photons: np.ndarray,
        kvp: int,
        target_transmission: np.ndarray,
    ) -> None:
        """
        Add tungsten characteristic lines using overvoltage scaling:
            I_k = C_k * (kVp - E_edge)^1.67
        """
        overvoltage = max(float(kvp) - self.W_K_EDGE, 0.0)
        if overvoltage <= 0.0:
            return

        overvoltage_term = overvoltage ** 1.67

        for line_energy, coeff in self.W_CHARACTERISTIC_COEFFS.items():
            if line_energy >= kvp:
                continue

            line_idx = int(np.argmin(np.abs(energies - line_energy)))
            line_intensity = coeff * overvoltage_term
            photons[line_idx] += line_intensity * target_transmission[line_idx]

    def _apply_filtration(
        self,
        energies: np.ndarray,
        photons: np.ndarray,
        thickness_mm: float,
    ) -> np.ndarray:
        """
        Apply aluminum filtration to spectrum.

        Uses Beer-Lambert law: I = I0 * exp(-mu * t)
        """
        if thickness_mm <= 0:
            return photons

        thickness_cm = thickness_mm / 10.0
        filtered = np.zeros_like(photons)

        for i, energy in enumerate(energies):
            mu_rho = self._interpolate_mu_rho(energy)
            mu = mu_rho * self.AL_DENSITY  # linear attenuation (cm^-1)
            filtered[i] = photons[i] * np.exp(-mu * thickness_cm)

        return filtered

    def _interpolate_mu_rho(self, energy: float) -> float:
        """Interpolate aluminum mass attenuation coefficient."""
        energies = np.array(list(self._AL_MU_RHO.keys()))
        values = np.array(list(self._AL_MU_RHO.values()))

        if energy <= energies[0]:
            return values[0]
        if energy >= energies[-1]:
            return values[-1]

        return np.interp(energy, energies, values)

    @staticmethod
    def typical_clinical_spectrum(kvp: int = 120) -> XRaySpectrum:
        """
        Generate a typical clinical CT spectrum.

        Uses standard filtration values for diagnostic CT.
        """
        generator = SpectrumGenerator()

        filtration_map = {
            80: 2.0,
            100: 2.5,
            120: 2.5,
            140: 3.0,
        }
        filtration = filtration_map.get(kvp, 2.5)

        return generator.generate(kvp=kvp, filtration_mm_al=filtration)
