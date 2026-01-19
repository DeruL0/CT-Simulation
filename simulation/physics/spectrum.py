"""
X-ray Spectrum Generation

Generates realistic X-ray spectra based on tube voltage (kVp) and filtration.
Uses Kramers' law approximation with characteristic radiation peaks.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


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
                filtration_mm_al=self.filtration_mm_al
            )
        return self


class SpectrumGenerator:
    """
    Generates X-ray spectra for CT simulation.
    
    Uses a simplified Kramers' law model with optional characteristic
    radiation peaks (K-alpha lines for Tungsten at ~59 and ~67 keV).
    """
    
    # Tungsten K-edge and characteristic lines
    W_K_ALPHA1 = 59.32  # keV
    W_K_ALPHA2 = 57.98  # keV
    W_K_BETA = 67.24    # keV
    
    # Aluminum mass attenuation coefficients (simplified, cm²/g)
    # Values at selected energies for filtration calculation
    _AL_MU_RHO = {
        20: 3.44, 30: 1.13, 40: 0.52, 50: 0.30, 60: 0.21,
        70: 0.16, 80: 0.14, 90: 0.12, 100: 0.11, 120: 0.09, 140: 0.08
    }
    AL_DENSITY = 2.7  # g/cm³
    
    def __init__(self, energy_bins: int = 140):
        """
        Initialize spectrum generator.
        
        Args:
            energy_bins: Number of energy bins (1 keV each by default)
        """
        self.energy_bins = energy_bins
    
    def generate(
        self,
        kvp: int = 120,
        filtration_mm_al: float = 2.5,
        add_characteristic: bool = True
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
        # Energy range: 1 keV to kVp
        energies = np.arange(1, kvp + 1, dtype=np.float64)
        
        # Kramers' continuous spectrum: I(E) ∝ Z * (E_max - E)
        # Simplified: photons ∝ (kvp - E) / E
        bremsstrahlung = np.zeros_like(energies)
        valid = energies < kvp
        bremsstrahlung[valid] = (kvp - energies[valid]) / energies[valid]
        bremsstrahlung = np.maximum(bremsstrahlung, 0)
        
        # Add characteristic radiation if tube voltage exceeds K-edge
        if add_characteristic and kvp > 70:
            # K-alpha peaks (combined)
            k_alpha_idx = int(self.W_K_ALPHA1) - 1
            if 0 <= k_alpha_idx < len(energies):
                # Characteristic radiation ~10-15% of bremsstrahlung at that energy
                bremsstrahlung[k_alpha_idx] += bremsstrahlung[k_alpha_idx] * 0.5
                bremsstrahlung[k_alpha_idx - 1] += bremsstrahlung[k_alpha_idx - 1] * 0.3
            
            # K-beta peak
            k_beta_idx = int(self.W_K_BETA) - 1
            if 0 <= k_beta_idx < len(energies):
                bremsstrahlung[k_beta_idx] += bremsstrahlung[k_beta_idx] * 0.15
        
        # Apply inherent + added filtration
        photons = self._apply_filtration(energies, bremsstrahlung, filtration_mm_al)
        
        return XRaySpectrum(
            energies=energies,
            photons=photons,
            kvp=kvp,
            filtration_mm_al=filtration_mm_al
        ).normalize()
    
    def _apply_filtration(
        self,
        energies: np.ndarray,
        photons: np.ndarray,
        thickness_mm: float
    ) -> np.ndarray:
        """
        Apply aluminum filtration to spectrum.
        
        Uses Beer-Lambert law: I = I0 * exp(-μ * t)
        """
        if thickness_mm <= 0:
            return photons
        
        thickness_cm = thickness_mm / 10.0
        filtered = np.zeros_like(photons)
        
        for i, E in enumerate(energies):
            # Interpolate mass attenuation coefficient
            mu_rho = self._interpolate_mu_rho(E)
            mu = mu_rho * self.AL_DENSITY  # Linear attenuation (cm⁻¹)
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
        
        # Typical filtration for different kVp settings
        filtration_map = {
            80: 2.0,
            100: 2.5,
            120: 2.5,
            140: 3.0
        }
        filtration = filtration_map.get(kvp, 2.5)
        
        return generator.generate(kvp=kvp, filtration_mm_al=filtration)
