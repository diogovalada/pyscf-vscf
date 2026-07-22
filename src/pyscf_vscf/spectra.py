"""Spectral intensity helpers independent of PySCF.

The integrated cross sections in this module are integrals over angular
frequency, not HITRAN line intensities.  The formulas follow Hilborn,
Am. J. Phys. 50, 982 (1982), DOI: 10.1119/1.12937.  Frequencies are explicit
arguments so a frequency-independent dipole-strength quantity cannot be
mistaken for an absolute absorption cross section.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    AVOGADRO_MOL_INV,
    DEBYE_TO_CM,
    HBAR_J_S,
    SPEED_OF_LIGHT_M_S,
    VACUUM_PERMITTIVITY_F_M,
)


def angular_frequency_from_wavenumber(frequency_cm: float) -> float:
    """Convert a positive spectroscopic wavenumber in cm^-1 to rad s^-1."""

    wavenumber = float(frequency_cm)
    if not np.isfinite(wavenumber) or wavenumber <= 0.0:
        raise ValueError("frequency_cm must be finite and positive")
    return float(2.0 * np.pi * SPEED_OF_LIGHT_M_S * 100.0 * wavenumber)


def integrated_cross_section_omega(
    mu_debye: float,
    frequency_cm: float,
    *,
    orientation_factor: float = 1.0 / 3.0,
) -> float:
    """Return ``integral sigma(omega) d omega`` in m^2 s^-1.

    ``orientation_factor=1/3`` is the isotropic average for the norm of a
    transition-dipole vector.  Use ``orientation_factor=1`` for light
    polarized along a specified dipole projection.
    """

    mu_cm = float(mu_debye) * DEBYE_TO_CM
    if not np.isfinite(mu_cm):
        raise ValueError("mu_debye must be finite")
    factor = float(orientation_factor)
    if not np.isfinite(factor) or factor < 0.0:
        raise ValueError("orientation_factor must be finite and non-negative")
    omega = angular_frequency_from_wavenumber(frequency_cm)
    return float(
        np.pi
        * omega
        * mu_cm
        * mu_cm
        * factor
        / (VACUUM_PERMITTIVITY_F_M * HBAR_J_S * SPEED_OF_LIGHT_M_S)
    )


def einstein_a_from_debye(mu_debye: float, frequency_cm: float) -> float:
    """Return the nondegenerate spontaneous-emission Einstein A in s^-1."""

    mu_cm = float(mu_debye) * DEBYE_TO_CM
    if not np.isfinite(mu_cm):
        raise ValueError("mu_debye must be finite")
    omega = angular_frequency_from_wavenumber(frequency_cm)
    return float(
        omega**3
        * mu_cm
        * mu_cm
        / (3.0 * np.pi * VACUUM_PERMITTIVITY_F_M * HBAR_J_S * SPEED_OF_LIGHT_M_S**3)
    )


def integrated_cross_section_omega_to_km_per_mol(value_m2_per_s: float) -> float:
    """Convert ``integral sigma(omega)domega`` to km mol^-1.

    The conversion uses ``domega = 2*pi*c*d(wavenumber_m^-1)`` and Avogadro's
    constant. It changes units only; temperature and population factors are
    not introduced.
    """

    value = float(value_m2_per_s)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("value_m2_per_s must be finite and non-negative")
    return float(AVOGADRO_MOL_INV * value / (2.0 * np.pi * SPEED_OF_LIGHT_M_S * 1000.0))


__all__ = [
    "angular_frequency_from_wavenumber",
    "einstein_a_from_debye",
    "integrated_cross_section_omega",
    "integrated_cross_section_omega_to_km_per_mol",
]
