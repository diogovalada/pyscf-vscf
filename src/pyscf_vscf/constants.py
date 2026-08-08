"""Physical constants used by the local-mode vibrational routines."""

from __future__ import annotations

AMU = 1822.888486209  # electron masses
ANG_TO_BOHR = 1.889726124565062
HARTREE_TO_CM = 219474.6313705
DEBYE_TO_CM = 3.33564e-30
SPEED_OF_LIGHT_M_S = 299792458.0
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
HBAR_J_S = 1.054571817e-34
AVOGADRO_MOL_INV = 6.02214076e23

MASS_AMU = {
    "C": 12.0,
    "CL": 34.968852682,
    "H": 1.00782503223,
    "D": 2.01410177812,
    "F": 18.99840316273,
    "N": 14.00307400443,
    "O": 15.99491461957,
}


def atomic_mass_amu(symbol: str) -> float:
    """Return the default analysis mass in amu for an atomic or isotope symbol."""

    key = str(symbol).upper()
    if key in MASS_AMU:
        return float(MASS_AMU[key])

    # PySCF is imported only for elements not covered by the exact-isotope
    # overrides above, keeping numerical-only imports lightweight.
    try:
        from pyscf.data import elements

        charge = int(elements.charge(key))
        if charge <= 0:
            raise ValueError
        return float(elements.MASSES[charge])
    except (ImportError, KeyError, TypeError, ValueError, IndexError) as exc:
        raise ValueError(f"Unknown atom or isotope symbol {symbol!r}") from exc
