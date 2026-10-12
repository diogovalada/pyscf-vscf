"""Physical constants used by the local-mode vibrational routines."""

from __future__ import annotations

AMU = 1822.888486209
"""One unified atomic mass unit (dalton) in electron masses."""
ANG_TO_BOHR = 1.889726124565062
"""Bohr radii per angstrom."""
HARTREE_TO_CM = 219474.6313705
"""Wavenumbers (cm^-1) per hartree."""
ATOMIC_DIPOLE_TO_DEBYE = 2.541746473
"""Debye per atomic unit of dipole (CODATA), used by the VSCF/VCI transition
moments; the scan stack uses ``contracts.AU_DIPOLE_TO_DEBYE`` (2.541746)."""
DEBYE_TO_CM = 3.33564e-30
"""Coulomb-metres (C m) per debye; "CM" is the SI unit, not wavenumbers."""
SPEED_OF_LIGHT_M_S = 299792458.0
"""Speed of light in vacuum, m s^-1."""
VACUUM_PERMITTIVITY_F_M = 8.8541878128e-12
"""Vacuum permittivity, F m^-1."""
HBAR_J_S = 1.054571817e-34
"""Reduced Planck constant, J s."""
AVOGADRO_MOL_INV = 6.02214076e23
"""Avogadro constant, mol^-1."""

MASS_AMU = {
    "C": 12.0,
    "CL": 34.968852682,
    "H": 1.00782503223,
    "D": 2.01410177812,
    "F": 18.99840316273,
    "N": 14.00307400443,
    "O": 15.99491461957,
}
"""Most-abundant-isotope masses (amu) for these elements, plus deuterium.
Other elements fall back to PySCF's ``elements.MASSES`` (standard atomic
weights, not single-isotope masses)."""


ELEMENT_SYMBOLS = tuple(
    """H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn
    Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr
    Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr
    Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc
    Lv Ts Og""".split()
)
"""Element symbols ordered by atomic number (``ELEMENT_SYMBOLS[0]`` is H)."""
_ATOMIC_NUMBERS = {symbol.upper(): index for index, symbol in enumerate(ELEMENT_SYMBOLS, 1)}


def atomic_number(symbol: str) -> int:
    """Return the atomic number of an element symbol; ``D`` (deuterium) gives 1."""

    key = str(symbol).strip().upper()
    if key == "D":
        return 1
    try:
        return _ATOMIC_NUMBERS[key]
    except KeyError:
        raise ValueError(f"Unknown element symbol {symbol!r}") from None


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
