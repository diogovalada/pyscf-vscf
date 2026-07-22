"""PES/DMS grid construction helpers.

The functions in this module are intentionally serial and accept injectable
single-point evaluators. The legacy driver can keep its multiprocessing wrapper
while sharing the same geometry/grid semantics.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .coordinates import Bond, stretch_along_bond
from .molecule import Molecule

EnergyDipoleFn = Callable[[Molecule, object], tuple[float, np.ndarray]]

AU_DIPOLE_TO_DEBYE = 2.541746


def energy_dipole(molecule: Molecule, cfg: object) -> tuple[float, np.ndarray]:
    """Run one PySCF-backed single point and return energy plus dipole in Debye."""

    from .backends import pyscf as pyscf_backend
    from .settings import coerce_es_settings

    settings = coerce_es_settings(cfg)
    pmol = pyscf_backend.molecule_to_pyscf(molecule, settings.basis)
    mf = pyscf_backend.make_mean_field(pmol, settings)
    energy = float(mf.e_tot)
    dm = mf.make_rdm1()
    try:
        mu_au = mf.dip_moment(dm=dm, unit="au", verbose=0)
    except TypeError:
        mu_au = mf.dip_moment(unit="au", verbose=0)
    return energy, np.asarray(mu_au, dtype=float) * AU_DIPOLE_TO_DEBYE


def _sub_molecule_like(molecule: Molecule, coords: np.ndarray) -> Molecule:
    return Molecule.from_arrays(
        molecule.symbols,
        coords,
        charge=molecule.charge,
        spin=molecule.spin,
        label=molecule.label,
        masses_amu=molecule.masses,
    )


def grid_1d_pes_dms(
    molecule: Molecule,
    cfg: object,
    bond: Bond,
    Rmin: float = 0.75,
    Rmax: float = 1.25,
    npts: int = 41,
    *,
    energy_dipole_fn: EnergyDipoleFn = energy_dipole,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a frozen local-bond 1D PES/DMS grid."""

    Rs = np.linspace(float(Rmin), float(Rmax), int(npts))
    energies = np.zeros(Rs.size, dtype=float)
    dipoles = np.zeros((Rs.size, 3), dtype=float)
    for idx, R in enumerate(Rs):
        coords = stretch_along_bond(molecule.coords, bond, float(R))
        energies[idx], dipoles[idx] = energy_dipole_fn(_sub_molecule_like(molecule, coords), cfg)
    energies -= float(np.min(energies))
    return Rs, energies, dipoles


def grid_1d_pes_dms_normal(
    molecule: Molecule,
    cfg: object,
    u_dir: np.ndarray,
    smin: float = -0.15,
    smax: float = 0.15,
    npts: int = 41,
    *,
    energy_dipole_fn: EnergyDipoleFn = energy_dipole,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 1D PES/DMS grid along a fixed Cartesian displacement direction."""

    direction = np.asarray(u_dir, dtype=float)
    if direction.shape != molecule.coords.shape:
        raise ValueError(f"u_dir shape {direction.shape} does not match molecule coordinates")
    S = np.linspace(float(smin), float(smax), int(npts))
    energies = np.zeros(S.size, dtype=float)
    dipoles = np.zeros((S.size, 3), dtype=float)
    for idx, s in enumerate(S):
        coords = molecule.coords + float(s) * direction
        energies[idx], dipoles[idx] = energy_dipole_fn(_sub_molecule_like(molecule, coords), cfg)
    energies -= float(np.min(energies))
    return S, energies, dipoles


def grid_2d_pes_dms(
    molecule: Molecule,
    cfg: object,
    b1: Bond,
    b2: Bond,
    R1: np.ndarray,
    R2: np.ndarray,
    *,
    energy_dipole_fn: EnergyDipoleFn = energy_dipole,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a frozen two-local-bond PES/DMS grid."""

    R1_arr = np.asarray(R1, dtype=float)
    R2_arr = np.asarray(R2, dtype=float)
    energies = np.zeros((R1_arr.size, R2_arr.size), dtype=float)
    dipoles = np.zeros((R1_arr.size, R2_arr.size, 3), dtype=float)
    for i, r1 in enumerate(R1_arr):
        coords1 = stretch_along_bond(molecule.coords, b1, float(r1))
        for j, r2 in enumerate(R2_arr):
            coords2 = stretch_along_bond(coords1, b2, float(r2))
            energies[i, j], dipoles[i, j] = energy_dipole_fn(
                _sub_molecule_like(molecule, coords2),
                cfg,
            )
    energies -= float(np.min(energies))
    return R1_arr, R2_arr, energies, dipoles


__all__ = [
    "AU_DIPOLE_TO_DEBYE",
    "energy_dipole",
    "grid_1d_pes_dms",
    "grid_1d_pes_dms_normal",
    "grid_2d_pes_dms",
]
