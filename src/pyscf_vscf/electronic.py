"""Single-point electronic-structure helpers used by scan workflows."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .molecule import Molecule

AU_DIPOLE_TO_DEBYE = 2.541746


class EnergyDipoleEvaluator(Protocol):
    """Callable boundary used by PES/DMS scan workflows.

    The returned energy must be in Hartree. The dipole must be a finite
    three-component Cartesian vector in Debye, expressed in the same fixed
    frame as the input geometry. When persisting a grid from a custom
    evaluator, pass a stable, non-``pyscf`` ``backend_identity`` to the cache
    helper and use the same identifier when loading it.
    """

    def __call__(self, molecule: Molecule, settings: object) -> tuple[float, np.ndarray]: ...


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


__all__ = ["AU_DIPOLE_TO_DEBYE", "EnergyDipoleEvaluator", "energy_dipole"]
