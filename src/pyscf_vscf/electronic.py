"""Electronic point contracts (re-exported) and the legacy scan evaluator API.

The contracts live in ``pyscf_vscf.contracts``; they are re-exported here
so existing imports keep working. ``energy_dipole`` and the
``EnergyDipoleEvaluator`` callable boundary belong to the 0.2 scan API.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from ._identity import FrozenJSONMapping
from ._warnings import deprecated
from .contracts import (
    AU_DIPOLE_TO_DEBYE,
    POINT_SCHEMA_VERSION,
    ElectronicPointRequest,
    ElectronicProvider,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from .molecule import Molecule


class EnergyDipoleEvaluator(Protocol):
    """Callable boundary used by PES/DMS scan workflows.

    The returned energy must be in Hartree. The dipole must be a finite
    three-component Cartesian vector in Debye, expressed in the same fixed
    frame as the input geometry. When persisting a grid from a custom
    evaluator, pass a stable, non-``pyscf`` ``backend_identity`` to the cache
    helper and use the same identifier when loading it.
    """

    def __call__(self, molecule: Molecule, settings: object) -> tuple[float, np.ndarray]: ...


def provider_fingerprint(provider: ElectronicProvider) -> str:
    """Deprecated spelling of ``provider_scientific_fingerprint``."""

    deprecated("provider_fingerprint() is deprecated; use provider_scientific_fingerprint().")
    return provider_scientific_fingerprint(provider)


def energy_dipole(molecule: Molecule, cfg: object) -> tuple[float, np.ndarray]:
    """Run one built-in PySCF single point; return Hartree energy and Debye dipole.

    This is the default evaluator of the 0.2 tuple-returning scan functions.
    """

    from .backends import pyscf as pyscf_backend

    return pyscf_backend.legacy_energy_dipole(molecule, cfg)


__all__ = [
    "AU_DIPOLE_TO_DEBYE",
    "ElectronicPointRequest",
    "ElectronicProvider",
    "ElectronicResult",
    "EnergyDipoleEvaluator",
    "FrozenJSONMapping",
    "POINT_SCHEMA_VERSION",
    "energy_dipole",
    "provider_fingerprint",
    "provider_scientific_fingerprint",
]
