"""Single-point evaluator helpers kept from the 0.2 surface API.

Fixed-geometry scans are built with ``pyscf_vscf.scans`` (``plan_bond_scan``,
``plan_normal_scan``, ``run_scan``); the 0.2 grid functions that lived here
were removed in 0.4.
"""

from .electronic import AU_DIPOLE_TO_DEBYE, EnergyDipoleEvaluator, energy_dipole

__all__ = [
    "AU_DIPOLE_TO_DEBYE",
    "EnergyDipoleEvaluator",
    "energy_dipole",
]
