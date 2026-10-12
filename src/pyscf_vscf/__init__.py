"""pyscf-vscf public package namespace.

Pure numerical modules are importable without PySCF; backend workflows import
PySCF only when explicitly used.
"""

import logging as _logging

from .assignments import ProductStateAssignment, assign_product_states_2d
from .constants import AMU, ANG_TO_BOHR, DEBYE_TO_CM, HARTREE_TO_CM, atomic_mass_amu
from .coordinates import Bond
from .electronic import EnergyDipoleEvaluator
from .backends.pyscf import MethodRecipe, PySCFMeanFieldProvider
from .harmonic import HarmonicResult, mass_weighted_freqs_modes
from .io import read_geometry, read_midas_mmol, read_xyz, write_midas_mmol, write_xyz
from .point_store import PointStore
from .runner import PlanIncompleteError, RunnerSettings, run_plan
from .scans import (
    ScanResult,
    bond_spectrum,
    normal_coordinate_spectrum,
    plan_bond_scan,
    plan_normal_scan,
    run_scan,
)
from .molecule import Molecule
from .settings import ESSettings, HarmonicSettings, RuntimeSettings
from .spectra import (
    einstein_a_from_debye,
    integrated_cross_section_omega,
    integrated_cross_section_omega_to_km_per_mol,
)
from .validation import ConvergenceReport, ExactProductDVR, convergence_report, exact_nmode_dvr
from .variational import TransitionRecord, variational_1d, variational_2d
from .vscf import (
    NModePotential,
    VSCFSettings,
    VSCFSpectrum,
    VSCFStateResult,
    VSCFTransition,
    dump_nmode_model,
    load_nmode_model,
    nmode_model_from_pair_surfaces,
    solve_vscf_state,
    vscf_spectrum,
)

__all__ = [
    "AMU",
    "MethodRecipe",
    "PlanIncompleteError",
    "PointStore",
    "PySCFMeanFieldProvider",
    "RunnerSettings",
    "ScanResult",
    "bond_spectrum",
    "normal_coordinate_spectrum",
    "plan_bond_scan",
    "plan_normal_scan",
    "read_geometry",
    "read_midas_mmol",
    "read_xyz",
    "run_plan",
    "run_scan",
    "write_midas_mmol",
    "write_xyz",
    "ANG_TO_BOHR",
    "Bond",
    "ConvergenceReport",
    "DEBYE_TO_CM",
    "ESSettings",
    "EnergyDipoleEvaluator",
    "ExactProductDVR",
    "HARTREE_TO_CM",
    "HarmonicResult",
    "HarmonicSettings",
    "Molecule",
    "ProductStateAssignment",
    "RuntimeSettings",
    "TransitionRecord",
    "NModePotential",
    "VSCFSettings",
    "VSCFSpectrum",
    "VSCFStateResult",
    "VSCFTransition",
    "assign_product_states_2d",
    "atomic_mass_amu",
    "convergence_report",
    "dump_nmode_model",
    "einstein_a_from_debye",
    "mass_weighted_freqs_modes",
    "integrated_cross_section_omega",
    "integrated_cross_section_omega_to_km_per_mol",
    "load_nmode_model",
    "nmode_model_from_pair_surfaces",
    "exact_nmode_dvr",
    "solve_vscf_state",
    "variational_1d",
    "variational_2d",
    "vscf_spectrum",
]

_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__version__ = "0.3.1"
