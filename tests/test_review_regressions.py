"""Regression tests for defects found in the 0.3.0 full-repository review."""

from __future__ import annotations

import dataclasses
import itertools
import json
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.contracts import (
    ElectronicPointRequest,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from pyscf_vscf.coordinates import Bond, LinearDisplacementCoordinateMap, parse_bond
from pyscf_vscf.molecule import Molecule

# --- shared analytic providers ------------------------------------------------


@dataclass(frozen=True)
class QuarticProvider:
    """Analytic energy and dipole of the displacement from ``REFERENCE``."""

    scale: float = 1.0

    def scientific_settings_payload(self) -> dict[str, object]:
        return {"schema": "test-quartic", "scale": self.scale}

    def execution_provenance(self) -> dict[str, object]:
        return {}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        x = np.asarray(request.coordinates_A, dtype=float) - REFERENCE
        energy = self.scale * float(np.sum(x**2) + 0.3 * np.sum(x**4))
        provider_id = provider_scientific_fingerprint(self)
        return ElectronicResult(
            total_energy_Eh=energy - 1.0,
            dipole_au=np.array([np.sum(x[:, 0]), np.sum(x[:, 1] ** 2), 0.2]),
            dipole_unit="atomic_unit",
            dipole_frame="input_cartesian",
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
        )


REFERENCE = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def _nmode_model(provider, nuclear_charges=(8, 1, 1), charge=0, spin=0):
    from pyscf_vscf.nmode import assemble_nmode_samples, fit_nmode_surface, plan_nmode_points
    from pyscf_vscf.runner import run_plan

    displacements = np.zeros((2, 3, 3))
    displacements[0, 1, 0] = 1.0
    displacements[1, 2, 1] = 1.0
    mapping = LinearDisplacementCoordinateMap(
        reference_geometry_A=REFERENCE,
        coordinate_ids=("q1", "q2"),
        units=("angstrom", "angstrom"),
        reference_values=np.zeros(2),
        displacements_A_per_unit=displacements,
    )
    axes = [np.linspace(-0.2, 0.2, 7)] * 2
    plan = plan_nmode_points(
        mapping, axes, provider, nuclear_charges=nuclear_charges, charge=charge, spin=spin
    )
    run = run_plan(plan, provider)
    return fit_nmode_surface(assemble_nmode_samples(plan, run.results), method="cubic")


# --- lane B: surfaces, coordinates, data --------------------------------------


def test_surface_interpolates_its_own_copy_of_the_inputs() -> None:
    from pyscf_vscf.nmode import _fit_tensor_surface

    axis = np.linspace(-1.0, 1.0, 5)
    values = axis**2
    surface = _fit_tensor_surface(
        (axis,),
        values,
        method="cubic",
        held_out_points=None,
        held_out_values=None,
        held_out_point_ids=(),
    )
    before = float(surface.evaluate(np.array([0.5])))
    values *= 10.0
    axis *= 2.0
    assert float(surface.evaluate(np.array([0.5]))) == before


def test_cubic_fits_compose_and_foreign_systems_are_rejected() -> None:
    from pyscf_vscf.multilevel import compose_multilevel_surface

    low = _nmode_model(QuarticProvider())
    high = _nmode_model(QuarticProvider(scale=1.1))
    model, _ = compose_multilevel_surface(low, high, energy_corrected_subsets=[(0,), (0, 1)])
    assert model.reference_energy_Eh == low.reference_energy_Eh

    cation = _nmode_model(QuarticProvider(scale=1.1), charge=1, spin=1)
    with pytest.raises(ValueError, match="different electronic systems"):
        compose_multilevel_surface(low, cation, energy_corrected_subsets=[(0,)])


def test_scan_definition_must_agree_with_its_axes_and_points() -> None:
    from pyscf_vscf.scans import ScanAxis, plan_bond_scan, run_scan

    provider = QuarticProvider()
    water = Molecule.from_arrays(["O", "H", "H"], REFERENCE)
    scan = run_scan(plan_bond_scan(water, provider, ["0-1"], [np.linspace(0.8, 1.2, 5)]), provider)
    stretched = ScanAxis(
        label=scan.axes[0].label,
        coordinate="bond-length",
        values=np.linspace(0.6, 1.4, 5),
        bond=(0, 1),
    )
    with pytest.raises(ValueError, match="sampling definition"):
        dataclasses.replace(scan, axes=(stretched,))
    sampling = json.loads(json.dumps(dict(scan.sampling), default=list))
    sampling["axes"] = [stretched.payload()]
    with pytest.raises(ValueError, match="does not match the geometry"):
        dataclasses.replace(scan, axes=(stretched,), sampling=sampling)


def test_legacy_loader_checks_arrays_against_the_declared_grid(tmp_path: Path) -> None:
    from pyscf_vscf.artifacts.scans import load_legacy_grid
    from pyscf_vscf.cache import dump_grid_npz
    from pyscf_vscf.settings import ESSettings
    from pyscf_vscf.workflows import scans as legacy

    water = Molecule.from_arrays(["O", "H", "H"], REFERENCE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = legacy.lbs_frozen_1d_cache_metadata(
            water, ESSettings(method="hf", basis="sto-3g"), "0-1", 0.8, 1.2, 5
        )
    R = np.linspace(1.3, 1.7, 5)
    path = tmp_path / "shifted.npz"
    dump_grid_npz(path, meta=meta, arrays={"R_A": R, "E_Eh": R**2, "MU_Debye": np.zeros((5, 3))})
    with pytest.raises(ValueError, match="does not match requested grid"):
        load_legacy_grid(path)


def test_fractional_charge_and_negative_bond_indices_are_refused() -> None:
    from pyscf_vscf.workflows.scans import normal_mode_direction_from_modes

    with pytest.raises(ValueError, match="charge must be an integer"):
        Molecule.from_arrays(["H", "H"], [[0, 0, 0], [0, 0, 0.74]], charge=0.9)
    with pytest.raises(ValueError, match="zero-based atom indices"):
        parse_bond("0--1")
    water = Molecule.from_arrays(["O", "H", "H"], REFERENCE)
    with pytest.raises(IndexError, match="out of range"):
        normal_mode_direction_from_modes(water, Bond(0, -1), np.eye(9)[:, :3], np.ones(3))


def test_relaxed_scans_reject_non_finite_constraint_diagnostics() -> None:
    from pyscf_vscf.backends.pyscf import NormalRelaxedPointResult
    from pyscf_vscf.workflows.scans import grid_1d_pes_dms_normal_relaxed

    def relaxed(molecule, cfg, u_dir, s, gtol, maxiter):
        return NormalRelaxedPointResult(
            energy_hartree=s * s,
            dipole_debye=np.zeros(3),
            coords_A=np.asarray(molecule.coords),
            requested_displacement_A=s,
            achieved_displacement_A=float("nan"),
            constraint_residual_A=float("nan"),
            converged=True,
            n_iterations=1,
            message="",
        )

    water = Molecule.from_arrays(["O", "H", "H"], REFERENCE)
    direction = np.zeros((3, 3))
    direction[1, 0] = 1.0
    with pytest.raises(RuntimeError, match="finite"):
        grid_1d_pes_dms_normal_relaxed(
            water, object(), direction, -0.1, 0.1, 5, relaxed_point_fn=relaxed
        )


# --- lane A: numerical core ----------------------------------------------------


def _harmonic_modes(n_modes: int, points: int, couplings=None, offset: float = 0.0):
    from pyscf_vscf.vscf import NModePotential

    q = np.linspace(-0.4, 0.4, points)
    potentials = [0.1 * q**2 for _ in range(n_modes)]
    potentials[0] = potentials[0] + offset
    return NModePotential(
        coordinates=(q,) * n_modes,
        masses_amu=(1.0,) * n_modes,
        one_mode_potentials_Eh=tuple(potentials),
        two_mode_couplings_Eh=couplings or {},
    )


def test_iterative_solvers_keep_every_degenerate_state() -> None:
    from pyscf_vscf.dvr import sinc_dvr_1d
    from pyscf_vscf.validation import exact_nmode_dvr
    from pyscf_vscf.vci import VCISettings, build_nmode_vscf_modal_basis, solve_vci

    model = _harmonic_modes(4, 7)
    single = sinc_dvr_1d(model.coordinates[0], 1.0, model.one_mode_potentials_Eh[0]).evals
    exact = np.sort([sum(single[list(c)]) for c in itertools.product(range(7), repeat=4)])[:12]

    hamiltonian, basis = build_nmode_vscf_modal_basis(model, 7)
    vci = solve_vci(hamiltonian, basis, settings=VCISettings(nstates=12, extra_eigenstates=2))
    np.testing.assert_allclose(vci.energies_Eh, exact, rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(exact_nmode_dvr(model, nstates=12).evals, exact, atol=1e-10)


def test_vci_is_invariant_to_an_energy_offset() -> None:
    from pyscf_vscf.vci import VCISettings, build_nmode_vscf_modal_basis, solve_vci

    q = np.linspace(-0.4, 0.4, 21)
    coupling = {(0, 1): 0.05 * q[:, None] ** 2 * q[None, :]}
    energies = []
    for offset in (0.0, -100.0):
        model = _harmonic_modes(2, 21, coupling, offset)
        hamiltonian, basis = build_nmode_vscf_modal_basis(model, 18)
        settings = VCISettings(nstates=5, dense_dimension_threshold=10)
        energies.append(np.asarray(solve_vci(hamiltonian, basis, settings=settings).energies_Eh))
    np.testing.assert_allclose(energies[1] + 100.0, energies[0], rtol=0.0, atol=1e-9)


def test_damping_does_not_fake_vscf_convergence() -> None:
    from pyscf_vscf.vscf import VSCFSettings, solve_vscf_state

    q = np.linspace(-0.45, 0.45, 31)
    from pyscf_vscf.vscf import NModePotential

    model = NModePotential(
        coordinates=(q, q),
        masses_amu=(1.0, 1.4),
        one_mode_potentials_Eh=(0.08 * q**2, 0.11 * q**2),
        two_mode_couplings_Eh={(0, 1): 2.0 * q[:, None] ** 2 * q[None, :] ** 2},
    )
    settings = VSCFSettings(modal_mixing=1e-9, max_iterations=5, raise_on_nonconvergence=False)
    assert not solve_vscf_state(model, (1, 0), settings=settings).converged


def test_dvr_checks_the_g_matrix_and_fixes_eigenvector_signs() -> None:
    from pyscf_vscf.dvr import product_dvr_2d, sinc_dvr_1d

    grid = np.linspace(-0.5, 0.5, 15)
    with pytest.raises(ValueError, match="positive definite"):
        product_dvr_2d(grid, grid, 1.0, 1.0, np.zeros((15, 15)), nmax=3, g12_inv_amu=2.0)
    vectors = sinc_dvr_1d(grid, 1.0, 0.1 * grid**2).evecs
    for column in vectors.T:
        pivot = int(np.flatnonzero(np.abs(column) >= np.max(np.abs(column)) * (1 - 1e-8))[0])
        assert column[pivot] > 0.0


def test_convergence_report_accepts_records_without_signatures() -> None:
    from pyscf_vscf.validation import convergence_report

    record = {
        "assignment": (1, 0),
        "assignment_signature": None,
        "freq_cm": 3600.0,
        "integrated_cross_section_isotropic_omega_m2_per_s": 1e-11,
    }
    report = convergence_report({"coarse": [record], "fine": [record]})
    assert report is not None


def test_harmonic_helpers_refuse_infinite_masses() -> None:
    from pyscf_vscf.harmonic import mass_weight

    with pytest.raises(ValueError, match="finite and positive"):
        mass_weight(np.eye(6), np.array([np.inf, np.inf]))


# --- lane C: electronic structure and execution --------------------------------


@dataclass(frozen=True)
class DriftingFieldProvider:
    """Field-capable energy provider whose configuration snapshot changes with field."""

    def scientific_settings_payload(self) -> dict[str, object]:
        from pyscf_vscf.backends.pyscf_correlated import _FIELD_CAPABILITY

        return {"schema": "test-drifting-field", "field_convention": dict(_FIELD_CAPABILITY)}

    def execution_provenance(self) -> dict[str, object]:
        return {}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        field = np.zeros(3) if request.field_au is None else np.asarray(request.field_au)
        provider_id = provider_scientific_fingerprint(self)
        return ElectronicResult(
            total_energy_Eh=-1.0 - float(field @ np.array([0.1, 0.2, 0.3])),
            dipole_au=None,
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
            scientific_diagnostics={
                "field_au": field.tolist(),
                "field_origin_A": [0.0, 0.0, 0.0],
                "nuclear_field_energy_Eh": 0.0,
                "field_hamiltonian": "H(F)=H(0)-F.mu",
                "nuclear_field_term_included": True,
            },
            provenance={"configuration_snapshot": {"grids_level": 4 if field[0] > 0 else 3}},
        )


def test_finite_field_dipoles_require_one_configuration() -> None:
    from pyscf_vscf.backends.pyscf_correlated import FiniteFieldDipoleProvider

    provider = FiniteFieldDipoleProvider(DriftingFieldProvider())
    request = ElectronicPointRequest(
        nuclear_charges=(1, 1), coordinates_A=[[0, 0, 0], [0, 0, 0.74]]
    )
    with pytest.raises(ValueError, match="grids_level"):
        provider.evaluate(request)


def test_relaxed_scans_compare_configuration_across_points() -> None:
    from pyscf_vscf.backends.pyscf import NormalRelaxedPointResult
    from pyscf_vscf.workflows.scans import grid_1d_pes_dms_normal_relaxed

    def relaxed(molecule, cfg, u_dir, s, gtol, maxiter):
        return NormalRelaxedPointResult(
            energy_hartree=s * s,
            dipole_debye=np.zeros(3),
            coords_A=np.asarray(molecule.coords),
            requested_displacement_A=s,
            achieved_displacement_A=s,
            constraint_residual_A=0.0,
            converged=True,
            n_iterations=1,
            message="",
            configuration_snapshot={"grids_level": 4 if s > 0 else 3},
        )

    water = Molecule.from_arrays(["O", "H", "H"], REFERENCE)
    direction = np.zeros((3, 3))
    direction[1, 0] = 1.0
    with pytest.raises(ValueError, match="grids_level"):
        grid_1d_pes_dms_normal_relaxed(
            water, object(), direction, -0.1, 0.1, 5, relaxed_point_fn=relaxed
        )


def test_spawned_workers_inherit_thread_settings_only_during_the_run(monkeypatch) -> None:
    from pyscf_vscf.runner import _ThreadEnvironment

    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "8")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    environment = _ThreadEnvironment(2)
    environment.apply()
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["OPENBLAS_NUM_THREADS"] == "1"
    environment.restore()
    assert os.environ["OPENBLAS_NUM_THREADS"] == "8"
    assert "OMP_NUM_THREADS" not in os.environ


# --- lane D: interfaces and scripts --------------------------------------------


def test_imaginary_frequencies_are_printed(capsys) -> None:
    from pyscf_vscf._cli_common import _print_frequencies

    _print_frequencies([-1682.7, 0.0, 4401.2])
    out = capsys.readouterr().out
    assert "-1682.7" in out and "4401.2" in out
    assert "1 imaginary mode" in out


def test_water_validation_exits_nonzero_when_acceptance_fails(monkeypatch) -> None:
    import importlib.util
    import sys

    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_archived_grids.py"
    spec = importlib.util.spec_from_file_location("validate_archived_grids", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    failing = {"systems": {"H2O": {"fundamental_convergence_acceptance": {"passed": False}}}}
    monkeypatch.setattr(module, "analyze", lambda *args, **kwargs: failing)
    monkeypatch.setattr(sys, "argv", ["validate_archived_grids.py"])
    assert module.main() == 1


def test_nh3_analysis_refuses_changed_inputs(tmp_path: Path) -> None:
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / "validate_nh3_three_mode.py"
    spec = importlib.util.spec_from_file_location("validate_nh3_three_mode", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (tmp_path / "nh3_optimized.xyz").write_text("changed\n")
    manifest = {"files": {"nh3_optimized.xyz": {"sha256": "0" * 64}}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="differ from manifest.json"):
        module._verify_archive(tmp_path)
