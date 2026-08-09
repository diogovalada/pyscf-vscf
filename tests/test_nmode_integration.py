from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pyscf_vscf.electronic import ElectronicPointRequest, ElectronicResult
from pyscf_vscf.coordinates import LinearDisplacementCoordinateMap
from pyscf_vscf.nmode import (
    assemble_nmode_samples,
    fit_nmode_surface,
    nmode_potential_from_surface,
    plan_nmode_points,
)
from pyscf_vscf.transition_moments import (
    build_vci_dipole_operator,
    nmode_dipole_on_grid,
    vci_transition_moments,
)
from pyscf_vscf.vci import VCISettings, build_nmode_vscf_modal_basis, solve_vci


@dataclass(frozen=True)
class _AnalyticHOFProvider:
    def scientific_settings_payload(self) -> dict[str, object]:
        return {
            "schema": "pyscf-vscf-test-analytic-hof",
            "schema_version": 1,
            "backend_family": "analytic",
            "method": "polynomial",
        }

    def execution_provenance(self) -> dict[str, object]:
        return {"implementation": "closed-form test fixture"}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        del request
        raise AssertionError("the portable fixture does not run electronic structure")


def _hof_coordinate_map() -> LinearDisplacementCoordinateMap:
    reference = np.array(
        [
            [-0.96, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.42, 1.32, 0.0],
        ]
    )
    displacements = np.zeros((3, 3, 3))
    displacements[0, 0, 0] = -1.0
    displacements[1, 2, :2] = np.array([0.3032, 0.9529])
    displacements[2, 0, 1] = -0.55
    displacements[2, 2, 0] = 0.55
    return LinearDisplacementCoordinateMap(
        reference_geometry_A=reference,
        coordinate_ids=("H_stretch", "OF_stretch", "bend"),
        units=("angstrom", "angstrom", "angstrom"),
        reference_values=np.zeros(3),
        displacements_A_per_unit=displacements,
    )


def _hof_energy_Eh(q: np.ndarray) -> float:
    q0, q1, q2 = q
    return float(
        -175.0 + 0.21 * q0**2 + 0.17 * q1**2 + 0.11 * q2**2 + 0.035 * q0 * q1 - 0.024 * q1 * q2
    )


def _hof_dipole_au(q: np.ndarray) -> np.ndarray:
    q0, q1, q2 = q
    return np.array(
        [
            0.14 + 0.72 * q0 + 0.18 * q1 * q2,
            -0.09 - 0.48 * q1 + 0.11 * q0 * q2,
            0.03 + 0.39 * q2 - 0.16 * q0 * q1,
        ]
    )


def test_portable_analytic_hof_nmode_vci_transition_pipeline() -> None:
    coordinate_map = _hof_coordinate_map()
    axes = tuple(np.linspace(-0.20, 0.20, 5) for _ in range(3))
    plan = plan_nmode_points(
        coordinate_map,
        axes,
        _AnalyticHOFProvider(),
        nuclear_charges=(1, 8, 9),
        max_rank=2,
    )
    assert len(plan.requests) == 61

    values_by_point = {
        lineage.point_causal_fingerprint: lineage.coordinate_values
        for lineage in plan.sampling_lineage.values()
    }
    results = {
        point: ElectronicResult(
            total_energy_Eh=_hof_energy_Eh(q),
            dipole_au=_hof_dipole_au(q),
            dipole_unit="atomic_unit",
            dipole_frame="input_cartesian",
            converged=True,
            point_causal_fingerprint=point,
            provider_scientific_fingerprint=plan.provider_scientific_fingerprint,
        )
        for point, q in values_by_point.items()
    }
    samples = assemble_nmode_samples(plan, results)
    surface = fit_nmode_surface(samples, method="cubic")

    probe = np.array([0.13, -0.07, 0.16])
    assert surface.energy_Eh(probe) == pytest.approx(_hof_energy_Eh(probe), abs=2e-12)
    np.testing.assert_allclose(surface.dipole_body_au(probe), _hof_dipole_au(probe), atol=2e-12)

    potential = nmode_potential_from_surface(
        surface,
        axes,
        masses_amu=(1.0, 10.0, 4.0),
    )
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(potential, 3)
    assert modal_basis.converged
    vci_result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=6, extra_eigenstates=2),
    )
    assert np.max(vci_result.residual_norms_Eh) < 1e-11
    assert np.all(np.diff(vci_result.energies_Eh) >= 0.0)

    grid_dipole = nmode_dipole_on_grid(surface, hamiltonian)
    dipole_operator = build_vci_dipole_operator(grid_dipole, modal_basis, vci_result)
    transitions = vci_transition_moments(dipole_operator, vci_result)

    assert vci_result.energies_Eh[0] == pytest.approx(0.006834129934, abs=5e-10)
    assert transitions[0].frequency_Eh == pytest.approx(0.002353320805, abs=5e-10)
    assert transitions[0].transition_dipole_body_au == pytest.approx(
        np.array([0.002536380, 0.026807249, -0.005410293]),
        abs=5e-8,
    )
