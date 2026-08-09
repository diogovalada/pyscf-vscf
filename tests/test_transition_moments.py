from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from pyscf_vscf.constants import ATOMIC_DIPOLE_TO_DEBYE, MASS_AMU
from pyscf_vscf.coordinates import (
    LinearDisplacementCoordinateMap,
    TriatomicValenceCoordinateMap,
    coordinate_map_fingerprint,
)
from pyscf_vscf.kinetic import (
    TriatomicJ0Hamiltonian,
    TriatomicJ0KineticOperator,
    TriatomicJacobiTransform,
    potential_on_jacobi_grid,
    solve_triatomic_direct_dvr,
)
from pyscf_vscf.nmode import FitDiagnostics, NModeSurfaceModel, TensorProductSurface
from pyscf_vscf.spectra import einstein_a_from_debye
from pyscf_vscf.transition_moments import (
    ConfigurationDipoleOperator,
    build_vci_dipole_operator,
    dipole_on_jacobi_grid,
    dump_vci_dipole_projection,
    dump_vci_transition_moments,
    load_vci_dipole_projection,
    load_vci_transition_moments,
    nmode_dipole_on_grid,
    vci_transition_moments,
)
from pyscf_vscf.validation import exact_nmode_dvr
from pyscf_vscf.vci import (
    VCISettings,
    build_nmode_vscf_modal_basis,
    build_triatomic_vscf_modal_basis,
    solve_vci,
)
from pyscf_vscf.vscf import NModePotential


def _fit_surface(
    axes: tuple[np.ndarray, ...],
    values: np.ndarray,
    name: str,
):
    del name
    interpolator = RegularGridInterpolator(
        axes,
        values,
        method="cubic_legacy",
        bounds_error=True,
    )
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack(mesh, axis=-1).reshape((-1, len(axes)))
    residual = np.asarray(interpolator(points)).reshape(values.shape) - values
    components = 1 if values.ndim == len(axes) else values.shape[-1]
    maximum = tuple(
        float(value) for value in np.max(np.abs(residual.reshape(-1, components)), axis=0)
    )
    return TensorProductSurface(
        axes=axes,
        node_values=values,
        method="cubic",
        diagnostics=FitDiagnostics(
            method="cubic",
            n_training_points=int(np.prod([axis.size for axis in axes])),
            training_max_abs_error=maximum,
        ),
    )


def _rectilinear_map() -> LinearDisplacementCoordinateMap:
    reference = np.zeros((2, 3))
    displacements = np.zeros((2, 2, 3))
    displacements[0, 0, 0] = 1.0
    displacements[1, 1, 1] = 1.0
    return LinearDisplacementCoordinateMap(
        reference_geometry_A=reference,
        coordinate_ids=("q1", "q2"),
        units=("angstrom", "angstrom"),
        reference_values=np.zeros(2),
        displacements_A_per_unit=displacements,
    )


def _source_lineage(label: str) -> dict[str, object]:
    return {
        "schema": "pyscf-vscf-electronic-source-lineage",
        "schema_version": 1,
        "provider_scientific_fingerprint": label,
        "point_causal_fingerprints": [f"{label}-point"],
    }


def _two_mode_models(
    *,
    rotation: np.ndarray | None = None,
    coupling: float = 0.0,
) -> tuple[NModePotential, NModeSurfaceModel]:
    first = np.linspace(-0.42, 0.42, 9)
    second = np.linspace(-0.36, 0.36, 9)
    one_first = 0.13 * first**2 + 0.02 * first**4
    one_second = 0.09 * second**2 + 0.015 * second**4
    pair = coupling * first[:, None] * second[None, :]
    coordinate_map = _rectilinear_map()
    map_id = coordinate_map_fingerprint(coordinate_map)
    potential = NModePotential(
        coordinates=(first, second),
        masses_amu=(1.0, 1.35),
        one_mode_potentials_Eh=(one_first, one_second),
        two_mode_couplings_Eh={(0, 1): pair},
        mode_labels=("q1", "q2"),
        coordinate_map_fingerprint=map_id,
    )
    transform = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    reference_dipole = transform @ np.array([0.12, -0.08, 0.03])
    dipole_first = np.zeros((first.size, 3))
    dipole_first[:, 0] = 0.75 * first
    dipole_second = np.zeros((second.size, 3))
    dipole_second[:, 1] = -0.55 * second
    dipole_pair = np.zeros((first.size, second.size, 3))
    dipole_pair[:, :, 2] = 0.65 * first[:, None] * second[None, :]
    dipole_first = dipole_first @ transform.T
    dipole_second = dipole_second @ transform.T
    dipole_pair = dipole_pair @ transform.T
    zeros_first = np.zeros_like(dipole_first)
    zeros_second = np.zeros_like(dipole_second)
    zeros_pair = np.zeros_like(dipole_pair)
    surface = NModeSurfaceModel(
        coordinate_ids=("q1", "q2"),
        coordinate_units=("angstrom", "angstrom"),
        coordinate_map_payload=coordinate_map.fingerprint_payload(),
        coordinate_map_fingerprint=map_id,
        reference_values=np.zeros(2),
        reference_energy_Eh=-1.0,
        reference_dipole_body_au=reference_dipole,
        energy_increments={
            (0,): _fit_surface((first,), one_first, "energy-q1"),
            (1,): _fit_surface((second,), one_second, "energy-q2"),
            (0, 1): _fit_surface((first, second), pair, "energy-q1-q2"),
        },
        dipole_increments={
            (0,): _fit_surface((first,), dipole_first + zeros_first, "dipole-q1"),
            (1,): _fit_surface((second,), dipole_second + zeros_second, "dipole-q2"),
            (0, 1): _fit_surface(
                (first, second),
                dipole_pair + zeros_pair,
                "dipole-q1-q2",
            ),
        },
        source_lineage=_source_lineage("analytic-vector-dms"),
    )
    return potential, surface


def _solve_models(
    potential: NModePotential,
    surface: NModeSurfaceModel,
    modal_count: int,
):
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(potential, modal_count)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=7, extra_eigenstates=2),
    )
    expansion = nmode_dipole_on_grid(surface, hamiltonian)
    dipole_operator = build_vci_dipole_operator(expansion, modal_basis, result)
    transitions = vci_transition_moments(dipole_operator, result)
    return hamiltonian, modal_basis, result, expansion, dipole_operator, transitions


def _state_for_configuration(result, configuration: tuple[int, ...]) -> int:
    matches = [
        item.state_index for item in result.assignments if item.configuration == configuration
    ]
    assert len(matches) == 1
    return matches[0]


def _transition_for_upper(transitions, upper: int):
    return next(item for item in transitions if item.upper_state == upper)


def _analytic_two_mode_dipole(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    q1, q2 = np.meshgrid(first, second, indexing="ij")
    values = np.empty((*q1.shape, 3))
    values[..., 0] = 0.12 + 0.75 * q1
    values[..., 1] = -0.08 - 0.55 * q2
    values[..., 2] = 0.03 + 0.65 * q1 * q2
    return values


def _triatomic_fixture():
    mass_sets = {
        "H2O": (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["H"]),
        "HDO-outer2": (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["D"]),
        "HDO-outer1": (MASS_AMU["D"], MASS_AMU["O"], MASS_AMU["H"]),
        "D2O": (MASS_AMU["D"], MASS_AMU["O"], MASS_AMU["D"]),
    }
    reference = np.array([0.98, 0.99, 1.82])
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [reference[0], 0.0, 0.0],
            [
                reference[1] * np.cos(reference[2]),
                reference[1] * np.sin(reference[2]),
                0.0,
            ],
        ]
    )
    coordinate_map = TriatomicValenceCoordinateMap(
        geometry,
        center_atom=0,
        outer_atom_1=1,
        outer_atom_2=2,
        coordinate_ids=("a", "b", "gamma"),
    )
    reference = coordinate_map.reference_values
    transforms = {
        label: TriatomicJacobiTransform(masses, atom_indices=(1, 0, 2))
        for label, masses in mass_sets.items()
    }
    kinetics = {
        label: TriatomicJ0KineticOperator(
            radial_r_A=np.linspace(0.76, 1.18, 4),
            radial_R_A=np.linspace(0.60, 1.28, 4),
            angular_order=4,
            masses_amu=masses,
            atom_indices=(1, 0, 2),
        )
        for label, masses in mass_sets.items()
    }
    valence_samples = []
    for label in mass_sets:
        meshes = np.meshgrid(*kinetics[label].coordinate_grids, indexing="ij")
        valence_samples.append(
            transforms[label].jacobi_to_valence(np.stack(meshes, axis=-1)).reshape((-1, 3))
        )
    pooled_valence = np.concatenate(valence_samples, axis=0)
    axes = tuple(
        np.unique(
            np.concatenate(
                (
                    np.linspace(
                        float(np.min(pooled_valence[:, mode])) - 0.03,
                        float(np.max(pooled_valence[:, mode])) + 0.03,
                        7,
                    ),
                    [reference[mode]],
                )
            )
        )
        for mode in range(3)
    )
    energy = {}
    dipole = {}
    force_constants = (0.16, 0.14, 0.035)
    slopes = np.array(
        [
            [0.55, 0.0, 0.0],
            [0.0, -0.47, 0.0],
            [0.0, 0.0, 0.32],
        ]
    )
    for mode in range(3):
        displacement = axes[mode] - reference[mode]
        energy[(mode,)] = _fit_surface(
            (axes[mode],),
            force_constants[mode] * displacement**2,
            f"triatomic-energy-{mode}",
        )
        dipole[(mode,)] = _fit_surface(
            (axes[mode],),
            displacement[:, None] * slopes[mode],
            f"triatomic-dipole-{mode}",
        )
    for first in range(3):
        for second in range(first + 1, 3):
            shape = (axes[first].size, axes[second].size)
            energy[(first, second)] = _fit_surface(
                (axes[first], axes[second]),
                np.zeros(shape),
                f"triatomic-energy-{first}-{second}",
            )
            dipole[(first, second)] = _fit_surface(
                (axes[first], axes[second]),
                np.zeros((*shape, 3)),
                f"triatomic-dipole-{first}-{second}",
            )
    triple = (
        (axes[0] - reference[0])[:, None, None]
        * (axes[1] - reference[1])[None, :, None]
        * (axes[2] - reference[2])[None, None, :]
    )
    energy[(0, 1, 2)] = _fit_surface(
        axes,
        np.zeros_like(triple),
        "triatomic-energy-triple",
    )
    triple_dipole = np.zeros((*triple.shape, 3))
    triple_dipole[..., 0] = 0.40 * triple
    triple_dipole[..., 2] = -0.25 * triple
    dipole[(0, 1, 2)] = _fit_surface(
        axes,
        triple_dipole,
        "triatomic-dipole-triple",
    )
    model = NModeSurfaceModel(
        coordinate_ids=("a", "b", "gamma"),
        coordinate_units=("angstrom", "angstrom", "radian"),
        coordinate_map_payload=coordinate_map.fingerprint_payload(),
        coordinate_map_fingerprint=coordinate_map_fingerprint(coordinate_map),
        reference_values=reference,
        reference_energy_Eh=-76.0,
        reference_dipole_body_au=np.array([0.10, -0.04, 0.02]),
        energy_increments=energy,
        dipole_increments=dipole,
        source_lineage=_source_lineage("analytic-triatomic-vector-dms"),
    )
    cases = {}
    for label in mass_sets:
        potential_projection = potential_on_jacobi_grid(
            model,
            coordinate_map,
            transforms[label],
            kinetics[label],
        )
        cases[label] = (
            transforms[label],
            TriatomicJ0Hamiltonian.from_projection(
                kinetics[label],
                potential_projection,
            ),
        )
    return model, coordinate_map, cases


def _analytic_triatomic_dipole(
    valence: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    displacement = valence - reference
    triple = displacement[..., 0] * displacement[..., 1] * displacement[..., 2]
    values = np.empty((*valence.shape[:-1], 3))
    values[..., 0] = 0.10 + 0.55 * displacement[..., 0] + 0.40 * triple
    values[..., 1] = -0.04 - 0.47 * displacement[..., 1]
    values[..., 2] = 0.02 + 0.32 * displacement[..., 2] - 0.25 * triple
    return values


def test_harmonic_selection_rules_and_pair_dms_combination_band() -> None:
    potential, surface = _two_mode_models()
    _, _, result, expansion, dipole_operator, transitions = _solve_models(
        potential,
        surface,
        9,
    )
    first = _transition_for_upper(transitions, _state_for_configuration(result, (1, 0)))
    second = _transition_for_upper(transitions, _state_for_configuration(result, (0, 1)))
    overtone = _transition_for_upper(transitions, _state_for_configuration(result, (2, 0)))
    combination = _transition_for_upper(transitions, _state_for_configuration(result, (1, 1)))

    assert abs(first.transition_dipole_body_au[0]) > 1e-3
    assert np.linalg.norm(first.transition_dipole_body_au[1:]) < 2e-12
    assert abs(second.transition_dipole_body_au[1]) > 1e-3
    assert abs(combination.transition_dipole_body_au[2]) > 1e-4
    assert np.linalg.norm(overtone.transition_dipole_body_au) < 2e-12
    assert set(expansion.increment_grids_au) == {(0,), (1,), (0, 1)}
    assert set(dipole_operator.increment_matrices_au) == {(0,), (1,), (0, 1)}
    assert not expansion.dipole_grid_au.flags.writeable


def test_vci_transition_moments_match_direct_product_dvr() -> None:
    potential, surface = _two_mode_models(coupling=0.018)
    _, _, result, expansion, _, transitions = _solve_models(potential, surface, 9)
    direct = exact_nmode_dvr(potential, nstates=12)
    analytic_grid = _analytic_two_mode_dipole(*potential.coordinates)
    np.testing.assert_allclose(expansion.dipole_grid_au, analytic_grid, atol=3e-15)
    dipole_grid = analytic_grid.reshape((-1, 3))

    for record in transitions[:6]:
        direct_vector = np.einsum(
            "i,ic,i->c",
            direct.evecs[:, 0],
            dipole_grid,
            direct.evecs[:, record.upper_state],
        )
        actual = record.transition_dipole_body_au
        sign = 1.0 if float(actual @ direct_vector) >= 0.0 else -1.0
        np.testing.assert_allclose(actual, sign * direct_vector, atol=3e-11)
        np.testing.assert_allclose(
            record.frequency_Eh,
            direct.evals[record.upper_state] - direct.evals[0],
            atol=3e-11,
        )


def test_jacobi_vci_transition_moments_match_direct_3d_dvr_with_3mr_dms() -> None:
    model, coordinate_map, cases = _triatomic_fixture()
    expansions = {}
    for label, (transform, hamiltonian) in cases.items():
        expansion = dipole_on_jacobi_grid(
            model,
            coordinate_map,
            transform,
            hamiltonian,
        )
        meshes = np.meshgrid(*hamiltonian.kinetic.coordinate_grids, indexing="ij")
        valence = transform.jacobi_to_valence(np.stack(meshes, axis=-1))
        analytic_grid = _analytic_triatomic_dipole(
            valence,
            coordinate_map.reference_values,
        )
        np.testing.assert_allclose(expansion.dipole_grid_au, analytic_grid, atol=8e-15)
        triple = (
            (valence[..., 0] - coordinate_map.reference_values[0])
            * (valence[..., 1] - coordinate_map.reference_values[1])
            * (valence[..., 2] - coordinate_map.reference_values[2])
        )
        expected_triple = np.zeros((*triple.shape, 3))
        expected_triple[..., 0] = 0.40 * triple
        expected_triple[..., 2] = -0.25 * triple
        np.testing.assert_allclose(
            expansion.increment_grids_au[(0, 1, 2)],
            expected_triple,
            atol=8e-15,
        )
        expansions[label] = (expansion, analytic_grid)

    transform, hamiltonian = cases["H2O"]
    expansion, analytic_grid = expansions["H2O"]
    modal_basis = build_triatomic_vscf_modal_basis(hamiltonian, 4)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=5, extra_eigenstates=2),
    )
    dipole_operator = build_vci_dipole_operator(expansion, modal_basis, result)
    transitions = vci_transition_moments(dipole_operator, result)
    direct = solve_triatomic_direct_dvr(
        hamiltonian,
        nstates=7,
        dense_dimension_threshold=hamiltonian.dimension,
    )
    dipole_grid = analytic_grid.reshape((-1, 3))

    assert (0, 1, 2) in expansion.increment_grids_au
    assert (0, 1, 2) in dipole_operator.increment_matrices_au
    for record in transitions:
        direct_vector = np.einsum(
            "i,ic,i->c",
            direct.eigenvectors[:, 0],
            dipole_grid,
            direct.eigenvectors[:, record.upper_state],
        )
        actual = record.transition_dipole_body_au
        sign = 1.0 if float(actual @ direct_vector) >= 0.0 else -1.0
        np.testing.assert_allclose(actual, sign * direct_vector, atol=4e-11)
        np.testing.assert_allclose(
            record.frequency_Eh,
            direct.energies_Eh[record.upper_state] - direct.energies_Eh[0],
            atol=3e-11,
        )


def test_signed_increment_cancellation_is_preserved() -> None:
    potential, surface = _two_mode_models(coupling=0.012)
    _, _, result, _, operator_template, _ = _solve_models(potential, surface, 9)
    lower = result.coefficients[:, 0]
    upper = result.coefficients[:, 1]
    cancellation_matrix = np.outer(lower, upper) + np.outer(upper, lower)
    positive = np.zeros((*cancellation_matrix.shape, 3))
    positive[:, :, 0] = cancellation_matrix
    negative = -positive
    zero = np.zeros_like(positive)
    dipole_operator = ConfigurationDipoleOperator.from_analytic_matrices(
        configurations=result.configurations,
        reference_matrices_au=zero,
        increment_matrices_au={(0,): positive, (1,): negative},
        provenance_label="signed-cancellation-fixture",
        modal_basis_fingerprint=result.modal_basis_fingerprint,
        hamiltonian_fingerprint=result.hamiltonian_fingerprint,
    )

    record = vci_transition_moments(
        dipole_operator,
        result,
        upper_states=(1,),
    )[0]

    assert np.linalg.norm(record.transition_dipole_body_au) < 2e-15
    assert record.isotropic_integrated_cross_section_omega_m2_per_s < 1e-45
    assert operator_template.fingerprint() != dipole_operator.fingerprint()


def test_transitions_touching_degenerate_states_require_manual_review() -> None:
    grid = np.linspace(-0.36, 0.36, 7)
    one_mode = 0.11 * grid**2 + 0.02 * grid**4
    coordinate_map = _rectilinear_map()
    map_id = coordinate_map_fingerprint(coordinate_map)
    potential = NModePotential(
        coordinates=(grid, grid),
        masses_amu=(1.0, 1.0),
        one_mode_potentials_Eh=(one_mode, one_mode),
        two_mode_couplings_Eh={(0, 1): np.zeros((grid.size, grid.size))},
        mode_labels=("q1", "q2"),
        coordinate_map_fingerprint=map_id,
    )
    dipole_first = np.zeros((grid.size, 3))
    dipole_first[:, 0] = grid
    dipole_second = np.zeros((grid.size, 3))
    dipole_second[:, 1] = grid
    surface = NModeSurfaceModel(
        coordinate_ids=("q1", "q2"),
        coordinate_units=("angstrom", "angstrom"),
        coordinate_map_payload=coordinate_map.fingerprint_payload(),
        coordinate_map_fingerprint=map_id,
        reference_values=np.zeros(2),
        reference_energy_Eh=-1.0,
        reference_dipole_body_au=np.zeros(3),
        energy_increments={
            (0,): _fit_surface((grid,), one_mode, "deg-energy-q1"),
            (1,): _fit_surface((grid,), one_mode, "deg-energy-q2"),
            (0, 1): _fit_surface(
                (grid, grid),
                np.zeros((grid.size, grid.size)),
                "deg-energy-pair",
            ),
        },
        dipole_increments={
            (0,): _fit_surface((grid,), dipole_first, "deg-dipole-q1"),
            (1,): _fit_surface((grid,), dipole_second, "deg-dipole-q2"),
            (0, 1): _fit_surface(
                (grid, grid),
                np.zeros((grid.size, grid.size, 3)),
                "deg-dipole-pair",
            ),
        },
        source_lineage=_source_lineage("degenerate-dms"),
    )
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(potential, 7)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=3, extra_eigenstates=2),
    )
    dipole_operator = build_vci_dipole_operator(
        nmode_dipole_on_grid(surface, hamiltonian),
        modal_basis,
        result,
    )
    records = vci_transition_moments(dipole_operator, result)

    assert result.assignments[1].degenerate_block == (1, 2)
    assert records[0].manual_review
    assert records[1].manual_review


def test_vector_rotation_covariance_and_absolute_intensity_identities() -> None:
    angle = 0.63
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    potential, surface = _two_mode_models(coupling=0.014)
    rotated_potential, rotated_surface = _two_mode_models(
        rotation=rotation,
        coupling=0.014,
    )
    *_, original = _solve_models(potential, surface, 9)
    *_, rotated = _solve_models(rotated_potential, rotated_surface, 9)

    for first, second in zip(original[:6], rotated[:6]):
        np.testing.assert_allclose(
            second.transition_dipole_body_au,
            rotation @ first.transition_dipole_body_au,
            atol=3e-12,
        )
        np.testing.assert_allclose(
            np.linalg.norm(second.transition_dipole_body_debye),
            np.linalg.norm(first.transition_dipole_body_au) * ATOMIC_DIPOLE_TO_DEBYE,
            atol=2e-14,
        )
        np.testing.assert_allclose(
            first.isotropic_integrated_cross_section_omega_m2_per_s,
            np.sum(first.polarized_integrated_cross_sections_omega_m2_per_s) / 3.0,
            rtol=2e-14,
        )
        np.testing.assert_allclose(
            first.einstein_a_s,
            einstein_a_from_debye(
                np.linalg.norm(first.transition_dipole_body_debye),
                first.frequency_cm,
            ),
            rtol=2e-14,
        )


def test_transition_records_bind_exact_vci_result_and_derive_immutable_outputs() -> None:
    potential, surface = _two_mode_models(coupling=0.016)
    _, _, result, _, dipole_operator, transitions = _solve_models(potential, surface, 9)
    record = transitions[0]
    modified_coefficients = np.array(result.coefficients, copy=True)
    modified_coefficients[:, [0, 1]] = modified_coefficients[:, [1, 0]]
    modified_diagnostic_coefficients = np.array(result.diagnostic_coefficients, copy=True)
    modified_diagnostic_coefficients[:, [0, 1]] = modified_diagnostic_coefficients[:, [1, 0]]
    modified_result = replace(
        result,
        coefficients=modified_coefficients,
        diagnostic_coefficients=modified_diagnostic_coefficients,
    )

    assert record.vci_result_fingerprint == result.fingerprint()
    assert modified_result.fingerprint() != result.fingerprint()
    modified_record = vci_transition_moments(
        dipole_operator,
        modified_result,
        upper_states=(1,),
    )[0]
    assert modified_record.vci_result_fingerprint == modified_result.fingerprint()
    assert modified_record.vci_result_fingerprint != record.vci_result_fingerprint

    with pytest.raises(ValueError):
        record.transition_dipole_body_au.setflags(write=True)
    with pytest.raises(ValueError):
        record.polarized_integrated_cross_sections_omega_m2_per_s.setflags(write=True)
    with pytest.raises(TypeError, match="must come from"):
        replace(
            record,
            transition_dipole_body_au=2.0 * record.transition_dipole_body_au,
        )


def test_dipole_projection_and_transition_artifacts_round_trip_and_reject_tampering(
    tmp_path: Path,
) -> None:
    potential, surface = _two_mode_models(coupling=0.016)
    _, _, _, expansion, dipole_operator, transitions = _solve_models(potential, surface, 7)
    projection_path = tmp_path / "projection.npz"
    transition_path = tmp_path / "transitions.json"

    dump_vci_dipole_projection(expansion, dipole_operator, projection_path)
    restored_expansion, restored_operator = load_vci_dipole_projection(projection_path)
    assert restored_expansion.fingerprint() == expansion.fingerprint()
    assert restored_operator.fingerprint() == dipole_operator.fingerprint()

    dump_vci_transition_moments(transitions, transition_path)
    restored_transitions = load_vci_transition_moments(transition_path)
    assert tuple(value.fingerprint() for value in restored_transitions) == tuple(
        value.fingerprint() for value in transitions
    )

    payload = json.loads(transition_path.read_text(encoding="utf-8"))
    payload["moments"][0]["transition_dipole_body_au"][0] += 1e-9
    transition_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprints do not match"):
        load_vci_transition_moments(transition_path)


def test_rectilinear_dms_requires_exact_coordinate_map_identity() -> None:
    potential, surface = _two_mode_models()
    compatible_hamiltonian, _ = build_nmode_vscf_modal_basis(potential, 4)
    expansion = nmode_dipole_on_grid(surface, compatible_hamiltonian)
    with pytest.raises(TypeError, match="projection function"):
        replace(
            expansion,
            increment_grids_au={
                subset: 2.0 * values for subset, values in expansion.increment_grids_au.items()
            },
        )

    foreign = replace(potential, coordinate_map_fingerprint="foreign-map")
    hamiltonian, _ = build_nmode_vscf_modal_basis(foreign, 4)

    with pytest.raises(ValueError, match="coordinate-map fingerprints"):
        nmode_dipole_on_grid(surface, hamiltonian)

    unspecified = replace(potential, coordinate_map_fingerprint=None)
    hamiltonian, _ = build_nmode_vscf_modal_basis(unspecified, 4)
    with pytest.raises(ValueError, match="lacks a coordinate-map fingerprint"):
        nmode_dipole_on_grid(surface, hamiltonian)


def test_transition_moments_converge_with_modal_basis_growth() -> None:
    potential, surface = _two_mode_models(coupling=0.02)
    exact = exact_nmode_dvr(potential, nstates=4)
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(potential, 9)
    full_result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=4, extra_eigenstates=2),
    )
    full_expansion = nmode_dipole_on_grid(surface, hamiltonian)
    full_operator = build_vci_dipole_operator(full_expansion, modal_basis, full_result)
    full_record = vci_transition_moments(
        full_operator,
        full_result,
        upper_states=(1,),
    )[0]
    direct_vector = np.einsum(
        "i,ic,i->c",
        exact.evecs[:, 0],
        full_expansion.dipole_grid_au.reshape((-1, 3)),
        exact.evecs[:, 1],
    )
    direct_strength = float(np.linalg.norm(direct_vector))
    errors = []
    for modal_count in (3, 5, 9):
        _, _, result, _, _, transitions = _solve_models(
            potential,
            surface,
            modal_count,
        )
        record = _transition_for_upper(transitions, 1)
        errors.append(abs(np.linalg.norm(record.transition_dipole_body_au) - direct_strength))
        assert result.residual_norms_Eh[0] < 1e-9

    assert errors[0] >= errors[1] - 2e-12
    assert errors[1] >= errors[2] - 2e-12
    assert errors[2] < 3e-11
    np.testing.assert_allclose(
        np.linalg.norm(full_record.transition_dipole_body_au),
        np.linalg.norm(direct_vector),
        atol=3e-11,
    )
