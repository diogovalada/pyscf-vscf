from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.constants import AMU, ANG_TO_BOHR, MASS_AMU
from pyscf_vscf.kinetic import (
    TriatomicJ0Hamiltonian,
    TriatomicJ0KineticOperator,
    solve_triatomic_direct_dvr,
)
from pyscf_vscf.validation import exact_nmode_dvr
from pyscf_vscf.vci import (
    VCIResult,
    VCISettings,
    build_nmode_vscf_modal_basis,
    build_triatomic_vscf_modal_basis,
    dump_ground_modal_basis,
    dump_vci_result,
    enumerate_vci_configurations,
    load_ground_modal_basis,
    load_vci_result,
    solve_vci,
    validate_vci_result_against_hamiltonian,
)
from pyscf_vscf.vci import (
    _ProductBasisProjector,
    _assign_vci_states,
    _phase_canonical_configurations,
)
from pyscf_vscf.vscf import NModePotential, VSCFSettings, nmode_model_fingerprint


def _coupled_two_mode_model(*, symmetric: bool = False) -> NModePotential:
    first = np.linspace(-0.32, 0.32, 9)
    second = first.copy() if symmetric else np.linspace(-0.28, 0.30, 8)
    first_potential = 0.12 * first**2 + 0.03 * first**4
    second_potential = (
        0.12 * second**2 + 0.03 * second**4 if symmetric else 0.09 * second**2 + 0.02 * second**4
    )
    coupling = np.zeros((first.size, second.size))
    if not symmetric:
        coupling = 0.035 * first[:, None] * second[None, :]
    return NModePotential(
        coordinates=(first, second),
        masses_amu=(1.0, 1.0 if symmetric else 1.3),
        one_mode_potentials_Eh=(first_potential, second_potential),
        two_mode_couplings_Eh={(0, 1): coupling},
        mode_labels=("q1", "q2"),
    )


def _coupled_three_mode_model() -> NModePotential:
    grids = tuple(np.linspace(-0.24, 0.24, 5) for _ in range(3))
    one_mode = tuple(
        depth * np.square(1.0 - np.exp(-width * grid))
        for grid, depth, width in zip(grids, (0.14, 0.11, 0.09), (1.4, 1.2, 1.1))
    )
    return NModePotential(
        coordinates=grids,
        masses_amu=(1.0, 1.2, 1.5),
        one_mode_potentials_Eh=one_mode,
        two_mode_couplings_Eh={
            (0, 1): 0.025 * grids[0][:, None] * grids[1][None, :],
            (0, 2): 0.018 * grids[0][:, None] ** 2 * grids[2][None, :] ** 2,
            (1, 2): -0.014 * grids[1][:, None] * grids[2][None, :],
        },
        mode_labels=("a", "b", "c"),
    )


def _symmetric_three_mode_model() -> NModePotential:
    grid = np.linspace(-0.30, 0.30, 7)
    potential = 0.12 * grid**2 + 0.025 * grid**4
    return NModePotential(
        coordinates=(grid, grid, grid),
        masses_amu=(1.0, 1.0, 1.0),
        one_mode_potentials_Eh=(potential, potential, potential),
        mode_labels=("q1", "q2", "q3"),
    )


def _triatomic_hamiltonian() -> TriatomicJ0Hamiltonian:
    kinetic = TriatomicJ0KineticOperator(
        radial_r_A=np.linspace(0.78, 1.20, 4),
        radial_R_A=np.linspace(0.62, 1.28, 4),
        angular_order=4,
        masses_amu=(MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["H"]),
        atom_indices=(1, 0, 2),
    )
    radial_r, radial_R, angular_x = np.meshgrid(
        *kinetic.coordinate_grids,
        indexing="ij",
    )
    potential = (
        0.16 * np.square(radial_r - 0.97)
        + 0.10 * np.square(radial_R - 0.92)
        + 0.04 * np.square(angular_x + 0.22)
        + 0.01 * (radial_r - 0.97) * (radial_R - 0.92) * angular_x
    )
    return TriatomicJ0Hamiltonian(kinetic, potential)


def test_full_modal_vci_matches_existing_exact_2d_dvr() -> None:
    model = _coupled_two_mode_model()
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(
        model,
        (9, 8),
        settings=VSCFSettings(modal_mixing=0.7),
    )
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=6, extra_eigenstates=2),
    )
    exact = exact_nmode_dvr(model, nstates=8)

    np.testing.assert_allclose(result.energies_Eh, exact.evals[:6], atol=3e-11)
    assert np.max(result.residual_norms_Eh) < 2e-11
    np.testing.assert_allclose(
        result.state_cutoff_margin_Eh,
        exact.evals[6] - exact.evals[5],
        atol=3e-11,
    )


def test_modal_and_polyad_expansions_converge_monotonically_to_2d_oracle() -> None:
    model = _coupled_two_mode_model()
    exact = exact_nmode_dvr(model, nstates=6).evals[:4]
    modal_errors = []
    for counts in ((3, 3), (5, 5), (9, 8)):
        hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, counts)
        result = solve_vci(
            hamiltonian,
            modal_basis,
            settings=VCISettings(nstates=4, extra_eigenstates=2),
        )
        modal_errors.append(float(np.max(result.energies_Eh - exact)))
    assert modal_errors[0] >= modal_errors[1] - 2e-12
    assert modal_errors[1] >= modal_errors[2] - 2e-12
    assert abs(modal_errors[2]) < 3e-11

    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, (9, 8))
    pruning_errors = []
    for maximum_total in (2, 4, None):
        result = solve_vci(
            hamiltonian,
            modal_basis,
            settings=VCISettings(
                nstates=4,
                extra_eigenstates=2,
                max_total_quanta=maximum_total,
            ),
        )
        pruning_errors.append(float(np.max(result.energies_Eh - exact)))
    assert pruning_errors[0] >= pruning_errors[1] - 2e-12
    assert pruning_errors[1] >= pruning_errors[2] - 2e-12
    assert abs(pruning_errors[2]) < 3e-11


def test_coupled_harmonic_vci_matches_analytic_normal_mode_frequencies() -> None:
    grids = (np.linspace(-1.05, 1.05, 25), np.linspace(-1.00, 1.00, 25))
    masses = (1.0, 1.35)
    force_constants = np.array([[0.19, 0.035], [0.035, 0.16]])
    q1_bohr = grids[0] * ANG_TO_BOHR
    q2_bohr = grids[1] * ANG_TO_BOHR
    model = NModePotential(
        coordinates=grids,
        masses_amu=masses,
        one_mode_potentials_Eh=(
            0.5 * force_constants[0, 0] * q1_bohr**2,
            0.5 * force_constants[1, 1] * q2_bohr**2,
        ),
        two_mode_couplings_Eh={
            (0, 1): force_constants[0, 1] * q1_bohr[:, None] * q2_bohr[None, :]
        },
        mode_labels=("q1", "q2"),
    )
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, 25)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(
            nstates=3,
            extra_eigenstates=2,
            dense_dimension_threshold=20,
            eigensolver_tolerance=1e-11,
        ),
    )
    inverse_sqrt_mass = np.diag(1.0 / np.sqrt(np.asarray(masses) * AMU))
    expected = np.sqrt(np.linalg.eigvalsh(inverse_sqrt_mass @ force_constants @ inverse_sqrt_mass))

    np.testing.assert_allclose(
        np.sort(result.energies_Eh[1:3] - result.energies_Eh[0]),
        expected,
        atol=2e-7,
    )


def test_configuration_pruning_is_deterministic_and_complete() -> None:
    model = _coupled_two_mode_model()
    _, modal_basis = build_nmode_vscf_modal_basis(model, (4, 3))
    settings = VCISettings(
        nstates=2,
        max_quanta_per_mode=(2, 1),
        max_total_quanta=2,
        max_modal_energy_Eh=10.0,
    )
    configurations = enumerate_vci_configurations(modal_basis, settings)

    assert configurations[0] == (0, 0)
    assert set(configurations) == {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1)}
    assert all(sum(configuration) <= 2 for configuration in configurations)
    assert all(configuration[0] <= 2 and configuration[1] <= 1 for configuration in configurations)


def test_sparse_three_mode_vci_converges_to_direct_product_oracle() -> None:
    model = _coupled_three_mode_model()
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(
        model,
        5,
        settings=VSCFSettings(modal_mixing=0.65),
    )
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(
            nstates=4,
            extra_eigenstates=2,
            dense_dimension_threshold=10,
            eigensolver_tolerance=1e-11,
        ),
    )
    exact = exact_nmode_dvr(model, nstates=6)

    np.testing.assert_allclose(result.energies_Eh, exact.evals[:4], atol=5e-10)
    assert np.max(result.residual_norms_Eh) < 2e-10
    assert all(assignment.leading_configurations for assignment in result.assignments)
    assert all(assignment.participation_ratio >= 1.0 for assignment in result.assignments)


def test_jacobi_vci_uses_complete_kinetic_operator_and_matches_direct_3d_dvr() -> None:
    hamiltonian = _triatomic_hamiltonian()
    modal_basis = build_triatomic_vscf_modal_basis(
        hamiltonian,
        4,
        settings=VSCFSettings(modal_mixing=0.6),
    )
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=4, extra_eigenstates=2),
    )
    direct = solve_triatomic_direct_dvr(
        hamiltonian,
        nstates=6,
        dense_dimension_threshold=hamiltonian.dimension,
    )

    np.testing.assert_allclose(result.energies_Eh, direct.energies_Eh[:4], atol=3e-11)
    assert modal_basis.kinetic_operator_fingerprint == hamiltonian.kinetic.fingerprint()
    assert result.hamiltonian_fingerprint == hamiltonian.fingerprint()


def test_truncated_jacobi_modal_spaces_converge_to_direct_3d_dvr() -> None:
    hamiltonian = _triatomic_hamiltonian()
    direct = solve_triatomic_direct_dvr(
        hamiltonian,
        nstates=5,
        dense_dimension_threshold=hamiltonian.dimension,
    )
    errors = []
    for modal_count in (2, 3, 4):
        modal_basis = build_triatomic_vscf_modal_basis(
            hamiltonian,
            modal_count,
            settings=VSCFSettings(modal_mixing=0.6),
        )
        result = solve_vci(
            hamiltonian,
            modal_basis,
            settings=VCISettings(nstates=3, extra_eigenstates=2),
        )
        errors.append(float(np.max(result.energies_Eh - direct.energies_Eh[:3])))
    assert errors[0] >= errors[1] - 2e-12
    assert errors[1] >= errors[2] - 2e-12
    assert abs(errors[2]) < 3e-11


def test_exact_degenerate_blocks_are_reported_for_manual_review() -> None:
    model = _coupled_two_mode_model(symmetric=True)
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, 6)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(
            nstates=3,
            extra_eigenstates=1,
            max_total_quanta=2,
            degeneracy_tolerance_Eh=1e-10,
        ),
    )

    assert result.assignments[0].configuration == (0, 0)
    assert not result.assignments[0].manual_review
    assert result.assignments[1].manual_review
    assert result.assignments[2].manual_review
    assert result.assignments[1].degenerate_block == (1, 2)
    assert result.assignments[1].subspace_overlap is not None
    block = result.degenerate_blocks[0]
    assert block.state_indices == (1, 2)
    np.testing.assert_allclose(
        block.invariant_projector @ block.invariant_projector,
        block.invariant_projector,
        atol=2e-12,
    )
    assert len(block.reference_configurations) == 2
    assert len(block.principal_cosines) == 2


def test_degenerate_block_crossing_requested_cutoff_is_not_assigned() -> None:
    model = _coupled_two_mode_model(symmetric=True)
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, 6)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(
            nstates=2,
            extra_eigenstates=2,
            max_total_quanta=2,
            degeneracy_tolerance_Eh=1e-10,
        ),
    )

    assert result.assignments[1].configuration is None
    assert result.assignments[1].manual_review
    assert result.assignments[1].degenerate_block == (1, 2)
    assert result.degenerate_blocks[0].state_indices == (1, 2)
    assert result.degenerate_blocks[0].crosses_requested_cutoff
    assert result.state_cutoff_margin_Eh < 1e-10
    assert result.diagnostic_energies_Eh.size == 4
    assert result.diagnostic_residual_norms_Eh.size == 4


def test_adaptive_diagnostic_window_closes_triply_degenerate_block() -> None:
    model = _symmetric_three_mode_model()
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, 4)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(
            nstates=2,
            extra_eigenstates=1,
            max_total_quanta=2,
            degeneracy_tolerance_Eh=1e-10,
        ),
    )

    block = result.degenerate_blocks[0]
    assert block.state_indices == (1, 2, 3)
    assert block.crosses_requested_cutoff
    assert result.diagnostic_energies_Eh.size == 5
    assert result.diagnostic_energies_Eh[4] - result.diagnostic_energies_Eh[3] > 1e-10
    assert np.trace(block.invariant_projector) == pytest.approx(3.0, abs=5e-12)


def test_vci_rejects_nonconverged_foreign_and_reordered_modal_bases() -> None:
    model = _coupled_two_mode_model()
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, (5, 5))
    settings = VCISettings(nstates=2, extra_eigenstates=2)
    with pytest.raises(ValueError, match="converged"):
        solve_vci(hamiltonian, replace(modal_basis, converged=False), settings=settings)

    changed_model = replace(
        model,
        one_mode_potentials_Eh=(
            model.one_mode_potentials_Eh[0] + 1e-4 * model.coordinates[0] ** 2,
            model.one_mode_potentials_Eh[1],
        ),
    )
    changed_hamiltonian, _ = build_nmode_vscf_modal_basis(changed_model, (5, 5))
    with pytest.raises(ValueError, match="not generated from"):
        solve_vci(changed_hamiltonian, modal_basis, settings=settings)

    reordered_model = replace(model, mode_labels=("q2", "q1"))
    reordered_hamiltonian, _ = build_nmode_vscf_modal_basis(reordered_model, (5, 5))
    with pytest.raises(ValueError, match="coordinate IDs"):
        solve_vci(reordered_hamiltonian, modal_basis, settings=settings)


def test_vci_validation_rejects_valid_eigenpairs_that_omit_the_lowest_root() -> None:
    model = _coupled_two_mode_model()
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, (5, 5))
    source = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=3, extra_eigenstates=2, max_total_quanta=3),
    )
    settings = replace(source.vci_settings, nstates=2)
    diagnostic_energies = source.diagnostic_energies_Eh[1:5]
    diagnostic_coefficients = source.diagnostic_coefficients[:, 1:5]
    diagnostic_residuals = source.diagnostic_residual_norms_Eh[1:5]
    assignments, blocks = _assign_vci_states(
        source.configurations,
        diagnostic_energies,
        diagnostic_coefficients,
        requested_count=2,
        degeneracy_tolerance=settings.degeneracy_tolerance_Eh,
        leading_count=settings.leading_configuration_count,
    )
    forged = VCIResult(
        configurations=source.configurations,
        energies_Eh=diagnostic_energies[:2],
        coefficients=diagnostic_coefficients[:, :2],
        residual_norms_Eh=diagnostic_residuals[:2],
        diagnostic_energies_Eh=diagnostic_energies,
        diagnostic_coefficients=diagnostic_coefficients,
        diagnostic_residual_norms_Eh=diagnostic_residuals,
        assignments=assignments,
        degenerate_blocks=blocks,
        state_cutoff_margin_Eh=float(diagnostic_energies[2] - diagnostic_energies[1]),
        modal_basis_fingerprint=source.modal_basis_fingerprint,
        hamiltonian_fingerprint=source.hamiltonian_fingerprint,
        modal_counts=source.modal_counts,
        vscf_settings=source.vscf_settings,
        vci_settings=settings,
    )

    with pytest.raises(ValueError, match="not the requested lowest"):
        validate_vci_result_against_hamiltonian(forged, modal_basis, hamiltonian)


def test_vci_validation_rejects_self_consistent_unconverged_eigenvectors() -> None:
    model = _coupled_two_mode_model()
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, (5, 5))
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=3, extra_eigenstates=2, max_total_quanta=3),
    )
    projector = _ProductBasisProjector(modal_basis.modals, result.configurations)

    def matvec(vector: np.ndarray) -> np.ndarray:
        return projector.from_grid(hamiltonian.apply(projector.to_grid(vector)))

    gap = float(result.diagnostic_energies_Eh[1] - result.diagnostic_energies_Eh[0])
    energy_shift = 10.0 * result.vci_settings.eigensolver_tolerance
    angle = float(np.arcsin(np.sqrt(energy_shift / gap)))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    coefficients = np.array(result.diagnostic_coefficients, copy=True)
    coefficients[:, :2] = coefficients[:, :2] @ rotation
    energies = np.array(
        [float(coefficients[:, state] @ matvec(coefficients[:, state])) for state in range(5)]
    )
    residuals = np.array(
        [
            float(
                np.linalg.norm(
                    matvec(coefficients[:, state]) - energies[state] * coefficients[:, state]
                )
            )
            for state in range(5)
        ]
    )
    assignments, blocks = _assign_vci_states(
        result.configurations,
        energies,
        coefficients,
        requested_count=3,
        degeneracy_tolerance=result.vci_settings.degeneracy_tolerance_Eh,
        leading_count=result.vci_settings.leading_configuration_count,
    )
    forged = replace(
        result,
        energies_Eh=energies[:3],
        coefficients=coefficients[:, :3],
        residual_norms_Eh=residuals[:3],
        diagnostic_energies_Eh=energies,
        diagnostic_coefficients=coefficients,
        diagnostic_residual_norms_Eh=residuals,
        assignments=assignments,
        degenerate_blocks=blocks,
        state_cutoff_margin_Eh=float(energies[3] - energies[2]),
    )

    with pytest.raises(ValueError, match="exceeds the retained eigensolver tolerance"):
        validate_vci_result_against_hamiltonian(forged, modal_basis, hamiltonian)


def test_modal_metadata_is_recursively_immutable_and_fingerprinted() -> None:
    model = _coupled_two_mode_model()
    _, modal_basis = build_nmode_vscf_modal_basis(model, (4, 4))
    sources = ["one", "two"]
    copied = replace(modal_basis, metadata={"nested": {"sources": sources}})
    fingerprint = copied.fingerprint()
    sources.append("late-mutation")

    assert tuple(copied.metadata["nested"]["sources"]) == ("one", "two")
    assert copied.fingerprint() == fingerprint


def test_nmode_metadata_is_recursively_immutable_and_hamiltonian_bound() -> None:
    sources = ["one", "two"]
    model = replace(
        _coupled_two_mode_model(),
        metadata={"nested": {"sources": sources}},
    )
    model_fingerprint = nmode_model_fingerprint(model)
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, (4, 4))
    hamiltonian_fingerprint = hamiltonian.fingerprint()
    sources.append("late-mutation")

    assert tuple(model.metadata["nested"]["sources"]) == ("one", "two")
    assert nmode_model_fingerprint(model) == model_fingerprint
    assert hamiltonian.fingerprint() == hamiltonian_fingerprint
    assert modal_basis.source_hamiltonian_fingerprint == hamiltonian_fingerprint


def test_nmode_numerical_state_cannot_mutate_after_hamiltonian_construction() -> None:
    model = _coupled_two_mode_model()
    hamiltonian, _ = build_nmode_vscf_modal_basis(model, (4, 4))
    fingerprint = hamiltonian.fingerprint()
    arrays = (
        *model.coordinates,
        *model.one_mode_potentials_Eh,
        *model.two_mode_couplings_Eh.values(),
    )

    for values in arrays:
        with pytest.raises(ValueError, match="WRITEABLE"):
            values.setflags(write=True)

    assert hamiltonian.fingerprint() == fingerprint


def test_vci_phase_ties_use_lexicographic_configuration_order() -> None:
    configurations = ((0, 0), (1, 0), (0, 1))
    scale = 1.0 / np.sqrt(2.0)
    vectors = np.array([[0.0], [-scale], [scale]])

    canonical = _phase_canonical_configurations(vectors, configurations)

    assert canonical[2, 0] > 0.0
    assert canonical[1, 0] < 0.0


def test_modal_basis_and_vci_result_round_trip_and_repeat_deterministically(
    tmp_path: Path,
) -> None:
    model = _coupled_two_mode_model()
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(model, (5, 5))
    settings = VCISettings(nstates=4, extra_eigenstates=2)
    first = solve_vci(hamiltonian, modal_basis, settings=settings)
    second = solve_vci(hamiltonian, modal_basis, settings=settings)
    basis_path = tmp_path / "modal-basis.npz"
    result_path = tmp_path / "vci-result.npz"

    dump_ground_modal_basis(modal_basis, basis_path)
    dump_vci_result(first, result_path)
    restored_basis = load_ground_modal_basis(basis_path)
    restored_result = load_vci_result(result_path)

    assert restored_basis.fingerprint() == modal_basis.fingerprint()
    assert restored_result.fingerprint() == first.fingerprint()
    np.testing.assert_array_equal(first.energies_Eh, second.energies_Eh)
    np.testing.assert_array_equal(first.coefficients, second.coefficients)
    with pytest.raises(ValueError):
        restored_result.coefficients.setflags(write=True)
