from __future__ import annotations

import numpy as np
import pytest

from pyscf_vscf.constants import HARTREE_TO_CM
from pyscf_vscf.dvr import sinc_dvr_1d
from pyscf_vscf.validation import exact_nmode_dvr
from pyscf_vscf.vscf import (
    NModePotential,
    VSCFSettings,
    dump_nmode_model,
    load_nmode_model,
    nmode_model_from_pair_surfaces,
    nmode_model_fingerprint,
    product_states,
    solve_vscf_state,
    vscf_spectrum,
)


def _two_mode_model(*, coupling: float = 0.0) -> NModePotential:
    q1 = np.linspace(-0.35, 0.35, 31)
    q2 = np.linspace(-0.30, 0.30, 29)
    v1 = 0.08 * q1**2 + 0.004 * q1**4
    v2 = 0.06 * q2**2 + 0.003 * q2**4
    v12 = coupling * q1[:, None] ** 2 * q2[None, :] ** 2
    return NModePotential(
        coordinates=(q1, q2),
        masses_amu=(1.0, 1.4),
        one_mode_potentials_Eh=(v1, v2),
        two_mode_couplings_Eh={(0, 1): v12},
        mode_labels=("stretch-a", "stretch-b"),
        metadata={"purpose": "test"},
    )


def _three_mode_model() -> NModePotential:
    coordinates = (
        np.linspace(-0.30, 0.30, 9),
        np.linspace(-0.28, 0.28, 7),
        np.linspace(-0.25, 0.25, 9),
    )
    potentials = tuple((0.05 + 0.01 * i) * q**2 for i, q in enumerate(coordinates))
    return NModePotential(
        coordinates=coordinates,
        masses_amu=(1.0, 1.2, 1.4),
        one_mode_potentials_Eh=potentials,
        two_mode_couplings_Eh={
            (0, 1): 0.02 * coordinates[0][:, None] * coordinates[1][None, :],
            (0, 2): 0.03 * coordinates[0][:, None] ** 2 * coordinates[2][None, :] ** 2,
            (1, 2): -0.01 * coordinates[1][:, None] * coordinates[2][None, :],
        },
    )


def test_pair_surface_assembly_recovers_three_mode_model_with_offsets() -> None:
    expected = _three_mode_model()
    references = (4, 3, 4)
    offsets = {(0, 1): -56.0, (0, 2): 3.5, (1, 2): 0.25}
    surfaces = {}
    for (i, j), coupling in expected.two_mode_couplings_Eh.items():
        surfaces[(i, j)] = (
            offsets[(i, j)]
            + expected.one_mode_potentials_Eh[i][:, None]
            + expected.one_mode_potentials_Eh[j][None, :]
            + coupling
        )

    assembled = nmode_model_from_pair_surfaces(
        expected.coordinates,
        expected.masses_amu,
        surfaces,
        reference_indices=references,
        consistency_tolerance_Eh=1e-12,
    )

    for actual, wanted in zip(assembled.one_mode_potentials_Eh, expected.one_mode_potentials_Eh):
        np.testing.assert_allclose(actual, wanted - wanted[actual.size // 2], atol=2e-14)
    for pair, wanted in expected.two_mode_couplings_Eh.items():
        i, j = pair
        ri, rj = references[i], references[j]
        normalized = wanted - wanted[:, rj][:, None] - wanted[ri, :][None, :] + wanted[ri, rj]
        np.testing.assert_allclose(assembled.two_mode_couplings_Eh[pair], normalized, atol=2e-14)


def test_pair_surface_assembly_rejects_inconsistent_shared_cuts() -> None:
    model = _three_mode_model()
    surfaces = {}
    for (i, j), coupling in model.two_mode_couplings_Eh.items():
        surfaces[(i, j)] = (
            model.one_mode_potentials_Eh[i][:, None]
            + model.one_mode_potentials_Eh[j][None, :]
            + coupling
        )
    surfaces[(0, 2)] = surfaces[(0, 2)] + 1e-4 * model.coordinates[0][:, None]

    with pytest.raises(ValueError, match="mode 0.*disagree"):
        nmode_model_from_pair_surfaces(
            model.coordinates,
            model.masses_amu,
            surfaces,
            consistency_tolerance_Eh=1e-8,
        )


def test_pair_surface_assembly_requires_covered_modes_and_valid_references() -> None:
    model = _three_mode_model()
    surface = (
        model.one_mode_potentials_Eh[0][:, None]
        + model.one_mode_potentials_Eh[1][None, :]
        + model.two_mode_couplings_Eh[(0, 1)]
    )
    with pytest.raises(ValueError, match="Mode 2 does not appear"):
        nmode_model_from_pair_surfaces(
            model.coordinates,
            model.masses_amu,
            {(0, 1): surface},
        )
    with pytest.raises(ValueError, match="outside the grid"):
        nmode_model_from_pair_surfaces(
            model.coordinates,
            model.masses_amu,
            {(0, 1): surface},
            reference_indices=(4, 99, 4),
        )


def test_exact_nmode_dvr_matches_separable_three_mode_sums() -> None:
    model = _three_mode_model()
    separable = NModePotential(
        coordinates=model.coordinates,
        masses_amu=model.masses_amu,
        one_mode_potentials_Eh=model.one_mode_potentials_Eh,
    )
    exact = exact_nmode_dvr(separable, nstates=8)
    one_mode_levels = [np.linalg.eigvalsh(h) for h in separable.one_mode_hamiltonians()]
    sums = sorted(
        a + b + c
        for a in one_mode_levels[0]
        for b in one_mode_levels[1]
        for c in one_mode_levels[2]
    )[:8]

    assert exact.shape == (9, 7, 9)
    np.testing.assert_allclose(exact.evals, sums, atol=2e-11)
    assert not exact.evals.flags.writeable
    assert not exact.evecs.flags.writeable


def test_exact_nmode_dvr_matches_dense_coupled_hamiltonian() -> None:
    coordinates = tuple(np.linspace(-0.2, 0.2, 5) for _ in range(3))
    model = NModePotential(
        coordinates=coordinates,
        masses_amu=(1.0, 1.2, 1.4),
        one_mode_potentials_Eh=tuple(
            coefficient * coordinate**2
            for coefficient, coordinate in zip((0.04, 0.05, 0.06), coordinates)
        ),
        two_mode_couplings_Eh={
            (0, 1): 0.015 * coordinates[0][:, None] * coordinates[1][None, :],
            (0, 2): 0.020 * coordinates[0][:, None] ** 2 * coordinates[2][None, :] ** 2,
            (1, 2): -0.010 * coordinates[1][:, None] * coordinates[2][None, :],
        },
    )
    exact = exact_nmode_dvr(model, nstates=6)

    shape = tuple(coordinate.size for coordinate in coordinates)
    identities = tuple(np.eye(size) for size in shape)
    dense = np.diag(
        (
            model.one_mode_potentials_Eh[0][:, None, None]
            + model.one_mode_potentials_Eh[1][None, :, None]
            + model.one_mode_potentials_Eh[2][None, None, :]
            + model.two_mode_couplings_Eh[(0, 1)][:, :, None]
            + model.two_mode_couplings_Eh[(0, 2)][:, None, :]
            + model.two_mode_couplings_Eh[(1, 2)][None, :, :]
        ).reshape(-1)
    )
    kinetic = model.one_mode_hamiltonians()
    for mode in range(3):
        one_mode_kinetic = kinetic[mode] - np.diag(model.one_mode_potentials_Eh[mode])
        factors = list(identities)
        factors[mode] = one_mode_kinetic
        term = np.kron(np.kron(factors[0], factors[1]), factors[2])
        dense += term

    expected = np.linalg.eigvalsh(dense)[:6]
    np.testing.assert_allclose(exact.evals, expected, atol=2e-11)


def test_coupled_three_mode_vscf_is_variational_against_exact_dvr() -> None:
    model = _three_mode_model()
    vscf = solve_vscf_state(model)
    exact = exact_nmode_dvr(model, nstates=2)

    assert vscf.converged
    assert vscf.energy_Eh >= exact.evals[0] - 1e-11


def test_uncoupled_vscf_matches_sum_of_one_mode_dvr_levels() -> None:
    model = _two_mode_model()
    result = solve_vscf_state(model, (1, 2))
    dvr1 = sinc_dvr_1d(model.coordinates[0], model.masses_amu[0], model.one_mode_potentials_Eh[0])
    dvr2 = sinc_dvr_1d(model.coordinates[1], model.masses_amu[1], model.one_mode_potentials_Eh[1])

    assert result.converged
    assert result.energy_Eh == pytest.approx(dvr1.evals[1] + dvr2.evals[2], abs=1e-12)
    assert result.history[-1].max_density_change < 1e-12


def test_coupled_vscf_converges_and_adds_pair_expectation_once() -> None:
    model = _two_mode_model(coupling=0.4)
    result = solve_vscf_state(
        model,
        settings=VSCFSettings(modal_mixing=0.7, energy_tolerance_Eh=1e-11),
    )
    hamiltonians = model.one_mode_hamiltonians()
    p0, p1 = (np.square(modal) for modal in result.modals)
    expected = (
        result.modals[0] @ hamiltonians[0] @ result.modals[0]
        + result.modals[1] @ hamiltonians[1] @ result.modals[1]
        + p0 @ model.two_mode_couplings_Eh[(0, 1)] @ p1
    )

    assert result.converged
    assert result.energy_Eh == pytest.approx(expected, abs=1e-12)
    assert result.iterations > 1

    from pyscf_vscf.dvr import product_dvr_2d

    exact = product_dvr_2d(
        model.coordinates[0],
        model.coordinates[1],
        model.masses_amu[0],
        model.masses_amu[1],
        model.one_mode_potentials_Eh[0][:, None]
        + model.one_mode_potentials_Eh[1][None, :]
        + model.two_mode_couplings_Eh[(0, 1)],
        k_eigs=2,
    )
    assert result.energy_Eh >= exact.evals[0] - 1e-12


def test_vscf_spectrum_reports_assigned_fundamentals_and_combination() -> None:
    model = _two_mode_model(coupling=0.05)
    spectrum = vscf_spectrum(model, max_quanta_per_mode=1, max_total_quanta=2)

    assert {transition.quanta for transition in spectrum.transitions} == {
        (0, 1),
        (1, 0),
        (1, 1),
    }
    assert all(transition.frequency_cm > 0.0 for transition in spectrum.transitions)
    np.testing.assert_allclose(
        [transition.frequency_cm for transition in spectrum.transitions],
        [transition.energy_Eh * HARTREE_TO_CM for transition in spectrum.transitions],
    )
    with pytest.raises(ValueError, match="duplicate"):
        vscf_spectrum(model, states=[(1, 0), (1, 0)])


def test_product_state_enumeration_is_polyad_bounded() -> None:
    assert product_states(2, max_quanta_per_mode=2, max_total_quanta=2) == (
        (0, 1),
        (1, 0),
        (0, 2),
        (1, 1),
        (2, 0),
    )


def test_nmode_archive_roundtrip_and_fingerprint(tmp_path) -> None:
    model = _two_mode_model(coupling=0.1)
    path = tmp_path / "model.npz"

    dump_nmode_model(path, model)
    restored = load_nmode_model(path)

    assert nmode_model_fingerprint(restored) == nmode_model_fingerprint(model)
    assert restored.mode_labels == model.mode_labels
    assert restored.metadata == model.metadata
    assert restored.coordinate_units == "angstrom"
    assert not restored.coordinates[0].flags.writeable
    assert not restored.one_mode_potentials_Eh[0].flags.writeable
    np.testing.assert_allclose(
        restored.two_mode_couplings_Eh[(0, 1)],
        model.two_mode_couplings_Eh[(0, 1)],
    )


def test_nmode_model_rejects_invalid_pair_shapes_and_single_mode() -> None:
    q = np.linspace(-1.0, 1.0, 5)
    with pytest.raises(ValueError, match="at least two"):
        NModePotential(
            coordinates=(q,),
            masses_amu=(1.0,),
            one_mode_potentials_Eh=(q**2,),
        )
    with pytest.raises(ValueError, match="shape"):
        NModePotential(
            coordinates=(q, q),
            masses_amu=(1.0, 1.0),
            one_mode_potentials_Eh=(q**2, q**2),
            two_mode_couplings_Eh={(0, 1): np.zeros((5, 4))},
        )


def test_nmode_model_rejects_unsupported_coordinate_units() -> None:
    q = np.linspace(-1.0, 1.0, 5)
    with pytest.raises(ValueError, match="Angstrom"):
        NModePotential(
            coordinates=(q, q),
            masses_amu=(1.0, 1.0),
            one_mode_potentials_Eh=(q**2, q**2),
            coordinate_units="bohr",
        )

    model = NModePotential(
        coordinates=(q, q),
        masses_amu=(1.0, 1.0),
        one_mode_potentials_Eh=(q**2, q**2),
        coordinate_units="AA",
    )
    assert model.coordinate_units == "angstrom"
