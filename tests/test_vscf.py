from __future__ import annotations

import numpy as np
import pytest

from pyscf_vscf.constants import HARTREE_TO_CM
from pyscf_vscf.dvr import sinc_dvr_1d
from pyscf_vscf.vscf import (
    NModePotential,
    VSCFSettings,
    dump_nmode_model,
    load_nmode_model,
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
