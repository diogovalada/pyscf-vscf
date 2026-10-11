from __future__ import annotations

import numpy as np
import pytest

from pyscf_vscf.dvr import sinc_dvr_1d, trans_mu_1d
from pyscf_vscf.validation import exact_nmode_dvr
from pyscf_vscf.vscf import NModeDipole, NModePotential, vscf_spectrum

Q1 = np.linspace(-0.45, 0.45, 41)
Q2 = np.linspace(-0.40, 0.40, 37)
MASSES = (1.0, 1.4)


def _model(coupling: float) -> NModePotential:
    return NModePotential(
        coordinates=(Q1, Q2),
        masses_amu=MASSES,
        one_mode_potentials_Eh=(0.08 * Q1**2 + 0.004 * Q1**4, 0.11 * Q2**2),
        two_mode_couplings_Eh={(0, 1): coupling * Q1[:, None] ** 2 * Q2[None, :]},
    )


def _dipole(cross: float = 0.0) -> NModeDipole:
    one = (
        np.stack([0.3 * Q1, 0.05 * Q1**2, np.zeros_like(Q1)], axis=-1),
        np.stack([np.zeros_like(Q2), 0.2 * Q2, 0.02 * Q2**2], axis=-1),
    )
    two = {
        (0, 1): cross
        * np.stack(
            [
                np.zeros((Q1.size, Q2.size)),
                np.zeros((Q1.size, Q2.size)),
                Q1[:, None] * Q2[None, :],
            ],
            axis=-1,
        )
    }
    return NModeDipole(
        coordinates=(Q1, Q2),
        reference_au=np.array([0.1, -0.2, 0.7]),
        one_mode_au=one,
        two_mode_au=two,
    )


def _exact_moments(model: NModePotential, dipole: NModeDipole, nstates: int) -> tuple:
    exact = exact_nmode_dvr(model, nstates=nstates)
    grid = np.broadcast_to(dipole.reference_au, (Q1.size, Q2.size, 3)).copy()
    grid += dipole.one_mode_au[0][:, None, :]
    grid += dipole.one_mode_au[1][None, :, :]
    for (i, j), values in dipole.two_mode_au.items():
        grid += values
    ground = exact.evecs[:, 0].reshape(Q1.size, Q2.size)
    moments = [
        np.einsum("ab,abx,ab->x", ground, grid, exact.evecs[:, n].reshape(Q1.size, Q2.size))
        for n in range(1, nstates)
    ]
    return exact.evals, moments


def test_uncoupled_fundamentals_match_one_dimensional_dvr() -> None:
    model = _model(0.0)
    dipole = _dipole()
    spectrum = vscf_spectrum(model, dipole=dipole)

    one = sinc_dvr_1d(Q1, MASSES[0], model.one_mode_potentials_Eh[0])
    expected_x = trans_mu_1d(one, lambda q: np.interp(q, Q1, dipole.one_mode_au[0][:, 0]), 1)
    fundamental = next(t for t in spectrum.transitions if t.quanta == (1, 0))
    moment = fundamental.transition_moment

    assert abs(moment.dipole_au[0]) == pytest.approx(abs(expected_x), rel=1e-10)
    assert moment.state_overlap == pytest.approx(0.0, abs=1e-12)
    assert not moment.approximate
    assert moment.integrated_cross_section_isotropic_km_per_mol > 0.0
    assert moment.einstein_a_s > 0.0


def test_weakly_coupled_moments_agree_with_exact_product_dvr() -> None:
    model = _model(0.01)
    dipole = _dipole(cross=0.05)
    spectrum = vscf_spectrum(model, dipole=dipole, max_quanta_per_mode=2, max_total_quanta=2)
    evals, exact_moments = _exact_moments(model, dipole, nstates=7)

    for transition in spectrum.transitions:
        index = int(np.argmin(np.abs(evals[1:] - evals[0] - transition.energy_Eh)))
        exact_norm = float(np.linalg.norm(exact_moments[index]))
        vscf_norm = float(np.linalg.norm(transition.transition_moment.dipole_au))
        if exact_norm > 1e-3:
            assert vscf_norm == pytest.approx(exact_norm, rel=0.05), transition.quanta


def test_moments_are_absent_without_a_dipole_and_grids_must_match() -> None:
    model = _model(0.01)
    assert vscf_spectrum(model).transitions[0].transition_moment is None

    wrong = NModeDipole(
        coordinates=(Q1[:-1], Q2),
        reference_au=np.zeros(3),
        one_mode_au=(np.zeros((Q1.size - 1, 3)), np.zeros((Q2.size, 3))),
    )
    with pytest.raises(ValueError, match="potential's grids"):
        vscf_spectrum(model, dipole=wrong)


def test_modal_signs_are_canonical() -> None:
    spectrum = vscf_spectrum(_model(0.01), max_quanta_per_mode=2, max_total_quanta=2)
    for state in (spectrum.ground, *spectrum.excited_states):
        for modal in state.modals:
            pivot = int(np.argmax(np.abs(modal) >= np.max(np.abs(modal)) * (1.0 - 1e-8)))
            assert modal[pivot] > 0.0
