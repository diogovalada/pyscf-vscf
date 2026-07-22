from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.constants import AMU, ANG_TO_BOHR, HARTREE_TO_CM
from pyscf_vscf.dvr import product_dvr_2d, sinc_dvr_1d, trans_mu_1d, trans_mu_2d
from pyscf_vscf.spectra import integrated_cross_section_omega


def morse_potential_shifted(
    R_ang: np.ndarray,
    *,
    De_Eh: float = 0.18,
    alpha_bohr_inv: float = 1.5,
    Re_ang: float = 1.0,
) -> np.ndarray:
    x = np.asarray(R_ang, dtype=float) * ANG_TO_BOHR
    xe = float(Re_ang) * ANG_TO_BOHR
    y = 1.0 - np.exp(-float(alpha_bohr_inv) * (x - xe))
    return float(De_Eh) * y * y


def morse_analytic_levels_shifted(
    *,
    De_Eh: float = 0.18,
    alpha_bohr_inv: float = 1.5,
    mu_amu: float = 0.94,
    vmax: int = 5,
) -> np.ndarray:
    mu = float(mu_amu) * AMU
    De = float(De_Eh)
    alpha = float(alpha_bohr_inv)
    lam = np.sqrt(2.0 * mu * De) / alpha
    levels: list[float] = []
    for v in range(int(vmax) + 1):
        n = v + 0.5
        if n >= lam:
            break
        levels.append(float((alpha * alpha / (2.0 * mu)) * (2.0 * lam * n - n * n)))
    return np.array(levels)


def test_1d_morse_bound_states_and_transition_strengths() -> None:
    R = np.linspace(0.6, 3.2, 81)
    V = morse_potential_shifted(R)
    dvr = sinc_dvr_1d(R, 0.94, V)

    assert dvr.R.shape == R.shape
    assert dvr.evals.shape == (R.size,)
    assert dvr.evecs.shape == (R.size, R.size)
    np.testing.assert_allclose(
        dvr.evecs.T @ dvr.evecs,
        np.eye(R.size),
        atol=2e-12,
    )

    ref = morse_analytic_levels_shifted(vmax=5)
    err_cm = np.abs(dvr.evals[: ref.size] - ref) * HARTREE_TO_CM
    assert float(np.max(err_cm)) < 1e-3

    const_moments = [abs(trans_mu_1d(dvr, lambda rr: np.ones_like(rr), v)) for v in range(1, 5)]
    assert max(const_moments) < 1e-12

    def linear_mu(rr):
        return np.asarray(rr, dtype=float) - 1.0

    moments = np.array([abs(trans_mu_1d(dvr, linear_mu, v)) for v in range(1, 5)])
    frequencies = (dvr.evals[1:5] - dvr.evals[0]) * HARTREE_TO_CM
    strengths = np.array(
        [
            integrated_cross_section_omega(moment, frequency)
            for moment, frequency in zip(moments, frequencies)
        ]
    )
    assert np.all(strengths > 0.0)
    assert strengths[0] > strengths[1] > strengths[2] > strengths[3]


def test_product_dvr_2d_separable_spectrum_and_transition_factors() -> None:
    R1 = np.linspace(0.7, 2.4, 11)
    R2 = np.linspace(0.75, 2.45, 11)
    V1 = morse_potential_shifted(
        R1,
        De_Eh=0.18,
        alpha_bohr_inv=1.5,
        Re_ang=1.0,
    )
    V2 = morse_potential_shifted(
        R2,
        De_Eh=0.14,
        alpha_bohr_inv=1.25,
        Re_ang=1.05,
    )
    V = V1[:, None] + V2[None, :]

    d1 = sinc_dvr_1d(R1, 0.94, V1)
    d2 = sinc_dvr_1d(R2, 1.2, V2)
    dvr2 = product_dvr_2d(R1, R2, 0.94, 1.2, V, nmax=5)

    assert dvr2.evals.shape == (6,)
    assert dvr2.evecs.shape == (R1.size * R2.size, 6)
    np.testing.assert_allclose(
        dvr2.evecs.T @ dvr2.evecs,
        np.eye(6),
        atol=2e-12,
    )

    pairs = sorted(
        (
            (i, j, float(e1 + e2))
            for i, e1 in enumerate(d1.evals[:6])
            for j, e2 in enumerate(d2.evals[:6])
        ),
        key=lambda item: item[2],
    )[:6]
    expected = np.array([energy for _, _, energy in pairs])
    err_cm = np.abs(dvr2.evals - expected) * HARTREE_TO_CM
    assert float(np.max(err_cm)) < 1e-6

    const_mu = np.ones_like(V)
    max_const_mu = max(abs(trans_mu_2d(dvr2, const_mu, n)) for n in range(1, 6))
    assert max_const_mu < 1e-12

    mu1_grid = (R1[:, None] - 1.0) + np.zeros_like(V)
    mu2_grid = np.zeros_like(V) + (R2[None, :] - 1.05)

    def mu1_model(rr):
        return np.asarray(rr, dtype=float) - 1.0

    def mu2_model(rr):
        return np.asarray(rr, dtype=float) - 1.05

    for n, (i, j, _) in enumerate(pairs[1:], start=1):
        expected_mu1 = abs(trans_mu_1d(d1, mu1_model, i)) if j == 0 else 0.0
        expected_mu2 = abs(trans_mu_1d(d2, mu2_model, j)) if i == 0 else 0.0
        got_mu1 = abs(trans_mu_2d(dvr2, mu1_grid, n))
        got_mu2 = abs(trans_mu_2d(dvr2, mu2_grid, n))
        assert abs(got_mu1 - expected_mu1) < 2e-10
        assert abs(got_mu2 - expected_mu2) < 2e-10


def test_product_dvr_2d_accepts_legacy_k_eigs_and_cross_keo() -> None:
    R1 = np.linspace(0.8, 1.4, 9)
    R2 = np.linspace(0.82, 1.42, 9)
    V = 0.04 * (R1[:, None] - 1.0) ** 2 + 0.05 * (R2[None, :] - 1.05) ** 2

    by_nmax = product_dvr_2d(R1, R2, 0.94, 1.2, V, nmax=4)
    by_k = product_dvr_2d(R1, R2, 0.94, 1.2, V, k_eigs=5)
    crossed = product_dvr_2d(R1, R2, 0.94, 1.2, V, k_eigs=5, g12_inv_amu=0.02)

    np.testing.assert_allclose(by_nmax.evals, by_k.evals, rtol=0.0, atol=1e-12)
    assert np.max(np.abs(crossed.evals - by_k.evals)) > 1e-8


def test_shape_and_unit_validation_errors_are_explicit() -> None:
    R = np.linspace(0.8, 1.2, 5)
    V = np.zeros_like(R)

    with pytest.raises(ValueError, match="uniformly spaced"):
        sinc_dvr_1d(np.array([0.8, 0.9, 1.05, 1.2]), 1.0, np.zeros(4))
    with pytest.raises(ValueError, match="strictly increasing"):
        sinc_dvr_1d(R[::-1], 1.0, V)
    with pytest.raises(ValueError, match="shape"):
        sinc_dvr_1d(R, 1.0, V[:, None])
    with pytest.raises(ValueError, match="positive finite mass"):
        sinc_dvr_1d(R, 0.0, V)

    dvr = sinc_dvr_1d(R, 1.0, V)
    with pytest.raises(ValueError, match="integer state index"):
        trans_mu_1d(dvr, lambda rr: rr, 1.5)
    with pytest.raises(ValueError, match="outside the available state range"):
        trans_mu_1d(dvr, lambda rr: rr, R.size)
    with pytest.raises(ValueError, match="shape"):
        trans_mu_1d(dvr, lambda rr: np.ones((rr.size, 1)), 1)

    V2 = np.zeros((R.size, R.size))
    with pytest.raises(ValueError, match="shape"):
        product_dvr_2d(
            R,
            R,
            1.0,
            1.0,
            np.zeros((R.size, R.size - 1)),
            nmax=1,
        )
    with pytest.raises(ValueError, match="nmax must be at least 1"):
        product_dvr_2d(R, R, 1.0, 1.0, V2, nmax=0)
    with pytest.raises(ValueError, match="Need at least 2 eigenpairs"):
        product_dvr_2d(R, R, 1.0, 1.0, V2, k_eigs=1)

    dvr2 = product_dvr_2d(R, R, 1.0, 1.0, V2, nmax=1)
    with pytest.raises(ValueError, match="shape"):
        trans_mu_2d(dvr2, np.ones((R.size, R.size - 1)), 1)


def test_importing_pure_dvr_modules_does_not_request_pyscf() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys

class BlockPySCF(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyscf" or fullname.startswith("pyscf."):
            raise RuntimeError(f"unexpected PySCF import: {fullname}")
        return None

sys.meta_path.insert(0, BlockPySCF())
import pyscf_vscf.dvr
import pyscf_vscf.spectra
print("pure-ok")
"""
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "pure-ok"
