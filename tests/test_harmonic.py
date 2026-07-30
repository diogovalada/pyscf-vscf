from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import numpy.linalg as npl
import pytest

from pyscf_vscf.constants import AMU, HARTREE_TO_CM
from pyscf_vscf.harmonic import (
    HarmonicResult,
    as_cart_hessian,
    cart_to_hess4,
    handle_imaginary_modes,
    mass_weight,
    mass_weighted_freqs_modes_from_coords,
    mw_rt_projector_explicit,
    mw_rt_projector_pyscf_like,
    signed_freqs_from_evals,
    zpe_cm_from_freqs,
)


def _evals_from_signed_cm(freqs_cm: list[float]) -> np.ndarray:
    freqs = np.asarray(freqs_cm, dtype=float)
    return np.sign(freqs) * (np.abs(freqs) / HARTREE_TO_CM) ** 2


def test_harmonic_result_and_zpe_match_legacy_cutoff() -> None:
    freqs = np.array([0.0, 1e-6, 100.0, 200.0])
    result = HarmonicResult(freqs_cm=freqs, modes=np.eye(4), zpe_cm=zpe_cm_from_freqs(freqs))

    assert result.zpe_cm == pytest.approx(150.0)


def test_signed_freqs_from_evals_preserves_imaginary_sign() -> None:
    w2 = _evals_from_signed_cm([-3.0, 0.0, 7.5])

    np.testing.assert_allclose(signed_freqs_from_evals(w2), [-3.0, 0.0, 7.5])


def test_hessian_shape_conversion_round_trips_pyscf_layout() -> None:
    natm = 2
    H4 = np.arange((3 * natm) ** 2, dtype=float).reshape(natm, natm, 3, 3)

    Hc = as_cart_hessian(H4, natm)

    np.testing.assert_allclose(
        Hc,
        H4.transpose(0, 2, 1, 3).reshape(3 * natm, 3 * natm),
    )
    np.testing.assert_allclose(cart_to_hess4(Hc, natm), H4)
    legacy_passthrough = np.zeros((5, 5))
    assert as_cart_hessian(legacy_passthrough, natm).shape == (5, 5)
    with pytest.raises(ValueError, match="Hc shape mismatch"):
        cart_to_hess4(np.zeros((5, 5)), natm)


def test_mass_weight_uses_amu_and_coordinate_repetition() -> None:
    masses = np.array([1.0, 4.0])
    H = np.arange(36, dtype=float).reshape(6, 6)
    M = np.repeat(masses, 3) * AMU

    got = mass_weight(H, masses)

    np.testing.assert_allclose(got, H / np.sqrt(np.outer(M, M)))
    with pytest.raises(ValueError, match="positive"):
        mass_weight(np.eye(3), np.array([0.0]))


def test_mass_weighted_freqs_modes_from_coords_rtproj_none() -> None:
    H = np.diag(np.arange(1, 7, dtype=float)) * AMU
    masses = np.array([1.0, 1.0])
    coords_bohr = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

    freqs, modes = mass_weighted_freqs_modes_from_coords(
        H,
        masses,
        coords_bohr,
        rtproj="none",
    )

    np.testing.assert_allclose(freqs, np.sqrt(np.arange(1, 7, dtype=float)) * HARTREE_TO_CM)
    assert modes.shape == (6, 6)


@pytest.mark.parametrize("strict", [False, True])
def test_mass_weighted_freqs_modes_rtproj_none_preserves_imaginary_sign(strict: bool) -> None:
    H = np.diag([-1.0, 2.0, 3.0, 4.0, 5.0, 6.0]) * AMU
    masses = np.array([1.0, 1.0])
    coords_bohr = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])

    freqs, modes = mass_weighted_freqs_modes_from_coords(
        H,
        masses,
        coords_bohr,
        rtproj="none",
        strict=strict,
    )

    np.testing.assert_allclose(
        freqs,
        np.array([-1.0, np.sqrt(2.0), np.sqrt(3.0), 2.0, np.sqrt(5.0), np.sqrt(6.0)])
        * HARTREE_TO_CM,
    )
    np.testing.assert_allclose(modes, np.eye(6))


def test_imaginary_mode_handling_strict_non_strict_and_rtproj_none() -> None:
    bad_vib = _evals_from_signed_cm([0.0, 0.0, 0.0, 0.0, 0.0, -11.0])

    with pytest.raises(RuntimeError, match="Imaginary vibrational modes"):
        handle_imaginary_modes(bad_vib, natm=2, rtproj="mw_explicit", strict=True, rt_rank=5)

    warnings: list[tuple[str, str]] = []
    returned = handle_imaginary_modes(
        bad_vib,
        natm=2,
        rtproj="mw_explicit",
        strict=False,
        rt_rank=5,
        warn_fn=lambda key, msg: warnings.append((key, msg)),
    )

    np.testing.assert_allclose(returned, bad_vib)
    assert warnings[0][0] == "imag_modes_non_strict"
    assert "NON-STRICT" in warnings[0][1]

    mid_vib = _evals_from_signed_cm([0.0, 0.0, 0.0, 0.0, 0.0, -5.0])
    warnings.clear()
    handle_imaginary_modes(
        mid_vib,
        natm=2,
        rtproj="mw_explicit",
        strict=True,
        rt_rank=5,
        warn_fn=lambda key, msg: warnings.append((key, msg)),
    )
    assert warnings[0][0] == "imag_modes_warn"

    np.testing.assert_allclose(
        handle_imaginary_modes(bad_vib, natm=2, rtproj="none", strict=True, rt_rank=5),
        bad_vib,
    )


def test_explicit_rt_projector_has_nonlinear_water_rank_and_projector_properties() -> None:
    coords_bohr = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8],
            [1.75, 0.0, -0.45],
        ]
    )
    masses_amu = np.array([15.99491461957, 1.00782503223, 1.00782503223])

    P, info = mw_rt_projector_explicit(np.eye(9), coords_bohr, masses_amu)

    assert int(info["rt_rank"]) == 6
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    assert float(np.trace(P)) == pytest.approx(3.0, abs=1e-11)
    evals = npl.eigvalsh(P)
    assert int(np.sum(evals > 0.5)) == 3
    assert int(np.sum(evals < 0.5)) == 6


def test_explicit_rt_projector_has_linear_diatomic_rank_and_projector_properties() -> None:
    coords_bohr = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    masses_amu = np.array([1.00782503223, 34.968852682])

    P, info = mw_rt_projector_explicit(np.eye(6), coords_bohr, masses_amu)

    assert int(info["rt_rank"]) == 5
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    assert float(np.trace(P)) == pytest.approx(1.0, abs=1e-11)
    evals = npl.eigvalsh(P)
    assert int(np.sum(evals > 0.5)) == 1
    assert int(np.sum(evals < 0.5)) == 5


def test_pyscf_like_rt_projector_accepts_injected_thermo_fake() -> None:
    class FakeThermo:
        @staticmethod
        def _get_TR(masses_amu: np.ndarray, coords_bohr: np.ndarray) -> np.ndarray:
            del masses_amu, coords_bohr
            TR = np.zeros((6, 9))
            TR[:6, :6] = np.eye(6)
            return TR

        @staticmethod
        def rotation_const(masses_amu: np.ndarray, coords_bohr: np.ndarray) -> np.ndarray:
            del masses_amu, coords_bohr
            return np.array([1.0, 2.0, 3.0])

        @staticmethod
        def _get_rotor_type(rot_const: np.ndarray) -> str:
            del rot_const
            return "REGULAR"

    coords_bohr = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.8],
            [1.75, 0.0, -0.45],
        ]
    )
    masses_amu = np.array([16.0, 1.0, 1.0])

    P, info = mw_rt_projector_pyscf_like(np.eye(9), coords_bohr, masses_amu, FakeThermo)

    assert int(info["rt_rank"]) == 6
    np.testing.assert_allclose(P, P.T, atol=1e-12)
    np.testing.assert_allclose(P @ P, P, atol=1e-12)
    assert float(np.trace(P)) == pytest.approx(3.0, abs=1e-12)


def test_importing_harmonic_module_does_not_request_pyscf() -> None:
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
import pyscf_vscf.harmonic
print("harmonic-ok")
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
    assert proc.stdout.strip() == "harmonic-ok"
