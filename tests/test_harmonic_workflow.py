from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf import harmonic as harmonic_helpers
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.settings import ESSettings
from pyscf_vscf.workflows import harmonic as workflow


class _FakePMol:
    natm = 2

    def atom_coords(self, unit: str = "Bohr") -> np.ndarray:
        assert unit == "Bohr"
        return np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])


class _FakeHessian:
    def __init__(self, hessian: np.ndarray):
        self._hessian = hessian

    def kernel(self) -> np.ndarray:
        return self._hessian


class _FakeMeanField:
    def __init__(self, hessian: np.ndarray):
        self._hessian = hessian

    def Hessian(self) -> _FakeHessian:
        return _FakeHessian(self._hessian)


def _h2_molecule() -> Molecule:
    return Molecule.from_arrays(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        label="h2",
    )


def test_importing_workflow_module_does_not_request_pyscf() -> None:
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
import pyscf_vscf.workflows.harmonic
print("workflow-ok")
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
    assert proc.stdout.strip() == "workflow-ok"


def test_harmonic_analysis_uses_analytic_hessian_and_existing_harmonic_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molecule = _h2_molecule()
    cfg = ESSettings(method="hf", basis="sto-3g", use_density_fit=False, dispersion=None)
    pmol = _FakePMol()
    hessian = np.eye(6)

    monkeypatch.setattr(workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: pmol)
    monkeypatch.setattr(
        workflow.pyscf_backend,
        "make_mean_field",
        lambda pmol_arg, cfg_arg: _FakeMeanField(hessian),
    )

    result = workflow.harmonic_analysis(molecule, cfg, rtproj="none")

    expected_freqs, expected_modes = harmonic_helpers.mass_weighted_freqs_modes(
        pmol,
        hessian,
        molecule.masses,
        rtproj="none",
        strict=True,
    )
    np.testing.assert_allclose(result.freqs_cm, expected_freqs)
    np.testing.assert_allclose(np.abs(result.modes), np.abs(expected_modes))
    assert result.zpe_cm == pytest.approx(harmonic_helpers.zpe_cm_from_freqs(expected_freqs))
    assert result.hessian_provenance == "analytic"


def test_harmonic_analysis_blocks_semi_numerical_dispersion_hessian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ESSettings(dispersion="d4", allow_fd_hessian=False)
    monkeypatch.setattr(
        workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: _FakePMol()
    )
    monkeypatch.setattr(
        workflow.pyscf_backend,
        "make_mean_field",
        lambda pmol_arg, cfg_arg: _FakeMeanField(np.eye(6)),
    )

    with pytest.raises(RuntimeError, match="semi-numerical.*--allow-fd-hessian"):
        workflow.harmonic_analysis(_h2_molecule(), cfg, rtproj="none")


def test_harmonic_analysis_records_semi_numerical_dispersion_hessian(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ESSettings(dispersion="d4", allow_fd_hessian=True)
    monkeypatch.setattr(
        workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: _FakePMol()
    )
    monkeypatch.setattr(
        workflow.pyscf_backend,
        "make_mean_field",
        lambda pmol_arg, cfg_arg: _FakeMeanField(np.eye(6)),
    )

    result = workflow.harmonic_analysis(_h2_molecule(), cfg, rtproj="none")

    assert result.hessian_provenance == "analytic-electronic+finite-difference-dispersion"


def test_harmonic_analysis_strict_analytic_failure_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ESSettings(strict=True, allow_fd_hessian=True, dispersion=None)
    monkeypatch.setattr(
        workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: _FakePMol()
    )
    monkeypatch.setattr(workflow.pyscf_backend, "make_mean_field", lambda pmol, cfg_arg: object())

    def fail_hessian(mf: object) -> np.ndarray:
        raise RuntimeError("synthetic hessian failure")

    monkeypatch.setattr(workflow, "analytic_hessian", fail_hessian)

    with pytest.raises(RuntimeError, match="synthetic hessian failure"):
        workflow.harmonic_analysis(_h2_molecule(), cfg, rtproj="none")


def test_harmonic_analysis_non_strict_analytic_failure_still_blocks_fd_unless_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = ESSettings(strict=False, allow_fd_hessian=False, dispersion=None)
    monkeypatch.setattr(
        workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: _FakePMol()
    )
    monkeypatch.setattr(workflow.pyscf_backend, "make_mean_field", lambda pmol, cfg_arg: object())

    def fail_hessian(mf: object) -> np.ndarray:
        raise RuntimeError("synthetic hessian failure")

    monkeypatch.setattr(workflow, "analytic_hessian", fail_hessian)

    with pytest.raises(RuntimeError, match="blocked; pass --allow-fd-hessian"):
        workflow.harmonic_analysis(_h2_molecule(), cfg, rtproj="none")


def test_harmonic_analysis_uses_fd_hessian_when_allowed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    molecule = _h2_molecule()
    cfg = ESSettings(strict=True, allow_fd_hessian=True, dispersion=None)
    pmol = _FakePMol()
    hessian = np.eye(6)
    captured_x0: list[np.ndarray] = []

    monkeypatch.setattr(workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: pmol)
    monkeypatch.setattr(
        workflow.pyscf_backend, "make_mean_field", lambda pmol_arg, cfg_arg: object()
    )
    monkeypatch.setattr(workflow, "analytic_hessian", lambda mf: None)

    def fake_fd(mol: object, cfg_arg: object, x0_bohr: np.ndarray) -> np.ndarray:
        captured_x0.append(np.asarray(x0_bohr, dtype=float))
        return hessian

    monkeypatch.setattr(workflow, "finite_difference_hessian_from_gradients", fake_fd)

    result = workflow.harmonic_analysis(molecule, cfg, rtproj="none")

    np.testing.assert_allclose(captured_x0[0], pmol.atom_coords(unit="Bohr").reshape(-1))
    assert result.freqs_cm.shape == (6,)
    assert result.zpe_cm > 0.0


def test_finite_difference_hessian_from_gradients_uses_central_columns() -> None:
    molecule = _h2_molecule()
    cfg = ESSettings()
    x0 = np.linspace(-0.3, 0.4, 6)
    force_constant = np.diag(np.linspace(1.0, 2.0, 6))
    progress: list[tuple[int, int, str]] = []

    def gradient(mol: object, cfg_arg: object, xflat_bohr: np.ndarray) -> np.ndarray:
        del mol, cfg_arg
        return force_constant @ xflat_bohr

    hessian = workflow.finite_difference_hessian_from_gradients(
        molecule,
        cfg,
        x0,
        h=1e-4,
        gradient_fn=gradient,
        progress_fn=lambda done, total, label: progress.append((done, total, label)),
    )

    np.testing.assert_allclose(hessian, force_constant, atol=1e-12)
    assert progress[-1] == (6, 6, "Hessian columns")


def test_finite_difference_fallback_preserves_all_es_settings() -> None:
    cfg = ESSettings(
        method="pbe0",
        basis="def2-tzvp",
        use_density_fit=False,
        auxbasis="custom-jkfit",
        dispersion=None,
        rtproj="mw_explicit",
        strict=False,
        allow_fd_hessian=True,
        scf_conv_tol=2e-9,
        dft_grid_level=5,
    )

    copied = workflow._legacy_fd_gradient_settings(cfg)

    assert copied == cfg
    assert copied is not cfg


def test_stationarity_diagnostic_summarizes_gradient_components() -> None:
    class Grad:
        @staticmethod
        def kernel() -> np.ndarray:
            return np.array([[1.0, 2.0, 2.0], [0.0, 0.0, 4.0]])

    class MF:
        @staticmethod
        def nuc_grad_method() -> Grad:
            return Grad()

    diagnostic = workflow.stationarity_diagnostic(MF())

    assert diagnostic.max_component == pytest.approx(4.0)
    assert diagnostic.rms_component == pytest.approx(np.sqrt(25.0 / 6.0))
    assert diagnostic.max_atom == pytest.approx(4.0)
    assert diagnostic.rms_atom == pytest.approx(np.sqrt((9.0 + 16.0) / 2.0))
    assert "Geometry stationarity" in workflow.format_stationarity_diagnostic(diagnostic)
