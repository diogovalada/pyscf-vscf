from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyscf_vscf.settings import ESSettings
from pyscf_vscf.workflows import optimization as workflow


class _FakePySCFMol:
    def __init__(self, coords_ang: np.ndarray):
        self._coords_ang = np.asarray(coords_ang, dtype=float)

    def atom_coords(self, unit: str = "Angstrom") -> np.ndarray:
        assert unit == "Angstrom"
        return self._coords_ang


class _FakeGrad:
    @staticmethod
    def kernel() -> np.ndarray:
        return np.array([[1.0e-5, -2.0e-5, 3.0e-5], [0.0, 4.0e-5, 0.0]])


class _FakeMeanField:
    def __init__(self) -> None:
        self.verbose = -1

    @staticmethod
    def nuc_grad_method() -> _FakeGrad:
        return _FakeGrad()


class _FakeMolecule:
    symbols = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    charge = 0
    spin = 0
    label = "h2"
    masses = np.array([1.00782503223, 1.00782503223])


def test_importing_optimization_module_does_not_request_optional_optimizers() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys

class BlockOptionalOptimizers(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        blocked = ("pyscf", "geometric", "berny")
        if fullname in blocked or fullname.startswith(tuple(f"{name}." for name in blocked)):
            raise RuntimeError(f"unexpected optional optimizer import: {fullname}")
        return None

sys.meta_path.insert(0, BlockOptionalOptimizers())
import pyscf_vscf.workflows.optimization
print("optimization-ok")
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
    assert proc.stdout.strip() == "optimization-ok"


def test_optimizer_profile_kwargs_match_legacy_thresholds() -> None:
    assert workflow.opt_kwargs_for_profile("orca") == {
        "convergence_gmax": 4.5e-5,
        "convergence_grms": 3.0e-5,
    }
    assert workflow.opt_kwargs_for_profile("orca-tight", backend="berny") == {
        "gradientmax": 4.5e-5,
        "gradientrms": 3.0e-5,
    }
    with pytest.raises(ValueError, match="Unknown --opt-conv profile"):
        workflow.opt_kwargs_for_profile("loose")


def test_run_opt_uses_geometric_path_writes_xyz_and_checks_stationarity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    molecule = _FakeMolecule()
    cfg = ESSettings(method="hf", basis="sto-3g", use_density_fit=False, dispersion=None)
    initial_pmol = _FakePySCFMol(molecule.coords)
    opt_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.70]])
    opt_pmol = _FakePySCFMol(opt_coords)
    mean_fields = [_FakeMeanField(), _FakeMeanField()]
    captured: dict[str, object] = {}

    class GeometricSolver:
        @staticmethod
        def kernel(mf: object, **kwargs: object) -> tuple[bool, _FakePySCFMol]:
            captured["mf"] = mf
            captured["kwargs"] = kwargs
            return True, opt_pmol

    def fake_molecule_to_pyscf(mol: object, basis: str) -> _FakePySCFMol:
        captured.setdefault("basis", basis)
        return initial_pmol if mol is molecule else opt_pmol

    monkeypatch.setattr(workflow.pyscf_backend, "molecule_to_pyscf", fake_molecule_to_pyscf)
    monkeypatch.setattr(
        workflow.pyscf_backend,
        "make_mean_field",
        lambda pmol, cfg_arg: mean_fields.pop(0),
    )
    monkeypatch.setattr(workflow, "_load_geometric_solver", lambda: GeometricSolver)
    monkeypatch.setattr(
        workflow,
        "_load_berny_solver",
        lambda: pytest.fail("Berny fallback should not be used"),
    )

    output_path = tmp_path / "optimized.xyz"
    result = workflow.run_opt(
        molecule,
        cfg,
        opt_out=output_path,
        opt_maxsteps=7,
        opt_conv="orca",
        verbose=True,
    )

    assert result.backend == "geomeTRIC"
    assert result.converged is True
    assert result.output_path == output_path
    np.testing.assert_allclose(result.molecule.coords, opt_coords)
    assert result.stationarity is not None
    assert captured["kwargs"] == {
        "convergence_gmax": 4.5e-5,
        "convergence_grms": 3.0e-5,
        "maxsteps": 7,
    }
    assert captured["mf"].verbose == 4
    assert "h2 (PySCF optimized)" in output_path.read_text()
    assert "Optimized-geometry gradient check" in capsys.readouterr().out


def test_run_opt_falls_back_to_berny_and_writes_mmol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    molecule = _FakeMolecule()
    cfg = {"basis": "sto-3g", "strict": True}
    opt_pmol = _FakePySCFMol(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.71]]))
    captured: dict[str, object] = {}
    warnings: list[tuple[str, str]] = []

    class BernySolver:
        @staticmethod
        def kernel(mf: object, **kwargs: object) -> tuple[bool, _FakePySCFMol]:
            captured["mf"] = mf
            captured["kwargs"] = kwargs
            return True, opt_pmol

    monkeypatch.setattr(workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: opt_pmol)
    monkeypatch.setattr(
        workflow.pyscf_backend, "make_mean_field", lambda pmol, cfg_arg: _FakeMeanField()
    )
    monkeypatch.setattr(
        workflow,
        "_load_geometric_solver",
        lambda: (_ for _ in ()).throw(ImportError("no geometric")),
    )
    monkeypatch.setattr(workflow, "_load_berny_solver", lambda: BernySolver)

    output_path = tmp_path / "optimized.mmol"
    result = workflow.run_opt(
        molecule,
        cfg,
        opt_out=output_path,
        opt_maxsteps=3,
        opt_conv="orca_tight",
        warn_fn=lambda key, msg: warnings.append((key, msg)),
    )

    assert result.backend == "berny"
    assert result.output_path == output_path
    assert captured["kwargs"] == {
        "gradientmax": 4.5e-5,
        "gradientrms": 3.0e-5,
        "maxsteps": 3,
    }
    assert warnings[0][0] == "geometric_missing_fallback_berny"
    assert "#0 MidasMolecule" in output_path.read_text()


def test_run_opt_warns_when_non_strict_optimization_does_not_converge(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    molecule = _FakeMolecule()
    cfg = SimpleNamespace(basis="sto-3g", strict=False)
    opt_pmol = _FakePySCFMol(molecule.coords)
    warnings: list[tuple[str, str]] = []

    class GeometricSolver:
        @staticmethod
        def kernel(mf: object, **kwargs: object) -> tuple[bool, _FakePySCFMol]:
            return False, opt_pmol

    monkeypatch.setattr(workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: opt_pmol)
    monkeypatch.setattr(
        workflow.pyscf_backend, "make_mean_field", lambda pmol, cfg_arg: _FakeMeanField()
    )
    monkeypatch.setattr(workflow, "_load_geometric_solver", lambda: GeometricSolver)

    result = workflow.run_opt(
        molecule,
        cfg,
        opt_out=tmp_path / "optimized.xyz",
        opt_maxsteps=1,
        opt_conv="orca",
        warn_fn=lambda key, msg: warnings.append((key, msg)),
    )

    assert result.converged is False
    assert (
        "opt_not_converged",
        "Geometry optimization did not converge within the allowed steps",
    ) in warnings


def test_run_opt_keeps_going_when_post_optimization_stationarity_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    molecule = _FakeMolecule()
    cfg = ESSettings(method="hf", basis="sto-3g", use_density_fit=False, dispersion=None)
    opt_pmol = _FakePySCFMol(molecule.coords)
    warnings: list[tuple[str, str]] = []

    class GeometricSolver:
        @staticmethod
        def kernel(mf: object, **kwargs: object) -> tuple[bool, _FakePySCFMol]:
            return True, opt_pmol

    def fake_make_mean_field(pmol: object, cfg_arg: object) -> _FakeMeanField:
        if fake_make_mean_field.calls:
            raise RuntimeError("gradient unavailable")
        fake_make_mean_field.calls += 1
        return _FakeMeanField()

    fake_make_mean_field.calls = 0

    monkeypatch.setattr(workflow.pyscf_backend, "molecule_to_pyscf", lambda mol, basis: opt_pmol)
    monkeypatch.setattr(workflow.pyscf_backend, "make_mean_field", fake_make_mean_field)
    monkeypatch.setattr(workflow, "_load_geometric_solver", lambda: GeometricSolver)

    result = workflow.run_opt(
        molecule,
        cfg,
        opt_out=tmp_path / "optimized.xyz",
        opt_maxsteps=1,
        opt_conv="orca",
        warn_fn=lambda key, msg: warnings.append((key, msg)),
    )

    assert result.stationarity is None
    assert warnings == [
        ("opt_grad_check_failed", "Post-opt gradient check failed: gradient unavailable")
    ]
