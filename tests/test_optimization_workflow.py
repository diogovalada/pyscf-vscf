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
        blocked = ("pyscf", "geometric")
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


def test_optimizer_convergence_kwargs() -> None:
    assert workflow.optimization_convergence_kwargs() == {
        "convergence_gmax": 4.5e-5,
        "convergence_grms": 3.0e-5,
    }


def _patch_engine(monkeypatch, *, converged: bool, opt_coords, stationarity_error=None):
    captured: dict[str, object] = {}
    scanner = object()

    class GeometricSolver:
        @staticmethod
        def kernel(method: object, **kwargs: object):
            captured["scanner"] = method
            captured["kwargs"] = kwargs
            return converged, _FakePySCFMol(opt_coords)

    def fake_converged(molecule: object, factory: object) -> SimpleNamespace:
        captured["stationarity_factory"] = factory
        if stationarity_error is not None:
            raise stationarity_error
        return SimpleNamespace(
            mean_field=_FakeMeanField(), build=SimpleNamespace(configuration_snapshot={})
        )

    monkeypatch.setattr(
        workflow, "_factory_gradient_scanner", lambda mol, factory, guard=None: scanner
    )
    monkeypatch.setattr(workflow, "_load_geometric_solver", lambda: GeometricSolver)
    monkeypatch.setattr(workflow.pyscf_backend, "converged_mean_field", fake_converged)
    return captured, scanner


def test_optimize_geometry_runs_geometric_and_writes_only_when_asked(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    molecule = _FakeMolecule()
    cfg = ESSettings(method="hf", basis="sto-3g", use_density_fit=False)
    opt_coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.70]])
    captured, scanner = _patch_engine(monkeypatch, converged=True, opt_coords=opt_coords)

    result = workflow.optimize_geometry(molecule, cfg, max_steps=7)

    assert captured["scanner"] is scanner
    assert captured["kwargs"] == {
        "convergence_gmax": 4.5e-5,
        "convergence_grms": 3.0e-5,
        "maxsteps": 7,
    }
    assert result.backend == "geomeTRIC"
    assert result.converged is True
    assert result.output_path is None
    np.testing.assert_allclose(result.molecule.coords, opt_coords)
    assert result.stationarity is not None
    assert list(tmp_path.iterdir()) == []

    output_path = tmp_path / "optimized.xyz"
    result = workflow.optimize_geometry(molecule, cfg, output_path=output_path)
    assert result.output_path == output_path
    assert "h2 (PySCF optimized)" in output_path.read_text()


def test_optimize_geometry_writes_mmol(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _patch_engine(monkeypatch, converged=True, opt_coords=_FakeMolecule.coords)
    output_path = tmp_path / "optimized.mmol"

    result = workflow.optimize_geometry(
        _FakeMolecule(), {"basis": "sto-3g"}, output_path=output_path, max_steps=3
    )

    assert result.output_path == output_path
    assert "#0 MidasMolecule" in output_path.read_text()


def test_optimize_geometry_rejects_unknown_suffix_before_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured, _ = _patch_engine(monkeypatch, converged=True, opt_coords=_FakeMolecule.coords)

    with pytest.raises(ValueError, match=".xyz or .mmol"):
        workflow.optimize_geometry(
            _FakeMolecule(), {"basis": "sto-3g"}, output_path=tmp_path / "x.pdb"
        )
    assert "scanner" not in captured


def test_non_strict_optimization_warns_when_not_converged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_engine(monkeypatch, converged=False, opt_coords=_FakeMolecule.coords)

    with pytest.warns(RuntimeWarning, match="did not converge within the allowed steps"):
        result = workflow.optimize_geometry(
            _FakeMolecule(), {"basis": "sto-3g"}, max_steps=1, strict=False
        )
    assert result.converged is False
    with pytest.raises(RuntimeError, match="did not converge"):
        workflow.optimize_geometry(_FakeMolecule(), {"basis": "sto-3g"}, max_steps=1)


def test_failed_stationarity_check_warns_and_keeps_the_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_engine(
        monkeypatch,
        converged=True,
        opt_coords=_FakeMolecule.coords,
        stationarity_error=RuntimeError("gradient unavailable"),
    )

    with pytest.warns(RuntimeWarning, match="gradient check failed: gradient unavailable"):
        result = workflow.optimize_geometry(_FakeMolecule(), {"basis": "sto-3g"})
    assert result.stationarity is None


def test_geometric_cannot_reconfigure_or_close_logging(monkeypatch, tmp_path) -> None:
    import logging
    import logging.config

    root = logging.getLogger()
    log_file = tmp_path / "caller.log"
    handler = logging.FileHandler(log_file)
    root.addHandler(handler)
    before = (list(root.handlers), root.level)
    seen = {}

    class GeometricSolver:
        @staticmethod
        def kernel(method: object, **kwargs: object):
            logging.config.fileConfig("does-not-exist.ini", disable_existing_loggers=False)
            seen["quiet"] = not logging.getLogger("geometric.nifty").isEnabledFor(logging.INFO)
            return True, _FakePySCFMol(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.7]]))

    monkeypatch.setattr(workflow, "_load_geometric_solver", lambda: GeometricSolver)
    try:
        workflow._run_geometric(object(), 5)
        assert seen["quiet"]
        assert (list(root.handlers), root.level) == before
        root.warning("still logging")
        handler.flush()
        assert "still logging" in log_file.read_text()
    finally:
        root.removeHandler(handler)
        handler.close()
