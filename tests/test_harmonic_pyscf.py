from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


pytestmark = pytest.mark.pyscf
pytest.importorskip("pyscf")


def _load_legacy_module():
    repo_root = Path(__file__).resolve().parents[1]
    legacy_path = repo_root / "pyscf_pme_pipeline.py"
    spec = importlib.util.spec_from_file_location("pyscf_pme_pipeline", legacy_path)
    assert spec is not None and spec.loader is not None
    legacy = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo_root))
    previous_module = sys.modules.get(spec.name)
    sys.modules[spec.name] = legacy
    try:
        spec.loader.exec_module(legacy)
    finally:
        sys.path.remove(str(repo_root))
        if previous_module is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous_module
    return legacy


def _water_like_pmol_and_masses():
    from pyscf_vscf.backends.pyscf import molecule_to_pyscf
    from pyscf_vscf.molecule import Molecule

    molecule = Molecule.from_arrays(
        ["O", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9572],
            [0.9266, 0.0, -0.2396],
        ],
        label="water",
    )
    return molecule_to_pyscf(molecule, basis="sto-3g"), molecule.masses


def test_legacy_cache_adapter_uses_package_checksum_manifest(tmp_path: Path) -> None:
    legacy = _load_legacy_module()
    from pyscf_vscf.cache import array_sha256

    path = tmp_path / "legacy-adapter.npz"
    values = np.arange(6, dtype=float).reshape(2, 3)
    legacy.dump_grid_npz(
        path,
        meta={"grid_cache_version": 2, "purpose": "adapter-test"},
        arrays={"values": values},
    )

    meta, arrays = legacy.load_grid_npz(path)

    assert meta["array_sha256"] == {"values": array_sha256(values)}
    np.testing.assert_allclose(arrays["values"], values)


def test_legacy_dev_fast_settings_are_materialized_before_backend_or_cache() -> None:
    legacy = _load_legacy_module()
    legacy.DEV_FAST = True

    settings = legacy._effective_es_settings(legacy.ESSettings())

    assert settings.method == "hf"
    assert settings.basis == "sto-3g"
    assert settings.scf_conv_tol == pytest.approx(1e-7)
    assert settings.scf_max_cycle == 50
    assert settings.dft_grid_level == 1


def test_legacy_normal_scan_uses_dedicated_centered_bounds(monkeypatch) -> None:
    legacy = _load_legacy_module()
    captured: dict[str, tuple[float, float]] = {}
    molecule = legacy.Molecule(
        ["O", "H"],
        np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]]),
        label="oh",
    )
    direction = np.zeros_like(molecule.coords)
    direction[1, 0] = 1.0

    monkeypatch.setattr(
        legacy,
        "calc_normal_mode_direction",
        lambda mol, cfg, bond: (direction, 1, 3000.0, np.eye(6), np.arange(6.0)),
    )

    def fake_grid(mol, cfg, u_dir, smin, smax, npts):
        del mol, cfg, u_dir
        captured["bounds"] = (float(smin), float(smax))
        return np.linspace(smin, smax, npts), np.zeros(npts), np.zeros((npts, 3))

    monkeypatch.setattr(legacy, "grid_1d_pes_dms_normal", fake_grid)
    monkeypatch.setattr(
        legacy,
        "_variational_helpers",
        lambda: (lambda value: value, lambda *args, **kwargs: [], object()),
    )

    legacy.run_1d(
        molecule,
        legacy.ESSettings(),
        legacy.Bond(0, 1),
        0.75,
        1.25,
        3,
        1,
        scan="normal",
        smin=-0.2,
        smax=0.25,
    )

    assert captured["bounds"] == (-0.2, 0.25)
    with pytest.raises(ValueError, match="--smin < 0 < --smax"):
        legacy.run_1d(
            molecule,
            legacy.ESSettings(),
            legacy.Bond(0, 1),
            0.75,
            1.25,
            3,
            1,
            scan="normal",
            smin=0.1,
            smax=0.2,
        )


def test_package_harmonic_helper_matches_legacy_on_real_pyscf_molecule() -> None:
    pytest.importorskip("pyscf")
    legacy = _load_legacy_module()
    from pyscf_vscf.harmonic import mass_weighted_freqs_modes

    assert legacy._pkg_harmonic is not None

    pmol, masses = _water_like_pmol_and_masses()
    H = np.diag(np.linspace(0.04, 0.16, 9))

    legacy.STRICT = True
    for rtproj in ["none", "mw_explicit", "pyscf"]:
        legacy_freqs, _ = legacy.mass_weighted_freqs_modes(
            pmol,
            H,
            masses,
            rtproj=rtproj,
            debug=False,
        )
        package_freqs, _ = mass_weighted_freqs_modes(
            pmol,
            H,
            masses,
            rtproj=rtproj,
            strict=True,
            debug=False,
        )
        np.testing.assert_allclose(package_freqs, legacy_freqs, rtol=0.0, atol=1e-8)


def test_workflow_analytic_hessian_helper_matches_legacy_direct_path() -> None:
    pytest.importorskip("pyscf")
    legacy = _load_legacy_module()
    from pyscf_vscf.workflows.harmonic import analytic_hessian

    hessian = np.arange(36, dtype=float).reshape(6, 6)

    class FakeHessian:
        @staticmethod
        def kernel() -> np.ndarray:
            return hessian

    class FakeMeanField:
        @staticmethod
        def Hessian() -> FakeHessian:
            return FakeHessian()

    np.testing.assert_allclose(
        analytic_hessian(FakeMeanField()),
        legacy._try_analytic_hessian(FakeMeanField()),
    )


def test_harmonic_workflow_real_pyscf_h2_hf_sto3g_smoke() -> None:
    pytest.importorskip("pyscf")
    from pyscf_vscf.molecule import Molecule
    from pyscf_vscf.settings import ESSettings
    from pyscf_vscf.workflows.harmonic import harmonic_analysis

    molecule = Molecule.from_arrays(
        ["H", "H"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
        label="h2",
    )
    cfg = ESSettings(
        method="hf",
        basis="sto-3g",
        use_density_fit=False,
        strict=True,
    )

    result = harmonic_analysis(molecule, cfg, rtproj="none")

    assert result.freqs_cm.shape == (6,)
    assert result.modes.shape == (6, 6)
    assert np.all(np.isfinite(result.freqs_cm))
    assert np.all(np.isfinite(result.modes))
    assert result.zpe_cm > 0.0


def test_harmonic_workflow_matches_legacy_h2_hf_sto3g() -> None:
    pytest.importorskip("pyscf")
    legacy = _load_legacy_module()
    from pyscf_vscf.molecule import Molecule
    from pyscf_vscf.settings import ESSettings
    from pyscf_vscf.workflows.harmonic import harmonic_analysis

    assert legacy._pkg_harmonic_workflow is not None

    symbols = ["H", "H"]
    coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    cfg = ESSettings(
        method="hf",
        basis="sto-3g",
        use_density_fit=False,
        strict=True,
    )
    package_result = harmonic_analysis(
        Molecule.from_arrays(symbols, coords, label="h2"),
        cfg,
        rtproj="none",
    )
    legacy_result = legacy.harmonic_analysis(
        legacy.Molecule(symbols, coords, label="h2"),
        legacy.ESSettings(
            method="hf",
            basis="sto-3g",
            use_density_fit=False,
            strict=True,
        ),
        rtproj="none",
    )

    assert isinstance(legacy_result, legacy.HarmonicResult)
    np.testing.assert_allclose(package_result.freqs_cm, legacy_result.freqs_cm, atol=1e-8)
    np.testing.assert_allclose(
        np.abs(package_result.modes), np.abs(legacy_result.modes), atol=1e-8
    )
    assert package_result.zpe_cm == pytest.approx(legacy_result.zpe_cm)


def test_legacy_run_opt_delegates_to_package_workflow(monkeypatch, tmp_path: Path) -> None:
    pytest.importorskip("pyscf")
    legacy = _load_legacy_module()
    calls = {}

    def fake_run_opt(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs

    monkeypatch.setattr(legacy, "_pkg_optimization", SimpleNamespace(run_opt=fake_run_opt))
    legacy.VERBOSE = True

    mol = legacy.Molecule(["H", "H"], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]), label="h2")
    cfg = legacy.ESSettings(method="hf", basis="sto-3g", use_density_fit=False)
    opt_out = tmp_path / "h2.xyz"

    legacy.run_opt(mol, cfg, opt_out=opt_out, opt_maxsteps=4, opt_conv="orca")

    assert calls["args"] == (mol, cfg)
    assert calls["kwargs"] == {
        "opt_out": opt_out,
        "opt_maxsteps": 4,
        "opt_conv": "orca",
        "verbose": True,
        "log_fn": legacy.log,
        "warn_fn": legacy.warn_once,
    }


def test_legacy_scan_grid_delegates_to_package_workflow(monkeypatch) -> None:
    pytest.importorskip("pyscf")
    legacy = _load_legacy_module()
    calls = {}

    def fake_grid_1d_pes_dms(*args, **kwargs):
        calls["args"] = args
        calls["kwargs"] = kwargs
        return np.array([0.8, 1.0]), np.array([0.0, 0.1]), np.zeros((2, 3))

    monkeypatch.setattr(
        legacy,
        "_pkg_scans",
        SimpleNamespace(grid_1d_pes_dms=fake_grid_1d_pes_dms),
    )

    mol = legacy.Molecule(["O", "H"], np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0]]), label="oh")
    cfg = legacy.ESSettings(method="hf", basis="sto-3g", use_density_fit=False)
    bond = legacy.Bond(0, 1)

    R, E, MU = legacy.grid_1d_pes_dms(mol, cfg, bond, Rmin=0.8, Rmax=1.0, npts=2)

    np.testing.assert_allclose(R, [0.8, 1.0])
    np.testing.assert_allclose(E, [0.0, 0.1])
    assert MU.shape == (2, 3)
    assert calls["args"] == (mol, cfg, bond, 0.8, 1.0, 2)
    assert calls["kwargs"]["energy_dipole_fn"] is legacy.energy_dipole
    assert calls["kwargs"]["executor_factory"] is legacy._pool_executor
    assert calls["kwargs"]["molecule_factory"] is legacy._legacy_molecule_with_coords
    assert calls["kwargs"]["progress_fn"] is legacy._progress_update
    assert calls["kwargs"]["log_fn"] is legacy.log
