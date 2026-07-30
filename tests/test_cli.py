from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyscf_vscf import cli
from pyscf_vscf.workflows import harmonic, optimization, scans


def _write_xyz(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "3",
                "water",
                "O 0.0 0.0 0.0",
                "H 1.0 0.0 0.0",
                "H 0.0 1.0 0.0",
                "",
            ]
        )
    )
    return path


def _write_mmol(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "#0 MidasMolecule",
                "",
                "#1 Xyz",
                "2 AA",
                "oh",
                "O 0.0 0.0 0.0",
                "H 1.0 0.0 0.0",
                "",
                "#0 MidasMoleculeEnd",
                "",
            ]
        )
    )
    return path


def test_help_and_version_do_not_import_pyscf_or_legacy_pipeline() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    blocker = """
import importlib.abc
import sys

class BlockExpensiveImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyscf" or fullname.startswith("pyscf."):
            raise RuntimeError(f"unexpected PySCF import: {fullname}")
        if fullname == "pyscf_pme_pipeline":
            raise RuntimeError("unexpected legacy pipeline import")
        return None

sys.meta_path.insert(0, BlockExpensiveImports())
from pyscf_vscf.cli import main
raise SystemExit(main(ARGV))
"""
    for argv, expected in ((["--help"], "--task"), (["--version"], "0.1.0a2")):
        code = blocker.replace("ARGV", repr(argv))
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            env=env,
            text=True,
            capture_output=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert expected in proc.stdout
        assert "pyscf_pme_pipeline" not in proc.stdout


def test_normal_relaxed_scan_is_exposed_in_package_cli() -> None:
    help_text = cli._build_parser().format_help()
    assert "normal-relaxed" in help_text
    assert "lbs-relaxed" not in help_text


def test_normal_coordinate_default_bounds_bracket_equilibrium() -> None:
    args = cli._build_parser().parse_args([])
    assert args.smin < 0.0 < args.smax


def test_development_fast_materializes_all_backend_settings() -> None:
    args = cli._build_parser().parse_args(["--dev-fast"])
    cfg, npts, tight_width = cli._build_es_settings(args)

    assert cfg.method == "hf"
    assert cfg.basis == "sto-3g"
    assert cfg.dispersion is None
    assert cfg.scf_conv_tol == pytest.approx(1e-7)
    assert cfg.scf_max_cycle == 50
    assert cfg.dft_grid_level == 1
    assert npts == 21
    assert tight_width == pytest.approx(0.2)


def test_executor_factory_uses_sequential_executor_for_single_worker() -> None:
    def plus_one(value: int) -> int:
        return value + 1

    factory = cli._executor_factory(1, 4)

    assert factory is cli._SequentialExecutor
    with factory() as executor:
        assert list(executor.map(plus_one, [1, 2])) == [2, 3]


def test_executor_factory_builds_process_pool_for_multiple_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeProcessPoolExecutor:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(cli, "ProcessPoolExecutor", FakeProcessPoolExecutor)

    factory = cli._executor_factory(3, 2)
    pool = factory()

    assert isinstance(pool, FakeProcessPoolExecutor)
    assert captured["max_workers"] == 3
    assert captured["initializer"] is cli._worker_init
    assert captured["initargs"] == (2,)
    assert captured["mp_context"].get_start_method() == "spawn"


def test_harmonic_task_dispatches_to_package_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = _write_mmol(tmp_path / "oh.mmol")
    captured: dict[str, object] = {}

    def fake_harmonic_analysis(molecule, cfg, *, rtproj: str, debug: bool):
        captured["molecule"] = molecule
        captured["cfg"] = cfg
        captured["rtproj"] = rtproj
        captured["debug"] = debug
        return SimpleNamespace(
            zpe_cm=123.45,
            freqs_cm=np.array([-1.0, 0.0, 111.1, 222.2]),
            modes=np.eye(6),
        )

    monkeypatch.setattr(harmonic, "harmonic_analysis", fake_harmonic_analysis)

    assert (
        cli.main(
            [
                "--mmol",
                str(input_path),
                "--task",
                "harmonic",
                "--method",
                "hf",
                "--basis",
                "sto-3g",
                "--dispersion",
                "none",
                "--no-ri",
                "--rtproj",
                "none",
                "-v",
            ]
        )
        == 0
    )

    cfg = captured["cfg"]
    assert captured["molecule"].label == "oh"
    assert cfg.method == "hf"
    assert cfg.basis == "sto-3g"
    assert cfg.use_density_fit is False
    assert cfg.dispersion is None
    assert captured["rtproj"] == "none"
    assert captured["debug"] is True
    out = capsys.readouterr().out
    assert "ZPE (harmonic): 123.45 cm^-1" in out
    assert "111.1 222.2" in out


def test_opt_task_reads_xyz_and_dispatches_to_package_workflow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    input_path = _write_xyz(tmp_path / "water.xyz")
    output_path = tmp_path / "opt.xyz"
    captured: dict[str, object] = {}

    def fake_run_opt(molecule, cfg, **kwargs):
        captured["molecule"] = molecule
        captured["cfg"] = cfg
        captured["kwargs"] = kwargs

    monkeypatch.setattr(optimization, "run_opt", fake_run_opt)

    assert (
        cli.main(
            [
                "--xyz",
                str(input_path),
                "--task",
                "opt",
                "--opt-out",
                str(output_path),
                "--opt-maxsteps",
                "7",
                "--opt-conv",
                "orca-tight",
                "--charge",
                "-1",
                "--spin",
                "1",
                "--no-strict",
                "--allow-fd-hessian",
            ]
        )
        == 0
    )

    molecule = captured["molecule"]
    cfg = captured["cfg"]
    kwargs = captured["kwargs"]
    assert molecule.label == "water"
    assert molecule.symbols == ["O", "H", "H"]
    np.testing.assert_allclose(molecule.coords[1], [1.0, 0.0, 0.0])
    assert molecule.charge == -1
    assert molecule.spin == 1
    assert cfg.strict is False
    assert cfg.allow_fd_hessian is True
    assert kwargs["opt_out"] == output_path
    assert kwargs["opt_maxsteps"] == 7
    assert kwargs["opt_conv"] == "orca-tight"


def test_1d_task_dispatches_grid_and_variational_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pyscf_vscf import variational

    input_path = _write_xyz(tmp_path / "water.xyz")
    captured: dict[str, object] = {}

    def fake_grid(molecule, cfg, bond, rmin, rmax, npts, **kwargs):
        captured["grid"] = (molecule, cfg, bond, rmin, rmax, npts, kwargs)
        return (
            np.linspace(float(rmin), float(rmax), int(npts)),
            np.array([0.0, 0.1, 0.2]),
            np.zeros((3, 3)),
        )

    def fake_variational(R, E, MU, redmass_amu, *, axis, vmax, intensity):
        captured["variational"] = (R, E, MU, redmass_amu, axis, vmax, intensity)
        return [
            {
                "v": 1,
                "freq_cm": 345.6,
                "transition_dipole_D": 0.1,
                "integrated_cross_section_omega_m2_per_s": 0.2,
                "orientation": "polarized-axis",
            }
        ]

    monkeypatch.setattr(scans, "grid_1d_pes_dms", fake_grid)
    monkeypatch.setattr(variational, "variational_1d", fake_variational)

    assert (
        cli.main(
            [
                "--xyz",
                str(input_path),
                "--task",
                "1d",
                "--bond",
                "O0-H1",
                "--rmin",
                "0.8",
                "--rmax",
                "1.0",
                "--npts",
                "3",
                "--vmax",
                "1",
                "--intensity",
                "axis",
                "--max-parallel",
                "4",
                "--pes-workers",
                "2",
            ]
        )
        == 0
    )

    _molecule, _cfg, bond, rmin, rmax, npts, kwargs = captured["grid"]
    assert (bond.O, bond.H) == (0, 1)
    assert (rmin, rmax, npts) == (0.8, 1.0, 3)
    assert kwargs["log_fn"] is not None
    assert kwargs["executor_factory"] is not cli._SequentialExecutor
    assert captured["variational"][5:] == (1, "axis")
    assert "345.6" in capsys.readouterr().out


def test_1d_task_supports_non_water_numeric_bond(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from pyscf_vscf import variational
    from pyscf_vscf.constants import atomic_mass_amu

    input_path = tmp_path / "hf.xyz"
    input_path.write_text(
        "\n".join(
            [
                "2",
                "HF",
                "H 0.0 0.0 0.0",
                "F 0.92 0.0 0.0",
                "",
            ]
        )
    )
    captured: dict[str, object] = {}

    def fake_grid(molecule, cfg, bond, rmin, rmax, npts, **kwargs):
        captured["grid"] = (molecule, cfg, bond, rmin, rmax, npts, kwargs)
        return (
            np.linspace(float(rmin), float(rmax), int(npts)),
            np.array([0.0, 0.1, 0.2]),
            np.zeros((3, 3)),
        )

    def fake_variational(R, E, MU, redmass_amu, *, axis, vmax, intensity):
        captured["variational"] = (R, E, MU, redmass_amu, axis, vmax, intensity)
        return [
            {
                "v": 1,
                "freq_cm": 123.4,
                "transition_dipole_D": 0.1,
                "integrated_cross_section_omega_m2_per_s": 0.2,
                "orientation": "polarized-axis",
            }
        ]

    monkeypatch.setattr(scans, "grid_1d_pes_dms", fake_grid)
    monkeypatch.setattr(variational, "variational_1d", fake_variational)

    assert (
        cli.main(
            [
                "--xyz",
                str(input_path),
                "--task",
                "1d",
                "--bond",
                "0-1",
                "--rmin",
                "0.8",
                "--rmax",
                "1.0",
                "--npts",
                "3",
            ]
        )
        == 0
    )

    _molecule, _cfg, bond, _rmin, _rmax, _npts, _kwargs = captured["grid"]
    assert (bond.i, bond.j) == (0, 1)
    expected_mu = (
        atomic_mass_amu("H") * atomic_mass_amu("F") / (atomic_mass_amu("H") + atomic_mass_amu("F"))
    )
    assert captured["variational"][3] == pytest.approx(expected_mu)


def test_1d_normal_task_dispatches_executor_to_normal_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pyscf_vscf import variational

    input_path = _write_xyz(tmp_path / "water.xyz")
    captured: dict[str, object] = {}

    def fake_normal_mode_direction(molecule, cfg, bond, **kwargs):
        captured["normal_mode"] = (molecule, cfg, bond, kwargs)
        u_dir = np.zeros_like(molecule.coords)
        u_dir[1, 0] = 1.0
        return u_dir, 3, 1234.5, np.eye(9), np.arange(9, dtype=float)

    def fake_grid(molecule, cfg, u_dir, smin, smax, npts, **kwargs):
        captured["grid"] = (molecule, cfg, u_dir, smin, smax, npts, kwargs)
        return (
            np.linspace(float(smin), float(smax), int(npts)),
            np.array([0.0, 0.1, 0.2]),
            np.zeros((3, 3)),
        )

    def fake_variational(R, E, MU, redmass_amu, *, axis, vmax, intensity):
        captured["variational"] = (R, E, MU, redmass_amu, axis, vmax, intensity)
        return [
            {
                "v": 1,
                "freq_cm": 2345.6,
                "transition_dipole_D": 0.1,
                "integrated_cross_section_omega_m2_per_s": 0.2,
                "orientation": "polarized-axis",
            }
        ]

    monkeypatch.setattr(scans, "calc_normal_mode_direction", fake_normal_mode_direction)
    monkeypatch.setattr(scans, "grid_1d_pes_dms_normal", fake_grid)
    monkeypatch.setattr(variational, "variational_1d", fake_variational)

    assert (
        cli.main(
            [
                "--xyz",
                str(input_path),
                "--task",
                "1d",
                "--scan",
                "normal",
                "--bond",
                "O0-H1",
                "--smin",
                "-0.1",
                "--smax",
                "0.1",
                "--npts",
                "3",
                "--vmax",
                "1",
            ]
        )
        == 0
    )

    _molecule, _cfg, _u_dir, smin, smax, npts, kwargs = captured["grid"]
    assert (smin, smax, npts) == (-0.1, 0.1, 3)
    assert kwargs["executor_factory"] is cli._SequentialExecutor
    assert "2345.6" in capsys.readouterr().out


def test_1d_normal_relaxed_task_wires_package_pyscf_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pyscf_vscf import variational
    from pyscf_vscf.backends import pyscf as pyscf_backend

    input_path = _write_xyz(tmp_path / "water.xyz")
    captured: dict[str, object] = {}

    def fake_normal_mode_direction(molecule, cfg, bond, **kwargs):
        captured["normal_mode"] = (molecule, cfg, bond, kwargs)
        u_dir = np.zeros_like(molecule.coords)
        u_dir[1, 0] = 1.0
        return u_dir, 2, 4321.0, np.eye(9), np.arange(9, dtype=float)

    def fake_grid(molecule, cfg, u_dir, smin, smax, npts, **kwargs):
        captured["grid"] = (molecule, cfg, u_dir, smin, smax, npts, kwargs)
        return (
            np.linspace(float(smin), float(smax), int(npts)),
            np.array([0.0, 0.1, 0.2]),
            np.zeros((3, 3)),
        )

    def fake_variational(R, E, MU, redmass_amu, *, axis, vmax, intensity):
        captured["variational"] = (R, E, MU, redmass_amu, axis, vmax, intensity)
        return [
            {
                "v": 1,
                "freq_cm": 3210.0,
                "transition_dipole_D": 0.1,
                "integrated_cross_section_omega_m2_per_s": 0.2,
                "orientation": "polarized-axis",
            }
        ]

    monkeypatch.setattr(scans, "calc_normal_mode_direction", fake_normal_mode_direction)
    monkeypatch.setattr(scans, "grid_1d_pes_dms_normal_relaxed", fake_grid)
    monkeypatch.setattr(variational, "variational_1d", fake_variational)

    assert (
        cli.main(
            [
                "--xyz",
                str(input_path),
                "--task",
                "1d",
                "--scan",
                "normal-relaxed",
                "--bond",
                "O0-H1",
                "--smin",
                "-0.1",
                "--smax",
                "0.1",
                "--npts",
                "3",
                "--vmax",
                "1",
            ]
        )
        == 0
    )

    _molecule, _cfg, _u_dir, smin, smax, npts, kwargs = captured["grid"]
    assert (smin, smax, npts) == (-0.1, 0.1, 3)
    assert kwargs["relaxed_point_fn"] is pyscf_backend.normal_relaxed_point
    assert kwargs["executor_factory"] is cli._SequentialExecutor
    assert "3210.0" in capsys.readouterr().out


def test_2d_task_dispatches_grid_and_variational_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pyscf_vscf import variational

    input_path = _write_xyz(tmp_path / "water.xyz")
    captured: dict[str, object] = {}

    def fake_grid(molecule, cfg, b1, b2, R1, R2, **kwargs):
        captured["grid"] = (molecule, cfg, b1, b2, R1, R2, kwargs)
        return R1, R2, np.zeros((3, 3)), np.zeros((3, 3, 3))

    def fake_variational(R1, R2, E, MU, mu1, mu2, *, axis, nmax, g12_inv_amu, intensity):
        captured["variational"] = (
            R1,
            R2,
            E,
            MU,
            mu1,
            mu2,
            axis,
            nmax,
            g12_inv_amu,
            intensity,
        )
        return [
            {
                "n": 1,
                "assignment": (1, 0),
                "assignment_weight": 0.99,
                "freq_cm": 456.7,
                "transition_dipole_D": 0.3,
                "integrated_cross_section_omega_m2_per_s": 0.4,
                "orientation": "isotropic",
            }
        ]

    monkeypatch.setattr(scans, "grid_2d_pes_dms", fake_grid)
    monkeypatch.setattr(variational, "variational_2d", fake_variational)

    assert (
        cli.main(
            [
                "--xyz",
                str(input_path),
                "--task",
                "2d",
                "--bond",
                "O0-H1",
                "--bond2",
                "O0-H2",
                "--rmin",
                "0.8",
                "--rmax",
                "1.0",
                "--npts",
                "3",
                "--vmax",
                "2",
                "--keo",
                "gmatrix",
                "--intensity",
                "vector",
                "--max-parallel",
                "4",
                "--pes-workers",
                "2",
            ]
        )
        == 0
    )

    _molecule, _cfg, b1, b2, R1, R2, kwargs = captured["grid"]
    assert (b1.O, b1.H) == (0, 1)
    assert (b2.O, b2.H) == (0, 2)
    np.testing.assert_allclose(R1, [0.8, 0.9, 1.0])
    np.testing.assert_allclose(R2, [0.8, 0.9, 1.0])
    assert kwargs["log_fn"] is not None
    assert kwargs["executor_factory"] is not cli._SequentialExecutor
    assert captured["variational"][7:] == (2, 0.0, "vector")
    assert "456.7" in capsys.readouterr().out
