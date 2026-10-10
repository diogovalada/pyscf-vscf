from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyscf_vscf import cli, settings, variational
from pyscf_vscf.settings import ESSettings
from pyscf_vscf.variational import TransitionRecord
from pyscf_vscf.workflows import optimization, scans


@pytest.fixture(autouse=True)
def _restore_process_environment():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _water_xyz(path: Path) -> Path:
    path.write_text("3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n")
    return path


def _record(frequency_cm: float, boundary: float | None = None) -> TransitionRecord:
    return TransitionRecord(
        state_index=1,
        quanta=(1,),
        frequency_cm=frequency_cm,
        transition_dipole_axis_D=0.1,
        integrated_cross_section_axis_omega_m2_per_s=2e-11,
        transition_dipole_norm_D=0.3,
        integrated_cross_section_isotropic_omega_m2_per_s=1e-11,
        grid_boundary_weight=boundary,
        ground_grid_boundary_weight=None if boundary is None else 1e-9,
    )


def _fake_grid(captured: dict[str, object]):
    def fake(molecule, cfg, bond, rmin, rmax, npts, **kwargs):
        captured["grid"] = (cfg, bond, rmin, rmax, npts, kwargs)
        R = np.linspace(float(rmin), float(rmax), int(npts))
        return R, (R - R.mean()) ** 2, np.zeros((R.size, 3))

    return fake


def _fake_variational(captured: dict[str, object], records):
    def fake(R, E, MU, redmass_amu, *, axis, vmax):
        captured["variational"] = (R, E, MU, redmass_amu, axis, vmax)
        return list(records)

    return fake


def test_help_documents_defaults_and_zero_based_bonds(capsys) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    assert "zero-based atom indices" in out
    assert "(default: wb97x)" in out
    assert "(default: aug-cc-pVTZ)" in out
    assert "--strict" not in out.replace("--no-strict", "")


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--bond", "0-5"], "outside 0..2 (indices are zero-based)"),
        (["--bond", "O-H"], "must be two zero-based atom indices"),
        (["--npts", "2"], "--npts must be at least 3"),
        (["--rmin", "1.3", "--rmax", "1.0"], "0 < --rmin < --rmax"),
        (["--scan", "normal", "--smin", "0.1"], "--smin < 0 < --smax"),
        (["--scan", "normal", "--dump-grid", "x.npz"], "only available for --scan lbs-frozen"),
    ],
)
def test_invalid_scan_options_fail_before_any_work(
    monkeypatch, tmp_path, capsys, extra, message
) -> None:
    monkeypatch.setattr(
        scans, "calc_normal_mode_direction", lambda *a, **k: pytest.fail("ran harmonic")
    )
    monkeypatch.setattr(scans, "grid_1d_pes_dms", lambda *a, **k: pytest.fail("ran scan"))
    xyz = _water_xyz(tmp_path / "water.xyz")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--xyz", str(xyz), "--task", "1d", *extra])

    assert excinfo.value.code == 2
    err = capsys.readouterr().err
    assert message in err
    assert "Runtime:" not in err


def test_missing_geometry_file_is_a_one_line_error(tmp_path, capsys) -> None:
    assert cli.main(["--xyz", str(tmp_path / "nope.xyz")]) == 1
    err = capsys.readouterr().err
    assert "pyscf-vscf: error:" in err
    assert "nope.xyz" in err
    assert "Traceback" not in err


def test_runtime_line_is_written_to_stderr(monkeypatch, tmp_path, capsys) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(scans, "grid_1d_pes_dms", _fake_grid(captured))
    monkeypatch.setattr(variational, "variational_1d", _fake_variational(captured, [_record(1.0)]))
    xyz = _water_xyz(tmp_path / "water.xyz")

    assert cli.main(["--xyz", str(xyz), "--task", "1d", "--npts", "5"]) == 0

    out, err = capsys.readouterr()
    assert "Runtime:" in err
    assert "Runtime:" not in out


def test_scan_defaults_to_memory_aware_process_parallelism(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(settings, "available_cpu_count", lambda **kwargs: 8)
    monkeypatch.setattr(settings, "available_memory_mb", lambda **kwargs: 6000.0)
    monkeypatch.setattr(scans, "grid_1d_pes_dms", _fake_grid(captured))
    monkeypatch.setattr(variational, "variational_1d", _fake_variational(captured, [_record(1.0)]))
    xyz = _water_xyz(tmp_path / "water.xyz")

    assert cli.main(["--xyz", str(xyz), "--task", "1d", "--npts", "21"]) == 0

    kwargs = captured["grid"][5]
    assert kwargs["executor_factory"] is not cli._SequentialExecutor
    # 0.75 * 6000 MB allows four 1 GiB workers; each gets an explicit budget.
    assert os.environ[settings.MAX_MEMORY_ENV_VAR] == "1125"
    assert kwargs["batch_size"] == 16


def test_explicit_workers_keep_previous_semantics(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(scans, "grid_1d_pes_dms", _fake_grid(captured))
    monkeypatch.setattr(variational, "variational_1d", _fake_variational(captured, [_record(1.0)]))
    xyz = _water_xyz(tmp_path / "water.xyz")

    assert cli.main(["--xyz", str(xyz), "--task", "1d", "--pes-workers", "1"]) == 0

    assert captured["grid"][5]["executor_factory"] is cli._SequentialExecutor
    assert settings.MAX_MEMORY_ENV_VAR not in os.environ


def test_reuse_grid_adopts_omitted_settings_from_the_cache(monkeypatch, tmp_path) -> None:
    from pyscf_vscf.io import read_xyz

    xyz = _water_xyz(tmp_path / "water.xyz")
    molecule = read_xyz(xyz)
    cfg = ESSettings(method="hf", basis="sto-3g", use_density_fit=False, scf_conv_tol=1e-9)
    R = np.linspace(0.7, 1.4, 15)
    cache = tmp_path / "oh.npz"
    scans.dump_lbs_frozen_1d_grid_cache(
        cache, molecule, cfg, "0-1", 0.7, 1.4, 15, R, (R - 1.0) ** 2, np.zeros((15, 3))
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(variational, "variational_1d", _fake_variational(captured, [_record(1.0)]))

    assert cli.main(["--xyz", str(xyz), "--task", "1d", "--reuse-grid", str(cache)]) == 0

    np.testing.assert_allclose(captured["variational"][0], R)


def test_reuse_grid_conflict_reports_the_differing_field(tmp_path, capsys) -> None:
    from pyscf_vscf.io import read_xyz

    xyz = _water_xyz(tmp_path / "water.xyz")
    molecule = read_xyz(xyz)
    cfg = ESSettings(method="hf", basis="sto-3g", use_density_fit=False)
    R = np.linspace(0.7, 1.4, 15)
    cache = tmp_path / "oh.npz"
    scans.dump_lbs_frozen_1d_grid_cache(
        cache, molecule, cfg, "0-1", 0.7, 1.4, 15, R, (R - 1.0) ** 2, np.zeros((15, 3))
    )

    argv = ["--xyz", str(xyz), "--task", "1d", "--reuse-grid", str(cache)]
    assert cli.main([*argv, "--basis", "cc-pvdz", "--npts", "21"]) == 1

    err = capsys.readouterr().err
    assert "electronic_structure.basis: cache='sto-3g', requested='cc-pvdz'" in err
    assert "scan.grid_A: cache has 15 values" in err
    assert "requested 21 values" in err


def test_scan_failure_writes_partial_file_and_explains(monkeypatch, tmp_path, capsys) -> None:
    original = scans.grid_1d_pes_dms

    def energy(molecule, cfg):
        del cfg
        coords = np.asarray(molecule.coords)
        if np.linalg.norm(coords[1] - coords[0]) > 1.0:
            raise RuntimeError("SCF did not converge")
        return 0.0, np.zeros(3)

    def failing_grid(molecule, cfg, bond, rmin, rmax, npts, **kwargs):
        return original(molecule, cfg, bond, rmin, rmax, npts, energy_dipole_fn=energy, **kwargs)

    monkeypatch.setattr(scans, "grid_1d_pes_dms", failing_grid)
    xyz = _water_xyz(tmp_path / "water.xyz")
    dump = tmp_path / "oh.npz"

    code = cli.main(
        [
            "--xyz",
            str(xyz),
            "--task",
            "1d",
            "--npts",
            "5",
            "--rmin",
            "0.8",
            "--rmax",
            "1.2",
            "--dump-grid",
            str(dump),
            "--pes-workers",
            "1",
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    partial = tmp_path / "oh.partial.npz"
    assert "3 of 5 point(s) completed, 1 failed" in err
    assert str(partial) in err
    assert "--continue-on-error" in err
    assert partial.exists()
    assert not dump.exists()


def test_json_output_contains_extended_records(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(scans, "grid_1d_pes_dms", _fake_grid(captured))
    monkeypatch.setattr(
        variational, "variational_1d", _fake_variational(captured, [_record(3700.0, 2e-6)])
    )
    xyz = _water_xyz(tmp_path / "water.xyz")
    out = tmp_path / "result.json"

    assert cli.main(["--xyz", str(xyz), "--task", "1d", "--npts", "5", "--json", str(out)]) == 0

    payload = json.loads(out.read_text())
    assert payload["schema"] == "pyscf-vscf-cli-result"
    assert payload["schema_version"] == 1
    assert payload["task"] == "1d"
    assert payload["scan"]["bond_zero_based"] == [0, 1]
    record = payload["transitions"][0]
    assert record["freq_cm"] == 3700.0
    assert record["grid_boundary_weight"] == 2e-6
    assert record["integrated_cross_section_axis_km_per_mol"] > 0.0
    assert payload["electronic_structure"]["resolved_scf_controls"]["max_cycle"] == 50


def test_boundary_flag_marks_squeezed_states(monkeypatch, tmp_path, capsys) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(scans, "grid_1d_pes_dms", _fake_grid(captured))
    monkeypatch.setattr(
        variational,
        "variational_1d",
        _fake_variational(captured, [_record(3700.0, 1e-7), _record(7000.0, 3e-3)]),
    )
    xyz = _water_xyz(tmp_path / "water.xyz")

    assert cli.main(["--xyz", str(xyz), "--task", "1d", "--intensity", "isotropic"]) == 0

    out, err = capsys.readouterr()
    lines = [line for line in out.splitlines() if "3700.0" in line or "7000.0" in line]
    assert not lines[0].endswith("*")
    assert lines[1].endswith("*")
    assert "1 state(s) place more than 1e-05" in err
    assert "orientation: isotropic" in out


def test_opt_default_output_follows_the_input_format(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def fake_run_opt(molecule, cfg, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            converged=True,
            backend="geomeTRIC",
            output_path=kwargs["opt_out"],
            molecule=molecule,
            stationarity=None,
        )

    monkeypatch.setattr(optimization, "run_opt", fake_run_opt)
    xyz = _water_xyz(tmp_path / "geometry.dat")

    assert cli.main(["--xyz", str(xyz), "--task", "opt"]) == 0

    assert captured["opt_out"] == Path("geometry.pyscf_opt.xyz")


def test_opt_rejects_unknown_output_suffix_before_optimizing(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(optimization, "run_opt", lambda *a, **k: pytest.fail("optimized"))
    xyz = _water_xyz(tmp_path / "water.xyz")

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--xyz", str(xyz), "--task", "opt", "--opt-out", str(tmp_path / "x.pdb")])
    assert excinfo.value.code == 2


def test_cache_options_are_rejected_for_non_scan_tasks(tmp_path) -> None:
    xyz = _water_xyz(tmp_path / "water.xyz")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--xyz", str(xyz), "--task", "harmonic", "--dump-grid", "x.npz"])
    assert excinfo.value.code == 2


def test_output_parent_must_be_a_directory(tmp_path) -> None:
    xyz = _water_xyz(tmp_path / "water.xyz")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--xyz", str(xyz), "--task", "harmonic", "--json", str(xyz / "out.json")])
    assert excinfo.value.code == 2


def test_infinite_normal_bounds_fail_before_harmonic_analysis(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        scans, "calc_normal_mode_direction", lambda *a, **k: pytest.fail("ran harmonic")
    )
    xyz = _water_xyz(tmp_path / "water.xyz")
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--xyz", str(xyz), "--task", "1d", "--scan", "normal", "--smax", "inf"])
    assert excinfo.value.code == 2


def test_partial_write_failure_keeps_the_scan_diagnostic(monkeypatch, tmp_path, capsys) -> None:
    original = scans.grid_1d_pes_dms

    def energy(molecule, cfg):
        del cfg
        if np.linalg.norm(np.asarray(molecule.coords)[1] - np.asarray(molecule.coords)[0]) > 1.0:
            raise RuntimeError("SCF did not converge")
        return 0.0, np.zeros(3)

    monkeypatch.setattr(
        scans,
        "grid_1d_pes_dms",
        lambda *a, **k: original(*a, energy_dipole_fn=energy, **k),
    )
    monkeypatch.setattr(
        scans,
        "dump_partial_scan_npz",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "Permission denied", "x")),
    )
    xyz = _water_xyz(tmp_path / "water.xyz")

    code = cli.main(
        [
            "--xyz",
            str(xyz),
            "--task",
            "1d",
            "--npts",
            "5",
            "--rmin",
            "0.8",
            "--rmax",
            "1.2",
            "--partial-out",
            str(tmp_path / "p.npz"),
            "--pes-workers",
            "1",
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "3 of 5 point(s) completed, 1 failed" in err
    assert "could not write completed points" in err
