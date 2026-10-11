from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf import cli
from pyscf_vscf.contracts import (
    ElectronicPointRequest,
    ElectronicResult,
    provider_scientific_fingerprint,
)


@pytest.fixture(autouse=True)
def _restore_process_environment():
    saved = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(saved)


def _water(path: Path) -> Path:
    path.write_text("3\nwater\nO 0.0 0.0 0.1173\nH 0.0 0.7572 -0.4692\nH 0.0 -0.7572 -0.4692\n")
    return path


@dataclass(frozen=True)
class MorseProvider:
    """Analytic O-H Morse surface along atom 1, linear dipole; fails beyond 2 A."""

    fail_beyond_A: float = 2.0

    def scientific_settings_payload(self) -> dict[str, object]:
        return {"schema": "test-morse", "fail_beyond_A": self.fail_beyond_A}

    def execution_provenance(self) -> dict[str, object]:
        return {}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        xyz = np.asarray(request.coordinates_A)
        r = float(np.linalg.norm(xyz[1] - xyz[0]))
        if r > self.fail_beyond_A:
            raise RuntimeError("SCF did not converge")
        energy = 0.18 * (1.0 - np.exp(-2.2 * (r - 0.96))) ** 2
        provider_id = provider_scientific_fingerprint(self)
        return ElectronicResult(
            total_energy_Eh=energy - 76.0,
            dipole_au=np.array([0.0, 0.3 * r, 0.1]),
            dipole_unit="atomic_unit",
            dipole_frame="input_cartesian",
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
        )


def _use_morse(monkeypatch, provider: MorseProvider) -> None:
    original = cli._prepare_method

    def prepare(context):
        original(context)
        context.provider = provider

    monkeypatch.setattr(cli, "_prepare_method", prepare)


def test_no_arguments_prints_help_and_fails(capsys) -> None:
    assert cli.main([]) == 2
    assert "scan1d" in capsys.readouterr().err


def test_flag_style_invocation_routes_to_legacy_cli_with_notice(monkeypatch, tmp_path, capsys):
    from pyscf_vscf import _cli_legacy

    seen = {}
    monkeypatch.setattr(_cli_legacy, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    cli.main(["--xyz", str(_water(tmp_path / "w.xyz")), "--task", "harmonic"])
    assert seen["argv"][0] == "--xyz"
    assert "deprecated" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["scan1d", "W", "--bond", "0-5"], "zero-based"),
        (["scan1d", "W", "--bond", "0-1", "--range", "1.0", "0.8", "21"], "MIN < MAX"),
        (["scan1d", "W", "--bond", "0-1", "--range", "0.8", "1.0", "2"], "at least 3"),
        (["scan2d", "W", "--bond", "0-1"], "exactly two --bond"),
        (["scan1d", "W", "--bond", "0-1", "--recipe", "m:r", "--basis", "sto-3g"], "--recipe"),
        (["opt", "W", "--output", "x.pdb"], ".xyz or .mmol"),
    ],
)
def test_invalid_options_fail_before_any_work(tmp_path, capsys, argv, message) -> None:
    geometry = str(_water(tmp_path / "w.xyz"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main([geometry if item == "W" else item for item in argv])
    assert excinfo.value.code == 2
    assert message in capsys.readouterr().err


def test_scan1d_computes_resumes_writes_and_reuses(monkeypatch, tmp_path, capsys) -> None:
    _use_morse(monkeypatch, MorseProvider())
    geometry = str(_water(tmp_path / "w.xyz"))
    store = tmp_path / "points.sqlite"
    output = tmp_path / "oh.npz"
    argv = ["scan1d", geometry, "--bond", "0-1", "--range", "0.6", "1.9", "41", "--workers", "1"]

    assert cli.main([*argv, "--store", str(store), "--output", str(output), "--states", "2"]) == 0
    first = capsys.readouterr().out
    assert cli.main([*argv, "--store", str(store), "--states", "2"]) == 0
    assert capsys.readouterr().out == first
    assert (
        cli.main(["scan1d", geometry, "--bond", "0-1", "--reuse", str(output), "--states", "2"])
        == 0
    )
    assert capsys.readouterr().out == first
    assert cli.main(["inspect", str(output)]) == 0
    assert "bond-1d" in capsys.readouterr().out


def test_scan_failure_keeps_points_and_explains_resume(monkeypatch, tmp_path, capsys) -> None:
    _use_morse(monkeypatch, MorseProvider(fail_beyond_A=1.5))
    geometry = str(_water(tmp_path / "w.xyz"))
    store = tmp_path / "points.sqlite"

    code = cli.main(
        [
            "scan1d",
            geometry,
            "--bond",
            "0-1",
            "--range",
            "0.6",
            "1.9",
            "14",
            "--workers",
            "1",
            "--store",
            str(store),
        ]
    )

    assert code == 1
    err = capsys.readouterr().err
    assert "1 failed" in err
    assert "rerun to resume" in err
    assert "--continue-on-error" in err


def test_json_output_and_boundary_flags(monkeypatch, tmp_path, capsys) -> None:
    _use_morse(monkeypatch, MorseProvider())
    geometry = str(_water(tmp_path / "w.xyz"))
    out = tmp_path / "r.json"

    assert (
        cli.main(
            [
                "scan1d",
                geometry,
                "--bond",
                "0-1",
                "--range",
                "0.8",
                "1.2",
                "21",
                "--workers",
                "1",
                "--states",
                "3",
                "--json",
                str(out),
            ]
        )
        == 0
    )

    payload = json.loads(out.read_text())
    assert payload["command"] == "scan1d"
    assert payload["scan"]["bonds_zero_based"] == [[0, 1]]
    assert len(payload["transitions"]) == 3
    assert "probability on the outermost grid points" in capsys.readouterr().err


def test_reuse_rejects_a_scan_of_another_bond(monkeypatch, tmp_path, capsys) -> None:
    _use_morse(monkeypatch, MorseProvider())
    geometry = str(_water(tmp_path / "w.xyz"))
    output = tmp_path / "oh.npz"
    cli.main(
        [
            "scan1d",
            geometry,
            "--bond",
            "0-1",
            "--range",
            "0.8",
            "1.2",
            "11",
            "--workers",
            "1",
            "--output",
            str(output),
        ]
    )
    capsys.readouterr()

    assert cli.main(["scan1d", geometry, "--bond", "0-2", "--reuse", str(output)]) == 1
    assert "covers bonds" in capsys.readouterr().err


def _write_v4_scan(tmp_path: Path, geometry: str, capsys) -> Path:
    output = tmp_path / "oh.npz"
    argv = ["scan1d", geometry, "--bond", "0-1", "--range", "0.6", "1.9", "31"]
    assert cli.main([*argv, "--workers", "1", "--output", str(output)]) == 0
    capsys.readouterr()
    return output


def test_reuse_takes_the_method_from_the_file(monkeypatch, tmp_path, capsys) -> None:
    _use_morse(monkeypatch, MorseProvider())
    geometry = str(_water(tmp_path / "w.xyz"))
    output = _write_v4_scan(tmp_path, geometry, capsys)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["scan1d", geometry, "--bond", "0-1", "--reuse", str(output), "--method", "hf"])
    assert excinfo.value.code == 2
    assert "takes the method from the scan file" in capsys.readouterr().err

    report = tmp_path / "reuse.json"
    argv = ["scan1d", geometry, "--bond", "0-1", "--reuse", str(output), "--json", str(report)]
    assert cli.main(argv) == 0
    payload = json.loads(report.read_text())
    assert payload["method"]["schema"] == "test-morse"
    assert payload["provider_scientific_fingerprint"] == provider_scientific_fingerprint(
        MorseProvider()
    )


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--charge", "1", "--spin", "1"], "charge or spin"),
        (["--coordinate", "normal"], "not normal-1d"),
    ],
)
def test_reuse_rejects_another_system_or_coordinate(
    monkeypatch, tmp_path, capsys, extra, message
) -> None:
    _use_morse(monkeypatch, MorseProvider())
    geometry = str(_water(tmp_path / "w.xyz"))
    output = _write_v4_scan(tmp_path, geometry, capsys)

    assert cli.main(["scan1d", geometry, "--bond", "0-1", "--reuse", str(output), *extra]) == 1
    assert message in capsys.readouterr().err


def _legacy_cache(tmp_path: Path, geometry: str, dimension: int) -> Path:
    import warnings

    from pyscf_vscf.cache import dump_grid_npz
    from pyscf_vscf.io import read_xyz
    from pyscf_vscf.settings import ESSettings
    from pyscf_vscf.workflows import scans as legacy

    molecule = read_xyz(geometry)
    settings = ESSettings(method="hf", basis="sto-3g")
    R = np.linspace(0.7, 1.6, 31)
    morse = 0.18 * (1.0 - np.exp(-2.2 * (R - 0.96))) ** 2
    path = tmp_path / f"legacy{dimension}d.npz"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if dimension == 1:
            dipoles = np.stack([np.zeros_like(R), 0.3 * R, np.full_like(R, 0.1)], axis=-1)
            legacy.dump_lbs_frozen_1d_grid_cache(
                path, molecule, settings, "0-1", R[0], R[-1], R.size, R, morse, dipoles
            )
        else:
            spec = (R[0], R[-1], R.size)
            meta = legacy.lbs_frozen_2d_cache_metadata(
                molecule, settings, "0-1", "0-2", spec, spec
            )
            energy = morse[:, None] + morse[None, :]
            dipoles = np.zeros((R.size, R.size, 3))
            dipoles[..., 1] = 0.3 * (R[:, None] - R[None, :])
            dump_grid_npz(
                path,
                meta=meta,
                arrays={"R1_A": R, "R2_A": R, "E_Eh": energy, "MU_Debye": dipoles},
            )
    return path


@pytest.mark.parametrize("dimension", [1, 2])
def test_reuse_reads_matching_legacy_caches(tmp_path, capsys, dimension) -> None:
    geometry = str(_water(tmp_path / "w.xyz"))
    cache = _legacy_cache(tmp_path, geometry, dimension)
    bonds = ["--bond", "0-1"] if dimension == 1 else ["--bond", "0-1", "--bond", "0-2"]
    command = f"scan{dimension}d"
    report = tmp_path / "legacy.json"

    argv = [command, geometry, *bonds, "--reuse", str(cache), "--json", str(report)]
    assert cli.main(argv) == 0
    payload = json.loads(report.read_text())
    assert payload["method"]["method"] == "hf"
    assert payload["provider_scientific_fingerprint"] is None
    assert payload["transitions"]
    capsys.readouterr()

    wrong = ["--bond", "0-2"] if dimension == 1 else ["--bond", "0-2", "--bond", "0-1"]
    assert cli.main([command, geometry, *wrong, "--reuse", str(cache)]) == 1
    assert "scans bond" in capsys.readouterr().err
