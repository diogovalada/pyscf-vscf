from __future__ import annotations

import dataclasses
import os
import pickle
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.backends.pyscf import PySCFMeanFieldProvider
from pyscf_vscf.electronic import (
    ElectronicPointRequest,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from pyscf_vscf.settings import ESSettings


def _request(**changes: object) -> ElectronicPointRequest:
    values: dict[str, object] = {
        "nuclear_charges": (1, 1),
        "coordinates_A": np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]]),
    }
    values.update(changes)
    return ElectronicPointRequest(**values)


class _SettingsProvider:
    def __init__(self, settings: Mapping[str, object]):
        self.settings = dict(settings)

    def scientific_settings_payload(self) -> Mapping[str, object]:
        return self.settings

    def execution_provenance(self) -> Mapping[str, object]:
        return {}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        raise NotImplementedError


def test_es_settings_preserves_released_positional_field_order() -> None:
    settings = ESSettings("hf", "sto-3g", False, "aux", 1e-9, 77, 5)

    assert settings.method == "hf"
    assert settings.basis == "sto-3g"
    assert settings.use_density_fit is False
    assert settings.auxbasis == "aux"
    assert settings.scf_conv_tol == 1e-9
    assert settings.scf_max_cycle == 77
    assert settings.dft_grid_level == 5


@pytest.mark.parametrize(
    ("name", "changed"),
    [
        ("geometry", {"coordinates_A": [[0.0, 0.0, -0.38], [0.0, 0.0, 0.37]]}),
        ("charge", {"charge": 1}),
        ("spin", {"spin": 2}),
        ("state", {"electronic_state": "state-1"}),
        ("property", {"requested_properties": ("energy",)}),
        ("field", {"field_au": [1e-4, 0.0, 0.0]}),
        (
            "field_origin",
            {"field_au": [1e-4, 0.0, 0.0], "field_origin_A": [0.1, 0.0, 0.0]},
        ),
    ],
)
def test_request_scientific_changes_change_causal_identity(
    name: str,
    changed: dict[str, object],
) -> None:
    del name
    baseline = _request()
    modified = _request(**changed)

    assert baseline.causal_fingerprint("provider") != modified.causal_fingerprint("provider")


@pytest.mark.parametrize(
    ("setting", "left", "right"),
    [
        ("method", "hf", "pbe"),
        ("basis", "sto-3g", "6-31g"),
        ("reference", "rhf", "uhf"),
        ("scf_tolerance", 1e-9, 1e-11),
        ("dft_grid", 2, 3),
        ("frozen_core", "none", "one-core"),
        ("integral_approximation", "conventional", "density-fitted"),
    ],
)
def test_provider_scientific_settings_change_causal_identity(
    setting: str,
    left: object,
    right: object,
) -> None:
    request = _request()
    left_id = provider_scientific_fingerprint(_SettingsProvider({setting: left}))
    right_id = provider_scientific_fingerprint(_SettingsProvider({setting: right}))

    assert left_id != right_id
    assert request.causal_fingerprint(left_id) != request.causal_fingerprint(right_id)


def test_mean_field_runtime_resources_do_not_change_scientific_identity() -> None:
    settings = ESSettings(method="hf", basis="sto-3g", use_density_fit=False)
    small = PySCFMeanFieldProvider(settings, threads=1, max_memory_mb=256)
    large = PySCFMeanFieldProvider(
        settings,
        threads=8,
        max_memory_mb=8192,
        user_annotations={"path": "/different", "comment": "runtime only"},
    )

    assert provider_scientific_fingerprint(small) == provider_scientific_fingerprint(large)
    assert small.execution_provenance() != large.execution_provenance()
    assert large.execution_provenance()["host"]


def test_mean_field_provider_normalizes_inactive_hf_grid() -> None:
    baseline = PySCFMeanFieldProvider(
        ESSettings(method="hf", basis="sto-3g", use_density_fit=False)
    )
    ignored_grid = PySCFMeanFieldProvider(
        ESSettings(
            method="hf",
            basis="sto-3g",
            use_density_fit=False,
            dft_grid_level=9,
        )
    )

    assert provider_scientific_fingerprint(baseline) == provider_scientific_fingerprint(
        ignored_grid
    )


def test_real_mean_field_provider_fingerprints_effective_settings() -> None:
    providers = [
        PySCFMeanFieldProvider(ESSettings(method="hf", basis="sto-3g", use_density_fit=False)),
        PySCFMeanFieldProvider(ESSettings(method="pbe", basis="sto-3g", use_density_fit=False)),
        PySCFMeanFieldProvider(ESSettings(method="hf", basis="6-31g", use_density_fit=False)),
        PySCFMeanFieldProvider(
            ESSettings(
                method="hf",
                basis="sto-3g",
                use_density_fit=False,
                scf_conv_tol=1e-12,
            )
        ),
        PySCFMeanFieldProvider(
            ESSettings(
                method="pbe",
                basis="sto-3g",
                use_density_fit=False,
                dft_grid_level=2,
            )
        ),
        PySCFMeanFieldProvider(
            ESSettings(
                method="hf",
                basis="sto-3g",
                use_density_fit=True,
                auxbasis="weigend",
            )
        ),
    ]

    fingerprints = {provider_scientific_fingerprint(provider) for provider in providers}
    assert len(fingerprints) == len(providers)


def test_result_scientific_and_content_identity_are_separate() -> None:
    base = ElectronicResult(
        total_energy_Eh=-1.0,
        dipole_au=np.array([0.1, -0.2, 0.3]),
        dipole_unit="atomic_unit",
        dipole_frame="input_cartesian",
        converged=True,
        point_causal_fingerprint="point",
        provider_scientific_fingerprint="provider",
        scientific_diagnostics={"t1": 0.01},
        execution_diagnostics={"runtime_seconds": 1.0, "warnings": ["first"]},
        provenance={"software_version": "1", "path": "/first", "threads": 1},
    )
    changed_runtime = dataclasses.replace(
        base,
        execution_diagnostics={"runtime_seconds": 9.0, "warnings": ["changed"]},
        provenance={"software_version": "2", "path": "/second", "threads": 8},
    )

    assert base.scientific_fingerprint() == changed_runtime.scientific_fingerprint()
    assert base.content_fingerprint() != changed_runtime.content_fingerprint()
    restored = pickle.loads(pickle.dumps(base))
    with pytest.raises(ValueError):
        restored.dipole_au.setflags(write=True)
    assert restored.content_fingerprint() == base.content_fingerprint()


@pytest.mark.parametrize(
    ("unit", "frame"),
    [("debye", "input_cartesian"), ("atomic_unit", "body")],
)
def test_result_rejects_contradictory_dipole_conventions(unit: str, frame: str) -> None:
    with pytest.raises(ValueError):
        ElectronicResult(
            total_energy_Eh=-1.0,
            dipole_au=np.ones(3),
            dipole_unit=unit,
            dipole_frame=frame,
            converged=True,
            point_causal_fingerprint="point",
            provider_scientific_fingerprint="provider",
        )


def test_importing_package_does_not_import_pyscf_or_evaluate_a_point() -> None:
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
import pyscf_vscf
from pyscf_vscf.backends.pyscf import PySCFMeanFieldProvider
print("import-ok")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )

    assert process.returncode == 0, process.stderr
    assert process.stdout.strip() == "import-ok"
