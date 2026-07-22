from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.cache
def test_archived_grid_and_log_hashes_match_manifest() -> None:
    root = Path(__file__).resolve().parents[1] / "validation_data"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["systems"].values():
        for path_key, hash_key in (
            ("grid", "grid_sha256"),
            ("log", "log_sha256"),
            ("geometry", "geometry_sha256"),
        ):
            payload = (root / entry[path_key]).resolve().read_bytes()
            assert hashlib.sha256(payload).hexdigest() == entry[hash_key]
        with np.load((root / entry["grid"]).resolve(), allow_pickle=False) as data:
            assert data["E_Eh"].shape == (41, 41)
            assert data["MU_Debye"].shape == (41, 41, 3)


@pytest.mark.cache
def test_checked_in_report_matches_independent_orca_intensity_scale() -> None:
    root = Path(__file__).resolve().parents[1] / "validation_data"
    report = json.loads((root / "convergence_report.json").read_text(encoding="utf-8"))

    comparisons = [
        comparison
        for system in report["systems"].values()
        for comparison in system["independent_orca_intensity_benchmark"]
    ]
    assert len(comparisons) == 6
    assert max(comparison["relative_intensity_error"] for comparison in comparisons) < 0.4


@pytest.mark.cache
def test_checked_in_molecular_vscf_benchmark_matches_exact_2d_dvr() -> None:
    root = Path(__file__).resolve().parents[1] / "validation_data"
    report = json.loads((root / "convergence_report.json").read_text(encoding="utf-8"))

    benchmarks = [
        system["molecular_vscf_vs_exact_2d_dvr"] for system in report["systems"].values()
    ]
    assert len(benchmarks) == 3
    assert all(benchmark["ground_converged"] for benchmark in benchmarks)
    assert all(len(benchmark["states"]) == 3 for benchmark in benchmarks)
    assert max(benchmark["maximum_absolute_error_cm"] for benchmark in benchmarks) < 25.0

    acceptance = [
        system["fundamental_convergence_acceptance"] for system in report["systems"].values()
    ]
    assert all(item["passed"] for item in acceptance)
    assert all(
        "cannot be cited as converged" in item["unmatched_state_policy"] for item in acceptance
    )
