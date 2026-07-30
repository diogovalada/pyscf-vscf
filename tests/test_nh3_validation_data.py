from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1] / "validation_data" / "nh3_three_mode"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _report() -> dict:
    return json.loads((ROOT / "report.json").read_text(encoding="utf-8"))


def test_nh3_validation_manifest_hashes_every_archived_artifact() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))

    assert len(manifest["files"]) >= 30
    for relative, record in manifest["files"].items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert _sha256(path) == record["sha256"], relative


def test_nh3_three_mode_validation_accepts_only_converged_manifolds() -> None:
    acceptance = _report()["acceptance"]

    assert acceptance["passed"] is True
    assert acceptance["accepted_variants"] == [
        "wider_37x37",
        "widest_43x43",
        "converged_49x49",
    ]
    assert set(acceptance["converged_manifolds"]) == {
        "fundamental",
        "binary_combination",
        "triple_combination",
    }
    assert "first_overtone" not in acceptance["converged_manifolds"]
    assert (
        acceptance["grid_spreads_by_manifold"]["first_overtone"]["vscf_centroid_spread_cm"]
        > acceptance["criteria"]["maximum_grid_centroid_spread_cm"]
    )
    observed = acceptance["observed"]
    assert observed["maximum_vscf_exact_centroid_error_cm"] < 5.0
    assert observed["minimum_exact_manifold_weight"] > 0.8
    assert observed["all_vscf_states_converged"] is True


def test_nh3_repeated_electronic_points_and_final_fundamental_are_stable() -> None:
    variants = _report()["variants"]
    original = variants["full_25x25"]["manifolds"]
    repeated = variants["repeated_25x25_from_wide31"]["manifolds"]
    for name in original:
        assert repeated[name]["exact_centroid_cm"] == pytest.approx(
            original[name]["exact_centroid_cm"], abs=2e-8
        )
        assert repeated[name]["vscf_centroid_cm"] == pytest.approx(
            original[name]["vscf_centroid_cm"], abs=2e-8
        )

    final = variants["converged_49x49"]["manifolds"]["fundamental"]
    assert final["exact_centroid_cm"] == pytest.approx(3492.701378566256, abs=1e-8)
    assert abs(final["vscf_minus_exact_centroid_cm"]) < 0.1
    assert final["minimum_exact_manifold_weight"] > 0.9999
