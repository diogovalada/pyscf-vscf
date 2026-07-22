"""NPZ cache helpers.

These names mirror :mod:`pyscf_vscf.io` for callers that prefer a cache-focused
module.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .io import dump_grid_npz, load_grid_npz
from .settings import coerce_es_settings, default_auxbasis

CACHE_SCHEMA_VERSION = 2


def canonical_json(value) -> str:
    """Serialize scientific provenance deterministically for fingerprinting."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scientific_fingerprint(value) -> str:
    """Return a SHA-256 fingerprint of canonical JSON-compatible provenance."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def array_sha256(array: np.ndarray) -> str:
    """Hash an array including dtype, shape, and contiguous byte content."""

    arr = np.ascontiguousarray(np.asarray(array))
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(canonical_json(list(arr.shape)).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def molecule_provenance(molecule) -> dict:
    """Return geometry, isotope, charge, and spin provenance."""

    symbols = [str(symbol) for symbol in getattr(molecule, "symbols")]
    coords = np.asarray(getattr(molecule, "coords"), dtype=float)
    masses_fn = getattr(molecule, "analysis_masses", None)
    if callable(masses_fn):
        masses = np.asarray(masses_fn(), dtype=float)
    else:
        masses = np.asarray(getattr(molecule, "masses"), dtype=float)
    return {
        "symbols": symbols,
        "coordinates_A": coords.tolist(),
        "masses_amu": masses.tolist(),
        "charge": int(getattr(molecule, "charge", 0)),
        "spin": int(getattr(molecule, "spin", 0)),
        "label": str(getattr(molecule, "label", "mol")),
    }


def electronic_structure_provenance(cfg) -> dict:
    """Return every electronic-structure setting and the effective RI basis."""

    settings = coerce_es_settings(cfg)
    values = {
        "method": settings.method,
        "basis": settings.basis,
        "use_density_fit": settings.use_density_fit,
        "auxbasis": settings.auxbasis,
        "dispersion": settings.dispersion,
        "rtproj": settings.rtproj,
        "strict": settings.strict,
        "allow_fd_hessian": settings.allow_fd_hessian,
        "scf_conv_tol": settings.scf_conv_tol,
        "scf_max_cycle": settings.scf_max_cycle,
        "dft_grid_level": settings.dft_grid_level,
    }
    values["effective_auxbasis"] = (
        settings.auxbasis or default_auxbasis(settings.basis) if settings.use_density_fit else None
    )
    values["backend"] = "pyscf"
    values["software_versions"] = runtime_provenance()["distributions"]
    return values


def runtime_provenance() -> dict:
    """Return reproducibility-relevant interpreter and distribution versions."""

    distributions = {}
    for name in (
        "pyscf-vscf",
        "numpy",
        "scipy",
        "pyscf",
        "pyscf-dispersion",
        "dftd4",
    ):
        try:
            distributions[name] = version(name)
        except PackageNotFoundError:
            distributions[name] = None
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "distributions": distributions,
    }


def scientific_cache_metadata(molecule, cfg, scan: dict) -> dict:
    """Build schema-v2 metadata with a complete scientific fingerprint."""

    scientific = {
        "molecule": molecule_provenance(molecule),
        "electronic_structure": electronic_structure_provenance(cfg),
        "scan": scan,
    }
    return {
        "grid_cache_version": CACHE_SCHEMA_VERSION,
        "scientific": scientific,
        "scientific_fingerprint_sha256": scientific_fingerprint(scientific),
        "runtime": runtime_provenance(),
    }


def validate_scientific_cache_metadata(actual: dict, expected: dict) -> None:
    """Fail closed unless schema and complete scientific fingerprints match."""

    version_actual = actual.get("grid_cache_version")
    if version_actual != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported grid cache schema {version_actual!r}; expected "
            f"{CACHE_SCHEMA_VERSION}. Legacy caches must be regenerated or explicitly migrated."
        )
    embedded = actual.get("scientific")
    embedded_fingerprint = actual.get("scientific_fingerprint_sha256")
    if embedded_fingerprint != scientific_fingerprint(embedded):
        raise ValueError("Grid cache scientific metadata fingerprint is corrupt")
    assert_meta_equal(
        "scientific_fingerprint_sha256",
        embedded_fingerprint,
        expected["scientific_fingerprint_sha256"],
    )


def assert_meta_equal(label: str, actual, expected) -> None:
    """Raise when a cache metadata field does not match exactly."""

    if actual != expected:
        raise ValueError(f"Grid cache mismatch for {label}: expected {expected!r}, got {actual!r}")


def assert_meta_close(label: str, actual: float, expected: float, tol: float = 1e-10) -> None:
    """Raise when a numeric cache metadata field differs beyond tolerance."""

    if abs(float(actual) - float(expected)) > tol:
        raise ValueError(f"Grid cache mismatch for {label}: expected {expected!r}, got {actual!r}")


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "array_sha256",
    "assert_meta_close",
    "assert_meta_equal",
    "canonical_json",
    "dump_grid_npz",
    "electronic_structure_provenance",
    "load_grid_npz",
    "molecule_provenance",
    "runtime_provenance",
    "scientific_cache_metadata",
    "scientific_fingerprint",
    "validate_scientific_cache_metadata",
]
