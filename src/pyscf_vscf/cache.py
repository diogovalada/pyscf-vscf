"""NPZ cache helpers.

These names mirror :mod:`pyscf_vscf.io` for callers that prefer a cache-focused
module.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import warnings
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from .io import dump_grid_npz, load_grid_npz
from .settings import coerce_es_settings, default_auxbasis, resolve_scf_controls

CACHE_SCHEMA_VERSION = 3


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


def molecule_electronic_identity(molecule) -> dict:
    """Return the mass-independent molecular identity of electronic points."""

    symbols = ["H" if str(symbol).upper() == "D" else str(symbol) for symbol in molecule.symbols]
    return {
        "symbols": symbols,
        "coordinates_A": np.asarray(molecule.coords, dtype=float).tolist(),
        "charge": int(molecule.charge),
        "spin": int(molecule.spin),
    }


def molecule_provenance(molecule) -> dict:
    """Return recorded geometry, isotope, mass, charge, spin, and label provenance."""

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


def electronic_structure_identity(cfg, *, backend_identity: str = "pyscf") -> dict:
    """Return settings that can change an electronic energy or dipole."""

    settings = coerce_es_settings(cfg)
    values = {
        "method": settings.method,
        "basis": settings.basis,
        "use_density_fit": settings.use_density_fit,
        "auxbasis": settings.auxbasis,
        "scf_conv_tol": settings.scf_conv_tol,
        "scf_max_cycle": settings.scf_max_cycle,
        "dft_grid_level": settings.dft_grid_level,
    }
    values["effective_auxbasis"] = (
        settings.auxbasis or default_auxbasis(settings.basis) if settings.use_density_fit else None
    )
    backend = str(backend_identity).strip()
    if not backend:
        raise ValueError("backend_identity must be a non-empty stable identifier")
    values["backend"] = backend
    return values


def electronic_structure_provenance(cfg, *, backend_identity: str = "pyscf") -> dict:
    """Return the electronic identity plus the runtime used to evaluate it.

    For the built-in PySCF backend the provenance also records the SCF
    controls actually applied after resolving unset fields to the package
    defaults. They are provenance rather than identity: caches written before
    0.2.2 recorded ``None`` for unset controls, and those calculations used
    PySCF's then-configurable defaults, which cannot be recovered afterwards.
    """

    identity = electronic_structure_identity(cfg, backend_identity=backend_identity)
    provenance = {
        "identity": identity,
        "software_versions": runtime_provenance()["distributions"],
    }
    if identity["backend"] == "pyscf":
        controls = resolve_scf_controls(coerce_es_settings(cfg))
        provenance["resolved_scf_controls"] = {
            "conv_tol": controls.conv_tol,
            "max_cycle": controls.max_cycle,
            "dft_grid_level": controls.dft_grid_level,
        }
    return provenance


def runtime_provenance() -> dict:
    """Return reproducibility-relevant interpreter and distribution versions."""

    distributions = {}
    for name in (
        "pyscf-vscf",
        "numpy",
        "scipy",
        "pyscf",
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


def scientific_cache_metadata(
    molecule,
    cfg,
    scan: dict,
    *,
    backend_identity: str = "pyscf",
) -> dict:
    """Build schema-v3 metadata with causal identity and recorded provenance."""

    identity = {
        "molecule": molecule_electronic_identity(molecule),
        "electronic_structure": electronic_structure_identity(
            cfg, backend_identity=backend_identity
        ),
        "scan": _scan_identity(scan),
    }
    provenance = {
        "molecule": molecule_provenance(molecule),
        "electronic_structure": electronic_structure_provenance(
            cfg, backend_identity=backend_identity
        ),
        "scan": scan,
        "runtime": runtime_provenance(),
    }
    return {
        "grid_cache_version": CACHE_SCHEMA_VERSION,
        "identity": identity,
        "cache_identity_sha256": scientific_fingerprint(identity),
        "provenance": provenance,
        "provenance_sha256": scientific_fingerprint(provenance),
    }


def validate_scientific_cache_metadata(actual: dict, expected: dict) -> None:
    """Validate causal identity and warn on noncausal runtime drift."""

    version_actual = actual.get("grid_cache_version")
    if version_actual == 2:
        actual = migrate_cache_metadata_v2(actual)
    elif version_actual != CACHE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported grid cache schema {version_actual!r}; expected "
            f"{CACHE_SCHEMA_VERSION}. Schema-2 caches can be migrated automatically."
        )
    _validate_schema3_metadata_integrity(actual, "Grid cache")
    _validate_schema3_metadata_integrity(expected, "Expected grid cache")
    embedded = actual.get("identity")
    embedded_fingerprint = actual.get("cache_identity_sha256")
    if embedded_fingerprint != scientific_fingerprint(embedded):
        raise ValueError("Grid cache identity fingerprint is corrupt")
    expected_embedded = expected.get("identity")
    expected_fingerprint = expected.get("cache_identity_sha256")
    if expected_fingerprint != scientific_fingerprint(expected_embedded):
        raise ValueError("Expected grid cache identity fingerprint is corrupt")
    if embedded_fingerprint != expected_fingerprint:
        differences = identity_differences(embedded, expected_embedded)
        shown = differences[:_MAX_REPORTED_DIFFERENCES]
        hidden = len(differences) - len(shown)
        lines = [
            "Grid cache mismatch for cache_identity_sha256: the cached grid was computed "
            "for a different calculation than the one requested"
        ]
        lines.extend(f"  {line}" for line in shown)
        if hidden > 0:
            lines.append(f"  ... and {hidden} more difference(s)")
        raise ValueError("\n".join(lines))
    actual_runtime = actual.get("provenance", {}).get("runtime", {})
    expected_runtime = expected.get("provenance", {}).get("runtime", {})
    if actual_runtime != expected_runtime:
        warnings.warn(
            "Grid cache runtime differs from the current environment; causal scientific "
            "identity matches, so immutable reuse is allowed",
            RuntimeWarning,
            stacklevel=2,
        )


_MAX_REPORTED_DIFFERENCES = 12


def identity_differences(cached, requested, path: str = "") -> list[str]:
    """Describe every field where a cached identity differs from a requested one.

    Numeric lists are summarized by shape and range, with the first differing
    index and the largest absolute difference when the shapes agree.
    """

    if isinstance(cached, dict) and isinstance(requested, dict):
        differences = []
        for key in sorted(set(cached) | set(requested), key=str):
            child = f"{path}.{key}" if path else str(key)
            if key not in cached:
                differences.append(
                    f"{child}: absent from the cache, requested {_summarize(requested[key])}"
                )
            elif key not in requested:
                differences.append(
                    f"{child}: cache has {_summarize(cached[key])}, absent from the request"
                )
            else:
                differences.extend(identity_differences(cached[key], requested[key], child))
        return differences
    if cached == requested:
        return []
    label = path or "<root>"
    left = _numeric_array(cached)
    right = _numeric_array(requested)
    if left is not None and right is not None:
        if left.shape != right.shape:
            return [f"{label}: cache has {_summarize(cached)}, requested {_summarize(requested)}"]
        delta = np.abs(left - right)
        first = tuple(int(value) for value in np.argwhere(left != right)[0])
        return [
            f"{label}: first difference at index {list(first)} "
            f"(cache {float(left[first])!r}, requested {float(right[first])!r}); "
            f"max |difference| {float(np.max(delta)):.3g}"
        ]
    return [f"{label}: cache={_summarize(cached)}, requested={_summarize(requested)}"]


def _numeric_array(value) -> np.ndarray | None:
    if not isinstance(value, (list, tuple)) or not value:
        return None
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.dtype == object:
        return None
    return array


def _summarize(value) -> str:
    array = _numeric_array(value)
    if array is not None and array.size > 6:
        return (
            f"{array.size} values with shape {list(array.shape)} in "
            f"[{float(np.min(array)):.6g}, {float(np.max(array)):.6g}]"
        )
    return repr(value)


def _validate_schema3_metadata_integrity(metadata: dict, label: str) -> None:
    identity = metadata.get("identity")
    identity_fingerprint = metadata.get("cache_identity_sha256")
    if not isinstance(identity, dict) or identity_fingerprint != scientific_fingerprint(identity):
        raise ValueError(f"{label} identity fingerprint is corrupt")

    provenance = metadata.get("provenance")
    provenance_fingerprint = metadata.get("provenance_sha256")
    if not isinstance(provenance, dict):
        raise ValueError(f"{label} provenance is missing")
    if provenance_fingerprint != scientific_fingerprint(provenance):
        raise ValueError(f"{label} provenance fingerprint is corrupt")

    molecule = provenance.get("molecule", {})
    molecule_identity = {
        "symbols": [
            "H" if str(symbol).upper() == "D" else str(symbol)
            for symbol in molecule.get("symbols", [])
        ],
        "coordinates_A": molecule.get("coordinates_A"),
        "charge": molecule.get("charge", 0),
        "spin": molecule.get("spin", 0),
    }
    if molecule_identity != identity.get("molecule"):
        raise ValueError(f"{label} molecule provenance conflicts with cache identity")
    electronic = provenance.get("electronic_structure", {}).get("identity")
    if electronic != identity.get("electronic_structure"):
        raise ValueError(f"{label} electronic provenance conflicts with cache identity")
    if _scan_identity(provenance.get("scan", {})) != identity.get("scan"):
        raise ValueError(f"{label} scan provenance conflicts with cache identity")


def migrate_cache_metadata_v2(metadata: dict) -> dict:
    """Convert schema-2 metadata to schema 3 without changing cached arrays."""

    scientific = metadata.get("scientific")
    fingerprint = metadata.get("scientific_fingerprint_sha256")
    if not isinstance(scientific, dict) or fingerprint != scientific_fingerprint(scientific):
        raise ValueError("Schema-2 grid cache scientific metadata fingerprint is corrupt")

    molecule = scientific.get("molecule", {})
    electronic = scientific.get("electronic_structure", {})
    identity = {
        "molecule": {
            "symbols": [
                "H" if str(symbol).upper() == "D" else str(symbol)
                for symbol in molecule.get("symbols", [])
            ],
            "coordinates_A": molecule.get("coordinates_A"),
            "charge": molecule.get("charge", 0),
            "spin": molecule.get("spin", 0),
        },
        "electronic_structure": {
            key: electronic.get(key)
            for key in (
                "method",
                "basis",
                "use_density_fit",
                "auxbasis",
                "scf_conv_tol",
                "scf_max_cycle",
                "dft_grid_level",
                "effective_auxbasis",
                "backend",
            )
        },
        "scan": _scan_identity(scientific.get("scan", {})),
    }
    runtime = metadata.get("runtime", {})
    provenance = {
        "molecule": molecule,
        "electronic_structure": {
            "identity": identity["electronic_structure"],
            "software_versions": electronic.get("software_versions", {}),
        },
        "scan": scientific.get("scan", {}),
        "runtime": runtime,
        "migrated_from_schema": 2,
    }
    return {
        "grid_cache_version": CACHE_SCHEMA_VERSION,
        "identity": identity,
        "cache_identity_sha256": scientific_fingerprint(identity),
        "provenance": provenance,
        "provenance_sha256": scientific_fingerprint(provenance),
    }


def _scan_identity(scan: dict) -> dict:
    noncausal = {
        "keo",
        "reduced_masses_amu",
        "g12_inv_amu",
        "gmatrix_reference_geometry",
    }
    return {key: value for key, value in scan.items() if key not in noncausal}


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
    "electronic_structure_identity",
    "identity_differences",
    "load_grid_npz",
    "molecule_provenance",
    "molecule_electronic_identity",
    "migrate_cache_metadata_v2",
    "runtime_provenance",
    "scientific_cache_metadata",
    "scientific_fingerprint",
    "validate_scientific_cache_metadata",
]
