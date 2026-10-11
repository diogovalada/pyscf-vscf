"""Scan artifacts: schema-v4 files and read-only access to v2/v3 grid caches.

A v4 file stores a :class:`~pyscf_vscf.scans.ScanResult`: absolute energies,
atomic-unit dipoles, axis definitions, point keys, the provider identity, the
sampling definition, array checksums, and a fingerprint of its complete
content. :func:`load_scan` verifies all of them.

Grid caches written by 0.1/0.2 (schema 2 and 3) hold energies relative to
the grid minimum and Debye dipoles. :func:`load_legacy_grid` returns them as a
:class:`LegacyRelativeScan` after the original checksum and identity checks;
it does not invent absolute energies or modern point identities.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .._artifacts import atomic_savez_compressed
from .._identity import (
    FrozenJSONMapping,
    canonical_json,
    immutable_json_mapping,
    payload_fingerprint,
)
from ..cache import (
    CACHE_SCHEMA_VERSION,
    _validate_schema3_metadata_integrity,
    array_sha256,
    migrate_cache_metadata_v2,
)
from ..io import load_grid_npz
from ..scans import ScanAxis, ScanResult

SCAN_ARTIFACT_SCHEMA = "pyscf-vscf-scan"
SCAN_ARTIFACT_SCHEMA_VERSION = 4


def dump_scan(path: Path | str, scan: ScanResult) -> None:
    """Write ``scan`` as a schema-v4 NPZ artifact (atomically)."""

    arrays = {
        "total_energies_Eh": np.asarray(scan.total_energies_Eh, dtype="<f8"),
        "dipoles_au": np.asarray(scan.dipoles_au, dtype="<f8"),
        "point_keys": np.asarray(scan.point_keys, dtype="<U64"),
    }
    for index, axis in enumerate(scan.axes):
        arrays[f"axis_{index}"] = np.asarray(axis.values, dtype="<f8")
    meta = {
        "schema": SCAN_ARTIFACT_SCHEMA,
        "schema_version": SCAN_ARTIFACT_SCHEMA_VERSION,
        "scan_kind": scan.scan_kind,
        "axes": [_axis_meta(axis) for axis in scan.axes],
        "energy_unit": "hartree",
        "energy_reference": "absolute",
        "dipole_unit": "atomic_unit",
        "dipole_frame": "input_cartesian",
        "provider_scientific_fingerprint": scan.provider_scientific_fingerprint,
        "provider_payload": dict(scan.provider_payload),
        "reference_geometry": dict(scan.reference_geometry),
        "sampling": dict(scan.sampling),
        "sampling_fingerprint": payload_fingerprint(dict(scan.sampling)),
        "provenance": dict(scan.provenance),
        "array_sha256": {name: array_sha256(value) for name, value in sorted(arrays.items())},
    }
    meta["content_fingerprint"] = payload_fingerprint(_content(meta))
    payload = {"meta_json": np.array(canonical_json(_jsonable(meta)))}
    payload.update(arrays)
    atomic_savez_compressed(path, payload)


def load_scan(path: Path | str) -> ScanResult:
    """Load and verify a schema-v4 scan artifact."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "meta_json" not in data.files:
            raise ValueError(f"{path} is not a pyscf-vscf scan artifact")
        meta = json.loads(str(data["meta_json"].tolist()))
        arrays = {name: data[name] for name in data.files if name != "meta_json"}
    if meta.get("schema") != SCAN_ARTIFACT_SCHEMA:
        if meta.get("grid_cache_version") in (2, CACHE_SCHEMA_VERSION):
            raise ValueError(
                f"{path} is a legacy (schema {meta['grid_cache_version']}) grid cache; "
                "read it with load_legacy_grid()"
            )
        raise ValueError(f"{path} is not a pyscf-vscf scan artifact")
    if meta.get("schema_version") != SCAN_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported scan artifact version {meta.get('schema_version')!r}")
    hashes = meta.get("array_sha256", {})
    if set(hashes) != set(arrays):
        raise ValueError(f"{path}: array manifest does not match the stored arrays")
    for name, expected in hashes.items():
        if array_sha256(arrays[name]) != expected:
            raise ValueError(f"{path}: checksum mismatch for array {name!r}")
    if payload_fingerprint(_content(meta)) != meta.get("content_fingerprint"):
        raise ValueError(f"{path}: content fingerprint mismatch")
    if payload_fingerprint(meta["sampling"]) != meta.get("sampling_fingerprint"):
        raise ValueError(f"{path}: sampling fingerprint mismatch")
    if payload_fingerprint(meta["provider_payload"]) != meta["provider_scientific_fingerprint"]:
        raise ValueError(f"{path}: provider payload does not match its fingerprint")
    axes = tuple(
        ScanAxis(
            label=axis["label"],
            coordinate=axis["coordinate"],
            values=arrays[f"axis_{index}"],
            unit=axis["unit"],
            bond=None if axis["bond_zero_based"] is None else tuple(axis["bond_zero_based"]),
        )
        for index, axis in enumerate(meta["axes"])
    )
    return ScanResult(
        scan_kind=meta["scan_kind"],
        axes=axes,
        total_energies_Eh=arrays["total_energies_Eh"],
        dipoles_au=arrays["dipoles_au"],
        point_keys=arrays["point_keys"],
        provider_scientific_fingerprint=meta["provider_scientific_fingerprint"],
        provider_payload=meta["provider_payload"],
        reference_geometry=meta["reference_geometry"],
        sampling=meta["sampling"],
        provenance=meta["provenance"],
    )


@dataclass(frozen=True)
class LegacyRelativeScan:
    """A 0.1/0.2 grid cache: relative energies (Hartree) and Debye dipoles.

    ``axes`` holds the grid values in angstrom, ``metadata`` the verified
    schema-3 metadata (schema-2 caches are migrated in memory). Absolute
    energies were never stored and are not available.
    """

    scan_kind: str
    axes: tuple[np.ndarray, ...]
    relative_energies_Eh: np.ndarray
    dipoles_debye: np.ndarray
    metadata: FrozenJSONMapping

    def as_legacy_tuple(self) -> tuple[np.ndarray, ...]:
        """Return ``(*axes, relative_energies_Eh, dipoles_debye)``."""

        return (*self.axes, self.relative_energies_Eh, self.dipoles_debye)


def load_legacy_grid(path: Path | str) -> LegacyRelativeScan:
    """Load a schema-2/3 grid cache after checksum and identity checks."""

    meta, arrays = load_grid_npz(Path(path))
    version = meta.get("grid_cache_version")
    if version == 2:
        meta = migrate_cache_metadata_v2(meta)
    elif version != CACHE_SCHEMA_VERSION:
        raise ValueError(f"{path} is not a legacy grid cache (schema {version!r})")
    _validate_schema3_metadata_integrity(meta, "Grid cache")
    from ..workflows.scans import _validate_grid_arrays_1d, _validate_grid_arrays_2d

    scan = meta["identity"]["scan"]
    kind = scan.get("kind")
    energies = np.asarray(arrays["E_Eh"], dtype=float)
    dipoles = np.asarray(arrays["MU_Debye"], dtype=float)
    axes: tuple[np.ndarray, ...]
    # The arrays must be the grid that the verified identity declares.
    if kind == "1d":
        axes = (np.asarray(arrays["R_A"], dtype=float),)
        _validate_grid_arrays_1d(axes[0], energies, dipoles, _declared_grid(scan, "grid_A"))
    elif kind == "2d":
        axes = (np.asarray(arrays["R1_A"], dtype=float), np.asarray(arrays["R2_A"], dtype=float))
        _validate_grid_arrays_2d(
            axes[0],
            axes[1],
            energies,
            dipoles,
            _declared_grid(scan, "grid1_A"),
            _declared_grid(scan, "grid2_A"),
        )
    else:
        raise ValueError(f"{path}: unknown legacy scan kind {kind!r}")
    return LegacyRelativeScan(
        scan_kind=f"legacy-{kind}",
        axes=axes,
        relative_energies_Eh=energies,
        dipoles_debye=dipoles,
        metadata=immutable_json_mapping(meta),
    )


def _declared_grid(scan: Mapping[str, Any], key: str) -> np.ndarray:
    if key not in scan:
        raise ValueError(f"Legacy grid cache identity lacks {key}")
    return np.asarray(scan[key], dtype=float)


def _axis_meta(axis: ScanAxis) -> dict[str, Any]:
    payload = axis.payload()
    payload.pop("values")
    return payload


def _content(meta: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in meta.items() if key != "content_fingerprint"}


def _jsonable(value: Any) -> Any:
    from .._identity import to_jsonable

    return to_jsonable(value)


__all__ = ["LegacyRelativeScan", "dump_scan", "load_legacy_grid", "load_scan"]
