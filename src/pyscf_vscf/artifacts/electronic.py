"""Exact JSON serialization of electronic point requests and results.

Floats are written with Python's shortest round-trip representation, so a
deserialized request or result reproduces the original values bit for bit;
deserialization verifies that by recomputing the stored fingerprints.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .._identity import to_jsonable
from ..contracts import ElectronicPointRequest, ElectronicResult

REQUEST_SCHEMA = "pyscf-vscf-electronic-request"
RESULT_SCHEMA = "pyscf-vscf-electronic-result"
SCHEMA_VERSION = 1


def request_to_json(request: ElectronicPointRequest) -> dict[str, Any]:
    """Return a JSON-ready mapping that reproduces ``request`` exactly."""

    return {
        "schema": REQUEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "nuclear_charges": list(request.nuclear_charges),
        "coordinates_A": np.asarray(request.coordinates_A, dtype=float).tolist(),
        "charge": request.charge,
        "spin": request.spin,
        "electronic_state": request.electronic_state,
        "requested_properties": list(request.requested_properties),
        "field_au": None
        if request.field_au is None
        else np.asarray(request.field_au, dtype=float).tolist(),
        "field_origin_A": None
        if request.field_origin_A is None
        else np.asarray(request.field_origin_A, dtype=float).tolist(),
        "point_schema_version": request.schema_version,
    }


def request_from_json(payload: Mapping[str, Any]) -> ElectronicPointRequest:
    """Rebuild a request written by ``request_to_json``."""

    _require_schema(payload, REQUEST_SCHEMA)
    return ElectronicPointRequest(
        nuclear_charges=tuple(payload["nuclear_charges"]),
        coordinates_A=np.asarray(payload["coordinates_A"], dtype=float),
        charge=payload["charge"],
        spin=payload["spin"],
        electronic_state=payload["electronic_state"],
        requested_properties=tuple(payload["requested_properties"]),
        field_au=None if payload["field_au"] is None else np.asarray(payload["field_au"]),
        field_origin_A=None
        if payload["field_origin_A"] is None
        else np.asarray(payload["field_origin_A"]),
        schema_version=payload["point_schema_version"],
    )


def result_to_json(result: ElectronicResult) -> dict[str, Any]:
    """Return a JSON-ready mapping that reproduces ``result`` exactly."""

    return {
        "schema": RESULT_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "total_energy_Eh": result.total_energy_Eh,
        "dipole_au": None
        if result.dipole_au is None
        else np.asarray(result.dipole_au, dtype=float).tolist(),
        "dipole_unit": result.dipole_unit,
        "dipole_frame": result.dipole_frame,
        "converged": result.converged,
        "point_causal_fingerprint": result.point_causal_fingerprint,
        "provider_scientific_fingerprint": result.provider_scientific_fingerprint,
        "scientific_diagnostics": to_jsonable(result.scientific_diagnostics),
        "execution_diagnostics": to_jsonable(result.execution_diagnostics),
        "provenance": to_jsonable(result.provenance),
        "scientific_fingerprint": result.scientific_fingerprint(),
        "content_fingerprint": result.content_fingerprint(),
    }


def result_from_json(payload: Mapping[str, Any]) -> ElectronicResult:
    """Rebuild a result written by ``result_to_json`` and verify its fingerprints."""

    _require_schema(payload, RESULT_SCHEMA)
    result = ElectronicResult(
        total_energy_Eh=payload["total_energy_Eh"],
        dipole_au=None if payload["dipole_au"] is None else np.asarray(payload["dipole_au"]),
        dipole_unit=payload["dipole_unit"],
        dipole_frame=payload["dipole_frame"],
        converged=payload["converged"],
        point_causal_fingerprint=payload["point_causal_fingerprint"],
        provider_scientific_fingerprint=payload["provider_scientific_fingerprint"],
        scientific_diagnostics=payload["scientific_diagnostics"],
        execution_diagnostics=payload["execution_diagnostics"],
        provenance=payload["provenance"],
    )
    if result.scientific_fingerprint() != payload["scientific_fingerprint"]:
        raise ValueError("Stored electronic result fails its scientific fingerprint")
    if result.content_fingerprint() != payload["content_fingerprint"]:
        raise ValueError("Stored electronic result fails its content fingerprint")
    return result


def _require_schema(payload: Mapping[str, Any], schema: str) -> None:
    if payload.get("schema") != schema or payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"Expected {schema} schema version {SCHEMA_VERSION}, got "
            f"{payload.get('schema')!r} version {payload.get('schema_version')!r}"
        )


__all__ = ["request_from_json", "request_to_json", "result_from_json", "result_to_json"]
