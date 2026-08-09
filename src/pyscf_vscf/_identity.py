"""Backend-neutral canonical identity and immutable JSON helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FrozenJSONMapping(Mapping[str, Any]):
    """Deeply immutable, pickleable mapping of JSON-compatible values."""

    _items: tuple[tuple[str, Any], ...]

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> FrozenJSONMapping:
        if not isinstance(values, Mapping):
            raise TypeError("Expected a mapping")
        items = []
        for key, value in values.items():
            if not isinstance(key, str):
                raise TypeError("Identity mapping keys must be strings")
            items.append((key, _freeze_json(value)))
        return cls(tuple(sorted(items)))

    def __getitem__(self, key: str) -> Any:
        for candidate, value in self._items:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> Iterator[str]:
        return (key for key, _ in self._items)

    def __len__(self) -> int:
        return len(self._items)


def immutable_json_mapping(values: Mapping[str, Any]) -> FrozenJSONMapping:
    """Return a deeply immutable copy of a JSON-compatible mapping."""

    return FrozenJSONMapping.from_mapping(values)


def to_jsonable(value: object) -> object:
    """Recursively convert immutable JSON containers into ordinary containers."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError("Identity payloads cannot contain NaN or infinity")
        return parsed
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError("Identity payloads cannot contain NaN or infinity")
        return parsed
    if isinstance(value, np.ndarray):
        raise TypeError("NumPy arrays require an explicit array identity")
    if isinstance(value, Mapping):
        converted = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Identity payload keys must be strings")
            converted[key] = to_jsonable(item)
        return converted
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    raise TypeError(f"Unsupported identity payload value {type(value).__name__}")


def canonical_json(value: object) -> str:
    """Return deterministic JSON for a recursively normalized payload."""

    return json.dumps(
        to_jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def payload_fingerprint(payload: Mapping[str, object]) -> str:
    """Return the SHA-256 identity of a canonical JSON payload."""

    return hashlib.sha256(canonical_json(dict(payload)).encode("utf-8")).hexdigest()


def array_identity(values: object, *, dtype: object | None = None) -> dict[str, object]:
    """Describe exact C-order array content with normalized byte order."""

    array = np.asarray(values, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError("Object arrays cannot be fingerprinted")
    normalized_dtype = array.dtype.newbyteorder("<")
    normalized = np.ascontiguousarray(array.astype(normalized_dtype, copy=False))
    if np.issubdtype(normalized.dtype, np.inexact) and not np.all(np.isfinite(normalized)):
        raise ValueError("Identity arrays cannot contain NaN or infinity")
    return {
        "dtype": normalized_dtype.str,
        "shape": list(normalized.shape),
        "sha256": hashlib.sha256(normalized.tobytes(order="C")).hexdigest(),
    }


def float64_array_identity(values: object) -> dict[str, object]:
    """Describe exact canonical little-endian float64 array content."""

    return array_identity(values, dtype="<f8")


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        parsed = float(value)
        if not np.isfinite(parsed):
            raise ValueError("Immutable JSON values cannot contain NaN or infinity")
        return parsed
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        raise TypeError("NumPy metadata values require explicit conversion")
    if isinstance(value, Mapping):
        return FrozenJSONMapping.from_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise TypeError(f"Unsupported immutable JSON value {type(value).__name__}")


__all__ = [
    "FrozenJSONMapping",
    "array_identity",
    "canonical_json",
    "float64_array_identity",
    "immutable_json_mapping",
    "payload_fingerprint",
    "to_jsonable",
]
