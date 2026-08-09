"""N-mode subset enumeration and geometry-keyed electronic point planning."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ._arrays import immutable_array
from ._identity import float64_array_identity, payload_fingerprint
from .coordinates import CoordinateMap, coordinate_map_fingerprint
from .electronic import (
    ElectronicPointRequest,
    ElectronicProvider,
    provider_scientific_fingerprint,
)

ModeSubset = tuple[int, ...]
LineageKey = tuple[ModeSubset, tuple[int, ...]]
NMODE_POINT_PLAN_SCHEMA_VERSION = 1


def enumerate_mode_subsets(
    n_modes: int,
    *,
    max_rank: int = 2,
    selected_subsets: Sequence[Sequence[int]] = (),
) -> tuple[ModeSubset, ...]:
    """Enumerate canonical non-empty subsets up to rank three."""

    count = operator.index(n_modes)
    rank = operator.index(max_rank)
    if count < 1:
        raise ValueError("n_modes must be positive")
    if rank < 1 or rank > 3:
        raise ValueError("max_rank must lie between one and three")
    rank = min(rank, count)
    subsets = {
        combination
        for size in range(1, rank + 1)
        for combination in itertools.combinations(range(count), size)
    }
    for raw_subset in selected_subsets:
        subset = _normalize_subset(raw_subset, count)
        if len(subset) > 3:
            raise ValueError("N-mode point planning supports rank at most three")
        subsets.add(subset)
    normalized = tuple(sorted(subsets, key=lambda value: (len(value), value)))
    _require_subset_closure(normalized)
    return normalized


@dataclass(frozen=True)
class SamplingLineage:
    """Immutable map-specific lineage for one geometry-keyed request."""

    coordinate_map_fingerprint: str
    coordinate_values: np.ndarray
    subset: ModeSubset
    point_causal_fingerprint: str
    geometry_A: np.ndarray

    def __post_init__(self) -> None:
        map_id = _nonempty("coordinate_map_fingerprint", self.coordinate_map_fingerprint)
        point_id = _nonempty("point_causal_fingerprint", self.point_causal_fingerprint)
        values = _finite_array("coordinate_values", self.coordinate_values, ndim=1)
        geometry = _finite_array("geometry_A", self.geometry_A, ndim=2)
        if geometry.shape[1:] != (3,):
            raise ValueError("geometry_A must have shape (n_atoms, 3)")
        subset = tuple(operator.index(mode) for mode in self.subset)
        if tuple(sorted(set(subset))) != subset or any(mode < 0 for mode in subset):
            raise ValueError("subset must contain unique increasing non-negative mode indices")
        if any(mode >= values.size for mode in subset):
            raise ValueError("subset contains a mode outside coordinate_values")
        object.__setattr__(self, "coordinate_map_fingerprint", map_id)
        object.__setattr__(self, "coordinate_values", immutable_array(values, dtype="<f8"))
        object.__setattr__(self, "subset", subset)
        object.__setattr__(self, "point_causal_fingerprint", point_id)
        object.__setattr__(self, "geometry_A", immutable_array(geometry, dtype="<f8"))

    def content_payload(self) -> dict[str, object]:
        """Return the complete retained sampling record for integrity checks."""

        return {
            "schema": "pyscf-vscf-sampling-lineage",
            "schema_version": 1,
            "coordinate_map_fingerprint": self.coordinate_map_fingerprint,
            "coordinate_values": float64_array_identity(self.coordinate_values),
            "subset": list(self.subset),
            "point_causal_fingerprint": self.point_causal_fingerprint,
            "geometry_A": float64_array_identity(self.geometry_A),
        }

    def content_fingerprint(self) -> str:
        """Fingerprint every retained lineage field."""

        return payload_fingerprint(self.content_payload())


@dataclass(frozen=True)
class NModePointPlan:
    """Unique electronic requests plus separate tensor-cut sampling lineage."""

    coordinate_map: CoordinateMap
    coordinate_map_fingerprint: str
    provider_scientific_fingerprint: str
    node_axes: tuple[np.ndarray, ...]
    subsets: tuple[ModeSubset, ...]
    requests: Mapping[str, ElectronicPointRequest]
    cut_point_fingerprints: Mapping[ModeSubset, np.ndarray]
    sampling_lineage: Mapping[LineageKey, SamplingLineage]
    schema_version: int = NMODE_POINT_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if operator.index(self.schema_version) != NMODE_POINT_PLAN_SCHEMA_VERSION:
            raise ValueError("Unsupported n-mode point-plan schema")
        n_modes = len(self.coordinate_map.coordinate_ids)
        axes = tuple(
            _validated_axis(f"node_axes[{mode}]", axis) for mode, axis in enumerate(self.node_axes)
        )
        if len(axes) != n_modes:
            raise ValueError("node_axes must contain one axis per coordinate")
        reference_values = np.asarray(self.coordinate_map.reference_values, dtype="<f8")
        for mode, (axis, reference) in enumerate(zip(axes, reference_values)):
            if np.count_nonzero(axis == reference) != 1:
                raise ValueError(
                    f"node_axes[{mode}] must contain the exact common reference value once"
                )
        subsets = tuple(_normalize_subset(subset, n_modes) for subset in self.subsets)
        if tuple(sorted(set(subsets), key=lambda value: (len(value), value))) != subsets:
            raise ValueError("subsets must be unique and sorted by rank then index")
        _require_subset_closure(subsets)
        provider_id = _nonempty(
            "provider_scientific_fingerprint", self.provider_scientific_fingerprint
        )
        map_id = coordinate_map_fingerprint(self.coordinate_map)
        if self.coordinate_map_fingerprint != map_id:
            raise ValueError("coordinate_map_fingerprint does not match the coordinate map")

        requests: dict[str, ElectronicPointRequest] = {}
        for raw_point, request in self.requests.items():
            point = str(raw_point)
            if request.causal_fingerprint(provider_id) != point:
                raise ValueError("A planned request does not match its causal fingerprint")
            if request.field_au is not None:
                raise ValueError("N-mode point plans do not support electric-field requests")
            requests[point] = request

        cut_points: dict[ModeSubset, np.ndarray] = {}
        expected_lineage_keys: set[LineageKey] = set()
        used_points: set[str] = set()
        invariant: tuple[object, ...] | None = None
        for subset in ((), *subsets):
            if subset not in self.cut_point_fingerprints:
                raise ValueError(f"Missing point-fingerprint grid for subset {subset}")
            raw_grid = np.asarray(self.cut_point_fingerprints[subset])
            expected_shape = tuple(axes[mode].size for mode in subset)
            if raw_grid.shape != expected_shape:
                raise ValueError(
                    f"Point-fingerprint grid for {subset} must have shape {expected_shape}"
                )
            grid = np.asarray(raw_grid, dtype="U64")
            indices = np.ndindex(grid.shape) if grid.shape else [()]
            for local_index in indices:
                key = (subset, tuple(local_index))
                expected_lineage_keys.add(key)
                if key not in self.sampling_lineage:
                    raise ValueError(f"Missing sampling lineage for {subset}{local_index}")
                point = str(grid[local_index])
                if point not in requests:
                    raise ValueError(
                        f"Point grid {subset}{local_index} references an unknown point"
                    )
                used_points.add(point)
                request = requests[point]
                request_invariant = _request_invariant(request)
                invariant = request_invariant if invariant is None else invariant
                if request_invariant != invariant:
                    raise ValueError(
                        "Planned requests do not share one electronic-system identity"
                    )
                expected_values = reference_values.copy()
                for position, mode in enumerate(subset):
                    expected_values[mode] = axes[mode][local_index[position]]
                lineage = self.sampling_lineage[key]
                _validate_sampling_lineage(
                    lineage,
                    coordinate_map=self.coordinate_map,
                    coordinate_map_fingerprint=map_id,
                    expected_values=expected_values,
                    subset=subset,
                    request=request,
                    point_causal_fingerprint=point,
                )
            cut_points[subset] = immutable_array(grid)

        if set(self.cut_point_fingerprints) != {(), *subsets}:
            raise ValueError("cut_point_fingerprints contains an unexpected subset")
        if set(self.sampling_lineage) != expected_lineage_keys:
            raise ValueError("sampling_lineage contains an unexpected cut point")
        if set(requests) != used_points:
            raise ValueError("requests contains an unreferenced electronic point")

        object.__setattr__(self, "coordinate_map_fingerprint", map_id)
        object.__setattr__(self, "provider_scientific_fingerprint", provider_id)
        object.__setattr__(self, "node_axes", tuple(immutable_array(axis) for axis in axes))
        object.__setattr__(self, "subsets", subsets)
        object.__setattr__(self, "requests", MappingProxyType(requests))
        object.__setattr__(self, "cut_point_fingerprints", MappingProxyType(cut_points))
        object.__setattr__(
            self,
            "sampling_lineage",
            MappingProxyType(dict(self.sampling_lineage)),
        )
        object.__setattr__(self, "schema_version", NMODE_POINT_PLAN_SCHEMA_VERSION)

    @property
    def reference_point_fingerprint(self) -> str:
        """Return the common-reference electronic point identity."""

        return str(self.cut_point_fingerprints[()].item())


def plan_nmode_points(
    coordinate_map: CoordinateMap,
    node_axes: Sequence[np.ndarray],
    provider: ElectronicProvider,
    *,
    nuclear_charges: Sequence[int],
    subsets: Sequence[Sequence[int]] | None = None,
    max_rank: int = 2,
    selected_subsets: Sequence[Sequence[int]] = (),
    charge: int = 0,
    spin: int = 0,
    electronic_state: str = "ground",
    requested_properties: tuple[str, ...] = ("energy", "dipole"),
) -> NModePointPlan:
    """Plan unique anchored requests without evaluating electronic structure."""

    n_modes = len(coordinate_map.coordinate_ids)
    properties = tuple(sorted({str(value).strip().lower() for value in requested_properties}))
    if properties != ("dipole", "energy"):
        raise ValueError("N-mode PES/DMS planning requires both energy and dipole properties")
    axes = tuple(
        _validated_axis(f"node_axes[{mode}]", axis) for mode, axis in enumerate(node_axes)
    )
    if len(axes) != n_modes:
        raise ValueError("node_axes must contain one axis per coordinate")
    reference = np.asarray(coordinate_map.reference_values, dtype="<f8")
    for mode, (axis, reference_value) in enumerate(zip(axes, reference)):
        if np.count_nonzero(axis == reference_value) != 1:
            raise ValueError(
                f"node_axes[{mode}] must contain the exact common reference value once"
            )
    if subsets is None:
        normalized_subsets = enumerate_mode_subsets(
            n_modes,
            max_rank=max_rank,
            selected_subsets=selected_subsets,
        )
    else:
        normalized_subsets = tuple(
            sorted(
                {_normalize_subset(subset, n_modes) for subset in subsets},
                key=lambda value: (len(value), value),
            )
        )
        _require_subset_closure(normalized_subsets)

    provider_id = provider_scientific_fingerprint(provider)
    map_id = coordinate_map_fingerprint(coordinate_map)
    requests: dict[str, ElectronicPointRequest] = {}
    cut_points: dict[ModeSubset, np.ndarray] = {}
    lineage: dict[LineageKey, SamplingLineage] = {}
    for subset in ((), *normalized_subsets):
        shape = tuple(axes[mode].size for mode in subset)
        points = np.empty(shape, dtype="U64")
        indices = np.ndindex(shape) if shape else [()]
        for local_index in indices:
            values = reference.copy()
            for position, mode in enumerate(subset):
                values[mode] = axes[mode][local_index[position]]
            geometry = np.ascontiguousarray(coordinate_map.to_cartesian(values), dtype="<f8")
            request = ElectronicPointRequest(
                nuclear_charges=tuple(nuclear_charges),
                coordinates_A=geometry,
                charge=charge,
                spin=spin,
                electronic_state=electronic_state,
                requested_properties=properties,
            )
            point = request.causal_fingerprint(provider_id)
            requests.setdefault(point, request)
            index = tuple(local_index)
            points[index] = point
            lineage[(subset, index)] = SamplingLineage(
                coordinate_map_fingerprint=map_id,
                coordinate_values=values,
                subset=subset,
                point_causal_fingerprint=point,
                geometry_A=request.coordinates_A,
            )
        cut_points[subset] = points
    return NModePointPlan(
        coordinate_map=coordinate_map,
        coordinate_map_fingerprint=map_id,
        provider_scientific_fingerprint=provider_id,
        node_axes=axes,
        subsets=normalized_subsets,
        requests=requests,
        cut_point_fingerprints=cut_points,
        sampling_lineage=lineage,
    )


def _validate_sampling_lineage(
    lineage: SamplingLineage,
    *,
    coordinate_map: CoordinateMap,
    coordinate_map_fingerprint: str,
    expected_values: np.ndarray,
    subset: ModeSubset,
    request: ElectronicPointRequest,
    point_causal_fingerprint: str,
) -> None:
    if lineage.coordinate_map_fingerprint != coordinate_map_fingerprint:
        raise ValueError("Sampling lineage does not match the coordinate map")
    if lineage.subset != subset or not np.array_equal(lineage.coordinate_values, expected_values):
        raise ValueError("Sampling lineage does not match its complete coordinate vector")
    if lineage.point_causal_fingerprint != point_causal_fingerprint:
        raise ValueError("Sampling lineage does not match its electronic point")
    if not np.array_equal(lineage.geometry_A, request.coordinates_A):
        raise ValueError("Sampling lineage does not retain the exact request geometry")
    reconstructed = coordinate_map.to_cartesian(lineage.coordinate_values)
    if not np.allclose(reconstructed, request.coordinates_A, rtol=0.0, atol=1e-12):
        raise ValueError("Sampling lineage geometry does not match map.to_cartesian(values)")


def _request_invariant(request: ElectronicPointRequest) -> tuple[object, ...]:
    return (
        request.nuclear_charges,
        request.charge,
        request.spin,
        request.electronic_state,
        request.requested_properties,
        request.field_au is None,
        request.field_origin_A is None,
    )


def _normalize_subset(raw_subset: Sequence[int], n_modes: int) -> ModeSubset:
    subset = tuple(operator.index(mode) for mode in raw_subset)
    if not subset or tuple(sorted(set(subset))) != subset:
        raise ValueError("Mode subsets must contain unique increasing indices")
    if subset[0] < 0 or subset[-1] >= n_modes:
        raise IndexError(f"Mode subset {subset} is out of range for {n_modes} modes")
    return subset


def _require_subset_closure(subsets: Sequence[ModeSubset]) -> None:
    available = set(subsets)
    for subset in subsets:
        for size in range(1, len(subset)):
            for lower in itertools.combinations(subset, size):
                if lower not in available:
                    raise ValueError(f"Subset {subset} requires lower-rank subset {lower}")


def _validated_axis(name: str, values: np.ndarray) -> np.ndarray:
    axis = _finite_array(name, values, ndim=1)
    if axis.size < 1 or np.any(np.diff(axis) <= 0.0):
        raise ValueError(f"{name} must be non-empty and strictly increasing")
    return axis


def _finite_array(name: str, values: object, *, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype="<f8")
    if array.ndim != ndim or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return np.ascontiguousarray(array)


def _nonempty(name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


__all__ = [
    "LineageKey",
    "ModeSubset",
    "NMODE_POINT_PLAN_SCHEMA_VERSION",
    "NModePointPlan",
    "SamplingLineage",
    "enumerate_mode_subsets",
    "plan_nmode_points",
]
