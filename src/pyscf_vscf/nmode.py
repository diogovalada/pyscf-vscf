"""N-mode subset enumeration and geometry-keyed electronic point planning."""

from __future__ import annotations

import itertools
import json
import operator
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ._arrays import immutable_array
from ._artifacts import atomic_savez_compressed
from ._identity import (
    canonical_json,
    float64_array_identity,
    immutable_json_mapping,
    payload_fingerprint,
    to_jsonable,
)
from .coordinates import (
    CoordinateMap,
    coordinate_map_fingerprint,
    coordinate_map_from_payload,
)
from .electronic import (
    ElectronicPointRequest,
    ElectronicProvider,
    ElectronicResult,
    provider_scientific_fingerprint,
)

ModeSubset = tuple[int, ...]
LineageKey = tuple[ModeSubset, tuple[int, ...]]
NMODE_POINT_PLAN_SCHEMA_VERSION = 1
NMODE_SURFACE_SCHEMA_VERSION = 1


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
        explicit_subsets = {_normalize_subset(subset, n_modes) for subset in subsets}
        if any(len(subset) > 3 for subset in explicit_subsets):
            raise ValueError("N-mode point planning supports rank at most three")
        normalized_subsets = tuple(sorted(explicit_subsets, key=lambda value: (len(value), value)))
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


@dataclass(frozen=True)
class NModeCutSamples:
    """Absolute electronic values sampled on one anchored coordinate cut."""

    subset: ModeSubset
    axes: tuple[np.ndarray, ...]
    energies_Eh: np.ndarray
    dipoles_input_au: np.ndarray
    dipoles_body_au: np.ndarray
    point_causal_fingerprints: np.ndarray

    def __post_init__(self) -> None:
        subset = tuple(operator.index(mode) for mode in self.subset)
        if not subset or tuple(sorted(set(subset))) != subset:
            raise ValueError("subset must contain unique increasing mode indices")
        axes = tuple(
            _validated_axis(f"axes[{position}]", axis) for position, axis in enumerate(self.axes)
        )
        if len(axes) != len(subset):
            raise ValueError("axes must contain one node array per subset mode")
        shape = tuple(axis.size for axis in axes)
        energies = _validated_shape("energies_Eh", self.energies_Eh, shape)
        dipoles_input = _validated_shape("dipoles_input_au", self.dipoles_input_au, (*shape, 3))
        dipoles_body = _validated_shape("dipoles_body_au", self.dipoles_body_au, (*shape, 3))
        points = np.asarray(self.point_causal_fingerprints, dtype="U64")
        if points.shape != shape or any(not str(value) for value in points.flat):
            raise ValueError("point fingerprints must align with the cut tensor")
        object.__setattr__(self, "subset", subset)
        object.__setattr__(self, "axes", tuple(immutable_array(axis) for axis in axes))
        object.__setattr__(self, "energies_Eh", immutable_array(energies))
        object.__setattr__(self, "dipoles_input_au", immutable_array(dipoles_input))
        object.__setattr__(self, "dipoles_body_au", immutable_array(dipoles_body))
        object.__setattr__(
            self,
            "point_causal_fingerprints",
            immutable_array(points),
        )


@dataclass(frozen=True)
class NModeSampleSet:
    """Absolute anchored samples with raw and fixed-body-frame vector dipoles."""

    coordinate_map: CoordinateMap
    coordinate_map_fingerprint: str
    provider_scientific_fingerprint: str
    reference_energy_Eh: float
    reference_dipole_input_au: np.ndarray
    reference_dipole_body_au: np.ndarray
    reference_point_causal_fingerprint: str
    cuts: Mapping[ModeSubset, NModeCutSamples]
    source_lineage: Mapping[str, Any]
    annotations: Mapping[str, Any] = field(default_factory=dict)
    reference_tolerance_Eh: float = 1e-12
    reference_tolerance_dipole_au: float = 1e-12

    def __post_init__(self) -> None:
        map_id = coordinate_map_fingerprint(self.coordinate_map)
        if self.coordinate_map_fingerprint != map_id:
            raise ValueError("coordinate_map_fingerprint does not match the coordinate map")
        provider_id = _nonempty(
            "provider_scientific_fingerprint", self.provider_scientific_fingerprint
        )
        energy = float(self.reference_energy_Eh)
        if not np.isfinite(energy):
            raise ValueError("reference_energy_Eh must be finite")
        dipole_input = _validated_shape(
            "reference_dipole_input_au", self.reference_dipole_input_au, (3,)
        )
        dipole_body = _validated_shape(
            "reference_dipole_body_au", self.reference_dipole_body_au, (3,)
        )
        expected_body = self.coordinate_map.vector_to_body(dipole_input)
        if not np.array_equal(expected_body, dipole_body):
            raise ValueError("Reference body-frame dipole does not match the raw input dipole")
        point = _nonempty(
            "reference_point_causal_fingerprint",
            self.reference_point_causal_fingerprint,
        )
        energy_tolerance = _nonnegative_finite(
            "reference_tolerance_Eh", self.reference_tolerance_Eh
        )
        dipole_tolerance = _nonnegative_finite(
            "reference_tolerance_dipole_au", self.reference_tolerance_dipole_au
        )
        cuts = dict(self.cuts)
        if not cuts:
            raise ValueError("NModeSampleSet requires at least one cut")
        subsets = tuple(sorted(cuts, key=lambda value: (len(value), value)))
        if set(subsets) != set(cuts):
            raise ValueError("Duplicate n-mode cut subsets")
        _require_subset_closure(subsets)
        reference = np.asarray(self.coordinate_map.reference_values, dtype=float)
        canonical_axes: list[np.ndarray | None] = [None] * reference.size
        referenced_points: set[str] = set()
        for subset in subsets:
            cut = cuts[subset]
            if cut.subset != subset or subset[-1] >= reference.size:
                raise ValueError("Cut key does not match a valid cut subset")
            for position, mode in enumerate(subset):
                axis = cut.axes[position]
                if canonical_axes[mode] is None:
                    canonical_axes[mode] = axis
                elif not np.array_equal(canonical_axes[mode], axis):
                    raise ValueError(f"Cuts do not share one canonical axis for mode {mode}")
            reference_index = tuple(
                _reference_index(cut.axes[position], reference[mode], mode)
                for position, mode in enumerate(subset)
            )
            if str(cut.point_causal_fingerprints[reference_index]) != point:
                raise ValueError(f"Cut {subset} does not reuse the common reference point")
            if abs(float(cut.energies_Eh[reference_index]) - energy) > energy_tolerance:
                raise ValueError(f"Cut {subset} does not share the reference energy")
            if (
                np.max(np.abs(cut.dipoles_input_au[reference_index] - dipole_input))
                > dipole_tolerance
                or np.max(np.abs(cut.dipoles_body_au[reference_index] - dipole_body))
                > dipole_tolerance
            ):
                raise ValueError(f"Cut {subset} does not share the reference dipole")
            transformed = self.coordinate_map.vector_to_body(cut.dipoles_input_au)
            if not np.array_equal(transformed, cut.dipoles_body_au):
                raise ValueError(f"Cut {subset} body dipoles do not match raw input dipoles")
            referenced_points.update(str(value) for value in cut.point_causal_fingerprints.flat)

        if any(axis is None for axis in canonical_axes):
            raise ValueError("Every coordinate must appear in at least one cut")
        for subset in subsets:
            cut = cuts[subset]
            for lower_subset in _proper_nonempty_subsets(subset):
                lower = cuts[lower_subset]
                selector: list[int | slice] = []
                for position, mode in enumerate(subset):
                    selector.append(
                        slice(None)
                        if mode in lower_subset
                        else _reference_index(cut.axes[position], reference[mode], mode)
                    )
                selection = tuple(selector)
                if not np.array_equal(
                    cut.point_causal_fingerprints[selection],
                    lower.point_causal_fingerprints,
                ):
                    raise ValueError("Cut intersections do not reuse electronic point identities")
                for name in (
                    "energies_Eh",
                    "dipoles_input_au",
                    "dipoles_body_au",
                ):
                    if not np.array_equal(getattr(cut, name)[selection], getattr(lower, name)):
                        raise ValueError(f"Cut intersections disagree for {name}")

        source = immutable_json_mapping(self.source_lineage)
        source_points = set(str(value) for value in source.get("point_causal_fingerprints", ()))
        if source_points != referenced_points:
            raise ValueError("Source lineage does not exactly cover sampled point identities")
        if source.get("provider_scientific_fingerprint") != provider_id:
            raise ValueError("Source lineage provider does not match the sample set")
        object.__setattr__(self, "coordinate_map_fingerprint", map_id)
        object.__setattr__(self, "provider_scientific_fingerprint", provider_id)
        object.__setattr__(self, "reference_energy_Eh", energy)
        object.__setattr__(self, "reference_dipole_input_au", immutable_array(dipole_input))
        object.__setattr__(self, "reference_dipole_body_au", immutable_array(dipole_body))
        object.__setattr__(self, "reference_point_causal_fingerprint", point)
        object.__setattr__(
            self,
            "cuts",
            MappingProxyType({subset: cuts[subset] for subset in subsets}),
        )
        object.__setattr__(self, "source_lineage", source)
        object.__setattr__(self, "annotations", immutable_json_mapping(self.annotations))
        object.__setattr__(self, "reference_tolerance_Eh", energy_tolerance)
        object.__setattr__(self, "reference_tolerance_dipole_au", dipole_tolerance)

    @property
    def coordinate_ids(self) -> tuple[str, ...]:
        return tuple(self.coordinate_map.coordinate_ids)

    @property
    def coordinate_units(self) -> tuple[str, ...]:
        return tuple(self.coordinate_map.units)

    @property
    def reference_values(self) -> np.ndarray:
        return self.coordinate_map.reference_values


def assemble_nmode_samples(
    plan: NModePointPlan,
    results: Mapping[str, ElectronicResult],
    *,
    annotations: Mapping[str, Any] | None = None,
) -> NModeSampleSet:
    """Assemble absolute PES and raw/body-frame DMS cuts from planned results."""

    if set(results) != set(plan.requests):
        missing = sorted(set(plan.requests) - set(results))
        extra = sorted(set(results) - set(plan.requests))
        raise ValueError(
            f"Electronic results do not exactly match the plan: missing={missing}, extra={extra}"
        )
    energies_by_cut: dict[ModeSubset, np.ndarray] = {}
    input_dipoles_by_cut: dict[ModeSubset, np.ndarray] = {}
    body_dipoles_by_cut: dict[ModeSubset, np.ndarray] = {}
    result_ids: dict[str, str] = {}
    for subset, point_grid in plan.cut_point_fingerprints.items():
        energies = np.empty(point_grid.shape, dtype=float)
        dipoles_input = np.empty((*point_grid.shape, 3), dtype=float)
        indices = np.ndindex(point_grid.shape) if point_grid.shape else [()]
        for index in indices:
            point = str(point_grid[index])
            result = results[point]
            if result.point_causal_fingerprint != point:
                raise ValueError("Electronic result does not match its planned point")
            if result.provider_scientific_fingerprint != plan.provider_scientific_fingerprint:
                raise ValueError("Electronic result provider does not match the n-mode plan")
            if not result.converged or result.dipole_au is None:
                raise ValueError(
                    "N-mode samples require converged energy and vector dipole results"
                )
            if result.dipole_unit != "atomic_unit" or result.dipole_frame != "input_cartesian":
                raise ValueError("Electronic dipoles must use atomic units in the input frame")
            energies[index] = result.total_energy_Eh
            dipoles_input[index] = result.dipole_au
            result_ids[point] = result.scientific_fingerprint()
        energies_by_cut[subset] = energies
        input_dipoles_by_cut[subset] = dipoles_input
        body_dipoles_by_cut[subset] = plan.coordinate_map.vector_to_body(dipoles_input)

    reference_energy = float(energies_by_cut[()].item())
    reference_input = input_dipoles_by_cut[()].reshape(3)
    reference_body = body_dipoles_by_cut[()].reshape(3)
    cuts = {
        subset: NModeCutSamples(
            subset=subset,
            axes=tuple(plan.node_axes[mode] for mode in subset),
            energies_Eh=energies_by_cut[subset],
            dipoles_input_au=input_dipoles_by_cut[subset],
            dipoles_body_au=body_dipoles_by_cut[subset],
            point_causal_fingerprints=plan.cut_point_fingerprints[subset],
        )
        for subset in plan.subsets
    }
    source_lineage = {
        "schema": "pyscf-vscf-electronic-source-lineage",
        "schema_version": 1,
        "provider_scientific_fingerprint": plan.provider_scientific_fingerprint,
        "point_causal_fingerprints": sorted(plan.requests),
        "result_scientific_fingerprints": {
            point: result_ids[point] for point in sorted(result_ids)
        },
        "sampling_lineage_fingerprints": sorted(
            lineage.content_fingerprint() for lineage in plan.sampling_lineage.values()
        ),
    }
    return NModeSampleSet(
        coordinate_map=plan.coordinate_map,
        coordinate_map_fingerprint=plan.coordinate_map_fingerprint,
        provider_scientific_fingerprint=plan.provider_scientific_fingerprint,
        reference_energy_Eh=reference_energy,
        reference_dipole_input_au=reference_input,
        reference_dipole_body_au=reference_body,
        reference_point_causal_fingerprint=plan.reference_point_fingerprint,
        cuts=cuts,
        source_lineage=source_lineage,
        annotations=dict(annotations or {}),
    )


@dataclass(frozen=True)
class HeldOutCutSamples:
    """Independent absolute values used only for interpolation diagnostics."""

    subset: ModeSubset
    points: np.ndarray
    energies_Eh: np.ndarray
    dipoles_body_au: np.ndarray
    point_causal_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        subset = tuple(operator.index(mode) for mode in self.subset)
        if not subset or tuple(sorted(set(subset))) != subset:
            raise ValueError("held-out subset must contain unique increasing indices")
        points = _finite_array("points", self.points, ndim=2)
        if points.shape[1] != len(subset) or points.shape[0] == 0:
            raise ValueError("held-out points must have shape (n_points, subset_rank)")
        energies = _validated_shape("energies_Eh", self.energies_Eh, (points.shape[0],))
        dipoles = _validated_shape("dipoles_body_au", self.dipoles_body_au, (points.shape[0], 3))
        point_ids = tuple(str(value) for value in self.point_causal_fingerprints)
        if point_ids and (
            len(point_ids) != points.shape[0] or any(not value for value in point_ids)
        ):
            raise ValueError("held-out point identities must align with held-out points")
        object.__setattr__(self, "subset", subset)
        object.__setattr__(self, "points", immutable_array(points))
        object.__setattr__(self, "energies_Eh", immutable_array(energies))
        object.__setattr__(self, "dipoles_body_au", immutable_array(dipoles))
        object.__setattr__(self, "point_causal_fingerprints", point_ids)


@dataclass(frozen=True)
class FitDiagnostics:
    """Deterministic interpolation residual summaries outside surface identity."""

    method: str
    n_training_points: int
    training_max_abs_error: tuple[float, ...]
    n_held_out_points: int = 0
    held_out_rmse: tuple[float, ...] | None = None
    held_out_max_abs_error: tuple[float, ...] | None = None
    held_out_point_causal_fingerprints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        method = _validated_interpolation_method(self.method)
        training_count = operator.index(self.n_training_points)
        held_count = operator.index(self.n_held_out_points)
        training_max = tuple(float(value) for value in self.training_max_abs_error)
        held_rmse = (
            None
            if self.held_out_rmse is None
            else tuple(float(value) for value in self.held_out_rmse)
        )
        held_max = (
            None
            if self.held_out_max_abs_error is None
            else tuple(float(value) for value in self.held_out_max_abs_error)
        )
        held_points = tuple(str(value) for value in self.held_out_point_causal_fingerprints)
        if training_count < 1 or held_count < 0 or not training_max:
            raise ValueError("Fit diagnostic counts and training errors are invalid")
        if any(not np.isfinite(value) or value < 0.0 for value in training_max):
            raise ValueError("Training errors must be finite and non-negative")
        if held_count == 0:
            if held_rmse is not None or held_max is not None or held_points:
                raise ValueError("Held-out diagnostics require held-out points")
        elif (
            held_rmse is None
            or held_max is None
            or len(held_rmse) != len(training_max)
            or len(held_max) != len(training_max)
            or (held_points and len(held_points) != held_count)
        ):
            raise ValueError("Held-out diagnostics do not match output components")
        if any(
            not np.isfinite(value) or value < 0.0
            for values in (held_rmse or (), held_max or ())
            for value in values
        ):
            raise ValueError("Held-out errors must be finite and non-negative")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "n_training_points", training_count)
        object.__setattr__(self, "training_max_abs_error", training_max)
        object.__setattr__(self, "n_held_out_points", held_count)
        object.__setattr__(self, "held_out_rmse", held_rmse)
        object.__setattr__(self, "held_out_max_abs_error", held_max)
        object.__setattr__(self, "held_out_point_causal_fingerprints", held_points)


@dataclass(frozen=True)
class TensorProductSurface:
    """Immutable tensor-grid interpolant with bounds errors enabled."""

    axes: tuple[np.ndarray, ...]
    node_values: np.ndarray
    method: str
    diagnostics: FitDiagnostics
    _interpolator: Any = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        axes = tuple(
            _validated_axis(f"axes[{position}]", axis) for position, axis in enumerate(self.axes)
        )
        if not axes:
            raise ValueError("TensorProductSurface requires at least one axis")
        method = _validated_interpolation_method(self.method)
        minimum = 2 if method == "linear" else 4
        if any(axis.size < minimum for axis in axes):
            raise ValueError(f"{method} interpolation requires at least {minimum} nodes per axis")
        values = np.asarray(self.node_values, dtype="<f8")
        if not np.all(np.isfinite(values)):
            raise ValueError("node_values must be finite")
        expected_prefix = tuple(axis.size for axis in axes)
        if values.shape[: len(axes)] != expected_prefix:
            raise ValueError(f"node_values must start with tensor shape {expected_prefix}")
        if values.ndim not in {len(axes), len(axes) + 1}:
            raise ValueError("TensorProductSurface supports scalar or one trailing vector axis")
        if values.ndim == len(axes) + 1 and values.shape[-1] != 3:
            raise ValueError("Vector tensor surfaces must have three components")
        if self.diagnostics.method != method:
            raise ValueError("Fit diagnostics method does not match the surface")
        if self.diagnostics.n_training_points != int(np.prod(expected_prefix)):
            raise ValueError("Fit diagnostics training count does not match the tensor grid")
        component_count = 1 if values.ndim == len(axes) else values.shape[-1]
        if len(self.diagnostics.training_max_abs_error) != component_count:
            raise ValueError("Fit diagnostics do not match surface output components")
        backend_method = "cubic_legacy" if method == "cubic" else method
        interpolator = RegularGridInterpolator(
            axes,
            values,
            method=backend_method,
            bounds_error=True,
        )
        training_prediction = np.asarray(interpolator(_mesh_points(axes)), dtype=float).reshape(
            values.shape
        )
        actual_training_max = _component_max_abs(
            training_prediction - values,
            len(axes),
        )
        if actual_training_max != self.diagnostics.training_max_abs_error:
            raise ValueError("Training diagnostics do not match the interpolation representation")
        object.__setattr__(self, "axes", tuple(immutable_array(axis) for axis in axes))
        object.__setattr__(self, "node_values", immutable_array(values))
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "_interpolator", interpolator)

    @property
    def rank(self) -> int:
        return len(self.axes)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate points whose final dimension follows this surface's mode order."""

        values = np.asarray(points, dtype=float)
        if values.shape == (self.rank,):
            if not np.all(np.isfinite(values)):
                raise ValueError("Surface evaluation points must be finite")
            evaluated = np.asarray(self._interpolator(values), dtype=float)
            return evaluated.reshape(self.node_values.shape[self.rank :])
        if values.ndim < 2 or values.shape[-1] != self.rank or not np.all(np.isfinite(values)):
            raise ValueError(f"Surface points must be finite with final dimension {self.rank}")
        leading_shape = values.shape[:-1]
        evaluated = np.asarray(self._interpolator(values.reshape((-1, self.rank))), dtype=float)
        return evaluated.reshape((*leading_shape, *self.node_values.shape[self.rank :]))

    def numerical_payload(self) -> dict[str, object]:
        """Return only the interpolation function definition."""

        return {
            "representation": "scipy-regular-grid-interpolator",
            "method": self.method,
            "backend_method": "cubic_legacy" if self.method == "cubic" else self.method,
            "bounds_error": True,
            "axes": [float64_array_identity(axis) for axis in self.axes],
            "node_values": float64_array_identity(self.node_values),
        }

    def artifact_payload(self) -> dict[str, object]:
        """Return numerical data identities and diagnostic content."""

        return {**self.numerical_payload(), "diagnostics": asdict(self.diagnostics)}


@dataclass(frozen=True)
class NModeSurfaceModel:
    """Anchored n-mode PES and signed fixed-body-frame vector-DMS increments."""

    coordinate_ids: tuple[str, ...]
    coordinate_units: tuple[str, ...]
    coordinate_map_payload: Mapping[str, Any]
    coordinate_map_fingerprint: str
    reference_values: np.ndarray
    reference_energy_Eh: float
    reference_dipole_body_au: np.ndarray
    energy_increments: Mapping[ModeSubset, TensorProductSurface]
    dipole_increments: Mapping[ModeSubset, TensorProductSurface]
    source_lineage: Mapping[str, Any]
    annotations: Mapping[str, Any] = field(default_factory=dict)
    anchor_tolerance_Eh: float = 1e-12
    anchor_tolerance_dipole_au: float = 1e-12
    schema_version: int = NMODE_SURFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        ids = tuple(str(value).strip() for value in self.coordinate_ids)
        units = tuple(str(value).strip().lower() for value in self.coordinate_units)
        if not ids or len(ids) != len(units) or len(set(ids)) != len(ids):
            raise ValueError("Coordinate IDs and units must be non-empty, unique, and aligned")
        map_payload = immutable_json_mapping(self.coordinate_map_payload)
        coordinate_map = coordinate_map_from_payload(to_jsonable(map_payload))
        map_id = coordinate_map_fingerprint(coordinate_map)
        if map_id != self.coordinate_map_fingerprint:
            raise ValueError("coordinate_map_fingerprint does not match its definition")
        if tuple(coordinate_map.coordinate_ids) != ids or tuple(coordinate_map.units) != units:
            raise ValueError("Coordinate order or units do not match the coordinate map")
        reference = _validated_shape("reference_values", self.reference_values, (len(ids),))
        if not np.array_equal(reference, coordinate_map.reference_values):
            raise ValueError("Reference values do not match the coordinate map")
        reference_energy = float(self.reference_energy_Eh)
        if not np.isfinite(reference_energy):
            raise ValueError("reference_energy_Eh must be finite")
        reference_dipole = _validated_shape(
            "reference_dipole_body_au", self.reference_dipole_body_au, (3,)
        )
        energy_tolerance = _nonnegative_finite("anchor_tolerance_Eh", self.anchor_tolerance_Eh)
        dipole_tolerance = _nonnegative_finite(
            "anchor_tolerance_dipole_au", self.anchor_tolerance_dipole_au
        )
        energy_surfaces = dict(self.energy_increments)
        dipole_surfaces = dict(self.dipole_increments)
        if not energy_surfaces or set(energy_surfaces) != set(dipole_surfaces):
            raise ValueError("PES and DMS increment subsets must be equal and non-empty")
        subsets = tuple(sorted(energy_surfaces, key=lambda value: (len(value), value)))
        _require_subset_closure(subsets)
        canonical_axes: list[np.ndarray | None] = [None] * len(ids)
        for subset in subsets:
            if _normalize_subset(subset, len(ids)) != subset:
                raise ValueError("Increment subsets must use canonical increasing indices")
            energy_surface = energy_surfaces[subset]
            dipole_surface = dipole_surfaces[subset]
            if energy_surface.rank != len(subset) or dipole_surface.rank != len(subset):
                raise ValueError("Increment surface rank does not match its subset")
            if energy_surface.node_values.ndim != len(subset):
                raise ValueError("Energy increments must be scalar-valued")
            if dipole_surface.node_values.shape[len(subset) :] != (3,):
                raise ValueError("Dipole increments must have three components")
            if any(
                not np.array_equal(energy_axis, dipole_axis)
                for energy_axis, dipole_axis in zip(energy_surface.axes, dipole_surface.axes)
            ):
                raise ValueError("PES and DMS increments must share axes")
            for position, mode in enumerate(subset):
                axis = energy_surface.axes[position]
                reference_index = _reference_index(axis, reference[mode], mode)
                if canonical_axes[mode] is None:
                    canonical_axes[mode] = axis
                elif not np.array_equal(canonical_axes[mode], axis):
                    raise ValueError("Increment surfaces do not share canonical mode axes")
                selector: list[int | slice] = [slice(None)] * len(subset)
                selector[position] = reference_index
                if np.max(np.abs(energy_surface.node_values[tuple(selector)])) > energy_tolerance:
                    raise ValueError(f"Energy increment {subset} is not zero on its anchor")
                if np.max(np.abs(dipole_surface.node_values[tuple(selector)])) > dipole_tolerance:
                    raise ValueError(f"Dipole increment {subset} is not zero on its anchor")
        if any(axis is None for axis in canonical_axes):
            raise ValueError("Every coordinate must appear in an increment surface")
        if operator.index(self.schema_version) != NMODE_SURFACE_SCHEMA_VERSION:
            raise ValueError("Unsupported n-mode surface schema")
        object.__setattr__(self, "coordinate_ids", ids)
        object.__setattr__(self, "coordinate_units", units)
        object.__setattr__(self, "coordinate_map_payload", map_payload)
        object.__setattr__(self, "coordinate_map_fingerprint", map_id)
        object.__setattr__(self, "reference_values", immutable_array(reference))
        object.__setattr__(self, "reference_energy_Eh", reference_energy)
        object.__setattr__(self, "reference_dipole_body_au", immutable_array(reference_dipole))
        object.__setattr__(
            self,
            "energy_increments",
            MappingProxyType({subset: energy_surfaces[subset] for subset in subsets}),
        )
        object.__setattr__(
            self,
            "dipole_increments",
            MappingProxyType({subset: dipole_surfaces[subset] for subset in subsets}),
        )
        object.__setattr__(self, "source_lineage", _validated_source_lineage(self.source_lineage))
        object.__setattr__(self, "annotations", immutable_json_mapping(self.annotations))
        object.__setattr__(self, "anchor_tolerance_Eh", energy_tolerance)
        object.__setattr__(self, "anchor_tolerance_dipole_au", dipole_tolerance)
        object.__setattr__(self, "schema_version", NMODE_SURFACE_SCHEMA_VERSION)

    @property
    def n_modes(self) -> int:
        return len(self.coordinate_ids)

    @property
    def subsets(self) -> tuple[ModeSubset, ...]:
        return tuple(self.energy_increments)

    def energy_Eh(self, q: np.ndarray, *, max_rank: int | None = None) -> float:
        values = _validated_shape("q", q, (self.n_modes,))
        total = self.reference_energy_Eh
        for subset, surface in self.energy_increments.items():
            if max_rank is None or len(subset) <= max_rank:
                total += float(surface.evaluate(values[list(subset)]))
        return float(total)

    def potential_Eh(self, q: np.ndarray, *, max_rank: int | None = None) -> float:
        return self.energy_Eh(q, max_rank=max_rank) - self.reference_energy_Eh

    def dipole_body_au(self, q: np.ndarray, *, max_rank: int | None = None) -> np.ndarray:
        values = _validated_shape("q", q, (self.n_modes,))
        total = np.array(self.reference_dipole_body_au, copy=True)
        for subset, surface in self.dipole_increments.items():
            if max_rank is None or len(subset) <= max_rank:
                total += np.asarray(surface.evaluate(values[list(subset)]), dtype=float)
        return total

    def pes_numerical_payload(self) -> dict[str, object]:
        return {
            "schema": "pyscf-vscf-nmode-pes",
            "schema_version": self.schema_version,
            "coordinate_ids": list(self.coordinate_ids),
            "coordinate_units": list(self.coordinate_units),
            "coordinate_map": to_jsonable(self.coordinate_map_payload),
            "reference_values": float64_array_identity(self.reference_values),
            "reference_energy_Eh": self.reference_energy_Eh,
            "increments": {
                _subset_key(subset): surface.numerical_payload()
                for subset, surface in self.energy_increments.items()
            },
        }

    def dms_numerical_payload(self) -> dict[str, object]:
        return {
            "schema": "pyscf-vscf-nmode-dms",
            "schema_version": self.schema_version,
            "coordinate_ids": list(self.coordinate_ids),
            "coordinate_units": list(self.coordinate_units),
            "coordinate_map": to_jsonable(self.coordinate_map_payload),
            "reference_values": float64_array_identity(self.reference_values),
            "reference_dipole_body_au": float64_array_identity(self.reference_dipole_body_au),
            "increments": {
                _subset_key(subset): surface.numerical_payload()
                for subset, surface in self.dipole_increments.items()
            },
        }

    def numerical_surface_fingerprint(self) -> str:
        return payload_fingerprint(
            {"pes": self.pes_numerical_payload(), "dms": self.dms_numerical_payload()}
        )

    def source_lineage_fingerprint(self) -> str:
        return payload_fingerprint(to_jsonable(self.source_lineage))

    def artifact_payload(self) -> dict[str, object]:
        return {
            "schema": "pyscf-vscf-nmode-surface-artifact",
            "schema_version": self.schema_version,
            "pes": self.pes_numerical_payload(),
            "dms": self.dms_numerical_payload(),
            "energy_diagnostics": {
                _subset_key(subset): surface.artifact_payload()
                for subset, surface in self.energy_increments.items()
            },
            "dipole_diagnostics": {
                _subset_key(subset): surface.artifact_payload()
                for subset, surface in self.dipole_increments.items()
            },
            "source_lineage": to_jsonable(self.source_lineage),
            "annotations": to_jsonable(self.annotations),
            "anchor_tolerance_Eh": self.anchor_tolerance_Eh,
            "anchor_tolerance_dipole_au": self.anchor_tolerance_dipole_au,
        }

    def artifact_integrity_fingerprint(self) -> str:
        return payload_fingerprint(self.artifact_payload())

    def fingerprint(self) -> str:
        """Compatibility spelling for complete artifact integrity."""

        return self.artifact_integrity_fingerprint()


def nmode_pes_fingerprint(model: NModeSurfaceModel) -> str:
    """Return only the numerical PES identity."""

    return payload_fingerprint(model.pes_numerical_payload())


def nmode_dms_fingerprint(model: NModeSurfaceModel) -> str:
    """Return only the numerical vector-DMS identity."""

    return payload_fingerprint(model.dms_numerical_payload())


def fit_nmode_surface(
    samples: NModeSampleSet,
    *,
    method: str = "linear",
    held_out: Mapping[ModeSubset, HeldOutCutSamples] | None = None,
    annotations: Mapping[str, Any] | None = None,
) -> NModeSurfaceModel:
    """Apply anchored inclusion-exclusion and fit deterministic tensor surfaces."""

    interpolation_method = _validated_interpolation_method(method)
    held_out_samples = dict(held_out or {})
    if not set(held_out_samples).issubset(samples.cuts):
        raise ValueError("Held-out subsets must correspond to retained training cuts")
    energy_surfaces: dict[ModeSubset, TensorProductSurface] = {}
    dipole_surfaces: dict[ModeSubset, TensorProductSurface] = {}
    reference = np.asarray(samples.reference_values, dtype=float)
    for subset, cut in samples.cuts.items():
        energy_increment = np.asarray(cut.energies_Eh - samples.reference_energy_Eh, dtype=float)
        dipole_increment = np.asarray(
            cut.dipoles_body_au - samples.reference_dipole_body_au,
            dtype=float,
        )
        mesh = _mesh_points(cut.axes)
        for lower_subset in _proper_nonempty_subsets(subset):
            positions = [subset.index(mode) for mode in lower_subset]
            lower_points = mesh[:, positions]
            energy_increment -= (
                energy_surfaces[lower_subset]
                .evaluate(lower_points)
                .reshape(energy_increment.shape)
            )
            dipole_increment -= (
                dipole_surfaces[lower_subset]
                .evaluate(lower_points)
                .reshape(dipole_increment.shape)
            )
        held_energy_increment = None
        held_dipole_increment = None
        held_points = held_out_samples.get(subset)
        if held_points is not None:
            if held_points.subset != subset:
                raise ValueError("Held-out mapping key does not match its subset")
            held_energy_increment = held_points.energies_Eh - samples.reference_energy_Eh
            held_dipole_increment = held_points.dipoles_body_au - samples.reference_dipole_body_au
            for lower_subset in _proper_nonempty_subsets(subset):
                positions = [subset.index(mode) for mode in lower_subset]
                lower_points = held_points.points[:, positions]
                held_energy_increment -= energy_surfaces[lower_subset].evaluate(lower_points)
                held_dipole_increment -= dipole_surfaces[lower_subset].evaluate(lower_points)
        energy_surfaces[subset] = _fit_tensor_surface(
            cut.axes,
            energy_increment,
            method=interpolation_method,
            held_out_points=None if held_points is None else held_points.points,
            held_out_values=held_energy_increment,
            held_out_point_ids=()
            if held_points is None
            else held_points.point_causal_fingerprints,
        )
        dipole_surfaces[subset] = _fit_tensor_surface(
            cut.axes,
            dipole_increment,
            method=interpolation_method,
            held_out_points=None if held_points is None else held_points.points,
            held_out_values=held_dipole_increment,
            held_out_point_ids=()
            if held_points is None
            else held_points.point_causal_fingerprints,
        )
    return NModeSurfaceModel(
        coordinate_ids=samples.coordinate_ids,
        coordinate_units=samples.coordinate_units,
        coordinate_map_payload=samples.coordinate_map.fingerprint_payload(),
        coordinate_map_fingerprint=samples.coordinate_map_fingerprint,
        reference_values=reference,
        reference_energy_Eh=samples.reference_energy_Eh,
        reference_dipole_body_au=samples.reference_dipole_body_au,
        energy_increments=energy_surfaces,
        dipole_increments=dipole_surfaces,
        source_lineage=to_jsonable(samples.source_lineage),
        annotations=dict(annotations or {}),
    )


def nmode_potential_from_surface(
    model: NModeSurfaceModel,
    coordinates: Sequence[np.ndarray],
    masses_amu: Sequence[float],
    *,
    mode_labels: Sequence[str] | None = None,
):
    """Adapt complete anchored 1MR/2MR increments to the released VSCF solver."""

    from .vscf import NModePotential

    if any(len(subset) >= 3 for subset in model.subsets):
        raise ValueError("Rectilinear VSCF adaptation cannot discard rank-3 increments")
    required_singletons = {(mode,) for mode in range(model.n_modes)}
    if not required_singletons.issubset(model.energy_increments):
        raise ValueError("VSCF adaptation requires complete one-mode closure")
    if any(unit != "angstrom" for unit in model.coordinate_units):
        raise ValueError("The released rectilinear VSCF solver requires Angstrom coordinates")
    grids = tuple(
        _validated_solver_grid(f"coordinates[{mode}]", values)
        for mode, values in enumerate(coordinates)
    )
    if len(grids) != model.n_modes:
        raise ValueError("coordinates must contain one solver grid per fitted coordinate")
    masses = tuple(
        _positive_finite(f"masses_amu[{mode}]", mass) for mode, mass in enumerate(masses_amu)
    )
    if len(masses) != model.n_modes:
        raise ValueError("masses_amu must contain one mass per fitted coordinate")
    labels = tuple(model.coordinate_ids) if mode_labels is None else tuple(mode_labels)
    one_mode = tuple(
        np.asarray(
            model.energy_increments[(mode,)].evaluate(grids[mode][:, None]),
            dtype=float,
        )
        for mode in range(model.n_modes)
    )
    couplings: dict[ModeSubset, np.ndarray] = {}
    for subset, surface in model.energy_increments.items():
        if len(subset) != 2:
            continue
        mesh = np.meshgrid(*(grids[mode] for mode in subset), indexing="ij")
        points = np.stack(mesh, axis=-1)
        couplings[subset] = np.asarray(surface.evaluate(points), dtype=float)
    provenance = {
        "adapter_schema": "pyscf-vscf-nmode-surface-to-vscf",
        "adapter_schema_version": 1,
        "source_pes_fingerprint": nmode_pes_fingerprint(model),
        "source_coordinate_map_fingerprint": model.coordinate_map_fingerprint,
        "source_reference_energy_Eh": model.reference_energy_Eh,
        "solver_grids": [float64_array_identity(grid) for grid in grids],
        "masses_amu": list(masses),
    }
    return NModePotential(
        coordinates=grids,
        masses_amu=masses,
        one_mode_potentials_Eh=one_mode,
        two_mode_couplings_Eh=couplings,
        mode_labels=labels,
        provenance=provenance,
        coordinate_map_fingerprint=model.coordinate_map_fingerprint,
        coordinate_units="angstrom",
    )


def dump_nmode_surface(model: NModeSurfaceModel, path: Path | str) -> None:
    """Atomically write one schema-versioned, integrity-protected surface archive."""

    arrays: dict[str, np.ndarray] = {
        "reference_values": np.asarray(model.reference_values),
        "reference_dipole_body_au": np.asarray(model.reference_dipole_body_au),
    }
    surface_specs: dict[str, object] = {}
    for subset in model.subsets:
        subset_name = _subset_key(subset)
        prefix = "subset_" + "_".join(str(mode) for mode in subset)
        axis_names = []
        for position, axis in enumerate(model.energy_increments[subset].axes):
            name = f"{prefix}_axis_{position}"
            arrays[name] = np.asarray(axis)
            axis_names.append(name)
        energy_name = f"{prefix}_energy"
        dipole_name = f"{prefix}_dipole"
        arrays[energy_name] = np.asarray(model.energy_increments[subset].node_values)
        arrays[dipole_name] = np.asarray(model.dipole_increments[subset].node_values)
        surface_specs[subset_name] = {
            "subset": list(subset),
            "axes": axis_names,
            "energy_values": energy_name,
            "dipole_values": dipole_name,
            "energy_method": model.energy_increments[subset].method,
            "dipole_method": model.dipole_increments[subset].method,
            "energy_diagnostics": asdict(model.energy_increments[subset].diagnostics),
            "dipole_diagnostics": asdict(model.dipole_increments[subset].diagnostics),
        }
    manifest = {
        "schema": "pyscf-vscf-nmode-surface-archive",
        "schema_version": NMODE_SURFACE_SCHEMA_VERSION,
        "coordinate_ids": list(model.coordinate_ids),
        "coordinate_units": list(model.coordinate_units),
        "coordinate_map_payload": to_jsonable(model.coordinate_map_payload),
        "coordinate_map_fingerprint": model.coordinate_map_fingerprint,
        "reference_energy_Eh": model.reference_energy_Eh,
        "source_lineage": to_jsonable(model.source_lineage),
        "annotations": to_jsonable(model.annotations),
        "anchor_tolerance_Eh": model.anchor_tolerance_Eh,
        "anchor_tolerance_dipole_au": model.anchor_tolerance_dipole_au,
        "surfaces": surface_specs,
        "array_identities": {
            name: float64_array_identity(array) for name, array in sorted(arrays.items())
        },
        "artifact_integrity_fingerprint": model.artifact_integrity_fingerprint(),
    }
    arrays["manifest_json"] = np.frombuffer(
        canonical_json(manifest).encode("utf-8"), dtype=np.uint8
    )
    atomic_savez_compressed(path, arrays)


def load_nmode_surface(path: Path | str) -> NModeSurfaceModel:
    """Load and verify a surface archive without pickle or implicit migration."""

    source = Path(path)
    try:
        archive = np.load(source, allow_pickle=False)
    except Exception as exc:
        raise ValueError(f"Unable to read n-mode surface archive '{source}'") from exc
    with archive as data:
        if "manifest_json" not in data.files:
            raise ValueError("N-mode surface archive is missing manifest_json")
        try:
            manifest = json.loads(np.asarray(data["manifest_json"], dtype=np.uint8).tobytes())
        except Exception as exc:
            raise ValueError("N-mode surface manifest is invalid") from exc
        if (
            manifest.get("schema") != "pyscf-vscf-nmode-surface-archive"
            or manifest.get("schema_version") != NMODE_SURFACE_SCHEMA_VERSION
        ):
            raise ValueError("Unsupported n-mode surface archive schema")
        identities = manifest.get("array_identities")
        if not isinstance(identities, Mapping):
            raise ValueError("N-mode surface archive is missing array identities")
        if set(data.files) != {"manifest_json", *identities}:
            raise ValueError("N-mode surface archive contains missing or unexpected arrays")
        arrays: dict[str, np.ndarray] = {}
        for name, identity in identities.items():
            values = np.asarray(data[name])
            if float64_array_identity(values) != identity:
                raise ValueError(f"N-mode surface array {name!r} failed its integrity check")
            arrays[name] = values

    energy_surfaces: dict[ModeSubset, TensorProductSurface] = {}
    dipole_surfaces: dict[ModeSubset, TensorProductSurface] = {}
    for specification in manifest.get("surfaces", {}).values():
        subset = tuple(operator.index(mode) for mode in specification["subset"])
        axes = tuple(arrays[name] for name in specification["axes"])
        energy_surfaces[subset] = TensorProductSurface(
            axes=axes,
            node_values=arrays[specification["energy_values"]],
            method=specification["energy_method"],
            diagnostics=_diagnostics_from_payload(specification["energy_diagnostics"]),
        )
        dipole_surfaces[subset] = TensorProductSurface(
            axes=axes,
            node_values=arrays[specification["dipole_values"]],
            method=specification["dipole_method"],
            diagnostics=_diagnostics_from_payload(specification["dipole_diagnostics"]),
        )
    model = NModeSurfaceModel(
        coordinate_ids=tuple(manifest["coordinate_ids"]),
        coordinate_units=tuple(manifest["coordinate_units"]),
        coordinate_map_payload=manifest["coordinate_map_payload"],
        coordinate_map_fingerprint=manifest["coordinate_map_fingerprint"],
        reference_values=arrays["reference_values"],
        reference_energy_Eh=manifest["reference_energy_Eh"],
        reference_dipole_body_au=arrays["reference_dipole_body_au"],
        energy_increments=energy_surfaces,
        dipole_increments=dipole_surfaces,
        source_lineage=manifest["source_lineage"],
        annotations=manifest["annotations"],
        anchor_tolerance_Eh=manifest["anchor_tolerance_Eh"],
        anchor_tolerance_dipole_au=manifest["anchor_tolerance_dipole_au"],
    )
    expected = str(manifest.get("artifact_integrity_fingerprint", ""))
    if not expected or model.artifact_integrity_fingerprint() != expected:
        raise ValueError("N-mode surface artifact integrity fingerprint mismatch")
    return model


def _fit_tensor_surface(
    axes: tuple[np.ndarray, ...],
    node_values: np.ndarray,
    *,
    method: str,
    held_out_points: np.ndarray | None,
    held_out_values: np.ndarray | None,
    held_out_point_ids: tuple[str, ...],
) -> TensorProductSurface:
    normalized_axes = tuple(_validated_axis("fit axis", axis) for axis in axes)
    values = np.asarray(node_values, dtype=float)
    backend_method = "cubic_legacy" if method == "cubic" else method
    interpolator = RegularGridInterpolator(
        normalized_axes,
        values,
        method=backend_method,
        bounds_error=True,
    )
    training_prediction = np.asarray(interpolator(_mesh_points(normalized_axes)), dtype=float)
    training_prediction = training_prediction.reshape(values.shape)
    training_error = training_prediction - values
    training_max = _component_max_abs(training_error, len(normalized_axes))
    held_count = 0
    held_rmse = None
    held_max = None
    if held_out_points is not None or held_out_values is not None:
        if held_out_points is None or held_out_values is None:
            raise ValueError("Held-out points and values must be supplied together")
        expected = np.asarray(held_out_values, dtype=float)
        prediction = np.asarray(interpolator(held_out_points), dtype=float)
        if prediction.shape != expected.shape:
            raise ValueError("Held-out values do not match interpolated output shape")
        error = prediction - expected
        held_count = int(np.asarray(held_out_points).shape[0])
        held_rmse = _component_rmse(error)
        held_max = _component_max_abs(error, 1)
    diagnostics = FitDiagnostics(
        method=method,
        n_training_points=int(np.prod([axis.size for axis in normalized_axes])),
        training_max_abs_error=training_max,
        n_held_out_points=held_count,
        held_out_rmse=held_rmse,
        held_out_max_abs_error=held_max,
        held_out_point_causal_fingerprints=held_out_point_ids,
    )
    return TensorProductSurface(
        axes=normalized_axes,
        node_values=values,
        method=method,
        diagnostics=diagnostics,
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


def _proper_nonempty_subsets(subset: ModeSubset) -> tuple[ModeSubset, ...]:
    return tuple(
        lower for size in range(1, len(subset)) for lower in itertools.combinations(subset, size)
    )


def _reference_index(axis: np.ndarray, reference: float, mode: int) -> int:
    matches = np.flatnonzero(axis == reference)
    if matches.size != 1:
        raise ValueError(f"Mode {mode} axis must contain its exact reference value once")
    return int(matches[0])


def _mesh_points(axes: Sequence[np.ndarray]) -> np.ndarray:
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape((-1, len(axes)))


def _validated_shape(name: str, values: object, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype="<f8")
    if array.shape != shape or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape {shape}")
    return np.ascontiguousarray(array)


def _validated_interpolation_method(value: object) -> str:
    method = str(value).strip().lower()
    if method not in {"linear", "cubic"}:
        raise ValueError("Interpolation method must be 'linear' or 'cubic'")
    return method


def _component_max_abs(error: np.ndarray, leading_rank: int) -> tuple[float, ...]:
    values = np.asarray(error, dtype=float)
    component_shape = values.shape[leading_rank:]
    component_count = int(np.prod(component_shape)) if component_shape else 1
    rows = values.reshape((-1, component_count))
    return tuple(float(value) for value in np.max(np.abs(rows), axis=0))


def _component_rmse(error: np.ndarray) -> tuple[float, ...]:
    values = np.asarray(error, dtype=float)
    rows = values.reshape((values.shape[0], -1))
    return tuple(float(value) for value in np.sqrt(np.mean(np.square(rows), axis=0)))


def _diagnostics_from_payload(payload: Mapping[str, object]) -> FitDiagnostics:
    return FitDiagnostics(
        method=payload["method"],
        n_training_points=payload["n_training_points"],
        training_max_abs_error=tuple(payload["training_max_abs_error"]),
        n_held_out_points=payload.get("n_held_out_points", 0),
        held_out_rmse=None
        if payload.get("held_out_rmse") is None
        else tuple(payload["held_out_rmse"]),
        held_out_max_abs_error=None
        if payload.get("held_out_max_abs_error") is None
        else tuple(payload["held_out_max_abs_error"]),
        held_out_point_causal_fingerprints=tuple(
            payload.get("held_out_point_causal_fingerprints", ())
        ),
    )


def _validated_source_lineage(values: Mapping[str, Any]):
    source = dict(values)
    required = {
        "schema",
        "schema_version",
        "provider_scientific_fingerprint",
        "point_causal_fingerprints",
    }
    optional = {
        "result_scientific_fingerprints",
        "sampling_lineage_fingerprints",
    }
    if set(source) - required - optional or not required.issubset(source):
        raise ValueError("Source lineage contains missing or unsupported fields")
    if (
        source["schema"] != "pyscf-vscf-electronic-source-lineage"
        or operator.index(source["schema_version"]) != 1
    ):
        raise ValueError("Unsupported source-lineage schema")
    provider = _nonempty(
        "provider_scientific_fingerprint",
        source["provider_scientific_fingerprint"],
    )
    points = tuple(sorted(str(value) for value in source["point_causal_fingerprints"]))
    if not points or any(not value for value in points) or len(set(points)) != len(points):
        raise ValueError("Source lineage point identities must be non-empty and unique")
    normalized: dict[str, object] = {
        "schema": source["schema"],
        "schema_version": 1,
        "provider_scientific_fingerprint": provider,
        "point_causal_fingerprints": list(points),
    }
    if "result_scientific_fingerprints" in source:
        results = {
            str(point): _nonempty("result scientific fingerprint", result)
            for point, result in dict(source["result_scientific_fingerprints"]).items()
        }
        if set(results) != set(points):
            raise ValueError("Result lineage must exactly cover source point identities")
        normalized["result_scientific_fingerprints"] = {
            point: results[point] for point in sorted(results)
        }
    if "sampling_lineage_fingerprints" in source:
        sampling = tuple(sorted(str(value) for value in source["sampling_lineage_fingerprints"]))
        if any(not value for value in sampling) or len(set(sampling)) != len(sampling):
            raise ValueError("Sampling-lineage fingerprints must be non-empty and unique")
        normalized["sampling_lineage_fingerprints"] = list(sampling)
    return immutable_json_mapping(normalized)


def _subset_key(subset: ModeSubset) -> str:
    return ",".join(str(mode) for mode in subset)


def _validated_solver_grid(name: str, values: object) -> np.ndarray:
    grid = _finite_array(name, values, ndim=1)
    if grid.size < 3 or np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} must contain at least three strictly increasing points")
    steps = np.diff(grid)
    if not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced")
    return grid


def _positive_finite(name: str, value: object) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _nonnegative_finite(name: str, value: object) -> float:
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


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
    "FitDiagnostics",
    "HeldOutCutSamples",
    "LineageKey",
    "ModeSubset",
    "NMODE_POINT_PLAN_SCHEMA_VERSION",
    "NMODE_SURFACE_SCHEMA_VERSION",
    "NModeCutSamples",
    "NModePointPlan",
    "NModeSampleSet",
    "NModeSurfaceModel",
    "SamplingLineage",
    "TensorProductSurface",
    "assemble_nmode_samples",
    "dump_nmode_surface",
    "enumerate_mode_subsets",
    "fit_nmode_surface",
    "load_nmode_surface",
    "nmode_dms_fingerprint",
    "nmode_pes_fingerprint",
    "nmode_potential_from_surface",
    "plan_nmode_points",
]
