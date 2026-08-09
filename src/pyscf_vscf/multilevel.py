"""Pure increment-level composition for fitted n-mode PES and vector DMS models."""

from __future__ import annotations

import itertools
import operator
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from ._identity import payload_fingerprint, to_jsonable
from .nmode import (
    FitDiagnostics,
    ModeSubset,
    NModeSurfaceModel,
    TensorProductSurface,
    _fit_tensor_surface,
    nmode_dms_fingerprint,
    nmode_pes_fingerprint,
)


MULTILEVEL_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ObservableCompositionDiagnostics:
    """Separate fit and correction diagnostics for one increment observable.

    ``correction_max_abs`` is measured on the low-level training nodes; an
    interpolant, especially a cubic one, can attain a larger between-node value.
    """

    status: str
    low_surface_fingerprint: str
    final_surface_fingerprint: str
    correction_max_abs: tuple[float, ...]
    final_fit: FitDiagnostics
    high_surface_fingerprint: str | None = None
    delta_surface_fingerprint: str | None = None
    delta_fit: FitDiagnostics | None = None

    def __post_init__(self) -> None:
        status = str(self.status).strip().lower()
        if status not in {"low_level", "delta"}:
            raise ValueError("Observable composition status must be low_level or delta")
        low = _nonempty("low_surface_fingerprint", self.low_surface_fingerprint)
        final = _nonempty("final_surface_fingerprint", self.final_surface_fingerprint)
        correction = tuple(float(value) for value in self.correction_max_abs)
        if not correction or any(not np.isfinite(value) or value < 0.0 for value in correction):
            raise ValueError("correction_max_abs must contain finite non-negative values")
        if not isinstance(self.final_fit, FitDiagnostics):
            raise TypeError("final_fit must be FitDiagnostics")
        high = _optional_nonempty("high_surface_fingerprint", self.high_surface_fingerprint)
        delta = _optional_nonempty("delta_surface_fingerprint", self.delta_surface_fingerprint)
        if status == "low_level":
            if high is not None or delta is not None or self.delta_fit is not None:
                raise ValueError("Low-level diagnostics cannot claim a Delta correction")
            if low != final or any(value != 0.0 for value in correction):
                raise ValueError("Uncorrected increments must retain the low-level surface")
        elif high is None or delta is None or not isinstance(self.delta_fit, FitDiagnostics):
            raise ValueError("Delta diagnostics require high-level and fitted correction data")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "low_surface_fingerprint", low)
        object.__setattr__(self, "final_surface_fingerprint", final)
        object.__setattr__(self, "correction_max_abs", correction)
        object.__setattr__(self, "high_surface_fingerprint", high)
        object.__setattr__(self, "delta_surface_fingerprint", delta)

    def payload(self) -> dict[str, object]:
        return {
            "status": self.status,
            "low_surface_fingerprint": self.low_surface_fingerprint,
            "high_surface_fingerprint": self.high_surface_fingerprint,
            "delta_surface_fingerprint": self.delta_surface_fingerprint,
            "final_surface_fingerprint": self.final_surface_fingerprint,
            "correction_max_abs": list(self.correction_max_abs),
            "delta_fit": None if self.delta_fit is None else asdict(self.delta_fit),
            "final_fit": asdict(self.final_fit),
        }


@dataclass(frozen=True)
class IncrementCompositionDiagnostics:
    """PES and vector-DMS diagnostics for one mode subset."""

    subset: ModeSubset
    energy: ObservableCompositionDiagnostics
    dipole: ObservableCompositionDiagnostics

    def __post_init__(self) -> None:
        subset = _normalize_subset(self.subset)
        if not isinstance(self.energy, ObservableCompositionDiagnostics) or not isinstance(
            self.dipole, ObservableCompositionDiagnostics
        ):
            raise TypeError("Increment diagnostics require energy and dipole records")
        if len(self.energy.correction_max_abs) != 1:
            raise ValueError("Energy correction diagnostics must contain one component")
        if len(self.dipole.correction_max_abs) != 3:
            raise ValueError("Dipole correction diagnostics must contain three components")
        object.__setattr__(self, "subset", subset)

    def payload(self) -> dict[str, object]:
        return {
            "subset": list(self.subset),
            "energy": self.energy.payload(),
            "dipole": self.dipole.payload(),
        }


@dataclass(frozen=True)
class MultilevelCompositionDiagnostics:
    """Composition policy and per-observable numerical diagnostics."""

    low_provider_scientific_fingerprint: str
    high_provider_scientific_fingerprint: str
    low_source_lineage_fingerprint: str
    high_source_lineage_fingerprint: str
    energy_corrected_subsets: tuple[ModeSubset, ...]
    dipole_corrected_subsets: tuple[ModeSubset, ...]
    records: Mapping[ModeSubset, IncrementCompositionDiagnostics]
    composed_pes_fingerprint: str
    composed_dms_fingerprint: str
    reference_policy: str = "preserve-low-level-absolute-reference"
    schema_version: int = MULTILEVEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        energy = _normalized_subset_collection(self.energy_corrected_subsets)
        dipole = _normalized_subset_collection(self.dipole_corrected_subsets)
        records = dict(self.records)
        canonical: dict[ModeSubset, IncrementCompositionDiagnostics] = {}
        for key in sorted(records, key=lambda value: (len(value), value)):
            subset = _normalize_subset(key)
            record = records[key]
            if not isinstance(record, IncrementCompositionDiagnostics) or record.subset != subset:
                raise ValueError("Composition record keys must match their subsets")
            canonical[subset] = record
        if not canonical or not (set(energy) | set(dipole)):
            raise ValueError("Multilevel diagnostics require at least one corrected increment")
        if set(energy) - set(canonical) or set(dipole) - set(canonical):
            raise ValueError("Corrected subsets must be represented in diagnostics")
        for subset, record in canonical.items():
            if (record.energy.status == "delta") != (subset in energy):
                raise ValueError("Energy correction status does not match the selected subsets")
            if (record.dipole.status == "delta") != (subset in dipole):
                raise ValueError("Dipole correction status does not match the selected subsets")
        policy = str(self.reference_policy).strip()
        if policy != "preserve-low-level-absolute-reference":
            raise ValueError("The minimal composer has one frozen reference policy")
        if operator.index(self.schema_version) != MULTILEVEL_SCHEMA_VERSION:
            raise ValueError("Unsupported multilevel diagnostics schema")
        object.__setattr__(self, "energy_corrected_subsets", energy)
        object.__setattr__(self, "dipole_corrected_subsets", dipole)
        object.__setattr__(self, "records", MappingProxyType(canonical))
        for name in (
            "low_provider_scientific_fingerprint",
            "high_provider_scientific_fingerprint",
            "low_source_lineage_fingerprint",
            "high_source_lineage_fingerprint",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        object.__setattr__(
            self,
            "composed_pes_fingerprint",
            _nonempty("composed_pes_fingerprint", self.composed_pes_fingerprint),
        )
        object.__setattr__(
            self,
            "composed_dms_fingerprint",
            _nonempty("composed_dms_fingerprint", self.composed_dms_fingerprint),
        )
        object.__setattr__(self, "reference_policy", policy)
        object.__setattr__(self, "schema_version", MULTILEVEL_SCHEMA_VERSION)

    def scientific_payload(self) -> dict[str, object]:
        """Return composition diagnostics without arbitrary model annotations."""

        return {
            "schema": "pyscf-vscf-multilevel-composition-diagnostics",
            "schema_version": self.schema_version,
            "low_provider_scientific_fingerprint": (self.low_provider_scientific_fingerprint),
            "high_provider_scientific_fingerprint": (self.high_provider_scientific_fingerprint),
            "low_source_lineage_fingerprint": self.low_source_lineage_fingerprint,
            "high_source_lineage_fingerprint": self.high_source_lineage_fingerprint,
            "energy_corrected_subsets": [list(value) for value in self.energy_corrected_subsets],
            "dipole_corrected_subsets": [list(value) for value in self.dipole_corrected_subsets],
            "reference_policy": self.reference_policy,
            "composed_pes_fingerprint": self.composed_pes_fingerprint,
            "composed_dms_fingerprint": self.composed_dms_fingerprint,
            "records": {
                _subset_key(subset): record.payload() for subset, record in self.records.items()
            },
        }

    def scientific_fingerprint(self) -> str:
        return payload_fingerprint(self.scientific_payload())


def compose_multilevel_surface(
    low: NModeSurfaceModel,
    high: NModeSurfaceModel,
    *,
    energy_corrected_subsets: Sequence[Sequence[int]] = (),
    dipole_corrected_subsets: Sequence[Sequence[int]] = (),
    annotations: Mapping[str, Any] | None = None,
) -> tuple[NModeSurfaceModel, MultilevelCompositionDiagnostics]:
    """Compose selected high-minus-low increments on the low-level grids.

    The low-level absolute energy and dipole reference are preserved. Only
    nonempty anchored increments are corrected, so the common reference cannot
    be counted once per subset or independently minimum-shifted.
    """

    _validate_model_compatibility(low, high)
    energy_selected = _selected_subsets(
        energy_corrected_subsets,
        low,
        high,
        observable="energy",
    )
    dipole_selected = _selected_subsets(
        dipole_corrected_subsets,
        low,
        high,
        observable="dipole",
    )
    if not energy_selected and not dipole_selected:
        raise ValueError("Multilevel composition requires at least one selected correction")
    _require_exact_model_anchors(
        low,
        subsets=low.subsets,
        level="low",
        observable="energy",
    )
    _require_exact_model_anchors(
        low,
        subsets=low.subsets,
        level="low",
        observable="dipole",
    )
    _require_exact_model_anchors(
        high,
        subsets=energy_selected,
        level="selected high",
        observable="energy",
    )
    _require_exact_model_anchors(
        high,
        subsets=dipole_selected,
        level="selected high",
        observable="dipole",
    )

    energy_surfaces: dict[ModeSubset, TensorProductSurface] = {}
    dipole_surfaces: dict[ModeSubset, TensorProductSurface] = {}
    records: dict[ModeSubset, IncrementCompositionDiagnostics] = {}
    for subset in low.subsets:
        final_energy, energy_record = _compose_observable(
            subset,
            low.energy_increments[subset],
            high.energy_increments.get(subset),
            selected=subset in energy_selected,
        )
        final_dipole, dipole_record = _compose_observable(
            subset,
            low.dipole_increments[subset],
            high.dipole_increments.get(subset),
            selected=subset in dipole_selected,
        )
        energy_surfaces[subset] = final_energy
        dipole_surfaces[subset] = final_dipole
        records[subset] = IncrementCompositionDiagnostics(
            subset=subset,
            energy=energy_record,
            dipole=dipole_record,
        )

    combined_annotations = dict(low.annotations)
    combined_annotations.update(dict(annotations or {}))
    model = NModeSurfaceModel(
        coordinate_ids=low.coordinate_ids,
        coordinate_units=low.coordinate_units,
        coordinate_map_payload=to_jsonable(low.coordinate_map_payload),
        coordinate_map_fingerprint=low.coordinate_map_fingerprint,
        reference_values=low.reference_values,
        reference_energy_Eh=low.reference_energy_Eh,
        reference_dipole_body_au=low.reference_dipole_body_au,
        energy_increments=energy_surfaces,
        dipole_increments=dipole_surfaces,
        source_lineage=_composite_source_lineage(
            low,
            high,
            energy_selected=energy_selected,
            dipole_selected=dipole_selected,
        ),
        annotations=combined_annotations,
        anchor_tolerance_Eh=max(low.anchor_tolerance_Eh, high.anchor_tolerance_Eh),
        anchor_tolerance_dipole_au=max(
            low.anchor_tolerance_dipole_au,
            high.anchor_tolerance_dipole_au,
        ),
    )
    if model.energy_Eh(model.reference_values) != model.reference_energy_Eh or not np.array_equal(
        model.dipole_body_au(model.reference_values),
        model.reference_dipole_body_au,
    ):
        raise RuntimeError("Composed n-mode model did not preserve its exact reference values")
    diagnostics = MultilevelCompositionDiagnostics(
        low_provider_scientific_fingerprint=str(
            low.source_lineage["provider_scientific_fingerprint"]
        ),
        high_provider_scientific_fingerprint=str(
            high.source_lineage["provider_scientific_fingerprint"]
        ),
        low_source_lineage_fingerprint=low.source_lineage_fingerprint(),
        high_source_lineage_fingerprint=high.source_lineage_fingerprint(),
        energy_corrected_subsets=energy_selected,
        dipole_corrected_subsets=dipole_selected,
        records=records,
        composed_pes_fingerprint=nmode_pes_fingerprint(model),
        composed_dms_fingerprint=nmode_dms_fingerprint(model),
    )
    return model, diagnostics


def _compose_observable(
    subset: ModeSubset,
    low: TensorProductSurface,
    high: TensorProductSurface | None,
    *,
    selected: bool,
) -> tuple[TensorProductSurface, ObservableCompositionDiagnostics]:
    low_fingerprint = _surface_fingerprint(low)
    component_count = 1 if low.node_values.ndim == len(subset) else 3
    if not selected:
        return low, ObservableCompositionDiagnostics(
            status="low_level",
            low_surface_fingerprint=low_fingerprint,
            final_surface_fingerprint=low_fingerprint,
            correction_max_abs=(0.0,) * component_count,
            final_fit=low.diagnostics,
        )
    if high is None:
        raise ValueError(f"Selected correction subset {subset} is absent from the high model")
    _require_common_domain(low.axes, high.axes, subset)
    points = _mesh_points(low.axes)
    high_on_low = np.asarray(high.evaluate(points), dtype=float).reshape(low.node_values.shape)
    delta_values = high_on_low - low.node_values
    delta_surface = _fit_tensor_surface(
        low.axes,
        delta_values,
        method=low.method,
        held_out_points=None,
        held_out_values=None,
        held_out_point_ids=(),
    )
    fitted_delta = np.asarray(delta_surface.evaluate(points), dtype=float).reshape(
        low.node_values.shape
    )
    final_surface = _fit_tensor_surface(
        low.axes,
        low.node_values + fitted_delta,
        method=low.method,
        held_out_points=None,
        held_out_values=None,
        held_out_point_ids=(),
    )
    return final_surface, ObservableCompositionDiagnostics(
        status="delta",
        low_surface_fingerprint=low_fingerprint,
        high_surface_fingerprint=_surface_fingerprint(high),
        delta_surface_fingerprint=_surface_fingerprint(delta_surface),
        final_surface_fingerprint=_surface_fingerprint(final_surface),
        correction_max_abs=_component_max_abs(delta_values, len(subset)),
        delta_fit=delta_surface.diagnostics,
        final_fit=final_surface.diagnostics,
    )


def _validate_model_compatibility(low: NModeSurfaceModel, high: NModeSurfaceModel) -> None:
    if not isinstance(low, NModeSurfaceModel) or not isinstance(high, NModeSurfaceModel):
        raise TypeError("Multilevel composition requires two NModeSurfaceModel objects")
    checks = (
        (low.coordinate_ids, high.coordinate_ids, "coordinate IDs"),
        (low.coordinate_units, high.coordinate_units, "coordinate units"),
    )
    for left, right, name in checks:
        if left != right:
            raise ValueError(f"Low- and high-level models use different {name}")
    if not np.array_equal(low.reference_values, high.reference_values):
        raise ValueError("Low- and high-level models require one exact reference coordinate")
    if low.coordinate_map_fingerprint != high.coordinate_map_fingerprint:
        raise ValueError("Low- and high-level models use different coordinate map/frame")


def _require_exact_model_anchors(
    model: NModeSurfaceModel,
    *,
    subsets: Sequence[ModeSubset],
    level: str,
    observable: str,
) -> None:
    if observable == "energy":
        surfaces = model.energy_increments
    elif observable == "dipole":
        surfaces = model.dipole_increments
    else:
        raise ValueError("Anchor observable must be energy or dipole")
    for subset in subsets:
        surface = surfaces[subset]
        for position, mode in enumerate(subset):
            matches = np.flatnonzero(surface.axes[position] == model.reference_values[mode])
            if matches.size != 1:
                raise ValueError(f"{level} subset {subset} lacks its exact reference node")
            selector: list[int | slice] = [slice(None)] * len(subset)
            selector[position] = int(matches[0])
            if np.any(surface.node_values[tuple(selector)] != 0.0):
                raise ValueError(
                    f"{level} {observable} increment {subset} is not exactly anchored"
                )


def _selected_subsets(
    values: Sequence[Sequence[int]],
    low: NModeSurfaceModel,
    high: NModeSurfaceModel,
    *,
    observable: str,
) -> tuple[ModeSubset, ...]:
    selected = _normalized_subset_collection(values)
    low_subsets = set(low.subsets)
    high_subsets = set(high.subsets)
    for subset in selected:
        if subset not in low_subsets:
            raise ValueError(f"Selected {observable} subset {subset} is absent from the low model")
        if subset not in high_subsets:
            raise ValueError(
                f"Selected {observable} subset {subset} is absent from the high model"
            )
        for rank in range(1, len(subset)):
            for proper in itertools.combinations(subset, rank):
                if proper not in low_subsets or proper not in high_subsets:
                    raise ValueError(
                        f"Selected {observable} subset {subset} requires common closure {proper}"
                    )
    return selected


def _require_common_domain(
    low_axes: tuple[np.ndarray, ...],
    high_axes: tuple[np.ndarray, ...],
    subset: ModeSubset,
) -> None:
    if len(low_axes) != len(high_axes):
        raise ValueError(f"Low/high surface rank differs for subset {subset}")
    for position, (low, high) in enumerate(zip(low_axes, high_axes)):
        if low[0] != high[0] or low[-1] != high[-1]:
            raise ValueError(
                f"Low/high Delta domains for mode {subset[position]} need identical bounds"
            )


def _composite_source_lineage(
    low: NModeSurfaceModel,
    high: NModeSurfaceModel,
    *,
    energy_selected: tuple[ModeSubset, ...],
    dipole_selected: tuple[ModeSubset, ...],
) -> dict[str, object]:
    low_source = to_jsonable(low.source_lineage)
    high_source = to_jsonable(high.source_lineage)
    points = tuple(
        sorted(
            set(low_source["point_causal_fingerprints"])
            | set(high_source["point_causal_fingerprints"])
        )
    )
    provider = payload_fingerprint(
        {
            "schema": "pyscf-vscf-multilevel-provider",
            "schema_version": MULTILEVEL_SCHEMA_VERSION,
            "low_provider_scientific_fingerprint": low_source["provider_scientific_fingerprint"],
            "high_provider_scientific_fingerprint": high_source["provider_scientific_fingerprint"],
            "energy_corrected_subsets": [list(value) for value in energy_selected],
            "dipole_corrected_subsets": [list(value) for value in dipole_selected],
            "reference_policy": "preserve-low-level-absolute-reference",
        }
    )
    source: dict[str, object] = {
        "schema": "pyscf-vscf-electronic-source-lineage",
        "schema_version": 1,
        "provider_scientific_fingerprint": provider,
        "point_causal_fingerprints": list(points),
    }
    low_results = low_source.get("result_scientific_fingerprints")
    high_results = high_source.get("result_scientific_fingerprints")
    if isinstance(low_results, Mapping) and isinstance(high_results, Mapping):
        merged_results = _merge_identity_mapping(low_results, high_results)
        if set(merged_results) == set(points):
            source["result_scientific_fingerprints"] = merged_results
    sampling = tuple(
        sorted(
            set(low_source.get("sampling_lineage_fingerprints", ()))
            | set(high_source.get("sampling_lineage_fingerprints", ()))
        )
    )
    if sampling:
        source["sampling_lineage_fingerprints"] = list(sampling)
    return source


def _merge_identity_mapping(
    first: Mapping[str, object],
    second: Mapping[str, object],
) -> dict[str, object]:
    merged = {str(key): value for key, value in first.items()}
    for key, value in second.items():
        normalized = str(key)
        if normalized in merged and merged[normalized] != value:
            raise ValueError("Source lineages disagree for a shared point identity")
        merged[normalized] = value
    return {key: merged[key] for key in sorted(merged)}


def _surface_fingerprint(surface: TensorProductSurface) -> str:
    return payload_fingerprint(surface.numerical_payload())


def _component_max_abs(values: np.ndarray, rank: int) -> tuple[float, ...]:
    trailing = values.shape[rank:]
    reshaped = values.reshape((-1, int(np.prod(trailing)) if trailing else 1))
    return tuple(float(value) for value in np.max(np.abs(reshaped), axis=0))


def _mesh_points(axes: tuple[np.ndarray, ...]) -> np.ndarray:
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([component.reshape(-1) for component in mesh], axis=-1)


def _normalized_subset_collection(values: Sequence[Sequence[int]]) -> tuple[ModeSubset, ...]:
    subsets = tuple(_normalize_subset(value) for value in values)
    if len(set(subsets)) != len(subsets):
        raise ValueError("Corrected mode subsets must be unique")
    return tuple(sorted(subsets, key=lambda value: (len(value), value)))


def _normalize_subset(values: Sequence[int]) -> ModeSubset:
    subset = tuple(operator.index(value) for value in values)
    if not subset or subset != tuple(sorted(set(subset))) or subset[0] < 0:
        raise ValueError("Mode subsets must be non-empty unique increasing indices")
    return subset


def _subset_key(subset: ModeSubset) -> str:
    return ",".join(str(value) for value in subset)


def _nonempty(name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _optional_nonempty(name: str, value: object | None) -> str | None:
    return None if value is None else _nonempty(name, value)


__all__ = [
    "IncrementCompositionDiagnostics",
    "MultilevelCompositionDiagnostics",
    "ObservableCompositionDiagnostics",
    "compose_multilevel_surface",
]
