"""Vector-DMS projection and transition moments for small VCI models."""

from __future__ import annotations

import json
import operator
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from ._arrays import immutable_array
from ._artifacts import atomic_savez_compressed, atomic_write_text
from ._identity import (
    FrozenJSONMapping,
    array_identity,
    payload_fingerprint,
    to_jsonable,
)
from .constants import ATOMIC_DIPOLE_TO_DEBYE, HARTREE_TO_CM
from .coordinates import TriatomicValenceCoordinateMap, coordinate_map_fingerprint
from .kinetic import TriatomicJ0Hamiltonian, TriatomicJacobiTransform
from .nmode import ModeSubset, NModeSurfaceModel, nmode_dms_fingerprint
from .spectra import einstein_a_from_debye, integrated_cross_section_omega
from .vci import (
    Configuration,
    GroundModalBasis,
    NModeGridHamiltonian,
    VCIResult,
)


_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True)
class GridDipoleExpansion:
    """Signed vector-DMS increments evaluated on one exact solver grid."""

    coordinate_ids: tuple[str, ...]
    shape: tuple[int, ...]
    coordinate_grids: tuple[np.ndarray, ...]
    reference_dipole_body_au: np.ndarray
    increment_grids_au: Mapping[ModeSubset, np.ndarray]
    source_dms_fingerprint: str
    hamiltonian_fingerprint: str
    metadata: Mapping[str, object] = field(default_factory=dict)
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("GridDipoleExpansion instances must come from a projection function")
        ids = tuple(str(value).strip() for value in self.coordinate_ids)
        shape = tuple(operator.index(value) for value in self.shape)
        grids = tuple(np.asarray(value, dtype=float) for value in self.coordinate_grids)
        reference = np.asarray(self.reference_dipole_body_au, dtype=float)
        if not ids or len(ids) != len(shape) or len(set(ids)) != len(ids):
            raise ValueError("coordinate_ids and shape must align and be unique")
        if any(value < 2 for value in shape):
            raise ValueError("Every dipole grid dimension must contain at least two points")
        if len(grids) != len(shape) or any(
            grid.shape != (size,) or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0)
            for grid, size in zip(grids, shape)
        ):
            raise ValueError("Dipole coordinate grids must be finite, increasing, and match shape")
        if reference.shape != (3,) or not np.all(np.isfinite(reference)):
            raise ValueError("reference_dipole_body_au must be a finite vector")
        increments = {}
        for raw_subset, raw_values in self.increment_grids_au.items():
            subset = tuple(operator.index(value) for value in raw_subset)
            if (
                not subset
                or subset != tuple(sorted(set(subset)))
                or min(subset) < 0
                or max(subset) >= len(shape)
            ):
                raise ValueError("Dipole increment subsets must be canonical mode indices")
            values = np.asarray(raw_values, dtype=float)
            if values.shape != (*shape, 3) or not np.all(np.isfinite(values)):
                raise ValueError("Every dipole increment must match the complete solver grid")
            increments[subset] = _readonly(values)
        if not increments:
            raise ValueError("At least one dipole increment is required")
        object.__setattr__(self, "coordinate_ids", ids)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "coordinate_grids", tuple(_readonly(value) for value in grids))
        object.__setattr__(self, "reference_dipole_body_au", _readonly(reference))
        object.__setattr__(
            self,
            "increment_grids_au",
            MappingProxyType(
                {
                    subset: increments[subset]
                    for subset in sorted(increments, key=lambda item: (len(item), item))
                }
            ),
        )
        object.__setattr__(
            self,
            "source_dms_fingerprint",
            _nonempty("source_dms_fingerprint", self.source_dms_fingerprint),
        )
        object.__setattr__(
            self,
            "hamiltonian_fingerprint",
            _nonempty("hamiltonian_fingerprint", self.hamiltonian_fingerprint),
        )
        object.__setattr__(self, "metadata", FrozenJSONMapping.from_mapping(self.metadata))

    @property
    def dipole_grid_au(self) -> np.ndarray:
        total = np.broadcast_to(self.reference_dipole_body_au, (*self.shape, 3)).copy()
        for values in self.increment_grids_au.values():
            total += values
        return _readonly(total)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "solver-grid-vector-dms-expansion",
            "schema_version": 1,
            "coordinate_ids": list(self.coordinate_ids),
            "shape": list(self.shape),
            "coordinate_grids": [array_identity(value) for value in self.coordinate_grids],
            "reference_dipole_body_au": array_identity(self.reference_dipole_body_au),
            "increment_grids_au": {
                ",".join(map(str, subset)): array_identity(values)
                for subset, values in self.increment_grids_au.items()
            },
            "source_dms_fingerprint": self.source_dms_fingerprint,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "metadata": to_jsonable(self.metadata),
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


@dataclass(frozen=True)
class ConfigurationDipoleOperator:
    """Vector dipole matrices in one pruned VCI configuration basis."""

    configurations: tuple[Configuration, ...]
    reference_matrices_au: np.ndarray
    increment_matrices_au: Mapping[ModeSubset, np.ndarray]
    grid_dipole_fingerprint: str
    modal_basis_fingerprint: str
    hamiltonian_fingerprint: str
    _construction_token: InitVar[object] = None

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError(
                "ConfigurationDipoleOperator instances must come from a reviewed factory"
            )
        configurations = tuple(_configuration(item) for item in self.configurations)
        if not configurations or len(set(configurations)) != len(configurations):
            raise ValueError("Configurations must be non-empty and unique")
        size = len(configurations)
        reference = _dipole_matrices("reference_matrices_au", self.reference_matrices_au, size)
        increments = {}
        for raw_subset, raw_values in self.increment_matrices_au.items():
            subset = tuple(operator.index(value) for value in raw_subset)
            if not subset or subset != tuple(sorted(set(subset))):
                raise ValueError("Dipole matrix subsets must be canonical")
            increments[subset] = _dipole_matrices(
                f"increment_matrices_au[{subset}]",
                raw_values,
                size,
            )
        if not increments:
            raise ValueError("At least one dipole increment matrix is required")
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "reference_matrices_au", reference)
        object.__setattr__(
            self,
            "increment_matrices_au",
            MappingProxyType(
                {
                    subset: increments[subset]
                    for subset in sorted(increments, key=lambda item: (len(item), item))
                }
            ),
        )
        for name in (
            "grid_dipole_fingerprint",
            "modal_basis_fingerprint",
            "hamiltonian_fingerprint",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))

    @classmethod
    def from_analytic_matrices(
        cls,
        *,
        configurations: tuple[Configuration, ...],
        reference_matrices_au: np.ndarray,
        increment_matrices_au: Mapping[ModeSubset, np.ndarray],
        provenance_label: str,
        modal_basis_fingerprint: str,
        hamiltonian_fingerprint: str,
    ) -> ConfigurationDipoleOperator:
        """Construct an explicitly analytic, non-surface matrix fixture."""

        label = _nonempty("provenance_label", provenance_label)
        return cls(
            configurations=configurations,
            reference_matrices_au=reference_matrices_au,
            increment_matrices_au=increment_matrices_au,
            grid_dipole_fingerprint=payload_fingerprint(
                {
                    "kind": "analytic-configuration-dipole-fixture",
                    "schema_version": 1,
                    "provenance_label": label,
                }
            ),
            modal_basis_fingerprint=modal_basis_fingerprint,
            hamiltonian_fingerprint=hamiltonian_fingerprint,
            _construction_token=_CONSTRUCTION_TOKEN,
        )

    @property
    def matrices_au(self) -> np.ndarray:
        total = np.array(self.reference_matrices_au, copy=True)
        for values in self.increment_matrices_au.values():
            total += values
        return _readonly(total)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "vci-configuration-vector-dipole-operator",
            "schema_version": 1,
            "configurations": [list(item) for item in self.configurations],
            "reference_matrices_au": array_identity(self.reference_matrices_au),
            "increment_matrices_au": {
                ",".join(map(str, subset)): array_identity(values)
                for subset, values in self.increment_matrices_au.items()
            },
            "grid_dipole_fingerprint": self.grid_dipole_fingerprint,
            "modal_basis_fingerprint": self.modal_basis_fingerprint,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


def dump_vci_dipole_projection(
    expansion: GridDipoleExpansion,
    operator_value: ConfigurationDipoleOperator,
    path: Path | str,
) -> None:
    """Serialize a grid DMS expansion and its VCI-basis projection together."""

    if not isinstance(expansion, GridDipoleExpansion) or not isinstance(
        operator_value,
        ConfigurationDipoleOperator,
    ):
        raise TypeError("VCI dipole projections require typed expansion and operator values")
    if operator_value.grid_dipole_fingerprint != expansion.fingerprint():
        raise ValueError("Dipole operator was not projected from the supplied grid expansion")
    arrays: dict[str, np.ndarray] = {
        "grid_reference_dipole_body_au": expansion.reference_dipole_body_au,
        "operator_configurations": np.asarray(operator_value.configurations, dtype="<i8"),
        "operator_reference_matrices_au": operator_value.reference_matrices_au,
    }
    for mode, values in enumerate(expansion.coordinate_grids):
        arrays[f"grid_coordinate_{mode}"] = values
    grid_subsets = []
    for index, (subset, values) in enumerate(expansion.increment_grids_au.items()):
        arrays[f"grid_increment_{index}"] = values
        grid_subsets.append(list(subset))
    operator_subsets = []
    for index, (subset, values) in enumerate(operator_value.increment_matrices_au.items()):
        arrays[f"operator_increment_{index}"] = values
        operator_subsets.append(list(subset))
    manifest = {
        "schema": "pyscf-vscf-vci-dipole-projection",
        "schema_version": 1,
        "grid": {
            "coordinate_ids": list(expansion.coordinate_ids),
            "shape": list(expansion.shape),
            "increment_subsets": grid_subsets,
            "source_dms_fingerprint": expansion.source_dms_fingerprint,
            "hamiltonian_fingerprint": expansion.hamiltonian_fingerprint,
            "metadata": to_jsonable(expansion.metadata),
            "fingerprint": expansion.fingerprint(),
        },
        "operator": {
            "increment_subsets": operator_subsets,
            "grid_dipole_fingerprint": operator_value.grid_dipole_fingerprint,
            "modal_basis_fingerprint": operator_value.modal_basis_fingerprint,
            "hamiltonian_fingerprint": operator_value.hamiltonian_fingerprint,
            "fingerprint": operator_value.fingerprint(),
        },
    }
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    atomic_savez_compressed(path, arrays)


def load_vci_dipole_projection(
    path: Path | str,
) -> tuple[GridDipoleExpansion, ConfigurationDipoleOperator]:
    """Load and fingerprint-check a retained grid/configuration DMS projection."""

    with np.load(Path(path), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        if manifest.get("schema") != "pyscf-vscf-vci-dipole-projection":
            raise ValueError("Not a pyscf-vscf VCI dipole projection artifact")
        if int(manifest.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported VCI dipole projection artifact schema version")
        grid_data = manifest["grid"]
        operator_data = manifest["operator"]
        expansion = GridDipoleExpansion(
            coordinate_ids=tuple(grid_data["coordinate_ids"]),
            shape=tuple(grid_data["shape"]),
            coordinate_grids=tuple(
                np.asarray(archive[f"grid_coordinate_{mode}"], dtype=float)
                for mode in range(len(grid_data["shape"]))
            ),
            reference_dipole_body_au=np.asarray(
                archive["grid_reference_dipole_body_au"], dtype=float
            ),
            increment_grids_au={
                tuple(subset): np.asarray(archive[f"grid_increment_{index}"], dtype=float)
                for index, subset in enumerate(grid_data["increment_subsets"])
            },
            source_dms_fingerprint=grid_data["source_dms_fingerprint"],
            hamiltonian_fingerprint=grid_data["hamiltonian_fingerprint"],
            metadata=grid_data["metadata"],
            _construction_token=_CONSTRUCTION_TOKEN,
        )
        operator_value = ConfigurationDipoleOperator(
            configurations=tuple(
                tuple(int(component) for component in row)
                for row in np.asarray(archive["operator_configurations"], dtype=np.int64)
            ),
            reference_matrices_au=np.asarray(
                archive["operator_reference_matrices_au"], dtype=float
            ),
            increment_matrices_au={
                tuple(subset): np.asarray(archive[f"operator_increment_{index}"], dtype=float)
                for index, subset in enumerate(operator_data["increment_subsets"])
            },
            grid_dipole_fingerprint=operator_data["grid_dipole_fingerprint"],
            modal_basis_fingerprint=operator_data["modal_basis_fingerprint"],
            hamiltonian_fingerprint=operator_data["hamiltonian_fingerprint"],
            _construction_token=_CONSTRUCTION_TOKEN,
        )
    if (
        grid_data.get("fingerprint") != expansion.fingerprint()
        or operator_data.get("fingerprint") != operator_value.fingerprint()
        or operator_value.grid_dipole_fingerprint != expansion.fingerprint()
    ):
        raise ValueError("Serialized VCI dipole projection fingerprints do not match")
    return expansion, operator_value


@dataclass(frozen=True)
class VCITransitionMoment:
    lower_state: int
    upper_state: int
    frequency_Eh: float
    transition_dipole_body_au: np.ndarray
    manual_review: bool
    dipole_operator_fingerprint: str
    vci_result_fingerprint: str
    _construction_token: InitVar[object] = None
    frequency_cm: float = field(init=False)
    transition_dipole_body_debye: np.ndarray = field(init=False)
    polarized_integrated_cross_sections_omega_m2_per_s: np.ndarray = field(init=False)
    isotropic_integrated_cross_section_omega_m2_per_s: float = field(init=False)
    einstein_a_s: float = field(init=False)

    def __post_init__(self, _construction_token: object) -> None:
        if _construction_token is not _CONSTRUCTION_TOKEN:
            raise TypeError("VCITransitionMoment instances must come from vci_transition_moments")
        lower = operator.index(self.lower_state)
        upper = operator.index(self.upper_state)
        frequency_Eh = float(self.frequency_Eh)
        dipole_au = np.asarray(self.transition_dipole_body_au, dtype=float)
        if lower < 0 or upper <= lower:
            raise ValueError("Transition state indices must satisfy 0 <= lower < upper")
        if not np.isfinite(frequency_Eh) or frequency_Eh <= 0.0:
            raise ValueError("frequency_Eh must be positive and finite")
        if dipole_au.shape != (3,) or not np.all(np.isfinite(dipole_au)):
            raise ValueError("Transition dipoles must be finite three-component vectors")
        frequency_cm = frequency_Eh * HARTREE_TO_CM
        dipole_debye = dipole_au * ATOMIC_DIPOLE_TO_DEBYE
        polarized = np.array(
            [
                integrated_cross_section_omega(
                    component,
                    frequency_cm,
                    orientation_factor=1.0,
                )
                for component in dipole_debye
            ]
        )
        norm_debye = float(np.linalg.norm(dipole_debye))
        isotropic = integrated_cross_section_omega(norm_debye, frequency_cm)
        einstein = einstein_a_from_debye(norm_debye, frequency_cm)
        object.__setattr__(self, "lower_state", lower)
        object.__setattr__(self, "upper_state", upper)
        object.__setattr__(self, "frequency_Eh", frequency_Eh)
        object.__setattr__(self, "frequency_cm", frequency_cm)
        object.__setattr__(self, "transition_dipole_body_au", _readonly(dipole_au))
        object.__setattr__(self, "transition_dipole_body_debye", _readonly(dipole_debye))
        object.__setattr__(
            self,
            "polarized_integrated_cross_sections_omega_m2_per_s",
            _readonly(polarized),
        )
        object.__setattr__(
            self,
            "isotropic_integrated_cross_section_omega_m2_per_s",
            isotropic,
        )
        object.__setattr__(self, "einstein_a_s", einstein)
        object.__setattr__(self, "manual_review", bool(self.manual_review))
        object.__setattr__(
            self,
            "dipole_operator_fingerprint",
            _nonempty("dipole_operator_fingerprint", self.dipole_operator_fingerprint),
        )
        object.__setattr__(
            self,
            "vci_result_fingerprint",
            _nonempty("vci_result_fingerprint", self.vci_result_fingerprint),
        )

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "vci-transition-moment",
            "schema_version": 1,
            "lower_state": self.lower_state,
            "upper_state": self.upper_state,
            "frequency_Eh": self.frequency_Eh,
            "transition_dipole_body_au": array_identity(self.transition_dipole_body_au),
            "manual_review": self.manual_review,
            "dipole_operator_fingerprint": self.dipole_operator_fingerprint,
            "vci_result_fingerprint": self.vci_result_fingerprint,
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


def dump_vci_transition_moments(
    moments: Sequence[VCITransitionMoment],
    path: Path | str,
) -> None:
    """Serialize typed VCI transition moments as canonical JSON."""

    records = tuple(moments)
    if not records or any(not isinstance(value, VCITransitionMoment) for value in records):
        raise TypeError("Transition artifacts require VCITransitionMoment values")
    if len({(value.lower_state, value.upper_state) for value in records}) != len(records):
        raise ValueError("Transition state pairs must be unique")
    if len({value.vci_result_fingerprint for value in records}) != 1:
        raise ValueError("Transition moments must share one VCI result")
    if len({value.dipole_operator_fingerprint for value in records}) != 1:
        raise ValueError("Transition moments must share one dipole operator")
    payload = {
        "schema": "pyscf-vscf-vci-transition-moments",
        "schema_version": 1,
        "moments": [
            {
                "lower_state": value.lower_state,
                "upper_state": value.upper_state,
                "frequency_Eh": value.frequency_Eh,
                "transition_dipole_body_au": value.transition_dipole_body_au.tolist(),
                "manual_review": value.manual_review,
                "dipole_operator_fingerprint": value.dipole_operator_fingerprint,
                "vci_result_fingerprint": value.vci_result_fingerprint,
                "fingerprint": value.fingerprint(),
            }
            for value in records
        ],
    }
    atomic_write_text(
        path,
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
    )


def load_vci_transition_moments(path: Path | str) -> tuple[VCITransitionMoment, ...]:
    """Load and fingerprint-check a transition artifact."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "pyscf-vscf-vci-transition-moments":
        raise ValueError("Not a pyscf-vscf VCI transition artifact")
    if int(payload.get("schema_version", -1)) != 1:
        raise ValueError("Unsupported VCI transition artifact schema version")
    moments = tuple(
        VCITransitionMoment(
            lower_state=value["lower_state"],
            upper_state=value["upper_state"],
            frequency_Eh=value["frequency_Eh"],
            transition_dipole_body_au=np.asarray(value["transition_dipole_body_au"], dtype=float),
            manual_review=value["manual_review"],
            dipole_operator_fingerprint=value["dipole_operator_fingerprint"],
            vci_result_fingerprint=value["vci_result_fingerprint"],
            _construction_token=_CONSTRUCTION_TOKEN,
        )
        for value in payload.get("moments", ())
    )
    if not moments or any(
        value["fingerprint"] != moment.fingerprint()
        for value, moment in zip(payload["moments"], moments)
    ):
        raise ValueError("Serialized transition fingerprints do not match")
    if len({(value.lower_state, value.upper_state) for value in moments}) != len(moments):
        raise ValueError("Serialized transition state pairs must be unique")
    if len({value.vci_result_fingerprint for value in moments}) != 1:
        raise ValueError("Serialized transitions do not share one VCI result")
    if len({value.dipole_operator_fingerprint for value in moments}) != 1:
        raise ValueError("Serialized transitions do not share one dipole operator")
    return moments


def nmode_dipole_on_grid(
    model: NModeSurfaceModel,
    hamiltonian: NModeGridHamiltonian,
) -> GridDipoleExpansion:
    """Evaluate each signed n-mode DMS increment on a rectilinear solver grid."""

    if model.coordinate_ids != hamiltonian.coordinate_ids:
        raise ValueError("DMS and Hamiltonian coordinate IDs do not match")
    if model.coordinate_units != ("angstrom",) * model.n_modes:
        raise ValueError("Rectilinear n-mode dipole projection requires Angstrom coordinates")
    if hamiltonian.model.coordinate_map_fingerprint is None:
        raise ValueError("Rectilinear Hamiltonian lacks a coordinate-map fingerprint")
    if model.coordinate_map_fingerprint != hamiltonian.model.coordinate_map_fingerprint:
        raise ValueError("DMS and Hamiltonian coordinate-map fingerprints do not match")
    grids = hamiltonian.model.coordinates
    increments = {
        subset: _increment_on_complete_grid(surface, subset, grids)
        for subset, surface in model.dipole_increments.items()
    }
    return GridDipoleExpansion(
        coordinate_ids=hamiltonian.coordinate_ids,
        shape=hamiltonian.shape,
        coordinate_grids=tuple(np.asarray(value, dtype=float) for value in grids),
        reference_dipole_body_au=model.reference_dipole_body_au,
        increment_grids_au=increments,
        source_dms_fingerprint=nmode_dms_fingerprint(model),
        hamiltonian_fingerprint=hamiltonian.fingerprint(),
        metadata={"projection": "rectilinear-nmode"},
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def nmode_dipole_on_product_grid(
    model: NModeSurfaceModel,
    coordinate_grids: Sequence[Sequence[float]],
    *,
    hamiltonian_fingerprint: str,
) -> GridDipoleExpansion:
    """Evaluate a vector DMS on an explicit product grid in its source coordinates."""

    if not isinstance(model, NModeSurfaceModel):
        raise TypeError("Product-grid DMS projection requires an NModeSurfaceModel")
    grids = tuple(np.asarray(value, dtype=float) for value in coordinate_grids)
    if len(grids) != model.n_modes or any(
        value.ndim != 1
        or value.size < 2
        or np.any(~np.isfinite(value))
        or np.any(np.diff(value) <= 0.0)
        for value in grids
    ):
        raise ValueError(
            "Product coordinate grids must be finite, increasing one-dimensional axes"
        )
    increments = {
        subset: _increment_on_complete_grid(surface, subset, grids)
        for subset, surface in model.dipole_increments.items()
    }
    return GridDipoleExpansion(
        coordinate_ids=model.coordinate_ids,
        shape=tuple(value.size for value in grids),
        coordinate_grids=grids,
        reference_dipole_body_au=model.reference_dipole_body_au,
        increment_grids_au=increments,
        source_dms_fingerprint=nmode_dms_fingerprint(model),
        hamiltonian_fingerprint=_nonempty("hamiltonian_fingerprint", hamiltonian_fingerprint),
        metadata={"projection": "explicit-product-grid"},
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def dipole_on_jacobi_grid(
    model: NModeSurfaceModel,
    coordinate_map: TriatomicValenceCoordinateMap,
    transform: TriatomicJacobiTransform,
    hamiltonian: TriatomicJ0Hamiltonian,
) -> GridDipoleExpansion:
    """Project signed valence DMS increments onto one bound Jacobi grid."""

    if model.n_modes != 3 or model.coordinate_units != (
        "angstrom",
        "angstrom",
        "radian",
    ):
        raise ValueError("Jacobi DMS projection requires three valence coordinates")
    map_fingerprint = coordinate_map_fingerprint(coordinate_map)
    if map_fingerprint != model.coordinate_map_fingerprint:
        raise ValueError("Source coordinate map fingerprint does not match the DMS model")
    if tuple(coordinate_map.coordinate_ids) != model.coordinate_ids:
        raise ValueError("Source coordinate IDs do not match the DMS model")
    if not np.array_equal(coordinate_map.reference_values, model.reference_values):
        raise ValueError("Source coordinate reference does not match the DMS model")
    expected_order = (
        coordinate_map.outer_atom_1,
        coordinate_map.center_atom,
        coordinate_map.outer_atom_2,
    )
    kinetic = hamiltonian.kinetic
    if transform.atom_indices != expected_order or kinetic.atom_indices != expected_order:
        raise ValueError("Jacobi transform atom order does not match the ordered valence map")
    if transform.masses_amu != kinetic.masses_amu:
        raise ValueError("Jacobi transform and Hamiltonian masses do not match")
    meshes = np.meshgrid(*kinetic.coordinate_grids, indexing="ij")
    valence = transform.jacobi_to_valence(np.stack(meshes, axis=-1))
    increments = {}
    for subset, surface in model.dipole_increments.items():
        values = surface.evaluate(valence[..., list(subset)])
        increments[subset] = np.asarray(values, dtype=float)
    return GridDipoleExpansion(
        coordinate_ids=kinetic.coordinate_ids,
        shape=kinetic.shape,
        coordinate_grids=tuple(kinetic.coordinate_grids),
        reference_dipole_body_au=model.reference_dipole_body_au,
        increment_grids_au=increments,
        source_dms_fingerprint=nmode_dms_fingerprint(model),
        hamiltonian_fingerprint=hamiltonian.fingerprint(),
        metadata={
            "projection": "ordered-valence-to-jacobi",
            "coordinate_map_fingerprint": map_fingerprint,
            "transform_fingerprint": transform.fingerprint(),
        },
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def build_vci_dipole_operator(
    expansion: GridDipoleExpansion,
    modal_basis: GroundModalBasis,
    vci_result: VCIResult,
) -> ConfigurationDipoleOperator:
    """Project a grid DMS expansion into the exact VCI configuration basis."""

    if expansion.hamiltonian_fingerprint != vci_result.hamiltonian_fingerprint:
        raise ValueError("Grid DMS was not generated for the VCI Hamiltonian")
    if modal_basis.fingerprint() != vci_result.modal_basis_fingerprint:
        raise ValueError("Modal basis does not match the VCI result")
    if expansion.coordinate_ids != modal_basis.coordinate_ids:
        raise ValueError("Grid DMS coordinate IDs do not match the modal basis")
    if expansion.shape != tuple(values.shape[0] for values in modal_basis.modals):
        raise ValueError("Grid DMS dimensions do not match the modal basis")
    basis_values = _product_basis_values(modal_basis.modals, vci_result.configurations)
    identity = np.eye(len(vci_result.configurations))
    reference = identity[:, :, None] * expansion.reference_dipole_body_au
    increments = {
        subset: _project_grid_dipole(values, basis_values)
        for subset, values in expansion.increment_grids_au.items()
    }
    return ConfigurationDipoleOperator(
        configurations=vci_result.configurations,
        reference_matrices_au=reference,
        increment_matrices_au=increments,
        grid_dipole_fingerprint=expansion.fingerprint(),
        modal_basis_fingerprint=modal_basis.fingerprint(),
        hamiltonian_fingerprint=vci_result.hamiltonian_fingerprint,
        _construction_token=_CONSTRUCTION_TOKEN,
    )


def vci_transition_moments(
    dipole_operator: ConfigurationDipoleOperator,
    vci_result: VCIResult,
    *,
    lower_state: int = 0,
    upper_states: Sequence[int] | None = None,
) -> tuple[VCITransitionMoment, ...]:
    """Contract a signed vector dipole operator with VCI eigenvectors."""

    if dipole_operator.configurations != vci_result.configurations:
        raise ValueError("Dipole operator configurations do not match the VCI result")
    if dipole_operator.modal_basis_fingerprint != vci_result.modal_basis_fingerprint:
        raise ValueError("Dipole operator modal basis does not match the VCI result")
    if dipole_operator.hamiltonian_fingerprint != vci_result.hamiltonian_fingerprint:
        raise ValueError("Dipole operator Hamiltonian does not match the VCI result")
    lower = operator.index(lower_state)
    if lower < 0 or lower >= vci_result.energies_Eh.size:
        raise ValueError("lower_state is outside the returned VCI states")
    if upper_states is None:
        uppers = tuple(range(lower + 1, vci_result.energies_Eh.size))
    else:
        uppers = tuple(operator.index(value) for value in upper_states)
        if (
            not uppers
            or len(set(uppers)) != len(uppers)
            or any(value <= lower or value >= vci_result.energies_Eh.size for value in uppers)
        ):
            raise ValueError("upper_states must be unique returned states above lower_state")
    matrices = dipole_operator.matrices_au
    lower_coefficients = vci_result.coefficients[:, lower]
    records = []
    for upper in uppers:
        upper_coefficients = vci_result.coefficients[:, upper]
        dipole_au = np.einsum(
            "i,ijc,j->c",
            lower_coefficients,
            matrices,
            upper_coefficients,
            optimize=True,
        )
        frequency_Eh = float(vci_result.energies_Eh[upper] - vci_result.energies_Eh[lower])
        manual = (
            vci_result.assignments[lower].manual_review
            or vci_result.assignments[upper].manual_review
        )
        records.append(
            VCITransitionMoment(
                lower_state=lower,
                upper_state=upper,
                frequency_Eh=frequency_Eh,
                transition_dipole_body_au=dipole_au,
                manual_review=manual,
                dipole_operator_fingerprint=dipole_operator.fingerprint(),
                vci_result_fingerprint=vci_result.fingerprint(),
                _construction_token=_CONSTRUCTION_TOKEN,
            )
        )
    return tuple(records)


def _increment_on_complete_grid(
    surface,
    subset: ModeSubset,
    grids: tuple[np.ndarray, ...],
) -> np.ndarray:
    meshes = np.meshgrid(*(grids[mode] for mode in subset), indexing="ij")
    points = np.stack(meshes, axis=-1)
    values = np.asarray(surface.evaluate(points), dtype=float)
    view = [1] * len(grids) + [3]
    for position, mode in enumerate(subset):
        view[mode] = grids[mode].size
    return np.broadcast_to(values.reshape(view), (*[grid.size for grid in grids], 3)).copy()


def _product_basis_values(
    modal_bases: tuple[np.ndarray, ...],
    configurations: tuple[Configuration, ...],
) -> np.ndarray:
    shape = tuple(values.shape[0] for values in modal_bases)
    output = np.empty((int(np.prod(shape)), len(configurations)), dtype=float)
    for column, configuration in enumerate(configurations):
        factors = [modal_bases[mode][:, quantum] for mode, quantum in enumerate(configuration)]
        product = factors[0]
        for factor in factors[1:]:
            product = np.multiply.outer(product, factor)
        output[:, column] = np.asarray(product).reshape(-1)
    if not np.allclose(output.T @ output, np.eye(len(configurations)), atol=3e-11):
        raise ValueError("Selected product configurations are not orthonormal")
    return output


def _project_grid_dipole(values: np.ndarray, basis_values: np.ndarray) -> np.ndarray:
    flattened = np.asarray(values, dtype=float).reshape((-1, 3))
    return _readonly(
        np.stack(
            [
                basis_values.T @ (flattened[:, component, None] * basis_values)
                for component in range(3)
            ],
            axis=-1,
        )
    )


def _dipole_matrices(name: str, values: np.ndarray, size: int) -> np.ndarray:
    matrices = np.asarray(values, dtype=float)
    if matrices.shape != (size, size, 3) or not np.all(np.isfinite(matrices)):
        raise ValueError(f"{name} must have shape {(size, size, 3)} and be finite")
    if not np.allclose(matrices, np.swapaxes(matrices, 0, 1), atol=3e-11):
        raise ValueError(f"{name} must be Hermitian")
    return _readonly(matrices)


def _configuration(values: Sequence[int]) -> Configuration:
    configuration = tuple(operator.index(value) for value in values)
    if not configuration or any(value < 0 for value in configuration):
        raise ValueError("Configurations must contain non-negative quanta")
    return configuration


def _nonempty(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _readonly(values: np.ndarray) -> np.ndarray:
    return immutable_array(values)


__all__ = [
    "ConfigurationDipoleOperator",
    "GridDipoleExpansion",
    "VCITransitionMoment",
    "build_vci_dipole_operator",
    "dipole_on_jacobi_grid",
    "dump_vci_dipole_projection",
    "dump_vci_transition_moments",
    "load_vci_dipole_projection",
    "load_vci_transition_moments",
    "nmode_dipole_on_grid",
    "nmode_dipole_on_product_grid",
    "vci_transition_moments",
]
