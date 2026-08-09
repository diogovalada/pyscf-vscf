"""Electronic point contracts and the released scan-workflow helper API."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .molecule import Molecule
from ._arrays import immutable_array
from ._identity import (
    FrozenJSONMapping,
    float64_array_identity,
    immutable_json_mapping,
    payload_fingerprint,
    to_jsonable,
)

AU_DIPOLE_TO_DEBYE = 2.541746
POINT_SCHEMA_VERSION = 1


class EnergyDipoleEvaluator(Protocol):
    """Callable boundary used by PES/DMS scan workflows.

    The returned energy must be in Hartree. The dipole must be a finite
    three-component Cartesian vector in Debye, expressed in the same fixed
    frame as the input geometry. When persisting a grid from a custom
    evaluator, pass a stable, non-``pyscf`` ``backend_identity`` to the cache
    helper and use the same identifier when loading it.
    """

    def __call__(self, molecule: Molecule, settings: object) -> tuple[float, np.ndarray]: ...


@dataclass(frozen=True)
class ElectronicPointRequest:
    """One immutable electronic evaluation request.

    Isotope labels, isotope masses, coordinate-map definitions, and coordinate
    values are deliberately absent. Exact Cartesian geometry and ordered
    nuclear charges define the electronic system; sampling lineage is retained
    separately by n-mode planning.
    """

    nuclear_charges: tuple[int, ...]
    coordinates_A: np.ndarray
    charge: int = 0
    spin: int = 0
    electronic_state: str = "ground"
    requested_properties: tuple[str, ...] = ("energy", "dipole")
    field_au: np.ndarray | None = None
    field_origin_A: np.ndarray | None = None
    schema_version: int = POINT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        charges = tuple(operator.index(value) for value in self.nuclear_charges)
        if not charges or any(value <= 0 for value in charges):
            raise ValueError("nuclear_charges must contain positive atomic numbers")
        coordinates = _finite_float64_array("coordinates_A", self.coordinates_A, ndim=2)
        if coordinates.shape != (len(charges), 3):
            raise ValueError("coordinates_A must have shape (len(nuclear_charges), 3)")
        state = str(self.electronic_state).strip()
        if not state:
            raise ValueError("electronic_state must be non-empty")
        properties = tuple(
            sorted({str(value).strip().lower() for value in self.requested_properties})
        )
        if not properties or any(not value for value in properties):
            raise ValueError("requested_properties must contain non-empty names")
        unsupported = set(properties) - {"energy", "dipole"}
        if unsupported:
            raise ValueError(f"Unsupported electronic properties: {sorted(unsupported)!r}")
        schema = operator.index(self.schema_version)
        if schema != POINT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported point schema {schema}; expected {POINT_SCHEMA_VERSION}")

        field_vector = None
        field_origin = None
        if self.field_au is not None:
            field_vector = _finite_float64_array("field_au", self.field_au, shape=(3,))
            origin_value = np.zeros(3) if self.field_origin_A is None else self.field_origin_A
            field_origin = _finite_float64_array("field_origin_A", origin_value, shape=(3,))
        elif self.field_origin_A is not None:
            raise ValueError("field_origin_A requires field_au")

        object.__setattr__(self, "nuclear_charges", charges)
        object.__setattr__(self, "coordinates_A", immutable_array(coordinates, dtype="<f8"))
        object.__setattr__(self, "charge", operator.index(self.charge))
        object.__setattr__(self, "spin", operator.index(self.spin))
        object.__setattr__(self, "electronic_state", state)
        object.__setattr__(self, "requested_properties", properties)
        object.__setattr__(
            self,
            "field_au",
            None if field_vector is None else immutable_array(field_vector, dtype="<f8"),
        )
        object.__setattr__(
            self,
            "field_origin_A",
            None if field_origin is None else immutable_array(field_origin, dtype="<f8"),
        )
        object.__setattr__(self, "schema_version", schema)

    def causal_payload(self, provider_scientific_fingerprint: str) -> dict[str, object]:
        """Return the complete electronic causal identity payload."""

        provider = str(provider_scientific_fingerprint).strip()
        if not provider:
            raise ValueError("provider_scientific_fingerprint must be non-empty")
        field_payload = None
        if self.field_au is not None:
            field_payload = {
                "vector_au": float64_array_identity(self.field_au),
                "origin_A": float64_array_identity(self.field_origin_A),
            }
        return {
            "schema": "pyscf-vscf-electronic-point",
            "schema_version": self.schema_version,
            "nuclear_charges": list(self.nuclear_charges),
            "coordinates_A": float64_array_identity(self.coordinates_A),
            "charge": self.charge,
            "spin": self.spin,
            "electronic_state": self.electronic_state,
            "requested_properties": list(self.requested_properties),
            "electric_field": field_payload,
            "provider_scientific_fingerprint": provider,
        }

    def causal_fingerprint(self, provider_scientific_fingerprint: str) -> str:
        """Return the electronic causal identity for this provider."""

        return payload_fingerprint(self.causal_payload(provider_scientific_fingerprint))

    def fingerprint(self, provider_scientific_fingerprint: str) -> str:
        """Compatibility spelling for :meth:`causal_fingerprint`."""

        return self.causal_fingerprint(provider_scientific_fingerprint)


@dataclass(frozen=True)
class ElectronicResult:
    """Raw scientific values with separate execution diagnostics and provenance."""

    total_energy_Eh: float
    dipole_au: np.ndarray | None
    converged: bool
    point_causal_fingerprint: str
    provider_scientific_fingerprint: str
    dipole_unit: str | None = None
    dipole_frame: str | None = None
    scientific_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    execution_diagnostics: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energy = float(self.total_energy_Eh)
        if not np.isfinite(energy):
            raise ValueError("total_energy_Eh must be finite")
        point = str(self.point_causal_fingerprint).strip()
        provider = str(self.provider_scientific_fingerprint).strip()
        if not point or not provider:
            raise ValueError("Point and provider fingerprints must be non-empty")
        dipole = None
        unit = None if self.dipole_unit is None else str(self.dipole_unit).strip()
        frame_name = None if self.dipole_frame is None else str(self.dipole_frame).strip()
        if self.dipole_au is not None:
            dipole = _finite_float64_array("dipole_au", self.dipole_au, shape=(3,))
            if not unit or not frame_name:
                raise ValueError("A retained dipole requires explicit unit and frame")
            if unit != "atomic_unit":
                raise ValueError("dipole_au requires dipole_unit='atomic_unit'")
            if frame_name != "input_cartesian":
                raise ValueError("Electronic dipoles must remain in the input Cartesian frame")
        elif unit is not None or frame_name is not None:
            raise ValueError("Dipole unit and frame require a retained dipole")

        object.__setattr__(self, "total_energy_Eh", energy)
        object.__setattr__(
            self,
            "dipole_au",
            None if dipole is None else immutable_array(dipole, dtype="<f8"),
        )
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "point_causal_fingerprint", point)
        object.__setattr__(self, "provider_scientific_fingerprint", provider)
        object.__setattr__(self, "dipole_unit", unit)
        object.__setattr__(self, "dipole_frame", frame_name)
        object.__setattr__(
            self,
            "scientific_diagnostics",
            immutable_json_mapping(self.scientific_diagnostics),
        )
        object.__setattr__(
            self,
            "execution_diagnostics",
            immutable_json_mapping(self.execution_diagnostics),
        )
        object.__setattr__(self, "provenance", immutable_json_mapping(self.provenance))

    @property
    def provider_fingerprint(self) -> str:
        """Compatibility alias for the provider scientific fingerprint."""

        return self.provider_scientific_fingerprint

    @property
    def point_fingerprint(self) -> str:
        """Compatibility alias for the point causal fingerprint."""

        return self.point_causal_fingerprint

    def scientific_payload(self) -> dict[str, object]:
        """Return only values that define the scientific result."""

        return {
            "schema": "pyscf-vscf-electronic-result-scientific",
            "schema_version": 1,
            "total_energy_Eh": self.total_energy_Eh,
            "dipole_au": (
                None if self.dipole_au is None else float64_array_identity(self.dipole_au)
            ),
            "dipole_unit": self.dipole_unit,
            "dipole_frame": self.dipole_frame,
            "converged": self.converged,
            "point_causal_fingerprint": self.point_causal_fingerprint,
            "provider_scientific_fingerprint": self.provider_scientific_fingerprint,
            "scientific_diagnostics": to_jsonable(self.scientific_diagnostics),
        }

    def scientific_fingerprint(self) -> str:
        """Fingerprint the scientific values without execution provenance."""

        return payload_fingerprint(self.scientific_payload())

    def content_payload(self) -> dict[str, object]:
        """Return every retained serialized field for integrity checking."""

        return {
            "schema": "pyscf-vscf-electronic-result-content",
            "schema_version": 1,
            "scientific": self.scientific_payload(),
            "execution_diagnostics": to_jsonable(self.execution_diagnostics),
            "provenance": to_jsonable(self.provenance),
        }

    def content_fingerprint(self) -> str:
        """Fingerprint every retained result field."""

        return payload_fingerprint(self.content_payload())


@runtime_checkable
class ElectronicProvider(Protocol):
    """Protocol implemented by electronic-structure point evaluators."""

    def scientific_settings_payload(self) -> Mapping[str, object]: ...

    def execution_provenance(self) -> Mapping[str, object]: ...

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult: ...


def provider_scientific_fingerprint(provider: ElectronicProvider) -> str:
    """Fingerprint every calculation-defining provider setting."""

    payload = dict(provider.scientific_settings_payload())
    if not payload:
        raise ValueError("Provider scientific settings payload must not be empty")
    return payload_fingerprint(payload)


def provider_fingerprint(provider: ElectronicProvider) -> str:
    """Compatibility spelling for :func:`provider_scientific_fingerprint`."""

    return provider_scientific_fingerprint(provider)


def energy_dipole(molecule: Molecule, cfg: object) -> tuple[float, np.ndarray]:
    """Run one PySCF-backed single point and return energy plus dipole in Debye."""

    from .backends import pyscf as pyscf_backend
    from .settings import coerce_es_settings

    settings = coerce_es_settings(cfg)
    pmol = pyscf_backend.molecule_to_pyscf(molecule, settings.basis)
    mf = pyscf_backend.make_mean_field(pmol, settings)
    energy = float(mf.e_tot)
    dm = mf.make_rdm1()
    try:
        mu_au = mf.dip_moment(dm=dm, unit="au", verbose=0)
    except TypeError:
        mu_au = mf.dip_moment(unit="au", verbose=0)
    return energy, np.asarray(mu_au, dtype=float) * AU_DIPOLE_TO_DEBYE


def _finite_float64_array(
    name: str,
    values: object,
    *,
    ndim: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    array = np.asarray(values, dtype="<f8")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


__all__ = [
    "AU_DIPOLE_TO_DEBYE",
    "ElectronicPointRequest",
    "ElectronicProvider",
    "ElectronicResult",
    "EnergyDipoleEvaluator",
    "FrozenJSONMapping",
    "POINT_SCHEMA_VERSION",
    "energy_dipole",
    "provider_fingerprint",
    "provider_scientific_fingerprint",
]
