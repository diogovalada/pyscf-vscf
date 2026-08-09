"""Backend-neutral coordinate maps and released local-mode scan helpers."""

from __future__ import annotations

import operator
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from ._arrays import immutable_array
from ._identity import canonical_json, float64_array_identity, payload_fingerprint


COORDINATE_MAP_SCHEMA_VERSION = 1
DEFAULT_FRAME_SINE_TOLERANCE = 1e-8


@runtime_checkable
class CoordinateMap(Protocol):
    """Joint map between internal-coordinate values and Cartesian geometry."""

    coordinate_ids: tuple[str, ...]
    units: tuple[str, ...]
    reference_values: np.ndarray
    active_atoms: tuple[int, ...]
    inactive_atoms: tuple[int, ...]
    frame_to_lab: np.ndarray

    def to_cartesian(self, q: np.ndarray) -> np.ndarray: ...

    def values_from_cartesian(self, xyz: np.ndarray) -> np.ndarray: ...

    def vector_to_body(self, vector_lab: np.ndarray) -> np.ndarray: ...

    def vector_to_lab(self, vector_body: np.ndarray) -> np.ndarray: ...

    def fingerprint_payload(self) -> Mapping[str, object]: ...


def coordinate_map_fingerprint(coordinate_map: CoordinateMap) -> str:
    """Return the deterministic scientific identity of a coordinate map."""

    return payload_fingerprint(dict(coordinate_map.fingerprint_payload()))


@dataclass(frozen=True)
class Bond:
    """A bond defined by two zero-based atom indices."""

    i: int
    j: int

    @property
    def O(self) -> int:  # noqa: E743 - retained compatibility spelling
        return self.i

    @property
    def H(self) -> int:
        return self.j


@dataclass(frozen=True)
class BondLengthCoordinateMap:
    """Set one ordered bond length along its fixed reference direction."""

    reference_geometry_A: np.ndarray
    bond: Bond
    coordinate_id: str = "r"
    reference_frame_to_lab: np.ndarray | None = None

    def __post_init__(self) -> None:
        reference = _validated_coordinates(self.reference_geometry_A)
        bond = Bond(
            _strict_atom_index("bond.i", self.bond.i),
            _strict_atom_index("bond.j", self.bond.j),
        )
        _validate_bond(bond, reference.shape[0])
        coordinate_id = str(self.coordinate_id).strip()
        if not coordinate_id:
            raise ValueError("coordinate_id must be non-empty")
        vector = reference[bond.j] - reference[bond.i]
        length = float(np.linalg.norm(vector))
        if length <= 0.0:
            raise ValueError(f"Zero-length bond {bond}")
        frame = (
            np.eye(3)
            if self.reference_frame_to_lab is None
            else _validated_frame(self.reference_frame_to_lab)
        )
        active = (bond.i, bond.j)

        object.__setattr__(self, "reference_geometry_A", immutable_array(reference))
        object.__setattr__(self, "bond", bond)
        object.__setattr__(self, "coordinate_id", coordinate_id)
        object.__setattr__(self, "reference_frame_to_lab", immutable_array(frame))
        object.__setattr__(self, "coordinate_ids", (coordinate_id,))
        object.__setattr__(self, "units", ("angstrom",))
        object.__setattr__(self, "reference_values", immutable_array([length], dtype="<f8"))
        object.__setattr__(self, "active_atoms", active)
        object.__setattr__(
            self,
            "inactive_atoms",
            tuple(index for index in range(reference.shape[0]) if index not in active),
        )
        object.__setattr__(self, "frame_to_lab", immutable_array(frame))

    def to_cartesian(self, q: np.ndarray) -> np.ndarray:
        values = _validated_values(q, 1)
        length = _validated_length(values[0])
        reference = self.reference_geometry_A
        direction = reference[self.bond.j] - reference[self.bond.i]
        direction = direction / np.linalg.norm(direction)
        out = np.array(reference, copy=True)
        out[self.bond.j] = reference[self.bond.i] + length * direction
        return out

    def values_from_cartesian(self, xyz: np.ndarray) -> np.ndarray:
        coordinates = _validated_coordinates_for_map(xyz, self.reference_geometry_A.shape)
        length = float(np.linalg.norm(coordinates[self.bond.j] - coordinates[self.bond.i]))
        if length <= 0.0:
            raise ValueError(f"Zero-length bond {self.bond}")
        return np.array([length], dtype=float)

    def vector_to_body(self, vector_lab: np.ndarray) -> np.ndarray:
        return _transform_vectors(vector_lab, self.frame_to_lab, to_body=True)

    def vector_to_lab(self, vector_body: np.ndarray) -> np.ndarray:
        return _transform_vectors(vector_body, self.frame_to_lab, to_body=False)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "schema": "pyscf-vscf-coordinate-map",
            "schema_version": COORDINATE_MAP_SCHEMA_VERSION,
            "kind": "ordered-bond-length",
            "coordinate_ids": list(self.coordinate_ids),
            "units": list(self.units),
            "bond": [self.bond.i, self.bond.j],
            "active_atoms": list(self.active_atoms),
            "inactive_atoms": list(self.inactive_atoms),
            "reference_geometry_A": _array_payload(self.reference_geometry_A),
            "reference_values": _array_payload(self.reference_values),
            "frame_convention": "fixed-explicit-reference",
            "frame_to_lab": _array_payload(self.frame_to_lab),
        }


@dataclass(frozen=True)
class TriatomicValenceCoordinateMap:
    """Ordered ``(r1, r2, theta)`` map embedded in a Cartesian geometry."""

    reference_geometry_A: np.ndarray
    center_atom: int
    outer_atom_1: int
    outer_atom_2: int
    coordinate_ids: tuple[str, str, str] = ("r1", "r2", "theta")
    frame_sine_tolerance: float = DEFAULT_FRAME_SINE_TOLERANCE

    def __post_init__(self) -> None:
        reference = _validated_coordinates(self.reference_geometry_A)
        atoms = tuple(
            _strict_atom_index(name, value)
            for name, value in (
                ("center_atom", self.center_atom),
                ("outer_atom_1", self.outer_atom_1),
                ("outer_atom_2", self.outer_atom_2),
            )
        )
        if len(set(atoms)) != 3:
            raise ValueError("center and outer atom indices must be distinct")
        if any(index < 0 or index >= reference.shape[0] for index in atoms):
            raise IndexError("Triatomic active atom index is out of range")
        ids = tuple(str(value).strip() for value in self.coordinate_ids)
        if len(ids) != 3 or len(set(ids)) != 3 or any(not value for value in ids):
            raise ValueError("coordinate_ids must contain three distinct non-empty names")
        tolerance = float(self.frame_sine_tolerance)
        if not np.isfinite(tolerance) or tolerance <= 0.0 or tolerance >= 1.0:
            raise ValueError("frame_sine_tolerance must lie strictly between zero and one")

        center, outer1, outer2 = atoms
        vector1 = reference[outer1] - reference[center]
        vector2 = reference[outer2] - reference[center]
        r1 = float(np.linalg.norm(vector1))
        r2 = float(np.linalg.norm(vector2))
        if r1 <= 0.0 or r2 <= 0.0:
            raise ValueError("Triatomic reference bonds must have positive length")
        e1 = vector1 / r1
        orthogonal = vector2 - float(np.dot(vector2, e1)) * e1
        orthogonal_norm = float(np.linalg.norm(orthogonal))
        if orthogonal_norm / r2 <= tolerance:
            raise ValueError("Triatomic reference geometry is linear or too nearly linear")
        e2 = orthogonal / orthogonal_norm
        e3 = np.cross(e1, e2)
        frame = _validated_frame(np.column_stack((e1, e2, e3)))
        theta = _angle_between(vector1, vector2)

        object.__setattr__(self, "reference_geometry_A", immutable_array(reference))
        object.__setattr__(self, "center_atom", center)
        object.__setattr__(self, "outer_atom_1", outer1)
        object.__setattr__(self, "outer_atom_2", outer2)
        object.__setattr__(self, "coordinate_ids", ids)
        object.__setattr__(self, "frame_sine_tolerance", tolerance)
        object.__setattr__(self, "units", ("angstrom", "angstrom", "radian"))
        object.__setattr__(self, "reference_values", immutable_array([r1, r2, theta]))
        object.__setattr__(self, "active_atoms", atoms)
        object.__setattr__(
            self,
            "inactive_atoms",
            tuple(index for index in range(reference.shape[0]) if index not in atoms),
        )
        object.__setattr__(self, "frame_to_lab", immutable_array(frame))

    def to_cartesian(self, q: np.ndarray) -> np.ndarray:
        values = _validated_values(q, 3)
        r1 = _validated_length(values[0])
        r2 = _validated_length(values[1])
        theta = float(values[2])
        if theta <= 0.0 or theta >= np.pi:
            raise ValueError("Triatomic bond angle must lie strictly between zero and pi")
        e1 = self.frame_to_lab[:, 0]
        e2 = self.frame_to_lab[:, 1]
        center_position = self.reference_geometry_A[self.center_atom]
        out = np.array(self.reference_geometry_A, copy=True)
        out[self.outer_atom_1] = center_position + r1 * e1
        out[self.outer_atom_2] = center_position + r2 * (np.cos(theta) * e1 + np.sin(theta) * e2)
        return out

    def values_from_cartesian(self, xyz: np.ndarray) -> np.ndarray:
        coordinates = _validated_coordinates_for_map(xyz, self.reference_geometry_A.shape)
        center = coordinates[self.center_atom]
        vector1 = coordinates[self.outer_atom_1] - center
        vector2 = coordinates[self.outer_atom_2] - center
        r1 = float(np.linalg.norm(vector1))
        r2 = float(np.linalg.norm(vector2))
        if r1 <= 0.0 or r2 <= 0.0:
            raise ValueError("Triatomic bond lengths must be positive")
        return np.array([r1, r2, _angle_between(vector1, vector2)], dtype=float)

    def vector_to_body(self, vector_lab: np.ndarray) -> np.ndarray:
        return _transform_vectors(vector_lab, self.frame_to_lab, to_body=True)

    def vector_to_lab(self, vector_body: np.ndarray) -> np.ndarray:
        return _transform_vectors(vector_body, self.frame_to_lab, to_body=False)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "schema": "pyscf-vscf-coordinate-map",
            "schema_version": COORDINATE_MAP_SCHEMA_VERSION,
            "kind": "ordered-triatomic-valence",
            "coordinate_ids": list(self.coordinate_ids),
            "units": list(self.units),
            "center_atom": self.center_atom,
            "outer_atoms": [self.outer_atom_1, self.outer_atom_2],
            "active_atoms": list(self.active_atoms),
            "inactive_atoms": list(self.inactive_atoms),
            "reference_geometry_A": _array_payload(self.reference_geometry_A),
            "reference_values": _array_payload(self.reference_values),
            "frame_convention": "fixed-ordered-reference-plane",
            "frame_sine_tolerance": self.frame_sine_tolerance,
            "frame_to_lab": _array_payload(self.frame_to_lab),
        }


@dataclass(frozen=True)
class LinearDisplacementCoordinateMap:
    """Map rectilinear coordinates onto fixed Cartesian displacement vectors."""

    reference_geometry_A: np.ndarray
    coordinate_ids: tuple[str, ...]
    units: tuple[str, ...]
    reference_values: np.ndarray
    displacements_A_per_unit: np.ndarray
    reference_frame_to_lab: np.ndarray | None = None

    def __post_init__(self) -> None:
        reference = _validated_coordinates(self.reference_geometry_A)
        ids = tuple(str(value).strip() for value in self.coordinate_ids)
        units = tuple(str(value).strip().lower() for value in self.units)
        if (
            not ids
            or len(ids) != len(units)
            or len(set(ids)) != len(ids)
            or any(not value for value in (*ids, *units))
        ):
            raise ValueError("coordinate IDs and units must be non-empty, unique, and aligned")
        values = _validated_values(self.reference_values, len(ids))
        displacements = np.asarray(self.displacements_A_per_unit, dtype=float)
        expected_shape = (len(ids), *reference.shape)
        if displacements.shape != expected_shape or not np.all(np.isfinite(displacements)):
            raise ValueError(f"displacements_A_per_unit must have shape {expected_shape}")
        if np.linalg.matrix_rank(displacements.reshape((len(ids), -1))) != len(ids):
            raise ValueError("Cartesian displacement vectors must be linearly independent")
        frame = (
            np.eye(3)
            if self.reference_frame_to_lab is None
            else _validated_frame(self.reference_frame_to_lab)
        )
        active = tuple(
            index
            for index in range(reference.shape[0])
            if np.any(displacements[:, index, :] != 0.0)
        )
        object.__setattr__(self, "reference_geometry_A", immutable_array(reference))
        object.__setattr__(self, "coordinate_ids", ids)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "reference_values", immutable_array(values))
        object.__setattr__(self, "displacements_A_per_unit", immutable_array(displacements))
        object.__setattr__(self, "reference_frame_to_lab", immutable_array(frame))
        object.__setattr__(self, "active_atoms", active)
        object.__setattr__(
            self,
            "inactive_atoms",
            tuple(index for index in range(reference.shape[0]) if index not in active),
        )
        object.__setattr__(self, "frame_to_lab", immutable_array(frame))

    def to_cartesian(self, q: np.ndarray) -> np.ndarray:
        values = _validated_values(q, len(self.coordinate_ids))
        offsets = values - self.reference_values
        return np.asarray(
            self.reference_geometry_A
            + np.tensordot(offsets, self.displacements_A_per_unit, axes=(0, 0)),
            dtype=float,
        )

    def values_from_cartesian(self, xyz: np.ndarray) -> np.ndarray:
        coordinates = _validated_coordinates_for_map(xyz, self.reference_geometry_A.shape)
        matrix = self.displacements_A_per_unit.reshape((len(self.coordinate_ids), -1)).T
        delta = (coordinates - self.reference_geometry_A).reshape(-1)
        offsets, _, _, _ = np.linalg.lstsq(matrix, delta, rcond=None)
        if not np.allclose(matrix @ offsets, delta, rtol=0.0, atol=1e-12):
            raise ValueError("Cartesian geometry is outside the linear coordinate-map subspace")
        return np.asarray(self.reference_values + offsets, dtype=float)

    def vector_to_body(self, vector_lab: np.ndarray) -> np.ndarray:
        return _transform_vectors(vector_lab, self.frame_to_lab, to_body=True)

    def vector_to_lab(self, vector_body: np.ndarray) -> np.ndarray:
        return _transform_vectors(vector_body, self.frame_to_lab, to_body=False)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "schema": "pyscf-vscf-coordinate-map",
            "schema_version": COORDINATE_MAP_SCHEMA_VERSION,
            "kind": "linear-cartesian-displacement",
            "coordinate_ids": list(self.coordinate_ids),
            "units": list(self.units),
            "active_atoms": list(self.active_atoms),
            "inactive_atoms": list(self.inactive_atoms),
            "reference_geometry_A": _array_payload(self.reference_geometry_A),
            "reference_values": _array_payload(self.reference_values),
            "displacements_A_per_unit": _array_payload(self.displacements_A_per_unit),
            "frame_convention": "fixed-explicit-reference",
            "frame_to_lab": _array_payload(self.frame_to_lab),
        }


def coordinate_map_from_payload(payload: Mapping[str, object]) -> CoordinateMap:
    """Reconstruct and integrity-check a supported coordinate map."""

    data = dict(payload)
    if (
        data.get("schema") != "pyscf-vscf-coordinate-map"
        or data.get("schema_version") != COORDINATE_MAP_SCHEMA_VERSION
    ):
        raise ValueError("Unsupported coordinate-map payload schema")
    kind = data.get("kind")
    reference = _array_from_payload("reference_geometry_A", data.get("reference_geometry_A"))
    if kind == "ordered-bond-length":
        bond = tuple(data.get("bond", ()))
        coordinate_ids = tuple(data.get("coordinate_ids", ()))
        if len(bond) != 2 or len(coordinate_ids) != 1:
            raise ValueError("Bond-length coordinate-map payload is incomplete")
        coordinate_map: CoordinateMap = BondLengthCoordinateMap(
            reference_geometry_A=reference,
            bond=Bond(*bond),
            coordinate_id=coordinate_ids[0],
            reference_frame_to_lab=_array_from_payload("frame_to_lab", data.get("frame_to_lab")),
        )
    elif kind == "ordered-triatomic-valence":
        outer_atoms = tuple(data.get("outer_atoms", ()))
        if len(outer_atoms) != 2:
            raise ValueError("Triatomic coordinate-map payload requires two outer atoms")
        coordinate_map = TriatomicValenceCoordinateMap(
            reference_geometry_A=reference,
            center_atom=data.get("center_atom"),
            outer_atom_1=outer_atoms[0],
            outer_atom_2=outer_atoms[1],
            coordinate_ids=tuple(data.get("coordinate_ids", ())),
            frame_sine_tolerance=data.get("frame_sine_tolerance"),
        )
    elif kind == "linear-cartesian-displacement":
        coordinate_map = LinearDisplacementCoordinateMap(
            reference_geometry_A=reference,
            coordinate_ids=tuple(data.get("coordinate_ids", ())),
            units=tuple(data.get("units", ())),
            reference_values=_array_from_payload("reference_values", data.get("reference_values")),
            displacements_A_per_unit=_array_from_payload(
                "displacements_A_per_unit", data.get("displacements_A_per_unit")
            ),
            reference_frame_to_lab=_array_from_payload("frame_to_lab", data.get("frame_to_lab")),
        )
    else:
        raise ValueError(f"Unsupported coordinate-map kind {kind!r}")
    if canonical_json(data) != canonical_json(dict(coordinate_map.fingerprint_payload())):
        raise ValueError("Coordinate-map payload does not reproduce its canonical definition")
    return coordinate_map


def parse_bond(s: str) -> Bond:
    """Parse a zero-based bond specification such as ``0-1``."""

    spec = s.strip().replace(" ", "")
    if "-" not in spec:
        raise ValueError(f"Bond specification must contain '-': {s!r}")
    left, right = spec.split("-", 1)
    try:
        return Bond(int(left), int(right))
    except ValueError as exc:
        raise ValueError(f"Invalid bond specification {s!r}; expected 0-1") from exc


def stretch_along_bond(coords: np.ndarray, bond: Bond, new_len_A: float) -> np.ndarray:
    out = _validated_coordinates(coords)
    i, j = int(bond.i), int(bond.j)
    _validate_bond(bond, out.shape[0])
    length = _validated_length(new_len_A)
    vec = out[j] - out[i]
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"Zero-length bond {bond}")
    out[j] = out[i] + vec / norm * length
    return out


def stretch_two_bonds(
    coords: np.ndarray,
    bond1: Bond,
    bond2: Bond,
    new_len1_A: float,
    new_len2_A: float,
) -> np.ndarray:
    """Set two bond lengths without one displacement invalidating the other.

    Bonds that share one atom are stretched simultaneously about that common
    atom, so the result is independent of bond orientation. For disjoint bonds,
    each bond retains the usual convention that ``i`` is fixed and ``j`` moves.
    Duplicate or reversed-duplicate bonds do not define two independent
    coordinates and are rejected.
    """

    source = _validated_coordinates(coords)
    _validate_bond(bond1, source.shape[0])
    _validate_bond(bond2, source.shape[0])
    length1 = _validated_length(new_len1_A)
    length2 = _validated_length(new_len2_A)

    atoms1 = {int(bond1.i), int(bond1.j)}
    atoms2 = {int(bond2.i), int(bond2.j)}
    shared = atoms1 & atoms2
    if len(shared) == 2:
        raise ValueError("Two-dimensional scans require two distinct bonds")

    if len(shared) == 1:
        anchor = shared.pop()
        endpoint1 = (atoms1 - {anchor}).pop()
        endpoint2 = (atoms2 - {anchor}).pop()
        out = source.copy()
        out[endpoint1] = _point_at_distance(source, anchor, endpoint1, length1)
        out[endpoint2] = _point_at_distance(source, anchor, endpoint2, length2)
    else:
        # Disjoint bonds cannot interfere. Their explicit i -> j orientation is
        # the anchor policy, matching the public one-bond coordinate semantics.
        out = stretch_along_bond(source, bond1, length1)
        out = stretch_along_bond(out, bond2, length2)

    achieved1 = float(np.linalg.norm(out[bond1.j] - out[bond1.i]))
    achieved2 = float(np.linalg.norm(out[bond2.j] - out[bond2.i]))
    if not np.isclose(achieved1, length1, rtol=0.0, atol=1e-12) or not np.isclose(
        achieved2,
        length2,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Two-bond stretch failed to realize the requested coordinate values")
    return out


def _validated_coordinates(coords: np.ndarray) -> np.ndarray:
    out = np.array(coords, dtype=float, copy=True)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(out)):
        raise ValueError("coordinates must contain only finite values")
    return out


def _strict_atom_index(name: str, value: object) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be an integer atom index, not a boolean")
    try:
        return operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer atom index") from exc


def _validated_coordinates_for_map(
    coords: np.ndarray,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    out = _validated_coordinates(coords)
    if out.shape != expected_shape:
        raise ValueError(f"coordinates must have shape {expected_shape}")
    return out


def _validated_values(values: np.ndarray, size: int) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if out.shape != (size,):
        raise ValueError(f"coordinate values must have shape ({size},)")
    if not np.all(np.isfinite(out)):
        raise ValueError("coordinate values must contain only finite values")
    return out


def _validated_frame(frame_to_lab: np.ndarray) -> np.ndarray:
    frame = np.asarray(frame_to_lab, dtype=float)
    if frame.shape != (3, 3) or not np.all(np.isfinite(frame)):
        raise ValueError("frame_to_lab must be a finite 3x3 matrix")
    if not np.allclose(frame.T @ frame, np.eye(3), rtol=0.0, atol=1e-12):
        raise ValueError("frame_to_lab must be orthonormal")
    if not np.isclose(np.linalg.det(frame), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("frame_to_lab must be right-handed with determinant +1")
    return np.array(frame, copy=True)


def _angle_between(vector1: np.ndarray, vector2: np.ndarray) -> float:
    norm1 = float(np.linalg.norm(vector1))
    norm2 = float(np.linalg.norm(vector2))
    if norm1 <= 0.0 or norm2 <= 0.0:
        raise ValueError("Cannot define an angle using a zero-length vector")
    cosine = float(np.dot(vector1, vector2) / (norm1 * norm2))
    if cosine < -1.0 - 1e-12 or cosine > 1.0 + 1e-12:
        raise ValueError("Computed angle cosine lies outside the physical interval")
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _transform_vectors(
    vectors: np.ndarray,
    frame_to_lab: np.ndarray,
    *,
    to_body: bool,
) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("vectors must have final dimension 3")
    if not np.all(np.isfinite(values)):
        raise ValueError("vectors must contain only finite values")
    transform = frame_to_lab if to_body else frame_to_lab.T
    return np.asarray(values @ transform, dtype=float)


def _array_payload(values: np.ndarray) -> dict[str, object]:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    return {**float64_array_identity(array), "values": array.tolist()}


def _array_from_payload(name: str, payload: object) -> np.ndarray:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} payload must be a mapping")
    if set(payload) != {"dtype", "shape", "sha256", "values"} or payload.get("dtype") != "<f8":
        raise ValueError(f"{name} payload has an unsupported array encoding")
    shape = tuple(operator.index(value) for value in payload["shape"])
    array = np.asarray(payload["values"], dtype="<f8")
    if array.shape != shape:
        raise ValueError(f"{name} payload shape does not match its values")
    if float64_array_identity(array) != {
        "dtype": payload["dtype"],
        "shape": list(shape),
        "sha256": payload["sha256"],
    }:
        raise ValueError(f"{name} payload hash does not match its values")
    return np.asarray(array, dtype=float)


def _validate_bond(bond: Bond, n_atoms: int) -> None:
    i, j = int(bond.i), int(bond.j)
    if i < 0 or j < 0 or i >= n_atoms or j >= n_atoms:
        raise IndexError(f"Bond {bond} is out of range for {n_atoms} atoms")
    if i == j:
        raise ValueError("A bond must reference two distinct atoms")


def _validated_length(new_len_A: float) -> float:
    length = float(new_len_A)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("Bond lengths must be positive and finite")
    return length


def _point_at_distance(
    coords: np.ndarray,
    anchor: int,
    endpoint: int,
    distance_A: float,
) -> np.ndarray:
    vector = coords[endpoint] - coords[anchor]
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"Zero-length bond Bond(i={anchor}, j={endpoint})")
    return coords[anchor] + vector / norm * distance_A


__all__ = [
    "Bond",
    "BondLengthCoordinateMap",
    "COORDINATE_MAP_SCHEMA_VERSION",
    "CoordinateMap",
    "DEFAULT_FRAME_SINE_TOLERANCE",
    "LinearDisplacementCoordinateMap",
    "TriatomicValenceCoordinateMap",
    "coordinate_map_fingerprint",
    "coordinate_map_from_payload",
    "parse_bond",
    "stretch_along_bond",
    "stretch_two_bonds",
]
