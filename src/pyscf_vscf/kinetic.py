"""Reviewed kinetic operators and direct-grid validation solvers."""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from ._arrays import immutable_array
from ._identity import (
    FrozenJSONMapping,
    array_identity,
    payload_fingerprint,
    to_jsonable,
)
from .constants import AMU, ANG_TO_BOHR
from .coordinates import TriatomicValenceCoordinateMap, coordinate_map_fingerprint
from .dvr import sinc_kinetic_1d
from .nmode import NModeSurfaceModel, nmode_pes_fingerprint


@runtime_checkable
class KineticOperator(Protocol):
    """Kinetic operator acting on one declared product-grid representation."""

    coordinate_ids: tuple[str, ...]

    @property
    def shape(self) -> tuple[int, ...]: ...

    def validate_grid(self, representation: object) -> None: ...

    def apply(self, vector: np.ndarray) -> np.ndarray: ...

    def matrix_elements(self, modal_bases: Sequence[np.ndarray]) -> ProjectedKineticTerms: ...

    def fingerprint_payload(self) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class ProjectedKineticTerms:
    """Modal-basis factors for the triatomic Jacobi kinetic operator."""

    radial_r_Eh: np.ndarray
    radial_R_Eh: np.ndarray
    inverse_r2_Eh: np.ndarray
    inverse_R2_Eh: np.ndarray
    angular_j2: np.ndarray

    def __post_init__(self) -> None:
        matrices = {
            "radial_r_Eh": self.radial_r_Eh,
            "radial_R_Eh": self.radial_R_Eh,
            "inverse_r2_Eh": self.inverse_r2_Eh,
            "inverse_R2_Eh": self.inverse_R2_Eh,
            "angular_j2": self.angular_j2,
        }
        for name, values in matrices.items():
            matrix = np.asarray(values, dtype=float)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError(f"{name} must be a square matrix")
            if not np.all(np.isfinite(matrix)):
                raise ValueError(f"{name} must be finite")
            if not np.allclose(matrix, matrix.T, rtol=0.0, atol=2e-13):
                raise ValueError(f"{name} must be Hermitian")
            object.__setattr__(self, name, _readonly(matrix))


@dataclass(frozen=True)
class TriatomicJacobiTransform:
    """Mass-dependent transform between ordered valence and Jacobi scalars.

    Masses and indices are ordered as ``(outer1, center, outer2)``. Valence
    coordinates are ``(a, b, gamma)`` in Angstrom, Angstrom, and radians;
    Jacobi coordinates are ``(r, R, x)`` in Angstrom, Angstrom, and cosine.
    """

    masses_amu: tuple[float, float, float]
    atom_indices: tuple[int, int, int] = (0, 1, 2)
    cosine_tolerance: float = 1e-12

    def __post_init__(self) -> None:
        masses = _positive_tuple("masses_amu", self.masses_amu, 3)
        indices = tuple(operator.index(value) for value in self.atom_indices)
        if len(set(indices)) != 3 or min(indices) < 0:
            raise ValueError("atom_indices must contain three distinct non-negative indices")
        tolerance = float(self.cosine_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("cosine_tolerance must be finite and non-negative")
        object.__setattr__(self, "masses_amu", masses)
        object.__setattr__(self, "atom_indices", indices)
        object.__setattr__(self, "cosine_tolerance", tolerance)

    @property
    def lambda_outer2(self) -> float:
        _, center, outer2 = self.masses_amu
        return outer2 / (center + outer2)

    def valence_to_jacobi(self, values: np.ndarray) -> np.ndarray:
        q = _coordinate_array("valence coordinates", values)
        a, b, gamma = (q[..., index] for index in range(3))
        if np.any(a <= 0.0) or np.any(b <= 0.0):
            raise ValueError("Valence bond lengths must be positive")
        if np.any(gamma <= 0.0) or np.any(gamma >= np.pi):
            raise ValueError("Valence angles must lie strictly between zero and pi")
        lam = self.lambda_outer2
        radial_R_squared = a * a + lam * lam * b * b - 2.0 * lam * a * b * np.cos(gamma)
        if np.any(radial_R_squared <= 0.0):
            raise ValueError("Valence coordinates produce a singular Jacobi vector")
        radial_R = np.sqrt(radial_R_squared)
        cosine = (a * np.cos(gamma) - lam * b) / radial_R
        cosine = _checked_cosine(cosine, self.cosine_tolerance)
        return np.stack((b, radial_R, cosine), axis=-1)

    def jacobi_to_valence(self, values: np.ndarray) -> np.ndarray:
        q = _coordinate_array("Jacobi coordinates", values)
        radial_r, radial_R, cosine = (q[..., index] for index in range(3))
        if np.any(radial_r <= 0.0) or np.any(radial_R <= 0.0):
            raise ValueError("Jacobi radial distances must be positive")
        cosine = _checked_cosine(cosine, self.cosine_tolerance)
        lam = self.lambda_outer2
        a_squared = (
            radial_R * radial_R
            + lam * lam * radial_r * radial_r
            + 2.0 * lam * radial_R * radial_r * cosine
        )
        if np.any(a_squared <= 0.0):
            raise ValueError("Jacobi coordinates produce a singular valence bond")
        a = np.sqrt(a_squared)
        gamma_cosine = (radial_R * cosine + lam * radial_r) / a
        gamma_cosine = _checked_cosine(gamma_cosine, self.cosine_tolerance)
        return np.stack((a, radial_r, np.arccos(gamma_cosine)), axis=-1)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "ordered-triatomic-jacobi-transform",
            "schema_version": 1,
            "masses_amu": list(self.masses_amu),
            "atom_indices_outer1_center_outer2": list(self.atom_indices),
            "valence_coordinates": ["a_angstrom", "b_angstrom", "gamma_radian"],
            "jacobi_coordinates": ["r_angstrom", "R_angstrom", "x_cosine"],
            "cosine_tolerance": self.cosine_tolerance,
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


@dataclass(frozen=True)
class TriatomicJ0KineticOperator:
    """Isolated-triatomic Jacobi kinetic operator at total ``J=0``.

    This implements the atom-diatom Jacobi convention of Tennyson et al.,
    Comput. Phys. Commun. 163, 85-116 (2004), DOI
    10.1016/j.cpc.2003.10.003. For ``Phi = r R Psi`` the integration measure is
    flat ``dr dR dx`` with ``x = cos(theta)``. Radial grids use the package's
    cardinal-sinc representation and the angular coordinate uses a normalized
    Gauss-Legendre DVR/FBR transform.
    """

    radial_r_A: np.ndarray
    radial_R_A: np.ndarray
    angular_order: int
    masses_amu: tuple[float, float, float]
    atom_indices: tuple[int, int, int] = (0, 1, 2)
    coordinate_ids: tuple[str, str, str] = field(init=False, default=("r", "R", "x"))
    angular_x: np.ndarray = field(init=False)
    angular_weights: np.ndarray = field(init=False)
    angular_j2: np.ndarray = field(init=False)
    radial_r_kinetic_Eh: np.ndarray = field(init=False)
    radial_R_kinetic_Eh: np.ndarray = field(init=False)
    inverse_r2_Eh: np.ndarray = field(init=False)
    inverse_R2_Eh: np.ndarray = field(init=False)
    reduced_masses_amu: tuple[float, float] = field(init=False)

    def __post_init__(self) -> None:
        radial_r = _uniform_positive_grid("radial_r_A", self.radial_r_A)
        radial_R = _uniform_positive_grid("radial_R_A", self.radial_R_A)
        order = operator.index(self.angular_order)
        if order < 2:
            raise ValueError("angular_order must be at least two")
        transform = TriatomicJacobiTransform(self.masses_amu, self.atom_indices)
        outer1, center, outer2 = transform.masses_amu
        mu_r = center * outer2 / (center + outer2)
        mu_R = outer1 * (center + outer2) / (outer1 + center + outer2)
        angular_x, angular_weights, angular_j2 = _gauss_legendre_j2(order)
        radial_r_kinetic = sinc_kinetic_1d(radial_r, mu_r)
        radial_R_kinetic = sinc_kinetic_1d(radial_R, mu_R)
        inverse_r2 = 1.0 / (2.0 * mu_r * AMU * np.square(radial_r * ANG_TO_BOHR))
        inverse_R2 = 1.0 / (2.0 * mu_R * AMU * np.square(radial_R * ANG_TO_BOHR))
        object.__setattr__(self, "radial_r_A", _readonly(radial_r))
        object.__setattr__(self, "radial_R_A", _readonly(radial_R))
        object.__setattr__(self, "angular_order", order)
        object.__setattr__(self, "masses_amu", transform.masses_amu)
        object.__setattr__(self, "atom_indices", transform.atom_indices)
        object.__setattr__(self, "angular_x", _readonly(angular_x))
        object.__setattr__(self, "angular_weights", _readonly(angular_weights))
        object.__setattr__(self, "angular_j2", _readonly(angular_j2))
        object.__setattr__(self, "radial_r_kinetic_Eh", _readonly(radial_r_kinetic))
        object.__setattr__(self, "radial_R_kinetic_Eh", _readonly(radial_R_kinetic))
        object.__setattr__(self, "inverse_r2_Eh", _readonly(inverse_r2))
        object.__setattr__(self, "inverse_R2_Eh", _readonly(inverse_R2))
        object.__setattr__(self, "reduced_masses_amu", (mu_r, mu_R))

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.radial_r_A.size, self.radial_R_A.size, self.angular_order)

    @property
    def dimension(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def coordinate_grids(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.radial_r_A, self.radial_R_A, self.angular_x

    def validate_grid(self, representation: object) -> None:
        shape = getattr(representation, "shape", None)
        if shape is None:
            try:
                shape = tuple(operator.index(value) for value in representation)  # type: ignore[arg-type]
            except (TypeError, ValueError) as exc:
                raise TypeError("representation must expose a shape") from exc
        if tuple(shape) != self.shape:
            raise ValueError(f"representation shape must be {self.shape}, got {tuple(shape)}")

    def apply(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.shape == (self.dimension,):
            wavefunction = values.reshape(self.shape)
            flattened = True
        elif values.shape == self.shape:
            wavefunction = values
            flattened = False
        else:
            raise ValueError(f"kinetic vector must have shape {(self.dimension,)} or {self.shape}")
        if not np.all(np.isfinite(wavefunction)):
            raise ValueError("kinetic vector must be finite")
        result = _apply_axis(self.radial_r_kinetic_Eh, wavefunction, 0)
        result += _apply_axis(self.radial_R_kinetic_Eh, wavefunction, 1)
        angular = _apply_axis(self.angular_j2, wavefunction, 2)
        coefficient = self.inverse_r2_Eh[:, None, None] + self.inverse_R2_Eh[None, :, None]
        result += coefficient * angular
        return result.reshape(-1) if flattened else result

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.dimension, self.dimension),
            matvec=self.apply,
            rmatvec=self.apply,
            dtype=float,
        )

    def matrix_elements(self, modal_bases: Sequence[np.ndarray]) -> ProjectedKineticTerms:
        bases = _validated_modal_bases(modal_bases, self.shape)
        radial_r, radial_R, angular = bases
        return ProjectedKineticTerms(
            radial_r_Eh=radial_r.T @ self.radial_r_kinetic_Eh @ radial_r,
            radial_R_Eh=radial_R.T @ self.radial_R_kinetic_Eh @ radial_R,
            inverse_r2_Eh=radial_r.T @ np.diag(self.inverse_r2_Eh) @ radial_r,
            inverse_R2_Eh=radial_R.T @ np.diag(self.inverse_R2_Eh) @ radial_R,
            angular_j2=angular.T @ self.angular_j2 @ angular,
        )

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "triatomic-j0-jacobi-kinetic",
            "schema_version": 1,
            "citation_doi": "10.1016/j.cpc.2003.10.003",
            "wavefunction_scaling": "Phi=r*R*Psi",
            "integration_measure": "dr_dR_dx",
            "radial_representation": "colbert-miller-cardinal-sinc",
            "angular_representation": "normalized-gauss-legendre-dvr",
            "coordinate_ids": list(self.coordinate_ids),
            "atom_indices_outer1_center_outer2": list(self.atom_indices),
            "masses_amu": list(self.masses_amu),
            "reduced_masses_amu": list(self.reduced_masses_amu),
            "radial_r_A": array_identity(self.radial_r_A),
            "radial_R_A": array_identity(self.radial_R_A),
            "angular_x": array_identity(self.angular_x),
            "angular_weights": array_identity(self.angular_weights),
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


@dataclass(frozen=True)
class TriatomicJ0Hamiltonian:
    """Triatomic Jacobi kinetic operator plus one grid-diagonal potential.

    Direct construction from an array is an explicitly unbound analytical-
    potential path. Use :meth:`from_projection` for a provenance-bound
    valence-PES projection.
    """

    kinetic: TriatomicJ0KineticOperator
    potential_Eh: np.ndarray | JacobiGridProjection
    metadata: Mapping[str, object] = field(default_factory=dict)
    _source_projection: JacobiGridProjection | None = field(
        init=False,
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        source = None
        if isinstance(self.potential_Eh, JacobiGridProjection):
            source = self.potential_Eh
            if source.kinetic_fingerprint != self.kinetic.fingerprint():
                raise ValueError(
                    "Jacobi projection was generated for a different kinetic operator"
                )
            potential = np.asarray(source.potential_Eh, dtype=float)
        else:
            potential = np.asarray(self.potential_Eh, dtype=float)
        self.kinetic.validate_grid(potential)
        if not np.all(np.isfinite(potential)):
            raise ValueError("potential_Eh must be finite")
        object.__setattr__(self, "potential_Eh", _readonly(potential))
        object.__setattr__(self, "metadata", FrozenJSONMapping.from_mapping(self.metadata))
        object.__setattr__(self, "_source_projection", source)

    @classmethod
    def from_projection(
        cls,
        kinetic: TriatomicJ0KineticOperator,
        projection: JacobiGridProjection,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> TriatomicJ0Hamiltonian:
        """Construct a Hamiltonian bound to the exact projected PES artifact."""

        return cls(
            kinetic=kinetic,
            potential_Eh=projection,
            metadata={} if metadata is None else metadata,
        )

    @property
    def source_projection_fingerprint(self) -> str | None:
        return None if self._source_projection is None else self._source_projection.fingerprint()

    @property
    def source_pes_fingerprint(self) -> str | None:
        return (
            None
            if self._source_projection is None
            else self._source_projection.source_pes_fingerprint
        )

    @property
    def kinetic_fingerprint(self) -> str:
        return self.kinetic.fingerprint()

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.kinetic.shape

    @property
    def dimension(self) -> int:
        return self.kinetic.dimension

    @property
    def coordinate_ids(self) -> tuple[str, str, str]:
        return self.kinetic.coordinate_ids

    def apply(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.dimension,):
            raise ValueError(f"Hamiltonian vector must have shape {(self.dimension,)}")
        return self.kinetic.apply(values) + self.potential_Eh.reshape(-1) * values

    def as_linear_operator(self) -> LinearOperator:
        return LinearOperator(
            (self.dimension, self.dimension),
            matvec=self.apply,
            rmatvec=self.apply,
            dtype=float,
        )

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "triatomic-j0-grid-hamiltonian",
            "schema_version": 1,
            "kinetic_fingerprint": self.kinetic.fingerprint(),
            "potential_Eh": array_identity(self.potential_Eh),
            "source_projection_fingerprint": self.source_projection_fingerprint,
            "metadata": to_jsonable(self.metadata),
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


@dataclass(frozen=True)
class DirectDVRResult:
    """Low-lying direct-grid eigenpairs used as a validation oracle."""

    shape: tuple[int, int, int]
    energies_Eh: np.ndarray
    eigenvectors: np.ndarray
    residual_norms_Eh: np.ndarray
    radial_edge_probabilities: np.ndarray
    hamiltonian_fingerprint: str

    def __post_init__(self) -> None:
        shape = tuple(operator.index(value) for value in self.shape)
        dimension = int(np.prod(shape, dtype=np.int64))
        energies = np.asarray(self.energies_Eh, dtype=float)
        vectors = np.asarray(self.eigenvectors, dtype=float)
        residuals = np.asarray(self.residual_norms_Eh, dtype=float)
        edges = np.asarray(self.radial_edge_probabilities, dtype=float)
        if energies.ndim != 1 or vectors.shape != (dimension, energies.size):
            raise ValueError("Direct DVR eigenpair arrays are inconsistent")
        if residuals.shape != energies.shape or edges.shape != (energies.size, 2, 2):
            raise ValueError("Direct DVR diagnostics are inconsistent")
        if not all(
            np.all(np.isfinite(values)) for values in (energies, vectors, residuals, edges)
        ):
            raise ValueError("Direct DVR arrays must be finite")
        if np.any(residuals < 0.0) or np.any(edges < 0.0) or np.any(edges > 1.0):
            raise ValueError("Direct DVR diagnostics are outside physical bounds")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "energies_Eh", _readonly(energies))
        object.__setattr__(self, "eigenvectors", _readonly(vectors))
        object.__setattr__(self, "residual_norms_Eh", _readonly(residuals))
        object.__setattr__(self, "radial_edge_probabilities", _readonly(edges))
        object.__setattr__(
            self,
            "hamiltonian_fingerprint",
            _nonempty("hamiltonian_fingerprint", self.hamiltonian_fingerprint),
        )


@dataclass(frozen=True)
class JacobiGridProjection:
    """Fingerprint-bound valence PES projected onto one Jacobi solver grid."""

    potential_Eh: np.ndarray
    source_pes_fingerprint: str
    source_coordinate_map_fingerprint: str
    transform_fingerprint: str
    kinetic_fingerprint: str
    coordinate_ids: tuple[str, str, str]
    atom_indices: tuple[int, int, int]

    def __post_init__(self) -> None:
        potential = np.asarray(self.potential_Eh, dtype=float)
        if potential.ndim != 3 or not np.all(np.isfinite(potential)):
            raise ValueError("Projected Jacobi potential must be a finite 3D array")
        ids = tuple(str(value).strip() for value in self.coordinate_ids)
        indices = tuple(operator.index(value) for value in self.atom_indices)
        if len(ids) != 3 or any(not value for value in ids):
            raise ValueError("coordinate_ids must contain three non-empty values")
        if len(indices) != 3 or len(set(indices)) != 3 or min(indices) < 0:
            raise ValueError("atom_indices must contain three distinct non-negative values")
        object.__setattr__(self, "potential_Eh", _readonly(potential))
        for name in (
            "source_pes_fingerprint",
            "source_coordinate_map_fingerprint",
            "transform_fingerprint",
            "kinetic_fingerprint",
        ):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        object.__setattr__(self, "coordinate_ids", ids)
        object.__setattr__(self, "atom_indices", indices)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "valence-to-jacobi-grid-projection",
            "schema_version": 1,
            "potential_Eh": array_identity(self.potential_Eh),
            "source_pes_fingerprint": self.source_pes_fingerprint,
            "source_coordinate_map_fingerprint": (self.source_coordinate_map_fingerprint),
            "transform_fingerprint": self.transform_fingerprint,
            "kinetic_fingerprint": self.kinetic_fingerprint,
            "coordinate_ids": list(self.coordinate_ids),
            "atom_indices_outer1_center_outer2": list(self.atom_indices),
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


def potential_on_jacobi_grid(
    model: NModeSurfaceModel,
    coordinate_map: TriatomicValenceCoordinateMap,
    transform: TriatomicJacobiTransform,
    kinetic: TriatomicJ0KineticOperator,
) -> JacobiGridProjection:
    """Evaluate a three-coordinate valence PES on a Jacobi solver grid.

    The source interpolants keep ``bounds_error=True``. Any Jacobi point whose
    inverse transform leaves a source cut domain therefore fails explicitly.
    """

    if model.n_modes != 3:
        raise ValueError("Jacobi projection requires exactly three valence coordinates")
    if model.coordinate_units != ("angstrom", "angstrom", "radian"):
        raise ValueError("Jacobi projection requires valence units (angstrom, angstrom, radian)")
    map_fingerprint = coordinate_map_fingerprint(coordinate_map)
    if map_fingerprint != model.coordinate_map_fingerprint:
        raise ValueError("Source coordinate map fingerprint does not match the PES model")
    if tuple(coordinate_map.coordinate_ids) != model.coordinate_ids:
        raise ValueError("Source coordinate IDs do not match the PES model")
    if not np.array_equal(coordinate_map.reference_values, model.reference_values):
        raise ValueError("Source coordinate reference does not match the PES model")
    expected_atom_order = (
        coordinate_map.outer_atom_1,
        coordinate_map.center_atom,
        coordinate_map.outer_atom_2,
    )
    if transform.atom_indices != expected_atom_order:
        raise ValueError("Jacobi transform atom order does not match the ordered valence map")
    if transform.masses_amu != kinetic.masses_amu:
        raise ValueError("Jacobi transform and kinetic operator masses must match")
    if transform.atom_indices != kinetic.atom_indices:
        raise ValueError("Jacobi transform and kinetic operator atom order must match")
    meshes = np.meshgrid(*kinetic.coordinate_grids, indexing="ij")
    jacobi = np.stack(meshes, axis=-1)
    valence = transform.jacobi_to_valence(jacobi)
    potential = np.empty(kinetic.shape, dtype=float)
    for index in np.ndindex(kinetic.shape):
        potential[index] = model.potential_Eh(valence[index])
    return JacobiGridProjection(
        potential_Eh=potential,
        source_pes_fingerprint=nmode_pes_fingerprint(model),
        source_coordinate_map_fingerprint=map_fingerprint,
        transform_fingerprint=transform.fingerprint(),
        kinetic_fingerprint=kinetic.fingerprint(),
        coordinate_ids=tuple(coordinate_map.coordinate_ids),
        atom_indices=transform.atom_indices,
    )


def solve_triatomic_direct_dvr(
    hamiltonian: TriatomicJ0Hamiltonian,
    *,
    nstates: int,
    residual_tolerance_Eh: float = 1e-9,
    dense_dimension_threshold: int = 256,
) -> DirectDVRResult:
    """Solve a small triatomic direct DVR and report numerical diagnostics."""

    count = operator.index(nstates)
    if count < 1 or count >= hamiltonian.dimension:
        raise ValueError("nstates must satisfy 1 <= nstates < Hamiltonian dimension")
    tolerance = float(residual_tolerance_Eh)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("residual_tolerance_Eh must be positive and finite")
    dense_threshold = operator.index(dense_dimension_threshold)
    if dense_threshold < 1:
        raise ValueError("dense_dimension_threshold must be positive")

    if hamiltonian.dimension <= dense_threshold:
        identity = np.eye(hamiltonian.dimension)
        dense = np.column_stack(
            [hamiltonian.apply(identity[:, column]) for column in range(identity.shape[1])]
        )
        if not np.allclose(dense, dense.T, rtol=0.0, atol=2e-12):
            raise RuntimeError("Direct DVR Hamiltonian is not Hermitian")
        all_energies, all_vectors = np.linalg.eigh(dense)
        energies = all_energies[:count]
        vectors = all_vectors[:, :count]
    else:
        initial = _deterministic_initial(hamiltonian.dimension)
        energies, vectors = eigsh(
            hamiltonian.as_linear_operator(),
            k=count,
            which="SA",
            v0=initial,
            tol=min(tolerance * 0.1, 1e-11),
        )
        order = np.argsort(energies)
        energies = energies[order]
        vectors = vectors[:, order]
    vectors = _phase_canonical_columns(vectors)
    residuals = np.array(
        [
            np.linalg.norm(
                hamiltonian.apply(vectors[:, state]) - energies[state] * vectors[:, state]
            )
            for state in range(count)
        ]
    )
    if np.any(residuals > tolerance):
        raise RuntimeError(
            "Direct DVR eigensolver residual exceeds tolerance: "
            f"max={float(np.max(residuals)):.3e} Eh"
        )
    edge_probabilities = np.empty((count, 2, 2), dtype=float)
    for state in range(count):
        density = np.square(vectors[:, state].reshape(hamiltonian.shape))
        edge_probabilities[state, 0] = (
            float(np.sum(density[0, :, :])),
            float(np.sum(density[-1, :, :])),
        )
        edge_probabilities[state, 1] = (
            float(np.sum(density[:, 0, :])),
            float(np.sum(density[:, -1, :])),
        )
    return DirectDVRResult(
        shape=hamiltonian.shape,
        energies_Eh=energies,
        eigenvectors=vectors,
        residual_norms_Eh=residuals,
        radial_edge_probabilities=edge_probabilities,
        hamiltonian_fingerprint=hamiltonian.fingerprint(),
    )


def _gauss_legendre_j2(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x, weights = np.polynomial.legendre.leggauss(order)
    transform = np.empty((order, order), dtype=float)
    for angular_momentum in range(order):
        coefficients = np.zeros(angular_momentum + 1)
        coefficients[-1] = 1.0
        polynomial = np.polynomial.legendre.legval(x, coefficients)
        normalized = np.sqrt((2 * angular_momentum + 1) / 2.0) * polynomial
        transform[:, angular_momentum] = np.sqrt(weights) * normalized
    if not np.allclose(transform.T @ transform, np.eye(order), rtol=0.0, atol=2e-13):
        raise RuntimeError("Gauss-Legendre DVR/FBR transform is not orthonormal")
    eigenvalues = np.arange(order, dtype=float)
    eigenvalues *= eigenvalues + 1.0
    j2 = (transform * eigenvalues[None, :]) @ transform.T
    return x, weights, 0.5 * (j2 + j2.T)


def _validated_modal_bases(
    modal_bases: Sequence[np.ndarray], shape: tuple[int, ...]
) -> tuple[np.ndarray, ...]:
    if len(modal_bases) != len(shape):
        raise ValueError("modal_bases must contain one matrix per coordinate")
    bases = []
    for mode, (raw, grid_size) in enumerate(zip(modal_bases, shape)):
        basis = np.asarray(raw, dtype=float)
        if basis.ndim != 2 or basis.shape[0] != grid_size or basis.shape[1] < 1:
            raise ValueError(f"modal_bases[{mode}] must have shape ({grid_size}, n_modals)")
        if not np.all(np.isfinite(basis)):
            raise ValueError(f"modal_bases[{mode}] must be finite")
        if not np.allclose(basis.T @ basis, np.eye(basis.shape[1]), rtol=0.0, atol=2e-11):
            raise ValueError(f"modal_bases[{mode}] must have orthonormal columns")
        bases.append(np.array(basis, copy=True))
    return tuple(bases)


def _apply_axis(matrix: np.ndarray, tensor: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(tensor, axis, 0)
    acted = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(acted, 0, axis)


def _uniform_positive_grid(name: str, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError(f"{name} must be a one-dimensional grid with at least 3 nodes")
    if not np.all(np.isfinite(grid)) or np.any(grid <= 0.0):
        raise ValueError(f"{name} must contain positive finite values")
    differences = np.diff(grid)
    if np.any(differences <= 0.0) or not np.allclose(
        differences, differences[0], rtol=1e-10, atol=1e-12
    ):
        raise ValueError(f"{name} must be strictly increasing and uniform")
    return np.array(grid, copy=True)


def _coordinate_array(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape == () or array.shape[-1] != 3:
        raise ValueError(f"{name} must have final dimension three")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _checked_cosine(values: np.ndarray, tolerance: float) -> np.ndarray:
    cosine = np.asarray(values, dtype=float)
    if np.any(cosine < -1.0 - tolerance) or np.any(cosine > 1.0 + tolerance):
        raise ValueError("Coordinate transform produced a cosine outside [-1, 1]")
    return np.clip(cosine, -1.0, 1.0)


def _positive_tuple(name: str, values: Sequence[float], length: int) -> tuple[float, ...]:
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values")
    parsed = tuple(float(value) for value in values)
    if any(not np.isfinite(value) or value <= 0.0 for value in parsed):
        raise ValueError(f"{name} must contain positive finite values")
    return parsed


def _phase_canonical_columns(vectors: np.ndarray) -> np.ndarray:
    canonical = np.array(vectors, dtype=float, copy=True)
    for column in range(canonical.shape[1]):
        absolute = np.abs(canonical[:, column])
        pivot = int(np.flatnonzero(absolute == np.max(absolute))[0])
        if canonical[pivot, column] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


def _deterministic_initial(dimension: int) -> np.ndarray:
    indices = np.arange(1, dimension + 1, dtype=float)
    vector = np.sin(np.sqrt(2.0) * indices) + np.cos(np.sqrt(3.0) * indices)
    return vector / np.linalg.norm(vector)


def _nonempty(name: str, value: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def _readonly(values: np.ndarray) -> np.ndarray:
    return immutable_array(values)


__all__ = [
    "DirectDVRResult",
    "JacobiGridProjection",
    "KineticOperator",
    "ProjectedKineticTerms",
    "TriatomicJ0Hamiltonian",
    "TriatomicJ0KineticOperator",
    "TriatomicJacobiTransform",
    "potential_on_jacobi_grid",
    "solve_triatomic_direct_dvr",
]
