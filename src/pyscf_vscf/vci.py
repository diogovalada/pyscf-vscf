"""Ground-state VSCF modal bases and vibrational configuration interaction."""

from __future__ import annotations

import itertools
import json
import operator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse.linalg import LinearOperator, eigsh

from ._arrays import immutable_array
from ._artifacts import atomic_savez_compressed
from ._identity import (
    FrozenJSONMapping,
    array_identity,
    payload_fingerprint,
    to_jsonable,
)
from .dvr import sinc_kinetic_1d
from .kinetic import TriatomicJ0Hamiltonian
from .vscf import (
    NModePotential,
    VSCFIteration,
    VSCFSettings,
    nmode_model_fingerprint,
)


Configuration = tuple[int, ...]


@runtime_checkable
class ProductGridHamiltonian(Protocol):
    @property
    def coordinate_ids(self) -> tuple[str, ...]: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def dimension(self) -> int: ...

    def apply(self, vector: np.ndarray) -> np.ndarray: ...

    def fingerprint(self) -> str: ...


@dataclass(frozen=True)
class NModeGridHamiltonian:
    """Direct-product grid Hamiltonian for the existing n-mode model."""

    model: NModePotential
    potential_Eh: np.ndarray = field(init=False)
    kinetic_matrices_Eh: tuple[np.ndarray, ...] = field(init=False)

    def __post_init__(self) -> None:
        shape = tuple(grid.size for grid in self.model.coordinates)
        potential = np.zeros(shape, dtype=float)
        for mode, values in enumerate(self.model.one_mode_potentials_Eh):
            view = [1] * self.model.n_modes
            view[mode] = shape[mode]
            potential += values.reshape(view)
        for (first, second), values in self.model.two_mode_couplings_Eh.items():
            view = [1] * self.model.n_modes
            view[first] = shape[first]
            view[second] = shape[second]
            potential += values.reshape(view)
        kinetic = tuple(
            sinc_kinetic_1d(grid, mass)
            for grid, mass in zip(self.model.coordinates, self.model.masses_amu)
        )
        object.__setattr__(self, "potential_Eh", _readonly(potential))
        object.__setattr__(
            self,
            "kinetic_matrices_Eh",
            tuple(_readonly(matrix) for matrix in kinetic),
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(grid.size for grid in self.model.coordinates)

    @property
    def dimension(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))

    @property
    def coordinate_ids(self) -> tuple[str, ...]:
        return tuple(self.model.mode_labels or ())

    def apply(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.shape != (self.dimension,):
            raise ValueError(f"Hamiltonian vector must have shape {(self.dimension,)}")
        wavefunction = values.reshape(self.shape)
        result = self.potential_Eh * wavefunction
        for mode, matrix in enumerate(self.kinetic_matrices_Eh):
            result += _apply_axis(matrix, wavefunction, mode)
        return result.reshape(-1)

    def fingerprint(self) -> str:
        return payload_fingerprint(
            {
                "kind": "nmode-direct-product-grid-hamiltonian",
                "schema_version": 1,
                "model_fingerprint": nmode_model_fingerprint(self.model),
            }
        )


@dataclass(frozen=True)
class GroundModalBasis:
    """Retained eigensystems of converged ground-state VSCF mean fields."""

    coordinate_ids: tuple[str, ...]
    modals: tuple[np.ndarray, ...]
    modal_energies_Eh: tuple[np.ndarray, ...]
    mean_field_hamiltonians_Eh: tuple[np.ndarray, ...]
    densities: tuple[np.ndarray, ...]
    vscf_energy_Eh: float
    converged: bool
    iterations: int
    history: tuple[VSCFIteration, ...]
    kinetic_operator_fingerprint: str
    source_hamiltonian_fingerprint: str
    vscf_settings: VSCFSettings
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = tuple(str(value).strip() for value in self.coordinate_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("coordinate_ids must be non-empty and unique")
        count = len(ids)
        if not all(
            len(values) == count
            for values in (
                self.modals,
                self.modal_energies_Eh,
                self.mean_field_hamiltonians_Eh,
                self.densities,
            )
        ):
            raise ValueError("Ground modal basis fields must align by mode")
        modals = []
        energies = []
        mean_fields = []
        densities = []
        for mode in range(count):
            basis = np.asarray(self.modals[mode], dtype=float)
            modal_energy = np.asarray(self.modal_energies_Eh[mode], dtype=float)
            mean_field = np.asarray(self.mean_field_hamiltonians_Eh[mode], dtype=float)
            density = np.asarray(self.densities[mode], dtype=float)
            if basis.ndim != 2 or basis.shape[1] < 1:
                raise ValueError(f"modals[{mode}] must be a two-dimensional modal basis")
            if modal_energy.shape != (basis.shape[1],):
                raise ValueError(f"modal_energies_Eh[{mode}] does not match its basis")
            if mean_field.shape != (basis.shape[0], basis.shape[0]):
                raise ValueError(f"mean_field_hamiltonians_Eh[{mode}] has the wrong shape")
            if density.shape != (basis.shape[0],):
                raise ValueError(f"densities[{mode}] has the wrong shape")
            if not all(
                np.all(np.isfinite(values))
                for values in (basis, modal_energy, mean_field, density)
            ):
                raise ValueError("Ground modal basis arrays must be finite")
            if not np.allclose(
                basis.T @ basis,
                np.eye(basis.shape[1]),
                rtol=0.0,
                atol=2e-11,
            ):
                raise ValueError(f"modals[{mode}] must have orthonormal columns")
            if not np.allclose(mean_field, mean_field.T, rtol=0.0, atol=2e-13):
                raise ValueError(f"mean_field_hamiltonians_Eh[{mode}] must be Hermitian")
            if np.any(np.diff(modal_energy) < 0.0) or not np.allclose(
                mean_field @ basis,
                basis * modal_energy[None, :],
                rtol=0.0,
                atol=2e-10,
            ):
                raise ValueError(f"modals[{mode}] do not reproduce the mean-field eigensystem")
            if np.any(density < 0.0) or not np.isclose(np.sum(density), 1.0, atol=2e-12):
                raise ValueError(f"densities[{mode}] must be normalized probabilities")
            if not np.allclose(
                density,
                np.square(basis[:, 0]),
                rtol=0.0,
                atol=2e-11,
            ):
                raise ValueError(f"densities[{mode}] does not match the ground modal")
            modals.append(_readonly(basis))
            energies.append(_readonly(modal_energy))
            mean_fields.append(_readonly(mean_field))
            densities.append(_readonly(density))
        energy = float(self.vscf_energy_Eh)
        iterations = operator.index(self.iterations)
        history = tuple(self.history)
        if not isinstance(self.vscf_settings, VSCFSettings):
            raise TypeError("vscf_settings must be VSCFSettings")
        if not np.isfinite(energy) or iterations < 0 or iterations != len(history):
            raise ValueError("Ground modal basis scalar diagnostics are inconsistent")
        if self.converged and (
            not history
            or history[-1].delta_energy_Eh > float(self.vscf_settings.energy_tolerance_Eh)
            or history[-1].max_density_change > float(self.vscf_settings.density_tolerance)
        ):
            raise ValueError("Converged modal basis does not satisfy its retained VSCF settings")
        object.__setattr__(self, "coordinate_ids", ids)
        object.__setattr__(self, "modals", tuple(modals))
        object.__setattr__(self, "modal_energies_Eh", tuple(energies))
        object.__setattr__(self, "mean_field_hamiltonians_Eh", tuple(mean_fields))
        object.__setattr__(self, "densities", tuple(densities))
        object.__setattr__(self, "vscf_energy_Eh", energy)
        object.__setattr__(self, "converged", bool(self.converged))
        object.__setattr__(self, "iterations", iterations)
        object.__setattr__(self, "history", history)
        object.__setattr__(
            self,
            "kinetic_operator_fingerprint",
            _nonempty("kinetic_operator_fingerprint", self.kinetic_operator_fingerprint),
        )
        object.__setattr__(
            self,
            "source_hamiltonian_fingerprint",
            _nonempty(
                "source_hamiltonian_fingerprint",
                self.source_hamiltonian_fingerprint,
            ),
        )
        object.__setattr__(self, "metadata", FrozenJSONMapping.from_mapping(self.metadata))

    @property
    def n_modes(self) -> int:
        return len(self.coordinate_ids)

    @property
    def n_modals_per_mode(self) -> tuple[int, ...]:
        return tuple(basis.shape[1] for basis in self.modals)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "ground-vscf-modal-basis",
            "schema_version": 1,
            "coordinate_ids": list(self.coordinate_ids),
            "modals": [array_identity(values) for values in self.modals],
            "modal_energies_Eh": [array_identity(values) for values in self.modal_energies_Eh],
            "mean_field_hamiltonians_Eh": [
                array_identity(values) for values in self.mean_field_hamiltonians_Eh
            ],
            "densities": [array_identity(values) for values in self.densities],
            "vscf_energy_Eh": self.vscf_energy_Eh,
            "converged": self.converged,
            "iterations": self.iterations,
            "history": [
                {
                    "iteration": item.iteration,
                    "energy_Eh": item.energy_Eh,
                    "delta_energy_Eh": item.delta_energy_Eh,
                    "max_density_change": item.max_density_change,
                }
                for item in self.history
            ],
            "kinetic_operator_fingerprint": self.kinetic_operator_fingerprint,
            "source_hamiltonian_fingerprint": self.source_hamiltonian_fingerprint,
            "vscf_settings": to_jsonable(asdict(self.vscf_settings)),
            "metadata": to_jsonable(self.metadata),
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


def dump_ground_modal_basis(basis: GroundModalBasis, path: Path | str) -> None:
    """Serialize a complete converged VSCF modal basis without pickle."""

    if not isinstance(basis, GroundModalBasis):
        raise TypeError("dump_ground_modal_basis requires a GroundModalBasis")
    arrays: dict[str, np.ndarray] = {}
    for mode in range(basis.n_modes):
        arrays[f"modals_{mode}"] = basis.modals[mode]
        arrays[f"modal_energies_Eh_{mode}"] = basis.modal_energies_Eh[mode]
        arrays[f"mean_field_hamiltonian_Eh_{mode}"] = basis.mean_field_hamiltonians_Eh[mode]
        arrays[f"density_{mode}"] = basis.densities[mode]
    manifest = {
        "schema": "pyscf-vscf-ground-modal-basis",
        "schema_version": 1,
        "coordinate_ids": list(basis.coordinate_ids),
        "vscf_energy_Eh": basis.vscf_energy_Eh,
        "converged": basis.converged,
        "iterations": basis.iterations,
        "history": [asdict(value) for value in basis.history],
        "kinetic_operator_fingerprint": basis.kinetic_operator_fingerprint,
        "source_hamiltonian_fingerprint": basis.source_hamiltonian_fingerprint,
        "vscf_settings": to_jsonable(asdict(basis.vscf_settings)),
        "metadata": to_jsonable(basis.metadata),
        "basis_fingerprint": basis.fingerprint(),
    }
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    atomic_savez_compressed(path, arrays)


def load_ground_modal_basis(path: Path | str) -> GroundModalBasis:
    """Load and fingerprint-check a modal basis artifact."""

    with np.load(Path(path), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        if manifest.get("schema") != "pyscf-vscf-ground-modal-basis":
            raise ValueError("Not a pyscf-vscf ground modal basis artifact")
        if int(manifest.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported ground modal basis artifact schema version")
        coordinate_ids = tuple(manifest["coordinate_ids"])
        basis = GroundModalBasis(
            coordinate_ids=coordinate_ids,
            modals=tuple(
                np.asarray(archive[f"modals_{mode}"], dtype=float)
                for mode in range(len(coordinate_ids))
            ),
            modal_energies_Eh=tuple(
                np.asarray(archive[f"modal_energies_Eh_{mode}"], dtype=float)
                for mode in range(len(coordinate_ids))
            ),
            mean_field_hamiltonians_Eh=tuple(
                np.asarray(archive[f"mean_field_hamiltonian_Eh_{mode}"], dtype=float)
                for mode in range(len(coordinate_ids))
            ),
            densities=tuple(
                np.asarray(archive[f"density_{mode}"], dtype=float)
                for mode in range(len(coordinate_ids))
            ),
            vscf_energy_Eh=manifest["vscf_energy_Eh"],
            converged=manifest["converged"],
            iterations=manifest["iterations"],
            history=tuple(VSCFIteration(**value) for value in manifest["history"]),
            kinetic_operator_fingerprint=manifest["kinetic_operator_fingerprint"],
            source_hamiltonian_fingerprint=manifest["source_hamiltonian_fingerprint"],
            vscf_settings=VSCFSettings(**manifest["vscf_settings"]),
            metadata=manifest["metadata"],
        )
    if manifest.get("basis_fingerprint") != basis.fingerprint():
        raise ValueError("Serialized modal basis fingerprint does not match")
    return basis


def validate_ground_modal_basis_against_hamiltonian(
    basis: GroundModalBasis,
    hamiltonian: ProductGridHamiltonian,
) -> None:
    """Reconstruct final VSCF mean fields from the retained Hamiltonian."""

    if not isinstance(basis, GroundModalBasis) or not isinstance(
        hamiltonian,
        ProductGridHamiltonian,
    ):
        raise TypeError("Modal validation requires a typed basis and product-grid Hamiltonian")
    if basis.source_hamiltonian_fingerprint != hamiltonian.fingerprint():
        raise ValueError("Modal basis does not identify the supplied Hamiltonian")
    if basis.coordinate_ids != tuple(hamiltonian.coordinate_ids) or tuple(
        value.shape[0] for value in basis.modals
    ) != tuple(hamiltonian.shape):
        raise ValueError("Modal basis and Hamiltonian grids do not align")
    ground_modals = tuple(value[:, 0] for value in basis.modals)
    product = _product_vector(ground_modals)
    expected_energy = float(product @ hamiltonian.apply(product))
    if not np.isclose(expected_energy, basis.vscf_energy_Eh, rtol=0.0, atol=2e-10):
        raise ValueError("Modal VSCF energy does not reproduce the retained Hamiltonian")
    for mode, expected in enumerate(basis.mean_field_hamiltonians_Eh):
        size = expected.shape[0]
        vectors = []
        for index in range(size):
            local = np.zeros(size, dtype=float)
            local[index] = 1.0
            factors = list(ground_modals)
            factors[mode] = local
            vectors.append(_product_vector(factors))
        columns = []
        for ket in vectors:
            applied = hamiltonian.apply(ket)
            columns.append(np.array([bra @ applied for bra in vectors]))
        reconstructed = np.column_stack(columns)
        difference = reconstructed - expected
        scalar_offset = float(np.trace(difference) / size)
        if not np.allclose(
            difference,
            scalar_offset * np.eye(size),
            rtol=0.0,
            atol=2e-10,
        ):
            raise ValueError(
                f"Modal mean-field Hamiltonian {mode} does not reproduce the full Hamiltonian"
            )


@dataclass(frozen=True)
class VCISettings:
    nstates: int = 12
    max_quanta_per_mode: int | tuple[int, ...] | None = None
    max_total_quanta: int | None = None
    max_modal_energy_Eh: float | None = None
    extra_eigenstates: int = 2
    dense_dimension_threshold: int = 256
    eigensolver_tolerance: float = 1e-10
    degeneracy_tolerance_Eh: float = 1e-9
    leading_configuration_count: int = 6

    def __post_init__(self) -> None:
        if operator.index(self.nstates) < 1:
            raise ValueError("nstates must be positive")
        if operator.index(self.extra_eigenstates) < 1:
            raise ValueError("extra_eigenstates must be positive")
        if operator.index(self.dense_dimension_threshold) < 1:
            raise ValueError("dense_dimension_threshold must be positive")
        if operator.index(self.leading_configuration_count) < 1:
            raise ValueError("leading_configuration_count must be positive")
        for name in ("eigensolver_tolerance", "degeneracy_tolerance_Eh"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if self.max_total_quanta is not None and operator.index(self.max_total_quanta) < 0:
            raise ValueError("max_total_quanta must be non-negative")
        if self.max_modal_energy_Eh is not None:
            cutoff = float(self.max_modal_energy_Eh)
            if not np.isfinite(cutoff) or cutoff < 0.0:
                raise ValueError("max_modal_energy_Eh must be finite and non-negative")


@dataclass(frozen=True)
class VCIStateAssignment:
    state_index: int
    configuration: Configuration | None
    overlap: float | None
    leading_configurations: tuple[tuple[Configuration, float], ...]
    participation_ratio: float
    degenerate_block: tuple[int, ...]
    reference_subspace: tuple[Configuration, ...]
    principal_cosines: tuple[float, ...]
    subspace_overlap: float | None
    manual_review: bool

    def __post_init__(self) -> None:
        state = operator.index(self.state_index)
        block = tuple(operator.index(value) for value in self.degenerate_block)
        if state < 0 or not block or state not in block or tuple(sorted(set(block))) != block:
            raise ValueError("degenerate_block must be ordered, unique, and contain state_index")
        configuration = _configuration_or_none(self.configuration)
        overlap = _probability_or_none("overlap", self.overlap)
        leading = tuple(
            (_configuration(item), _probability("leading overlap", weight))
            for item, weight in self.leading_configurations
        )
        participation = float(self.participation_ratio)
        if not np.isfinite(participation) or participation < 1.0 - 1e-12:
            raise ValueError("participation_ratio must be finite and at least one")
        reference = tuple(_configuration(item) for item in self.reference_subspace)
        cosines = tuple(float(value) for value in self.principal_cosines)
        if any(
            not np.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12 for value in cosines
        ):
            raise ValueError("principal_cosines must lie between zero and one")
        subspace_overlap = _probability_or_none("subspace_overlap", self.subspace_overlap)
        manual = bool(self.manual_review)
        if len(block) == 1:
            if configuration is None or overlap is None or reference or cosines:
                raise ValueError("Nondegenerate assignments require one configuration only")
            if subspace_overlap is not None or manual:
                raise ValueError("Nondegenerate assignments cannot require subspace review")
        else:
            if configuration is not None or overlap is not None or not manual:
                raise ValueError("Degenerate assignments must be marked for manual review")
            if len(reference) != len(block) or len(cosines) != len(block):
                raise ValueError("Degenerate subspace diagnostics must match the block size")
            expected = min(cosines) ** 2
            if subspace_overlap is None or not np.isclose(
                subspace_overlap, expected, rtol=0.0, atol=2e-12
            ):
                raise ValueError("subspace_overlap must be the squared minimum principal cosine")
        object.__setattr__(self, "state_index", state)
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(self, "leading_configurations", leading)
        object.__setattr__(self, "participation_ratio", participation)
        object.__setattr__(self, "degenerate_block", block)
        object.__setattr__(self, "reference_subspace", reference)
        object.__setattr__(self, "principal_cosines", cosines)
        object.__setattr__(self, "subspace_overlap", subspace_overlap)
        object.__setattr__(self, "manual_review", manual)


@dataclass(frozen=True)
class VCIDegenerateBlock:
    """Basis-invariant diagnostics for a numerically degenerate root block."""

    state_indices: tuple[int, ...]
    reference_configurations: tuple[Configuration, ...]
    invariant_projector: np.ndarray
    principal_cosines: tuple[float, ...]
    crosses_requested_cutoff: bool

    def __post_init__(self) -> None:
        states = tuple(operator.index(value) for value in self.state_indices)
        references = tuple(_configuration(item) for item in self.reference_configurations)
        projector = np.asarray(self.invariant_projector, dtype=float)
        cosines = tuple(float(value) for value in self.principal_cosines)
        if len(states) < 2 or tuple(sorted(set(states))) != states:
            raise ValueError("A degenerate block needs at least two ordered unique states")
        if len(references) != len(states) or len(set(references)) != len(references):
            raise ValueError("Reference configurations must be unique and match the block size")
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise ValueError("invariant_projector must be square")
        if not np.all(np.isfinite(projector)):
            raise ValueError("invariant_projector must be finite")
        if not np.allclose(projector, projector.T, rtol=0.0, atol=2e-11):
            raise ValueError("invariant_projector must be Hermitian")
        if not np.allclose(projector @ projector, projector, rtol=0.0, atol=5e-10):
            raise ValueError("invariant_projector must be idempotent")
        if not np.isclose(np.trace(projector), len(states), rtol=0.0, atol=5e-10):
            raise ValueError("invariant_projector rank does not match the block size")
        if len(cosines) != len(states) or any(
            not np.isfinite(value) or value < -1e-12 or value > 1.0 + 1e-12 for value in cosines
        ):
            raise ValueError("principal_cosines must match the block and lie in [0, 1]")
        object.__setattr__(self, "state_indices", states)
        object.__setattr__(self, "reference_configurations", references)
        object.__setattr__(self, "invariant_projector", _readonly(projector))
        object.__setattr__(self, "principal_cosines", cosines)
        object.__setattr__(self, "crosses_requested_cutoff", bool(self.crosses_requested_cutoff))


@dataclass(frozen=True)
class VCIResult:
    configurations: tuple[Configuration, ...]
    energies_Eh: np.ndarray
    coefficients: np.ndarray
    residual_norms_Eh: np.ndarray
    diagnostic_energies_Eh: np.ndarray
    diagnostic_coefficients: np.ndarray
    diagnostic_residual_norms_Eh: np.ndarray
    assignments: tuple[VCIStateAssignment, ...]
    degenerate_blocks: tuple[VCIDegenerateBlock, ...]
    state_cutoff_margin_Eh: float
    modal_basis_fingerprint: str
    hamiltonian_fingerprint: str
    modal_counts: tuple[int, ...]
    vscf_settings: VSCFSettings
    vci_settings: VCISettings

    def __post_init__(self) -> None:
        configurations = tuple(
            tuple(operator.index(value) for value in item) for item in self.configurations
        )
        energies = np.asarray(self.energies_Eh, dtype=float)
        coefficients = np.asarray(self.coefficients, dtype=float)
        residuals = np.asarray(self.residual_norms_Eh, dtype=float)
        diagnostic_energies = np.asarray(self.diagnostic_energies_Eh, dtype=float)
        diagnostic_coefficients = np.asarray(self.diagnostic_coefficients, dtype=float)
        diagnostic_residuals = np.asarray(self.diagnostic_residual_norms_Eh, dtype=float)
        if not configurations or len(set(configurations)) != len(configurations):
            raise ValueError("VCI configurations must be non-empty and unique")
        if energies.ndim != 1 or coefficients.shape != (len(configurations), energies.size):
            raise ValueError("VCI eigenpair arrays are inconsistent")
        if residuals.shape != energies.shape or len(self.assignments) != energies.size:
            raise ValueError("VCI diagnostics do not align with returned states")
        if (
            diagnostic_energies.ndim != 1
            or diagnostic_coefficients.shape != (len(configurations), diagnostic_energies.size)
            or diagnostic_residuals.shape != diagnostic_energies.shape
            or diagnostic_energies.size <= energies.size
        ):
            raise ValueError("VCI cutoff diagnostics must include at least one extra root")
        if not np.array_equal(diagnostic_energies[: energies.size], energies):
            raise ValueError("Returned VCI energies must prefix the diagnostic roots")
        if not np.array_equal(diagnostic_coefficients[:, : energies.size], coefficients):
            raise ValueError("Returned VCI coefficients must prefix the diagnostic roots")
        if not np.array_equal(diagnostic_residuals[: energies.size], residuals):
            raise ValueError("Returned residuals must prefix the diagnostic residuals")
        if not all(
            np.all(np.isfinite(values))
            for values in (
                energies,
                coefficients,
                residuals,
                diagnostic_energies,
                diagnostic_coefficients,
                diagnostic_residuals,
            )
        ):
            raise ValueError("VCI arrays must be finite")
        if np.any(diagnostic_residuals < 0.0):
            raise ValueError("VCI residuals must be non-negative")
        if not np.allclose(
            diagnostic_coefficients.T @ diagnostic_coefficients,
            np.eye(diagnostic_energies.size),
            rtol=0.0,
            atol=2e-11,
        ):
            raise ValueError("Diagnostic VCI coefficients must be orthonormal")
        assignments = tuple(self.assignments)
        if tuple(item.state_index for item in assignments) != tuple(range(energies.size)):
            raise ValueError("VCI assignments must be ordered by returned state")
        blocks = tuple(self.degenerate_blocks)
        if any(max(block.state_indices) >= diagnostic_energies.size for block in blocks):
            raise ValueError("A degenerate block references an unavailable diagnostic root")
        if any(block.invariant_projector.shape != (len(configurations),) * 2 for block in blocks):
            raise ValueError("Degenerate projectors must act in the VCI configuration basis")
        margin = float(self.state_cutoff_margin_Eh)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("state_cutoff_margin_Eh must be finite and non-negative")
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "energies_Eh", _readonly(energies))
        object.__setattr__(self, "coefficients", _readonly(coefficients))
        object.__setattr__(self, "residual_norms_Eh", _readonly(residuals))
        object.__setattr__(self, "diagnostic_energies_Eh", _readonly(diagnostic_energies))
        object.__setattr__(
            self,
            "diagnostic_coefficients",
            _readonly(diagnostic_coefficients),
        )
        object.__setattr__(
            self,
            "diagnostic_residual_norms_Eh",
            _readonly(diagnostic_residuals),
        )
        object.__setattr__(self, "assignments", assignments)
        object.__setattr__(self, "degenerate_blocks", blocks)
        object.__setattr__(self, "state_cutoff_margin_Eh", margin)
        object.__setattr__(
            self,
            "modal_basis_fingerprint",
            _nonempty("modal_basis_fingerprint", self.modal_basis_fingerprint),
        )
        object.__setattr__(
            self,
            "hamiltonian_fingerprint",
            _nonempty("hamiltonian_fingerprint", self.hamiltonian_fingerprint),
        )
        counts = tuple(operator.index(value) for value in self.modal_counts)
        if len(counts) != len(configurations[0]) or any(value < 1 for value in counts):
            raise ValueError("modal_counts must contain one positive count per mode")
        if any(
            quantum >= count
            for configuration in configurations
            for quantum, count in zip(configuration, counts)
        ):
            raise ValueError("VCI configurations exceed the retained modal counts")
        if not isinstance(self.vscf_settings, VSCFSettings):
            raise TypeError("vscf_settings must be VSCFSettings")
        if not isinstance(self.vci_settings, VCISettings):
            raise TypeError("vci_settings must be VCISettings")
        if operator.index(self.vci_settings.nstates) != energies.size:
            raise ValueError("VCI settings nstates does not match the returned roots")
        if diagnostic_energies.size < energies.size + operator.index(
            self.vci_settings.extra_eigenstates
        ):
            raise ValueError("VCI diagnostics do not retain the requested extra roots")
        quanta_limits = _quanta_limits(self.vci_settings.max_quanta_per_mode, counts)
        if any(
            any(value > limit for value, limit in zip(configuration, quanta_limits))
            or (
                self.vci_settings.max_total_quanta is not None
                and sum(configuration) > operator.index(self.vci_settings.max_total_quanta)
            )
            for configuration in configurations
        ):
            raise ValueError("VCI configurations violate the retained pruning settings")
        object.__setattr__(self, "modal_counts", counts)

    def fingerprint_payload(self) -> Mapping[str, object]:
        return {
            "kind": "vci-result",
            "schema_version": 1,
            "configurations": [list(item) for item in self.configurations],
            "energies_Eh": array_identity(self.energies_Eh),
            "coefficients": array_identity(self.coefficients),
            "residual_norms_Eh": array_identity(self.residual_norms_Eh),
            "diagnostic_energies_Eh": array_identity(self.diagnostic_energies_Eh),
            "diagnostic_coefficients": array_identity(self.diagnostic_coefficients),
            "diagnostic_residual_norms_Eh": array_identity(self.diagnostic_residual_norms_Eh),
            "assignments": [
                {
                    "state_index": item.state_index,
                    "configuration": (
                        None if item.configuration is None else list(item.configuration)
                    ),
                    "overlap": item.overlap,
                    "leading_configurations": [
                        [list(configuration), weight]
                        for configuration, weight in item.leading_configurations
                    ],
                    "participation_ratio": item.participation_ratio,
                    "degenerate_block": list(item.degenerate_block),
                    "reference_subspace": [
                        list(configuration) for configuration in item.reference_subspace
                    ],
                    "principal_cosines": list(item.principal_cosines),
                    "subspace_overlap": item.subspace_overlap,
                    "manual_review": item.manual_review,
                }
                for item in self.assignments
            ],
            "degenerate_blocks": [
                {
                    "state_indices": list(block.state_indices),
                    "reference_configurations": [
                        list(configuration) for configuration in block.reference_configurations
                    ],
                    "invariant_projector": array_identity(block.invariant_projector),
                    "principal_cosines": list(block.principal_cosines),
                    "crosses_requested_cutoff": block.crosses_requested_cutoff,
                }
                for block in self.degenerate_blocks
            ],
            "state_cutoff_margin_Eh": self.state_cutoff_margin_Eh,
            "modal_basis_fingerprint": self.modal_basis_fingerprint,
            "hamiltonian_fingerprint": self.hamiltonian_fingerprint,
            "modal_counts": list(self.modal_counts),
            "vscf_settings": to_jsonable(asdict(self.vscf_settings)),
            "vci_settings": to_jsonable(asdict(self.vci_settings)),
        }

    def fingerprint(self) -> str:
        return payload_fingerprint(self.fingerprint_payload())


def dump_vci_result(result: VCIResult, path: Path | str) -> None:
    """Serialize a VCI result without pickled Python objects."""

    if not isinstance(result, VCIResult):
        raise TypeError("dump_vci_result requires a VCIResult")
    arrays: dict[str, np.ndarray] = {
        "configurations": np.asarray(result.configurations, dtype="<i8"),
        "energies_Eh": result.energies_Eh,
        "coefficients": result.coefficients,
        "residual_norms_Eh": result.residual_norms_Eh,
        "diagnostic_energies_Eh": result.diagnostic_energies_Eh,
        "diagnostic_coefficients": result.diagnostic_coefficients,
        "diagnostic_residual_norms_Eh": result.diagnostic_residual_norms_Eh,
    }
    blocks = []
    for index, block in enumerate(result.degenerate_blocks):
        arrays[f"degenerate_projector_{index}"] = block.invariant_projector
        blocks.append(
            {
                "state_indices": list(block.state_indices),
                "reference_configurations": [
                    list(value) for value in block.reference_configurations
                ],
                "principal_cosines": list(block.principal_cosines),
                "crosses_requested_cutoff": block.crosses_requested_cutoff,
            }
        )
    manifest = {
        "schema": "pyscf-vscf-vci-result",
        "schema_version": 1,
        "assignments": [
            {
                "state_index": value.state_index,
                "configuration": (
                    None if value.configuration is None else list(value.configuration)
                ),
                "overlap": value.overlap,
                "leading_configurations": [
                    [list(configuration), weight]
                    for configuration, weight in value.leading_configurations
                ],
                "participation_ratio": value.participation_ratio,
                "degenerate_block": list(value.degenerate_block),
                "reference_subspace": [
                    list(configuration) for configuration in value.reference_subspace
                ],
                "principal_cosines": list(value.principal_cosines),
                "subspace_overlap": value.subspace_overlap,
                "manual_review": value.manual_review,
            }
            for value in result.assignments
        ],
        "degenerate_blocks": blocks,
        "state_cutoff_margin_Eh": result.state_cutoff_margin_Eh,
        "modal_basis_fingerprint": result.modal_basis_fingerprint,
        "hamiltonian_fingerprint": result.hamiltonian_fingerprint,
        "modal_counts": list(result.modal_counts),
        "vscf_settings": to_jsonable(asdict(result.vscf_settings)),
        "vci_settings": to_jsonable(asdict(result.vci_settings)),
        "result_fingerprint": result.fingerprint(),
    }
    arrays["manifest_json"] = np.asarray(
        json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    atomic_savez_compressed(path, arrays)


def load_vci_result(path: Path | str) -> VCIResult:
    """Load and fingerprint-check a result written by :func:`dump_vci_result`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        manifest = json.loads(str(archive["manifest_json"].item()))
        if manifest.get("schema") != "pyscf-vscf-vci-result":
            raise ValueError("Not a pyscf-vscf VCI result artifact")
        if int(manifest.get("schema_version", -1)) != 1:
            raise ValueError("Unsupported VCI result artifact schema version")
        assignments = tuple(
            VCIStateAssignment(
                state_index=value["state_index"],
                configuration=(
                    None if value["configuration"] is None else tuple(value["configuration"])
                ),
                overlap=value["overlap"],
                leading_configurations=tuple(
                    (tuple(configuration), weight)
                    for configuration, weight in value["leading_configurations"]
                ),
                participation_ratio=value["participation_ratio"],
                degenerate_block=tuple(value["degenerate_block"]),
                reference_subspace=tuple(
                    tuple(configuration) for configuration in value["reference_subspace"]
                ),
                principal_cosines=tuple(value["principal_cosines"]),
                subspace_overlap=value["subspace_overlap"],
                manual_review=value["manual_review"],
            )
            for value in manifest["assignments"]
        )
        blocks = tuple(
            VCIDegenerateBlock(
                state_indices=tuple(value["state_indices"]),
                reference_configurations=tuple(
                    tuple(configuration) for configuration in value["reference_configurations"]
                ),
                invariant_projector=np.asarray(
                    archive[f"degenerate_projector_{index}"], dtype=float
                ),
                principal_cosines=tuple(value["principal_cosines"]),
                crosses_requested_cutoff=value["crosses_requested_cutoff"],
            )
            for index, value in enumerate(manifest["degenerate_blocks"])
        )
        result = VCIResult(
            configurations=tuple(
                tuple(int(component) for component in row)
                for row in np.asarray(archive["configurations"], dtype=np.int64)
            ),
            energies_Eh=np.asarray(archive["energies_Eh"], dtype=float),
            coefficients=np.asarray(archive["coefficients"], dtype=float),
            residual_norms_Eh=np.asarray(archive["residual_norms_Eh"], dtype=float),
            diagnostic_energies_Eh=np.asarray(archive["diagnostic_energies_Eh"], dtype=float),
            diagnostic_coefficients=np.asarray(archive["diagnostic_coefficients"], dtype=float),
            diagnostic_residual_norms_Eh=np.asarray(
                archive["diagnostic_residual_norms_Eh"], dtype=float
            ),
            assignments=assignments,
            degenerate_blocks=blocks,
            state_cutoff_margin_Eh=manifest["state_cutoff_margin_Eh"],
            modal_basis_fingerprint=manifest["modal_basis_fingerprint"],
            hamiltonian_fingerprint=manifest["hamiltonian_fingerprint"],
            modal_counts=tuple(manifest["modal_counts"]),
            vscf_settings=VSCFSettings(**manifest["vscf_settings"]),
            vci_settings=VCISettings(
                **{
                    **manifest["vci_settings"],
                    "max_quanta_per_mode": (
                        None
                        if manifest["vci_settings"]["max_quanta_per_mode"] is None
                        else tuple(manifest["vci_settings"]["max_quanta_per_mode"])
                        if isinstance(manifest["vci_settings"]["max_quanta_per_mode"], list)
                        else manifest["vci_settings"]["max_quanta_per_mode"]
                    ),
                }
            ),
        )
    if manifest.get("result_fingerprint") != result.fingerprint():
        raise ValueError("Serialized VCI result fingerprint does not match")
    return result


def build_nmode_vscf_modal_basis(
    model: NModePotential,
    n_modals_per_mode: int | Sequence[int],
    *,
    settings: VSCFSettings | None = None,
) -> tuple[NModeGridHamiltonian, GroundModalBasis]:
    """Converge a ground-state VSCF basis for the separable sinc operator."""

    policy = settings or VSCFSettings()
    counts = _modal_counts(n_modals_per_mode, tuple(grid.size for grid in model.coordinates))
    hamiltonian = NModeGridHamiltonian(model)
    one_mode = tuple(
        kinetic + np.diag(potential)
        for kinetic, potential in zip(
            hamiltonian.kinetic_matrices_Eh,
            model.one_mode_potentials_Eh,
        )
    )
    modals = [np.linalg.eigh(matrix)[1][:, 0] for matrix in one_mode]

    def mean_fields(current: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        probabilities = tuple(np.square(modal) for modal in current)
        fields = []
        for mode, base in enumerate(one_mode):
            correction = np.zeros(base.shape[0], dtype=float)
            for (first, second), coupling in model.two_mode_couplings_Eh.items():
                if mode == first:
                    correction += coupling @ probabilities[second]
                elif mode == second:
                    correction += probabilities[first] @ coupling
            fields.append(base + np.diag(correction))
        return tuple(fields)

    def product_energy(current: Sequence[np.ndarray]) -> float:
        wavefunction = _product_vector(current)
        return float(wavefunction @ hamiltonian.apply(wavefunction))

    modals, mean_field_matrices, history, converged = _iterate_ground_vscf(
        modals,
        mean_fields,
        product_energy,
        policy,
    )
    basis = _finalize_modal_basis(
        coordinate_ids=tuple(model.mode_labels or ()),
        counts=counts,
        mean_fields=mean_field_matrices,
        product_energy=product_energy,
        converged=converged,
        history=history,
        kinetic_fingerprint=payload_fingerprint(
            {
                "kind": "separable-colbert-miller-sinc",
                "schema_version": 1,
                "model_fingerprint": nmode_model_fingerprint(model),
            }
        ),
        source_hamiltonian_fingerprint=hamiltonian.fingerprint(),
        vscf_settings=policy,
        metadata={"source": "NModePotential"},
    )
    return hamiltonian, basis


def build_triatomic_vscf_modal_basis(
    hamiltonian: TriatomicJ0Hamiltonian,
    n_modals_per_mode: int | Sequence[int],
    *,
    settings: VSCFSettings | None = None,
) -> GroundModalBasis:
    """Converge ground-state modals with the complete Jacobi kinetic operator."""

    policy = settings or VSCFSettings()
    counts = _modal_counts(n_modals_per_mode, hamiltonian.shape)
    kinetic = hamiltonian.kinetic
    uniform_modals = tuple(np.full(size, 1.0 / np.sqrt(size)) for size in hamiltonian.shape)
    initial_fields = _triatomic_mean_fields(hamiltonian, uniform_modals)
    modals = [np.linalg.eigh(matrix)[1][:, 0] for matrix in initial_fields]

    def mean_fields(current: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
        return _triatomic_mean_fields(hamiltonian, current)

    def product_energy(current: Sequence[np.ndarray]) -> float:
        wavefunction = _product_vector(current)
        return float(wavefunction @ hamiltonian.apply(wavefunction))

    modals, mean_field_matrices, history, converged = _iterate_ground_vscf(
        modals,
        mean_fields,
        product_energy,
        policy,
    )
    del modals
    return _finalize_modal_basis(
        coordinate_ids=kinetic.coordinate_ids,
        counts=counts,
        mean_fields=mean_field_matrices,
        product_energy=product_energy,
        converged=converged,
        history=history,
        kinetic_fingerprint=kinetic.fingerprint(),
        source_hamiltonian_fingerprint=hamiltonian.fingerprint(),
        vscf_settings=policy,
        metadata={
            "source": "TriatomicJ0Hamiltonian",
            "hamiltonian_fingerprint": hamiltonian.fingerprint(),
        },
    )


def enumerate_vci_configurations(
    modal_basis: GroundModalBasis,
    settings: VCISettings,
) -> tuple[Configuration, ...]:
    """Enumerate deterministic modal configurations under all pruning rules."""

    counts = modal_basis.n_modals_per_mode
    per_mode = _quanta_limits(settings.max_quanta_per_mode, counts)
    energy_cutoff = (
        None if settings.max_modal_energy_Eh is None else float(settings.max_modal_energy_Eh)
    )
    candidates = []
    for configuration in itertools.product(*(range(count) for count in counts)):
        if any(value > limit for value, limit in zip(configuration, per_mode)):
            continue
        if settings.max_total_quanta is not None and sum(configuration) > operator.index(
            settings.max_total_quanta
        ):
            continue
        excitation = _uncoupled_excitation(modal_basis, configuration)
        if energy_cutoff is not None and excitation > energy_cutoff:
            continue
        candidates.append((tuple(configuration), excitation))
    ground = (0,) * modal_basis.n_modes
    if ground not in {configuration for configuration, _ in candidates}:
        raise ValueError("VCI pruning settings exclude the all-zero configuration")
    return tuple(
        configuration
        for configuration, _ in sorted(
            candidates,
            key=lambda item: (sum(item[0]), item[1], item[0]),
        )
    )


def validate_vci_result_against_modal_basis(
    result: VCIResult,
    modal_basis: GroundModalBasis,
) -> None:
    """Authenticate VCI configuration and assignment claims against a modal basis."""

    if not isinstance(result, VCIResult) or not isinstance(modal_basis, GroundModalBasis):
        raise TypeError("VCI validation requires typed result and modal basis values")
    if result.modal_basis_fingerprint != modal_basis.fingerprint():
        raise ValueError("VCI result does not derive from the retained modal basis")
    if result.modal_counts != modal_basis.n_modals_per_mode:
        raise ValueError("VCI modal counts do not match the retained modal basis")
    if result.vscf_settings != modal_basis.vscf_settings:
        raise ValueError("VCI VSCF settings do not match the retained modal basis")
    if result.hamiltonian_fingerprint != modal_basis.source_hamiltonian_fingerprint:
        raise ValueError("VCI and modal-basis Hamiltonian identities do not match")
    canonical = enumerate_vci_configurations(modal_basis, result.vci_settings)
    if result.configurations != canonical:
        raise ValueError("VCI configurations are not the complete canonical pruned basis")
    assignments, blocks = _assign_vci_states(
        result.configurations,
        result.diagnostic_energies_Eh,
        result.diagnostic_coefficients,
        requested_count=result.energies_Eh.size,
        degeneracy_tolerance=float(result.vci_settings.degeneracy_tolerance_Eh),
        leading_count=operator.index(result.vci_settings.leading_configuration_count),
    )
    if _assignment_identity(assignments, blocks) != _assignment_identity(
        result.assignments,
        result.degenerate_blocks,
    ):
        raise ValueError("VCI assignments do not reproduce the retained eigenvectors")
    expected_margin = float(
        result.diagnostic_energies_Eh[result.energies_Eh.size]
        - result.diagnostic_energies_Eh[result.energies_Eh.size - 1]
    )
    if result.state_cutoff_margin_Eh != expected_margin:
        raise ValueError("VCI cutoff margin does not reproduce the diagnostic roots")


def validate_vci_result_against_hamiltonian(
    result: VCIResult,
    modal_basis: GroundModalBasis,
    hamiltonian: ProductGridHamiltonian,
) -> None:
    """Replay every retained diagnostic VCI eigenpair through the Hamiltonian."""

    validate_vci_result_against_modal_basis(result, modal_basis)
    if not isinstance(hamiltonian, ProductGridHamiltonian):
        raise TypeError("VCI Hamiltonian validation requires a ProductGridHamiltonian")
    if result.hamiltonian_fingerprint != hamiltonian.fingerprint():
        raise ValueError("VCI result does not identify the supplied Hamiltonian")
    projector = _ProductBasisProjector(modal_basis.modals, result.configurations)

    def matvec(vector: np.ndarray) -> np.ndarray:
        grid_vector = projector.to_grid(vector)
        return projector.from_grid(hamiltonian.apply(grid_vector))

    replayed_energies = []
    replayed_residuals = []
    for state, energy in enumerate(result.diagnostic_energies_Eh):
        coefficients = result.diagnostic_coefficients[:, state]
        projected = matvec(coefficients)
        replayed_energies.append(float(coefficients @ projected))
        replayed_residuals.append(float(np.linalg.norm(projected - energy * coefficients)))
    if not np.allclose(
        replayed_energies,
        result.diagnostic_energies_Eh,
        rtol=0.0,
        atol=2e-11,
    ):
        raise ValueError("VCI diagnostic energies do not reproduce the Hamiltonian")
    if not np.allclose(
        replayed_residuals,
        result.diagnostic_residual_norms_Eh,
        rtol=1e-9,
        atol=2e-13,
    ):
        raise ValueError("VCI diagnostic residuals do not reproduce the Hamiltonian")
    residual_limit = 10.0 * float(result.vci_settings.eigensolver_tolerance)
    if any(value > residual_limit for value in replayed_residuals):
        raise ValueError("VCI diagnostic residual exceeds the retained eigensolver tolerance")

    lowest_energies, _ = _solve_projected_roots(
        matvec,
        dimension=len(result.configurations),
        count=result.diagnostic_energies_Eh.size,
        dense_dimension_threshold=operator.index(result.vci_settings.dense_dimension_threshold),
        eigensolver_tolerance=float(result.vci_settings.eigensolver_tolerance),
    )
    spectral_tolerance = max(
        2e-11,
        20.0 * float(result.vci_settings.eigensolver_tolerance),
    )
    if not np.allclose(
        lowest_energies,
        result.diagnostic_energies_Eh,
        rtol=0.0,
        atol=spectral_tolerance,
    ):
        raise ValueError("VCI diagnostics are not the requested lowest Hamiltonian roots")


def vci_state_probability_marginals(
    modal_basis: GroundModalBasis,
    result: VCIResult,
    state_index: int,
) -> tuple[np.ndarray, ...]:
    """Reconstruct normalized DVR probability marginals for one retained VCI state."""

    validate_vci_result_against_modal_basis(result, modal_basis)
    state = operator.index(state_index)
    if state < 0 or state >= result.energies_Eh.size:
        raise IndexError("state_index is outside the retained VCI roots")
    projector = _ProductBasisProjector(modal_basis.modals, result.configurations)
    wavefunction = projector.to_grid(result.coefficients[:, state]).reshape(projector.grid_shape)
    density = np.square(wavefunction)
    normalization = float(np.sum(density))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("Retained VCI state has invalid grid normalization")
    density /= normalization
    marginals = []
    for mode in range(density.ndim):
        axes = tuple(index for index in range(density.ndim) if index != mode)
        marginal = density if not axes else np.sum(density, axis=axes)
        marginals.append(_readonly(np.asarray(marginal, dtype=float)))
    return tuple(marginals)


def solve_vci(
    hamiltonian: ProductGridHamiltonian,
    modal_basis: GroundModalBasis,
    *,
    settings: VCISettings | None = None,
) -> VCIResult:
    """Solve VCI in a deterministic pruned product of ground-state modals.

    The projected Hamiltonian is matrix-free, but every matrix-vector product
    expands through the complete product DVR grid. This implementation is
    therefore intended for small reduced-dimensional and triatomic bases.
    """

    policy = settings or VCISettings()
    if not modal_basis.converged:
        raise ValueError("VCI requires a converged ground-state modal basis")
    if modal_basis.coordinate_ids != tuple(hamiltonian.coordinate_ids):
        raise ValueError("Modal coordinate IDs do not match the Hamiltonian")
    hamiltonian_fingerprint = hamiltonian.fingerprint()
    if modal_basis.source_hamiltonian_fingerprint != hamiltonian_fingerprint:
        raise ValueError("Modal basis was not generated from this Hamiltonian")
    if tuple(basis.shape[0] for basis in modal_basis.modals) != tuple(hamiltonian.shape):
        raise ValueError("Modal basis grid dimensions do not match the Hamiltonian")
    configurations = enumerate_vci_configurations(modal_basis, policy)
    requested = operator.index(policy.nstates)
    extra = operator.index(policy.extra_eigenstates)
    solve_count = requested + extra
    if solve_count > len(configurations):
        raise ValueError("VCI basis is too small for the requested states and cutoff margin")
    projector = _ProductBasisProjector(modal_basis.modals, configurations)

    def matvec(vector: np.ndarray) -> np.ndarray:
        grid = projector.to_grid(vector)
        return projector.from_grid(hamiltonian.apply(grid))

    dimension = len(configurations)
    computed_energies, computed_coefficients = _solve_projected_roots(
        matvec,
        dimension=dimension,
        count=solve_count,
        dense_dimension_threshold=operator.index(policy.dense_dimension_threshold),
        eigensolver_tolerance=float(policy.eigensolver_tolerance),
    )
    while _diagnostic_window_is_open(
        computed_energies,
        requested_count=requested,
        full_dimension=dimension,
        degeneracy_tolerance=float(policy.degeneracy_tolerance_Eh),
    ):
        solve_count = min(dimension, solve_count + max(1, extra))
        computed_energies, computed_coefficients = _solve_projected_roots(
            matvec,
            dimension=dimension,
            count=solve_count,
            dense_dimension_threshold=operator.index(policy.dense_dimension_threshold),
            eigensolver_tolerance=float(policy.eigensolver_tolerance),
        )
    computed_coefficients = _phase_canonical_configurations(
        computed_coefficients,
        configurations,
    )
    residuals = np.array(
        [
            np.linalg.norm(
                matvec(computed_coefficients[:, state])
                - computed_energies[state] * computed_coefficients[:, state]
            )
            for state in range(solve_count)
        ]
    )
    if np.any(residuals > float(policy.eigensolver_tolerance) * 10.0):
        raise RuntimeError(
            f"VCI eigenpair residual exceeds tolerance: max={float(np.max(residuals)):.3e} Eh"
        )
    energies = computed_energies[:requested]
    coefficients = computed_coefficients[:, :requested]
    assignments, degenerate_blocks = _assign_vci_states(
        configurations,
        computed_energies,
        computed_coefficients,
        requested_count=requested,
        degeneracy_tolerance=float(policy.degeneracy_tolerance_Eh),
        leading_count=operator.index(policy.leading_configuration_count),
    )
    margin = float(computed_energies[requested] - computed_energies[requested - 1])
    return VCIResult(
        configurations=configurations,
        energies_Eh=energies,
        coefficients=coefficients,
        residual_norms_Eh=residuals[:requested],
        diagnostic_energies_Eh=computed_energies,
        diagnostic_coefficients=computed_coefficients,
        diagnostic_residual_norms_Eh=residuals,
        assignments=assignments,
        degenerate_blocks=degenerate_blocks,
        state_cutoff_margin_Eh=margin,
        modal_basis_fingerprint=modal_basis.fingerprint(),
        hamiltonian_fingerprint=hamiltonian_fingerprint,
        modal_counts=modal_basis.n_modals_per_mode,
        vscf_settings=modal_basis.vscf_settings,
        vci_settings=policy,
    )


def _solve_projected_roots(
    matvec: Callable[[np.ndarray], np.ndarray],
    *,
    dimension: int,
    count: int,
    dense_dimension_threshold: int,
    eigensolver_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    if dimension <= dense_dimension_threshold or count == dimension:
        identity = np.eye(dimension)
        dense = np.column_stack([matvec(identity[:, column]) for column in range(dimension)])
        asymmetry = float(np.max(np.abs(dense - dense.T)))
        if asymmetry > 2e-11:
            raise RuntimeError(f"Projected VCI Hamiltonian is not Hermitian: {asymmetry:.3e}")
        all_energies, all_coefficients = np.linalg.eigh(0.5 * (dense + dense.T))
        return all_energies[:count], all_coefficients[:, :count]

    linear = LinearOperator(
        (dimension, dimension),
        matvec=matvec,
        rmatvec=matvec,
        dtype=float,
    )
    energies, coefficients = eigsh(
        linear,
        k=count,
        which="SA",
        v0=_deterministic_initial(dimension),
        tol=eigensolver_tolerance,
    )
    order = np.argsort(energies)
    return energies[order], coefficients[:, order]


def _diagnostic_window_is_open(
    energies: np.ndarray,
    *,
    requested_count: int,
    full_dimension: int,
    degeneracy_tolerance: float,
) -> bool:
    if energies.size == full_dimension:
        return False
    final_block = _degenerate_blocks(energies, degeneracy_tolerance)[-1]
    return final_block[0] < requested_count


@dataclass(frozen=True)
class _ProductBasisProjector:
    modal_bases: tuple[np.ndarray, ...]
    configurations: tuple[Configuration, ...]
    modal_shape: tuple[int, ...] = field(init=False)
    grid_shape: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        bases = tuple(np.asarray(values, dtype=float) for values in self.modal_bases)
        if not bases:
            raise ValueError("Product basis requires at least one mode")
        modal_shape = tuple(values.shape[1] for values in bases)
        grid_shape = tuple(values.shape[0] for values in bases)
        for mode, values in enumerate(bases):
            if values.ndim != 2 or not np.allclose(
                values.T @ values,
                np.eye(values.shape[1]),
                atol=2e-11,
            ):
                raise ValueError(f"modal_bases[{mode}] is not orthonormal")
        configurations = tuple(self.configurations)
        if any(
            len(configuration) != len(bases)
            or any(
                value < 0 or value >= modal_shape[mode] for mode, value in enumerate(configuration)
            )
            for configuration in configurations
        ):
            raise ValueError("A product-basis configuration is outside the modal basis")
        object.__setattr__(self, "modal_bases", bases)
        object.__setattr__(self, "configurations", configurations)
        object.__setattr__(self, "modal_shape", modal_shape)
        object.__setattr__(self, "grid_shape", grid_shape)

    def to_grid(self, coefficients: np.ndarray) -> np.ndarray:
        values = np.asarray(coefficients, dtype=float)
        if values.shape != (len(self.configurations),):
            raise ValueError("Configuration coefficient vector has the wrong shape")
        modal_tensor = np.zeros(self.modal_shape, dtype=float)
        for index, configuration in enumerate(self.configurations):
            modal_tensor[configuration] = values[index]
        return _modal_to_grid(self.modal_bases, modal_tensor).reshape(-1)

    def from_grid(self, vector: np.ndarray) -> np.ndarray:
        values = np.asarray(vector, dtype=float)
        if values.shape != (int(np.prod(self.grid_shape)),):
            raise ValueError("Grid vector has the wrong shape")
        modal_tensor = _grid_to_modal(self.modal_bases, values.reshape(self.grid_shape))
        return np.array([modal_tensor[configuration] for configuration in self.configurations])


def _iterate_ground_vscf(
    initial_modals: Sequence[np.ndarray],
    mean_fields,
    product_energy,
    policy: VSCFSettings,
) -> tuple[list[np.ndarray], tuple[np.ndarray, ...], tuple[VSCFIteration, ...], bool]:
    modals = [np.asarray(modal, dtype=float) for modal in initial_modals]
    previous_energy = product_energy(modals)
    history = []
    converged = False
    final_fields = mean_fields(modals)
    for iteration in range(1, int(policy.max_iterations) + 1):
        old_modals = [modal.copy() for modal in modals]
        fields = mean_fields(old_modals)
        new_modals = []
        for mode, matrix in enumerate(fields):
            _, eigenvectors = np.linalg.eigh(matrix)
            candidate = _phase_aligned(eigenvectors[:, 0], old_modals[mode])
            mixed = _normalized(
                (1.0 - float(policy.modal_mixing)) * old_modals[mode]
                + float(policy.modal_mixing) * candidate
            )
            new_modals.append(mixed)
        energy = product_energy(new_modals)
        density_change = max(
            float(np.max(np.abs(np.square(new) - np.square(old))))
            for new, old in zip(new_modals, old_modals)
        )
        delta_energy = abs(float(energy - previous_energy))
        history.append(VSCFIteration(iteration, float(energy), delta_energy, density_change))
        modals = new_modals
        previous_energy = energy
        final_fields = mean_fields(modals)
        if delta_energy <= float(policy.energy_tolerance_Eh) and density_change <= float(
            policy.density_tolerance
        ):
            converged = True
            break
    if not converged and policy.raise_on_nonconvergence:
        last = history[-1]
        raise RuntimeError(
            "Ground-state VSCF modal basis did not converge after "
            f"{policy.max_iterations} iterations: dE={last.delta_energy_Eh:.3e} Eh, "
            f"density={last.max_density_change:.3e}"
        )
    return modals, final_fields, tuple(history), converged


def _finalize_modal_basis(
    *,
    coordinate_ids: tuple[str, ...],
    counts: tuple[int, ...],
    mean_fields: tuple[np.ndarray, ...],
    product_energy,
    converged: bool,
    history: tuple[VSCFIteration, ...],
    kinetic_fingerprint: str,
    source_hamiltonian_fingerprint: str,
    vscf_settings: VSCFSettings,
    metadata: Mapping[str, object],
) -> GroundModalBasis:
    modals = []
    energies = []
    densities = []
    for count, matrix in zip(counts, mean_fields):
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        basis = _phase_canonical_columns(eigenvectors[:, :count])
        modals.append(basis)
        energies.append(eigenvalues[:count])
        densities.append(np.square(basis[:, 0]))
    energy = product_energy([basis[:, 0] for basis in modals])
    return GroundModalBasis(
        coordinate_ids=coordinate_ids,
        modals=tuple(modals),
        modal_energies_Eh=tuple(energies),
        mean_field_hamiltonians_Eh=mean_fields,
        densities=tuple(densities),
        vscf_energy_Eh=energy,
        converged=converged,
        iterations=len(history),
        history=history,
        kinetic_operator_fingerprint=kinetic_fingerprint,
        source_hamiltonian_fingerprint=source_hamiltonian_fingerprint,
        vscf_settings=vscf_settings,
        metadata=metadata,
    )


def _triatomic_mean_fields(
    hamiltonian: TriatomicJ0Hamiltonian,
    modals: Sequence[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kinetic = hamiltonian.kinetic
    probabilities = tuple(np.square(modal) for modal in modals)
    radial_r_probability, radial_R_probability, angular_probability = probabilities
    potential = hamiltonian.potential_Eh
    potential_r = np.einsum(
        "ijk,j,k->i", potential, radial_R_probability, angular_probability, optimize=True
    )
    potential_R = np.einsum(
        "ijk,i,k->j", potential, radial_r_probability, angular_probability, optimize=True
    )
    potential_x = np.einsum(
        "ijk,i,j->k", potential, radial_r_probability, radial_R_probability, optimize=True
    )
    angular_j2_expectation = float(modals[2] @ kinetic.angular_j2 @ modals[2])
    inverse_r_expectation = float(radial_r_probability @ kinetic.inverse_r2_Eh)
    inverse_R_expectation = float(radial_R_probability @ kinetic.inverse_R2_Eh)
    radial_r = kinetic.radial_r_kinetic_Eh + np.diag(
        potential_r + angular_j2_expectation * kinetic.inverse_r2_Eh
    )
    radial_R = kinetic.radial_R_kinetic_Eh + np.diag(
        potential_R + angular_j2_expectation * kinetic.inverse_R2_Eh
    )
    angular = (
        np.diag(potential_x) + (inverse_r_expectation + inverse_R_expectation) * kinetic.angular_j2
    )
    return radial_r, radial_R, angular


def _assign_vci_states(
    configurations: tuple[Configuration, ...],
    energies: np.ndarray,
    coefficients: np.ndarray,
    *,
    requested_count: int,
    degeneracy_tolerance: float,
    leading_count: int,
) -> tuple[tuple[VCIStateAssignment, ...], tuple[VCIDegenerateBlock, ...]]:
    weights = np.square(coefficients)
    blocks = _degenerate_blocks(energies, degeneracy_tolerance)
    state_to_block = {state: block for block in blocks for state in block}
    degenerate_blocks = []
    block_diagnostics = {}
    for block in blocks:
        if len(block) == 1 or not any(state < requested_count for state in block):
            continue
        block_coefficients = coefficients[:, list(block)]
        projector = block_coefficients @ block_coefficients.T
        diagonal = np.diag(projector)
        reference_indices = tuple(
            sorted(
                range(len(configurations)),
                key=lambda index: (-diagonal[index], configurations[index]),
            )[: len(block)]
        )
        reference = tuple(configurations[index] for index in reference_indices)
        principal_cosines = tuple(
            float(value)
            for value in np.linalg.svd(
                block_coefficients[np.asarray(reference_indices), :],
                compute_uv=False,
            )
        )
        diagnostic = VCIDegenerateBlock(
            state_indices=block,
            reference_configurations=reference,
            invariant_projector=projector,
            principal_cosines=principal_cosines,
            crosses_requested_cutoff=(min(block) < requested_count <= max(block)),
        )
        degenerate_blocks.append(diagnostic)
        block_diagnostics[block] = diagnostic

    nondegenerate_states = tuple(
        state for state in range(requested_count) if len(state_to_block[state]) == 1
    )
    matched = {}
    if nondegenerate_states:
        state_rows, configuration_columns = linear_sum_assignment(
            -weights[:, nondegenerate_states].T
        )
        matched = {
            nondegenerate_states[int(row)]: int(column)
            for row, column in zip(state_rows, configuration_columns)
        }
    assignments = []
    for state in range(requested_count):
        block = state_to_block[state]
        order = sorted(
            range(len(configurations)),
            key=lambda index: (-weights[index, state], configurations[index]),
        )[:leading_count]
        leading = tuple((configurations[index], float(weights[index, state])) for index in order)
        participation = float(1.0 / np.sum(np.square(weights[:, state])))
        if len(block) == 1:
            index = matched[state]
            assignment = configurations[index]
            overlap = float(weights[index, state])
            reference_subspace = ()
            principal_cosines = ()
            subspace_overlap = None
            manual = False
        else:
            diagnostic = block_diagnostics[block]
            assignment = None
            overlap = None
            reference_subspace = diagnostic.reference_configurations
            principal_cosines = diagnostic.principal_cosines
            subspace_overlap = min(principal_cosines) ** 2
            manual = True
        assignments.append(
            VCIStateAssignment(
                state_index=state,
                configuration=assignment,
                overlap=overlap,
                leading_configurations=leading,
                participation_ratio=participation,
                degenerate_block=block,
                reference_subspace=reference_subspace,
                principal_cosines=principal_cosines,
                subspace_overlap=subspace_overlap,
                manual_review=manual,
            )
        )
    return tuple(assignments), tuple(degenerate_blocks)


def _degenerate_blocks(energies: np.ndarray, tolerance: float) -> tuple[tuple[int, ...], ...]:
    blocks = []
    start = 0
    for index in range(1, energies.size):
        if float(energies[index] - energies[index - 1]) > tolerance:
            blocks.append(tuple(range(start, index)))
            start = index
    blocks.append(tuple(range(start, energies.size)))
    return tuple(blocks)


def _configuration(values: Sequence[int]) -> Configuration:
    configuration = tuple(operator.index(value) for value in values)
    if not configuration or any(value < 0 for value in configuration):
        raise ValueError("Configurations must contain non-negative quanta")
    return configuration


def _configuration_or_none(values: Sequence[int] | None) -> Configuration | None:
    return None if values is None else _configuration(values)


def _probability(name: str, value: float) -> float:
    probability = float(value)
    if not np.isfinite(probability) or probability < -1e-12 or probability > 1.0 + 1e-12:
        raise ValueError(f"{name} must lie between zero and one")
    return min(1.0, max(0.0, probability))


def _probability_or_none(name: str, value: float | None) -> float | None:
    return None if value is None else _probability(name, value)


def _modal_counts(values: int | Sequence[int], grid_shape: tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(values, int):
        counts = (operator.index(values),) * len(grid_shape)
    else:
        if len(values) != len(grid_shape):
            raise ValueError("n_modals_per_mode must align with the grid dimensions")
        counts = tuple(operator.index(value) for value in values)
    if any(count < 1 or count > size for count, size in zip(counts, grid_shape)):
        raise ValueError("Each retained modal count must lie within its grid dimension")
    return counts


def _quanta_limits(
    values: int | tuple[int, ...] | None,
    modal_counts: tuple[int, ...],
) -> tuple[int, ...]:
    if values is None:
        return tuple(count - 1 for count in modal_counts)
    if isinstance(values, int):
        limits = (operator.index(values),) * len(modal_counts)
    else:
        if len(values) != len(modal_counts):
            raise ValueError("max_quanta_per_mode must align with the modal basis")
        limits = tuple(operator.index(value) for value in values)
    if any(value < 0 for value in limits):
        raise ValueError("max_quanta_per_mode values must be non-negative")
    return tuple(min(value, count - 1) for value, count in zip(limits, modal_counts))


def _uncoupled_excitation(
    modal_basis: GroundModalBasis,
    configuration: Configuration,
) -> float:
    return float(
        sum(
            energies[quantum] - energies[0]
            for energies, quantum in zip(
                modal_basis.modal_energies_Eh,
                configuration,
            )
        )
    )


def _product_vector(modals: Sequence[np.ndarray]) -> np.ndarray:
    product = np.asarray(modals[0], dtype=float)
    for modal in modals[1:]:
        product = np.multiply.outer(product, modal)
    return product.reshape(-1)


def _modal_to_grid(modal_bases: tuple[np.ndarray, ...], modal_tensor: np.ndarray) -> np.ndarray:
    n_modes = len(modal_bases)
    grid_labels = list(range(n_modes))
    modal_labels = list(range(n_modes, 2 * n_modes))
    arguments = []
    for mode, basis in enumerate(modal_bases):
        arguments.extend((basis, [grid_labels[mode], modal_labels[mode]]))
    arguments.extend((modal_tensor, modal_labels, grid_labels))
    return np.einsum(*arguments, optimize=True)


def _grid_to_modal(modal_bases: tuple[np.ndarray, ...], grid_tensor: np.ndarray) -> np.ndarray:
    n_modes = len(modal_bases)
    grid_labels = list(range(n_modes))
    modal_labels = list(range(n_modes, 2 * n_modes))
    arguments = []
    for mode, basis in enumerate(modal_bases):
        arguments.extend((basis, [grid_labels[mode], modal_labels[mode]]))
    arguments.extend((grid_tensor, grid_labels, modal_labels))
    return np.einsum(*arguments, optimize=True)


def _apply_axis(matrix: np.ndarray, tensor: np.ndarray, axis: int) -> np.ndarray:
    moved = np.moveaxis(tensor, axis, 0)
    acted = np.tensordot(matrix, moved, axes=(1, 0))
    return np.moveaxis(acted, 0, axis)


def _phase_aligned(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return candidate if float(candidate @ reference) >= 0.0 else -candidate


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Cannot normalize a zero or non-finite modal")
    return vector / norm


def _phase_canonical_columns(vectors: np.ndarray) -> np.ndarray:
    canonical = np.array(vectors, dtype=float, copy=True)
    for column in range(canonical.shape[1]):
        absolute = np.abs(canonical[:, column])
        pivot = int(np.flatnonzero(absolute == np.max(absolute))[0])
        if canonical[pivot, column] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


def _phase_canonical_configurations(
    vectors: np.ndarray,
    configurations: tuple[Configuration, ...],
) -> np.ndarray:
    """Canonicalize VCI signs with lexicographic configuration tie-breaking."""

    canonical = np.array(vectors, dtype=float, copy=True)
    if canonical.shape[0] != len(configurations):
        raise ValueError("VCI vectors and configurations do not align")
    for column in range(canonical.shape[1]):
        absolute = np.abs(canonical[:, column])
        maximum = float(np.max(absolute))
        tied = np.flatnonzero(absolute == maximum)
        pivot = min(
            (int(index) for index in tied),
            key=lambda index: configurations[index],
        )
        if canonical[pivot, column] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


def _assignment_identity(
    assignments: Sequence[VCIStateAssignment],
    blocks: Sequence[VCIDegenerateBlock],
) -> str:
    return payload_fingerprint(
        {
            "assignments": [
                {
                    "state_index": value.state_index,
                    "configuration": (
                        None if value.configuration is None else list(value.configuration)
                    ),
                    "overlap": value.overlap,
                    "leading_configurations": [
                        [list(configuration), weight]
                        for configuration, weight in value.leading_configurations
                    ],
                    "participation_ratio": value.participation_ratio,
                    "degenerate_block": list(value.degenerate_block),
                    "reference_subspace": [
                        list(configuration) for configuration in value.reference_subspace
                    ],
                    "principal_cosines": list(value.principal_cosines),
                    "subspace_overlap": value.subspace_overlap,
                    "manual_review": value.manual_review,
                }
                for value in assignments
            ],
            "degenerate_blocks": [
                {
                    "state_indices": list(value.state_indices),
                    "reference_configurations": [
                        list(configuration) for configuration in value.reference_configurations
                    ],
                    "invariant_projector": array_identity(value.invariant_projector),
                    "principal_cosines": list(value.principal_cosines),
                    "crosses_requested_cutoff": value.crosses_requested_cutoff,
                }
                for value in blocks
            ],
        }
    )


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
    "Configuration",
    "GroundModalBasis",
    "NModeGridHamiltonian",
    "ProductGridHamiltonian",
    "VCIDegenerateBlock",
    "VCIResult",
    "VCISettings",
    "VCIStateAssignment",
    "build_nmode_vscf_modal_basis",
    "build_triatomic_vscf_modal_basis",
    "dump_ground_modal_basis",
    "dump_vci_result",
    "enumerate_vci_configurations",
    "load_ground_modal_basis",
    "load_vci_result",
    "solve_vci",
    "validate_ground_modal_basis_against_hamiltonian",
    "validate_vci_result_against_hamiltonian",
    "validate_vci_result_against_modal_basis",
    "vci_state_probability_marginals",
]
