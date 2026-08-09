"""Reduced-dimensional vibrational self-consistent-field solver.

The implemented Hamiltonian uses one-mode reference potentials plus optional
two-mode coupling corrections on direct-product uniform sinc-DVR grids::

    H = sum_i [T_i + V_i(q_i)] + sum_{i<j} V_ij(q_i, q_j)

``V_ij`` must be a coupling correction, not a complete two-mode potential, so
that one-mode terms are not counted twice. The solver is state-specific VSCF;
it is not vibrational configuration interaction (VCI).
"""

from __future__ import annotations

import hashlib
import json
import operator
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from ._arrays import immutable_array
from ._artifacts import atomic_savez_compressed
from ._identity import immutable_json_mapping, to_jsonable
from .constants import HARTREE_TO_CM
from .dvr import sinc_kinetic_1d


Pair = tuple[int, int]


@dataclass(frozen=True)
class NModePotential:
    """One-mode plus two-mode representation of a vibrational Hamiltonian."""

    coordinates: tuple[np.ndarray, ...]
    masses_amu: tuple[float, ...]
    one_mode_potentials_Eh: tuple[np.ndarray, ...]
    two_mode_couplings_Eh: Mapping[Pair, np.ndarray] = field(default_factory=dict)
    mode_labels: tuple[str, ...] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    coordinate_units: str = "angstrom"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    coordinate_map_fingerprint: str | None = None

    def __post_init__(self) -> None:
        coordinates = tuple(
            _validated_grid(f"coordinates[{i}]", q) for i, q in enumerate(self.coordinates)
        )
        n_modes = len(coordinates)
        if n_modes < 2:
            raise ValueError("VSCF requires at least two vibrational modes")
        if len(self.masses_amu) != n_modes:
            raise ValueError("masses_amu must contain one mass per mode")
        if len(self.one_mode_potentials_Eh) != n_modes:
            raise ValueError("one_mode_potentials_Eh must contain one array per mode")
        coordinate_units = str(self.coordinate_units).strip().lower()
        if coordinate_units not in {"angstrom", "ang", "a", "aa"}:
            raise ValueError(
                "coordinate_units must identify Angstrom coordinates; other units are not supported"
            )

        masses = tuple(
            _positive_finite(f"masses_amu[{i}]", value) for i, value in enumerate(self.masses_amu)
        )
        one_mode = tuple(
            _validated_array(f"one_mode_potentials_Eh[{i}]", values, coordinates[i].shape)
            for i, values in enumerate(self.one_mode_potentials_Eh)
        )

        couplings: dict[Pair, np.ndarray] = {}
        for raw_pair, values in self.two_mode_couplings_Eh.items():
            if len(raw_pair) != 2:
                raise ValueError(f"Invalid two-mode key {raw_pair!r}; expected (i, j)")
            i, j = (operator.index(raw_pair[0]), operator.index(raw_pair[1]))
            if i < 0 or j < 0 or i >= n_modes or j >= n_modes or i >= j:
                raise ValueError(
                    f"Invalid two-mode key {(i, j)!r}; require 0 <= i < j < {n_modes}"
                )
            couplings[(i, j)] = _validated_array(
                f"two_mode_couplings_Eh[{i}, {j}]",
                values,
                (coordinates[i].size, coordinates[j].size),
            )

        labels = self.mode_labels
        if labels is None:
            labels = tuple(f"q{i}" for i in range(n_modes))
        else:
            labels = tuple(str(label) for label in labels)
            if len(labels) != n_modes or any(not label for label in labels):
                raise ValueError("mode_labels must contain one non-empty label per mode")
            if len(set(labels)) != len(labels):
                raise ValueError("mode_labels must be unique")

        object.__setattr__(self, "coordinates", tuple(_readonly_copy(q) for q in coordinates))
        object.__setattr__(self, "masses_amu", masses)
        object.__setattr__(
            self,
            "one_mode_potentials_Eh",
            tuple(_readonly_copy(v) for v in one_mode),
        )
        object.__setattr__(
            self,
            "two_mode_couplings_Eh",
            MappingProxyType({pair: _readonly_copy(values) for pair, values in couplings.items()}),
        )
        object.__setattr__(self, "mode_labels", labels)
        object.__setattr__(self, "metadata", immutable_json_mapping(self.metadata))
        object.__setattr__(self, "provenance", immutable_json_mapping(self.provenance))
        object.__setattr__(self, "coordinate_units", "angstrom")
        map_id = self.coordinate_map_fingerprint
        if map_id is not None:
            map_id = str(map_id).strip()
            if not map_id:
                raise ValueError("coordinate_map_fingerprint must be non-empty when present")
        object.__setattr__(self, "coordinate_map_fingerprint", map_id)

    @property
    def n_modes(self) -> int:
        return len(self.coordinates)

    def one_mode_hamiltonians(self) -> tuple[np.ndarray, ...]:
        return tuple(
            sinc_kinetic_1d(q, mass) + np.diag(potential)
            for q, mass, potential in zip(
                self.coordinates,
                self.masses_amu,
                self.one_mode_potentials_Eh,
            )
        )


def nmode_model_from_pair_surfaces(
    coordinates: Sequence[np.ndarray],
    masses_amu: Sequence[float],
    pair_surfaces_Eh: Mapping[Pair, np.ndarray],
    *,
    reference_indices: Sequence[int] | None = None,
    mode_labels: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    consistency_tolerance_Eh: float = 1e-8,
) -> NModePotential:
    """Assemble a one-plus-two-mode model from overlapping pair surfaces.

    Each input surface may have an independent additive energy offset. The
    one-mode cuts shared by different pair surfaces must agree within
    ``consistency_tolerance_Eh``; their mean is used in the assembled model.
    """

    grids = tuple(_validated_grid(f"coordinates[{i}]", q) for i, q in enumerate(coordinates))
    n_modes = len(grids)
    if n_modes < 2:
        raise ValueError("At least two coordinates are required")
    if len(masses_amu) != n_modes:
        raise ValueError("masses_amu must contain one mass per mode")

    tolerance = float(consistency_tolerance_Eh)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("consistency_tolerance_Eh must be finite and non-negative")

    if reference_indices is None:
        references = tuple(q.size // 2 for q in grids)
    else:
        if len(reference_indices) != n_modes:
            raise ValueError("reference_indices must contain one index per mode")
        references = tuple(operator.index(index) for index in reference_indices)
        for mode, (index, grid) in enumerate(zip(references, grids)):
            if index < 0 or index >= grid.size:
                raise ValueError(f"reference_indices[{mode}]={index} is outside the grid")

    surfaces: dict[Pair, np.ndarray] = {}
    cuts: list[list[np.ndarray]] = [[] for _ in range(n_modes)]
    for raw_pair, values in pair_surfaces_Eh.items():
        if len(raw_pair) != 2:
            raise ValueError(f"Invalid two-mode key {raw_pair!r}; expected (i, j)")
        i, j = operator.index(raw_pair[0]), operator.index(raw_pair[1])
        if i < 0 or j < 0 or i >= n_modes or j >= n_modes or i >= j:
            raise ValueError(f"Invalid two-mode key {(i, j)!r}; require 0 <= i < j < {n_modes}")
        pair = (i, j)
        if pair in surfaces:
            raise ValueError(f"Duplicate two-mode surface for pair {pair}")
        surface = _validated_array(
            f"pair_surfaces_Eh[{i}, {j}]",
            values,
            (grids[i].size, grids[j].size),
        )
        surfaces[pair] = surface
        ri, rj = references[i], references[j]
        reference_energy = surface[ri, rj]
        cuts[i].append(surface[:, rj] - reference_energy)
        cuts[j].append(surface[ri, :] - reference_energy)

    if not surfaces:
        raise ValueError("pair_surfaces_Eh must contain at least one surface")

    one_mode: list[np.ndarray] = []
    maximum_disagreements: list[float] = []
    for mode, candidates in enumerate(cuts):
        if not candidates:
            raise ValueError(f"Mode {mode} does not appear in any pair surface")
        stacked = np.stack(candidates)
        mean_cut = np.mean(stacked, axis=0)
        disagreement = float(np.max(np.abs(stacked - mean_cut)))
        if disagreement > tolerance:
            raise ValueError(
                f"Shared one-mode cuts for mode {mode} disagree by {disagreement:.3e} Eh, "
                f"exceeding consistency_tolerance_Eh={tolerance:.3e}"
            )
        one_mode.append(mean_cut)
        maximum_disagreements.append(disagreement)

    couplings: dict[Pair, np.ndarray] = {}
    for (i, j), surface in surfaces.items():
        ri, rj = references[i], references[j]
        couplings[(i, j)] = surface - surface[ri, rj] - one_mode[i][:, None] - one_mode[j][None, :]

    assembled_metadata = dict(metadata or {})
    assembled_metadata["pair_surface_assembly"] = {
        "reference_indices": list(references),
        "consistency_tolerance_Eh": tolerance,
        "maximum_cut_disagreement_Eh": maximum_disagreements,
    }
    return NModePotential(
        coordinates=grids,
        masses_amu=tuple(masses_amu),
        one_mode_potentials_Eh=tuple(one_mode),
        two_mode_couplings_Eh=couplings,
        mode_labels=None if mode_labels is None else tuple(mode_labels),
        metadata=assembled_metadata,
    )


@dataclass(frozen=True)
class VSCFSettings:
    """Numerical policy for state-specific VSCF iteration."""

    max_iterations: int = 100
    energy_tolerance_Eh: float = 1e-10
    density_tolerance: float = 1e-8
    modal_mixing: float = 1.0
    root_following: bool = True
    raise_on_nonconvergence: bool = True

    def __post_init__(self) -> None:
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be at least 1")
        _positive_finite("energy_tolerance_Eh", self.energy_tolerance_Eh)
        _positive_finite("density_tolerance", self.density_tolerance)
        mixing = float(self.modal_mixing)
        if not np.isfinite(mixing) or mixing <= 0.0 or mixing > 1.0:
            raise ValueError("modal_mixing must satisfy 0 < modal_mixing <= 1")


@dataclass(frozen=True)
class VSCFIteration:
    iteration: int
    energy_Eh: float
    delta_energy_Eh: float
    max_density_change: float


@dataclass(frozen=True)
class VSCFStateResult:
    quanta: tuple[int, ...]
    energy_Eh: float
    modal_eigenvalues_Eh: tuple[float, ...]
    modals: tuple[np.ndarray, ...]
    converged: bool
    iterations: int
    history: tuple[VSCFIteration, ...]


@dataclass(frozen=True)
class VSCFTransition:
    quanta: tuple[int, ...]
    energy_Eh: float
    frequency_cm: float


@dataclass(frozen=True)
class VSCFSpectrum:
    ground: VSCFStateResult
    excited_states: tuple[VSCFStateResult, ...]
    transitions: tuple[VSCFTransition, ...]


def solve_vscf_state(
    model: NModePotential,
    quanta: Sequence[int] | None = None,
    *,
    settings: VSCFSettings | None = None,
) -> VSCFStateResult:
    """Solve one state-specific VSCF product state."""

    policy = settings or VSCFSettings()
    target = _validate_quanta(model, quanta)
    one_mode_hamiltonians = model.one_mode_hamiltonians()

    modals: list[np.ndarray] = []
    modal_eigenvalues: list[float] = []
    for mode, (hamiltonian, root) in enumerate(zip(one_mode_hamiltonians, target)):
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        if root >= eigenvalues.size:
            raise ValueError(
                f"quanta[{mode}]={root} exceeds the {eigenvalues.size} available DVR roots"
            )
        modals.append(eigenvectors[:, root].copy())
        modal_eigenvalues.append(float(eigenvalues[root]))

    previous_energy = _product_energy(model, one_mode_hamiltonians, modals)
    history: list[VSCFIteration] = []
    converged = False

    for iteration in range(1, int(policy.max_iterations) + 1):
        old_modals = tuple(modal.copy() for modal in modals)
        new_modals: list[np.ndarray] = []
        new_eigenvalues: list[float] = []

        for mode, root in enumerate(target):
            effective = one_mode_hamiltonians[mode] + np.diag(
                _mean_field_potential(model, mode, old_modals)
            )
            eigenvalues, eigenvectors = np.linalg.eigh(effective)
            chosen = root
            if policy.root_following and iteration > 1:
                overlaps = np.abs(eigenvectors.T @ old_modals[mode])
                chosen = int(np.argmax(overlaps))
            candidate = _phase_aligned(eigenvectors[:, chosen], old_modals[mode])
            mixed = _normalized(
                (1.0 - float(policy.modal_mixing)) * old_modals[mode]
                + float(policy.modal_mixing) * candidate
            )
            new_modals.append(mixed)
            new_eigenvalues.append(float(mixed @ effective @ mixed))

        energy = _product_energy(model, one_mode_hamiltonians, new_modals)
        density_change = max(
            float(np.max(np.abs(np.square(new) - np.square(old))))
            for new, old in zip(new_modals, old_modals)
        )
        delta_energy = abs(float(energy - previous_energy))
        history.append(
            VSCFIteration(
                iteration=iteration,
                energy_Eh=float(energy),
                delta_energy_Eh=delta_energy,
                max_density_change=density_change,
            )
        )
        modals = new_modals
        modal_eigenvalues = new_eigenvalues
        previous_energy = energy
        if delta_energy <= float(policy.energy_tolerance_Eh) and density_change <= float(
            policy.density_tolerance
        ):
            converged = True
            break

    if not converged and policy.raise_on_nonconvergence:
        last = history[-1]
        raise RuntimeError(
            "VSCF did not converge after "
            f"{policy.max_iterations} iterations: dE={last.delta_energy_Eh:.3e} Eh, "
            f"density={last.max_density_change:.3e}"
        )

    return VSCFStateResult(
        quanta=target,
        energy_Eh=float(previous_energy),
        modal_eigenvalues_Eh=tuple(modal_eigenvalues),
        modals=tuple(modal.copy() for modal in modals),
        converged=converged,
        iterations=len(history),
        history=tuple(history),
    )


def vscf_spectrum(
    model: NModePotential,
    *,
    states: Sequence[Sequence[int]] | None = None,
    max_quanta_per_mode: int = 1,
    max_total_quanta: int = 1,
    settings: VSCFSettings | None = None,
) -> VSCFSpectrum:
    """Solve a ground state and selected fundamental/overtone/combination states."""

    ground_quanta = (0,) * model.n_modes
    if states is None:
        requested = product_states(
            model.n_modes,
            max_quanta_per_mode=max_quanta_per_mode,
            max_total_quanta=max_total_quanta,
        )
    else:
        requested = tuple(_validate_quanta(model, state) for state in states)
        requested = tuple(state for state in requested if state != ground_quanta)
    if len(set(requested)) != len(requested):
        raise ValueError("states contains duplicate product-state quantum labels")

    ground = solve_vscf_state(model, ground_quanta, settings=settings)
    excited = tuple(solve_vscf_state(model, state, settings=settings) for state in requested)
    ordered = tuple(sorted(excited, key=lambda state: state.energy_Eh))
    transitions_list = []
    for state in ordered:
        transition_energy = float(state.energy_Eh - ground.energy_Eh)
        if transition_energy <= 0.0:
            raise RuntimeError(
                f"VSCF state {state.quanta} is not above the ground state; "
                "the state-specific root assignment collapsed"
            )
        transitions_list.append(
            VSCFTransition(
                quanta=state.quanta,
                energy_Eh=transition_energy,
                frequency_cm=transition_energy * HARTREE_TO_CM,
            )
        )
    transitions = tuple(transitions_list)
    return VSCFSpectrum(ground=ground, excited_states=ordered, transitions=transitions)


def product_states(
    n_modes: int,
    *,
    max_quanta_per_mode: int,
    max_total_quanta: int,
) -> tuple[tuple[int, ...], ...]:
    """Enumerate non-ground product-state quantum labels in a bounded polyad."""

    n = operator.index(n_modes)
    per_mode = operator.index(max_quanta_per_mode)
    total = operator.index(max_total_quanta)
    if n < 1 or per_mode < 1 or total < 1:
        raise ValueError("n_modes, max_quanta_per_mode, and max_total_quanta must be positive")

    states: list[tuple[int, ...]] = []
    shape = (per_mode + 1,) * n
    for state in np.ndindex(shape):
        quanta = tuple(int(value) for value in state)
        if 0 < sum(quanta) <= total:
            states.append(quanta)
    return tuple(sorted(states, key=lambda state: (sum(state), state)))


def nmode_model_fingerprint(model: NModePotential) -> str:
    """Return a stable SHA-256 fingerprint of a complete n-mode model."""

    digest = hashlib.sha256()
    header = {
        "schema": "pyscf-vscf-nmode",
        "schema_version": 2,
        "coordinate_units": model.coordinate_units,
        "masses_amu": model.masses_amu,
        "mode_labels": model.mode_labels,
        "metadata": to_jsonable(model.metadata),
        "pairs": [list(pair) for pair in sorted(model.two_mode_couplings_Eh)],
    }
    if model.coordinate_map_fingerprint is not None:
        header["coordinate_map_fingerprint"] = model.coordinate_map_fingerprint
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    for array in (*model.coordinates, *model.one_mode_potentials_Eh):
        _update_array_hash(digest, array)
    for pair in sorted(model.two_mode_couplings_Eh):
        _update_array_hash(digest, model.two_mode_couplings_Eh[pair])
    return digest.hexdigest()


def nmode_model_content_fingerprint(model: NModePotential) -> str:
    """Return an all-fields fingerprint for retained model content."""

    payload = {
        "schema": "pyscf-vscf-nmode-content",
        "schema_version": 1,
        "scientific_fingerprint_sha256": nmode_model_fingerprint(model),
        "provenance": to_jsonable(model.provenance),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def dump_nmode_model(path: Path | str, model: NModePotential) -> None:
    """Write a self-describing, fingerprinted n-mode model archive."""

    meta = {
        "schema": "pyscf-vscf-nmode",
        "schema_version": 3,
        "coordinate_units": model.coordinate_units,
        "masses_amu": list(model.masses_amu),
        "mode_labels": list(model.mode_labels or ()),
        "metadata": to_jsonable(model.metadata),
        "pairs": [list(pair) for pair in sorted(model.two_mode_couplings_Eh)],
        "fingerprint_sha256": nmode_model_fingerprint(model),
        "content_fingerprint_sha256": nmode_model_content_fingerprint(model),
    }
    if model.provenance:
        meta["provenance"] = to_jsonable(model.provenance)
    if model.coordinate_map_fingerprint is not None:
        meta["coordinate_map_fingerprint"] = model.coordinate_map_fingerprint
    arrays: dict[str, np.ndarray] = {}
    for mode, (coordinate, potential) in enumerate(
        zip(model.coordinates, model.one_mode_potentials_Eh)
    ):
        arrays[f"q_{mode}"] = coordinate
        arrays[f"v1_{mode}"] = potential
    for (i, j), coupling in model.two_mode_couplings_Eh.items():
        arrays[f"v2_{i}_{j}"] = coupling
    arrays["meta_json"] = np.array(json.dumps(meta, sort_keys=True, separators=(",", ":")))
    atomic_savez_compressed(path, arrays)


def load_nmode_model(path: Path | str) -> NModePotential:
    """Load and verify an n-mode model archive."""

    source = Path(path)
    with np.load(source, allow_pickle=False) as data:
        if "meta_json" not in data.files:
            raise ValueError(f"N-mode archive '{source}' is missing meta_json")
        meta = json.loads(str(data["meta_json"].tolist()))
        schema_version = meta.get("schema_version")
        if meta.get("schema") != "pyscf-vscf-nmode" or schema_version not in (2, 3):
            raise ValueError(f"Unsupported n-mode archive schema in '{source}'")
        labels = tuple(str(value) for value in meta["mode_labels"])
        n_modes = len(labels)
        coordinates = tuple(np.asarray(data[f"q_{i}"], dtype=float) for i in range(n_modes))
        one_mode = tuple(np.asarray(data[f"v1_{i}"], dtype=float) for i in range(n_modes))
        couplings = {
            (int(i), int(j)): np.asarray(data[f"v2_{int(i)}_{int(j)}"], dtype=float)
            for i, j in meta.get("pairs", [])
        }
    model = NModePotential(
        coordinates=coordinates,
        masses_amu=tuple(float(value) for value in meta["masses_amu"]),
        one_mode_potentials_Eh=one_mode,
        two_mode_couplings_Eh=couplings,
        mode_labels=labels,
        metadata=meta.get("metadata", {}),
        provenance=meta.get("provenance", {}),
        coordinate_map_fingerprint=meta.get("coordinate_map_fingerprint"),
        coordinate_units=str(meta["coordinate_units"]),
    )
    expected = str(meta.get("fingerprint_sha256", ""))
    actual = nmode_model_fingerprint(model)
    if not expected or actual != expected:
        raise ValueError(
            f"N-mode archive fingerprint mismatch: expected {expected!r}, calculated {actual!r}"
        )
    if schema_version == 3:
        expected_content = str(meta.get("content_fingerprint_sha256", ""))
        actual_content = nmode_model_content_fingerprint(model)
        if not expected_content or actual_content != expected_content:
            raise ValueError(
                "N-mode archive content fingerprint mismatch: "
                f"expected {expected_content!r}, calculated {actual_content!r}"
            )
    return model


def _validate_quanta(
    model: NModePotential,
    quanta: Sequence[int] | None,
) -> tuple[int, ...]:
    if quanta is None:
        return (0,) * model.n_modes
    if len(quanta) != model.n_modes:
        raise ValueError(f"quanta must contain {model.n_modes} entries")
    result = tuple(operator.index(value) for value in quanta)
    if any(value < 0 for value in result):
        raise ValueError("quanta must be non-negative")
    return result


def _mean_field_potential(
    model: NModePotential,
    mode: int,
    modals: Sequence[np.ndarray],
) -> np.ndarray:
    effective = np.zeros(model.coordinates[mode].size, dtype=float)
    probabilities = tuple(np.square(modal) for modal in modals)
    for (i, j), coupling in model.two_mode_couplings_Eh.items():
        if mode == i:
            effective += coupling @ probabilities[j]
        elif mode == j:
            effective += probabilities[i] @ coupling
    return effective


def _product_energy(
    model: NModePotential,
    one_mode_hamiltonians: Sequence[np.ndarray],
    modals: Sequence[np.ndarray],
) -> float:
    energy = sum(
        float(modal @ hamiltonian @ modal)
        for modal, hamiltonian in zip(modals, one_mode_hamiltonians)
    )
    probabilities = tuple(np.square(modal) for modal in modals)
    for (i, j), coupling in model.two_mode_couplings_Eh.items():
        energy += float(probabilities[i] @ coupling @ probabilities[j])
    return float(energy)


def _validated_grid(name: str, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < 3:
        raise ValueError(f"{name} must be a 1D array with at least 3 points")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain only finite values")
    steps = np.diff(grid)
    if not np.all(steps > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(steps, steps[0], rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced")
    return grid


def _validated_array(name: str, values: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _positive_finite(name: str, value: float) -> float:
    number = float(value)
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    return immutable_array(array, dtype=float)


def _phase_aligned(candidate: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return -candidate if float(candidate @ reference) < 0.0 else candidate


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-14:
        raise RuntimeError("VSCF modal mixing produced a zero-norm vector")
    return vector / norm


def _update_array_hash(digest: Any, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))


__all__ = [
    "NModePotential",
    "VSCFIteration",
    "VSCFSettings",
    "VSCFSpectrum",
    "VSCFStateResult",
    "VSCFTransition",
    "dump_nmode_model",
    "load_nmode_model",
    "nmode_model_from_pair_surfaces",
    "nmode_model_fingerprint",
    "product_states",
    "solve_vscf_state",
    "vscf_spectrum",
]
