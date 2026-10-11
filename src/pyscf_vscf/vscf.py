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

from ._validation import positive_finite as _positive_finite
from ._arrays import immutable_array
from ._artifacts import atomic_savez_compressed
from ._identity import immutable_json_mapping, to_jsonable
from ._phase import phase_canonical_columns
from .constants import ATOMIC_DIPOLE_TO_DEBYE, HARTREE_TO_CM
from .dvr import sinc_kinetic_1d
from .spectra import (
    einstein_a_from_debye,
    integrated_cross_section_omega,
    integrated_cross_section_omega_to_km_per_mol,
)

DEFAULT_OVERLAP_TOLERANCE = 0.05


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
        """Number of vibrational coordinates."""

        return len(self.coordinates)

    def one_mode_hamiltonians(self) -> tuple[np.ndarray, ...]:
        """Return the sinc-DVR one-mode Hamiltonians ``T_i + V_i`` in Hartree."""

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
    """Energy and density change after one VSCF iteration."""

    iteration: int
    energy_Eh: float
    delta_energy_Eh: float
    max_density_change: float


@dataclass(frozen=True)
class VSCFStateResult:
    """A converged (or last-iterate) state-specific VSCF product state.

    ``modals`` are the normalized one-mode functions on each DVR grid,
    ``modal_eigenvalues_Eh`` their effective one-mode energies, and ``history``
    the per-iteration record.
    """

    quanta: tuple[int, ...]
    energy_Eh: float
    modal_eigenvalues_Eh: tuple[float, ...]
    modals: tuple[np.ndarray, ...]
    converged: bool
    iterations: int
    history: tuple[VSCFIteration, ...]


@dataclass(frozen=True)
class NModeDipole:
    """Vector dipole on the VSCF grids: reference plus one- and two-mode increments.

    ``reference_au`` is the dipole (atomic units, a fixed molecular frame) at
    the reference point; ``one_mode_au[i]`` has shape ``(n_i, 3)`` and
    ``two_mode_au[(i, j)]`` shape ``(n_i, n_j, 3)``, both as increments that
    vanish at the reference values (the same convention as the potential).
    """

    coordinates: tuple[np.ndarray, ...]
    reference_au: np.ndarray
    one_mode_au: tuple[np.ndarray, ...]
    two_mode_au: Mapping[Pair, np.ndarray] = field(default_factory=dict)
    source_fingerprint: str | None = None

    def __post_init__(self) -> None:
        grids = tuple(
            _validated_grid(f"coordinates[{i}]", q) for i, q in enumerate(self.coordinates)
        )
        reference = _validated_array("reference_au", self.reference_au, (3,))
        if len(self.one_mode_au) != len(grids):
            raise ValueError("one_mode_au must contain one array per coordinate")
        one = tuple(
            _validated_array(f"one_mode_au[{i}]", values, (grids[i].size, 3))
            for i, values in enumerate(self.one_mode_au)
        )
        two = {}
        for raw_pair, values in self.two_mode_au.items():
            i, j = (operator.index(raw_pair[0]), operator.index(raw_pair[1]))
            if not 0 <= i < j < len(grids):
                raise ValueError(f"Invalid two-mode key {(i, j)!r}")
            two[(i, j)] = _readonly_copy(
                _validated_array(
                    f"two_mode_au[{i}, {j}]", values, (grids[i].size, grids[j].size, 3)
                )
            )
        object.__setattr__(self, "coordinates", tuple(_readonly_copy(q) for q in grids))
        object.__setattr__(self, "reference_au", _readonly_copy(reference))
        object.__setattr__(self, "one_mode_au", tuple(_readonly_copy(v) for v in one))
        object.__setattr__(self, "two_mode_au", MappingProxyType(two))

    @classmethod
    def from_surface(cls, model: Any, coordinates: Sequence[np.ndarray]) -> NModeDipole:
        """Evaluate a fitted n-mode dipole surface (body frame) on VSCF grids.

        Requires every one-mode increment and no increment above rank two,
        matching :func:`~pyscf_vscf.nmode.nmode_potential_from_surface`.
        """

        from .nmode import nmode_dms_fingerprint

        increments = model.dipole_increments
        if any(len(subset) >= 3 for subset in increments):
            raise ValueError("VSCF intensities cannot discard rank-3 dipole increments")
        grids = tuple(np.asarray(grid, dtype=float) for grid in coordinates)
        one = []
        for mode in range(len(grids)):
            if (mode,) not in increments:
                raise ValueError("VSCF intensities require every one-mode dipole increment")
            one.append(np.asarray(increments[(mode,)].evaluate(grids[mode][:, None]), dtype=float))
        two = {}
        for subset, surface in increments.items():
            if len(subset) != 2:
                continue
            mesh = np.meshgrid(*(grids[mode] for mode in subset), indexing="ij")
            two[subset] = np.asarray(surface.evaluate(np.stack(mesh, axis=-1)), dtype=float)
        return cls(
            coordinates=grids,
            reference_au=np.asarray(model.reference_dipole_body_au, dtype=float),
            one_mode_au=tuple(one),
            two_mode_au=two,
            source_fingerprint=nmode_dms_fingerprint(model),
        )


@dataclass(frozen=True)
class VSCFTransitionMoment:
    """Transition dipole between the VSCF ground state and one excited state.

    State-specific VSCF optimizes each state's modals separately, so the two
    product states need not be orthogonal. ``state_overlap`` is their overlap
    ``<0|n>``; ``approximate`` is true when its magnitude exceeds the
    tolerance, in which case the moment contains a reference-dipole term that
    an orthogonal treatment would not have. The isotropic intensity uses the
    vector norm with the 1/3 orientation average.
    """

    dipole_au: np.ndarray
    state_overlap: float
    approximate: bool
    integrated_cross_section_isotropic_omega_m2_per_s: float
    einstein_a_s: float

    @property
    def dipole_debye(self) -> np.ndarray:
        """Transition dipole in Debye."""

        return np.asarray(self.dipole_au) * ATOMIC_DIPOLE_TO_DEBYE

    @property
    def norm_debye(self) -> float:
        """Norm of the transition dipole in Debye."""

        return float(np.linalg.norm(self.dipole_debye))

    @property
    def integrated_cross_section_isotropic_km_per_mol(self) -> float:
        """Isotropic integrated absorption in km mol^-1."""

        return integrated_cross_section_omega_to_km_per_mol(
            self.integrated_cross_section_isotropic_omega_m2_per_s
        )


@dataclass(frozen=True)
class VSCFTransition:
    """Transition from the VSCF ground state to one excited product state.

    ``transition_moment`` is present when a dipole expansion was supplied.
    """

    quanta: tuple[int, ...]
    energy_Eh: float
    frequency_cm: float
    transition_moment: VSCFTransitionMoment | None = None


@dataclass(frozen=True)
class VSCFSpectrum:
    """VSCF ground state, excited states sorted by energy, and their transitions."""

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
        modals=tuple(_canonical_modal(modal) for modal in modals),
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
    dipole: NModeDipole | None = None,
    overlap_tolerance: float = DEFAULT_OVERLAP_TOLERANCE,
) -> VSCFSpectrum:
    """Solve a ground state and selected fundamental/overtone/combination states.

    With ``dipole`` (an :class:`NModeDipole` on the same grids), each
    transition carries a :class:`VSCFTransitionMoment`.
    """

    if dipole is not None:
        _check_dipole_grids(model, dipole)

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
        frequency_cm = transition_energy * HARTREE_TO_CM
        moment = None
        if dipole is not None:
            moment = vscf_transition_moment(
                dipole, ground, state, frequency_cm, overlap_tolerance=overlap_tolerance
            )
        transitions_list.append(
            VSCFTransition(
                quanta=state.quanta,
                energy_Eh=transition_energy,
                frequency_cm=frequency_cm,
                transition_moment=moment,
            )
        )
    transitions = tuple(transitions_list)
    return VSCFSpectrum(ground=ground, excited_states=ordered, transitions=transitions)


def vscf_transition_moment(
    dipole: NModeDipole,
    ground: VSCFStateResult,
    excited: VSCFStateResult,
    frequency_cm: float,
    *,
    overlap_tolerance: float = DEFAULT_OVERLAP_TOLERANCE,
) -> VSCFTransitionMoment:
    """Return ``<ground|mu|excited>`` for product states of normalized DVR modals.

    With modal overlaps ``S_k = <g_k|e_k>`` the moment is
    ``mu_ref prod_k S_k + sum_i <g_i|mu_i|e_i> prod_{k!=i} S_k
    + sum_{i<j} <g_i g_j|mu_ij|e_i e_j> prod_{k!=i,j} S_k``.
    """

    ground_modals = ground.modals
    excited_modals = excited.modals
    n_modes = len(ground_modals)
    overlaps = np.array([float(g @ e) for g, e in zip(ground_modals, excited_modals)], dtype=float)

    def others(*skip: int) -> float:
        return float(np.prod([overlaps[k] for k in range(n_modes) if k not in skip]))

    vector = np.asarray(dipole.reference_au, dtype=float) * float(np.prod(overlaps))
    for mode, increment in enumerate(dipole.one_mode_au):
        density = ground_modals[mode] * excited_modals[mode]
        vector = vector + (density @ increment) * others(mode)
    for (i, j), increment in dipole.two_mode_au.items():
        first = ground_modals[i] * excited_modals[i]
        second = ground_modals[j] * excited_modals[j]
        vector = vector + np.einsum("a,b,abx->x", first, second, increment) * others(i, j)
    state_overlap = float(np.prod(overlaps))
    norm_debye = float(np.linalg.norm(vector) * ATOMIC_DIPOLE_TO_DEBYE)
    frequency = float(frequency_cm)
    return VSCFTransitionMoment(
        dipole_au=_readonly_copy(np.asarray(vector, dtype=float)),
        state_overlap=state_overlap,
        approximate=abs(state_overlap) > float(overlap_tolerance),
        integrated_cross_section_isotropic_omega_m2_per_s=integrated_cross_section_omega(
            norm_debye, frequency
        ),
        einstein_a_s=einstein_a_from_debye(norm_debye, frequency),
    )


def _check_dipole_grids(model: NModePotential, dipole: NModeDipole) -> None:
    if len(dipole.coordinates) != model.n_modes or any(
        q.shape != d.shape or not np.array_equal(q, d)
        for q, d in zip(model.coordinates, dipole.coordinates)
    ):
        raise ValueError("The dipole expansion must use the potential's grids")


def _canonical_modal(modal: np.ndarray) -> np.ndarray:
    return _readonly_copy(phase_canonical_columns(np.asarray(modal, dtype=float)[:, None])[:, 0])


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
    "NModeDipole",
    "NModePotential",
    "VSCFIteration",
    "VSCFSettings",
    "VSCFSpectrum",
    "VSCFStateResult",
    "VSCFTransition",
    "VSCFTransitionMoment",
    "dump_nmode_model",
    "load_nmode_model",
    "nmode_model_from_pair_surfaces",
    "nmode_model_fingerprint",
    "product_states",
    "solve_vscf_state",
    "vscf_spectrum",
    "vscf_transition_moment",
]
