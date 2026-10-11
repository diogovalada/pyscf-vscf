"""Scan workflow helpers (0.2 tuple API).

The tuple-returning grid functions and schema-3 cache helpers are deprecated
in favour of :mod:`pyscf_vscf.scans` (plans, runner, ``ScanResult``) and
:mod:`pyscf_vscf.artifacts.scans` (schema-4 files). Normal-mode selection,
constrained relaxed scans, and coordinate/mass helpers remain current.

This module sits above the pure coordinate/surface helpers and below the CLI.
All expensive electronic-structure work is injectable.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any
import traceback
import warnings

import numpy as np

from pyscf_vscf._warnings import deprecated
from pyscf_vscf.cache import (
    dump_grid_npz,
    electronic_structure_provenance,
    load_grid_npz,
    molecule_provenance,
    runtime_provenance,
    scientific_cache_metadata,
    validate_scientific_cache_metadata,
)
from pyscf_vscf.coordinates import Bond, parse_bond
from pyscf_vscf.coordinates import stretch_along_bond as _stretch_along_bond
from pyscf_vscf.coordinates import stretch_two_bonds as _stretch_two_bonds
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.electronic import EnergyDipoleEvaluator, energy_dipole


EnergyDipoleFn = EnergyDipoleEvaluator
ExecutorFactory = Callable[[], AbstractContextManager[Any]]
MoleculeFactory = Callable[[Any, np.ndarray], Any]
ProgressFn = Callable[[int, int, str], None]
LogFn = Callable[[str], None]
HarmonicFn = Callable[..., Any]
RelaxedNormalPointFn = Callable[
    [Any, Any, np.ndarray, float, float, int],
    Any,
]


@dataclass(frozen=True)
class NormalModeDirection:
    """Selected normal-mode scan direction and diagnostics."""

    u_dir: np.ndarray
    mode_index: int
    frequency_cm: float
    modes: np.ndarray
    freqs_cm: np.ndarray


@dataclass(frozen=True)
class NormalRelaxedGrid:
    """Exactly constrained scan arrays plus pointwise optimization diagnostics."""

    displacements_A: np.ndarray
    energies_hartree: np.ndarray
    dipoles_debye: np.ndarray
    achieved_displacements_A: np.ndarray
    constraint_residuals_A: np.ndarray
    converged: np.ndarray
    iterations: np.ndarray
    messages: tuple[str, ...]

    @property
    def failed_indices(self) -> tuple[int, ...]:
        """Indices whose constrained optimizations did not converge."""

        return tuple(int(index) for index in np.flatnonzero(~self.converged))

    def failure_summary(self) -> str:
        """Return an actionable aggregate description of failed points."""

        failed = self.failed_indices
        if not failed:
            return "normal-relaxed scan converged at every point"
        max_residual = float(np.max(np.abs(self.constraint_residuals_A)))
        return (
            f"normal-relaxed scan failed at {len(failed)} point(s); "
            f"indices={list(failed)}; max constraint residual={max_residual:.3e} A"
        )


DEFAULT_SCAN_BATCH_SIZE = 32
PARTIAL_SCAN_SCHEMA = "pyscf-vscf-partial-scan"
PARTIAL_SCAN_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ScanPointFailure:
    """One scan point whose electronic evaluation raised an exception."""

    index: tuple[int, ...]
    scan_values: tuple[float, ...]
    axis_labels: tuple[str, ...]
    error_type: str
    message: str
    traceback: str = ""

    def describe(self) -> str:
        """Return a one-line description naming the point and the error."""

        position = ", ".join(
            f"{label}={value:.6g}" for label, value in zip(self.axis_labels, self.scan_values)
        )
        index = ",".join(str(value) for value in self.index)
        return f"point [{index}] ({position}): {self.error_type}: {self.message}"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready failure record."""

        return {
            "index": list(self.index),
            "scan_values": list(self.scan_values),
            "axis_labels": list(self.axis_labels),
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
        }


@dataclass(frozen=True)
class PartialScanGrid:
    """Completed points of a scan that could not finish.

    Energies are absolute Hartree values; points that were not completed hold
    NaN and are ``False`` in ``completed``. A partial grid is diagnostic data:
    it is never a valid reusable grid cache.
    """

    scan_kind: str
    axis_labels: tuple[str, ...]
    axes: tuple[np.ndarray, ...]
    completed: np.ndarray
    energies_hartree: np.ndarray
    dipoles_debye: np.ndarray
    failures: tuple[ScanPointFailure, ...]
    stopped_early: bool
    executor_error: str | None = None

    @property
    def n_completed(self) -> int:
        """Number of points with an energy and dipole."""

        return int(np.count_nonzero(self.completed))

    @property
    def n_points(self) -> int:
        """Total number of requested points."""

        return int(self.completed.size)

    def summary(self, *, max_failures: int = 3) -> str:
        """Return an actionable description of the failed scan."""

        parts = [
            f"{self.scan_kind} scan incomplete: {self.n_completed} of {self.n_points} "
            f"point(s) completed, {len(self.failures)} failed"
        ]
        if self.stopped_early:
            parts.append("stopped scheduling new points after the first failure")
        if self.executor_error is not None:
            parts.append(f"executor failure: {self.executor_error}")
        shown = [failure.describe() for failure in self.failures[: int(max_failures)]]
        if shown:
            parts.append("; ".join(shown))
        hidden = len(self.failures) - len(shown)
        if hidden > 0:
            parts.append(f"and {hidden} more failure(s)")
        return "; ".join(parts)


class ScanFailedError(RuntimeError):
    """Raised when scan points fail; ``partial`` holds the completed points."""

    def __init__(self, message: str, partial: PartialScanGrid) -> None:
        super().__init__(message)
        self.partial = partial

    def __reduce__(self):
        return (type(self), (str(self), self.partial))


@dataclass(frozen=True)
class _PointOutcome:
    value: Any = None
    error_type: str | None = None
    message: str = ""
    traceback: str = ""

    @property
    def ok(self) -> bool:
        return self.error_type is None


@dataclass(frozen=True)
class _RunOutcome:
    values: dict[int, Any]
    failures: dict[int, _PointOutcome]
    stopped_early: bool
    executor_error: str | None


def _guarded_call(worker: Callable[[tuple[Any, ...]], Any], task: tuple[Any, ...]) -> Any:
    """Run one scan task and return an outcome instead of raising."""

    try:
        return _PointOutcome(value=worker(task))
    except Exception as exc:
        return _PointOutcome(
            error_type=f"{type(exc).__module__}.{type(exc).__qualname__}".removeprefix(
                "builtins."
            ),
            message=str(exc),
            traceback=traceback.format_exc(),
        )


def normalize_bond(bond: Bond | str) -> Bond:
    """Return a package ``Bond`` from a bond object or index string."""

    if isinstance(bond, str):
        return parse_bond(bond)
    if isinstance(bond, Bond):
        return bond
    raise TypeError("bond must be a Bond or an index specification such as '0-1'")


def bond_label(bond: Bond | str) -> str:
    """Return the canonical cache label for a bond."""

    b = normalize_bond(bond)
    return f"{b.i}-{b.j}"


def stretch_along_bond(coords: np.ndarray, bond: Bond | str, new_len_A: float) -> np.ndarray:
    """Return coordinates with the second bond atom moved to a new bond length."""

    return _stretch_along_bond(coords, normalize_bond(bond), new_len_A)


def stretch_two_bonds(
    coords: np.ndarray,
    bond1: Bond | str,
    bond2: Bond | str,
    new_len1_A: float,
    new_len2_A: float,
) -> np.ndarray:
    """Return coordinates realizing two independent bond lengths exactly."""

    return _stretch_two_bonds(
        coords,
        normalize_bond(bond1),
        normalize_bond(bond2),
        new_len1_A,
        new_len2_A,
    )


def molecule_with_coords(molecule: Molecule, coords: np.ndarray) -> Molecule:
    """Copy a package molecule with new coordinates."""

    return Molecule.from_arrays(
        molecule.symbols,
        np.asarray(coords, dtype=float),
        charge=molecule.charge,
        spin=molecule.spin,
        label=molecule.label,
        masses_amu=molecule.masses,
    )


def normal_mode_displaced_coords(
    coords: np.ndarray,
    u_dir: np.ndarray,
    displacement_A: float,
) -> np.ndarray:
    """Return coordinates displaced by ``displacement_A * u_dir``."""

    coords_arr = np.asarray(coords, dtype=float)
    direction = np.asarray(u_dir, dtype=float)
    if direction.shape != coords_arr.shape:
        raise ValueError(
            f"u_dir shape {direction.shape} does not match coordinates {coords_arr.shape}"
        )
    return coords_arr + float(displacement_A) * direction


def normal_mode_effective_mass_amu(molecule: Any, u_dir: np.ndarray) -> float:
    """Return the effective mass for a unit-normalized Cartesian path."""

    direction = _normal_mode_direction_for_molecule(molecule, u_dir)
    masses = molecule_analysis_masses(molecule)
    return float(np.sum(masses * np.sum(direction * direction, axis=1)))


def local_bond_reduced_mass_amu(molecule: Any, bond: Bond | str | Any) -> float:
    """Return the diatomic reduced mass for the selected local stretch."""

    b = normalize_bond(bond)
    masses = molecule_analysis_masses(molecule)
    _validate_bond_indices(b, masses.size)
    mi = float(masses[b.i])
    mj = float(masses[b.j])
    if mi <= 0.0 or mj <= 0.0:
        raise ValueError("Selected bond atoms must have positive masses")
    return mi * mj / (mi + mj)


def bond_bond_g12_inv_amu(molecule: Any, b1: Bond | str | Any, b2: Bond | str | Any) -> float:
    """Return the constant Wilson-G cross term for two local bond stretches."""

    bond1 = normalize_bond(b1)
    bond2 = normalize_bond(b2)
    coords = _molecule_coords(molecule)
    masses = molecule_analysis_masses(molecule)
    _validate_bond_indices(bond1, masses.size)
    _validate_bond_indices(bond2, masses.size)
    shared = set((bond1.i, bond1.j)) & set((bond2.i, bond2.j))
    if not shared:
        return 0.0
    if len(shared) != 1:
        raise ValueError("gmatrix KEO requires two distinct bonds that share at most one atom")

    shared_idx = shared.pop()
    shared_mass = float(masses[shared_idx])
    if shared_mass <= 0.0:
        raise ValueError("Shared atom mass must be positive for gmatrix KEO")

    u1 = _bond_unit_vector(coords, bond1)
    u2 = _bond_unit_vector(coords, bond2)
    sign1 = _bond_derivative_sign(bond1, shared_idx)
    sign2 = _bond_derivative_sign(bond2, shared_idx)
    cos_term = max(-1.0, min(1.0, float(np.dot(u1, u2))))
    return sign1 * sign2 * cos_term / shared_mass


def normal_mode_direction_from_modes(
    molecule: Any,
    bond: Bond | str | Any,
    modes: np.ndarray,
    freqs_cm: np.ndarray,
) -> NormalModeDirection:
    """Select the normal mode with the strongest target-bond projection."""

    b = normalize_bond(bond)
    coords = _molecule_coords(molecule)
    modes_arr = np.asarray(modes, dtype=float)
    freqs_arr = np.asarray(freqs_cm, dtype=float)
    masses = molecule_analysis_masses(molecule)
    natm = len(molecule.symbols)

    if coords.shape != (natm, 3):
        raise ValueError("molecule coordinates must have shape (n_atoms, 3)")
    if masses.shape != (natm,):
        raise ValueError("molecule masses must have one entry per atom")
    if modes_arr.ndim != 2 or modes_arr.shape[0] != 3 * natm:
        raise ValueError("modes must have shape (3*n_atoms, n_modes)")
    if freqs_arr.ndim != 1 or freqs_arr.shape[0] != modes_arr.shape[1]:
        raise ValueError("freqs_cm must have one entry per mode column")

    axis_vec = coords[b.j] - coords[b.i]
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1e-12:
        raise ValueError("Zero target-bond axis length")
    axis_unit = axis_vec / axis_norm

    mass_rep = np.repeat(masses, 3)
    scores = []
    for k in range(modes_arr.shape[1]):
        u_cart = (modes_arr[:, k] / np.sqrt(mass_rep)).reshape(natm, 3)
        rel = u_cart[b.j] - u_cart[b.i]
        scores.append(abs(float(np.dot(rel, axis_unit))))
    mode_index = int(np.argmax(scores))

    u_best = (modes_arr[:, mode_index] / np.sqrt(mass_rep)).reshape(natm, 3)
    norm = float(np.linalg.norm(u_best))
    if norm < 1e-14:
        raise RuntimeError("Normal-mode direction has near-zero norm")
    return NormalModeDirection(
        u_dir=u_best / norm,
        mode_index=mode_index,
        frequency_cm=float(freqs_arr[mode_index]),
        modes=modes_arr,
        freqs_cm=freqs_arr,
    )


def calc_normal_mode_direction(
    molecule: Any,
    cfg: Any,
    bond: Bond | str,
    *,
    rtproj: str = "pyscf",
    strict: bool = True,
    allow_fd_hessian: bool = False,
    harmonic_fn: HarmonicFn | None = None,
    log_fn: LogFn | None = None,
) -> NormalModeDirection:
    """Run/select the normal-mode scan direction with strongest target-bond projection."""

    if harmonic_fn is None:
        from pyscf_vscf.workflows.harmonic import harmonic_analysis

        harmonic_fn = harmonic_analysis

    result = harmonic_fn(
        molecule,
        cfg,
        rtproj=rtproj,
        strict=strict,
        allow_fd_hessian=allow_fd_hessian,
        debug=False,
    )
    selected = normal_mode_direction_from_modes(
        molecule,
        bond,
        getattr(result, "modes"),
        getattr(result, "freqs_cm"),
    )
    if log_fn is not None:
        log_fn(
            "Selected normal mode index "
            f"{selected.mode_index} with freq {selected.frequency_cm:.1f} cm^-1 "
            f"for {bond_label(bond)} scan"
        )
    return selected


def grid_1d_pes_dms(
    molecule: Any,
    cfg: Any,
    bond: Bond | str | Any,
    Rmin: float = 0.75,
    Rmax: float = 1.25,
    npts: int = 41,
    *,
    energy_dipole_fn: EnergyDipoleFn = energy_dipole,
    executor_factory: ExecutorFactory | None = None,
    molecule_factory: MoleculeFactory = molecule_with_coords,
    progress_fn: ProgressFn | None = None,
    log_fn: LogFn | None = None,
    stop_on_error: bool = True,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a frozen local-bond 1D PES/DMS grid with injectable execution.

    If any point raises, :class:`ScanFailedError` is raised after the active
    batch finishes; its ``partial`` attribute holds every completed point.
    With ``stop_on_error=False`` all points are attempted before raising.
    """

    deprecated(
        "grid_1d_pes_dms() is deprecated; use pyscf_vscf.scans.plan_bond_scan() and run_scan()."
    )

    b = normalize_bond(bond)
    Rs = np.linspace(float(Rmin), float(Rmax), _npts(npts))
    energies = np.full(Rs.size, np.nan, dtype=float)
    dipoles = np.full((Rs.size, 3), np.nan, dtype=float)
    completed = np.zeros(Rs.size, dtype=bool)
    tasks = [
        (idx, float(R), molecule, cfg, b, energy_dipole_fn, molecule_factory)
        for idx, R in enumerate(Rs)
    ]
    if log_fn is not None:
        log_fn(
            f"Evaluating 1D grid: {Rs.size} points from {float(Rmin):.2f} to {float(Rmax):.2f} A"
        )

    run = _run_tasks(
        _grid_1d_lbs_worker,
        tasks,
        total=Rs.size,
        progress_label="1D grid points",
        executor_factory=executor_factory,
        progress_fn=progress_fn,
        stop_on_error=stop_on_error,
        batch_size=batch_size,
    )
    for idx, energy, dipole in run.values.values():
        energies[idx] = energy
        dipoles[idx] = dipole
        completed[idx] = True
    _raise_if_incomplete(
        run,
        scan_kind=f"1D local-bond ({bond_label(b)})",
        axis_labels=("R_A",),
        axes=(Rs,),
        completed=completed,
        energies=energies,
        dipoles=dipoles,
        point_of=lambda position: ((position,), (float(Rs[position]),)),
    )
    energies -= float(np.min(energies))
    return Rs, energies, dipoles


def grid_1d_pes_dms_normal(
    molecule: Any,
    cfg: Any,
    u_dir: np.ndarray,
    smin: float = -0.15,
    smax: float = 0.15,
    npts: int = 41,
    *,
    energy_dipole_fn: EnergyDipoleFn = energy_dipole,
    executor_factory: ExecutorFactory | None = None,
    molecule_factory: MoleculeFactory = molecule_with_coords,
    progress_fn: ProgressFn | None = None,
    log_fn: LogFn | None = None,
    stop_on_error: bool = True,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 1D PES/DMS grid along a fixed normal-mode displacement direction.

    Failed points raise :class:`ScanFailedError` as in :func:`grid_1d_pes_dms`.
    """

    deprecated(
        "grid_1d_pes_dms_normal() is deprecated; use pyscf_vscf.scans.plan_normal_scan() and run_scan()."
    )

    direction = _normal_mode_direction_for_molecule(molecule, u_dir)
    S = np.linspace(float(smin), float(smax), _npts(npts))
    energies = np.full(S.size, np.nan, dtype=float)
    dipoles = np.full((S.size, 3), np.nan, dtype=float)
    completed = np.zeros(S.size, dtype=bool)
    tasks = [
        (idx, float(s), molecule, cfg, direction, energy_dipole_fn, molecule_factory)
        for idx, s in enumerate(S)
    ]
    if log_fn is not None:
        log_fn(
            "Evaluating 1D grid (normal-mode path): "
            f"{S.size} points from {float(smin):.2f} to {float(smax):.2f} A"
        )

    run = _run_tasks(
        _grid_1d_normal_worker,
        tasks,
        total=S.size,
        progress_label="1D grid points",
        executor_factory=executor_factory,
        progress_fn=progress_fn,
        stop_on_error=stop_on_error,
        batch_size=batch_size,
    )
    for idx, energy, dipole in run.values.values():
        energies[idx] = energy
        dipoles[idx] = dipole
        completed[idx] = True
    _raise_if_incomplete(
        run,
        scan_kind="1D normal-coordinate",
        axis_labels=("s_A",),
        axes=(S,),
        completed=completed,
        energies=energies,
        dipoles=dipoles,
        point_of=lambda position: ((position,), (float(S[position]),)),
    )
    energies -= float(np.min(energies))
    return S, energies, dipoles


def grid_1d_pes_dms_normal_relaxed(
    molecule: Any,
    cfg: Any,
    u_dir: np.ndarray,
    smin: float = -0.15,
    smax: float = 0.15,
    npts: int = 41,
    *,
    relaxed_point_fn: RelaxedNormalPointFn | None = None,
    executor_factory: ExecutorFactory | None = None,
    progress_fn: ProgressFn | None = None,
    log_fn: LogFn | None = None,
    gtol: float = 1e-4,
    maxiter: int = 100,
    strict: bool = True,
    stop_on_error: bool = True,
    batch_size: int | None = None,
) -> NormalRelaxedGrid:
    """Orchestrate an exactly constrained relaxed normal-coordinate scan.

    Points whose optimizer does not converge are reported through
    :class:`NormalRelaxedGrid` (``strict`` decides between raising and
    warning). Points that raise an exception produce :class:`ScanFailedError`.
    """

    if relaxed_point_fn is None:
        raise NotImplementedError("normal-relaxed scans require an injected relaxed_point_fn")

    direction = _normal_mode_direction_for_molecule(molecule, u_dir)
    S = np.linspace(float(smin), float(smax), _npts(npts))
    energies = np.full(S.size, np.nan, dtype=float)
    dipoles = np.full((S.size, 3), np.nan, dtype=float)
    completed = np.zeros(S.size, dtype=bool)
    achieved = np.zeros(S.size, dtype=float)
    residuals = np.zeros(S.size, dtype=float)
    converged = np.zeros(S.size, dtype=bool)
    iterations = np.zeros(S.size, dtype=int)
    messages = [""] * S.size
    tasks = [
        (idx, float(s), molecule, cfg, direction, relaxed_point_fn, float(gtol), int(maxiter))
        for idx, s in enumerate(S)
    ]
    if log_fn is not None:
        log_fn(
            "Evaluating 1D grid (normal-mode path, constrained relax): "
            f"{S.size} points from {float(smin):.2f} to {float(smax):.2f} A"
        )

    run = _run_tasks(
        _grid_1d_normal_relaxed_worker,
        tasks,
        total=S.size,
        progress_label="1D grid points",
        executor_factory=executor_factory,
        progress_fn=progress_fn,
        stop_on_error=stop_on_error,
        batch_size=batch_size,
    )
    for idx, energy, dipole, actual, residual, ok, n_iter, message in run.values.values():
        energies[idx] = energy
        dipoles[idx] = dipole
        completed[idx] = True
        achieved[idx] = actual
        residuals[idx] = residual
        converged[idx] = ok
        iterations[idx] = n_iter
        messages[idx] = message
    _raise_if_incomplete(
        run,
        scan_kind="1D normal-coordinate relaxed",
        axis_labels=("s_A",),
        axes=(S,),
        completed=completed,
        energies=energies,
        dipoles=dipoles,
        point_of=lambda position: ((position,), (float(S[position]),)),
    )
    if np.max(np.abs(residuals)) > 1e-10:
        raise RuntimeError(
            "normal-relaxed constraint residual exceeded 1e-10 A: "
            f"max={np.max(np.abs(residuals)):.3e} A"
        )
    energies -= float(np.min(energies))
    result = NormalRelaxedGrid(
        displacements_A=S,
        energies_hartree=energies,
        dipoles_debye=dipoles,
        achieved_displacements_A=achieved,
        constraint_residuals_A=residuals,
        converged=converged,
        iterations=iterations,
        messages=tuple(messages),
    )
    if result.failed_indices:
        summary = result.failure_summary()
        if bool(strict):
            raise RuntimeError(summary)
        warnings.warn(summary, RuntimeWarning, stacklevel=2)
    return result


def grid_2d_pes_dms(
    molecule: Any,
    cfg: Any,
    b1: Bond | str | Any,
    b2: Bond | str | Any,
    R1: Sequence[float] | np.ndarray,
    R2: Sequence[float] | np.ndarray,
    *,
    energy_dipole_fn: EnergyDipoleFn = energy_dipole,
    executor_factory: ExecutorFactory | None = None,
    molecule_factory: MoleculeFactory = molecule_with_coords,
    progress_fn: ProgressFn | None = None,
    log_fn: LogFn | None = None,
    stop_on_error: bool = True,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a frozen two-local-bond PES/DMS grid with injectable execution.

    Failed points raise :class:`ScanFailedError` as in :func:`grid_1d_pes_dms`.
    """

    deprecated(
        "grid_2d_pes_dms() is deprecated; use pyscf_vscf.scans.plan_bond_scan() with two bonds and run_scan()."
    )

    bond1 = normalize_bond(b1)
    bond2 = normalize_bond(b2)
    R1_arr = np.asarray(R1, dtype=float)
    R2_arr = np.asarray(R2, dtype=float)
    _validate_scan_grid(R1_arr, "R1")
    _validate_scan_grid(R2_arr, "R2")
    # Validate topology and source geometry before scheduling any electronic point.
    stretch_two_bonds(
        _molecule_coords(molecule),
        bond1,
        bond2,
        float(R1_arr[0]),
        float(R2_arr[0]),
    )
    shape = (R1_arr.size, R2_arr.size)
    energies = np.full(shape, np.nan, dtype=float)
    dipoles = np.full(shape + (3,), np.nan, dtype=float)
    completed = np.zeros(shape, dtype=bool)
    tasks = [
        (
            i,
            j,
            molecule,
            cfg,
            bond1,
            bond2,
            float(r1),
            float(r2),
            energy_dipole_fn,
            molecule_factory,
        )
        for i, r1 in enumerate(R1_arr)
        for j, r2 in enumerate(R2_arr)
    ]
    if log_fn is not None:
        log_fn(f"Evaluating 2D grid: {R1_arr.size}x{R2_arr.size} = {len(tasks)} points")

    run = _run_tasks(
        _grid_2d_lbs_worker,
        tasks,
        total=len(tasks),
        progress_label="2D grid points",
        executor_factory=executor_factory,
        progress_fn=progress_fn,
        stop_on_error=stop_on_error,
        batch_size=batch_size,
    )
    for i, j, energy, dipole in run.values.values():
        energies[i, j] = energy
        dipoles[i, j] = dipole
        completed[i, j] = True

    def point_of(position: int) -> tuple[tuple[int, ...], tuple[float, ...]]:
        i, j = divmod(position, R2_arr.size)
        return (i, j), (float(R1_arr[i]), float(R2_arr[j]))

    _raise_if_incomplete(
        run,
        scan_kind=f"2D local-bond ({bond_label(bond1)}, {bond_label(bond2)})",
        axis_labels=("R1_A", "R2_A"),
        axes=(R1_arr, R2_arr),
        completed=completed,
        energies=energies,
        dipoles=dipoles,
        point_of=point_of,
    )
    energies -= float(np.min(energies))
    return R1_arr, R2_arr, energies, dipoles


def dump_partial_scan_npz(
    path: Path | str,
    partial: PartialScanGrid,
    molecule: Any,
    cfg: Any,
    *,
    backend_identity: str = "pyscf",
) -> None:
    """Write completed points of a failed scan as a diagnostic NPZ file.

    The file is checksummed like a grid cache but deliberately carries no
    schema-v3 cache identity, so it can never be reused as a grid cache.
    """

    meta = {
        "schema": PARTIAL_SCAN_SCHEMA,
        "schema_version": PARTIAL_SCAN_SCHEMA_VERSION,
        "reusable_grid_cache": False,
        "scan_kind": partial.scan_kind,
        "axis_labels": list(partial.axis_labels),
        "n_points": partial.n_points,
        "n_completed": partial.n_completed,
        "stopped_early": bool(partial.stopped_early),
        "executor_error": partial.executor_error,
        "failures": [failure.as_dict() for failure in partial.failures],
        "energy_reference": "absolute",
        "dipole_units": "Debye",
        "molecule": molecule_provenance(molecule),
        "electronic_structure": electronic_structure_provenance(
            cfg, backend_identity=backend_identity
        ),
        "runtime": runtime_provenance(),
    }
    arrays = {
        label: np.asarray(axis, dtype=float)
        for label, axis in zip(partial.axis_labels, partial.axes)
    }
    arrays["completed"] = np.asarray(partial.completed, dtype=bool)
    arrays["E_abs_Eh"] = np.asarray(partial.energies_hartree, dtype=float)
    arrays["MU_Debye"] = np.asarray(partial.dipoles_debye, dtype=float)
    dump_grid_npz(Path(path), meta=meta, arrays=arrays)


def lbs_frozen_1d_cache_metadata(
    molecule: Any,
    cfg: Any,
    bond: Bond | str | Any,
    rmin: float,
    rmax: float,
    npts: int,
    *,
    scan: str = "lbs-frozen",
    backend_identity: str = "pyscf",
) -> dict[str, Any]:
    """Return schema-v3 identity and provenance for a frozen 1D bond grid."""

    b = normalize_bond(bond)
    grid = np.linspace(float(rmin), float(rmax), _npts(npts))
    scan_meta = {
        "kind": "1d",
        "coordinate": scan,
        "bond_zero_based": [b.i, b.j],
        "grid_A": grid.tolist(),
        "energy_reference": "minimum-grid-energy",
        "dipole_units": "Debye",
    }
    return scientific_cache_metadata(molecule, cfg, scan_meta, backend_identity=backend_identity)


def validate_lbs_frozen_1d_cache_metadata(
    meta: Mapping[str, Any],
    molecule: Any,
    cfg: Any,
    bond: Bond | str | Any,
    rmin: float,
    rmax: float,
    npts: int,
    *,
    scan: str = "lbs-frozen",
    backend_identity: str = "pyscf",
) -> None:
    """Validate the complete scientific fingerprint before reusing a 1D cache."""

    expected = lbs_frozen_1d_cache_metadata(
        molecule,
        cfg,
        bond,
        rmin,
        rmax,
        npts,
        scan=scan,
        backend_identity=backend_identity,
    )
    validate_scientific_cache_metadata(dict(meta), expected)


def dump_lbs_frozen_1d_grid_cache(
    path: Path,
    molecule: Any,
    cfg: Any,
    bond: Bond | str | Any,
    rmin: float,
    rmax: float,
    npts: int,
    R: np.ndarray,
    E: np.ndarray,
    MU: np.ndarray,
    *,
    scan: str = "lbs-frozen",
    backend_identity: str = "pyscf",
) -> None:
    """Write a frozen 1D local-bond grid cache."""

    deprecated("Schema-3 grid caches are deprecated; use pyscf_vscf.artifacts.scans.dump_scan().")

    meta = lbs_frozen_1d_cache_metadata(
        molecule,
        cfg,
        bond,
        rmin,
        rmax,
        npts,
        scan=scan,
        backend_identity=backend_identity,
    )
    dump_grid_npz(path, meta=meta, arrays={"R_A": R, "E_Eh": E, "MU_Debye": MU})


def load_lbs_frozen_1d_grid_cache(
    path: Path,
    molecule: Any,
    cfg: Any,
    bond: Bond | str | Any,
    rmin: float,
    rmax: float,
    npts: int,
    *,
    scan: str = "lbs-frozen",
    backend_identity: str = "pyscf",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a frozen 1D local-bond grid cache."""

    deprecated(
        "Schema-3 grid cache loaders are deprecated; use pyscf_vscf.artifacts.scans.load_scan() or load_legacy_grid()."
    )

    meta, arrays = load_grid_npz(path)
    validate_lbs_frozen_1d_cache_metadata(
        meta,
        molecule,
        cfg,
        bond,
        rmin,
        rmax,
        npts,
        scan=scan,
        backend_identity=backend_identity,
    )
    R = np.asarray(arrays["R_A"], dtype=float)
    E = np.asarray(arrays["E_Eh"], dtype=float)
    MU = np.asarray(arrays["MU_Debye"], dtype=float)
    requested = np.linspace(float(rmin), float(rmax), _npts(npts))
    _validate_grid_arrays_1d(R, E, MU, requested)
    return R, E, MU


def lbs_frozen_2d_cache_metadata(
    molecule: Any,
    cfg: Any,
    b1: Bond | str | Any,
    b2: Bond | str | Any,
    r1: Sequence[float],
    r2: Sequence[float],
    *,
    keo: str = "gmatrix",
    backend_identity: str = "pyscf",
) -> dict[str, Any]:
    """Return schema-v3 identity and provenance for a frozen 2D bond grid."""

    bond1 = normalize_bond(b1)
    bond2 = normalize_bond(b2)
    keo_lc = (keo or "gmatrix").lower()
    if keo_lc not in {"reduced", "gmatrix"}:
        raise ValueError(f"Unknown KEO {keo!r}; expected 'reduced' or 'gmatrix'")
    r1_min, r1_max, r1_npts = _grid_spec(r1, "r1")
    r2_min, r2_max, r2_npts = _grid_spec(r2, "r2")
    reduced_masses = [
        local_bond_reduced_mass_amu(molecule, bond1),
        local_bond_reduced_mass_amu(molecule, bond2),
    ]
    g12_inv_amu = 0.0
    if keo_lc == "gmatrix":
        g12_inv_amu = bond_bond_g12_inv_amu(molecule, bond1, bond2)
    scan_meta = {
        "kind": "2d",
        "coordinate": "lbs-frozen",
        "keo": keo_lc,
        "reduced_masses_amu": reduced_masses,
        "g12_inv_amu": g12_inv_amu,
        "gmatrix_reference_geometry": "identity.molecule.coordinates_A",
        "bond1_zero_based": [bond1.i, bond1.j],
        "bond2_zero_based": [bond2.i, bond2.j],
        "grid1_A": np.linspace(r1_min, r1_max, r1_npts).tolist(),
        "grid2_A": np.linspace(r2_min, r2_max, r2_npts).tolist(),
        "energy_reference": "minimum-grid-energy",
        "dipole_units": "Debye",
    }
    return scientific_cache_metadata(molecule, cfg, scan_meta, backend_identity=backend_identity)


def validate_lbs_frozen_2d_cache(
    meta: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    molecule: Any,
    cfg: Any,
    b1: Bond | str | Any,
    b2: Bond | str | Any,
    r1: Sequence[float],
    r2: Sequence[float],
    *,
    keo: str = "gmatrix",
    backend_identity: str = "pyscf",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return arrays from a frozen 2D local-bond grid cache."""

    r1_min, r1_max, r1_npts = _grid_spec(r1, "r1")
    R1_req = np.linspace(float(r1_min), float(r1_max), int(r1_npts))
    r2_min, r2_max, r2_npts = _grid_spec(r2, "r2")
    R2_req = np.linspace(float(r2_min), float(r2_max), int(r2_npts))

    expected = lbs_frozen_2d_cache_metadata(
        molecule,
        cfg,
        b1,
        b2,
        r1,
        r2,
        keo=keo,
        backend_identity=backend_identity,
    )
    validate_scientific_cache_metadata(dict(meta), expected)

    R1 = np.asarray(arrays["R1_A"], dtype=float)
    R2 = np.asarray(arrays["R2_A"], dtype=float)
    E = np.asarray(arrays["E_Eh"], dtype=float)
    MU = np.asarray(arrays["MU_Debye"], dtype=float)
    _validate_grid_arrays_2d(R1, R2, E, MU, R1_req, R2_req)
    return R1, R2, E, MU


def load_lbs_frozen_2d_grid_cache(
    path: Path,
    molecule: Any,
    cfg: Any,
    b1: Bond | str | Any,
    b2: Bond | str | Any,
    r1: Sequence[float],
    r2: Sequence[float],
    *,
    keo: str = "gmatrix",
    backend_identity: str = "pyscf",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a frozen 2D local-bond grid cache."""

    deprecated(
        "Schema-3 grid cache loaders are deprecated; use pyscf_vscf.artifacts.scans.load_scan() or load_legacy_grid()."
    )

    meta, arrays = load_grid_npz(path)
    return validate_lbs_frozen_2d_cache(
        meta,
        arrays,
        molecule,
        cfg,
        b1,
        b2,
        r1,
        r2,
        keo=keo,
        backend_identity=backend_identity,
    )


def _grid_1d_lbs_worker(task: tuple[Any, ...]) -> tuple[int, float, np.ndarray]:
    idx, R, molecule, cfg, bond, energy_fn, molecule_factory = task
    coords = stretch_along_bond(_molecule_coords(molecule), bond, R)
    return _indexed_energy_dipole(idx, molecule, cfg, coords, energy_fn, molecule_factory)


def _grid_1d_normal_worker(task: tuple[Any, ...]) -> tuple[int, float, np.ndarray]:
    idx, s, molecule, cfg, u_dir, energy_fn, molecule_factory = task
    coords = normal_mode_displaced_coords(_molecule_coords(molecule), u_dir, s)
    return _indexed_energy_dipole(idx, molecule, cfg, coords, energy_fn, molecule_factory)


def _grid_1d_normal_relaxed_worker(task: tuple[Any, ...]) -> tuple[Any, ...]:
    idx, s, molecule, cfg, u_dir, relaxed_point_fn, gtol, maxiter = task
    result = relaxed_point_fn(molecule, cfg, u_dir, s, gtol, maxiter)
    energy, dipole = _coerce_energy_dipole((result.energy_hartree, result.dipole_debye))
    return (
        int(idx),
        energy,
        dipole,
        float(result.achieved_displacement_A),
        float(result.constraint_residual_A),
        bool(result.converged),
        int(result.n_iterations),
        str(result.message),
    )


def _grid_2d_lbs_worker(task: tuple[Any, ...]) -> tuple[int, int, float, np.ndarray]:
    i, j, molecule, cfg, b1, b2, r1, r2, energy_fn, molecule_factory = task
    coords = stretch_two_bonds(_molecule_coords(molecule), b1, b2, r1, r2)
    energy, dipole = _point_energy_dipole(molecule, cfg, coords, energy_fn, molecule_factory)
    return int(i), int(j), energy, dipole


def _run_tasks(
    worker: Callable[[tuple[Any, ...]], Any],
    tasks: Sequence[tuple[Any, ...]],
    *,
    total: int,
    progress_label: str,
    executor_factory: ExecutorFactory | None,
    progress_fn: ProgressFn | None,
    stop_on_error: bool = True,
    batch_size: int | None = None,
) -> _RunOutcome:
    """Run scan tasks in bounded batches and collect every outcome.

    Executors only need a ``map`` method. Work is submitted one batch at a
    time so that, after a failure, no further batch is scheduled when
    ``stop_on_error`` is true. Cancellation of work already submitted is best
    effort: the active batch always finishes. Exceptions raised by the
    executor itself (for example a broken process pool) end the run while
    keeping every result collected so far.
    """

    guarded = partial(_guarded_call, worker)
    values: dict[int, Any] = {}
    failures: dict[int, _PointOutcome] = {}
    done = 0

    def record(position: int, outcome: _PointOutcome) -> None:
        nonlocal done
        done += 1
        if outcome.ok:
            values[position] = outcome.value
        else:
            failures[position] = outcome
        if progress_fn is not None:
            progress_fn(done, total, progress_label)

    if executor_factory is None:
        for position, task in enumerate(tasks):
            record(position, guarded(task))
            if failures and stop_on_error:
                return _RunOutcome(values, failures, position + 1 < len(tasks), None)
        return _RunOutcome(values, failures, False, None)

    size = _batch_size(batch_size)
    with executor_factory() as executor:
        for start in range(0, len(tasks), size):
            batch = list(tasks[start : start + size])
            try:
                for offset, outcome in enumerate(executor.map(guarded, batch)):
                    record(start + offset, outcome)
            except Exception as exc:
                return _RunOutcome(
                    values,
                    failures,
                    start + size < len(tasks),
                    f"{type(exc).__name__}: {exc}",
                )
            if failures and stop_on_error:
                return _RunOutcome(values, failures, start + size < len(tasks), None)
    return _RunOutcome(values, failures, False, None)


def _batch_size(batch_size: int | None) -> int:
    if batch_size is None:
        return DEFAULT_SCAN_BATCH_SIZE
    size = int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be at least 1")
    return size


def _raise_if_incomplete(
    run: _RunOutcome,
    *,
    scan_kind: str,
    axis_labels: tuple[str, ...],
    axes: tuple[np.ndarray, ...],
    completed: np.ndarray,
    energies: np.ndarray,
    dipoles: np.ndarray,
    point_of: Callable[[int], tuple[tuple[int, ...], tuple[float, ...]]],
) -> None:
    if not run.failures and run.executor_error is None and bool(np.all(completed)):
        return
    failures = []
    for position, outcome in sorted(run.failures.items()):
        index, scan_values = point_of(position)
        failures.append(
            ScanPointFailure(
                index=tuple(int(value) for value in index),
                scan_values=tuple(float(value) for value in scan_values),
                axis_labels=axis_labels,
                error_type=str(outcome.error_type),
                message=outcome.message,
                traceback=outcome.traceback,
            )
        )
    partial_grid = PartialScanGrid(
        scan_kind=scan_kind,
        axis_labels=axis_labels,
        axes=tuple(np.array(axis, dtype=float, copy=True) for axis in axes),
        completed=np.array(completed, dtype=bool, copy=True),
        energies_hartree=np.array(energies, dtype=float, copy=True),
        dipoles_debye=np.array(dipoles, dtype=float, copy=True),
        failures=tuple(failures),
        stopped_early=bool(run.stopped_early),
        executor_error=run.executor_error,
    )
    raise ScanFailedError(partial_grid.summary(), partial_grid)


def _indexed_energy_dipole(
    idx: int,
    molecule: Any,
    cfg: Any,
    coords: np.ndarray,
    energy_fn: EnergyDipoleFn,
    molecule_factory: MoleculeFactory,
) -> tuple[int, float, np.ndarray]:
    energy, dipole = _point_energy_dipole(molecule, cfg, coords, energy_fn, molecule_factory)
    return int(idx), energy, dipole


def _point_energy_dipole(
    molecule: Any,
    cfg: Any,
    coords: np.ndarray,
    energy_fn: EnergyDipoleFn,
    molecule_factory: MoleculeFactory,
) -> tuple[float, np.ndarray]:
    return _coerce_energy_dipole(energy_fn(molecule_factory(molecule, coords), cfg))


def _coerce_energy_dipole(value: tuple[float, np.ndarray]) -> tuple[float, np.ndarray]:
    energy, dipole = value
    energy_value = float(energy)
    # Copy: results are buffered before assembly, and an evaluator may reuse
    # one scratch array for every point.
    dipole_arr = np.array(dipole, dtype=float, copy=True)
    if dipole_arr.shape != (3,):
        raise ValueError(f"dipole must have shape (3,), got {dipole_arr.shape}")
    if not np.isfinite(energy_value):
        raise ValueError("energy must be finite and expressed in Hartree")
    if not np.all(np.isfinite(dipole_arr)):
        raise ValueError("dipole must be finite and expressed in Debye")
    return energy_value, dipole_arr


def _validate_grid_arrays_1d(
    R: np.ndarray,
    E: np.ndarray,
    MU: np.ndarray,
    requested: np.ndarray,
) -> None:
    if R.shape != requested.shape or not np.allclose(R, requested, rtol=0.0, atol=1e-12):
        raise ValueError("Grid cache mismatch: R array does not match requested grid")
    if E.shape != R.shape:
        raise ValueError("Grid cache mismatch: E_Eh shape must match R_A")
    if MU.shape != (R.size, 3):
        raise ValueError("Grid cache mismatch: MU_Debye shape must be (npts, 3)")
    if not all(np.all(np.isfinite(value)) for value in (R, E, MU)):
        raise ValueError("Grid cache contains non-finite values")


def _validate_grid_arrays_2d(
    R1: np.ndarray,
    R2: np.ndarray,
    E: np.ndarray,
    MU: np.ndarray,
    requested1: np.ndarray,
    requested2: np.ndarray,
) -> None:
    if R1.shape != requested1.shape or not np.allclose(R1, requested1, rtol=0.0, atol=1e-12):
        raise ValueError("Grid cache mismatch: R1 array does not match requested grid")
    if R2.shape != requested2.shape or not np.allclose(R2, requested2, rtol=0.0, atol=1e-12):
        raise ValueError("Grid cache mismatch: R2 array does not match requested grid")
    if E.shape != (R1.size, R2.size):
        raise ValueError("Grid cache mismatch: E_Eh shape must be (npts1, npts2)")
    if MU.shape != (R1.size, R2.size, 3):
        raise ValueError("Grid cache mismatch: MU_Debye shape must be (npts1, npts2, 3)")
    if not all(np.all(np.isfinite(value)) for value in (R1, R2, E, MU)):
        raise ValueError("Grid cache contains non-finite values")


def _validate_bond_indices(bond: Bond, n_atoms: int) -> None:
    if bond.i < 0 or bond.j < 0 or bond.i >= n_atoms or bond.j >= n_atoms:
        raise IndexError(f"Bond {bond_label(bond)} is out of range for {n_atoms} atoms")
    if bond.i == bond.j:
        raise ValueError("A bond must reference two distinct atoms")


def _bond_unit_vector(coords: np.ndarray, bond: Bond) -> np.ndarray:
    _validate_bond_indices(bond, coords.shape[0])
    vec = coords[bond.j] - coords[bond.i]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-12:
        raise ValueError("Invalid bond geometry for gmatrix KEO (zero bond vector norm)")
    return vec / norm


def _bond_derivative_sign(bond: Bond, atom_index: int) -> float:
    if atom_index == bond.i:
        return -1.0
    if atom_index == bond.j:
        return 1.0
    raise ValueError(f"Atom {atom_index} is not part of bond {bond_label(bond)}")


def _normal_mode_direction_for_molecule(molecule: Any, u_dir: np.ndarray) -> np.ndarray:
    direction = np.asarray(u_dir, dtype=float)
    coords = _molecule_coords(molecule)
    if direction.shape != coords.shape:
        raise ValueError(f"u_dir shape {direction.shape} does not match molecule coordinates")
    norm = float(np.linalg.norm(direction))
    if not np.isfinite(norm) or norm < 1e-14:
        raise ValueError("u_dir must have a finite non-zero norm")
    return direction / norm


def _molecule_coords(molecule: Any) -> np.ndarray:
    coords = np.asarray(getattr(molecule, "coords"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("molecule coordinates must have shape (n_atoms, 3)")
    return coords


def molecule_analysis_masses(molecule: Molecule) -> np.ndarray:
    """Return isotope masses in amu from a package molecule."""

    return molecule.analysis_masses()


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _npts(npts: int) -> int:
    count = int(npts)
    if count <= 0:
        raise ValueError("npts must be positive")
    return count


def _grid_spec(spec: Sequence[float], name: str) -> tuple[float, float, int]:
    values = list(spec)
    if len(values) != 3:
        raise ValueError(f"{name} must contain (min, max, npts)")
    return float(values[0]), float(values[1]), int(values[2])


def _validate_scan_grid(values: np.ndarray, name: str) -> None:
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional grid")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError(f"{name} bond lengths must be positive and finite")


__all__ = [
    "Bond",
    "EnergyDipoleFn",
    "ExecutorFactory",
    "HarmonicFn",
    "LogFn",
    "MoleculeFactory",
    "DEFAULT_SCAN_BATCH_SIZE",
    "NormalModeDirection",
    "NormalRelaxedGrid",
    "PartialScanGrid",
    "ProgressFn",
    "RelaxedNormalPointFn",
    "ScanFailedError",
    "ScanPointFailure",
    "bond_bond_g12_inv_amu",
    "bond_label",
    "calc_normal_mode_direction",
    "dump_lbs_frozen_1d_grid_cache",
    "dump_partial_scan_npz",
    "grid_1d_pes_dms",
    "grid_1d_pes_dms_normal",
    "grid_1d_pes_dms_normal_relaxed",
    "grid_2d_pes_dms",
    "lbs_frozen_1d_cache_metadata",
    "lbs_frozen_2d_cache_metadata",
    "load_lbs_frozen_1d_grid_cache",
    "load_lbs_frozen_2d_grid_cache",
    "local_bond_reduced_mass_amu",
    "molecule_analysis_masses",
    "molecule_with_coords",
    "normal_mode_direction_from_modes",
    "normal_mode_displaced_coords",
    "normal_mode_effective_mass_amu",
    "normalize_bond",
    "parse_bond",
    "stretch_along_bond",
    "stretch_two_bonds",
    "validate_lbs_frozen_1d_cache_metadata",
    "validate_lbs_frozen_2d_cache",
]
