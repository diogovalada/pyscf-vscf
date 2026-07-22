"""Legacy-compatible scan workflow helpers.

This module sits above the pure coordinate/surface helpers and below the CLI.
It preserves the legacy frozen local-bond and normal-mode grid semantics while
keeping all expensive electronic-structure work injectable.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyscf_vscf.cache import (
    dump_grid_npz,
    load_grid_npz,
    scientific_cache_metadata,
    validate_scientific_cache_metadata,
)
from pyscf_vscf.constants import atomic_mass_amu
from pyscf_vscf.coordinates import Bond, parse_bond
from pyscf_vscf.coordinates import stretch_along_bond as _stretch_along_bond
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.surfaces import energy_dipole


EnergyDipoleFn = Callable[[Any, Any], tuple[float, np.ndarray]]
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
    """Selected normal-mode scan direction and legacy diagnostics."""

    u_dir: np.ndarray
    mode_index: int
    frequency_cm: float
    modes: np.ndarray
    freqs_cm: np.ndarray

    def as_legacy_tuple(self) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
        """Return the tuple shape used by ``pyscf_pme_pipeline.py``."""

        return self.u_dir, self.mode_index, self.frequency_cm, self.modes, self.freqs_cm


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

    def __iter__(self):
        """Allow legacy ``S, E, MU = result`` unpacking."""

        yield self.displacements_A
        yield self.energies_hartree
        yield self.dipoles_debye


def normalize_bond(bond: Bond | str | Any) -> Bond:
    """Return a package ``Bond`` from package, legacy, or string bond specs."""

    if isinstance(bond, str):
        return parse_bond(bond)
    if isinstance(bond, Bond):
        return bond
    if hasattr(bond, "i") and hasattr(bond, "j"):
        return Bond(int(getattr(bond, "i")), int(getattr(bond, "j")))
    if hasattr(bond, "O") and hasattr(bond, "H"):
        return Bond(int(getattr(bond, "O")), int(getattr(bond, "H")))
    raise TypeError("bond must be a Bond, legacy O/H bond, or bond specification string")


def bond_label(bond: Bond | str | Any) -> str:
    """Return the canonical cache label for a bond."""

    b = normalize_bond(bond)
    return f"{b.i}-{b.j}"


def legacy_bond_label(bond: Bond | str | Any) -> str:
    """Return the historical O/H-style cache label for compatibility."""

    b = normalize_bond(bond)
    return f"O{b.i}-H{b.j}"


def stretch_along_bond(coords: np.ndarray, bond: Bond | str | Any, new_len_A: float) -> np.ndarray:
    """Return coordinates with the second bond atom moved to a new bond length."""

    return _stretch_along_bond(coords, normalize_bond(bond), new_len_A)


def molecule_with_coords(molecule: Any, coords: np.ndarray) -> Molecule:
    """Build a package molecule from a duck-typed molecule and new coordinates."""

    masses_fn = getattr(molecule, "analysis_masses", None)
    masses = masses_fn() if callable(masses_fn) else getattr(molecule, "masses", None)
    return Molecule.from_arrays(
        _molecule_symbols(molecule),
        np.asarray(coords, dtype=float),
        charge=int(getattr(molecule, "charge", 0)),
        spin=int(getattr(molecule, "spin", 0)),
        label=str(getattr(molecule, "label", "mol")),
        masses_amu=masses,
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
    """Return legacy effective mass for a unit-normalized Cartesian path."""

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
    natm = len(_molecule_symbols(molecule))

    if coords.shape != (natm, 3):
        raise ValueError("molecule coordinates must have shape (n_atoms, 3)")
    if masses.shape != (natm,):
        raise ValueError("molecule masses must have one entry per atom")
    if modes_arr.ndim != 2 or modes_arr.shape[0] != 3 * natm:
        raise ValueError("modes must have shape (3*n_atoms, n_modes)")
    if freqs_arr.ndim != 1 or freqs_arr.shape[0] != modes_arr.shape[1]:
        raise ValueError("freqs_cm must have one entry per mode column")

    axis_vec = coords[b.H] - coords[b.O]
    axis_norm = float(np.linalg.norm(axis_vec))
    if axis_norm < 1e-12:
        raise ValueError("Zero target-bond axis length")
    axis_unit = axis_vec / axis_norm

    mass_rep = np.repeat(masses, 3)
    scores = []
    for k in range(modes_arr.shape[1]):
        u_cart = (modes_arr[:, k] / np.sqrt(mass_rep)).reshape(natm, 3)
        rel = u_cart[b.H] - u_cart[b.O]
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
    bond: Bond | str | Any,
    *,
    harmonic_fn: HarmonicFn | None = None,
    log_fn: LogFn | None = None,
) -> tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    """Run/select the normal-mode scan direction with strongest target-bond projection."""

    if harmonic_fn is None:
        from pyscf_vscf.workflows.harmonic import harmonic_analysis

        harmonic_fn = harmonic_analysis

    result = harmonic_fn(
        molecule,
        cfg,
        rtproj=_cfg_get(cfg, "rtproj", "pyscf"),
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
    return selected.as_legacy_tuple()


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a frozen local-bond 1D PES/DMS grid with injectable execution."""

    b = normalize_bond(bond)
    Rs = np.linspace(float(Rmin), float(Rmax), _npts(npts))
    energies = np.zeros(Rs.size, dtype=float)
    dipoles = np.zeros((Rs.size, 3), dtype=float)
    tasks = [
        (idx, float(R), molecule, cfg, b, energy_dipole_fn, molecule_factory)
        for idx, R in enumerate(Rs)
    ]
    if log_fn is not None:
        log_fn(
            f"Evaluating 1D grid: {Rs.size} points from {float(Rmin):.2f} to {float(Rmax):.2f} A"
        )

    for idx, energy, dipole in _run_tasks(
        _grid_1d_lbs_worker,
        tasks,
        total=Rs.size,
        progress_label="1D grid points",
        executor_factory=executor_factory,
        progress_fn=progress_fn,
    ):
        energies[idx] = energy
        dipoles[idx] = dipole
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a 1D PES/DMS grid along a fixed normal-mode displacement direction."""

    direction = _normal_mode_direction_for_molecule(molecule, u_dir)
    S = np.linspace(float(smin), float(smax), _npts(npts))
    energies = np.zeros(S.size, dtype=float)
    dipoles = np.zeros((S.size, 3), dtype=float)
    tasks = [
        (idx, float(s), molecule, cfg, direction, energy_dipole_fn, molecule_factory)
        for idx, s in enumerate(S)
    ]
    if log_fn is not None:
        log_fn(
            "Evaluating 1D grid (normal-mode path): "
            f"{S.size} points from {float(smin):.2f} to {float(smax):.2f} A"
        )

    for idx, energy, dipole in _run_tasks(
        _grid_1d_normal_worker,
        tasks,
        total=S.size,
        progress_label="1D grid points",
        executor_factory=executor_factory,
        progress_fn=progress_fn,
    ):
        energies[idx] = energy
        dipoles[idx] = dipole
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
) -> NormalRelaxedGrid:
    """Orchestrate an exactly constrained relaxed normal-coordinate scan."""

    if relaxed_point_fn is None:
        raise NotImplementedError("normal-relaxed scans require an injected relaxed_point_fn")

    direction = _normal_mode_direction_for_molecule(molecule, u_dir)
    S = np.linspace(float(smin), float(smax), _npts(npts))
    energies = np.zeros(S.size, dtype=float)
    dipoles = np.zeros((S.size, 3), dtype=float)
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

    for idx, energy, dipole, actual, residual, ok, n_iter, message in _run_tasks(
        _grid_1d_normal_relaxed_worker,
        tasks,
        total=S.size,
        progress_label="1D grid points",
        executor_factory=executor_factory,
        progress_fn=progress_fn,
    ):
        energies[idx] = energy
        dipoles[idx] = dipole
        achieved[idx] = actual
        residuals[idx] = residual
        converged[idx] = ok
        iterations[idx] = n_iter
        messages[idx] = message
    if np.max(np.abs(residuals)) > 1e-10:
        raise RuntimeError(
            "normal-relaxed constraint residual exceeded 1e-10 A: "
            f"max={np.max(np.abs(residuals)):.3e} A"
        )
    if not np.all(converged) and bool(_cfg_get(cfg, "strict", True)):
        failed = np.flatnonzero(~converged).tolist()
        raise RuntimeError(f"normal-relaxed points did not converge at indices {failed}")
    energies -= float(np.min(energies))
    return NormalRelaxedGrid(
        displacements_A=S,
        energies_hartree=energies,
        dipoles_debye=dipoles,
        achieved_displacements_A=achieved,
        constraint_residuals_A=residuals,
        converged=converged,
        iterations=iterations,
        messages=tuple(messages),
    )


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a frozen two-local-bond PES/DMS grid with injectable execution."""

    bond1 = normalize_bond(b1)
    bond2 = normalize_bond(b2)
    R1_arr = np.asarray(R1, dtype=float)
    R2_arr = np.asarray(R2, dtype=float)
    energies = np.zeros((R1_arr.size, R2_arr.size), dtype=float)
    dipoles = np.zeros((R1_arr.size, R2_arr.size, 3), dtype=float)
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

    for i, j, energy, dipole in _run_tasks(
        _grid_2d_lbs_worker,
        tasks,
        total=len(tasks),
        progress_label="2D grid points",
        executor_factory=executor_factory,
        progress_fn=progress_fn,
    ):
        energies[i, j] = energy
        dipoles[i, j] = dipole
    energies -= float(np.min(energies))
    return R1_arr, R2_arr, energies, dipoles


def lbs_frozen_1d_cache_metadata(
    molecule: Any,
    cfg: Any,
    bond: Bond | str | Any,
    rmin: float,
    rmax: float,
    npts: int,
    *,
    scan: str = "lbs-frozen",
) -> dict[str, Any]:
    """Return complete schema-v2 metadata for a frozen 1D bond grid."""

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
    return scientific_cache_metadata(molecule, cfg, scan_meta)


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
) -> None:
    """Write a frozen 1D local-bond grid cache with legacy array names."""

    meta = lbs_frozen_1d_cache_metadata(molecule, cfg, bond, rmin, rmax, npts, scan=scan)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a frozen 1D local-bond grid cache."""

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
) -> dict[str, Any]:
    """Return complete schema-v2 metadata for a frozen 2D bond grid."""

    bond1 = normalize_bond(b1)
    bond2 = normalize_bond(b2)
    r1_min, r1_max, r1_npts = _grid_spec(r1, "r1")
    r2_min, r2_max, r2_npts = _grid_spec(r2, "r2")
    scan_meta = {
        "kind": "2d",
        "coordinate": "lbs-frozen",
        "keo": keo or "gmatrix",
        "bond1_zero_based": [bond1.i, bond1.j],
        "bond2_zero_based": [bond2.i, bond2.j],
        "grid1_A": np.linspace(r1_min, r1_max, r1_npts).tolist(),
        "grid2_A": np.linspace(r2_min, r2_max, r2_npts).tolist(),
        "energy_reference": "minimum-grid-energy",
        "dipole_units": "Debye",
    }
    return scientific_cache_metadata(molecule, cfg, scan_meta)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate and return arrays from a frozen 2D local-bond grid cache."""

    r1_min, r1_max, r1_npts = _grid_spec(r1, "r1")
    R1_req = np.linspace(float(r1_min), float(r1_max), int(r1_npts))
    r2_min, r2_max, r2_npts = _grid_spec(r2, "r2")
    R2_req = np.linspace(float(r2_min), float(r2_max), int(r2_npts))

    expected = lbs_frozen_2d_cache_metadata(molecule, cfg, b1, b2, r1, r2, keo=keo)
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and validate a frozen 2D local-bond grid cache."""

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
    coords1 = stretch_along_bond(_molecule_coords(molecule), b1, r1)
    coords2 = stretch_along_bond(coords1, b2, r2)
    energy, dipole = _point_energy_dipole(molecule, cfg, coords2, energy_fn, molecule_factory)
    return int(i), int(j), energy, dipole


def _run_tasks(
    worker: Callable[[tuple[Any, ...]], Any],
    tasks: Sequence[tuple[Any, ...]],
    *,
    total: int,
    progress_label: str,
    executor_factory: ExecutorFactory | None,
    progress_fn: ProgressFn | None,
) -> Iterable[Any]:
    if executor_factory is None:
        results = (worker(task) for task in tasks)
        yield from _with_progress(results, total, progress_label, progress_fn)
        return

    with executor_factory() as executor:
        results = executor.map(worker, tasks)
        yield from _with_progress(results, total, progress_label, progress_fn)


def _with_progress(
    results: Iterable[Any],
    total: int,
    label: str,
    progress_fn: ProgressFn | None,
) -> Iterable[Any]:
    for done, result in enumerate(results, 1):
        if progress_fn is not None:
            progress_fn(done, total, label)
        yield result


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
    dipole_arr = np.asarray(dipole, dtype=float)
    if dipole_arr.shape != (3,):
        raise ValueError(f"dipole must have shape (3,), got {dipole_arr.shape}")
    return float(energy), dipole_arr


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


def _molecule_symbols(molecule: Any) -> list[str]:
    return [str(symbol) for symbol in getattr(molecule, "symbols")]


def molecule_analysis_masses(molecule: Any) -> np.ndarray:
    """Return isotope masses in amu from a duck-typed molecule."""

    analysis_masses = getattr(molecule, "analysis_masses", None)
    if callable(analysis_masses):
        return np.asarray(analysis_masses(), dtype=float)

    masses = getattr(molecule, "masses", None)
    if masses is not None:
        return np.asarray(masses, dtype=float)

    values = []
    for symbol in _molecule_symbols(molecule):
        key = symbol.upper()
        try:
            values.append(atomic_mass_amu(key))
        except ValueError as exc:
            raise ValueError(f"No mass is defined for atom symbol {key!r}") from exc
    return np.asarray(values, dtype=float)


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


__all__ = [
    "Bond",
    "EnergyDipoleFn",
    "ExecutorFactory",
    "HarmonicFn",
    "LogFn",
    "MoleculeFactory",
    "NormalModeDirection",
    "NormalRelaxedGrid",
    "ProgressFn",
    "RelaxedNormalPointFn",
    "bond_bond_g12_inv_amu",
    "bond_label",
    "calc_normal_mode_direction",
    "dump_lbs_frozen_1d_grid_cache",
    "grid_1d_pes_dms",
    "grid_1d_pes_dms_normal",
    "grid_1d_pes_dms_normal_relaxed",
    "grid_2d_pes_dms",
    "lbs_frozen_1d_cache_metadata",
    "lbs_frozen_2d_cache_metadata",
    "load_lbs_frozen_1d_grid_cache",
    "load_lbs_frozen_2d_grid_cache",
    "legacy_bond_label",
    "local_bond_reduced_mass_amu",
    "molecule_analysis_masses",
    "molecule_with_coords",
    "normal_mode_direction_from_modes",
    "normal_mode_displaced_coords",
    "normal_mode_effective_mass_amu",
    "normalize_bond",
    "parse_bond",
    "stretch_along_bond",
    "validate_lbs_frozen_1d_cache_metadata",
    "validate_lbs_frozen_2d_cache",
]
