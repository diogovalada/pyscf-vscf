"""Typed potential and dipole scans built on electronic point plans.

A scan is planned (:func:`plan_bond_scan`, :func:`plan_normal_scan`),
evaluated by :func:`pyscf_vscf.runner.run_plan`, and assembled into a
:class:`ScanResult` with absolute energies in Hartree and dipoles in atomic
units in the input Cartesian frame. :func:`run_scan` does all three.
Planning is pure: it needs no electronic-structure backend.

The 0.2 tuple-returning functions in :mod:`pyscf_vscf.workflows.scans` remain
available (deprecated) for one release.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ._arrays import immutable_array
from ._identity import FrozenJSONMapping, immutable_json_mapping, payload_fingerprint, to_jsonable
from .constants import atomic_number
from .contracts import (
    AU_DIPOLE_TO_DEBYE,
    ElectronicPointRequest,
    ElectronicProvider,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from .coordinates import Bond, parse_bond, stretch_along_bond, stretch_two_bonds
from .point_store import PointStore
from .runner import (
    PlanIncompleteError,
    PlanRun,
    ProgressEvent,
    RunnerSettings,
    run_plan,
    validate_result,
)

SCAN_PLAN_SCHEMA = "pyscf-vscf-scan-plan"
SCAN_PLAN_SCHEMA_VERSION = 1

# Default bond grid relative to the equilibrium bond length r_e (angstrom).
# Chosen by benchmark; see docs/scans.md.
DEFAULT_BOND_GRID_INNER_A = 0.30
DEFAULT_BOND_GRID_OUTER_A = 0.70
DEFAULT_BOND_GRID_POINTS = 41
MIN_GRID_POINTS = 3


@dataclass(frozen=True)
class ScanAxis:
    """One scanned coordinate: its kind, unit, grid values, and definition."""

    label: str
    coordinate: str
    values: np.ndarray
    unit: str = "angstrom"
    bond: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1 or values.size < MIN_GRID_POINTS or not np.all(np.isfinite(values)):
            raise ValueError(f"Axis {self.label!r} needs at least {MIN_GRID_POINTS} finite values")
        if not np.all(np.diff(values) > 0.0):
            raise ValueError(f"Axis {self.label!r} must be strictly increasing")
        if self.coordinate not in {"bond-length", "normal-displacement"}:
            raise ValueError(f"Unknown axis coordinate {self.coordinate!r}")
        object.__setattr__(self, "values", immutable_array(values, dtype="<f8"))
        if self.bond is not None:
            object.__setattr__(self, "bond", (int(self.bond[0]), int(self.bond[1])))

    def payload(self) -> dict[str, object]:
        """Return the axis definition as a JSON-ready mapping."""

        return {
            "label": self.label,
            "coordinate": self.coordinate,
            "unit": self.unit,
            "bond_zero_based": None if self.bond is None else list(self.bond),
            "values": self.values.tolist(),
        }


@dataclass(frozen=True)
class ScanPlan:
    """Electronic point requests for every grid point of a scan.

    ``point_keys`` has the grid shape and holds the point causal fingerprint
    of each grid point; identical geometries share a request. ``sampling``
    records how geometries were constructed and is fingerprinted separately
    from the electronic points.
    """

    scan_kind: str
    axes: tuple[ScanAxis, ...]
    point_keys: np.ndarray
    requests: Mapping[str, ElectronicPointRequest]
    provider_scientific_fingerprint: str
    provider_payload: FrozenJSONMapping
    reference_geometry: FrozenJSONMapping
    sampling: FrozenJSONMapping

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape (one entry per axis)."""

        return tuple(axis.values.size for axis in self.axes)

    def sampling_fingerprint(self) -> str:
        """Fingerprint of the sampling definition (geometry construction and axes)."""

        return payload_fingerprint(dict(self.sampling))

    def plan_fingerprint(self) -> str:
        """Fingerprint of the complete plan: provider, sampling, and point keys."""

        return payload_fingerprint(
            {
                "schema": SCAN_PLAN_SCHEMA,
                "schema_version": SCAN_PLAN_SCHEMA_VERSION,
                "provider": self.provider_scientific_fingerprint,
                "sampling": self.sampling_fingerprint(),
                "point_keys": np.asarray(self.point_keys).ravel().tolist(),
            }
        )


@dataclass(frozen=True)
class ScanResult:
    """Absolute energies and dipoles on a scan grid.

    ``total_energies_Eh`` and ``dipoles_au`` (input Cartesian frame) have the
    grid shape (dipoles with a trailing axis of 3). Use
    :attr:`relative_energies_Eh` and :attr:`dipoles_debye` for the 0.2
    conventions.
    """

    scan_kind: str
    axes: tuple[ScanAxis, ...]
    total_energies_Eh: np.ndarray
    dipoles_au: np.ndarray
    point_keys: np.ndarray
    provider_scientific_fingerprint: str
    provider_payload: Mapping[str, Any]
    reference_geometry: Mapping[str, Any]
    sampling: Mapping[str, Any]
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = tuple(axis.values.size for axis in self.axes)
        energies = np.asarray(self.total_energies_Eh, dtype=float)
        dipoles = np.asarray(self.dipoles_au, dtype=float)
        if energies.shape != shape or dipoles.shape != shape + (3,):
            raise ValueError("Scan arrays do not match the axes")
        if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(dipoles)):
            raise ValueError("Scan arrays must be finite")
        keys = np.asarray(self.point_keys, dtype="<U64")
        if keys.shape != shape:
            raise ValueError("point_keys must have the grid shape")
        object.__setattr__(self, "total_energies_Eh", immutable_array(energies, dtype="<f8"))
        object.__setattr__(self, "dipoles_au", immutable_array(dipoles, dtype="<f8"))
        object.__setattr__(self, "point_keys", immutable_array(keys))
        for name in ("provider_payload", "reference_geometry", "sampling", "provenance"):
            object.__setattr__(self, name, immutable_json_mapping(getattr(self, name)))

    @property
    def shape(self) -> tuple[int, ...]:
        """Grid shape."""

        return tuple(axis.values.size for axis in self.axes)

    @property
    def reference_energy_Eh(self) -> float:
        """The lowest energy on the grid."""

        return float(np.min(self.total_energies_Eh))

    @property
    def relative_energies_Eh(self) -> np.ndarray:
        """Energies relative to the grid minimum (the 0.2 convention)."""

        return np.asarray(self.total_energies_Eh) - self.reference_energy_Eh

    @property
    def dipoles_debye(self) -> np.ndarray:
        """Dipoles in Debye (input Cartesian frame)."""

        return np.asarray(self.dipoles_au) * AU_DIPOLE_TO_DEBYE

    def as_legacy_tuple(self) -> tuple[np.ndarray, ...]:
        """Return ``(*axis_values, relative_energies_Eh, dipoles_debye)`` as in 0.2."""

        return (
            *(np.array(axis.values) for axis in self.axes),
            self.relative_energies_Eh,
            self.dipoles_debye,
        )


def default_bond_grid(r_e_A: float) -> tuple[float, float, int]:
    """Return the default ``(r_min, r_max, n_points)`` around an equilibrium length.

    Raises ``ValueError`` if the inner bound would not be positive.
    """

    r_e = float(r_e_A)
    lower = r_e - DEFAULT_BOND_GRID_INNER_A
    if not np.isfinite(r_e) or lower <= 0.0:
        raise ValueError(
            f"The default bond grid needs r_e > {DEFAULT_BOND_GRID_INNER_A} A (got {r_e:g}); "
            "give the grid explicitly"
        )
    return lower, r_e + DEFAULT_BOND_GRID_OUTER_A, DEFAULT_BOND_GRID_POINTS


def plan_bond_scan(
    molecule: Any,
    provider: ElectronicProvider,
    bonds: Sequence[Bond | str],
    grids: Sequence[Sequence[float] | np.ndarray] | None = None,
) -> ScanPlan:
    """Plan a frozen scan of one or two bond lengths (angstrom).

    The second atom of each bond moves (two bonds sharing an atom are
    stretched about that atom). ``grids`` defaults to :func:`default_bond_grid`
    around each bond's length in ``molecule``.
    """

    parsed = tuple(_bond(value) for value in bonds)
    if len(parsed) not in (1, 2):
        raise ValueError("Bond scans need one or two bonds")
    coordinates = np.asarray(molecule.coords, dtype=float)
    if grids is None:
        grids = []
        for bond in parsed:
            r_e = float(np.linalg.norm(coordinates[bond.j] - coordinates[bond.i]))
            lower, upper, count = default_bond_grid(r_e)
            grids.append(np.linspace(lower, upper, count))
    if len(grids) != len(parsed):
        raise ValueError("Give one grid per bond")
    axes = tuple(
        ScanAxis(
            label=f"r({bond.i}-{bond.j})",
            coordinate="bond-length",
            values=np.asarray(grid, dtype=float),
            bond=(bond.i, bond.j),
        )
        for bond, grid in zip(parsed, grids)
    )
    if np.any(axes[0].values <= 0.0) or (len(axes) == 2 and np.any(axes[1].values <= 0.0)):
        raise ValueError("Bond lengths must be positive")
    geometries: dict[tuple[int, ...], np.ndarray]
    if len(axes) == 1:
        geometries = {
            (index,): stretch_along_bond(coordinates, parsed[0], value)
            for index, value in enumerate(axes[0].values)
        }
        kind = "bond-1d"
    else:
        geometries = {
            (i, j): stretch_two_bonds(coordinates, parsed[0], parsed[1], r1, r2)
            for i, r1 in enumerate(axes[0].values)
            for j, r2 in enumerate(axes[1].values)
        }
        kind = "bond-2d"
    sampling = {
        "schema": SCAN_PLAN_SCHEMA,
        "schema_version": SCAN_PLAN_SCHEMA_VERSION,
        "scan_kind": kind,
        "geometry_construction": "frozen-bond-stretch",
        "axes": [axis.payload() for axis in axes],
    }
    return _plan(molecule, provider, kind, axes, geometries, sampling)


def plan_normal_scan(
    molecule: Any,
    provider: ElectronicProvider,
    direction: np.ndarray,
    displacements_A: Sequence[float] | np.ndarray,
    *,
    label: str = "s",
) -> ScanPlan:
    """Plan a rigid scan along a Cartesian direction (angstrom displacements).

    ``direction`` has shape ``(n_atoms, 3)`` and is normalized to unit
    Euclidean norm; grid point ``s`` places the atoms at ``x0 + s * u``.
    """

    coordinates = np.asarray(molecule.coords, dtype=float)
    unit = np.asarray(direction, dtype=float)
    if unit.shape != coordinates.shape:
        raise ValueError("direction must have the molecule's coordinate shape")
    norm = float(np.linalg.norm(unit))
    if not np.isfinite(norm) or norm < 1e-14:
        raise ValueError("direction must have a finite non-zero norm")
    unit = unit / norm
    axis = ScanAxis(
        label=label, coordinate="normal-displacement", values=np.asarray(displacements_A)
    )
    geometries: dict[tuple[int, ...], np.ndarray] = {
        (index,): coordinates + float(value) * unit for index, value in enumerate(axis.values)
    }
    sampling = {
        "schema": SCAN_PLAN_SCHEMA,
        "schema_version": SCAN_PLAN_SCHEMA_VERSION,
        "scan_kind": "normal-1d",
        "geometry_construction": "rigid-cartesian-displacement",
        "direction": unit.tolist(),
        "axes": [axis.payload()],
    }
    return _plan(molecule, provider, "normal-1d", (axis,), geometries, sampling)


def assemble_scan(
    plan: ScanPlan,
    results: Mapping[str, ElectronicResult],
    *,
    provenance: Mapping[str, Any] | None = None,
) -> ScanResult:
    """Place the results of every planned point on the scan grid.

    Every result must be converged, come from the plan's provider, and belong
    to its grid point.
    """

    energies = np.empty(plan.shape, dtype=float)
    dipoles = np.empty(plan.shape + (3,), dtype=float)
    for index in np.ndindex(plan.shape):
        key = str(plan.point_keys[index])
        if key not in results:
            raise ValueError(f"Missing result for grid point {list(index)}")
        result = results[key]
        try:
            validate_result(result, plan.requests[key], plan.provider_scientific_fingerprint)
        except ValueError as exc:
            raise ValueError(f"Grid point {list(index)}: {exc}") from exc
        if result.dipole_au is None:
            raise ValueError("Scan results need dipoles")
        energies[index] = result.total_energy_Eh
        dipoles[index] = result.dipole_au
    return ScanResult(
        scan_kind=plan.scan_kind,
        axes=plan.axes,
        total_energies_Eh=energies,
        dipoles_au=dipoles,
        point_keys=plan.point_keys,
        provider_scientific_fingerprint=plan.provider_scientific_fingerprint,
        provider_payload=plan.provider_payload,
        reference_geometry=plan.reference_geometry,
        sampling=plan.sampling,
        provenance=dict(provenance or {}),
    )


def run_scan(
    plan: ScanPlan,
    provider: ElectronicProvider,
    *,
    store: PointStore | None = None,
    settings: RunnerSettings | None = None,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> ScanResult:
    """Evaluate a scan plan and assemble it.

    Raises :class:`~pyscf_vscf.runner.PlanIncompleteError` (carrying the
    :class:`~pyscf_vscf.runner.PlanRun`) when any point is missing.
    """

    run = run_plan(plan, provider, store=store, settings=settings, progress=progress)
    if not run.complete:
        raise PlanIncompleteError(f"{plan.scan_kind} scan incomplete: {run.summary()}", run)
    return assemble_scan(
        plan,
        run.results,
        provenance={
            "run_id": run.run_id,
            "points_reused": len(run.reused),
            "wall_seconds": run.wall_seconds,
        },
    )


def bond_spectrum(
    scan: ScanResult,
    molecule: Any,
    *,
    max_state_index: int = 8,
    kinetic_operator: str = "gmatrix",
    intensity_axis: Sequence[float] | np.ndarray | None = None,
) -> list[Any]:
    """Return DVR transition records for a one- or two-bond scan.

    Masses come from ``molecule``, so one scan serves every isotopologue on
    the same surface. ``max_state_index`` is the highest reported state (the
    vibrational quantum number in 1D, the eigenstate index in 2D).
    ``kinetic_operator`` (2D only) is ``"gmatrix"`` (with the Wilson-G cross
    term) or ``"reduced"``. Polarized intensities use ``intensity_axis``,
    default the first bond's direction.
    """

    from .variational import variational_1d, variational_2d
    from .workflows.scans import bond_bond_g12_inv_amu, local_bond_reduced_mass_amu

    if scan.scan_kind not in {"bond-1d", "bond-2d"}:
        raise ValueError(f"bond_spectrum needs a bond scan, got {scan.scan_kind!r}")
    _check_same_system(scan, molecule)
    bonds = []
    for axis_definition in scan.axes:
        if axis_definition.bond is None:
            raise ValueError("bond scan axes must name their bond")
        bonds.append(Bond(*axis_definition.bond))
    coordinates = np.asarray(molecule.coords, dtype=float)
    axis = (
        coordinates[bonds[0].j] - coordinates[bonds[0].i]
        if intensity_axis is None
        else np.asarray(intensity_axis, dtype=float)
    )
    energies = scan.relative_energies_Eh
    dipoles = scan.dipoles_debye
    if scan.scan_kind == "bond-1d":
        return variational_1d(
            scan.axes[0].values,
            energies,
            dipoles,
            local_bond_reduced_mass_amu(molecule, bonds[0]),
            axis=axis,
            vmax=int(max_state_index),
        )
    keo = str(kinetic_operator).lower()
    if keo not in {"gmatrix", "reduced"}:
        raise ValueError("kinetic_operator must be 'gmatrix' or 'reduced'")
    g12 = bond_bond_g12_inv_amu(molecule, bonds[0], bonds[1]) if keo == "gmatrix" else 0.0
    return variational_2d(
        scan.axes[0].values,
        scan.axes[1].values,
        energies,
        dipoles,
        local_bond_reduced_mass_amu(molecule, bonds[0]),
        local_bond_reduced_mass_amu(molecule, bonds[1]),
        axis=axis,
        nmax=int(max_state_index),
        g12_inv_amu=g12,
    )


def normal_coordinate_spectrum(
    scan: ScanResult,
    molecule: Any,
    *,
    max_state_index: int = 8,
    intensity_axis: Sequence[float] | np.ndarray | None = None,
) -> list[Any]:
    """Return DVR transition records for a rigid normal-coordinate scan.

    The effective mass of the unit Cartesian direction uses ``molecule``'s
    masses. Polarized intensities default to the z axis.
    """

    from .variational import variational_1d

    if scan.scan_kind != "normal-1d":
        raise ValueError(f"normal_coordinate_spectrum needs a normal scan, got {scan.scan_kind!r}")
    _check_same_system(scan, molecule)
    direction = np.asarray(scan.sampling["direction"], dtype=float)
    masses = np.asarray(molecule.analysis_masses(), dtype=float)
    effective_mass = float(np.sum(masses * np.sum(direction * direction, axis=1)))
    return variational_1d(
        scan.axes[0].values,
        scan.relative_energies_Eh,
        scan.dipoles_debye,
        effective_mass,
        axis=None if intensity_axis is None else np.asarray(intensity_axis, dtype=float),
        vmax=int(max_state_index),
    )


def _check_same_system(scan: ScanResult, molecule: Any) -> None:
    """Require ``molecule`` to be the scan's reference system (isotopes may differ)."""

    charges = [atomic_number(symbol) for symbol in molecule.symbols]
    reference = scan.reference_geometry
    coordinates = np.asarray(molecule.coords, dtype=float)
    stored = np.asarray(reference["coordinates_A"], dtype=float)
    if (
        list(reference["nuclear_charges"]) != charges
        or stored.shape != coordinates.shape
        or not np.allclose(stored, coordinates, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("The molecule does not match the scan's reference geometry")
    if (int(reference["charge"]), int(reference["spin"])) != (
        int(getattr(molecule, "charge", 0)),
        int(getattr(molecule, "spin", 0)),
    ):
        raise ValueError("The molecule's charge or spin differs from the scan's")


def _plan(
    molecule: Any,
    provider: ElectronicProvider,
    kind: str,
    axes: tuple[ScanAxis, ...],
    geometries: Mapping[tuple[int, ...], np.ndarray],
    sampling: Mapping[str, Any],
) -> ScanPlan:
    provider_id = provider_scientific_fingerprint(provider)
    charges = tuple(atomic_number(symbol) for symbol in molecule.symbols)
    charge = int(getattr(molecule, "charge", 0))
    spin = int(getattr(molecule, "spin", 0))
    shape = tuple(axis.values.size for axis in axes)
    keys = np.empty(shape, dtype="<U64")
    requests: dict[str, ElectronicPointRequest] = {}
    for index, coordinates in geometries.items():
        request = ElectronicPointRequest(
            nuclear_charges=charges,
            coordinates_A=coordinates,
            charge=charge,
            spin=spin,
        )
        key = request.causal_fingerprint(provider_id)
        requests.setdefault(key, request)
        keys[index] = key
    reference = {
        "nuclear_charges": list(charges),
        "coordinates_A": np.asarray(molecule.coords, dtype=float).tolist(),
        "charge": charge,
        "spin": spin,
    }
    return ScanPlan(
        scan_kind=kind,
        axes=axes,
        point_keys=immutable_array(keys),
        requests=requests,
        provider_scientific_fingerprint=provider_id,
        provider_payload=immutable_json_mapping(
            cast(Mapping[str, Any], to_jsonable(dict(provider.scientific_settings_payload())))
        ),
        reference_geometry=immutable_json_mapping(reference),
        sampling=immutable_json_mapping(sampling),
    )


def _bond(value: Bond | str) -> Bond:
    if isinstance(value, Bond):
        return value
    return parse_bond(str(value))


__all__ = [
    "DEFAULT_BOND_GRID_INNER_A",
    "DEFAULT_BOND_GRID_OUTER_A",
    "DEFAULT_BOND_GRID_POINTS",
    "PlanIncompleteError",
    "PlanRun",
    "RunnerSettings",
    "ScanAxis",
    "ScanPlan",
    "ScanResult",
    "assemble_scan",
    "bond_spectrum",
    "default_bond_grid",
    "normal_coordinate_spectrum",
    "plan_bond_scan",
    "plan_normal_scan",
    "run_scan",
]
