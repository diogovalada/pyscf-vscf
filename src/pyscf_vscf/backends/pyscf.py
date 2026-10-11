"""PySCF backend: the mean-field electronic provider and legacy helpers.

PySCF is imported only when a calculation runs, so this module imports in
environments without PySCF. Every calculation builds its molecule and mean
field through the construction seam in
:mod:`pyscf_vscf.backends._pyscf_factory`; convergence and properties come
from :mod:`pyscf_vscf.backends._pyscf_properties`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from socket import gethostname
from typing import Any

import numpy as np

from .._identity import immutable_json_mapping, to_jsonable
from .._warnings import deprecated
from ..cache import runtime_provenance
from ..constants import ANG_TO_BOHR, MASS_AMU, atomic_mass_amu
from ..contracts import (
    AU_DIPOLE_TO_DEBYE,
    ElectronicPointRequest,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from ..molecule import Molecule
from ..settings import (
    DEFAULT_PROVIDER_MAX_MEMORY_MB,
    ESSettings,
    coerce_es_settings,
    default_auxbasis,
    runtime_max_memory_mb,
)
from ._pyscf_factory import (
    BackendUnavailableError,
    ConfigurationGuard,
    ElectronicSystem,
    MeanFieldFactory,
    MethodRecipe,
    PySCFResources,
    SettingsFactory,
    build_mean_field,
    configure_settings_mean_field,
    require_pyscf,
)
from ._pyscf_properties import (
    CONTINUITY_SUCCESS_SCOPE,
    ConvergedMeanField,
    continuity_diagnostics,
    converge,
    current_pyscf_threads,
    dipole_au,
    max_rss_mb,
    nuclear_gradient,
)

_WARNED_ONCE: set[str] = set()
_CONTINUITY_SUCCESS_SCOPE = CONTINUITY_SUCCESS_SCOPE


@dataclass(frozen=True)
class NormalRelaxedPointResult:
    """Result of an exactly constrained normal-coordinate relaxation."""

    energy_hartree: float
    dipole_debye: np.ndarray
    coords_A: np.ndarray
    requested_displacement_A: float
    achieved_displacement_A: float
    constraint_residual_A: float
    converged: bool
    n_iterations: int
    message: str
    configuration_snapshot: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class PySCFMeanFieldProvider:
    """Mean-field electronic points with split scientific and execution identity.

    Pass ``ESSettings`` (or a mapping/object with the same fields) for the
    built-in method, or ``factory=`` a :class:`MethodRecipe` (any
    :class:`MeanFieldFactory`) for a user-supplied PySCF method. ``threads``
    is advisory: worker processes apply it at start-up, and :meth:`evaluate`
    never changes PySCF's process-wide thread count. ``max_memory_mb`` is the
    PySCF memory budget of each calculation; by default the runner's worker
    budget (``RunnerSettings.max_memory_mb``) applies, else 4000 MB.
    """

    settings: object = None
    threads: int = 1
    max_memory_mb: int | None = None
    user_annotations: Mapping[str, Any] = field(default_factory=dict)
    retain_occupied_mo_coefficients: bool = False
    continuity_diagnostics: str = "off"
    factory: Any = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if self.factory is None:
            if self.settings is None:
                raise TypeError("Provide electronic settings or factory=")
            factory = SettingsFactory(coerce_es_settings(self.settings))
            object.__setattr__(self, "factory", factory)
            object.__setattr__(self, "settings", factory.settings)
        else:
            if self.settings is not None:
                raise TypeError("Provide either settings or factory=, not both")
            if not isinstance(self.factory, MeanFieldFactory):
                raise TypeError(
                    "factory must provide scientific_settings_payload(), build_molecule(), "
                    "and build_mean_field()"
                )
        threads = int(self.threads)
        memory = None if self.max_memory_mb is None else int(self.max_memory_mb)
        if threads <= 0 or (memory is not None and memory <= 0):
            raise ValueError("threads and max_memory_mb must be positive")
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "max_memory_mb", memory)
        object.__setattr__(
            self,
            "retain_occupied_mo_coefficients",
            bool(self.retain_occupied_mo_coefficients),
        )
        continuity_mode = _normalized_continuity_mode(self.continuity_diagnostics)
        if self.retain_occupied_mo_coefficients and continuity_mode != "strict":
            raise ValueError(
                "retain_occupied_mo_coefficients requires continuity_diagnostics='strict'"
            )
        object.__setattr__(self, "continuity_diagnostics", continuity_mode)
        object.__setattr__(
            self,
            "user_annotations",
            immutable_json_mapping(self.user_annotations),
        )

    def scientific_settings_payload(self) -> Mapping[str, object]:
        """Return every setting that can change the intended calculation.

        For built-in settings the payload is unchanged from 0.2, so provider
        fingerprints and stored results remain valid.
        """

        if isinstance(self.factory, SettingsFactory):
            method_fields: dict[str, object] = {
                "provider": "mean-field",
                "reference_policy": "restricted-if-spin-zero-otherwise-unrestricted",
                "settings": dict(self.factory.scientific_settings_payload()),
                "frozen_core_policy": "not-applicable",
                "orbital_policy": "self-consistent-canonical",
                "post_scf_tolerance": None,
                "integral_policy": "pyscf-default-in-core-or-direct",
            }
        else:
            method_fields = {
                "provider": "mean-field-recipe",
                "method": dict(self.factory.scientific_settings_payload()),
            }
        return {
            "schema": "pyscf-vscf-provider-settings",
            "schema_version": 1,
            "backend_family": "pyscf",
            **method_fields,
            "continuity_diagnostics": {
                "mode": self.continuity_diagnostics,
                "scope": (
                    "disabled"
                    if self.continuity_diagnostics == "off"
                    else CONTINUITY_SUCCESS_SCOPE
                ),
                "primary_density_population_partition": "meta-lowdin",
                "meta_lowdin_pre_orthogonalization": "ANO",
                "secondary_density_population_partition": "mulliken",
                "occupied_orbital_population_partition": "mulliken",
                "retain_occupied_mo_coefficients": self.retain_occupied_mo_coefficients,
            },
            "field_convention": {
                "field_units": "atomic_unit",
                "origin_units": "angstrom",
                "hamiltonian": "H(F)=H(0)-F.mu",
                "nuclear_term": "included-in-total-energy",
            },
        }

    def memory_budget_mb(self) -> int:
        """Return the PySCF memory budget: ``max_memory_mb``, else the
        ``PYSCF_VSCF_MAX_MEMORY_MB`` worker setting, else 4000 MB."""

        if self.max_memory_mb is not None:
            return int(self.max_memory_mb)
        return runtime_max_memory_mb() or DEFAULT_PROVIDER_MAX_MEMORY_MB

    def execution_provenance(self) -> Mapping[str, object]:
        """Return runtime resources and software versions outside causal identity."""

        return {
            "threads_requested": self.threads,
            "max_memory_mb": self.memory_budget_mb(),
            "host": gethostname(),
            "user_annotations": to_jsonable(self.user_annotations),
            "runtime": runtime_provenance(),
        }

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        """Evaluate one immutable request without changing its causal identity."""

        if request.electronic_state != "ground":
            raise ValueError("PySCFMeanFieldProvider supports only electronic_state='ground'")
        if request.field_au is not None and "dipole" in request.requested_properties:
            raise ValueError("Finite-field requests support energy only")
        if "dipole" in request.requested_properties and request.charge != 0:
            raise ValueError(
                "Charged-system dipoles require an independently selected origin convention"
            )

        started = time.perf_counter()
        _, _, _, _ = require_pyscf()
        build = build_mean_field(
            ElectronicSystem.from_request(request),
            self.factory,
            resources=PySCFResources(max_memory_mb=self.memory_budget_mb()),
        )
        converged = converge(
            build,
            field_au=request.field_au,
            field_origin_A=request.field_origin_A,
        )
        dipole = None
        if "dipole" in request.requested_properties:
            dipole = dipole_au(converged)
        field_vector = np.zeros(3) if request.field_au is None else request.field_au
        field_origin = np.zeros(3) if request.field_origin_A is None else request.field_origin_A
        scientific_diagnostics = {
            "field_au": np.asarray(field_vector, dtype=float).tolist(),
            "field_origin_A": np.asarray(field_origin, dtype=float).tolist(),
            "field_hamiltonian": "H(F)=H(0)-F.mu",
            "nuclear_field_energy_Eh": converged.nuclear_field_energy_Eh,
            "nuclear_field_term_included": True,
        }
        if self.continuity_diagnostics != "off":
            scientific_diagnostics.update(
                continuity_diagnostics(
                    converged,
                    mode=self.continuity_diagnostics,
                    retain_occupied_mo_coefficients=self.retain_occupied_mo_coefficients,
                )
            )
        provider_id = provider_scientific_fingerprint(self)
        provenance = dict(self.execution_provenance())
        provenance["configuration_snapshot"] = to_jsonable(build.configuration_snapshot)
        return ElectronicResult(
            total_energy_Eh=converged.total_energy_Eh,
            dipole_au=dipole,
            dipole_unit=None if dipole is None else "atomic_unit",
            dipole_frame=None if dipole is None else "input_cartesian",
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
            scientific_diagnostics=scientific_diagnostics,
            execution_diagnostics={
                "reference_class": type(converged.mean_field).__name__,
                "scf_cycles": converged.cycles,
                "runtime_seconds": time.perf_counter() - started,
                "max_rss_mb": max_rss_mb(),
                "threads_actual": current_pyscf_threads(),
                "max_memory_mb": self.memory_budget_mb(),
                "warnings": [],
            },
            provenance=provenance,
        )


def _normalized_continuity_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("continuity_diagnostics must be a string")
    mode = value.strip().lower()
    if mode not in {"off", "best-effort", "strict"}:
        raise ValueError("continuity_diagnostics must be 'off', 'best-effort', or 'strict'")
    return mode


def method_factory(cfg: Any) -> MeanFieldFactory:
    """Return the factory for ``cfg``: itself if it is a factory, else built-in settings."""

    if isinstance(cfg, MeanFieldFactory) and not isinstance(cfg, Mapping):
        return cfg
    return SettingsFactory(coerce_es_settings(cfg))


def converged_mean_field(
    molecule: Any,
    cfg: Any,
    *,
    resources: PySCFResources | None = None,
) -> ConvergedMeanField:
    """Build through the seam and converge the mean field for a package molecule.

    ``cfg`` is ``ESSettings`` (or compatible) or a :class:`MeanFieldFactory`.
    """

    return converge(
        build_mean_field(
            ElectronicSystem.from_molecule(molecule), method_factory(cfg), resources=resources
        )
    )


def legacy_energy_dipole(
    molecule: Any, cfg: Any, *, guard: ConfigurationGuard | None = None
) -> tuple[float, np.ndarray]:
    """Return the Hartree energy and Debye dipole of one point (0.2 evaluator contract).

    ``guard`` rejects a method configuration that differs from earlier builds.
    """

    converged = converged_mean_field(molecule, cfg)
    if guard is not None:
        guard.check(converged.build.configuration_snapshot)
    return converged.total_energy_Eh, dipole_au(converged) * AU_DIPOLE_TO_DEBYE


def warn_once(key: str, msg: str) -> None:
    """Emit a backend warning once per key."""

    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    import warnings

    warnings.warn(msg, RuntimeWarning, stacklevel=2)


def electronic_symbol(symbol: str) -> str:
    """Return the electronic-structure symbol, mapping deuterium to hydrogen."""

    sym = str(symbol)
    return "H" if sym.upper() == "D" else sym


def is_available() -> bool:
    """Return whether the core PySCF backend imports successfully."""

    try:
        require_pyscf()
    except BackendUnavailableError:
        return False
    return True


def molecule_to_pyscf(molecule: Any, basis: str = "aug-cc-pVTZ"):
    """Build a PySCF ``gto.Mole`` from a ``pyscf_vscf.molecule.Molecule``.

    Deuterium is mapped to hydrogen for electronic structure, while isotope
    masses are passed to PySCF through ``mol.nucprop`` for vibrational analysis.
    When ``PYSCF_VSCF_MAX_MEMORY_MB`` is set, it becomes the molecule's
    ``max_memory`` budget, overriding PySCF's configured default.
    """

    gto, _, _, elements = require_pyscf()
    symbols = list(getattr(molecule, "symbols"))
    coords = getattr(molecule, "coords")
    sym_elec = [electronic_symbol(s) for s in symbols]

    pmol = gto.Mole()
    pmol.unit = "Angstrom"
    pmol.basis = basis
    pmol.charge = int(getattr(molecule, "charge", 0))
    pmol.spin = int(getattr(molecule, "spin", 0))
    pmol.verbose = 0
    pmol.atom = [(s, tuple(map(float, xyz))) for s, xyz in zip(sym_elec, coords)]
    memory_budget = runtime_max_memory_mb()
    if memory_budget is not None:
        pmol.max_memory = memory_budget

    default_masses = _default_pyscf_masses(sym_elec, gto, elements)
    analysis_masses = _analysis_masses(molecule, symbols)
    nucprop = {}
    for i, (default_mass, analysis_mass) in enumerate(
        zip(default_masses, analysis_masses), start=1
    ):
        if abs(float(analysis_mass) - float(default_mass)) > 1e-3:
            nucprop[i] = {"mass": float(analysis_mass)}
    if nucprop:
        pmol.nucprop = nucprop

    pmol.build()
    return pmol


def make_mean_field(
    pmol: Any,
    cfg: Any,
    *,
    field_au: np.ndarray | None = None,
    field_origin_A: np.ndarray | None = None,
):
    """Deprecated: create and run the built-in mean field on an existing molecule.

    Use :func:`converged_mean_field` or :class:`PySCFMeanFieldProvider`.
    """

    deprecated(
        "make_mean_field() is deprecated; use converged_mean_field() or PySCFMeanFieldProvider."
    )
    settings = coerce_es_settings(cfg)
    mean_field = configure_settings_mean_field(pmol, settings)
    build = _legacy_build(pmol, mean_field, SettingsFactory(settings))
    converge(build, field_au=field_au, field_origin_A=field_origin_A)
    return mean_field


def energy_gradient_at_coords_bohr(
    molecule: Any,
    cfg: Any,
    xflat_bohr: np.ndarray,
    *,
    guard: ConfigurationGuard | None = None,
) -> tuple[float, np.ndarray]:
    """Return single-point energy and flattened nuclear gradient at Bohr coordinates."""

    coords = _coords_from_flat_bohr(molecule, xflat_bohr)
    converged = converged_mean_field(_molecule_with_coords(molecule, coords), cfg)
    if guard is not None:
        guard.check(converged.build.configuration_snapshot)
    return converged.total_energy_Eh, nuclear_gradient(converged).reshape(-1)


def normal_relaxed_point(
    molecule: Any,
    cfg: Any,
    u_dir: np.ndarray,
    s: float,
    gtol: float,
    maxiter: int,
    *,
    strict: bool = True,
) -> NormalRelaxedPointResult:
    """Minimize all coordinates at an exact mass-metric normal displacement.

    With a Euclidean-normalized Cartesian direction ``u`` and diagonal mass
    matrix ``M``, the constrained coordinate is
    ``a @ (x - x0) = s``, where ``a = M u / (u.T M u)``.  The affine
    parameterization ``x = x0 + s*u + Q*y`` enforces that coordinate by
    construction; columns of ``Q`` span the null space of ``a`` and therefore
    describe displacements that are mass-orthogonal to ``u``.
    """

    from scipy import linalg, optimize

    coords = _molecule_coords(molecule)
    direction = np.asarray(u_dir, dtype=float)
    if direction.shape != coords.shape:
        raise ValueError(f"u_dir shape {direction.shape} does not match molecule coordinates")

    iterations = int(maxiter)
    if iterations <= 0:
        raise ValueError("normal-relaxed maxiter must be positive")
    gradient_tolerance = float(gtol)
    if not np.isfinite(gradient_tolerance) or gradient_tolerance <= 0.0:
        raise ValueError("normal-relaxed gtol must be finite and positive")

    x0_bohr = (coords * ANG_TO_BOHR).reshape(-1)
    u_flat = direction.reshape(-1)
    u_norm = float(np.linalg.norm(u_flat))
    if not np.isfinite(u_norm) or u_norm < 1e-14:
        raise ValueError("u_dir must have a finite non-zero norm")
    u_flat = u_flat / u_norm
    symbols = [str(symbol) for symbol in getattr(molecule, "symbols")]
    masses_amu = np.asarray(_analysis_masses(molecule, symbols), dtype=float)
    if masses_amu.shape != (coords.shape[0],):
        raise ValueError("molecule must provide one analysis mass per atom")
    if not np.all(np.isfinite(masses_amu)) or np.any(masses_amu <= 0.0):
        raise ValueError("analysis masses must be finite and positive")
    mass_flat = np.repeat(masses_amu, 3)
    effective_mass_amu = float(np.dot(u_flat, mass_flat * u_flat))
    constraint_covector = mass_flat * u_flat / effective_mass_amu
    s_bohr = float(s) * ANG_TO_BOHR
    orthogonal_basis = linalg.null_space(constraint_covector.reshape(1, -1))
    constrained_origin = x0_bohr + s_bohr * u_flat
    # Every build of this point (optimizer steps and the final evaluation) must
    # share one method configuration; the scan compares the points' snapshots.
    guard = ConfigurationGuard(cfg)

    def objective_and_gradient(y: np.ndarray) -> tuple[float, np.ndarray]:
        xflat_bohr = constrained_origin + orthogonal_basis @ y
        energy, gradient = energy_gradient_at_coords_bohr(molecule, cfg, xflat_bohr, guard=guard)
        return energy, orthogonal_basis.T @ gradient

    result = optimize.minimize(
        objective_and_gradient,
        np.zeros(orthogonal_basis.shape[1], dtype=float),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": iterations, "gtol": gradient_tolerance},
    )
    if not result.success:
        message = f"normal-relaxed optimizer did not converge: {result.message}"
        if bool(strict):
            raise RuntimeError(message)
        warn_once("normal_relaxed_optimizer", message)

    final_x_bohr = constrained_origin + orthogonal_basis @ np.asarray(result.x, dtype=float)
    final_coords = _coords_from_flat_bohr(molecule, final_x_bohr)
    achieved_bohr = float(np.dot(constraint_covector, final_x_bohr - x0_bohr))
    residual_A = (achieved_bohr - s_bohr) / ANG_TO_BOHR
    sub = _molecule_with_coords(molecule, final_coords)
    energy, dipole = legacy_energy_dipole(sub, cfg, guard=guard)
    snapshot = guard.reference_snapshot
    return NormalRelaxedPointResult(
        energy_hartree=float(energy),
        dipole_debye=np.asarray(dipole, dtype=float),
        coords_A=final_coords,
        requested_displacement_A=float(s),
        achieved_displacement_A=achieved_bohr / ANG_TO_BOHR,
        constraint_residual_A=residual_A,
        converged=bool(result.success),
        n_iterations=int(result.nit),
        message=str(result.message),
        configuration_snapshot=None if snapshot is None else to_jsonable(snapshot),
    )


def _legacy_build(molecule: Any, mean_field: Any, factory: MeanFieldFactory):
    from .._identity import immutable_json_mapping as _freeze
    from ._pyscf_factory import MeanFieldBuild, configuration_snapshot

    return MeanFieldBuild(
        mean_field=mean_field,
        molecule=molecule,
        method_identity=_freeze(factory.scientific_settings_payload()),
        configuration_snapshot=_freeze(configuration_snapshot(mean_field)),
    )


def _molecule_coords(molecule: Any) -> np.ndarray:
    coords = np.asarray(getattr(molecule, "coords"), dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("molecule coordinates must have shape (n_atoms, 3)")
    return coords


def _coords_from_flat_bohr(molecule: Any, xflat_bohr: np.ndarray) -> np.ndarray:
    coords = np.asarray(xflat_bohr, dtype=float).reshape(_molecule_coords(molecule).shape)
    return coords / ANG_TO_BOHR


def _molecule_with_coords(molecule: Any, coords: np.ndarray) -> Molecule:
    masses_fn = getattr(molecule, "analysis_masses", None)
    masses = masses_fn() if callable(masses_fn) else getattr(molecule, "masses", None)
    return Molecule.from_arrays(
        [str(symbol) for symbol in getattr(molecule, "symbols")],
        np.asarray(coords, dtype=float),
        charge=int(getattr(molecule, "charge", 0)),
        spin=int(getattr(molecule, "spin", 0)),
        label=str(getattr(molecule, "label", "mol")),
        masses_amu=masses,
    )


def _analysis_masses(molecule: Any, symbols: list[str]) -> list[float]:
    analysis_masses = getattr(molecule, "analysis_masses", None)
    if callable(analysis_masses):
        return [float(mass) for mass in analysis_masses()]

    masses = getattr(molecule, "masses", None)
    if masses is not None:
        return [float(mass) for mass in masses]

    return [atomic_mass_amu(symbol) for symbol in symbols]


def _default_pyscf_masses(symbols: list[str], gto: Any, elements: Any) -> list[float]:
    result = []
    for symbol in symbols:
        try:
            z = gto.mole.charge(symbol)
            result.append(float(elements.MASSES[z]))
        except Exception as exc:
            key = str(symbol).upper()
            if key in MASS_AMU:
                warn_once(
                    "fallback_mass_lookup",
                    f"Falling back to internal mass table for symbol '{symbol}' ({exc})",
                )
                result.append(atomic_mass_amu(symbol))
            else:
                raise ValueError(
                    f"Unknown element symbol '{symbol}' (failed to resolve atomic number)"
                ) from exc
    return result


__all__ = [
    "BackendUnavailableError",
    "ESSettings",
    "ElectronicSystem",
    "MeanFieldFactory",
    "MethodRecipe",
    "NormalRelaxedPointResult",
    "PySCFMeanFieldProvider",
    "PySCFResources",
    "SettingsFactory",
    "converged_mean_field",
    "default_auxbasis",
    "electronic_symbol",
    "energy_gradient_at_coords_bohr",
    "is_available",
    "legacy_energy_dipole",
    "make_mean_field",
    "method_factory",
    "molecule_to_pyscf",
    "normal_relaxed_point",
    "warn_once",
]
