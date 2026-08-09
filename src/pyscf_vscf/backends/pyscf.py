"""PySCF backend helpers.

This module intentionally does not import PySCF at module import time. Callers
can import ``pyscf_vscf.backends.pyscf`` in environments without PySCF; functions
that need the backend raise :class:`BackendUnavailableError` when it is missing.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from socket import gethostname
from typing import Any

import numpy as np

from ..constants import ANG_TO_BOHR, MASS_AMU, atomic_mass_amu
from ..cache import runtime_provenance
from ..electronic import (
    ElectronicPointRequest,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from .._identity import immutable_json_mapping, to_jsonable
from ..molecule import Molecule
from ..settings import ESSettings, coerce_es_settings, default_auxbasis, normalize_dispersion


_WARNED_ONCE: set[str] = set()


class BackendUnavailableError(ImportError):
    """Raised when the required PySCF backend dependency is unavailable."""


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


@dataclass(frozen=True)
class _MeanFieldSettings:
    method: str
    basis: str
    use_density_fit: bool
    auxbasis: str | None
    dispersion: str | None
    scf_conv_tol: float
    scf_max_cycle: int
    dft_grid_level: int | None

    @classmethod
    def from_value(cls, value: object) -> _MeanFieldSettings:
        settings = coerce_es_settings(value)
        method = str(settings.method).strip()
        basis = str(settings.basis).strip()
        if not method or not basis:
            raise ValueError("Electronic method and basis must be non-empty")
        use_density_fit = bool(settings.use_density_fit)
        auxbasis = None
        if use_density_fit:
            auxbasis = settings.auxbasis or default_auxbasis(basis)
            auxbasis = str(auxbasis).strip()
            if not auxbasis:
                raise ValueError("Density fitting requires a non-empty auxiliary basis")
        dispersion = normalize_dispersion(settings.dispersion)
        tolerance = 1e-10 if settings.scf_conv_tol is None else float(settings.scf_conv_tol)
        max_cycle = 50 if settings.scf_max_cycle is None else int(settings.scf_max_cycle)
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("scf_conv_tol must be finite and positive")
        if max_cycle <= 0:
            raise ValueError("scf_max_cycle must be positive")
        grid_level = settings.dft_grid_level
        if method.lower() == "hf":
            grid_level = None
        elif grid_level is None:
            grid_level = 3
        if grid_level is not None and int(grid_level) < 0:
            raise ValueError("dft_grid_level must be non-negative")
        return cls(
            method=method,
            basis=basis,
            use_density_fit=use_density_fit,
            auxbasis=auxbasis,
            dispersion=dispersion,
            scf_conv_tol=tolerance,
            scf_max_cycle=max_cycle,
            dft_grid_level=None if grid_level is None else int(grid_level),
        )


@dataclass(frozen=True)
class PySCFMeanFieldProvider:
    """Mean-field electronic points with split scientific and execution identity."""

    settings: object
    threads: int = 1
    max_memory_mb: int = 4000
    user_annotations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "settings", _MeanFieldSettings.from_value(self.settings))
        threads = int(self.threads)
        memory = int(self.max_memory_mb)
        if threads <= 0 or memory <= 0:
            raise ValueError("threads and max_memory_mb must be positive")
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "max_memory_mb", memory)
        object.__setattr__(
            self,
            "user_annotations",
            immutable_json_mapping(self.user_annotations),
        )

    def scientific_settings_payload(self) -> Mapping[str, object]:
        """Return every setting that can change the intended calculation."""

        return {
            "schema": "pyscf-vscf-provider-settings",
            "schema_version": 1,
            "backend_family": "pyscf",
            "provider": "mean-field",
            "reference_policy": "restricted-if-spin-zero-otherwise-unrestricted",
            "settings": asdict(self.settings),
            "frozen_core_policy": "not-applicable",
            "orbital_policy": "self-consistent-canonical",
            "post_scf_tolerance": None,
            "integral_policy": "pyscf-default-in-core-or-direct",
            "field_convention": {
                "field_units": "atomic_unit",
                "origin_units": "angstrom",
                "hamiltonian": "H(F)=H(0)-F.mu",
                "nuclear_term": "included-in-total-energy",
            },
        }

    def execution_provenance(self) -> Mapping[str, object]:
        """Return runtime resources and software versions outside causal identity."""

        return {
            "threads_requested": self.threads,
            "max_memory_mb": self.max_memory_mb,
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
        actual_threads = _configure_pyscf_threads(self.threads)
        _, _, _, elements = _require_pyscf()
        symbols = [elements._symbol(charge) for charge in request.nuclear_charges]
        molecule = Molecule.from_arrays(
            symbols,
            request.coordinates_A,
            charge=request.charge,
            spin=request.spin,
            label="electronic-point",
        )
        pmol = molecule_to_pyscf(molecule, self.settings.basis)
        pmol.max_memory = self.max_memory_mb
        mf = make_mean_field(
            pmol,
            self.settings,
            field_au=request.field_au,
            field_origin_A=request.field_origin_A,
        )
        dipole = None
        if "dipole" in request.requested_properties:
            density = mf.make_rdm1()
            try:
                dipole = mf.dip_moment(dm=density, unit="au", verbose=0)
            except TypeError:
                dipole = mf.dip_moment(unit="au", verbose=0)
        provider_id = provider_scientific_fingerprint(self)
        nuclear_field_energy = float(getattr(mf, "_pyscf_vscf_nuclear_field_energy_Eh", 0.0))
        field_vector = np.zeros(3) if request.field_au is None else request.field_au
        field_origin = np.zeros(3) if request.field_origin_A is None else request.field_origin_A
        runtime_seconds = time.perf_counter() - started
        return ElectronicResult(
            total_energy_Eh=float(mf.e_tot) + nuclear_field_energy,
            dipole_au=None if dipole is None else np.asarray(dipole, dtype=float),
            dipole_unit=None if dipole is None else "atomic_unit",
            dipole_frame=None if dipole is None else "input_cartesian",
            converged=bool(mf.converged),
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
            scientific_diagnostics={
                "field_au": np.asarray(field_vector, dtype=float).tolist(),
                "field_origin_A": np.asarray(field_origin, dtype=float).tolist(),
                "field_hamiltonian": "H(F)=H(0)-F.mu",
                "nuclear_field_energy_Eh": nuclear_field_energy,
                "nuclear_field_term_included": True,
            },
            execution_diagnostics={
                "reference_class": type(mf).__name__,
                "scf_cycles": _optional_int(getattr(mf, "cycles", None)),
                "runtime_seconds": runtime_seconds,
                "max_rss_mb": _max_rss_mb(),
                "threads_actual": actual_threads,
                "max_memory_mb": self.max_memory_mb,
                "warnings": [],
            },
            provenance=self.execution_provenance(),
        )


def warn_once(key: str, msg: str) -> None:
    """Emit a backend warning once."""

    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def electronic_symbol(symbol: str) -> str:
    """Return the electronic-structure symbol, mapping deuterium to hydrogen."""

    sym = str(symbol)
    return "H" if sym.upper() == "D" else sym


def is_available() -> bool:
    """Return whether the core PySCF backend imports successfully."""

    try:
        _require_pyscf()
    except BackendUnavailableError:
        return False
    return True


def molecule_to_pyscf(molecule: Any, basis: str = "aug-cc-pVTZ"):
    """Build a PySCF ``gto.Mole`` from a ``pyscf_vscf.molecule.Molecule``.

    Deuterium is mapped to hydrogen for electronic structure, while isotope
    masses are passed to PySCF through ``mol.nucprop`` for vibrational analysis.
    """

    gto, _, _, elements = _require_pyscf()
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
    """Create and run a PySCF mean-field object from an ESSettings-like config."""

    _, scf, dft, _ = _require_pyscf()
    method = str(_cfg_get(cfg, "method", "wb97x"))
    open_shell = int(getattr(pmol, "spin", 0)) != 0
    is_dft = method.lower() != "hf"
    if method.lower() == "hf":
        mf = scf.UHF(pmol) if open_shell else scf.RHF(pmol)
    else:
        mf = dft.UKS(pmol, xc=method) if open_shell else dft.RKS(pmol, xc=method)

    if bool(_cfg_get(cfg, "use_density_fit", True)):
        auxbasis = _cfg_get(cfg, "auxbasis", None)
        aux = auxbasis if auxbasis else default_auxbasis(str(_cfg_get(cfg, "basis", "")))
        try:
            mf = mf.density_fit(auxbasis=aux)
        except Exception as exc:
            msg = (
                f"RI density_fit(auxbasis='{aux}') failed. Refusing an implicit "
                f"auxiliary-basis substitution because it would change the recorded "
                f"electronic-structure model ({exc})"
            )
            raise RuntimeError(msg) from exc

    dispersion = normalize_dispersion(_cfg_get(cfg, "dispersion", None))
    if dispersion is not None:
        _require_pyscf_dispersion(dispersion)
        mf.disp = dispersion

    dft_grid_level = _cfg_get(cfg, "dft_grid_level", None)
    if dft_grid_level is not None and is_dft:
        mf.grids.level = int(dft_grid_level)

    nuclear_field_energy = _apply_electric_field(
        mf,
        pmol,
        np.zeros(3) if field_au is None else field_au,
        np.zeros(3) if field_origin_A is None else field_origin_A,
    )
    mf._pyscf_vscf_nuclear_field_energy_Eh = nuclear_field_energy

    mf.conv_tol = 1e-10

    scf_conv_tol = _cfg_get(cfg, "scf_conv_tol", None)
    if scf_conv_tol is not None:
        mf.conv_tol = float(scf_conv_tol)
    scf_max_cycle = _cfg_get(cfg, "scf_max_cycle", None)
    if scf_max_cycle is not None:
        max_cycle = int(scf_max_cycle)
        if max_cycle <= 0:
            raise ValueError("scf_max_cycle must be positive")
        mf.max_cycle = max_cycle
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf


def _apply_electric_field(
    mean_field: Any,
    molecule: Any,
    field_au: np.ndarray,
    origin_A: np.ndarray,
) -> float:
    """Apply ``H(F)=H(0)-F.mu`` and return the nuclear-field energy."""

    field_vector = np.asarray(field_au, dtype=float)
    origin = np.asarray(origin_A, dtype=float)
    if field_vector.shape != (3,) or origin.shape != (3,):
        raise ValueError("Electric field and origin must be three-component vectors")
    if not np.all(np.isfinite(field_vector)) or not np.all(np.isfinite(origin)):
        raise ValueError("Electric field and origin must be finite")
    if not np.any(field_vector):
        return 0.0
    origin_bohr = origin * ANG_TO_BOHR
    with molecule.with_common_orig(origin_bohr):
        position_ao = molecule.intor_symmetric("int1e_r", comp=3)
    field_hcore = np.einsum("x,xij->ij", field_vector, position_ao, optimize=True)
    base_get_hcore = mean_field.get_hcore

    def get_hcore(mol=None):
        return base_get_hcore(mol) + field_hcore

    mean_field.get_hcore = get_hcore
    nuclear_dipole = np.einsum(
        "i,ix->x",
        molecule.atom_charges(),
        molecule.atom_coords() - origin_bohr[None, :],
        optimize=True,
    )
    return -float(field_vector @ nuclear_dipole)


def mean_field_dispersion(mf: Any) -> str | None:
    """Return PySCF's effective D3/D4 label without loading its extension."""

    explicit = getattr(mf, "disp", None)
    method = getattr(mf, "xc", None)
    method_text = "" if method is None else str(method).lower()
    if explicit in (None, False, 0, "") and "-d3" not in method_text and "-d4" not in method_text:
        return None

    from pyscf.scf.dispersion import check_disp

    detected = check_disp(mf)
    return None if not detected else str(detected)


def energy_gradient_at_coords_bohr(
    molecule: Any,
    cfg: Any,
    xflat_bohr: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Return single-point energy and flattened nuclear gradient at Bohr coordinates."""

    coords = _coords_from_flat_bohr(molecule, xflat_bohr)
    sub = _molecule_with_coords(molecule, coords)
    pmol = molecule_to_pyscf(sub, str(_cfg_get(cfg, "basis", "aug-cc-pVTZ")))
    mf = make_mean_field(pmol, cfg)
    gradient = _nuclear_gradient(mf, warn_key="grad_api_fallback_cfg")
    return float(mf.e_tot), gradient.reshape(-1)


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

    def objective_and_gradient(y: np.ndarray) -> tuple[float, np.ndarray]:
        xflat_bohr = constrained_origin + orthogonal_basis @ y
        energy, gradient = energy_gradient_at_coords_bohr(molecule, cfg, xflat_bohr)
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
    from pyscf_vscf.surfaces import energy_dipole

    energy, dipole = energy_dipole(sub, cfg)
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
    )


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _nuclear_gradient(mf: Any, *, warn_key: str) -> np.ndarray:
    try:
        gradient = mf.nuc_grad_method().kernel()
    except AttributeError:
        warn_once(
            warn_key,
            "Falling back to mf.Gradients().kernel() (mf.nuc_grad_method unavailable)",
        )
        gradient = mf.Gradients().kernel()
    return np.asarray(gradient, dtype=float)


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


def _require_pyscf():
    try:
        from pyscf import dft, gto, scf
        from pyscf.data import elements
    except Exception as exc:
        raise BackendUnavailableError(
            "PySCF is unavailable. Reinstall pyscf-vscf and its required dependencies."
        ) from exc
    return gto, scf, dft, elements


def _require_pyscf_dispersion(dispersion: str) -> None:
    try:
        import pyscf.dispersion  # noqa: F401
    except Exception as exc:
        raise BackendUnavailableError(
            f"Dispersion was requested ({dispersion!r}) but pyscf.dispersion is unavailable"
        ) from exc


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


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _configure_pyscf_threads(threads: int) -> int:
    try:
        from pyscf import lib
    except Exception as exc:
        raise BackendUnavailableError(
            "PySCF is unavailable. Reinstall pyscf-vscf and its required dependencies."
        ) from exc
    lib.num_threads(int(threads))
    return int(lib.num_threads())


def _max_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return maximum / (1024.0 * 1024.0)
    return maximum / 1024.0


__all__ = [
    "BackendUnavailableError",
    "ESSettings",
    "NormalRelaxedPointResult",
    "PySCFMeanFieldProvider",
    "default_auxbasis",
    "electronic_symbol",
    "energy_gradient_at_coords_bohr",
    "is_available",
    "make_mean_field",
    "mean_field_dispersion",
    "molecule_to_pyscf",
    "normal_relaxed_point",
    "warn_once",
]
