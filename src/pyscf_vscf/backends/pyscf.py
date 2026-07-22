"""PySCF backend helpers.

This module intentionally does not import PySCF at module import time. Callers
can import ``pyscf_vscf.backends.pyscf`` in environments without PySCF; functions
that need the backend raise :class:`BackendUnavailableError` when it is missing.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constants import ANG_TO_BOHR, MASS_AMU, atomic_mass_amu
from ..molecule import Molecule
from ..settings import ESSettings, default_auxbasis, normalize_dispersion


STRICT: bool = True
DEV_FAST: bool = False

_WARNED_ONCE: set[str] = set()


class BackendUnavailableError(ImportError):
    """Raised when the optional PySCF backend dependency is unavailable."""


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


def warn_once(key: str, msg: str) -> None:
    """Emit a backend warning once, matching the legacy helper's behavior."""

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


def make_mean_field(pmol: Any, cfg: Any):
    """Create and run a PySCF mean-field object from an ESSettings-like config."""

    _, scf, dft, _ = _require_pyscf()
    method = str(_cfg_get(cfg, "method", "wb97x"))
    open_shell = int(getattr(pmol, "spin", 0)) != 0
    is_dft = method.lower() != "hf"
    if method.lower() == "hf":
        mf = scf.UHF(pmol) if open_shell else scf.RHF(pmol)
    else:
        mf = dft.UKS(pmol) if open_shell else dft.RKS(pmol)
        mf.xc = method

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

    dispersion = normalize_dispersion(_cfg_get(cfg, "dispersion", "d4"))
    if dispersion:
        _require_pyscf_dispersion(dispersion)
        mf.disp = dispersion

    dft_grid_level = _cfg_get(cfg, "dft_grid_level", None)
    if dft_grid_level is not None and is_dft:
        mf.grids.level = int(dft_grid_level)

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
        if bool(_cfg_get(cfg, "strict", STRICT)):
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
            "PySCF backend is unavailable. Install the 'pyscf' optional dependencies "
            "to use this backend."
        ) from exc
    return gto, scf, dft, elements


def _require_pyscf_dispersion(dispersion: Any) -> None:
    try:
        import pyscf.dispersion  # noqa: F401
    except Exception as exc:
        raise BackendUnavailableError(
            f"Dispersion was requested (cfg.dispersion={dispersion!r}) but "
            "pyscf.dispersion is unavailable. Install pyscf-dispersion or pass "
            "--dispersion none."
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


__all__ = [
    "BackendUnavailableError",
    "DEV_FAST",
    "ESSettings",
    "NormalRelaxedPointResult",
    "STRICT",
    "default_auxbasis",
    "electronic_symbol",
    "energy_gradient_at_coords_bohr",
    "is_available",
    "make_mean_field",
    "molecule_to_pyscf",
    "normal_relaxed_point",
    "warn_once",
]
