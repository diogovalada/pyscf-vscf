"""PySCF-backed harmonic workflow.

This module contains the package-level orchestration extracted from the legacy
``pyscf_pme_pipeline.py`` script. It remains import-light: PySCF is only imported
by backend calls or by the analytic Hessian fallback path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from pyscf_vscf import harmonic as harmonic_helpers
from pyscf_vscf.backends import pyscf as pyscf_backend
from pyscf_vscf.constants import ANG_TO_BOHR, atomic_mass_amu
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.settings import (
    DEFAULT_ALLOW_FD_HESSIAN,
    DEFAULT_STRICT,
    ESSettings,
    coerce_es_settings,
)


_DEFAULT_ES = ESSettings()
BOHR_TO_ANG = 1.0 / ANG_TO_BOHR


@dataclass(frozen=True)
class StationarityDiagnostic:
    """Gradient summary for checking whether a geometry is stationary."""

    gradient: np.ndarray
    max_component: float
    rms_component: float
    max_atom: float
    rms_atom: float


def harmonic_analysis(
    molecule: Any,
    cfg: Any,
    *,
    rtproj: str = "pyscf",
    debug: bool = False,
) -> harmonic_helpers.HarmonicResult:
    """Run the legacy-compatible PySCF harmonic workflow.

    Analytic Hessians are preferred. If no analytic Hessian is available, finite
    differences of analytic gradients are only used when ``cfg.allow_fd_hessian``
    is truthy. Analytic Hessian failures raise in strict mode and warn/fall back
    in non-strict mode, matching the legacy driver policy.
    """

    strict = bool(_cfg_get(cfg, "strict", DEFAULT_STRICT))
    basis = str(_cfg_get(cfg, "basis", _DEFAULT_ES.basis))
    allow_fd_hessian = bool(_cfg_get(cfg, "allow_fd_hessian", DEFAULT_ALLOW_FD_HESSIAN))
    dispersion = _cfg_get(cfg, "dispersion", _DEFAULT_ES.dispersion)
    has_dispersion = dispersion is not None and str(dispersion).strip().lower() != "none"

    pmol = pyscf_backend.molecule_to_pyscf(molecule, basis=basis)
    mf = pyscf_backend.make_mean_field(pmol, cfg)

    if debug:
        _print_stationarity_diagnostic(mf)

    hessian_provenance = "analytic"
    if has_dispersion:
        msg = (
            f"{dispersion} Hessians include a numerically differentiated dispersion "
            "component and are therefore semi-numerical"
        )
        if not allow_fd_hessian:
            raise RuntimeError(msg + " (blocked; pass --allow-fd-hessian to proceed)")
        harmonic_helpers.warn_once(
            "semi_numerical_dispersion_hessian",
            msg + " (enabled by --allow-fd-hessian)",
        )
        hessian_provenance = "analytic-electronic+finite-difference-dispersion"

    hessian = None
    try:
        hessian = analytic_hessian(mf)
    except Exception as exc:
        msg = f"Analytic Hessian failed: {exc}"
        if strict:
            raise
        harmonic_helpers.warn_once("analytic_hessian_failed", msg)
        hessian = None

    if hessian is None:
        msg = "Analytic Hessian unavailable; would fall back to finite-difference Hessian"
        if not allow_fd_hessian:
            raise RuntimeError(msg + " (blocked; pass --allow-fd-hessian to proceed)")
        harmonic_helpers.warn_once("fd_hessian", msg + " (enabled by --allow-fd-hessian)")
        x0_bohr = atom_coords_bohr(pmol).reshape(-1)
        hessian = finite_difference_hessian_from_gradients(molecule, cfg, x0_bohr)
        hessian_provenance = "finite-difference-analytic-gradients"

    freqs_cm, modes = harmonic_helpers.mass_weighted_freqs_modes(
        pmol,
        hessian,
        molecule_analysis_masses(molecule),
        rtproj=rtproj,
        strict=strict,
        debug=debug,
    )
    return harmonic_helpers.HarmonicResult(
        freqs_cm=freqs_cm,
        modes=modes,
        zpe_cm=harmonic_helpers.zpe_cm_from_freqs(freqs_cm),
        hessian_provenance=hessian_provenance,
    )


def analytic_hessian(mf: Any) -> np.ndarray | None:
    """Return an analytic Hessian from a converged PySCF mean-field object.

    Missing APIs return ``None`` so the caller can decide whether finite
    differences are allowed. Runtime failures are wrapped as
    ``RuntimeError("Analytic Hessian computation failed")`` to preserve legacy
    behavior.
    """

    try:
        return mf.Hessian().kernel()
    except AttributeError:
        pass
    except Exception as exc:
        raise RuntimeError("Analytic Hessian computation failed") from exc

    try:
        from pyscf import dft, scf
        from pyscf.hessian import rhf as h_rhf
        from pyscf.hessian import rks as h_rks
    except Exception:
        return None

    try:
        if isinstance(mf, scf.hf.RHF):
            return h_rhf.Hessian(mf).kernel()
        if isinstance(mf, dft.rks.RKS):
            return h_rks.Hessian(mf).kernel()
    except AttributeError:
        return None
    except Exception as exc:
        raise RuntimeError("Analytic Hessian computation failed") from exc
    return None


def nuclear_gradient(
    mf: Any,
    *,
    warn_fn: Callable[[str, str], None] | None = None,
    warn_key: str = "grad_api_fallback",
) -> np.ndarray:
    """Return a PySCF nuclear gradient using the version-agnostic API first."""

    emit_warning = warn_fn or harmonic_helpers.warn_once
    try:
        gradient = mf.nuc_grad_method().kernel()
    except AttributeError:
        emit_warning(
            warn_key,
            "Falling back to mf.Gradients().kernel() (mf.nuc_grad_method unavailable)",
        )
        gradient = mf.Gradients().kernel()
    return np.asarray(gradient, dtype=float)


def stationarity_diagnostic(mf: Any) -> StationarityDiagnostic:
    """Return gradient norms used by the legacy debug stationarity check."""

    gradient = nuclear_gradient(
        mf,
        warn_key="grad_api_fallback_stationarity",
    )
    gradient = np.asarray(gradient, dtype=float)
    g_flat = gradient.reshape(-1)
    max_component = float(np.max(np.abs(g_flat))) if g_flat.size else 0.0
    rms_component = float(np.sqrt(np.mean(g_flat * g_flat))) if g_flat.size else 0.0
    g_atom = gradient.reshape(-1, 3)
    atom_norms = np.linalg.norm(g_atom, axis=1)
    max_atom = float(np.max(atom_norms)) if atom_norms.size else 0.0
    rms_atom = float(np.sqrt(np.mean(atom_norms * atom_norms))) if atom_norms.size else 0.0
    return StationarityDiagnostic(
        gradient=gradient,
        max_component=max_component,
        rms_component=rms_component,
        max_atom=max_atom,
        rms_atom=rms_atom,
    )


def format_stationarity_diagnostic(diagnostic: StationarityDiagnostic) -> str:
    """Format the stationarity diagnostic with legacy text and thresholds."""

    return "\n".join(
        [
            "Geometry stationarity (gradient) in Eh/Bohr:",
            "  "
            f"max|g_comp|={diagnostic.max_component:.3e}  "
            f"rms|g_comp|={diagnostic.rms_component:.3e}  "
            f"max|g_atom|={diagnostic.max_atom:.3e}  "
            f"rms|g_atom|={diagnostic.rms_atom:.3e}",
            "  Heuristic: max|g_comp| <= 1e-4 (good), "
            "1e-4-5e-4 (maybe), >5e-4 (likely non-stationary)",
        ]
    )


def gradient_at(molecule: Any, cfg: Any, xflat_bohr: np.ndarray) -> np.ndarray:
    """Return the flattened nuclear gradient at displaced Bohr coordinates."""

    coords_ang = np.asarray(xflat_bohr, dtype=float).reshape(-1, 3) * BOHR_TO_ANG
    masses_fn = getattr(molecule, "analysis_masses", None)
    masses = masses_fn() if callable(masses_fn) else getattr(molecule, "masses", None)
    displaced = Molecule.from_arrays(
        list(getattr(molecule, "symbols")),
        coords_ang,
        charge=int(getattr(molecule, "charge", 0)),
        spin=int(getattr(molecule, "spin", 0)),
        label=str(getattr(molecule, "label", "mol")),
        masses_amu=masses,
    )
    fd_cfg = _legacy_fd_gradient_settings(cfg)
    pmol = pyscf_backend.molecule_to_pyscf(displaced, basis=fd_cfg.basis)
    mf = pyscf_backend.make_mean_field(pmol, fd_cfg)
    return nuclear_gradient(mf).reshape(-1)


def finite_difference_hessian_from_gradients(
    molecule: Any,
    cfg: Any,
    x0_bohr: np.ndarray,
    h: float = 2e-3,
    *,
    gradient_fn: Callable[[Any, Any, np.ndarray], np.ndarray] | None = None,
    progress_fn: Callable[[int, int, str], None] | None = None,
) -> np.ndarray:
    """Build a central-difference Hessian from flattened gradients."""

    x0 = np.asarray(x0_bohr, dtype=float).reshape(-1)
    step = float(h)
    if step <= 0.0:
        raise ValueError("h must be positive")

    gradient = gradient_fn or gradient_at
    size = x0.size
    hessian = np.zeros((size, size), dtype=float)
    for column in range(size):
        direction = np.zeros_like(x0)
        direction[column] = 1.0
        g_plus = np.asarray(gradient(molecule, cfg, x0 + step * direction), dtype=float).reshape(
            -1
        )
        g_minus = np.asarray(gradient(molecule, cfg, x0 - step * direction), dtype=float).reshape(
            -1
        )
        if g_plus.shape != (size,) or g_minus.shape != (size,):
            raise ValueError("Gradient shape is inconsistent with x0_bohr")
        hessian[:, column] = (g_plus - g_minus) / (2.0 * step)
        if progress_fn is not None:
            progress_fn(column + 1, size, "Hessian columns")
    return hessian


def atom_coords_bohr(pmol: Any) -> np.ndarray:
    """Return PySCF molecule coordinates in Bohr."""

    atom_coords = getattr(pmol, "atom_coords")
    try:
        coords = atom_coords(unit="Bohr")
    except TypeError:
        coords = atom_coords()
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("pmol.atom_coords() returned an unexpected shape")
    return coords


def molecule_analysis_masses(molecule: Any) -> np.ndarray:
    """Return isotope masses in amu for harmonic analysis."""

    analysis_masses = getattr(molecule, "analysis_masses", None)
    if callable(analysis_masses):
        return np.asarray(analysis_masses(), dtype=float)

    masses = getattr(molecule, "masses", None)
    if masses is not None:
        return np.asarray(masses, dtype=float)

    try:
        symbols = list(getattr(molecule, "symbols"))
    except Exception as exc:
        raise ValueError("molecule must expose analysis masses, masses, or symbols") from exc
    return np.asarray([atomic_mass_amu(symbol) for symbol in symbols], dtype=float)


def _print_stationarity_diagnostic(mf: Any) -> None:
    try:
        print(format_stationarity_diagnostic(stationarity_diagnostic(mf)))
    except Exception as exc:
        print(f"Geometry stationarity check (gradient) failed: {exc}")


def _legacy_fd_gradient_settings(cfg: Any) -> ESSettings:
    return coerce_es_settings(cfg)


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


_try_analytic_hessian = analytic_hessian
_grad_at = gradient_at
_num_hessian_from_gradients = finite_difference_hessian_from_gradients


__all__ = [
    "StationarityDiagnostic",
    "analytic_hessian",
    "atom_coords_bohr",
    "finite_difference_hessian_from_gradients",
    "format_stationarity_diagnostic",
    "gradient_at",
    "harmonic_analysis",
    "molecule_analysis_masses",
    "nuclear_gradient",
    "stationarity_diagnostic",
]
