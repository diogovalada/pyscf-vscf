"""PySCF-backed geometry optimization workflow.

This module contains the package-level orchestration extracted from the legacy
``pyscf_pme_pipeline.py`` script. It remains import-light: PySCF, geomeTRIC, and
PyBerny are imported only when an optimization is actually run.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyscf_vscf.backends import pyscf as pyscf_backend
from pyscf_vscf.io import write_midas_mmol, write_xyz
from pyscf_vscf.settings import DEFAULT_STRICT, ESSettings
from pyscf_vscf.workflows.harmonic import StationarityDiagnostic, stationarity_diagnostic


_DEFAULT_ES = ESSettings()


@dataclass
class OptimizedMolecule:
    """Minimal optimized geometry container accepted by package I/O helpers."""

    symbols: list[str]
    coords: np.ndarray
    charge: int = 0
    spin: int = 0
    label: str = "mol"
    masses: np.ndarray | None = None


@dataclass(frozen=True)
class OptimizationResult:
    """Result returned by :func:`run_opt`."""

    molecule: OptimizedMolecule
    converged: bool
    backend: str
    output_path: Path | None = None
    stationarity: StationarityDiagnostic | None = None


def opt_kwargs_for_profile(profile: str | None, *, backend: str = "geometric") -> dict[str, float]:
    """Return optimizer convergence kwargs for a named legacy profile."""

    profile = (profile or "orca").lower()
    if profile not in ("orca", "orca-tight", "orca_tight"):
        raise ValueError(f"Unknown --opt-conv profile '{profile}'")

    optimizer = backend.lower()
    if optimizer == "geometric":
        # Match the ORCA Opt thresholds used in legacy AiiDA inputs (Eh/Bohr).
        # geomeTRIC uses convergence_{gmax,grms} in Eh/Bohr.
        return {
            "convergence_gmax": 4.5e-5,
            "convergence_grms": 3.0e-5,
        }
    if optimizer == "berny":
        # PyBerny uses gradient{max,rms} for the same force thresholds.
        return {
            "gradientmax": 4.5e-5,
            "gradientrms": 3.0e-5,
        }
    raise ValueError(f"Unknown optimizer backend '{backend}'")


def run_opt(
    mol: Any,
    cfg: Any,
    *,
    opt_out: Path | str | None,
    opt_maxsteps: int | None,
    opt_conv: str | None,
    verbose: bool = False,
    log_fn: Callable[[str], None] | None = None,
    warn_fn: Callable[[str, str], None] | None = None,
) -> OptimizationResult:
    """Run a legacy-compatible PySCF geometry optimization and write the result."""

    emit_log = log_fn or _noop_log
    emit_warning = warn_fn or pyscf_backend.warn_once

    emit_log("Geometry optimization: setting up PySCF molecule and mean-field")
    basis = str(_cfg_get(cfg, "basis", _DEFAULT_ES.basis))
    pmol = _molecule_to_pyscf(mol, basis)
    mf = pyscf_backend.make_mean_field(pmol, cfg)
    mf.verbose = 4 if verbose else 0

    maxsteps = None if opt_maxsteps is None else int(opt_maxsteps)
    if maxsteps is not None:
        emit_log(
            "Starting geometry optimization "
            f"(backend=geomeTRIC; maxsteps={maxsteps}; conv='{opt_conv or 'orca'}')"
        )
    else:
        emit_log(
            "Starting geometry optimization "
            f"(backend=geomeTRIC; maxsteps=None; conv='{opt_conv or 'orca'}')"
        )

    try:
        converged, optmol, backend = _run_geometric(mf, opt_conv, maxsteps)
    except ImportError as exc:
        emit_warning(
            "geometric_missing_fallback_berny",
            f"geomeTRIC optimizer unavailable ({exc}); falling back to PySCF Berny optimizer",
        )
        converged, optmol, backend = _run_berny(mf, opt_conv, maxsteps)

    if not converged:
        msg = "Geometry optimization did not converge within the allowed steps"
        if bool(_cfg_get(cfg, "strict", DEFAULT_STRICT)):
            raise RuntimeError(msg)
        emit_warning("opt_not_converged", msg)

    mol_opt = _optimized_molecule_from_pyscf(mol, optmol)
    stationarity = _post_optimization_stationarity(mol_opt, cfg, warn_fn=emit_warning)
    output_path = _write_optimized_geometry(mol_opt, mol, opt_out)
    print(f"Optimized geometry written to: {output_path}", flush=True)

    return OptimizationResult(
        molecule=mol_opt,
        converged=bool(converged),
        backend=backend,
        output_path=output_path,
        stationarity=stationarity,
    )


def _run_geometric(
    mf: Any,
    opt_conv: str | None,
    maxsteps: int | None,
) -> tuple[bool, Any, str]:
    solver = _load_geometric_solver()
    kwargs = opt_kwargs_for_profile(opt_conv, backend="geometric")
    if maxsteps is not None:
        kwargs["maxsteps"] = maxsteps
    converged, optmol = solver.kernel(mf, **kwargs)
    return bool(converged), optmol, "geomeTRIC"


def _run_berny(
    mf: Any,
    opt_conv: str | None,
    maxsteps: int | None,
) -> tuple[bool, Any, str]:
    solver = _load_berny_solver()
    kwargs = opt_kwargs_for_profile(opt_conv, backend="berny")
    if maxsteps is not None:
        kwargs["maxsteps"] = maxsteps
    converged, optmol = solver.kernel(mf, **kwargs)
    return bool(converged), optmol, "berny"


def _post_optimization_stationarity(
    mol_opt: OptimizedMolecule,
    cfg: Any,
    *,
    warn_fn: Callable[[str, str], None],
) -> StationarityDiagnostic | None:
    try:
        basis = str(_cfg_get(cfg, "basis", _DEFAULT_ES.basis))
        pmol = _molecule_to_pyscf(mol_opt, basis)
        mf = pyscf_backend.make_mean_field(pmol, cfg)
        diagnostic = stationarity_diagnostic(mf)
        print(format_optimization_stationarity_diagnostic(diagnostic))
        return diagnostic
    except Exception as exc:
        warn_fn("opt_grad_check_failed", f"Post-opt gradient check failed: {exc}")
        return None


def format_optimization_stationarity_diagnostic(diagnostic: StationarityDiagnostic) -> str:
    """Format the post-optimization gradient check with legacy text."""

    return "\n".join(
        [
            "Optimized-geometry gradient check (Eh/Bohr):",
            "  "
            f"max|g_comp|={diagnostic.max_component:.3e}  "
            f"rms|g_comp|={diagnostic.rms_component:.3e}",
        ]
    )


def _write_optimized_geometry(
    mol_opt: OptimizedMolecule,
    source_mol: Any,
    opt_out: Path | str | None,
) -> Path:
    if opt_out is None:
        opt_out = Path(f"{mol_opt.label}.pyscf_opt.mmol")
    output_path = Path(opt_out)
    suffix = output_path.suffix.lower()
    source_label = str(_mol_get(source_mol, "label", mol_opt.label))
    if suffix == ".xyz":
        write_xyz(output_path, mol_opt, comment=f"{source_label} (PySCF optimized)")
    elif suffix == ".mmol":
        write_midas_mmol(output_path, mol_opt, title=f"{source_label} (PySCF optimized)")
    else:
        raise ValueError(f"--opt-out must end with .xyz or .mmol (got '{output_path.name}')")
    return output_path


def _optimized_molecule_from_pyscf(source_mol: Any, optmol: Any) -> OptimizedMolecule:
    atom_coords = getattr(optmol, "atom_coords")
    coords_opt = np.asarray(atom_coords(unit="Angstrom"), dtype=float)
    if coords_opt.ndim != 2 or coords_opt.shape[1] != 3:
        raise ValueError("Optimized PySCF molecule returned unexpected coordinates")

    masses = _mol_get(source_mol, "masses", None)
    if masses is not None:
        masses = np.asarray(masses, dtype=float)
    return OptimizedMolecule(
        symbols=list(_mol_get(source_mol, "symbols", [])),
        coords=coords_opt,
        charge=int(_mol_get(source_mol, "charge", 0)),
        spin=int(_mol_get(source_mol, "spin", 0)),
        label=str(_mol_get(source_mol, "label", "mol")),
        masses=masses,
    )


def _molecule_to_pyscf(molecule: Any, basis: str) -> Any:
    as_pyscf = getattr(molecule, "as_pyscf", None)
    if callable(as_pyscf):
        return as_pyscf(basis)
    return pyscf_backend.molecule_to_pyscf(molecule, basis=basis)


def _load_geometric_solver() -> Any:
    try:
        from pyscf.geomopt import geometric_solver
    except ImportError as exc:
        raise ImportError("geomeTRIC optimizer is unavailable") from exc
    return geometric_solver


def _load_berny_solver() -> Any:
    try:
        from pyscf.geomopt import berny_solver
    except ImportError as exc:
        raise ImportError("PySCF Berny optimizer is unavailable") from exc
    return berny_solver


def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _mol_get(molecule: Any, name: str, default: Any) -> Any:
    if isinstance(molecule, Mapping):
        return molecule.get(name, default)
    return getattr(molecule, name, default)


def _noop_log(_msg: str) -> None:
    return None


_opt_kwargs_for_profile = opt_kwargs_for_profile


__all__ = [
    "OptimizationResult",
    "OptimizedMolecule",
    "format_optimization_stationarity_diagnostic",
    "opt_kwargs_for_profile",
    "run_opt",
]
