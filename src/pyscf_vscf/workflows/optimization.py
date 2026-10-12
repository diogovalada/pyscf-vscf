"""PySCF-backed geometry optimization workflow.

This module remains import-light: PySCF, geomeTRIC, and PyBerny are imported
only when an optimization is actually run.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pyscf_vscf.backends import pyscf as pyscf_backend
from pyscf_vscf.backends._pyscf_factory import ConfigurationGuard
from pyscf_vscf.io import write_midas_mmol, write_xyz
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.settings import ESSettings
from pyscf_vscf.workflows.harmonic import StationarityDiagnostic, stationarity_diagnostic


_DEFAULT_ES = ESSettings()
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptimizationResult:
    """Result returned by ``optimize_geometry``."""

    molecule: Molecule
    converged: bool
    backend: str
    output_path: Path | None = None
    stationarity: StationarityDiagnostic | None = None


def optimization_convergence_kwargs() -> dict[str, float]:
    """Return the package optimization convergence thresholds."""

    return {
        "convergence_gmax": 4.5e-5,
        "convergence_grms": 3.0e-5,
    }


def optimize_geometry(
    molecule: Any,
    settings: Any,
    *,
    output_path: Path | str | None = None,
    max_steps: int | None = 50,
    strict: bool = True,
) -> OptimizationResult:
    """Optimize a geometry with geomeTRIC on the package's PySCF method.

    ``settings`` is ``ESSettings`` (or compatible) or a method factory such as
    ``MethodRecipe``; the method is rebuilt
    through the construction seam at every geometry. The optimized geometry is
    written only when ``output_path`` (``.xyz`` or ``.mmol``) is given. A
    non-converged optimization raises in strict mode and warns otherwise.
    """

    if output_path is not None:
        _validate_output_suffix(Path(output_path))
    factory = pyscf_backend.method_factory(settings)
    maxsteps = None if max_steps is None else int(max_steps)
    _LOG.info("Starting geometry optimization (backend=geomeTRIC; maxsteps=%s)", maxsteps)
    guard = ConfigurationGuard(factory)
    scanner = _factory_gradient_scanner(molecule, factory, guard)
    converged, optmol, backend = _run_geometric(scanner, maxsteps)
    if not converged:
        message = "Geometry optimization did not converge within the allowed steps"
        if bool(strict):
            raise RuntimeError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    mol_opt = _optimized_molecule_from_pyscf(molecule, optmol)
    stationarity = _post_optimization_stationarity(mol_opt, factory, guard)
    written = None
    if output_path is not None:
        written = _write_optimized_geometry(mol_opt, molecule, output_path)
        _LOG.info("Optimized geometry written to %s", written)
    return OptimizationResult(
        molecule=mol_opt,
        converged=bool(converged),
        backend=backend,
        output_path=written,
        stationarity=stationarity,
    )


def _factory_gradient_scanner(
    molecule: Any, factory: Any, guard: ConfigurationGuard | None = None
) -> Any:
    """Return a geomeTRIC-compatible gradient scanner that rebuilds the method.

    PySCF's own scanners reuse one mean-field object across geometries; this
    one builds the molecule and mean field through the seam at every step, so
    every setting of a user recipe applies at every geometry. ``guard``
    rejects builds whose method configuration changes between steps.
    """

    import sys

    from pyscf import lib

    from pyscf_vscf.backends._pyscf_factory import ElectronicSystem, build_mean_field
    from pyscf_vscf.backends._pyscf_properties import converge, nuclear_gradient

    system = ElectronicSystem.from_molecule(molecule)
    guard = guard or ConfigurationGuard(factory)
    initial = build_mean_field(system, factory)
    guard.check(initial.configuration_snapshot)

    class _State:
        converged = False
        e_tot = 0.0

    class FactoryGradScanner(lib.GradScanner):
        def __init__(self) -> None:
            self.mol = initial.molecule
            self.base = _State()
            self.verbose = 0
            self.stdout = sys.stdout
            self.atmlst = None

        def __call__(self, mol: Any) -> tuple[float, np.ndarray]:
            point = ElectronicSystem(
                nuclear_charges=system.nuclear_charges,
                coordinates_A=np.asarray(mol.atom_coords(unit="Angstrom"), dtype=float),
                charge=system.charge,
                spin=system.spin,
            )
            build = build_mean_field(point, factory)
            guard.check(build.configuration_snapshot)
            converged = converge(build)
            self.base.e_tot = converged.total_energy_Eh
            self.base.converged = True
            self.mol = mol
            return converged.total_energy_Eh, nuclear_gradient(converged)

    return FactoryGradScanner()


def _run_geometric(
    scanner: Any,
    maxsteps: int | None,
) -> tuple[bool, Any, str]:
    solver = _load_geometric_solver()
    kwargs: dict[str, Any] = dict(optimization_convergence_kwargs())
    if maxsteps is not None:
        kwargs["maxsteps"] = maxsteps
    with _geometric_logging():
        converged, optmol = solver.kernel(scanner, **kwargs)
    return bool(converged), optmol, "geomeTRIC"


@contextmanager
def _geometric_logging() -> Iterator[None]:
    """Run geomeTRIC without letting it reconfigure logging.

    geomeTRIC calls ``logging.config.fileConfig``, which replaces the root
    logger's handlers and closes every existing handler. That call is skipped;
    geomeTRIC's progress log is shown (on standard error) only when the
    ``pyscf_vscf`` logger shows INFO messages, and its warnings always.
    """

    import logging.config
    import sys

    original = logging.config.fileConfig
    loggers = [logging.getLogger("geometric"), logging.getLogger("geometric.nifty")]
    levels = [logger.level for logger in loggers]
    handler: logging.Handler | None = None
    logging.config.fileConfig = _skip_logging_configuration  # type: ignore[assignment]
    if logging.getLogger("pyscf_vscf").isEnabledFor(logging.INFO):
        handler = logging.StreamHandler(sys.stderr)
        handler.terminator = ""  # geomeTRIC messages carry their own newlines
        loggers[0].addHandler(handler)
        loggers[0].setLevel(logging.INFO)
    else:
        for logger in loggers:
            logger.setLevel(logging.WARNING)
    try:
        yield
    finally:
        logging.config.fileConfig = original  # type: ignore[assignment]
        if handler is not None:
            loggers[0].removeHandler(handler)
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


def _skip_logging_configuration(*args: Any, **kwargs: Any) -> None:
    return None


def _post_optimization_stationarity(
    mol_opt: Molecule,
    factory: Any,
    guard: ConfigurationGuard,
) -> StationarityDiagnostic | None:
    try:
        converged = pyscf_backend.converged_mean_field(mol_opt, factory)
    except Exception as exc:
        warnings.warn(
            f"Post-optimization gradient check failed: {exc}", RuntimeWarning, stacklevel=3
        )
        return None
    guard.check(converged.build.configuration_snapshot)
    try:
        return stationarity_diagnostic(converged.mean_field)
    except Exception as exc:
        warnings.warn(
            f"Post-optimization gradient check failed: {exc}", RuntimeWarning, stacklevel=3
        )
        return None


def format_optimization_stationarity_diagnostic(diagnostic: StationarityDiagnostic) -> str:
    """Format the post-optimization gradient check."""

    return "\n".join(
        [
            "Optimized-geometry gradient check (Eh/Bohr):",
            "  "
            f"max|g_comp|={diagnostic.max_component:.3e}  "
            f"rms|g_comp|={diagnostic.rms_component:.3e}",
        ]
    )


def _write_optimized_geometry(
    mol_opt: Molecule,
    source_mol: Any,
    opt_out: Path | str | None,
) -> Path:
    if opt_out is None:
        opt_out = Path(f"{mol_opt.label}.pyscf_opt.mmol")
    output_path = Path(opt_out)
    suffix = _validate_output_suffix(output_path)
    source_label = str(_mol_get(source_mol, "label", mol_opt.label))
    if suffix == ".xyz":
        write_xyz(output_path, mol_opt, comment=f"{source_label} (PySCF optimized)")
    else:
        write_midas_mmol(output_path, mol_opt, title=f"{source_label} (PySCF optimized)")
    return output_path


def _validate_output_suffix(output_path: Path) -> str:
    suffix = output_path.suffix.lower()
    if suffix not in {".xyz", ".mmol"}:
        raise ValueError(
            f"Optimized-geometry output must end with .xyz or .mmol (got '{output_path.name}')"
        )
    return suffix


def _optimized_molecule_from_pyscf(source_mol: Any, optmol: Any) -> Molecule:
    atom_coords = getattr(optmol, "atom_coords")
    coords_opt = np.asarray(atom_coords(unit="Angstrom"), dtype=float)
    if coords_opt.ndim != 2 or coords_opt.shape[1] != 3:
        raise ValueError("Optimized PySCF molecule returned unexpected coordinates")

    masses = _mol_get(source_mol, "masses", None)
    if masses is not None:
        masses = np.asarray(masses, dtype=float)
    return Molecule.from_arrays(
        list(_mol_get(source_mol, "symbols", [])),
        coords_opt,
        charge=int(_mol_get(source_mol, "charge", 0)),
        spin=int(_mol_get(source_mol, "spin", 0)),
        label=str(_mol_get(source_mol, "label", "mol")),
        masses_amu=masses,
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


__all__ = [
    "OptimizationResult",
    "format_optimization_stationarity_diagnostic",
    "optimization_convergence_kwargs",
    "optimize_geometry",
]
