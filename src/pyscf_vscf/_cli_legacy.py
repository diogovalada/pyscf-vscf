"""The 0.2 flag-style command line (``pyscf-vscf --task ...``), deprecated.

``pyscf-vscf`` routes flag-style invocations here unchanged for one release;
the subcommand interface in :mod:`pyscf_vscf.cli` replaces it in 0.4.
"""

from __future__ import annotations

import argparse
import math
import multiprocessing as mp
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from ._warnings import PyscfVscfDeprecationWarning
from ._cli_common import (
    CLI_RESULT_SCHEMA,
    CLI_RESULT_SCHEMA_VERSION,
    DEFAULT_BOUNDARY_WARN,
    _CLIError,
    _describe_exception,
    _log_fn,
    _package_version,
    _print_error,
    _print_records,
    _print_worker_traceback,
    _warn_boundary,
    _write_json,
)
from ._cli_common import _worker_init as _common_worker_init

MIN_GRID_POINTS = 3

# Defaults for options whose omission matters when settings are adopted from a
# reused grid cache. The parser stores ``None`` for an omitted option.
_DEFAULT_RMIN = 0.75
_DEFAULT_RMAX = 1.25
_DEFAULT_NPTS = 41
_DEFAULT_BOND = "0-1"

_EPILOG = """\
examples:
  pyscf-vscf --xyz water.xyz --task opt --method b3lyp --basis cc-pvdz
  pyscf-vscf --xyz water.xyz --task harmonic --method b3lyp --basis cc-pvdz
  pyscf-vscf --xyz water.xyz --task 1d --bond 0-1 --rmin 0.65 --rmax 1.6 \\
      --npts 41 --method b3lyp --basis cc-pvdz --dump-grid oh.npz
  pyscf-vscf --xyz water.xyz --task 1d --reuse-grid oh.npz --vmax 4

Bonds are written as zero-based atom indices, e.g. 0-1 for the first two atoms
in the geometry file. Scan grids are bond lengths (or normal-coordinate
displacements) in angstrom. Check that reported states are converged with
respect to the grid: states flagged with '*' place a significant probability
on the outermost grid points.
"""


def _build_parser() -> argparse.ArgumentParser:
    from .settings import ESSettings, HarmonicSettings

    defaults = ESSettings()
    harmonic_defaults = HarmonicSettings()
    parser = argparse.ArgumentParser(
        prog="pyscf-vscf",
        description=(
            "Variational anharmonic vibrational spectroscopy on PySCF potential and dipole "
            "surfaces: geometry optimization, harmonic analysis, and 1D/2D local-stretch or "
            "normal-coordinate DVR spectra."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_package_version())

    geometry = parser.add_argument_group("geometry")
    geometry.add_argument("--xyz", type=Path, help="XYZ geometry file (angstrom)")
    geometry.add_argument("--mmol", type=Path, help="Midas MMOL geometry file")
    geometry.add_argument(
        "--charge",
        type=int,
        default=None,
        help="total molecular charge (default: from the file, usually 0)",
    )
    geometry.add_argument(
        "--spin",
        type=int,
        default=None,
        help="number of unpaired electrons, N(alpha) - N(beta) (default: from the file)",
    )

    task = parser.add_argument_group("task")
    task.add_argument(
        "--task",
        choices=["harmonic", "opt", "1d", "2d"],
        default="harmonic",
        help="calculation to run (default: harmonic)",
    )

    electronic = parser.add_argument_group("electronic structure")
    electronic.add_argument(
        "--method",
        default=None,
        help=f"'hf' or a PySCF exchange-correlation functional (default: {defaults.method})",
    )
    electronic.add_argument(
        "--basis",
        default=None,
        help=f"PySCF basis set name (default: {defaults.basis})",
    )
    ri = electronic.add_mutually_exclusive_group()
    ri.add_argument(
        "--ri",
        dest="use_ri",
        action="store_true",
        default=None,
        help="use RI density fitting (default)",
    )
    ri.add_argument(
        "--no-ri",
        dest="use_ri",
        action="store_false",
        help="disable RI density fitting",
    )
    electronic.add_argument(
        "--ri-aux",
        dest="ri_aux",
        default=None,
        help="auxiliary basis for RI (default: chosen from the main basis)",
    )
    electronic.add_argument(
        "--scf-conv-tol",
        type=float,
        default=None,
        help="SCF energy convergence threshold in Hartree (default: 1e-10)",
    )
    electronic.add_argument(
        "--scf-max-cycle",
        type=int,
        default=None,
        help="maximum SCF iterations (default: 50)",
    )
    electronic.add_argument(
        "--dft-grid-level",
        type=int,
        default=None,
        help="PySCF DFT integration grid level (default: 3; ignored for HF)",
    )

    scan = parser.add_argument_group("scans (--task 1d/2d)")
    scan.add_argument(
        "--bond",
        default=None,
        help=f"scanned bond as zero-based atom indices i-j (default: {_DEFAULT_BOND})",
    )
    scan.add_argument("--bond2", default=None, help="second bond for --task 2d, e.g. 0-2")
    scan.add_argument(
        "--scan",
        choices=["lbs-frozen", "normal", "normal-relaxed"],
        default="lbs-frozen",
        help=(
            "1D scan type: frozen local bond stretch, rigid normal-mode displacement, or "
            "normal-mode displacement with all other coordinates relaxed "
            "(default: lbs-frozen)"
        ),
    )
    scan.add_argument(
        "--rmin",
        type=float,
        default=None,
        help=f"smallest bond length in angstrom (default: {_DEFAULT_RMIN})",
    )
    scan.add_argument(
        "--rmax",
        type=float,
        default=None,
        help=f"largest bond length in angstrom (default: {_DEFAULT_RMAX})",
    )
    scan.add_argument(
        "--npts",
        type=int,
        default=None,
        help=f"grid points per scanned coordinate, at least 3 (default: {_DEFAULT_NPTS})",
    )
    scan.add_argument(
        "--rmin2",
        type=float,
        default=None,
        help="smallest second-bond length for --task 2d (default: --rmin)",
    )
    scan.add_argument(
        "--rmax2",
        type=float,
        default=None,
        help="largest second-bond length for --task 2d (default: --rmax)",
    )
    scan.add_argument(
        "--npts2",
        type=int,
        default=None,
        help="grid points for the second bond in --task 2d (default: --npts)",
    )
    scan.add_argument(
        "--smin",
        type=float,
        default=-0.15,
        help="smallest normal-coordinate displacement in angstrom, < 0 (default: -0.15)",
    )
    scan.add_argument(
        "--smax",
        type=float,
        default=0.15,
        help="largest normal-coordinate displacement in angstrom, > 0 (default: 0.15)",
    )
    scan.add_argument(
        "--keo",
        choices=["reduced", "gmatrix"],
        default="gmatrix",
        help=(
            "2D kinetic energy operator: separable reduced masses, or including the "
            "Wilson G-matrix cross term (default: gmatrix)"
        ),
    )
    scan.add_argument(
        "--vmax",
        type=int,
        default=8,
        help=(
            "number of excited states to report: highest quantum number in 1D, number of "
            "eigenstates above the ground state in 2D (default: 8)"
        ),
    )
    scan.add_argument(
        "--dump-grid",
        type=Path,
        default=None,
        help="write the computed PES/DMS grid to this .npz cache (lbs-frozen scans)",
    )
    scan.add_argument(
        "--reuse-grid",
        type=Path,
        default=None,
        help=(
            "load a PES/DMS grid cache instead of computing it; omitted electronic and "
            "grid options are taken from the cache"
        ),
    )
    scan.add_argument(
        "--partial-out",
        type=Path,
        default=None,
        help=(
            "if scan points fail, write the completed points to this diagnostic .npz "
            "(default with --dump-grid: <dump-grid>.partial.npz)"
        ),
    )
    scan.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep evaluating the remaining points after a scan point fails",
    )
    scan.add_argument(
        "--intensity",
        choices=["axis", "isotropic", "vector", "both"],
        default="axis",
        help=(
            "intensity convention to print: light polarized along the first bond axis, "
            "isotropic average ('vector' is an alias), or both (default: axis)"
        ),
    )
    scan.add_argument(
        "--boundary-warn",
        type=float,
        default=DEFAULT_BOUNDARY_WARN,
        help=(
            "flag states whose probability on the outermost grid points exceeds this "
            f"value (default: {DEFAULT_BOUNDARY_WARN:g})"
        ),
    )

    harmonic = parser.add_argument_group("harmonic analysis and optimization")
    harmonic.add_argument(
        "--rtproj",
        choices=["pyscf", "mw_explicit", "none"],
        default=harmonic_defaults.rtproj,
        help="projection of rotations and translations (default: pyscf)",
    )
    harmonic.add_argument(
        "--allow-fd-hessian",
        action="store_true",
        help="allow a finite-difference Hessian when no analytic Hessian is available",
    )
    harmonic.add_argument(
        "--opt-out",
        type=Path,
        default=None,
        help=(
            "optimized geometry output (.xyz or .mmol; default: "
            "<label>.pyscf_opt.xyz or .mmol, matching the input format)"
        ),
    )
    harmonic.add_argument(
        "--opt-maxsteps",
        type=int,
        default=50,
        help="maximum optimization steps (default: 50)",
    )

    execution = parser.add_argument_group("execution and output")
    execution.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="total CPU threads to use (default: all CPUs available to this process)",
    )
    execution.add_argument(
        "--pes-workers",
        type=int,
        default=None,
        help=(
            "parallel processes for scan points (default: automatic, up to one worker per "
            "CPU, limited by points and available memory; CPUs left over are split among "
            "the workers as threads)"
        ),
    )
    execution.add_argument(
        "--max-memory",
        type=int,
        default=None,
        help="PySCF memory budget per process in MB (default: automatic for scans)",
    )
    execution.add_argument(
        "--json",
        dest="json_out",
        type=Path,
        default=None,
        help="also write machine-readable results to this JSON file",
    )
    strict_group = execution.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict", dest="strict", action="store_true", help=argparse.SUPPRESS
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="downgrade some failures (Hessian, optimization) to warnings",
    )
    parser.set_defaults(strict=True)
    execution.add_argument("-v", "--verbose", action="store_true", help="print progress")
    execution.add_argument(
        "--debug",
        action="store_true",
        help="show full tracebacks instead of one-line error messages",
    )
    return parser


@dataclass
class _Prepared:
    args: argparse.Namespace
    molecule: Any
    cfg: Any
    runtime: Any
    bond: Any = None
    bond2: Any = None
    grid1: tuple[float, float, int] | None = None
    grid2: tuple[float, float, int] | None = None
    opt_out: Path | None = None
    partial_out: Path | None = None
    adopted: dict[str, object] = field(default_factory=dict)


def main(argv: list[str] | None = None) -> int:
    """Run the ``pyscf-vscf`` command line and return its exit code."""

    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        prepared = _prepare(args, parser)
    except _CLIError as exc:
        if args.debug:
            raise
        _print_error(str(exc))
        return 1
    except Exception as exc:
        if args.debug:
            raise
        _print_error(_describe_exception(exc))
        return 1

    from .settings import format_runtime

    log = _log_fn(args.verbose)
    start = time.perf_counter()
    payload: dict[str, Any] | None = None
    try:
        # This interface itself is the deprecated path; do not repeat the warning
        # for every 0.2 function it calls.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyscfVscfDeprecationWarning)
            payload = _run(prepared, log)
        if args.json_out is not None and payload is not None:
            payload["runtime_seconds"] = time.perf_counter() - start
            _write_json(args.json_out, payload)
            log(f"Wrote JSON results: {args.json_out}")
        return 0
    except KeyboardInterrupt:
        _print_error("interrupted")
        return 130
    except Exception as exc:
        if args.debug:
            _print_worker_traceback(exc)
            raise
        _print_error(_describe_exception(exc))
        return 1
    finally:
        elapsed = time.perf_counter() - start
        sys.stdout.flush()
        print(f"Runtime: {elapsed:.2f} s ({format_runtime(elapsed)})", file=sys.stderr)


def _prepare(args: argparse.Namespace, parser: argparse.ArgumentParser) -> _Prepared:
    """Validate everything that can be checked before any expensive work."""

    from .settings import (
        DEFAULT_MAX_PARALLEL,
        apply_thread_env_updates,
        available_cpu_count,
        available_memory_mb,
        derive_parallel_settings,
        MAX_MEMORY_ENV_VAR,
    )

    if bool(args.mmol) == bool(args.xyz):
        parser.error("provide exactly one of --mmol or --xyz")
    for name in ("max_parallel", "pes_workers", "max_memory", "opt_maxsteps"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.vmax < 1:
        parser.error("--vmax must be at least 1")
    if not (math.isfinite(args.boundary_warn) and args.boundary_warn > 0.0):
        parser.error("--boundary-warn must be a positive number")
    if args.task not in {"1d", "2d"}:
        for option in ("dump_grid", "reuse_grid", "partial_out"):
            if getattr(args, option) is not None:
                parser.error(f"--{option.replace('_', '-')} applies only to --task 1d or 2d")
    if args.dump_grid is not None and args.reuse_grid is not None:
        parser.error("use only one of --dump-grid or --reuse-grid")
    for option in ("dump_grid", "reuse_grid", "partial_out"):
        path = getattr(args, option)
        if path is not None and path.suffix.lower() != ".npz":
            parser.error(f"--{option.replace('_', '-')} must end with .npz")
    if args.task == "1d" and args.scan != "lbs-frozen":
        if args.dump_grid is not None or args.reuse_grid is not None:
            parser.error("grid caching is only available for --scan lbs-frozen")
    if args.task == "2d" and args.scan != "lbs-frozen":
        parser.error("--scan applies only to --task 1d")
    if args.json_out is not None and args.json_out.suffix.lower() != ".json":
        parser.error("--json must end with .json")
    for option in ("json_out", "dump_grid", "partial_out", "opt_out"):
        path = getattr(args, option)
        if path is None:
            continue
        if not path.parent.is_dir():
            parser.error(f"output directory does not exist: {path.parent}")
        if path.is_dir():
            parser.error(f"output path is a directory: {path}")

    molecule = _load_molecule(args)
    prepared = _Prepared(args=args, molecule=molecule, cfg=None, runtime=None)
    n_atoms = len(molecule.symbols)

    if args.reuse_grid is not None:
        _adopt_cache_settings(prepared, parser)

    if args.task in {"1d", "2d"}:
        prepared.bond = _parse_bond_option(args.bond or _DEFAULT_BOND, "--bond", n_atoms, parser)
        if args.task == "2d":
            if not args.bond2:
                parser.error("--bond2 is required for --task 2d")
            prepared.bond2 = _parse_bond_option(args.bond2, "--bond2", n_atoms, parser)
            if {prepared.bond.i, prepared.bond.j} == {prepared.bond2.i, prepared.bond2.j}:
                parser.error("--bond and --bond2 must be different bonds")
        rmin = _DEFAULT_RMIN if args.rmin is None else args.rmin
        rmax = _DEFAULT_RMAX if args.rmax is None else args.rmax
        npts = _DEFAULT_NPTS if args.npts is None else args.npts
        if npts < MIN_GRID_POINTS:
            parser.error(f"--npts must be at least {MIN_GRID_POINTS}")
        if args.task == "1d" and args.scan != "lbs-frozen":
            if not (
                math.isfinite(args.smin)
                and math.isfinite(args.smax)
                and args.smin < 0.0 < args.smax
            ):
                parser.error("normal-coordinate bounds must satisfy --smin < 0 < --smax")
        else:
            _check_bond_range(rmin, rmax, "--rmin", "--rmax", parser)
        prepared.grid1 = (float(rmin), float(rmax), int(npts))
        if args.task == "2d":
            rmin2 = rmin if args.rmin2 is None else args.rmin2
            rmax2 = rmax if args.rmax2 is None else args.rmax2
            npts2 = npts if args.npts2 is None else args.npts2
            if npts2 < MIN_GRID_POINTS:
                parser.error(f"--npts2 must be at least {MIN_GRID_POINTS}")
            _check_bond_range(rmin2, rmax2, "--rmin2", "--rmax2", parser)
            prepared.grid2 = (float(rmin2), float(rmax2), int(npts2))
            from .workflows import scans

            try:
                scans.stretch_two_bonds(
                    molecule.coords, prepared.bond, prepared.bond2, rmin, rmin2
                )
            except (ValueError, IndexError) as exc:
                parser.error(f"invalid --bond/--bond2 pair: {exc}")
        elif any(value is not None for value in (args.rmin2, args.rmax2, args.npts2)):
            parser.error("--rmin2, --rmax2 and --npts2 apply only to --task 2d")
        if args.partial_out is not None:
            prepared.partial_out = args.partial_out
        elif args.dump_grid is not None:
            prepared.partial_out = args.dump_grid.with_name(args.dump_grid.stem + ".partial.npz")

    if args.task == "opt":
        prepared.opt_out = args.opt_out
        if prepared.opt_out is None:
            suffix = ".xyz" if args.xyz else ".mmol"
            prepared.opt_out = Path(f"{molecule.label}.pyscf_opt{suffix}")
        elif prepared.opt_out.suffix.lower() not in {".xyz", ".mmol"}:
            parser.error("--opt-out must end with .xyz or .mmol")

    prepared.cfg = _build_es_settings(args)

    max_parallel = args.max_parallel
    if max_parallel is None:
        max_parallel = available_cpu_count() or DEFAULT_MAX_PARALLEL
    computing_grid = args.task in {"1d", "2d"} and args.reuse_grid is None
    n_points = None
    memory = None
    if computing_grid:
        n_points = prepared.grid1[2] * (prepared.grid2[2] if prepared.grid2 else 1)
        if args.pes_workers is None:
            memory = available_memory_mb()
    # Reusing a cached grid runs no electronic points, so it uses one process.
    parallel_task = args.task if computing_grid else "harmonic"
    try:
        prepared.runtime = derive_parallel_settings(
            parallel_task,
            max_parallel=max_parallel,
            pes_workers=args.pes_workers,
            verbose=args.verbose,
            strict=args.strict,
            allow_fd_hessian=args.allow_fd_hessian,
            n_points=n_points,
            available_memory_mb=memory,
            max_memory_mb=args.max_memory,
        )
    except ValueError as exc:
        parser.error(str(exc))
    apply_thread_env_updates(prepared.runtime.threads_per_worker, os.environ)
    if prepared.runtime.max_memory_mb is not None:
        os.environ[MAX_MEMORY_ENV_VAR] = str(prepared.runtime.max_memory_mb)
    return prepared


def _run(prepared: _Prepared, log) -> dict[str, Any] | None:
    from .settings import default_auxbasis

    args = prepared.args
    cfg = prepared.cfg
    runtime = prepared.runtime
    log(f"Selected task: {args.task}")
    for name, value in prepared.adopted.items():
        log(f"Using {name}={value!r} from {args.reuse_grid}")
    log(
        f"Method: {cfg.method} | Basis: {cfg.basis} | "
        f"RI {'enabled' if cfg.use_density_fit else 'disabled'}"
    )
    log(
        "Parallel: "
        f"max_parallel={runtime.max_parallel} "
        f"pes_workers={'auto' if runtime.auto_workers else runtime.pes_workers} "
        f"=> workers={runtime.workers} threads={runtime.threads_per_worker} (BLAS=1)"
        + (
            ""
            if runtime.max_memory_mb is None
            else f" max_memory={runtime.max_memory_mb} MB per process"
        )
    )
    if cfg.use_density_fit:
        log(f"RI auxiliary basis: {cfg.auxbasis or default_auxbasis(cfg.basis)}")
    executor_factory = _executor_factory(
        runtime.workers, runtime.threads_per_worker, runtime.max_memory_mb
    )
    if args.task == "harmonic":
        return _run_harmonic(prepared)
    if args.task == "opt":
        return _run_opt(prepared, log)
    if args.task == "1d":
        return _run_1d(prepared, log, executor_factory)
    return _run_2d(prepared, log, executor_factory)


def _load_molecule(args: argparse.Namespace) -> Any:
    if args.mmol:
        from .io import read_midas_mmol

        molecule = read_midas_mmol(args.mmol)
    else:
        from .io import read_xyz

        molecule = read_xyz(args.xyz)

    if args.charge is None and args.spin is None:
        return molecule

    from .molecule import Molecule

    return Molecule.from_arrays(
        molecule.symbols,
        molecule.coords,
        charge=molecule.charge if args.charge is None else args.charge,
        spin=molecule.spin if args.spin is None else args.spin,
        label=molecule.label,
        masses_amu=molecule.analysis_masses(),
    )


def _parse_bond_option(text: str, option: str, n_atoms: int, parser) -> Any:
    from .coordinates import parse_bond

    try:
        bond = parse_bond(text)
    except ValueError:
        parser.error(f"{option} must be two zero-based atom indices such as 0-1 (got {text!r})")
    if not (0 <= bond.i < n_atoms and 0 <= bond.j < n_atoms):
        parser.error(
            f"{option} {bond.i}-{bond.j} refers to an atom outside 0..{n_atoms - 1} "
            "(indices are zero-based)"
        )
    if bond.i == bond.j:
        parser.error(f"{option} must reference two different atoms")
    return bond


def _check_bond_range(rmin: float, rmax: float, low: str, high: str, parser) -> None:
    if not (math.isfinite(rmin) and math.isfinite(rmax) and 0.0 < rmin < rmax):
        parser.error(f"bond-length grid must satisfy 0 < {low} < {high}")


def _build_es_settings(args: argparse.Namespace) -> Any:
    from .settings import ESSettings

    defaults = ESSettings()
    return ESSettings(
        method=defaults.method if args.method is None else args.method,
        basis=defaults.basis if args.basis is None else args.basis,
        use_density_fit=defaults.use_density_fit if args.use_ri is None else args.use_ri,
        auxbasis=args.ri_aux,
        scf_conv_tol=args.scf_conv_tol,
        scf_max_cycle=args.scf_max_cycle,
        dft_grid_level=args.dft_grid_level,
    )


_ADOPTED_ELECTRONIC_OPTIONS = (
    ("method", "method"),
    ("basis", "basis"),
    ("use_density_fit", "use_ri"),
    ("auxbasis", "ri_aux"),
    ("scf_conv_tol", "scf_conv_tol"),
    ("scf_max_cycle", "scf_max_cycle"),
    ("dft_grid_level", "dft_grid_level"),
)


def _adopt_cache_settings(prepared: _Prepared, parser) -> None:
    """Fill omitted electronic and grid options from a verified grid cache."""

    from .cache import (
        _validate_schema3_metadata_integrity,
        load_grid_npz,
        migrate_cache_metadata_v2,
    )

    args = prepared.args
    path = args.reuse_grid
    if not path.exists():
        raise _CLIError(f"grid cache not found: {path}")
    meta, _arrays = load_grid_npz(path)
    if meta.get("grid_cache_version") == 2:
        meta = migrate_cache_metadata_v2(meta)
    if meta.get("grid_cache_version") != 3:
        raise _CLIError(
            f"{path} is not a reusable grid cache (schema {meta.get('grid_cache_version')!r})"
        )
    _validate_schema3_metadata_integrity(meta, "Grid cache")
    identity = meta["identity"]
    electronic = identity.get("electronic_structure", {})
    if electronic.get("backend") != "pyscf":
        raise _CLIError(
            f"{path} was produced by backend {electronic.get('backend')!r}; the CLI can "
            "only reuse grids from the built-in PySCF backend"
        )
    for key, option in _ADOPTED_ELECTRONIC_OPTIONS:
        if getattr(args, option) is None and electronic.get(key) is not None:
            setattr(args, option, electronic[key])
            prepared.adopted[option] = electronic[key]

    scan = identity.get("scan", {})
    kind = scan.get("kind")
    if args.task == "1d":
        if kind != "1d":
            raise _CLIError(f"{path} holds a {kind!r} grid, not a 1D grid")
        _adopt_bond(args, "bond", scan.get("bond_zero_based"), prepared)
        _adopt_grid(args, ("rmin", "rmax", "npts"), scan.get("grid_A"), path, prepared)
    else:
        if kind != "2d":
            raise _CLIError(f"{path} holds a {kind!r} grid, not a 2D grid")
        _adopt_bond(args, "bond", scan.get("bond1_zero_based"), prepared)
        _adopt_bond(args, "bond2", scan.get("bond2_zero_based"), prepared)
        _adopt_grid(args, ("rmin", "rmax", "npts"), scan.get("grid1_A"), path, prepared)
        _adopt_grid(args, ("rmin2", "rmax2", "npts2"), scan.get("grid2_A"), path, prepared)


def _adopt_bond(args, option: str, value, prepared: _Prepared) -> None:
    if getattr(args, option) is None and isinstance(value, list) and len(value) == 2:
        text = f"{int(value[0])}-{int(value[1])}"
        setattr(args, option, text)
        prepared.adopted[option] = text


def _adopt_grid(args, options: tuple[str, str, str], grid, path: Path, prepared) -> None:
    import numpy as np

    if not isinstance(grid, list) or len(grid) < 2:
        return
    values = np.asarray(grid, dtype=float)
    expressible = np.allclose(
        values, np.linspace(values[0], values[-1], values.size), rtol=0.0, atol=1e-12
    )
    omitted = [option for option in options if getattr(args, option) is None]
    if not omitted:
        return
    if not expressible:
        raise _CLIError(
            f"the grid stored in {path} is not uniform and cannot be expressed with "
            f"--{options[0]}/--{options[1]}/--{options[2]}"
        )
    adopted = {options[0]: float(values[0]), options[1]: float(values[-1])}
    adopted[options[2]] = int(values.size)
    for option in omitted:
        setattr(args, option, adopted[option])
        prepared.adopted[option] = adopted[option]


def _run_harmonic(prepared: _Prepared) -> dict[str, Any]:
    from .workflows import harmonic as harmonic_workflow

    args = prepared.args
    result = harmonic_workflow.harmonic_analysis(
        prepared.molecule,
        prepared.cfg,
        rtproj=args.rtproj,
        strict=args.strict,
        allow_fd_hessian=args.allow_fd_hessian,
        debug=args.verbose,
    )
    print(f"ZPE (harmonic): {result.zpe_cm:.2f} cm^-1")
    print("Frequencies (cm^-1):")
    print(" ".join(f"{float(freq):.1f}" for freq in result.freqs_cm if float(freq) > 1e-2))
    payload = _base_payload(prepared)
    payload["harmonic"] = {
        "zpe_cm": float(result.zpe_cm),
        "frequencies_cm": [float(freq) for freq in result.freqs_cm],
        "hessian_provenance": getattr(result, "hessian_provenance", None),
        "rtproj": args.rtproj,
    }
    return payload


def _run_opt(prepared: _Prepared, log) -> dict[str, Any]:
    from .settings import warn_once
    from .workflows import optimization

    args = prepared.args
    result = optimization.run_opt(
        prepared.molecule,
        prepared.cfg,
        opt_out=prepared.opt_out,
        opt_maxsteps=args.opt_maxsteps,
        strict=args.strict,
        verbose=args.verbose,
        log_fn=log,
        warn_fn=warn_once,
    )
    payload = _base_payload(prepared)
    payload["optimization"] = None
    if result is not None:
        stationarity = getattr(result, "stationarity", None)
        payload["optimization"] = {
            "converged": bool(result.converged),
            "backend": result.backend,
            "output_path": None if result.output_path is None else str(result.output_path),
            "coordinates_A": [[float(x) for x in row] for row in result.molecule.coords],
            "max_gradient_component_Eh_per_bohr": (
                None if stationarity is None else float(stationarity.max_component)
            ),
        }
    return payload


def _run_1d(prepared: _Prepared, log, executor_factory: Any = None) -> dict[str, Any]:
    import numpy as np

    from .variational import parse_intensity_mode, variational_1d
    from .workflows import scans

    args = prepared.args
    molecule = prepared.molecule
    cfg = prepared.cfg
    b = prepared.bond
    rmin, rmax, npts = prepared.grid1
    axis_vec = np.asarray(molecule.coords[b.j], dtype=float) - np.asarray(
        molecule.coords[b.i],
        dtype=float,
    )
    scan_options = _scan_options(prepared)
    grid_source = "computed"

    if args.scan == "lbs-frozen":
        if args.reuse_grid is not None:
            R, E, MU = scans.load_lbs_frozen_1d_grid_cache(
                args.reuse_grid,
                molecule,
                cfg,
                b,
                rmin,
                rmax,
                npts,
                scan=args.scan,
            )
            grid_source = f"cache:{args.reuse_grid}"
            log(f"Reusing cached 1D grid from {args.reuse_grid}")
        else:
            R, E, MU = _guarded_scan(
                prepared,
                scans.grid_1d_pes_dms,
                molecule,
                cfg,
                b,
                rmin,
                rmax,
                npts,
                executor_factory=executor_factory,
                log_fn=log,
                **scan_options,
            )
            if args.dump_grid is not None:
                scans.dump_lbs_frozen_1d_grid_cache(
                    args.dump_grid,
                    molecule,
                    cfg,
                    b,
                    rmin,
                    rmax,
                    npts,
                    R,
                    E,
                    MU,
                    scan=args.scan,
                )
                log(f"Wrote 1D grid cache: {args.dump_grid}")
        redmass_amu = scans.local_bond_reduced_mass_amu(molecule, b)
        normal_summary = None
    else:
        selected_mode = scans.calc_normal_mode_direction(
            molecule,
            cfg,
            b,
            rtproj=args.rtproj,
            strict=args.strict,
            allow_fd_hessian=args.allow_fd_hessian,
            log_fn=log,
        )
        u_dir = selected_mode.u_dir
        smin = float(args.smin)
        smax = float(args.smax)
        if args.scan == "normal":
            R, E, MU = _guarded_scan(
                prepared,
                scans.grid_1d_pes_dms_normal,
                molecule,
                cfg,
                u_dir,
                smin,
                smax,
                npts,
                executor_factory=executor_factory,
                log_fn=log,
                **scan_options,
            )
        else:
            from .backends import pyscf as pyscf_backend

            relaxed_grid = _guarded_scan(
                prepared,
                scans.grid_1d_pes_dms_normal_relaxed,
                molecule,
                cfg,
                u_dir,
                smin,
                smax,
                npts,
                relaxed_point_fn=partial(pyscf_backend.normal_relaxed_point, strict=args.strict),
                executor_factory=executor_factory,
                log_fn=log,
                strict=args.strict,
                **scan_options,
            )
            if relaxed_grid.failed_indices:
                raise _CLIError(
                    relaxed_grid.failure_summary()
                    + "; refusing to compute a spectrum from unconverged points"
                )
            R = relaxed_grid.displacements_A
            E = relaxed_grid.energies_hartree
            MU = relaxed_grid.dipoles_debye
        redmass_amu = scans.normal_mode_effective_mass_amu(molecule, u_dir)
        normal_summary = (selected_mode.mode_index, selected_mode.frequency_cm, redmass_amu)

    records = variational_1d(
        R,
        E,
        MU,
        redmass_amu,
        axis=axis_vec,
        vmax=args.vmax,
    )

    if normal_summary is not None:
        mode_index, freq_cm, mu_eff = normal_summary
        if records:
            print(
                f"Harmonic nu (mode {mode_index}) = {freq_cm:.1f} cm^-1; "
                f"DVR v=1 = {records[0].frequency_cm:.1f} cm^-1; "
                f"mu_eff = {mu_eff:.6f} amu"
            )
        else:
            print(
                f"Harmonic nu (mode {mode_index}) = {freq_cm:.1f} cm^-1; mu_eff = {mu_eff:.6f} amu"
            )

    _print_records(records, parse_intensity_mode(args.intensity), args.boundary_warn, two_d=False)
    _warn_boundary(
        records, args.boundary_warn, "--rmin/--rmax" if normal_summary is None else "--smin/--smax"
    )
    payload = _base_payload(prepared)
    payload["scan"] = {
        "type": args.scan,
        "bond_zero_based": [b.i, b.j],
        "grid_A": [float(value) for value in np.asarray(R, dtype=float)],
        "grid_source": grid_source,
        "mass_amu": float(redmass_amu),
        "intensity_axis": [float(value) for value in axis_vec],
    }
    if normal_summary is not None:
        payload["scan"]["normal_mode"] = {
            "index": int(normal_summary[0]),
            "harmonic_frequency_cm": float(normal_summary[1]),
        }
    payload["transitions"] = [record.as_dict(extended=True) for record in records]
    return payload


def _run_2d(prepared: _Prepared, log, executor_factory: Any = None) -> dict[str, Any]:
    import numpy as np

    from .cache import dump_grid_npz
    from .variational import parse_intensity_mode, variational_2d
    from .workflows import scans

    args = prepared.args
    molecule = prepared.molecule
    cfg = prepared.cfg
    b1 = prepared.bond
    b2 = prepared.bond2
    r1 = prepared.grid1
    r2 = prepared.grid2
    R1_req = np.linspace(*r1)
    R2_req = np.linspace(*r2)
    keo = args.keo
    grid_source = "computed"

    if args.reuse_grid is not None:
        R1, R2, E, MU = scans.load_lbs_frozen_2d_grid_cache(
            args.reuse_grid,
            molecule,
            cfg,
            b1,
            b2,
            r1,
            r2,
            keo=keo,
        )
        grid_source = f"cache:{args.reuse_grid}"
        log(f"Reusing cached 2D grid from {args.reuse_grid}")
    else:
        R1, R2, E, MU = _guarded_scan(
            prepared,
            scans.grid_2d_pes_dms,
            molecule,
            cfg,
            b1,
            b2,
            R1_req,
            R2_req,
            executor_factory=executor_factory,
            log_fn=log,
            **_scan_options(prepared),
        )
        if args.dump_grid is not None:
            meta = scans.lbs_frozen_2d_cache_metadata(
                molecule,
                cfg,
                b1,
                b2,
                r1,
                r2,
                keo=keo,
            )
            dump_grid_npz(
                args.dump_grid,
                meta=meta,
                arrays={"R1_A": R1, "R2_A": R2, "E_Eh": E, "MU_Debye": MU},
            )
            log(f"Wrote 2D grid cache: {args.dump_grid}")

    axis_vec = np.asarray(molecule.coords[b1.j], dtype=float) - np.asarray(
        molecule.coords[b1.i],
        dtype=float,
    )
    mu1 = scans.local_bond_reduced_mass_amu(molecule, b1)
    mu2 = scans.local_bond_reduced_mass_amu(molecule, b2)
    g12_inv_amu = 0.0
    if keo == "gmatrix":
        g12_inv_amu = scans.bond_bond_g12_inv_amu(molecule, b1, b2)

    records = variational_2d(
        R1,
        R2,
        E,
        MU,
        mu1,
        mu2,
        axis=axis_vec,
        nmax=args.vmax,
        g12_inv_amu=g12_inv_amu,
    )
    _print_records(records, parse_intensity_mode(args.intensity), args.boundary_warn, two_d=True)
    _warn_boundary(records, args.boundary_warn, "--rmin/--rmax (and --rmin2/--rmax2)")
    payload = _base_payload(prepared)
    payload["scan"] = {
        "type": "lbs-frozen-2d",
        "bond1_zero_based": [b1.i, b1.j],
        "bond2_zero_based": [b2.i, b2.j],
        "grid1_A": [float(value) for value in np.asarray(R1, dtype=float)],
        "grid2_A": [float(value) for value in np.asarray(R2, dtype=float)],
        "grid_source": grid_source,
        "keo": keo,
        "reduced_masses_amu": [float(mu1), float(mu2)],
        "g12_inv_amu": float(g12_inv_amu),
        "intensity_axis": [float(value) for value in axis_vec],
    }
    payload["transitions"] = [record.as_dict(extended=True) for record in records]
    return payload


def _scan_options(prepared: _Prepared) -> dict[str, Any]:
    # A serial run stops right after a failing point; parallel runs submit
    # bounded batches so that work already scheduled is kept.
    workers = int(prepared.runtime.workers or 1)
    return {
        "stop_on_error": not prepared.args.continue_on_error,
        "batch_size": 1 if workers <= 1 else max(16, 4 * workers),
    }


def _guarded_scan(prepared: _Prepared, scan_fn, *args: Any, **kwargs: Any) -> Any:
    """Run a scan; on point failures, save completed points and explain."""

    from .workflows import scans

    try:
        return scan_fn(*args, **kwargs)
    except scans.ScanFailedError as exc:
        message = str(exc)
        if prepared.partial_out is not None:
            try:
                scans.dump_partial_scan_npz(
                    prepared.partial_out, exc.partial, prepared.molecule, prepared.cfg
                )
            except Exception as write_error:
                message += (
                    f"\ncould not write completed points to {prepared.partial_out}: "
                    f"{_describe_exception(write_error)}"
                )
            else:
                message += (
                    f"\ncompleted points were written to {prepared.partial_out} "
                    "(diagnostic only; not a reusable grid cache)"
                )
        if not prepared.args.continue_on_error and exc.partial.stopped_early:
            message += "\nuse --continue-on-error to attempt the remaining points"
        error = _CLIError(message)
        error.scan_failure = exc
        raise error from exc


def _base_payload(prepared: _Prepared) -> dict[str, Any]:
    from .cache import electronic_structure_provenance, molecule_provenance

    args = prepared.args
    return {
        "schema": CLI_RESULT_SCHEMA,
        "schema_version": CLI_RESULT_SCHEMA_VERSION,
        "package_version": _package_version(),
        "task": args.task,
        "input": str(args.xyz or args.mmol),
        "molecule": molecule_provenance(prepared.molecule),
        "electronic_structure": electronic_structure_provenance(prepared.cfg),
        "settings_from_cache": dict(prepared.adopted),
        "boundary_warning_threshold": float(args.boundary_warn),
    }


class _SequentialExecutor:
    def __enter__(self) -> "_SequentialExecutor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def map(self, fn, iterable):
        return (fn(item) for item in iterable)


def _executor_factory(
    workers: int | None,
    threads_per_worker: int | None,
    max_memory_mb: int | None = None,
):
    if workers is None or int(workers) <= 1:
        return _SequentialExecutor

    worker_count = int(workers)
    initargs: tuple[Any, ...] = (threads_per_worker,)
    if max_memory_mb is not None:
        initargs = (threads_per_worker, int(max_memory_mb))

    def make_executor():
        return ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_init,
            initargs=initargs,
            mp_context=_multiprocessing_context(),
        )

    return make_executor


def _multiprocessing_context():
    method = os.environ.get("VSCF_PYSCF_START_METHOD", "spawn")
    try:
        return mp.get_context(method)
    except ValueError:
        return mp.get_context("spawn")


def _worker_init(threads_per_worker: int | None, max_memory_mb: int | None = None) -> None:
    _common_worker_init(threads_per_worker, max_memory_mb)


if __name__ == "__main__":
    raise SystemExit(main())
