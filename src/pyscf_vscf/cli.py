"""The ``pyscf-vscf`` command line.

Subcommands::

    pyscf-vscf harmonic GEOMETRY [electronic options]
    pyscf-vscf opt      GEOMETRY [--output FILE]
    pyscf-vscf scan1d   GEOMETRY --bond I-J [--range RMIN RMAX NPTS] [--store FILE]
    pyscf-vscf scan2d   GEOMETRY --bond I-J --bond K-L [--range ... --range ...]
    pyscf-vscf inspect  FILE

Flag-style 0.2 invocations (``pyscf-vscf --xyz ... --task ...``) still work
for one release through :mod:`pyscf_vscf._cli_legacy`, with a notice.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, cast

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
    _warn_boundary,
    _write_json,
)

SUBCOMMANDS = ("harmonic", "opt", "scan1d", "scan2d", "inspect")
_LEGACY_NOTICE = (
    "pyscf-vscf: note: flag-style options (--xyz/--mmol/--task) are deprecated and will be "
    "removed in 0.4; use subcommands such as 'pyscf-vscf scan1d GEOMETRY --bond 0-1'. "
    "Run 'pyscf-vscf --help' for details."
)

_EPILOG = """\
examples:
  pyscf-vscf opt water.xyz --method b3lyp --basis cc-pvdz --output water-opt.xyz
  pyscf-vscf harmonic water-opt.xyz --method b3lyp --basis cc-pvdz
  pyscf-vscf scan1d water-opt.xyz --bond 0-1 --method b3lyp --basis cc-pvdz \\
      --store water.sqlite --output oh.npz
  pyscf-vscf scan1d water-opt.xyz --bond 0-1 --reuse oh.npz --states 4
  pyscf-vscf scan2d water-opt.xyz --bond 0-1 --bond 0-2 --store water.sqlite

GEOMETRY is an .xyz or Midas .mmol file in angstrom; 'D' marks deuterium.
Bonds are zero-based atom indices (0-1 is the first two atoms). Without
--range, a bond grid is placed around the bond's length in GEOMETRY, so start
from an optimized geometry. Reported states marked '*' place a significant
probability on the outermost grid points and are not converged.
"""


@dataclass
class _Context:
    args: argparse.Namespace
    molecule: Any
    log: Any
    settings: Any = None
    factory: Any = None
    provider: Any = None
    parallel: Any = None


def main(argv: list[str] | None = None) -> int:
    """Run the ``pyscf-vscf`` command line and return its exit code."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if (
        arguments
        and arguments[0].startswith("-")
        and arguments[0]
        not in {
            "-h",
            "--help",
            "--version",
        }
    ):
        from . import _cli_legacy

        print(_LEGACY_NOTICE, file=sys.stderr)
        return _cli_legacy.main(arguments)

    parser = _build_parser()
    if not arguments:
        parser.print_help(sys.stderr)
        return 2
    args = parser.parse_args(arguments)
    log = _log_fn(getattr(args, "verbose", False))
    try:
        context = _prepare(args, parser, log)
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        _print_error(_describe_exception(exc))
        return 1

    from .settings import format_runtime

    start = time.perf_counter()
    try:
        payload = args.handler(context)
        if getattr(args, "json_out", None) is not None and payload is not None:
            payload["runtime_seconds"] = time.perf_counter() - start
            _write_json(args.json_out, payload)
            log(f"Wrote JSON results: {args.json_out}")
        return 0
    except KeyboardInterrupt:
        _print_error("interrupted")
        return 130
    except Exception as exc:
        if args.debug:
            _print_first_traceback(exc)
            raise
        _print_error(_describe_exception(exc))
        return 1
    finally:
        if args.command != "inspect":
            elapsed = time.perf_counter() - start
            sys.stdout.flush()
            print(f"Runtime: {elapsed:.2f} s ({format_runtime(elapsed)})", file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    from .settings import ESSettings

    defaults = ESSettings()
    parser = argparse.ArgumentParser(
        prog="pyscf-vscf",
        description=(
            "Variational anharmonic vibrational spectroscopy (DVR, VSCF, VCI) on PySCF "
            "potential and dipole surfaces."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=_package_version())
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")

    def electronic_parent() -> argparse.ArgumentParser:
        parent = argparse.ArgumentParser(add_help=False)
        group = parent.add_argument_group("electronic structure")
        group.add_argument(
            "--method",
            default=None,
            help=f"'hf' or a PySCF exchange-correlation functional (default: {defaults.method})",
        )
        group.add_argument(
            "--basis", default=None, help=f"PySCF basis set (default: {defaults.basis})"
        )
        ri = group.add_mutually_exclusive_group()
        ri.add_argument(
            "--ri", dest="use_ri", action="store_true", default=None, help="use RI (default)"
        )
        ri.add_argument("--no-ri", dest="use_ri", action="store_false", help="disable RI")
        group.add_argument(
            "--ri-aux", default=None, help="RI auxiliary basis (default: automatic)"
        )
        group.add_argument(
            "--scf-conv-tol", type=float, default=None, help="SCF threshold in Eh (default: 1e-10)"
        )
        group.add_argument(
            "--scf-max-cycle", type=int, default=None, help="maximum SCF cycles (default: 50)"
        )
        group.add_argument(
            "--dft-grid-level", type=int, default=None, help="DFT grid level (default: 3)"
        )
        group.add_argument(
            "--recipe",
            default=None,
            metavar="MODULE:NAME",
            help=(
                "use a MethodRecipe object from an importable module instead of the "
                "options above (any PySCF method; see the Python documentation)"
            ),
        )
        geometry = parent.add_argument_group("molecule")
        geometry.add_argument("--charge", type=int, default=None, help="total charge")
        geometry.add_argument(
            "--spin", type=int, default=None, help="unpaired electrons, N(alpha) - N(beta)"
        )
        return parent

    def execution_parent() -> argparse.ArgumentParser:
        parent = argparse.ArgumentParser(add_help=False)
        group = parent.add_argument_group("execution and output")
        group.add_argument(
            "--max-parallel",
            type=int,
            default=None,
            help="total CPU threads (default: all CPUs available to this process)",
        )
        group.add_argument(
            "--workers",
            type=int,
            default=None,
            help=(
                "worker processes for scan points (default: automatic, limited by CPUs, "
                "points, and available memory)"
            ),
        )
        group.add_argument(
            "--max-memory", type=int, default=None, help="PySCF memory per process in MB"
        )
        group.add_argument(
            "--json", dest="json_out", type=Path, default=None, help="write JSON results"
        )
        group.add_argument("-v", "--verbose", action="store_true", help="print progress")
        group.add_argument("--debug", action="store_true", help="show full tracebacks")
        return parent

    electronic = electronic_parent()
    execution = execution_parent()

    harmonic = commands.add_parser(
        "harmonic",
        parents=[electronic, execution],
        help="harmonic frequencies and zero-point energy",
        description="Harmonic frequencies from the analytic Hessian.",
    )
    harmonic.add_argument("geometry", type=Path, help="geometry file (.xyz or .mmol)")
    harmonic.add_argument(
        "--rtproj",
        choices=["pyscf", "mw_explicit", "none"],
        default="pyscf",
        help="projection of rotations and translations (default: pyscf)",
    )
    harmonic.add_argument(
        "--allow-fd-hessian",
        action="store_true",
        help="allow finite differences of gradients when no analytic Hessian exists",
    )
    harmonic.add_argument(
        "--no-strict", dest="strict", action="store_false", help="downgrade Hessian failures"
    )
    harmonic.set_defaults(handler=_run_harmonic, strict=True)

    opt = commands.add_parser(
        "opt",
        parents=[electronic, execution],
        help="geometry optimization (geomeTRIC)",
        description="Optimize the geometry; the method is rebuilt at every step.",
    )
    opt.add_argument("geometry", type=Path, help="geometry file (.xyz or .mmol)")
    opt.add_argument(
        "--output",
        type=Path,
        default=None,
        help="optimized geometry (.xyz or .mmol; default: GEOMETRY stem + -opt + same suffix)",
    )
    opt.add_argument("--max-steps", type=int, default=50, help="optimization steps (default: 50)")
    opt.add_argument(
        "--no-strict", dest="strict", action="store_false", help="warn instead of failing"
    )
    opt.set_defaults(handler=_run_opt, strict=True)

    def scan_parent() -> argparse.ArgumentParser:
        parent = argparse.ArgumentParser(add_help=False)
        group = parent.add_argument_group("scan and spectrum")
        group.add_argument(
            "--store",
            type=Path,
            default=None,
            help="SQLite point store: completed points are kept and reused (resume)",
        )
        group.add_argument(
            "--retry-failed", action="store_true", help="re-evaluate points that failed before"
        )
        group.add_argument(
            "--continue-on-error",
            action="store_true",
            help="keep evaluating the remaining points after a point fails",
        )
        group.add_argument("--output", type=Path, default=None, help="write the scan (.npz, v4)")
        group.add_argument(
            "--reuse",
            type=Path,
            default=None,
            help="read a scan (.npz, v4 or a 0.2 grid cache) instead of computing it",
        )
        group.add_argument(
            "--states",
            type=int,
            default=4,
            help="highest reported state (1D: quantum number; 2D: eigenstate index; default: 4)",
        )
        group.add_argument(
            "--intensity",
            choices=["axis", "isotropic", "both"],
            default="axis",
            help="intensity convention to print (default: axis, along the first bond)",
        )
        group.add_argument(
            "--boundary-warn",
            type=float,
            default=DEFAULT_BOUNDARY_WARN,
            help=f"flag states above this grid-edge probability (default: {DEFAULT_BOUNDARY_WARN:g})",
        )
        return parent

    scan = scan_parent()
    scan1d = commands.add_parser(
        "scan1d",
        parents=[electronic, scan, execution],
        help="1D bond or normal-coordinate scan and DVR spectrum",
        description="Scan one coordinate, then solve the 1D vibrational problem.",
    )
    scan1d.add_argument("geometry", type=Path, help="geometry file (.xyz or .mmol)")
    scan1d.add_argument("--bond", required=True, help="bond as zero-based atom indices, e.g. 0-1")
    scan1d.add_argument(
        "--coordinate",
        choices=["bond", "normal", "normal-relaxed"],
        default="bond",
        help=(
            "scan the bond length, the normal mode most aligned with it, or that mode with "
            "all other coordinates relaxed (default: bond)"
        ),
    )
    scan1d.add_argument(
        "--range",
        nargs=3,
        type=float,
        metavar=("MIN", "MAX", "NPTS"),
        default=None,
        help=(
            "grid in angstrom: bond lengths, or normal displacements (default: centred on "
            "the input geometry)"
        ),
    )
    scan1d.set_defaults(handler=_run_scan, dimension=1)

    scan2d = commands.add_parser(
        "scan2d",
        parents=[electronic, scan, execution],
        help="2D two-bond scan and DVR spectrum",
        description="Scan two bond lengths, then solve the coupled 2D vibrational problem.",
    )
    scan2d.add_argument("geometry", type=Path, help="geometry file (.xyz or .mmol)")
    scan2d.add_argument(
        "--bond", action="append", required=True, help="a bond (give exactly two), e.g. 0-1"
    )
    scan2d.add_argument(
        "--range",
        nargs=3,
        type=float,
        action="append",
        metavar=("MIN", "MAX", "NPTS"),
        default=None,
        help="grid for the corresponding --bond (give none or two)",
    )
    scan2d.add_argument(
        "--keo",
        choices=["gmatrix", "reduced"],
        default="gmatrix",
        help="kinetic operator with or without the Wilson G cross term (default: gmatrix)",
    )
    scan2d.set_defaults(handler=_run_scan, dimension=2)

    inspect = commands.add_parser(
        "inspect", help="describe a scan file or a point store", description="Describe a file."
    )
    inspect.add_argument("path", type=Path, help="a .npz scan/grid file or a .sqlite point store")
    inspect.add_argument("--debug", action="store_true", help="show full tracebacks")
    inspect.set_defaults(handler=_run_inspect, verbose=False, json_out=None)
    return parser


def _prepare(args: argparse.Namespace, parser: argparse.ArgumentParser, log) -> _Context:
    if args.command is None:
        parser.error("choose a command: " + ", ".join(SUBCOMMANDS))
    if args.command == "inspect":
        return _Context(args=args, molecule=None, log=log)
    for name in ("max_parallel", "workers", "max_memory"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    for option in ("json_out", "output"):
        path = getattr(args, option, None)
        if path is not None:
            if not path.parent.is_dir():
                parser.error(f"output directory does not exist: {path.parent}")
            if path.is_dir():
                parser.error(f"output path is a directory: {path}")
    if args.json_out is not None and args.json_out.suffix.lower() != ".json":
        parser.error("--json must end with .json")
    if args.recipe is not None and any(
        getattr(args, name) is not None for flag, name in _METHOD_OPTIONS[:-1]
    ):
        parser.error("--recipe replaces --method/--basis/--ri/... ; do not combine them")

    molecule = _load_molecule(args)
    context = _Context(args=args, molecule=molecule, log=log)
    if args.command in {"scan1d", "scan2d"}:
        _prepare_scan(context, parser)
    if args.command == "opt":
        if args.output is not None and args.output.suffix.lower() not in {".xyz", ".mmol"}:
            parser.error("--output must end with .xyz or .mmol")
        if args.max_steps < 1:
            parser.error("--max-steps must be at least 1")
    _prepare_method(context)
    return context


def _prepare_scan(context: _Context, parser: argparse.ArgumentParser) -> None:
    import numpy as np

    from .coordinates import parse_bond

    args = context.args
    n_atoms = len(context.molecule.symbols)
    bonds_text = [args.bond] if args.dimension == 1 else list(args.bond)
    if args.dimension == 2 and len(bonds_text) != 2:
        parser.error("scan2d needs exactly two --bond options")
    bonds = []
    for text in bonds_text:
        try:
            bond = parse_bond(text)
        except ValueError:
            parser.error(f"--bond must be two zero-based atom indices such as 0-1 (got {text!r})")
        if not (0 <= bond.i < n_atoms and 0 <= bond.j < n_atoms) or bond.i == bond.j:
            parser.error(
                f"--bond {text} must name two different atoms in 0..{n_atoms - 1} "
                "(indices are zero-based)"
            )
        bonds.append(bond)
    if args.dimension == 2 and {bonds[0].i, bonds[0].j} == {bonds[1].i, bonds[1].j}:
        parser.error("the two --bond options must be different bonds")
    context.bonds = bonds
    ranges = args.range
    if args.dimension == 1 and ranges is not None:
        ranges = [ranges]
    if args.dimension == 2 and ranges is not None and len(ranges) != 2:
        parser.error("give either no --range or one --range per --bond")
    grids = None
    if ranges is not None:
        grids = []
        for low, high, count in ranges:
            if not count.is_integer() or count < 3:
                parser.error("the NPTS of --range must be an integer of at least 3")
            if not (math.isfinite(low) and math.isfinite(high) and low < high):
                parser.error("--range must satisfy MIN < MAX")
            if getattr(args, "coordinate", "bond") == "bond" and low <= 0.0:
                parser.error("bond-length ranges must be positive")
            if getattr(args, "coordinate", "bond") != "bond" and not low < 0.0 < high:
                parser.error("normal-coordinate ranges must satisfy MIN < 0 < MAX")
            grids.append(np.linspace(low, high, int(count)))
    context.grids = grids
    if args.states < 1:
        parser.error("--states must be at least 1")
    if not (math.isfinite(args.boundary_warn) and args.boundary_warn > 0.0):
        parser.error("--boundary-warn must be a positive number")
    if args.output is not None and args.output.suffix.lower() != ".npz":
        parser.error("--output must end with .npz")
    if getattr(args, "coordinate", "bond") == "normal-relaxed" and any(
        getattr(args, name) is not None for name in ("store", "output", "reuse")
    ):
        parser.error("--coordinate normal-relaxed does not support --store, --output, or --reuse")
    if args.reuse is not None:
        if args.store is not None or args.output is not None:
            parser.error(
                "--reuse reads a finished scan; it cannot be combined with --store/--output"
            )
        explicit = [flag for flag, name in _METHOD_OPTIONS if getattr(args, name) is not None]
        if explicit:
            parser.error(
                f"--reuse takes the method from the scan file; remove {', '.join(explicit)}"
            )
        if not args.reuse.is_file():
            parser.error(f"scan file not found: {args.reuse}")


def _prepare_method(context: _Context) -> None:
    from .backends.pyscf import PySCFMeanFieldProvider, SettingsFactory
    from .settings import (
        DEFAULT_MAX_PARALLEL,
        MAX_MEMORY_ENV_VAR,
        ESSettings,
        apply_thread_env_updates,
        available_cpu_count,
        available_memory_mb,
        derive_parallel_settings,
    )
    import os

    args = context.args
    if args.recipe is not None:
        context.factory = _load_recipe(args.recipe)
    else:
        defaults = ESSettings()
        context.settings = ESSettings(
            method=defaults.method if args.method is None else args.method,
            basis=defaults.basis if args.basis is None else args.basis,
            use_density_fit=defaults.use_density_fit if args.use_ri is None else args.use_ri,
            auxbasis=args.ri_aux,
            scf_conv_tol=args.scf_conv_tol,
            scf_max_cycle=args.scf_max_cycle,
            dft_grid_level=args.dft_grid_level,
        )
        context.factory = SettingsFactory(context.settings)

    max_parallel = args.max_parallel or available_cpu_count() or DEFAULT_MAX_PARALLEL
    computing = args.command in {"scan1d", "scan2d"} and args.reuse is None
    n_points = None
    if computing:
        from .scans import DEFAULT_BOND_GRID_POINTS

        if context.grids:
            n_points = int(math.prod(grid.size for grid in context.grids))
        else:
            n_points = DEFAULT_BOND_GRID_POINTS ** int(args.dimension)
    parallel = derive_parallel_settings(
        args.command if computing else "harmonic",
        max_parallel=max_parallel,
        pes_workers=args.workers,
        n_points=n_points,
        available_memory_mb=available_memory_mb() if computing and args.workers is None else None,
        max_memory_mb=args.max_memory,
    )
    context.parallel = parallel
    apply_thread_env_updates(parallel.threads_per_worker, os.environ)
    if parallel.max_memory_mb is not None:
        os.environ[MAX_MEMORY_ENV_VAR] = str(parallel.max_memory_mb)
    provider_kwargs = {"threads": parallel.threads_per_worker or 1}
    if parallel.max_memory_mb is not None:
        provider_kwargs["max_memory_mb"] = parallel.max_memory_mb
    if args.recipe is not None:
        context.provider = PySCFMeanFieldProvider(factory=context.factory, **provider_kwargs)
    else:
        context.provider = PySCFMeanFieldProvider(context.settings, **provider_kwargs)


# Electronic-structure options, as (flag, argparse destination); --recipe last.
_METHOD_OPTIONS = (
    ("--method", "method"),
    ("--basis", "basis"),
    ("--ri/--no-ri", "use_ri"),
    ("--ri-aux", "ri_aux"),
    ("--scf-conv-tol", "scf_conv_tol"),
    ("--scf-max-cycle", "scf_max_cycle"),
    ("--dft-grid-level", "dft_grid_level"),
    ("--recipe", "recipe"),
)


def _load_recipe(spec: str) -> Any:
    from .backends.pyscf import MeanFieldFactory

    module_name, _, name = spec.partition(":")
    if not module_name or not name:
        raise _CLIError("--recipe must look like MODULE:NAME")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise _CLIError(f"cannot import recipe module {module_name!r}: {exc}") from exc
    recipe = getattr(module, name, None)
    if recipe is None:
        raise _CLIError(f"module {module_name!r} has no attribute {name!r}")
    if not isinstance(recipe, MeanFieldFactory):
        raise _CLIError(f"{spec} is not a method recipe (MethodRecipe or MeanFieldFactory)")
    return recipe


def _load_molecule(args: argparse.Namespace) -> Any:
    path = args.geometry
    if not path.is_file():
        raise _CLIError(f"geometry file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".mmol":
        from .io import read_midas_mmol

        molecule = read_midas_mmol(path)
    elif suffix == ".xyz":
        from .io import read_xyz

        molecule = read_xyz(path)
    else:
        raise _CLIError(f"unknown geometry format {path.suffix!r}; use .xyz or .mmol")
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


def _base_payload(context: _Context) -> dict[str, Any]:
    from .cache import molecule_provenance

    return {
        "schema": CLI_RESULT_SCHEMA,
        "schema_version": CLI_RESULT_SCHEMA_VERSION,
        "package_version": _package_version(),
        "command": context.args.command,
        "input": str(context.args.geometry),
        "molecule": molecule_provenance(context.molecule),
        "method": dict(context.provider.scientific_settings_payload()),
        "provider_scientific_fingerprint": _provider_fingerprint(context.provider),
    }


def _provider_fingerprint(provider: Any) -> str:
    from .contracts import provider_scientific_fingerprint

    return provider_scientific_fingerprint(provider)


def _run_harmonic(context: _Context) -> dict[str, Any]:
    from .workflows import harmonic as harmonic_workflow

    args = context.args
    result = harmonic_workflow.harmonic_analysis(
        context.molecule,
        context.factory,
        rtproj=args.rtproj,
        strict=args.strict,
        allow_fd_hessian=args.allow_fd_hessian,
        debug=args.verbose,
    )
    print(f"ZPE (harmonic): {result.zpe_cm:.2f} cm^-1")
    print("Frequencies (cm^-1):")
    print(" ".join(f"{float(freq):.1f}" for freq in result.freqs_cm if float(freq) > 1e-2))
    payload = _base_payload(context)
    payload["harmonic"] = {
        "zpe_cm": float(result.zpe_cm),
        "frequencies_cm": [float(freq) for freq in result.freqs_cm],
        "hessian_provenance": getattr(result, "hessian_provenance", None),
        "rtproj": args.rtproj,
    }
    return payload


def _run_opt(context: _Context) -> dict[str, Any]:
    from .workflows import optimization
    from .workflows.optimization import format_optimization_stationarity_diagnostic

    args = context.args
    output = args.output
    if output is None:
        output = args.geometry.with_name(f"{args.geometry.stem}-opt{args.geometry.suffix}")
    result = optimization.optimize_geometry(
        context.molecule,
        context.factory,
        output_path=output,
        max_steps=args.max_steps,
        strict=args.strict,
    )
    if result.stationarity is not None:
        print(format_optimization_stationarity_diagnostic(result.stationarity))
    print(f"Optimized geometry written to: {result.output_path}")
    payload = _base_payload(context)
    payload["optimization"] = {
        "converged": bool(result.converged),
        "backend": result.backend,
        "output_path": str(result.output_path),
        "coordinates_A": [[float(x) for x in row] for row in result.molecule.coords],
        "max_gradient_component_Eh_per_bohr": None
        if result.stationarity is None
        else float(result.stationarity.max_component),
    }
    return payload


def _run_scan(context: _Context) -> dict[str, Any]:
    import numpy as np

    from .variational import parse_intensity_mode

    args = context.args
    source = "computed"
    if getattr(args, "coordinate", "bond") == "normal-relaxed":
        return _run_relaxed_scan(context)
    if args.reuse is not None:
        scan = _load_reused_scan(context)
        source = f"file:{args.reuse}"
    else:
        scan = _compute_scan(context)
        if args.output is not None:
            from .artifacts.scans import dump_scan

            dump_scan(args.output, scan)
            context.log(f"Wrote scan: {args.output}")
    records = _spectrum(context, scan)
    normal = getattr(args, "coordinate", "bond") == "normal"
    _print_records(
        records,
        parse_intensity_mode(args.intensity),
        args.boundary_warn,
        two_d=args.dimension == 2,
    )
    _warn_boundary(records, args.boundary_warn, "--range" if not normal else "--range (normal)")
    payload = _base_payload(context)
    if args.reuse is not None:
        payload["method"], payload["provider_scientific_fingerprint"] = _scan_method(scan)
    axes = getattr(scan, "axes", ())
    payload["scan"] = {
        "kind": getattr(scan, "scan_kind", "legacy"),
        "bonds_zero_based": [[bond.i, bond.j] for bond in context.bonds],
        "axes_A": [
            [float(v) for v in np.asarray(getattr(axis, "values", axis), dtype=float)]
            for axis in axes
        ],
        "source": source,
        "keo": getattr(args, "keo", None),
    }
    payload["transitions"] = [record.as_dict(extended=True) for record in records]
    return payload


def _run_relaxed_scan(context: _Context) -> dict[str, Any]:
    """Constrained relaxed normal-coordinate scan (not yet on the plan runner)."""

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial

    import numpy as np

    from ._cli_common import _worker_init
    from .backends.pyscf import normal_relaxed_point
    from .variational import parse_intensity_mode, variational_1d
    from .workflows.scans import (
        calc_normal_mode_direction,
        grid_1d_pes_dms_normal_relaxed,
        normal_mode_effective_mass_amu,
    )

    args = context.args
    molecule = context.molecule
    bond = context.bonds[0]
    direction = calc_normal_mode_direction(molecule, context.factory, bond, log_fn=context.log)
    grid = context.grids[0] if context.grids else _default_normal_grid()
    parallel = context.parallel
    executor_factory = None
    if parallel.workers and parallel.workers > 1:
        executor_factory = partial(
            ProcessPoolExecutor,
            max_workers=parallel.workers,
            mp_context=mp.get_context("spawn"),
            initializer=_worker_init,
            initargs=(parallel.threads_per_worker, parallel.max_memory_mb),
        )
    relaxed = grid_1d_pes_dms_normal_relaxed(
        molecule,
        context.factory,
        direction.u_dir,
        float(grid[0]),
        float(grid[-1]),
        int(grid.size),
        relaxed_point_fn=partial(normal_relaxed_point, strict=True),
        executor_factory=executor_factory,
        log_fn=context.log,
        stop_on_error=not args.continue_on_error,
    )
    axis = np.asarray(molecule.coords[bond.j]) - np.asarray(molecule.coords[bond.i])
    records = variational_1d(
        relaxed.displacements_A,
        relaxed.energies_hartree,
        relaxed.dipoles_debye,
        normal_mode_effective_mass_amu(molecule, direction.u_dir),
        axis=axis,
        vmax=args.states,
    )
    _print_records(records, parse_intensity_mode(args.intensity), args.boundary_warn, two_d=False)
    _warn_boundary(records, args.boundary_warn, "--range")
    payload = _base_payload(context)
    payload["scan"] = {
        "kind": "normal-relaxed-1d",
        "bonds_zero_based": [[bond.i, bond.j]],
        "axes_A": [[float(value) for value in relaxed.displacements_A]],
        "normal_mode_index": int(direction.mode_index),
        "harmonic_frequency_cm": float(direction.frequency_cm),
        "source": "computed",
    }
    payload["transitions"] = [record.as_dict(extended=True) for record in records]
    return payload


def _compute_scan(context: _Context) -> Any:
    from . import scans
    from .point_store import PointStore
    from .runner import RunnerSettings

    args = context.args
    molecule = context.molecule
    provider = context.provider
    if getattr(args, "coordinate", "bond") == "normal":
        from .workflows.scans import calc_normal_mode_direction

        direction = calc_normal_mode_direction(
            molecule, context.factory, context.bonds[0], log_fn=context.log
        )
        grid = context.grids[0] if context.grids else _default_normal_grid()
        plan = scans.plan_normal_scan(molecule, provider, direction.u_dir, grid)
        context.normal_direction = direction
    else:
        plan = scans.plan_bond_scan(molecule, provider, context.bonds, context.grids)
    parallel = context.parallel
    workers = parallel.workers if parallel.workers and parallel.workers > 1 else None
    runner = RunnerSettings(
        workers=workers,
        threads_per_worker=parallel.threads_per_worker if workers else None,
        max_memory_mb=parallel.max_memory_mb,
        stop_on_error=not args.continue_on_error,
        retry_failed=args.retry_failed,
    )
    context.log(
        f"Evaluating {len(plan.requests)} point(s) "
        f"({'serial' if workers is None else f'{workers} workers x {parallel.threads_per_worker} threads'})"
    )

    def progress(event: Any) -> None:
        if args.verbose:
            print(f"[INFO] point {event.completed}/{event.total}: {event.status}", flush=True)

    store = PointStore(args.store) if args.store is not None else None
    try:
        return scans.run_scan(plan, provider, store=store, settings=runner, progress=progress)
    except scans.PlanIncompleteError as exc:
        message = str(exc)
        if args.store is not None:
            message += f"\ncompleted points are kept in {args.store}; rerun to resume"
        else:
            message += "\nuse --store FILE to keep completed points and resume later"
        if not args.continue_on_error and exc.run.not_attempted:
            message += "\nuse --continue-on-error to attempt the remaining points"
        error = _CLIError(message)
        error.plan_run = exc.run
        raise error from exc
    finally:
        if store is not None:
            store.close()


def _default_normal_grid():
    import numpy as np

    return np.linspace(-0.15, 0.15, 41)


def _load_reused_scan(context: _Context) -> Any:
    from .artifacts.scans import load_legacy_grid, load_scan

    path = context.args.reuse
    try:
        scan = load_scan(path)
    except ValueError as exc:
        if "load_legacy_grid" not in str(exc):
            raise
        scan = load_legacy_grid(path)
        context.log(f"Read legacy (relative-energy) grid cache {path}")
    return scan


def _spectrum(context: _Context, scan: Any) -> list[Any]:
    import numpy as np

    from . import scans

    args = context.args
    molecule = context.molecule
    coordinate = getattr(args, "coordinate", "bond")
    if isinstance(scan, scans.ScanResult):
        expected = "normal-1d" if coordinate == "normal" else f"bond-{args.dimension}d"
        if scan.scan_kind != expected:
            raise _CLIError(f"the scan file holds a {scan.scan_kind} scan, not {expected}")
        try:
            scans._check_same_system(scan, molecule)
        except ValueError as exc:
            raise _CLIError(str(exc)) from exc
        if scan.scan_kind == "normal-1d":
            bond = context.bonds[0]
            axis = np.asarray(molecule.coords[bond.j]) - np.asarray(molecule.coords[bond.i])
            return scans.normal_coordinate_spectrum(
                scan, molecule, max_state_index=args.states, intensity_axis=axis
            )
        _check_scan_bonds(scan, context.bonds)
        return scans.bond_spectrum(
            scan,
            molecule,
            max_state_index=args.states,
            kinetic_operator=getattr(args, "keo", "gmatrix"),
        )
    # A 0.2 relative-energy grid: check that it belongs to this geometry and bonds.
    from .constants import atomic_number
    from .variational import variational_1d, variational_2d
    from .workflows.scans import bond_bond_g12_inv_amu, local_bond_reduced_mass_amu

    if coordinate != "bond" or scan.scan_kind != f"legacy-{args.dimension}d":
        raise _CLIError(
            f"the 0.2 grid cache holds a {scan.scan_kind} bond scan, not a "
            f"{args.dimension}D {coordinate} scan"
        )
    identity = scan.metadata["identity"]
    stored_molecule = identity["molecule"]
    reference = np.asarray(stored_molecule["coordinates_A"], dtype=float)
    # Schema-3 identities store deuterium as hydrogen: compare nuclear charges.
    if (
        [atomic_number(symbol) for symbol in stored_molecule["symbols"]]
        != [atomic_number(symbol) for symbol in molecule.symbols]
        or reference.shape != np.asarray(molecule.coords).shape
        or not np.allclose(reference, molecule.coords, rtol=0.0, atol=1e-10)
    ):
        raise _CLIError("the grid cache was computed for a different geometry")
    if (int(stored_molecule.get("charge", 0)), int(stored_molecule.get("spin", 0))) != (
        molecule.charge,
        molecule.spin,
    ):
        raise _CLIError("the grid cache was computed for a different charge or spin")
    stored = identity["scan"]
    keys = ["bond_zero_based"] if args.dimension == 1 else ["bond1_zero_based", "bond2_zero_based"]
    for key, bond in zip(keys, context.bonds):
        stored_bond = stored.get(key)
        if stored_bond is None or tuple(stored_bond) != (bond.i, bond.j):
            raise _CLIError(f"the grid cache scans bond {stored_bond}, not {bond.i}-{bond.j}")
    bond0 = context.bonds[0]
    axis = np.asarray(molecule.coords[bond0.j]) - np.asarray(molecule.coords[bond0.i])
    if args.dimension == 1:
        return variational_1d(
            scan.axes[0],
            scan.relative_energies_Eh,
            scan.dipoles_debye,
            local_bond_reduced_mass_amu(molecule, bond0),
            axis=axis,
            vmax=args.states,
        )
    g12 = (
        bond_bond_g12_inv_amu(molecule, context.bonds[0], context.bonds[1])
        if args.keo == "gmatrix"
        else 0.0
    )
    return variational_2d(
        scan.axes[0],
        scan.axes[1],
        scan.relative_energies_Eh,
        scan.dipoles_debye,
        local_bond_reduced_mass_amu(molecule, context.bonds[0]),
        local_bond_reduced_mass_amu(molecule, context.bonds[1]),
        axis=axis,
        nmax=args.states,
        g12_inv_amu=g12,
    )


def _scan_method(scan: Any) -> tuple[dict[str, Any], str | None]:
    """Return the method payload and provider fingerprint recorded in a reused scan."""

    from ._identity import to_jsonable

    if hasattr(scan, "provider_payload"):
        return (
            cast(dict[str, Any], to_jsonable(scan.provider_payload)),
            scan.provider_scientific_fingerprint,
        )
    electronic = scan.metadata["identity"]["electronic_structure"]
    return cast(dict[str, Any], to_jsonable(electronic)), None


def _check_scan_bonds(scan: Any, bonds: Sequence[Any]) -> None:
    stored = [axis.bond for axis in scan.axes]
    wanted = [(bond.i, bond.j) for bond in bonds]
    if stored != wanted:
        raise _CLIError(f"the scan file covers bonds {stored}, not {wanted}")


def _run_inspect(context: _Context) -> None:
    path = context.args.path
    if not path.is_file():
        raise _CLIError(f"file not found: {path}")
    if path.suffix.lower() in {".sqlite", ".db"}:
        _inspect_store(path)
        return None
    from .artifacts.scans import load_legacy_grid, load_scan

    try:
        scan = load_scan(path)
    except ValueError as exc:
        if "load_legacy_grid" not in str(exc):
            raise
        legacy = load_legacy_grid(path)
        identity = legacy.metadata["identity"]
        print(f"{path}: 0.2 grid cache ({legacy.scan_kind}), energies relative to the minimum")
        print(f"  method: {identity['electronic_structure']}")
        print(
            f"  scan: { {k: v for k, v in identity['scan'].items() if not k.startswith('grid')} }"
        )
        for index, axis in enumerate(legacy.axes):
            print(f"  axis {index}: {axis.size} points, {axis[0]:.4f}-{axis[-1]:.4f} A")
        return None
    print(f"{path}: scan ({scan.scan_kind}), schema 4")
    method = scan.provider_payload.get("settings") or scan.provider_payload.get("method")
    print(f"  method: {dict(method) if method is not None else '?'}")
    for axis in scan.axes:
        print(
            f"  axis {axis.label}: {axis.values.size} points, "
            f"{axis.values[0]:.4f}-{axis.values[-1]:.4f} {axis.unit}"
        )
    print(f"  energy range: {float(scan.relative_energies_Eh.max()):.6f} Eh above the minimum")
    return None


def _inspect_store(path: Path) -> None:
    import sqlite3

    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        counts = {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("providers", "points", "results", "runs")
        }
        failed = connection.execute(
            "SELECT COUNT(DISTINCT point_key) FROM attempts a WHERE status = 'failed' AND "
            "attempt_id = (SELECT MAX(attempt_id) FROM attempts b WHERE b.point_key = a.point_key)"
        ).fetchone()[0]
    finally:
        connection.close()
    print(
        f"{path}: point store with {counts['results']} stored result(s) for "
        f"{counts['providers']} method(s), {failed} point(s) whose latest attempt failed, "
        f"{counts['runs']} run(s)"
    )


def _print_first_traceback(exc: BaseException) -> None:
    run = getattr(exc, "plan_run", None)
    if run is None:
        return
    for failure in run.failures.values():
        if failure.traceback:
            print(f"Traceback of failed point {failure.point_key[:12]}:", file=sys.stderr)
            print(failure.traceback, file=sys.stderr)
            return


if __name__ == "__main__":
    raise SystemExit(main())
