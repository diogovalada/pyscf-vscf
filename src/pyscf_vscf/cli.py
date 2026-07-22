"""Command-line entry point for package-native PySCF local-stretch DVR workflows."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence


def _package_version() -> str:
    try:
        return version("pyscf-vscf")
    except PackageNotFoundError:
        from . import __version__

        return __version__


def _build_parser() -> argparse.ArgumentParser:
    from .settings import DEFAULT_MAX_PARALLEL, DEFAULT_PES_WORKERS, ESSettings

    defaults = ESSettings()
    parser = argparse.ArgumentParser(
        prog="pyscf-vscf",
        description="PySCF-backed reduced-dimensional local-stretch DVR workflow driver.",
    )
    parser.add_argument("--version", action="store_true", help="print the package version")
    parser.add_argument("--mmol", type=Path, help="Midas .mmol geometry input")
    parser.add_argument("--xyz", type=Path, help="XYZ geometry input")
    parser.add_argument("--charge", type=int, default=None, help="Total molecular charge")
    parser.add_argument(
        "--spin",
        type=int,
        default=None,
        help="PySCF spin value (number of alpha electrons minus beta electrons)",
    )
    parser.add_argument("--task", choices=["harmonic", "opt", "1d", "2d"], default="harmonic")
    parser.add_argument("--basis", default=defaults.basis)
    parser.add_argument("--method", default=defaults.method)
    parser.add_argument(
        "--dispersion",
        choices=["d3", "d4", "none"],
        default=defaults.dispersion or "none",
        help="DFT dispersion correction; use 'none' to disable",
    )
    parser.add_argument(
        "--rtproj",
        choices=["pyscf", "mw_explicit", "none"],
        default=defaults.rtproj,
        help="RT projection for harmonic frequencies",
    )
    parser.add_argument(
        "--no-ri",
        dest="use_ri",
        action="store_false",
        help="Disable RI density fitting",
    )
    parser.add_argument("--ri-aux", dest="ri_aux", help="Auxiliary basis for RI density fitting")
    parser.add_argument("--scf-conv-tol", type=float, default=None)
    parser.add_argument("--scf-max-cycle", type=int, default=None)
    parser.add_argument("--dft-grid-level", type=int, default=None)
    parser.add_argument("--bond", default="0-1")
    parser.add_argument(
        "--scan",
        choices=[
            "lbs-frozen",
            "normal",
            "normal-relaxed",
        ],
        default="lbs-frozen",
        help="1D scan type",
    )
    parser.add_argument("--bond2")
    parser.add_argument(
        "--keo",
        choices=["reduced", "gmatrix"],
        default="gmatrix",
        help="2D kinetic energy operator",
    )
    parser.add_argument("--rmin", type=float, default=0.75)
    parser.add_argument("--rmax", type=float, default=1.25)
    parser.add_argument(
        "--smin",
        type=float,
        default=-0.15,
        help="Minimum normal-coordinate displacement in angstrom",
    )
    parser.add_argument(
        "--smax",
        type=float,
        default=0.15,
        help="Maximum normal-coordinate displacement in angstrom",
    )
    parser.add_argument("--npts", type=int, default=41)
    parser.add_argument("--vmax", type=int, default=8)
    parser.add_argument("--dump-grid", type=Path, default=None)
    parser.add_argument("--reuse-grid", type=Path, default=None)
    parser.add_argument(
        "--intensity",
        choices=["axis", "vector", "both"],
        default="axis",
        help="Transition-dipole convention for variational intensities",
    )
    parser.add_argument(
        "--opt-out",
        type=Path,
        default=None,
        help="Optimization output path (.mmol or .xyz)",
    )
    parser.add_argument("--opt-maxsteps", type=int, default=50)
    parser.add_argument(
        "--opt-conv",
        choices=["orca", "orca-tight", "orca_tight"],
        default="orca",
    )
    parser.add_argument("--max-parallel", type=int, default=DEFAULT_MAX_PARALLEL)
    parser.add_argument("--pes-workers", type=int, default=DEFAULT_PES_WORKERS)
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument("--strict", dest="strict", action="store_true")
    strict_group.add_argument("--no-strict", dest="strict", action="store_false")
    parser.set_defaults(strict=True, use_ri=True)
    parser.add_argument("--allow-fd-hessian", action="store_true")
    parser.add_argument("--dev-fast", action="store_true")
    parser.add_argument("--fast-npts", type=int, default=21)
    parser.add_argument("--fast-width", type=float, default=0.20)
    parser.add_argument("--tight-scan", action="store_true")
    parser.add_argument("--tight-width", type=float, default=0.30)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    if args.version:
        print(_package_version())
        return 0
    if args.dump_grid is not None and args.reuse_grid is not None:
        parser.error("Use only one of --dump-grid or --reuse-grid (not both)")

    from .settings import (
        apply_thread_env_updates,
        default_auxbasis,
        derive_parallel_settings,
        format_runtime,
    )

    try:
        runtime = derive_parallel_settings(
            args.task,
            max_parallel=args.max_parallel,
            pes_workers=args.pes_workers,
            verbose=args.verbose,
            dev_fast=args.dev_fast,
            strict=args.strict,
            allow_fd_hessian=args.allow_fd_hessian,
        )
    except ValueError as exc:
        parser.error(str(exc))

    apply_thread_env_updates(runtime.threads_per_worker, os.environ)
    log = _log_fn(args.verbose)
    executor_factory = _executor_factory(runtime.workers, runtime.threads_per_worker)

    start = time.perf_counter()
    try:
        molecule = _load_molecule(args, parser)
        cfg, npts, tight_width = _build_es_settings(args)

        log(f"Selected task: {args.task}")
        log(
            f"Method: {cfg.method} | Basis: {cfg.basis} | "
            f"RI {'enabled' if cfg.use_density_fit else 'disabled'}"
        )
        log(f"Dispersion: {cfg.dispersion}")
        log(
            "Parallel: "
            f"max_parallel={runtime.max_parallel} pes_workers={runtime.pes_workers} "
            f"=> workers={runtime.workers} threads={runtime.threads_per_worker} (BLAS=1)"
        )
        if cfg.use_density_fit:
            log(f"RI auxiliary basis: {cfg.auxbasis or default_auxbasis(cfg.basis)}")
        if args.dev_fast:
            log(
                "FAST MODE: "
                f"method={cfg.method}, basis={cfg.basis}, npts={npts}, "
                f"tight-width={tight_width:.2f}"
            )

        if args.task == "harmonic":
            _run_harmonic(molecule, cfg, verbose=args.verbose)
        elif args.task == "opt":
            _run_opt(
                molecule,
                cfg,
                opt_out=args.opt_out,
                opt_maxsteps=args.opt_maxsteps,
                opt_conv=args.opt_conv,
                verbose=args.verbose,
                log_fn=log,
            )
        elif args.task == "1d":
            _run_1d(
                molecule,
                cfg,
                bond=args.bond,
                rmin=args.rmin,
                rmax=args.rmax,
                smin=args.smin,
                smax=args.smax,
                npts=npts,
                vmax=args.vmax,
                tight_scan=args.tight_scan,
                tight_width=tight_width,
                scan=args.scan,
                dump_grid=args.dump_grid,
                reuse_grid=args.reuse_grid,
                intensity=args.intensity,
                log_fn=log,
                executor_factory=executor_factory,
            )
        elif args.task == "2d":
            if not args.bond2:
                parser.error("--bond2 required for 2D")
            _run_2d(
                molecule,
                cfg,
                bond=args.bond,
                bond2=args.bond2,
                rmin=args.rmin,
                rmax=args.rmax,
                npts=npts,
                nmax=args.vmax,
                keo=args.keo,
                dump_grid=args.dump_grid,
                reuse_grid=args.reuse_grid,
                intensity=args.intensity,
                log_fn=log,
                executor_factory=executor_factory,
            )
        return 0
    finally:
        elapsed = time.perf_counter() - start
        print(f"Runtime: {elapsed:.2f} s ({format_runtime(elapsed)})")


def _load_molecule(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Any:
    if bool(args.mmol) == bool(args.xyz):
        parser.error("Provide exactly one of --mmol or --xyz")

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


def _build_es_settings(args: argparse.Namespace) -> tuple[Any, int, float]:
    from .settings import ESSettings, apply_development_fast_policy, normalize_dispersion

    cfg = ESSettings(
        method=args.method,
        basis=args.basis,
        use_density_fit=args.use_ri,
        auxbasis=args.ri_aux,
        dispersion=normalize_dispersion(args.dispersion),
        rtproj=args.rtproj,
        strict=bool(args.strict),
        allow_fd_hessian=bool(args.allow_fd_hessian),
        scf_conv_tol=args.scf_conv_tol,
        scf_max_cycle=args.scf_max_cycle,
        dft_grid_level=args.dft_grid_level,
    )
    npts = int(args.npts)
    tight_width = float(args.tight_width)
    if args.dev_fast:
        dev = apply_development_fast_policy(
            cfg,
            npts=npts,
            tight_width=tight_width,
            fast_npts=args.fast_npts,
            fast_width=args.fast_width,
        )
        cfg = dev.es
        npts = npts if dev.npts is None else dev.npts
        tight_width = tight_width if dev.tight_width is None else dev.tight_width
    return cfg, npts, tight_width


def _run_harmonic(molecule: Any, cfg: Any, *, verbose: bool = False) -> None:
    from .workflows import harmonic as harmonic_workflow

    result = harmonic_workflow.harmonic_analysis(
        molecule,
        cfg,
        rtproj=getattr(cfg, "rtproj", "pyscf"),
        debug=verbose,
    )
    print(f"ZPE (harmonic): {result.zpe_cm:.2f} cm^-1")
    print("Frequencies (cm^-1):")
    print(" ".join(f"{float(freq):.1f}" for freq in result.freqs_cm if float(freq) > 1e-2))


def _run_opt(
    molecule: Any,
    cfg: Any,
    *,
    opt_out: Path | None,
    opt_maxsteps: int | None,
    opt_conv: str | None,
    verbose: bool,
    log_fn: Any,
) -> None:
    from .settings import warn_once
    from .workflows import optimization

    optimization.run_opt(
        molecule,
        cfg,
        opt_out=opt_out,
        opt_maxsteps=opt_maxsteps,
        opt_conv=opt_conv,
        verbose=verbose,
        log_fn=log_fn,
        warn_fn=warn_once,
    )


def _run_1d(
    molecule: Any,
    cfg: Any,
    *,
    bond: str,
    rmin: float,
    rmax: float,
    smin: float,
    smax: float,
    npts: int,
    vmax: int,
    tight_scan: bool,
    tight_width: float,
    scan: str,
    dump_grid: Path | None,
    reuse_grid: Path | None,
    intensity: str,
    log_fn: Any,
    executor_factory: Any = None,
) -> None:
    import numpy as np

    from .variational import parse_intensity_mode, variational_1d
    from .workflows import scans

    b = scans.normalize_bond(bond)
    axis_vec = np.asarray(molecule.coords[b.H], dtype=float) - np.asarray(
        molecule.coords[b.O],
        dtype=float,
    )

    if scan == "lbs-frozen":
        if tight_scan:
            r_eq = float(np.linalg.norm(axis_vec))
            half_width = 0.5 * float(tight_width)
            if half_width <= 0.0:
                raise ValueError("--tight-width must be positive")
            rmin = r_eq - half_width
            rmax = r_eq + half_width
            log_fn(f"Tight scan enabled: r_eq={r_eq:.4f} A; window [{rmin:.4f}, {rmax:.4f}] A")

        if reuse_grid is not None:
            R, E, MU = scans.load_lbs_frozen_1d_grid_cache(
                reuse_grid,
                molecule,
                cfg,
                b,
                rmin,
                rmax,
                npts,
                scan=scan,
            )
            log_fn(f"Reusing cached 1D grid from {reuse_grid}")
        else:
            _validate_npz_cache_path(dump_grid, "--dump-grid")
            R, E, MU = scans.grid_1d_pes_dms(
                molecule,
                cfg,
                b,
                rmin,
                rmax,
                npts,
                executor_factory=executor_factory,
                log_fn=log_fn,
            )
            if dump_grid is not None:
                scans.dump_lbs_frozen_1d_grid_cache(
                    dump_grid,
                    molecule,
                    cfg,
                    b,
                    rmin,
                    rmax,
                    npts,
                    R,
                    E,
                    MU,
                    scan=scan,
                )
                log_fn(f"Wrote 1D grid cache: {dump_grid}")
        redmass_amu = scans.local_bond_reduced_mass_amu(molecule, b)
        normal_summary = None

    elif scan in {"normal", "normal-relaxed"}:
        if dump_grid is not None or reuse_grid is not None:
            raise NotImplementedError(
                "Grid caching is only implemented for --scan lbs-frozen in --task 1d."
            )

        u_dir, mode_index, freq_cm, _modes, _freqs_cm = scans.calc_normal_mode_direction(
            molecule,
            cfg,
            b,
            log_fn=log_fn,
        )
        if tight_scan:
            half_width = 0.5 * float(tight_width)
            if half_width <= 0.0:
                raise ValueError("--tight-width must be positive")
            smin = -half_width
            smax = half_width
            log_fn(f"Tight scan (normal) enabled: window [{smin:.4f}, {smax:.4f}] A")
        else:
            smin = float(smin)
            smax = float(smax)
            if not smin < 0.0 < smax:
                raise ValueError("Normal-coordinate bounds must satisfy --smin < 0 < --smax")

        if scan == "normal":
            R, E, MU = scans.grid_1d_pes_dms_normal(
                molecule,
                cfg,
                u_dir,
                smin,
                smax,
                npts,
                executor_factory=executor_factory,
                log_fn=log_fn,
            )
        else:
            from .backends import pyscf as pyscf_backend

            R, E, MU = scans.grid_1d_pes_dms_normal_relaxed(
                molecule,
                cfg,
                u_dir,
                smin,
                smax,
                npts,
                relaxed_point_fn=pyscf_backend.normal_relaxed_point,
                executor_factory=executor_factory,
                log_fn=log_fn,
            )
        redmass_amu = scans.normal_mode_effective_mass_amu(molecule, u_dir)
        normal_summary = (mode_index, freq_cm, redmass_amu)

    elif scan == "lbs-relaxed":
        raise NotImplementedError(
            "Scan type 'lbs-relaxed' is planned but not available in the alpha CLI."
        )
    else:
        raise ValueError(f"Unknown --scan '{scan}'")

    records = variational_1d(
        R,
        E,
        MU,
        redmass_amu,
        axis=axis_vec,
        vmax=vmax,
        intensity=intensity,
    )

    if normal_summary is not None:
        mode_index, freq_cm, mu_eff = normal_summary
        if records:
            print(
                f"Harmonic nu (mode {mode_index}) = {freq_cm:.1f} cm^-1; "
                f"DVR v=1 = {records[0]['freq_cm']:.1f} cm^-1; "
                f"mu_eff = {mu_eff:.6f} amu"
            )
        else:
            print(
                f"Harmonic nu (mode {mode_index}) = {freq_cm:.1f} cm^-1; mu_eff = {mu_eff:.6f} amu"
            )

    _print_1d_records(records, parse_intensity_mode(intensity))


def _run_2d(
    molecule: Any,
    cfg: Any,
    *,
    bond: str,
    bond2: str,
    rmin: float,
    rmax: float,
    npts: int,
    nmax: int,
    keo: str,
    dump_grid: Path | None,
    reuse_grid: Path | None,
    intensity: str,
    log_fn: Any,
    executor_factory: Any = None,
) -> None:
    import numpy as np

    from .cache import dump_grid_npz
    from .variational import parse_intensity_mode, variational_2d
    from .workflows import scans

    b1 = scans.normalize_bond(bond)
    b2 = scans.normalize_bond(bond2)
    r1 = (float(rmin), float(rmax), int(npts))
    r2 = (float(rmin), float(rmax), int(npts))
    R1_req = np.linspace(*r1)
    R2_req = np.linspace(*r2)

    if reuse_grid is not None:
        R1, R2, E, MU = scans.load_lbs_frozen_2d_grid_cache(
            reuse_grid,
            molecule,
            cfg,
            b1,
            b2,
            r1,
            r2,
            keo=keo,
        )
        log_fn(f"Reusing cached 2D grid from {reuse_grid}")
    else:
        _validate_npz_cache_path(dump_grid, "--dump-grid")
        R1, R2, E, MU = scans.grid_2d_pes_dms(
            molecule,
            cfg,
            b1,
            b2,
            R1_req,
            R2_req,
            executor_factory=executor_factory,
            log_fn=log_fn,
        )
        if dump_grid is not None:
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
                dump_grid,
                meta=meta,
                arrays={"R1_A": R1, "R2_A": R2, "E_Eh": E, "MU_Debye": MU},
            )
            log_fn(f"Wrote 2D grid cache: {dump_grid}")

    axis_vec = np.asarray(molecule.coords[b1.H], dtype=float) - np.asarray(
        molecule.coords[b1.O],
        dtype=float,
    )
    mu1 = scans.local_bond_reduced_mass_amu(molecule, b1)
    mu2 = scans.local_bond_reduced_mass_amu(molecule, b2)
    keo_lc = (keo or "gmatrix").lower()
    if keo_lc not in {"reduced", "gmatrix"}:
        raise ValueError(f"Unknown --keo '{keo}' (expected reduced|gmatrix)")

    g12_inv_amu = 0.0
    if keo_lc == "gmatrix":
        g12_inv_amu = scans.bond_bond_g12_inv_amu(molecule, b1, b2)

    records = variational_2d(
        R1,
        R2,
        E,
        MU,
        mu1,
        mu2,
        axis=axis_vec,
        nmax=nmax,
        g12_inv_amu=g12_inv_amu,
        intensity=intensity,
    )
    _print_2d_records(records, parse_intensity_mode(intensity))


def _print_1d_records(records: Sequence[dict[str, Any]], intensity: str) -> None:
    if intensity == "both":
        print(
            "v  nu/cm^-1   mu_axis/D   int_axis/domega (m^2/s)   |mu|/D   int_iso/domega (m^2/s)"
        )
        for record in records:
            print(
                f"{record['v']:>1d}  {record['freq_cm']:8.1f}   "
                f"{record['transition_dipole_axis_D']:>12.4e}   "
                f"{record['integrated_cross_section_axis_omega_m2_per_s']:>12.4e}   "
                f"{record['transition_dipole_norm_D']:>12.4e}   "
                f"{record['integrated_cross_section_isotropic_omega_m2_per_s']:>12.4e}"
            )
        return

    print("v  nu/cm^-1   mu/D   integral sigma(omega)domega (m^2/s)   orientation")
    for record in records:
        print(
            f"{record['v']:>1d}  {record['freq_cm']:8.1f}   "
            f"{record['transition_dipole_D']:>12.4e}   "
            f"{record['integrated_cross_section_omega_m2_per_s']:>12.4e}   "
            f"{record['orientation']}"
        )


def _print_2d_records(records: Sequence[dict[str, Any]], intensity: str) -> None:
    if intensity == "both":
        print(
            "n assignment weight  nu/cm^-1   mu_axis/D   int_axis/domega (m^2/s)   "
            "|mu|/D   int_iso/domega (m^2/s)"
        )
        for record in records:
            print(
                f"{record['n']:>1d} {str(record['assignment']):>10s} "
                f"{record['assignment_weight']:6.3f} {record['freq_cm']:8.1f}   "
                f"{record['transition_dipole_axis_D']:>12.4e}   "
                f"{record['integrated_cross_section_axis_omega_m2_per_s']:>12.4e}   "
                f"{record['transition_dipole_norm_D']:>12.4e}   "
                f"{record['integrated_cross_section_isotropic_omega_m2_per_s']:>12.4e}"
            )
        return

    print(
        "n assignment weight  nu/cm^-1   mu/D   integral sigma(omega)domega (m^2/s)   orientation"
    )
    for record in records:
        print(
            f"{record['n']:>1d} {str(record['assignment']):>10s} "
            f"{record['assignment_weight']:6.3f} {record['freq_cm']:8.1f}   "
            f"{record['transition_dipole_D']:>12.4e}   "
            f"{record['integrated_cross_section_omega_m2_per_s']:>12.4e}   "
            f"{record['orientation']}"
        )


def _validate_npz_cache_path(path: Path | None, option_name: str) -> None:
    if path is not None and path.suffix.lower() != ".npz":
        raise ValueError(f"{option_name} must end with .npz")


def _log_fn(verbose: bool):
    def emit(message: str) -> None:
        if verbose:
            print(f"[INFO] {message}", flush=True)

    return emit


class _SequentialExecutor:
    def __enter__(self) -> "_SequentialExecutor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def map(self, fn, iterable):
        return (fn(item) for item in iterable)


def _executor_factory(workers: int | None, threads_per_worker: int | None):
    if workers is None or int(workers) <= 1:
        return _SequentialExecutor

    worker_count = int(workers)

    def make_executor():
        return ProcessPoolExecutor(
            max_workers=worker_count,
            initializer=_worker_init,
            initargs=(threads_per_worker,),
            mp_context=_multiprocessing_context(),
        )

    return make_executor


def _multiprocessing_context():
    method = os.environ.get("VSCF_PYSCF_START_METHOD", "spawn")
    try:
        return mp.get_context(method)
    except ValueError:
        return mp.get_context("spawn")


def _worker_init(threads_per_worker: int | None) -> None:
    from .settings import apply_thread_env_updates

    apply_thread_env_updates(threads_per_worker, os.environ)
    try:
        from pyscf import lib
    except Exception:
        return
    if threads_per_worker is not None:
        lib.num_threads(int(threads_per_worker))


if __name__ == "__main__":
    raise SystemExit(main())
