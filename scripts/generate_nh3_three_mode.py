#!/usr/bin/env python3
"""Generate the archived non-water, three-mode NH3 validation surfaces."""

from __future__ import annotations

import argparse
import contextlib
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

import numpy as np

from pyscf_vscf import Bond, ESSettings, __version__
from pyscf_vscf.cache import runtime_provenance
from pyscf_vscf.io import dump_grid_npz, read_xyz
from pyscf_vscf.settings import apply_thread_env_updates
from pyscf_vscf.workflows.optimization import run_opt
from pyscf_vscf.workflows.scans import (
    grid_2d_pes_dms,
    lbs_frozen_2d_cache_metadata,
    load_lbs_frozen_2d_grid_cache,
)


PAIRS = ((0, 1), (0, 2), (1, 2))


class _Tee:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _worker_init(threads: int) -> None:
    apply_thread_env_updates(threads, os.environ)
    from pyscf import lib

    lib.num_threads(threads)


def _executor_factory(workers: int, threads: int):
    context = mp.get_context(os.environ.get("VSCF_PYSCF_START_METHOD", "spawn"))

    def make_executor() -> ProcessPoolExecutor:
        return ProcessPoolExecutor(
            max_workers=workers,
            initializer=_worker_init,
            initargs=(threads,),
            mp_context=context,
        )

    return make_executor


def _cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.partition(":")[2].strip()
    return platform.processor() or "unknown"


def _bond_length(coords: np.ndarray, atom: int) -> float:
    return float(np.linalg.norm(coords[atom] - coords[0]))


def _progress(done: int, total: int, label: str) -> None:
    interval = max(1, total // 20)
    if done == 1 or done == total or done % interval == 0:
        print(f"{label}: {done}/{total}", flush=True)


def _settings() -> ESSettings:
    return ESSettings(
        method="wb97x",
        basis="aug-cc-pVTZ",
        use_density_fit=True,
        auxbasis="aug-cc-pVTZ-jkfit",
        strict=True,
        scf_conv_tol=1e-10,
        scf_max_cycle=100,
        dft_grid_level=3,
    )


def generate(
    data_root: Path,
    *,
    npts: int,
    half_width_A: float,
    workers: int,
    threads_per_worker: int,
    force: bool,
    tag: str,
) -> dict:
    data_root.mkdir(parents=True, exist_ok=True)
    initial_path = data_root / "nh3_initial.xyz"
    optimized_path = data_root / "nh3_optimized.xyz"
    cfg = _settings()

    apply_thread_env_updates(max(1, workers * threads_per_worker), os.environ)
    initial = read_xyz(initial_path)
    if force or not optimized_path.exists():
        print("Optimizing NH3 geometry", flush=True)
        run_opt(
            initial,
            cfg,
            opt_out=optimized_path,
            opt_maxsteps=100,
            verbose=False,
            log_fn=lambda message: print(message, flush=True),
        )
    else:
        print(f"Reusing optimized geometry: {optimized_path}", flush=True)

    molecule = read_xyz(optimized_path)
    molecule.label = "NH3"
    bonds = (Bond(0, 1), Bond(0, 2), Bond(0, 3))
    centers = tuple(_bond_length(molecule.coords, atom) for atom in (1, 2, 3))
    grids = tuple(
        np.linspace(center - half_width_A, center + half_width_A, npts) for center in centers
    )
    executor_factory = _executor_factory(workers, threads_per_worker)

    pair_records = []
    total_start = time.perf_counter()
    artifact_suffix = f"_{tag}" if tag else ""
    for i, j in PAIRS:
        path = data_root / f"nh3_pair_{i + 1}{j + 1}_{npts}x{npts}{artifact_suffix}.npz"
        spec_i = (float(grids[i][0]), float(grids[i][-1]), npts)
        spec_j = (float(grids[j][0]), float(grids[j][-1]), npts)
        pair_start = time.perf_counter()
        reused = False
        if path.exists() and not force:
            try:
                r1, r2, energy, dipole = load_lbs_frozen_2d_grid_cache(
                    path,
                    molecule,
                    cfg,
                    bonds[i],
                    bonds[j],
                    spec_i,
                    spec_j,
                    keo="separable-local-reduced-mass",
                )
                reused = True
                print(f"Validated and reused {path.name}", flush=True)
            except ValueError as exc:
                print(f"Cache rejected ({exc}); regenerating {path.name}", flush=True)
        if not reused:
            print(f"Generating pair surface ({i}, {j}) -> {path.name}", flush=True)
            r1, r2, energy, dipole = grid_2d_pes_dms(
                molecule,
                cfg,
                bonds[i],
                bonds[j],
                grids[i],
                grids[j],
                executor_factory=executor_factory,
                progress_fn=_progress,
                log_fn=lambda message: print(message, flush=True),
            )
            metadata = lbs_frozen_2d_cache_metadata(
                molecule,
                cfg,
                bonds[i],
                bonds[j],
                spec_i,
                spec_j,
                keo="separable-local-reduced-mass",
            )
            dump_grid_npz(
                path,
                meta=metadata,
                arrays={"R1_A": r1, "R2_A": r2, "E_Eh": energy, "MU_Debye": dipole},
            )
        pair_records.append(
            {
                "pair_zero_based": [i, j],
                "path": path.name,
                "reused": reused,
                "elapsed_seconds": time.perf_counter() - pair_start,
                "energy_range_Eh": float(np.ptp(energy)),
            }
        )

    summary = {
        "purpose": "non-water three-local-mode VSCF validation",
        "package_version": __version__,
        "electronic_structure": asdict(cfg),
        "runtime": runtime_provenance(),
        "hardware": {
            "cpu": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "workers": workers,
            "threads_per_worker": threads_per_worker,
        },
        "grid": {
            "npts_per_mode": npts,
            "half_width_A": half_width_A,
            "equilibrium_bond_lengths_A": centers,
            "coordinate": "frozen local N-H bond lengths",
        },
        "artifact_tag": tag or None,
        "optimized_geometry": {
            "path": optimized_path.name,
            "symbols": molecule.symbols,
            "coordinates_A": molecule.coords.tolist(),
        },
        "pairs": pair_records,
        "total_elapsed_seconds": time.perf_counter() - total_start,
    }
    (data_root / f"generation_summary_{npts}{artifact_suffix}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("validation_data/nh3_three_mode"),
    )
    parser.add_argument("--npts", type=int, default=25)
    parser.add_argument("--half-width", type=float, default=0.24)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    parser.add_argument("--tag", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.npts < 5 or args.npts % 2 == 0:
        parser.error("--npts must be an odd integer of at least 5")
    if args.half_width <= 0.0:
        parser.error("--half-width must be positive")
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("worker counts must be positive")
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.tag.strip()).strip(".-_")
    if args.tag and not tag:
        parser.error("--tag must contain at least one letter or number")

    args.data_root.mkdir(parents=True, exist_ok=True)
    artifact_suffix = f"_{tag}" if tag else ""
    log_path = args.data_root / f"generation_{args.npts}{artifact_suffix}.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        tee = _Tee(sys.stdout, log_stream)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"Command: {' '.join(sys.argv)}")
            print(f"Working directory: {Path.cwd().resolve()}")
            print(f"Python: {sys.version}")
            summary = generate(
                args.data_root,
                npts=args.npts,
                half_width_A=args.half_width,
                workers=args.workers,
                threads_per_worker=args.threads_per_worker,
                force=args.force,
                tag=tag,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
