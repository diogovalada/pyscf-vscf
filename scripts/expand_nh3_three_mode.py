#!/usr/bin/env python3
"""Expand archived NH3 pair grids by evaluating only a nested outer ring."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from generate_nh3_three_mode import PAIRS, _Tee, _cpu_model, _executor_factory, _settings
from pyscf_vscf import Bond, __version__
from pyscf_vscf.cache import runtime_provenance
from pyscf_vscf.io import dump_grid_npz, read_xyz
from pyscf_vscf.settings import apply_thread_env_updates
from pyscf_vscf.surfaces import energy_dipole
from pyscf_vscf.workflows.scans import (
    lbs_frozen_2d_cache_metadata,
    load_lbs_frozen_2d_grid_cache,
    molecule_with_coords,
    stretch_along_bond,
)


def _ring_worker(task):
    i, j, molecule, cfg, bond1, bond2, r1, r2 = task
    coords = stretch_along_bond(molecule.coords, bond1, r1)
    coords = stretch_along_bond(coords, bond2, r2)
    energy, dipole = energy_dipole(molecule_with_coords(molecule, coords), cfg)
    return int(i), int(j), float(energy), np.asarray(dipole, dtype=float)


def _nested_slice(target: np.ndarray, source: np.ndarray) -> slice:
    matches = np.flatnonzero(np.isclose(target, source[0], rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise ValueError("Source grid does not begin at exactly one target-grid point")
    start = int(matches[0])
    stop = start + source.size
    if stop > target.size or not np.allclose(target[start:stop], source, rtol=0.0, atol=1e-12):
        raise ValueError("Source grid is not an exact contiguous subset of the target grid")
    return slice(start, stop)


def _expand_pair(
    molecule,
    cfg,
    bond1: Bond,
    bond2: Bond,
    target1: np.ndarray,
    target2: np.ndarray,
    source1: np.ndarray,
    source2: np.ndarray,
    source_energy: np.ndarray,
    source_dipole: np.ndarray,
    *,
    reference_energy_Eh: float,
    executor_factory,
) -> tuple[np.ndarray, np.ndarray, int]:
    slice1 = _nested_slice(target1, source1)
    slice2 = _nested_slice(target2, source2)
    energies = np.full((target1.size, target2.size), np.nan, dtype=float)
    dipoles = np.full((target1.size, target2.size, 3), np.nan, dtype=float)
    center1 = source1.size // 2
    center2 = source2.size // 2
    energies[slice1, slice2] = (
        source_energy - source_energy[center1, center2] + reference_energy_Eh
    )
    dipoles[slice1, slice2] = source_dipole

    tasks = [
        (i, j, molecule, cfg, bond1, bond2, float(r1), float(r2))
        for i, r1 in enumerate(target1)
        for j, r2 in enumerate(target2)
        if not (slice1.start <= i < slice1.stop and slice2.start <= j < slice2.stop)
    ]
    print(
        f"Expanding nested pair grid: {len(tasks)} new points, "
        f"{source1.size * source2.size} reused points",
        flush=True,
    )
    with executor_factory() as executor:
        for done, (i, j, energy, dipole) in enumerate(executor.map(_ring_worker, tasks), start=1):
            energies[i, j] = energy
            dipoles[i, j] = dipole
            interval = max(1, len(tasks) // 20)
            if done == 1 or done == len(tasks) or done % interval == 0:
                print(f"Outer-ring points: {done}/{len(tasks)}", flush=True)

    if not np.all(np.isfinite(energies)) or not np.all(np.isfinite(dipoles)):
        raise RuntimeError("Expanded pair grid contains unfilled or non-finite points")
    energies -= float(np.min(energies))
    return energies, dipoles, len(tasks)


def expand(
    data_root: Path,
    *,
    source_npts: int,
    source_tag: str,
    target_npts: int,
    target_half_width_A: float,
    target_tag: str,
    workers: int,
    threads_per_worker: int,
) -> dict:
    source_suffix = f"_{source_tag}" if source_tag else ""
    source_summary_path = data_root / f"generation_summary_{source_npts}{source_suffix}.json"
    source_summary = json.loads(source_summary_path.read_text(encoding="utf-8"))
    source_records = {
        tuple(record["pair_zero_based"]): record for record in source_summary["pairs"]
    }
    molecule = read_xyz(data_root / source_summary["optimized_geometry"]["path"])
    molecule.label = "NH3"
    cfg = _settings()
    centers = tuple(
        float(np.linalg.norm(molecule.coords[atom] - molecule.coords[0])) for atom in (1, 2, 3)
    )
    source_width = float(source_summary["grid"]["half_width_A"])
    source_grids = tuple(
        np.linspace(center - source_width, center + source_width, source_npts)
        for center in centers
    )
    target_grids = tuple(
        np.linspace(
            center - target_half_width_A,
            center + target_half_width_A,
            target_npts,
        )
        for center in centers
    )
    for target, source in zip(target_grids, source_grids):
        _nested_slice(target, source)

    apply_thread_env_updates(max(1, workers * threads_per_worker), os.environ)
    reference_energy, _ = energy_dipole(molecule, cfg)
    executor_factory = _executor_factory(workers, threads_per_worker)
    bonds = (Bond(0, 1), Bond(0, 2), Bond(0, 3))
    target_suffix = f"_{target_tag}" if target_tag else ""
    pair_records = []
    total_start = time.perf_counter()
    for i, j in PAIRS:
        pair_start = time.perf_counter()
        source_path = data_root / source_records[(i, j)]["path"]
        source_spec_i = (
            float(source_grids[i][0]),
            float(source_grids[i][-1]),
            source_npts,
        )
        source_spec_j = (
            float(source_grids[j][0]),
            float(source_grids[j][-1]),
            source_npts,
        )
        r1_source, r2_source, energy_source, dipole_source = load_lbs_frozen_2d_grid_cache(
            source_path,
            molecule,
            cfg,
            bonds[i],
            bonds[j],
            source_spec_i,
            source_spec_j,
            keo="separable-local-reduced-mass",
        )
        energy, dipole, new_points = _expand_pair(
            molecule,
            cfg,
            bonds[i],
            bonds[j],
            target_grids[i],
            target_grids[j],
            r1_source,
            r2_source,
            energy_source,
            dipole_source,
            reference_energy_Eh=reference_energy,
            executor_factory=executor_factory,
        )
        path = (
            data_root / f"nh3_pair_{i + 1}{j + 1}_{target_npts}x{target_npts}{target_suffix}.npz"
        )
        target_spec_i = (
            float(target_grids[i][0]),
            float(target_grids[i][-1]),
            target_npts,
        )
        target_spec_j = (
            float(target_grids[j][0]),
            float(target_grids[j][-1]),
            target_npts,
        )
        metadata = lbs_frozen_2d_cache_metadata(
            molecule,
            cfg,
            bonds[i],
            bonds[j],
            target_spec_i,
            target_spec_j,
            keo="separable-local-reduced-mass",
        )
        metadata["nested_expansion"] = {
            "source_cache": source_path.name,
            "source_npts": source_npts,
            "source_tag": source_tag or None,
            "reused_points": source_npts**2,
            "new_points": new_points,
        }
        dump_grid_npz(
            path,
            meta=metadata,
            arrays={
                "R1_A": target_grids[i],
                "R2_A": target_grids[j],
                "E_Eh": energy,
                "MU_Debye": dipole,
            },
        )
        pair_records.append(
            {
                "pair_zero_based": [i, j],
                "path": path.name,
                "source_path": source_path.name,
                "reused_points": source_npts**2,
                "new_points": new_points,
                "elapsed_seconds": time.perf_counter() - pair_start,
                "energy_range_Eh": float(np.ptp(energy)),
            }
        )

    summary = {
        "purpose": "nested expansion of non-water three-local-mode VSCF validation grids",
        "package_version": __version__,
        "artifact_tag": target_tag or None,
        "electronic_structure": asdict(cfg),
        "runtime": runtime_provenance(),
        "hardware": {
            "cpu": _cpu_model(),
            "logical_cpu_count": os.cpu_count(),
            "workers": workers,
            "threads_per_worker": threads_per_worker,
        },
        "grid": {
            "npts_per_mode": target_npts,
            "half_width_A": target_half_width_A,
            "equilibrium_bond_lengths_A": centers,
            "coordinate": "frozen local N-H bond lengths",
        },
        "optimized_geometry": source_summary["optimized_geometry"],
        "source_summary": source_summary_path.name,
        "pairs": pair_records,
        "total_elapsed_seconds": time.perf_counter() - total_start,
    }
    summary_path = data_root / f"generation_summary_{target_npts}{target_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("validation_data/nh3_three_mode"),
    )
    parser.add_argument("--source-npts", type=int, required=True)
    parser.add_argument("--source-tag", default="")
    parser.add_argument("--target-npts", type=int, required=True)
    parser.add_argument("--target-half-width", type=float, required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--threads-per-worker", type=int, default=1)
    args = parser.parse_args()
    if args.target_npts <= args.source_npts:
        parser.error("--target-npts must exceed --source-npts")
    if args.target_half_width <= 0.0:
        parser.error("--target-half-width must be positive")
    if args.workers < 1 or args.threads_per_worker < 1:
        parser.error("worker counts must be positive")

    log_path = args.data_root / f"generation_{args.target_npts}_{args.target_tag}.log"
    with log_path.open("w", encoding="utf-8") as log_stream:
        tee = _Tee(sys.stdout, log_stream)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"Command: {' '.join(sys.argv)}")
            print(f"Working directory: {Path.cwd().resolve()}")
            summary = expand(
                args.data_root,
                source_npts=args.source_npts,
                source_tag=args.source_tag,
                target_npts=args.target_npts,
                target_half_width_A=args.target_half_width,
                target_tag=args.target_tag,
                workers=args.workers,
                threads_per_worker=args.threads_per_worker,
            )
            print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
