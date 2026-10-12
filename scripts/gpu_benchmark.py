#!/usr/bin/env python3
"""Benchmark device="gpu" against device="cpu" through the package.

Two measurements, written to a JSON report:

- ``points``: wall time of one electronic point (energy and dipole) on one
  CPU core and on the GPU, for several molecules, methods, and basis sets.
  ``cores_replaced`` = CPU time / GPU time is the number of single-threaded
  CPU workers that match one GPU.
- ``throughput``: points per minute of a water bond scan run by ``run_plan``
  with 1..N workers sharing each GPU, and with one single-threaded worker per
  available CPU core.

    python scripts/gpu_benchmark.py --output gpu-benchmark.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from pyscf_vscf import (
    ESSettings,
    PySCFMeanFieldProvider,
    RunnerSettings,
    __version__,
    plan_bond_scan,
    run_scan,
)
from pyscf_vscf.backends._gpu import gpu_provenance
from pyscf_vscf.backends._pyscf_factory import ElectronicSystem
from pyscf_vscf.contracts import ElectronicPointRequest
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.settings import available_cpu_count

WATER = Molecule.from_arrays(
    ["O", "H", "H"],
    [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    label="water",
)
DIMER = Molecule.from_arrays(
    ["O", "H", "H", "O", "H", "H"],
    [
        [-1.551007, -0.114520, 0.000000],
        [-1.934259, 0.762503, 0.000000],
        [-0.599677, 0.040712, 0.000000],
        [1.350625, 0.111469, 0.000000],
        [1.680398, -0.373741, -0.758561],
        [1.680398, -0.373741, 0.758561],
    ],
    label="water dimer",
)
PYRIDINE = Molecule.from_arrays(
    ["N", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H"],
    [
        [0.0000, 1.4045, 0.0],
        [1.1393, 0.7116, 0.0],
        [1.1946, -0.6824, 0.0],
        [0.0000, -1.3980, 0.0],
        [-1.1946, -0.6824, 0.0],
        [-1.1393, 0.7116, 0.0],
        [2.0655, 1.2832, 0.0],
        [2.1563, -1.1886, 0.0],
        [0.0000, -2.4846, 0.0],
        [-2.1563, -1.1886, 0.0],
        [-2.0655, 1.2832, 0.0],
    ],
    label="pyridine",
)
POINT_CASES = [
    (WATER, "hf", "cc-pvdz"),
    (WATER, "b3lyp", "cc-pvdz"),
    (WATER, "b3lyp", "aug-cc-pvtz"),
    (WATER, "wb97x", "aug-cc-pvtz"),
    (DIMER, "b3lyp", "aug-cc-pvtz"),
    (DIMER, "wb97x", "aug-cc-pvtz"),
    (PYRIDINE, "b3lyp", "cc-pvdz"),
]


def _evaluate(molecule: Molecule, settings: ESSettings) -> float:
    system = ElectronicSystem.from_molecule(molecule)
    request = ElectronicPointRequest(
        nuclear_charges=system.nuclear_charges, coordinates_A=system.coordinates_A
    )
    started = time.perf_counter()
    PySCFMeanFieldProvider(settings).evaluate(request)
    return time.perf_counter() - started


def point_times(gpu_threads: int) -> list[dict[str, object]]:
    from pyscf import lib

    rows = []
    for molecule, method, basis in POINT_CASES:
        for density_fit in (False, True):
            cpu = ESSettings(method=method, basis=basis, use_density_fit=density_fit)
            gpu = replace(cpu, device="gpu")
            lib.num_threads(gpu_threads)
            _evaluate(molecule, gpu)  # warm-up: kernel compilation and memory pools
            gpu_seconds = _evaluate(molecule, gpu)
            lib.num_threads(1)
            cpu_seconds = _evaluate(molecule, cpu)
            row = {
                "molecule": molecule.label,
                "method": method,
                "basis": basis,
                "density_fit": density_fit,
                "cpu_one_core_seconds": cpu_seconds,
                "gpu_seconds": gpu_seconds,
                "cores_replaced": cpu_seconds / gpu_seconds,
            }
            rows.append(row)
            print(
                f"{molecule.label:12s} {method:6s} {basis:12s} df={int(density_fit)} "
                f"cpu1={cpu_seconds:8.2f}s gpu={gpu_seconds:7.2f}s "
                f"cores_replaced={row['cores_replaced']:6.1f}",
                flush=True,
            )
    return rows


def throughput(settings: ESSettings, points: int, max_gpu_workers: int) -> list[dict[str, object]]:
    rows = []
    cpus = available_cpu_count() or 1
    runs = [("gpu", workers) for workers in range(1, max_gpu_workers + 1)] + [("cpu", cpus)]
    for index, (device, workers) in enumerate(runs):
        method = replace(settings, device=device)
        provider = PySCFMeanFieldProvider(method, threads=1 if device == "cpu" else 2)
        # A fresh grid per run, so nothing is reused between runs.
        grid = np.linspace(0.80 + 1e-4 * index, 1.30, points)
        plan = plan_bond_scan(WATER, provider, ["0-1"], [grid])
        runner = RunnerSettings(workers=workers, threads_per_worker=1 if device == "cpu" else 2)
        started = time.perf_counter()
        run_scan(plan, provider, settings=runner)
        seconds = time.perf_counter() - started
        row = {
            "device": device,
            "workers": workers,
            "points": points,
            "seconds": seconds,
            "points_per_minute": 60.0 * points / seconds,
        }
        rows.append(row)
        print(
            f"throughput {device} workers={workers}: {row['points_per_minute']:.1f} points/min",
            flush=True,
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-threads", type=int, default=2, help="CPU threads for GPU runs")
    parser.add_argument("--points", type=int, default=24, help="points in the throughput scan")
    parser.add_argument("--max-gpu-workers", type=int, default=3)
    parser.add_argument("--skip-points", action="store_true")
    args = parser.parse_args(argv)

    report: dict[str, object] = {
        "schema": "pyscf-vscf-gpu-benchmark",
        "schema_version": 1,
        "package_version": __version__,
        "gpu": gpu_provenance(),
        "cpu": platform.processor() or platform.machine(),
        "available_cpus": available_cpu_count(),
        "python": platform.python_version(),
    }
    if not args.skip_points:
        report["points"] = point_times(args.gpu_threads)
    report["throughput"] = throughput(
        ESSettings(method="wb97x", basis="aug-cc-pvtz", use_density_fit=True),
        args.points,
        args.max_gpu_workers,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
