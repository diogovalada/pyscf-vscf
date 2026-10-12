#!/usr/bin/env python3
"""Compare GPU4PySCF (device="gpu") with PySCF (device="cpu") through the package.

Runs on a machine with GPU4PySCF and a CUDA device. Every check computes the
same quantity on both devices through the public API and compares it with a
tolerance fixed in advance (``TOLERANCES``). Writes a JSON report and exits 1
if any check fails.

    python scripts/gpu_parity.py --output gpu-parity.json
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
    bond_spectrum,
    plan_bond_scan,
    run_scan,
)
from pyscf_vscf.backends._gpu import gpu_provenance
from pyscf_vscf.backends._pyscf_properties import nuclear_gradient
from pyscf_vscf.backends.pyscf import converged_mean_field
from pyscf_vscf.backends._pyscf_factory import ElectronicSystem
from pyscf_vscf.contracts import ElectronicPointRequest
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.workflows.harmonic import harmonic_analysis
from pyscf_vscf.workflows.optimization import optimize_geometry

# Fixed before the first GPU run, except harmonic_frequencies (first 0.1 cm^-1).
# Energies: Eh; dipoles: atomic units; gradients: Eh/bohr; frequencies: cm^-1;
# geometry: A.
#
# harmonic_frequencies was raised to 0.5 cm^-1 after the first run, where
# NH3 at wB97X/aug-cc-pVDZ (density fitting) differed by 0.24 cm^-1. Neither
# implementation is wrong: both analytic DFT Hessians omit the grid response,
# and each differs from a finite-difference Hessian of its own gradients by
# 4.4e-4 Eh/bohr^2 (about 1.8 cm^-1); those finite-difference Hessians agree
# between CPU and GPU to 1.2e-5 Eh/bohr^2. The CPU-GPU difference (3.5e-5) is
# ten times smaller than either implementation's own approximation. Evidence:
# validation_data/gpu/hessian-referee-nh3-wb97x.txt.
TOLERANCES = {
    "energy": 1e-8,
    "dipole": 1e-5,
    "finite_field_energy": 1e-8,
    "gradient": 1e-6,
    "harmonic_frequencies": 0.5,
    "optimized_geometry": 1e-4,
    "optimized_energy": 1e-8,
    "scan_energies": 1e-8,
    "scan_dipoles": 1e-5,
    "scan_transition_wavenumbers": 0.01,
}

WATER = (["O", "H", "H"], [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]])
AMMONIA = (
    ["N", "H", "H", "H"],
    [
        [0.0, 0.0, 0.1127],
        [0.0, 0.9377, -0.2629],
        [0.8121, -0.4689, -0.2629],
        [-0.8121, -0.4689, -0.2629],
    ],
)
SYSTEMS = {
    "H2O": Molecule.from_arrays(*WATER, label="H2O"),
    "D2O": Molecule.from_arrays(["O", "D", "D"], WATER[1], label="D2O"),
    "NH3": Molecule.from_arrays(*AMMONIA, label="NH3"),
}
METHODS = {
    "hf/cc-pvdz": ESSettings(method="hf", basis="cc-pvdz", use_density_fit=False),
    "b3lyp/cc-pvdz/df": ESSettings(method="b3lyp", basis="cc-pvdz", use_density_fit=True),
    "wb97x/aug-cc-pvdz/df": ESSettings(method="wb97x", basis="aug-cc-pvdz", use_density_fit=True),
}


def _gpu(settings: ESSettings) -> ESSettings:
    return replace(settings, device="gpu")


def _max_diff(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


class Report:
    def __init__(self) -> None:
        self.checks: list[dict[str, object]] = []

    def compare(self, system: str, method: str, quantity: str, cpu, gpu) -> None:
        difference = _max_diff(cpu, gpu)
        tolerance = TOLERANCES[quantity]
        self.checks.append(
            {
                "system": system,
                "method": method,
                "quantity": quantity,
                "max_abs_difference": difference,
                "tolerance": tolerance,
                "passed": bool(difference <= tolerance),
            }
        )
        status = "ok  " if difference <= tolerance else "FAIL"
        print(
            f"{status} {system:4s} {method:22s} {quantity:28s} {difference:.2e} (tol {tolerance:.0e})"
        )
        sys.stdout.flush()

    def error(self, system: str, method: str, quantity: str, exc: BaseException) -> None:
        self.checks.append(
            {
                "system": system,
                "method": method,
                "quantity": quantity,
                "error": f"{type(exc).__name__}: {exc}",
                "passed": False,
            }
        )
        print(f"FAIL {system:4s} {method:22s} {quantity:28s} {type(exc).__name__}: {exc}")
        sys.stdout.flush()


def _point(molecule: Molecule, settings: ESSettings, field=None):
    provider = PySCFMeanFieldProvider(settings)
    system = ElectronicSystem.from_molecule(molecule)
    request = ElectronicPointRequest(
        nuclear_charges=system.nuclear_charges,
        coordinates_A=system.coordinates_A,
        requested_properties=("energy",) if field is not None else ("energy", "dipole"),
        field_au=field,
    )
    return provider.evaluate(request)


def check_points(report: Report, name: str, molecule: Molecule, label: str, settings) -> None:
    try:
        cpu, gpu = _point(molecule, settings), _point(molecule, _gpu(settings))
        report.compare(name, label, "energy", cpu.total_energy_Eh, gpu.total_energy_Eh)
        report.compare(name, label, "dipole", cpu.dipole_au, gpu.dipole_au)
    except Exception as exc:
        report.error(name, label, "energy", exc)
    field = np.array([0.0, 0.0, 1e-3])
    try:
        cpu = _point(molecule, settings, field)
        gpu = _point(molecule, _gpu(settings), field)
        report.compare(
            name, label, "finite_field_energy", cpu.total_energy_Eh, gpu.total_energy_Eh
        )
    except Exception as exc:
        report.error(name, label, "finite_field_energy", exc)
    try:
        cpu = nuclear_gradient(converged_mean_field(molecule, settings))
        gpu = nuclear_gradient(converged_mean_field(molecule, _gpu(settings)))
        report.compare(name, label, "gradient", cpu, gpu)
    except Exception as exc:
        report.error(name, label, "gradient", exc)


def check_harmonic(report: Report, name: str, molecule: Molecule, label: str, settings) -> None:
    try:
        cpu = harmonic_analysis(molecule, settings)
        gpu = harmonic_analysis(molecule, _gpu(settings))
        report.compare(name, label, "harmonic_frequencies", cpu.freqs_cm, gpu.freqs_cm)
    except Exception as exc:
        report.error(name, label, "harmonic_frequencies", exc)


def check_optimization(
    report: Report, name: str, molecule: Molecule, label: str, settings
) -> None:
    try:
        cpu = optimize_geometry(molecule, settings)
        gpu = optimize_geometry(molecule, _gpu(settings))
        report.compare(name, label, "optimized_geometry", cpu.molecule.coords, gpu.molecule.coords)
        report.compare(
            name,
            label,
            "optimized_energy",
            _point(cpu.molecule, settings).total_energy_Eh,
            _point(gpu.molecule, settings).total_energy_Eh,
        )
    except Exception as exc:
        report.error(name, label, "optimized_geometry", exc)


def check_scan(report: Report, label: str, settings, workers: int) -> None:
    """A bond scan on 2 GPU workers against the same scan on the CPU."""

    molecule = SYSTEMS["H2O"]
    grid = [np.linspace(0.75, 1.35, 11)]
    try:
        scans = {}
        for device, runner in (
            ("cpu", RunnerSettings(workers=workers)),
            ("gpu", RunnerSettings(workers=workers)),
        ):
            method = settings if device == "cpu" else _gpu(settings)
            provider = PySCFMeanFieldProvider(method)
            plan = plan_bond_scan(molecule, provider, ["0-1"], grid)
            scans[device] = run_scan(plan, provider, settings=runner)
        cpu, gpu = scans["cpu"], scans["gpu"]
        report.compare("H2O", label, "scan_energies", cpu.total_energies_Eh, gpu.total_energies_Eh)
        report.compare("H2O", label, "scan_dipoles", cpu.dipoles_au, gpu.dipoles_au)
        report.compare(
            "H2O",
            label,
            "scan_transition_wavenumbers",
            [r.frequency_cm for r in bond_spectrum(cpu, molecule, max_state_index=3)],
            [r.frequency_cm for r in bond_spectrum(gpu, molecule, max_state_index=3)],
        )
    except Exception as exc:
        report.error("H2O", label, "scan_energies", exc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON report to write")
    parser.add_argument("--quick", action="store_true", help="H2O with one method only")
    parser.add_argument("--workers", type=int, default=2, help="workers for the scan check")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    report = Report()
    methods = dict(list(METHODS.items())[1:2]) if args.quick else METHODS
    systems = {"H2O": SYSTEMS["H2O"]} if args.quick else SYSTEMS
    for label, settings in methods.items():
        for name, molecule in systems.items():
            if name != "D2O":  # same electrons as H2O; only the frequencies differ
                check_points(report, name, molecule, label, settings)
            check_harmonic(report, name, molecule, label, settings)
        check_optimization(report, "H2O", SYSTEMS["H2O"], label, settings)
        check_scan(report, label, settings, args.workers)

    passed = all(check["passed"] for check in report.checks)
    payload = {
        "schema": "pyscf-vscf-gpu-parity",
        "schema_version": 1,
        "package_version": __version__,
        "passed": passed,
        "tolerances": TOLERANCES,
        "gpu": gpu_provenance(),
        "python": platform.python_version(),
        "wall_seconds": time.perf_counter() - started,
        "checks": report.checks,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    failed = [check for check in report.checks if not check["passed"]]
    print(
        f"{len(report.checks) - len(failed)} of {len(report.checks)} checks passed; wrote {args.output}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
