"""Water: harmonic, VSCF, and VCI fundamentals from PySCF surfaces.

Run with ``python -m pyscf_vscf.examples.water_vscf_vci``. The workflow:

1. optimize the geometry and compute harmonic normal modes;
2. plan a 1MR/2MR n-mode grid along the three normal coordinates;
3. evaluate the electronic points (in parallel, resumable with ``--store``);
4. fit potential and dipole surfaces, then solve VSCF and VCI with
   intensities.

The default HF/6-31G model and small grids keep the run to about a minute on
a laptop; ``--method``, ``--basis``, ``--nodes``, and ``--solver-points``
increase the accuracy. The numbers illustrate the workflow; converge the
grids and electronic model before using them scientifically.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyscf_vscf.backends.pyscf import PySCFMeanFieldProvider
from pyscf_vscf.constants import AMU, ANG_TO_BOHR, HARTREE_TO_CM
from pyscf_vscf.coordinates import LinearDisplacementCoordinateMap
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.nmode import (
    assemble_nmode_samples,
    fit_nmode_surface,
    nmode_potential_from_surface,
    plan_nmode_points,
)
from pyscf_vscf.point_store import PointStore
from pyscf_vscf.runner import RunnerSettings, run_plan
from pyscf_vscf.settings import ESSettings, available_cpu_count
from pyscf_vscf.spectra import integrated_cross_section_omega_to_km_per_mol
from pyscf_vscf.transition_moments import (
    build_vci_dipole_operator,
    nmode_dipole_on_grid,
    vci_transition_moments,
)
from pyscf_vscf.vci import VCISettings, build_nmode_vscf_modal_basis, solve_vci
from pyscf_vscf.vscf import NModeDipole, vscf_spectrum
from pyscf_vscf.workflows.harmonic import harmonic_analysis
from pyscf_vscf.workflows.optimization import optimize_geometry

WATER = Molecule.from_arrays(
    ["O", "H", "H"],
    [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    label="water",
)


def normal_coordinates(molecule: Molecule, settings: ESSettings):
    """Return harmonic frequencies, unit Cartesian directions, and effective masses."""

    harmonic = harmonic_analysis(molecule, settings)
    masses = np.repeat(molecule.analysis_masses(), 3)
    selected = [k for k, freq in enumerate(harmonic.freqs_cm) if freq > 100.0]
    frequencies, directions, effective_masses = [], [], []
    for k in selected:
        cartesian = harmonic.modes[:, k] / np.sqrt(masses)
        unit = cartesian / np.linalg.norm(cartesian)
        frequencies.append(float(harmonic.freqs_cm[k]))
        directions.append(unit.reshape(-1, 3))
        effective_masses.append(float(np.sum(masses * unit * unit)))
    return np.array(frequencies), np.array(directions), np.array(effective_masses)


def displacement_extent_A(frequency_cm: float, mass_amu: float, quanta: int = 4) -> float:
    """Classical turning point of harmonic level ``quanta``, plus a 30% margin."""

    omega = frequency_cm / HARTREE_TO_CM
    turning_bohr = np.sqrt((2 * quanta + 1) / (mass_amu * AMU * omega))
    return 1.3 * turning_bohr / ANG_TO_BOHR


def run(args: argparse.Namespace) -> dict:
    settings = ESSettings(method=args.method, basis=args.basis)
    optimized = optimize_geometry(WATER, settings).molecule
    frequencies, directions, masses = normal_coordinates(optimized, settings)
    labels = ("bend", "symmetric stretch", "antisymmetric stretch")[: len(frequencies)]

    coordinate_map = LinearDisplacementCoordinateMap(
        reference_geometry_A=optimized.coords,
        coordinate_ids=tuple(f"q{k + 1}" for k in range(len(frequencies))),
        units=("angstrom",) * len(frequencies),
        reference_values=np.zeros(len(frequencies)),
        displacements_A_per_unit=directions,
    )
    extents = [displacement_extent_A(f, m) for f, m in zip(frequencies, masses)]
    node_axes = [np.linspace(-a, a, args.nodes) for a in extents]
    provider = PySCFMeanFieldProvider(settings)
    plan = plan_nmode_points(
        coordinate_map, node_axes, provider, nuclear_charges=(8, 1, 1), max_rank=2
    )
    workers = max(1, min(available_cpu_count(), len(plan.requests)))
    runner = RunnerSettings(workers=workers if workers > 1 else None, threads_per_worker=1)
    store = PointStore(args.store) if args.store else None
    try:
        evaluation = run_plan(plan, provider, store=store, settings=runner)
    finally:
        if store is not None:
            store.close()
    if not evaluation.complete:
        raise RuntimeError(evaluation.summary())

    surface = fit_nmode_surface(assemble_nmode_samples(plan, evaluation.results), method="cubic")
    solver_grids = tuple(np.linspace(-0.98 * a, 0.98 * a, args.solver_points) for a in extents)
    potential = nmode_potential_from_surface(surface, solver_grids, masses)
    dipole = NModeDipole.from_surface(surface, solver_grids)

    vscf = vscf_spectrum(potential, dipole=dipole)
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(potential, args.modals)
    vci = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=7, max_total_quanta=3, extra_eigenstates=2),
    )
    vci_moments = vci_transition_moments(
        build_vci_dipole_operator(nmode_dipole_on_grid(surface, hamiltonian), modal_basis, vci),
        vci,
    )

    rows = []
    for mode, label in enumerate(labels):
        quanta = tuple(1 if k == mode else 0 for k in range(len(labels)))
        vscf_line = next(t for t in vscf.transitions if t.quanta == quanta)
        vci_index = next(
            (
                state
                for state, assignment in enumerate(vci.assignments)
                if assignment.configuration == quanta
            ),
            None,
        )
        vci_line = next((m for m in vci_moments if m.upper_state == vci_index), None)
        rows.append(
            {
                "mode": label,
                "harmonic_cm": float(frequencies[mode]),
                "vscf_cm": vscf_line.frequency_cm,
                "vscf_km_per_mol": vscf_line.transition_moment.integrated_cross_section_isotropic_km_per_mol,
                "vci_cm": None if vci_line is None else float(vci_line.frequency_cm),
                "vci_km_per_mol": None
                if vci_line is None
                else integrated_cross_section_omega_to_km_per_mol(
                    vci_line.isotropic_integrated_cross_section_omega_m2_per_s
                ),
            }
        )
    return {
        "method": f"{args.method}/{args.basis}",
        "electronic_points": evaluation.total,
        "points_reused": len(evaluation.reused),
        "fundamentals": rows,
    }


def main(argv: list[str] | None = None) -> None:
    """Run the water example and print a table (or JSON with ``--json``)."""

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", default="hf")
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--nodes", type=int, default=9, help="odd number of fit nodes per mode")
    parser.add_argument("--solver-points", type=int, default=21)
    parser.add_argument("--modals", type=int, default=6)
    parser.add_argument("--store", type=Path, default=None, help="SQLite point store (resume)")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a table")
    args = parser.parse_args(argv)
    if args.nodes < 5 or args.nodes % 2 == 0:
        parser.error("--nodes must be an odd number of at least 5")
    result = run(args)
    if args.json:
        print(json.dumps(result, indent=2))
        return
    print(f"Water fundamentals, {result['method']} ({result['electronic_points']} points)")
    print(
        f"{'mode':<22} {'harmonic':>9} {'VSCF':>9} {'VCI':>9} {'VSCF km/mol':>12} {'VCI km/mol':>11}"
    )
    for row in result["fundamentals"]:
        vci = "-" if row["vci_cm"] is None else f"{row['vci_cm']:9.1f}"
        vci_i = "-" if row["vci_km_per_mol"] is None else f"{row['vci_km_per_mol']:11.2f}"
        print(
            f"{row['mode']:<22} {row['harmonic_cm']:9.1f} {row['vscf_cm']:9.1f} {vci:>9} "
            f"{row['vscf_km_per_mol']:12.2f} {vci_i:>11}"
        )


if __name__ == "__main__":
    main()
