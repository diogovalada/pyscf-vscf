#!/usr/bin/env python3
"""Reanalyze archived monomer grids and emit assigned convergence budgets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pyscf_vscf import __version__
from pyscf_vscf.io import load_grid_npz, read_midas_mmol
from pyscf_vscf.spectra import integrated_cross_section_omega_to_km_per_mol
from pyscf_vscf.validation import convergence_report
from pyscf_vscf.variational import variational_2d
from pyscf_vscf.vscf import NModePotential, VSCFSettings, vscf_spectrum
from pyscf_vscf.workflows.scans import (
    bond_bond_g12_inv_amu,
    local_bond_reduced_mass_amu,
)
from pyscf_vscf.coordinates import Bond


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verified_path(root: Path, relative: str, expected: str) -> Path:
    path = (root / relative).resolve()
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")
    return path


def _records_for_variant(
    arrays: dict[str, np.ndarray],
    indices: np.ndarray,
    molecule,
    *,
    nmax: int,
) -> list[dict]:
    bond1 = Bond(0, 1)
    bond2 = Bond(0, 2)
    R1 = np.asarray(arrays["R1_A"])[indices]
    R2 = np.asarray(arrays["R2_A"])[indices]
    E = np.asarray(arrays["E_Eh"])[np.ix_(indices, indices)]
    MU = np.asarray(arrays["MU_Debye"])[np.ix_(indices, indices, np.arange(3))]
    axis = molecule.coords[bond1.j] - molecule.coords[bond1.i]
    records = variational_2d(
        R1,
        R2,
        E,
        MU,
        local_bond_reduced_mass_amu(molecule, bond1),
        local_bond_reduced_mass_amu(molecule, bond2),
        axis=axis,
        nmax=nmax,
        g12_inv_amu=bond_bond_g12_inv_amu(molecule, bond1, bond2),
    )
    return [record.as_dict() for record in records]


def _molecular_vscf_benchmark(arrays: dict[str, np.ndarray], molecule) -> dict:
    """Compare VSCF with exact 2D DVR on the same archived molecular PES."""

    bond1 = Bond(0, 1)
    bond2 = Bond(0, 2)
    R1 = np.asarray(arrays["R1_A"], dtype=float)
    R2 = np.asarray(arrays["R2_A"], dtype=float)
    energy = np.asarray(arrays["E_Eh"], dtype=float)
    minimum = np.unravel_index(int(np.argmin(energy)), energy.shape)
    reference = float(energy[minimum])
    one_mode_1 = energy[:, minimum[1]] - reference
    one_mode_2 = energy[minimum[0], :] - reference
    coupling = energy - reference - one_mode_1[:, None] - one_mode_2[None, :]
    mu1 = local_bond_reduced_mass_amu(molecule, bond1)
    mu2 = local_bond_reduced_mass_amu(molecule, bond2)
    model = NModePotential(
        coordinates=(R1, R2),
        masses_amu=(mu1, mu2),
        one_mode_potentials_Eh=(one_mode_1, one_mode_2),
        two_mode_couplings_Eh={(0, 1): coupling},
        mode_labels=("bond-1", "bond-2"),
        metadata={"benchmark": "archived molecular PES"},
        coordinate_units="angstrom",
    )
    requested_states = ((1, 0), (0, 1), (1, 1))
    vscf = vscf_spectrum(
        model,
        states=requested_states,
        settings=VSCFSettings(
            max_iterations=200,
            energy_tolerance_Eh=1e-11,
            density_tolerance=1e-9,
        ),
    )
    exact_records = variational_2d(
        R1,
        R2,
        energy,
        np.zeros((*energy.shape, 3), dtype=float),
        mu1,
        mu2,
        nmax=12,
        g12_inv_amu=0.0,
        reference_potentials_Eh=(one_mode_1, one_mode_2),
    )
    exact_by_state = {record.quanta: record for record in exact_records}
    vscf_by_state = {transition.quanta: transition for transition in vscf.transitions}
    state_results = []
    for state in requested_states:
        if state not in exact_by_state:
            raise ValueError(f"Exact 2D DVR did not assign benchmark state {state}")
        exact = exact_by_state[state]
        approximate = vscf_by_state[state]
        error_cm = float(approximate.frequency_cm - exact.frequency_cm)
        state_results.append(
            {
                "assignment": state,
                "exact_dvr_frequency_cm": float(exact.frequency_cm),
                "vscf_frequency_cm": float(approximate.frequency_cm),
                "signed_error_cm": error_cm,
                "absolute_error_cm": abs(error_cm),
                "exact_assignment_weight": float(exact.assignment_weight),
            }
        )
    return {
        "reference": "exact 2D sinc-DVR on the identical archived PES",
        "kinetic_model": "separable constant reduced masses; g12_inv_amu=0 to match VSCF",
        "scope": (
            "tests the state-specific VSCF mean-field implementation on molecular surfaces; "
            "it does not validate the electronic-structure PES or omitted kinetic couplings"
        ),
        "ground_converged": bool(vscf.ground.converged),
        "ground_iterations": int(vscf.ground.iterations),
        "states": state_results,
        "maximum_absolute_error_cm": max(result["absolute_error_cm"] for result in state_results),
    }


def analyze(data_root: Path, *, nmax: int) -> dict:
    manifest_path = data_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    benchmark_entry = manifest["independent_intensity_benchmark"]
    benchmark_path = _verified_path(
        data_root,
        benchmark_entry["path"],
        benchmark_entry["sha256"],
    )
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    reports = {}
    for species, entry in manifest["systems"].items():
        grid_path = _verified_path(data_root, entry["grid"], entry["grid_sha256"])
        _verified_path(data_root, entry["log"], entry["log_sha256"])
        geometry_path = _verified_path(data_root, entry["geometry"], entry["geometry_sha256"])
        _meta, arrays = load_grid_npz(grid_path)
        molecule = read_midas_mmol(geometry_path)
        variants = {
            "full_41x41_0.70-1.70A": np.arange(41),
            "nested_21x21_0.70-1.70A": np.arange(0, 41, 2),
            "window_37x37_0.75-1.65A": np.arange(2, 39),
        }
        records = {
            name: _records_for_variant(arrays, indices, molecule, nmax=nmax)
            for name, indices in variants.items()
        }
        error_budget = convergence_report(records)
        fundamental_budgets = [
            state for state in error_budget.states if _is_fundamental_signature(state.assignment)
        ]
        if len(fundamental_budgets) != 2:
            raise ValueError(f"Expected two matched fundamental error budgets for {species}")
        fundamental_acceptance = {
            "criteria": {
                "maximum_frequency_spread_cm": 20.0,
                "maximum_intensity_relative_spread": 0.01,
                "minimum_dominant_manifold_weight": 0.99,
            },
            "observed": {
                "maximum_frequency_spread_cm": max(
                    state.frequency_spread_cm for state in fundamental_budgets
                ),
                "maximum_intensity_relative_spread": max(
                    float(state.intensity_relative_spread) for state in fundamental_budgets
                ),
                "minimum_dominant_manifold_weight": min(
                    float(state.minimum_dominant_manifold_weight) for state in fundamental_budgets
                ),
            },
        }
        criteria = fundamental_acceptance["criteria"]
        observed = fundamental_acceptance["observed"]
        fundamental_acceptance["passed"] = bool(
            observed["maximum_frequency_spread_cm"] <= criteria["maximum_frequency_spread_cm"]
            and observed["maximum_intensity_relative_spread"]
            <= criteria["maximum_intensity_relative_spread"]
            and observed["minimum_dominant_manifold_weight"]
            >= criteria["minimum_dominant_manifold_weight"]
        )
        fundamental_acceptance["unmatched_state_policy"] = (
            "Only states present under the same phase-canonical assignment signature in every "
            "variant enter an error budget. Unmatched states are reported and cannot be cited as "
            "converged; the two accepted fundamentals are present in all three variants."
        )
        full_fundamentals = sorted(
            (
                record
                for record in records["full_41x41_0.70-1.70A"]
                if _is_stretch_fundamental(record)
            ),
            key=lambda record: record["freq_cm"],
        )
        references = sorted(
            benchmark["systems"][species],
            key=lambda record: record["frequency_cm"],
        )
        if len(full_fundamentals) != 2 or len(references) != 2:
            raise ValueError(f"Expected two stretch fundamentals for {species}")
        intensity_benchmark = []
        for calculated, reference in zip(full_fundamentals, references):
            calculated_intensity = integrated_cross_section_omega_to_km_per_mol(
                calculated["integrated_cross_section_isotropic_omega_m2_per_s"]
            )
            reference_intensity = float(reference["intensity_km_per_mol"])
            intensity_benchmark.append(
                {
                    "assignment_signature": calculated["assignment_signature"],
                    "calculated_frequency_cm": calculated["freq_cm"],
                    "reference_frequency_cm": reference["frequency_cm"],
                    "calculated_intensity_km_per_mol": calculated_intensity,
                    "reference_intensity_km_per_mol": reference_intensity,
                    "relative_intensity_error": abs(calculated_intensity - reference_intensity)
                    / reference_intensity,
                    "reference_mode_index": reference["mode_index"],
                }
            )
        reports[species] = {
            "source_grid_sha256": entry["grid_sha256"],
            "variants": records,
            "error_budget": error_budget.as_dict(),
            "fundamental_convergence_acceptance": fundamental_acceptance,
            "independent_orca_intensity_benchmark": intensity_benchmark,
            "molecular_vscf_vs_exact_2d_dvr": _molecular_vscf_benchmark(arrays, molecule),
        }
    return {
        "package_version": __version__,
        "source_manifest_sha256": _sha256(manifest_path),
        "intensity_benchmark_source": benchmark["source"],
        "intensity_benchmark_limitations": benchmark["limitations"],
        "method": (
            "assigned cached-grid convergence and VSCF-vs-exact-DVR molecular benchmark; "
            "no electronic-structure recomputation"
        ),
        "nmax": int(nmax),
        "systems": reports,
    }


def _is_stretch_fundamental(record: dict) -> bool:
    signature = record.get("assignment_signature", ())
    return bool(signature) and all(sum(quanta) == 1 for quanta, _sign in signature)


def _is_fundamental_signature(signature: tuple) -> bool:
    return bool(signature) and all(sum(quanta) == 1 for quanta, _sign in signature)


def main() -> None:
    parser = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--data-root", type=Path, default=repo_root / "validation_data")
    parser.add_argument("--nmax", type=int, default=12)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = analyze(args.data_root.resolve(), nmax=args.nmax)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
