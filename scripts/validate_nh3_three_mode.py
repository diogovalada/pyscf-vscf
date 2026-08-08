#!/usr/bin/env python3
"""Validate three-mode NH3 VSCF against exact DVR on archived pair surfaces."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from pyscf_vscf import Bond, HARTREE_TO_CM, __version__
from pyscf_vscf.io import load_grid_npz, read_xyz
from pyscf_vscf.validation import exact_nmode_dvr
from pyscf_vscf.vscf import (
    VSCFSettings,
    nmode_model_from_pair_surfaces,
    vscf_spectrum,
)
from pyscf_vscf.workflows.scans import local_bond_reduced_mass_amu


PAIRS = ((0, 1), (0, 2), (1, 2))
MANIFOLDS = {
    "fundamental": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    "binary_combination": ((1, 1, 0), (1, 0, 1), (0, 1, 1)),
    "first_overtone": ((2, 0, 0), (0, 2, 0), (0, 0, 2)),
    "triple_combination": ((1, 1, 1),),
}
CRITERIA = {
    "maximum_vscf_exact_centroid_error_cm": 75.0,
    "maximum_grid_centroid_spread_cm": 25.0,
    "maximum_equivalent_vscf_state_spread_cm": 5.0,
    "minimum_exact_manifold_weight": 0.70,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_pair_data(data_root: Path, npts: int, *, tag: str = "") -> tuple[dict, dict, dict]:
    suffix = f"_{tag}" if tag else ""
    summary_path = data_root / f"generation_summary_{npts}{suffix}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pair_data = {}
    shared_scientific = None
    for record in summary["pairs"]:
        pair = tuple(int(value) for value in record["pair_zero_based"])
        meta, arrays = load_grid_npz(data_root / record["path"])
        scientific = meta["scientific"]
        shared = {
            "molecule": scientific["molecule"],
            "electronic_structure": scientific["electronic_structure"],
        }
        if shared_scientific is None:
            shared_scientific = shared
        elif shared != shared_scientific:
            raise ValueError(
                "Pair caches do not share molecule and electronic-structure provenance"
            )
        pair_data[pair] = {
            "meta": meta,
            "R1_A": np.asarray(arrays["R1_A"], dtype=float),
            "R2_A": np.asarray(arrays["R2_A"], dtype=float),
            "E_Eh": np.asarray(arrays["E_Eh"], dtype=float),
        }
    if set(pair_data) != set(PAIRS):
        raise ValueError(f"Expected pair surfaces {PAIRS}, found {tuple(sorted(pair_data))}")
    return summary, pair_data, shared_scientific or {}


def _model_for_indices(pair_data: dict, molecule, indices: np.ndarray, name: str):
    grid_candidates: list[list[np.ndarray]] = [[], [], []]
    surfaces = {}
    for i, j in PAIRS:
        data = pair_data[(i, j)]
        grid_candidates[i].append(data["R1_A"][indices])
        grid_candidates[j].append(data["R2_A"][indices])
        surfaces[(i, j)] = data["E_Eh"][np.ix_(indices, indices)]

    coordinates = []
    for mode, candidates in enumerate(grid_candidates):
        reference = candidates[0]
        for candidate in candidates[1:]:
            if not np.allclose(reference, candidate, rtol=0.0, atol=1e-12):
                raise ValueError(f"Pair caches disagree on the coordinate grid for mode {mode}")
        coordinates.append(reference)

    bonds = (Bond(0, 1), Bond(0, 2), Bond(0, 3))
    masses = tuple(local_bond_reduced_mass_amu(molecule, bond) for bond in bonds)
    return nmode_model_from_pair_surfaces(
        coordinates,
        masses,
        surfaces,
        reference_indices=tuple(len(indices) // 2 for _ in range(3)),
        mode_labels=("N-H1", "N-H2", "N-H3"),
        metadata={"validation_variant": name, "molecule": "NH3"},
        consistency_tolerance_Eh=5e-8,
    )


def _product_vector(modals: tuple[np.ndarray, ...], quanta: tuple[int, ...]) -> np.ndarray:
    vector = modals[0][:, quanta[0]]
    for mode in range(1, len(modals)):
        vector = np.multiply.outer(vector, modals[mode][:, quanta[mode]])
    return np.asarray(vector, dtype=float).reshape(-1)


def _analyze_variant(name: str, model, *, exact_states: int) -> dict:
    requested = tuple(state for states in MANIFOLDS.values() for state in states)
    settings = VSCFSettings(
        max_iterations=300,
        energy_tolerance_Eh=1e-11,
        density_tolerance=1e-9,
        modal_mixing=0.7,
        root_following=True,
    )
    spectrum = vscf_spectrum(model, states=requested, settings=settings)
    vscf_by_state = {transition.quanta: transition for transition in spectrum.transitions}
    state_results = {result.quanta: result for result in spectrum.excited_states}

    exact = exact_nmode_dvr(model, nstates=exact_states)
    one_mode_modals = tuple(np.linalg.eigh(matrix)[1] for matrix in model.one_mode_hamiltonians())
    basis = np.column_stack([_product_vector(one_mode_modals, state) for state in requested])
    overlaps = np.abs(basis.T @ exact.evecs[:, 1:]) ** 2
    manifold_rows = {
        name: [requested.index(state) for state in states] for name, states in MANIFOLDS.items()
    }
    slot_names = tuple(name for name, states in MANIFOLDS.items() for _ in states)
    slot_weights = np.stack([np.sum(overlaps[manifold_rows[name]], axis=0) for name in slot_names])
    rows, columns = linear_sum_assignment(-slot_weights)
    selected_by_manifold: dict[str, list[int]] = {name: [] for name in MANIFOLDS}
    for row, column in zip(rows, columns):
        selected_by_manifold[slot_names[row]].append(int(column + 1))

    manifolds = {}
    for manifold_name, states in MANIFOLDS.items():
        exact_indices = tuple(sorted(selected_by_manifold[manifold_name]))
        if len(exact_indices) != len(states):
            raise RuntimeError(f"Incomplete exact-state assignment for {manifold_name}")
        exact_frequencies = tuple(
            float((exact.evals[index] - exact.evals[0]) * HARTREE_TO_CM) for index in exact_indices
        )
        vscf_frequencies = tuple(float(vscf_by_state[state].frequency_cm) for state in states)
        state_rows = manifold_rows[manifold_name]
        manifold_weights = tuple(
            float(np.sum(overlaps[state_rows, index - 1])) for index in exact_indices
        )
        exact_centroid = float(np.mean(exact_frequencies))
        vscf_centroid = float(np.mean(vscf_frequencies))
        manifolds[manifold_name] = {
            "product_states": states,
            "exact_state_indices": exact_indices,
            "exact_frequencies_cm": exact_frequencies,
            "vscf_frequencies_cm": vscf_frequencies,
            "exact_centroid_cm": exact_centroid,
            "vscf_centroid_cm": vscf_centroid,
            "vscf_minus_exact_centroid_cm": vscf_centroid - exact_centroid,
            "vscf_equivalent_state_spread_cm": float(np.ptp(vscf_frequencies)),
            "exact_manifold_weights": manifold_weights,
            "minimum_exact_manifold_weight": min(manifold_weights),
            "vscf_iterations": tuple(state_results[state].iterations for state in states),
            "vscf_converged": all(state_results[state].converged for state in states),
        }

    assembly = model.metadata["pair_surface_assembly"]
    return {
        "name": name,
        "grid_shape": tuple(len(q) for q in model.coordinates),
        "grid_ranges_A": tuple((float(q[0]), float(q[-1])) for q in model.coordinates),
        "maximum_shared_cut_disagreement_Eh": max(assembly["maximum_cut_disagreement_Eh"]),
        "exact_states_computed": exact_states,
        "exact_ground_energy_Eh": float(exact.evals[0]),
        "vscf_ground_energy_Eh": float(spectrum.ground.energy_Eh),
        "vscf_ground_minus_exact_cm": float(
            (spectrum.ground.energy_Eh - exact.evals[0]) * HARTREE_TO_CM
        ),
        "vscf_ground_converged": spectrum.ground.converged,
        "vscf_ground_iterations": spectrum.ground.iterations,
        "manifolds": manifolds,
    }


def _acceptance(variants: dict, accepted_variant_names: tuple[str, ...]) -> dict:
    accepted = {name: variants[name] for name in accepted_variant_names}
    manifold_names = tuple(MANIFOLDS)
    centroid_errors = [
        abs(variant["manifolds"][name]["vscf_minus_exact_centroid_cm"])
        for variant in accepted.values()
        for name in manifold_names
    ]
    equivalent_spreads = [
        variant["manifolds"][name]["vscf_equivalent_state_spread_cm"]
        for variant in accepted.values()
        for name in manifold_names
        if len(MANIFOLDS[name]) > 1
    ]
    manifold_weights = [
        variant["manifolds"][name]["minimum_exact_manifold_weight"]
        for variant in accepted.values()
        for name in manifold_names
    ]
    grid_spreads = {}
    for name in manifold_names:
        exact_values = [
            variant["manifolds"][name]["exact_centroid_cm"] for variant in accepted.values()
        ]
        vscf_values = [
            variant["manifolds"][name]["vscf_centroid_cm"] for variant in accepted.values()
        ]
        grid_spreads[name] = {
            "exact_centroid_spread_cm": float(np.ptp(exact_values)),
            "vscf_centroid_spread_cm": float(np.ptp(vscf_values)),
        }
    converged_manifolds = tuple(
        name
        for name, spreads in grid_spreads.items()
        if max(spreads.values()) <= CRITERIA["maximum_grid_centroid_spread_cm"]
    )
    maximum_converged_spread = (
        max(max(grid_spreads[name].values()) for name in converged_manifolds)
        if converged_manifolds
        else None
    )
    observed = {
        "maximum_vscf_exact_centroid_error_cm": max(centroid_errors),
        "maximum_converged_manifold_grid_spread_cm": maximum_converged_spread,
        "maximum_equivalent_vscf_state_spread_cm": max(equivalent_spreads),
        "minimum_exact_manifold_weight": min(manifold_weights),
        "all_vscf_states_converged": all(
            variant["vscf_ground_converged"]
            and all(manifold["vscf_converged"] for manifold in variant["manifolds"].values())
            for variant in accepted.values()
        ),
    }
    passed = bool(
        converged_manifolds
        and observed["maximum_vscf_exact_centroid_error_cm"]
        <= CRITERIA["maximum_vscf_exact_centroid_error_cm"]
        and observed["maximum_converged_manifold_grid_spread_cm"]
        <= CRITERIA["maximum_grid_centroid_spread_cm"]
        and observed["maximum_equivalent_vscf_state_spread_cm"]
        <= CRITERIA["maximum_equivalent_vscf_state_spread_cm"]
        and observed["minimum_exact_manifold_weight"] >= CRITERIA["minimum_exact_manifold_weight"]
        and observed["all_vscf_states_converged"]
        and "fundamental" in converged_manifolds
    )
    return {
        "criteria": CRITERIA,
        "accepted_variants": accepted_variant_names,
        "converged_manifolds": converged_manifolds,
        "manifold_policy": (
            "All target manifolds are reported. A manifold enters the converged molecular "
            "error budget only when both exact and VSCF centroid spreads across the final "
            "three window variants do not exceed 25 cm^-1; the fundamental must pass."
        ),
        "excluded_diagnostic_variants": {
            "coarse_13x13_from_25": (
                "Retained to document the failed coarse-grid test; excluded from the "
                "converged plateau because it violates the predefined 25 cm^-1 criterion."
            ),
            "narrow_21x21_from_25": (
                "Retained to document artificial confinement in the narrowed domain; "
                "excluded because it violates the predefined 25 cm^-1 criterion."
            ),
            "repeated_25x25_from_wide31": (
                "Independent same-coordinate electronic points retained as a reproducibility "
                "diagnostic; the primary 25-point archive is used in the plateau."
            ),
            "wide_31x31": (
                "Retained as the first expanded-window diagnostic; not on the final plateau."
            ),
        },
        "observed": observed,
        "grid_spreads_by_manifold": grid_spreads,
        "passed": passed,
    }


def analyze(data_root: Path, *, exact_states: int) -> dict:
    summary25, pair_data25, scientific25 = _load_pair_data(data_root, 25)
    summary31, pair_data31, scientific31 = _load_pair_data(data_root, 31)
    summary_wide, pair_data_wide, scientific_wide = _load_pair_data(data_root, 31, tag="wide")
    summary37, pair_data37, scientific37 = _load_pair_data(data_root, 37, tag="wider")
    summary43, pair_data43, scientific43 = _load_pair_data(data_root, 43, tag="widest")
    summary49, pair_data49, scientific49 = _load_pair_data(data_root, 49, tag="converged")
    if not (
        scientific25
        == scientific31
        == scientific_wide
        == scientific37
        == scientific43
        == scientific49
    ):
        raise ValueError("NH3 grid caches have incompatible molecular or electronic provenance")
    molecule = read_xyz(data_root / summary25["optimized_geometry"]["path"])
    variant_inputs = {
        "coarse_13x13_from_25": (pair_data25, np.arange(0, 25, 2)),
        "narrow_21x21_from_25": (pair_data25, np.arange(2, 23)),
        "full_25x25": (pair_data25, np.arange(25)),
        "dense_31x31": (pair_data31, np.arange(31)),
        "wide_31x31": (pair_data_wide, np.arange(31)),
        "repeated_25x25_from_wide31": (pair_data_wide, np.arange(3, 28)),
        "wider_37x37": (pair_data37, np.arange(37)),
        "widest_43x43": (pair_data43, np.arange(43)),
        "converged_49x49": (pair_data49, np.arange(49)),
    }
    variants = {}
    for name, (pair_data, indices) in variant_inputs.items():
        variants[name] = _analyze_variant(
            name,
            _model_for_indices(pair_data, molecule, indices, name),
            exact_states=exact_states,
        )
    accepted_variant_names = ("wider_37x37", "widest_43x43", "converged_49x49")
    return {
        "schema_version": 1,
        "package_version": __version__,
        "system": "NH3",
        "coordinate_model": "three frozen local N-H bond lengths",
        "potential_model": "one-mode representation plus all three two-mode corrections",
        "kinetic_model": "separable constant local N-H reduced masses",
        "exact_reference": "iterative exact 3D sinc-DVR diagonalization of the identical model",
        "scope": (
            "This validates the state-specific VSCF implementation on a non-water, three-mode "
            "molecular Hamiltonian. It does not establish full-dimensional NH3 spectroscopic "
            "accuracy or validate omitted bend, inversion, and kinetic-coupling terms."
        ),
        "electronic_structure_provenance": scientific25,
        "variants": variants,
        "acceptance": _acceptance(variants, accepted_variant_names),
    }


def _write_manifest(data_root: Path, report_path: Path) -> None:
    summary_paths = (
        "generation_summary_25.json",
        "generation_summary_31.json",
        "generation_summary_31_wide.json",
        "generation_summary_37_wider.json",
        "generation_summary_43_widest.json",
        "generation_summary_49_converged.json",
    )
    summaries = [
        json.loads((data_root / path).read_text(encoding="utf-8")) for path in summary_paths
    ]
    relative_paths = [
        "nh3_initial.xyz",
        "nh3_optimized.xyz",
        "generation_25.log",
        "generation_31.log",
        "generation_31_wide.log",
        "generation_37_wider.log",
        "generation_43_widest.log",
        "generation_49_converged.log",
        *summary_paths,
        *(record["path"] for summary in summaries for record in summary["pairs"]),
        "report_coarse_13_diagnostic.json",
        "report_window_030_diagnostic.json",
        report_path.name,
    ]
    manifest = {
        "schema_version": 1,
        "purpose": "reproducible non-water, three-mode package validation",
        "files": {
            relative: {"sha256": _sha256(data_root / relative)} for relative in relative_paths
        },
        "regeneration_command": (
            "uv run python scripts/generate_nh3_three_mode.py --npts 25 "
            "--workers 8 --threads-per-worker 1 && uv run python "
            "scripts/generate_nh3_three_mode.py --npts 31 --workers 8 "
            "--threads-per-worker 1 && uv run python "
            "scripts/generate_nh3_three_mode.py --npts 31 --half-width 0.30 --tag wide "
            "--workers 8 --threads-per-worker 1 && uv run python "
            "scripts/expand_nh3_three_mode.py --source-npts 31 --source-tag wide "
            "--target-npts 37 --target-half-width 0.36 --target-tag wider --workers 8 "
            "--threads-per-worker 1 && uv run python "
            "scripts/expand_nh3_three_mode.py --source-npts 37 --source-tag wider "
            "--target-npts 43 --target-half-width 0.42 --target-tag widest --workers 8 "
            "--threads-per-worker 1 && uv run python "
            "scripts/expand_nh3_three_mode.py --source-npts 43 --source-tag widest "
            "--target-npts 49 --target-half-width 0.48 --target-tag converged --workers 8 "
            "--threads-per-worker 1"
        ),
        "analysis_command": "uv run python scripts/validate_nh3_three_mode.py",
    }
    (data_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("validation_data/nh3_three_mode"),
    )
    parser.add_argument("--exact-states", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.exact_states < 20:
        parser.error("--exact-states must be at least 20 for the target manifolds")
    output = args.output or args.data_root / "report.json"
    report = analyze(args.data_root, exact_states=args.exact_states)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if output.resolve().parent == args.data_root.resolve():
        _write_manifest(args.data_root, output)
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    return 0 if report["acceptance"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
