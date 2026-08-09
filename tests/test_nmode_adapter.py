from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.coordinates import LinearDisplacementCoordinateMap, coordinate_map_fingerprint
from pyscf_vscf.nmode import (
    FitDiagnostics,
    NModeSurfaceModel,
    TensorProductSurface,
    nmode_pes_fingerprint,
    nmode_potential_from_surface,
)
from pyscf_vscf.validation import exact_nmode_dvr
from pyscf_vscf.vscf import (
    dump_nmode_model,
    load_nmode_model,
    nmode_model_fingerprint,
    solve_vscf_state,
)


def _surface(axes: tuple[np.ndarray, ...], values: np.ndarray) -> TensorProductSurface:
    components = 1 if values.ndim == len(axes) else values.shape[-1]
    return TensorProductSurface(
        axes=axes,
        node_values=values,
        method="linear",
        diagnostics=FitDiagnostics(
            method="linear",
            n_training_points=int(np.prod([axis.size for axis in axes])),
            training_max_abs_error=(0.0,) * components,
        ),
    )


def _model(*, include_pairs: bool = True, units: tuple[str, ...] | None = None):
    reference_geometry = np.array([[0.0, 0.0, 0.0], [0.97, 0.0, 0.0], [-0.42, 1.31, 0.0]])
    displacements = np.zeros((3, 3, 3))
    displacements[0, 1, 0] = 1.0
    displacements[1, 2, 1] = 1.0
    displacements[2, 0, 2] = 1.0
    coordinate_units = units or ("angstrom", "angstrom", "angstrom")
    coordinate_map = LinearDisplacementCoordinateMap(
        reference_geometry_A=reference_geometry,
        coordinate_ids=("H_x", "F_y", "O_z"),
        units=coordinate_units,
        reference_values=np.zeros(3),
        displacements_A_per_unit=displacements,
    )
    axes = tuple(np.linspace(-0.3, 0.3, 7) for _ in range(3))
    energy = {
        (mode,): _surface((axes[mode],), coefficient * axes[mode] ** 2)
        for mode, coefficient in enumerate((0.04, 0.05, 0.06))
    }
    dipole = {
        (mode,): _surface((axes[mode],), np.zeros((axes[mode].size, 3))) for mode in range(3)
    }
    if include_pairs:
        pair_values = {
            (0, 1): 0.015 * axes[0][:, None] * axes[1][None, :],
            (0, 2): 0.008 * axes[0][:, None] ** 2 * axes[2][None, :] ** 2,
            (1, 2): -0.010 * axes[1][:, None] * axes[2][None, :],
        }
        for subset, values in pair_values.items():
            subset_axes = tuple(axes[mode] for mode in subset)
            energy[subset] = _surface(subset_axes, values)
            dipole[subset] = _surface(subset_axes, np.zeros((*values.shape, 3)))
    return NModeSurfaceModel(
        coordinate_ids=coordinate_map.coordinate_ids,
        coordinate_units=coordinate_map.units,
        coordinate_map_payload=coordinate_map.fingerprint_payload(),
        coordinate_map_fingerprint=coordinate_map_fingerprint(coordinate_map),
        reference_values=coordinate_map.reference_values,
        reference_energy_Eh=-100.0,
        reference_dipole_body_au=np.array([0.2, -0.1, 0.3]),
        energy_increments=energy,
        dipole_increments=dipole,
        source_lineage={
            "schema": "pyscf-vscf-electronic-source-lineage",
            "schema_version": 1,
            "provider_scientific_fingerprint": "analytic",
            "point_causal_fingerprints": ["p0"],
        },
        annotations={"path": "/not-scientific"},
    )


def test_adapter_maps_anchored_increments_without_minimum_shift_or_reference_duplication() -> None:
    model = _model()
    grids = tuple(np.linspace(-0.2, 0.2, 5) for _ in range(3))
    adapted = nmode_potential_from_surface(model, grids, (1.0, 1.2, 1.4))

    for mode in range(3):
        expected = model.energy_increments[(mode,)].evaluate(grids[mode][:, None])
        np.testing.assert_allclose(adapted.one_mode_potentials_Eh[mode], expected)
        assert adapted.one_mode_potentials_Eh[mode][2] == pytest.approx(0.0, abs=1e-15)
    for subset, coupling in adapted.two_mode_couplings_Eh.items():
        mesh = np.stack(np.meshgrid(*(grids[mode] for mode in subset), indexing="ij"), axis=-1)
        np.testing.assert_allclose(coupling, model.energy_increments[subset].evaluate(mesh))
    assert adapted.provenance["source_reference_energy_Eh"] == -100.0
    assert adapted.provenance["source_pes_fingerprint"] == nmode_pes_fingerprint(model)


def test_non_water_three_mode_adapter_reaches_vscf_and_exact_dvr() -> None:
    model = _model(include_pairs=False)
    grids = tuple(np.linspace(-0.3, 0.3, 7) for _ in range(3))
    adapted = nmode_potential_from_surface(model, grids, (1.0, 1.2, 1.4))

    vscf = solve_vscf_state(adapted)
    exact = exact_nmode_dvr(adapted, nstates=2)

    assert vscf.converged
    assert vscf.energy_Eh == pytest.approx(exact.evals[0], abs=2e-11)


def test_adapter_rejects_rank_three_instead_of_dropping_it() -> None:
    model = _model()
    axes = tuple(np.linspace(-0.3, 0.3, 7) for _ in range(3))
    values = 0.1 * axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
    triple_energy = _surface(axes, values)
    triple_dipole = _surface(axes, np.zeros((*values.shape, 3)))
    expanded = replace(
        model,
        energy_increments={**model.energy_increments, (0, 1, 2): triple_energy},
        dipole_increments={**model.dipole_increments, (0, 1, 2): triple_dipole},
    )

    with pytest.raises(ValueError, match="cannot discard rank-3"):
        nmode_potential_from_surface(expanded, axes, (1.0, 1.2, 1.4))


def test_adapter_rejects_units_grid_shape_uniformity_and_bounds() -> None:
    model = _model()
    grids = tuple(np.linspace(-0.2, 0.2, 5) for _ in range(3))
    mixed_units = _model(units=("angstrom", "dimensionless", "angstrom"))
    with pytest.raises(ValueError, match="requires Angstrom"):
        nmode_potential_from_surface(mixed_units, grids, (1.0, 1.2, 1.4))
    with pytest.raises(ValueError, match="one solver grid"):
        nmode_potential_from_surface(model, grids[:2], (1.0, 1.2, 1.4))
    nonuniform = (np.array([-0.2, -0.1, 0.01, 0.1, 0.2]), *grids[1:])
    with pytest.raises(ValueError, match="uniformly spaced"):
        nmode_potential_from_surface(model, nonuniform, (1.0, 1.2, 1.4))
    outside = (np.linspace(-0.4, 0.4, 5), *grids[1:])
    with pytest.raises(ValueError, match="out of bounds"):
        nmode_potential_from_surface(model, outside, (1.0, 1.2, 1.4))


def test_surface_model_rejects_missing_one_mode_closure() -> None:
    model = _model()
    energies = dict(model.energy_increments)
    dipoles = dict(model.dipole_increments)
    del energies[(0,)]
    del dipoles[(0,)]

    with pytest.raises(ValueError, match=r"requires lower-rank subset \(0,\)"):
        replace(model, energy_increments=energies, dipole_increments=dipoles)


def test_adapter_provenance_round_trips_without_changing_released_model_identity(
    tmp_path: Path,
) -> None:
    model = _model(include_pairs=False)
    grids = tuple(np.linspace(-0.3, 0.3, 7) for _ in range(3))
    adapted = nmode_potential_from_surface(model, grids, (1.0, 1.2, 1.4))
    without_provenance = replace(adapted, provenance={"host": "different"})
    path = tmp_path / "adapted.npz"

    assert nmode_model_fingerprint(without_provenance) == nmode_model_fingerprint(adapted)
    dump_nmode_model(path, adapted)
    restored = load_nmode_model(path)
    assert dict(restored.provenance) == dict(adapted.provenance)
    assert nmode_model_fingerprint(restored) == nmode_model_fingerprint(adapted)
