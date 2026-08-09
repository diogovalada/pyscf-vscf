from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from scipy.interpolate import RegularGridInterpolator

from pyscf_vscf.constants import MASS_AMU
from pyscf_vscf.coordinates import (
    TriatomicValenceCoordinateMap,
    coordinate_map_fingerprint,
)
from pyscf_vscf.kinetic import (
    JacobiGridProjection,
    TriatomicJ0Hamiltonian,
    TriatomicJ0KineticOperator,
    TriatomicJacobiTransform,
    potential_on_jacobi_grid,
    solve_triatomic_direct_dvr,
)
from pyscf_vscf.nmode import (
    FitDiagnostics,
    NModeSurfaceModel,
    TensorProductSurface,
    nmode_pes_fingerprint,
)


def _operator(
    masses: tuple[float, float, float] | None = None,
) -> TriatomicJ0KineticOperator:
    return TriatomicJ0KineticOperator(
        radial_r_A=np.linspace(0.75, 1.25, 5),
        radial_R_A=np.linspace(0.55, 1.35, 5),
        angular_order=5,
        masses_amu=masses or (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["H"]),
        atom_indices=(1, 0, 2),
    )


def _analytic_potential(operator: TriatomicJ0KineticOperator) -> np.ndarray:
    radial_r, radial_R, angular_x = np.meshgrid(
        *operator.coordinate_grids,
        indexing="ij",
    )
    return (
        0.18 * np.square(radial_r - 0.98)
        + 0.11 * np.square(radial_R - 0.91)
        + 0.035 * np.square(angular_x + 0.25)
        + 0.012 * (radial_r - 0.98) * (radial_R - 0.91) * angular_x
    )


def _surface(
    axis: np.ndarray,
    values: np.ndarray,
    *,
    method: str,
) -> TensorProductSurface:
    backend_method = "cubic_legacy" if method == "cubic" else method
    kwargs = {"method": backend_method, "bounds_error": True}
    interpolator = RegularGridInterpolator((axis,), values, **kwargs)
    residual = np.asarray(interpolator(axis[:, None])).reshape(values.shape) - values
    components = 1 if values.ndim == 1 else values.shape[-1]
    maximum = tuple(
        float(value) for value in np.max(np.abs(residual.reshape(-1, components)), axis=0)
    )
    return TensorProductSurface(
        axes=(axis,),
        node_values=values,
        method=method,
        diagnostics=FitDiagnostics(
            method=method,
            n_training_points=axis.size,
            training_max_abs_error=maximum,
        ),
    )


def _source_lineage(label: str) -> dict[str, object]:
    return {
        "schema": "pyscf-vscf-electronic-source-lineage",
        "schema_version": 1,
        "provider_scientific_fingerprint": label,
        "point_causal_fingerprints": [f"{label}-point"],
    }


def _linear_valence_surface(
    transform: TriatomicJacobiTransform,
    operator: TriatomicJ0KineticOperator,
) -> tuple[
    NModeSurfaceModel,
    TriatomicValenceCoordinateMap,
    np.ndarray,
    np.ndarray,
]:
    meshes = np.meshgrid(*operator.coordinate_grids, indexing="ij")
    jacobi = np.stack(meshes, axis=-1)
    valence = transform.jacobi_to_valence(jacobi)
    reference = transform.jacobi_to_valence(
        np.array(
            [
                operator.radial_r_A[operator.radial_r_A.size // 2],
                operator.radial_R_A[operator.radial_R_A.size // 2],
                operator.angular_x[operator.angular_x.size // 2],
            ]
        )
    )
    coefficients = np.array([0.07, -0.05, 0.02])
    coordinate_map = _coordinate_map(reference)
    energy_surfaces = {}
    dipole_surfaces = {}
    for mode in range(3):
        lower = float(np.min(valence[..., mode])) - 0.02
        upper = float(np.max(valence[..., mode])) + 0.02
        axis = np.unique(np.concatenate((np.linspace(lower, upper, 5), [reference[mode]])))
        energy_surfaces[(mode,)] = _surface(
            axis,
            coefficients[mode] * (axis - reference[mode]),
            method="linear",
        )
        dipole_surfaces[(mode,)] = _surface(
            axis,
            np.zeros((axis.size, 3)),
            method="linear",
        )
    model = NModeSurfaceModel(
        coordinate_ids=("a", "b", "gamma"),
        coordinate_units=("angstrom", "angstrom", "radian"),
        coordinate_map_payload=coordinate_map.fingerprint_payload(),
        coordinate_map_fingerprint=coordinate_map_fingerprint(coordinate_map),
        reference_values=reference,
        reference_energy_Eh=-76.0,
        reference_dipole_body_au=np.zeros(3),
        energy_increments=energy_surfaces,
        dipole_increments=dipole_surfaces,
        source_lineage=_source_lineage("analytic-provider"),
    )
    return model, coordinate_map, reference, coefficients


def _coordinate_map(reference: np.ndarray) -> TriatomicValenceCoordinateMap:
    a, b, gamma = reference
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [a, 0.0, 0.0],
            [b * np.cos(gamma), b * np.sin(gamma), 0.0],
        ]
    )
    return TriatomicValenceCoordinateMap(
        geometry,
        center_atom=0,
        outer_atom_1=1,
        outer_atom_2=2,
        coordinate_ids=("a", "b", "gamma"),
    )


def _common_harmonic_valence_surface(
    transforms: tuple[TriatomicJacobiTransform, ...],
    operator: TriatomicJ0KineticOperator,
) -> tuple[NModeSurfaceModel, TriatomicValenceCoordinateMap]:
    reference = np.array([0.98, 0.98, 1.82])
    coordinate_map = _coordinate_map(reference)
    all_valence = []
    meshes = np.meshgrid(*operator.coordinate_grids, indexing="ij")
    jacobi = np.stack(meshes, axis=-1)
    for transform in transforms:
        all_valence.append(transform.jacobi_to_valence(jacobi).reshape(-1, 3))
    pooled = np.concatenate(all_valence, axis=0)
    force_constants = (0.18, 0.16, 0.04)
    energy_surfaces = {}
    dipole_surfaces = {}
    for mode, force_constant in enumerate(force_constants):
        lower = float(np.min(pooled[:, mode])) - 0.03
        upper = float(np.max(pooled[:, mode])) + 0.03
        axis = np.unique(np.concatenate((np.linspace(lower, upper, 7), [reference[mode]])))
        energy_surfaces[(mode,)] = _surface(
            axis,
            force_constant * np.square(axis - reference[mode]),
            method="cubic",
        )
        dipole_surfaces[(mode,)] = _surface(
            axis,
            np.zeros((axis.size, 3)),
            method="cubic",
        )
    return (
        NModeSurfaceModel(
            coordinate_ids=("a", "b", "gamma"),
            coordinate_units=("angstrom", "angstrom", "radian"),
            coordinate_map_payload=coordinate_map.fingerprint_payload(),
            coordinate_map_fingerprint=coordinate_map_fingerprint(coordinate_map),
            reference_values=reference,
            reference_energy_Eh=-76.0,
            reference_dipole_body_au=np.zeros(3),
            energy_increments=energy_surfaces,
            dipole_increments=dipole_surfaces,
            source_lineage=_source_lineage("common-born-oppenheimer-surface"),
        ),
        coordinate_map,
    )


def test_valence_jacobi_transform_round_trip_and_mass_dependence() -> None:
    light = TriatomicJacobiTransform(
        (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["H"]),
        atom_indices=(1, 0, 2),
    )
    heavy = TriatomicJacobiTransform(
        (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["D"]),
        atom_indices=(1, 0, 2),
    )
    valence = np.array(
        [
            [0.96, 0.97, 1.82],
            [0.83, 1.14, 1.37],
            [1.21, 0.79, 2.31],
        ]
    )

    jacobi = light.valence_to_jacobi(valence)
    np.testing.assert_allclose(light.jacobi_to_valence(jacobi), valence, atol=3e-15)
    assert not np.allclose(jacobi[:, 1:], heavy.valence_to_jacobi(valence)[:, 1:])
    assert light.fingerprint() != heavy.fingerprint()

    with pytest.raises(ValueError, match="cosine outside"):
        light.jacobi_to_valence(np.array([1.0, 1.0, 1.01]))
    with pytest.raises(ValueError, match="strictly between"):
        light.valence_to_jacobi(np.array([1.0, 1.0, 0.0]))


def test_gauss_legendre_j2_spectrum_and_operator_hermiticity() -> None:
    kinetic = _operator()
    angular_momenta = np.arange(kinetic.angular_order, dtype=float)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(kinetic.angular_j2),
        angular_momenta * (angular_momenta + 1.0),
        atol=2e-13,
    )
    identity = np.eye(kinetic.dimension)
    dense = np.column_stack(
        [kinetic.apply(identity[:, column]) for column in range(kinetic.dimension)]
    )
    np.testing.assert_allclose(dense, dense.T, atol=2e-13)

    rng = np.random.default_rng(91)
    left = rng.normal(size=kinetic.dimension)
    right = rng.normal(size=kinetic.dimension)
    assert left @ kinetic.apply(right) == pytest.approx(
        kinetic.apply(left) @ right,
        abs=2e-12,
    )


def test_operator_declares_scaled_measure_citation_and_reduced_mass_placement() -> None:
    masses = (1.25, 15.75, 18.50)
    kinetic = _operator(masses)
    payload = kinetic.fingerprint_payload()

    assert kinetic.reduced_masses_amu == pytest.approx(
        (
            masses[1] * masses[2] / (masses[1] + masses[2]),
            masses[0] * (masses[1] + masses[2]) / sum(masses),
        )
    )
    assert payload["citation_doi"] == "10.1016/j.cpc.2003.10.003"
    assert payload["wavefunction_scaling"] == "Phi=r*R*Psi"
    assert payload["integration_measure"] == "dr_dR_dx"
    assert payload["atom_indices_outer1_center_outer2"] == [1, 0, 2]


def test_projected_kinetic_terms_reconstruct_complete_modal_matrix() -> None:
    kinetic = _operator()
    rng = np.random.default_rng(17)
    bases = tuple(np.linalg.qr(rng.normal(size=(size, size)))[0] for size in kinetic.shape)
    projected = kinetic.matrix_elements(bases)
    identity = np.eye(kinetic.shape[0])
    expected = np.kron(
        np.kron(projected.radial_r_Eh, identity),
        identity,
    )
    expected += np.kron(
        np.kron(identity, projected.radial_R_Eh),
        identity,
    )
    expected += np.kron(
        np.kron(projected.inverse_r2_Eh, identity),
        projected.angular_j2,
    )
    expected += np.kron(
        np.kron(identity, projected.inverse_R2_Eh),
        projected.angular_j2,
    )

    modal_to_grid = np.kron(np.kron(bases[0], bases[1]), bases[2])
    grid_identity = np.eye(kinetic.dimension)
    dense_grid = np.column_stack(
        [kinetic.apply(grid_identity[:, column]) for column in range(kinetic.dimension)]
    )
    np.testing.assert_allclose(
        expected,
        modal_to_grid.T @ dense_grid @ modal_to_grid,
        atol=2e-12,
    )


def test_direct_triatomic_dvr_matches_dense_oracle_and_reports_edges() -> None:
    kinetic = _operator()
    hamiltonian = TriatomicJ0Hamiltonian(kinetic, _analytic_potential(kinetic))
    result = solve_triatomic_direct_dvr(
        hamiltonian,
        nstates=6,
        dense_dimension_threshold=kinetic.dimension,
    )

    assert result.energies_Eh.shape == (6,)
    assert np.max(result.residual_norms_Eh) < 2e-12
    assert result.radial_edge_probabilities.shape == (6, 2, 2)
    assert np.all(result.radial_edge_probabilities >= 0.0)
    np.testing.assert_allclose(result.eigenvectors.T @ result.eigenvectors, np.eye(6), atol=2e-13)
    assert result.hamiltonian_fingerprint == hamiltonian.fingerprint()
    assert hamiltonian.source_projection_fingerprint is None


def test_common_valence_pes_isotope_replay_preserves_provenance() -> None:
    mass_sets = {
        "H2O": (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["H"]),
        "HDO": (MASS_AMU["D"], MASS_AMU["O"], MASS_AMU["H"]),
        "D2O": (MASS_AMU["D"], MASS_AMU["O"], MASS_AMU["D"]),
    }
    transforms = tuple(
        TriatomicJacobiTransform(masses, atom_indices=(1, 0, 2)) for masses in mass_sets.values()
    )
    model, coordinate_map = _common_harmonic_valence_surface(
        transforms,
        _operator(mass_sets["H2O"]),
    )
    ground_energies = {}
    maximum_edges = {}
    projection_fingerprints = set()
    for (label, masses), transform in zip(mass_sets.items(), transforms):
        operator = _operator(masses)
        projection = potential_on_jacobi_grid(
            model,
            coordinate_map,
            transform,
            operator,
        )
        result = solve_triatomic_direct_dvr(
            TriatomicJ0Hamiltonian.from_projection(
                operator,
                projection,
                metadata={"purpose": "coarse isotope provenance regression"},
            ),
            nstates=2,
            dense_dimension_threshold=operator.dimension,
        )
        ground_energies[label] = result.energies_Eh[0]
        maximum_edges[label] = float(np.max(result.radial_edge_probabilities))
        assert projection.source_pes_fingerprint == nmode_pes_fingerprint(model)
        projection_fingerprints.add(projection.fingerprint())

    # This coarse-grid trend is a smoke test, not a converged isotope benchmark.
    assert ground_energies["H2O"] > ground_energies["HDO"] > ground_energies["D2O"]
    assert max(maximum_edges.values()) < 0.04
    assert len(projection_fingerprints) == 3


def test_valence_nmode_surface_projects_without_jacobi_extrapolation() -> None:
    operator = _operator()
    transform = TriatomicJacobiTransform(operator.masses_amu, operator.atom_indices)
    model, coordinate_map, reference, coefficients = _linear_valence_surface(transform, operator)
    projected = potential_on_jacobi_grid(model, coordinate_map, transform, operator)
    meshes = np.meshgrid(*operator.coordinate_grids, indexing="ij")
    valence = transform.jacobi_to_valence(np.stack(meshes, axis=-1))
    expected = np.sum(coefficients * (valence - reference), axis=-1)

    assert isinstance(projected, JacobiGridProjection)
    np.testing.assert_allclose(projected.potential_Eh, expected, atol=2e-15)
    assert not projected.potential_Eh.flags.writeable
    assert projected.source_pes_fingerprint == nmode_pes_fingerprint(model)
    assert projected.source_coordinate_map_fingerprint == coordinate_map_fingerprint(
        coordinate_map
    )
    assert projected.transform_fingerprint == transform.fingerprint()
    assert projected.kinetic_fingerprint == operator.fingerprint()
    expanded = TriatomicJ0KineticOperator(
        radial_r_A=np.linspace(0.45, 1.25, 5),
        radial_R_A=operator.radial_R_A,
        angular_order=operator.angular_order,
        masses_amu=operator.masses_amu,
        atom_indices=operator.atom_indices,
    )
    with pytest.raises(ValueError, match="out of bounds"):
        potential_on_jacobi_grid(model, coordinate_map, transform, expanded)


def test_valence_projection_rejects_foreign_map_and_atom_order() -> None:
    operator = _operator()
    transform = TriatomicJacobiTransform(operator.masses_amu, operator.atom_indices)
    model, coordinate_map, _, _ = _linear_valence_surface(transform, operator)
    swapped_map = TriatomicValenceCoordinateMap(
        coordinate_map.reference_geometry_A,
        center_atom=0,
        outer_atom_1=2,
        outer_atom_2=1,
        coordinate_ids=("a", "b", "gamma"),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        potential_on_jacobi_grid(model, swapped_map, transform, operator)

    wrong_order = TriatomicJacobiTransform(
        operator.masses_amu,
        atom_indices=(2, 0, 1),
    )
    with pytest.raises(ValueError, match="atom order"):
        potential_on_jacobi_grid(model, coordinate_map, wrong_order, operator)

    projection = potential_on_jacobi_grid(model, coordinate_map, transform, operator)
    heavy_operator = _operator((MASS_AMU["D"], MASS_AMU["O"], MASS_AMU["D"]))
    with pytest.raises(ValueError, match="different kinetic operator"):
        TriatomicJ0Hamiltonian.from_projection(heavy_operator, projection)

    bound = TriatomicJ0Hamiltonian.from_projection(operator, projection)
    assert bound.source_projection_fingerprint == projection.fingerprint()
    assert bound.source_pes_fingerprint == nmode_pes_fingerprint(model)

    changed = replace(bound, potential_Eh=bound.potential_Eh + 1e-6)
    assert changed.source_projection_fingerprint is None
    with pytest.raises(TypeError, match="unexpected keyword"):
        TriatomicJ0Hamiltonian(
            operator,
            projection.potential_Eh,
            source_projection_fingerprint="forged",
        )


def test_triatomic_hamiltonian_metadata_is_recursively_immutable() -> None:
    kinetic = _operator()
    nested = ["source-a", "source-b"]
    hamiltonian = TriatomicJ0Hamiltonian(
        kinetic,
        _analytic_potential(kinetic),
        metadata={"nested": {"sources": nested}},
    )
    fingerprint = hamiltonian.fingerprint()
    nested.append("late-mutation")

    assert tuple(hamiltonian.metadata["nested"]["sources"]) == (
        "source-a",
        "source-b",
    )
    assert hamiltonian.fingerprint() == fingerprint
