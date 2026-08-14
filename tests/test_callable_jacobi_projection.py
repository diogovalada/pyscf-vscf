from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.constants import MASS_AMU
from pyscf_vscf.coordinates import (
    TriatomicValenceCoordinateMap,
    coordinate_map_fingerprint,
)
from pyscf_vscf.kinetic import (
    TriatomicJ0Hamiltonian,
    TriatomicJ0KineticOperator,
    TriatomicJacobiTransform,
    potential_on_jacobi_grid_from_callable,
    solve_triatomic_direct_dvr,
)
from pyscf_vscf.transition_moments import (
    build_vci_dipole_operator,
    dipole_on_jacobi_grid_from_callable,
    dump_vci_dipole_projection,
    load_vci_dipole_projection,
    vci_transition_moments,
)
from pyscf_vscf.vci import (
    VCISettings,
    build_triatomic_vscf_modal_basis,
    solve_vci,
)


SOURCE_PES_FINGERPRINT = hashlib.sha256(
    b"test_callable_jacobi_projection:_energy_formula:v1"
).hexdigest()
SOURCE_DMS_FINGERPRINT = hashlib.sha256(
    b"test_callable_jacobi_projection:_dipole_formula:v1"
).hexdigest()


def _coordinate_map(
    *,
    coordinate_ids: tuple[str, str, str] = ("a", "b", "gamma"),
) -> TriatomicValenceCoordinateMap:
    reference = np.array([0.97, 1.01, 1.80])
    geometry = np.array(
        [
            [0.0, 0.0, 0.0],
            [reference[0], 0.0, 0.0],
            [
                reference[1] * np.cos(reference[2]),
                reference[1] * np.sin(reference[2]),
                0.0,
            ],
        ]
    )
    return TriatomicValenceCoordinateMap(
        geometry,
        center_atom=0,
        outer_atom_1=1,
        outer_atom_2=2,
        coordinate_ids=coordinate_ids,
    )


def _kinetic(
    masses_amu: tuple[float, float, float] | None = None,
    *,
    atom_indices: tuple[int, int, int] = (1, 0, 2),
) -> TriatomicJ0KineticOperator:
    return TriatomicJ0KineticOperator(
        radial_r_A=np.linspace(0.78, 1.18, 4),
        radial_R_A=np.linspace(0.62, 1.26, 4),
        angular_order=4,
        masses_amu=masses_amu or (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["H"]),
        atom_indices=atom_indices,
    )


def _energy_formula(valence: np.ndarray, reference: np.ndarray) -> np.ndarray:
    displacement = valence - reference
    return (
        -76.123456789
        + 0.16 * displacement[..., 0] ** 2
        + 0.13 * displacement[..., 1] ** 2
        + 0.04 * displacement[..., 2] ** 2
        + 0.012 * displacement[..., 0] * displacement[..., 1]
        + 0.006 * np.prod(displacement, axis=-1)
    )


def _dipole_formula(valence: np.ndarray, reference: np.ndarray) -> np.ndarray:
    displacement = valence - reference
    triple = np.prod(displacement, axis=-1)
    return np.stack(
        (
            0.11
            + 0.52 * displacement[..., 0]
            + 0.08 * displacement[..., 1] * displacement[..., 2],
            -0.03 - 0.44 * displacement[..., 1] + 0.05 * triple,
            0.02 + 0.31 * displacement[..., 2] - 0.07 * triple,
        ),
        axis=-1,
    )


class _RecordedEnergy:
    def __init__(self, reference: np.ndarray, identity_label: str) -> None:
        self.reference = reference
        self.identity_label = identity_label
        self.call_shapes: list[tuple[int, ...]] = []

    def __call__(self, valence: np.ndarray) -> np.ndarray:
        self.call_shapes.append(valence.shape)
        return _energy_formula(valence, self.reference)


class _RecordedDipole:
    def __init__(self, reference: np.ndarray, identity_label: str) -> None:
        self.reference = reference
        self.identity_label = identity_label
        self.call_shapes: list[tuple[int, ...]] = []

    def __call__(self, valence: np.ndarray) -> np.ndarray:
        self.call_shapes.append(valence.shape)
        return _dipole_formula(valence, self.reference)


def _bound_hamiltonian():
    coordinate_map = _coordinate_map()
    kinetic = _kinetic()
    transform = TriatomicJacobiTransform(kinetic.masses_amu, kinetic.atom_indices)
    projection = potential_on_jacobi_grid_from_callable(
        lambda valence: _energy_formula(valence, coordinate_map.reference_values),
        coordinate_map,
        transform,
        kinetic,
        source_pes_fingerprint=SOURCE_PES_FINGERPRINT,
    )
    hamiltonian = TriatomicJ0Hamiltonian.from_projection(kinetic, projection)
    return coordinate_map, transform, kinetic, projection, hamiltonian


def test_callable_pes_matches_nodes_and_subtracts_reference_without_callable_identity() -> None:
    coordinate_map = _coordinate_map()
    kinetic = _kinetic()
    transform = TriatomicJacobiTransform(kinetic.masses_amu, kinetic.atom_indices)
    first_callable = _RecordedEnergy(coordinate_map.reference_values, "first-object")
    first = potential_on_jacobi_grid_from_callable(
        first_callable,
        coordinate_map,
        transform,
        kinetic,
        source_pes_fingerprint=SOURCE_PES_FINGERPRINT,
    )
    second = potential_on_jacobi_grid_from_callable(
        lambda valence: _energy_formula(valence, coordinate_map.reference_values),
        coordinate_map,
        transform,
        kinetic,
        source_pes_fingerprint=SOURCE_PES_FINGERPRINT,
    )
    meshes = np.meshgrid(*kinetic.coordinate_grids, indexing="ij")
    valence = transform.jacobi_to_valence(np.stack(meshes, axis=-1))
    absolute_grid = _energy_formula(valence, coordinate_map.reference_values)
    reference_energy = _energy_formula(
        coordinate_map.reference_values,
        coordinate_map.reference_values,
    )

    assert first_callable.call_shapes == [(*kinetic.shape, 3), (3,)]
    np.testing.assert_allclose(first.potential_Eh, absolute_grid - reference_energy, atol=2e-15)
    assert first.source_pes_fingerprint == SOURCE_PES_FINGERPRINT
    assert first.source_coordinate_map_fingerprint == coordinate_map_fingerprint(coordinate_map)
    assert first.fingerprint() == second.fingerprint()
    assert not first.potential_Eh.flags.writeable


def test_callable_dms_and_typed_vci_moments_match_direct_3d_dvr(tmp_path: Path) -> None:
    coordinate_map, transform, kinetic, _, hamiltonian = _bound_hamiltonian()
    first_callable = _RecordedDipole(coordinate_map.reference_values, "first-object")
    expansion = dipole_on_jacobi_grid_from_callable(
        first_callable,
        coordinate_map,
        transform,
        hamiltonian,
        source_dms_fingerprint=SOURCE_DMS_FINGERPRINT,
    )
    second = dipole_on_jacobi_grid_from_callable(
        lambda valence: _dipole_formula(valence, coordinate_map.reference_values),
        coordinate_map,
        transform,
        hamiltonian,
        source_dms_fingerprint=SOURCE_DMS_FINGERPRINT,
    )
    meshes = np.meshgrid(*kinetic.coordinate_grids, indexing="ij")
    valence = transform.jacobi_to_valence(np.stack(meshes, axis=-1))
    analytic_grid = _dipole_formula(valence, coordinate_map.reference_values)
    analytic_reference = _dipole_formula(
        coordinate_map.reference_values,
        coordinate_map.reference_values,
    )

    assert first_callable.call_shapes == [(*kinetic.shape, 3), (3,)]
    assert set(expansion.increment_grids_au) == {(0, 1, 2)}
    np.testing.assert_allclose(expansion.reference_dipole_body_au, analytic_reference, atol=0.0)
    np.testing.assert_allclose(
        expansion.increment_grids_au[(0, 1, 2)],
        analytic_grid - analytic_reference,
        atol=2e-15,
    )
    np.testing.assert_allclose(expansion.dipole_grid_au, analytic_grid, atol=2e-15)
    assert expansion.source_dms_fingerprint == SOURCE_DMS_FINGERPRINT
    assert expansion.hamiltonian_fingerprint == hamiltonian.fingerprint()
    assert expansion.metadata["coordinate_map_fingerprint"] == coordinate_map_fingerprint(
        coordinate_map
    )
    assert expansion.metadata["transform_fingerprint"] == transform.fingerprint()
    assert expansion.fingerprint() == second.fingerprint()
    assert not expansion.reference_dipole_body_au.flags.writeable
    assert not expansion.increment_grids_au[(0, 1, 2)].flags.writeable
    assert not expansion.dipole_grid_au.flags.writeable
    assert all(not grid.flags.writeable for grid in expansion.coordinate_grids)
    with pytest.raises(TypeError):
        expansion.metadata["projection"] = "changed"

    modal_basis = build_triatomic_vscf_modal_basis(hamiltonian, 4)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=5, extra_eigenstates=2),
    )
    dipole_operator = build_vci_dipole_operator(expansion, modal_basis, result)
    artifact_path = tmp_path / "callable-dms-projection.npz"
    dump_vci_dipole_projection(expansion, dipole_operator, artifact_path)
    restored_expansion, restored_operator = load_vci_dipole_projection(artifact_path)
    assert restored_expansion.fingerprint() == expansion.fingerprint()
    assert restored_operator.fingerprint() == dipole_operator.fingerprint()
    transitions = vci_transition_moments(dipole_operator, result)
    direct = solve_triatomic_direct_dvr(
        hamiltonian,
        nstates=7,
        dense_dimension_threshold=hamiltonian.dimension,
    )
    flattened_dipole = analytic_grid.reshape((-1, 3))
    for record in transitions:
        direct_vector = np.einsum(
            "i,ic,i->c",
            direct.eigenvectors[:, 0],
            flattened_dipole,
            direct.eigenvectors[:, record.upper_state],
        )
        sign = 1.0 if float(record.transition_dipole_body_au @ direct_vector) >= 0.0 else -1.0
        np.testing.assert_allclose(
            record.transition_dipole_body_au,
            sign * direct_vector,
            atol=4e-11,
        )
        np.testing.assert_allclose(
            record.frequency_Eh,
            direct.energies_Eh[record.upper_state] - direct.energies_Eh[0],
            atol=3e-11,
        )


def test_callable_isotope_replay_is_source_stable_and_mass_bound() -> None:
    coordinate_map = _coordinate_map()
    mass_sets = {
        "H2O": (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["H"]),
        "HDO-outer2": (MASS_AMU["H"], MASS_AMU["O"], MASS_AMU["D"]),
        "D2O": (MASS_AMU["D"], MASS_AMU["O"], MASS_AMU["D"]),
    }
    projections = {}
    expansions = {}
    for label, masses in mass_sets.items():
        kinetic = _kinetic(masses)
        transform = TriatomicJacobiTransform(masses, kinetic.atom_indices)
        projection = potential_on_jacobi_grid_from_callable(
            lambda valence: _energy_formula(valence, coordinate_map.reference_values),
            coordinate_map,
            transform,
            kinetic,
            source_pes_fingerprint=SOURCE_PES_FINGERPRINT,
        )
        hamiltonian = TriatomicJ0Hamiltonian.from_projection(kinetic, projection)
        expansion = dipole_on_jacobi_grid_from_callable(
            lambda valence: _dipole_formula(valence, coordinate_map.reference_values),
            coordinate_map,
            transform,
            hamiltonian,
            source_dms_fingerprint=SOURCE_DMS_FINGERPRINT,
        )
        projections[label] = projection
        expansions[label] = expansion

    assert {value.source_pes_fingerprint for value in projections.values()} == {
        SOURCE_PES_FINGERPRINT
    }
    assert {value.source_dms_fingerprint for value in expansions.values()} == {
        SOURCE_DMS_FINGERPRINT
    }
    assert len({value.fingerprint() for value in projections.values()}) == 3
    assert len({value.fingerprint() for value in expansions.values()}) == 3
    assert not np.array_equal(
        projections["H2O"].potential_Eh,
        projections["HDO-outer2"].potential_Eh,
    )
    assert not np.array_equal(
        expansions["H2O"].dipole_grid_au,
        expansions["HDO-outer2"].dipole_grid_au,
    )
    np.testing.assert_array_equal(
        projections["HDO-outer2"].potential_Eh,
        projections["D2O"].potential_Eh,
    )
    np.testing.assert_array_equal(
        expansions["HDO-outer2"].dipole_grid_au,
        expansions["D2O"].dipole_grid_au,
    )


def test_callable_projection_rejects_fingerprint_and_coordinate_relationship_errors() -> None:
    coordinate_map, transform, kinetic, _, hamiltonian = _bound_hamiltonian()

    def energy(valence: np.ndarray) -> np.ndarray:
        return _energy_formula(valence, coordinate_map.reference_values)

    def dipole(valence: np.ndarray) -> np.ndarray:
        return _dipole_formula(valence, coordinate_map.reference_values)

    with pytest.raises(ValueError, match="source_pes_fingerprint must be non-empty"):
        potential_on_jacobi_grid_from_callable(
            energy,
            coordinate_map,
            transform,
            kinetic,
            source_pes_fingerprint="  ",
        )
    with pytest.raises(ValueError, match="source_dms_fingerprint must be non-empty"):
        dipole_on_jacobi_grid_from_callable(
            dipole,
            coordinate_map,
            transform,
            hamiltonian,
            source_dms_fingerprint="",
        )

    swapped_map = TriatomicValenceCoordinateMap(
        coordinate_map.reference_geometry_A,
        center_atom=0,
        outer_atom_1=2,
        outer_atom_2=1,
        coordinate_ids=coordinate_map.coordinate_ids,
    )
    with pytest.raises(ValueError, match="atom order"):
        potential_on_jacobi_grid_from_callable(
            energy,
            swapped_map,
            transform,
            kinetic,
            source_pes_fingerprint=SOURCE_PES_FINGERPRINT,
        )

    heavy_transform = TriatomicJacobiTransform(
        (MASS_AMU["D"], MASS_AMU["O"], MASS_AMU["D"]),
        transform.atom_indices,
    )
    with pytest.raises(ValueError, match="masses"):
        potential_on_jacobi_grid_from_callable(
            energy,
            coordinate_map,
            heavy_transform,
            kinetic,
            source_pes_fingerprint=SOURCE_PES_FINGERPRINT,
        )

    renamed_map = _coordinate_map(coordinate_ids=("r1", "r2", "theta"))
    with pytest.raises(ValueError, match="Coordinate map fingerprint"):
        dipole_on_jacobi_grid_from_callable(
            dipole,
            renamed_map,
            transform,
            hamiltonian,
            source_dms_fingerprint=SOURCE_DMS_FINGERPRINT,
        )

    changed_transform = TriatomicJacobiTransform(
        transform.masses_amu,
        transform.atom_indices,
        cosine_tolerance=1e-10,
    )
    with pytest.raises(ValueError, match="transform fingerprint"):
        dipole_on_jacobi_grid_from_callable(
            dipole,
            coordinate_map,
            changed_transform,
            hamiltonian,
            source_dms_fingerprint=SOURCE_DMS_FINGERPRINT,
        )


@pytest.mark.parametrize(
    "invalid_fingerprint",
    (pytest.param(None, id="none"), pytest.param(object(), id="object")),
)
def test_callable_projection_rejects_non_string_source_fingerprints(
    invalid_fingerprint: object,
) -> None:
    coordinate_map, transform, kinetic, _, hamiltonian = _bound_hamiltonian()

    with pytest.raises(TypeError, match="source_pes_fingerprint must be a string"):
        potential_on_jacobi_grid_from_callable(
            lambda valence: _energy_formula(valence, coordinate_map.reference_values),
            coordinate_map,
            transform,
            kinetic,
            source_pes_fingerprint=invalid_fingerprint,  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="source_dms_fingerprint must be a string"):
        dipole_on_jacobi_grid_from_callable(
            lambda valence: _dipole_formula(valence, coordinate_map.reference_values),
            coordinate_map,
            transform,
            hamiltonian,
            source_dms_fingerprint=invalid_fingerprint,  # type: ignore[arg-type]
        )


def _energy_grid_shape(valence: np.ndarray) -> np.ndarray:
    return np.zeros(valence.shape[:-1])


@pytest.mark.parametrize(
    ("energy", "match"),
    (
        (lambda valence: -76.0, "must return shape"),
        (lambda valence: np.zeros((*valence.shape[:-1], 1)), "must return shape"),
        (
            lambda valence: np.full(valence.shape[:-1], 1.0 + 2.0j),
            "real-valued Jacobi-grid",
        ),
        (lambda valence: np.full(valence.shape[:-1], np.nan), "non-finite Jacobi-grid"),
        (
            lambda valence: np.zeros(1) if valence.ndim == 1 else _energy_grid_shape(valence),
            "return a scalar",
        ),
        (
            lambda valence: (
                np.asarray(np.nan) if valence.ndim == 1 else _energy_grid_shape(valence)
            ),
            "non-finite reference",
        ),
        (
            lambda valence: (
                np.asarray(1.0 + 2.0j) if valence.ndim == 1 else _energy_grid_shape(valence)
            ),
            "real-valued reference",
        ),
    ),
)
def test_callable_pes_rejects_invalid_outputs(
    energy: Callable[[np.ndarray], object],
    match: str,
) -> None:
    coordinate_map = _coordinate_map()
    kinetic = _kinetic()
    transform = TriatomicJacobiTransform(kinetic.masses_amu, kinetic.atom_indices)
    with pytest.raises(ValueError, match=match):
        potential_on_jacobi_grid_from_callable(
            energy,
            coordinate_map,
            transform,
            kinetic,
            source_pes_fingerprint=SOURCE_PES_FINGERPRINT,
        )


def _dipole_grid_shape(valence: np.ndarray) -> np.ndarray:
    return np.zeros((*valence.shape[:-1], 3))


@pytest.mark.parametrize(
    ("dipole", "match"),
    (
        (lambda valence: np.zeros(3), "must return shape"),
        (lambda valence: np.zeros(valence.shape[:-1]), "must return shape"),
        (
            lambda valence: np.full((*valence.shape[:-1], 3), np.nan),
            "non-finite Jacobi-grid",
        ),
        (
            lambda valence: np.full((*valence.shape[:-1], 3), 1.0 + 2.0j),
            "real-valued Jacobi-grid",
        ),
        (
            lambda valence: np.zeros((1, 3)) if valence.ndim == 1 else _dipole_grid_shape(valence),
            r"return shape \(3,\)",
        ),
        (
            lambda valence: (
                np.full(3, np.nan) if valence.ndim == 1 else _dipole_grid_shape(valence)
            ),
            "non-finite reference",
        ),
        (
            lambda valence: (
                np.full(3, 1.0 + 2.0j) if valence.ndim == 1 else _dipole_grid_shape(valence)
            ),
            "real-valued reference",
        ),
    ),
)
def test_callable_dms_rejects_invalid_outputs(
    dipole: Callable[[np.ndarray], object],
    match: str,
) -> None:
    coordinate_map, transform, _, _, hamiltonian = _bound_hamiltonian()
    with pytest.raises(ValueError, match=match):
        dipole_on_jacobi_grid_from_callable(
            dipole,
            coordinate_map,
            transform,
            hamiltonian,
            source_dms_fingerprint=SOURCE_DMS_FINGERPRINT,
        )
