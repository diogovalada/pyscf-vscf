from __future__ import annotations

import copy

import numpy as np
import pytest

from pyscf_vscf.coordinates import (
    Bond,
    BondLengthCoordinateMap,
    CoordinateMap,
    LinearDisplacementCoordinateMap,
    TriatomicValenceCoordinateMap,
    coordinate_map_fingerprint,
    coordinate_map_from_payload,
)


def _embedded_water() -> np.ndarray:
    return np.array(
        [
            [2.0, -1.0, 0.4],
            [2.9572, -1.0, 0.4],
            [1.7600, -0.0700, 0.4],
            [-3.0, 2.0, 1.0],
        ]
    )


@pytest.mark.parametrize(
    "values",
    [
        np.array([0.82, 1.12, 1.55]),
        np.array([1.25, 0.91, 2.35]),
        np.array([0.96, 0.97, 1.82]),
    ],
)
def test_triatomic_map_joint_reconstruction_and_round_trip(values: np.ndarray) -> None:
    coordinate_map = TriatomicValenceCoordinateMap(_embedded_water(), 0, 1, 2)

    geometry = coordinate_map.to_cartesian(values)

    np.testing.assert_allclose(
        coordinate_map.values_from_cartesian(geometry), values, rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(geometry[3], _embedded_water()[3], rtol=0.0, atol=0.0)
    assert np.linalg.norm(geometry[1] - geometry[0]) == pytest.approx(values[0])
    assert np.linalg.norm(geometry[2] - geometry[0]) == pytest.approx(values[1])


def test_non_water_hof_map_is_generic_and_keeps_frozen_atoms() -> None:
    hof_plus_spectator = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.970, 0.0, 0.0],
            [-0.410, 1.310, 0.0],
            [4.2, -2.1, 0.8],
        ]
    )
    coordinate_map = TriatomicValenceCoordinateMap(
        hof_plus_spectator,
        center_atom=0,
        outer_atom_1=1,
        outer_atom_2=2,
        coordinate_ids=("r_OH", "r_OF", "angle_HOF"),
    )
    values = np.array([1.03, 1.44, 1.74])

    geometry = coordinate_map.to_cartesian(values)

    assert coordinate_map.active_atoms == (0, 1, 2)
    assert coordinate_map.inactive_atoms == (3,)
    np.testing.assert_array_equal(geometry[3], hof_plus_spectator[3])
    np.testing.assert_allclose(coordinate_map.values_from_cartesian(geometry), values)
    assert isinstance(coordinate_map, CoordinateMap)


def test_fixed_body_frame_covaries_with_rigid_rotation_and_translation() -> None:
    reference = _embedded_water()[:3]
    angle = 0.71
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([3.0, -2.0, 0.8])
    rotated_reference = reference @ rotation.T + translation
    original_map = TriatomicValenceCoordinateMap(reference, 0, 1, 2)
    rotated_map = TriatomicValenceCoordinateMap(rotated_reference, 0, 1, 2)
    dipoles_lab = np.array([[0.4, -0.8, 0.2], [-0.3, 0.1, 0.9]])
    rotated_dipoles = dipoles_lab @ rotation.T

    np.testing.assert_allclose(
        original_map.vector_to_body(dipoles_lab),
        rotated_map.vector_to_body(rotated_dipoles),
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        original_map.vector_to_lab(original_map.vector_to_body(dipoles_lab)),
        dipoles_lab,
        rtol=0.0,
        atol=1e-14,
    )
    assert np.linalg.det(original_map.frame_to_lab) == pytest.approx(1.0)


def test_outer_atom_exchange_is_an_explicit_coordinate_permutation() -> None:
    reference = _embedded_water()[:3]
    original = TriatomicValenceCoordinateMap(reference, 0, 1, 2)
    exchanged = TriatomicValenceCoordinateMap(reference, 0, 2, 1)
    values = np.array([1.14, 0.84, 1.93])
    permuted = np.array([values[1], values[0], values[2]])

    original_geometry = original.to_cartesian(values)
    exchanged_geometry = exchanged.to_cartesian(permuted)

    np.testing.assert_allclose(
        original.values_from_cartesian(original_geometry), values, atol=1e-14
    )
    np.testing.assert_allclose(
        exchanged.values_from_cartesian(exchanged_geometry), permuted, atol=1e-14
    )
    assert coordinate_map_fingerprint(original) != coordinate_map_fingerprint(exchanged)
    assert np.linalg.det(exchanged.frame_to_lab) == pytest.approx(1.0)


def test_bond_length_map_preserves_frozen_atoms_and_explicit_frame() -> None:
    reference = _embedded_water()
    frame = TriatomicValenceCoordinateMap(reference, 0, 1, 2).frame_to_lab
    coordinate_map = BondLengthCoordinateMap(
        reference,
        Bond(0, 1),
        coordinate_id="r_OH",
        reference_frame_to_lab=frame,
    )

    geometry = coordinate_map.to_cartesian(np.array([1.31]))

    np.testing.assert_array_equal(geometry[[0, 2, 3]], reference[[0, 2, 3]])
    assert coordinate_map.values_from_cartesian(geometry)[0] == pytest.approx(1.31)
    np.testing.assert_allclose(
        coordinate_map.vector_to_lab(coordinate_map.vector_to_body([1.0, 2.0, 3.0])),
        [1.0, 2.0, 3.0],
    )


def test_linear_displacement_map_round_trips_and_reconstructs_from_payload() -> None:
    reference = _embedded_water()
    displacements = np.zeros((2, *reference.shape))
    displacements[0, 1, 0] = 1.0
    displacements[1, 2, 1] = 0.5
    coordinate_map = LinearDisplacementCoordinateMap(
        reference_geometry_A=reference,
        coordinate_ids=("qx", "qy"),
        units=("angstrom", "dimensionless"),
        reference_values=np.array([0.0, 2.0]),
        displacements_A_per_unit=displacements,
    )
    values = np.array([0.23, 1.7])

    geometry = coordinate_map.to_cartesian(values)
    restored = coordinate_map_from_payload(coordinate_map.fingerprint_payload())

    np.testing.assert_allclose(coordinate_map.values_from_cartesian(geometry), values)
    np.testing.assert_array_equal(geometry[[0, 3]], reference[[0, 3]])
    np.testing.assert_array_equal(restored.to_cartesian(values), geometry)
    assert coordinate_map_fingerprint(restored) == coordinate_map_fingerprint(coordinate_map)


def test_coordinate_map_payload_detects_retained_array_mutation() -> None:
    coordinate_map = TriatomicValenceCoordinateMap(_embedded_water(), 0, 1, 2)
    payload = copy.deepcopy(coordinate_map.fingerprint_payload())
    payload["reference_geometry_A"]["values"][0][0] += 1e-6

    with pytest.raises(ValueError, match="hash does not match"):
        coordinate_map_from_payload(payload)


def test_coordinate_map_arrays_are_deeply_immutable() -> None:
    coordinate_map = TriatomicValenceCoordinateMap(_embedded_water(), 0, 1, 2)

    for array in (
        coordinate_map.reference_geometry_A,
        coordinate_map.reference_values,
        coordinate_map.frame_to_lab,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
        with pytest.raises(ValueError):
            array.flat[0] = 0.0


@pytest.mark.parametrize(
    "reference",
    [
        np.zeros((3, 3)),
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1e-12, 0.0]]),
    ],
)
def test_triatomic_map_rejects_degenerate_reference(reference: np.ndarray) -> None:
    with pytest.raises(ValueError, match="positive length|linear or too nearly linear"):
        TriatomicValenceCoordinateMap(reference, 0, 1, 2)


def test_linear_map_rejects_geometry_outside_its_subspace() -> None:
    reference = np.zeros((2, 3))
    displacement = np.zeros((1, 2, 3))
    displacement[0, 1, 0] = 1.0
    coordinate_map = LinearDisplacementCoordinateMap(
        reference, ("q",), ("angstrom",), np.array([0.0]), displacement
    )
    outside = coordinate_map.to_cartesian(np.array([0.2]))
    outside[1, 1] = 1e-6

    with pytest.raises(ValueError, match="outside the linear coordinate-map subspace"):
        coordinate_map.values_from_cartesian(outside)
