from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from pyscf_vscf.coordinates import (
    Bond,
    BondLengthCoordinateMap,
    LinearDisplacementCoordinateMap,
    TriatomicValenceCoordinateMap,
    coordinate_map_fingerprint,
)
from pyscf_vscf.electronic import ElectronicPointRequest, ElectronicResult
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.nmode import (
    NModePointPlan,
    SamplingLineage,
    enumerate_mode_subsets,
    plan_nmode_points,
)


@dataclass(frozen=True)
class _NoopProvider:
    method: str = "analytic"

    def scientific_settings_payload(self) -> dict[str, object]:
        return {
            "schema": "test-provider",
            "backend_family": "analytic",
            "method": self.method,
        }

    def execution_provenance(self) -> dict[str, object]:
        return {"threads": 99, "path": "/ignored"}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        del request
        raise AssertionError("point planning must not evaluate electronic structure")


def _water_map() -> TriatomicValenceCoordinateMap:
    return TriatomicValenceCoordinateMap(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.96, 0.0, 0.0],
                [-0.24, 0.93, 0.0],
            ]
        ),
        0,
        1,
        2,
    )


def _three_axes(coordinate_map: TriatomicValenceCoordinateMap) -> tuple[np.ndarray, ...]:
    r1, r2, theta = coordinate_map.reference_values
    return (
        np.array([r1 - 0.1, r1, r1 + 0.1]),
        np.array([r2 - 0.1, r2, r2 + 0.1]),
        np.array([theta - 0.15, theta, theta + 0.15]),
    )


def test_subset_enumeration_is_canonical_closed_and_selected_rank_three() -> None:
    assert enumerate_mode_subsets(1) == ((0,),)
    assert enumerate_mode_subsets(3, max_rank=2) == (
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
    )
    assert enumerate_mode_subsets(3, max_rank=2, selected_subsets=[(0, 1, 2)])[-1] == (
        0,
        1,
        2,
    )
    with pytest.raises(ValueError, match="unique increasing"):
        enumerate_mode_subsets(3, selected_subsets=[(1, 0)])


def test_point_plan_deduplicates_anchored_cuts_and_retains_complete_lineage() -> None:
    coordinate_map = _water_map()
    plan = plan_nmode_points(
        coordinate_map,
        _three_axes(coordinate_map),
        _NoopProvider(),
        nuclear_charges=(8, 1, 1),
        max_rank=2,
    )

    assert len(plan.requests) == 19
    assert plan.subsets == enumerate_mode_subsets(3, max_rank=2)
    assert plan.cut_point_fingerprints[()].shape == ()
    assert plan.cut_point_fingerprints[(0, 2)].shape == (3, 3)
    reference = plan.reference_point_fingerprint
    for subset, grid in plan.cut_point_fingerprints.items():
        reference_index = tuple(1 for _ in subset)
        assert str(grid[reference_index]) == reference
    for (subset, local_index), lineage in plan.sampling_lineage.items():
        assert lineage.subset == subset
        assert lineage.coordinate_values.shape == (3,)
        assert lineage.coordinate_map_fingerprint == plan.coordinate_map_fingerprint
        assert lineage.point_causal_fingerprint == str(
            plan.cut_point_fingerprints[subset][local_index]
        )
        np.testing.assert_array_equal(
            lineage.geometry_A,
            plan.requests[lineage.point_causal_fingerprint].coordinates_A,
        )


def test_distinct_maps_share_geometry_key_but_keep_distinct_sampling_lineage() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    bond_map = BondLengthCoordinateMap(reference, Bond(0, 1), coordinate_id="r")
    displacement = np.zeros((1, 2, 3))
    displacement[0, 1, 0] = 1.0
    linear_map = LinearDisplacementCoordinateMap(
        reference,
        coordinate_ids=("x_H",),
        units=("angstrom",),
        reference_values=np.array([1.0]),
        displacements_A_per_unit=displacement,
    )
    axis = (np.array([1.0, 1.2]),)
    provider = _NoopProvider()

    bond_plan = plan_nmode_points(bond_map, axis, provider, nuclear_charges=(1, 1), max_rank=1)
    linear_plan = plan_nmode_points(linear_map, axis, provider, nuclear_charges=(1, 1), max_rank=1)

    np.testing.assert_array_equal(
        bond_map.to_cartesian(np.array([1.2])),
        linear_map.to_cartesian(np.array([1.2])),
    )
    bond_point = str(bond_plan.cut_point_fingerprints[(0,)][1])
    linear_point = str(linear_plan.cut_point_fingerprints[(0,)][1])
    assert bond_point == linear_point
    assert coordinate_map_fingerprint(bond_map) != coordinate_map_fingerprint(linear_map)
    assert (
        bond_plan.sampling_lineage[((0,), (1,))].content_fingerprint()
        != linear_plan.sampling_lineage[((0,), (1,))].content_fingerprint()
    )


def test_isotope_mass_or_label_cannot_enter_electronic_point_identity() -> None:
    geometry = np.array(_water_map().reference_geometry_A)
    h2o = Molecule.from_arrays(["O", "H", "H"], geometry, label="ordinary")
    d2o = Molecule.from_arrays(["O", "D", "D"], geometry, label="isotope")
    coordinate_map = TriatomicValenceCoordinateMap(h2o.coords, 0, 1, 2)
    axes = _three_axes(coordinate_map)
    ordinary = plan_nmode_points(
        coordinate_map,
        axes,
        _NoopProvider(),
        nuclear_charges=(8, 1, 1),
        max_rank=1,
    )
    isotope_relabelled = plan_nmode_points(
        TriatomicValenceCoordinateMap(d2o.coords, 0, 1, 2),
        axes,
        _NoopProvider(),
        nuclear_charges=(8, 1, 1),
        max_rank=1,
    )

    assert not np.array_equal(h2o.analysis_masses(), d2o.analysis_masses())
    np.testing.assert_array_equal(h2o.coords, d2o.coords)
    assert ordinary.requests.keys() == isotope_relabelled.requests.keys()
    assert not hasattr(next(iter(ordinary.requests.values())), "masses")
    assert not hasattr(next(iter(ordinary.requests.values())), "symbols")


def test_plan_fails_closed_when_request_geometry_disagrees_with_map() -> None:
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    coordinate_map = BondLengthCoordinateMap(reference, Bond(0, 1))
    provider = _NoopProvider()
    provider_id = plan_nmode_points(
        coordinate_map,
        (np.array([1.0]),),
        provider,
        nuclear_charges=(1, 1),
        max_rank=1,
    ).provider_scientific_fingerprint
    inconsistent_geometry = reference.copy()
    inconsistent_geometry[1, 0] += 2e-12
    request = ElectronicPointRequest(
        nuclear_charges=(1, 1),
        coordinates_A=inconsistent_geometry,
        requested_properties=("energy", "dipole"),
    )
    point = request.causal_fingerprint(provider_id)
    lineage = SamplingLineage(
        coordinate_map_fingerprint=coordinate_map_fingerprint(coordinate_map),
        coordinate_values=np.array([1.0]),
        subset=(),
        point_causal_fingerprint=point,
        geometry_A=request.coordinates_A,
    )

    with pytest.raises(
        ValueError, match=r"Sampling lineage geometry does not match map\.to_cartesian"
    ):
        NModePointPlan(
            coordinate_map=coordinate_map,
            coordinate_map_fingerprint=coordinate_map_fingerprint(coordinate_map),
            provider_scientific_fingerprint=provider_id,
            node_axes=(np.array([1.0]),),
            subsets=(),
            requests={point: request},
            cut_point_fingerprints={(): np.array(point)},
            sampling_lineage={((), ()): lineage},
        )


def test_point_plan_and_lineage_arrays_cannot_be_reenabled_for_writing() -> None:
    coordinate_map = _water_map()
    plan = plan_nmode_points(
        coordinate_map,
        _three_axes(coordinate_map),
        _NoopProvider(),
        nuclear_charges=(8, 1, 1),
        max_rank=1,
    )
    lineage = next(iter(plan.sampling_lineage.values()))

    for array in (
        plan.node_axes[0],
        plan.cut_point_fingerprints[(0,)],
        lineage.coordinate_values,
        lineage.geometry_A,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_point_plan_rejects_missing_subset_closure() -> None:
    coordinate_map = _water_map()

    with pytest.raises(ValueError, match=r"requires lower-rank subset \(0,\)"):
        plan_nmode_points(
            coordinate_map,
            _three_axes(coordinate_map),
            _NoopProvider(),
            nuclear_charges=(8, 1, 1),
            subsets=((0, 1),),
        )
