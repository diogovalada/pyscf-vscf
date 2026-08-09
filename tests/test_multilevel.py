from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from pyscf_vscf.coordinates import (
    LinearDisplacementCoordinateMap,
    coordinate_map_fingerprint,
)
from pyscf_vscf.multilevel import compose_multilevel_surface
from pyscf_vscf.nmode import (
    NModeSurfaceModel,
    _fit_tensor_surface,
    nmode_dms_fingerprint,
    nmode_pes_fingerprint,
)


def _coordinate_map(
    *,
    reference_values: np.ndarray | None = None,
    frame_to_lab: np.ndarray | None = None,
    coordinate_ids: tuple[str, str] = ("q0", "q1"),
    units: tuple[str, str] = ("angstrom", "angstrom"),
) -> LinearDisplacementCoordinateMap:
    return LinearDisplacementCoordinateMap(
        reference_geometry_A=np.array([[0.0, 0.0, 0.0]]),
        coordinate_ids=coordinate_ids,
        units=units,
        reference_values=(
            np.zeros(2) if reference_values is None else np.asarray(reference_values)
        ),
        displacements_A_per_unit=np.array(
            [
                [[1.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0]],
            ]
        ),
        reference_frame_to_lab=frame_to_lab,
    )


def _surface(
    axes: tuple[np.ndarray, ...],
    values: np.ndarray,
    *,
    method: str = "linear",
):
    return _fit_tensor_surface(
        axes,
        values,
        method=method,
        held_out_points=None,
        held_out_values=None,
        held_out_point_ids=(),
    )


def _model(
    level: str,
    *,
    axes: tuple[np.ndarray, np.ndarray] | None = None,
    coordinate_map: LinearDisplacementCoordinateMap | None = None,
    include_pair: bool = True,
    annotations: dict[str, object] | None = None,
    energy_anchor_residual: float = 0.0,
    dipole_anchor_residual: float = 0.0,
    anchor_tolerance_Eh: float = 1e-12,
    anchor_tolerance_dipole_au: float = 1e-12,
    interpolation_method: str = "linear",
    curved_dipole: bool = False,
) -> NModeSurfaceModel:
    if level not in {"low", "high"}:
        raise ValueError("unknown fixture level")
    mapping = _coordinate_map() if coordinate_map is None else coordinate_map
    default_axis = np.linspace(-1.0, 1.0, 5)
    mode_axes = (default_axis, default_axis) if axes is None else axes
    q0, q1 = mode_axes
    d0 = q0 - mapping.reference_values[0]
    d1 = q1 - mapping.reference_values[1]
    high = level == "high"

    energy0 = (1.0 + 0.25 * high) * d0**2
    if energy_anchor_residual:
        energy0 = np.array(energy0, copy=True)
        energy0[np.flatnonzero(q0 == mapping.reference_values[0])[0]] = energy_anchor_residual
    energy_increments = {
        (0,): _surface((q0,), energy0, method=interpolation_method),
        (1,): _surface(
            (q1,),
            (2.0 - 0.3 * high) * d1**2,
            method=interpolation_method,
        ),
    }
    dipole0 = np.column_stack(
        (
            (1.0 + 0.1 * high) * d0 + (0.8 * d0**2 if curved_dipole else 0.0),
            (2.0 - 0.2 * high) * d0,
            (-1.0 + 0.3 * high) * d0,
        )
    )
    if dipole_anchor_residual:
        dipole0 = np.array(dipole0, copy=True)
        dipole0[np.flatnonzero(q0 == mapping.reference_values[0])[0], 0] = dipole_anchor_residual
    dipole_increments = {
        (0,): _surface((q0,), dipole0, method=interpolation_method),
        (1,): _surface(
            (q1,),
            np.column_stack(
                (
                    (0.5 - 0.1 * high) * d1,
                    (-1.0 + 0.4 * high) * d1,
                    (2.0 + 0.2 * high) * d1,
                )
            ),
            method=interpolation_method,
        ),
    }
    if include_pair:
        mesh0, mesh1 = np.meshgrid(q0, q1, indexing="ij")
        product = (mesh0 - mapping.reference_values[0]) * (mesh1 - mapping.reference_values[1])
        energy_increments[(0, 1)] = _surface(
            (q0, q1),
            (0.3 + 0.1 * high) * product,
            method=interpolation_method,
        )
        dipole_increments[(0, 1)] = _surface(
            (q0, q1),
            product[..., None]
            * np.asarray(
                [
                    1.0 + 0.2 * high,
                    -2.0 + 0.1 * high,
                    0.5 - 0.3 * high,
                ]
            ),
            method=interpolation_method,
        )
    return NModeSurfaceModel(
        coordinate_ids=mapping.coordinate_ids,
        coordinate_units=mapping.units,
        coordinate_map_payload=mapping.fingerprint_payload(),
        coordinate_map_fingerprint=coordinate_map_fingerprint(mapping),
        reference_values=mapping.reference_values,
        reference_energy_Eh=-10.0 if level == "low" else -20.0,
        reference_dipole_body_au=(
            np.array([0.2, -0.1, 0.3]) if level == "low" else np.array([0.8, 0.6, -0.4])
        ),
        energy_increments=energy_increments,
        dipole_increments=dipole_increments,
        source_lineage={
            "schema": "pyscf-vscf-electronic-source-lineage",
            "schema_version": 1,
            "provider_scientific_fingerprint": f"{level}-provider",
            "point_causal_fingerprints": [f"{level}-point-0", f"{level}-point-1"],
        },
        annotations={} if annotations is None else annotations,
        anchor_tolerance_Eh=anchor_tolerance_Eh,
        anchor_tolerance_dipole_au=anchor_tolerance_dipole_au,
    )


def test_exact_increment_replacement_preserves_one_absolute_reference() -> None:
    low = _model("low")
    high = _model("high")
    corrected = ((0,), (1,), (0, 1))
    model, diagnostics = compose_multilevel_surface(
        low,
        high,
        energy_corrected_subsets=corrected,
        dipole_corrected_subsets=corrected,
    )

    for q in (
        np.array([-0.5, 0.5]),
        np.array([0.0, -1.0]),
        np.array([1.0, 1.0]),
    ):
        expected_energy = low.reference_energy_Eh + (high.energy_Eh(q) - high.reference_energy_Eh)
        expected_dipole = low.reference_dipole_body_au + (
            high.dipole_body_au(q) - high.reference_dipole_body_au
        )
        assert model.energy_Eh(q) == pytest.approx(expected_energy, abs=1e-14)
        np.testing.assert_allclose(model.dipole_body_au(q), expected_dipole, atol=1e-14)

    assert model.reference_energy_Eh == low.reference_energy_Eh
    np.testing.assert_array_equal(
        model.reference_dipole_body_au,
        low.reference_dipole_body_au,
    )
    assert model.energy_Eh(model.reference_values) == model.reference_energy_Eh
    np.testing.assert_array_equal(
        model.dipole_body_au(model.reference_values),
        model.reference_dipole_body_au,
    )

    assert diagnostics.low_provider_scientific_fingerprint == "low-provider"
    assert diagnostics.high_provider_scientific_fingerprint == "high-provider"
    assert diagnostics.low_source_lineage_fingerprint == low.source_lineage_fingerprint()
    assert diagnostics.high_source_lineage_fingerprint == high.source_lineage_fingerprint()
    pair = diagnostics.records[(0, 1)]
    assert pair.energy.status == "delta"
    assert pair.dipole.status == "delta"
    assert len(pair.energy.correction_max_abs) == 1
    assert len(pair.dipole.correction_max_abs) == 3
    assert pair.energy.delta_fit is not None
    assert pair.dipole.delta_fit is not None


def test_pair_only_correction_does_not_double_count_singletons() -> None:
    low = _model("low")
    high = _model("high")
    model, diagnostics = compose_multilevel_surface(
        low,
        high,
        energy_corrected_subsets=((0, 1),),
    )
    q = np.array([0.5, -0.5])
    expected = (
        low.reference_energy_Eh
        + float(low.energy_increments[(0,)].evaluate(q[[0]]))
        + float(low.energy_increments[(1,)].evaluate(q[[1]]))
        + float(high.energy_increments[(0, 1)].evaluate(q))
    )

    assert model.energy_Eh(q) == pytest.approx(expected, abs=2e-15)
    np.testing.assert_array_equal(model.dipole_body_au(q), low.dipole_body_au(q))
    assert diagnostics.records[(0,)].energy.status == "low_level"
    assert diagnostics.records[(1,)].energy.status == "low_level"
    assert diagnostics.records[(0, 1)].energy.status == "delta"
    assert diagnostics.records[(0, 1)].dipole.status == "low_level"


def test_energy_and_each_dipole_component_have_separate_diagnostics() -> None:
    model, diagnostics = compose_multilevel_surface(
        _model("low"),
        _model("high"),
        energy_corrected_subsets=((1,),),
        dipole_corrected_subsets=((0,), (0, 1)),
    )

    assert diagnostics.records[(1,)].energy.status == "delta"
    assert diagnostics.records[(1,)].dipole.status == "low_level"
    assert diagnostics.records[(0,)].energy.status == "low_level"
    assert diagnostics.records[(0,)].dipole.status == "delta"
    assert len(diagnostics.records[(0,)].dipole.correction_max_abs) == 3
    assert diagnostics.composed_pes_fingerprint == nmode_pes_fingerprint(model)
    assert diagnostics.composed_dms_fingerprint == nmode_dms_fingerprint(model)
    with pytest.raises(TypeError):
        diagnostics.records[(0,)] = diagnostics.records[(1,)]


def test_annotations_do_not_change_composed_scientific_identity() -> None:
    low = _model("low", annotations={"path": "/first", "comment": "one"})
    high = _model("high", annotations={"campaign": "first"})
    first, first_diagnostics = compose_multilevel_surface(
        low,
        high,
        energy_corrected_subsets=((0,), (0, 1)),
        dipole_corrected_subsets=((1,),),
        annotations={"output": "first"},
    )
    second, second_diagnostics = compose_multilevel_surface(
        replace(low, annotations={"path": "/other", "comment": "two"}),
        replace(high, annotations={"campaign": "other"}),
        energy_corrected_subsets=((0,), (0, 1)),
        dipole_corrected_subsets=((1,),),
        annotations={"output": "second"},
    )

    assert nmode_pes_fingerprint(first) == nmode_pes_fingerprint(second)
    assert nmode_dms_fingerprint(first) == nmode_dms_fingerprint(second)
    assert (
        first_diagnostics.scientific_fingerprint() == second_diagnostics.scientific_fingerprint()
    )
    assert first.artifact_integrity_fingerprint() != second.artifact_integrity_fingerprint()


def test_high_grid_is_fitted_to_low_grid_on_a_common_closed_domain() -> None:
    low_axes = (np.linspace(-1.0, 1.0, 5), np.linspace(-1.0, 1.0, 5))
    high_axes = (np.linspace(-1.0, 1.0, 9), np.linspace(-1.0, 1.0, 9))
    low = _model("low", axes=low_axes)
    high = _model("high", axes=high_axes)
    model, _ = compose_multilevel_surface(
        low,
        high,
        energy_corrected_subsets=((0,), (0, 1)),
    )

    for subset in ((0,), (0, 1)):
        points = np.stack(
            [
                component.reshape(-1)
                for component in np.meshgrid(
                    *low.energy_increments[subset].axes,
                    indexing="ij",
                )
            ],
            axis=-1,
        )
        np.testing.assert_allclose(
            model.energy_increments[subset].evaluate(points),
            high.energy_increments[subset].evaluate(points),
            atol=2e-15,
        )


def test_composition_rejects_incompatible_models_and_selection() -> None:
    low = _model("low")
    high = _model("high")
    with pytest.raises(ValueError, match="at least one selected"):
        compose_multilevel_surface(low, high)
    with pytest.raises(ValueError, match="unique increasing"):
        compose_multilevel_surface(low, high, energy_corrected_subsets=((1, 0),))
    with pytest.raises(ValueError, match="absent from the high"):
        compose_multilevel_surface(
            low,
            _model("high", include_pair=False),
            energy_corrected_subsets=((0, 1),),
        )

    narrow = (np.linspace(-0.8, 0.8, 5), np.linspace(-1.0, 1.0, 5))
    with pytest.raises(ValueError, match="identical bounds"):
        compose_multilevel_surface(
            low,
            _model("high", axes=narrow),
            energy_corrected_subsets=((0,),),
        )

    rotated = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    with pytest.raises(ValueError, match="coordinate map/frame"):
        compose_multilevel_surface(
            low,
            _model("high", coordinate_map=_coordinate_map(frame_to_lab=rotated)),
            energy_corrected_subsets=((0,),),
        )

    with pytest.raises(ValueError, match="coordinate IDs"):
        compose_multilevel_surface(
            low,
            _model(
                "high",
                coordinate_map=_coordinate_map(coordinate_ids=("q1", "q0")),
            ),
            energy_corrected_subsets=((0,),),
        )
    with pytest.raises(ValueError, match="coordinate units"):
        compose_multilevel_surface(
            low,
            _model(
                "high",
                coordinate_map=_coordinate_map(units=("bohr", "angstrom")),
            ),
            energy_corrected_subsets=((0,),),
        )
    with pytest.raises(ValueError, match="exact reference coordinate"):
        compose_multilevel_surface(
            low,
            _model(
                "high",
                coordinate_map=_coordinate_map(reference_values=np.array([0.5, 0.0])),
            ),
            energy_corrected_subsets=((0,),),
        )


def test_composition_rejects_tolerated_but_nonzero_anchor_residual() -> None:
    high = _model(
        "high",
        energy_anchor_residual=1e-10,
        anchor_tolerance_Eh=1e-9,
    )
    assert high.energy_increments[(0,)].node_values[2] != 0.0

    with pytest.raises(ValueError, match="not exactly anchored"):
        compose_multilevel_surface(
            _model("low"),
            high,
            energy_corrected_subsets=((0,),),
        )


def test_high_anchor_validation_is_observable_specific() -> None:
    high_with_energy_residual = _model(
        "high",
        energy_anchor_residual=1e-10,
        anchor_tolerance_Eh=1e-9,
    )
    compose_multilevel_surface(
        _model("low"),
        high_with_energy_residual,
        dipole_corrected_subsets=((0,),),
    )
    with pytest.raises(ValueError, match="energy increment .* is not exactly anchored"):
        compose_multilevel_surface(
            _model("low"),
            high_with_energy_residual,
            energy_corrected_subsets=((0,),),
        )

    high_with_dipole_residual = _model(
        "high",
        dipole_anchor_residual=1e-10,
        anchor_tolerance_dipole_au=1e-9,
    )
    compose_multilevel_surface(
        _model("low"),
        high_with_dipole_residual,
        energy_corrected_subsets=((0,),),
    )
    with pytest.raises(ValueError, match="dipole increment .* is not exactly anchored"):
        compose_multilevel_surface(
            _model("low"),
            high_with_dipole_residual,
            dipole_corrected_subsets=((0,),),
        )


def test_cubic_composition_preserves_exact_evaluated_reference() -> None:
    low = replace(
        _model("low", interpolation_method="cubic", curved_dipole=True),
        reference_energy_Eh=0.0,
        reference_dipole_body_au=np.zeros(3),
    )
    high = replace(
        _model("high", interpolation_method="cubic", curved_dipole=True),
        reference_energy_Eh=0.0,
        reference_dipole_body_au=np.zeros(3),
    )
    corrected = ((0,), (1,), (0, 1))
    model, _ = compose_multilevel_surface(
        low,
        high,
        energy_corrected_subsets=corrected,
        dipole_corrected_subsets=corrected,
    )

    raw_energy = model.reference_energy_Eh + sum(
        float(surface.evaluate(model.reference_values[list(subset)]))
        for subset, surface in model.energy_increments.items()
    )
    raw_dipole = np.array(model.reference_dipole_body_au, copy=True)
    for subset, surface in model.dipole_increments.items():
        raw_dipole += surface.evaluate(model.reference_values[list(subset)])
    assert raw_energy != model.reference_energy_Eh
    assert not np.array_equal(raw_dipole, model.reference_dipole_body_au)
    assert model.energy_Eh(model.reference_values) == model.reference_energy_Eh
    np.testing.assert_array_equal(
        model.dipole_body_au(model.reference_values),
        model.reference_dipole_body_au,
    )

    for q, active_subset in (
        (np.array([0.0, 0.5]), (1,)),
        (np.array([0.5, 0.0]), (0,)),
    ):
        raw_pair_energy = float(model.energy_increments[(0, 1)].evaluate(q))
        raw_pair_dipole = model.dipole_increments[(0, 1)].evaluate(q)
        assert raw_pair_energy != 0.0
        assert np.any(raw_pair_dipole != 0.0)
        expected_energy = model.reference_energy_Eh + float(
            model.energy_increments[active_subset].evaluate(q[list(active_subset)])
        )
        expected_dipole = model.reference_dipole_body_au + model.dipole_increments[
            active_subset
        ].evaluate(q[list(active_subset)])
        assert model.energy_Eh(q) == expected_energy
        np.testing.assert_array_equal(model.dipole_body_au(q), expected_dipole)
