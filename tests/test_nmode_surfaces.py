from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.coordinates import LinearDisplacementCoordinateMap
from pyscf_vscf.electronic import ElectronicPointRequest, ElectronicResult
from pyscf_vscf.nmode import (
    HeldOutCutSamples,
    assemble_nmode_samples,
    dump_nmode_surface,
    fit_nmode_surface,
    load_nmode_surface,
    nmode_dms_fingerprint,
    nmode_pes_fingerprint,
    plan_nmode_points,
)


@dataclass(frozen=True)
class _AnalyticProvider:
    def scientific_settings_payload(self) -> dict[str, object]:
        return {"backend_family": "analytic", "method": "polynomial", "schema": 1}

    def execution_provenance(self) -> dict[str, object]:
        return {"threads": 1}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        del request
        raise AssertionError("fixture results are assembled analytically")


def _coordinate_map() -> LinearDisplacementCoordinateMap:
    reference = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.97, 0.0, 0.0],
            [-0.42, 1.31, 0.0],
        ]
    )
    displacements = np.zeros((3, 3, 3))
    displacements[0, 1, 0] = 1.0
    displacements[1, 2, 1] = 1.0
    displacements[2, 0, 2] = 1.0
    angle = 0.43
    frame = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return LinearDisplacementCoordinateMap(
        reference_geometry_A=reference,
        coordinate_ids=("x_H", "y_F", "z_O"),
        units=("angstrom", "angstrom", "angstrom"),
        reference_values=np.zeros(3),
        displacements_A_per_unit=displacements,
        reference_frame_to_lab=frame,
    )


def _energy(q: np.ndarray) -> float:
    q0, q1, q2 = q
    return float(
        -75.25
        + 0.07 * q0
        + 0.20 * q0**2
        + 0.16 * q1**2
        + 0.04 * q1**3
        + 0.11 * q2**2
        + 0.03 * q0 * q1
        - 0.02 * q0 * q2
        + 0.05 * q1 * q2
        + 0.40 * q0 * q1 * q2
    )


def _dipole_body(q: np.ndarray) -> np.ndarray:
    q0, q1, q2 = q
    return np.array(
        [
            0.1 + 0.8 * q0 + 0.2 * q1 * q2,
            -0.2 - 0.5 * q1 + 0.3 * q0 * q2,
            0.3 + 0.7 * q2 - 0.4 * q0 * q1 + 0.5 * q0 * q1 * q2,
        ]
    )


def _sample_set():
    coordinate_map = _coordinate_map()
    axes = tuple(np.linspace(-0.2, 0.2, 5) for _ in range(3))
    plan = plan_nmode_points(
        coordinate_map,
        axes,
        _AnalyticProvider(),
        nuclear_charges=(8, 1, 9),
        max_rank=2,
        selected_subsets=((0, 1, 2),),
    )
    values_by_point: dict[str, np.ndarray] = {}
    for lineage in plan.sampling_lineage.values():
        point = lineage.point_causal_fingerprint
        if point in values_by_point:
            np.testing.assert_array_equal(values_by_point[point], lineage.coordinate_values)
        values_by_point[point] = np.asarray(lineage.coordinate_values)
    results = {}
    for point, values in values_by_point.items():
        dipole_input = coordinate_map.vector_to_lab(_dipole_body(values))
        results[point] = ElectronicResult(
            total_energy_Eh=_energy(values),
            dipole_au=dipole_input,
            dipole_unit="atomic_unit",
            dipole_frame="input_cartesian",
            converged=True,
            point_causal_fingerprint=point,
            provider_scientific_fingerprint=plan.provider_scientific_fingerprint,
            execution_diagnostics={"path": f"/tmp/{point}"},
            provenance={"software_version": "ignored"},
        )
    return assemble_nmode_samples(plan, results, annotations={"campaign": "not-scientific"})


def _fitted_model():
    samples = _sample_set()
    held_points = np.array([[-0.15, 0.07, 0.11], [0.12, -0.04, 0.18]])
    held = HeldOutCutSamples(
        subset=(0, 1, 2),
        points=held_points,
        energies_Eh=np.array([_energy(q) for q in held_points]),
        dipoles_body_au=np.array([_dipole_body(q) for q in held_points]),
        point_causal_fingerprints=("held-1", "held-2"),
    )
    return fit_nmode_surface(
        samples,
        method="cubic",
        held_out={(0, 1, 2): held},
        annotations={"path": "/first/path", "comment": "first"},
    )


def test_absolute_samples_retain_input_and_fixed_body_frame_dipoles() -> None:
    samples = _sample_set()
    cut = samples.cuts[(0, 2)]

    assert not np.array_equal(cut.dipoles_input_au, cut.dipoles_body_au)
    np.testing.assert_array_equal(
        samples.coordinate_map.vector_to_body(cut.dipoles_input_au),
        cut.dipoles_body_au,
    )
    with pytest.raises(ValueError):
        cut.dipoles_input_au.setflags(write=True)


def test_cubic_fit_reconstructs_polynomial_pes_vector_dms_and_held_out_points() -> None:
    model = _fitted_model()
    probes = np.array(
        [
            [-0.15, 0.07, 0.11],
            [0.12, -0.04, 0.18],
            [0.03, -0.13, -0.08],
        ]
    )

    for q in probes:
        assert model.energy_Eh(q) == pytest.approx(_energy(q), abs=2e-12)
        np.testing.assert_allclose(model.dipole_body_au(q), _dipole_body(q), atol=2e-12)
    diagnostics = model.energy_increments[(0, 1, 2)].diagnostics
    assert diagnostics.n_held_out_points == 2
    assert max(diagnostics.held_out_max_abs_error or ()) < 2e-12


def test_inclusion_exclusion_recovers_pair_and_triple_increments_without_double_counting() -> None:
    model = _fitted_model()
    q = np.array([0.13, -0.09, 0.17])

    assert model.energy_increments[(0, 1)].evaluate(q[[0, 1]]) == pytest.approx(
        0.03 * q[0] * q[1], abs=1e-12
    )
    assert model.energy_increments[(0, 1, 2)].evaluate(q) == pytest.approx(
        0.40 * np.prod(q), abs=1e-12
    )
    assert model.dipole_increments[(0, 1, 2)].evaluate(q)[2] == pytest.approx(
        0.5 * np.prod(q), abs=1e-12
    )
    assert model.potential_Eh(np.zeros(3)) == pytest.approx(0.0, abs=1e-14)


def test_numerical_source_and_artifact_fingerprints_have_separate_domains() -> None:
    model = _fitted_model()
    changed_annotations = replace(
        model,
        annotations={"path": "/different/path", "software_version": "different"},
    )
    changed_source = replace(
        model,
        source_lineage={
            **dict(model.source_lineage),
            "provider_scientific_fingerprint": "different-provider",
        },
    )

    assert nmode_pes_fingerprint(changed_annotations) == nmode_pes_fingerprint(model)
    assert nmode_dms_fingerprint(changed_annotations) == nmode_dms_fingerprint(model)
    assert changed_annotations.source_lineage_fingerprint() == model.source_lineage_fingerprint()
    assert (
        changed_annotations.artifact_integrity_fingerprint()
        != model.artifact_integrity_fingerprint()
    )
    assert nmode_pes_fingerprint(changed_source) == nmode_pes_fingerprint(model)
    assert changed_source.source_lineage_fingerprint() != model.source_lineage_fingerprint()
    with pytest.raises(ValueError, match="unsupported fields"):
        replace(model, source_lineage={**dict(model.source_lineage), "host": "not-lineage"})


def test_surface_serialization_round_trip_and_retained_array_integrity(tmp_path: Path) -> None:
    model = _fitted_model()
    path = tmp_path / "surface.npz"
    dump_nmode_surface(model, path)

    restored = load_nmode_surface(path)

    assert restored.artifact_integrity_fingerprint() == model.artifact_integrity_fingerprint()
    assert nmode_pes_fingerprint(restored) == nmode_pes_fingerprint(model)
    assert nmode_dms_fingerprint(restored) == nmode_dms_fingerprint(model)
    for subset in model.subsets:
        np.testing.assert_array_equal(
            restored.energy_increments[subset].node_values,
            model.energy_increments[subset].node_values,
        )

    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    energy_key = next(name for name in arrays if name.endswith("_energy"))
    arrays[energy_key].flat[0] += 1e-8
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="failed its integrity check"):
        load_nmode_surface(tampered)


def test_interpolation_rejects_out_of_domain_evaluation() -> None:
    model = _fitted_model()

    with pytest.raises(ValueError, match="out of bounds"):
        model.energy_Eh(np.array([0.21, 0.0, 0.0]))
