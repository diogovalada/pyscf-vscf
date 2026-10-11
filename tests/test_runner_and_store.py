from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.artifacts.electronic import (
    request_from_json,
    request_to_json,
    result_from_json,
    result_to_json,
)
from pyscf_vscf.artifacts.scans import dump_scan, load_legacy_grid, load_scan
from pyscf_vscf.contracts import (
    ElectronicPointRequest,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.point_store import PointStore, StoreConflictError
from pyscf_vscf.runner import PlanIncompleteError, RunnerSettings, run_plan
from pyscf_vscf.scans import (
    assemble_scan,
    default_bond_grid,
    plan_bond_scan,
    plan_normal_scan,
    run_scan,
)


@dataclass(frozen=True)
class FakeProvider:
    """Analytic provider: energy = sum of squared coordinates; dipole = coordinate sum.

    Points whose first-bond length exceeds ``fail_above_A`` raise.
    """

    offset: float = 0.0
    fail_above_A: float = np.inf

    def scientific_settings_payload(self) -> dict[str, object]:
        return {"schema": "test-fake-provider", "offset": self.offset}

    def execution_provenance(self) -> dict[str, object]:
        return {}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        coordinates = np.asarray(request.coordinates_A, dtype=float)
        if np.linalg.norm(coordinates[1] - coordinates[0]) > self.fail_above_A:
            raise RuntimeError("SCF did not converge")
        provider_id = provider_scientific_fingerprint(self)
        return ElectronicResult(
            total_energy_Eh=float(np.sum(coordinates**2)) + self.offset,
            dipole_au=coordinates.sum(axis=0),
            dipole_unit="atomic_unit",
            dipole_frame="input_cartesian",
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
        )


@dataclass(frozen=True)
class DriftingProvider(FakeProvider):
    """Same identity as FakeProvider(offset=0) but a different energy."""

    def scientific_settings_payload(self) -> dict[str, object]:
        return {"schema": "test-fake-provider", "offset": 0.0}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        result = super().evaluate(request)
        provider_id = provider_scientific_fingerprint(self)
        return ElectronicResult(
            total_energy_Eh=result.total_energy_Eh + 1.0,
            dipole_au=result.dipole_au,
            dipole_unit="atomic_unit",
            dipole_frame="input_cartesian",
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
        )


def _water() -> Molecule:
    return Molecule.from_arrays(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        label="water",
    )


def test_request_and_result_json_round_trip_exactly() -> None:
    provider = FakeProvider(offset=1.0 / 3.0)
    request = ElectronicPointRequest(
        nuclear_charges=(8, 1, 1), coordinates_A=_water().coords + 1e-13
    )
    result = provider.evaluate(request)

    restored_request = request_from_json(request_to_json(request))
    assert restored_request.causal_fingerprint("p") == request.causal_fingerprint("p")
    np.testing.assert_array_equal(restored_request.coordinates_A, request.coordinates_A)
    restored = result_from_json(result_to_json(result))
    assert restored.content_fingerprint() == result.content_fingerprint()
    assert restored.total_energy_Eh == result.total_energy_Eh

    tampered = result_to_json(result)
    tampered["total_energy_Eh"] += 1e-15
    with pytest.raises(ValueError, match="fingerprint"):
        result_from_json(tampered)


def test_scan_plan_is_pure_and_deduplicates_geometries() -> None:
    provider = FakeProvider()
    plan = plan_bond_scan(_water(), provider, ["0-1", "0-2"], [[0.9, 1.0, 1.1]] * 2)

    assert plan.shape == (3, 3)
    assert len(plan.requests) == 9
    assert plan.axes[0].bond == (0, 1)
    assert (
        plan.plan_fingerprint()
        == plan_bond_scan(
            _water(), provider, ["0-1", "0-2"], [[0.9, 1.0, 1.1]] * 2
        ).plan_fingerprint()
    )

    direction = np.zeros((3, 3))
    direction[1, 0] = 2.0
    normal = plan_normal_scan(_water(), provider, direction, [-0.1, 0.0, 0.1])
    first = normal.requests[str(normal.point_keys[0])]
    np.testing.assert_allclose(first.coordinates_A[1], [0.86, 0.0, 0.0])


def test_default_bond_grid_is_equilibrium_relative_and_rejects_short_bonds() -> None:
    lower, upper, count = default_bond_grid(0.96)
    assert lower > 0.0 and upper > 0.96 and count >= 3
    with pytest.raises(ValueError, match="give the grid explicitly"):
        default_bond_grid(0.1)


def test_store_resumes_and_reuses_verified_results(tmp_path: Path) -> None:
    provider = FakeProvider()
    plan = plan_bond_scan(_water(), provider, ["0-1"], [np.linspace(0.8, 1.2, 5)])

    with PointStore(tmp_path / "points.sqlite") as store:
        first = run_plan(plan, provider, store=store)
        second = run_plan(plan, provider, store=store)

    assert first.complete and not first.reused
    assert second.complete and second.reused == frozenset(plan.requests)
    for key in plan.requests:
        assert (
            second.results[key].content_fingerprint() == first.results[key].content_fingerprint()
        )


def test_failed_points_are_recorded_and_retried_only_on_request(tmp_path: Path) -> None:
    failing = FakeProvider(fail_above_A=1.05)
    plan = plan_bond_scan(_water(), failing, ["0-1"], [np.linspace(0.8, 1.2, 5)])

    with PointStore(tmp_path / "points.sqlite") as store:
        run = run_plan(plan, failing, store=store, settings=RunnerSettings(stop_on_error=False))
        assert len(run.results) == 3 and len(run.failures) == 2
        failed_key = next(iter(run.failures))
        assert store.latest_attempt(failed_key).status == "failed"

        again = run_plan(plan, failing, store=store, settings=RunnerSettings(stop_on_error=False))
        assert again.reused == frozenset(run.results)
        assert all(failure.from_store for failure in again.failures.values())
        assert "retry_failed=True" in next(iter(again.failures.values())).message

        retried = run_plan(
            plan,
            failing,
            store=store,
            settings=RunnerSettings(stop_on_error=False, retry_failed=True),
        )
        assert not any(failure.from_store for failure in retried.failures.values())
        assert len(store.attempts(failed_key)) == 2


def test_stop_on_error_leaves_later_points_unattempted() -> None:
    failing = FakeProvider(fail_above_A=0.85)
    plan = plan_bond_scan(_water(), failing, ["0-1"], [np.linspace(0.8, 1.2, 5)])

    run = run_plan(plan, failing)

    assert len(run.failures) == 1
    assert len(run.results) + len(run.failures) + len(run.not_attempted) == 5
    assert not run.complete
    with pytest.raises(PlanIncompleteError, match="1 failed") as excinfo:
        run_scan(plan, failing)
    assert excinfo.value.run.failures


def test_conflicting_results_are_recorded_not_overwritten(tmp_path: Path) -> None:
    provider = FakeProvider()
    plan = plan_bond_scan(_water(), provider, ["0-1"], [np.linspace(0.9, 1.1, 3)])
    with PointStore(tmp_path / "points.sqlite") as store:
        first = run_plan(plan, provider, store=store)
        key = next(iter(plan.requests))
        drifting = DriftingProvider()
        attempt = store.begin_attempt(store.begin_run(provider_fingerprint="x", n_points=1), key)
        with pytest.raises(StoreConflictError):
            store.record_success(
                attempt, plan.requests[key], drifting.evaluate(plan.requests[key])
            )
        assert (
            store.get_result(key).content_fingerprint() == first.results[key].content_fingerprint()
        )
        assert store.attempts(key)[-1].status == "conflict"


def test_provider_must_match_the_plan() -> None:
    plan = plan_bond_scan(_water(), FakeProvider(), ["0-1"], [[0.9, 1.0, 1.1]])
    with pytest.raises(ValueError, match="provider"):
        run_plan(plan, FakeProvider(offset=1.0))


def test_spawned_workers_match_serial_evaluation() -> None:
    provider = FakeProvider()
    plan = plan_bond_scan(_water(), provider, ["0-1", "0-2"], [np.linspace(0.9, 1.1, 3)] * 2)

    serial = run_scan(plan, provider)
    parallel = run_scan(plan, provider, settings=RunnerSettings(workers=2, max_in_flight=3))

    np.testing.assert_array_equal(serial.total_energies_Eh, parallel.total_energies_Eh)
    np.testing.assert_array_equal(serial.dipoles_au, parallel.dipoles_au)


def test_unpicklable_providers_fail_before_any_work() -> None:
    @dataclass(frozen=True)
    class LocalProvider(FakeProvider):
        pass

    provider = LocalProvider()
    plan = plan_bond_scan(_water(), provider, ["0-1"], [[0.9, 1.0, 1.1]])
    with pytest.raises(TypeError, match="module-level"):
        run_plan(plan, provider, settings=RunnerSettings(workers=2))


def test_scan_result_views_and_v4_round_trip(tmp_path: Path) -> None:
    provider = FakeProvider()
    plan = plan_bond_scan(_water(), provider, ["0-1"], [np.linspace(0.9, 1.1, 3)])
    scan = run_scan(plan, provider)

    R, E, MU = scan.as_legacy_tuple()
    assert np.min(E) == 0.0
    np.testing.assert_allclose(MU, scan.dipoles_au * 2.541746)
    path = tmp_path / "scan.npz"
    dump_scan(path, scan)
    loaded = load_scan(path)
    np.testing.assert_array_equal(loaded.total_energies_Eh, scan.total_energies_Eh)
    assert loaded.axes[0].bond == (0, 1)
    assert loaded.sampling == scan.sampling

    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["total_energies_Eh"] = arrays["total_energies_Eh"] + 1e-12
    np.savez_compressed(tmp_path / "tampered.npz", **arrays)
    with pytest.raises(ValueError, match="checksum"):
        load_scan(tmp_path / "tampered.npz")


def test_assembly_requires_every_point() -> None:
    provider = FakeProvider()
    plan = plan_bond_scan(_water(), provider, ["0-1"], [np.linspace(0.9, 1.1, 3)])
    results = run_plan(plan, provider).results
    partial = dict(list(results.items())[:2])
    with pytest.raises(ValueError, match="Missing result"):
        assemble_scan(plan, partial)


def test_legacy_v3_cache_loads_as_relative_scan(tmp_path: Path) -> None:
    from pyscf_vscf.settings import ESSettings
    from pyscf_vscf.workflows import scans as legacy

    molecule = _water()
    cfg = ESSettings(method="hf", basis="sto-3g", use_density_fit=False)
    R = np.linspace(0.8, 1.2, 5)
    path = tmp_path / "legacy.npz"
    from pyscf_vscf._warnings import PyscfVscfDeprecationWarning

    with pytest.warns(PyscfVscfDeprecationWarning, match="dump_scan"):
        legacy.dump_lbs_frozen_1d_grid_cache(
            path, molecule, cfg, "0-1", 0.8, 1.2, 5, R, (R - 1.0) ** 2, np.zeros((5, 3))
        )

    loaded = load_legacy_grid(path)
    assert loaded.scan_kind == "legacy-1d"
    np.testing.assert_array_equal(loaded.axes[0], R)
    with pytest.raises(ValueError, match="load_legacy_grid"):
        load_scan(path)
