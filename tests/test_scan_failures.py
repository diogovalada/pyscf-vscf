"""Failure handling of the task runner behind relaxed normal-coordinate scans."""

from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyscf_vscf.cache import load_grid_npz
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.workflows import scans

GRID = np.linspace(-0.3, 0.3, 7)


class _MapOnlyExecutor:
    """Executor fake that evaluates each map batch eagerly, like a test double."""

    def __init__(self, batches: list[int], *, fail_on_batch: int | None = None) -> None:
        self.batches = batches
        self.fail_on_batch = fail_on_batch

    def __enter__(self) -> "_MapOnlyExecutor":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    def map(self, func, tasks):
        self.batches.append(len(tasks))
        if self.fail_on_batch is not None and len(self.batches) == self.fail_on_batch:
            raise RuntimeError("worker process died")
        return [func(task) for task in tasks]


def _water() -> Molecule:
    return Molecule.from_arrays(
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    )


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(method="hf", basis="sto-3g", use_density_fit=False, auxbasis=None)


def _direction() -> np.ndarray:
    direction = np.zeros((3, 3))
    direction[1, 0] = 1.0
    return direction


def _point(s: float, dipole: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        energy_hartree=s * s,
        dipole_debye=np.array([s, 0.0, 0.0]) if dipole is None else dipole,
        achieved_displacement_A=s,
        constraint_residual_A=0.0,
        converged=True,
        n_iterations=1,
        message="ok",
    )


def _fails_beyond_005(molecule, cfg, u_dir, s, gtol, maxiter):
    del molecule, cfg, u_dir, gtol, maxiter
    if s > 0.05:
        raise RuntimeError(f"optimizer failed at s={s:.1f} A")
    return _point(s)


def _relaxed(**kwargs) -> scans.NormalRelaxedGrid:
    return scans.grid_1d_pes_dms_normal_relaxed(
        _water(),
        _cfg(),
        _direction(),
        float(GRID[0]),
        float(GRID[-1]),
        GRID.size,
        **{"relaxed_point_fn": _fails_beyond_005, **kwargs},
    )


def test_sequential_scan_stops_at_first_failure_and_keeps_completed_points() -> None:
    with pytest.raises(scans.ScanFailedError) as excinfo:
        _relaxed()
    partial = excinfo.value.partial
    assert partial.axis_labels == ("s_A",)
    np.testing.assert_allclose(partial.axes[0], GRID)
    assert partial.completed.tolist() == [True, True, True, True, False, False, False]
    np.testing.assert_allclose(partial.energies_hartree[:4], GRID[:4] ** 2)
    assert np.all(np.isnan(partial.energies_hartree[4:]))
    assert np.all(np.isnan(partial.dipoles_debye[4:]))
    assert partial.stopped_early
    assert len(partial.failures) == 1
    failure = partial.failures[0]
    assert failure.index == (4,)
    assert failure.scan_values == pytest.approx((0.1,))
    assert failure.error_type == "RuntimeError"
    assert "optimizer failed at s=0.1 A" in failure.message
    assert "Traceback" in failure.traceback
    message = str(excinfo.value)
    assert "4 of 7 point(s) completed, 1 failed" in message
    assert "point [4] (s_A=0.1)" in message


def test_continue_on_error_attempts_every_point() -> None:
    with pytest.raises(scans.ScanFailedError) as excinfo:
        _relaxed(stop_on_error=False)
    partial = excinfo.value.partial
    assert not partial.stopped_early
    assert [failure.index for failure in partial.failures] == [(4,), (5,), (6,)]
    assert partial.n_completed == 4
    assert "4 of 7 point(s) completed, 3 failed" in str(excinfo.value)


def test_batched_executor_stops_scheduling_after_failing_batch() -> None:
    batches: list[int] = []
    with pytest.raises(scans.ScanFailedError) as excinfo:
        _relaxed(executor_factory=lambda: _MapOnlyExecutor(batches), batch_size=3)
    # Points 0-2 succeed, batch 2 (points 3-5) contains the first failure.
    assert batches == [3, 3]
    partial = excinfo.value.partial
    assert partial.completed.tolist() == [True, True, True, True, False, False, False]
    assert [failure.index for failure in partial.failures] == [(4,), (5,)]
    assert partial.stopped_early


def test_executor_failure_preserves_collected_results() -> None:
    batches: list[int] = []
    with pytest.raises(scans.ScanFailedError) as excinfo:
        _relaxed(executor_factory=lambda: _MapOnlyExecutor(batches, fail_on_batch=2), batch_size=2)
    partial = excinfo.value.partial
    assert partial.completed.tolist() == [True, True, False, False, False, False, False]
    assert partial.failures == ()
    assert partial.executor_error == "RuntimeError: worker process died"
    assert "executor failure: RuntimeError: worker process died" in str(excinfo.value)


def test_scan_failed_error_round_trips_through_pickle() -> None:
    with pytest.raises(scans.ScanFailedError) as excinfo:
        _relaxed()
    restored = pickle.loads(pickle.dumps(excinfo.value))
    assert str(restored) == str(excinfo.value)
    assert restored.partial.failures == excinfo.value.partial.failures


def test_partial_scan_file_is_checksummed_but_never_a_reusable_cache(tmp_path: Path) -> None:
    from pyscf_vscf.artifacts.scans import load_legacy_grid

    molecule = _water()
    with pytest.raises(scans.ScanFailedError) as excinfo:
        _relaxed()
    path = tmp_path / "scan.partial.npz"
    scans.dump_partial_scan_npz(path, excinfo.value.partial, molecule, _cfg())
    meta, arrays = load_grid_npz(path)
    assert meta["schema"] == "pyscf-vscf-partial-scan"
    assert meta["reusable_grid_cache"] is False
    assert meta["n_completed"] == 4
    assert meta["failures"][0]["index"] == [4]
    assert arrays["completed"].tolist() == [True, True, True, True, False, False, False]
    np.testing.assert_allclose(arrays["s_A"], GRID)
    with pytest.raises(ValueError):
        load_legacy_grid(path)


def test_buffered_results_do_not_alias_a_reused_dipole_array() -> None:
    scratch = np.zeros(3)

    def reusing_point(molecule, cfg, u_dir, s, gtol, maxiter):
        del molecule, cfg, u_dir, gtol, maxiter
        scratch[:] = [s, 0.0, 0.0]
        return _point(s, dipole=scratch)

    for executor_factory in (None, lambda: _MapOnlyExecutor([])):
        result = _relaxed(relaxed_point_fn=reusing_point, executor_factory=executor_factory)
        np.testing.assert_allclose(result.dipoles_debye[:, 0], GRID)
