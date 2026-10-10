from __future__ import annotations

import multiprocessing as mp
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyscf_vscf.cache import load_grid_npz
from pyscf_vscf.coordinates import Bond
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.workflows import scans


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


def _bond_length_energy(molecule: object, cfg: object) -> tuple[float, np.ndarray]:
    del cfg
    coords = np.asarray(getattr(molecule, "coords"), dtype=float)
    length = float(np.linalg.norm(coords[1] - coords[0]))
    if length > 1.05:
        raise RuntimeError(f"SCF did not converge at {length:.3f} A")
    return length**2, coords.sum(axis=0)


def test_sequential_scan_stops_at_first_failure_and_keeps_completed_points() -> None:
    with pytest.raises(scans.ScanFailedError) as excinfo:
        scans.grid_1d_pes_dms(
            _water(),
            _cfg(),
            Bond(0, 1),
            0.9,
            1.2,
            7,
            energy_dipole_fn=_bond_length_energy,
        )

    partial = excinfo.value.partial
    grid = np.linspace(0.9, 1.2, 7)
    assert partial.axis_labels == ("R_A",)
    np.testing.assert_allclose(partial.axes[0], grid)
    assert partial.completed.tolist() == [True, True, True, True, False, False, False]
    np.testing.assert_allclose(partial.energies_hartree[:4], grid[:4] ** 2)
    assert np.all(np.isnan(partial.energies_hartree[4:]))
    assert np.all(np.isnan(partial.dipoles_debye[4:]))
    assert partial.stopped_early
    assert len(partial.failures) == 1
    failure = partial.failures[0]
    assert failure.index == (4,)
    assert failure.scan_values == pytest.approx((1.1,))
    assert failure.error_type == "RuntimeError"
    assert "SCF did not converge at 1.100 A" in failure.message
    assert "Traceback" in failure.traceback
    message = str(excinfo.value)
    assert "4 of 7 point(s) completed, 1 failed" in message
    assert "point [4] (R_A=1.1)" in message


def test_continue_on_error_attempts_every_point() -> None:
    with pytest.raises(scans.ScanFailedError) as excinfo:
        scans.grid_1d_pes_dms(
            _water(),
            _cfg(),
            Bond(0, 1),
            0.9,
            1.2,
            7,
            energy_dipole_fn=_bond_length_energy,
            stop_on_error=False,
        )

    partial = excinfo.value.partial
    assert not partial.stopped_early
    assert [failure.index for failure in partial.failures] == [(4,), (5,), (6,)]
    assert partial.n_completed == 4
    assert "more failure" not in str(excinfo.value)
    assert "4 of 7 point(s) completed, 3 failed" in str(excinfo.value)


def test_batched_executor_stops_scheduling_after_failing_batch() -> None:
    batches: list[int] = []
    with pytest.raises(scans.ScanFailedError) as excinfo:
        scans.grid_1d_pes_dms(
            _water(),
            _cfg(),
            Bond(0, 1),
            0.9,
            1.2,
            7,
            energy_dipole_fn=_bond_length_energy,
            executor_factory=lambda: _MapOnlyExecutor(batches),
            batch_size=3,
        )

    # Points 0-2 succeed, batch 2 (points 3-5) contains the first failure.
    assert batches == [3, 3]
    partial = excinfo.value.partial
    assert partial.completed.tolist() == [True, True, True, True, False, False, False]
    assert [failure.index for failure in partial.failures] == [(4,), (5,)]
    assert partial.stopped_early


def test_executor_failure_preserves_collected_results() -> None:
    batches: list[int] = []
    with pytest.raises(scans.ScanFailedError) as excinfo:
        scans.grid_1d_pes_dms(
            _water(),
            _cfg(),
            Bond(0, 1),
            0.9,
            1.0,
            6,
            energy_dipole_fn=_bond_length_energy,
            executor_factory=lambda: _MapOnlyExecutor(batches, fail_on_batch=2),
            batch_size=2,
        )

    partial = excinfo.value.partial
    assert partial.completed.tolist() == [True, True, False, False, False, False]
    assert partial.failures == ()
    assert partial.executor_error == "RuntimeError: worker process died"
    assert "executor failure: RuntimeError: worker process died" in str(excinfo.value)


def test_scan_failed_error_round_trips_through_pickle() -> None:
    with pytest.raises(scans.ScanFailedError) as excinfo:
        scans.grid_1d_pes_dms(
            _water(), _cfg(), Bond(0, 1), 1.0, 1.2, 3, energy_dipole_fn=_bond_length_energy
        )

    restored = pickle.loads(pickle.dumps(excinfo.value))
    assert str(restored) == str(excinfo.value)
    assert restored.partial.failures == excinfo.value.partial.failures


def test_2d_failures_report_both_grid_indices() -> None:
    def energy(molecule: object, cfg: object) -> tuple[float, np.ndarray]:
        del cfg
        coords = np.asarray(getattr(molecule, "coords"), dtype=float)
        r2 = float(np.linalg.norm(coords[2] - coords[0]))
        if r2 > 0.99:
            raise ValueError("bad point")
        return r2, coords.sum(axis=0)

    with pytest.raises(scans.ScanFailedError) as excinfo:
        scans.grid_2d_pes_dms(
            _water(),
            _cfg(),
            Bond(0, 1),
            Bond(0, 2),
            [0.9, 1.0],
            [0.9, 0.95, 1.0],
            energy_dipole_fn=energy,
            stop_on_error=False,
        )

    partial = excinfo.value.partial
    assert partial.axis_labels == ("R1_A", "R2_A")
    assert partial.completed.tolist() == [[True, True, False], [True, True, False]]
    assert [failure.index for failure in partial.failures] == [(0, 2), (1, 2)]
    assert partial.failures[1].scan_values == pytest.approx((1.0, 1.0))


def test_relaxed_scan_converts_raised_points_into_scan_failures() -> None:
    def relaxed_point(molecule, cfg, u_dir, s, gtol, maxiter):
        del molecule, cfg, u_dir, gtol, maxiter
        if s > 0.0:
            raise RuntimeError("optimizer exploded")
        return SimpleNamespace(
            energy_hartree=s * s,
            dipole_debye=np.zeros(3),
            achieved_displacement_A=s,
            constraint_residual_A=0.0,
            converged=True,
            n_iterations=1,
            message="ok",
        )

    direction = np.zeros((3, 3))
    direction[1, 0] = 1.0
    with pytest.raises(scans.ScanFailedError, match="optimizer exploded") as excinfo:
        scans.grid_1d_pes_dms_normal_relaxed(
            _water(),
            _cfg(),
            direction,
            -0.1,
            0.1,
            3,
            relaxed_point_fn=relaxed_point,
        )
    assert excinfo.value.partial.completed.tolist() == [True, True, False]


def test_partial_scan_file_is_checksummed_but_never_a_reusable_cache(tmp_path: Path) -> None:
    molecule = _water()
    with pytest.raises(scans.ScanFailedError) as excinfo:
        scans.grid_1d_pes_dms(
            molecule, _cfg(), Bond(0, 1), 0.9, 1.2, 7, energy_dipole_fn=_bond_length_energy
        )
    path = tmp_path / "scan.partial.npz"

    scans.dump_partial_scan_npz(path, excinfo.value.partial, molecule, _cfg())

    meta, arrays = load_grid_npz(path)
    assert meta["schema"] == "pyscf-vscf-partial-scan"
    assert meta["reusable_grid_cache"] is False
    assert meta["n_completed"] == 4
    assert meta["failures"][0]["index"] == [4]
    assert arrays["completed"].tolist() == [True, True, True, True, False, False, False]
    np.testing.assert_allclose(arrays["R_A"], np.linspace(0.9, 1.2, 7))
    with pytest.raises(ValueError, match="Unsupported grid cache schema"):
        scans.load_lbs_frozen_1d_grid_cache(path, molecule, _cfg(), Bond(0, 1), 0.9, 1.2, 7)


@pytest.mark.pyscf
def test_process_pool_scan_matches_serial_scan_for_fresh_pyscf_points() -> None:
    pytest.importorskip("pyscf")
    from pyscf_vscf.settings import ESSettings

    molecule = Molecule.from_arrays(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
        label="water",
    )
    settings = ESSettings(method="hf", basis="sto-3g", use_density_fit=False)

    serial = scans.grid_1d_pes_dms(molecule, settings, "0-1", 0.9, 1.1, 3)
    context = mp.get_context("spawn")
    parallel = scans.grid_1d_pes_dms(
        molecule,
        settings,
        "0-1",
        0.9,
        1.1,
        3,
        executor_factory=lambda: ProcessPoolExecutor(max_workers=2, mp_context=context),
        batch_size=2,
    )

    for left, right in zip(serial, parallel):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-10)


def test_buffered_results_do_not_alias_a_reused_dipole_array() -> None:
    scratch = np.zeros(3)

    def reusing_evaluator(molecule: object, cfg: object) -> tuple[float, np.ndarray]:
        del cfg
        coords = np.asarray(getattr(molecule, "coords"), dtype=float)
        scratch[:] = coords[1]
        return 0.0, scratch

    for executor_factory in (None, lambda: _MapOnlyExecutor([])):
        R, _E, MU = scans.grid_1d_pes_dms(
            _water(),
            _cfg(),
            Bond(0, 1),
            0.9,
            1.1,
            3,
            energy_dipole_fn=reusing_evaluator,
            executor_factory=executor_factory,
        )
        np.testing.assert_allclose(MU[:, 0], R)
