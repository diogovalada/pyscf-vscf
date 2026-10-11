"""Evaluate the electronic points of a plan, in parallel and resumably.

A *plan* is anything with a ``provider_scientific_fingerprint`` and a
``requests`` mapping from point causal fingerprint to
:class:`~pyscf_vscf.contracts.ElectronicPointRequest`: scan plans and n-mode
point plans both qualify. :func:`run_plan` evaluates every point that is not
already in the optional :class:`~pyscf_vscf.point_store.PointStore`, keeps
going or stops at the first failure, and returns a :class:`PlanRun` report.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
import traceback
from collections.abc import Callable, Mapping
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Protocol

from .backends._pyscf_factory import ConfigurationGuard
from .contracts import (
    ElectronicPointRequest,
    ElectronicProvider,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from .point_store import PointStore, StoreConflictError


class ElectronicPlan(Protocol):
    """A set of electronic point requests for one provider."""

    @property
    def provider_scientific_fingerprint(self) -> str:
        """Fingerprint of the provider the plan was made for."""
        ...

    @property
    def requests(self) -> Mapping[str, ElectronicPointRequest]:
        """Requests keyed by point causal fingerprint."""
        ...


@dataclass(frozen=True)
class RunnerSettings:
    """Execution policy for :func:`run_plan`.

    ``workers=None`` evaluates points in the calling process and leaves its
    PySCF thread settings alone. ``workers >= 1`` uses that many spawned worker
    processes; each sets ``threads_per_worker`` PySCF/BLAS threads and the
    PySCF memory budget ``max_memory_mb`` at start-up. ``stop_on_error`` stops
    scheduling new points after the first failure; points whose latest stored
    attempt failed are retried only with ``retry_failed``. ``max_in_flight``
    bounds submitted work (default: twice the worker count). Thread limits
    need the default ``spawn`` start method, because forked workers inherit
    the parent's already-started thread pools.
    """

    workers: int | None = None
    threads_per_worker: int | None = None
    max_memory_mb: int | None = None
    stop_on_error: bool = True
    retry_failed: bool = False
    max_in_flight: int | None = None
    start_method: str = "spawn"

    def __post_init__(self) -> None:
        for name in ("workers", "threads_per_worker", "max_memory_mb", "max_in_flight"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be at least 1")
        if self.start_method not in mp.get_all_start_methods():
            raise ValueError(f"Unsupported multiprocessing start method {self.start_method!r}")
        if self.threads_per_worker is not None and self.start_method != "spawn":
            raise ValueError(
                "threads_per_worker requires start_method='spawn': forked workers inherit "
                "the parent's already-started thread pools"
            )


@dataclass(frozen=True)
class PointFailure:
    """A point whose evaluation raised, or was refused because it failed before."""

    point_key: str
    error_type: str
    message: str
    traceback: str = ""
    from_store: bool = False


@dataclass(frozen=True)
class ProgressEvent:
    """Emitted after each point outcome is recorded."""

    completed: int
    total: int
    point_key: str
    status: str


@dataclass(frozen=True)
class PlanRun:
    """Outcome of :func:`run_plan`.

    ``results`` holds every available result, both newly computed and
    ``reused`` from the store. ``failures`` lists failed points; ``not_attempted``
    those never started because the run stopped early or was interrupted.
    """

    provider_scientific_fingerprint: str
    total: int
    results: Mapping[str, ElectronicResult]
    reused: frozenset[str]
    failures: Mapping[str, PointFailure]
    not_attempted: frozenset[str]
    executor_error: str | None = None
    interrupted: bool = False
    run_id: str | None = None
    wall_seconds: float = 0.0
    conflicts: tuple[str, ...] = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        """True when every planned point has a result."""

        return len(self.results) == self.total

    def summary(self, *, max_failures: int = 3) -> str:
        """Return a one-line description of the run."""

        parts = [
            f"{len(self.results)} of {self.total} point(s) available "
            f"({len(self.reused)} reused), {len(self.failures)} failed, "
            f"{len(self.not_attempted)} not attempted"
        ]
        if self.executor_error:
            parts.append(f"executor failure: {self.executor_error}")
        if self.interrupted:
            parts.append("interrupted")
        shown = list(self.failures.values())[: int(max_failures)]
        parts.extend(
            f"point {failure.point_key[:12]}: {failure.error_type}: {failure.message}"
            for failure in shown
        )
        if len(self.failures) > len(shown):
            parts.append(f"and {len(self.failures) - len(shown)} more failure(s)")
        return "; ".join(parts)


class PlanIncompleteError(RuntimeError):
    """Raised by convenience workflows when a plan could not be completed."""

    def __init__(self, message: str, run: PlanRun) -> None:
        super().__init__(message)
        self.run = run

    def __reduce__(self) -> tuple[Any, ...]:
        return (type(self), (str(self), self.run))


def run_plan(
    plan: ElectronicPlan,
    provider: ElectronicProvider,
    *,
    store: PointStore | None = None,
    settings: RunnerSettings | None = None,
    progress: Callable[[ProgressEvent], None] | None = None,
) -> PlanRun:
    """Evaluate every point of ``plan`` with ``provider``; return a :class:`PlanRun`."""

    policy = settings or RunnerSettings()
    started = time.perf_counter()
    provider_id = provider_scientific_fingerprint(provider)
    if provider_id != plan.provider_scientific_fingerprint:
        raise ValueError("The provider does not match the plan's provider fingerprint")
    requests = dict(sorted(plan.requests.items()))
    for key, request in requests.items():
        if request.causal_fingerprint(provider_id) != key:
            raise ValueError("A plan request does not match its point causal fingerprint")

    results: dict[str, ElectronicResult] = {}
    failures: dict[str, PointFailure] = {}
    reused: set[str] = set()
    pending: list[str] = []
    run_id = None
    if store is not None:
        store.register_provider(provider_id, provider.scientific_settings_payload())
    for key in requests:
        stored = store.get_result(key) if store is not None else None
        if stored is not None:
            validate_result(stored, requests[key], provider_id)
            results[key] = stored
            reused.add(key)
            continue
        latest = store.latest_attempt(key) if store is not None else None
        if latest is not None and latest.status == "failed" and not policy.retry_failed:
            failures[key] = PointFailure(
                point_key=key,
                error_type=latest.error_type or "Error",
                message=f"{latest.message} (failed in an earlier run; retry_failed=True retries it)",
                traceback=latest.traceback or "",
                from_store=True,
            )
            continue
        pending.append(key)

    guard = ConfigurationGuard(getattr(provider, "factory", None))

    def check_configuration(result: ElectronicResult) -> None:
        """Require the same invariant method configuration for every point."""

        snapshot = result.provenance.get("configuration_snapshot")
        guard.check(None if snapshot is None else dict(snapshot))

    for key in sorted(reused):
        check_configuration(results[key])

    total = len(requests)
    counter = {"done": len(reused) + len(failures)}
    conflicts: list[str] = []
    if store is not None:
        run_id = store.begin_run(
            provider_fingerprint=provider_id,
            n_points=total,
            policy={
                "stop_on_error": policy.stop_on_error,
                "retry_failed": policy.retry_failed,
                "workers": policy.workers,
            },
        )

    def finish(key: str, attempt_id: int | None, outcome: dict[str, Any]) -> None:
        if outcome["ok"]:
            result = outcome["result"]
            try:
                validate_result(result, requests[key], provider_id)
                check_configuration(result)
                if store is not None:
                    assert attempt_id is not None
                    store.record_success(attempt_id, requests[key], result)
                results[key] = result
                status = "succeeded"
            except StoreConflictError as exc:
                conflicts.append(key)
                failures[key] = PointFailure(key, "StoreConflictError", str(exc))
                status = "conflict"
            except Exception as exc:
                failures[key] = PointFailure(key, _type_name(exc), str(exc))
                if store is not None:
                    assert attempt_id is not None
                    store.record_failure(attempt_id, error_type=_type_name(exc), message=str(exc))
                status = "failed"
        else:
            failures[key] = PointFailure(
                key, outcome["error_type"], outcome["message"], outcome["traceback"]
            )
            if store is not None:
                assert attempt_id is not None
                store.record_failure(
                    attempt_id,
                    error_type=outcome["error_type"],
                    message=outcome["message"],
                    traceback=outcome["traceback"],
                )
            status = "failed"
        counter["done"] += 1
        if progress is not None:
            progress(ProgressEvent(counter["done"], total, key, status))

    def begin(key: str) -> int | None:
        if store is None:
            return None
        assert run_id is not None
        return store.begin_attempt(run_id, key)

    def abandon(key: str, attempt_id: int | None) -> None:
        """Close an attempt whose evaluation never ran (cancelled before it started)."""

        attempted.discard(key)
        if store is not None and attempt_id is not None:
            store.record_failure(
                attempt_id,
                error_type="Cancelled",
                message="cancelled before evaluation",
                status="interrupted",
            )

    executor_error = None
    interrupted = False
    attempted: set[str] = set()
    try:
        if policy.workers is None:
            for key in pending:
                if failures and policy.stop_on_error:
                    break
                attempted.add(key)
                attempt_id = begin(key)
                finish(key, attempt_id, _evaluate_point(provider, requests[key], provider_id))
        else:
            executor_error = _run_parallel(
                provider,
                requests,
                pending,
                provider_id,
                policy,
                begin,
                finish,
                abandon,
                failures,
                attempted,
            )
    except KeyboardInterrupt:
        interrupted = True
        if store is not None and run_id is not None:
            store.finish_run(run_id, "interrupted")
        raise
    finally:
        if store is not None and run_id is not None and not interrupted:
            complete = len(results) == total
            store.finish_run(run_id, "complete" if complete else "incomplete")

    not_attempted = frozenset(set(pending) - attempted)
    return PlanRun(
        provider_scientific_fingerprint=provider_id,
        total=total,
        results=dict(results),
        reused=frozenset(reused),
        failures=dict(failures),
        not_attempted=not_attempted,
        executor_error=executor_error,
        interrupted=interrupted,
        run_id=run_id,
        wall_seconds=time.perf_counter() - started,
        conflicts=tuple(conflicts),
    )


def _run_parallel(
    provider: ElectronicProvider,
    requests: Mapping[str, ElectronicPointRequest],
    pending: list[str],
    provider_id: str,
    policy: RunnerSettings,
    begin: Callable[[str], int | None],
    finish: Callable[[str, int | None, dict[str, Any]], None],
    abandon: Callable[[str, int | None], None],
    failures: Mapping[str, PointFailure],
    attempted: set[str],
) -> str | None:
    """Evaluate ``pending`` in worker processes; return an executor error, if any.

    Every submitted point ends up finished (result or failure) or abandoned
    (cancelled before it ran, so it counts as not attempted), including when
    the pool breaks or the run is interrupted.
    """

    try:
        pickle.dumps(provider)
    except Exception as exc:
        raise TypeError(
            "The provider cannot be sent to worker processes; method recipes must use "
            f"importable module-level functions ({_type_name(exc)}: {exc})"
        ) from exc
    workers = int(policy.workers or 1)
    limit = int(policy.max_in_flight or 2 * workers)
    queue = list(pending)
    in_flight: dict[Future, tuple[str, int | None]] = {}
    executor_error: str | None = None

    def collect(future: Future) -> None:
        nonlocal executor_error
        key, attempt_id = in_flight.pop(future)
        if future.cancelled():
            abandon(key, attempt_id)
            return
        try:
            outcome = future.result()
        except BaseException as exc:  # the worker's exception, e.g. a broken pool
            if executor_error is None:
                executor_error = f"{_type_name(exc)}: {exc}"
            outcome = {
                "ok": False,
                "error_type": _type_name(exc),
                "message": f"worker process failed: {exc}",
                "traceback": "".join(traceback.format_exception(exc)),
            }
        finish(key, attempt_id, outcome)

    # Spawned workers import NumPy (and its BLAS pool) before their initializer
    # runs, so the thread settings must already be in the environment they inherit.
    environment = _ThreadEnvironment(policy.threads_per_worker)
    try:
        environment.apply()
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=mp.get_context(policy.start_method),
            initializer=_worker_init,
            initargs=(policy.threads_per_worker, policy.max_memory_mb),
        )
    except Exception as exc:
        environment.restore()
        return f"{_type_name(exc)}: {exc}"
    except BaseException:  # e.g. KeyboardInterrupt while the pool starts
        environment.restore()
        raise
    try:
        while queue or in_flight:
            while (
                executor_error is None
                and queue
                and len(in_flight) < limit
                and not (failures and policy.stop_on_error)
            ):
                key = queue.pop(0)
                attempt_id = begin(key)
                attempted.add(key)
                try:
                    future = executor.submit(_evaluate_point, provider, requests[key], provider_id)
                except Exception as exc:
                    executor_error = f"{_type_name(exc)}: {exc}"
                    abandon(key, attempt_id)
                    break
                in_flight[future] = (key, attempt_id)
            if not in_flight:
                break
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                collect(future)
            if executor_error is not None:
                break
    finally:
        try:
            for future in list(in_flight):
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
        finally:
            environment.restore()
        # Account for every point still in flight: finished during shutdown, or cancelled.
        for future in list(in_flight):
            collect(future)
    return executor_error


class _ThreadEnvironment:
    """Temporarily set the thread variables that spawned workers inherit."""

    def __init__(self, threads: int | None) -> None:
        from .settings import thread_env_updates

        self._updates = thread_env_updates(threads)
        self._saved: dict[str, str | None] = {}

    def apply(self) -> None:
        self._saved = {name: os.environ.get(name) for name in self._updates}
        os.environ.update(self._updates)

    def restore(self) -> None:
        for name, value in self._saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        self._saved = {}


def _evaluate_point(
    provider: ElectronicProvider,
    request: ElectronicPointRequest,
    expected_provider_id: str,
) -> dict[str, Any]:
    """Evaluate one request; never raise (worker processes return outcomes)."""

    try:
        actual = provider_scientific_fingerprint(provider)
        if actual != expected_provider_id:
            raise RuntimeError("The provider's identity changed inside the worker process")
        return {"ok": True, "result": provider.evaluate(request)}
    except Exception as exc:
        return {
            "ok": False,
            "error_type": _type_name(exc),
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }


def validate_result(
    result: ElectronicResult,
    request: ElectronicPointRequest,
    provider_id: str,
) -> None:
    """Raise ``ValueError`` unless ``result`` is a converged answer to ``request``."""

    if result.provider_scientific_fingerprint != provider_id:
        raise ValueError("Result was produced by a different provider")
    if result.point_causal_fingerprint != request.causal_fingerprint(provider_id):
        raise ValueError("Result does not belong to the requested point")
    if not result.converged:
        raise ValueError("Result is not converged")
    if "dipole" in request.requested_properties and result.dipole_au is None:
        raise ValueError("Result is missing the requested dipole")


def _worker_init(threads: int | None, max_memory_mb: int | None) -> None:
    from .settings import MAX_MEMORY_ENV_VAR, apply_thread_env_updates

    apply_thread_env_updates(threads, os.environ)
    if max_memory_mb is not None:
        os.environ[MAX_MEMORY_ENV_VAR] = str(int(max_memory_mb))
    if threads is not None:
        try:
            from pyscf import lib
        except Exception:
            return
        lib.num_threads(int(threads))


def _type_name(exc: BaseException) -> str:
    name = f"{type(exc).__module__}.{type(exc).__qualname__}"
    return name.removeprefix("builtins.")


__all__ = [
    "ElectronicPlan",
    "PlanIncompleteError",
    "PlanRun",
    "PointFailure",
    "ProgressEvent",
    "RunnerSettings",
    "run_plan",
]
