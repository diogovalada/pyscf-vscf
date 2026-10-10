"""Pure runtime and electronic-structure settings for pyscf-vscf."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Any, MutableMapping, TextIO

DEFAULT_MAX_PARALLEL = 8
DEFAULT_PES_WORKERS = 1
DEFAULT_WORKERS = None
DEFAULT_THREADS_PER_WORKER = None
DEFAULT_VERBOSE = False
DEFAULT_STRICT = True
DEFAULT_ALLOW_FD_HESSIAN = False

IMAG_FREQ_WARN_CM = 1.0
IMAG_FREQ_ERR_CM = 10.0

# Package-owned SCF controls applied when ``ESSettings`` leaves a field unset.
# The tolerance is the package's long-standing value (PySCF's stock default is
# 1e-9); the cycle limit and grid level equal PySCF's stock defaults. They are
# set explicitly so that a user's PySCF configuration file cannot change a
# calculation whose cache identity records ``None``.
DEFAULT_SCF_CONV_TOL = 1e-10
DEFAULT_SCF_MAX_CYCLE = 50
DEFAULT_DFT_GRID_LEVEL = 3

# Automatic scan parallelism and per-process memory budgets.
MAX_MEMORY_ENV_VAR = "PYSCF_VSCF_MAX_MEMORY_MB"
AUTO_MEMORY_FRACTION = 0.75
AUTO_WORKER_MIN_MEMORY_MB = 1024
AUTO_WORKER_MAX_MEMORY_MB = 4000
AUTO_WORKERS_WITHOUT_MEMORY_INFO = 4

BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class ESSettings:
    """Inputs that determine an electronic-structure calculation."""

    method: str = "wb97x"
    basis: str = "aug-cc-pVTZ"
    use_density_fit: bool = True
    auxbasis: str | None = None
    scf_conv_tol: float | None = None
    scf_max_cycle: int | None = None
    dft_grid_level: int | None = None


@dataclass(frozen=True)
class HarmonicSettings:
    """Policy for harmonic analysis, separate from electronic inputs."""

    rtproj: str = "pyscf"
    strict: bool = DEFAULT_STRICT
    allow_fd_hessian: bool = DEFAULT_ALLOW_FD_HESSIAN


def coerce_es_settings(cfg: Any) -> ESSettings:
    """Copy every electronic-structure field from an object or mapping."""

    defaults = ESSettings()
    values = {}
    for item in fields(ESSettings):
        if isinstance(cfg, Mapping):
            values[item.name] = cfg.get(item.name, getattr(defaults, item.name))
        else:
            values[item.name] = getattr(cfg, item.name, getattr(defaults, item.name))
    return ESSettings(**values)


@dataclass(frozen=True)
class ResolvedSCFControls:
    """Effective SCF convergence and DFT-grid controls for one calculation."""

    conv_tol: float
    max_cycle: int
    dft_grid_level: int | None


def resolve_scf_controls(settings: ESSettings) -> ResolvedSCFControls:
    """Resolve unset SCF controls to the package defaults and validate them.

    ``dft_grid_level`` resolves to ``None`` for Hartree-Fock, which has no
    exchange-correlation grid.
    """

    tolerance = (
        DEFAULT_SCF_CONV_TOL if settings.scf_conv_tol is None else float(settings.scf_conv_tol)
    )
    if not math.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("scf_conv_tol must be finite and positive")
    max_cycle = (
        DEFAULT_SCF_MAX_CYCLE if settings.scf_max_cycle is None else int(settings.scf_max_cycle)
    )
    if max_cycle <= 0:
        raise ValueError("scf_max_cycle must be positive")
    grid_level = None
    if str(settings.method).strip().lower() != "hf":
        grid_level = (
            DEFAULT_DFT_GRID_LEVEL
            if settings.dft_grid_level is None
            else int(settings.dft_grid_level)
        )
        if grid_level < 0:
            raise ValueError("dft_grid_level must be non-negative")
    return ResolvedSCFControls(
        conv_tol=tolerance,
        max_cycle=max_cycle,
        dft_grid_level=grid_level,
    )


@dataclass
class RuntimeSettings:
    """Runtime policy after deriving worker and per-worker thread counts.

    ``max_memory_mb`` is the per-process PySCF memory budget applied to every
    calculation object, or ``None`` to leave PySCF's own default in place.
    ``auto_workers`` records whether the worker count was chosen
    automatically.
    """

    workers: int | None = None
    threads_per_worker: int | None = None
    max_parallel: int = DEFAULT_MAX_PARALLEL
    pes_workers: int = DEFAULT_PES_WORKERS
    verbose: bool = DEFAULT_VERBOSE
    strict: bool = DEFAULT_STRICT
    allow_fd_hessian: bool = DEFAULT_ALLOW_FD_HESSIAN
    max_memory_mb: int | None = None
    auto_workers: bool = False


@dataclass
class WarningRegistry:
    """Explicit state for warnings that should be emitted once per key."""

    warned_once: set[str] = field(default_factory=set)

    def should_emit(self, key: str) -> bool:
        """Record key and return True only the first time it is seen."""

        if key in self.warned_once:
            return False
        self.warned_once.add(key)
        return True


_DEFAULT_WARNING_REGISTRY = WarningRegistry()


def default_auxbasis(main_basis: str) -> str:
    """Return the RI auxiliary basis choice for a main basis name."""

    bas = main_basis.lower().replace(" ", "")
    if bas.startswith("aug-cc-pvtz"):
        return "aug-cc-pVTZ-jkfit"
    if bas.startswith("cc-pvtz"):
        return "cc-pVTZ-jkfit"
    if bas.startswith("def2-tzvp"):
        return "def2-universal-jkfit"
    return "weigend+etb"


def derive_parallel_settings(
    task: str,
    *,
    max_parallel: int = DEFAULT_MAX_PARALLEL,
    pes_workers: int | None = DEFAULT_PES_WORKERS,
    verbose: bool = DEFAULT_VERBOSE,
    strict: bool = DEFAULT_STRICT,
    allow_fd_hessian: bool = DEFAULT_ALLOW_FD_HESSIAN,
    n_points: int | None = None,
    available_memory_mb: float | None = None,
    max_memory_mb: int | None = None,
) -> RuntimeSettings:
    """Derive worker and thread counts for the requested task.

    ``harmonic`` and ``opt`` run one process with ``max_parallel`` threads.
    Scans with an explicit ``pes_workers`` split ``max_parallel`` threads
    across that many processes. With ``pes_workers=None`` scans choose the
    worker count automatically: up to one worker per CPU, capped by the number
    of points and by available memory (at least
    ``AUTO_WORKER_MIN_MEMORY_MB`` per worker from ``AUTO_MEMORY_FRACTION`` of
    the available memory). Each automatic worker then receives an explicit
    PySCF memory budget unless ``max_memory_mb`` is given. In both modes the
    ``max_parallel`` threads are split evenly among the workers. When available
    memory is unknown, at most ``AUTO_WORKERS_WITHOUT_MEMORY_INFO`` workers
    are used and PySCF's memory default is left unchanged.
    """

    max_parallel_int = int(max_parallel)
    if max_parallel_int < 1:
        raise ValueError("max_parallel must be >= 1")
    if pes_workers is not None and int(pes_workers) < 1:
        raise ValueError("pes_workers must be >= 1")
    if max_memory_mb is not None and int(max_memory_mb) < 1:
        raise ValueError("max_memory_mb must be >= 1")
    memory_budget = None if max_memory_mb is None else int(max_memory_mb)
    auto = False

    if task in ("harmonic", "opt"):
        workers = 1
        threads_per_worker = max_parallel_int
    elif pes_workers is None:
        auto = True
        cap = max_parallel_int
        if n_points is not None:
            cap = max(1, min(cap, int(n_points)))
        if available_memory_mb is None:
            workers = max(1, min(cap, AUTO_WORKERS_WITHOUT_MEMORY_INFO))
        else:
            usable = max(0.0, float(available_memory_mb)) * AUTO_MEMORY_FRACTION
            workers = max(1, min(cap, int(usable // AUTO_WORKER_MIN_MEMORY_MB)))
            if memory_budget is None:
                memory_budget = max(1, int(min(AUTO_WORKER_MAX_MEMORY_MB, usable / workers)))
        threads_per_worker = max(1, max_parallel_int // workers)
    else:
        workers = min(int(pes_workers), max_parallel_int)
        threads_per_worker = max(1, max_parallel_int // workers)

    return RuntimeSettings(
        workers=workers,
        threads_per_worker=threads_per_worker,
        max_parallel=max_parallel_int,
        pes_workers=workers if pes_workers is None else int(pes_workers),
        verbose=bool(verbose),
        strict=bool(strict),
        allow_fd_hessian=bool(allow_fd_hessian),
        max_memory_mb=memory_budget,
        auto_workers=auto,
    )


def available_cpu_count(*, cgroup_root: Path | str = "/sys/fs/cgroup") -> int:
    """Return the CPUs this process may use, honouring affinity and cgroup quotas."""

    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    for directory in _visible_cgroup_directories(Path(cgroup_root)):
        quota = _read_text(directory / "cpu.max")
        if not quota:
            continue
        fields_ = quota.split()
        if len(fields_) == 2 and fields_[0] != "max":
            try:
                limit = math.ceil(int(fields_[0]) / int(fields_[1]))
            except (ValueError, ZeroDivisionError):
                continue
            count = min(count, max(1, limit))
    return max(1, int(count))


def available_memory_mb(
    *,
    proc_root: Path | str = "/proc",
    cgroup_root: Path | str = "/sys/fs/cgroup",
) -> float | None:
    """Return available memory in MiB, or ``None`` if it cannot be determined.

    The result is the minimum of the host ``MemAvailable`` and, for every
    cgroup-v2 ancestor visible from this process, ``memory.max`` minus
    ``memory.current``. Limits outside the visible cgroup namespace cannot be
    detected.
    """

    candidates: list[float] = []
    meminfo = _read_text(Path(proc_root) / "meminfo")
    for line in (meminfo or "").splitlines():
        if line.startswith("MemAvailable:"):
            try:
                candidates.append(int(line.split()[1]) / 1024.0)
            except (IndexError, ValueError):
                pass
            break
    for directory in _visible_cgroup_directories(Path(cgroup_root), proc_root=Path(proc_root)):
        limit = _read_text(directory / "memory.max")
        current = _read_text(directory / "memory.current")
        if not limit or limit == "max" or not current:
            continue
        try:
            candidates.append(max(0, int(limit) - int(current)) / (1024.0 * 1024.0))
        except ValueError:
            continue
    return min(candidates) if candidates else None


def runtime_max_memory_mb() -> int | None:
    """Return the per-process PySCF memory budget from the environment."""

    value = os.environ.get(MAX_MEMORY_ENV_VAR)
    if value is None or not value.strip():
        return None
    try:
        budget = int(value)
    except ValueError as exc:
        raise ValueError(f"{MAX_MEMORY_ENV_VAR} must be a positive integer (MiB)") from exc
    if budget < 1:
        raise ValueError(f"{MAX_MEMORY_ENV_VAR} must be a positive integer (MiB)")
    return budget


def _visible_cgroup_directories(
    cgroup_root: Path,
    *,
    proc_root: Path = Path("/proc"),
) -> list[Path]:
    text = _read_text(proc_root / "self" / "cgroup")
    relative = None
    for line in (text or "").splitlines():
        if line.startswith("0::"):
            relative = line[3:].strip()
            break
    if relative is None:
        return []
    root = cgroup_root.resolve() if cgroup_root.exists() else cgroup_root
    directory = (root / relative.lstrip("/")).resolve() if root.exists() else root
    directories = []
    while True:
        if directory == root or root in directory.parents:
            directories.append(directory)
        if directory == root or directory == directory.parent:
            break
        directory = directory.parent
    return directories


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def thread_env_updates(threads: int | None) -> dict[str, str]:
    """Return environment updates that cap OpenMP and BLAS thread pools."""

    try:
        parsed = int(threads) if threads is not None else 0
    except Exception as exc:
        raise ValueError(
            f"Failed to parse threads={threads!r}; expected an integer or None"
        ) from exc
    if parsed == 0:
        return {}

    updates = {"OMP_NUM_THREADS": str(parsed)}
    updates.update({name: "1" for name in BLAS_THREAD_ENV_VARS})
    return updates


def apply_thread_env_updates(
    threads: int | None,
    environ: MutableMapping[str, str],
) -> dict[str, str]:
    """Apply :func:`thread_env_updates` to an explicit environment mapping."""

    updates = thread_env_updates(threads)
    environ.update(updates)
    return updates


def format_runtime(seconds: float) -> str:
    """Format elapsed seconds as a human-readable runtime string."""

    hours, rem = divmod(float(seconds), 3600.0)
    minutes, secs = divmod(rem, 60.0)
    return f"{int(hours)}h {int(minutes):02d}m {secs:05.2f}s"


def warn(msg: str, *, stream: TextIO | None = None) -> None:
    """Emit a warning line to stderr or an explicit stream."""

    out = stream if stream is not None else sys.stderr
    out.write(f"[WARN] {msg}\n")
    out.flush()


def warn_once(
    key: str,
    msg: str,
    *,
    registry: WarningRegistry | None = None,
    stream: TextIO | None = None,
) -> bool:
    """Emit a warning once per registry key; return True when emitted."""

    warning_registry = registry if registry is not None else _DEFAULT_WARNING_REGISTRY
    if not warning_registry.should_emit(key):
        return False
    warn(msg, stream=stream)
    return True


def log(msg: str, *, verbose: bool = False, stream: TextIO | None = None) -> None:
    """Emit an informational line when verbose mode is enabled."""

    if not verbose:
        return
    out = stream if stream is not None else sys.stdout
    print(f"[INFO] {msg}", file=out, flush=True)


__all__ = [
    "AUTO_MEMORY_FRACTION",
    "AUTO_WORKER_MAX_MEMORY_MB",
    "AUTO_WORKER_MIN_MEMORY_MB",
    "AUTO_WORKERS_WITHOUT_MEMORY_INFO",
    "BLAS_THREAD_ENV_VARS",
    "DEFAULT_ALLOW_FD_HESSIAN",
    "DEFAULT_DFT_GRID_LEVEL",
    "DEFAULT_MAX_PARALLEL",
    "DEFAULT_PES_WORKERS",
    "DEFAULT_SCF_CONV_TOL",
    "DEFAULT_SCF_MAX_CYCLE",
    "DEFAULT_STRICT",
    "DEFAULT_THREADS_PER_WORKER",
    "DEFAULT_VERBOSE",
    "DEFAULT_WORKERS",
    "ESSettings",
    "HarmonicSettings",
    "ResolvedSCFControls",
    "coerce_es_settings",
    "resolve_scf_controls",
    "IMAG_FREQ_ERR_CM",
    "IMAG_FREQ_WARN_CM",
    "MAX_MEMORY_ENV_VAR",
    "RuntimeSettings",
    "WarningRegistry",
    "apply_thread_env_updates",
    "available_cpu_count",
    "available_memory_mb",
    "default_auxbasis",
    "derive_parallel_settings",
    "format_runtime",
    "log",
    "runtime_max_memory_mb",
    "thread_env_updates",
    "warn",
    "warn_once",
]
