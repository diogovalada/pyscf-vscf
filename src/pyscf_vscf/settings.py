"""Pure runtime and electronic-structure settings for pyscf-vscf."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from typing import Any, MutableMapping, TextIO

DEFAULT_MAX_PARALLEL = 8
DEFAULT_PES_WORKERS = 1
DEFAULT_WORKERS = None
DEFAULT_THREADS_PER_WORKER = None
DEFAULT_VERBOSE = False
DEFAULT_DEV_FAST = False
DEFAULT_STRICT = True
DEFAULT_ALLOW_FD_HESSIAN = False

IMAG_FREQ_WARN_CM = 1.0
IMAG_FREQ_ERR_CM = 10.0

BLAS_THREAD_ENV_VARS = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass
class ESSettings:
    """Electronic-structure knobs consumed by PySCF-backed workflows."""

    method: str = "wb97x"
    basis: str = "aug-cc-pVTZ"
    use_density_fit: bool = True
    auxbasis: str | None = None
    dispersion: str | None = "d4"
    rtproj: str = "pyscf"
    strict: bool = True
    allow_fd_hessian: bool = False
    scf_conv_tol: float | None = None
    scf_max_cycle: int | None = None
    dft_grid_level: int | None = None


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


@dataclass
class RuntimeSettings:
    """Runtime policy after deriving worker and per-worker thread counts."""

    workers: int | None = None
    threads_per_worker: int | None = None
    max_parallel: int = DEFAULT_MAX_PARALLEL
    pes_workers: int = DEFAULT_PES_WORKERS
    verbose: bool = DEFAULT_VERBOSE
    dev_fast: bool = DEFAULT_DEV_FAST
    strict: bool = DEFAULT_STRICT
    allow_fd_hessian: bool = DEFAULT_ALLOW_FD_HESSIAN


@dataclass(frozen=True)
class DevFastPolicy:
    """Legacy development-fast defaults without importing or mutating PySCF objects."""

    method: str = "hf"
    basis: str = "sto-3g"
    dispersion: str | None = None
    fast_npts: int = 21
    fast_width: float = 0.20
    min_fast_npts: int = 5
    min_fast_width: float = 0.05
    scf_conv_tol: float = 1e-7
    dft_grid_level: int = 1
    max_cycle: int = 50

    def npts_cap(self) -> int:
        """Return the normalized PES grid cap used by legacy --dev-fast."""

        return int(max(self.min_fast_npts, self.fast_npts))

    def tight_width_cap(self) -> float:
        """Return the normalized tight-scan width cap used by legacy --dev-fast."""

        return float(max(self.min_fast_width, self.fast_width))


@dataclass(frozen=True)
class DevFastResult:
    """Settings and scan caps after applying :class:`DevFastPolicy`."""

    es: ESSettings
    npts: int | None = None
    tight_width: float | None = None
    npts_cap: int = 21
    tight_width_cap: float = 0.20
    max_cycle: int = 50


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


def es_default(field_name: str):
    """Return a default value from :class:`ESSettings` by field name."""

    return getattr(ESSettings, field_name)


_es_default = es_default


def normalize_dispersion(value: str | None) -> str | None:
    """Map legacy dispersion CLI values onto ESSettings values."""

    if value is None:
        return None
    text = str(value)
    if text.lower() == "none":
        return None
    return text


def default_auxbasis(main_basis: str) -> str:
    """Return the legacy RI auxiliary basis choice for a main basis name."""

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
    pes_workers: int = DEFAULT_PES_WORKERS,
    verbose: bool = DEFAULT_VERBOSE,
    dev_fast: bool = DEFAULT_DEV_FAST,
    strict: bool = DEFAULT_STRICT,
    allow_fd_hessian: bool = DEFAULT_ALLOW_FD_HESSIAN,
) -> RuntimeSettings:
    """Derive worker and thread counts using the legacy CLI policy."""

    max_parallel_int = int(max_parallel)
    pes_workers_int = int(pes_workers)
    if max_parallel_int < 1:
        raise ValueError("max_parallel must be >= 1")
    if pes_workers_int < 1:
        raise ValueError("pes_workers must be >= 1")

    if task in ("harmonic", "opt"):
        workers = 1
        threads_per_worker = max_parallel_int
    else:
        workers = min(pes_workers_int, max_parallel_int)
        threads_per_worker = max(1, max_parallel_int // workers)

    return RuntimeSettings(
        workers=workers,
        threads_per_worker=threads_per_worker,
        max_parallel=max_parallel_int,
        pes_workers=pes_workers_int,
        verbose=bool(verbose),
        dev_fast=bool(dev_fast),
        strict=bool(strict),
        allow_fd_hessian=bool(allow_fd_hessian),
    )


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


def development_fast_settings(
    cfg: ESSettings,
    *,
    policy: DevFastPolicy | None = None,
) -> ESSettings:
    """Return ES settings with legacy --dev-fast substitutions applied."""

    policy = policy or DevFastPolicy()
    defaults = ESSettings()

    return replace(
        cfg,
        method=policy.method if cfg.method == defaults.method else cfg.method,
        basis=policy.basis if cfg.basis == defaults.basis else cfg.basis,
        dispersion=policy.dispersion if cfg.dispersion == defaults.dispersion else cfg.dispersion,
        scf_conv_tol=policy.scf_conv_tol if cfg.scf_conv_tol is None else cfg.scf_conv_tol,
        scf_max_cycle=policy.max_cycle if cfg.scf_max_cycle is None else cfg.scf_max_cycle,
        dft_grid_level=policy.dft_grid_level if cfg.dft_grid_level is None else cfg.dft_grid_level,
    )


def apply_development_fast_policy(
    cfg: ESSettings,
    *,
    npts: int | None = None,
    tight_width: float | None = None,
    fast_npts: int = 21,
    fast_width: float = 0.20,
) -> DevFastResult:
    """Apply the legacy --dev-fast ES substitutions and scan caps."""

    policy = DevFastPolicy(fast_npts=fast_npts, fast_width=fast_width)
    npts_cap = policy.npts_cap()
    width_cap = policy.tight_width_cap()

    capped_npts = None if npts is None else min(int(npts), npts_cap)
    capped_width = None if tight_width is None else min(float(tight_width), width_cap)

    return DevFastResult(
        es=development_fast_settings(cfg, policy=policy),
        npts=capped_npts,
        tight_width=capped_width,
        npts_cap=npts_cap,
        tight_width_cap=width_cap,
        max_cycle=policy.max_cycle,
    )


apply_dev_fast_policy = apply_development_fast_policy


def format_runtime(seconds: float) -> str:
    """Format elapsed seconds as the legacy human-readable runtime string."""

    hours, rem = divmod(float(seconds), 3600.0)
    minutes, secs = divmod(rem, 60.0)
    return f"{int(hours)}h {int(minutes):02d}m {secs:05.2f}s"


_format_runtime = format_runtime


def warn(msg: str, *, stream: TextIO | None = None) -> None:
    """Emit a warning line using the legacy prefix."""

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
    "ALLOW_FD_HESSIAN",
    "BLAS_THREAD_ENV_VARS",
    "DEFAULT_ALLOW_FD_HESSIAN",
    "DEFAULT_DEV_FAST",
    "DEFAULT_MAX_PARALLEL",
    "DEFAULT_PES_WORKERS",
    "DEFAULT_STRICT",
    "DEFAULT_THREADS_PER_WORKER",
    "DEFAULT_VERBOSE",
    "DEFAULT_WORKERS",
    "DEV_FAST",
    "DevFastPolicy",
    "DevFastResult",
    "ESSettings",
    "coerce_es_settings",
    "IMAG_FREQ_ERR_CM",
    "IMAG_FREQ_WARN_CM",
    "MAX_PARALLEL",
    "PES_WORKERS",
    "RuntimeSettings",
    "STRICT",
    "THREADS_PER_WORKER",
    "VERBOSE",
    "WORKERS",
    "WarningRegistry",
    "apply_dev_fast_policy",
    "apply_development_fast_policy",
    "apply_thread_env_updates",
    "default_auxbasis",
    "derive_parallel_settings",
    "development_fast_settings",
    "es_default",
    "format_runtime",
    "log",
    "normalize_dispersion",
    "thread_env_updates",
    "warn",
    "warn_once",
]

# Legacy global names retained as constants for migration consumers.
WORKERS = DEFAULT_WORKERS
THREADS_PER_WORKER = DEFAULT_THREADS_PER_WORKER
MAX_PARALLEL = DEFAULT_MAX_PARALLEL
PES_WORKERS = DEFAULT_PES_WORKERS
VERBOSE = DEFAULT_VERBOSE
STRICT = DEFAULT_STRICT
ALLOW_FD_HESSIAN = DEFAULT_ALLOW_FD_HESSIAN
DEV_FAST = DEFAULT_DEV_FAST
