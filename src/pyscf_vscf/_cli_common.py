"""Output, error, and worker helpers shared by the command-line interfaces."""

from __future__ import annotations

import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Sequence

CLI_RESULT_SCHEMA = "pyscf-vscf-cli-result"
CLI_RESULT_SCHEMA_VERSION = 1
DEFAULT_BOUNDARY_WARN = 1e-5


class _CLIError(RuntimeError):
    """A user-facing error whose message needs no traceback."""


def _package_version() -> str:
    try:
        return version("pyscf-vscf")
    except PackageNotFoundError:
        from . import __version__

        return __version__


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    from .io import _json_default

    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _flag(record: Any, threshold: float) -> str:
    weights = (record.grid_boundary_weight, record.ground_grid_boundary_weight)
    return " *" if any(w is not None and w > threshold for w in weights) else ""


def _print_records(
    records: Sequence[Any], intensity: str, threshold: float, *, two_d: bool
) -> None:
    label = "n assignment weight " if two_d else "v "
    if intensity == "both":
        print(
            f"{label} nu/cm^-1   mu_axis/D   I_axis/(km/mol)   |mu|/D   I_iso/(km/mol)   "
            "int_axis/(m^2/s)   int_iso/(m^2/s)"
        )
    elif intensity == "axis":
        print(
            f"{label} nu/cm^-1   mu_axis/D   I_axis/(km/mol)   int_axis/(m^2/s)   "
            "orientation: polarized along bond axis"
        )
    else:
        print(
            f"{label} nu/cm^-1   |mu|/D   I_iso/(km/mol)   int_iso/(m^2/s)   "
            "orientation: isotropic"
        )
    for record in records:
        if two_d:
            prefix = (
                f"{record.state_index:>1d} {str(record.quanta):>10s} "
                f"{_format_optional(record.assignment_weight, '6.3f')}"
            )
        else:
            prefix = f"{record.quanta[0]:>1d}"
        if intensity == "both":
            columns = (
                f"{record.transition_dipole_axis_D:>12.4e}   "
                f"{record.integrated_cross_section_axis_km_per_mol:>12.4e}   "
                f"{record.transition_dipole_norm_D:>12.4e}   "
                f"{record.integrated_cross_section_isotropic_km_per_mol:>12.4e}   "
                f"{record.integrated_cross_section_axis_omega_m2_per_s:>12.4e}   "
                f"{record.integrated_cross_section_isotropic_omega_m2_per_s:>12.4e}"
            )
        elif intensity == "axis":
            columns = (
                f"{record.transition_dipole_axis_D:>12.4e}   "
                f"{record.integrated_cross_section_axis_km_per_mol:>12.4e}   "
                f"{record.integrated_cross_section_axis_omega_m2_per_s:>12.4e}"
            )
        else:
            columns = (
                f"{record.transition_dipole_norm_D:>12.4e}   "
                f"{record.integrated_cross_section_isotropic_km_per_mol:>12.4e}   "
                f"{record.integrated_cross_section_isotropic_omega_m2_per_s:>12.4e}"
            )
        print(f"{prefix} {record.frequency_cm:8.1f}   {columns}{_flag(record, threshold)}")


def _format_optional(value: float | None, spec: str) -> str:
    return "     -" if value is None else format(value, spec)


def _warn_boundary(records: Sequence[Any], threshold: float, options: str) -> None:
    flagged = [record for record in records if _flag(record, threshold)]
    if not flagged:
        return
    ground = records[0].ground_grid_boundary_weight if records else None
    note = (
        f"* {len(flagged)} state(s) place more than {threshold:g} of their probability on "
        f"the outermost grid points; widen {options} and check that frequencies converge."
    )
    if ground is not None and ground > threshold:
        note += " The ground state is affected too, so every transition is suspect."
    sys.stdout.flush()
    print(note, file=sys.stderr)


def _describe_exception(exc: BaseException) -> str:
    if isinstance(exc, FileNotFoundError) and exc.filename:
        return f"{exc.strerror or 'file not found'}: {exc.filename}"
    if isinstance(exc, KeyError) and len(exc.args) == 1 and isinstance(exc.args[0], str):
        return exc.args[0]
    text = str(exc).strip()
    name = type(exc).__name__
    if not text:
        return name
    if isinstance(exc, (_CLIError, ValueError, RuntimeError)):
        return text
    return f"{name}: {text}"


def _print_error(message: str) -> None:
    sys.stdout.flush()
    lines = message.splitlines() or [""]
    print(f"pyscf-vscf: error: {lines[0]}", file=sys.stderr)
    for line in lines[1:]:
        print(f"  {line}", file=sys.stderr)
    print("  (run with --debug for a full traceback)", file=sys.stderr)


def _print_worker_traceback(exc: BaseException) -> None:
    failure = getattr(exc, "scan_failure", None)
    if failure is None or not failure.partial.failures:
        return
    first = failure.partial.failures[0]
    if first.traceback:
        print(f"Traceback of the first failed scan point {list(first.index)}:", file=sys.stderr)
        print(first.traceback, file=sys.stderr)


def _print_frequencies(freqs_cm: Sequence[float]) -> None:
    """Print harmonic frequencies; imaginary modes appear as negative numbers."""

    shown = [float(freq) for freq in freqs_cm if abs(float(freq)) > 1e-2]
    print("Frequencies (cm^-1):")
    print(" ".join(f"{freq:.1f}" for freq in shown))
    imaginary = sum(1 for freq in shown if freq < 0.0)
    if imaginary:
        print(f"{imaginary} imaginary mode(s), shown as negative numbers: not a minimum")


def _configure_logging(verbose: bool) -> None:
    """Show the package's INFO log messages on standard output when verbose."""

    import logging

    logger = logging.getLogger("pyscf_vscf")
    if verbose and not any(getattr(h, "_pyscf_vscf_cli", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[INFO] %(message)s"))
        handler._pyscf_vscf_cli = True
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)


def _log_fn(verbose: bool):
    _configure_logging(verbose)

    def emit(message: str) -> None:
        if verbose:
            print(f"[INFO] {message}", flush=True)

    return emit


def _limit_process_threads(threads: int | None) -> None:
    """Apply a thread count to pools this process has already loaded.

    PySCF's OpenMP count is set directly. NumPy's BLAS pool is fixed when NumPy
    is imported; it is limited only if ``threadpoolctl`` is installed.
    """

    if threads is None:
        return
    try:
        from pyscf import lib
    except Exception:
        pass
    else:
        lib.num_threads(int(threads))
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(int(threads))


def _worker_init(threads_per_worker: int | None, max_memory_mb: int | None = None) -> None:
    from .settings import MAX_MEMORY_ENV_VAR, apply_thread_env_updates

    apply_thread_env_updates(threads_per_worker, os.environ)
    if max_memory_mb is not None:
        os.environ[MAX_MEMORY_ENV_VAR] = str(int(max_memory_mb))
    try:
        from pyscf import lib
    except Exception:
        return
    if threads_per_worker is not None:
        lib.num_threads(int(threads_per_worker))
