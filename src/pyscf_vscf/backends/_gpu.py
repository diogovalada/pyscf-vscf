"""Optional GPU execution through GPU4PySCF.

A method built for ``device="gpu"`` is configured on the CPU exactly as for
``device="cpu"`` and then converted with PySCF's ``to_gpu()``. GPU4PySCF and
CuPy are imported only when such a method is built, so the package imports
and runs on machines without them.
"""

from __future__ import annotations

import os
from typing import Any

DEVICES = ("cpu", "gpu")
GPU_INSTALL_HINT = (
    'install the GPU extra for your CUDA version, e.g. pip install "pyscf-vscf[gpu-cuda12]"'
)


def normalize_device(value: object) -> str:
    """Return ``"cpu"`` or ``"gpu"``; raise ``ValueError`` for anything else."""

    if not isinstance(value, str):
        raise TypeError("device must be 'cpu' or 'gpu'")
    device = value.strip().lower()
    if device not in DEVICES:
        raise ValueError(f"device must be 'cpu' or 'gpu' (got {value!r})")
    return device


def is_gpu_object(value: Any) -> bool:
    """True if ``value`` is a GPU4PySCF object (or an instance of a subclass of one)."""

    return any(cls.__module__.split(".")[0] == "gpu4pyscf" for cls in type(value).__mro__)


def require_gpu() -> tuple[Any, Any]:
    """Import GPU4PySCF and CuPy and require a visible CUDA device.

    Raises ``BackendUnavailableError`` (an ``ImportError``) with an install hint
    when either package is missing, and ``RuntimeError`` when no device is
    visible.
    """

    from ._pyscf_factory import BackendUnavailableError

    try:
        import cupy
        import gpu4pyscf
    except Exception as exc:
        raise BackendUnavailableError(
            f"device='gpu' needs GPU4PySCF and CuPy ({type(exc).__name__}: {exc}); "
            f"{GPU_INSTALL_HINT}"
        ) from exc
    try:
        count = int(cupy.cuda.runtime.getDeviceCount())
    except Exception as exc:
        raise RuntimeError(f"device='gpu': CUDA is unavailable ({exc})") from exc
    if count < 1:
        raise RuntimeError("device='gpu': no CUDA device is visible")
    return gpu4pyscf, cupy


def to_gpu(mean_field: Any) -> Any:
    """Return the GPU4PySCF version of a configured, unconverged mean field."""

    from ._pyscf_factory import BackendUnavailableError

    require_gpu()
    converter = getattr(mean_field, "to_gpu", None)
    if not callable(converter):
        raise BackendUnavailableError(
            f"{type(mean_field).__name__} has no to_gpu(); device='gpu' needs a PySCF "
            f"version that GPU4PySCF supports ({GPU_INSTALL_HINT})"
        )
    converted = converter()
    if not is_gpu_object(converted):
        raise RuntimeError(
            f"{type(mean_field).__name__}.to_gpu() returned {type(converted).__name__}, "
            "not a GPU4PySCF object"
        )
    return converted


def visible_gpu_ids() -> tuple[str, ...]:
    """Return the CUDA device identifiers worker processes can be pinned to.

    ``CUDA_VISIBLE_DEVICES`` is honoured when set, as CUDA reads it: entries
    after the first invalid one (such as ``-1``) are hidden, and the list is
    cut to the number of devices CUDA reports. Otherwise every device CuPy
    reports is used.
    """

    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is None:
        _, cupy = require_gpu()
        return tuple(str(index) for index in range(int(cupy.cuda.runtime.getDeviceCount())))
    ids: list[str] = []
    for item in configured.split(","):
        item = item.strip()
        if not item or item.startswith("-"):
            break
        ids.append(item)
    if not ids:
        raise RuntimeError("device='gpu': CUDA_VISIBLE_DEVICES hides every device")
    try:
        _, cupy = require_gpu()
        count = int(cupy.cuda.runtime.getDeviceCount())
    except ImportError:
        count = len(ids)
    return tuple(ids[:count])


def gpu_provenance() -> dict[str, object]:
    """Return the device and GPU software of this process, for provenance."""

    try:
        gpu4pyscf, cupy = require_gpu()
        runtime = cupy.cuda.runtime
        device = int(cupy.cuda.Device().id)
        properties = runtime.getDeviceProperties(device)
        name = properties.get("name", b"")
        return {
            "device_name": name.decode() if isinstance(name, bytes) else str(name),
            "compute_capability": f"{properties.get('major')}.{properties.get('minor')}",
            "cuda_runtime_version": int(runtime.runtimeGetVersion()),
            "cuda_driver_version": int(runtime.driverGetVersion()),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu4pyscf": str(getattr(gpu4pyscf, "__version__", "unknown")),
            "cupy": str(getattr(cupy, "__version__", "unknown")),
        }
    except Exception as exc:  # provenance must never fail a calculation
        return {"error": f"{type(exc).__name__}: {exc}"}


__all__ = [
    "DEVICES",
    "GPU_INSTALL_HINT",
    "gpu_provenance",
    "is_gpu_object",
    "normalize_device",
    "require_gpu",
    "to_gpu",
    "visible_gpu_ids",
]
