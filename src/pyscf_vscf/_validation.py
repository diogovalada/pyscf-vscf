"""Small validators and helpers shared by the numerical modules."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def nonempty(name: str, value: object) -> str:
    """Return ``str(value)`` stripped; raise ``ValueError`` if it is empty."""

    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


def nonempty_string(name: str, value: object) -> str:
    """Like :func:`nonempty`, but ``value`` must already be a string."""

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return nonempty(name, value)


def positive_finite(name: str, value: object) -> float:
    """Return ``float(value)``; raise ``ValueError`` unless it is finite and positive."""

    number = float(value)  # type: ignore[arg-type]
    if not np.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return number


def subset_key(subset: Sequence[int]) -> str:
    """Return the canonical ``"i,j,..."`` key of a mode subset."""

    return ",".join(str(mode) for mode in subset)


def mesh_points(axes: Sequence[np.ndarray]) -> np.ndarray:
    """Return all grid points of a tensor-product mesh, shape ``(n_points, n_axes)``."""

    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack(mesh, axis=-1).reshape((-1, len(axes)))


def component_max_abs(values: np.ndarray, leading_rank: int) -> tuple[float, ...]:
    """Return the maximum absolute value of each trailing component."""

    array = np.asarray(values, dtype=float)
    trailing = array.shape[leading_rank:]
    rows = array.reshape((-1, int(np.prod(trailing)) if trailing else 1))
    return tuple(float(value) for value in np.max(np.abs(rows), axis=0))


__all__ = [
    "component_max_abs",
    "mesh_points",
    "nonempty",
    "nonempty_string",
    "positive_finite",
    "subset_key",
]
