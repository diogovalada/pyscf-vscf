"""Lowest eigenpairs of a symmetric matrix-free operator, degeneracies included."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh


def deterministic_start(dimension: int) -> np.ndarray:
    """Return a fixed, dense, normalized starting vector."""

    indices = np.arange(1, dimension + 1, dtype=float)
    vector = np.sin(np.sqrt(2.0) * indices) + np.cos(np.sqrt(3.0) * indices)
    return vector / np.linalg.norm(vector)


def lowest_eigenpairs(
    matvec: Callable[[np.ndarray], np.ndarray],
    *,
    dimension: int,
    count: int,
    tolerance: float,
    degeneracy_tolerance: float,
    shift: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return at least the ``count`` lowest eigenpairs, in ascending order.

    The operator is shifted by ``shift`` (default: its first diagonal element;
    pass a value near the lowest roots, such as the smallest diagonal element)
    so that ARPACK's relative tolerance acts on energies near zero. Lanczos
    finds one vector per degenerate eigenvalue in exact arithmetic, so after
    the first solve the operator is deflated against the vectors found and
    solved again until no eigenvalue at or below the highest one found
    (within ``degeneracy_tolerance``) remains. The result can therefore hold
    more than ``count`` pairs, completing a degenerate block at the cutoff.
    """

    if not 1 <= count < dimension:
        raise ValueError("count must satisfy 1 <= count < dimension")
    if shift is None:
        unit = np.zeros(dimension)
        unit[0] = 1.0
        shift = float(np.asarray(matvec(unit))[0])

    def shifted(vector: np.ndarray) -> np.ndarray:
        return np.asarray(matvec(vector), dtype=float) - shift * vector

    energies, vectors = _eigsh(
        shifted, dimension, count, tolerance, deterministic_start(dimension)
    )
    margin = float(degeneracy_tolerance) + 10.0 * float(tolerance)
    while vectors.shape[1] < dimension - 1:
        cutoff = float(energies[-1])
        raise_by = cutoff + (cutoff - float(energies[0])) + 1.0
        basis = vectors

        def deflated(vector: np.ndarray, basis: np.ndarray = basis) -> np.ndarray:
            projection = basis.T @ vector
            orthogonal = vector - basis @ projection
            acted = shifted(orthogonal)
            acted = acted - basis @ (basis.T @ acted)
            return acted + raise_by * (basis @ projection)

        start = deterministic_start(dimension)
        start = start - basis @ (basis.T @ start)
        start /= np.linalg.norm(start)
        k = min(count, dimension - basis.shape[1] - 1)
        found, candidates = _eigsh(deflated, dimension, k, tolerance, start)
        missing = found <= cutoff + margin
        if not np.any(missing):
            break
        combined, _ = np.linalg.qr(np.column_stack([vectors, candidates[:, missing]]))
        projected = combined.T @ np.column_stack(
            [shifted(combined[:, column]) for column in range(combined.shape[1])]
        )
        energies, rotation = np.linalg.eigh(0.5 * (projected + projected.T))
        vectors = combined @ rotation
    return energies + shift, vectors


def _eigsh(
    matvec: Callable[[np.ndarray], np.ndarray],
    dimension: int,
    count: int,
    tolerance: float,
    start: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    operator = LinearOperator((dimension, dimension), matvec=matvec, rmatvec=matvec, dtype=float)
    energies, vectors = eigsh(operator, k=count, which="SA", v0=start, tol=tolerance)
    order = np.argsort(energies)
    return np.asarray(energies[order], dtype=float), np.asarray(vectors[:, order], dtype=float)


__all__ = ["deterministic_start", "lowest_eigenpairs"]
