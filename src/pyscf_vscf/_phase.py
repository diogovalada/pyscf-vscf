"""Deterministic sign conventions for real eigenvectors.

A real eigenvector is defined only up to an overall sign. The package fixes
that sign by making a pivot component positive. Choosing the pivot with an
exact ``argmax`` is not portable: symmetric problems produce components with
equal magnitude at mirror-image grid points, and LAPACK/BLAS roundoff then
decides which one wins, so signed quantities such as transition dipoles can
flip between NumPy builds. Components whose magnitude is within ``rtol`` of
the maximum are therefore treated as tied, and ties are broken by a fixed
order (the lowest index unless the caller supplies an ordering key).

The tolerance is set from measured mirror-component noise on symmetric
sinc-DVR problems: about 3e-12 relative for dense ``eigh`` and at most about
the solver tolerance for ``eigsh`` (5e-11 at 1e-10, the default VCI
iterative tolerance; the kinetic solver uses at most 1e-11). ``1e-8`` leaves
two orders of magnitude of margin while keeping clearly unequal components
distinct.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

PHASE_TIE_RTOL = 1e-8


def phase_pivot(
    values: np.ndarray,
    *,
    rtol: float = PHASE_TIE_RTOL,
    order_key: Callable[[int], object] | None = None,
) -> int:
    """Return the index whose component sign defines the vector's phase."""

    magnitudes = np.abs(np.asarray(values, dtype=float))
    maximum = float(np.max(magnitudes))
    if maximum == 0.0:
        return 0
    tied = np.flatnonzero(magnitudes >= maximum * (1.0 - float(rtol)))
    if order_key is None:
        return int(tied[0])
    return min((int(index) for index in tied), key=order_key)


def phase_canonical_columns(
    vectors: np.ndarray,
    *,
    rtol: float = PHASE_TIE_RTOL,
    order_key: Callable[[int], object] | None = None,
) -> np.ndarray:
    """Return a copy whose columns have a positive pivot component."""

    canonical = np.array(vectors, dtype=float, copy=True)
    for column in range(canonical.shape[1]):
        pivot = phase_pivot(canonical[:, column], rtol=rtol, order_key=order_key)
        if canonical[pivot, column] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


__all__ = ["PHASE_TIE_RTOL", "phase_canonical_columns", "phase_pivot"]
