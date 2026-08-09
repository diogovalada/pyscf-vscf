"""Internal immutable NumPy-array storage."""

from __future__ import annotations

import numpy as np


class _ImmutableArray(np.ndarray):
    """Array view backed by immutable bytes, including after pickling."""

    def __reduce__(self):
        return (
            _restore_immutable_array,
            (self.dtype.str, self.shape, self.tobytes(order="C")),
        )


def _restore_immutable_array(
    dtype: str,
    shape: tuple[int, ...],
    content: bytes,
) -> np.ndarray:
    values = np.frombuffer(content, dtype=np.dtype(dtype)).reshape(shape)
    return values.view(_ImmutableArray)


def immutable_array(values: object, *, dtype: object | None = None) -> np.ndarray:
    """Return a C-contiguous copy whose write flag cannot be re-enabled."""

    array = np.array(values, dtype=dtype, copy=True, order="C", subok=False)
    return _restore_immutable_array(array.dtype.str, array.shape, array.tobytes(order="C"))


__all__ = ["immutable_array"]
