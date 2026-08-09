"""Small atomic-write primitives for retained scientific artifacts."""

from __future__ import annotations

import io
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np


def atomic_write_bytes(path: Path | str, content: bytes) -> None:
    """Atomically replace *path* with *content* after syncing the temporary file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_text(path: Path | str, content: str) -> None:
    """Atomically replace a UTF-8 text artifact using Unix newlines."""

    atomic_write_bytes(path, content.encode("utf-8"))


def atomic_savez_compressed(path: Path | str, arrays: Mapping[str, np.ndarray]) -> None:
    """Serialize one compressed NumPy archive before atomically publishing it."""

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    atomic_write_bytes(path, buffer.getvalue())


__all__ = ["atomic_savez_compressed", "atomic_write_bytes", "atomic_write_text"]
