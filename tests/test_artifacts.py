from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf import _artifacts


def test_atomic_text_and_npz_writes(tmp_path: Path) -> None:
    text_path = tmp_path / "nested" / "artifact.json"
    npz_path = tmp_path / "arrays.npz"

    _artifacts.atomic_write_text(text_path, '{"value":1}\n')
    _artifacts.atomic_savez_compressed(npz_path, {"values": np.arange(4.0)})

    assert text_path.read_text(encoding="utf-8") == '{"value":1}\n'
    with np.load(npz_path, allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive["values"], np.arange(4.0))


def test_failed_atomic_replace_preserves_existing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "artifact.json"
    target.write_text("old\n", encoding="utf-8")

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("simulated replace failure")

    monkeypatch.setattr(_artifacts.os, "replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        _artifacts.atomic_write_text(target, "new\n")

    assert target.read_text(encoding="utf-8") == "old\n"
    assert tuple(tmp_path.iterdir()) == (target,)
