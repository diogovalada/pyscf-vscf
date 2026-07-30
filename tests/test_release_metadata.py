from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import pyscf_vscf


ROOT = Path(__file__).resolve().parents[1]


def test_release_versions_are_consistent() -> None:
    installed = version("pyscf-vscf")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    match = re.search(r"^version:\s*([^\s]+)\s*$", citation, flags=re.MULTILINE)

    assert match is not None
    assert installed == pyscf_vscf.__version__ == match.group(1)
    assert f"## [{installed}] - " in changelog
