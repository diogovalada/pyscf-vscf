from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "release_notes", ROOT / "scripts" / "release_notes.py"
)
release_notes = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(release_notes)

CHANGELOG = """# Changelog

## [Unreleased]

## [1.1.0] - 2026-01-02

### Fixed

- A fix; see the [guide](docs/guide.md) and [PySCF](https://pyscf.org).

## [1.1.0a1] - 2026-01-01

- A prerelease.

## [1.0.0] - 2025-12-31
"""


def test_section_is_the_text_between_version_headings() -> None:
    notes = release_notes.changelog_section(CHANGELOG, "1.1.0")
    assert notes.startswith("### Fixed")
    assert "prerelease" not in notes
    assert release_notes.changelog_section(CHANGELOG, "1.1.0a1") == "- A prerelease."


def test_missing_and_empty_sections_are_errors(capsys) -> None:
    with pytest.raises(ValueError, match="no section"):
        release_notes.changelog_section(CHANGELOG, "2.0.0")
    with pytest.raises(ValueError, match="empty"):
        release_notes.changelog_section(CHANGELOG, "1.0.0")
    path = ROOT / "CHANGELOG.md"
    assert release_notes.main(["9.9.9", "--changelog", str(path)]) == 1
    assert "no section" in capsys.readouterr().err


def test_relative_links_become_absolute() -> None:
    notes = release_notes.absolute_links(
        release_notes.changelog_section(CHANGELOG, "1.1.0"),
        "https://github.com/o/r/blob/v1.1.0/",
    )
    assert "[guide](https://github.com/o/r/blob/v1.1.0/docs/guide.md)" in notes
    assert "[PySCF](https://pyscf.org)" in notes


def test_the_current_version_has_notes() -> None:
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    import pyscf_vscf

    assert release_notes.changelog_section(text, pyscf_vscf.__version__)


def test_link_rewriting_respects_markdown() -> None:
    text = "\n".join(
        [
            '- [titled](docs/a.md "A guide") and ![image](assets/b.png)',
            "- [angle](<docs/c d.md>), [anchor](#top), [mail](mailto:x@y.z)",
            "- [protocol-relative](//cdn.example.org/x.js) and [abs](https://e.org/y)",
            "- code `[not](a/link)` stays",
            "[ref]: docs/ref.md 'Reference'",
            "```",
            "[fenced](not/a/link)",
            "```",
        ]
    )
    out = release_notes.absolute_links(text, "https://b/blob/v1")
    assert '[titled](https://b/blob/v1/docs/a.md "A guide")' in out
    assert "![image](https://b/blob/v1/assets/b.png)" in out
    assert "[angle](<https://b/blob/v1/docs/c d.md>)" in out
    assert "[anchor](#top)" in out and "[mail](mailto:x@y.z)" in out
    assert "[protocol-relative](//cdn.example.org/x.js)" in out
    assert "[abs](https://e.org/y)" in out
    assert "`[not](a/link)`" in out
    assert "[ref]: https://b/blob/v1/docs/ref.md 'Reference'" in out
    assert "[fenced](not/a/link)" in out
