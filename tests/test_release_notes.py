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

- A fix; see the [guide](https://example.org/guide) and [PySCF](https://pyscf.org).

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


def test_relative_links_are_found_with_commonmark_rules() -> None:
    pytest.importorskip("markdown_it")
    text = "\n".join(
        [
            '- [titled](docs/a.md "A `guide`") and ![image](assets/b.png)',
            "- [angle](<docs/c d.md>), [anchor](#top), [mail](mailto:x@y.z)",
            "- [protocol-relative](//cdn.example.org/x.js) and [abs](https://e.org/y)",
            "- a code span `[not](a/link)` and a [reference][ref]",
            "- a multiline code span `[not](b/link)",
            "  continued` here",
            "",
            "[ref]: docs/ref.md 'Reference'",
            "",
            "```python",
            "[fenced](not/a/link)",
            "```python is not a closing fence",
            "[still fenced](not/c/link)",
            "```",
        ]
    )
    found = release_notes.relative_links(text)
    assert sorted(found) == sorted(["docs/a.md", "assets/b.png", "docs/c%20d.md", "docs/ref.md"])


def test_the_changelog_has_notes_and_absolute_links_for_every_version() -> None:
    pytest.importorskip("markdown_it")
    import pyscf_vscf

    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    versions = [
        version for version in release_notes.changelog_versions(text) if version != "Unreleased"
    ]
    assert pyscf_vscf.__version__ in versions
    for version in versions:
        section = release_notes.changelog_section(text, version)
        assert release_notes.relative_links(section) == [], version
    unreleased = text.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
    assert release_notes.relative_links(unreleased) == []
