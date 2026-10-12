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


def _rendered(markdown: str) -> tuple[str, list[str]]:
    """Whitespace-normalized HTML, plus the exact content of every code span."""

    import re

    from markdown_it import MarkdownIt

    parser = MarkdownIt("commonmark").enable("table")
    code: list[str] = []

    def walk(tokens: list) -> None:
        for token in tokens:
            if token.type == "code_inline":
                code.append(token.content)
            walk(token.children or [])

    walk(parser.parse(markdown))
    return re.sub(r"\s+", " ", parser.render(markdown)).strip(), code


def test_unwrapping_joins_paragraph_lines_without_changing_the_rendering() -> None:
    pytest.importorskip("markdown_it")
    text = "\n".join(
        [
            "Intro sentence that",
            "continues here with a `code",
            "span` and a [link](https://e.org/x). A `spaced   ",
            "code` span.",
            "",
            "An indented `split",
            " code` span stays wrapped.",
            "",
            "So does ![an image `label",
            " code`](https://e.org/i.png) span.",
            "",
            "- A list item",
            "  wrapped over lines.",
            "  - Nested item",
            "    also wrapped.",
            "",
            "> A quote",
            "> wrapped.",
            "",
            "Hard break\\",
            "kept.",
            "",
            "```",
            "code line one",
            "code line two",
            "```",
            "",
            "| a | b |",
            "| - | - |",
            "| 1 | 2 |",
        ]
    )
    unwrapped = release_notes.unwrap_paragraphs(text)
    assert "Intro sentence that continues here with a `code span` and a [link]" in unwrapped
    assert "A `spaced    code` span." in unwrapped
    assert "An indented `split\n code` span stays wrapped." in unwrapped
    assert "So does ![an image `label\n code`](https://e.org/i.png) span." in unwrapped
    assert "- A list item wrapped over lines." in unwrapped
    assert "  - Nested item also wrapped." in unwrapped
    assert "> A quote\n> wrapped." in unwrapped
    assert "Hard break\\\nkept." in unwrapped
    assert "code line one\ncode line two" in unwrapped
    assert "| a | b |\n| - | - |\n| 1 | 2 |" in unwrapped
    assert _rendered(unwrapped) == _rendered(text)


def test_unwrapped_changelog_sections_render_unchanged() -> None:
    pytest.importorskip("markdown_it")
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    for version in release_notes.changelog_versions(text):
        if version == "Unreleased":
            continue
        section = release_notes.changelog_section(text, version)
        assert _rendered(release_notes.unwrap_paragraphs(section)) == _rendered(section), version
