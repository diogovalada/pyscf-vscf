#!/usr/bin/env python3
"""Print the CHANGELOG.md section of one version as GitHub release notes.

The release workflow publishes this text verbatim as the body of the GitHub
release, so the changelog is the single source of release notes. A release
body has no file context, so links in the changelog must be absolute URLs;
``--check-links`` parses the section as CommonMark (with ``markdown-it-py``)
and fails if any link or image points to a relative path. Exits with status 1
if the version has no section, the section is empty, or a checked link is
relative.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")
_ABSOLUTE = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//|#)")


def changelog_section(text: str, version: str) -> str:
    """Return the body of ``## [version]`` up to the next version heading."""

    lines = text.splitlines()
    start = None
    for index, line in enumerate(lines):
        match = _HEADING.match(line)
        if match is None:
            continue
        if start is not None:
            body = "\n".join(lines[start:index]).strip()
            break
        if match.group("version") == version:
            start = index + 1
    else:
        body = "" if start is None else "\n".join(lines[start:]).strip()
    if start is None:
        raise ValueError(f"CHANGELOG.md has no section for version {version}")
    if not body:
        raise ValueError(f"The CHANGELOG.md section for version {version} is empty")
    return body


def changelog_versions(text: str) -> list[str]:
    """Return every version heading in the changelog, newest first."""

    return [match.group("version") for match in map(_HEADING.match, text.splitlines()) if match]


def relative_links(markdown: str) -> list[str]:
    """Return link and image destinations that are not absolute URLs or anchors."""

    from markdown_it import MarkdownIt

    found: list[str] = []

    def walk(tokens: list) -> None:
        for token in tokens:
            if token.type == "link_open":
                target = token.attrGet("href")
            elif token.type == "image":
                target = token.attrGet("src")
            else:
                target = None
            if target is not None and not _ABSOLUTE.match(str(target)):
                found.append(str(target))
            if token.children:
                walk(token.children)

    walk(MarkdownIt("commonmark").parse(markdown))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="version without the leading v, e.g. 0.3.1")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument(
        "--check-links",
        action="store_true",
        help="fail if the section links to a relative path (needs markdown-it-py)",
    )
    args = parser.parse_args(argv)
    try:
        notes = changelog_section(args.changelog.read_text(encoding="utf-8"), args.version)
    except ValueError as exc:
        print(f"release_notes: {exc}", file=sys.stderr)
        return 1
    if args.check_links:
        relative = relative_links(notes)
        if relative:
            print(
                f"release_notes: the {args.version} section links to relative paths "
                f"(use absolute URLs): {', '.join(relative)}",
                file=sys.stderr,
            )
            return 1
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
