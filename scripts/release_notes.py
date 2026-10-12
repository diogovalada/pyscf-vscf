#!/usr/bin/env python3
"""Print the CHANGELOG.md section of one version as GitHub release notes.

The release workflow publishes this text as the body of the GitHub release, so
the changelog is the single source of release notes. Relative links are made
absolute with ``--link-base`` (for example the repository's blob URL at the
release tag), because a release body has no file context. Exits with status 1
if the version has no section or the section is empty.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_HEADING = re.compile(r"^## \[(?P<version>[^\]]+)\]")
# Inline link or image destination, optionally in angle brackets, then an
# optional title: ](dest), ](<dest>), ](dest "title").
_INLINE = re.compile(r"\]\(\s*(?P<dest><[^>\n]*>|[^\s)]+)(?P<rest>(?:\s+[^)]*)?)\)")
# Reference definition: [label]: dest "title"
_REFERENCE = re.compile(r"^(?P<lead> {0,3}\[[^\]]+\]:\s*)(?P<dest><[^>]*>|\S+)(?P<rest>.*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_CODE_SPAN = re.compile(r"(`+)(?:.+?)\1")
_EXTERNAL = re.compile(r"^(?:[A-Za-z][A-Za-z0-9+.-]*:|//|#)")


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


def absolute_links(text: str, base: str) -> str:
    """Prefix relative Markdown link and image destinations with ``base``.

    Inline links (with or without titles), images, and reference definitions
    are rewritten; destinations with a URL scheme, protocol-relative URLs, and
    in-page anchors are left alone, as is everything inside code spans and
    fenced code blocks.
    """

    root = base.rstrip("/") + "/"

    def absolute(destination: str) -> str:
        bracketed = destination.startswith("<") and destination.endswith(">")
        target = destination[1:-1] if bracketed else destination
        if not target or _EXTERNAL.match(target):
            return destination
        target = root + target.lstrip("/")
        return f"<{target}>" if bracketed else target

    def inline(match: re.Match[str]) -> str:
        return f"]({absolute(match.group('dest'))}{match.group('rest')})"

    def outside_code(line: str) -> str:
        pieces, position = [], 0
        for span in _CODE_SPAN.finditer(line):
            pieces.append(_INLINE.sub(inline, line[position : span.start()]))
            pieces.append(span.group(0))
            position = span.end()
        pieces.append(_INLINE.sub(inline, line[position:]))
        return "".join(pieces)

    lines, fence = [], None
    for line in text.split("\n"):
        marker = _FENCE.match(line)
        if fence is not None:
            if marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence):
                fence = None
            lines.append(line)
            continue
        if marker:
            fence = marker.group(1)
            lines.append(line)
            continue
        reference = _REFERENCE.match(line)
        if reference:
            line = f"{reference.group('lead')}{absolute(reference.group('dest'))}{reference.group('rest')}"
        else:
            line = outside_code(line)
        lines.append(line)
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="version without the leading v, e.g. 0.3.1")
    parser.add_argument("--changelog", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--link-base", help="URL prepended to relative link targets")
    args = parser.parse_args(argv)
    try:
        notes = changelog_section(args.changelog.read_text(encoding="utf-8"), args.version)
    except ValueError as exc:
        print(f"release_notes: {exc}", file=sys.stderr)
        return 1
    if args.link_base:
        notes = absolute_links(notes, args.link_base)
    print(notes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
