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
_RELATIVE_LINK = re.compile(r"\]\((?!https?://|mailto:|#)(?P<target>[^)\s]+)\)")


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
    """Prefix relative Markdown link targets with ``base``."""

    root = base.rstrip("/") + "/"
    return _RELATIVE_LINK.sub(lambda match: f"]({root}{match.group('target')})", text)


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
