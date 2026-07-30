#!/usr/bin/env python3
"""Compare regenerated validation JSON with a checked-in reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pyscf_vscf.report_compare import assert_reports_close


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected", type=Path)
    parser.add_argument("actual", type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=1e-9)
    parser.add_argument("--absolute-tolerance", type=float, default=1e-15)
    args = parser.parse_args()

    expected = json.loads(args.expected.read_text(encoding="utf-8"))
    actual = json.loads(args.actual.read_text(encoding="utf-8"))
    assert_reports_close(
        expected,
        actual,
        relative_tolerance=args.relative_tolerance,
        absolute_tolerance=args.absolute_tolerance,
    )
    print(
        "Validation reports match "
        f"(rtol={args.relative_tolerance:g}, atol={args.absolute_tolerance:g})."
    )


if __name__ == "__main__":
    main()
