"""Cross-platform comparison of regenerated JSON validation reports."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from numbers import Integral, Real


class ReportComparisonError(AssertionError):
    """Raised when regenerated validation data differ materially."""


def assert_reports_close(
    expected: object,
    actual: object,
    *,
    relative_tolerance: float = 1e-9,
    absolute_tolerance: float = 1e-15,
    wavenumber_absolute_tolerance: float = 1e-9,
    intensity_absolute_tolerance: float = 1e-24,
) -> None:
    """Require identical JSON structure and tightly matching floating-point values."""

    tolerances = (
        relative_tolerance,
        absolute_tolerance,
        wavenumber_absolute_tolerance,
        intensity_absolute_tolerance,
    )
    if any(tolerance < 0.0 for tolerance in tolerances):
        raise ValueError("Report comparison tolerances must be non-negative")
    _compare(
        expected,
        actual,
        path="$",
        relative_tolerance=relative_tolerance,
        absolute_tolerance=absolute_tolerance,
        wavenumber_absolute_tolerance=wavenumber_absolute_tolerance,
        intensity_absolute_tolerance=intensity_absolute_tolerance,
    )


def _compare(
    expected: object,
    actual: object,
    *,
    path: str,
    relative_tolerance: float,
    absolute_tolerance: float,
    wavenumber_absolute_tolerance: float,
    intensity_absolute_tolerance: float,
) -> None:
    if isinstance(expected, bool) or isinstance(actual, bool):
        _require_exact(expected, actual, path)
        return

    if isinstance(expected, Integral) or isinstance(actual, Integral):
        _require_exact(expected, actual, path)
        return

    if isinstance(expected, Real) and isinstance(actual, Real):
        expected_float = float(expected)
        actual_float = float(actual)
        if not math.isfinite(expected_float) or not math.isfinite(actual_float):
            _require_exact(expected_float, actual_float, path)
            return
        effective_absolute_tolerance = _absolute_tolerance_for_path(
            path,
            default=absolute_tolerance,
            wavenumber=wavenumber_absolute_tolerance,
            intensity=intensity_absolute_tolerance,
        )
        if not math.isclose(
            expected_float,
            actual_float,
            rel_tol=relative_tolerance,
            abs_tol=effective_absolute_tolerance,
        ):
            difference = abs(expected_float - actual_float)
            raise ReportComparisonError(
                f"{path}: expected {expected_float!r}, got {actual_float!r} "
                f"(absolute difference {difference:.17g})"
            )
        return

    if isinstance(expected, Mapping) and isinstance(actual, Mapping):
        expected_keys = set(expected)
        actual_keys = set(actual)
        if expected_keys != actual_keys:
            missing = sorted(expected_keys - actual_keys, key=str)
            unexpected = sorted(actual_keys - expected_keys, key=str)
            raise ReportComparisonError(
                f"{path}: object keys differ; missing={missing!r}, unexpected={unexpected!r}"
            )
        for key in expected:
            _compare(
                expected[key],
                actual[key],
                path=f"{path}.{key}",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                wavenumber_absolute_tolerance=wavenumber_absolute_tolerance,
                intensity_absolute_tolerance=intensity_absolute_tolerance,
            )
        return

    if _is_json_sequence(expected) and _is_json_sequence(actual):
        if len(expected) != len(actual):
            raise ReportComparisonError(
                f"{path}: array lengths differ; expected {len(expected)}, got {len(actual)}"
            )
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare(
                expected_item,
                actual_item,
                path=f"{path}[{index}]",
                relative_tolerance=relative_tolerance,
                absolute_tolerance=absolute_tolerance,
                wavenumber_absolute_tolerance=wavenumber_absolute_tolerance,
                intensity_absolute_tolerance=intensity_absolute_tolerance,
            )
        return

    _require_exact(expected, actual, path)


def _is_json_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def _absolute_tolerance_for_path(
    path: str,
    *,
    default: float,
    wavenumber: float,
    intensity: float,
) -> float:
    key = path.rsplit(".", maxsplit=1)[-1].split("[", maxsplit=1)[0].lower()
    if "intens" in key or "cross_section" in key:
        return intensity
    if key.endswith("_cm"):
        return wavenumber
    return default


def _require_exact(expected: object, actual: object, path: str) -> None:
    if type(expected) is not type(actual) or expected != actual:
        raise ReportComparisonError(f"{path}: expected {expected!r}, got {actual!r}")


__all__ = ["ReportComparisonError", "assert_reports_close"]
