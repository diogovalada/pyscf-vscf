from __future__ import annotations

import re

import pytest

from pyscf_vscf.report_compare import ReportComparisonError, assert_reports_close


def test_report_comparison_accepts_tight_cross_platform_roundoff() -> None:
    expected = {
        "frequency_cm": 2829.814620568318,
        "intensity_m2_per_s": 9.588574779090422e-11,
        "assignments": [[1, 0], [0, 1]],
        "passed": True,
    }
    actual = {
        "frequency_cm": expected["frequency_cm"] + 2e-8,
        "intensity_m2_per_s": expected["intensity_m2_per_s"] * (1.0 + 2e-10),
        "assignments": [[1, 0], [0, 1]],
        "passed": True,
    }

    assert_reports_close(expected, actual)


def test_report_comparison_uses_unit_aware_absolute_tolerances() -> None:
    assert_reports_close(
        {"vscf_minus_exact_centroid_cm": 0.009976373608424183},
        {"vscf_minus_exact_centroid_cm": 0.00997637371347082},
    )

    with pytest.raises(ReportComparisonError, match="intensity_m2_per_s"):
        assert_reports_close(
            {"intensity_m2_per_s": 1e-18},
            {"intensity_m2_per_s": 1.1e-18},
        )


@pytest.mark.parametrize(
    ("actual", "path"),
    [
        ({"value": 1.0001}, "$.value"),
        ({"value": [1, 3]}, "$.value[1]"),
        ({"other": 1.0}, "$"),
        ({"value": 1}, "$.value"),
    ],
)
def test_report_comparison_rejects_material_or_structural_changes(actual: dict, path: str) -> None:
    expected = {"value": 1.0} if path != "$.value[1]" else {"value": [1, 2]}

    with pytest.raises(ReportComparisonError, match=rf"^{re.escape(path)}"):
        assert_reports_close(expected, actual)


def test_report_comparison_rejects_negative_tolerances() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        assert_reports_close({}, {}, relative_tolerance=-1.0)
