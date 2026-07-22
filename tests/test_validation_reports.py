from __future__ import annotations

import pytest

from pyscf_vscf.validation import convergence_report


def _record(assignment, frequency, intensity, weight=0.9):
    return {
        "assignment": assignment,
        "assignment_weight": weight,
        "assignment_dominant_manifold_weight": 0.98,
        "assignment_participation_ratio": 1.5,
        "freq_cm": frequency,
        "integrated_cross_section_isotropic_omega_m2_per_s": intensity,
    }


def test_convergence_report_matches_assignments_and_quantifies_spreads() -> None:
    report = convergence_report(
        {
            "coarse": [_record((1, 0), 1000.0, 2.0e-10, 0.85)],
            "fine": [
                _record((1, 0), 1001.5, 2.2e-10, 0.95),
                _record((0, 1), 1100.0, 1.0e-10),
            ],
        }
    )

    assert report.run_names == ("coarse", "fine")
    assert len(report.states) == 1
    state = report.states[0]
    assert state.assignment == (1, 0)
    assert state.frequency_spread_cm == pytest.approx(1.5)
    assert state.intensity_relative_spread == pytest.approx(0.2 / 2.2)
    assert state.minimum_assignment_weight == pytest.approx(0.85)
    assert state.minimum_dominant_manifold_weight == pytest.approx(0.98)
    assert state.maximum_participation_ratio == pytest.approx(1.5)
    assert report.unmatched["coarse"] == ((0, 1),)
    assert report.unmatched["fine"] == ()


def test_convergence_report_rejects_frequency_only_and_duplicate_matching() -> None:
    with pytest.raises(ValueError, match="explicit"):
        convergence_report({"a": [{"freq_cm": 1.0}], "b": [{"freq_cm": 1.0}]})
    duplicate = [_record((1, 0), 1.0, 1.0), _record((1, 0), 2.0, 2.0)]
    with pytest.raises(ValueError, match="duplicate"):
        convergence_report({"a": duplicate, "b": [_record((1, 0), 1.0, 1.0)]})
