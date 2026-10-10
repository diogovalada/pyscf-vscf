from __future__ import annotations

import numpy as np

from pyscf_vscf._phase import PHASE_TIE_RTOL, phase_canonical_columns, phase_pivot


def test_mirror_ties_within_roundoff_use_the_lowest_index() -> None:
    antisymmetric = np.array([0.1, -0.7, 0.0, 0.7, -0.1])
    noisy = antisymmetric.copy()
    noisy[3] += 4e-16  # roundoff makes the later mirror component the argmax

    assert phase_pivot(noisy) == 1
    for vector in (antisymmetric, noisy, -noisy):
        canonical = phase_canonical_columns(vector[:, None])[:, 0]
        assert canonical[1] > 0.0


def test_components_outside_the_tolerance_keep_the_true_maximum() -> None:
    vector = np.array([0.5, -0.7, 0.0, 0.7 * (1.0 + 1e-6), -0.1])

    assert phase_pivot(vector) == 3
    assert phase_canonical_columns(-vector[:, None])[3, 0] > 0.0


def test_ties_at_the_tolerance_boundary_are_deterministic() -> None:
    # The earlier component is slightly smaller in both cases, so only
    # tolerant tie handling can select index 0.
    inside = np.array([0.7 * (1.0 - 0.5 * PHASE_TIE_RTOL), -0.7])
    outside = np.array([0.7 * (1.0 - 10.0 * PHASE_TIE_RTOL), -0.7])

    assert phase_pivot(inside) == 0
    assert phase_pivot(outside) == 1


def test_order_key_breaks_ties_and_zero_vectors_are_left_alone() -> None:
    vector = np.array([0.6, -0.6, 0.1])
    labels = [(1, 0), (0, 1), (0, 0)]

    assert phase_pivot(vector, order_key=lambda index: labels[index]) == 1
    canonical = phase_canonical_columns(vector[:, None], order_key=lambda index: labels[index])
    np.testing.assert_allclose(canonical[:, 0], -vector)
    np.testing.assert_array_equal(phase_canonical_columns(np.zeros((3, 1))), np.zeros((3, 1)))
