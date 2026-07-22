from __future__ import annotations

import numpy as np
import pytest

from pyscf_vscf.assignments import assign_product_states_2d
from pyscf_vscf.dvr import product_dvr_2d


def test_separable_product_states_have_unique_unit_weight_assignments() -> None:
    q1 = np.linspace(-0.5, 0.5, 13)
    q2 = np.linspace(-0.45, 0.45, 11)
    v1 = 0.08 * q1**2
    v2 = 0.13 * q2**2
    dvr = product_dvr_2d(
        q1,
        q2,
        1.0,
        1.4,
        v1[:, None] + v2[None, :],
        k_eigs=7,
    )

    assignments = assign_product_states_2d(dvr, 1.0, 1.4, v1, v2)

    assert assignments[0].quanta == (0, 0)
    assert len({assignment.quanta for assignment in assignments}) == len(assignments)
    assert min(assignment.weight for assignment in assignments) > 1.0 - 1e-10


def test_assignment_weights_expose_strongly_mixed_near_degenerate_states() -> None:
    q = np.linspace(-0.5, 0.5, 15)
    one_mode = 0.1 * q**2
    coupling = 0.02 * q[:, None] * q[None, :]
    dvr = product_dvr_2d(
        q,
        q,
        1.0,
        1.0,
        one_mode[:, None] + one_mode[None, :] + coupling,
        k_eigs=4,
    )

    assignments = assign_product_states_2d(dvr, 1.0, 1.0, one_mode, one_mode)

    first_excited = assignments[1]
    assert first_excited.weight < 0.8
    assert first_excited.participation_ratio == pytest.approx(2.0, rel=0.05)
    assert first_excited.dominant_manifold_weight > 0.99
    assert len(first_excited.signature) == 2
    assert sum(weight for _label, _coefficient, weight in first_excited.top_components[:2]) > 0.99
