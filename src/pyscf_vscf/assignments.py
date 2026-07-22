"""Wavefunction-overlap state assignment for product-grid DVR solutions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from .dvr import DVR2D, sinc_dvr_1d


@dataclass(frozen=True)
class ProductStateAssignment:
    """One-to-one product-state label and its overlap diagnostics."""

    state_index: int
    quanta: tuple[int, int]
    weight: float
    signature: tuple[tuple[tuple[int, int], int], ...]
    participation_ratio: float
    dominant_manifold_weight: float
    top_components: tuple[tuple[tuple[int, int], float, float], ...]


def assign_product_states_2d(
    dvr: DVR2D,
    mu1_amu: float,
    mu2_amu: float,
    reference_potential1_Eh: np.ndarray,
    reference_potential2_Eh: np.ndarray,
    *,
    top_n: int = 5,
) -> tuple[ProductStateAssignment, ...]:
    """Assign coupled states by overlaps with a separable 1D DVR basis.

    A Hungarian assignment gives unique ``(v1, v2)`` labels.  The reported
    weight and leading components make mixed or unstable assignments visible.
    """

    if int(top_n) < 1:
        raise ValueError("top_n must be at least 1")
    one = sinc_dvr_1d(dvr.R1, mu1_amu, reference_potential1_Eh)
    two = sinc_dvr_1d(dvr.R2, mu2_amu, reference_potential2_Eh)
    n_states = dvr.evecs.shape[1]
    one_basis = _phase_canonical_columns(one.evecs)
    two_basis = _phase_canonical_columns(two.evecs)
    n_products = one_basis.shape[1] * two_basis.shape[1]
    coefficients_by_state = np.empty((n_products, n_states))

    for state in range(n_states):
        wavefunction = dvr.evecs[:, state].reshape(dvr.R1.size, dvr.R2.size)
        coefficients = (one_basis.T @ wavefunction @ two_basis).ravel()
        coefficient_magnitudes = np.abs(coefficients)
        phase_candidates = np.flatnonzero(
            coefficient_magnitudes >= np.max(coefficient_magnitudes) * (1.0 - 1e-10)
        )
        phase_anchor = int(phase_candidates[0])
        if coefficients[phase_anchor] < 0.0:
            coefficients = -coefficients
        coefficients_by_state[:, state] = coefficients

    weights = np.square(np.abs(coefficients_by_state))

    product_rows, state_columns = linear_sum_assignment(-weights)
    assigned_rows = {int(state): int(row) for row, state in zip(product_rows, state_columns)}
    assignments = []
    for state in range(n_states):
        row = assigned_rows[state]
        quanta = np.unravel_index(row, (one.evecs.shape[1], two.evecs.shape[1]))
        order = np.argsort(weights[:, state])[::-1][: int(top_n)]
        maximum_weight = float(weights[order[0], state])
        signature_threshold = max(0.05, 0.1 * maximum_weight)
        signature_rows = [
            int(component)
            for component in np.argsort(weights[:, state])[::-1]
            if weights[component, state] >= signature_threshold
        ]
        signature = tuple(
            sorted(
                (
                    tuple(
                        int(value)
                        for value in np.unravel_index(
                            component,
                            (one_basis.shape[1], two_basis.shape[1]),
                        )
                    ),
                    1 if coefficients_by_state[component, state] >= 0.0 else -1,
                )
                for component in signature_rows
            )
        )
        components = tuple(
            (
                tuple(
                    int(value)
                    for value in np.unravel_index(
                        int(component),
                        (one.evecs.shape[1], two.evecs.shape[1]),
                    )
                ),
                float(coefficients_by_state[component, state]),
                float(weights[component, state]),
            )
            for component in order
        )
        assignments.append(
            ProductStateAssignment(
                state_index=state,
                quanta=(int(quanta[0]), int(quanta[1])),
                weight=float(weights[row, state]),
                signature=signature,
                participation_ratio=float(1.0 / np.sum(np.square(weights[:, state]))),
                dominant_manifold_weight=float(
                    sum(weights[component, state] for component in signature_rows)
                ),
                top_components=components,
            )
        )
    return tuple(assignments)


def _phase_canonical_columns(vectors: np.ndarray) -> np.ndarray:
    canonical = np.asarray(vectors, dtype=float).copy()
    for column in range(canonical.shape[1]):
        anchor = int(np.argmax(np.abs(canonical[:, column])))
        if canonical[anchor, column] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


__all__ = ["ProductStateAssignment", "assign_product_states_2d"]
