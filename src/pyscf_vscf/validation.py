"""Convergence and numerical error-budget reports for assigned spectra."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


StateLabel = tuple


@dataclass(frozen=True)
class StateErrorBudget:
    """Observed spread for one consistently assigned transition."""

    assignment: StateLabel
    runs: tuple[str, ...]
    frequencies_cm: tuple[float, ...]
    frequency_min_cm: float
    frequency_max_cm: float
    frequency_spread_cm: float
    intensities_m2_per_s: tuple[float, ...]
    intensity_min_m2_per_s: float
    intensity_max_m2_per_s: float
    intensity_relative_spread: float | None
    minimum_assignment_weight: float | None
    minimum_dominant_manifold_weight: float | None
    maximum_participation_ratio: float | None


@dataclass(frozen=True)
class ConvergenceReport:
    """Matched-state numerical spreads across named calculation variants."""

    run_names: tuple[str, ...]
    states: tuple[StateErrorBudget, ...]
    unmatched: Mapping[str, tuple[StateLabel, ...]]

    def as_dict(self) -> dict:
        return asdict(self)


def convergence_report(
    runs: Mapping[str, Sequence[Mapping]],
    *,
    intensity_key: str = "integrated_cross_section_isotropic_omega_m2_per_s",
) -> ConvergenceReport:
    """Build an error budget by matching explicit quantum assignments.

    2D records should contain a phase-canonical ``assignment_signature`` and
    fall back to ``assignment``. One-dimensional records may use ``v``.
    Frequency-nearest matching is deliberately unsupported.
    """

    if len(runs) < 2:
        raise ValueError("At least two named runs are required for a convergence report")
    indexed: dict[str, dict[StateLabel, Mapping]] = {}
    for run_name, records in runs.items():
        name = str(run_name)
        if not name:
            raise ValueError("Run names must be non-empty")
        by_state: dict[StateLabel, Mapping] = {}
        for record in records:
            label = _state_label(record)
            if label in by_state:
                raise ValueError(f"Run {name!r} contains duplicate assignment {label}")
            by_state[label] = record
        indexed[name] = by_state

    common = set.intersection(*(set(records) for records in indexed.values()))
    union = set.union(*(set(records) for records in indexed.values()))
    unmatched = {name: tuple(sorted(union - set(records))) for name, records in indexed.items()}
    budgets = []
    names = tuple(indexed)
    for label in sorted(common):
        records = [indexed[name][label] for name in names]
        frequencies = tuple(float(record["freq_cm"]) for record in records)
        intensities = tuple(float(record[intensity_key]) for record in records)
        _finite_values("frequency", frequencies)
        _finite_values("intensity", intensities)
        if any(frequency <= 0.0 for frequency in frequencies):
            raise ValueError("Transition frequencies must be positive")
        if any(intensity < 0.0 for intensity in intensities):
            raise ValueError("Integrated cross sections must be non-negative")
        intensity_max = max(intensities)
        intensity_min = min(intensities)
        relative_spread = None
        if intensity_max > 0.0:
            relative_spread = (intensity_max - intensity_min) / intensity_max
        weights = [record.get("assignment_weight") for record in records]
        minimum_weight = None
        if all(weight is not None for weight in weights):
            minimum_weight = min(float(weight) for weight in weights)
        manifold_weights = [
            record.get("assignment_dominant_manifold_weight") for record in records
        ]
        minimum_manifold_weight = None
        if all(weight is not None for weight in manifold_weights):
            minimum_manifold_weight = min(float(weight) for weight in manifold_weights)
        participation = [record.get("assignment_participation_ratio") for record in records]
        maximum_participation = None
        if all(value is not None for value in participation):
            maximum_participation = max(float(value) for value in participation)
        budgets.append(
            StateErrorBudget(
                assignment=label,
                runs=names,
                frequencies_cm=frequencies,
                frequency_min_cm=min(frequencies),
                frequency_max_cm=max(frequencies),
                frequency_spread_cm=max(frequencies) - min(frequencies),
                intensities_m2_per_s=intensities,
                intensity_min_m2_per_s=intensity_min,
                intensity_max_m2_per_s=intensity_max,
                intensity_relative_spread=relative_spread,
                minimum_assignment_weight=minimum_weight,
                minimum_dominant_manifold_weight=minimum_manifold_weight,
                maximum_participation_ratio=maximum_participation,
            )
        )
    return ConvergenceReport(run_names=names, states=tuple(budgets), unmatched=unmatched)


def _state_label(record: Mapping) -> StateLabel:
    if "assignment_signature" in record:
        values = tuple(
            (tuple(int(value) for value in quanta), int(sign))
            for quanta, sign in record["assignment_signature"]
        )
        if not values:
            raise ValueError("assignment_signature must not be empty")
        if any(
            sign not in {-1, 1} or any(value < 0 for value in quanta) for quanta, sign in values
        ):
            raise ValueError(f"Invalid assignment signature {values}")
    elif "assignment" in record:
        values = tuple(int(value) for value in record["assignment"])
    elif "v" in record:
        values = (int(record["v"]),)
    else:
        raise ValueError("Each record must contain an explicit 'assignment' or 'v' label")
    if "assignment_signature" not in record and any(value < 0 for value in values):
        raise ValueError(f"Invalid negative state assignment {values}")
    return values


def _finite_values(label: str, values: Sequence[float]) -> None:
    if not np.all(np.isfinite(np.asarray(values, dtype=float))):
        raise ValueError(f"Non-finite {label} values in convergence input")


__all__ = ["ConvergenceReport", "StateErrorBudget", "convergence_report"]
