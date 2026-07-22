"""Coordinate helpers for local-mode scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Bond:
    i: int
    j: int

    @property
    def O(self) -> int:  # noqa: E743 - legacy O/H bond attribute spelling
        return self.i

    @property
    def H(self) -> int:
        return self.j


def parse_bond(s: str) -> Bond:
    spec = s.strip().upper().replace(" ", "")
    if "-" not in spec:
        raise ValueError(f"Bond specification must contain '-': {s!r}")
    left, right = spec.split("-", 1)
    if left.startswith("O") and right.startswith("H"):
        left = left[1:]
        right = right[1:]
    try:
        return Bond(int(left), int(right))
    except ValueError as exc:
        raise ValueError(
            f"Invalid bond specification {s!r}; expected 0-1 "
            "(O0-H1 is accepted as compatibility syntax)"
        ) from exc


def stretch_along_bond(coords: np.ndarray, bond: Bond, new_len_A: float) -> np.ndarray:
    out = np.array(coords, dtype=float, copy=True)
    i, j = int(bond.i), int(bond.j)
    vec = out[j] - out[i]
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"Zero-length bond {bond}")
    out[j] = out[i] + vec / norm * float(new_len_A)
    return out
