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
    out = _validated_coordinates(coords)
    i, j = int(bond.i), int(bond.j)
    _validate_bond(bond, out.shape[0])
    length = _validated_length(new_len_A)
    vec = out[j] - out[i]
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        raise ValueError(f"Zero-length bond {bond}")
    out[j] = out[i] + vec / norm * length
    return out


def stretch_two_bonds(
    coords: np.ndarray,
    bond1: Bond,
    bond2: Bond,
    new_len1_A: float,
    new_len2_A: float,
) -> np.ndarray:
    """Set two bond lengths without one displacement invalidating the other.

    Bonds that share one atom are stretched simultaneously about that common
    atom, so the result is independent of bond orientation. For disjoint bonds,
    each bond retains the usual convention that ``i`` is fixed and ``j`` moves.
    Duplicate or reversed-duplicate bonds do not define two independent
    coordinates and are rejected.
    """

    source = _validated_coordinates(coords)
    _validate_bond(bond1, source.shape[0])
    _validate_bond(bond2, source.shape[0])
    length1 = _validated_length(new_len1_A)
    length2 = _validated_length(new_len2_A)

    atoms1 = {int(bond1.i), int(bond1.j)}
    atoms2 = {int(bond2.i), int(bond2.j)}
    shared = atoms1 & atoms2
    if len(shared) == 2:
        raise ValueError("Two-dimensional scans require two distinct bonds")

    if len(shared) == 1:
        anchor = shared.pop()
        endpoint1 = (atoms1 - {anchor}).pop()
        endpoint2 = (atoms2 - {anchor}).pop()
        out = source.copy()
        out[endpoint1] = _point_at_distance(source, anchor, endpoint1, length1)
        out[endpoint2] = _point_at_distance(source, anchor, endpoint2, length2)
    else:
        # Disjoint bonds cannot interfere. Their explicit i -> j orientation is
        # the anchor policy, matching the public one-bond coordinate semantics.
        out = stretch_along_bond(source, bond1, length1)
        out = stretch_along_bond(out, bond2, length2)

    achieved1 = float(np.linalg.norm(out[bond1.j] - out[bond1.i]))
    achieved2 = float(np.linalg.norm(out[bond2.j] - out[bond2.i]))
    if not np.isclose(achieved1, length1, rtol=0.0, atol=1e-12) or not np.isclose(
        achieved2,
        length2,
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Two-bond stretch failed to realize the requested coordinate values")
    return out


def _validated_coordinates(coords: np.ndarray) -> np.ndarray:
    out = np.array(coords, dtype=float, copy=True)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError("coordinates must have shape (n_atoms, 3)")
    if not np.all(np.isfinite(out)):
        raise ValueError("coordinates must contain only finite values")
    return out


def _validate_bond(bond: Bond, n_atoms: int) -> None:
    i, j = int(bond.i), int(bond.j)
    if i < 0 or j < 0 or i >= n_atoms or j >= n_atoms:
        raise IndexError(f"Bond {bond} is out of range for {n_atoms} atoms")
    if i == j:
        raise ValueError("A bond must reference two distinct atoms")


def _validated_length(new_len_A: float) -> float:
    length = float(new_len_A)
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("Bond lengths must be positive and finite")
    return length


def _point_at_distance(
    coords: np.ndarray,
    anchor: int,
    endpoint: int,
    distance_A: float,
) -> np.ndarray:
    vector = coords[endpoint] - coords[anchor]
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError(f"Zero-length bond Bond(i={anchor}, j={endpoint})")
    return coords[anchor] + vector / norm * distance_A
