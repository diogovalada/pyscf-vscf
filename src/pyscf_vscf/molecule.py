"""Molecule containers independent of any electronic-structure backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .constants import atomic_mass_amu


def _normalize_symbol(symbol: str) -> str:
    raw = str(symbol).strip()
    if not raw:
        raise ValueError("Atom symbols must be non-empty")
    if raw.upper() == "D":
        return "D"
    return raw[:1].upper() + raw[1:].lower()


@dataclass
class Molecule:
    symbols: list[str]
    coords: np.ndarray
    charge: int = 0
    spin: int = 0
    label: str = "mol"
    mass_overrides_amu: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.symbols = [_normalize_symbol(s) for s in self.symbols]
        self.coords = np.asarray(self.coords, dtype=float)
        if self.coords.ndim != 2 or self.coords.shape[1] != 3:
            raise ValueError("Molecule coordinates must have shape (n_atoms, 3)")
        if len(self.symbols) != self.coords.shape[0]:
            raise ValueError(
                f"Expected one symbol per coordinate row, got {len(self.symbols)} symbols "
                f"and {self.coords.shape[0]} coordinate rows"
            )
        self.charge = int(self.charge)
        self.spin = int(self.spin)
        self.label = str(self.label)
        if self.mass_overrides_amu is not None:
            masses = np.asarray(self.mass_overrides_amu, dtype=float)
            if masses.shape != (len(self.symbols),):
                raise ValueError("mass_overrides_amu must contain one value per atom")
            if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
                raise ValueError("mass_overrides_amu values must be finite and positive")
            self.mass_overrides_amu = masses.copy()

    @property
    def masses(self) -> np.ndarray:
        if self.mass_overrides_amu is not None:
            return self.mass_overrides_amu.copy()
        return np.array([atomic_mass_amu(symbol) for symbol in self.symbols], dtype=float)

    def analysis_masses(self) -> np.ndarray:
        """Return per-atom isotope masses in atomic mass units."""

        return self.masses

    @classmethod
    def from_arrays(
        cls,
        symbols: Sequence[str],
        coords: Sequence[Sequence[float]],
        *,
        charge: int = 0,
        spin: int = 0,
        label: str = "mol",
        masses_amu: Sequence[float] | None = None,
    ) -> "Molecule":
        return cls(
            symbols=list(symbols),
            coords=np.asarray(coords, dtype=float),
            charge=int(charge),
            spin=int(spin),
            label=str(label),
            mass_overrides_amu=(
                None if masses_amu is None else np.asarray(masses_amu, dtype=float)
            ),
        )
