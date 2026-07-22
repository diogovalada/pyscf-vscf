from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, Sequence


class _MolLike(Protocol):
    symbols: Sequence[str]
    coords: Sequence[Sequence[float]]
    label: str


def write_xyz(path: Path, mol: _MolLike, comment: Optional[str] = None):
    path = Path(path)
    n = len(mol.symbols)
    lines = [str(n), (comment if comment is not None else mol.label)]
    for sym, xyz in zip(mol.symbols, mol.coords):
        sym_out = "D" if str(sym).upper() == "D" else str(sym)
        x, y, z = map(float, xyz)
        lines.append(f"{sym_out} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n")


def write_midas_mmol(path: Path, mol: _MolLike, title: Optional[str] = None):
    """
    Write a minimal MidasMolecule .mmol file in Angstrom units.
    Deuterium is encoded as 'H ... ISO=2' to preserve round-tripping with read_midas_mmol().
    """
    path = Path(path)
    n = len(mol.symbols)
    lines = [
        "#0 MidasMolecule",
        "",
        "#1 Xyz",
        f"{n} AA",
        (title if title is not None else mol.label),
    ]
    for sym, xyz in zip(mol.symbols, mol.coords):
        x, y, z = map(float, xyz)
        if str(sym).upper() == "D":
            lines.append(f"H   {x: .10f}  {y: .10f}  {z: .10f}  ISO=2")
        else:
            lines.append(f"{sym}   {x: .10f}  {y: .10f}  {z: .10f}")
    lines += ["", "#0 MidasMoleculeEnd", ""]
    path.write_text("\n".join(lines))
