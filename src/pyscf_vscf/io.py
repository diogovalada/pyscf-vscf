"""Geometry and cache I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Protocol, Sequence

import numpy as np

from .constants import ANG_TO_BOHR
from .molecule import Molecule


class MolLike(Protocol):
    symbols: Sequence[str]
    coords: Sequence[Sequence[float]]
    label: str


_ANGSTROM_UNITS = {"A", "AA", "ANG", "ANGSTROM", "ANGSTROMS"}
_BOHR_UNITS = {"AU", "BOHR", "BOHRS"}


def _parse_iso_number(token: str, *, path: Path, line_number: int) -> int | None:
    key, sep, value = token.partition("=")
    if not sep or key.upper() != "ISO":
        return None
    value = value.strip().rstrip(",;")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid ISO value {token!r} in {path} on line {line_number}") from exc


def _atom_symbol_from_mmol(sym: str, attrs: Sequence[str], *, path: Path, line_number: int) -> str:
    if sym.upper() == "D":
        return "D"
    if sym.upper() != "H":
        return sym
    for attr in attrs:
        if _parse_iso_number(attr, path=path, line_number=line_number) == 2:
            return "D"
    return sym


def read_midas_mmol(path: Path) -> Molecule:
    path = Path(path)
    lines = path.read_text().splitlines()
    xyz_idx = next(
        (
            i
            for i, line in enumerate(lines)
            if line.strip().upper().startswith("#1") and "XYZ" in line.upper()
        ),
        None,
    )
    if xyz_idx is None:
        raise ValueError(f"No #1 Xyz block found in {path}")

    header_idx = xyz_idx + 1
    while header_idx < len(lines) and not lines[header_idx].strip():
        header_idx += 1
    if header_idx >= len(lines):
        raise ValueError(f"Missing atom-count header after #1 Xyz block in {path}")

    header = lines[header_idx].strip().split()
    if len(header) < 2:
        raise ValueError(f"Invalid MMOL XYZ header in {path} on line {header_idx + 1}")
    try:
        expected_atoms = int(header[0])
    except ValueError as exc:
        raise ValueError(f"Invalid atom count in {path} on line {header_idx + 1}") from exc
    unit = header[1].upper()
    if unit in _ANGSTROM_UNITS:
        scale = 1.0
    elif unit in _BOHR_UNITS:
        scale = 1.0 / ANG_TO_BOHR
    else:
        raise ValueError(f"Unsupported MMOL coordinate unit {header[1]!r} in {path}")

    symbols: list[str] = []
    coords: list[list[float]] = []
    atom_idx = header_idx + 2
    while len(symbols) < expected_atoms and atom_idx < len(lines):
        line_number = atom_idx + 1
        s = lines[atom_idx].strip()
        atom_idx += 1
        if not s or s.startswith("#"):
            continue
        parts = s.split()
        if len(parts) < 4:
            continue
        sym = _atom_symbol_from_mmol(parts[0], parts[4:], path=path, line_number=line_number)
        try:
            xyz = [float(x) * scale for x in parts[1:4]]
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates in {path} on line {line_number}") from exc
        symbols.append(sym)
        coords.append(xyz)
    if len(symbols) != expected_atoms:
        raise ValueError(f"Expected {expected_atoms} atoms in {path}, got {len(symbols)}")
    return Molecule(symbols=symbols, coords=np.array(coords, dtype=float), label=path.stem)


def read_xyz(path: Path) -> Molecule:
    path = Path(path)
    lines = path.read_text().splitlines()
    if not lines:
        raise ValueError(f"Empty XYZ file: {path}")
    try:
        expected_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count in {path} on line 1") from exc
    if expected_atoms < 0:
        raise ValueError(f"Invalid XYZ atom count in {path} on line 1")
    if len(lines) < expected_atoms + 2:
        raise ValueError(
            f"Expected {expected_atoms} atoms in {path}, got {max(0, len(lines) - 2)}"
        )

    symbols: list[str] = []
    coords: list[list[float]] = []
    for offset, line in enumerate(lines[2 : expected_atoms + 2], start=3):
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Invalid XYZ atom line in {path} on line {offset}")
        try:
            xyz = [float(x) for x in parts[1:4]]
        except ValueError as exc:
            raise ValueError(f"Invalid coordinates in {path} on line {offset}") from exc
        symbols.append(parts[0])
        coords.append(xyz)
    return Molecule(symbols=symbols, coords=np.array(coords, dtype=float), label=path.stem)


def write_xyz(path: Path, mol: MolLike, comment: Optional[str] = None) -> None:
    path = Path(path)
    n = len(mol.symbols)
    lines = [str(n), (comment if comment is not None else mol.label)]
    for sym, xyz in zip(mol.symbols, mol.coords):
        sym_out = "D" if str(sym).upper() == "D" else str(sym)
        x, y, z = map(float, xyz)
        lines.append(f"{sym_out} {x: .10f} {y: .10f} {z: .10f}")
    path.write_text("\n".join(lines) + "\n")


def write_midas_mmol(path: Path, mol: MolLike, title: Optional[str] = None) -> None:
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


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def dump_grid_npz(path: Path, *, meta: dict, arrays: dict[str, np.ndarray]) -> None:
    from .cache import array_sha256

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_arrays = {name: np.asarray(value) for name, value in arrays.items()}
    stored_meta = dict(meta)
    stored_meta["array_sha256"] = {
        name: array_sha256(value) for name, value in sorted(normalized_arrays.items())
    }
    payload = {
        "meta_json": np.array(json.dumps(stored_meta, sort_keys=True, default=_json_default))
    }
    payload.update(normalized_arrays)
    np.savez_compressed(path, **payload)


def load_grid_npz(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    from .cache import array_sha256

    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        if "meta_json" not in data.files:
            raise ValueError(f"Grid cache '{path}' missing key 'meta_json'")
        meta = json.loads(str(data["meta_json"].tolist()))
        arrays = {k: data[k] for k in data.files if k != "meta_json"}
    expected_hashes = meta.get("array_sha256")
    if meta.get("grid_cache_version") in {2, 3} and expected_hashes is None:
        raise ValueError(
            f"Schema-v{meta['grid_cache_version']} grid cache '{path}' is missing its "
            "array checksum manifest"
        )
    if expected_hashes is not None:
        if not isinstance(expected_hashes, dict):
            raise ValueError(f"Grid cache '{path}' has an invalid array checksum manifest")
        actual_names = set(arrays)
        if set(expected_hashes) != actual_names:
            raise ValueError(f"Grid cache '{path}' array manifest does not match stored arrays")
        for name, expected in expected_hashes.items():
            actual = array_sha256(arrays[name])
            if actual != expected:
                raise ValueError(f"Grid cache '{path}' checksum mismatch for array '{name}'")
    return meta, arrays
