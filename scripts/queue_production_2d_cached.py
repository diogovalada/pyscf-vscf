#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np


def _base_stem_from_opt_mmol(path: Path) -> str:
    stem = path.stem
    if stem.endswith(".pyscf_opt"):
        stem = stem[: -len(".pyscf_opt")]
    return stem


def _pick_h_index(symbols, coords, *, o_idx: int, h_candidates: Tuple[int, int], target_type: str, other_o_idx: int) -> int:
    target_type = target_type.upper()
    if target_type not in {"H", "D"}:
        raise ValueError(f"Unknown target_type '{target_type}' (expected H|D)")
    cands = [h for h in h_candidates if symbols[h].upper() == target_type]
    if not cands:
        raise ValueError(
            f"No atoms of type {target_type} among indices {h_candidates} (symbols={symbols[h_candidates[0]]},{symbols[h_candidates[1]]})"
        )

    def score(h: int) -> float:
        return float(np.linalg.norm(coords[h] - coords[other_o_idx]))

    return min(cands, key=score)


def bonds_for_system(mmol_path: Path) -> Tuple[str, str]:
    """
    Returns (bond1, bond2) strings for pyscf_pme_pipeline.py.

    - Monomers: O0-H1 and O0-H2 (H may be D in the underlying symbol list; index is what matters).
    - Dimers: choose one stretch on each monomer, guided by the filename tag:
        <donor>-<acceptor>-<HH|HD|DH|DD>.pyscf_opt.mmol
      tag[0] selects donor monomer stretch type (H or D),
      tag[1] selects acceptor monomer stretch type (H or D).
    """
    from pyscf_pme_pipeline import read_midas_mmol

    mol = read_midas_mmol(mmol_path)
    base = _base_stem_from_opt_mmol(mmol_path)

    if base in {"H2O", "HDO", "D2O"}:
        return "O0-H1", "O0-H2"

    parts = base.split("-")
    if len(parts) != 3:
        raise ValueError(f"Unrecognized system naming for '{base}' (expected monomer or <donor>-<acceptor>-<tag>)")
    tag = parts[2].upper()
    if len(tag) != 2 or any(c not in "HD" for c in tag):
        raise ValueError(f"Unrecognized dimer tag '{tag}' in '{base}' (expected HH/HD/DH/DD)")

    if len(mol.symbols) != 6:
        raise ValueError(f"Expected 6 atoms for dimer '{base}', got {len(mol.symbols)}")

    # By convention in this repo: donor monomer atoms 0-2, acceptor monomer atoms 3-5.
    donor_o = 0
    accept_o = 3
    donor_h = _pick_h_index(
        mol.symbols, mol.coords, o_idx=donor_o, h_candidates=(1, 2), target_type=tag[0], other_o_idx=accept_o
    )
    accept_h = _pick_h_index(
        mol.symbols, mol.coords, o_idx=accept_o, h_candidates=(4, 5), target_type=tag[1], other_o_idx=donor_o
    )
    return f"O0-H{donor_h}", f"O3-H{accept_h}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom-dir", type=Path, default=Path("geom"), help="Directory containing *.pyscf_opt.mmol (default geom/)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"), help="Directory for NPZ caches (default results/)")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--pes-workers", type=int, default=4)
    ap.add_argument("--npts", type=int, default=41)
    ap.add_argument("--rmin", type=float, default=0.70)
    ap.add_argument("--rmax", type=float, default=1.70)
    ap.add_argument("--nmax", type=int, default=80, help="Number of excited states to print for --task 2d (passed via --vmax)")
    ap.add_argument("--keo", choices=["reduced", "gmatrix"], default="gmatrix")
    ap.add_argument("--dispersion", choices=["d3", "d4", "none"], default="none")
    ap.add_argument("--intensity", choices=["axis", "vector", "both"], default="vector")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing NPZ caches (default: hard error)")
    ap.add_argument("--dry-run", action="store_true", help="Print commands and exit")
    args = ap.parse_args(argv)

    if args.max_parallel < 1:
        ap.error("--max-parallel must be >= 1")
    if args.pes_workers < 1:
        ap.error("--pes-workers must be >= 1")
    if args.npts < 5:
        ap.error("--npts too small")
    if args.nmax < 2:
        ap.error("--nmax must be >= 2")

    geom_dir = args.geom_dir
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    mmols = sorted(geom_dir.glob("*.pyscf_opt.mmol"))
    if not mmols:
        raise SystemExit(f"No '*.pyscf_opt.mmol' files found in {geom_dir}")

    # Deterministic run order: monomers first, then dimers by filename.
    monomers = [p for p in mmols if _base_stem_from_opt_mmol(p) in {"H2O", "HDO", "D2O"}]
    dimers = [p for p in mmols if p not in monomers]
    runlist = monomers + dimers

    for mmol in runlist:
        base = _base_stem_from_opt_mmol(mmol)
        b1, b2 = bonds_for_system(mmol)
        cache_name = f"prod_{base}_2d_npts{args.npts}_nmax{args.nmax}_r{args.rmin:.2f}-{args.rmax:.2f}_keo{args.keo}_disp{args.dispersion}_mu{args.intensity}.npz"
        cache_path = results_dir / cache_name
        if cache_path.exists() and not args.overwrite:
            raise SystemExit(f"Refusing to overwrite existing cache: {cache_path} (use --overwrite)")

        cmd = [
            sys.executable,
            "scripts/run_logged.py",
            "--tag",
            f"prod_cache_{base}",
            "--pythonnousersite",
            "--",
            sys.executable,
            "pyscf_pme_pipeline.py",
            "--mmol",
            str(mmol),
            "--task",
            "2d",
            "--bond",
            b1,
            "--bond2",
            b2,
            "--rmin",
            str(args.rmin),
            "--rmax",
            str(args.rmax),
            "--npts",
            str(args.npts),
            "--vmax",
            str(args.nmax),
            "--max-parallel",
            str(args.max_parallel),
            "--pes-workers",
            str(args.pes_workers),
            "--keo",
            args.keo,
            "--dispersion",
            args.dispersion,
            "--intensity",
            args.intensity,
            "--dump-grid",
            str(cache_path),
            "-v",
        ]

        print(f"[queue] {base}: {mmol} | bonds: {b1} & {b2} | cache: {cache_path}", flush=True)
        if args.dry_run:
            print(" ".join(cmd), flush=True)
            continue

        proc = subprocess.run(cmd)
        if proc.returncode != 0:
            raise SystemExit(f"[queue] FAILED rc={proc.returncode} for {base}")

    print("[queue] All production runs completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
