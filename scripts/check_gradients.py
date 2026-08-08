#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Gradient stationarity check for saved geometries (strict guardrail).\n"
            "Runs a single SCF + nuclear gradient and errors if the geometry is not stationary."
        )
    )
    p.add_argument(
        "--mmol",
        action="append",
        default=[],
        help="Path to MIDAS MMOL geometry file. May be passed multiple times. Default: geom/*.pyscf_opt.mmol",
    )
    p.add_argument("--method", default="wb97x")
    p.add_argument("--basis", default="aug-cc-pVTZ")
    p.add_argument("--no-density-fit", action="store_true", help="Disable density fitting (RI)")
    p.add_argument("--scf-conv-tol", type=float, default=1e-10)
    p.add_argument("--dft-grid-level", type=int, default=3)

    p.add_argument(
        "--warn-max-grad",
        type=float,
        default=3e-5,
        help="Warn if max |grad component| (Eh/Bohr) exceeds this threshold",
    )
    p.add_argument(
        "--fail-max-grad",
        type=float,
        default=1e-4,
        help="Fail if max |grad component| (Eh/Bohr) exceeds this threshold",
    )
    p.add_argument("--no-fail", action="store_true", help="Never fail on gradients (still prints + warns)")

    return p.parse_args(argv)


def _expand_paths(paths: List[str]) -> List[Path]:
    if paths:
        return [Path(p) for p in paths]
    return [Path(p) for p in sorted(glob.glob("geom/*.pyscf_opt.mmol"))]


def _grad(pipe, mol, cfg) -> np.ndarray:
    pmol = mol.as_pyscf(cfg.basis)
    mf = pipe.make_mean_field(pmol, cfg)
    try:
        g = mf.nuc_grad_method().kernel()
    except AttributeError:
        g = mf.Gradients().kernel()
    return np.asarray(g, float)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    # Keep all chemistry logic centralized in the main pipeline module.
    try:
        import pyscf_pme_pipeline as pipe
    except Exception as exc:
        sys.stderr.write(f"ERROR: failed to import pyscf_pme_pipeline (activate the pyscf env?).\n{exc}\n")
        return 2

    paths = _expand_paths(list(args.mmol))
    if not paths:
        sys.stderr.write("ERROR: no geometries found (expected geom/*.pyscf_opt.mmol or pass --mmol).\n")
        return 2

    # Build ES settings with strict SCF, since this is a guardrail tool.
    cfg = pipe.ESSettings(
        method=str(args.method),
        basis=str(args.basis),
        use_density_fit=not bool(args.no_density_fit),
        strict=True,
        scf_conv_tol=float(args.scf_conv_tol),
        dft_grid_level=int(args.dft_grid_level),
    )

    warn_thr = float(args.warn_max_grad)
    fail_thr = float(args.fail_max_grad)
    rc = 0

    for path in paths:
        mol = pipe.read_midas_mmol(path)
        g = _grad(pipe, mol, cfg)
        g_flat = g.reshape(-1)
        max_comp = float(np.max(np.abs(g_flat))) if g_flat.size else 0.0
        rms_comp = float(np.sqrt(np.mean(g_flat * g_flat))) if g_flat.size else 0.0
        g_atom = g.reshape(-1, 3)
        atom_norms = np.linalg.norm(g_atom, axis=1)
        max_atom = float(np.max(atom_norms)) if atom_norms.size else 0.0
        rms_atom = float(np.sqrt(np.mean(atom_norms * atom_norms))) if atom_norms.size else 0.0

        print(f"{path}:")
        print(f"  max|g_comp|={max_comp:.3e}  rms|g_comp|={rms_comp:.3e}  max|g_atom|={max_atom:.3e}  rms|g_atom|={rms_atom:.3e}  (Eh/Bohr)")

        if max_comp > warn_thr:
            sys.stderr.write(
                f"[WARN] {path}: max|g_comp|={max_comp:.3e} exceeds warn threshold {warn_thr:.1e} Eh/Bohr\n"
            )
        if (not args.no_fail) and max_comp > fail_thr:
            sys.stderr.write(
                f"[ERROR] {path}: max|g_comp|={max_comp:.3e} exceeds fail threshold {fail_thr:.1e} Eh/Bohr\n"
            )
            rc = 1

    return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
