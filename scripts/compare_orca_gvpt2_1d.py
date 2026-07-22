#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _load_ref(ref_json: Path) -> dict:
    with ref_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _match_orca_mode_by_harmonic(ref_species: dict, *, target_w_harm_cm: float) -> dict:
    funds = ref_species.get("fundamentals_anharmonic", [])
    if not funds:
        raise RuntimeError("Reference species has no fundamentals_anharmonic entries")
    best = min(funds, key=lambda r: abs(float(r["w_harm_cm_1"]) - float(target_w_harm_cm)))
    return best


def _orca_overtone(ref_species: dict, *, mode_index: int) -> Optional[dict]:
    # ORCA stores overtones/combination bands by a list of mode indices; [i,i] is the first overtone of mode i.
    entries = ref_species.get("overtones_combinations_anharmonic", [])
    for r in entries:
        modes = r.get("modes")
        if modes == [int(mode_index), int(mode_index)]:
            return r
    return None


def _base_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    return env


def _run_1d(
    *,
    pipeline: Path,
    mmol: Path,
    bond: str,
    scan: str,
    npts: int,
    vmax: int,
    tight_scan: bool,
    tight_width: float,
    max_parallel: int,
    pes_workers: int,
    method: Optional[str],
    basis: Optional[str],
    scf_conv_tol: Optional[float],
    dft_grid_level: Optional[int],
    no_ri: bool,
) -> Tuple[int, str, str]:
    cmd: List[str] = [
        sys.executable,
        str(pipeline),
        "--mmol",
        str(mmol),
        "--task",
        "1d",
        "--bond",
        bond,
        "--scan",
        scan,
        "--npts",
        str(int(npts)),
        "--vmax",
        str(int(vmax)),
        "--max-parallel",
        str(int(max_parallel)),
        "--pes-workers",
        str(int(pes_workers)),
    ]
    if tight_scan:
        cmd += ["--tight-scan", "--tight-width", str(float(tight_width))]
    if method is not None:
        cmd += ["--method", method]
    if basis is not None:
        cmd += ["--basis", basis]
    if scf_conv_tol is not None:
        cmd += ["--scf-conv-tol", str(float(scf_conv_tol))]
    if dft_grid_level is not None:
        cmd += ["--dft-grid-level", str(int(dft_grid_level))]
    if no_ri:
        cmd += ["--no-ri"]

    env = _base_env()
    env["OMP_NUM_THREADS"] = str(int(max_parallel))
    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    return int(p.returncode), p.stdout, p.stderr


def _parse_selected_harmonic(stdout: str) -> float:
    # run_1d prints:
    #   Harmonic ν (mode kbest) = XXXX.X cm^-1; DVR v=1 = YYYY.Y cm^-1; μ_eff = ...
    for line in stdout.splitlines():
        if line.startswith("Harmonic ν (mode") and "cm^-1" in line:
            # Extract first float after '='.
            try:
                rhs = line.split("=", 1)[1]
                val = float(rhs.strip().split()[0])
                return val
            except Exception as exc:
                raise RuntimeError(f"Failed to parse harmonic ν from line: {line!r}") from exc
    raise RuntimeError("Did not find 'Harmonic ν (mode ... ) =' line in 1D stdout; use --scan normal.")


def _parse_dvr_table(stdout: str) -> Dict[int, float]:
    # Table:
    # v  nu/cm^-1   mu_tr (arb)   ∫σ dω (m^2 s)
    out: Dict[int, float] = {}
    in_table = False
    for line in stdout.splitlines():
        if line.strip().startswith("v  nu/cm^-1"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip():
            break
        if line.startswith("Runtime:"):
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            v = int(parts[0])
            nu = float(parts[1])
        except Exception:
            continue
        out[v] = nu
    if not out:
        raise RuntimeError("Failed to parse any (v, nu) entries from the 1D table")
    return out


def _rms(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compare a 1D DVR scan (normal-mode path) against ORCA GVPT2 fundamentals/overtones.\n"
            "Mapping is approximate: we match the ORCA mode by closest harmonic frequency."
        )
    )
    ap.add_argument("--species", required=True, help="Species key in reference/all_gvpt2.json (e.g. H2O, HDO)")
    ap.add_argument("--bond", default="O0-H1")
    ap.add_argument(
        "--level",
        choices=["default", "orca_like"],
        default="orca_like",
        help="Preset for method/basis/SCF/grid (default orca_like)",
    )
    ap.add_argument("--ref-json", type=Path, default=Path("reference/all_gvpt2.json"))
    ap.add_argument("--pipeline", type=Path, default=Path("pyscf_pme_pipeline.py"))
    ap.add_argument(
        "--mmol",
        type=Path,
        default=None,
        help="Geometry file (default: geom/<species>.pyscf_opt.mmol if present, else geom/<species>.mmol)",
    )
    ap.add_argument("--npts", type=int, default=81)
    ap.add_argument("--vmax", type=int, default=2, help="Compare v=1 and (if available) v=2 (default 2)")
    ap.add_argument(
        "--tight-width",
        type=float,
        default=1.00,
        help=(
            "Full scan width (Å) used with --tight-scan. "
            "For high-frequency stretches, too-small windows artificially push DVR levels up (box confinement)."
        ),
    )
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--pes-workers", type=int, default=1)
    ap.add_argument("--method", default=None, help="Override preset method (e.g. wb97x)")
    ap.add_argument("--basis", default=None, help="Override preset basis (e.g. aug-cc-pVTZ)")
    ap.add_argument("--no-ri", action="store_true")
    ap.add_argument("--fail-fund", type=float, default=None, help="If set, fail if |Δν_fund| exceeds this (cm^-1)")
    ap.add_argument("--fail-ov2", type=float, default=None, help="If set, fail if |Δν_overtone(v=2)| exceeds this (cm^-1)")
    args = ap.parse_args(argv)

    try:
        import pyscf  # noqa: F401
    except Exception:
        sys.stderr.write(
            "ERROR: PySCF is not importable. Install the validation dependencies "
            "with 'uv sync --extra validation' or activate an environment containing PySCF.\n"
        )
        return 2

    ref = _load_ref(args.ref_json)
    if args.species not in ref:
        sys.stderr.write(f"ERROR: species '{args.species}' not found in {args.ref_json}\n")
        return 2

    if args.mmol is not None:
        mmol = Path(args.mmol)
    else:
        opt = Path("geom") / f"{args.species}.pyscf_opt.mmol"
        raw = Path("geom") / f"{args.species}.mmol"
        if opt.exists():
            mmol = opt
        elif raw.exists():
            sys.stderr.write(f"[WARN] Missing PySCF-optimized geometry {opt}; falling back to {raw}\n")
            mmol = raw
        else:
            mmol = raw
    if not mmol.exists():
        sys.stderr.write(f"ERROR: geometry file not found: {mmol}\n")
        return 2

    # Presets (overrideable by explicit --method/--basis).
    if args.level == "orca_like":
        method = "wb97x"
        basis = "aug-cc-pVTZ"
        scf_conv_tol = 1e-10  # "VeryTight-ish" in PySCF terms
        dft_grid_level = 3    # ORCA DefGrid3 analogue
    else:
        method = None
        basis = None
        scf_conv_tol = None
        dft_grid_level = None
    if args.method is not None:
        method = args.method
    if args.basis is not None:
        basis = args.basis

    rc, out, err = _run_1d(
        pipeline=args.pipeline,
        mmol=mmol,
        bond=args.bond,
        scan="normal",
        npts=int(args.npts),
        vmax=int(args.vmax),
        tight_scan=True,
        tight_width=float(args.tight_width),
        max_parallel=int(args.max_parallel),
        pes_workers=int(args.pes_workers),
        method=method,
        basis=basis,
        scf_conv_tol=scf_conv_tol,
        dft_grid_level=dft_grid_level,
        no_ri=bool(args.no_ri),
    )
    if rc != 0:
        sys.stderr.write("ERROR: PySCF 1D run failed.\n")
        sys.stderr.write(err)
        sys.stderr.write(out)
        return rc

    w_harm = _parse_selected_harmonic(out)
    dvr = _parse_dvr_table(out)

    ref_sp = ref[args.species]
    fund = _match_orca_mode_by_harmonic(ref_sp, target_w_harm_cm=w_harm)
    mode_idx = int(fund["mode_index"])
    orca_w = float(fund["w_harm_cm_1"])
    orca_fund = float(fund["v_fund_cm_1"])

    d_fund = float(dvr.get(1))
    diff_fund = d_fund - orca_fund
    abs_fund = abs(diff_fund)

    print(f"Species: {args.species} | bond={args.bond} | scan=normal (tight width {float(args.tight_width):.2f} Å)")
    print(f"PySCF selected harmonic: {w_harm:.2f} cm^-1")
    print(f"Matched ORCA GVPT2 mode_index={mode_idx} by closest w_harm: ORCA w_harm={orca_w:.2f} cm^-1")
    print(f"Fundamental: PySCF DVR v=1 {d_fund:.2f} vs ORCA GVPT2 {orca_fund:.2f}  (Δ={diff_fund:+.2f} cm^-1)")

    failures: List[str] = []
    if args.fail_fund is not None and abs_fund > float(args.fail_fund):
        failures.append(f"|Δfund|={abs_fund:.2f} > {float(args.fail_fund):.2f} cm^-1")

    if int(args.vmax) >= 2 and 2 in dvr:
        ov = _orca_overtone(ref_sp, mode_index=mode_idx)
        if ov is None:
            print("Overtone(v=2): ORCA [i,i] entry not present; skipped")
        else:
            orca_ov2 = float(ov["frequency_cm_1"])
            d_ov2 = float(dvr[2])
            diff_ov2 = d_ov2 - orca_ov2
            abs_ov2 = abs(diff_ov2)
            print(f"Overtone(v=2): PySCF DVR {d_ov2:.2f} vs ORCA {orca_ov2:.2f}  (Δ={diff_ov2:+.2f} cm^-1)")
            if args.fail_ov2 is not None and abs_ov2 > float(args.fail_ov2):
                failures.append(f"|Δov2|={abs_ov2:.2f} > {float(args.fail_ov2):.2f} cm^-1")

    if failures:
        sys.stderr.write("FAIL: " + "; ".join(failures) + "\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
