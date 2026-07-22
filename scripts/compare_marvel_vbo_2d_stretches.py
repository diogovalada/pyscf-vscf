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


def _load_targets(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _base_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    return env


def _default_geom(species: str) -> Path:
    opt = Path("geom") / f"{species}.pyscf_opt.mmol"
    raw = Path("geom") / f"{species}.mmol"
    if opt.exists():
        return opt
    if raw.exists():
        sys.stderr.write(f"[WARN] Missing PySCF-optimized geometry {opt}; falling back to {raw}\n")
        return raw
    return raw


def _run_2d(
    *,
    pipeline: Path,
    mmol: Path,
    bond1: str,
    bond2: str,
    rmin: float,
    rmax: float,
    npts: int,
    nmax: int,
    level: str,
    keo: str,
    max_parallel: int,
    pes_workers: int,
    method: Optional[str],
    basis: Optional[str],
    no_ri: bool,
) -> Tuple[int, str, str]:
    cmd: List[str] = [
        sys.executable,
        str(pipeline),
        "--mmol",
        str(mmol),
        "--task",
        "2d",
        "--bond",
        bond1,
        "--bond2",
        bond2,
        "--rmin",
        str(float(rmin)),
        "--rmax",
        str(float(rmax)),
        "--npts",
        str(int(npts)),
        "--vmax",
        str(int(nmax)),
        "--keo",
        str(keo),
        "--max-parallel",
        str(int(max_parallel)),
        "--pes-workers",
        str(int(pes_workers)),
    ]

    if level == "orca_like":
        cmd += ["--method", "wb97x", "--basis", "aug-cc-pVTZ", "--scf-conv-tol", "1e-10", "--dft-grid-level", "3"]
    if method is not None:
        cmd += ["--method", method]
    if basis is not None:
        cmd += ["--basis", basis]
    if no_ri:
        cmd += ["--no-ri"]

    env = _base_env()
    env["OMP_NUM_THREADS"] = str(int(max_parallel))
    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    return int(p.returncode), p.stdout, p.stderr


def _parse_2d_table(stdout: str) -> List[Tuple[int, float]]:
    rows: List[Tuple[int, float]] = []
    in_table = False
    for line in stdout.splitlines():
        if line.strip().startswith("n  nu/cm^-1"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.strip() or line.startswith("Runtime:"):
            break
        parts = line.split()
        if len(parts) < 2:
            continue
        try:
            n = int(parts[0])
            nu = float(parts[1])
        except Exception:
            continue
        rows.append((n, nu))
    if not rows:
        raise RuntimeError("Failed to parse any 2D transitions from stdout")
    return rows


def _monotone_optimal_match(
    targets: List[Tuple[str, float]], candidates: List[Tuple[int, float]]
) -> List[Tuple[str, float, int, float, float]]:
    """
    Optimal injective matching under the assumption that both targets and candidates are ordered by frequency and
    the best assignment is order-preserving (monotone).

    This avoids factorial blowups vs brute-force permutations when targets>~6.
    """
    t = list(targets)
    c = list(candidates)
    if not t:
        return []
    if len(c) < len(t):
        # Greedy best-effort: match in order to the nearest remaining candidate.
        remaining = list(c)
        out: List[Tuple[str, float, int, float, float]] = []
        for label, nu_ref in t:
            if not remaining:
                out.append((label, float(nu_ref), -1, float("nan"), float("nan")))
                continue
            best_i = min(range(len(remaining)), key=lambda k: abs(remaining[k][1] - nu_ref))
            n, nu = remaining.pop(best_i)
            out.append((label, float(nu_ref), int(n), float(nu), float(nu) - float(nu_ref)))
        return out

    # DP: dp[i][j] = min cost to match targets[0..i] with candidates[0..j] using candidate[j] for target[i].
    nt = len(t)
    nc = len(c)
    inf = float("inf")
    dp = [[inf] * nc for _ in range(nt)]
    prev = [[-1] * nc for _ in range(nt)]

    for j in range(nc):
        d = float(c[j][1]) - float(t[0][1])
        dp[0][j] = d * d

    for i in range(1, nt):
        best_val = inf
        best_k = -1
        for j in range(nc):
            # Update prefix minimum from dp[i-1][0..j-1]
            if j - 1 >= 0 and dp[i - 1][j - 1] < best_val:
                best_val = dp[i - 1][j - 1]
                best_k = j - 1
            if best_k < 0:
                continue
            d = float(c[j][1]) - float(t[i][1])
            dp[i][j] = best_val + d * d
            prev[i][j] = best_k

    # Choose the best ending candidate for the last target.
    j_best = min(range(nc), key=lambda j: dp[nt - 1][j])
    out_rev: List[Tuple[str, float, int, float, float]] = []
    j = j_best
    for i in range(nt - 1, -1, -1):
        label, nu_ref = t[i]
        n, nu = c[j]
        d = float(nu) - float(nu_ref)
        out_rev.append((label, float(nu_ref), int(n), float(nu), float(d)))
        j = prev[i][j]
        if i > 0 and j < 0:
            break
    out = list(reversed(out_rev))
    if len(out) != nt:
        # Should be rare; fall back to greedy.
        remaining = list(c)
        out2: List[Tuple[str, float, int, float, float]] = []
        for label, nu_ref in t:
            best_i = min(range(len(remaining)), key=lambda k: abs(remaining[k][1] - nu_ref))
            n, nu = remaining.pop(best_i)
            out2.append((label, float(nu_ref), int(n), float(nu), float(nu) - float(nu_ref)))
        return out2
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compare 2D local-stretch DVR transition energies to curated MARVEL/IUPAC band origins (J=0 term values).\n"
            "This is intended as a real-molecule high-overtone regression check."
        )
    )
    ap.add_argument("--species", required=True, choices=["H2O", "HDO", "D2O"])
    ap.add_argument("--targets-json", type=Path, default=Path("reference/marvel_iupac/vbo_stretch_targets.json"))
    ap.add_argument("--pipeline", type=Path, default=Path("pyscf_pme_pipeline.py"))
    ap.add_argument("--mmol", type=Path, default=None, help="Geometry file (default: prefer geom/<species>.pyscf_opt.mmol)")
    ap.add_argument("--bond1", default="O0-H1")
    ap.add_argument("--bond2", default="O0-H2")
    ap.add_argument("--level", choices=["default", "orca_like"], default="orca_like")
    ap.add_argument("--method", default=None, help="Override preset method")
    ap.add_argument("--basis", default=None, help="Override preset basis")
    ap.add_argument("--no-ri", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--pes-workers", type=int, default=8)
    ap.add_argument("--keo", choices=["reduced", "gmatrix"], default="gmatrix", help="2D kinetic energy operator (default gmatrix)")
    ap.add_argument("--rmin", type=float, default=0.70)
    ap.add_argument("--rmax", type=float, default=1.70)
    ap.add_argument("--npts", type=int, default=31)
    ap.add_argument("--nmax", type=int, default=30, help="Number of 2D eigenstates/transitions to report/compare (default 30)")
    ap.add_argument("--fail-rms", type=float, default=None, help="If set, fail if RMS(|Δ|) exceeds this (cm^-1)")
    ap.add_argument("--fail-max", type=float, default=None, help="If set, fail if max(|Δ|) exceeds this (cm^-1)")
    args = ap.parse_args(argv)

    try:
        import pyscf  # noqa: F401
    except Exception:
        sys.stderr.write(
            "ERROR: PySCF is not importable. Install the validation dependencies "
            "with 'uv sync --extra validation' or activate an environment containing PySCF.\n"
        )
        return 2

    db = _load_targets(args.targets_json)
    all_targets = [t for t in db.get("targets", []) if t.get("species") == args.species]
    if not all_targets:
        sys.stderr.write(f"ERROR: no targets found for species {args.species} in {args.targets_json}\n")
        return 2

    targets: List[Tuple[str, float]] = []
    for t in all_targets:
        label = str(t.get("label", ""))
        nu = float(t["band_origin_cm1"])
        targets.append((label, nu))
    targets_sorted = sorted(targets, key=lambda x: x[1])

    mmol = Path(args.mmol) if args.mmol is not None else _default_geom(args.species)
    if not mmol.exists():
        sys.stderr.write(f"ERROR: geometry file not found: {mmol}\n")
        return 2

    rc, out, err = _run_2d(
        pipeline=args.pipeline,
        mmol=mmol,
        bond1=str(args.bond1),
        bond2=str(args.bond2),
        rmin=float(args.rmin),
        rmax=float(args.rmax),
        npts=int(args.npts),
        nmax=int(args.nmax),
        level=str(args.level),
        keo=str(args.keo),
        max_parallel=int(args.max_parallel),
        pes_workers=int(args.pes_workers),
        method=args.method,
        basis=args.basis,
        no_ri=bool(args.no_ri),
    )
    if rc != 0:
        sys.stderr.write("ERROR: PySCF 2D run failed.\n")
        sys.stderr.write(err)
        sys.stderr.write(out)
        return rc

    cand = _parse_2d_table(out)
    cand = [(n, nu) for (n, nu) in cand if nu > 1e-6]
    if not cand:
        sys.stderr.write("ERROR: parsed no positive-frequency transitions\n")
        return 2

    cand_sorted = sorted(cand, key=lambda x: x[1])
    matched = _monotone_optimal_match(targets_sorted, cand_sorted)

    diffs = [abs(d) for (_lab, _ref, n, _nu, d) in matched if n >= 0 and math.isfinite(d)]
    rms = math.sqrt(sum(x * x for x in diffs) / len(diffs)) if diffs else float("nan")
    mx = max(diffs) if diffs else float("nan")

    print(f"Species: {args.species} | geom={mmol}")
    print(f"2D scan: bonds {args.bond1} & {args.bond2} | window [{args.rmin:.2f},{args.rmax:.2f}] Å | npts={int(args.npts)} | nmax={int(args.nmax)}")
    print(f"Targets: {len(targets_sorted)} (MARVEL/IUPAC VBOs) | RMS(|Δ|)={rms:.2f} cm^-1 | max(|Δ|)={mx:.2f} cm^-1")
    for label, nu_ref, n, nu, d in matched:
        if n < 0:
            print(f"  {label:14s} MARVEL {nu_ref:10.3f}  PySCF <missing>     Δ  <missing>")
        else:
            print(f"  {label:14s} MARVEL {nu_ref:10.3f}  PySCF n={n:2d} {nu:10.3f}  Δ {d:+10.3f}")

    failures: List[str] = []
    if args.fail_rms is not None and rms > float(args.fail_rms):
        failures.append(f"RMS(|Δ|) {rms:.2f} > {float(args.fail_rms):.2f} cm^-1")
    if args.fail_max is not None and mx > float(args.fail_max):
        failures.append(f"max(|Δ|) {mx:.2f} > {float(args.fail_max):.2f} cm^-1")
    if failures:
        sys.stderr.write("FAIL: " + "; ".join(failures) + "\n")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
