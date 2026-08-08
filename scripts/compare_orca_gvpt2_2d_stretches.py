#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def _load_ref(ref_json: Path) -> dict:
    with ref_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_mmol_symbols(path: Path) -> List[str]:
    """
    Minimal MMOL parser to get per-atom symbols including ISO=2 -> D.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip().upper().startswith("#1"):
        i += 1
    if i >= len(lines):
        raise ValueError(f"Failed to find #1 block in {path}")
    i += 1
    header = lines[i].strip().split()
    nat = int(header[0])
    i += 2  # header + title
    syms: List[str] = []
    while i < len(lines) and len(syms) < nat:
        line = lines[i].strip()
        i += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        sym = parts[0]
        if sym.upper() == "H":
            for p in parts[4:]:
                if p.upper().startswith("ISO=") and p.split("=", 1)[1].strip() == "2":
                    sym = "D"
                    break
        syms.append(sym)
    if len(syms) != nat:
        raise ValueError(f"Expected {nat} atoms in {path}, parsed {len(syms)}")
    return syms


def _parse_bond_h_index(bond: str) -> int:
    # Accept "O3-H5" with arbitrary whitespace.
    s = bond.strip()
    if "-" not in s:
        raise ValueError(f"Invalid bond spec '{bond}' (expected Oi-Hj)")
    left, right = s.split("-", 1)
    if not right.upper().startswith("H"):
        raise ValueError(f"Invalid bond spec '{bond}' (expected ...-Hj)")
    return int(right[1:])


def _pick_stretch_modes(ref_species: dict, *, kind: str) -> List[dict]:
    funds = list(ref_species.get("fundamentals_anharmonic", []))
    if len(funds) < 2:
        raise RuntimeError("Not enough fundamentals_anharmonic entries to pick two stretches")
    kind_lc = (kind or "auto").lower()
    if kind_lc not in {"auto", "oh", "od", "any"}:
        raise ValueError(f"Unknown stretch kind '{kind}'")

    # Harmonic windows to target likely stretches.
    # (Wide on purpose; ORCA harmonic w and GVPT2 fundamentals differ by ~100-200 cm^-1.)
    if kind_lc == "oh":
        wmin, wmax = 3200.0, 4200.0
    elif kind_lc == "od":
        wmin, wmax = 2200.0, 3200.0
    else:
        wmin, wmax = -1e9, 1e9

    cand = [r for r in funds if wmin <= float(r["w_harm_cm_1"]) <= wmax]
    if len(cand) < 2 and kind_lc in {"oh", "od"}:
        # If the window is too strict for a given species, fall back to all modes rather than
        # silently returning too few targets.
        sys.stderr.write(f"[WARN] Only {len(cand)} modes found in {kind_lc.upper()} window; falling back to top-2 overall.\n")
        cand = funds

    cand.sort(key=lambda r: float(r["w_harm_cm_1"]))
    return cand[-2:]


def _find_overtone(ref_species: dict, mode_index: int) -> Optional[dict]:
    for r in ref_species.get("overtones_combinations_anharmonic", []):
        if r.get("modes") == [int(mode_index), int(mode_index)]:
            return r
    return None


def _find_combination(ref_species: dict, i: int, j: int) -> Optional[dict]:
    pair1 = [int(i), int(j)]
    pair2 = [int(j), int(i)]
    for r in ref_species.get("overtones_combinations_anharmonic", []):
        modes = r.get("modes")
        if modes == pair1 or modes == pair2:
            return r
    return None


def _base_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    return env


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

    # Presets (overrideable).
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
    # Output table:
    # n  nu/cm^-1   mu_tr (arb)   ∫σ dω (m^2 s)
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


def _greedy_match(targets: List[Tuple[str, float]], candidates: List[Tuple[int, float]]) -> List[Tuple[str, float, int, float, float]]:
    remaining = list(candidates)
    out: List[Tuple[str, float, int, float, float]] = []
    for label, nu_ref in targets:
        if not remaining:
            out.append((label, nu_ref, -1, float("nan"), float("nan")))
            continue
        best_i = min(range(len(remaining)), key=lambda k: abs(remaining[k][1] - nu_ref))
        n, nu = remaining.pop(best_i)
        out.append((label, nu_ref, n, nu, nu - nu_ref))
    return out


def _optimal_match(
    targets: List[Tuple[str, float]], candidates: List[Tuple[int, float]]
) -> List[Tuple[str, float, int, float, float]]:
    """
    Brute-force optimal injective matching minimizing sum of squared frequency errors.
    Sizes here are small (targets<=5, candidates<=~12), so permutations are fine and robust.
    """
    if not targets:
        return []
    if len(candidates) < len(targets):
        # Fall back to greedy (will emit <missing> rows).
        return _greedy_match(targets, candidates)

    best_perm = None
    best_cost = float("inf")
    for perm in itertools.permutations(candidates, len(targets)):
        cost = 0.0
        for (_, nu_ref), (_, nu) in zip(targets, perm):
            d = float(nu) - float(nu_ref)
            cost += d * d
            if cost >= best_cost:
                break
        if cost < best_cost:
            best_cost = cost
            best_perm = perm

    assert best_perm is not None
    out: List[Tuple[str, float, int, float, float]] = []
    for (label, nu_ref), (n, nu) in zip(targets, best_perm):
        out.append((label, float(nu_ref), int(n), float(nu), float(nu) - float(nu_ref)))
    return out


def _default_geom(species: str) -> Path:
    opt = Path("geom") / f"{species}.pyscf_opt.mmol"
    raw = Path("geom") / f"{species}.mmol"
    if opt.exists():
        return opt
    if raw.exists():
        sys.stderr.write(f"[WARN] Missing PySCF-optimized geometry {opt}; falling back to {raw}\n")
        return raw
    return raw


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compare a 2D local-stretch DVR spectrum to ORCA GVPT2 for monomers.\n"
            "We extract the two highest-frequency ORCA fundamentals as 'stretches', and compare:\n"
            "  fundamentals (2), overtones (2), and combination band (1) by nearest-frequency matching."
        )
    )
    ap.add_argument("--species", required=True, help="Species key (e.g. H2O, HDO)")
    ap.add_argument("--bond1", default="O0-H1", help="First stretch coordinate (default O0-H1)")
    ap.add_argument("--bond2", default="O0-H2", help="Second stretch coordinate (default O0-H2)")
    ap.add_argument("--ref-json", type=Path, default=Path("reference/all_gvpt2.json"))
    ap.add_argument("--pipeline", type=Path, default=Path("pyscf_pme_pipeline.py"))
    ap.add_argument("--mmol", type=Path, default=None, help="Geometry file (default: prefer geom/<species>.pyscf_opt.mmol)")
    ap.add_argument("--level", choices=["default", "orca_like"], default="orca_like")
    ap.add_argument(
        "--stretch-kind",
        choices=["auto", "oh", "od", "any"],
        default="auto",
        help="How to select the two ORCA stretch modes (default auto: inferred from bond H/D types)",
    )
    ap.add_argument("--method", default=None, help="Override preset method")
    ap.add_argument("--basis", default=None, help="Override preset basis")
    ap.add_argument("--no-ri", action="store_true")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--pes-workers", type=int, default=8)
    ap.add_argument("--rmin", type=float, default=0.70)
    ap.add_argument("--rmax", type=float, default=1.70)
    ap.add_argument("--npts", type=int, default=31)
    ap.add_argument("--nmax", type=int, default=12, help="Number of 2D eigenstates to report/compare (default 12)")
    ap.add_argument("--keo", choices=["reduced", "gmatrix"], default="gmatrix", help="2D KEO for PySCF runs (default gmatrix)")
    ap.add_argument("--match", choices=["optimal", "greedy"], default="optimal", help="Matching strategy (default optimal)")
    ap.add_argument("--fail-rms", type=float, default=None, help="If set, fail if RMS(|Δ|) exceeds this (cm^-1)")
    ap.add_argument("--fail-max", type=float, default=None, help="If set, fail if max(|Δ|) exceeds this (cm^-1)")
    args = ap.parse_args(argv)

    try:
        import pyscf  # noqa: F401
    except Exception:
        sys.stderr.write(
            "ERROR: PySCF is not importable. Install the project dependencies "
            "with 'uv sync' or activate an environment containing PySCF.\n"
        )
        return 2

    ref = _load_ref(args.ref_json)
    if args.species not in ref:
        sys.stderr.write(f"ERROR: species '{args.species}' not found in {args.ref_json}\n")
        return 2
    ref_sp = ref[args.species]

    mmol = Path(args.mmol) if args.mmol is not None else _default_geom(args.species)
    if not mmol.exists():
        sys.stderr.write(f"ERROR: geometry file not found: {mmol}\n")
        return 2

    stretch_kind = str(args.stretch_kind).lower()
    if stretch_kind == "auto":
        try:
            syms = _read_mmol_symbols(mmol)
            h1 = _parse_bond_h_index(str(args.bond1))
            h2 = _parse_bond_h_index(str(args.bond2))
            t1 = "od" if syms[h1].upper() == "D" else "oh"
            t2 = "od" if syms[h2].upper() == "D" else "oh"
            if t1 == t2:
                stretch_kind = t1
            else:
                # Mixed case: fall back to 'any' (still reasonable for monomers).
                sys.stderr.write(f"[WARN] Mixed bond types ({t1},{t2}) for auto stretch selection; using 'any'.\n")
                stretch_kind = "any"
        except Exception as exc:
            sys.stderr.write(f"[WARN] Failed to infer stretch kind from geometry ({exc}); using 'any'.\n")
            stretch_kind = "any"

    stretches = _pick_stretch_modes(ref_sp, kind=stretch_kind)
    i = int(stretches[0]["mode_index"])
    j = int(stretches[1]["mode_index"])

    targets: List[Tuple[str, float]] = []
    for r in stretches:
        targets.append((f"fund[{int(r['mode_index'])}]", float(r["v_fund_cm_1"])))
    ov_i = _find_overtone(ref_sp, i)
    ov_j = _find_overtone(ref_sp, j)
    comb = _find_combination(ref_sp, i, j)
    if ov_i is not None:
        targets.append((f"2*[{i}]", float(ov_i["frequency_cm_1"])))
    if ov_j is not None:
        targets.append((f"2*[{j}]", float(ov_j["frequency_cm_1"])))
    if comb is not None:
        targets.append((f"[{i}]+[{j}]", float(comb["frequency_cm_1"])))

    # Run the 2D scan + DVR.
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

    # Compare by nearest-frequency matching.
    targets_sorted = sorted(targets, key=lambda t: t[1])
    cand_sorted = sorted(cand, key=lambda c: c[1])
    matched = _optimal_match(targets_sorted, cand_sorted) if args.match == "optimal" else _greedy_match(targets_sorted, cand_sorted)

    diffs = [abs(d) for (_lab, _ref, n, _nu, d) in matched if n >= 0 and math.isfinite(d)]
    rms = math.sqrt(sum(x * x for x in diffs) / len(diffs)) if diffs else float("nan")
    mx = max(diffs) if diffs else float("nan")

    print(f"Species: {args.species} | geom={mmol}")
    print(f"2D scan: bonds {args.bond1} & {args.bond2} | window [{args.rmin:.2f},{args.rmax:.2f}] Å | npts={int(args.npts)} | nmax={int(args.nmax)}")
    print(f"ORCA stretch modes picked by highest w_harm: {i}, {j}")
    print(f"Matched targets: {len(matched)} | RMS(|Δ|)={rms:.2f} cm^-1 | max(|Δ|)={mx:.2f} cm^-1")
    for label, nu_ref, n, nu, d in matched:
        if n < 0:
            print(f"  {label:10s} ORCA {nu_ref:8.2f}  PySCF <missing>     Δ  <missing>")
        else:
            print(f"  {label:10s} ORCA {nu_ref:8.2f}  PySCF n={n:2d} {nu:8.2f}  Δ {d:+8.2f}")

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
