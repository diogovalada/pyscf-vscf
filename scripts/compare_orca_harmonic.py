#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


def _load_ref(ref_json: Path) -> dict:
    with ref_json.open("r", encoding="utf-8") as f:
        return json.load(f)


def _orca_harmonic_vib_freqs_cm(ref: dict, species: str, *, rtmodes: int) -> List[float]:
    try:
        modes = ref[species]["harmonic_modes"]
    except KeyError as exc:
        raise KeyError(f"Species '{species}' not found in reference JSON") from exc

    freqs = [float(m["frequency_cm_1"]) for m in modes]
    if rtmodes < 0 or rtmodes >= len(freqs):
        raise ValueError(f"Invalid --rtmodes={rtmodes} for 3N={len(freqs)}")

    # ORCA includes RT modes (near 0, sometimes with small negative values).
    order = sorted(range(len(freqs)), key=lambda i: abs(freqs[i]))
    vib_idx = order[int(rtmodes) :]
    vib = [abs(freqs[i]) for i in vib_idx]
    vib.sort()
    return vib


_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?")


def _parse_pyscf_freqs_from_stdout(stdout: str) -> List[float]:
    lines = stdout.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "Frequencies (cm^-1):":
            # Frequencies should appear immediately after the header, but allow for occasional
            # empty/progress lines when stdout is captured.
            for j in range(i + 1, min(i + 6, len(lines))):
                if lines[j].lstrip().startswith("Runtime:"):
                    break
                vals = [float(x) for x in _FLOAT_RE.findall(lines[j])]
                if vals:
                    return vals
            raise RuntimeError("Failed to parse frequencies after 'Frequencies (cm^-1):' header")
    raise RuntimeError("Did not find 'Frequencies (cm^-1):' in PySCF pipeline output")


def _run_pyscf_harmonic(
    *,
    pipeline: Path,
    mmol: Path,
    rtproj: str,
    max_parallel: int,
    method: str | None,
    basis: str | None,
    no_ri: bool,
    no_strict: bool,
) -> Tuple[int, str, str]:
    cmd: List[str] = [
        sys.executable,
        str(pipeline),
        "--mmol",
        str(mmol),
        "--task",
        "harmonic",
        "--rtproj",
        rtproj,
        "--max-parallel",
        str(int(max_parallel)),
    ]
    if method is not None:
        cmd += ["--method", method]
    if basis is not None:
        cmd += ["--basis", basis]
    if no_ri:
        cmd += ["--no-ri"]
    if no_strict:
        cmd += ["--no-strict"]

    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    # Keep BLAS thread pools pinned to 1; max_parallel controls OpenMP.
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")

    p = subprocess.run(cmd, text=True, capture_output=True, env=env)
    return p.returncode, p.stdout, p.stderr


def _rms(xs: Sequence[float]) -> float:
    if not xs:
        return float("nan")
    return math.sqrt(sum(x * x for x in xs) / len(xs))


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Compare PySCF harmonic frequencies to ORCA reference (drop RT modes by |nu|).")
    ap.add_argument("--species", required=True, help="Species key in reference/all_gvpt2.json (e.g. H2O, HDO)")
    ap.add_argument("--ref-json", type=Path, default=Path("reference/all_gvpt2.json"))
    ap.add_argument("--pipeline", type=Path, default=Path("pyscf_pme_pipeline.py"))
    ap.add_argument("--mmol", type=Path, default=None, help="Geometry file to run PySCF on (default: geom/<species>.mmol)")
    ap.add_argument("--rtmodes", type=int, default=6, help="Number of RT modes to drop (default 6 for non-linear molecules)")
    ap.add_argument("--rtproj", choices=["pyscf", "mw_explicit", "none"], default="pyscf")
    ap.add_argument("--max-parallel", type=int, default=8, help="Parallel budget passed to the pipeline (default 8)")
    ap.add_argument("--method", default=None)
    ap.add_argument("--basis", default=None)
    ap.add_argument("--no-ri", action="store_true", help="Disable RI density fitting for the PySCF run")
    ap.add_argument("--no-strict", action="store_true", help="Run pipeline with --no-strict (useful for debugging non-stationary geometries)")
    ap.add_argument("--fail-rms", type=float, default=25.0, help="Fail if RMS(|Δν|) exceeds this (cm^-1)")
    ap.add_argument("--fail-max", type=float, default=80.0, help="Fail if max(|Δν|) exceeds this (cm^-1)")
    args = ap.parse_args(argv)

    # Ensure user runs in the expected environment (otherwise the subprocess will fail noisily).
    try:
        import pyscf  # noqa: F401
    except Exception:
        sys.stderr.write(
            "ERROR: PySCF is not importable. Install the validation dependencies "
            "with 'uv sync --extra validation' or activate an environment containing PySCF.\n"
        )
        return 2

    ref = _load_ref(args.ref_json)
    orca_vib = _orca_harmonic_vib_freqs_cm(ref, args.species, rtmodes=int(args.rtmodes))

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

    rc, out, err = _run_pyscf_harmonic(
        pipeline=args.pipeline,
        mmol=mmol,
        rtproj=args.rtproj,
        max_parallel=int(args.max_parallel),
        method=args.method,
        basis=args.basis,
        no_ri=bool(args.no_ri),
        no_strict=bool(args.no_strict),
    )
    if rc != 0:
        sys.stderr.write("ERROR: PySCF pipeline failed.\n")
        sys.stderr.write(err)
        sys.stderr.write(out)
        return rc

    pyscf_vib = _parse_pyscf_freqs_from_stdout(out)
    pyscf_vib = [float(x) for x in pyscf_vib if float(x) > 1e-8]
    pyscf_vib.sort()

    if len(pyscf_vib) != len(orca_vib):
        sys.stderr.write(
            f"ERROR: mode count mismatch after RT removal: PySCF has {len(pyscf_vib)} vib modes, ORCA has {len(orca_vib)}.\n"
            "This likely indicates a parsing issue or inconsistent RT handling.\n"
        )
        return 2

    diffs = [p - o for p, o in zip(pyscf_vib, orca_vib)]
    abs_diffs = [abs(d) for d in diffs]

    rms = _rms(abs_diffs)
    mx = max(abs_diffs) if abs_diffs else float("nan")

    print(f"Species: {args.species}")
    print(f"Geometry: {mmol}")
    print(f"RT removal: drop {int(args.rtmodes)} smallest |nu| modes")
    print(f"Counts: vib={len(pyscf_vib)}")
    print(f"RMS(|Δν|) = {rms:.3f} cm^-1")
    print(f"max(|Δν|) = {mx:.3f} cm^-1")

    if args.fail_rms is not None and rms > float(args.fail_rms):
        sys.stderr.write(f"FAIL: RMS(|Δν|) {rms:.3f} > {float(args.fail_rms):.3f} cm^-1\n")
        return 1
    if args.fail_max is not None and mx > float(args.fail_max):
        sys.stderr.write(f"FAIL: max(|Δν|) {mx:.3f} > {float(args.fail_max):.3f} cm^-1\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
