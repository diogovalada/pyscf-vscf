#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


def _base_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.setdefault("PYTHONNOUSERSITE", "1")
    # Default to moderate OpenMP threading for faster iteration; keep BLAS pinned to 1.
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    return env


def _run(cmd: List[str], *, env: Dict[str, str]) -> int:
    p = subprocess.run(cmd, env=env)
    return int(p.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    # Keep it intentionally tiny: the point is a cheap regression sanity check.
    env = _base_env()

    try:
        import pyscf  # noqa: F401
    except Exception:
        sys.stderr.write(
            "ERROR: PySCF is not importable. Install the validation dependencies "
            "with 'uv sync --extra validation' or activate an environment containing PySCF.\n"
        )
        return 2

    root = Path(__file__).resolve().parents[1]
    morse = root / "scripts" / "validate_morse_dvr.py"
    scan = root / "scripts" / "validate_1d_scan_regression.py"
    compare = root / "scripts" / "compare_orca_harmonic.py"
    cmp2d = root / "scripts" / "compare_orca_gvpt2_2d_stretches.py"
    gradcheck = root / "scripts" / "check_gradients.py"
    marvel2d = root / "scripts" / "compare_marvel_vbo_2d_stretches.py"

    rc = 0
    # Guardrail: ensure saved PySCF-optimized geometries are still stationary at the current ES settings.
    for sp in ["H2O", "HDO", "D2O"]:
        rc |= _run([sys.executable, str(gradcheck), "--mmol", f"geom/{sp}.pyscf_opt.mmol"], env=env)
    rc |= _run([sys.executable, str(morse)], env=env)
    rc |= _run([sys.executable, str(scan)], env=env)

    for sp in ["H2O", "HDO", "D2O"]:
        rc |= _run([sys.executable, str(compare), "--species", sp, "--max-parallel", env.get("OMP_NUM_THREADS", "8")], env=env)

    # Optional longer checks (run occasionally): enable with VSCF_LONG_VALIDATIONS=1
    if env.get("VSCF_LONG_VALIDATIONS", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        rc |= _run([sys.executable, str(gradcheck), "--mmol", "geom/HDO-HDO-HD.pyscf_opt.mmol"], env=env)
        # 2D stretch local-mode check vs ORCA GVPT2 (expensive).
        for sp in ["H2O", "HDO", "D2O"]:
            rc |= _run(
                [
                    sys.executable,
                    str(cmp2d),
                    "--species",
                    sp,
                    "--max-parallel",
                    env.get("OMP_NUM_THREADS", "8"),
                    "--pes-workers",
                    env.get("OMP_NUM_THREADS", "8"),
                    "--fail-rms",
                    "120",
                    "--fail-max",
                    "150",
                ],
                env=env,
            )

        # Dimer harmonic vs ORCA (note: ORCA geometry may be non-stationary in PySCF; use --no-strict).
        rc |= _run(
            [
                sys.executable,
                str(compare),
                "--species",
                "HDO-HDO-HD",
                "--mmol",
                "geom/HDO-HDO-HD.mmol",
                "--no-strict",
                "--max-parallel",
                env.get("OMP_NUM_THREADS", "8"),
                "--fail-rms",
                "120",
                "--fail-max",
                "250",
            ],
            env=env,
        )

    # Real-molecule high-overtone regression vs MARVEL/IUPAC band origins (2D stretch model).
    # Enable with VSCF_MARVEL_VALIDATIONS=1 (moderately expensive; runs 3 monomer 2D calculations).
    if env.get("VSCF_MARVEL_VALIDATIONS", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        for sp in ["H2O", "HDO", "D2O"]:
            reveal = env.get("OMP_NUM_THREADS", "8")
            rc |= _run(
                [
                    sys.executable,
                    str(marvel2d),
                    "--species",
                    sp,
                    "--max-parallel",
                    reveal,
                    "--pes-workers",
                    reveal,
                    "--npts",
                    "21",
                    "--nmax",
                    "30",
                ],
                env=env,
            )

    # Ultra-long checks (dimers 2D PES): enable with VSCF_DIMER_VALIDATIONS=1
    if env.get("VSCF_DIMER_VALIDATIONS", "").strip() in {"1", "true", "TRUE", "yes", "YES"}:
        cmp2d = root / "scripts" / "compare_orca_gvpt2_2d_stretches.py"
        # HDO-HDO-HD OH–OH stretches: O0-H1 and O3-H5
        rc |= _run(
            [
                sys.executable,
                str(cmp2d),
                "--species",
                "HDO-HDO-HD",
                "--mmol",
                "geom/HDO-HDO-HD.pyscf_opt.mmol",
                "--bond1",
                "O0-H1",
                "--bond2",
                "O3-H5",
                "--stretch-kind",
                "oh",
                "--max-parallel",
                env.get("OMP_NUM_THREADS", "8"),
                "--pes-workers",
                env.get("OMP_NUM_THREADS", "8"),
                "--fail-rms",
                "120",
                "--fail-max",
                "150",
            ],
            env=env,
        )
        # HDO-HDO-HD OD–OD stretches: O0-D2 and O3-D4 (indices 2 and 4 are D in the MMOL)
        rc |= _run(
            [
                sys.executable,
                str(cmp2d),
                "--species",
                "HDO-HDO-HD",
                "--mmol",
                "geom/HDO-HDO-HD.pyscf_opt.mmol",
                "--bond1",
                "O0-H2",
                "--bond2",
                "O3-H4",
                "--stretch-kind",
                "od",
                "--max-parallel",
                env.get("OMP_NUM_THREADS", "8"),
                "--pes-workers",
                env.get("OMP_NUM_THREADS", "8"),
                "--fail-rms",
                "120",
                "--fail-max",
                "150",
            ],
            env=env,
        )

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
