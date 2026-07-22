#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


def _set_single_thread_env() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")


def main() -> int:
    _set_single_thread_env()
    try:
        import pyscf  # noqa: F401
    except Exception:
        sys.stderr.write(
            "ERROR: PySCF is not importable. Install the validation dependencies "
            "with 'uv sync --extra validation' or activate an environment containing PySCF.\n"
        )
        return 2

    # Import after setting env vars (repo root is not necessarily on sys.path).
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import pyscf_pme_pipeline as p  # type: ignore

    # Keep the check fast and deterministic.
    p.WORKERS = 1
    p.THREADS_PER_WORKER = 1
    p.VERBOSE = False
    p.DEV_FAST = True
    p.STRICT = True

    geom = Path("geom/H2O.pyscf_opt.mmol")
    if not geom.exists():
        sys.stderr.write(f"ERROR: missing geometry: {geom}\n")
        return 2

    mol = p.read_midas_mmol(geom)
    cfg = p.ESSettings(method="hf", basis="sto-3g", use_density_fit=False, dispersion=None, rtproj="pyscf", strict=True)

    bond = p.parse_bond("O0-H1")
    axis_vec = mol.coords[bond.H] - mol.coords[bond.O]

    # Pick an OH-like normal mode and scan a small symmetric displacement window.
    u_dir, kbest, f_cm, modes, freqs_cm = p.calc_normal_mode_direction(mol, cfg, bond)

    npts = 21
    half = 0.15
    smin, smax = -half, +half
    R, E, MU = p.grid_1d_pes_dms_normal(mol, cfg, u_dir, smin, smax, npts)

    s = np.asarray(R, float)
    e = np.asarray(E, float)
    if s.size != npts or e.size != npts:
        raise RuntimeError("Unexpected grid size in 1D scan")

    # Energy minimum should be near the center (s≈0).
    imin = int(np.argmin(e))
    smin_at = float(s[imin])
    if abs(smin_at) > 0.03:
        raise RuntimeError(f"1D scan minimum not near 0: s_min={smin_at:.4f} Å (expected |s_min|<=0.03 Å)")

    # Build a small DVR spectrum and ensure it behaves sensibly.
    masses = mol.analysis_masses()
    mu_eff = float(np.sum(masses * np.sum(u_dir * u_dir, axis=1)))
    spec = p.variational_1d(s, e, MU, mu_eff, axis=axis_vec, vmax=4)
    if not spec:
        raise RuntimeError("No DVR excitations returned")

    nu1 = float(spec[0]["freq_cm"])
    if not np.isfinite(nu1) or nu1 <= 0.0:
        raise RuntimeError(f"Invalid DVR v=1 frequency: {nu1}")

    # v=1 should be on the same scale as the selected harmonic frequency.
    rel = abs(nu1 - float(f_cm)) / float(f_cm)
    if rel > 0.30:
        raise RuntimeError(f"DVR v=1 deviates too far from harmonic: nu1={nu1:.1f}, harm={float(f_cm):.1f}, rel={rel:.3f}")

    # Frequencies should increase with v.
    nus = [float(s["freq_cm"]) for s in spec]
    if any((not np.isfinite(x) or x <= 0.0) for x in nus):
        raise RuntimeError("Non-finite/negative DVR frequencies detected")
    if any(nus[i] <= nus[i - 1] for i in range(1, len(nus))):
        raise RuntimeError(f"DVR frequencies not strictly increasing: {nus}")

    print(f"[OK] 1D scan regression (H2O HF/sto-3g): mode={kbest} harm={float(f_cm):.1f} cm^-1; DVR v=1={nu1:.1f} cm^-1; min at s={smin_at:.4f} Å")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
