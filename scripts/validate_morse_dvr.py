#!/usr/bin/env python3
"""
Step-1 validation: isolate the DVR + transition-dipole numerics (no electronic structure).

Validates:
- 1D sinc DVR energy levels against analytic Morse bound-state energies
- 2D product DVR on a separable potential against sums of 1D levels
- Transition dipole integral sanity: constant dipole -> zero transitions (orthogonality)

Run inside the pyscf conda env:
  PYTHONNOUSERSITE=1 python scripts/validate_morse_dvr.py
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from pathlib import Path

# Ensure repo root is importable when running as a script.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pyscf_pme_pipeline import (
    AMU,
    ANG_TO_BOHR,
    HARTREE_TO_CM,
    DVR1D,
    DVR2D,
    product_dvr_2d,
    sinc_dvr_1d,
    trans_mu_1d,
    trans_mu_2d,
)


def morse_potential_shifted(
    R_ang: np.ndarray, *, De_Eh: float, alpha_bohr_inv: float, Re_ang: float
) -> np.ndarray:
    """
    Morse potential with minimum at 0 and dissociation at De:
        V(x) = De * (1 - exp(-alpha*(x-xe)))^2
    x,xe in Bohr; alpha in 1/Bohr; De in Eh.
    """
    x = np.asarray(R_ang, float) * ANG_TO_BOHR
    xe = float(Re_ang) * ANG_TO_BOHR
    a = float(alpha_bohr_inv)
    De = float(De_Eh)
    y = 1.0 - np.exp(-a * (x - xe))
    return De * (y * y)


def morse_analytic_levels_shifted(
    *, De_Eh: float, alpha_bohr_inv: float, mu_amu: float, vmax: int
) -> List[float]:
    """
    Analytic Morse bound-state energies (above minimum, in Eh), for the shifted Morse potential.

    Using:
      mu = mu_amu * AMU  (electron masses)
      lambda = sqrt(2*mu*De)/alpha
      n = v + 1/2
      E_v (above min) = (alpha^2/(2*mu)) * (2*lambda*n - n^2)
    Valid for v < lambda - 1/2.
    """
    mu = float(mu_amu) * AMU
    De = float(De_Eh)
    a = float(alpha_bohr_inv)
    lam = np.sqrt(2.0 * mu * De) / a
    out: List[float] = []
    for v in range(int(vmax) + 1):
        n = v + 0.5
        if n >= lam:
            break
        Ev = (a * a / (2.0 * mu)) * (2.0 * lam * n - n * n)
        out.append(float(Ev))
    return out


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


def _fmt_cm(x_Eh: float) -> str:
    return f"{x_Eh * HARTREE_TO_CM: .6f}"


def check_1d_morse_levels(
    *, Rmin: float, Rmax: float, npts: int, mu_amu: float, De_Eh: float, alpha_bohr_inv: float, Re_ang: float, nv: int
) -> CheckResult:
    R = np.linspace(Rmin, Rmax, int(npts))
    V = morse_potential_shifted(R, De_Eh=De_Eh, alpha_bohr_inv=alpha_bohr_inv, Re_ang=Re_ang)
    dvr: DVR1D = sinc_dvr_1d(R, mu_amu, V)
    ref = morse_analytic_levels_shifted(
        De_Eh=De_Eh, alpha_bohr_inv=alpha_bohr_inv, mu_amu=mu_amu, vmax=nv
    )
    n = min(len(ref), nv + 1, len(dvr.evals))
    if n < 3:
        return CheckResult("1D Morse levels", False, f"Too few levels computed (n={n})")
    err_Eh = np.abs(dvr.evals[:n] - np.array(ref[:n]))
    err_cm = err_Eh * HARTREE_TO_CM
    max_err = float(np.max(err_cm))
    # For a moderate grid (npts ~ 300–500), sub-cm^-1 agreement for low v is realistic.
    tol_cm = 1.0
    ok = max_err < tol_cm
    details = (
        f"npts={npts} window=[{Rmin},{Rmax}]Å max|ΔE_v|={max_err:.3f} cm^-1 (tol {tol_cm:.1f}); "
        f"E0_dvr={_fmt_cm(dvr.evals[0])} cm^-1"
    )
    return CheckResult("1D Morse levels", ok, details)


def check_1d_constant_dipole_orthogonality(dvr: DVR1D, vmax: int = 5) -> CheckResult:
    mu_of_R = lambda rr: np.ones_like(rr, dtype=float)  # constant dipole -> <0|mu|v> = 0 for v!=0
    vals = []
    for v in range(1, min(int(vmax) + 1, dvr.evecs.shape[1])):
        vals.append(abs(trans_mu_1d(dvr, mu_of_R, v)))
    max_abs = float(max(vals)) if vals else 0.0
    tol = 5e-6
    ok = max_abs < tol
    return CheckResult(
        "1D constant dipole",
        ok,
        f"max|<0|mu_const|v>|={max_abs:.3e} (tol {tol:.1e})",
    )


def check_1d_overtone_convergence(
    *,
    mu_amu: float,
    De_Eh: float,
    alpha_bohr_inv: float,
    Re_ang: float,
    Rmin: float,
    Rmax: float,
    npts_lo: int,
    npts_hi: int,
    vmax: int,
) -> CheckResult:
    """
    High-overtone sanity: check that <0|mu(R)|v> converges w.r.t. grid resolution
    for a simple dipole model mu(R) = (R - Re).
    """
    Rlo = np.linspace(Rmin, Rmax, int(npts_lo))
    Rhi = np.linspace(Rmin, Rmax, int(npts_hi))
    Vlo = morse_potential_shifted(Rlo, De_Eh=De_Eh, alpha_bohr_inv=alpha_bohr_inv, Re_ang=Re_ang)
    Vhi = morse_potential_shifted(Rhi, De_Eh=De_Eh, alpha_bohr_inv=alpha_bohr_inv, Re_ang=Re_ang)
    dlo = sinc_dvr_1d(Rlo, mu_amu, Vlo)
    dhi = sinc_dvr_1d(Rhi, mu_amu, Vhi)

    mu_of_R_lo = lambda rr: (np.asarray(rr, float) - float(Re_ang))
    mu_of_R_hi = lambda rr: (np.asarray(rr, float) - float(Re_ang))

    nlo = min(int(vmax) + 1, dlo.evecs.shape[1])
    nhi = min(int(vmax) + 1, dhi.evecs.shape[1])
    n = min(nlo, nhi)
    if n < 6:
        return CheckResult("1D overtone convergence", False, f"Too few states available (n={n})")

    rel_errs = []
    abs_errs = []
    for v in range(1, n):
        mlo = float(trans_mu_1d(dlo, mu_of_R_lo, v))
        mhi = float(trans_mu_1d(dhi, mu_of_R_hi, v))
        abs_err = abs(mhi - mlo)
        abs_errs.append(abs_err)
        if abs(mhi) > 1e-4:
            rel_errs.append(abs_err / abs(mhi))

    max_abs = float(max(abs_errs)) if abs_errs else 0.0
    max_rel = float(max(rel_errs)) if rel_errs else 0.0

    # This is a numerical convergence check, not a physics check.
    # High-v moments get small; use a mild relative tolerance with an absolute backstop.
    tol_rel = 2e-2
    tol_abs = 5e-4
    ok = (max_rel < tol_rel) and (max_abs < tol_abs)
    details = (
        f"mu(R)=(R-Re), v=1..{n-1}; npts {npts_lo}->{npts_hi}; "
        f"max|Δmu|={max_abs:.3e}, max rel={max_rel:.3e} (tol rel {tol_rel:.1e} and abs {tol_abs:.1e})"
    )
    return CheckResult("1D overtone convergence", ok, details)


def check_1d_highv_energy_convergence(
    *,
    mu_amu: float,
    De_Eh: float,
    alpha_bohr_inv: float,
    Re_ang: float,
    Rmin: float,
    Rmax: float,
    npts_lo: int,
    npts_hi: int,
    vmax: int,
) -> CheckResult:
    Rlo = np.linspace(Rmin, Rmax, int(npts_lo))
    Rhi = np.linspace(Rmin, Rmax, int(npts_hi))
    Vlo = morse_potential_shifted(Rlo, De_Eh=De_Eh, alpha_bohr_inv=alpha_bohr_inv, Re_ang=Re_ang)
    Vhi = morse_potential_shifted(Rhi, De_Eh=De_Eh, alpha_bohr_inv=alpha_bohr_inv, Re_ang=Re_ang)
    dlo = sinc_dvr_1d(Rlo, mu_amu, Vlo)
    dhi = sinc_dvr_1d(Rhi, mu_amu, Vhi)
    n = min(int(vmax) + 1, len(dlo.evals), len(dhi.evals))
    if n < 6:
        return CheckResult("1D high-v energy convergence", False, f"Too few states (n={n})")
    dE_cm = np.abs(dhi.evals[:n] - dlo.evals[:n]) * HARTREE_TO_CM
    max_dE = float(np.max(dE_cm))
    tol_cm = 2.0
    ok = max_dE < tol_cm
    return CheckResult(
        "1D high-v energy convergence",
        ok,
        f"v=0..{n-1}; npts {npts_lo}->{npts_hi}; max|ΔE|={max_dE:.3f} cm^-1 (tol {tol_cm:.1f})",
    )

def check_2d_separable_spectrum(
    *,
    Rmin: float,
    Rmax: float,
    npts: int,
    mu1_amu: float,
    mu2_amu: float,
    De1_Eh: float,
    De2_Eh: float,
    alpha1_bohr_inv: float,
    alpha2_bohr_inv: float,
    Re1_ang: float,
    Re2_ang: float,
) -> Tuple[CheckResult, DVR2D]:
    R1 = np.linspace(Rmin, Rmax, int(npts))
    R2 = np.linspace(Rmin, Rmax, int(npts))
    V1 = morse_potential_shifted(R1, De_Eh=De1_Eh, alpha_bohr_inv=alpha1_bohr_inv, Re_ang=Re1_ang)
    V2 = morse_potential_shifted(R2, De_Eh=De2_Eh, alpha_bohr_inv=alpha2_bohr_inv, Re_ang=Re2_ang)
    V = V1[:, None] + V2[None, :]

    d1 = sinc_dvr_1d(R1, mu1_amu, V1)
    d2 = sinc_dvr_1d(R2, mu2_amu, V2)
    dvr2: DVR2D = product_dvr_2d(R1, R2, mu1_amu, mu2_amu, V)

    # Expected lowest k energies are the sorted sums of the 1D levels
    k = int(min(len(dvr2.evals), 12))
    sums = []
    for i in range(min(8, len(d1.evals))):
        for j in range(min(8, len(d2.evals))):
            sums.append(float(d1.evals[i] + d2.evals[j]))
    sums = np.array(sorted(sums)[:k], float)
    err_cm = np.abs(dvr2.evals[:k] - sums) * HARTREE_TO_CM
    max_err = float(np.max(err_cm)) if err_cm.size else float("inf")
    tol_cm = 2.0
    ok = max_err < tol_cm
    details = f"npts={npts} max|ΔE_k|={max_err:.3f} cm^-1 (tol {tol_cm:.1f})"
    return CheckResult("2D separable spectrum", ok, details), dvr2


def check_2d_constant_dipole_orthogonality(dvr2: DVR2D, vmax: int = 5) -> CheckResult:
    mu_proj = np.ones((len(dvr2.R1), len(dvr2.R2)), dtype=float)
    vals = []
    for n in range(1, min(int(vmax) + 1, dvr2.evecs.shape[1])):
        vals.append(abs(trans_mu_2d(dvr2, mu_proj, n)))
    max_abs = float(max(vals)) if vals else 0.0
    tol = 1e-5
    ok = max_abs < tol
    return CheckResult(
        "2D constant dipole",
        ok,
        f"max|<0|mu_const|n>|={max_abs:.3e} (tol {tol:.1e})",
    )


def main() -> int:
    # Pick a reasonably bound Morse with several levels (numbers are generic; units are consistent).
    mu_amu = 0.94
    De_Eh = 0.18
    alpha_bohr_inv = 1.5
    Re_ang = 1.0

    # 1D: choose a window wide enough for higher-v wavefunctions (overtones).
    Rmin, Rmax, npts = 0.6, 3.2, 601
    nv_ref = 8

    checks: List[CheckResult] = []

    c1 = check_1d_morse_levels(
        Rmin=Rmin,
        Rmax=Rmax,
        npts=npts,
        mu_amu=mu_amu,
        De_Eh=De_Eh,
        alpha_bohr_inv=alpha_bohr_inv,
        Re_ang=Re_ang,
        nv=nv_ref,
    )
    checks.append(c1)

    # Build a DVR once for the dipole orthogonality check
    R = np.linspace(Rmin, Rmax, int(npts))
    V = morse_potential_shifted(R, De_Eh=De_Eh, alpha_bohr_inv=alpha_bohr_inv, Re_ang=Re_ang)
    dvr = sinc_dvr_1d(R, mu_amu, V)
    checks.append(check_1d_constant_dipole_orthogonality(dvr, vmax=6))
    checks.append(
        check_1d_overtone_convergence(
            mu_amu=mu_amu,
            De_Eh=De_Eh,
            alpha_bohr_inv=alpha_bohr_inv,
            Re_ang=Re_ang,
            Rmin=Rmin,
            Rmax=Rmax,
            npts_lo=601,
            npts_hi=1201,
            vmax=12,
        )
    )
    checks.append(
        check_1d_highv_energy_convergence(
            mu_amu=mu_amu,
            De_Eh=De_Eh,
            alpha_bohr_inv=alpha_bohr_inv,
            Re_ang=Re_ang,
            Rmin=Rmin,
            Rmax=Rmax,
            npts_lo=601,
            npts_hi=1201,
            vmax=14,
        )
    )

    # 2D separable check (smaller grid to keep it quick; still exercises product DVR)
    c2, dvr2 = check_2d_separable_spectrum(
        Rmin=0.7,
        Rmax=2.4,
        npts=21,
        mu1_amu=mu_amu,
        mu2_amu=mu_amu,
        De1_Eh=De_Eh,
        De2_Eh=De_Eh,
        alpha1_bohr_inv=alpha_bohr_inv,
        alpha2_bohr_inv=alpha_bohr_inv,
        Re1_ang=Re_ang,
        Re2_ang=Re_ang,
    )
    checks.append(c2)
    checks.append(check_2d_constant_dipole_orthogonality(dvr2, vmax=6))

    ok_all = all(c.ok for c in checks)
    for c in checks:
        status = "OK" if c.ok else "FAIL"
        print(f"[{status}] {c.name}: {c.details}")

    if not ok_all:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
