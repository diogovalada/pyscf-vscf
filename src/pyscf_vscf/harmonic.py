"""Import-light harmonic-analysis helpers.

The routines in this module are extracted from the legacy
``pyscf_pme_pipeline.py`` driver. They only depend on NumPy at import time; the
optional PySCF thermo helpers are imported lazily by call sites that request the
``rtproj="pyscf"`` path.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.linalg as npl

from .constants import AMU, HARTREE_TO_CM


IMAG_FREQ_WARN_CM: float = 1.0
IMAG_FREQ_ERR_CM: float = 10.0

_WARNED_ONCE: set[str] = set()


@dataclass
class HarmonicResult:
    freqs_cm: np.ndarray
    modes: np.ndarray
    zpe_cm: float
    hessian_provenance: str = "analytic"


def warn_once(key: str, msg: str) -> None:
    """Emit a warning once, matching the legacy pipeline's stderr behavior."""

    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def signed_freqs_from_evals(w2: np.ndarray) -> np.ndarray:
    """Convert Hessian eigenvalues in a.u. to signed frequencies in cm^-1."""

    w2 = np.asarray(w2, dtype=float)
    return np.sign(w2) * np.sqrt(np.abs(w2)) * HARTREE_TO_CM


def handle_imaginary_modes(
    w2: np.ndarray,
    *,
    natm: int,
    rtproj: str,
    strict: bool,
    rt_rank: int | None = None,
    warn_fn: Callable[[str, str], None] | None = None,
) -> np.ndarray:
    """Enforce the legacy imaginary-mode policy after RT projection.

    Imaginary vibrational modes are identified after excluding the rotational
    and translational subspace. Low-magnitude imaginary modes above
    ``IMAG_FREQ_WARN_CM`` warn; modes above ``IMAG_FREQ_ERR_CM`` raise in strict
    mode and warn in non-strict mode. The returned eigenvalues are unchanged.
    """

    rtproj_lc = (rtproj or "pyscf").lower()
    if rtproj_lc == "none":
        return np.asarray(w2, dtype=float)

    emit_warning = warn_fn or warn_once
    w2 = np.asarray(w2, dtype=float)
    imag_cm = np.sqrt(np.clip(-w2, 0.0, None)) * HARTREE_TO_CM
    abs_cm = np.sqrt(np.abs(w2)) * HARTREE_TO_CM

    if rt_rank is not None:
        ntr = int(rt_rank)
    else:
        ntr = 6 if int(natm) > 2 else 5
    order = np.argsort(abs_cm)
    vib_idx = order[ntr:]

    bad = [(int(i), float(imag_cm[i])) for i in vib_idx if imag_cm[i] > IMAG_FREQ_ERR_CM]
    mid = [
        (int(i), float(imag_cm[i]))
        for i in vib_idx
        if IMAG_FREQ_WARN_CM < imag_cm[i] <= IMAG_FREQ_ERR_CM
    ]

    if bad:
        msg = (
            "Imaginary vibrational modes detected after RT projection "
            f"(|nu_imag| > {IMAG_FREQ_ERR_CM:.1f} cm^-1): "
            + ", ".join(f"mode[{i}]={v:.1f}i" for i, v in bad)
            + ". Geometry is likely not a minimum (re-optimize) or settings are inconsistent."
        )
        if strict:
            raise RuntimeError(msg)
        emit_warning(
            "imag_modes_non_strict",
            "NON-STRICT: " + msg + " (continuing; values will be clipped)",
        )

    if mid:
        emit_warning(
            "imag_modes_warn",
            "Possible low-magnitude imaginary vibrational modes after RT projection "
            f"({IMAG_FREQ_WARN_CM:.1f} < |nu_imag| <= {IMAG_FREQ_ERR_CM:.1f} cm^-1): "
            + ", ".join(f"mode[{i}]={v:.1f}i" for i, v in mid)
            + ". This may indicate a very floppy coordinate or insufficient "
            "optimization/SCF/grid tightness.",
        )

    return w2


def format_low_mode_summary(tag: str, w2: np.ndarray, nshow: int = 10) -> str:
    """Return the legacy low-mode diagnostic summary as a string."""

    w2 = np.asarray(w2, dtype=float)
    order = np.argsort(w2)
    w2s = w2[order][:nshow]
    f_signed = signed_freqs_from_evals(w2s)
    f_abs = np.abs(f_signed)

    lines = [f"{tag}: lowest {min(nshow, w2.size)} eigenvalues/frequencies"]
    for k in range(len(w2s)):
        lines.append(
            f"  {k:2d}  w2={w2s[k]: .6e}  "
            f"nu_signed={f_signed[k]: .3f} cm^-1  |nu|={f_abs[k]: .3f} cm^-1"
        )
    return "\n".join(lines)


def print_low_mode_summary(tag: str, w2: np.ndarray, nshow: int = 10) -> None:
    """Print the legacy low-mode diagnostic summary."""

    print(format_low_mode_summary(tag, w2, nshow=nshow))


def as_cart_hessian(H: np.ndarray, natm: int) -> np.ndarray:
    """Return a Cartesian ``(3N, 3N)`` Hessian from 2D or PySCF 4D shapes."""

    H = np.asarray(H, dtype=float)
    expected = (3 * int(natm), 3 * int(natm))
    if H.ndim == 2:
        return H
    if H.ndim == 4 and H.shape == (int(natm), int(natm), 3, 3):
        return H.transpose(0, 2, 1, 3).reshape(expected)
    raise ValueError(f"Unexpected Hessian shape {H.shape}")


def cart_to_hess4(Hc: np.ndarray, natm: int) -> np.ndarray:
    """Convert a Cartesian ``(3N, 3N)`` Hessian to PySCF ``(N, N, 3, 3)`` shape."""

    Hc = np.asarray(Hc, dtype=float)
    expected = (3 * int(natm), 3 * int(natm))
    if Hc.shape != expected:
        raise ValueError("Hc shape mismatch")
    return Hc.reshape(int(natm), 3, int(natm), 3).transpose(0, 2, 1, 3)


def mass_weight(H_au: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    """Mass-weight a Cartesian Hessian using isotope masses in amu."""

    H_au = np.asarray(H_au, dtype=float)
    masses_amu = np.asarray(masses_amu, dtype=float)
    if masses_amu.ndim != 1:
        raise ValueError("masses_amu must be a one-dimensional array")
    if H_au.shape != (3 * masses_amu.size, 3 * masses_amu.size):
        raise ValueError("Hessian shape is inconsistent with masses_amu")
    M = np.repeat(masses_amu, 3) * AMU
    if np.any(M <= 0.0):
        raise ValueError("masses_amu must be positive")
    return H_au / np.sqrt(np.outer(M, M))


def mw_rt_projector_explicit(
    W: np.ndarray,
    coords_bohr: np.ndarray,
    masses_amu: np.ndarray,
    svd_tol: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project translations and rotations from a mass-weighted Cartesian Hessian."""

    W = np.asarray(W, dtype=float)
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    masses_amu = np.asarray(masses_amu, dtype=float)
    if coords_bohr.ndim != 2 or coords_bohr.shape[1] != 3:
        raise ValueError("coords_bohr must have shape (N, 3)")
    if masses_amu.shape != (coords_bohr.shape[0],):
        raise ValueError("masses_amu shape mismatch")

    natm = coords_bohr.shape[0]
    if W.shape != (3 * natm, 3 * natm):
        raise ValueError("W shape mismatch for explicit RT projector")

    mass_au = masses_amu * AMU
    if np.any(mass_au <= 0.0):
        raise ValueError("masses_amu must be positive")
    sqrt_mass = np.sqrt(np.repeat(mass_au, 3))

    msum = float(np.sum(masses_amu))
    if msum <= 0.0:
        raise ValueError("Non-positive total mass")
    com = np.sum(coords_bohr * masses_amu[:, None], axis=0) / msum
    r = coords_bohr - com

    basis = []
    for d in range(3):
        v = np.zeros(3 * natm)
        v[d::3] = 1.0
        basis.append(v * sqrt_mass)

    axes = np.eye(3)
    for a in range(3):
        omega = axes[a]
        v = np.zeros((natm, 3))
        v[:, :] = np.cross(np.broadcast_to(omega, (natm, 3)), r)
        basis.append(v.reshape(-1) * sqrt_mass)

    B = np.column_stack(basis)
    U, s, _ = npl.svd(B, full_matrices=False)
    smax = float(s.max()) if s.size else 0.0
    keep = s > (float(svd_tol) * max(smax, 1.0))
    U = U[:, keep]
    rank = int(U.shape[1])

    P = np.eye(3 * natm) - U @ U.T
    Wp = P @ W @ P
    Wp = 0.5 * (Wp + Wp.T)
    info = {
        "rt_rank": float(rank),
        "sv_min_kept": float(s[keep].min()) if np.any(keep) else 0.0,
        "sv_max": float(smax),
    }
    return Wp, info


def mw_rt_projector_pyscf_like(
    W: np.ndarray,
    coords_bohr: np.ndarray,
    masses_amu: np.ndarray,
    thermo_mod: Any,
    svd_tol: float = 1e-12,
) -> tuple[np.ndarray, dict[str, float]]:
    """Build an RT projector from an injected PySCF thermo-like module."""

    W = np.asarray(W, dtype=float)
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    masses_amu = np.asarray(masses_amu, dtype=float)
    natm = coords_bohr.shape[0]
    if W.shape != (3 * natm, 3 * natm):
        raise ValueError("W shape mismatch for pyscf-like RT projector")

    try:
        get_tr = getattr(thermo_mod, "_get_TR")
        rotation_const = getattr(thermo_mod, "rotation_const")
        get_rotor_type = getattr(thermo_mod, "_get_rotor_type")
    except Exception as exc:
        raise RuntimeError(
            "PySCF thermo module lacks _get_TR/rotation_const/_get_rotor_type"
        ) from exc

    mass_center = np.einsum("z,zx->x", masses_amu, coords_bohr) / float(np.sum(masses_amu))
    coords = coords_bohr - mass_center

    TR = get_tr(masses_amu, coords)
    TRspace = [TR[:3]]

    rot_const = rotation_const(masses_amu, coords)
    rotor_type = get_rotor_type(rot_const)
    if rotor_type == "ATOM":
        pass
    elif rotor_type == "LINEAR":
        TRspace.append(TR[3:5])
    else:
        TRspace.append(TR[3:])

    A = np.vstack(TRspace)
    q, _ = npl.qr(A.T)
    _, s, _ = npl.svd(A, full_matrices=False)
    smax = float(s.max()) if s.size else 0.0
    keep = s > (float(svd_tol) * max(smax, 1.0))
    q = q[:, : int(np.sum(keep))]
    rank = int(q.shape[1])

    P = np.eye(3 * natm) - q @ q.T
    Wp = P @ W @ P
    Wp = 0.5 * (Wp + Wp.T)
    info = {
        "rt_rank": float(rank),
        "sv_max": float(smax),
        "sv_min_kept": float(s[keep].min()) if np.any(keep) else 0.0,
    }
    return Wp, info


def mass_weighted_freqs_modes_from_coords(
    H_au: np.ndarray,
    masses_amu: np.ndarray,
    coords_bohr: np.ndarray,
    *,
    rtproj: str = "pyscf",
    strict: bool = True,
    debug: bool = False,
    thermo_mod: Any | None = None,
    warn_fn: Callable[[str, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return harmonic frequencies and modes from pure coordinate arrays."""

    coords_bohr = np.asarray(coords_bohr, dtype=float)
    natm = int(coords_bohr.shape[0])
    Hc = as_cart_hessian(H_au, natm)
    W = mass_weight(Hc, masses_amu)

    if debug:
        w2_raw, _ = npl.eigh(W)
        print_low_mode_summary("Raw mass-weighted Hessian W (no RT projection)", w2_raw)

    rtproj_lc = (rtproj or "pyscf").lower()
    emit_warning = warn_fn or warn_once

    if rtproj_lc == "pyscf":
        if thermo_mod is None:
            try:
                from pyscf.hessian import thermo as thermo_mod  # type: ignore[import-not-found]
            except Exception as exc:
                msg = f"Requested --rtproj pyscf but pyscf.hessian.thermo unavailable ({exc})"
                if strict:
                    raise RuntimeError(msg) from exc
                emit_warning(
                    "rtproj_pyscf_missing",
                    msg + "; falling back to --rtproj mw_explicit",
                )
                rtproj_lc = "mw_explicit"

        if rtproj_lc == "pyscf":
            Wp, info = mw_rt_projector_pyscf_like(W, coords_bohr, masses_amu, thermo_mod)
            if debug:
                print(
                    "PySCF-like RT projector: "
                    f"rt_rank={int(info['rt_rank'])} "
                    f"sv_max={info['sv_max']:.3e} "
                    f"sv_min_kept={info['sv_min_kept']:.3e}"
                )
            w2, vec = npl.eigh(Wp)
            w2 = handle_imaginary_modes(
                w2,
                natm=natm,
                rtproj=rtproj_lc,
                strict=strict,
                rt_rank=int(info["rt_rank"]),
                warn_fn=warn_fn,
            )
            if debug:
                print_low_mode_summary("PySCF-like projected Hessian", w2)
            return np.sqrt(np.clip(w2, 0.0, None)) * HARTREE_TO_CM, vec

    if rtproj_lc == "none":
        Wp = W
        rt_rank = None
    elif rtproj_lc == "mw_explicit":
        Wp, info = mw_rt_projector_explicit(W, coords_bohr, masses_amu)
        rt_rank = int(info["rt_rank"])
        if debug:
            print(
                "Explicit MW RT projector: "
                f"rt_rank={rt_rank} "
                f"sv_max={info['sv_max']:.3e} "
                f"sv_min_kept={info['sv_min_kept']:.3e}"
            )
    else:
        raise ValueError(f"Unknown rtproj '{rtproj}'")

    w2, vec = npl.eigh(Wp)
    w2 = handle_imaginary_modes(
        w2,
        natm=natm,
        rtproj=rtproj_lc,
        strict=strict,
        rt_rank=rt_rank,
        warn_fn=warn_fn,
    )
    if debug:
        print_low_mode_summary("Final projected Hessian used for modes", w2)
    return np.sqrt(np.clip(w2, 0.0, None)) * HARTREE_TO_CM, vec


def mass_weighted_freqs_modes(
    pmol: Any,
    H_au: np.ndarray,
    masses_amu: np.ndarray,
    *,
    rtproj: str = "pyscf",
    strict: bool = True,
    debug: bool = False,
    thermo_mod: Any | None = None,
    warn_fn: Callable[[str, str], None] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Legacy-compatible adapter for PySCF-like molecule objects."""

    natm = int(getattr(pmol, "natm"))
    atom_coords = getattr(pmol, "atom_coords")
    try:
        coords_bohr = atom_coords(unit="Bohr")
    except TypeError:
        coords_bohr = atom_coords()
    coords_bohr = np.asarray(coords_bohr, dtype=float)
    if coords_bohr.shape != (natm, 3):
        raise ValueError("pmol.atom_coords() returned an unexpected shape")

    return mass_weighted_freqs_modes_from_coords(
        H_au,
        masses_amu,
        coords_bohr,
        rtproj=rtproj,
        strict=strict,
        debug=debug,
        thermo_mod=thermo_mod,
        warn_fn=warn_fn,
    )


def zpe_cm_from_freqs(freqs_cm: np.ndarray, cutoff_cm: float = 1e-5) -> float:
    """Return the legacy harmonic zero-point energy in cm^-1."""

    freqs_cm = np.asarray(freqs_cm, dtype=float)
    return float(0.5 * np.sum(freqs_cm[freqs_cm > float(cutoff_cm)]))


_signed_freqs_from_evals = signed_freqs_from_evals
_handle_imaginary_modes = handle_imaginary_modes
_print_low_mode_summary = print_low_mode_summary
_as_cart_hessian = as_cart_hessian
_mass_weight = mass_weight
_cart_to_hess4 = cart_to_hess4
_mw_rt_projector_explicit = mw_rt_projector_explicit
_mw_rt_projector_pyscf_like = mw_rt_projector_pyscf_like


__all__ = [
    "HarmonicResult",
    "IMAG_FREQ_ERR_CM",
    "IMAG_FREQ_WARN_CM",
    "as_cart_hessian",
    "cart_to_hess4",
    "format_low_mode_summary",
    "handle_imaginary_modes",
    "mass_weight",
    "mass_weighted_freqs_modes",
    "mass_weighted_freqs_modes_from_coords",
    "mw_rt_projector_explicit",
    "mw_rt_projector_pyscf_like",
    "print_low_mode_summary",
    "signed_freqs_from_evals",
    "warn_once",
    "zpe_cm_from_freqs",
]
