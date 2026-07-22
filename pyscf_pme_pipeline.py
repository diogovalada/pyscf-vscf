#!/usr/bin/env python3
"""
PySCF-based pipeline matching your Summary2 intent
- Robust MMOL reader with isotope handling (ISO=2 => D mass)
- Harmonic frequencies + ZPE (analytic Hessian if available, else FD of gradients)
- 1D / 2D PES+DMS grids from PySCF single points (parallelized with ProcessPool)
- Variational 1D/2D DVR on fitted grids to get overtone stick positions & intensities
- Parallel knobs: --max-parallel (budget) and --pes-workers (processes for PES/FD farming)

Note: For PES grids, prefer higher --pes-workers with lower effective threads per worker; see --max-parallel.
"""
# Changes implemented:
# - Development fast mode (--dev-fast): lighter method/basis caps, fewer grid points, and looser SCF thresholds for rapid iteration
from __future__ import annotations
import argparse
import time
import sys
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict
import multiprocessing as mp
import os
import numpy as np
import numpy.linalg as npl
from concurrent.futures import ProcessPoolExecutor
from scipy import sparse
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.linalg import eigsh

from pyscf import gto, scf, dft, grad, lib
from pyscf.data import elements
from vscf_io import write_xyz, write_midas_mmol

try:
    from pyscf_vscf import harmonic as _pkg_harmonic
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("pyscf_vscf"):
        _pkg_harmonic = None
    else:
        raise

try:
    from pyscf_vscf.workflows import harmonic as _pkg_harmonic_workflow
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("pyscf_vscf"):
        _pkg_harmonic_workflow = None
    else:
        raise

try:
    from pyscf_vscf.workflows import optimization as _pkg_optimization
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("pyscf_vscf"):
        _pkg_optimization = None
    else:
        raise

try:
    from pyscf_vscf.workflows import scans as _pkg_scans
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("pyscf_vscf"):
        _pkg_scans = None
    else:
        raise

# -------------------- Parallel controls --------------------
WORKERS: Optional[int] = None
THREADS_PER_WORKER: Optional[int] = None
MAX_PARALLEL: int = 8
PES_WORKERS: int = 1
VERBOSE: bool = False
DEV_FAST: bool = False
STRICT: bool = True
ALLOW_FD_HESSIAN: bool = False

_WARNED_ONCE: set[str] = set()

IMAG_FREQ_WARN_CM: float = 1.0
IMAG_FREQ_ERR_CM: float = 10.0

def warn(msg: str):
    sys.stderr.write(f"[WARN] {msg}\n")
    sys.stderr.flush()

def warn_once(key: str, msg: str):
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    warn(msg)

# Allow overriding multiprocessing start method via env; default to spawn for fork-safety with MKL/OpenMP.
# Spawn avoids post-SCF fork deadlocks but costs extra startup latency, so for tiny grids the sequential
# path (workers <= 1) keeps things faster. When you cap per-process OpenMP threads to 1, the inherited BLAS
# state is already single-threaded, so classic fork stays safe (set VSCF_PYSCF_START_METHOD=fork if you prefer
# that path).
_START_METHOD = os.environ.get('VSCF_PYSCF_START_METHOD', 'spawn')
try:
    _MP_CONTEXT = mp.get_context(_START_METHOD)
except ValueError:
    warn_once("mp_context_fallback", f"Invalid multiprocessing start method '{_START_METHOD}'; falling back to 'fork'")
    _MP_CONTEXT = mp.get_context('fork')

def _pool_init(threads: Optional[int], strict: bool=False):
    """
    Initializer for worker processes:
    - Set OpenMP threads per process for PySCF
    - Keep BLAS threadpools capped to 1 (for predictability; avoids oversubscription in multiprocessing)
    - Configure PySCF thread count
    This helps avoid deadlocks/oversubscription when using process pools.
    """
    global STRICT
    STRICT = bool(strict)
    try:
        t = int(threads) if threads is not None else 0
        if t==0:
            return
    except Exception:
        raise ValueError(f"Failed to parse threads={threads!r}; expected an integer or None")
    try:
        import os
        os.environ['OMP_NUM_THREADS'] = str(t)
        # For now, keep BLAS thread pools pinned to 1 for reproducibility and to avoid
        # oversubscription when using multiprocessing workers.
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['NUMEXPR_NUM_THREADS'] = '1'
        os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    except Exception:
        warn_once("threads_env", "Failed to set OMP/MKL/OPENBLAS/NUMEXPR thread env vars; oversubscription possible")
    try:
        lib.num_threads(t)
    except Exception:
        msg = f"Failed to set pyscf.lib.num_threads({t}); oversubscription possible"
        if STRICT:
            raise RuntimeError(msg)
        warn_once("pyscf_num_threads", msg)


def _shutdown_pool(ex: ProcessPoolExecutor, wait: bool, cancel_futures: bool=False):
    try:
        ex.shutdown(wait=wait, cancel_futures=cancel_futures)
    except TypeError:
        ex.shutdown(wait=wait)


def _terminate_pool(ex: ProcessPoolExecutor, timeout: float=1.0):
    procs = getattr(ex, "_processes", None)
    if not procs:
        return
    for proc in list(procs.values()):
        try:
            if proc.is_alive():
                proc.terminate()
        except Exception:
            pass
    deadline = time.time() + max(float(timeout), 0.0)
    for proc in list(procs.values()):
        try:
            remaining = deadline - time.time()
            if remaining <= 0:
                remaining = 0.0
            proc.join(remaining)
        except Exception:
            pass
    for proc in list(procs.values()):
        try:
            if proc.is_alive():
                if hasattr(proc, "kill"):
                    proc.kill()
                else:
                    proc.terminate()
                proc.join(timeout=0.0)
        except Exception:
            pass
    mgr_thread = getattr(ex, "_queue_management_thread", None)
    if mgr_thread and mgr_thread.is_alive():
        try:
            mgr_thread.join(timeout=max(float(timeout), 0.0))
        except Exception:
            pass


@contextmanager
def _pool_executor():
    if WORKERS is None or WORKERS <= 1:
        _pool_init(THREADS_PER_WORKER, STRICT)
        class _SequentialExecutor:
            def map(self, fn, iterable):
                return (fn(item) for item in iterable)
        yield _SequentialExecutor()
        return

    ex = ProcessPoolExecutor(max_workers=WORKERS,
                             initializer=_pool_init,
                             initargs=(THREADS_PER_WORKER, STRICT),
                             mp_context=_MP_CONTEXT)
    cancelled = False
    try:
        yield ex
    except KeyboardInterrupt:
        cancelled = True
        _terminate_pool(ex)
        _shutdown_pool(ex, wait=False, cancel_futures=True)
        raise
    finally:
        if not cancelled:
            _shutdown_pool(ex, wait=True)


def _format_runtime(seconds: float) -> str:
    hours, rem = divmod(seconds, 3600.0)
    minutes, secs = divmod(rem, 60.0)
    return f"{int(hours)}h {int(minutes):02d}m {secs:05.2f}s"


def log(msg: str):
    if VERBOSE:
        print(f"[INFO] {msg}", flush=True)


def _progress_update(completed: int, total: int, desc: str, step_pct: int = 5):
    if not VERBOSE or total <= 0:
        return
    if step_pct <= 0:
        step_pct = 1

    frac = completed / total
    pct = int(frac * 100.0)
    bucket = (pct // step_pct) * step_pct
    if completed >= total:
        bucket = 100

    state = getattr(_progress_update, "_state", None)
    if state is None:
        state = {}
        setattr(_progress_update, "_state", state)

    prev = state.get(desc)
    if prev is None or prev[0] != total or completed < prev[2]:
        prev_bucket = -1
    else:
        prev_bucket = prev[1]

    # Avoid extremely chatty output: only print when we cross a new percent bucket,
    # or at completion. Also skip the initial 0% bucket.
    if bucket <= prev_bucket and completed < total:
        return
    if bucket == 0 and completed < total:
        return

    state[desc] = (total, bucket, completed)
    bar_len = 20
    filled = int(bar_len * frac)
    bar = '#' * filled + '-' * (bar_len - filled)
    sys.stderr.write(f"\r{desc}: [{bar}] {completed}/{total} ({frac*100:5.1f}%)")
    if completed >= total:
        sys.stderr.write('\n')
    sys.stderr.flush()


def default_auxbasis(main_basis: str) -> str:
    bas = main_basis.lower().replace(' ', '')
    if bas.startswith('aug-cc-pvtz'):
        return 'aug-cc-pVTZ-jkfit'
    if bas.startswith('cc-pvtz'):
        return 'cc-pVTZ-jkfit'
    if bas.startswith('def2-tzvp'):
        return 'def2-universal-jkfit'
    return 'weigend+etb'

# -------------------- Grid cache (NPZ) --------------------
def _json_default(obj):
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def dump_grid_npz(path: Path, *, meta: Dict, arrays: Dict[str, np.ndarray]):
    from pyscf_vscf.cache import dump_grid_npz as package_dump_grid_npz

    package_dump_grid_npz(path, meta=meta, arrays=arrays)


def load_grid_npz(path: Path) -> Tuple[Dict, Dict[str, np.ndarray]]:
    from pyscf_vscf.cache import load_grid_npz as package_load_grid_npz

    return package_load_grid_npz(path)


def _assert_meta_equal(label: str, a, b):
    if a != b:
        raise ValueError(f"Grid cache mismatch for {label}: expected {a!r}, got {b!r}")


def _assert_meta_close(label: str, a: float, b: float, tol: float = 1e-10):
    if abs(float(a) - float(b)) > tol:
        raise ValueError(f"Grid cache mismatch for {label}: expected {a!r}, got {b!r}")

# -------------------- Constants --------------------
BOHR_TO_ANG = 0.529177210903
ANG_TO_BOHR = 1.0 / BOHR_TO_ANG
HARTREE_TO_CM = 219474.6313705
AMU = 1822.888486209
DEBYE_TO_C_M = 3.33564e-30
EPS0 = 8.8541878128e-12
HBAR = 1.054571817e-34
C_LIGHT_M_S = 299792458.0
MASS = { 'H': 1.00782503223, 'D': 2.01410177812, 'O': 15.99491461957 }

# -------------------- Molecule --------------------
@dataclass
class Molecule:
    symbols: List[str]
    coords: np.ndarray
    charge: int = 0
    spin: int = 0
    label: str = "mol"
    masses: Optional[np.ndarray] = None

    def analysis_masses(self) -> np.ndarray:
        if self.masses is not None:
            return np.asarray(self.masses, float)
        return np.array([MASS.get(s.upper(), MASS['H'] if s.upper() in ('H','D') else MASS['O'])
                         for s in self.symbols], float)

    def as_pyscf(self, basis: str = "aug-cc-pVTZ"):
        # map D->H for electronic structure, but push isotope masses via nucprop
        sym_elec = ['H' if s.upper()=='D' else s for s in self.symbols]
        mol = gto.Mole()
        mol.unit = 'Angstrom'
        mol.basis = basis
        mol.charge = self.charge
        mol.spin = self.spin
        mol.verbose = 0
        mol.atom = [(s, tuple(map(float,xyz))) for s,xyz in zip(sym_elec, self.coords)]
        # set per-atom masses when analysis masses differ
        def_mass = []
        for s in sym_elec:
            try:
                z = gto.mole.charge(s)
                def_mass.append(float(elements.MASSES[z]))
            except Exception as exc:
                # If PySCF can't resolve the element symbol, it's safer to fail loudly than
                # to silently substitute a mass (which can corrupt any vibrational analysis).
                if s.upper() in MASS:
                    warn_once("fallback_mass_lookup", f"Falling back to internal mass table for symbol '{s}' ({exc})")
                    def_mass.append(float(MASS[s.upper()]))
                else:
                    raise ValueError(f"Unknown element symbol '{s}' (failed to resolve atomic number)") from exc
        ana = self.analysis_masses()
        nucprop = {}
        for i,(md,ma) in enumerate(zip(def_mass, ana), start=1):
            if abs(float(ma)-float(md))>1e-3:
                nucprop[i] = {'mass': float(ma)}
        if nucprop:
            mol.nucprop = nucprop
        mol.build()
        return mol


def _legacy_molecule_with_coords(mol: Molecule, coords: np.ndarray) -> Molecule:
    return Molecule(
        mol.symbols,
        np.asarray(coords, float),
        mol.charge,
        mol.spin,
        label=mol.label,
        masses=mol.masses,
    )

# -------------------- MMOL reader --------------------
def read_midas_mmol(path: Path) -> Molecule:
    txt = Path(path).read_text().splitlines()
    i = 0
    while i < len(txt) and not txt[i].strip().upper().startswith('#1'):
        i += 1
    if i >= len(txt):
        raise ValueError('No #1 Xyz block')
    i += 1
    header = txt[i].strip().split()
    n = int(header[0]); unit = header[1].strip().upper()
    i += 2  # skip title line
    symbols, coords, masses = [], [], []
    while len(symbols) < n and i < len(txt):
        line = txt[i].strip(); i += 1
        if not line or line.startswith('#'): continue
        parts = line.split()
        if len(parts) < 4: continue
        sym = parts[0]
        x,y,z = map(float, parts[1:4])
        if unit in ('AU','BOHR'):
            x,y,z = x*BOHR_TO_ANG, y*BOHR_TO_ANG, z*BOHR_TO_ANG
        sym_out = sym
        for p in parts[4:]:
            if p.upper().startswith('ISO=') and sym.upper()=='H' and int(p.split('=')[1])==2:
                sym_out = 'D'
        symbols.append(sym_out)
        coords.append([x,y,z])
        masses.append(MASS['D'] if sym_out.upper()=='D' else MASS.get(sym_out.upper(), MASS['H']))
    if len(symbols)!=n:
        raise ValueError(f'Expected {n} atoms, got {len(symbols)}')
    return Molecule(symbols, np.array(coords,float), masses=np.array(masses,float), label=Path(path).stem)

# -------------------- Electronic structure --------------------
@dataclass
class ESSettings:
    method: str = 'wb97x'
    basis: str = 'aug-cc-pVTZ'
    use_density_fit: bool = True
    auxbasis: Optional[str] = None
    dispersion: Optional[str] = 'd4'
    rtproj: str = 'pyscf'
    strict: bool = True
    allow_fd_hessian: bool = False
    scf_conv_tol: Optional[float] = None
    scf_max_cycle: Optional[int] = None
    dft_grid_level: Optional[int] = None

def make_mean_field(pmol: gto.Mole, cfg: ESSettings):
    from pyscf_vscf.backends.pyscf import make_mean_field as package_make_mean_field

    # Keep the historical driver on the package's single fail-closed backend
    # path so open-shell selection, RI provenance, and all ES settings agree.
    return package_make_mean_field(pmol, _effective_es_settings(cfg))


def _effective_es_settings(cfg: ESSettings):
    from pyscf_vscf.settings import coerce_es_settings, development_fast_settings

    settings = coerce_es_settings(cfg)
    return development_fast_settings(settings) if DEV_FAST else settings

# -------------------- Hessian / Harmonic --------------------
@dataclass
class HarmonicResult:
    freqs_cm: np.ndarray
    modes: np.ndarray
    zpe_cm: float

def _signed_freqs_from_evals(w2: np.ndarray) -> np.ndarray:
    if _pkg_harmonic is not None:
        return _pkg_harmonic.signed_freqs_from_evals(w2)
    w2 = np.asarray(w2, float)
    return np.sign(w2) * np.sqrt(np.abs(w2)) * HARTREE_TO_CM


def _handle_imaginary_modes(
    w2: np.ndarray, *, natm: int, rtproj: str, strict: bool, rt_rank: Optional[int] = None
) -> np.ndarray:
    """
    Enforce strict handling of imaginary vibrational modes after RT projection.

    Behavior:
    - Only applies when RT projection is enabled (rtproj != 'none').
    - Imaginary modes with |nu_imag| <= IMAG_FREQ_WARN_CM are treated as numerical noise.
    - If any vibrational mode has |nu_imag| > IMAG_FREQ_ERR_CM:
        - strict=True: raise
        - strict=False: warn loudly and continue (will be clipped to 0 for printing)
    - If any vibrational mode falls in (IMAG_FREQ_WARN_CM, IMAG_FREQ_ERR_CM]:
        - warn loudly (even in strict mode), but continue.
    """
    if _pkg_harmonic is not None:
        return _pkg_harmonic.handle_imaginary_modes(
            w2,
            natm=natm,
            rtproj=rtproj,
            strict=strict,
            rt_rank=rt_rank,
            warn_fn=warn_once,
        )

    rtproj_lc = (rtproj or "pyscf").lower()
    if rtproj_lc == "none":
        return np.asarray(w2, float)

    w2 = np.asarray(w2, float)
    # Convert to magnitude of imaginary frequency in cm^-1
    imag_cm = np.sqrt(np.clip(-w2, 0.0, None)) * HARTREE_TO_CM
    abs_cm = np.sqrt(np.abs(w2)) * HARTREE_TO_CM

    # Exclude RT modes in the projected spectrum. Prefer projector rank (handles LINEAR/ATOM),
    # otherwise fall back to the usual 3N-6 (nonlinear) / 3N-5 (linear/diatomic) convention.
    if rt_rank is not None:
        ntr = int(rt_rank)
    else:
        ntr = 6 if int(natm) > 2 else 5
    order = np.argsort(abs_cm)
    vib_idx = order[ntr:]

    # Identify imaginary vibrational modes
    bad = [(int(i), float(imag_cm[i])) for i in vib_idx if imag_cm[i] > IMAG_FREQ_ERR_CM]
    mid = [(int(i), float(imag_cm[i])) for i in vib_idx if IMAG_FREQ_WARN_CM < imag_cm[i] <= IMAG_FREQ_ERR_CM]

    if bad:
        msg = (
            f"Imaginary vibrational modes detected after RT projection (|nu_imag| > {IMAG_FREQ_ERR_CM:.1f} cm^-1): "
            + ", ".join(f"mode[{i}]={v:.1f}i" for i, v in bad)
            + ". Geometry is likely not a minimum (re-optimize) or settings are inconsistent."
        )
        if strict:
            raise RuntimeError(msg)
        warn_once("imag_modes_non_strict", "NON-STRICT: " + msg + " (continuing; values will be clipped)")

    if mid:
        warn_once(
            "imag_modes_warn",
            "Possible low-magnitude imaginary vibrational modes after RT projection "
            f"({IMAG_FREQ_WARN_CM:.1f} < |nu_imag| <= {IMAG_FREQ_ERR_CM:.1f} cm^-1): "
            + ", ".join(f"mode[{i}]={v:.1f}i" for i, v in mid)
            + ". This may indicate a very floppy coordinate or insufficient optimization/SCF/grid tightness.",
        )

    return w2


def _print_low_mode_summary(tag: str, w2: np.ndarray, nshow: int=10):
    if _pkg_harmonic is not None:
        _pkg_harmonic.print_low_mode_summary(tag, w2, nshow=nshow)
        return
    w2 = np.asarray(w2, float)
    order = np.argsort(w2)
    w2s = w2[order][:nshow]
    f_signed = _signed_freqs_from_evals(w2s)
    f_abs = np.abs(f_signed)
    print(f"{tag}: lowest {min(nshow, w2.size)} eigenvalues/frequencies")
    for k in range(len(w2s)):
        print(f"  {k:2d}  w2={w2s[k]: .6e}  nu_signed={f_signed[k]: .3f} cm^-1  |nu|={f_abs[k]: .3f} cm^-1")


def _mw_rt_projector_explicit(W: np.ndarray, coords_bohr: np.ndarray, masses_amu: np.ndarray,
                              svd_tol: float=1e-12) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Explicit 6D translation+rotation projector in mass-weighted Cartesian coordinates.

    W: mass-weighted Hessian (3N x 3N) in a.u.
    coords_bohr: (N,3) in Bohr
    masses_amu: (N,) in amu
    """
    if _pkg_harmonic is not None:
        return _pkg_harmonic.mw_rt_projector_explicit(
            W,
            coords_bohr,
            masses_amu,
            svd_tol=svd_tol,
        )

    coords_bohr = np.asarray(coords_bohr, float)
    masses_amu = np.asarray(masses_amu, float)
    natm = coords_bohr.shape[0]
    if W.shape != (3*natm, 3*natm):
        raise ValueError("W shape mismatch for explicit RT projector")

    mass_au = masses_amu * AMU
    sqrt_mass = np.sqrt(np.repeat(mass_au, 3))

    # COM-center coordinates for rotations
    msum = float(np.sum(masses_amu))
    if msum <= 0:
        raise ValueError("Non-positive total mass")
    com = np.sum(coords_bohr * masses_amu[:, None], axis=0) / msum
    r = coords_bohr - com

    basis = []
    # Translations: sqrt(m_i) * e_dir
    for d in range(3):
        v = np.zeros(3*natm)
        v[d::3] = 1.0
        basis.append(v * sqrt_mass)
    # Rotations: sqrt(m_i) * (omega x r_i)
    axes = np.eye(3)
    for a in range(3):
        omega = axes[a]
        v = np.zeros((natm, 3))
        v[:, :] = np.cross(np.broadcast_to(omega, (natm, 3)), r)
        basis.append(v.reshape(-1) * sqrt_mass)

    B = np.column_stack(basis)  # (3N, 6)
    # Orthonormalize with SVD; drop near-null vectors (handles near-linear / COM issues)
    U, s, _ = npl.svd(B, full_matrices=False)
    smax = float(s.max()) if s.size else 0.0
    keep = s > (svd_tol * max(smax, 1.0))
    U = U[:, keep]
    rank = int(U.shape[1])

    P = np.eye(3*natm) - U @ U.T
    Wp = P @ W @ P
    Wp = 0.5 * (Wp + Wp.T)
    info = {
        "rt_rank": float(rank),
        "sv_min_kept": float(s[keep].min()) if np.any(keep) else 0.0,
        "sv_max": float(smax),
    }
    return Wp, info


def _mw_rt_projector_pyscf_like(W: np.ndarray, coords_bohr: np.ndarray, masses_amu: np.ndarray, thermo_mod,
                                svd_tol: float=1e-12) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Build RT projector using PySCF's thermo._get_TR convention (mass^0.5 scaling + principal axes).
    Produces a full-space projector P and returns Wp = P W P in mass-weighted Cartesian space.
    """
    if _pkg_harmonic is not None:
        return _pkg_harmonic.mw_rt_projector_pyscf_like(
            W,
            coords_bohr,
            masses_amu,
            thermo_mod,
            svd_tol=svd_tol,
        )

    coords_bohr = np.asarray(coords_bohr, float)
    masses_amu = np.asarray(masses_amu, float)
    natm = coords_bohr.shape[0]
    if W.shape != (3*natm, 3*natm):
        raise ValueError("W shape mismatch for pyscf-like RT projector")

    try:
        get_tr = getattr(thermo_mod, "_get_TR")
        rotation_const = getattr(thermo_mod, "rotation_const")
        get_rotor_type = getattr(thermo_mod, "_get_rotor_type")
    except Exception as exc:
        raise RuntimeError("PySCF thermo module lacks _get_TR/rotation_const/_get_rotor_type") from exc

    mass_center = np.einsum('z,zx->x', masses_amu, coords_bohr) / float(np.sum(masses_amu))
    coords = coords_bohr - mass_center

    TR = get_tr(masses_amu, coords)
    TRspace = []
    TRspace.append(TR[:3])  # translations

    rot_const = rotation_const(masses_amu, coords)
    rotor_type = get_rotor_type(rot_const)
    if rotor_type == 'ATOM':
        pass
    elif rotor_type == 'LINEAR':
        TRspace.append(TR[3:5])
    else:
        TRspace.append(TR[3:])

    A = np.vstack(TRspace)  # (nTR, 3N)
    # Orthonormalize (QR/SVD) and build full-space projector
    q, _ = npl.qr(A.T)  # (3N, nTR)
    # Drop near-dependent directions (numerical safety)
    # SVD on A gives singular values that indicate dependence.
    _, s, _ = npl.svd(A, full_matrices=False)
    smax = float(s.max()) if s.size else 0.0
    keep = s > (svd_tol * max(smax, 1.0))
    q = q[:, :int(np.sum(keep))]
    rank = int(q.shape[1])

    P = np.eye(3*natm) - q @ q.T
    Wp = P @ W @ P
    Wp = 0.5 * (Wp + Wp.T)
    info = {"rt_rank": float(rank), "sv_max": float(smax), "sv_min_kept": float(s[keep].min()) if np.any(keep) else 0.0}
    return Wp, info


def _try_analytic_hessian(mf):
    # Prefer explicit failure over silent algorithm changes; only treat "not available"
    # (missing attribute/import) as None so the caller can decide what to do.
    try:
        return mf.Hessian().kernel()
    except AttributeError:
        pass
    except Exception as exc:
        raise RuntimeError("Analytic Hessian computation failed") from exc

    try:
        from pyscf.hessian import rhf as h_rhf, rks as h_rks
    except Exception:
        return None
    try:
        if isinstance(mf, scf.hf.RHF):
            return h_rhf.Hessian(mf).kernel()
        if isinstance(mf, dft.rks.RKS):
            return h_rks.Hessian(mf).kernel()
    except AttributeError:
        return None
    except Exception as exc:
        raise RuntimeError("Analytic Hessian computation failed") from exc
    return None


def _grad_at(symbols, charge, spin, basis, method, use_df, auxbasis, xflat):
    xyz_ang = xflat.reshape(-1,3)*BOHR_TO_ANG
    pm = Molecule(symbols, xyz_ang, charge, spin)
    pmol2 = pm.as_pyscf(basis)
    mf2 = make_mean_field(pmol2, ESSettings(method=method, basis=basis, use_density_fit=use_df, auxbasis=auxbasis))
    # Use PySCF's version-agnostic API for nuclear gradients; fall back for older builds
    try:
        gvec = mf2.nuc_grad_method().kernel()
    except AttributeError:
        warn_once("grad_api_fallback", "Falling back to mf.Gradients().kernel() (mf.nuc_grad_method unavailable)")
        gvec = mf2.Gradients().kernel()
    return gvec.reshape(-1)


def _hessian_column_worker(task):
    j, x0, hh, symbols, charge, spin, basis, method, use_df, aux = task
    eye = np.zeros_like(x0); eye[j] = 1.0
    gp = _grad_at(symbols, charge, spin, basis, method, use_df, aux, x0 + hh*eye)
    gm = _grad_at(symbols, charge, spin, basis, method, use_df, aux, x0 - hh*eye)
    return j, (gp-gm)/(2*hh)


def _num_hessian_from_gradients(mol: Molecule, cfg: ESSettings, x0_bohr: np.ndarray, h: float=2e-3) -> np.ndarray:
    n = x0_bohr.size
    jobs = [(j, x0_bohr, h, mol.symbols, mol.charge, mol.spin,
             cfg.basis, cfg.method, cfg.use_density_fit, cfg.auxbasis)
            for j in range(n)]
    H = np.zeros((n,n))
    log(f"Finite-difference Hessian: computing {n} columns (step {h:g})")
    with _pool_executor() as ex:
        for idx, (j, col) in enumerate(ex.map(_hessian_column_worker, jobs), 1):
            H[:,j] = col
            _progress_update(idx, n, "Hessian columns")
    return H


def _as_cart_hessian(H, natm):
    if _pkg_harmonic is not None:
        return _pkg_harmonic.as_cart_hessian(H, natm)
    if H.ndim==2: return H
    if H.ndim==4 and H.shape==(natm,natm,3,3):
        return H.transpose(0,2,1,3).reshape(3*natm,3*natm)
    raise ValueError(f'Unexpected Hessian shape {H.shape}')


def _mass_weight(H_au: np.ndarray, masses_amu: np.ndarray) -> np.ndarray:
    if _pkg_harmonic is not None:
        return _pkg_harmonic.mass_weight(H_au, masses_amu)
    M = np.repeat(masses_amu,3)*AMU
    return H_au/np.sqrt(np.outer(M,M))

def _cart_to_hess4(Hc: np.ndarray, natm: int) -> np.ndarray:
    if _pkg_harmonic is not None:
        return _pkg_harmonic.cart_to_hess4(Hc, natm)
    Hc = np.asarray(Hc, float)
    if Hc.shape != (3*natm, 3*natm):
        raise ValueError("Hc shape mismatch")
    return Hc.reshape(natm, 3, natm, 3).transpose(0, 2, 1, 3)


def mass_weighted_freqs_modes(pmol: gto.Mole, H_au: np.ndarray, masses_amu: np.ndarray,
                              rtproj: str='pyscf', debug: bool=False):
    Hc = _as_cart_hessian(H_au, pmol.natm)
    W = _mass_weight(Hc, masses_amu)
    coords = pmol.atom_coords(unit='Bohr')

    natm = pmol.natm
    expected_total = 3 * natm
    expected_vib = expected_total - (6 if natm > 2 else 5)
    if debug:
        print(f"Harmonic diagnostics: natm={natm} | expected 3N={expected_total} | expected vib={expected_vib}")

    Wp = None
    vec = None

    thermo = None
    thermo_err = None
    try:
        from pyscf.hessian import thermo as _thermo
        thermo = _thermo
    except Exception as exc:
        thermo_err = exc

    if debug:
        w2_raw, _ = npl.eigh(W)
        _print_low_mode_summary("Raw mass-weighted Hessian W (no RT projection)", w2_raw, nshow=10)

    # If PySCF is available, print the relevant PySCF helper entrypoint.
    if thermo is not None and debug:
        try:
            import inspect
            print(f"PySCF thermo module: {getattr(thermo, '__file__', '(unknown)')}")
            print(f"PySCF harmonic_analysis signature: {inspect.signature(thermo.harmonic_analysis)}")
            try:
                src = inspect.getsource(thermo.harmonic_analysis)
                head = "\n".join(src.splitlines()[:30])
                print("PySCF harmonic_analysis source (first ~30 lines):")
                print(head)
            except Exception:
                warn_once("thermo_getsource", "Failed to fetch PySCF thermo.harmonic_analysis source for debug output")
        except Exception as exc:
            warn_once("thermo_inspect", f"Failed to inspect PySCF thermo helpers for debug output: {exc}")

        # Note: This PySCF build does not expose a standalone RT-projector helper.
        # `thermo.harmonic_analysis(...)` performs mass-weighting and TR removal internally.

    rtproj = (rtproj or 'pyscf').lower()
    if rtproj == 'pyscf':
        if thermo is None:
            msg = f"Requested --rtproj pyscf but pyscf.hessian.thermo unavailable ({thermo_err})"
            if STRICT:
                raise RuntimeError(msg)
            warn_once("rtproj_pyscf_missing", msg + "; falling back to --rtproj mw_explicit")
            rtproj = 'mw_explicit'
        else:
            # Use PySCF's internal TR basis convention to build a full-space projector in MW coordinates.
            Wp, info = _mw_rt_projector_pyscf_like(W, coords, masses_amu, thermo)
            if debug:
                print(f"PySCF-like RT projector: rt_rank={int(info['rt_rank'])} sv_max={info['sv_max']:.3e} sv_min_kept={info['sv_min_kept']:.3e}")
            w2, vec = npl.eigh(Wp)
            w2 = _handle_imaginary_modes(w2, natm=natm, rtproj=rtproj, strict=STRICT, rt_rank=int(info["rt_rank"]))
            if debug:
                _print_low_mode_summary("PySCF-like projected Hessian", w2, nshow=10)
                f_signed = _signed_freqs_from_evals(w2)
                print(f"  PySCF-like counts: |nu|<1={int(np.sum(np.abs(f_signed)<1.0))}  |nu|<10={int(np.sum(np.abs(f_signed)<10.0))}  printed(|nu|>1e-2)={int(np.sum(np.abs(f_signed)>1e-2))}")
            w2_pos = np.clip(w2, 0.0, None)
            freqs_cm = np.sqrt(w2_pos) * HARTREE_TO_CM
            return freqs_cm, vec

    if rtproj == 'none':
        Wp = W
    elif rtproj == 'mw_explicit':
        Wp, info = _mw_rt_projector_explicit(W, coords, masses_amu)
        if debug:
            print(f"Explicit MW RT projector: rt_rank={int(info['rt_rank'])} sv_max={info['sv_max']:.3e} sv_min_kept={info['sv_min_kept']:.3e}")
    else:
        raise ValueError(f"Unknown rtproj '{rtproj}'")

    w2, vec = npl.eigh(Wp)
    if rtproj == 'mw_explicit':
        w2 = _handle_imaginary_modes(w2, natm=natm, rtproj=rtproj, strict=STRICT, rt_rank=int(info["rt_rank"]))
    else:
        w2 = _handle_imaginary_modes(w2, natm=natm, rtproj=rtproj, strict=STRICT)
    if debug:
        _print_low_mode_summary("Final projected Hessian used for modes", w2, nshow=10)
        f_signed = _signed_freqs_from_evals(w2)
        print(f"  Final counts: |nu|<1={int(np.sum(np.abs(f_signed)<1.0))}  |nu|<10={int(np.sum(np.abs(f_signed)<10.0))}  printed(|nu|>1e-2)={int(np.sum(np.abs(f_signed)>1e-2))}")
    w2_pos = np.clip(w2, 0.0, None)
    freqs_cm = np.sqrt(w2_pos) * HARTREE_TO_CM
    return freqs_cm, vec


def harmonic_analysis(mol: Molecule, cfg: ESSettings, *, rtproj: str='pyscf', debug: bool=False) -> HarmonicResult:
    if _pkg_harmonic_workflow is not None:
        res = _pkg_harmonic_workflow.harmonic_analysis(mol, cfg, rtproj=rtproj, debug=debug)
        return HarmonicResult(res.freqs_cm, res.modes, res.zpe_cm)

    log(f"Generating PySCF molecule for basis {cfg.basis}")
    pmol = mol.as_pyscf(cfg.basis)
    log(f"Running mean-field calculation ({cfg.method}-{cfg.dispersion or ''})")
    mf = make_mean_field(pmol, cfg)

    if debug:
        # Stationarity check: gradient at the supplied geometry
        try:
            try:
                g = mf.nuc_grad_method().kernel()
            except AttributeError:
                warn_once("grad_api_fallback_stationarity", "Stationarity check: falling back to mf.Gradients().kernel() (mf.nuc_grad_method unavailable)")
                g = mf.Gradients().kernel()
            g = np.asarray(g, float)
            g_flat = g.reshape(-1)
            max_comp = float(np.max(np.abs(g_flat))) if g_flat.size else 0.0
            rms_comp = float(np.sqrt(np.mean(g_flat*g_flat))) if g_flat.size else 0.0
            g_atom = g.reshape(-1, 3)
            atom_norms = np.linalg.norm(g_atom, axis=1)
            max_atom = float(np.max(atom_norms)) if atom_norms.size else 0.0
            rms_atom = float(np.sqrt(np.mean(atom_norms*atom_norms))) if atom_norms.size else 0.0
            print("Geometry stationarity (gradient) in Eh/Bohr:")
            print(f"  max|g_comp|={max_comp:.3e}  rms|g_comp|={rms_comp:.3e}  max|g_atom|={max_atom:.3e}  rms|g_atom|={rms_atom:.3e}")
            print("  Heuristic: max|g_comp| <= 1e-4 (good), 1e-4–5e-4 (maybe), >5e-4 (likely non-stationary)")
        except Exception as exc:
            print(f"Geometry stationarity check (gradient) failed: {exc}")

    H = None
    try:
        H = _try_analytic_hessian(mf)
    except Exception as exc:
        msg = f"Analytic Hessian failed: {exc}"
        if getattr(cfg, "strict", STRICT):
            raise
        warn_once("analytic_hessian_failed", msg)
        H = None

    if H is None:
        msg = "Analytic Hessian unavailable; would fall back to finite-difference Hessian"
        if not getattr(cfg, "allow_fd_hessian", ALLOW_FD_HESSIAN):
            raise RuntimeError(msg + " (blocked; pass --allow-fd-hessian to proceed)")
        warn_once("fd_hessian", msg + " (enabled by --allow-fd-hessian)")
        # pmol.atom_coords() is Bohr by default; do not re-scale.
        x0 = pmol.atom_coords(unit='Bohr').reshape(-1)
        H = _num_hessian_from_gradients(mol, cfg, x0)
    else:
        log("Analytic Hessian obtained from PySCF")
    freqs_cm, modes = mass_weighted_freqs_modes(pmol, H, mol.analysis_masses(), rtproj=rtproj, debug=debug)
    zpe_cm = 0.5*np.sum(freqs_cm[freqs_cm>1e-5])
    return HarmonicResult(freqs_cm, modes, zpe_cm)

# -------------------- PES/DMS grids --------------------
@dataclass
class Bond:
    O: int
    H: int

def stretch_along_bond(coords: np.ndarray, bond: Bond, new_len_A: float) -> np.ndarray:
    A,B = bond.O, bond.H
    rA = coords[A]; rB = coords[B]
    v = rB-rA; L = npl.norm(v)
    if L<1e-10: raise ValueError('Zero bond length')
    u = v/L
    new = coords.copy(); new[B] = rA + u*new_len_A
    return new


def energy_dipole(mol: Molecule, cfg: ESSettings):
    pm = mol.as_pyscf(cfg.basis)
    mf = make_mean_field(pm, cfg)
    e = mf.e_tot
    dm = mf.make_rdm1()
    try:
        mu_au = mf.dip_moment(dm=dm, unit='au', verbose=0)
    except TypeError:
        warn_once("dip_moment_signature", "mf.dip_moment(dm=..., unit='au', verbose=0) unsupported; falling back to mf.dip_moment(unit='au', verbose=0)")
        mu_au = mf.dip_moment(unit='au', verbose=0)
    mu_debye = np.array(mu_au)*2.541746
    return e, mu_debye


def _grid_1d_worker(task):
    R, mol, cfg, bond = task
    coords = stretch_along_bond(mol.coords, bond, R)
    sub = Molecule(mol.symbols, coords, mol.charge, mol.spin)
    return energy_dipole(sub, cfg)


def grid_1d_pes_dms(mol: Molecule, cfg: ESSettings, bond: Bond, Rmin=0.75, Rmax=1.25, npts=41):
    if _pkg_scans is not None:
        return _pkg_scans.grid_1d_pes_dms(
            mol,
            cfg,
            bond,
            Rmin,
            Rmax,
            npts,
            energy_dipole_fn=energy_dipole,
            executor_factory=_pool_executor,
            molecule_factory=_legacy_molecule_with_coords,
            progress_fn=_progress_update,
            log_fn=log,
        )

    Rs = np.linspace(Rmin, Rmax, npts)
    E = np.zeros(npts); MU = np.zeros((npts,3))
    tasks = [(R, mol, cfg, bond) for R in Rs]
    log(f"Evaluating 1D grid: {npts} points from {Rmin:.2f} to {Rmax:.2f} Å")
    with _pool_executor() as ex:
        for idx,(e,mu) in enumerate(ex.map(_grid_1d_worker, tasks), 1):
            E[idx-1]=e; MU[idx-1]=mu
            _progress_update(idx, npts, "1D grid points")
    E -= E.min()
    return Rs, E, MU


def _grid_1d_normal_worker(task):
    s, mol, cfg, u_dir = task
    coords = mol.coords + s * u_dir
    sub = Molecule(mol.symbols, coords, mol.charge, mol.spin)
    return energy_dipole(sub, cfg)


def grid_1d_pes_dms_normal(mol: Molecule, cfg: ESSettings, u_dir: np.ndarray, smin=-0.15, smax=0.15, npts=41):
    if _pkg_scans is not None:
        return _pkg_scans.grid_1d_pes_dms_normal(
            mol,
            cfg,
            u_dir,
            smin,
            smax,
            npts,
            energy_dipole_fn=energy_dipole,
            executor_factory=_pool_executor,
            molecule_factory=_legacy_molecule_with_coords,
            progress_fn=_progress_update,
            log_fn=log,
        )

    S = np.linspace(smin, smax, npts)
    E = np.zeros(npts); MU = np.zeros((npts,3))
    tasks = [(s, mol, cfg, u_dir) for s in S]
    log(f"Evaluating 1D grid (normal-mode path): {npts} points from {smin:.2f} to {smax:.2f} Å")
    with _pool_executor() as ex:
        for idx,(e,mu) in enumerate(ex.map(_grid_1d_normal_worker, tasks), 1):
            E[idx-1]=e; MU[idx-1]=mu
            _progress_update(idx, npts, "1D grid points")
    E -= E.min()
    return S, E, MU


def calc_normal_mode_direction(mol: Molecule, cfg: ESSettings, bond: Bond) -> Tuple[np.ndarray, int, float, np.ndarray, np.ndarray]:
    """
    Selects the OH-like normal mode and returns:
    - u_dir: unit Cartesian displacement pattern (Å) for the chosen mode
    - kbest: mode index
    - f_cm: harmonic frequency (cm^-1)
    - modes: mass-weighted eigenvectors (3N x 3N)
    - freqs_cm: harmonic frequencies (cm^-1)
    """
    if _pkg_scans is not None:
        return _pkg_scans.calc_normal_mode_direction(
            mol,
            cfg,
            bond,
            harmonic_fn=harmonic_analysis,
            log_fn=log,
        )

    # Obtain mass-weighted modes at the input geometry and pick the donor OH-stretch-like mode
    res = harmonic_analysis(mol, cfg)
    modes = res.modes  # shape (3N, 3N), mass-weighted eigenvectors
    masses = mol.analysis_masses()
    mass_rep = np.repeat(masses, 3)
    natm = len(mol.symbols)
    axis_vec = mol.coords[bond.H] - mol.coords[bond.O]
    axis_norm = np.linalg.norm(axis_vec)
    if axis_norm < 1e-12:
        raise ValueError("Zero O–H axis length")
    axis_unit = axis_vec / axis_norm
    scores = []
    for k in range(modes.shape[1]):
        u_cart = (modes[:, k] / np.sqrt(mass_rep)).reshape(natm, 3)
        rel = u_cart[bond.H] - u_cart[bond.O]
        score = abs(float(np.dot(rel, axis_unit)))
        scores.append(score)
    kbest = int(np.argmax(scores))
    u_best = (modes[:, kbest] / np.sqrt(mass_rep)).reshape(natm, 3)
    norm = float(np.linalg.norm(u_best))
    if norm < 1e-14:
        raise RuntimeError("Normal-mode direction has near-zero norm")
    u_dir = u_best / norm
    log(f"Selected normal mode index {kbest} with freq {res.freqs_cm[kbest]:.1f} cm^-1 for O{bond.O}-H{bond.H} scan")
    return u_dir, kbest, float(res.freqs_cm[kbest]), modes, res.freqs_cm


def _energy_grad_at_cfg(symbols, charge, spin, cfg: ESSettings, xflat_bohr: np.ndarray):
    xyz_ang = xflat_bohr.reshape(-1,3)*BOHR_TO_ANG
    pm = Molecule(symbols, xyz_ang, charge, spin)
    pmol2 = pm.as_pyscf(cfg.basis)
    mf2 = make_mean_field(pmol2, cfg)
    e = mf2.e_tot
    # Use PySCF's version-agnostic API for nuclear gradients; fall back for older builds
    try:
        gvec = mf2.nuc_grad_method().kernel()
    except AttributeError:
        warn_once("grad_api_fallback_cfg", "Falling back to mf.Gradients().kernel() (mf.nuc_grad_method unavailable)")
        gvec = mf2.Gradients().kernel()
    return e, gvec.reshape(-1)


def _normal_relaxed_point(
    mol: Molecule,
    cfg: ESSettings,
    u_dir: np.ndarray,
    s: float,
    gtol: float,
    maxiter: int,
):
    from pyscf_vscf.backends.pyscf import normal_relaxed_point

    return normal_relaxed_point(mol, cfg, u_dir, s, gtol, maxiter)


def grid_1d_pes_dms_normal_relaxed(
    mol: Molecule,
    cfg: ESSettings,
    u_dir: np.ndarray,
    smin=-0.15,
    smax=0.15,
    npts=41,
    gtol: float = 1e-4,
    maxiter: int = 100,
):
    if _pkg_scans is None:
        raise RuntimeError("Exact normal-relaxed scans require the installed pyscf-vscf package")
    return _pkg_scans.grid_1d_pes_dms_normal_relaxed(
        mol,
        cfg,
        u_dir,
        smin,
        smax,
        npts,
        relaxed_point_fn=_normal_relaxed_point,
        executor_factory=_pool_executor,
        progress_fn=_progress_update,
        log_fn=log,
        gtol=gtol,
        maxiter=maxiter,
    )


def _grid_2d_worker(task):
    i, j, mol, cfg, b1, b2, R1_val, R2_val = task
    c1 = stretch_along_bond(mol.coords, b1, R1_val)
    c2 = stretch_along_bond(c1, b2, R2_val)
    sub = Molecule(mol.symbols, c2, mol.charge, mol.spin)
    e, mu = energy_dipole(sub, cfg)
    return i, j, e, mu


def grid_2d_pes_dms(mol: Molecule, cfg: ESSettings, b1: Bond, b2: Bond, R1: np.ndarray, R2: np.ndarray):
    if _pkg_scans is not None:
        return _pkg_scans.grid_2d_pes_dms(
            mol,
            cfg,
            b1,
            b2,
            R1,
            R2,
            energy_dipole_fn=energy_dipole,
            executor_factory=_pool_executor,
            molecule_factory=_legacy_molecule_with_coords,
            progress_fn=_progress_update,
            log_fn=log,
        )

    n1,n2 = len(R1), len(R2)
    E = np.zeros((n1,n2)); MU = np.zeros((n1,n2,3))
    tasks = [(i, j, mol, cfg, b1, b2, R1[i], R2[j])
             for i in range(n1) for j in range(n2)]
    total = len(tasks)
    log(f"Evaluating 2D grid: {n1}x{n2} = {total} points")
    with _pool_executor() as ex:
        for idx,(i,j,e,mu) in enumerate(ex.map(_grid_2d_worker, tasks), 1):
            E[i,j]=e; MU[i,j]=mu
            _progress_update(idx, total, "2D grid points")
    E -= E.min()
    return R1, R2, E, MU

# -------------------- DVR & intensities --------------------
@dataclass
class DVR1D:
    R: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray

def sinc_dvr_1d(R: np.ndarray, mu_red_amu: float, V_Eh: np.ndarray) -> DVR1D:
    # Colbert–Miller sinc DVR kinetic + diagonal potential
    N = len(R)
    x = R*ANG_TO_BOHR
    dx = (x[-1] - x[0]) / (N - 1)
    mu = mu_red_amu*AMU
    T = np.empty((N,N))
    coef = 1.0/(2.0*mu*dx*dx)
    for i in range(N):
        for j in range(N):
            if i == j:
                T[i, i] = coef*(np.pi**2/3.0)
            else:
                n = i - j
                T[i, j] = coef*(2.0*((-1.0)**n)/(n*n))
    H = T + np.diag(V_Eh)
    evals, evecs = npl.eigh(H)
    return DVR1D(R, evals, evecs)


def trans_mu_1d(dvr: DVR1D, mu_of_R: Callable[[np.ndarray], np.ndarray], v: int) -> float:
    psi0 = dvr.evecs[:,0]; psiv = dvr.evecs[:,v]
    mu = mu_of_R(dvr.R)
    # In an orthonormal DVR basis, <0|mu(x)|v> reduces to a weighted dot product
    # with uniform quadrature weights (no dx factor).
    return float(np.dot(psi0, mu * psiv))


def sigma_int(
    mu_Cm: float,
    frequency_cm: float,
    orientation_factor: float = 1.0 / 3.0,
) -> float:
    omega = 2.0 * np.pi * C_LIGHT_M_S * 100.0 * float(frequency_cm)
    return (
        np.pi
        * omega
        * float(orientation_factor)
        * mu_Cm**2
        / (EPS0 * HBAR * C_LIGHT_M_S)
    )


def _parse_intensity_mode(mode: str) -> str:
    mode = (mode or "axis").lower()
    if mode not in {"axis", "vector", "both"}:
        raise ValueError(f"Unknown intensity mode '{mode}' (expected axis|vector|both)")
    return mode


def variational_1d(
    R,
    E,
    MU,
    redmass_amu: float,
    axis: Optional[np.ndarray] = None,
    vmax: int = 8,
    *,
    intensity: str = "axis",
) -> List[Dict]:
    dvr = sinc_dvr_1d(R, redmass_amu, E)
    if VERBOSE:
        try:
            psi0 = dvr.evecs[:,0]
            psi1 = dvr.evecs[:,1] if dvr.evecs.shape[1] > 1 else None
            edge0 = 0.5*(abs(psi0[0]) + abs(psi0[-1]))
            edge1 = 0.5*(abs(psi1[0]) + abs(psi1[-1])) if psi1 is not None else 0.0
            log(f"Edge amplitudes |psi0|: {edge0:.3e} |psi1|: {edge1:.3e}")
            if max(edge0, edge1) > 1e-2:
                log("DVR warning: wavefunction amplitude at box edges is large; increase npts or scan window for accuracy")
        except Exception as exc:
            warn_once("edge_amp_diag", f"Edge-amplitude diagnostic failed (continuing): {exc}")
    if axis is None:
        axis_vec = np.array([0.0, 0.0, 1.0])
    else:
        axis_vec = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis_vec)
    if norm < 1e-12:
        raise ValueError("Dipole projection axis must be non-zero")
    axis_unit = axis_vec / norm
    intensity = _parse_intensity_mode(intensity)

    mux = lambda rr: np.interp(rr, R, MU[:, 0])
    muy = lambda rr: np.interp(rr, R, MU[:, 1])
    muz = lambda rr: np.interp(rr, R, MU[:, 2])
    mu_axis_of_R = lambda rr: axis_unit[0] * mux(rr) + axis_unit[1] * muy(rr) + axis_unit[2] * muz(rr)
    out = []
    e0 = dvr.evals[0]
    for v in range(1, min(vmax+1, len(dvr.evals))):
        nu_cm = (dvr.evals[v]-e0)*HARTREE_TO_CM
        mu_axis = trans_mu_1d(dvr, mu_axis_of_R, v)
        mu_vec = None
        if intensity in {"vector", "both"}:
            mx = trans_mu_1d(dvr, mux, v)
            my = trans_mu_1d(dvr, muy, v)
            mz = trans_mu_1d(dvr, muz, v)
            mu_vec = float(np.sqrt(mx * mx + my * my + mz * mz))

        if intensity == "axis":
            mu_use = mu_axis
        else:
            assert mu_vec is not None
            mu_use = mu_vec

        orientation_factor = 1.0 if intensity == "axis" else 1.0 / 3.0
        rec = {
            'v': v,
            'freq_cm': nu_cm,
            'mu_tr_arb': mu_use,
            'sigma_int_m2s': sigma_int(
                mu_use * DEBYE_TO_C_M, nu_cm, orientation_factor
            ),
        }
        if intensity == "both":
            rec.update(
                {
                    "mu_tr_axis_arb": mu_axis,
                    "sigma_int_axis_m2s": sigma_int(
                        mu_axis * DEBYE_TO_C_M, nu_cm, 1.0
                    ),
                    "mu_tr_vec_arb": mu_vec,
                    "sigma_int_vec_m2s": sigma_int(
                        mu_vec * DEBYE_TO_C_M, nu_cm, 1.0 / 3.0
                    ),
                }
            )
        out.append(rec)
    return out

# 2D product DVR
@dataclass
class DVR2D:
    R1: np.ndarray
    R2: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray  # flattened grid eigenvectors (N1*N2, k)

def product_dvr_2d(
    R1: np.ndarray,
    R2: np.ndarray,
    mu1_amu: float,
    mu2_amu: float,
    V: np.ndarray,
    *,
    k_eigs: Optional[int] = None,
    g12_inv_amu: float = 0.0,
) -> DVR2D:
    # Build kinetic via 2D product sinc DVR. Default is separable reduced-mass KEO.
    # If g12_inv_amu != 0, include a constant cross-derivative term:
    #   T = -(1/2)[ g11 d2/dr1^2 + g22 d2/dr2^2 + 2 g12 d2/(dr1 dr2) ]
    # where g11 = 1/mu1, g22 = 1/mu2, g12 = g12_inv_amu / AMU, and r are in Bohr.
    def _sinc_d1_d2(R):
        N = len(R)
        x = R*ANG_TO_BOHR
        dx = (x[-1] - x[0])/(N - 1)
        D1 = np.empty((N, N))
        D2 = np.empty((N, N))
        for i in range(N):
            for j in range(N):
                if i == j:
                    D1[i, i] = 0.0
                    D2[i, i] = -(np.pi**2/3.0)/(dx*dx)
                else:
                    n = i - j
                    D1[i, j] = ((-1.0)**n) / (dx * n)
                    D2[i, j] = -2.0 * (((-1.0)**n) / (dx*dx * n*n))
        return D1, D2

    D1_1, D2_1 = _sinc_d1_d2(R1)
    D1_2, D2_2 = _sinc_d1_d2(R2)

    n1 = len(R1)
    n2 = len(R2)
    eye1 = sparse.eye(n1, format="csr")
    eye2 = sparse.eye(n2, format="csr")

    g11 = 1.0 / (float(mu1_amu) * AMU)
    g22 = 1.0 / (float(mu2_amu) * AMU)
    g12 = float(g12_inv_amu) / AMU

    T = (-0.5 * g11) * sparse.kron(sparse.csr_matrix(D2_1), eye2) + (-0.5 * g22) * sparse.kron(eye1, sparse.csr_matrix(D2_2))
    if abs(g12) > 1e-18:
        # Cross term: -(1/2) * (2 g12 d2/(dr1 dr2)) = -g12 (d/dr1)(d/dr2)
        T = T + (-g12) * sparse.kron(sparse.csr_matrix(D1_1), sparse.csr_matrix(D1_2))

    H = T
    H = H + sparse.diags(V.ravel())
    dim = int(H.shape[0])
    if k_eigs is None:
        # Historical default: compute a small number of low-lying states (used by toy-model validation scripts).
        k = min(12, dim - 1)
        if k < 2:
            raise ValueError(f"DVR Hamiltonian dimension is {dim}; cannot compute at least 2 eigenpairs")
    else:
        k = int(k_eigs)
        if k < 2:
            raise ValueError(f"Need at least 2 eigenpairs (ground + 1 excited); got k_eigs={k}")
        if k >= dim:
            raise ValueError(
                f"Requested k_eigs={k} but DVR Hamiltonian dimension is {dim}; reduce --vmax/--nmax or grid size."
            )
    evals, evecs = eigsh(H, k=k, which='SA')
    idx = np.argsort(evals); evals=evals[idx]; evecs=evecs[:,idx]
    return DVR2D(R1,R2,evals,evecs)


def trans_mu_2d(dvr: DVR2D, mu_proj_grid: np.ndarray, n: int) -> float:
    N1,N2 = len(dvr.R1), len(dvr.R2)
    psi0 = dvr.evecs[:,0].reshape(N1,N2)
    psin = dvr.evecs[:,n].reshape(N1,N2)
    # Product DVR is orthonormal; matrix element is simple sum over grid points.
    return float(np.dot(psi0.reshape(-1), (mu_proj_grid * psin).reshape(-1)))


def variational_2d(
    R1,
    R2,
    E,
    MU,
    mu1_amu,
    mu2_amu,
    axis: Optional[np.ndarray] = None,
    nmax=8,
    *,
    g12_inv_amu: float = 0.0,
    intensity: str = "axis",
) -> List[Dict]:
    # Need at least (nmax + 1) eigenpairs (ground + nmax excited states).
    dvr = product_dvr_2d(R1, R2, mu1_amu, mu2_amu, E, k_eigs=int(nmax) + 1, g12_inv_amu=float(g12_inv_amu))
    if axis is None:
        axis_vec = np.array([0.0, 0.0, 1.0])
    else:
        axis_vec = np.array(axis, dtype=float)
    norm = np.linalg.norm(axis_vec)
    if norm < 1e-12:
        raise ValueError("Dipole projection axis must be non-zero")
    axis_unit = axis_vec / norm
    intensity = _parse_intensity_mode(intensity)
    # MU is already evaluated on the DVR grid, so no interpolation is needed.
    mu_x = MU[:, :, 0]
    mu_y = MU[:, :, 1]
    mu_z = MU[:, :, 2]
    mu_proj = axis_unit[0] * mu_x + axis_unit[1] * mu_y + axis_unit[2] * mu_z
    e0 = dvr.evals[0]; out=[]
    for n in range(1, min(nmax+1, len(dvr.evals))):
        nu_cm = (dvr.evals[n]-e0)*HARTREE_TO_CM
        mu_axis = trans_mu_2d(dvr, mu_proj, n)
        mu_vec = None
        if intensity in {"vector", "both"}:
            mx = trans_mu_2d(dvr, mu_x, n)
            my = trans_mu_2d(dvr, mu_y, n)
            mz = trans_mu_2d(dvr, mu_z, n)
            mu_vec = float(np.sqrt(mx * mx + my * my + mz * mz))

        if intensity == "axis":
            mu_use = mu_axis
        else:
            assert mu_vec is not None
            mu_use = mu_vec

        orientation_factor = 1.0 if intensity == "axis" else 1.0 / 3.0
        rec = {
            'n': n,
            'freq_cm': nu_cm,
            'mu_tr_arb': mu_use,
            'sigma_int_m2s': sigma_int(
                mu_use * DEBYE_TO_C_M, nu_cm, orientation_factor
            ),
        }
        if intensity == "both":
            rec.update(
                {
                    "mu_tr_axis_arb": mu_axis,
                    "sigma_int_axis_m2s": sigma_int(
                        mu_axis * DEBYE_TO_C_M, nu_cm, 1.0
                    ),
                    "mu_tr_vec_arb": mu_vec,
                    "sigma_int_vec_m2s": sigma_int(
                        mu_vec * DEBYE_TO_C_M, nu_cm, 1.0 / 3.0
                    ),
                }
            )
        out.append(rec)
    return out

# -------------------- CLI tasks --------------------
def _variational_helpers():
    try:
        from pyscf_vscf.variational import (
            parse_intensity_mode as parse_mode,
            variational_1d as pkg_variational_1d,
            variational_2d as pkg_variational_2d,
        )
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("pyscf_vscf"):
            raise RuntimeError(
                "The legacy driver now requires the installed pyscf-vscf package "
                "for validated variational intensities and state assignment"
            ) from exc
        raise
    return parse_mode, pkg_variational_1d, pkg_variational_2d


def parse_bond(s: str) -> Bond:
    s=s.upper(); o=int(s.split('O')[1].split('-')[0]); h=int(s.split('H')[1]); return Bond(o,h)

def run_harmonic(mol: Molecule, cfg: ESSettings):
    log("Harmonic analysis: building frequencies and ZPE")
    rtproj = getattr(cfg, "rtproj", "pyscf")
    res = harmonic_analysis(mol, cfg, rtproj=rtproj, debug=VERBOSE)
    print(f"ZPE (harmonic): {res.zpe_cm:.2f} cm^-1")
    print("Frequencies (cm^-1):")
    print(" ".join(f"{f:.1f}" for f in res.freqs_cm[res.freqs_cm>1e-2]))

def _opt_kwargs_for_profile(profile: str) -> Dict[str, float]:
    profile = (profile or "orca").lower()
    if profile in ("orca", "orca-tight", "orca_tight"):
        # Match the ORCA Opt thresholds used in your AiiDA job input (Eh/Bohr).
        # geomeTRIC uses convergence_{gmax,grms} in Eh/Bohr.
        return {
            "convergence_gmax": 4.5e-5,
            "convergence_grms": 3.0e-5,
        }
    raise ValueError(f"Unknown --opt-conv profile '{profile}'")


def run_opt(mol: Molecule, cfg: ESSettings, *, opt_out: Optional[Path], opt_maxsteps: int, opt_conv: str):
    if _pkg_optimization is not None:
        _pkg_optimization.run_opt(
            mol,
            cfg,
            opt_out=opt_out,
            opt_maxsteps=opt_maxsteps,
            opt_conv=opt_conv,
            verbose=VERBOSE,
            log_fn=log,
            warn_fn=warn_once,
        )
        return

    log("Geometry optimization: setting up PySCF molecule and mean-field")
    pmol = mol.as_pyscf(cfg.basis)
    mf = make_mean_field(pmol, cfg)
    # Let PySCF/geomeTRIC show iteration info only when -v is set
    mf.verbose = 4 if VERBOSE else 0

    from pyscf.geomopt import geometric_solver

    kwargs: Dict[str, float] = {}
    kwargs.update(_opt_kwargs_for_profile(opt_conv))
    if opt_maxsteps is not None:
        kwargs["maxsteps"] = int(opt_maxsteps)

    log(f"Starting geometry optimization (backend=geomeTRIC; maxsteps={int(opt_maxsteps)}; conv='{opt_conv}')")
    converged, optmol = geometric_solver.kernel(mf, **kwargs)
    if not converged:
        msg = "Geometry optimization did not converge within the allowed steps"
        if getattr(cfg, "strict", STRICT):
            raise RuntimeError(msg)
        warn_once("opt_not_converged", msg)

    coords_opt = np.asarray(optmol.atom_coords(unit='Angstrom'), float)
    mol_opt = Molecule(mol.symbols, coords_opt, mol.charge, mol.spin, label=mol.label, masses=mol.masses)

    # Stationarity check at the optimized geometry
    try:
        pmol2 = mol_opt.as_pyscf(cfg.basis)
        mf2 = make_mean_field(pmol2, cfg)
        g = mf2.nuc_grad_method().kernel()
        g = np.asarray(g, float).reshape(-1)
        max_comp = float(np.max(np.abs(g))) if g.size else 0.0
        rms_comp = float(np.sqrt(np.mean(g*g))) if g.size else 0.0
        print("Optimized-geometry gradient check (Eh/Bohr):")
        print(f"  max|g_comp|={max_comp:.3e}  rms|g_comp|={rms_comp:.3e}")
    except Exception as exc:
        warn_once("opt_grad_check_failed", f"Post-opt gradient check failed: {exc}")

    if opt_out is None:
        opt_out = Path(f"{mol.label}.pyscf_opt.mmol")
    opt_out = Path(opt_out)
    suffix = opt_out.suffix.lower()
    if suffix == ".xyz":
        write_xyz(opt_out, mol_opt, comment=f"{mol.label} (PySCF optimized)")
    elif suffix == ".mmol":
        write_midas_mmol(opt_out, mol_opt, title=f"{mol.label} (PySCF optimized)")
    else:
        raise ValueError(f"--opt-out must end with .xyz or .mmol (got '{opt_out.name}')")
    print(f"Optimized geometry written to: {opt_out}", flush=True)


def run_1d(
    mol: Molecule,
    cfg: ESSettings,
    bond: Bond,
    rmin,
    rmax,
    npts,
    vmax,
    tight_scan: bool = False,
    tight_width: float = 0.30,
    scan: str = 'lbs-frozen',
    *,
    smin: float = -0.15,
    smax: float = 0.15,
    dump_grid: Optional[Path] = None,
    reuse_grid: Optional[Path] = None,
    intensity: str = "axis",
):
    axis_vec = mol.coords[bond.H] - mol.coords[bond.O]

    if scan == 'lbs-frozen':
        if tight_scan:
            r_eq = np.linalg.norm(mol.coords[bond.H] - mol.coords[bond.O])
            half = 0.5*max(float(tight_width), 0.30)
            rmin = r_eq - half
            rmax = r_eq + half
            log(f"Tight scan enabled: r_eq={r_eq:.4f} Å → window [{rmin:.4f}, {rmax:.4f}] Å (full width {2*half:.3f} Å)")
        log(f"Constructing 1D PES/DMS (frozen LBS) for bond O{bond.O}-H{bond.H} with {npts} points from {rmin:.2f} to {rmax:.2f} Å")
        if reuse_grid is not None:
            if _pkg_scans is None:
                raise RuntimeError("Schema-v2 cache reuse requires the installed package")
            R, E, MU = _pkg_scans.load_lbs_frozen_1d_grid_cache(
                reuse_grid, mol, cfg, bond, rmin, rmax, npts, scan=scan
            )
            log(f"Reusing cached 1D grid from {reuse_grid}")
        else:
            if dump_grid is not None and dump_grid.suffix.lower() != ".npz":
                raise ValueError("--dump-grid must end with .npz")
            R, E, MU = grid_1d_pes_dms(mol, cfg, bond, rmin, rmax, npts)
            if dump_grid is not None:
                if _pkg_scans is None:
                    raise RuntimeError("Schema-v2 cache writing requires the installed package")
                _pkg_scans.dump_lbs_frozen_1d_grid_cache(
                    dump_grid, mol, cfg, bond, rmin, rmax, npts, R, E, MU, scan=scan
                )
                log(f"Wrote 1D grid cache: {dump_grid}")
        # Diatomic reduced mass for O–X stretch
        mO = MASS['O']; mX = MASS['D'] if mol.symbols[bond.H].upper()=='D' else MASS['H']
        mu = mO*mX/(mO+mX)

    elif scan in ('normal', 'normal-relaxed'):
        if dump_grid is not None or reuse_grid is not None:
            raise NotImplementedError("Grid caching is only implemented for --scan lbs-frozen in --task 1d.")
        # Select OH-like normal mode and build grid along that direction
        u_dir, kbest, f_cm, modes, freqs_cm = calc_normal_mode_direction(mol, cfg, bond)
        if tight_scan:
            half = 0.5*max(float(tight_width), 0.30)
            smin = -half
            smax = +half
            log(f"Tight scan (normal) enabled: window [{smin:.4f}, {smax:.4f}] Å (full width {2*half:.3f} Å)")
        else:
            smin = float(smin)
            smax = float(smax)
            if not smin < 0.0 < smax:
                raise ValueError("Normal-coordinate bounds must satisfy --smin < 0 < --smax")
        if scan == 'normal':
            log(f"Constructing 1D PES/DMS (normal-mode path) with {npts} points from {smin:.2f} to {smax:.2f} Å")
            R, E, MU = grid_1d_pes_dms_normal(mol, cfg, u_dir, smin, smax, npts)
        else:
            log(f"Constructing 1D PES/DMS (normal-mode constrained relax) with {npts} points from {smin:.2f} to {smax:.2f} Å")
            R, E, MU = grid_1d_pes_dms_normal_relaxed(mol, cfg, u_dir, smin, smax, npts)
        # Effective mass for the chosen path: μ_eff = Σ m_i ||u_i||^2  (u_dir is unit-normalized)
        masses = mol.analysis_masses()
        mu = float(np.sum(masses * np.sum(u_dir*u_dir, axis=1)))
        log(f"Normal-mode scan using mode {kbest} (harmonic {f_cm:.1f} cm^-1); μ_eff = {mu:.6f} amu")

    elif scan == 'lbs-relaxed':
        raise NotImplementedError("Scan type 'lbs-relaxed' is not implemented yet.")
    else:
        raise ValueError(f"Unknown --scan '{scan}'")

    parse_intensity_mode, variational_1d_impl, _ = _variational_helpers()
    spec = variational_1d_impl(
        R,
        E,
        MU,
        mu,
        axis=axis_vec,
        vmax=vmax,
        intensity=intensity,
    )

    if scan in ('normal', 'normal-relaxed'):
        if len(spec) >= 1:
            print(f"Harmonic ν (mode {kbest}) = {f_cm:.1f} cm^-1; DVR v=1 = {spec[0]['freq_cm']:.1f} cm^-1; μ_eff = {mu:.6f} amu")
        else:
            print(f"Harmonic ν (mode {kbest}) = {f_cm:.1f} cm^-1; μ_eff = {mu:.6f} amu")

    intensity = parse_intensity_mode(intensity)
    if intensity == "both":
        print("v  nu/cm^-1   mu_axis/D   int_axis/domega (m^2/s)   |mu|/D   int_iso/domega (m^2/s)")
        for s in spec:
            print(
                f"{s['v']:>1d}  {s['freq_cm']:8.1f}   {s['transition_dipole_axis_D']:>12.4e}   "
                f"{s['integrated_cross_section_axis_omega_m2_per_s']:>12.4e}   "
                f"{s['transition_dipole_norm_D']:>12.4e}   "
                f"{s['integrated_cross_section_isotropic_omega_m2_per_s']:>12.4e}"
            )
    else:
        print("v  nu/cm^-1   mu/D   integral sigma(omega)domega (m^2/s)   orientation")
        for s in spec:
            print(
                f"{s['v']:>1d}  {s['freq_cm']:8.1f}   {s['transition_dipole_D']:>12.4e}   "
                f"{s['integrated_cross_section_omega_m2_per_s']:>12.4e}   {s['orientation']}"
            )


def run_2d(
    mol: Molecule,
    cfg: ESSettings,
    b1: Bond,
    b2: Bond,
    r1,
    r2,
    nmax,
    *,
    keo: str = "gmatrix",
    dump_grid: Optional[Path] = None,
    reuse_grid: Optional[Path] = None,
    intensity: str = "axis",
):
    log(f"Constructing 2D PES/DMS for bonds O{b1.O}-H{b1.H} and O{b2.O}-H{b2.H}")
    if dump_grid is not None and dump_grid.suffix.lower() != ".npz":
        raise ValueError("--dump-grid must end with .npz")
    R1_req = np.linspace(*r1)
    R2_req = np.linspace(*r2)
    if reuse_grid is not None:
        if _pkg_scans is None:
            raise RuntimeError("Schema-v2 cache reuse requires the installed package")
        R1, R2, E, MU = _pkg_scans.load_lbs_frozen_2d_grid_cache(
            reuse_grid, mol, cfg, b1, b2, r1, r2, keo=keo
        )
        log(f"Reusing cached 2D grid from {reuse_grid}")
    else:
        R1 = R1_req
        R2 = R2_req
        R1, R2, E, MU = grid_2d_pes_dms(mol, cfg, b1, b2, R1, R2)
        if dump_grid is not None:
            if _pkg_scans is None:
                raise RuntimeError("Schema-v2 cache writing requires the installed package")
            meta = _pkg_scans.lbs_frozen_2d_cache_metadata(
                mol, cfg, b1, b2, r1, r2, keo=keo
            )
            dump_grid_npz(dump_grid, meta=meta, arrays={"R1_A": R1, "R2_A": R2, "E_Eh": E, "MU_Debye": MU})
            log(f"Wrote 2D grid cache: {dump_grid}")
    mO = MASS['O']
    mX1 = MASS['D'] if mol.symbols[b1.H].upper()=='D' else MASS['H']
    mX2 = MASS['D'] if mol.symbols[b2.H].upper()=='D' else MASS['H']
    mu1 = mO*mX1/(mO+mX1); mu2 = mO*mX2/(mO+mX2)
    axis_vec = mol.coords[b1.H] - mol.coords[b1.O]
    keo_lc = (keo or "gmatrix").lower()
    if keo_lc not in {"reduced", "gmatrix"}:
        raise ValueError(f"Unknown --keo '{keo}' (expected reduced|gmatrix)")

    use_gmatrix = (keo_lc == "gmatrix")
    g12_inv_amu = 0.0
    if use_gmatrix and int(b1.O) == int(b2.O):
        # For valence coordinates r1=|O-H1|, r2=|O-H2| at fixed directions, the Wilson G element is:
        #   g12 = (u1·u2)/mO   (in 1/mass units). We store g12_inv_amu = (u1·u2)/mO_amu.
        u1 = mol.coords[b1.H] - mol.coords[b1.O]
        u2 = mol.coords[b2.H] - mol.coords[b2.O]
        n1 = float(np.linalg.norm(u1))
        n2 = float(np.linalg.norm(u2))
        if n1 < 1e-12 or n2 < 1e-12:
            raise ValueError("Invalid bond geometry for gmatrix KEO (zero bond vector norm)")
        cos_th = float(np.dot(u1, u2) / (n1 * n2))
        # Clamp tiny numerical drift
        if cos_th > 1.0:
            cos_th = 1.0
        if cos_th < -1.0:
            cos_th = -1.0
        g12_inv_amu = cos_th / float(mO)
    elif keo_lc == "gmatrix":
        # gmatrix with no shared atom is identical to reduced-mass separable KEO in this frozen-coordinate model.
        g12_inv_amu = 0.0

    parse_intensity_mode, _, variational_2d_impl = _variational_helpers()
    spec = variational_2d_impl(
        R1,
        R2,
        E,
        MU,
        mu1,
        mu2,
        axis=axis_vec,
        nmax=nmax,
        g12_inv_amu=g12_inv_amu,
        intensity=intensity,
    )
    intensity = parse_intensity_mode(intensity)
    if intensity == "both":
        print("n assignment weight  nu/cm^-1   mu_axis/D   int_axis/domega (m^2/s)   |mu|/D   int_iso/domega (m^2/s)")
        for s in spec:
            print(
                f"{s['n']:>1d} {str(s['assignment']):>10s} {s['assignment_weight']:6.3f} "
                f"{s['freq_cm']:8.1f}   {s['transition_dipole_axis_D']:>12.4e}   "
                f"{s['integrated_cross_section_axis_omega_m2_per_s']:>12.4e}   "
                f"{s['transition_dipole_norm_D']:>12.4e}   "
                f"{s['integrated_cross_section_isotropic_omega_m2_per_s']:>12.4e}"
            )
    else:
        print("n assignment weight  nu/cm^-1   mu/D   integral sigma(omega)domega (m^2/s)   orientation")
        for s in spec:
            print(
                f"{s['n']:>1d} {str(s['assignment']):>10s} {s['assignment_weight']:6.3f} "
                f"{s['freq_cm']:8.1f}   {s['transition_dipole_D']:>12.4e}   "
                f"{s['integrated_cross_section_omega_m2_per_s']:>12.4e}   {s['orientation']}"
            )


def _es_default(field: str):
    return getattr(ESSettings, field)


def main():
    global WORKERS, THREADS_PER_WORKER, VERBOSE, DEV_FAST, MAX_PARALLEL, PES_WORKERS, STRICT, ALLOW_FD_HESSIAN
    ap = argparse.ArgumentParser()
    ap.add_argument('--mmol', type=Path)
    ap.add_argument('--xyz', type=Path)
    ap.add_argument('--task', choices=['harmonic','opt','1d','2d'], default='harmonic')
    ap.add_argument('--basis', default=_es_default('basis'))
    ap.add_argument('--method', default=_es_default('method'))
    ap.add_argument(
        '--dispersion',
        choices=['d3', 'd4', 'none'],
        default='d4',
        help="DFT dispersion correction via PySCF's optional pyscf-dispersion package (default d4). Use 'none' to disable.",
    )
    ap.add_argument('--rtproj', choices=['pyscf','mw_explicit','none'], default=_es_default('rtproj'),
                    help="RT projection for harmonic frequencies: 'pyscf' (default), 'mw_explicit', or 'none'")
    ap.add_argument('--no-ri', dest='use_ri', action='store_false', help='Disable RI density fitting (enabled by default)')
    ap.add_argument('--ri-aux', dest='ri_aux', help="Auxiliary basis for RI density fitting (default 'weigend+etb')")
    ap.add_argument('--scf-conv-tol', type=float, default=None, help="Override SCF convergence tolerance (PySCF mf.conv_tol)")
    ap.add_argument('--scf-max-cycle', type=int, default=None, help="Override maximum PySCF SCF cycles")
    ap.add_argument('--dft-grid-level', type=int, default=None, help="Override PySCF DFT grid level (mf.grids.level)")
    ap.add_argument('--bond', default='O0-H1')
    ap.add_argument('--scan', choices=['lbs-frozen','normal','normal-relaxed','lbs-relaxed'], default='lbs-frozen', help='Choose 1D scan type')
    ap.add_argument('--bond2')
    ap.add_argument(
        '--keo',
        choices=['reduced', 'gmatrix'],
        default='gmatrix',
        help="For --task 2d: kinetic energy operator (default gmatrix).",
    )
    ap.add_argument('--rmin', type=float, default=0.75)
    ap.add_argument('--rmax', type=float, default=1.25)
    ap.add_argument('--smin', type=float, default=-0.15)
    ap.add_argument('--smax', type=float, default=0.15)
    ap.add_argument('--npts', type=int, default=41)
    ap.add_argument('--vmax', type=int, default=8)
    ap.add_argument('--dump-grid', type=Path, default=None, help="For 1d/2d tasks: write PES/DMS grid to a .npz cache file")
    ap.add_argument('--reuse-grid', type=Path, default=None, help="For 1d/2d tasks: load PES/DMS grid from a .npz cache file (skip PySCF point farming)")
    ap.add_argument(
        '--intensity',
        choices=['axis', 'vector', 'both'],
        default='axis',
        help="Transition-dipole convention for variational intensities: axis (bond1-projected), vector (|μ⃗|), or both (default axis).",
    )
    ap.add_argument('--opt-out', type=Path, default=None, help="For --task opt: output path (.mmol or .xyz). Default: <label>.pyscf_opt.mmol")
    ap.add_argument('--opt-maxsteps', type=int, default=50, help="For --task opt: maximum geometry steps (default 50)")
    ap.add_argument(
        '--opt-conv',
        choices=['orca'],
        default='orca',
        help="For --task opt: convergence profile (default 'orca')",
    )
    ap.add_argument(
        '--max-parallel',
        type=int,
        default=MAX_PARALLEL,
        help="Max parallel execution units budget (default 8). Used to derive threads/workers.",
    )
    ap.add_argument(
        '--pes-workers',
        type=int,
        default=PES_WORKERS,
        help="For PES/FD farming tasks: number of worker processes (default 1).",
    )
    strict_group = ap.add_mutually_exclusive_group()
    strict_group.add_argument('--strict', dest='strict', action='store_true', help='Fail hard on fallbacks that could hide errors (default)')
    strict_group.add_argument('--no-strict', dest='strict', action='store_false', help='Allow certain fallbacks with loud warnings')
    ap.set_defaults(strict=True)
    ap.add_argument('--allow-fd-hessian', action='store_true', help='Allow finite-difference Hessian fallback (otherwise error)')
    ap.add_argument('--dev-fast', action='store_true', help='Use lighter method/basis, fewer points, looser SCF for faster dev runs')
    ap.add_argument('--fast-npts', type=int, default=21, help='Cap npts in --dev-fast mode (default 21)')
    ap.add_argument('--fast-width', type=float, default=0.20, help='Cap tight-scan full width in Å for --dev-fast (default 0.20 Å)')
    ap.add_argument('-v', '--verbose', action='store_true', help='Enable verbose logging and progress bars')
    ap.set_defaults(use_ri=True)
    ap.add_argument('--tight-scan', action='store_true', help='For 1D task, center window at equilibrium bond length; see --tight-width')
    ap.add_argument('--tight-width', type=float, default=0.30, help='For 1D with --tight-scan, full scan width in Å (default 0.30 Å)')
    args = ap.parse_args()

    if args.dump_grid is not None and args.reuse_grid is not None:
        ap.error("Use only one of --dump-grid or --reuse-grid (not both)")

    MAX_PARALLEL = int(args.max_parallel)
    PES_WORKERS = int(args.pes_workers)
    if MAX_PARALLEL < 1:
        ap.error("--max-parallel must be >= 1")
    if PES_WORKERS < 1:
        ap.error("--pes-workers must be >= 1")
    VERBOSE = args.verbose
    DEV_FAST = args.dev_fast
    STRICT = bool(args.strict)
    ALLOW_FD_HESSIAN = bool(args.allow_fd_hessian)

    # Derive effective parallel settings.
    if args.task in ("harmonic", "opt"):
        WORKERS = 1
        THREADS_PER_WORKER = int(MAX_PARALLEL)
    else:
        # PES/FD farming tasks: split budget across processes.
        WORKERS = min(int(PES_WORKERS), int(MAX_PARALLEL))
        THREADS_PER_WORKER = max(1, int(MAX_PARALLEL) // int(WORKERS))

    # Apply thread limits early so all paths respect the effective threading.
    _pool_init(THREADS_PER_WORKER, STRICT)

    log(f"Selected task: {args.task}")
    log(f"Method: {args.method} | Basis: {args.basis} | RI {'enabled' if args.use_ri else 'disabled'}")
    log(f"Dispersion: {args.dispersion}")
    log(f"Parallel: max_parallel={MAX_PARALLEL} pes_workers={PES_WORKERS} => workers={WORKERS} threads={THREADS_PER_WORKER} (BLAS=1)")

    start = time.perf_counter()
    try:
        if not args.mmol and not args.xyz:
            ap.error('Provide --mmol or --xyz')
        if args.mmol:
            log(f"Reading MMOL geometry from {args.mmol}")
        elif args.xyz:
            log(f"Reading XYZ geometry from {args.xyz}")
        mol = read_midas_mmol(args.mmol) if args.mmol else Molecule([],np.zeros((0,3)))
        log(f"Loaded molecule '{mol.label}' with {len(mol.symbols)} atoms")
        if DEV_FAST:
            if args.method == _es_default('method'):
                args.method = 'hf'
            if args.basis == _es_default('basis'):
                args.basis = 'sto-3g'
            if args.dispersion == 'd4':
                args.dispersion = 'none'
            # Cap npts and tight-width using configurable dev-fast caps
            cap_npts = int(max(5, args.fast_npts))
            cap_width = float(max(0.05, args.fast_width))
            if args.npts > cap_npts:
                args.npts = cap_npts
            if args.tight_width > cap_width:
                args.tight_width = cap_width
            log(f"FAST MODE: method={args.method}, basis={args.basis}, npts={args.npts} (cap {cap_npts}), tight-width={args.tight_width:.2f} Å (cap {cap_width:.2f})")
        disp = None if (args.dispersion or "d4").lower() == "none" else str(args.dispersion)
        cfg = ESSettings(method=args.method, basis=args.basis, use_density_fit=args.use_ri, auxbasis=args.ri_aux,
                         dispersion=disp, rtproj=args.rtproj,
                         strict=STRICT, allow_fd_hessian=ALLOW_FD_HESSIAN,
                         scf_conv_tol=args.scf_conv_tol, scf_max_cycle=args.scf_max_cycle,
                         dft_grid_level=args.dft_grid_level)
        cfg = _effective_es_settings(cfg)
        if cfg.use_density_fit:
            aux_name = cfg.auxbasis if cfg.auxbasis else default_auxbasis(cfg.basis)
            log(f"RI auxiliary basis: {aux_name}")
        if args.task=='harmonic':
            log("Starting harmonic analysis")
            run_harmonic(mol,cfg)
        elif args.task=='opt':
            log("Starting geometry optimization")
            run_opt(mol, cfg, opt_out=args.opt_out, opt_maxsteps=args.opt_maxsteps, opt_conv=args.opt_conv)
        elif args.task=='1d':
            log(f"Starting 1D PES/VSCF task for bond {args.bond} (scan={args.scan})")
            b = parse_bond(args.bond)
            run_1d(
                mol,
                cfg,
                b,
                args.rmin,
                args.rmax,
                args.npts,
                args.vmax,
                args.tight_scan,
                args.tight_width,
                args.scan,
                smin=args.smin,
                smax=args.smax,
                dump_grid=args.dump_grid,
                reuse_grid=args.reuse_grid,
                intensity=args.intensity,
            )
        elif args.task=='2d':
            if not args.bond2: ap.error('--bond2 required for 2D')
            log(f"Starting 2D PES/VSCF task for bonds {args.bond} & {args.bond2}")
            b1 = parse_bond(args.bond); b2 = parse_bond(args.bond2)
            run_2d(
                mol,
                cfg,
                b1,
                b2,
                (args.rmin, args.rmax, args.npts),
                (args.rmin, args.rmax, args.npts),
                args.vmax,
                keo=args.keo,
                dump_grid=args.dump_grid,
                reuse_grid=args.reuse_grid,
                intensity=args.intensity,
            )
    finally:
        elapsed = time.perf_counter() - start
        print(f"Runtime: {elapsed:.2f} s ({_format_runtime(elapsed)})")

if __name__ == '__main__':
    main()

# --- Examples (run with one thread per process) ---
# python pyscf_pme_pipeline.py --mmol geom/HDO-HDO-HD.mmol --task harmonic --max-parallel 8
# python pyscf_pme_pipeline.py --mmol geom/HDO-HDO-HD.mmol --task opt --opt-out HDO-HDO-HD.pyscf_opt.mmol --max-parallel 8
# python pyscf_pme_pipeline.py --mmol geom/HDO-HDO-HD.mmol --task 1d --bond O0-H1 --npts 61 --max-parallel 8 --pes-workers 8
# python pyscf_pme_pipeline.py --mmol geom/HDO-HDO-HD.mmol --task 2d --bond O0-H1 --bond2 O3-H5 --npts 31 --max-parallel 8 --pes-workers 8
