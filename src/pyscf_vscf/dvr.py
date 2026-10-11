"""Discrete-variable-representation solvers and transition moments."""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import sparse


from ._eigen import lowest_eigenpairs
from ._phase import phase_canonical_columns
from .constants import AMU, ANG_TO_BOHR


@dataclass
class DVR1D:
    """Eigenpairs of a 1D sinc-DVR Hamiltonian.

    ``R`` is the uniform grid in angstrom, ``evals`` the energies in Hartree in
    ascending order, and ``evecs`` the normalized grid-point coefficients (one
    column per state).
    """

    R: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray


def _as_uniform_grid(name: str, values: np.ndarray) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1:
        raise ValueError(f"{name} must be a 1D array")
    if grid.size < 3:
        raise ValueError(f"{name} must contain at least 3 grid points")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain only finite values")

    step = np.diff(grid)
    if not np.all(step > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    if not np.allclose(step, step[0], rtol=1e-10, atol=1e-12):
        raise ValueError(f"{name} must be uniformly spaced")
    return grid


def _as_grid_values(
    name: str,
    values: np.ndarray,
    shape: tuple[int, ...],
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match expected shape {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _as_positive_mass(name: str, value: float) -> float:
    mass = float(value)
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{name} must be a positive finite mass in amu")
    return mass


def _state_index(value: int, nstates: int, name: str) -> int:
    try:
        idx = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer state index") from exc
    if idx < 0 or idx >= nstates:
        raise ValueError(f"{name}={idx} is outside the available state range 0..{nstates - 1}")
    return idx


def _sinc_kinetic_1d(R: np.ndarray, mu_amu: float) -> np.ndarray:
    n = R.size
    dx = (R[1] - R[0]) * ANG_TO_BOHR
    mu = _as_positive_mass("mu_amu", mu_amu) * AMU

    idx = np.arange(n)
    delta = idx[:, None] - idx[None, :]
    T = np.empty((n, n), dtype=float)
    diag = delta == 0
    T[diag] = (np.pi**2) / (6.0 * mu * dx * dx)
    off = ~diag
    T[off] = ((-1.0) ** delta[off]) / (mu * dx * dx * delta[off] ** 2)
    return T


def sinc_kinetic_1d(R: np.ndarray, mu_amu: float) -> np.ndarray:
    """Return the Colbert-Miller sinc-DVR kinetic-energy matrix in Hartree."""

    grid = _as_uniform_grid("R", R)
    return _sinc_kinetic_1d(grid, mu_amu)


def _sinc_derivatives_1d(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = (R[1] - R[0]) * ANG_TO_BOHR
    idx = np.arange(R.size)
    delta = idx[:, None] - idx[None, :]
    D1 = np.empty((R.size, R.size), dtype=float)
    D2 = np.empty((R.size, R.size), dtype=float)
    diag = delta == 0
    D1[diag] = 0.0
    D2[diag] = -(np.pi**2 / 3.0) / (dx * dx)
    off = ~diag
    D1[off] = ((-1.0) ** delta[off]) / (dx * delta[off])
    D2[off] = -2.0 * ((-1.0) ** delta[off]) / (dx * dx * delta[off] ** 2)
    return D1, D2


def sinc_dvr_1d(R: np.ndarray, mu_red_amu: float, V_Eh: np.ndarray) -> DVR1D:
    """Solve a 1D Colbert-Miller sinc DVR for potential ``V_Eh`` on grid ``R`` (angstrom)."""

    R = _as_uniform_grid("R", R)
    V_Eh = _as_grid_values("V_Eh", V_Eh, R.shape)
    T = _sinc_kinetic_1d(R, mu_red_amu)
    H = T + np.diag(V_Eh)
    evals, evecs = np.linalg.eigh(H)
    return DVR1D(R=R, evals=evals, evecs=phase_canonical_columns(evecs))


def trans_mu_1d(
    dvr: DVR1D,
    mu_of_R: Callable[[np.ndarray], np.ndarray],
    v: int,
) -> float:
    """Return the transition moment ``<0|mu|v>`` for a dipole function of ``R``."""

    v_idx = _state_index(v, dvr.evecs.shape[1], "v")
    mu = _as_grid_values("mu_of_R(dvr.R)", mu_of_R(dvr.R), dvr.R.shape)
    return float(np.dot(dvr.evecs[:, 0] * mu, dvr.evecs[:, v_idx]))


@dataclass
class DVR2D:
    """Lowest eigenpairs of a 2D product sinc-DVR Hamiltonian.

    Eigenvector columns are flattened with the second coordinate varying fastest
    (shape ``(R1.size, R2.size)`` in C order).
    """

    R1: np.ndarray
    R2: np.ndarray
    evals: np.ndarray
    evecs: np.ndarray


def product_dvr_2d(
    R1: np.ndarray,
    R2: np.ndarray,
    mu1_amu: float,
    mu2_amu: float,
    V_Eh: np.ndarray,
    *,
    nmax: int = 12,
    g12_inv_amu: float = 0.0,
) -> DVR2D:
    """Solve a 2D product sinc DVR.

    ``g12_inv_amu`` is the constant Wilson-G cross element in inverse amu. The
    default ``0`` gives the separable reduced-mass kinetic energy operator.
    """

    R1 = _as_uniform_grid("R1", R1)
    R2 = _as_uniform_grid("R2", R2)
    V_Eh = _as_grid_values("V_Eh", V_Eh, (R1.size, R2.size))
    mu1 = _as_positive_mass("mu1_amu", mu1_amu) * AMU
    mu2 = _as_positive_mass("mu2_amu", mu2_amu) * AMU
    g12 = float(g12_inv_amu) / AMU
    if not np.isfinite(g12):
        raise ValueError("g12_inv_amu must be finite")
    # The kinetic metric [[1/mu1, g12], [g12, 1/mu2]] must be positive definite.
    if g12 * g12 >= 1.0 / (mu1 * mu2):
        raise ValueError(
            "g12_inv_amu must satisfy g12^2 < 1/(mu1*mu2); the G matrix is not positive definite"
        )

    n1, n2 = R1.size, R2.size
    I1 = sparse.identity(n1, format="csr")
    I2 = sparse.identity(n2, format="csr")
    D1_1, D2_1 = _sinc_derivatives_1d(R1)
    D1_2, D2_2 = _sinc_derivatives_1d(R2)
    T1 = sparse.csr_matrix((-0.5 / mu1) * D2_1)
    T2 = sparse.csr_matrix((-0.5 / mu2) * D2_2)
    H = sparse.kron(T1, I2) + sparse.kron(I1, T2) + sparse.diags(V_Eh.ravel(), 0)
    if abs(g12) > 1e-18:
        H = H + (-g12) * sparse.kron(
            sparse.csr_matrix(D1_1),
            sparse.csr_matrix(D1_2),
        )
    dim = n1 * n2
    try:
        nmax_int = operator.index(nmax)
    except TypeError as exc:
        raise ValueError("nmax must be an integer") from exc
    if nmax_int < 1:
        raise ValueError("nmax must be at least 1")
    k = nmax_int + 1
    if k >= dim:
        raise ValueError(f"Requested {k} eigenpairs for 2D DVR dimension {dim}")
    # Deflated solves recover every copy of a degenerate level (for example the
    # two fundamentals of uncoupled identical bonds).
    evals, evecs = lowest_eigenpairs(
        lambda vector: H @ vector,
        dimension=dim,
        count=k,
        tolerance=0.0,
        degeneracy_tolerance=1e-12,
    )
    return DVR2D(R1=R1, R2=R2, evals=evals[:k], evecs=phase_canonical_columns(evecs[:, :k]))


def trans_mu_2d(dvr: DVR2D, mu_proj_grid: np.ndarray, n: int) -> float:
    """Return ``<0|mu|n>`` for a dipole component tabulated on the 2D grid."""

    n_idx = _state_index(n, dvr.evecs.shape[1], "n")
    mu = _as_grid_values(
        "mu_proj_grid",
        mu_proj_grid,
        (dvr.R1.size, dvr.R2.size),
    ).ravel()
    return float(np.dot(dvr.evecs[:, 0] * mu, dvr.evecs[:, n_idx]))
