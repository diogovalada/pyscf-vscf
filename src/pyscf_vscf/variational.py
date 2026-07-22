"""Pure variational spectrum assembly for cached or freshly computed grids."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .assignments import assign_product_states_2d
from .constants import HARTREE_TO_CM
from .dvr import product_dvr_2d, sinc_dvr_1d, trans_mu_1d, trans_mu_2d
from .spectra import integrated_cross_section_omega


def parse_intensity_mode(mode: str | None) -> str:
    """Normalize and validate the legacy intensity mode string."""

    parsed = (mode or "axis").lower()
    if parsed not in {"axis", "vector", "both"}:
        raise ValueError(f"Unknown intensity mode '{mode}' (expected axis|vector|both)")
    return parsed


def _axis_unit(axis: Sequence[float] | np.ndarray | None) -> np.ndarray:
    if axis is None:
        axis_vec = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        axis_vec = np.asarray(axis, dtype=float)
    if axis_vec.shape != (3,):
        raise ValueError("Dipole projection axis must have shape (3,)")
    norm = float(np.linalg.norm(axis_vec))
    if norm < 1e-12:
        raise ValueError("Dipole projection axis must be non-zero")
    return axis_vec / norm


def _as_mu_1d(MU: np.ndarray, npts: int) -> np.ndarray:
    mu = np.asarray(MU, dtype=float)
    if mu.shape != (npts, 3):
        raise ValueError(f"MU shape {mu.shape} does not match expected shape {(npts, 3)}")
    if not np.all(np.isfinite(mu)):
        raise ValueError("MU must contain only finite values")
    return mu


def _as_mu_2d(MU: np.ndarray, n1: int, n2: int) -> np.ndarray:
    mu = np.asarray(MU, dtype=float)
    if mu.shape != (n1, n2, 3):
        raise ValueError(f"MU shape {mu.shape} does not match expected shape {(n1, n2, 3)}")
    if not np.all(np.isfinite(mu)):
        raise ValueError("MU must contain only finite values")
    return mu


def variational_1d(
    R,
    E,
    MU,
    redmass_amu: float,
    axis: Sequence[float] | np.ndarray | None = None,
    vmax: int = 8,
    *,
    intensity: str = "axis",
) -> list[dict]:
    """Return legacy-format 1D variational stick records."""

    R_arr = np.asarray(R, dtype=float)
    E_arr = np.asarray(E, dtype=float)
    MU_arr = _as_mu_1d(MU, R_arr.size)
    dvr = sinc_dvr_1d(R_arr, redmass_amu, E_arr)
    axis_unit = _axis_unit(axis)
    intensity = parse_intensity_mode(intensity)

    def mux(rr):
        return np.interp(rr, R_arr, MU_arr[:, 0])

    def muy(rr):
        return np.interp(rr, R_arr, MU_arr[:, 1])

    def muz(rr):
        return np.interp(rr, R_arr, MU_arr[:, 2])

    def mu_axis_of_R(rr):
        return axis_unit[0] * mux(rr) + axis_unit[1] * muy(rr) + axis_unit[2] * muz(rr)

    out: list[dict] = []
    e0 = dvr.evals[0]
    for v in range(1, min(int(vmax) + 1, len(dvr.evals))):
        nu_cm = float((dvr.evals[v] - e0) * HARTREE_TO_CM)
        mu_axis = trans_mu_1d(dvr, mu_axis_of_R, v)
        mu_vec = None
        if intensity in {"vector", "both"}:
            mx = trans_mu_1d(dvr, mux, v)
            my = trans_mu_1d(dvr, muy, v)
            mz = trans_mu_1d(dvr, muz, v)
            mu_vec = float(np.sqrt(mx * mx + my * my + mz * mz))

        mu_use = mu_axis if intensity == "axis" else mu_vec
        assert mu_use is not None
        orientation = "polarized-axis" if intensity == "axis" else "isotropic"
        orientation_factor = 1.0 if intensity == "axis" else 1.0 / 3.0
        rec = {
            "v": v,
            "freq_cm": nu_cm,
            "transition_dipole_D": float(mu_use),
            "integrated_cross_section_omega_m2_per_s": integrated_cross_section_omega(
                float(mu_use), nu_cm, orientation_factor=orientation_factor
            ),
            "orientation": orientation,
        }
        if intensity == "axis":
            rec["transition_dipole_axis_D"] = float(mu_axis)
            rec["integrated_cross_section_axis_omega_m2_per_s"] = rec[
                "integrated_cross_section_omega_m2_per_s"
            ]
        elif intensity == "vector":
            rec["transition_dipole_norm_D"] = float(mu_use)
            rec["integrated_cross_section_isotropic_omega_m2_per_s"] = rec[
                "integrated_cross_section_omega_m2_per_s"
            ]
        if intensity == "both":
            assert mu_vec is not None
            rec.update(
                {
                    "transition_dipole_axis_D": float(mu_axis),
                    "integrated_cross_section_axis_omega_m2_per_s": (
                        integrated_cross_section_omega(
                            float(mu_axis), nu_cm, orientation_factor=1.0
                        )
                    ),
                    "transition_dipole_norm_D": float(mu_vec),
                    "integrated_cross_section_isotropic_omega_m2_per_s": (
                        integrated_cross_section_omega(float(mu_vec), nu_cm)
                    ),
                }
            )
        out.append(rec)
    return out


def variational_2d(
    R1,
    R2,
    E,
    MU,
    mu1_amu: float,
    mu2_amu: float,
    axis: Sequence[float] | np.ndarray | None = None,
    nmax=8,
    *,
    g12_inv_amu: float = 0.0,
    intensity: str = "axis",
    reference_potentials_Eh: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict]:
    """Return assigned 2D variational stick records."""

    R1_arr = np.asarray(R1, dtype=float)
    R2_arr = np.asarray(R2, dtype=float)
    E_arr = np.asarray(E, dtype=float)
    MU_arr = _as_mu_2d(MU, R1_arr.size, R2_arr.size)
    axis_unit = _axis_unit(axis)
    intensity = parse_intensity_mode(intensity)
    dvr = product_dvr_2d(
        R1_arr,
        R2_arr,
        mu1_amu,
        mu2_amu,
        E_arr,
        k_eigs=int(nmax) + 1,
        g12_inv_amu=float(g12_inv_amu),
    )
    if reference_potentials_Eh is None:
        minimum = np.unravel_index(int(np.argmin(E_arr)), E_arr.shape)
        reference1 = E_arr[:, minimum[1]]
        reference2 = E_arr[minimum[0], :]
        assignment_reference = "minimum-energy cuts"
    else:
        reference1, reference2 = reference_potentials_Eh
        assignment_reference = "caller-provided one-mode potentials"
    assignments = assign_product_states_2d(
        dvr,
        mu1_amu,
        mu2_amu,
        np.asarray(reference1, dtype=float),
        np.asarray(reference2, dtype=float),
    )

    mu_x = MU_arr[:, :, 0]
    mu_y = MU_arr[:, :, 1]
    mu_z = MU_arr[:, :, 2]
    mu_proj = axis_unit[0] * mu_x + axis_unit[1] * mu_y + axis_unit[2] * mu_z

    out: list[dict] = []
    e0 = dvr.evals[0]
    for n in range(1, min(int(nmax) + 1, len(dvr.evals))):
        nu_cm = float((dvr.evals[n] - e0) * HARTREE_TO_CM)
        mu_axis = trans_mu_2d(dvr, mu_proj, n)
        mu_vec = None
        if intensity in {"vector", "both"}:
            mx = trans_mu_2d(dvr, mu_x, n)
            my = trans_mu_2d(dvr, mu_y, n)
            mz = trans_mu_2d(dvr, mu_z, n)
            mu_vec = float(np.sqrt(mx * mx + my * my + mz * mz))

        mu_use = mu_axis if intensity == "axis" else mu_vec
        assert mu_use is not None
        orientation = "polarized-axis" if intensity == "axis" else "isotropic"
        orientation_factor = 1.0 if intensity == "axis" else 1.0 / 3.0
        rec = {
            "n": n,
            "assignment": assignments[n].quanta,
            "assignment_weight": assignments[n].weight,
            "assignment_signature": assignments[n].signature,
            "assignment_participation_ratio": assignments[n].participation_ratio,
            "assignment_dominant_manifold_weight": assignments[n].dominant_manifold_weight,
            "assignment_top_components": assignments[n].top_components,
            "assignment_method": "phase-canonical-overlap-hungarian",
            "assignment_reference": assignment_reference,
            "freq_cm": nu_cm,
            "transition_dipole_D": float(mu_use),
            "integrated_cross_section_omega_m2_per_s": integrated_cross_section_omega(
                float(mu_use), nu_cm, orientation_factor=orientation_factor
            ),
            "orientation": orientation,
        }
        if intensity == "axis":
            rec["transition_dipole_axis_D"] = float(mu_axis)
            rec["integrated_cross_section_axis_omega_m2_per_s"] = rec[
                "integrated_cross_section_omega_m2_per_s"
            ]
        elif intensity == "vector":
            rec["transition_dipole_norm_D"] = float(mu_use)
            rec["integrated_cross_section_isotropic_omega_m2_per_s"] = rec[
                "integrated_cross_section_omega_m2_per_s"
            ]
        if intensity == "both":
            assert mu_vec is not None
            rec.update(
                {
                    "transition_dipole_axis_D": float(mu_axis),
                    "integrated_cross_section_axis_omega_m2_per_s": (
                        integrated_cross_section_omega(
                            float(mu_axis), nu_cm, orientation_factor=1.0
                        )
                    ),
                    "transition_dipole_norm_D": float(mu_vec),
                    "integrated_cross_section_isotropic_omega_m2_per_s": (
                        integrated_cross_section_omega(float(mu_vec), nu_cm)
                    ),
                }
            )
        out.append(rec)
    return out


__all__ = ["parse_intensity_mode", "variational_1d", "variational_2d"]
