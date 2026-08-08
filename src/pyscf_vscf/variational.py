"""Pure variational spectrum assembly for cached or freshly computed grids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .assignments import assign_product_states_2d
from .constants import HARTREE_TO_CM
from .dvr import product_dvr_2d, sinc_dvr_1d, trans_mu_1d, trans_mu_2d
from .spectra import integrated_cross_section_omega


@dataclass(frozen=True)
class TransitionRecord:
    """One assigned vibrational transition with both intensity conventions."""

    state_index: int
    quanta: tuple[int, ...]
    frequency_cm: float
    transition_dipole_axis_D: float
    integrated_cross_section_axis_omega_m2_per_s: float
    transition_dipole_norm_D: float
    integrated_cross_section_isotropic_omega_m2_per_s: float
    assignment_weight: float | None = None
    assignment_signature: tuple[tuple[tuple[int, int], int], ...] | None = None
    assignment_participation_ratio: float | None = None
    assignment_dominant_manifold_weight: float | None = None
    assignment_top_components: tuple[tuple[tuple[int, int], float, float], ...] | None = None
    assignment_method: str | None = None
    assignment_reference: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-ready record without ambiguous intensity aliases."""

        record: dict[str, Any] = {
            "state_index": self.state_index,
            "quanta": self.quanta,
            "freq_cm": self.frequency_cm,
            "transition_dipole_axis_D": self.transition_dipole_axis_D,
            "integrated_cross_section_axis_omega_m2_per_s": (
                self.integrated_cross_section_axis_omega_m2_per_s
            ),
            "transition_dipole_norm_D": self.transition_dipole_norm_D,
            "integrated_cross_section_isotropic_omega_m2_per_s": (
                self.integrated_cross_section_isotropic_omega_m2_per_s
            ),
        }
        if len(self.quanta) == 1:
            record["v"] = self.quanta[0]
        else:
            record.update(
                {
                    "n": self.state_index,
                    "assignment": self.quanta,
                    "assignment_weight": self.assignment_weight,
                    "assignment_signature": self.assignment_signature,
                    "assignment_participation_ratio": self.assignment_participation_ratio,
                    "assignment_dominant_manifold_weight": (
                        self.assignment_dominant_manifold_weight
                    ),
                    "assignment_top_components": self.assignment_top_components,
                    "assignment_method": self.assignment_method,
                    "assignment_reference": self.assignment_reference,
                }
            )
        return record


def parse_intensity_mode(mode: str | None) -> str:
    """Normalize and validate a CLI intensity display mode."""

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
) -> list[TransitionRecord]:
    """Return 1D transitions with axis and isotropic intensities."""

    R_arr = np.asarray(R, dtype=float)
    E_arr = np.asarray(E, dtype=float)
    MU_arr = _as_mu_1d(MU, R_arr.size)
    dvr = sinc_dvr_1d(R_arr, redmass_amu, E_arr)
    axis_unit = _axis_unit(axis)

    def mux(rr):
        return np.interp(rr, R_arr, MU_arr[:, 0])

    def muy(rr):
        return np.interp(rr, R_arr, MU_arr[:, 1])

    def muz(rr):
        return np.interp(rr, R_arr, MU_arr[:, 2])

    def mu_axis_of_R(rr):
        return axis_unit[0] * mux(rr) + axis_unit[1] * muy(rr) + axis_unit[2] * muz(rr)

    out: list[TransitionRecord] = []
    e0 = dvr.evals[0]
    for v in range(1, min(int(vmax) + 1, len(dvr.evals))):
        nu_cm = float((dvr.evals[v] - e0) * HARTREE_TO_CM)
        mu_axis = trans_mu_1d(dvr, mu_axis_of_R, v)
        mx = trans_mu_1d(dvr, mux, v)
        my = trans_mu_1d(dvr, muy, v)
        mz = trans_mu_1d(dvr, muz, v)
        mu_vec = float(np.sqrt(mx * mx + my * my + mz * mz))
        out.append(
            TransitionRecord(
                state_index=v,
                quanta=(v,),
                frequency_cm=nu_cm,
                transition_dipole_axis_D=float(mu_axis),
                integrated_cross_section_axis_omega_m2_per_s=integrated_cross_section_omega(
                    float(mu_axis), nu_cm, orientation_factor=1.0
                ),
                transition_dipole_norm_D=mu_vec,
                integrated_cross_section_isotropic_omega_m2_per_s=(
                    integrated_cross_section_omega(mu_vec, nu_cm)
                ),
            )
        )
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
    reference_potentials_Eh: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[TransitionRecord]:
    """Return assigned 2D variational stick records."""

    R1_arr = np.asarray(R1, dtype=float)
    R2_arr = np.asarray(R2, dtype=float)
    E_arr = np.asarray(E, dtype=float)
    MU_arr = _as_mu_2d(MU, R1_arr.size, R2_arr.size)
    axis_unit = _axis_unit(axis)
    dvr = product_dvr_2d(
        R1_arr,
        R2_arr,
        mu1_amu,
        mu2_amu,
        E_arr,
        nmax=int(nmax),
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

    out: list[TransitionRecord] = []
    e0 = dvr.evals[0]
    for n in range(1, min(int(nmax) + 1, len(dvr.evals))):
        nu_cm = float((dvr.evals[n] - e0) * HARTREE_TO_CM)
        mu_axis = trans_mu_2d(dvr, mu_proj, n)
        mx = trans_mu_2d(dvr, mu_x, n)
        my = trans_mu_2d(dvr, mu_y, n)
        mz = trans_mu_2d(dvr, mu_z, n)
        mu_vec = float(np.sqrt(mx * mx + my * my + mz * mz))
        assignment = assignments[n]
        out.append(
            TransitionRecord(
                state_index=n,
                quanta=assignment.quanta,
                frequency_cm=nu_cm,
                transition_dipole_axis_D=float(mu_axis),
                integrated_cross_section_axis_omega_m2_per_s=integrated_cross_section_omega(
                    float(mu_axis), nu_cm, orientation_factor=1.0
                ),
                transition_dipole_norm_D=mu_vec,
                integrated_cross_section_isotropic_omega_m2_per_s=(
                    integrated_cross_section_omega(mu_vec, nu_cm)
                ),
                assignment_weight=assignment.weight,
                assignment_signature=assignment.signature,
                assignment_participation_ratio=assignment.participation_ratio,
                assignment_dominant_manifold_weight=assignment.dominant_manifold_weight,
                assignment_top_components=assignment.top_components,
                assignment_method="phase-canonical-overlap-hungarian",
                assignment_reference=assignment_reference,
            )
        )
    return out


__all__ = ["TransitionRecord", "parse_intensity_mode", "variational_1d", "variational_2d"]
