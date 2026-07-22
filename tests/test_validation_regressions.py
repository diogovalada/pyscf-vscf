from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.constants import AMU, ANG_TO_BOHR, HARTREE_TO_CM, SPEED_OF_LIGHT_M_S
from pyscf_vscf.dvr import sinc_dvr_1d
from pyscf_vscf.spectra import (
    angular_frequency_from_wavenumber,
    einstein_a_from_debye,
    integrated_cross_section_omega,
    integrated_cross_section_omega_to_km_per_mol,
)
from pyscf_vscf.variational import variational_1d, variational_2d


def _morse_potential_shifted(
    R_ang: np.ndarray,
    *,
    De_Eh: float = 0.18,
    alpha_bohr_inv: float = 1.5,
    Re_ang: float = 1.0,
) -> np.ndarray:
    x = np.asarray(R_ang, dtype=float) * ANG_TO_BOHR
    xe = float(Re_ang) * ANG_TO_BOHR
    y = 1.0 - np.exp(-float(alpha_bohr_inv) * (x - xe))
    return float(De_Eh) * y * y


def _morse_analytic_levels_shifted(
    *,
    De_Eh: float = 0.18,
    alpha_bohr_inv: float = 1.5,
    mu_amu: float = 0.94,
    vmax: int = 5,
) -> np.ndarray:
    mu = float(mu_amu) * AMU
    lam = np.sqrt(2.0 * mu * float(De_Eh)) / float(alpha_bohr_inv)
    levels: list[float] = []
    for v in range(int(vmax) + 1):
        n = v + 0.5
        levels.append(
            float((alpha_bohr_inv * alpha_bohr_inv / (2.0 * mu)) * (2.0 * lam * n - n * n))
        )
    return np.array(levels)


def test_morse_fundamental_converges_with_grid_refinement() -> None:
    ref = _morse_analytic_levels_shifted(vmax=5)
    errors_cm = []
    for npts in (31, 41, 61):
        R = np.linspace(0.6, 3.2, npts)
        dvr = sinc_dvr_1d(R, 0.94, _morse_potential_shifted(R))
        fundamental_error = abs((dvr.evals[1] - dvr.evals[0]) - (ref[1] - ref[0]))
        errors_cm.append(float(fundamental_error * HARTREE_TO_CM))

    assert errors_cm[1] < errors_cm[0] / 100.0
    assert errors_cm[2] < errors_cm[1] / 100.0
    assert errors_cm[2] < 1e-4


def test_frozen_1d_pes_dms_regression_records_are_stable() -> None:
    R = np.linspace(0.78, 1.22, 13)
    q = R - 0.96
    E = 0.05 * q * q + 0.004 * q**3 + 0.0008 * q**4
    MU = np.column_stack(
        (
            0.13 + 0.62 * q + 0.08 * q * q,
            -0.04 + 0.18 * q,
            0.02 + 0.015 * q * q,
        )
    )

    records = variational_1d(
        R,
        E,
        MU,
        0.94,
        axis=[1.0, 0.0, 0.0],
        vmax=4,
        intensity="both",
    )

    np.testing.assert_allclose(
        [record["freq_cm"] for record in records],
        [2196.7237731495443, 5696.625636449846, 10571.21481844496, 16837.81172853942],
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        [abs(record["transition_dipole_axis_D"]) for record in records],
        [0.05557448820371811, 0.0001864638820441968, 0.004117818476811359, 0.00003468330284332917],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [record["integrated_cross_section_isotropic_omega_m2_per_s"] for record in records],
        [
            5.759494034796283e-11,
            5.6736016026760215e-15,
            1.5226816312454092e-12,
            5.43360167754274e-16,
        ],
        rtol=1e-10,
        atol=1e-27,
    )


def test_frozen_2d_pes_dms_regression_records_are_stable() -> None:
    R1 = np.linspace(0.82, 1.18, 7)
    R2 = np.linspace(0.85, 1.21, 7)
    q1 = R1 - 0.98
    q2 = R2 - 1.02
    E = (
        0.045 * q1[:, None] ** 2
        + 0.055 * q2[None, :] ** 2
        + 0.003 * q1[:, None] * q2[None, :]
        + 0.001 * q1[:, None] ** 3
        - 0.0007 * q2[None, :] ** 3
    )
    MU = np.zeros((R1.size, R2.size, 3))
    MU[:, :, 0] = 0.1 + 0.4 * q1[:, None] + 0.05 * q2[None, :] + 0.02 * q1[:, None] * q2[None, :]
    MU[:, :, 1] = -0.03 + 0.12 * q2[None, :]
    MU[:, :, 2] = 0.02 * q1[:, None] ** 2 - 0.015 * q2[None, :] ** 2

    records = variational_2d(
        R1,
        R2,
        E,
        MU,
        0.94,
        1.2,
        axis=[1.0, 0.0, 0.0],
        nmax=4,
        g12_inv_amu=0.01,
        intensity="both",
    )

    np.testing.assert_allclose(
        [record["freq_cm"] for record in records],
        [2017.4759611582933, 2531.4049474811245, 4547.579013681504, 5243.334272615995],
        rtol=0.0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        [abs(record["transition_dipole_axis_D"]) for record in records],
        [
            0.003145742455344773,
            0.03319098393961768,
            0.00014259858345289415,
            0.000016821996397453295,
        ],
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        [record["integrated_cross_section_isotropic_omega_m2_per_s"] for record in records],
        [
            1.6867072076018246e-12,
            2.186685473084918e-11,
            7.254532354278308e-16,
            6.951341156327545e-16,
        ],
        rtol=1e-10,
        atol=1e-27,
    )


def test_absolute_intensity_matches_independent_einstein_a_relation() -> None:
    frequency_cm = 1000.0
    one_debye = integrated_cross_section_omega(1.0, frequency_cm)
    omega = angular_frequency_from_wavenumber(frequency_cm)
    einstein_a = einstein_a_from_debye(1.0, frequency_cm)

    assert integrated_cross_section_omega(0.0, frequency_cm) == pytest.approx(0.0)
    assert integrated_cross_section_omega(-1.0, frequency_cm) == pytest.approx(one_debye)
    assert integrated_cross_section_omega(2.0, frequency_cm) == pytest.approx(4.0 * one_debye)
    assert one_debye == pytest.approx(7.840467192254704e-09, rel=1e-12)
    assert one_debye == pytest.approx(
        np.pi**2 * SPEED_OF_LIGHT_M_S**2 * einstein_a / omega**2,
        rel=1e-12,
    )
    assert integrated_cross_section_omega_to_km_per_mol(one_debye) == pytest.approx(
        2506.641773636364,
        rel=1e-12,
    )
    with pytest.raises(ValueError, match="positive"):
        integrated_cross_section_omega(1.0, 0.0)


def test_linear_dipole_harmonic_oscillator_has_analytic_transition_moment() -> None:
    frequency_cm = 1500.0
    mass_amu = 1.0
    slope_D_per_A = 0.7
    omega_au = frequency_cm / HARTREE_TO_CM
    mass_electron = mass_amu * AMU
    R = np.linspace(-0.8, 0.8, 101)
    q_bohr = R * ANG_TO_BOHR
    potential = 0.5 * mass_electron * omega_au**2 * q_bohr**2
    dipole = np.column_stack((slope_D_per_A * R, np.zeros_like(R), np.zeros_like(R)))

    record = variational_1d(
        R,
        potential,
        dipole,
        mass_amu,
        axis=[1.0, 0.0, 0.0],
        vmax=1,
        intensity="axis",
    )[0]
    zero_point_amplitude_A = np.sqrt(1.0 / (2.0 * mass_electron * omega_au)) / ANG_TO_BOHR
    expected_dipole_D = slope_D_per_A * zero_point_amplitude_A

    assert record["freq_cm"] == pytest.approx(frequency_cm, abs=1e-6)
    assert abs(record["transition_dipole_D"]) == pytest.approx(expected_dipole_D, rel=1e-10)


def test_curated_marvel_iupac_targets_are_parseable_and_well_formed() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "reference" / "marvel_iupac" / "vbo_stretch_targets.json"
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["meta"]["units_frequency"] == "cm^-1"
    assert data["meta"]["units_uncertainty"] == "cm^-1"
    source_dois = {source["doi"] for source in data["meta"]["sources"]}
    assert source_dois

    targets = data["targets"]
    species = {target["species"] for target in targets}
    assert {"H2O", "HDO", "D2O"}.issubset(species)
    assert len(targets) >= 10
    for target in targets:
        assert target["source"]["doi"] in source_dois
        assert np.isfinite(target["band_origin_cm1"])
        assert target["band_origin_cm1"] > 0.0
        if target["unc_cm1"] is not None:
            assert np.isfinite(target["unc_cm1"])
            assert target["unc_cm1"] >= 0.0
        assert set(target["quanta"]) == {"v1", "v2", "v3"}
