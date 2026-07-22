from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.variational import parse_intensity_mode, variational_1d, variational_2d


def test_variational_1d_axis_vector_and_both_records() -> None:
    R = np.linspace(0.75, 1.35, 31)
    E = 0.06 * (R - 1.0) ** 2
    MU = np.column_stack((R - 1.0, 2.0 * (R - 1.0), np.zeros_like(R)))

    axis_records = variational_1d(R, E, MU, 1.0, axis=[1.0, 0.0, 0.0], vmax=3)
    both_records = variational_1d(R, E, MU, 1.0, axis=[1.0, 0.0, 0.0], vmax=3, intensity="both")
    vector_records = variational_1d(
        R,
        E,
        MU,
        1.0,
        axis=[1.0, 0.0, 0.0],
        vmax=3,
        intensity="vector",
    )

    assert [rec["v"] for rec in axis_records] == [1, 2, 3]
    assert all(rec["freq_cm"] > 0.0 for rec in axis_records)
    assert both_records[0]["transition_dipole_axis_D"] == pytest.approx(
        axis_records[0]["transition_dipole_D"]
    )
    assert both_records[0]["transition_dipole_norm_D"] == pytest.approx(
        vector_records[0]["transition_dipole_D"]
    )
    assert abs(vector_records[0]["transition_dipole_D"]) > abs(
        axis_records[0]["transition_dipole_D"]
    )
    assert axis_records[0]["orientation"] == "polarized-axis"
    assert vector_records[0]["orientation"] == "isotropic"


def test_variational_2d_records_and_intensity_validation() -> None:
    R1 = np.linspace(0.8, 1.3, 9)
    R2 = np.linspace(0.82, 1.32, 9)
    E = 0.04 * (R1[:, None] - 1.0) ** 2 + 0.05 * (R2[None, :] - 1.05) ** 2
    MU = np.zeros((R1.size, R2.size, 3))
    MU[:, :, 0] = R1[:, None] - 1.0
    MU[:, :, 1] = 0.5 * (R2[None, :] - 1.05)

    records = variational_2d(
        R1,
        R2,
        E,
        MU,
        0.94,
        1.2,
        axis=[1.0, 0.0, 0.0],
        nmax=3,
        intensity="both",
    )

    assert [rec["n"] for rec in records] == [1, 2, 3]
    assert all(rec["freq_cm"] > 0.0 for rec in records)
    assert {
        "assignment",
        "assignment_weight",
        "transition_dipole_axis_D",
        "transition_dipole_norm_D",
        "integrated_cross_section_axis_omega_m2_per_s",
        "integrated_cross_section_isotropic_omega_m2_per_s",
    } <= set(records[0])
    with pytest.raises(ValueError, match="Unknown intensity mode"):
        parse_intensity_mode("bad")
    with pytest.raises(ValueError, match="non-zero"):
        variational_1d(R1, np.zeros_like(R1), np.zeros((R1.size, 3)), 1.0, axis=[0, 0, 0])


def test_importing_variational_module_does_not_request_pyscf() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    code = """
import importlib.abc
import sys

class BlockPySCF(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyscf" or fullname.startswith("pyscf."):
            raise RuntimeError(f"unexpected PySCF import: {fullname}")
        return None

sys.meta_path.insert(0, BlockPySCF())
import pyscf_vscf.variational
print("variational-ok")
"""
    env = os.environ.copy()
    src_path = str(repo_root / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "variational-ok"
