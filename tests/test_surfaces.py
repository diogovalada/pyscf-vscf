from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyscf_vscf.coordinates import Bond
from pyscf_vscf.molecule import Molecule
from pyscf_vscf import surfaces
from pyscf_vscf.backends.pyscf import PySCFMeanFieldProvider
from pyscf_vscf.scans import plan_bond_scan, plan_normal_scan
from pyscf_vscf.settings import ESSettings

pytestmark = pytest.mark.filterwarnings("ignore::pyscf_vscf._warnings.PyscfVscfDeprecationWarning")


def _water() -> Molecule:
    return Molecule.from_arrays(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        label="water",
    )


PROVIDER = PySCFMeanFieldProvider(ESSettings(method="hf", basis="sto-3g"))


def _geometry(plan, index: tuple[int, ...]) -> np.ndarray:
    return np.asarray(plan.requests[str(plan.point_keys[index])].coordinates_A)


def test_bond_scan_plan_stretches_one_bond_and_leaves_the_molecule_alone() -> None:
    mol = _water()
    grid = np.linspace(0.8, 1.2, 5)
    plan = plan_bond_scan(mol, PROVIDER, [Bond(0, 1)], [grid])

    assert plan.shape == (5,)
    for index, length in enumerate(grid):
        coords = _geometry(plan, (index,))
        assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(length)
        np.testing.assert_allclose(coords[2], mol.coords[2])
    np.testing.assert_allclose(mol.coords[1], [0.96, 0.0, 0.0])


def test_normal_scan_plan_normalizes_the_direction_to_angstrom_displacements() -> None:
    mol = _water()
    u_dir = np.zeros_like(mol.coords)
    u_dir[1, 2] = 2.0
    plan = plan_normal_scan(mol, PROVIDER, u_dir, [-0.1, 0.0, 0.1])

    np.testing.assert_allclose(
        _geometry(plan, (2,)) - mol.coords, 0.1 * u_dir / np.linalg.norm(u_dir)
    )
    with pytest.raises(ValueError, match="coordinate shape"):
        plan_normal_scan(mol, PROVIDER, np.zeros((2, 3)), [0.0])


def test_two_bond_plan_has_the_grid_shape_and_keeps_the_molecule() -> None:
    mol = _water()
    plan = plan_bond_scan(
        mol,
        PROVIDER,
        [Bond(0, 1), Bond(0, 2)],
        [np.linspace(0.85, 1.05, 3), np.linspace(0.9, 1.1, 4)],
    )

    assert plan.shape == (3, 4)
    assert plan.point_keys.shape == (3, 4)
    np.testing.assert_allclose(mol.coords[1], [0.96, 0.0, 0.0])
    np.testing.assert_allclose(mol.coords[2], [-0.24, 0.93, 0.0])


@pytest.mark.parametrize(
    ("b1", "b2"),
    [
        (Bond(0, 1), Bond(0, 2)),
        (Bond(1, 0), Bond(2, 0)),
        (Bond(1, 0), Bond(0, 2)),
    ],
)
def test_two_bond_plan_realizes_shared_bond_lengths_for_all_orientations(
    b1: Bond,
    b2: Bond,
) -> None:
    plan = plan_bond_scan(
        _water(), PROVIDER, [b1, b2], [np.array([1.10, 1.15, 1.20]), np.array([1.20, 1.25, 1.30])]
    )

    coords = _geometry(plan, (0, 0))
    assert np.linalg.norm(coords[b1.j] - coords[b1.i]) == pytest.approx(1.10)
    assert np.linalg.norm(coords[b2.j] - coords[b2.i]) == pytest.approx(1.20)


def test_two_bond_plan_realizes_disjoint_bond_lengths() -> None:
    mol = Molecule.from_arrays(
        ["H", "F", "H", "Cl"],
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [3.0, 0.0, 0.0], [4.2, 0.0, 0.0]],
    )
    plan = plan_bond_scan(
        mol,
        PROVIDER,
        [Bond(0, 1), Bond(2, 3)],
        [np.array([1.0, 1.1, 1.2]), np.array([1.4, 1.5, 1.6])],
    )

    coords = _geometry(plan, (0, 0))
    assert np.linalg.norm(coords[1] - coords[0]) == pytest.approx(1.0)
    assert np.linalg.norm(coords[3] - coords[2]) == pytest.approx(1.4)


def test_energy_dipole_coerces_mapping_before_selecting_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyscf_vscf.backends import pyscf as backend
    from pyscf_vscf.settings import ESSettings

    captured: dict[str, object] = {}

    def fake_converged_mean_field(molecule: Molecule, cfg: object) -> object:
        captured["molecule"] = molecule
        captured["factory"] = backend.method_factory(cfg)
        return SimpleNamespace(total_energy_Eh=-1.0)

    monkeypatch.setattr(backend, "converged_mean_field", fake_converged_mean_field)
    monkeypatch.setattr(backend, "dipole_au", lambda converged: np.array([1.0, 0.0, 0.0]))

    energy, dipole = surfaces.energy_dipole(
        _water(),
        {"basis": "6-31g", "method": "hf", "scf_max_cycle": 37},
    )

    settings = captured["factory"].settings
    assert isinstance(settings, ESSettings)
    assert settings.basis == "6-31g"
    assert settings.scf_max_cycle == 37
    assert energy == pytest.approx(-1.0)
    assert dipole[0] == pytest.approx(surfaces.AU_DIPOLE_TO_DEBYE)


def test_importing_surfaces_module_does_not_request_pyscf() -> None:
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
import pyscf_vscf.surfaces

print("surfaces-ok")
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
    assert proc.stdout.strip() == "surfaces-ok"
