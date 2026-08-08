from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pyscf_vscf.coordinates import Bond
from pyscf_vscf.molecule import Molecule
from pyscf_vscf import surfaces
from pyscf_vscf import grid_2d_pes_dms as public_grid_2d_pes_dms
from pyscf_vscf.surfaces import grid_1d_pes_dms, grid_1d_pes_dms_normal, grid_2d_pes_dms


def _water() -> Molecule:
    return Molecule.from_arrays(
        ["O", "H", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        label="water",
    )


def _fake_energy_dipole(molecule: Molecule, cfg: object) -> tuple[float, np.ndarray]:
    coords = np.asarray(molecule.coords, dtype=float)
    energy = float(np.sum(coords * coords))
    dipole = np.sum(coords, axis=0)
    return energy, dipole


def test_grid_1d_pes_dms_uses_frozen_bond_geometry_and_zeroes_energy() -> None:
    mol = _water()
    R, E, MU = grid_1d_pes_dms(
        mol,
        object(),
        Bond(0, 1),
        Rmin=0.8,
        Rmax=1.2,
        npts=5,
        energy_dipole_fn=_fake_energy_dipole,
    )

    np.testing.assert_allclose(R, np.linspace(0.8, 1.2, 5))
    assert E.shape == (5,)
    assert MU.shape == (5, 3)
    assert np.min(E) == pytest.approx(0.0)
    np.testing.assert_allclose(mol.coords[1], [0.96, 0.0, 0.0])
    assert MU[0, 0] < MU[-1, 0]


def test_grid_1d_normal_validates_direction_shape() -> None:
    mol = _water()
    u_dir = np.zeros_like(mol.coords)
    u_dir[1, 2] = 1.0

    S, E, MU = grid_1d_pes_dms_normal(
        mol,
        object(),
        u_dir,
        smin=-0.1,
        smax=0.1,
        npts=3,
        energy_dipole_fn=_fake_energy_dipole,
    )

    np.testing.assert_allclose(S, [-0.1, 0.0, 0.1])
    assert E.shape == (3,)
    assert MU.shape == (3, 3)
    with pytest.raises(ValueError, match="u_dir shape"):
        grid_1d_pes_dms_normal(
            mol, object(), np.zeros((2, 3)), energy_dipole_fn=_fake_energy_dipole
        )


def test_public_normal_grid_normalizes_direction_to_angstrom_displacement() -> None:
    mol = _water()
    u_dir = np.zeros_like(mol.coords)
    u_dir[1, 2] = 2.0
    seen: list[np.ndarray] = []

    def capture(molecule: Molecule, cfg: object) -> tuple[float, np.ndarray]:
        del cfg
        seen.append(molecule.coords.copy())
        return 0.0, np.zeros(3)

    grid_1d_pes_dms_normal(
        mol,
        object(),
        u_dir,
        smin=0.1,
        smax=0.1,
        npts=1,
        energy_dipole_fn=capture,
    )

    np.testing.assert_allclose(seen[0] - mol.coords, 0.1 * u_dir / np.linalg.norm(u_dir))


def test_grid_2d_pes_dms_shapes_and_original_geometry() -> None:
    mol = _water()
    R1 = np.linspace(0.85, 1.05, 3)
    R2 = np.linspace(0.9, 1.1, 4)

    out_R1, out_R2, E, MU = grid_2d_pes_dms(
        mol,
        object(),
        Bond(0, 1),
        Bond(0, 2),
        R1,
        R2,
        energy_dipole_fn=_fake_energy_dipole,
    )

    np.testing.assert_allclose(out_R1, R1)
    np.testing.assert_allclose(out_R2, R2)
    assert E.shape == (3, 4)
    assert MU.shape == (3, 4, 3)
    assert np.min(E) == pytest.approx(0.0)
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
def test_public_grid_2d_realizes_shared_bond_lengths_for_all_orientations(
    b1: Bond,
    b2: Bond,
) -> None:
    mol = _water()
    seen: list[np.ndarray] = []

    def capture(molecule: Molecule, cfg: object) -> tuple[float, np.ndarray]:
        del cfg
        seen.append(molecule.coords.copy())
        return 0.0, np.zeros(3)

    public_grid_2d_pes_dms(
        mol,
        object(),
        b1,
        b2,
        np.array([1.10]),
        np.array([1.20]),
        energy_dipole_fn=capture,
    )

    assert len(seen) == 1
    assert np.linalg.norm(seen[0][b1.j] - seen[0][b1.i]) == pytest.approx(1.10)
    assert np.linalg.norm(seen[0][b2.j] - seen[0][b2.i]) == pytest.approx(1.20)


def test_public_grid_2d_realizes_disjoint_bond_lengths() -> None:
    mol = Molecule.from_arrays(
        ["H", "F", "H", "Cl"],
        [[0.0, 0.0, 0.0], [0.9, 0.0, 0.0], [3.0, 0.0, 0.0], [4.2, 0.0, 0.0]],
    )
    seen: list[np.ndarray] = []

    def capture(molecule: Molecule, cfg: object) -> tuple[float, np.ndarray]:
        del cfg
        seen.append(molecule.coords.copy())
        return 0.0, np.zeros(3)

    grid_2d_pes_dms(
        mol,
        object(),
        Bond(0, 1),
        Bond(2, 3),
        np.array([1.0]),
        np.array([1.4]),
        energy_dipole_fn=capture,
    )

    assert np.linalg.norm(seen[0][1] - seen[0][0]) == pytest.approx(1.0)
    assert np.linalg.norm(seen[0][3] - seen[0][2]) == pytest.approx(1.4)


def test_energy_dipole_coerces_mapping_before_selecting_basis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pyscf_vscf.backends import pyscf as backend
    from pyscf_vscf.settings import ESSettings

    captured: dict[str, object] = {}

    class FakeMeanField:
        e_tot = -1.0

        @staticmethod
        def make_rdm1() -> np.ndarray:
            return np.eye(1)

        @staticmethod
        def dip_moment(**kwargs) -> np.ndarray:
            del kwargs
            return np.array([1.0, 0.0, 0.0])

    def fake_molecule_to_pyscf(molecule: Molecule, basis: str) -> object:
        captured["molecule"] = molecule
        captured["basis"] = basis
        return object()

    def fake_make_mean_field(pmol: object, cfg: object) -> FakeMeanField:
        del pmol
        captured["cfg"] = cfg
        return FakeMeanField()

    monkeypatch.setattr(backend, "molecule_to_pyscf", fake_molecule_to_pyscf)
    monkeypatch.setattr(backend, "make_mean_field", fake_make_mean_field)

    energy, dipole = surfaces.energy_dipole(
        _water(),
        {"basis": "6-31g", "method": "hf", "scf_max_cycle": 37},
    )

    assert captured["basis"] == "6-31g"
    assert isinstance(captured["cfg"], ESSettings)
    assert captured["cfg"].basis == "6-31g"
    assert captured["cfg"].scf_max_cycle == 37
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
