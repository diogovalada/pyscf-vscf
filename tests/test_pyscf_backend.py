from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


BACKEND_MODULE = "pyscf_vscf.backends.pyscf"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _import_backend_or_skip():
    try:
        return importlib.import_module(BACKEND_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name in {BACKEND_MODULE, "pyscf"}:
            pytest.skip(f"{exc.name} is not available")
        raise


def _import_pyscf_or_skip() -> None:
    try:
        importlib.import_module("pyscf")
    except Exception as exc:
        pytest.skip(f"PySCF is not available: {exc}")


def _hf_sto3g_settings(backend):
    settings_cls = getattr(backend, "ESSettings", None)
    if settings_cls is None:
        try:
            from pyscf_vscf.settings import ESSettings
        except ModuleNotFoundError:
            return {
                "method": "hf",
                "basis": "sto-3g",
                "use_density_fit": False,
                "dispersion": None,
            }
        settings_cls = ESSettings

    return settings_cls(
        method="hf",
        basis="sto-3g",
        use_density_fit=False,
        dispersion=None,
    )


def _hdo_molecule():
    from pyscf_vscf.molecule import Molecule

    return Molecule.from_arrays(
        ["O", "D", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9572],
            [0.9266, 0.0, -0.2396],
        ],
        label="HDO",
    )


def test_normal_relaxed_point_enforces_exact_constraint_without_importing_pyscf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = importlib.import_module(BACKEND_MODULE)
    from pyscf_vscf import surfaces
    from pyscf_vscf.molecule import Molecule

    molecule = Molecule.from_arrays(["H"], [[0.0, 0.0, 0.0]])
    direction = np.array([[1.0, 0.0, 0.0]])
    seen: dict[str, np.ndarray] = {}
    orthogonal_minimum_bohr = np.array([0.0, 2.0, -1.0])

    def fake_energy_gradient(molecule_arg, cfg_arg, xflat_bohr):
        del molecule_arg, cfg_arg
        xflat = np.asarray(xflat_bohr, dtype=float)
        delta = xflat - orthogonal_minimum_bohr
        return 0.5 * float(np.dot(delta, delta)), delta

    def fake_energy_dipole(molecule_arg, cfg_arg):
        del cfg_arg
        seen["coords"] = np.asarray(molecule_arg.coords, dtype=float).copy()
        return 123.0, np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(backend, "energy_gradient_at_coords_bohr", fake_energy_gradient)
    monkeypatch.setattr(surfaces, "energy_dipole", fake_energy_dipole)

    result = backend.normal_relaxed_point(
        molecule,
        SimpleNamespace(),
        direction,
        s=0.2,
        gtol=1e-10,
        maxiter=50,
    )

    assert result.energy_hartree == pytest.approx(123.0)
    np.testing.assert_allclose(result.dipole_debye, [1.0, 2.0, 3.0])
    from pyscf_vscf.constants import ANG_TO_BOHR

    np.testing.assert_allclose(
        seen["coords"][0],
        [0.2, 2.0 / ANG_TO_BOHR, -1.0 / ANG_TO_BOHR],
        atol=1e-10,
    )
    assert result.achieved_displacement_A == pytest.approx(0.2, abs=1e-12)
    assert result.constraint_residual_A == pytest.approx(0.0, abs=1e-12)


def test_normal_relaxed_point_uses_mass_metric_for_heteronuclear_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = importlib.import_module(BACKEND_MODULE)
    from pyscf_vscf import surfaces
    from pyscf_vscf.constants import ANG_TO_BOHR
    from pyscf_vscf.molecule import Molecule

    molecule = Molecule.from_arrays(
        ["H", "F"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        masses_amu=[1.0, 19.0],
    )
    direction = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    seen: dict[str, np.ndarray] = {}

    def fake_energy_gradient(molecule_arg, cfg_arg, xflat_bohr):
        del molecule_arg, cfg_arg
        xflat = np.asarray(xflat_bohr, dtype=float)
        return 0.5 * float(np.dot(xflat, xflat)), xflat

    def fake_energy_dipole(molecule_arg, cfg_arg):
        del cfg_arg
        seen["coords"] = np.asarray(molecule_arg.coords, dtype=float).copy()
        return 0.0, np.zeros(3)

    monkeypatch.setattr(backend, "energy_gradient_at_coords_bohr", fake_energy_gradient)
    monkeypatch.setattr(surfaces, "energy_dipole", fake_energy_dipole)

    result = backend.normal_relaxed_point(
        molecule,
        SimpleNamespace(strict=True),
        direction,
        s=0.2,
        gtol=1e-12,
        maxiter=100,
    )

    u_flat = direction.reshape(-1) / np.linalg.norm(direction)
    mass_flat = np.repeat(np.array([1.0, 19.0]), 3)
    effective_mass = np.dot(u_flat, mass_flat * u_flat)
    covector = mass_flat * u_flat / effective_mass
    final_bohr = seen["coords"].reshape(-1) * ANG_TO_BOHR
    s_bohr = 0.2 * ANG_TO_BOHR
    relaxation = final_bohr - s_bohr * u_flat

    assert np.dot(covector, final_bohr) == pytest.approx(s_bohr, abs=1e-10)
    assert np.dot(mass_flat * u_flat, relaxation) == pytest.approx(0.0, abs=1e-10)
    assert abs(float(np.dot(u_flat, relaxation))) > 1e-3
    assert result.achieved_displacement_A == pytest.approx(0.2, abs=1e-10)
    assert result.constraint_residual_A == pytest.approx(0.0, abs=1e-10)


def test_make_mean_field_selects_unrestricted_method_for_open_shell(monkeypatch) -> None:
    backend = importlib.import_module(BACKEND_MODULE)
    calls: list[str] = []

    class FakeMF:
        converged = True

        def kernel(self):
            return 0.0

    class FakeSCF:
        @staticmethod
        def RHF(pmol):
            del pmol
            calls.append("RHF")
            return FakeMF()

        @staticmethod
        def UHF(pmol):
            del pmol
            calls.append("UHF")
            return FakeMF()

    class FakeDFT:
        RKS = FakeSCF.RHF
        UKS = FakeSCF.UHF

    monkeypatch.setattr(
        backend,
        "_require_pyscf",
        lambda: (object(), FakeSCF, FakeDFT, object()),
    )
    monkeypatch.setattr(
        backend,
        "_require_pyscf_dispersion",
        lambda value: pytest.fail(f"dispersion backend unexpectedly requested for {value!r}"),
    )

    mean_field = backend.make_mean_field(
        SimpleNamespace(spin=1),
        SimpleNamespace(
            method="hf",
            use_density_fit=False,
            dispersion="none",
            dev_fast=False,
            scf_conv_tol=1e-9,
            scf_max_cycle=37,
        ),
    )

    assert isinstance(mean_field, FakeMF)
    assert calls == ["UHF"]
    assert mean_field.max_cycle == 37


@pytest.mark.pyscf
def test_backend_module_import_does_not_request_pyscf() -> None:
    code = f"""
import importlib
import importlib.abc


class BlockPySCF(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyscf" or fullname.startswith("pyscf."):
            raise RuntimeError(f"unexpected PySCF import: {{fullname}}")
        return None


import sys
sys.meta_path.insert(0, BlockPySCF())
try:
    importlib.import_module({BACKEND_MODULE!r})
except ModuleNotFoundError as exc:
    if exc.name == {BACKEND_MODULE!r}:
        raise SystemExit(77)
    raise
print("backend-import-ok")
"""
    env = os.environ.copy()
    src_path = str(_repo_root() / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        env=env,
        text=True,
        capture_output=True,
    )

    if proc.returncode == 77:
        pytest.skip(f"{BACKEND_MODULE} is not available yet")
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "backend-import-ok"


@pytest.mark.pyscf
def test_molecule_to_pyscf_preserves_deuterium_isotope_mass() -> None:
    _import_pyscf_or_skip()
    backend = _import_backend_or_skip()
    molecule_to_pyscf = backend.molecule_to_pyscf

    from pyscf_vscf.constants import MASS_AMU

    pmol = molecule_to_pyscf(_hdo_molecule(), basis="sto-3g")

    assert pmol.natm == 3
    np.testing.assert_allclose(pmol.atom_charges(), [8, 1, 1])
    masses = np.asarray(pmol.atom_mass_list(), dtype=float)
    nucprop = getattr(pmol, "nucprop", {})
    assert masses[1] == pytest.approx(MASS_AMU["D"])
    assert masses[1] > 1.5 * masses[2]
    assert nucprop[2]["mass"] == pytest.approx(MASS_AMU["D"])


@pytest.mark.pyscf
def test_make_mean_field_constructs_tiny_hf_water() -> None:
    _import_pyscf_or_skip()
    backend = _import_backend_or_skip()

    from pyscf_vscf.molecule import Molecule

    molecule_to_pyscf = backend.molecule_to_pyscf
    make_mean_field = backend.make_mean_field
    water = Molecule.from_arrays(
        ["O", "H", "H"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.9572],
            [0.9266, 0.0, -0.2396],
        ],
        label="water",
    )
    pmol = molecule_to_pyscf(water, basis="sto-3g")

    mf = make_mean_field(pmol, _hf_sto3g_settings(backend))

    assert mf.mol is pmol
    assert mf.mol.natm == 3
    assert callable(mf.kernel)


@pytest.mark.pyscf
def test_energy_gradient_at_coords_bohr_tiny_hf_is_finite() -> None:
    _import_pyscf_or_skip()
    backend = _import_backend_or_skip()

    from pyscf_vscf.constants import ANG_TO_BOHR
    from pyscf_vscf.molecule import Molecule

    hf = Molecule.from_arrays(["H", "F"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]], label="HF")
    energy, gradient = backend.energy_gradient_at_coords_bohr(
        hf,
        _hf_sto3g_settings(backend),
        hf.coords * ANG_TO_BOHR,
    )

    assert -105.0 < energy < -90.0
    assert gradient.shape == (6,)
    assert np.all(np.isfinite(gradient))


@pytest.mark.pyscf
def test_energy_dipole_tiny_hf_is_finite() -> None:
    _import_pyscf_or_skip()
    _import_backend_or_skip()

    from pyscf_vscf.molecule import Molecule
    from pyscf_vscf.surfaces import energy_dipole

    backend = importlib.import_module(BACKEND_MODULE)
    hf = Molecule.from_arrays(["H", "F"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]], label="HF")
    energy, dipole = energy_dipole(hf, _hf_sto3g_settings(backend))

    assert -105.0 < energy < -90.0
    assert dipole.shape == (3,)
    assert np.all(np.isfinite(dipole))
