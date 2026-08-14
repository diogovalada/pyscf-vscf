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
            }
        settings_cls = ESSettings

    return settings_cls(
        method="hf",
        basis="sto-3g",
        use_density_fit=False,
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


def _complete_continuity_diagnostics() -> dict[str, object]:
    return {
        "continuity_descriptor_scope": ("closed-shell-mulliken-and-meta-lowdin-ao-metrics"),
        "mulliken_density_atom_populations": [1.9, 0.1],
        "meta_lowdin_density_atom_populations": [1.8, 0.2],
        "meta_lowdin_pre_orthogonalization": "ANO",
        "mulliken_occupied_orbital_atom_populations": [[0.95, 0.05]],
        "occupied_orbital_energies_Eh": [-0.5],
        "reference_homo_lumo_gap_Eh": 0.3,
        "occupied_mo_coefficients_ao": [[0.8], [0.6]],
    }


def test_continuity_success_schema_rejects_malformed_fields() -> None:
    backend = importlib.import_module(BACKEND_MODULE)
    valid = _complete_continuity_diagnostics()
    frozen = backend._validated_continuity_diagnostics(
        valid,
        require_coefficients=True,
    )
    assert frozen["reference_homo_lumo_gap_Eh"] == pytest.approx(0.3)

    malformed = (
        {**valid, "mulliken_density_atom_populations": "not-an-array"},
        {**valid, "meta_lowdin_density_atom_populations": [2.0]},
        {**valid, "mulliken_occupied_orbital_atom_populations": [[1.0]]},
        {**valid, "occupied_orbital_energies_Eh": []},
        {**valid, "reference_homo_lumo_gap_Eh": None},
        {**valid, "occupied_mo_coefficients_ao": [[0.8, 0.1], [0.6, 0.2]]},
    )
    for diagnostics in malformed:
        with pytest.raises(ValueError):
            backend._validated_continuity_diagnostics(
                diagnostics,
                require_coefficients=True,
            )


@pytest.mark.pyscf
def test_mean_field_provider_agrees_with_released_energy_dipole() -> None:
    _import_pyscf_or_skip()
    from pyscf_vscf.electronic import (
        AU_DIPOLE_TO_DEBYE,
        ElectronicPointRequest,
        energy_dipole,
    )
    from pyscf_vscf.molecule import Molecule

    backend = importlib.import_module(BACKEND_MODULE)
    molecule = Molecule.from_arrays(
        ["H", "F"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]],
        label="HF",
    )
    settings = _hf_sto3g_settings(backend)
    provider = backend.PySCFMeanFieldProvider(
        settings,
        threads=1,
        max_memory_mb=512,
        continuity_diagnostics="strict",
        retain_occupied_mo_coefficients=True,
    )
    request = ElectronicPointRequest(
        nuclear_charges=(1, 9),
        coordinates_A=molecule.coords,
    )

    expected_energy, expected_dipole_debye = energy_dipole(molecule, settings)
    result = provider.evaluate(request)

    assert result.total_energy_Eh == pytest.approx(expected_energy, abs=1e-12)
    np.testing.assert_allclose(
        result.dipole_au * AU_DIPOLE_TO_DEBYE,
        expected_dipole_debye,
        rtol=0.0,
        atol=1e-11,
    )
    assert result.dipole_unit == "atomic_unit"
    assert result.dipole_frame == "input_cartesian"
    diagnostics = result.scientific_diagnostics
    assert diagnostics["continuity_descriptor_scope"] == (
        "closed-shell-mulliken-and-meta-lowdin-ao-metrics"
    )
    assert len(diagnostics["mulliken_density_atom_populations"]) == 2
    assert len(diagnostics["meta_lowdin_density_atom_populations"]) == 2
    assert diagnostics["meta_lowdin_pre_orthogonalization"] == "ANO"
    assert sum(diagnostics["meta_lowdin_density_atom_populations"]) == pytest.approx(
        10.0, abs=1e-10
    )
    assert len(diagnostics["mulliken_occupied_orbital_atom_populations"]) == 5
    assert "density_atom_populations" not in diagnostics
    assert "occupied_orbital_atom_populations" not in diagnostics
    assert diagnostics["reference_homo_lumo_gap_Eh"] > 0.0
    assert np.asarray(diagnostics["occupied_mo_coefficients_ao"]).shape[1] == 5


@pytest.mark.pyscf
def test_continuity_collection_modes_do_not_discard_valid_energy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _import_pyscf_or_skip()
    from pyscf_vscf.electronic import ElectronicPointRequest

    backend = importlib.import_module(BACKEND_MODULE)
    settings = _hf_sto3g_settings(backend)
    request = ElectronicPointRequest(
        nuclear_charges=(1, 1),
        coordinates_A=[[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
        requested_properties=("energy",),
    )

    def fail_diagnostics(*_args, **_kwargs):
        raise ValueError("simulated population-analysis failure")

    monkeypatch.setattr(backend, "_mean_field_continuity_diagnostics", fail_diagnostics)

    disabled = backend.PySCFMeanFieldProvider(settings).evaluate(request)
    assert "continuity_descriptor_scope" not in disabled.scientific_diagnostics

    best_effort = backend.PySCFMeanFieldProvider(
        settings,
        continuity_diagnostics="best-effort",
    ).evaluate(request)
    assert best_effort.total_energy_Eh == pytest.approx(disabled.total_energy_Eh, abs=1e-12)
    assert (
        best_effort.scientific_diagnostics["continuity_descriptor_scope"]
        == "unavailable-after-error"
    )
    assert (
        best_effort.scientific_diagnostics["continuity_descriptor_error_type"]
        == "builtins.ValueError"
    )

    with pytest.raises(RuntimeError, match="Strict continuity diagnostics failed"):
        backend.PySCFMeanFieldProvider(
            settings,
            continuity_diagnostics="strict",
        ).evaluate(request)

    def nonfinite_diagnostics(*_args, **_kwargs):
        return {
            "continuity_descriptor_scope": "closed-shell-test-descriptor",
            "invalid_value": np.nan,
        }

    monkeypatch.setattr(backend, "_mean_field_continuity_diagnostics", nonfinite_diagnostics)

    best_effort_nonfinite = backend.PySCFMeanFieldProvider(
        settings,
        continuity_diagnostics="best-effort",
    ).evaluate(request)
    assert best_effort_nonfinite.total_energy_Eh == pytest.approx(
        disabled.total_energy_Eh, abs=1e-12
    )
    assert (
        best_effort_nonfinite.scientific_diagnostics["continuity_descriptor_scope"]
        == "unavailable-after-error"
    )
    assert (
        best_effort_nonfinite.scientific_diagnostics["continuity_descriptor_error_type"]
        == "builtins.ValueError"
    )

    monkeypatch.setattr(
        backend,
        "_mean_field_continuity_diagnostics",
        lambda *_args, **_kwargs: {"unexpected": "JSON-valid but incomplete"},
    )
    best_effort_malformed = backend.PySCFMeanFieldProvider(
        settings,
        continuity_diagnostics="best-effort",
    ).evaluate(request)
    assert best_effort_malformed.total_energy_Eh == pytest.approx(
        disabled.total_energy_Eh, abs=1e-12
    )
    assert (
        best_effort_malformed.scientific_diagnostics["continuity_descriptor_scope"]
        == "unavailable-after-error"
    )


@pytest.mark.pyscf
@pytest.mark.parametrize(
    "diagnostics",
    [
        {"continuity_descriptor_scope": "unavailable-for-open-shell-reference"},
        {
            "continuity_descriptor_scope": ("closed-shell-mulliken-and-meta-lowdin-ao-metrics"),
            "mulliken_density_atom_populations": [2.0],
        },
    ],
)
def test_strict_continuity_rejects_unavailable_or_incomplete_descriptors(
    monkeypatch: pytest.MonkeyPatch,
    diagnostics: dict[str, object],
) -> None:
    _import_pyscf_or_skip()
    from pyscf_vscf.electronic import ElectronicPointRequest

    backend = importlib.import_module(BACKEND_MODULE)
    settings = _hf_sto3g_settings(backend)
    request = ElectronicPointRequest(
        nuclear_charges=(1, 1),
        coordinates_A=[[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]],
        requested_properties=("energy",),
    )
    monkeypatch.setattr(
        backend,
        "_mean_field_continuity_diagnostics",
        lambda *_args, **_kwargs: diagnostics,
    )

    with pytest.raises(RuntimeError, match="Strict continuity diagnostics failed"):
        backend.PySCFMeanFieldProvider(
            settings,
            continuity_diagnostics="strict",
        ).evaluate(request)


def test_continuity_diagnostics_reject_open_shell_occupations() -> None:
    backend = _import_backend_or_skip()
    mean_field = SimpleNamespace(
        mo_coeff=np.eye(2),
        mo_occ=np.array([2.0, 1.0]),
        mo_energy=np.array([-0.5, -0.1]),
        make_rdm1=lambda: np.diag([2.0, 1.0]),
        get_ovlp=lambda: np.eye(2),
        mol=SimpleNamespace(aoslice_by_atom=lambda: np.array([[0, 0, 0, 2]])),
    )

    diagnostics = backend._mean_field_continuity_diagnostics(
        mean_field,
        retain_occupied_mo_coefficients=True,
    )

    assert diagnostics == {"continuity_descriptor_scope": "unavailable-for-open-shell-reference"}


@pytest.mark.pyscf
def test_mean_field_finite_difference_matches_signed_analytic_dipole_and_origin() -> None:
    _import_pyscf_or_skip()
    from pyscf_vscf.electronic import ElectronicPointRequest
    from pyscf_vscf.molecule import Molecule

    backend = importlib.import_module(BACKEND_MODULE)
    molecule = Molecule.from_arrays(
        ["H", "F"],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.92]],
        label="HF",
    )
    settings = _hf_sto3g_settings(backend)
    provider = backend.PySCFMeanFieldProvider(settings, threads=1, max_memory_mb=512)
    base = {
        "nuclear_charges": (1, 9),
        "coordinates_A": molecule.coords,
    }
    analytic = provider.evaluate(ElectronicPointRequest(**base))
    step = 1e-4
    plus_request = ElectronicPointRequest(
        **base,
        requested_properties=("energy",),
        field_au=[0.0, 0.0, step],
    )
    minus_request = ElectronicPointRequest(
        **base,
        requested_properties=("energy",),
        field_au=[0.0, 0.0, -step],
    )
    shifted_request = ElectronicPointRequest(
        **base,
        requested_properties=("energy",),
        field_au=[0.0, 0.0, step],
        field_origin_A=[0.2, -0.1, 0.3],
    )
    plus = provider.evaluate(plus_request)
    minus = provider.evaluate(minus_request)
    shifted = provider.evaluate(shifted_request)

    finite_difference_dipole = -(plus.total_energy_Eh - minus.total_energy_Eh) / (2 * step)
    assert finite_difference_dipole == pytest.approx(analytic.dipole_au[2], abs=2e-5)
    assert shifted.total_energy_Eh == pytest.approx(plus.total_energy_Eh, abs=1e-10)
    assert plus.point_causal_fingerprint != minus.point_causal_fingerprint
    assert plus.point_causal_fingerprint != shifted.point_causal_fingerprint


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
        @staticmethod
        def RKS(pmol, *, xc):
            assert xc == "pbe"
            return FakeSCF.RHF(pmol)

        @staticmethod
        def UKS(pmol, *, xc):
            assert xc == "pbe"
            return FakeSCF.UHF(pmol)

    monkeypatch.setattr(
        backend,
        "_require_pyscf",
        lambda: (object(), FakeSCF, FakeDFT, object()),
    )
    mean_field = backend.make_mean_field(
        SimpleNamespace(spin=1),
        SimpleNamespace(
            method="pbe",
            use_density_fit=False,
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
