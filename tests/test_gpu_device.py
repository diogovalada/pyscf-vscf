"""The GPU device switch, exercised on the CPU with stand-in device objects.

Real GPU4PySCF parity is checked by ``tests/test_gpu_parity.py`` (``-m gpu``)
on GPU hardware.
"""

from __future__ import annotations

import multiprocessing as mp
import sys

import numpy as np
import pytest

from pyscf_vscf.backends import _gpu
from pyscf_vscf.backends._pyscf_factory import (
    BackendUnavailableError,
    ElectronicSystem,
    MethodRecipe,
    SettingsFactory,
    build_mean_field,
)
from pyscf_vscf.backends._pyscf_properties import to_numpy
from pyscf_vscf.backends.pyscf import PySCFMeanFieldProvider
from pyscf_vscf.contracts import provider_scientific_fingerprint
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.runner import RunnerSettings, _worker_init, run_plan
from pyscf_vscf.scans import plan_bond_scan
from pyscf_vscf.settings import ESSettings

WATER = Molecule.from_arrays(
    ["O", "H", "H"],
    [[0.0, 0.0, 0.1173], [0.0, 0.7572, -0.4692], [0.0, -0.7572, -0.4692]],
    label="water",
)
HF_STO3G = ESSettings(method="hf", basis="sto-3g", use_density_fit=False)


class FakeDeviceArray:
    """Behaves like a CuPy array at the boundary: no implicit NumPy conversion."""

    __cuda_array_interface__ = {"version": 3}

    def __init__(self, value) -> None:
        self._value = np.asarray(value)

    def get(self) -> np.ndarray:
        return self._value.copy()

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed")


class FakeGPUMeanField:
    """Wraps a CPU mean field and returns device arrays, as GPU4PySCF does."""

    _DEVICE_ATTRIBUTES = frozenset({"mo_coeff", "mo_occ", "mo_energy"})

    def __init__(self, cpu) -> None:
        object.__setattr__(self, "_cpu", cpu)

    def __getattr__(self, name):
        value = getattr(self._cpu, name)
        if name in self._DEVICE_ATTRIBUTES and isinstance(value, np.ndarray):
            return FakeDeviceArray(value)
        return value

    def __setattr__(self, name, value) -> None:
        setattr(self._cpu, name, value)

    def kernel(self, *args, **kwargs):
        return self._cpu.kernel(*args, **kwargs)

    def make_rdm1(self, *args, **kwargs):
        return FakeDeviceArray(self._cpu.make_rdm1(*args, **kwargs))

    def get_ovlp(self, *args, **kwargs):
        return FakeDeviceArray(self._cpu.get_ovlp(*args, **kwargs))

    def nuc_grad_method(self):
        return _FakeGPUGradients(self._cpu.nuc_grad_method())


class _FakeGPUGradients:
    def __init__(self, cpu) -> None:
        self._cpu = cpu

    def kernel(self, *args, **kwargs):
        return FakeDeviceArray(self._cpu.kernel(*args, **kwargs))


class FakeGPUSubclass(FakeGPUMeanField):
    """A user subclass of a GPU4PySCF class: still a GPU object."""


FakeGPUMeanField.__module__ = "gpu4pyscf.fake"
FakeGPUSubclass.__module__ = "mylab.methods"


def _fake_to_gpu(mean_field):
    return FakeGPUMeanField(mean_field)


@pytest.fixture
def fake_gpu(monkeypatch):
    """Make ``device='gpu'`` builds use the stand-in instead of GPU4PySCF."""

    monkeypatch.setattr(_gpu, "require_gpu", lambda: (None, None))
    from pyscf import dft, scf

    for method in (scf.hf.RHF, dft.rks.RKS):
        for cls in method.__mro__:
            if "to_gpu" in vars(cls):
                monkeypatch.setattr(cls, "to_gpu", _fake_to_gpu)


@pytest.fixture
def no_gpu4pyscf(monkeypatch):
    monkeypatch.setitem(sys.modules, "gpu4pyscf", None)


def test_device_names_are_validated() -> None:
    assert _gpu.normalize_device(" GPU ") == "gpu"
    with pytest.raises(ValueError, match="'cpu' or 'gpu'"):
        _gpu.normalize_device("cuda")
    with pytest.raises(TypeError):
        _gpu.normalize_device(None)
    with pytest.raises(ValueError, match="'cpu' or 'gpu'"):
        SettingsFactory(ESSettings(device="tpu"))


def test_cpu_identity_is_unchanged_and_gpu_identity_differs() -> None:
    cpu = SettingsFactory(ESSettings())
    gpu = SettingsFactory(ESSettings(device="gpu"))
    assert "device" not in cpu.scientific_settings_payload()
    assert gpu.scientific_settings_payload()["device"] == "gpu"
    assert {
        **gpu.scientific_settings_payload(),
        "device": None,
    } != cpu.scientific_settings_payload()
    cpu_provider = PySCFMeanFieldProvider(ESSettings())
    gpu_provider = PySCFMeanFieldProvider(ESSettings(device="gpu"))
    assert (cpu_provider.device, gpu_provider.device) == ("cpu", "gpu")
    assert provider_scientific_fingerprint(cpu_provider) != provider_scientific_fingerprint(
        gpu_provider
    )
    # CPU fingerprints are those of 0.3.1 (computed with the released code).
    assert provider_scientific_fingerprint(cpu_provider) == (
        "e5f2ab15abaa6101488b6d346f988b3b545802a96016f5c5a660b0c9419881de"
    )
    assert provider_scientific_fingerprint(PySCFMeanFieldProvider(HF_STO3G)) == (
        "3d0f98b055284c56edfc70117e34ddab209c6d0ee0dca154166dec64acd93265"
    )
    assert provider_scientific_fingerprint(PySCFMeanFieldProvider(factory=_recipe())) == (
        "83e0397c2f18fad2f1aca4a3135250c900099b348667a649e0fef526c4bf9cef"
    )


def test_legacy_cache_identity_records_only_a_gpu_device() -> None:
    from pyscf_vscf.cache import electronic_structure_identity

    cpu = electronic_structure_identity(ESSettings())
    assert "device" not in cpu
    assert electronic_structure_identity(ESSettings(device="gpu")) == {**cpu, "device": "gpu"}


class _CustomFactory:
    """A factory outside the package whose manifest omits the device."""

    def __init__(self, device: str) -> None:
        self.device = device

    def scientific_settings_payload(self):
        return {"method": "hf", "basis": "sto-3g"}

    def build_molecule(self, system):
        return SettingsFactory(HF_STO3G).build_molecule(system)

    def build_mean_field(self, molecule):
        return molecule.RHF()


def test_the_provider_records_the_device_of_any_factory() -> None:
    cpu = PySCFMeanFieldProvider(factory=_CustomFactory("cpu"))
    gpu = PySCFMeanFieldProvider(factory=_CustomFactory("gpu"))
    assert "device" not in cpu.scientific_settings_payload()
    assert gpu.scientific_settings_payload()["device"] == "gpu"
    assert provider_scientific_fingerprint(cpu) != provider_scientific_fingerprint(gpu)


def _hf(mol):
    return mol.RHF()


def _recipe(**kwargs) -> MethodRecipe:
    return MethodRecipe(
        recipe_id="test.hf",
        recipe_version="1",
        basis="sto-3g",
        configuration={"method": "hf"},
        mean_field=_hf,
        **kwargs,
    )


def test_recipe_device_sets_the_implementation_and_identity() -> None:
    cpu = _recipe()
    gpu = _recipe(device="gpu")
    assert (cpu.implementation, gpu.implementation) == ("pyscf-cpu", "gpu4pyscf")
    assert "device" not in cpu.scientific_settings_payload()
    assert gpu.scientific_settings_payload()["device"] == "gpu"
    assert cpu.on_device("gpu") == gpu
    assert gpu.on_device("cpu") == cpu
    assert cpu.on_device("cpu") is cpu
    custom = _recipe(implementation="mylab-build", device="gpu")
    assert custom.implementation == "mylab-build"
    assert custom.on_device("cpu").implementation == "mylab-build"
    with pytest.raises(ValueError, match="needs device='gpu'"):
        _recipe(implementation="gpu4pyscf")


@pytest.mark.pyscf
def test_missing_gpu4pyscf_is_reported_with_an_install_hint(no_gpu4pyscf) -> None:
    pytest.importorskip("pyscf")
    system = ElectronicSystem.from_molecule(WATER)
    with pytest.raises(BackendUnavailableError, match="gpu-cuda12"):
        build_mean_field(
            system, SettingsFactory(ESSettings(**{**vars(HF_STO3G), "device": "gpu"}))
        )


@pytest.mark.pyscf
def test_gpu_build_converts_the_configured_mean_field(fake_gpu) -> None:
    pytest.importorskip("pyscf")
    settings = ESSettings(
        method="b3lyp", basis="sto-3g", use_density_fit=False, dft_grid_level=2, device="gpu"
    )
    build = build_mean_field(ElectronicSystem.from_molecule(WATER), SettingsFactory(settings))
    assert isinstance(build.mean_field, FakeGPUMeanField)
    assert build.configuration_snapshot["class"] == "gpu4pyscf.fake.FakeGPUMeanField"
    assert build.configuration_snapshot["grids_level"] == 2
    assert build.configuration_snapshot["conv_tol"] == 1e-10


@pytest.mark.pyscf
def test_gpu_results_cross_the_numpy_boundary(fake_gpu) -> None:
    pytest.importorskip("pyscf")
    from pyscf_vscf.contracts import ElectronicPointRequest

    gpu = PySCFMeanFieldProvider(ESSettings(**{**vars(HF_STO3G), "device": "gpu"}))
    cpu = PySCFMeanFieldProvider(HF_STO3G)

    def evaluate(provider):
        request = ElectronicPointRequest(nuclear_charges=(8, 1, 1), coordinates_A=WATER.coords)
        return provider.evaluate(request)

    on_gpu, on_cpu = evaluate(gpu), evaluate(cpu)
    assert on_gpu.total_energy_Eh == pytest.approx(on_cpu.total_energy_Eh, abs=1e-10)
    np.testing.assert_allclose(on_gpu.dipole_au, on_cpu.dipole_au, atol=1e-8)
    assert on_gpu.provider_scientific_fingerprint != on_cpu.provider_scientific_fingerprint
    assert "gpu" in on_gpu.provenance


@pytest.mark.pyscf
def test_a_recipe_that_returns_a_gpu_object_must_declare_the_device(fake_gpu) -> None:
    pytest.importorskip("pyscf")

    recipe = MethodRecipe(
        recipe_id="test.sneaky",
        recipe_version="1",
        basis="sto-3g",
        configuration={"method": "hf", "device": "gpu (undeclared)"},
        mean_field=_hf_to_gpu,
    )
    with pytest.raises(ValueError, match="set device='gpu'"):
        build_mean_field(ElectronicSystem.from_molecule(WATER), recipe)


def _hf_to_gpu(mol):
    return mol.RHF().to_gpu()


def test_subclasses_of_gpu_classes_are_gpu_objects() -> None:
    assert _gpu.is_gpu_object(FakeGPUSubclass(object()))
    assert not _gpu.is_gpu_object(np.zeros(1))


@pytest.mark.pyscf
def test_gpu_orbitals_and_gradients_cross_the_numpy_boundary(fake_gpu) -> None:
    pytest.importorskip("pyscf")
    from pyscf_vscf.backends._pyscf_properties import converge, nuclear_gradient
    from pyscf_vscf.contracts import ElectronicPointRequest

    gpu = PySCFMeanFieldProvider(
        ESSettings(**{**vars(HF_STO3G), "device": "gpu"}), continuity_diagnostics="strict"
    )
    cpu = PySCFMeanFieldProvider(HF_STO3G, continuity_diagnostics="strict")
    request = ElectronicPointRequest(nuclear_charges=(8, 1, 1), coordinates_A=WATER.coords)
    on_gpu, on_cpu = gpu.evaluate(request), cpu.evaluate(request)
    for key in ("mulliken_density_atom_populations", "occupied_orbital_energies_Eh"):
        np.testing.assert_allclose(
            on_gpu.scientific_diagnostics[key], on_cpu.scientific_diagnostics[key], atol=1e-8
        )
    system = ElectronicSystem.from_molecule(WATER)
    gradients = [
        nuclear_gradient(converge(build_mean_field(system, SettingsFactory(settings))))
        for settings in (HF_STO3G, ESSettings(**{**vars(HF_STO3G), "device": "gpu"}))
    ]
    np.testing.assert_allclose(gradients[0], gradients[1], atol=1e-8)


def test_device_arrays_are_copied_not_implicitly_converted() -> None:
    array = FakeDeviceArray([[1.0, 2.0]])
    with pytest.raises(TypeError):
        np.asarray(array)
    np.testing.assert_array_equal(to_numpy(array), [[1.0, 2.0]])


def test_gpu_workers_need_spawn() -> None:
    if "fork" not in mp.get_all_start_methods():
        pytest.skip("fork is unavailable on this platform")
    provider = PySCFMeanFieldProvider(ESSettings(**{**vars(HF_STO3G), "device": "gpu"}))
    plan = plan_bond_scan(WATER, provider, ["0-1"], [np.linspace(0.9, 1.1, 3)])
    with pytest.raises(ValueError, match="spawn"):
        run_plan(plan, provider, settings=RunnerSettings(workers=2, start_method="fork"))


def test_workers_are_pinned_to_gpus_round_robin(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    counter = mp.get_context("spawn").Value("i", 0)
    pinned = []
    for _ in range(3):
        _worker_init(None, None, ("3", "5"), counter)
        pinned.append(__import__("os").environ["CUDA_VISIBLE_DEVICES"])
    assert pinned == ["3", "5", "3"]


def test_pinning_refuses_a_worker_where_cuda_may_already_run(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cupy", object())
    counter = mp.get_context("spawn").Value("i", 0)
    with pytest.raises(RuntimeError, match="imported before the worker started"):
        _worker_init(None, None, ("0", "1"), counter)


def test_visible_gpu_ids_honour_cuda_visible_devices(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2, 3")
    assert _gpu.visible_gpu_ids() == ("2", "3")
    # CUDA stops at the first invalid entry.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,2,-1,1")
    assert _gpu.visible_gpu_ids() == ("0", "2")
    for hidden in ("", "-1"):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", hidden)
        with pytest.raises(RuntimeError, match="hides every device"):
            _gpu.visible_gpu_ids()


def test_cli_device_gpu_without_gpu4pyscf_fails_before_computing(
    tmp_path, capsys, no_gpu4pyscf
) -> None:
    from pyscf_vscf import cli

    geometry = tmp_path / "water.xyz"
    geometry.write_text(
        "3\nwater\nO 0 0 0.1173\nH 0 0.7572 -0.4692\nH 0 -0.7572 -0.4692\n", encoding="utf-8"
    )
    assert cli.main(["harmonic", str(geometry), "--device", "gpu"]) == 1
    assert "gpu-cuda12" in capsys.readouterr().err


def test_cli_reuse_refuses_device(tmp_path, capsys) -> None:
    from pyscf_vscf import cli

    geometry = tmp_path / "water.xyz"
    geometry.write_text("3\nwater\nO 0 0 0\nH 0 0.76 0.59\nH 0 -0.76 0.59\n", encoding="utf-8")
    scan = tmp_path / "scan.npz"
    scan.write_bytes(b"")
    with pytest.raises(SystemExit) as exc:
        cli.main(
            ["scan1d", str(geometry), "--bond", "0-1", "--reuse", str(scan), "--device", "gpu"]
        )
    assert exc.value.code == 2
    assert "--device" in capsys.readouterr().err
