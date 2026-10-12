from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pyscf_vscf.coordinates import Bond
from pyscf_vscf.cache import scientific_fingerprint
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.workflows import scans

pytestmark = pytest.mark.filterwarnings("ignore::pyscf_vscf._warnings.PyscfVscfDeprecationWarning")


class _RecordingExecutor:
    def __init__(self, calls: list[object]):
        self.calls = calls

    def __enter__(self) -> "_RecordingExecutor":
        self.calls.append("enter")
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.calls.append("exit")

    def map(self, func, tasks):
        self.calls.append(("map", len(tasks)))
        return [func(task) for task in tasks]


def _water() -> Molecule:
    return Molecule.from_arrays(
        ["O", "H", "H"],
        np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
    )


def _fake_energy_dipole(molecule: object, cfg: object) -> tuple[float, np.ndarray]:
    del cfg
    coords = np.asarray(getattr(molecule, "coords"), dtype=float)
    return float(np.sum(coords * coords)), np.sum(coords, axis=0)


def _cfg(**overrides: object) -> SimpleNamespace:
    values = {
        "method": "hf",
        "basis": "sto-3g",
        "use_density_fit": False,
        "auxbasis": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_importing_scan_workflow_module_does_not_request_pyscf() -> None:
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
import pyscf_vscf.workflows.scans

print("scans-ok")
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
    assert proc.stdout.strip() == "scans-ok"


def test_local_bond_reduced_mass_uses_selected_atom_masses_for_non_water() -> None:
    mol = Molecule.from_arrays(
        ["H", "F"],
        np.array([[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]]),
        label="HF",
    )

    expected = 1.00782503223 * 18.99840316273 / (1.00782503223 + 18.99840316273)

    assert scans.local_bond_reduced_mass_amu(mol, "0-1") == pytest.approx(expected)
    with pytest.raises(IndexError, match="out of range"):
        scans.local_bond_reduced_mass_amu(mol, "0-2")


def test_bond_bond_g12_uses_actual_shared_atom_mass_and_orientation() -> None:
    mol = Molecule.from_arrays(
        ["C", "H", "H"],
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        label="methylenelike",
    )

    assert scans.bond_bond_g12_inv_amu(mol, "0-1", "0-2") == pytest.approx(0.0)
    assert scans.bond_bond_g12_inv_amu(mol, "0-1", "1-2") == pytest.approx(
        1.0 / np.sqrt(2.0) / 1.00782503223
    )

    bent_carbon = Molecule.from_arrays(
        ["C", "H", "H"],
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]]),
        label="bent_ch2",
    )
    assert scans.bond_bond_g12_inv_amu(bent_carbon, "0-1", "0-2") == pytest.approx(0.5 / 12.0)
    with pytest.raises(ValueError, match="distinct bonds"):
        scans.bond_bond_g12_inv_amu(mol, "0-1", "1-0")


def test_normal_mode_selection_uses_bond_projection_scoring() -> None:
    mol = Molecule.from_arrays(
        ["O", "H"],
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    masses = mol.analysis_masses()
    modes = np.zeros((6, 2), dtype=float)
    modes[4, 0] = np.sqrt(masses[1])
    modes[3, 1] = 2.0 * np.sqrt(masses[1])
    modes[0, 1] = -0.25 * np.sqrt(masses[0])
    freqs = np.array([111.0, 222.0])

    selected = scans.normal_mode_direction_from_modes(mol, Bond(0, 1), modes, freqs)

    assert selected.mode_index == 1
    assert selected.frequency_cm == pytest.approx(222.0)
    assert np.linalg.norm(selected.u_dir) == pytest.approx(1.0)
    expected_mu = np.sum(masses * np.sum(selected.u_dir * selected.u_dir, axis=1))
    assert scans.normal_mode_effective_mass_amu(mol, selected.u_dir) == pytest.approx(expected_mu)


def test_calc_normal_mode_direction_uses_injected_harmonic_function() -> None:
    mol = Molecule.from_arrays(
        ["O", "H"],
        np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    masses = mol.analysis_masses()
    modes = np.zeros((6, 1), dtype=float)
    modes[3, 0] = np.sqrt(masses[1])
    logs: list[str] = []
    kwargs_seen: dict[str, object] = {}

    def fake_harmonic(molecule, cfg, **kwargs):
        del molecule, cfg
        kwargs_seen.update(kwargs)
        return SimpleNamespace(modes=modes, freqs_cm=np.array([321.0]))

    result = scans.calc_normal_mode_direction(
        mol,
        _cfg(),
        "0-1",
        rtproj="mw_explicit",
        harmonic_fn=fake_harmonic,
        log_fn=logs.append,
    )

    assert result.mode_index == 0
    assert result.frequency_cm == pytest.approx(321.0)
    assert kwargs_seen == {
        "rtproj": "mw_explicit",
        "strict": True,
        "allow_fd_hessian": False,
        "debug": False,
    }
    assert "Selected normal mode index 0" in logs[0]


@pytest.mark.parametrize(
    ("energy", "dipole", "message"),
    [
        (np.nan, np.zeros(3), "energy must be finite"),
        (0.0, np.array([0.0, np.inf, 0.0]), "dipole must be finite"),
    ],
)
def test_relaxed_scan_rejects_nonfinite_point_results(energy, dipole, message) -> None:
    def point(molecule, cfg, u_dir, s, gtol, maxiter):
        del molecule, cfg, u_dir, gtol, maxiter
        return SimpleNamespace(
            energy_hartree=energy,
            dipole_debye=dipole,
            achieved_displacement_A=s,
            constraint_residual_A=0.0,
            converged=True,
            n_iterations=1,
            message="ok",
        )

    direction = np.zeros((3, 3))
    direction[1, 0] = 1.0
    with pytest.raises(scans.ScanFailedError, match=message) as excinfo:
        scans.grid_1d_pes_dms_normal_relaxed(
            _water(), _cfg(), direction, -0.1, 0.1, 3, relaxed_point_fn=point
        )
    failure = excinfo.value.partial.failures[0]
    assert failure.error_type == "ValueError"
    assert message in failure.message


def test_two_bond_stretch_is_orientation_independent_for_shared_atom() -> None:
    coords = np.array(
        [
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    canonical = scans.stretch_two_bonds(coords, Bond(1, 0), Bond(1, 2), 1.2, 0.8)
    reversed_bonds = scans.stretch_two_bonds(coords, Bond(0, 1), Bond(2, 1), 1.2, 0.8)

    np.testing.assert_allclose(reversed_bonds, canonical)
    assert np.linalg.norm(reversed_bonds[0] - reversed_bonds[1]) == pytest.approx(1.2)
    assert np.linalg.norm(reversed_bonds[2] - reversed_bonds[1]) == pytest.approx(0.8)


def test_two_bond_stretch_uses_explicit_anchors_for_disjoint_bonds() -> None:
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [3.0, 1.0, 0.0],
        ]
    )

    stretched = scans.stretch_two_bonds(coords, Bond(0, 1), Bond(2, 3), 1.4, 0.7)

    np.testing.assert_allclose(stretched[[0, 2]], coords[[0, 2]])
    assert np.linalg.norm(stretched[1] - stretched[0]) == pytest.approx(1.4)
    assert np.linalg.norm(stretched[3] - stretched[2]) == pytest.approx(0.7)


def test_two_bond_plans_reject_invalid_bonds_and_lengths() -> None:
    from pyscf_vscf.backends.pyscf import PySCFMeanFieldProvider
    from pyscf_vscf.scans import plan_bond_scan
    from pyscf_vscf.settings import ESSettings

    provider = PySCFMeanFieldProvider(ESSettings(method="hf", basis="sto-3g"))
    grid = np.linspace(0.9, 1.1, 3)
    for bonds, grids in (
        ([Bond(0, 1), Bond(1, 0)], [grid, grid]),
        ([Bond(0, 1), Bond(0, 2)], [grid, np.array([0.0, 0.5, 1.0])]),
        ([Bond(0, 0), Bond(0, 2)], [grid, grid]),
    ):
        with pytest.raises(ValueError):
            plan_bond_scan(_water(), provider, bonds, grids)


def test_normal_relaxed_grid_is_represented_by_injected_point_optimizer() -> None:
    mol = _water()
    u_dir = np.zeros_like(mol.coords)
    u_dir[1, 0] = 1.0
    calls: list[tuple[float, float, int]] = []

    with pytest.raises(NotImplementedError, match="injected relaxed_point_fn"):
        scans.grid_1d_pes_dms_normal_relaxed(mol, _cfg(), u_dir, npts=1)

    def relaxed_point(
        molecule: object,
        cfg: object,
        direction: np.ndarray,
        s: float,
        gtol: float,
        maxiter: int,
    ) -> object:
        del molecule, cfg, direction
        calls.append((s, gtol, maxiter))
        return SimpleNamespace(
            energy_hartree=1.0 + s * s,
            dipole_debye=np.array([s, 0.0, 0.0]),
            achieved_displacement_A=s,
            constraint_residual_A=0.0,
            converged=True,
            n_iterations=2,
            message="converged",
        )

    result = scans.grid_1d_pes_dms_normal_relaxed(
        mol,
        _cfg(),
        u_dir,
        smin=-0.1,
        smax=0.1,
        npts=3,
        relaxed_point_fn=relaxed_point,
        gtol=1e-5,
        maxiter=7,
    )
    np.testing.assert_allclose(result.displacements_A, [-0.1, 0.0, 0.1])
    np.testing.assert_allclose(result.energies_hartree, [0.01, 0.0, 0.01])
    np.testing.assert_allclose(result.dipoles_debye[:, 0], [-0.1, 0.0, 0.1])
    assert calls[-1] == (0.1, 1e-5, 7)
    np.testing.assert_allclose(result.constraint_residuals_A, 0.0)
    assert result.failed_indices == ()


def test_normal_relaxed_grid_warns_once_and_retains_diagnostics_when_not_strict() -> None:
    mol = _water()
    cfg = _cfg(strict=False)
    u_dir = np.zeros_like(mol.coords)
    u_dir[1, 0] = 1.0

    def relaxed_point(molecule, cfg, direction, s, gtol, maxiter):
        del molecule, cfg, direction, gtol, maxiter
        return SimpleNamespace(
            energy_hartree=s * s,
            dipole_debye=np.zeros(3),
            achieved_displacement_A=s,
            constraint_residual_A=0.0,
            converged=s != 0.0,
            n_iterations=3,
            message="failed" if s == 0.0 else "ok",
        )

    with pytest.warns(RuntimeWarning, match=r"failed at 1 point.*indices=\[1\]"):
        result = scans.grid_1d_pes_dms_normal_relaxed(
            mol,
            cfg,
            u_dir,
            smin=-0.1,
            smax=0.1,
            npts=3,
            strict=False,
            relaxed_point_fn=relaxed_point,
        )

    assert result.failed_indices == (1,)
    assert "max constraint residual=0.000e+00 A" in result.failure_summary()
    with pytest.raises(TypeError):
        tuple(result)


def test_1d_lbs_cache_metadata_validation_and_legacy_read(tmp_path: Path) -> None:
    from pyscf_vscf.artifacts.scans import load_legacy_grid
    from pyscf_vscf.cache import dump_grid_npz

    mol = _water()
    cfg = _cfg()
    R = np.linspace(0.8, 1.2, 5)
    E = np.linspace(0.0, 0.4, 5)
    MU = np.zeros((5, 3))
    path = tmp_path / "grid_1d.npz"

    meta = scans.lbs_frozen_1d_cache_metadata(mol, cfg, Bond(0, 1), 0.8, 1.2, 5)
    assert meta["identity"]["scan"]["bond_zero_based"] == [0, 1]
    dump_grid_npz(path, meta=meta, arrays={"R_A": R, "E_Eh": E, "MU_Debye": MU})
    legacy = load_legacy_grid(path)
    np.testing.assert_allclose(legacy.axes[0], R)
    np.testing.assert_allclose(legacy.relative_energies_Eh, E)
    np.testing.assert_allclose(legacy.dipoles_debye, MU)
    scans.validate_lbs_frozen_1d_cache_metadata(meta, mol, cfg, Bond(0, 1), 0.8, 1.2, 5)
    legacy_meta = dict(meta, grid_cache_version=1)
    with pytest.raises(ValueError, match="Unsupported grid cache schema"):
        scans.validate_lbs_frozen_1d_cache_metadata(legacy_meta, mol, cfg, Bond(0, 1), 0.8, 1.2, 5)
    with pytest.raises(ValueError, match="cache_identity_sha256"):
        scans.validate_lbs_frozen_1d_cache_metadata(
            meta, mol, _cfg(basis="6-31g"), Bond(0, 1), 0.8, 1.2, 5
        )


def test_cache_identity_covers_causal_inputs_but_not_policy_or_isotope() -> None:
    mol = _water()
    cfg = _cfg(
        rtproj="mw_explicit",
        strict=False,
        allow_fd_hessian=True,
        scf_conv_tol=3e-9,
        scf_max_cycle=77,
        dft_grid_level=4,
    )
    meta = scans.lbs_frozen_1d_cache_metadata(mol, cfg, Bond(0, 1), 0.8, 1.2, 5)
    es = meta["identity"]["electronic_structure"]
    assert "rtproj" not in es
    assert "strict" not in es
    assert "allow_fd_hessian" not in es
    assert es["scf_conv_tol"] == pytest.approx(3e-9)
    assert es["scf_max_cycle"] == 77
    assert es["dft_grid_level"] == 4
    assert "software_versions" not in es
    assert "software_versions" in meta["provenance"]["electronic_structure"]

    isotope = Molecule.from_arrays(
        ["O", "D", "D"],
        mol.coords,
        label="different-label",
    )
    scans.validate_lbs_frozen_1d_cache_metadata(meta, isotope, cfg, Bond(0, 1), 0.8, 1.2, 5)

    moved = _water()
    moved.coords[1, 0] += 1e-4
    with pytest.raises(ValueError, match="cache_identity_sha256"):
        scans.validate_lbs_frozen_1d_cache_metadata(meta, moved, cfg, Bond(0, 1), 0.8, 1.2, 5)
    with pytest.raises(ValueError, match="cache_identity_sha256"):
        scans.validate_lbs_frozen_1d_cache_metadata(
            meta,
            mol,
            _cfg(
                rtproj="mw_explicit",
                strict=False,
                allow_fd_hessian=True,
                scf_conv_tol=4e-9,
                scf_max_cycle=77,
                dft_grid_level=4,
            ),
            Bond(0, 1),
            0.8,
            1.2,
            5,
        )


def test_schema_v2_cache_metadata_is_migrated_and_runtime_drift_warns() -> None:
    mol = _water()
    cfg = _cfg()
    current = scans.lbs_frozen_1d_cache_metadata(mol, cfg, Bond(0, 1), 0.8, 1.2, 5)
    electronic = dict(current["identity"]["electronic_structure"])
    electronic["software_versions"] = {"pyscf": "old"}
    electronic.update({"rtproj": "pyscf", "strict": True, "allow_fd_hessian": False})
    scientific = {
        "molecule": current["provenance"]["molecule"],
        "electronic_structure": electronic,
        "scan": current["provenance"]["scan"],
    }
    schema_v2 = {
        "grid_cache_version": 2,
        "scientific": scientific,
        "scientific_fingerprint_sha256": scientific_fingerprint(scientific),
        "runtime": {"distributions": {"pyscf": "old"}},
    }

    with pytest.warns(RuntimeWarning, match="runtime differs"):
        scans.validate_lbs_frozen_1d_cache_metadata(schema_v2, mol, cfg, Bond(0, 1), 0.8, 1.2, 5)

    drifted = dict(current)
    drifted["provenance"] = dict(current["provenance"])
    drifted["provenance"]["runtime"] = {"distributions": {"pyscf": "different"}}
    drifted["provenance_sha256"] = scientific_fingerprint(drifted["provenance"])
    with pytest.warns(RuntimeWarning, match="runtime differs"):
        scans.validate_lbs_frozen_1d_cache_metadata(drifted, mol, cfg, Bond(0, 1), 0.8, 1.2, 5)


def test_schema_v3_cache_requires_integral_consistent_provenance_and_backend_identity() -> None:
    mol = _water()
    cfg = _cfg()
    custom = scans.lbs_frozen_1d_cache_metadata(
        mol,
        cfg,
        Bond(0, 1),
        0.8,
        1.2,
        5,
        backend_identity="custom-qc/v1",
    )
    assert custom["identity"]["electronic_structure"]["backend"] == "custom-qc/v1"
    with pytest.raises(ValueError, match="cache_identity_sha256"):
        scans.validate_lbs_frozen_1d_cache_metadata(custom, mol, cfg, Bond(0, 1), 0.8, 1.2, 5)

    missing = dict(custom)
    missing.pop("provenance")
    with pytest.raises(ValueError, match="provenance is missing"):
        scans.validate_lbs_frozen_1d_cache_metadata(
            missing,
            mol,
            cfg,
            Bond(0, 1),
            0.8,
            1.2,
            5,
            backend_identity="custom-qc/v1",
        )

    corrupted = dict(custom)
    corrupted["provenance"] = dict(custom["provenance"])
    corrupted["provenance"]["molecule"] = dict(custom["provenance"]["molecule"])
    corrupted["provenance"]["molecule"]["label"] = "tampered"
    with pytest.raises(ValueError, match="provenance fingerprint is corrupt"):
        scans.validate_lbs_frozen_1d_cache_metadata(
            corrupted,
            mol,
            cfg,
            Bond(0, 1),
            0.8,
            1.2,
            5,
            backend_identity="custom-qc/v1",
        )


def test_2d_lbs_cache_validation_checks_metadata_and_requested_arrays() -> None:
    mol = _water()
    cfg = _cfg()
    r1 = (0.8, 1.0, 3)
    r2 = (0.9, 1.1, 3)
    R1 = np.linspace(*r1)
    R2 = np.linspace(*r2)
    meta = scans.lbs_frozen_2d_cache_metadata(mol, cfg, Bond(0, 1), Bond(0, 2), r1, r2)
    scan_meta = meta["identity"]["scan"]
    assert scan_meta["bond1_zero_based"] == [0, 1]
    assert scan_meta["bond2_zero_based"] == [0, 2]
    assert "keo" not in scan_meta
    assert "reduced_masses_amu" not in scan_meta
    assert "g12_inv_amu" not in scan_meta
    full_scan = meta["provenance"]["scan"]
    assert full_scan["keo"] == "gmatrix"
    assert full_scan["reduced_masses_amu"] == pytest.approx(
        [
            scans.local_bond_reduced_mass_amu(mol, Bond(0, 1)),
            scans.local_bond_reduced_mass_amu(mol, Bond(0, 2)),
        ]
    )
    assert full_scan["g12_inv_amu"] == pytest.approx(
        scans.bond_bond_g12_inv_amu(mol, Bond(0, 1), Bond(0, 2))
    )
    assert full_scan["gmatrix_reference_geometry"] == "identity.molecule.coordinates_A"
    arrays = {
        "R1_A": R1,
        "R2_A": R2,
        "E_Eh": np.zeros((3, 3)),
        "MU_Debye": np.zeros((3, 3, 3)),
    }

    out_R1, out_R2, E, MU = scans.validate_lbs_frozen_2d_cache(
        meta,
        arrays,
        mol,
        cfg,
        Bond(0, 1),
        Bond(0, 2),
        r1,
        r2,
    )
    np.testing.assert_allclose(out_R1, R1)
    np.testing.assert_allclose(out_R2, R2)
    assert E.shape == (3, 3)
    assert MU.shape == (3, 3, 3)

    isotope = Molecule.from_arrays(["O", "D", "D"], mol.coords)
    scans.validate_lbs_frozen_2d_cache(
        meta,
        arrays,
        isotope,
        cfg,
        Bond(0, 1),
        Bond(0, 2),
        r1,
        r2,
        keo="reduced",
    )

    bad_arrays = dict(arrays)
    bad_arrays["R2_A"] = R2 + 0.01
    with pytest.raises(ValueError, match="R2 array does not match requested grid"):
        scans.validate_lbs_frozen_2d_cache(
            meta,
            bad_arrays,
            mol,
            cfg,
            Bond(0, 1),
            Bond(0, 2),
            r1,
            r2,
        )
