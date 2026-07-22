from __future__ import annotations

import builtins
import importlib
import json
import sys

import numpy as np
import pytest


def test_pure_imports_do_not_request_pyscf(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pyscf" or name.startswith("pyscf."):
            raise AssertionError(f"pure import unexpectedly requested {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for module_name in [
        "pyscf_vscf",
        "pyscf_vscf.molecule",
        "pyscf_vscf.io",
        "pyscf_vscf.cache",
        "pyscf_vscf.coordinates",
    ]:
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)


def test_midas_mmol_roundtrip_preserves_deuterium_iso2(tmp_path):
    from pyscf_vscf.io import read_midas_mmol, write_midas_mmol
    from pyscf_vscf.molecule import Molecule

    mol = Molecule.from_arrays(
        ["O", "D", "H"],
        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]],
        label="HDO",
    )
    path = tmp_path / "hdo_roundtrip.mmol"

    write_midas_mmol(path, mol, title="HDO title")
    text = path.read_text()

    assert "ISO=2" in text
    assert "\nD " not in text
    parsed = read_midas_mmol(path)
    assert parsed.label == "hdo_roundtrip"
    assert parsed.symbols == ["O", "D", "H"]
    np.testing.assert_allclose(parsed.coords, mol.coords)
    assert parsed.masses[1] == pytest.approx(2.01410177812)


def test_read_midas_mmol_supports_bohr_iso_variants_and_declared_count(tmp_path):
    from pyscf_vscf.constants import ANG_TO_BOHR
    from pyscf_vscf.io import read_midas_mmol

    path = tmp_path / "bohr.mmol"
    path.write_text(
        "\n".join(
            [
                "#0 MidasMolecule",
                "#1 Xyz",
                "2 BOHR",
                "title ignored for label",
                "H  1.0  0.0  0.0  ISO=02 SUB=1",
                "O  0.0  0.0  0.0",
                "C  9.0  9.0  9.0",
                "#0 MidasMoleculeEnd",
                "",
            ]
        )
    )

    mol = read_midas_mmol(path)

    assert mol.label == "bohr"
    assert mol.symbols == ["D", "O"]
    np.testing.assert_allclose(mol.coords[0], [1.0 / ANG_TO_BOHR, 0.0, 0.0])


def test_write_xyz_uses_xyz_deuterium_symbol_and_comment(tmp_path):
    from pyscf_vscf.io import write_xyz
    from pyscf_vscf.molecule import Molecule

    mol = Molecule.from_arrays(["D", "O"], [[1.0, 2.0, 3.0], [0.0, 0.5, -1.0]])
    path = tmp_path / "geom.xyz"

    write_xyz(path, mol, comment="custom comment")

    assert path.read_text().splitlines() == [
        "2",
        "custom comment",
        "D  1.0000000000  2.0000000000  3.0000000000",
        "O  0.0000000000  0.5000000000 -1.0000000000",
    ]


def test_bond_parse_and_stretch_behavior():
    from pyscf_vscf.coordinates import Bond, parse_bond, stretch_along_bond

    assert parse_bond("O0-H1") == Bond(0, 1)
    assert parse_bond("2-3") == Bond(2, 3)
    assert parse_bond(" o4 - h5 ").O == 4
    assert parse_bond(" o4 - h5 ").H == 5

    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    stretched = stretch_along_bond(coords, parse_bond("O0-H1"), 2.5)

    np.testing.assert_allclose(stretched[0], coords[0])
    np.testing.assert_allclose(stretched[1], [2.5, 0.0, 0.0])
    np.testing.assert_allclose(stretched[2], coords[2])
    np.testing.assert_allclose(coords[1], [1.0, 0.0, 0.0])

    with pytest.raises(ValueError, match="Zero-length bond"):
        stretch_along_bond(np.zeros((2, 3)), Bond(0, 1), 1.0)
    with pytest.raises(ValueError, match="Bond specification"):
        parse_bond("O0/O1")


def test_molecule_default_masses_include_non_water_alpha_example():
    from pyscf_vscf.molecule import Molecule

    mol = Molecule.from_arrays(
        ["H", "F"],
        [[0.0, 0.0, 0.0], [0.92, 0.0, 0.0]],
        label="HF",
    )

    np.testing.assert_allclose(mol.masses, [1.00782503223, 18.99840316273])


def test_mass_overrides_support_arbitrary_elements_and_survive_displacement():
    from pyscf_vscf.molecule import Molecule
    from pyscf_vscf.workflows.scans import molecule_with_coords

    molecule = Molecule.from_arrays(
        ["Xe", "H"],
        [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]],
        masses_amu=[131.904155, 1.00782503223],
    )
    displaced = molecule_with_coords(molecule, molecule.coords + 0.1)

    np.testing.assert_allclose(molecule.masses, [131.904155, 1.00782503223])
    np.testing.assert_allclose(displaced.masses, molecule.masses)


def test_npz_cache_roundtrip_preserves_metadata_and_arrays(tmp_path):
    from pyscf_vscf.cache import assert_meta_close, assert_meta_equal, dump_grid_npz, load_grid_npz

    path = tmp_path / "nested" / "grid.npz"
    meta = {
        "kind": "1d",
        "symbols": ["O", "D", "H"],
        "npts": np.int64(3),
        "axis_vec_A": np.array([1.0, 0.0, 0.0]),
        "rmin": np.float64(0.75),
    }
    arrays = {
        "R_A": np.array([0.75, 1.0, 1.25]),
        "E_Eh": np.array([0.2, 0.0, 0.3]),
    }

    dump_grid_npz(path, meta=meta, arrays=arrays)
    loaded_meta, loaded_arrays = load_grid_npz(path)

    assert loaded_meta["axis_vec_A"] == [1.0, 0.0, 0.0]
    assert loaded_meta["kind"] == "1d"
    assert loaded_meta["npts"] == 3
    assert loaded_meta["rmin"] == 0.75
    assert loaded_meta["symbols"] == ["O", "D", "H"]
    assert set(loaded_meta["array_sha256"]) == {"R_A", "E_Eh"}
    assert set(loaded_arrays) == {"R_A", "E_Eh"}
    np.testing.assert_allclose(loaded_arrays["R_A"], arrays["R_A"])
    np.testing.assert_allclose(loaded_arrays["E_Eh"], arrays["E_Eh"])
    assert_meta_equal("kind", loaded_meta["kind"], "1d")
    assert_meta_close("rmin", loaded_meta["rmin"], 0.75)
    with pytest.raises(ValueError, match="Grid cache mismatch for kind"):
        assert_meta_equal("kind", loaded_meta["kind"], "2d")
    with pytest.raises(ValueError, match="Grid cache mismatch for rmin"):
        assert_meta_close("rmin", loaded_meta["rmin"], 0.76)

    corrupted = dict(loaded_arrays)
    corrupted["E_Eh"] = corrupted["E_Eh"].copy()
    corrupted["E_Eh"][0] += 1e-6
    np.savez_compressed(
        path,
        meta_json=np.array(json.dumps(loaded_meta, sort_keys=True)),
        **corrupted,
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_grid_npz(path)


def test_schema_v2_cache_requires_array_checksum_manifest(tmp_path):
    from pyscf_vscf.cache import load_grid_npz

    path = tmp_path / "unchecked-v2.npz"
    np.savez_compressed(
        path,
        meta_json=np.array(json.dumps({"grid_cache_version": 2})),
        E_Eh=np.array([0.0, 0.1]),
    )

    with pytest.raises(ValueError, match="missing its array checksum manifest"):
        load_grid_npz(path)
