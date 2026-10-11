from __future__ import annotations

import numpy as np
import pytest

from pyscf_vscf.backends.pyscf import MethodRecipe, PySCFMeanFieldProvider
from pyscf_vscf.contracts import provider_scientific_fingerprint
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.runner import run_plan
from pyscf_vscf.scans import plan_bond_scan
from pyscf_vscf.settings import ESSettings

pytestmark = pytest.mark.pyscf


def _h2() -> Molecule:
    return Molecule.from_arrays(["H", "H"], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]], label="h2")


def lda_grid3(mol):
    mf = mol.RKS(xc="lda,vwn")
    mf.grids.level = 3
    return mf


def lda_grid_by_length(mol):
    mf = mol.RKS(xc="lda,vwn")
    length = float(np.linalg.norm(mol.atom_coords()[1] - mol.atom_coords()[0]))
    mf.grids.level = 4 if length > 1.5 else 3
    return mf


def _recipe(function, **kwargs) -> MethodRecipe:
    return MethodRecipe(
        recipe_id="test.lda",
        recipe_version="1",
        basis="sto-3g",
        configuration={"xc": "lda,vwn", "grids": "level 3 (or 4 beyond 1.5 bohr)"},
        mean_field=function,
        **kwargs,
    )


def test_recipe_identity_is_the_manifest_and_differs_from_built_in_settings() -> None:
    pytest.importorskip("pyscf")
    recipe_provider = PySCFMeanFieldProvider(factory=_recipe(lda_grid3))
    built_in = PySCFMeanFieldProvider(
        ESSettings(method="lda,vwn", basis="sto-3g", use_density_fit=False)
    )
    payload = recipe_provider.scientific_settings_payload()

    assert payload["provider"] == "mean-field-recipe"
    assert payload["method"]["recipe_id"] == "test.lda"
    assert provider_scientific_fingerprint(recipe_provider) != provider_scientific_fingerprint(
        built_in
    )
    bumped = PySCFMeanFieldProvider(
        factory=MethodRecipe(
            recipe_id="test.lda",
            recipe_version="2",
            basis="sto-3g",
            configuration={"xc": "lda,vwn"},
            mean_field=lda_grid3,
        )
    )
    assert provider_scientific_fingerprint(bumped) != provider_scientific_fingerprint(
        recipe_provider
    )


def test_recipe_energy_matches_equivalent_built_in_settings() -> None:
    pytest.importorskip("pyscf")
    recipe_provider = PySCFMeanFieldProvider(factory=_recipe(lda_grid3))
    built_in = PySCFMeanFieldProvider(
        ESSettings(method="lda,vwn", basis="sto-3g", use_density_fit=False, scf_conv_tol=1e-9)
    )
    plan_a = plan_bond_scan(_h2(), recipe_provider, ["0-1"], [[0.7, 0.74, 0.8]])
    plan_b = plan_bond_scan(_h2(), built_in, ["0-1"], [[0.7, 0.74, 0.8]])

    a = run_plan(plan_a, recipe_provider).results
    b = run_plan(plan_b, built_in).results
    energies_a = sorted(result.total_energy_Eh for result in a.values())
    energies_b = sorted(result.total_energy_Eh for result in b.values())
    np.testing.assert_allclose(energies_a, energies_b, atol=1e-8)
    snapshot = next(iter(a.values())).provenance["configuration_snapshot"]
    assert snapshot["grids_level"] == 3
    assert snapshot["xc"] == "lda,vwn"


def test_undeclared_geometry_dependent_configuration_is_rejected() -> None:
    pytest.importorskip("pyscf")
    provider = PySCFMeanFieldProvider(factory=_recipe(lda_grid_by_length))
    plan = plan_bond_scan(_h2(), provider, ["0-1"], [[0.6, 0.74, 1.0]])

    run = run_plan(plan, provider, settings=None)

    assert not run.complete
    failure = next(iter(run.failures.values()))
    assert "grids_level" in failure.message
    assert "variable_snapshot_fields" in failure.message

    declared = PySCFMeanFieldProvider(
        factory=_recipe(lda_grid_by_length, variable_snapshot_fields=("grids_level",))
    )
    plan = plan_bond_scan(_h2(), declared, ["0-1"], [[0.6, 0.74, 1.0]])
    assert run_plan(plan, declared).complete


def test_recipe_factories_must_not_run_the_scf() -> None:
    pytest.importorskip("pyscf")
    from pyscf_vscf.backends._pyscf_factory import ElectronicSystem, build_mean_field

    def converged(mol):
        mf = mol.RHF()
        mf.kernel()
        return mf

    recipe = MethodRecipe(
        recipe_id="bad",
        recipe_version="1",
        basis="sto-3g",
        configuration={"x": 1},
        mean_field=converged,
    )
    system = ElectronicSystem(nuclear_charges=(1, 1), coordinates_A=_h2().coords)
    with pytest.raises(ValueError, match="must not run the SCF"):
        build_mean_field(system, recipe)


def test_optimization_and_finite_difference_hessians_reject_configuration_drift() -> None:
    pytest.importorskip("pyscf")
    from pyscf import gto

    from pyscf_vscf.backends.pyscf import converged_mean_field
    from pyscf_vscf.workflows import harmonic, optimization

    recipe = _recipe(lda_grid_by_length)
    stretched = gto.M(atom="H 0 0 0; H 0 0 1.0", basis="sto-3g", unit="Angstrom", verbose=0)
    scanner = optimization._factory_gradient_scanner(_h2(), recipe)
    with pytest.raises(ValueError, match="grids_level"):
        scanner(stretched)

    reference = converged_mean_field(_h2(), recipe)
    displaced = reference.build.molecule.atom_coords().reshape(-1).copy()
    displaced[5] += 0.3
    gradient = harmonic._guarded_gradient(recipe, reference)
    with pytest.raises(ValueError, match="grids_level"):
        gradient(_h2(), recipe, displaced)

    declared = _recipe(lda_grid_by_length, variable_snapshot_fields=("grids_level",))
    tolerant = harmonic._guarded_gradient(declared, converged_mean_field(_h2(), declared))
    assert np.all(np.isfinite(tolerant(_h2(), declared, displaced)))


def test_runner_memory_budget_reaches_the_calculation(monkeypatch) -> None:
    from pyscf_vscf.settings import MAX_MEMORY_ENV_VAR

    provider = PySCFMeanFieldProvider(ESSettings(method="hf", basis="sto-3g"))
    monkeypatch.delenv(MAX_MEMORY_ENV_VAR, raising=False)
    assert provider.memory_budget_mb() == 4000
    monkeypatch.setenv(MAX_MEMORY_ENV_VAR, "128")
    assert provider.memory_budget_mb() == 128
    assert provider.execution_provenance()["max_memory_mb"] == 128
    explicit = PySCFMeanFieldProvider(ESSettings(method="hf", basis="sto-3g"), max_memory_mb=512)
    assert explicit.memory_budget_mb() == 512
