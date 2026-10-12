# Your own PySCF method

The built-in options (`--method`, `--basis`, `--ri`, ...) cover HF and
standard DFT. Anything else PySCF can do (another functional, dispersion,
solvent, ECPs, finer grids) runs through a `MethodRecipe`: a function that
configures a mean field, plus a manifest that identifies it.

## Write a recipe

Save this as `mymethods.py`:

```python
from pyscf_vscf import MethodRecipe


def pbe0(mol):
    """PBE0 with density fitting on a finer DFT grid (configured, not run)."""
    mf = mol.RKS(xc="pbe0").density_fit()
    mf.grids.level = 4
    mf.conv_tol = 1e-10
    return mf


recipe = MethodRecipe(
    recipe_id="tutorial.pbe0",
    recipe_version="1",
    basis="def2-svp",
    configuration={"xc": "pbe0", "density_fit": "default", "grids_level": 4, "conv_tol": 1e-10},
    mean_field=pbe0,
)
```

The function receives a built PySCF `Mole` and returns a mean field that has
**not** been run; the package converges it. It must be defined at module
level, because worker processes import it.

## Use it on the command line

`--recipe MODULE:NAME` replaces the method options. The module must be
importable, for example from the current directory with `PYTHONPATH=.`:

```bash
PYTHONPATH=. pyscf-vscf opt water.xyz --recipe mymethods:recipe --output water-pbe0.xyz
PYTHONPATH=. pyscf-vscf scan1d water-pbe0.xyz --bond 0-1 --recipe mymethods:recipe --states 3
```

```text
v  nu/cm^-1   mu_axis/D   I_axis/(km/mol)   int_axis/(m^2/s)   orientation: polarized along bond axis
1   3761.8     4.2431e-02     5.0930e+01     1.5930e-10
2   7371.0    -9.8376e-03     5.3643e+00     1.6779e-11
3  10832.3     1.9680e-03     3.1548e-01     9.8679e-13
```

`harmonic`, `opt`, `scan1d`, and `scan2d` all accept `--recipe`;
`optimize_geometry` rebuilds the method at every step.

## Use it from Python

```python
from pyscf_vscf import PySCFMeanFieldProvider

from mymethods import recipe

provider = PySCFMeanFieldProvider(factory=recipe)
```

The provider works anywhere the built-in settings do: `plan_bond_scan`,
`plan_nmode_points`, `run_scan`, `run_plan`, `harmonic_analysis(molecule,
recipe)`, and `optimize_geometry(molecule, recipe)`.

## Identity: what the manifest is for

Stored points and scan files are keyed by the method's identity. For a
recipe, that identity is the manifest: `recipe_id`, `recipe_version`,
`basis`, `ecp`, and `configuration`. The package cannot read your function's
code, so:

- list every setting that changes a result in `configuration`;
- bump `recipe_version` whenever the function changes, so old points are not
  reused for a different method.

As a safety net, the package records a snapshot of the configured mean field
(functional, grids, tolerances, density fitting, solvent, ...) with every
result, and refuses to mix points whose snapshots differ within one scan,
optimization, finite-difference Hessian, or relaxed scan. If a setting
legitimately changes with geometry, declare it in
`MethodRecipe(variable_snapshot_fields=(...))`.

## Limits

- Dipoles and finite electric fields are computed with non-relativistic
  spatial-orbital integrals. They need a restricted or unrestricted HF or KS
  mean field (with or without density fitting or solvent); generalized
  (GHF/GKS), Dirac, and X2C methods are refused for them.
- GPU execution (GPU4PySCF) is not supported yet; the roadmap describes the
  plan.
