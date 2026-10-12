# Migrating from 0.2 to 0.3

!!! note
    0.4 removed the interfaces this page lists as deprecated; see
    [migrating to 0.4](migration-0.4.md).

0.3 replaces the two parallel electronic-structure code paths of 0.2 with one:
every calculation is built through a single PySCF construction seam, and scans
are plans of electronic points evaluated by a resumable runner. The 0.2
interfaces that 0.3 replaces are deprecated, not removed: they still work in
0.3, emit a `PyscfVscfDeprecationWarning` (a `DeprecationWarning`) naming
their replacement, and will be removed in 0.4.

Python 3.11 or newer is required.

## Command line

| 0.2 | 0.3 |
| --- | --- |
| `pyscf-vscf --xyz w.xyz --task harmonic` | `pyscf-vscf harmonic w.xyz` |
| `pyscf-vscf --xyz w.xyz --task opt --opt-out o.xyz` | `pyscf-vscf opt w.xyz --output o.xyz` |
| `pyscf-vscf --xyz w.xyz --task 1d --bond 0-1 --rmin A --rmax B --npts N` | `pyscf-vscf scan1d w.xyz --bond 0-1 --range A B N` |
| `--scan normal` / `--scan normal-relaxed` | `scan1d ... --coordinate normal` / `normal-relaxed` |
| `pyscf-vscf --xyz w.xyz --task 2d --bond 0-1 --bond2 0-2` | `pyscf-vscf scan2d w.xyz --bond 0-1 --bond 0-2` |
| `--dump-grid g.npz` / `--reuse-grid g.npz` | `--output s.npz` / `--reuse s.npz` (reads 0.2 caches too; the file supplies the method) |
| `--vmax N` | `--states N` (default 4) |
| `--pes-workers N` | `--workers N` |
| (none) | `--store points.sqlite` keeps every point and resumes interrupted scans |
| (none) | `--recipe MODULE:NAME` runs any PySCF method (see below) |

Flag-style 0.2 invocations still run unchanged, with the 0.2 defaults, and
print a deprecation notice. Two defaults differ in the subcommands:

- Without `--range`, the bond grid is placed around the bond's length in the
  input geometry (see [scans](scans.md)); the 0.2 default of 0.75-1.25 A was
  too narrow for O-H overtones and meaningless for longer bonds. Start from
  an optimized geometry.
- `opt` writes `<stem>-opt<suffix>` next to the input unless `--output` is
  given.

## Python

### Scans

```python
# 0.2
R, E_rel, MU_debye = grid_1d_pes_dms(molecule, settings, "0-1", 0.7, 1.5, 41)

# 0.3
from pyscf_vscf import PySCFMeanFieldProvider, plan_bond_scan, run_scan, bond_spectrum

provider = PySCFMeanFieldProvider(settings)
plan = plan_bond_scan(molecule, provider, ["0-1"], [np.linspace(0.7, 1.5, 41)])
scan = run_scan(plan, provider)          # ScanResult: absolute Eh, atomic-unit dipoles
records = bond_spectrum(scan, molecule, max_state_index=4)
R, E_rel, MU_debye = scan.as_legacy_tuple()   # the 0.2 arrays, if you need them
```

`run_scan` accepts a `PointStore` for resumable evaluation and
`RunnerSettings(workers=...)` for parallel workers. Failed points raise
`PlanIncompleteError`, whose `run` attribute lists completed, failed, and
unattempted points.

| Deprecated (removed in 0.4) | Replacement |
| --- | --- |
| `grid_1d_pes_dms`, `grid_1d_pes_dms_normal`, `grid_2d_pes_dms` | `plan_bond_scan` / `plan_normal_scan` + `run_scan` |
| `dump_lbs_frozen_1d_grid_cache`, schema-3 loaders | `pyscf_vscf.artifacts.scans.dump_scan` / `load_scan`; `load_legacy_grid` reads old caches |
| `energy_dipole_fn=` custom evaluators | an `ElectronicProvider` (see [caching](caching.md)) |
| `make_mean_field` | `converged_mean_field`, `PySCFMeanFieldProvider` |
| `run_opt` (always wrote a file) | `optimize_geometry(output_path=None)` |
| `harmonic_analysis(cfg=...)` | `harmonic_analysis(settings=...)` |
| `ElectronicResult.provider_fingerprint`, `.point_fingerprint`, `ElectronicPointRequest.fingerprint()`, `provider_fingerprint()` | the `*_scientific_*` / `causal_*` spellings |
| `NModeSurfaceModel.fingerprint()` | `artifact_integrity_fingerprint()` |
| `Bond.O`, `Bond.H` | `Bond.i`, `Bond.j` |

`grid_1d_pes_dms_normal_relaxed` (constrained relaxed scans) is not
deprecated and is not on the plan runner: relaxed scans cannot use a point
store, so an interrupted relaxed scan starts again from the beginning.

### Your own PySCF method

```python
from pyscf_vscf import MethodRecipe, PySCFMeanFieldProvider

def wb97mv(mol):                      # module-level, so worker processes can import it
    mf = mol.RKS(xc="wb97m-v").density_fit()
    mf.grids.level = 4
    mf.nlcgrids.level = 2
    return mf                         # configured, not run

recipe = MethodRecipe(
    recipe_id="mylab.wb97m-v",
    recipe_version="1",
    basis="def2-tzvp",
    configuration={"xc": "wb97m-v", "grids_level": 4, "nlcgrids_level": 2, "density_fit": "default"},
    mean_field=wb97mv,
)
provider = PySCFMeanFieldProvider(factory=recipe)
```

The recipe's manifest is its identity: list every setting that changes a
result in `configuration` and bump `recipe_version` whenever the method
changes. The package also records a snapshot of the configured mean field as
provenance and refuses builds whose snapshot changes between geometries
unless you declare those fields in `variable_snapshot_fields`. Recipes work
everywhere the built-in settings do: scans, n-mode plans, `harmonic_analysis`,
and `optimize_geometry` (which rebuilds the method at every step). Dipoles
and finite fields are computed with non-relativistic spatial-orbital
integrals, so they need a restricted or unrestricted HF or KS mean field
(with or without density fitting or solvent); generalized (GHF/GKS), Dirac,
and X2C methods are refused for them.

### Other changes

- Built-in provider fingerprints are unchanged, so stored 0.2 n-mode results
  remain valid.
- `PySCFMeanFieldProvider.evaluate` no longer changes PySCF's process-wide
  thread count; worker processes set it at start-up. Without an explicit
  `max_memory_mb`, the provider uses `RunnerSettings.max_memory_mb` in workers,
  else 4000 MB as before.
- Library code logs through the `pyscf_vscf` logger and warns with
  `RuntimeWarning` instead of printing.
- `read_xyz`, `read_midas_mmol`, and `read_geometry` are available from the
  package top level.
- VSCF transitions carry `transition_moment` when `vscf_spectrum` receives a
  `dipole=NModeDipole(...)`.
