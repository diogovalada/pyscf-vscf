# Migrating to 0.4

0.4 removes the interfaces that 0.3 deprecated. Code that ran on 0.3 without
a `PyscfVscfDeprecationWarning` runs unchanged; stored point stores, scan
files, and 0.2 grid caches remain readable.

| Removed in 0.4 | Use instead |
| --- | --- |
| `pyscf-vscf --xyz ... --task ...` (flag-style command line) | the subcommands: `harmonic`, `opt`, `scan1d`, `scan2d`, `inspect` |
| `grid_1d_pes_dms`, `grid_1d_pes_dms_normal`, `grid_2d_pes_dms` | `plan_bond_scan` / `plan_normal_scan` + `run_scan` |
| `energy_dipole_fn=` custom evaluators | an `ElectronicProvider` |
| `dump_lbs_frozen_1d_grid_cache`, `load_lbs_frozen_1d_grid_cache`, `load_lbs_frozen_2d_grid_cache` | `dump_scan` / `load_scan`; `load_legacy_grid` reads 0.2 caches |
| `make_mean_field` | `converged_mean_field`, `PySCFMeanFieldProvider` |
| `run_opt` | `optimize_geometry(output_path=...)` |
| `harmonic_analysis(cfg=...)` | `harmonic_analysis(settings=...)` |
| `ElectronicResult.provider_fingerprint`, `.point_fingerprint`, `ElectronicPointRequest.fingerprint()`, `provider_fingerprint()` | `provider_scientific_fingerprint`, `point_causal_fingerprint`, `causal_fingerprint()`, `provider_scientific_fingerprint()` |
| `NModeSurfaceModel.fingerprint()` | `artifact_integrity_fingerprint()` |
| `Bond.O`, `Bond.H` | `Bond.i`, `Bond.j` |

The [0.3 migration guide](migration-0.3.md) shows each replacement with an
example.

`scripts/generate_nh3_three_mode.py` and `scripts/expand_nh3_three_mode.py`,
which generated the archived NH3 grids with the 0.2 scan API, now need
pyscf-vscf 0.3 (`uv run --no-project --with 'pyscf-vscf==0.3.1' python
scripts/generate_nh3_three_mode.py ...`). Validating the archived data does
not need them.

0.4 also adds optional GPU execution; see [GPU execution](gpu.md).
