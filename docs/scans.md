# Scans, the point store, and parallel runs

A scan evaluates the electronic energy and dipole at every point of a
coordinate grid. In 0.3 it has three steps:

1. **Plan** (`plan_bond_scan`, `plan_normal_scan`): build one
   `ElectronicPointRequest` per distinct geometry. Planning needs no
   electronic-structure code and is deterministic; identical geometries share a
   request.
2. **Run** (`run_plan`, or `run_scan` for plan + run + assemble): evaluate the
   requests with a provider, in the calling process or in spawned workers.
3. **Assemble** (`assemble_scan`): place the results on the grid as a
   `ScanResult` with absolute energies (Hartree) and dipoles (atomic units,
   input Cartesian frame).

```python
import numpy as np
from pyscf_vscf import (
    ESSettings, PointStore, PySCFMeanFieldProvider, RunnerSettings,
    bond_spectrum, plan_bond_scan, read_geometry, run_scan,
)

molecule = read_geometry("water-opt.xyz")
provider = PySCFMeanFieldProvider(ESSettings(method="b3lyp", basis="cc-pvdz"))
plan = plan_bond_scan(molecule, provider, ["0-1", "0-2"])      # default grids
with PointStore("water.sqlite") as store:
    scan = run_scan(plan, provider, store=store, settings=RunnerSettings(workers=8))
records = bond_spectrum(scan, molecule, max_state_index=6)
```

## The point store

`PointStore` is a SQLite file. Every completed point is stored under its
causal fingerprint, which combines the exact geometry, charge, spin,
requested properties, and the provider's scientific fingerprint. Running the
same plan again reuses stored points, so an interrupted or partly failed scan
resumes where it stopped, and different scans that share geometries share
points. Stored results are verified (fingerprints, provider, convergence,
requested properties) before reuse.

Failed attempts are kept as history. A point whose latest attempt failed is
not retried unless `RunnerSettings(retry_failed=True)` (CLI:
`--retry-failed`) is given; changing a scientific setting such as the SCF
cycle limit changes the provider fingerprint and therefore the points. A
different result stored under an existing key is recorded as a conflict and
never overwrites the original.

One process at a time should write a store, on a local file system.

## Execution

`RunnerSettings(workers=None)` evaluates in the calling process and leaves
its PySCF thread setting alone. With `workers >= 1`, points run in that many
spawned processes, each with `threads_per_worker` threads and an optional
`max_memory_mb` PySCF budget; providers are sent to the workers by pickling,
so a `MethodRecipe` must use module-level functions. By default the runner
stops scheduling after the first failure (`stop_on_error=True`); otherwise it
attempts every point. In both cases completed points are kept.

The CLI chooses workers automatically: up to one per CPU, limited by the
number of points and by memory (see [installation](installation.md)).

## Grids

Bond grids default to a range around the bond length in the input geometry,
so plan from an optimized geometry:

    r_min = r_e - DEFAULT_BOND_GRID_INNER_A,  r_max = r_e + DEFAULT_BOND_GRID_OUTER_A

with `DEFAULT_BOND_GRID_POINTS` points. The default was chosen by benchmark
(next section); it is a starting point, not a convergence guarantee. Check
convergence for the states you report: every `TransitionRecord` carries
`grid_boundary_weight` and `ground_grid_boundary_weight`, the probability on
the outermost grid points, and the CLI marks states above `1e-5` with `*`.
The boundary weight grows with the grid spacing for the same physical
wavefunction, so treat it as a warning sign and confirm by widening or
refining the grid.

### How the default was chosen

<!-- filled from the benchmark report -->

## Files

`pyscf_vscf.artifacts.scans.dump_scan` writes a schema-4 NPZ file with the
arrays, axis definitions, point keys, provider identity, sampling definition,
array checksums, and a content fingerprint; `load_scan` verifies all of them.
`load_legacy_grid` reads 0.1/0.2 grid caches (energies relative to the grid
minimum, Debye dipoles) without inventing absolute energies. `pyscf-vscf
inspect FILE` describes either kind, and point stores.
