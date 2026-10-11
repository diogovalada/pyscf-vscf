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
from pyscf_vscf import (
    ESSettings, PointStore, PySCFMeanFieldProvider, RunnerSettings,
    bond_spectrum, plan_bond_scan, read_geometry, run_scan,
)

def main():
    molecule = read_geometry("water-opt.xyz")
    provider = PySCFMeanFieldProvider(ESSettings(method="b3lyp", basis="cc-pvdz"))
    plan = plan_bond_scan(molecule, provider, ["0-1", "0-2"])      # default grids
    with PointStore("water.sqlite") as store:
        scan = run_scan(plan, provider, store=store, settings=RunnerSettings(workers=8))
    return bond_spectrum(scan, molecule, max_state_index=6)


if __name__ == "__main__":
    records = main()
```

Worker processes are started with `spawn`, which imports your script again in
each worker; code that runs a plan with `workers` must therefore sit under
`if __name__ == "__main__":` (as above) when it is in a script.

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

with `DEFAULT_BOND_GRID_POINTS` points: 0.40 A inside and 0.70 A outside the
bond length, 41 points (a spacing of 0.0275 A). The default was chosen by
benchmark (next section); it is a starting point, not a convergence
guarantee. Check convergence for the states you report:

- **Range.** Every `TransitionRecord` carries `grid_boundary_weight` and
  `ground_grid_boundary_weight`, the probability on the outermost grid
  points, and the CLI marks states above `1e-5` with `*`. A marked state
  needs a wider range. The weight also grows with the spacing for the same
  wavefunction, so it is a warning sign, not a measure of the error.
- **Density.** The boundary weight does not detect a grid that is too coarse
  for the wavefunction. Heavy-atom bonds and high overtones need more points;
  confirm that the reported states do not change when the number of points
  grows.

### How the default was chosen

The rule was selected on B3LYP/cc-pVDZ frozen-bond stretches of O-H in water
(with H2O and D2O masses on the same surface), H-Cl, C-O, and the C-C bond of
ethane, from optimized geometries. For each bond, a dense reference scan
(0.55-2.30 r_e with 176 points for the X-H bonds, 0.75-2.00 r_e with 201
points for the heavy-atom bonds; the CO reference ends at 1.82 A, before the
first SCF failure of its dense scan) gave transitions v = 1-3 converged to better than
1e-4 cm^-1 in both range and density. Candidate rules were evaluated by
interpolating the reference energies and dipoles onto each candidate grid
(checked against directly computed candidates to 0.001 cm^-1); a rule
passed if every v <= 3 transition of every bond was within 0.5 cm^-1 of its
reference.

The two kinds of bond limit the grid differently. X-H bonds are limited by
range: with only 0.30 A inside the bond length, the H-Cl fundamental and
overtones are 0.4-7 cm^-1 too high; with a range of 0.70-1.70 r_e, the O-H
v = 3 transition is 1 cm^-1 too high. C-O is limited by density: 31 points
over the default range put its v = 3 transition 2.6 cm^-1 off, with boundary
weights near 1e-10. The narrowest passing rules use 31 points (r_e - 0.40 A
to r_e + 0.60 A, largest error 0.19 cm^-1), but they sit close to failure:
at 31 or 35 points, slightly wider rules fail for C-O. The default uses 41
points (32% more than 31 in 1D, 75% more in 2D), where every nearby rule
(inner 0.35-0.45 A, outer 0.70-0.90 A) passes; its largest error over the
five systems is 0.003 cm^-1, confirmed by direct calculation on the default
grids. These results cover v <= 3 frequencies of these frozen-bond models
with this method; they do not establish intensities, 2D scans, or other
bonds, which is why the convergence checks above remain necessary.

| Bond | r_e / A | Reference v = 1, 2, 3 / cm^-1 | Largest default-grid error / cm^-1 |
| --- | --- | --- | --- |
| O-H (H2O) | 0.9687 | 3644.8, 7136.8, 10481.2 | < 0.001 |
| O-D (D2O) | 0.9687 | 2684.4, 5286.6, 7808.7 | < 0.001 |
| H-Cl | 1.2839 | 2930.1, 5738.3, 8432.3 | 0.003 |
| C-O | 1.1346 | 2176.6, 4329.1, 6457.7 | < 0.001 |
| C-C (ethane) | 1.5292 | 1430.6, 2845.5, 4244.0 | < 0.001 |

## Files

`pyscf_vscf.artifacts.scans.dump_scan` writes a schema-4 NPZ file with the
arrays, axis definitions, point keys, provider identity, sampling definition,
array checksums, and a content fingerprint; `load_scan` verifies all of them.
`load_legacy_grid` reads 0.1/0.2 grid caches (energies relative to the grid
minimum, Debye dipoles) without inventing absolute energies. `pyscf-vscf
inspect FILE` describes either kind, and point stores.
