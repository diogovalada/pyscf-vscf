# Water VSCF and VCI from PySCF

This tutorial runs the packaged example `pyscf_vscf.examples.water_vscf_vci`,
which goes from a PySCF method to VSCF and VCI fundamentals with intensities
for all three vibrations of water:

1. optimize the geometry and compute harmonic normal modes;
2. plan a 1MR/2MR n-mode grid along the three normal coordinates;
3. evaluate the electronic points (in parallel, resumable with `--store`);
4. fit potential and dipole surfaces, then solve VSCF and VCI with
   intensities.

## Run it

```bash
python -m pyscf_vscf.examples.water_vscf_vci
```

```text
Water fundamentals, hf/6-31g (217 points)
mode                    harmonic      VSCF       VCI  VSCF km/mol  VCI km/mol
bend                      1736.9    1660.4    1677.1       123.48      124.25
symmetric stretch         3988.1    3876.1    3853.3         1.83        1.37
antisymmetric stretch     4144.8    3967.2    3994.8        42.77       44.26
```

With the default HF/6-31G model and small grids it takes about 13 s on an
8-core machine. The numbers illustrate the workflow, not converged
spectroscopy: converge the grids and the electronic model before using them.

## What it does

The source is short and is meant to be copied:
[`water_vscf_vci.py`](https://github.com/diogovalada/pyscf-vscf/blob/main/src/pyscf_vscf/examples/water_vscf_vci.py).

- `optimize_geometry` and `harmonic_analysis` give the optimized geometry and
  the normal modes; modes above 100 cm^-1 become unit Cartesian
  displacement directions with effective masses.
- A `LinearDisplacementCoordinateMap` maps the three normal coordinates onto
  Cartesian geometries. Each axis spans the classical turning point of the
  fourth harmonic level plus 30%.
- `plan_nmode_points` lists the unique electronic points of all one- and
  two-mode cuts (217 here), and `run_plan` evaluates them with
  `PySCFMeanFieldProvider`, in parallel and optionally into a `PointStore`.
- `fit_nmode_surface` builds anchored cubic 1MR/2MR potential and dipole
  increments; `nmode_potential_from_surface` and `NModeDipole.from_surface`
  tabulate them on the solver grids.
- `vscf_spectrum(potential, dipole=...)` solves state-specific VSCF and
  attaches transition moments; `build_nmode_vscf_modal_basis`, `solve_vci`,
  and `vci_transition_moments` do the same with VCI on the ground-state VSCF
  modals.

## Options

| Option | Default | Meaning |
| --- | --- | --- |
| `--method`, `--basis` | `hf`, `6-31g` | the PySCF method for every point |
| `--nodes` | 9 | fit nodes per mode (odd, at least 5) |
| `--solver-points` | 21 | DVR grid points per mode for VSCF/VCI |
| `--modals` | 6 | VSCF modals per mode in the VCI basis |
| `--store FILE` | none | SQLite point store; rerunning resumes |
| `--json` | off | print JSON instead of the table |

For example, a more realistic and resumable run:

```bash
python -m pyscf_vscf.examples.water_vscf_vci --method b3lyp --basis cc-pvdz \
    --nodes 13 --solver-points 31 --modals 8 --store water-nmode.sqlite
```

The [n-mode and VCI guide](../nmode-vci.md) covers the surface workflow,
coordinate maps, and multilevel composition in more detail.
