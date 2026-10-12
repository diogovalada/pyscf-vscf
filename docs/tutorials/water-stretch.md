# Water O-H stretch spectrum

This tutorial computes the anharmonic O-H stretch fundamental and overtones of
water from a B3LYP/cc-pVDZ potential and dipole scan, then reuses the same
scan for D2O. The command line and the Python API are shown side by side. The
outputs below come from a real run.

## Geometry

Save the starting geometry as `water.xyz` (angstrom):

```text
3
water
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
```

Bonds are zero-based atom indices, so `0-1` is the first O-H bond.

## Optimize

The default bond grid is placed around the bond length in the input
geometry, so start from an optimized geometry:

```bash
pyscf-vscf opt water.xyz --method b3lyp --basis cc-pvdz
```

```text
Optimized-geometry gradient check (Eh/Bohr):
  max|g_comp|=1.039e-05  rms|g_comp|=5.609e-06
Optimized geometry written to: water-opt.xyz
```

`opt` writes `water-opt.xyz` next to the input unless `--output` is given.
Add `-v` to see geomeTRIC's progress.

## Scan and solve

```bash
pyscf-vscf scan1d water-opt.xyz --bond 0-1 --method b3lyp --basis cc-pvdz \
    --store water.sqlite --output oh.npz
```

```text
v  nu/cm^-1   mu_axis/D   I_axis/(km/mol)   int_axis/(m^2/s)   orientation: polarized along bond axis
1   3644.8     2.6930e-02     1.9877e+01     6.2173e-11
2   7136.8    -8.8835e-03     4.2353e+00     1.3247e-11
3  10481.2     2.0528e-03     3.3214e-01     1.0389e-12
4  13683.0    -4.9706e-04     2.5422e-02     7.9517e-14
```

The scan evaluates 41 points, from 0.40 A inside to 0.70 A outside the
optimized bond length (0.569-1.669 A here), in parallel worker processes;
this took about 11 s on an 8-core machine. Each row is a transition from the
ground state to `v` quanta of the stretch:

- `nu/cm^-1` is the transition wavenumber;
- `mu_axis/D` is the transition dipole along the bond axis. Its sign follows
  the package's eigenvector sign convention and has no physical meaning on
  its own;
- `I_axis/(km/mol)` and `int_axis/(m^2/s)` are integrated absorption cross
  sections for light polarized along the bond (see
  [intensity conventions](../intensity-conventions.md)).

`--store water.sqlite` keeps every electronic point: if the run is
interrupted, running the same command again resumes from the stored points.
Points that failed (for example, an SCF that did not converge) are recorded
and not retried automatically; add `--retry-failed` to try them again, and
`--continue-on-error` to evaluate the remaining points after a failure.
`--output oh.npz` keeps the finished scan.

## More states, isotropic intensities

The scan file holds the surfaces, so further spectra need no
electronic-structure calculations; the file also supplies the method:

```bash
pyscf-vscf scan1d water-opt.xyz --bond 0-1 --reuse oh.npz --states 6 --intensity both
```

```text
v  nu/cm^-1   mu_axis/D   I_axis/(km/mol)   |mu|/D   I_iso/(km/mol)   int_axis/(m^2/s)   int_iso/(m^2/s)
1   3644.8     2.6930e-02     1.9877e+01     2.7687e-02     7.0038e+00     6.2173e-11     2.1907e-11
2   7136.8    -8.8835e-03     4.2353e+00     9.2786e-03     1.5401e+00     1.3247e-11     4.8174e-12
3  10481.2     2.0528e-03     3.3214e-01     2.2216e-03     1.2967e-01     1.0389e-12     4.0559e-13
4  13683.0    -4.9706e-04     2.5422e-02     5.7111e-04     1.1187e-02     7.9517e-14     3.4991e-14
5  16746.9     1.2196e-04     1.8732e-03     1.5940e-04     1.0666e-03     5.8592e-15     3.3362e-15
6  19679.6    -2.4211e-05     8.6749e-05     4.8327e-05     1.1521e-04     2.7134e-16     3.6036e-16 *
* 1 state(s) place more than 1e-05 of their probability on the outermost grid points; widen --range and check that frequencies converge.
```

The `*` marks `v = 6`: its wavefunction reaches the edge of the grid, so it
is not converged. The `I_iso` columns are for isotropically oriented
molecules.

## D2O from the same surface

Isotopologues share the potential and dipole surfaces; only the masses
change. Replace the hydrogen labels with `D` and reuse the scan:

```bash
sed 's/^H /D /' water-opt.xyz > d2o-opt.xyz
pyscf-vscf scan1d d2o-opt.xyz --bond 0-1 --reuse oh.npz --states 4
```

```text
v  nu/cm^-1   mu_axis/D   I_axis/(km/mol)   int_axis/(m^2/s)   orientation: polarized along bond axis
1   2684.4     2.3757e-02     1.1393e+01     3.5635e-11
2   5286.6    -6.5485e-03     1.7048e+00     5.3324e-12
3   7808.7     1.2813e-03     9.6403e-02     3.0154e-13
4  10252.8    -2.6140e-04     5.2681e-03     1.6478e-14
```

`--reuse` checks that the geometry, charge, spin, bond, and coordinate kind
match the scan file; isotope labels may differ.

## Check convergence

The default grid was chosen by benchmark (see [scans](../scans.md)), but it
is a starting point. Before citing a transition, check both:

- **range**: no reported state is marked `*`; widen with `--range MIN MAX NPTS`
  and confirm the frequencies do not move;
- **density**: the `*` mark does not detect a grid that is too coarse, so
  also add points and confirm the frequencies do not move.

For example, a wider and denser grid for the same bond:

```bash
pyscf-vscf scan1d water-opt.xyz --bond 0-1 --method b3lyp --basis cc-pvdz \
    --range 0.52 1.80 61 --store water.sqlite
```

Any geometry already in the store is reused rather than recomputed.

## The same in Python

```python
from pyscf_vscf import (
    ESSettings, PointStore, PySCFMeanFieldProvider, RunnerSettings,
    bond_spectrum, plan_bond_scan, read_geometry, run_scan,
)


def main():
    water = read_geometry("water-opt.xyz")
    provider = PySCFMeanFieldProvider(ESSettings(method="b3lyp", basis="cc-pvdz"))
    plan = plan_bond_scan(water, provider, ["0-1"])          # the default grid
    with PointStore("water.sqlite") as store:
        scan = run_scan(plan, provider, store=store, settings=RunnerSettings(workers=4))
    print(f"{len(plan.requests)} points, {scan.provenance['points_reused']} reused from the store")
    for record in bond_spectrum(scan, water, max_state_index=3):
        print(record.quanta, round(record.frequency_cm, 1),
              round(record.integrated_cross_section_axis_km_per_mol, 3))


if __name__ == "__main__":    # workers are spawned processes that import this file
    main()
```

```text
41 points, 41 reused from the store
(1,) 3644.8 19.877
(2,) 7136.8 4.235
(3,) 10481.2 0.332
```

Every point was already in `water.sqlite` from the command-line run, so no
SCF was repeated. `scan` is a `ScanResult` with absolute energies (Hartree)
and dipoles (atomic units); `dump_scan` and `load_scan` in
`pyscf_vscf.artifacts.scans` write and read it.

## Local and normal modes

`pyscf-vscf harmonic water-opt.xyz --method b3lyp --basis cc-pvdz` reports
harmonic normal-mode frequencies of 1658.5 cm^-1 (the bend) and 3749.9 and
3851.0 cm^-1. The latter two are the symmetric and antisymmetric
combinations of both stretches, not the single local O-H stretch scanned
here, so they are not directly comparable with the 3644.8 cm^-1
fundamental. For coupled stretches, use
`scan2d` with both bonds, or the n-mode workflow in the
[next tutorial](water-vscf-vci.md).
