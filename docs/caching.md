# Grid caches

PES/DMS scans are the expensive part of a calculation. Frozen local-bond 1D
and 2D grids can be written to a checksummed NPZ cache and reused for any
later spectrum, including other isotopologues on the same Born-Oppenheimer
surface.

```bash
pyscf-vscf --xyz water.xyz --task 1d --bond 0-1 --rmin 0.65 --rmax 1.6 \
    --npts 41 --method b3lyp --basis cc-pvdz --dump-grid oh.npz
pyscf-vscf --xyz water.xyz --task 1d --reuse-grid oh.npz --vmax 4
```

## What makes a cache reusable

Schema-v3 caches separate the causal identity of the electronic points from
recorded provenance:

| Part of the identity | Recorded only as provenance |
| --- | --- |
| element symbols (deuterium maps to hydrogen), Cartesian geometry, charge, spin | isotope masses and labels |
| method, basis, density fitting and auxiliary basis, SCF tolerance, SCF cycle limit, DFT grid level, backend identifier | software versions, platform, host |
| scan type, bond indices, grid values | kinetic-energy operator and reduced masses of 2D scans |

Isotope masses, labels, runtime versions, and the vibrational analysis policy
therefore do not invalidate the same surface: an `H2O` grid can be reused for
`D2O` by supplying a geometry file with `D` atoms. Array checksums are always
verified when a cache is loaded.

## Reusing a cache from the command line

With `--reuse-grid`, every electronic and grid option you omit is taken from
the cache, after its checksums and fingerprints have been verified. Options
you give explicitly must agree with the cache. A mismatch names each
differing field:

```text
pyscf-vscf: error: Grid cache mismatch for cache_identity_sha256: the cached grid was computed for a different calculation than the one requested
  electronic_structure.basis: cache='sto-3g', requested='cc-pvdz'
  scan.grid_A: cache has 15 values with shape [15] in [0.7, 1.4], requested 21 values with shape [21] in [0.7, 1.4]
```

The kinetic-energy operator (`--keo`) and isotope masses come from the
current command line and geometry file, never from the cache.

## SCF controls

Unset SCF controls resolve to the package defaults: a convergence threshold
of `1e-10` Hartree, at most 50 SCF cycles, and DFT grid level 3. Since 0.2.2
these values are applied explicitly, so a PySCF configuration file cannot
change them, and new caches record the resolved values under
`provenance.electronic_structure.resolved_scf_controls`.

Caches written by earlier versions record `None` for unset controls. Those
calculations used PySCF's configured defaults at the time, which equal the
values above unless a PySCF configuration file changed them. Such a change
cannot be detected afterwards; regenerate the cache if that is a concern.

## Failed scans and partial files

If electronic points fail, the scan stops scheduling new points (or, with
`--continue-on-error`, attempts every point) and reports each failed point by
grid index and coordinate. The completed points are written to a diagnostic
file: `--partial-out PATH`, or `<dump-grid>.partial.npz` when `--dump-grid`
was given. A partial file stores absolute energies, a completion mask, and
the failure messages. It is checksummed but deliberately is not a reusable
grid cache.

## Custom evaluators in Python

Scan functions accept any energy/dipole evaluator through `energy_dipole_fn`.
The evaluator returns Hartree energies and Cartesian Debye dipoles in the
geometry frame. When writing or loading caches from a custom evaluator, pass
a stable, non-`pyscf` `backend_identity` so its grids can never be confused
with built-in PySCF grids.
