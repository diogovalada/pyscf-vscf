# Reusing electronic points and scans

Electronic points are the expensive part of a calculation. 0.3 keeps them in
two complementary ways:

- a **point store** (`--store points.sqlite`, `PointStore`) keeps every
  evaluated point, resumes interrupted scans, and shares points between
  scans; see [scans](scans.md);
- a **scan file** (`--output scan.npz`, `dump_scan`) keeps one finished scan,
  from which spectra for any isotopologue on the same surface can be
  recomputed without electronic-structure calculations.

```bash
pyscf-vscf scan1d water-opt.xyz --bond 0-1 --method b3lyp --basis cc-pvdz \
    --store water.sqlite --output oh.npz
pyscf-vscf scan1d d2o-opt.xyz --bond 0-1 --reuse oh.npz --states 6
```

A point's identity combines its exact geometry, charge, spin, requested
properties, and the provider's scientific fingerprint (every setting that
can change the result). Scan files additionally record the axes and the
sampling definition, and are verified by array checksums and a content
fingerprint when loaded.

The rest of this page describes the schema-3 grid caches written by 0.1 to
0.3 (`--dump-grid`). 0.4 no longer writes them but still reads them
(`--reuse`, `load_legacy_grid`).

## What makes a 0.2 cache reusable

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

## Reusing a 0.2 cache

`--reuse` reads a 0.2 cache like a scan file, after verifying its checksums
and fingerprints. The method and grid come from the cache, so method options
and `--range` are refused; the geometry, charge, spin, and bonds must match
the cache:

```bash
pyscf-vscf scan1d water-opt.xyz --bond 0-1 --reuse old-grid.npz --states 4
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

## Failed scans

Points that fail are recorded in the point store (`--store`) with their
error, and the run reports each one. A later run resumes from the stored
points; `--retry-failed` attempts the failed points again, and
`--continue-on-error` evaluates the remaining points after a failure. See
[scans](scans.md).
