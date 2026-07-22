# Archived monomer grids

This directory contains the exact H2O, HDO, and D2O 41x41 PES/DMS cache files
used by the original research driver, plus their generation logs. The archive
allows DVR, state-assignment, intensity-formula, and convergence work to be
repeated without 5,043 new electronic-structure points.

The files use legacy cache schema 1. Their embedded provenance omits software
versions and several settings, so the package's production cache loader rejects
them. `manifest.json` records every fact recoverable from the files and logs,
including SHA-256 hashes and the exact commands. They are accepted only by the
validation script after manifest hash verification.

The intensity columns printed in the archived logs used the superseded
frequency-independent formula. They are retained as historical output and must
not be cited. The PES/DMS arrays are reanalyzed by the current package.

Run the no-PySCF validation:

```bash
uv run python scripts/validate_archived_grids.py --output validation_data/report.json
```

The source geometries are copied under `validation_data/geometries/` and are
hash-pinned in the manifest, making the archive self-contained.

The validation report also decomposes each archived molecular 2D PES into
one-mode cuts plus an exact two-mode coupling correction, then compares
state-specific VSCF with exact 2D sinc-DVR for the two fundamentals and their
combination state. Both solvers use constant reduced masses with the kinetic
cross term disabled so the Hamiltonians match. This tests the VSCF mean-field
implementation on molecular surfaces, not the electronic-structure PES or
omitted curvilinear/kinetic coupling physics.

`orca_stretch_intensity_benchmarks.json` is a hash-pinned six-transition
summary derived from the repository's independent ORCA 6.1 reference data.
The validation script compares it with the assigned H2O/HDO/D2O stretch
fundamentals as a broad units, scale, and trend check. It is not experimental
or like-for-like rovibrational validation.

The original computed data and derived summaries in this directory are
licensed under CC BY 4.0 as described in the repository's `DATA_LICENSE.md`.
