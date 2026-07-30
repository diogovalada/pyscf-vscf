# Running validations

The package has three validation tiers. The default tier is deliberately light:
it does not require PySCF, does not launch electronic-structure scans, and does
not require external quantum-chemistry programs.

## Tier 1: pure package checks

Run these first. They exercise the DVR solver, variational spectrum assembly,
analytic Morse and harmonic-oscillator convergence, frozen PES/DMS regression
arrays, the Einstein-A intensity identity, overlap-based state assignment,
VSCF convergence, and shipped MARVEL/IUPAC reference-data schema.

```bash
uv run --with pytest pytest tests/test_dvr.py tests/test_variational.py tests/test_validation_regressions.py -q
```

Full default suite:

```bash
uv run --with pytest pytest -q
```

If PySCF is not installed, PySCF-marked tests are skipped.

## Tier 2: optional PySCF smoke checks

These checks require only the package's optional PySCF backend dependencies.
They use tiny HF/STO-3G systems and are intended to catch backend API, shape,
isotope-mass, gradient, and dipole regressions. They are not heavy production
simulations.

```bash
uv run --extra pyscf --with pytest pytest -m pyscf -q
```

## Tier 3: research validation scripts

These scripts compare against cached or curated reference data and can launch
new PySCF calculations. They do not require ORCA, CFOUR, MidasCpp, or another
external quantum-chemistry program unless you are regenerating the reference
data yourself.

Short validation script:

```bash
uv run --extra validation python scripts/run_logged.py --tag validations -- python scripts/run_validations.py
```

Longer opt-in validations:

```bash
VSCF_LONG_VALIDATIONS=1 uv run --extra validation python scripts/run_logged.py --tag long_validations -- python scripts/run_validations.py
VSCF_DIMER_VALIDATIONS=1 uv run --extra validation python scripts/run_logged.py --tag dimer_validations -- python scripts/run_validations.py
VSCF_MARVEL_VALIDATIONS=1 uv run --extra validation python scripts/run_logged.py --tag marvel_validations -- python scripts/run_validations.py
```

Use these when you are validating research conclusions, not for every package
edit.

The checked-in H2O/HDO/D2O grids support an intermediate validation that needs
neither PySCF nor an external program:

```bash
uv run python scripts/validate_archived_grids.py --output validation_data/report.json
```

It verifies file hashes, computes corrected intensities, assigns states by
phase-canonical wavefunction overlap, and reports frequency/intensity spreads
for 41x41, nested 21x21, and narrowed-window variants. It also checks the two
assigned stretch fundamentals for each isotopologue against independent ORCA
harmonic IR intensities. All six are within the stated 40% model-comparison
tolerance; the largest relative intensity deviation is 35.4%.

The same report compares state-specific VSCF with exact 2D DVR on each archived
molecular PES using a matched separable kinetic model. The regression criterion
is a maximum 25 cm^-1 discrepancy for the two fundamentals and their
combination state; the observed maximum is below 20 cm^-1. For grid/window
convergence, both stretch fundamentals must be present under the same
phase-canonical assignment signature in all variants, with at most 20 cm^-1
frequency spread, 1% relative intensity spread, and at least 0.99 dominant
manifold weight. Unmatched higher states are reported but are not accepted as
converged or suitable for citation.

The ORCA comparison is an independent computational scale and trend check, not
an experimental or like-for-like rovibrational benchmark. The archived grids
use legacy schema 1 and are isolated from production cache reuse; their known
provenance gaps are explicit in `validation_data/manifest.json`.

### Non-water three-mode validation

The checked-in NH3 archive tests the state-specific VSCF solver on a
three-local-mode molecular Hamiltonian with all three pair corrections. Exact
3D DVR and VSCF use the identical potential expansion, coordinate grids, and
separable local reduced masses:

```bash
uv run python scripts/validate_nh3_three_mode.py
```

The report preserves failed coarse and narrow-window trials, verifies an
independently repeated central 25x25 electronic grid, and defines the accepted
window sequence from 37, 43, and 49 points. The fundamental,
binary-combination, and triple-combination manifolds pass the 25 cm^-1
centroid-spread criterion. The first overtone does not and cannot be cited as
converged from this archive. Reanalysis takes about 17 minutes on the recorded
Ryzen 7 6800HS and requires neither PySCF nor an external program.

Regeneration is opt-in. The exact commands and source-cache chain are stored
in `validation_data/nh3_three_mode/manifest.json`. The complete archive
represents 11,961 independently evaluated electronic points and about 2.9
hours of measured surface-generation time with eight one-thread workers on
that CPU.

## Heavy or external work

Installing and running external quantum-chemistry programs is only needed if
you want to regenerate external benchmark data. Consuming the checked-in
reference summaries does not require those programs.

Examples of work that should remain opt-in:

- Dense PySCF PES/DMS scans for new molecules or dimers.
- Large convergence matrices over `npts`, DVR windows, `vmax`, and scan modes.
- Regenerating ORCA/GVPT2 or other external reference datasets. The checked-in
  six-transition summary can be consumed without ORCA.

Example convergence matrix command, edited for the specific case under study:

```bash
uv run --extra validation python scripts/run_logged.py --tag h2o_2d_npts31_vmax12 -- python scripts/compare_orca_gvpt2_2d_stretches.py --species H2O --npts 31 --vmax 12 --max-parallel 8 --pes-workers 8
```
