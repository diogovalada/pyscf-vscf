# Changelog

All notable changes are recorded here. The project follows Semantic Versioning
once a public release is tagged.

## [Unreleased]

### Added

- Documentation at <https://pyscf-vscf.readthedocs.io/>, with tutorials and
  an API reference.

### Changed

- GitHub release notes are taken from this changelog.
- The command-line help and the guides state that relaxed normal-coordinate
  scans cannot resume.

### Fixed

- The README's example output shows the transition-dipole signs that 0.3
  prints.

## [0.3.0] - 2026-10-11

0.3 builds every calculation through one PySCF code path and runs
fixed-geometry scans as resumable plans; results stored by 0.2 remain valid. The 0.2 interfaces it
replaces are deprecated, not removed: they still work, with a
`PyscfVscfDeprecationWarning`, and will be removed in 0.4. See the
[migration guide](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/migration-0.3.md).

### Added

- Any PySCF method through `MethodRecipe` (functional, dispersion, solvent,
  ECP, ...), in scans, n-mode plans, harmonic analysis, and optimization, and
  with `--recipe` on the command line.
- Resumable, parallel scans: `run_plan` evaluates any scan or n-mode plan in
  worker processes, and `PointStore` (`--store`) keeps every point, resumes
  interrupted runs, and retries failed points on request (`--retry-failed`).
  Relaxed normal-coordinate scans do not use the runner and cannot resume.
- Typed scans (`plan_bond_scan`, `plan_normal_scan`, `run_scan`,
  `ScanResult`) and scan files (`dump_scan`, `load_scan`; `load_legacy_grid`
  reads 0.1/0.2 caches).
- A subcommand command line: `pyscf-vscf harmonic | opt | scan1d | scan2d |
  inspect`.
- A default bond grid around the bond length (r_e - 0.40 A to r_e + 0.70 A,
  41 points), chosen by benchmark.
- VSCF transition moments and intensities (`vscf_spectrum(dipole=...)`).
- A water example from PySCF surfaces to VSCF and VCI with intensities
  (`python -m pyscf_vscf.examples.water_vscf_vci`).
- Type information (`py.typed`) and docstrings for the public API.

### Changed

- Requires Python 3.11, NumPy 1.23.2, and geomeTRIC 1.0.2 or newer.
- `--reuse` takes the method and grid from the scan file, and refuses method
  options and `--range`.
- Evaluating a point no longer changes PySCF's process-wide thread count.
  Provider memory defaults to the runner's per-worker budget, else 4000 MB.
- Library messages go to the `pyscf_vscf` logger and `RuntimeWarning`
  instead of standard output; `-v` shows them on the command line.
- Signed transition dipoles and VSCF modals follow a fixed sign convention,
  so some signs differ from 0.2. Intensities are unchanged.

### Fixed

- VCI and the exact reference solvers could drop copies of degenerate states,
  and VCI failed for potentials with a large constant offset.
- Damped VSCF (`modal_mixing < 1`) could report convergence too early.
- The command line's harmonic output omitted imaginary frequencies.
- Optimization let geomeTRIC replace and close the caller's logging handlers.
- Fitted surfaces could change if the caller later modified the input arrays.
- Multilevel composition accepted models of different molecules or charge
  states, and rejected cubic fits.
- Fractional charge or spin was truncated, and negative bond indices,
  non-finite masses, and invalid kinetic G matrices were accepted.

### Deprecated

- The 0.2 tuple scan functions (`grid_1d_pes_dms` and relatives), the
  schema-3 cache writers and loaders (`load_legacy_grid` still reads existing
  caches), `energy_dipole_fn` evaluators, `make_mean_field`, `run_opt`,
  the flag-style command line, and several older spellings. The migration
  guide lists each with its replacement.

## [0.2.2] - 2026-10-10

### Added

- Command-line options: `--json` results, km/mol intensities,
  `--intensity isotropic`, `--continue-on-error`, `--partial-out`,
  `--max-memory`, `--ri`, `--debug`, and second-bond grids for 2D scans
  (`--rmin2`, `--rmax2`, `--npts2`).
- States that reach the edge of the grid are flagged with `*` (threshold
  `--boundary-warn`), and `TransitionRecord` reports their boundary weights.
- A failed scan point raises `ScanFailedError`, which keeps every completed
  point and reports each failure's grid point and traceback.
- Python 3.14 support.

### Changed

- Scans run in parallel by default (up to one worker per CPU, limited by the
  number of points and by memory); `--max-parallel` defaults to the number of
  available CPUs.
- SCF defaults (convergence 1e-10 Eh, 50 cycles, DFT grid level 3) are set
  explicitly, so a PySCF configuration file can no longer change them. Cache
  identities are unchanged.
- Command-line errors are short messages without a traceback (`--debug`
  shows it), and inputs are checked before any calculation.
- Options that were silently ignored are now errors: cache options for
  harmonic and optimization tasks, and `--scan` values other than
  `lbs-frozen` for 2D scans.
- `--reuse-grid` takes omitted options from the cache and lists each field
  that does not match.
- The default optimization output keeps the input format.
- Requires SciPy 1.13 or newer.

### Fixed

- Signed transition dipoles could flip sign between NumPy/LAPACK builds.
- The analytic-Hessian fallback used the Hartree-Fock Hessian for some DFT
  methods.
- The n-mode, VCI, and transition-moment APIs failed with SciPy older than
  1.13.

### Known limitations

- The default bond grid (0.75-1.25 A) is too narrow for converged O-H states:
  the water fundamental is 76 cm^-1 too high at B3LYP/cc-pVDZ. Choose a wider
  `--rmin`/`--rmax`; 0.3.0 replaced the default.

## [0.2.1] - 2026-08-17

### Changed

- Updated the distribution summary to describe the package as providing
  variational anharmonic vibrational calculations with PySCF. The API,
  numerical methods, and archived scientific results are unchanged.

## [0.2.0] - 2026-08-17

### Changed

- Promoted the reviewed `0.2.0a1` code to the final `0.2` release. The
  numerical methods and archived scientific results are unchanged.

## [0.2.0a1] - 2026-08-14

### Added

- Optional raw-Mulliken and meta-Lowdin/ANO continuity descriptors for
  closed-shell PySCF mean-field points, with explicit off, best-effort, and
  strict collection modes.
- Backend-neutral electronic point identities, immutable result artifacts, and
  reusable bond, linear-displacement, and triatomic valence coordinate maps.
- Anchored fitted 1MR/2MR and selected 3MR scalar/vector n-mode surfaces with an
  explicit adapter to the released rectilinear VSCF Hamiltonian.
- A complete rotationless triatomic Jacobi `J=0` kinetic operator, VCI on
  converged VSCF modals, and signed vector-DMS transition moments.
- Direct, fingerprint-bound projection of vectorized analytic triatomic PES
  and fixed-body-frame DMS callables onto Jacobi solver grids.
- A portable analytic HOF end-to-end regression, an advanced API guide, and a
  packaged PySCF-free VCI example.

### Changed

- Advanced generated artifacts bind numerical surfaces, source lineage,
  coordinate maps, Hamiltonians, modal bases, and VCI results through separate
  fail-closed fingerprint domains.

## [0.1.0] - 2026-08-14

### Changed

- Promoted the reviewed `0.1.0a7` code to the first final `0.1` release. The
  numerical methods and archived scientific results are unchanged.

## [0.1.0a7] - 2026-08-08

### Changed

- Public scan and optimization workflows now use package-native `Molecule`,
  `Bond`, settings, and result types instead of project-driver compatibility
  objects.
- Variational DVR calculations return immutable `TransitionRecord` objects
  that always retain both polarized-axis and isotropic intensities.
- Electronic settings are separated from harmonic and execution policy.
- Grid-cache schema 3 fingerprints only inputs that can change a
  Born-Oppenheimer PES/DMS. Isotope masses, labels, runtime versions, and
  kinetic-model choices remain recorded provenance without preventing valid
  surface reuse. Schema-2 metadata is migrated during validation.
- The supported optimization path is geomeTRIC, matching the package's declared
  dependencies.

### Removed

- Water-project geometries, reference tables, orchestration scripts, and the
  monolithic research driver from the public package repository.
- Development convenience profiles, legacy global aliases, duck-typed bond
  syntax, and the duplicate `k_eigs` DVR argument.

## [0.1.0a6] - 2026-08-08

### Fixed

- The public two-bond PES/DMS builder now uses the same orientation-independent
  geometry construction as the package workflow and realizes both requested
  shared-atom bond lengths exactly.
- Relaxed normal-coordinate scans no longer discard convergence diagnostics by
  tuple unpacking or continue into spectroscopy after a failed point.
- Default molecular masses now cover the periodic table through PySCF while
  preserving exact H/D and other existing isotope overrides.
- PyPI rendering now uses a stable banner URL, and installed-example and
  validation documentation only advertise commands present in the release.
- Public normal-coordinate grid construction now normalizes its direction
  vector, so the scan coordinate consistently measures displacement in
  Angstrom.

### Removed

- The `--opt-conv` CLI option, whose two advertised profiles used identical
  thresholds.

## [0.1.0a5] - 2026-08-08

### Changed

- PySCF and geomeTRIC are now installed by default. Electronic methods use
  PySCF's native method specification instead of a separate package-level
  dispersion option.

## [0.1.0a4] - 2026-07-30

### Changed

- Enabled the first PyPI publication through GitHub Actions trusted publishing.
- Updated release and validation provenance to `0.1.0a4`; numerical methods and
  archived scientific results are unchanged from `0.1.0a3`.

## [0.1.0a3] - 2026-07-30

### Fixed

- Two-bond scans now realize both requested coordinates for any shared-atom
  bond orientation and reject dependent or invalid coordinates before running
  electronic-structure calculations.
- Unprojected harmonic analyses preserve imaginary modes as negative
  frequencies instead of silently clipping them to zero.
- Installation and KEO benchmark documentation now match the public release
  channel and current implementation.

### Changed

- Two-dimensional cache provenance now records reduced masses, the numerical
  constant G-matrix cross term, and its reference geometry.

## [0.1.0a2] - 2026-07-30

### Added

- Offset-invariant 1MR/2MR model assembly from overlapping complete pair
  surfaces, including fail-closed shared-cut consistency checks.
- A matrix-free exact n-dimensional sinc-DVR reference solver for small
  validation Hamiltonians.
- Reproducible NH3 validation drivers for a non-water, three-local-mode VSCF
  comparison against exact 3D DVR on the identical Hamiltonian.

## [0.1.0a1] - 2026-07-22

### Added

- State-specific VSCF for 1MR/2MR n-mode Hamiltonians on uniform sinc-DVR grids.
- Wavefunction-overlap assignment for coupled 2D DVR states.
- Schema-v2 scientific cache fingerprints and array checksums.
- Convergence/error-budget reporting and archived monomer validation grids.
- Independent six-transition ORCA intensity scale/trend benchmark.
- Molecular H2O/HDO/D2O VSCF comparison against exact 2D DVR on matched
  archived Hamiltonians.
- CI, release automation, citation metadata, and a packaged runnable example.

### Changed

- Absolute integrated cross sections now include transition frequency and state
  whether they are polarized or isotropically averaged.
- `normal-relaxed` now uses an exact mass-metric affine constraint and records
  residuals.
- Electronic-structure fallbacks preserve settings or fail closed; development
  fast mode is represented by ordinary fingerprinted SCF/grid settings.
- D3/D4 harmonic Hessians are explicitly classified as semi-numerical and
  require finite-difference opt-in.
- N-mode coordinates are explicitly and exclusively Angstrom, and model archive
  schema 2 records the unit.
- Public CLI inputs expose charge, spin, normal-coordinate bounds, and maximum
  SCF cycles without overloading bond-length controls.

### Removed

- Ambiguous frequency-independent `sigma_int` and arbitrary-unit dipole labels.

[Unreleased]: https://github.com/diogovalada/pyscf-vscf/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/diogovalada/pyscf-vscf/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/diogovalada/pyscf-vscf/compare/v0.2.0a1...v0.2.0
[0.2.0a1]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0...v0.2.0a1
[0.1.0]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a7...v0.1.0
[0.1.0a7]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a6...v0.1.0a7
[0.1.0a6]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a5...v0.1.0a6
[0.1.0a5]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a4...v0.1.0a5
[0.1.0a4]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/diogovalada/pyscf-vscf/releases/tag/v0.1.0a1
