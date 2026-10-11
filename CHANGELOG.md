# Changelog

All notable changes are recorded here. The project follows Semantic Versioning
once a public release is tagged.

## [Unreleased]

## [0.3.0] - 2026-10-11

0.3 replaces the two parallel electronic-structure paths of 0.2 with one. See
the [migration guide](docs/migration-0.3.md); everything removed in 0.4 still
works in 0.3 with a `PyscfVscfDeprecationWarning`.

### Added

- One construction seam for every PySCF calculation
  (`backends/_pyscf_factory.py`): `SettingsFactory` for the built-in method and
  `MethodRecipe` for any user-supplied PySCF method (functional, dispersion,
  solvent, ECP, ...), identified by a declared manifest. A canonical snapshot of
  the configured mean field is recorded as provenance, and undeclared
  geometry-dependent configuration is rejected in plan runs, optimizations,
  and finite-difference Hessians.
  `PySCFMeanFieldProvider(factory=...)`, `harmonic_analysis`, and
  `optimize_geometry` accept recipes.
- `contracts.py`: backend-independent point requests, results, and the provider
  protocol.
- `PointStore` (SQLite): resumable, shared storage of electronic points with
  attempt history, explicit retry of failed points, and conflict detection.
- `run_plan` / `RunnerSettings`: evaluates any plan (scans and n-mode plans)
  serially or in spawned workers, with completion-driven submission,
  provider-identity checks, and fail-fast or continue policies.
- Typed scans (`pyscf_vscf.scans`): `plan_bond_scan`, `plan_normal_scan`,
  `run_scan`, `ScanResult` (absolute energies, atomic-unit dipoles),
  `bond_spectrum`, `normal_coordinate_spectrum`, and equilibrium-relative
  default grids.
- Schema-4 scan files (`artifacts.scans.dump_scan` / `load_scan`) and
  `load_legacy_grid` for 0.1/0.2 caches.
- Subcommand command line: `pyscf-vscf harmonic|opt|scan1d|scan2d|inspect`
  with `--store`, `--output`, `--reuse`, `--recipe`, and `--workers`. `--reuse`
  takes the method from the scan file (method options are refused) and checks
  the geometry, charge, spin, bonds, and coordinate kind.
- VSCF intensities: `NModeDipole` and `vscf_spectrum(dipole=...)` attach a
  `VSCFTransitionMoment` (with state overlap and an `approximate` flag) to each
  transition.
- `optimize_geometry`, which writes a file only when asked and rebuilds the
  method at every step.
- A packaged water example from PySCF surfaces to VSCF and VCI with
  intensities (`python -m pyscf_vscf.examples.water_vscf_vci`).
- `read_geometry` and top-level exports of the readers and the new API;
  `py.typed` and strict mypy checking of the new core modules; docstrings for
  every public function and class; `docs/scans.md` and
  `docs/migration-0.3.md`.

### Changed

- Python 3.11 or newer is required; NumPy 1.23.2 or newer.
- Flag-style command lines (`--xyz ... --task ...`) run unchanged through the
  0.2 interface and print a deprecation notice.
- `PySCFMeanFieldProvider.evaluate` no longer changes PySCF's process-wide
  thread count; dipoles are computed from the density with CPU integrals (the
  same formula as PySCF's `dip_moment`). Built-in provider fingerprints are
  unchanged. Its `max_memory_mb` (and the CCSD(T) provider's) now defaults to
  the runner's per-worker budget, else 4000 MB.
- The correlated CCSD(T) provider builds its RHF reference through the seam
  (still conventional, without density fitting).
- Library diagnostics use the `pyscf_vscf` logger and `RuntimeWarning`
  instead of printing; the command lines show them with `-v`.
- Final VSCF modals follow the package's deterministic sign convention.
- Duplicated private validators are shared through one module.

### Deprecated

- `grid_1d_pes_dms`, `grid_1d_pes_dms_normal`, `grid_2d_pes_dms`, schema-3 cache
  writers and loaders, custom `energy_dipole_fn` evaluators, `make_mean_field`,
  `run_opt`, `harmonic_analysis(cfg=...)`, `ElectronicResult.provider_fingerprint`
  and `.point_fingerprint`, `ElectronicPointRequest.fingerprint()`,
  `provider_fingerprint()`, `NModeSurfaceModel.fingerprint()`, `Bond.O`,
  `Bond.H`, and the flag-style command line. All are removed in 0.4.

## [0.2.2] - 2026-10-10

### Added

- Command line: `--json` structured results, km/mol intensity columns,
  grid-boundary flags (`*`, threshold `--boundary-warn`, default `1e-5`),
  `--continue-on-error`, `--partial-out`, `--debug`, `--max-memory`, `--ri`,
  `--rmin2/--rmax2/--npts2` for 2D scans, `--intensity isotropic`, and help
  text that states every default.
- Python: `ScanFailedError`, `PartialScanGrid`, and `ScanPointFailure` report
  failed scan points with their grid index, coordinates, and traceback and
  keep every completed point; scan functions accept `stop_on_error` and
  `batch_size`; `dump_partial_scan_npz` writes a diagnostic (never reusable)
  file of completed points.
- `TransitionRecord.grid_boundary_weight`, `ground_grid_boundary_weight`, km/mol
  intensity properties, and `as_dict(extended=True)`. The default `as_dict()`
  output is unchanged.
- `settings.resolve_scf_controls`, `available_cpu_count`, `available_memory_mb`,
  and the per-process memory budget `PYSCF_VSCF_MAX_MEMORY_MB`.
- `docs/caching.md`, issue templates, Python 3.14 support, and CI jobs for the
  lowest supported dependency versions and a branch-coverage gate.

### Changed

- Scans run in parallel by default: up to one worker per available CPU,
  limited by the number of points and by available memory, each with an
  explicit PySCF memory budget; leftover CPUs become threads of the workers. `--max-parallel` now defaults to the available
  CPU count instead of 8, which also sets the thread count of harmonic
  analysis and optimization. An explicit `--pes-workers` keeps the previous
  behaviour.
- Unset SCF controls are applied explicitly as package defaults (convergence
  `1e-10` Eh, 50 cycles, DFT grid level 3), so a PySCF configuration file can
  no longer change them. New caches record the resolved values as provenance;
  cache identities and existing caches are unchanged.
- Exceptions raised while evaluating scan points now surface as
  `ScanFailedError` (a `RuntimeError`) carrying the completed points; the
  original exception type and message are kept in its failure records.
- The command line reports errors as one line (exit code 1; argument errors
  exit with 2) instead of a traceback, validates bonds, grids, normal-coordinate
  bounds, and output paths before any calculation, and writes the runtime line
  to standard error.
- `--reuse-grid` takes omitted electronic and grid options from the verified
  cache, and cache mismatches list each differing field.
- The default optimization output follows the input format (`.xyz` for
  `--xyz`). Optimization output suffixes are validated before optimizing.
- Cache options are rejected for harmonic and optimization tasks, and `--scan`
  is rejected for 2D scans, instead of being ignored.
- SciPy 1.13 or newer is required.
- `FUTURE WORK.md` is now `ROADMAP.md`; the README was rewritten.

### Fixed

- Signed transition dipoles and eigenvector phases could flip sign between
  NumPy/LAPACK builds because the sign convention picked its pivot by exact
  maximum, and symmetric problems have mirror components equal to roundoff.
  Components within a relative `1e-8` of the maximum are now treated as tied
  and resolved by a fixed order. Results without such near ties, including
  the archived validation reports, are unchanged.
- The n-mode, VCI, and transition-moment APIs failed with SciPy older than
  1.13 on the package's read-only arrays; the dependency floor is corrected.
- The analytic-Hessian fallback selected the Hartree-Fock Hessian for DFT
  objects without a `Hessian` method, and reported an `AttributeError` raised
  inside PySCF's Hessian code as a missing Hessian.
- Python error messages no longer tell API users to pass command-line flags.

### Known limitations

- The default 1D/2D bond grid (0.75-1.25 A) is too narrow for converged O-H
  states: for water at B3LYP/cc-pVDZ the fundamental is 76 cm^-1 too high.
  Such states are now flagged; choose a wider `--rmin`/`--rmax`. The defaults
  are planned to change in 0.3.

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
