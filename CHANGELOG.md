# Changelog

All notable changes are recorded here. The project follows Semantic Versioning
once a public release is tagged.

## [Unreleased]

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

[Unreleased]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a7...HEAD
[0.1.0a7]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a6...v0.1.0a7
[0.1.0a6]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a5...v0.1.0a6
[0.1.0a5]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a4...v0.1.0a5
[0.1.0a4]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a3...v0.1.0a4
[0.1.0a3]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a2...v0.1.0a3
[0.1.0a2]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/diogovalada/pyscf-vscf/releases/tag/v0.1.0a1
