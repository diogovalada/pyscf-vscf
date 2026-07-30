# Changelog

All notable changes are recorded here. The project follows Semantic Versioning
once a public release is tagged.

## [Unreleased]

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

[Unreleased]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a2...HEAD
[0.1.0a2]: https://github.com/diogovalada/pyscf-vscf/compare/v0.1.0a1...v0.1.0a2
[0.1.0a1]: https://github.com/diogovalada/pyscf-vscf/releases/tag/v0.1.0a1
