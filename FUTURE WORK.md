# Future Work

## Upgrades

The package now includes fitted scalar/vector n-mode surfaces, reusable
coordinate maps, a complete triatomic Jacobi `J=0` kinetic operator, VCI on
converged VSCF modals, and signed vector-DMS transition moments. These advanced
interfaces remain evolving alpha APIs.

The following capabilities can wait for later releases:

- Backend-neutral VPT2/GVPT2 support, staged from a solver that consumes
  user-supplied harmonic, cubic, and semi-diagonal quartic force fields to
  optional backend adapters for automatic force-field generation. Reliable
  resonance detection, deperturbation, and polyad diagonalization require
  independent reference benchmarks before being exposed as supported features.
- Automated PySCF generation of general n-mode PES/DMS expansions.
- GPU4PySCF acceleration for electronic surface generation, after persistent
  GPU-worker execution, CPU/GPU parity, and CUDA CI can be tested on supported
  NVIDIA hardware.
- General polyatomic curvilinear, rovibrational, and coordinate-dependent
  kinetic-energy operators beyond the current triatomic `J=0` implementation.
- Full-dimensional workflows and broader chemical-coordinate support.
- Exactly constrained relaxed local-bond (`lbs-relaxed`) scans.
- DIIS or quasi-Newton acceleration for difficult VSCF iterations.
- Larger-system sparse modal solvers and restart/checkpoint support.

## Validation

The package now has overlap-based state assignment, analytic model benchmarks,
an independent Einstein-A intensity identity, explicit convergence reports,
and archived reproducible molecular grids. The water validation compares
state-specific VSCF against exact 2D DVR on three isotopologue surfaces and
checks six intensities against independent ORCA harmonic data. A separate NH3
archive exercises a non-water, three-local-mode 1MR/2MR Hamiltonian against
exact 3D DVR. These checks establish software and reduced-dimensional numerical
correctness; they do not certify every new electronic-structure surface or
establish experimental accuracy.

In the NH3 validation, the final three coordinate windows converge the
fundamental, binary-combination, and triple-combination centroid frequencies
within 15.32 cm^-1, and VSCF differs from exact 3D DVR by at most 4.43 cm^-1.
The first-overtone centroid remains window-sensitive at 67.91 cm^-1 and is not
part of the converged error budget.

Before a quantitative molecular result set is published:

- Compare assigned frequencies against independent experimental band origins.
- Add experimental absolute-intensity benchmarks where like-for-like
  vibrational or rovibrational conventions are available; the current ORCA
  comparison is an independent computational benchmark only.
- Report grid-density, coordinate-window, state-count, and assignment-weight
  error budgets for every cited transition.
- Test sensitivity to electronic-structure method, basis, DFT grid, SCF
  tolerance, density fitting, and any provider-level dispersion model when
  conclusions depend on small differences.
- Archive current-schema source grids, commands, software versions, geometries,
  hashes, and generated reports with the publication.

The expensive boundary remains PES/DMS generation, not DVR/VSCF analysis.
Typical electronic-structure costs are 31-61 points for one 1D frozen scan,
441/961/1681 points for 21/31/41-point 2D grids, and approximately
`npts * optimizer iterations` gradient calculations for constrained relaxed
scans. Cached-grid convergence analysis is normally seconds to minutes on a
laptop and requires no PySCF installation.
