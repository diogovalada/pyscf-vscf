# Future Work

## Upgrades

The minimum honest `pyscf-vscf` alpha scope is implemented: a multi-mode data
model, 1MR/2MR potentials, state-specific VSCF iteration, convergence
diagnostics, transition energies, and a packaged runnable example.

The following capabilities can wait for later releases:

- VCI on top of converged VSCF modals, with polyad and energy pruning.
- Dipole-surface n-mode expansion and VSCF/VCI transition intensities.
- Automated PySCF generation of general n-mode PES/DMS expansions.
- Curvilinear and coordinate-dependent kinetic-energy operators.
- Full-dimensional workflows and broader chemical-coordinate support.
- Exactly constrained relaxed local-bond (`lbs-relaxed`) scans.
- DIIS or quasi-Newton acceleration for difficult VSCF iterations.
- Larger-system sparse modal solvers and restart/checkpoint support.

## Validation

The package now has overlap-based state assignment, analytic model benchmarks,
an independent Einstein-A intensity identity, explicit convergence reports,
archived reproducible H2O/HDO/D2O grids, and a six-transition comparison with
independent ORCA harmonic IR intensities. It also compares state-specific VSCF
against exact 2D DVR on the same three archived molecular surfaces under a
matched separable kinetic model. The ORCA check reproduces the molecular
intensity scale and trend within 35.4% relative error. These checks establish
software and reduced-dimensional numerical correctness; they do not certify
every new electronic-structure surface or establish experimental accuracy.

Before a quantitative molecular result set is published:

- Compare assigned frequencies against independent experimental band origins.
- Add experimental absolute-intensity benchmarks where like-for-like
  vibrational or rovibrational conventions are available; the current ORCA
  comparison is an independent computational benchmark only.
- Report grid-density, coordinate-window, state-count, and assignment-weight
  error budgets for every cited transition.
- Test sensitivity to electronic-structure method, basis, DFT grid, SCF
  tolerance, density fitting, and dispersion when conclusions depend on small
  differences.
- Archive schema-v2 source grids, commands, software versions, geometries,
  hashes, and generated reports with the publication.

The expensive boundary remains PES/DMS generation, not DVR/VSCF analysis.
Typical electronic-structure costs are 31-61 points for one 1D frozen scan,
441/961/1681 points for 21/31/41-point 2D grids, and approximately
`npts * optimizer iterations` gradient calculations for constrained relaxed
scans. Cached-grid convergence analysis is normally seconds to minutes on a
laptop and requires no PySCF installation.
