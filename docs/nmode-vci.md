# Advanced n-mode, triatomic, and VCI APIs

The advanced APIs in `pyscf_vscf.nmode`, `pyscf_vscf.kinetic`,
`pyscf_vscf.vci`, and `pyscf_vscf.transition_moments` are evolving. Their
artifact schemas and class names may change between alpha releases. The
released 1D/2D scan and VSCF workflows do not require these APIs.

## Physical scope and units

- `LinearDisplacementCoordinateMap` uses user-defined rectilinear coordinates.
  The released separable sinc-DVR/VSCF adapter requires every coordinate and
  solver grid to be in Angstrom and every effective mass to be in atomic mass
  units.
- `TriatomicValenceCoordinateMap` uses two bond lengths in Angstrom and one
  angle in radians. `TriatomicJ0KineticOperator` implements the complete
  rotationless triatomic Jacobi operator for total angular momentum `J=0`; it
  is not a general curvilinear or rovibrational kinetic-energy operator.
- Electronic energies and fitted potentials are in Hartree. Raw electronic
  dipoles are signed three-component vectors in atomic units in the input
  Cartesian frame. Fitted surfaces and transition dipoles use the coordinate
  map's fixed molecular body frame.

## Surface workflow

Use a coordinate map and an `ElectronicProvider` to plan unique electronic
points without evaluating them:

```python
from pyscf_vscf.nmode import plan_nmode_points

plan = plan_nmode_points(
    coordinate_map,
    node_axes,
    provider,
    nuclear_charges=(1, 8, 9),
    max_rank=2,
)
```

Evaluate `plan.requests` with the same provider, then assemble and fit the
complete anchored cuts:

```python
from pyscf_vscf.nmode import assemble_nmode_samples, fit_nmode_surface

samples = assemble_nmode_samples(plan, results_by_point_fingerprint)
surface = fit_nmode_surface(samples, method="cubic")
```

`ElectronicResult` dipoles must be in the input Cartesian frame. Assembly
retains those vectors and transforms them to the coordinate map's fixed body
frame. Inclusion-exclusion produces signed 1MR/2MR and selected 3MR increments;
fits reject evaluation outside their training axes.

### Electronic continuity descriptors

The PySCF mean-field provider leaves continuity analysis off by default so an
ancillary population analysis cannot invalidate an otherwise successful SCF
point. Surface campaigns can request descriptors explicitly:

```python
from pyscf_vscf.backends.pyscf import PySCFMeanFieldProvider

provider = PySCFMeanFieldProvider(
    settings,
    continuity_diagnostics="strict",
    retain_occupied_mo_coefficients=True,
)
```

`best-effort` retains an unavailable/error marker instead of failing the
electronic point. `strict` requires the complete supported closed-shell
Mulliken and meta-Lowdin/ANO descriptor set. Canonical-orbital population
fields are Mulliken quantities and must not be treated as invariant state
labels.

Retained occupied MO coefficients are AO coefficients at one geometry. Compare
neighboring occupied spaces with the cross-geometry AO overlap and singular
values of `C_left.T @ S_left,right @ C_right`; direct coefficient, sign, or
orbital-row comparisons are not meaningful. The package deliberately supplies
descriptors rather than molecule-specific continuity thresholds.

For a complete rectilinear 1MR/2MR PES, adapt the fit to the existing VSCF
Hamiltonian:

```python
from pyscf_vscf.nmode import nmode_potential_from_surface

potential = nmode_potential_from_surface(
    surface,
    solver_grids,
    masses_amu,
)
```

The adapter rejects rank-3 energy increments because the separable
`NModePotential` contract cannot represent them. Triatomic valence surfaces
instead use `TriatomicJacobiTransform`, `TriatomicJ0KineticOperator`, and
`potential_on_jacobi_grid` from `pyscf_vscf.kinetic`.

Analytic triatomic surfaces can bypass n-mode tensors and interpolation with
`potential_on_jacobi_grid_from_callable` and
`dipole_on_jacobi_grid_from_callable`. Both callables receive one vectorized
`(..., 3)` array in the map's ordered Angstrom, Angstrom, radian convention.
The PES returns absolute Hartree energies with shape `(...)`; projection
subtracts a separate evaluation at the map reference. The DMS returns
`(..., 3)` atomic-unit vectors in the map's fixed package body frame and is
stored as the reference vector plus one full-rank increment. Callers must
supply stable, non-empty scientific source fingerprints: Python callable
identity is deliberately excluded. Each result also binds the exact map,
mass-dependent transform, solver grid, and, for the DMS, Hamiltonian.
The fixed-frame axes are the columns of `coordinate_map.frame_to_lab`: axis 0
points from the center to `outer_atom_1` in the reference geometry, axis 1 is
the in-plane orthogonal direction toward `outer_atom_2`, and axis 2 is their
right-handed cross product. No lab-frame, Eckart-frame, molecular-bisector, or
other geometry-dependent DMS rotation is performed; the callable must return
components already transformed into these fixed axes.

For the strongest DMS provenance checks, construct the Hamiltonian with
`TriatomicJ0Hamiltonian.from_projection`. A Hamiltonian built directly from an
unbound analytical potential array has no source map or transform fingerprint,
so callable DMS projection can bind the kinetic grid and Hamiltonian but cannot
verify those absent source identities.

## VCI and transitions

Build converged ground-state modals and solve a deterministic pruned VCI:

```python
from pyscf_vscf.vci import (
    VCISettings,
    build_nmode_vscf_modal_basis,
    solve_vci,
)

hamiltonian, modal_basis = build_nmode_vscf_modal_basis(potential, 6)
result = solve_vci(
    hamiltonian,
    modal_basis,
    settings=VCISettings(nstates=8, max_total_quanta=5),
)
```

Inspect `result.residual_norms_Eh`, `state_cutoff_margin_Eh`, assignments,
leading configurations, participation ratios, and near-degenerate blocks
before using the states quantitatively. Assignments touching an unresolved or
cutoff-crossing block are marked for manual review.

Project a fitted vector DMS into that exact Hamiltonian and modal basis:

```python
from pyscf_vscf.transition_moments import (
    build_vci_dipole_operator,
    nmode_dipole_on_grid,
    vci_transition_moments,
)

grid_dms = nmode_dipole_on_grid(surface, hamiltonian)
dipole_operator = build_vci_dipole_operator(grid_dms, modal_basis, result)
transitions = vci_transition_moments(dipole_operator, result)
```

Each transition retains the signed body-frame vector, its Debye conversion,
polarized component strengths, the isotropic `1/3` average, and the Einstein A
coefficient. Projection fails closed if coordinate-map, Hamiltonian, modal-basis,
or VCI identities disagree.

Surface, modal-basis, VCI, DMS-projection, and transition artifacts have typed
load/dump functions in their defining modules. Numerical surface identity,
electronic source lineage, and retained-artifact integrity are separate
fingerprint domains.

Run the packaged PySCF-free example with:

```bash
python -m pyscf_vscf.examples.nmode_vci
```
