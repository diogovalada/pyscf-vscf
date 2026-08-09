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
