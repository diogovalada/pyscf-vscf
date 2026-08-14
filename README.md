# pyscf-vscf

![A coupled two-coordinate potential-energy surface](https://raw.githubusercontent.com/diogovalada/pyscf-vscf/main/docs/assets/pyscf-vscf-banner.png)

`pyscf-vscf` provides reduced-dimensional vibrational calculations with a
PySCF electronic-structure backend. It includes:

- state-specific VSCF for n-mode Hamiltonians with one-mode potentials and
  explicit two-mode coupling corrections;
- 1D and 2D sinc-DVR solvers;
- frozen local-bond, normal-coordinate, and constrained relaxed scans;
- harmonic analysis and geometry optimization;
- assigned transition frequencies and axis-projected and isotropic
  intensities; and
- provenance-checked PES/DMS caches.

The package does not implement VCI, a full-dimensional PES generator, or
general curvilinear kinetic-energy operators.

## Install

```bash
pip install pyscf-vscf
```

PySCF and geomeTRIC are installed with the package. PySCF does not support
native Windows, so Windows users should install it under WSL.

For development:

```bash
git clone https://github.com/diogovalada/pyscf-vscf.git
cd pyscf-vscf
uv sync --extra dev
```

## VSCF Example

Run the PySCF-free example included in the wheel:

```bash
python -m pyscf_vscf.examples.vscf_two_mode
```

The corresponding API is:

```python
from pyscf_vscf import NModePotential, VSCFSettings, vscf_spectrum
```

`NModePotential` represents

```text
H = sum_i [T_i + V_i(q_i)] + sum_{i<j} V_ij(q_i, q_j)
```

Each `V_ij` is a coupling correction, not a complete pair potential.
Coordinates are uniform grids in Angstrom.

## PySCF Workflows

The command-line interface accepts XYZ and Midas MMOL geometries:

```bash
pyscf-vscf --xyz molecule.xyz --task harmonic
pyscf-vscf --xyz molecule.xyz --task 1d --bond 0-1 --npts 41
pyscf-vscf --help
```

The built-in backend supports Hartree-Fock (`--method hf`) and PySCF DFT
exchange-correlation specifications. Scan functions also accept an injected
energy/dipole evaluator through the Python API; there is not yet a packaged
adapter for another electronic-structure program. Custom evaluators return
Hartree energies and Cartesian Debye dipoles in the geometry frame. Use a
stable custom `backend_identity` when writing and loading their caches.

Charge and spin are not reliably encoded in XYZ or MMOL files. Pass
`--charge` and `--spin` for ions and open-shell systems.

## Results And Caches

`variational_1d` and `variational_2d` return immutable `TransitionRecord`
objects. Every record contains both the polarized-axis and isotropically
averaged intensity conventions. Use `record.as_dict()` for JSON output.

Schema-v3 grid caches separate the causal cache identity from recorded
provenance. Electronic method, basis, geometry, charge, spin, and scan grid
control reuse; isotope masses, labels, runtime versions, and vibrational
analysis policy do not invalidate the same Born-Oppenheimer surface. Array
checksums are always verified.

See the [intensity conventions](https://github.com/diogovalada/pyscf-vscf/blob/v0.1.0/docs/intensity-conventions.md),
[installation guide](https://github.com/diogovalada/pyscf-vscf/blob/v0.1.0/docs/installation.md), and
[validation guide](https://github.com/diogovalada/pyscf-vscf/blob/v0.1.0/docs/running-validations.md)
for details.

## Validation

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pytest -q
uv run python scripts/validate_archived_grids.py --output /tmp/water-report.json
uv run python scripts/validate_nh3_three_mode.py --output /tmp/nh3-report.json
```

The test suite includes analytic numerical checks, tiny PySCF smoke tests, and
archived H2O/HDO/D2O and NH3 molecular validations. The archives test solver
correctness and convergence on stated reduced-dimensional Hamiltonians; they
do not establish general spectroscopic accuracy.

## License

MIT
