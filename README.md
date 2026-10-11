# pyscf-vscf

![A coupled two-coordinate potential-energy surface](https://raw.githubusercontent.com/diogovalada/pyscf-vscf/v0.3.0/docs/assets/pyscf-vscf-banner.jpg)

`pyscf-vscf` computes anharmonic vibrational spectra variationally, on
potential-energy and dipole-moment surfaces evaluated with
[PySCF](https://pyscf.org). It scans one or two bond stretches (or a normal
coordinate), solves the vibrational Schrodinger equation on that grid, and
reports assigned transition frequencies with absolute infrared intensities.
For more coordinates it offers vibrational self-consistent field (VSCF) and
configuration-interaction (VCI) solvers on fitted n-mode surfaces.

It is meant for researchers who need anharmonic frequencies and intensities
of a few strongly anharmonic modes, such as X-H stretches, overtones, and
isotopologues, beyond what harmonic analysis or second-order perturbation
theory provide, and who want every step to be inspectable, cached, and
reproducible from Python.

## Install

```bash
pip install pyscf-vscf
```

Python 3.11 or newer. PySCF and geomeTRIC are installed with the package.
PySCF does not support native Windows, so Windows users should install under
WSL. See the [installation guide](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/installation.md) for development
installs, parallelism, and memory. Upgrading from 0.2? See the
[migration guide](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/migration-0.3.md).

## Quickstart

Optimize a geometry, then compute the O-H stretch spectrum of water from a
potential and dipole scan around the optimized bond length:

```bash
pyscf-vscf opt water.xyz --method b3lyp --basis cc-pvdz
pyscf-vscf scan1d water-opt.xyz --bond 0-1 --method b3lyp --basis cc-pvdz \
    --store water.sqlite --output oh.npz
```

```text
OUTPUT_SCAN
```

Bonds are zero-based atom indices. The points run in parallel automatically;
SCAN_TIMING. `--store` keeps every
electronic point in a SQLite file, so an interrupted or partly failed scan
resumes where it stopped. `--output` writes the finished scan, from which
spectra are recomputed without electronic-structure calculations, for any
isotopologue on the same surface (the file supplies the method):

```bash
pyscf-vscf scan1d water-opt.xyz --bond 0-1 --reuse oh.npz --states 6 --intensity both
```

Check that the grid is wide and dense enough for the states you report.
Without `--range MIN MAX NPTS`, the grid spans DEFAULT_RULE around the
bond length in the input geometry, a starting point chosen by benchmark
([scans](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/scans.md)). States that place more than `1e-5` of their
probability on the outermost grid points are marked with `*`; confirm the
reported states do not change when the range grows or the grid is refined.

Other commands: `pyscf-vscf harmonic` (harmonic frequencies), `scan2d` (two
coupled bond stretches), `scan1d --coordinate normal|normal-relaxed`, and
`inspect FILE` (describe a scan file or point store). Run
`pyscf-vscf COMMAND --help` for every option; `--json results.json` also
writes machine-readable results.

## Python

The same workflow from Python: plan the scan, run it (optionally in parallel
and resumably), and solve the vibrational problem:

```python
from pyscf_vscf import (
    ESSettings, PointStore, PySCFMeanFieldProvider, RunnerSettings,
    bond_spectrum, plan_bond_scan, read_geometry, run_scan,
)

water = read_geometry("water-opt.xyz")
provider = PySCFMeanFieldProvider(ESSettings(method="b3lyp", basis="cc-pvdz"))
plan = plan_bond_scan(water, provider, ["0-1"])          # default grid
with PointStore("water.sqlite") as store:
    scan = run_scan(plan, provider, store=store, settings=RunnerSettings(workers=4))
for record in bond_spectrum(scan, water, max_state_index=3):
    print(record.quanta, round(record.frequency_cm, 1))
```

Any PySCF method can be used through a `MethodRecipe`: a module-level
function that configures the mean field, plus a manifest that identifies it.
Recipes work in scans, n-mode plans, harmonic analysis, and optimization, and
on the command line with `--recipe MODULE:NAME`:

```python
from pyscf_vscf import MethodRecipe, PySCFMeanFieldProvider

def wb97mv(mol):
    mf = mol.RKS(xc="wb97m-v").density_fit()
    mf.grids.level = 4
    return mf                                            # configured, not run

recipe = MethodRecipe(
    recipe_id="mylab.wb97m-v", recipe_version="1", basis="def2-tzvp",
    configuration={"xc": "wb97m-v", "grids_level": 4, "density_fit": "default"},
    mean_field=wb97mv,
)
provider = PySCFMeanFieldProvider(factory=recipe)
```

VSCF and VCI on n-mode surfaces need no electronic-structure code of their
own; this model runs in a second:

```python
import numpy as np
from pyscf_vscf import NModePotential, vscf_spectrum

q1 = np.linspace(-0.35, 0.35, 31)
q2 = np.linspace(-0.30, 0.30, 29)
model = NModePotential(
    coordinates=(q1, q2),                       # angstrom
    masses_amu=(1.0, 1.5),
    one_mode_potentials_Eh=(0.08 * q1**2, 0.11 * q2**2),
    two_mode_couplings_Eh={(0, 1): 0.20 * q1[:, None] ** 2 * q2[None, :] ** 2},
)
spectrum = vscf_spectrum(model, max_quanta_per_mode=2, max_total_quanta=2)
for transition in spectrum.transitions:
    print(transition.quanta, round(transition.frequency_cm, 1))
```

`python -m pyscf_vscf.examples.water_vscf_vci` runs the whole chain for water:
optimization, harmonic modes, a 1MR/2MR n-mode surface from PySCF points,
then VSCF and VCI fundamentals with intensities. `vscf_two_mode` and
`nmode_vci` are smaller examples without electronic-structure calculations.

## What it computes

- 1D and 2D sinc-DVR spectra on frozen local-bond scans, rigid
  normal-coordinate scans, and exactly constrained relaxed normal-coordinate
  scans, with overlap-based state assignments.
- Absolute intensities for light polarized along a bond and for isotropic
  samples, in km/mol and as integrated cross sections.
- State-specific VSCF for n-mode Hamiltonians with one-mode potentials and
  two-mode couplings, with transition moments and intensities.
- Fitted 1MR/2MR and selected 3MR scalar and vector n-mode surfaces, VCI on
  VSCF modals, and a complete triatomic Jacobi `J=0` kinetic-energy operator.
- Harmonic analysis and geometry optimization through PySCF and geomeTRIC,
  with the built-in methods or any PySCF method as a recipe.
- Resumable, parallel evaluation of electronic points with a SQLite point
  store, and checksummed scan files whose identity is independent of isotope
  masses.

The package does not generate full-dimensional potential surfaces and does
not implement a general curvilinear or rovibrational kinetic-energy operator.
Reduced-dimensional results inherit the error of the omitted coordinates.

## Stability

| Area | Modules | Status |
| --- | --- | --- |
| Command line, scans, point store and runner, 1D/2D DVR spectra, method recipes | `cli`, `scans`, `runner`, `point_store`, `contracts`, `variational`, `dvr`, `backends.pyscf` | Stable; changes are announced in the changelog |
| VSCF on n-mode potentials | `vscf` | Stable |
| n-mode surfaces, triatomic `J=0`, VCI, transition moments, multilevel surfaces, correlated providers | `nmode`, `kinetic`, `vci`, `transition_moments`, `multilevel`, `backends.pyscf_correlated` | Evolving; may change between minor releases |

APIs deprecated in 0.3 (the 0.2 tuple scans, schema-3 cache writers, and the
flag-style command line) keep working with a warning and are removed in 0.4.

## Documentation

- [Scans, the point store, and parallel runs](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/scans.md)
- [Migrating from 0.2](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/migration-0.3.md)
- [Reusing points and scans](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/caching.md)
- [Intensity conventions](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/intensity-conventions.md)
- [Advanced n-mode and VCI guide](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/nmode-vci.md)
- [Correlated and finite-field providers](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/correlated-providers.md)
- [Running the validations](https://github.com/diogovalada/pyscf-vscf/blob/v0.3.0/docs/running-validations.md)
- [Changelog](https://github.com/diogovalada/pyscf-vscf/blob/main/CHANGELOG.md) and [roadmap](https://github.com/diogovalada/pyscf-vscf/blob/main/ROADMAP.md)

## Validation

The test suite combines analytic numerical checks, small PySCF calculations,
and archived H2O/HDO/D2O and NH3 molecular validations against exact DVR.
These establish solver correctness and convergence on the stated
reduced-dimensional Hamiltonians; they do not establish general spectroscopic
accuracy.

## Citing and contributing

If you use `pyscf-vscf` in research, cite it with the metadata in
[`CITATION.cff`](https://github.com/diogovalada/pyscf-vscf/blob/main/CITATION.cff)
(GitHub's "Cite this repository" button). Bug reports, questions, and
contributions are welcome; see
[CONTRIBUTING](https://github.com/diogovalada/pyscf-vscf/blob/main/CONTRIBUTING.md).

## License

The code is MIT-licensed. The archived validation data are licensed under
CC BY 4.0; see
[`DATA_LICENSE.md`](https://github.com/diogovalada/pyscf-vscf/blob/main/DATA_LICENSE.md).
