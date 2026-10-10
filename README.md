# pyscf-vscf

![A coupled two-coordinate potential-energy surface](https://raw.githubusercontent.com/diogovalada/pyscf-vscf/v0.2.2/docs/assets/pyscf-vscf-banner.jpg)

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

PySCF and geomeTRIC are installed with the package. PySCF does not support
native Windows, so Windows users should install under WSL. See the
[installation guide](https://github.com/diogovalada/pyscf-vscf/blob/v0.2.2/docs/installation.md)
for development installs and thread settings.

## Quickstart

Optimize a geometry, then compute the O-H stretch spectrum of water from a
41-point potential and dipole scan:

```bash
pyscf-vscf --xyz water.xyz --task opt --method b3lyp --basis cc-pvdz
pyscf-vscf --xyz water.pyscf_opt.xyz --task 1d --bond 0-1 \
    --rmin 0.65 --rmax 1.6 --npts 41 --vmax 3 \
    --method b3lyp --basis cc-pvdz --dump-grid oh.npz
```

```text
v  nu/cm^-1   mu_axis/D   I_axis/(km/mol)   int_axis/(m^2/s)   orientation: polarized along bond axis
1   3644.8     2.6930e-02     1.9877e+01     6.2173e-11
2   7136.8     8.8836e-03     4.2354e+00     1.3248e-11
3  10481.3     2.0529e-03     3.3216e-01     1.0390e-12
```

Bonds are zero-based atom indices. The scan runs in parallel automatically;
this one takes about 15 s on an 8-core laptop. The grid is cached in
`oh.npz`, so further spectra need no electronic-structure calculations; the
cache supplies the method, basis, bond, and grid:

```bash
pyscf-vscf --xyz water.pyscf_opt.xyz --task 1d --reuse-grid oh.npz --intensity both
```

Check that the grid is wide enough. States that place more than `1e-5` of
their probability on the outermost grid points are marked with `*`; their
frequencies are not converged. On the default 0.75-1.25 A range the O-H
fundamental above is 76 cm^-1 too high, so always choose `--rmin`/`--rmax`
for your bond and confirm that the reported states do not change when the
range grows.

Run `pyscf-vscf --help` for every option. `--json results.json` also writes
machine-readable results.

## Python

The same building blocks are available from Python. This VSCF example needs
no electronic-structure calculation:

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

`python -m pyscf_vscf.examples.vscf_two_mode` and
`python -m pyscf_vscf.examples.nmode_vci` run packaged examples. Scans from
Python accept your own energy/dipole evaluator, so other electronic-structure
programs can supply the surfaces.

## What it computes

- 1D and 2D sinc-DVR spectra on frozen local-bond scans, rigid
  normal-coordinate scans, and exactly constrained relaxed normal-coordinate
  scans, with overlap-based state assignments.
- Absolute intensities for light polarized along a bond and for isotropic
  samples, in km/mol and as integrated cross sections.
- State-specific VSCF for n-mode Hamiltonians with one-mode potentials and
  two-mode couplings.
- Fitted 1MR/2MR and selected 3MR scalar and vector n-mode surfaces, VCI on
  VSCF modals, and a complete triatomic Jacobi `J=0` kinetic-energy operator.
- Harmonic analysis and geometry optimization through PySCF and geomeTRIC.
- Checksummed PES/DMS caches whose identity is independent of isotope masses.

The package does not generate full-dimensional potential surfaces and does
not implement a general curvilinear or rovibrational kinetic-energy operator.
Reduced-dimensional results inherit the error of the omitted coordinates.

## Stability

| Area | Modules | Status |
| --- | --- | --- |
| Command line, scans, 1D/2D DVR spectra, caches | `cli`, `workflows`, `variational`, `dvr`, `cache` | Stable; changes are announced in the changelog |
| VSCF on n-mode potentials | `vscf` | Stable |
| n-mode surfaces, triatomic `J=0`, VCI, transition moments, multilevel surfaces, correlated providers | `nmode`, `kinetic`, `vci`, `transition_moments`, `multilevel`, `backends.pyscf_correlated` | Evolving; may change between minor releases |

## Documentation

- [Grid caches and reuse](https://github.com/diogovalada/pyscf-vscf/blob/v0.2.2/docs/caching.md)
- [Intensity conventions](https://github.com/diogovalada/pyscf-vscf/blob/v0.2.2/docs/intensity-conventions.md)
- [Advanced n-mode and VCI guide](https://github.com/diogovalada/pyscf-vscf/blob/v0.2.2/docs/nmode-vci.md)
- [Correlated and finite-field providers](https://github.com/diogovalada/pyscf-vscf/blob/v0.2.2/docs/correlated-providers.md)
- [Running the validations](https://github.com/diogovalada/pyscf-vscf/blob/v0.2.2/docs/running-validations.md)
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
