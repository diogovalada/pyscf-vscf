# GPU execution

The electronic points can run on an NVIDIA GPU through
[GPU4PySCF](https://github.com/pyscf/gpu4pyscf). The method is the same:
pyscf-vscf configures it on the CPU exactly as for a CPU run and converts it
with PySCF's `to_gpu()`. Everything after the SCF (dipoles, surfaces, DVR,
VSCF, VCI) is unchanged.

## Requirements and installation

- Linux on x86-64 (on Windows, use WSL2), an NVIDIA GPU with compute
  capability 7.0 or newer (Volta and later), and an NVIDIA driver for CUDA 12
  or 13.
- The GPU extra for your CUDA version:

```bash
pip install "pyscf-vscf[gpu-cuda12]"    # or gpu-cuda13
```

It installs GPU4PySCF, CuPy, and the CUDA libraries CuPy needs.

## Use

On the command line, add `--device gpu`:

```bash
pyscf-vscf opt water.xyz --device gpu
pyscf-vscf harmonic water-opt.xyz --device gpu
pyscf-vscf scan1d water-opt.xyz --bond 0-1 --device gpu --store water-gpu.sqlite
```

In Python, set the device on the method:

```python
from pyscf_vscf import ESSettings, PySCFMeanFieldProvider

provider = PySCFMeanFieldProvider(ESSettings(method="wb97x", basis="aug-cc-pvtz", device="gpu"))
```

A `MethodRecipe` takes `device="gpu"` too, or `recipe.on_device("gpu")`; its
`mean_field` function still returns the CPU object. With `--recipe`,
`--device gpu` moves the recipe to the GPU.

## Workers

Scans run one worker per visible GPU by default (`CUDA_VISIBLE_DEVICES`
selects the GPUs). Each worker is pinned to one GPU, round-robin, so
`--workers 2` on one GPU runs two workers sharing it, and on two GPUs one
worker on each. For small molecules a single worker leaves the GPU partly
idle; see the throughput benchmark below. In Python, `RunnerSettings(workers=N)`
does the same.

With more than one GPU, a script that runs workers must not import CuPy or
GPU4PySCF at its top level: spawned workers re-import the main script before
they are pinned to a GPU, and the runner refuses to pin a worker in which
CUDA may already have started. Multi-GPU pinning is covered by unit tests;
the GPU runs reported here used a single GPU.

Relaxed normal-coordinate scans (`--coordinate normal-relaxed`) run their
points one after another in the main process on the GPU.

## Identity and reuse

The device is part of the method's identity. A GPU method has a different
fingerprint from the same method on the CPU, so a point store never mixes
CPU and GPU points in one scan, and `--reuse` uses the device recorded in
the scan file. CPU fingerprints are the same as in earlier releases.

CPU and GPU agree closely but not to the last digit: the parity suite below
requires agreement within 1e-8 Eh for energies and 1e-5 atomic units for
dipoles, and the observed differences are far smaller for energies
(below 1e-10 Eh).

## When the GPU helps

The speed-up depends on the molecule's size, the method, the basis set,
density fitting, and the GPU's double-precision throughput, so measure it
for your own calculations (`scripts/gpu_benchmark.py`). As one example, an
RTX 3060 (a consumer card, whose double precision is 1/64 of its single
precision) against one core of an AMD EPYC 7402, energy and dipole of one
point (`validation_data/gpu/benchmark-rtx3060.json`). *CPU cores matched* is
the CPU time divided by the GPU time:

| Molecule | Method | Density fitting | CPU, 1 core (s) | GPU (s) | CPU cores matched |
| --- | --- | --- | --- | --- | --- |
| Water | HF/cc-pVDZ | no / yes | 0.05 / 0.05 | 1.85 / 0.21 | 0.0 / 0.3 |
| Water | B3LYP/cc-pVDZ | no / yes | 0.45 / 0.48 | 1.62 / 0.31 | 0.3 / 1.5 |
| Water | B3LYP/aug-cc-pVTZ | no / yes | 2.04 / 1.72 | 4.14 / 0.46 | 0.5 / 3.8 |
| Water | ωB97X/aug-cc-pVTZ | no / yes | 4.62 / 1.89 | 4.62 / 0.55 | 1.0 / 3.4 |
| Water dimer | B3LYP/aug-cc-pVTZ | no / yes | 15.4 / 6.03 | 10.9 / 1.20 | 1.4 / 5.0 |
| Water dimer | ωB97X/aug-cc-pVTZ | no / yes | 53.2 / 7.23 | 12.7 / 1.43 | 4.2 / 5.1 |
| Pyridine | B3LYP/cc-pVDZ | no / yes | 11.2 / 9.49 | 6.63 / 1.72 | 1.7 / 5.5 |

A 24-point water bond scan at ωB97X/aug-cc-pVTZ with density fitting ran at
95, 120, and 124 points per minute with one, two, and three workers sharing
the GPU, and at 122 points per minute on three CPU cores.

- On this card the ratio ranged from 0.03 to 5.5. It was largest (1.5-5.5)
  for DFT with density fitting, and grew with the molecule and basis set.
- For small molecules the CPU was faster in most cases: HF on water, with or
  without density fitting, and DFT without density fitting in small basis
  sets (ωB97X/aug-cc-pVTZ water without density fitting was a tie).
- GPUs with fast double precision (V100, A100, H100) can be expected to do
  considerably better; these numbers do not measure them.
- Several workers can share one GPU (`--workers`); for small molecules this
  raises throughput.

## Parity suite

`scripts/gpu_parity.py` compares the GPU with the CPU through the package
for H2O, HDO, D2O, and NH3 with HF, B3LYP (density fitting), and ωB97X
(density fitting): energies, dipoles, finite-field energies, gradients,
analytic Hessians, harmonic frequencies, optimized geometries, and a bond
scan with its spectrum. The reports are archived in `validation_data/gpu/`.

The tolerances were fixed before the first GPU run, with one revision. In
that run 47 of 48 checks passed: for NH3 at ωB97X the harmonic frequencies
differed by 0.24 cm⁻¹, against a tolerance of 0.1 cm⁻¹. A finite-difference
check (`validation_data/gpu/hessian-referee-nh3-wb97x.txt`) showed that both
analytic Hessians differ by about ten times more (1.8 cm⁻¹ in the lowest
mode) from a finite-difference Hessian of their own gradients (consistent
with neither including the DFT grid response), while those finite-difference
Hessians agree between CPU and GPU to 0.03 cm⁻¹. The frequency tolerance
was raised to 0.5 cm⁻¹; the first report is archived alongside the final
one. The analytic-Hessian check and HDO were added afterwards; its
tolerance, 1e-4 Eh/bohr², was set from the NH3 difference (3.5e-5), and
water at ωB97X later came closest to it (9.0e-5). Harmonic frequencies
agree within 0.001 cm⁻¹ for HF, 0.021 cm⁻¹ for B3LYP, and 0.24 cm⁻¹ for ωB97X.

```bash
python scripts/gpu_parity.py --output gpu-parity.json
pytest -m gpu      # a shorter run of the same checks
```

## Limits

- CCSD(T) and the other correlated providers run on the CPU only.
- GPU4PySCF supports most, not all, PySCF features; a recipe that uses a
  feature without a GPU implementation fails when it is built or run.
