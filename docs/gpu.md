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

Measured with `scripts/gpu_benchmark.py` on an RTX 3060 (a consumer card
with slow double precision) against one core of an AMD EPYC 7402.
*Cores replaced* is the number of single-threaded CPU workers that match one
GPU:

BENCHMARK_TABLE

- With density fitting (the default), the GPU replaces many CPU cores, even
  for water.
- Without density fitting, small molecules gain little or nothing; the GPU
  pays off for larger molecules and basis sets.
- Datacenter GPUs (V100, A100, H100) have much faster double precision than
  consumer cards.

Run the benchmark on your own hardware:

```bash
python scripts/gpu_benchmark.py --output gpu-benchmark.json
```

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
one. The analytic-Hessian check and HDO were added afterwards.

```bash
python scripts/gpu_parity.py --output gpu-parity.json
pytest -m gpu      # a shorter run of the same checks
```

## Limits

- CCSD(T) and the other correlated providers run on the CPU only.
- GPU4PySCF supports most, not all, PySCF features; a recipe that uses a
  feature without a GPU implementation fails when it is built or run.
