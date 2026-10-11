# Installation

For a consumer installation with Python 3.11 or newer, install from PyPI:

```bash
pip install pyscf-vscf
```

PySCF is the supported electronic-structure backend and is installed with the
package.
geomeTRIC is also installed for the built-in geometry-optimization workflow.
The constrained normal-relaxed scan uses SciPy. Install optional PySCF
extensions directly through PySCF and select
them with PySCF's native method specification; `pyscf-vscf` does not duplicate
those dependencies or configuration options.

PySCF does not support native Windows installations. Use WSL on Windows.

For a source checkout, use `uv`:

```bash
uv sync --extra dev
```

Prevent unrelated user-site packages from leaking into managed environments:

```bash
export PYTHONNOUSERSITE=1
```

On AMD systems using OpenBLAS, `OPENBLAS_CORETYPE=Zen` may improve performance.

## Parallelism and memory

Scans evaluate independent electronic points. By default the CLI runs up to
one worker process per available CPU, limited by the number of points and by
available memory (at least 1 GiB per worker from 75% of the available memory,
honouring cgroup limits that are visible to the process). When fewer workers
than CPUs are started, the remaining CPUs are split among the workers as
threads.
Each automatic worker receives an explicit PySCF memory budget of at most
4000 MB. When available memory cannot be determined, at most four workers
are used.

Override the defaults with `--max-parallel` (total threads),
`--pes-workers` (processes; the threads are split between them), and
`--max-memory` (MB per process). Harmonic analysis and optimization use one
process with `--max-parallel` threads. From Python, the environment variable
`PYSCF_VSCF_MAX_MEMORY_MB` sets the same per-process budget for every PySCF
molecule the package builds. Avoid combining uncontrolled process and BLAS
parallelism.

XYZ and MMOL geometry files do not reliably preserve electronic charge or
spin. Pass `--charge` and `--spin` explicitly when reusing non-neutral or
open-shell geometries.
