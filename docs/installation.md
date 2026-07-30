# Installation

For a consumer installation with Python 3.10 or newer, install the current
alpha from its immutable GitHub release tag:

```bash
pip install "pyscf-vscf @ git+https://github.com/diogovalada/pyscf-vscf.git@v0.1.0a3"
pip install "pyscf-vscf[pyscf] @ git+https://github.com/diogovalada/pyscf-vscf.git@v0.1.0a3"
```

After the project is published on PyPI, the shorter `pip install pyscf-vscf`
form will become available.

For a source checkout, use `uv`:

```bash
uv sync --extra dev
```

For PySCF-backed harmonic, optimization, and PES/DMS workflows:

```bash
uv sync --extra dev --extra pyscf
```

Prevent unrelated user-site packages from leaking into managed environments:

```bash
export PYTHONNOUSERSITE=1
```

On AMD systems using OpenBLAS, `OPENBLAS_CORETYPE=Zen` may improve performance.
Set thread counts through the CLI's `--max-parallel` and `--pes-workers`
options rather than combining uncontrolled process and BLAS parallelism.

XYZ and MMOL geometry files do not reliably preserve electronic charge or
spin. Pass `--charge` and `--spin` explicitly when reusing non-neutral or
open-shell geometries.
