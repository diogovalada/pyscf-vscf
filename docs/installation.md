# Installation

For a consumer installation with Python 3.10 or newer, install from PyPI:

```bash
pip install pyscf-vscf
```

PySCF is the supported electronic-structure backend and is installed with the
package.
geomeTRIC is also installed for the built-in optimization and relaxed-scan
workflows. Install optional PySCF extensions directly through PySCF and select
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
Set thread counts through the CLI's `--max-parallel` and `--pes-workers`
options rather than combining uncontrolled process and BLAS parallelism.

XYZ and MMOL geometry files do not reliably preserve electronic charge or
spin. Pass `--charge` and `--spin` explicitly when reusing non-neutral or
open-shell geometries.
