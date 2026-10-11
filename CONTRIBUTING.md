# Contributing

Bug reports, questions, and contributions are welcome.

## Getting help and reporting problems

- **Questions** about using the package or interpreting results: open a
  GitHub issue with the "Question" template.
- **Bugs**: open an issue with the "Bug report" template. Include the
  `pyscf-vscf --version` output, the exact command or a minimal Python script,
  the geometry file, and the full output. Rerun failing commands with
  `--debug` and include the traceback.
- **Scientific concerns** (for example a result that disagrees with a
  reference value): include the reference, the electronic-structure settings,
  and the grid used.

## Development setup

Use Python 3.11 or newer and [`uv`](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/diogovalada/pyscf-vscf.git
cd pyscf-vscf
uv sync --extra dev
```

## Checks

These are the checks run by continuous integration:

```bash
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run mypy                                                   # strict on the 0.3 core modules
uv run pytest -m "not pyscf and not slow and not cache" -q   # fast, no PySCF calculations
uv run pytest -q --cov=pyscf_vscf --cov-fail-under=78        # everything, with coverage
uv run python scripts/validate_archived_grids.py --nmax 12 --output /tmp/water-report.json
uv run python scripts/compare_validation_reports.py validation_data/convergence_report.json /tmp/water-report.json
uv run python scripts/validate_nh3_three_mode.py --output /tmp/nh3-report.json
uv run python scripts/compare_validation_reports.py validation_data/nh3_three_mode/report.json /tmp/nh3-report.json
uv run python -m build && uv run twine check dist/*
```

Continuous integration also tests the lowest supported dependency versions
(`uv pip install --resolution lowest-direct .` on Python 3.11). The
`pyscf`-marked tests run small electronic-structure calculations; the archived
validations do not run PySCF.

## Scientific changes

Scientific changes must include units, coordinate definitions, provenance, and
a focused analytic or archived-data regression. Do not update numerical
reference values without documenting why the scientific expectation changed.
Releases follow [docs/releasing.md](docs/releasing.md).
