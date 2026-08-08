# Contributing

Use Python 3.10 or newer and `uv`.

```bash
uv sync --extra dev
uv run ruff check src tests scripts/validate_archived_grids.py
uv run ruff format --check src tests scripts/validate_archived_grids.py
uv run pytest -m "not pyscf" -q
uv run python -m build
uv run twine check dist/*
```

PySCF-backed checks are separated because they execute electronic-structure
calculations:

```bash
uv run pytest -m pyscf -q
```

Scientific changes must include units, coordinate definitions, provenance, and
a focused analytic or archived-data regression. Do not update numerical
reference values without documenting why the scientific expectation changed.
