# Releasing

## Public-history policy

The public repository was initialized from a sanitized source export. Do not
merge private research history or refs into it. Before every release, verify
that the complete public history contains no redistributed third-party papers,
copied web pages, credentials, private data, or generated build artifacts.
Removing a file from the current tree is not sufficient if it has already been
committed.

1. Run `git status --short --branch` and
   `git ls-files --others --exclude-standard`; account for every local file and
   verify that the release branch is synchronized with its upstream.
2. Confirm `CHANGELOG.md`, `CITATION.cff`, `pyproject.toml`, and
   `pyscf_vscf.__version__` agree.
3. Run the pure and PySCF test suites, regenerate both archived
   validation reports, and compare them with the committed reports using
   `scripts/compare_validation_reports.py`. The comparison requires identical
   structure and discrete values and uses only tight floating-point tolerances
   for cross-platform eigensolver roundoff. The defaults use `rtol=1e-9`, a
   general `atol=1e-15`, `1e-9 cm^-1` for wavenumber-valued fields, and
   `1e-24 m^2 s^-1` for intensity-valued fields.
4. Build and inspect both distributions with `uv build` and
   `uvx twine check dist/*`.
5. Confirm that the sdist contains the scripts, geometries, and archived data
   needed for documented source-based validation.
6. Install both the wheel and sdist into isolated environments and run the
   packaged examples.
7. Tag the reviewed commit as `vX.Y.Z` and push the tag.
8. The release workflow creates a GitHub release, attaches the wheel and source
   distribution, and marks PEP 440 alpha, beta, and release-candidate versions
   as prereleases.
9. PyPI publishing is disabled by default. Before enabling it, configure the
   GitHub `pypi` environment and the PyPI trusted publisher, then set the
   repository variable `PUBLISH_TO_PYPI` to `true`.
10. Review the generated GitHub release notes against the matching changelog
   section.

Never release from a dirty worktree or by uploading an unreviewed local build.
