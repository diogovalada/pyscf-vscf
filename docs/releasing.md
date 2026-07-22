# Releasing

## Public-history policy

The public repository was initialized from a sanitized source export. Do not
merge private research history or refs into it. Before every release, verify
that the complete public history contains no redistributed third-party papers,
copied web pages, credentials, private data, or generated build artifacts.
Removing a file from the current tree is not sufficient if it has already been
committed.

1. Confirm `CHANGELOG.md`, `CITATION.cff`, `pyproject.toml`, and
   `pyscf_vscf.__version__` agree.
2. Run the pure and optional PySCF test suites.
3. Build and inspect both distributions with `uv build` and
   `uvx twine check dist/*`.
4. Install the wheel into an isolated environment and run the packaged example.
5. Tag the reviewed commit as `vX.Y.Z` and push the tag.
6. The release workflow creates a GitHub release, attaches the wheel and source
   distribution, and marks PEP 440 alpha, beta, and release-candidate versions
   as prereleases.
7. PyPI publishing is disabled by default. Before enabling it, configure the
   GitHub `pypi` environment and the PyPI trusted publisher, then set the
   repository variable `PUBLISH_TO_PYPI` to `true`.
8. Review the generated GitHub release notes against the matching changelog
   section.

Never release from a dirty worktree or by uploading an unreviewed local build.
