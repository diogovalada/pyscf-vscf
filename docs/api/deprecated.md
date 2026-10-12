# 0.2 scan layer and cache helpers

The 0.2 tuple-returning scan functions, the `energy_dipole` evaluator API,
and the schema-3 cache writers and loaders still work in 0.3; each emits a
`PyscfVscfDeprecationWarning` naming its replacement and is removed in 0.4.
See the [migration guide](../migration-0.3.md). Functions on this page that
do not warn, such as the provenance helpers in `cache`, are not deprecated.
`pyscf_vscf.surfaces` re-exports the 0.2 scan functions and the evaluator API
under the same names.

::: pyscf_vscf.workflows.scans

::: pyscf_vscf.electronic
    options:
      filters: ["!^_", "!^AU_DIPOLE_TO_DEBYE$", "!^ElectronicPointRequest$", "!^ElectronicProvider$", "!^ElectronicResult$", "!^FrozenJSONMapping$", "!^POINT_SCHEMA_VERSION$", "!^provider_scientific_fingerprint$"]

::: pyscf_vscf.cache
    options:
      filters: ["!^_", "!^dump_grid_npz$", "!^load_grid_npz$"]
