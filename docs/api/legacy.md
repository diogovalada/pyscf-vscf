# Relaxed scans, 0.2 caches, and scan helpers

`pyscf_vscf.workflows.scans` holds what remains of the 0.2 scan layer:
constrained relaxed normal-coordinate scans, normal-mode selection,
coordinate and mass helpers, and the schema-3 cache metadata helpers used to
read 0.2 caches. `pyscf_vscf.electronic.energy_dipole` evaluates one
built-in PySCF point as a Hartree energy and Debye dipole. The 0.2 tuple
scan functions and schema-3 cache writers and loaders were removed in 0.4;
see the [migration guide](../migration-0.4.md).

::: pyscf_vscf.workflows.scans
    options:
      filters: ["!^_", "!^Bond$", "!^parse_bond$"]

::: pyscf_vscf.electronic
    options:
      filters: ["!^_", "!^AU_DIPOLE_TO_DEBYE$", "!^ElectronicPointRequest$", "!^ElectronicProvider$", "!^ElectronicResult$", "!^FrozenJSONMapping$", "!^POINT_SCHEMA_VERSION$", "!^provider_scientific_fingerprint$"]

::: pyscf_vscf.cache
    options:
      filters: ["!^_", "!^dump_grid_npz$", "!^load_grid_npz$"]
