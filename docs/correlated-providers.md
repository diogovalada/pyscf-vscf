# Correlated and finite-field providers

The optional correlated backend exposes closed-shell RHF-CCSD(T) energies and
finite-field dipoles through the electronic-point contract:

```python
from pyscf_vscf.backends.pyscf_correlated import (
    CCSDPerturbativeTriplesSettings,
    FiniteFieldDipoleProvider,
    PySCFCCSDPerturbativeTriplesProvider,
)

energy_provider = PySCFCCSDPerturbativeTriplesProvider(
    CCSDPerturbativeTriplesSettings(
        basis="cc-pvdz",
        density_fit=False,
        frozen_orbitals=(),
    ),
    threads=1,
)
dipole_provider = FiniteFieldDipoleProvider(
    energy_provider,
    field_magnitudes_au=(1e-4, 5e-5),
)
```

`PySCFCCSDPerturbativeTriplesProvider` returns energies only. The name is
deliberately explicit: `(T)` is a perturbative triples correction, not full
CCSDT. The implementation currently rejects open-shell references,
unconverged SCF/CCSD results, non-finite corrections, and property requests
other than energy.

When `density_fit=True`, density fitting applies to the CCSD correlation
integrals. The RHF reference remains conventional; the auxiliary basis and
density-fitting policy remain part of the scientific provider identity.

`FiniteFieldDipoleProvider` evaluates one zero-field point and the positive and
negative field along each Cartesian axis at every configured magnitude. It
uses

```text
H(F) = H(0) - F.mu
mu = -dE/dF
```

with fields in atomic units, origins in Angstrom, and the nuclear field term
included in each total energy. Neutral systems default to the zero-Angstrom
origin; charged systems are rejected. Every signed field has a distinct causal
identity. Assembly fails unless every result proves the expected field vector,
origin, Hamiltonian, and nuclear-term convention.

The reported dipole is the intercept of a linear fit versus squared field
magnitude. Scientific diagnostics retain the dipole from each step, the fit
condition number, step spread, fit residual, and all constituent point
identities. A CCSD density is never labeled as a CCSD(T) dipole.

Scientific fingerprints include method, basis, density fitting and auxiliary
basis, frozen orbitals, convergence settings, integral policy, diagnostics,
field steps, and conventions. Threads, memory limits, host data, software
versions, paths, and user annotations are retained separately as execution
provenance and do not change scientific identity.
