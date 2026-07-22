# Package examples

## State-specific VSCF

```python
import numpy as np

from pyscf_vscf import NModePotential, vscf_spectrum

q1 = np.linspace(-0.35, 0.35, 31)
q2 = np.linspace(-0.30, 0.30, 29)
model = NModePotential(
    coordinates=(q1, q2),
    masses_amu=(1.0, 1.5),
    one_mode_potentials_Eh=(0.08 * q1**2, 0.11 * q2**2),
    two_mode_couplings_Eh={(0, 1): 0.20 * q1[:, None] ** 2 * q2[None, :] ** 2},
    mode_labels=("q1", "q2"),
)
spectrum = vscf_spectrum(model, max_quanta_per_mode=2, max_total_quanta=2)
for transition in spectrum.transitions:
    print(transition.quanta, transition.frequency_cm)
```

This is VSCF, not VCI. `V_ij` arrays are coupling corrections.

## Cached 1D DVR

Production cache reuse should go through the scan workflow loaders so the full
schema-v2 scientific fingerprint is validated. The variational layer consumes
arrays in Angstrom, Hartree relative to the grid minimum, and Debye:

```python
from pyscf_vscf.variational import variational_1d

records = variational_1d(
    R_A,
    E_Eh,
    MU_Debye,
    redmass_amu,
    axis=[1.0, 0.0, 0.0],
    vmax=8,
    intensity="both",
)
for record in records:
    print(
        record["v"],
        record["freq_cm"],
        record["transition_dipole_norm_D"],
        record["integrated_cross_section_isotropic_omega_m2_per_s"],
    )
```

## Archived convergence data

```bash
uv run python scripts/validate_archived_grids.py --output validation_data/report.json
```

This command verifies archive hashes, recomputes assigned spectra with the
current intensity convention, and reports numerical spreads. It does not import
or run PySCF.
