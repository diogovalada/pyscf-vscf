"""Small PySCF-free VCI calculation on an analytic two-mode Hamiltonian."""

from __future__ import annotations

import json

import numpy as np

from pyscf_vscf.constants import HARTREE_TO_CM
from pyscf_vscf.vci import VCISettings, build_nmode_vscf_modal_basis, solve_vci
from pyscf_vscf.vscf import NModePotential


def build_model() -> NModePotential:
    """Return a deterministic coupled two-mode model in Angstrom and Hartree."""

    q1 = np.linspace(-0.36, 0.36, 17)
    q2 = np.linspace(-0.32, 0.32, 15)
    v1 = 0.10 * q1**2 + 0.008 * q1**4
    v2 = 0.13 * q2**2 + 0.006 * q2**4
    coupling = 0.018 * q1[:, None] * q2[None, :]
    return NModePotential(
        coordinates=(q1, q2),
        masses_amu=(1.0, 1.4),
        one_mode_potentials_Eh=(v1, v2),
        two_mode_couplings_Eh={(0, 1): coupling},
        mode_labels=("q1", "q2"),
        metadata={"example": "analytic-two-mode-vci"},
    )


def main() -> None:
    hamiltonian, modal_basis = build_nmode_vscf_modal_basis(build_model(), 6)
    result = solve_vci(
        hamiltonian,
        modal_basis,
        settings=VCISettings(nstates=6, max_total_quanta=6, extra_eigenstates=2),
    )
    payload = {
        "method": "VCI on converged VSCF modals",
        "vscf_converged": modal_basis.converged,
        "state_cutoff_margin_Eh": result.state_cutoff_margin_Eh,
        "maximum_residual_Eh": float(np.max(result.residual_norms_Eh)),
        "transition_frequencies_cm": [
            float((energy - result.energies_Eh[0]) * HARTREE_TO_CM)
            for energy in result.energies_Eh[1:]
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
