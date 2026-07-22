"""Small PySCF-free two-mode VSCF example.

Run with ``python -m pyscf_vscf.examples.vscf_two_mode``.
"""

from __future__ import annotations

import json

import numpy as np

from pyscf_vscf.vscf import NModePotential, vscf_spectrum


def build_model() -> NModePotential:
    """Return a deterministic coupled two-mode model in Angstrom and Hartree."""

    q1 = np.linspace(-0.35, 0.35, 31)
    q2 = np.linspace(-0.30, 0.30, 29)
    v1 = 0.08 * q1**2 + 0.006 * q1**4
    v2 = 0.11 * q2**2 + 0.004 * q2**4
    coupling = 0.20 * q1[:, None] ** 2 * q2[None, :] ** 2
    return NModePotential(
        coordinates=(q1, q2),
        masses_amu=(1.0, 1.5),
        one_mode_potentials_Eh=(v1, v2),
        two_mode_couplings_Eh={(0, 1): coupling},
        mode_labels=("q1", "q2"),
        metadata={"example": "coupled-two-mode"},
    )


def main() -> None:
    spectrum = vscf_spectrum(
        build_model(),
        max_quanta_per_mode=2,
        max_total_quanta=2,
    )
    payload = {
        "method": "state-specific VSCF",
        "ground_converged": spectrum.ground.converged,
        "ground_iterations": spectrum.ground.iterations,
        "transitions": [
            {"quanta": transition.quanta, "frequency_cm": transition.frequency_cm}
            for transition in spectrum.transitions
        ],
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
