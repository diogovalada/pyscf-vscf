"""Referee the CPU/GPU analytic Hessian difference with finite differences.

Run on GPU hardware: python validation_data/gpu/hessian_referee.py
Output archived in hessian-referee-nh3-wb97x.txt.
"""

import warnings

import numpy as np
from dataclasses import replace
from pyscf_vscf import ESSettings
from pyscf_vscf.molecule import Molecule
from pyscf_vscf.backends.pyscf import converged_mean_field
from pyscf_vscf.backends._pyscf_properties import to_numpy
from pyscf_vscf import harmonic as hh
from pyscf_vscf.workflows import harmonic as hw
from pyscf_vscf.workflows.scans import molecule_analysis_masses

warnings.filterwarnings("ignore")

NH3 = Molecule.from_arrays(
    ["N", "H", "H", "H"],
    [
        [0, 0, 0.1127],
        [0, 0.9377, -0.2629],
        [0.8121, -0.4689, -0.2629],
        [-0.8121, -0.4689, -0.2629],
    ],
    label="NH3",
)
cpu = ESSettings(method="wb97x", basis="aug-cc-pvdz", use_density_fit=True, scf_conv_tol=1e-12)
gpu = replace(cpu, device="gpu")
masses = molecule_analysis_masses(NH3)


def flat(H):
    H = np.asarray(H)
    n = H.shape[0]
    return H.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)


H = {}
for name, s in (("cpu", cpu), ("gpu", gpu)):
    c = converged_mean_field(NH3, s)
    c.mean_field.conv_tol_cpscf = 1e-10
    H["analytic-" + name] = flat(to_numpy(c.mean_field.Hessian().kernel()))
    pmol = c.build.molecule
    x0 = hw.atom_coords_bohr(pmol).reshape(-1)
    fd = hw.finite_difference_hessian_from_gradients(NH3, s, x0, h=1e-3)
    H["fd-" + name] = 0.5 * (fd + fd.T)
    print("done", name, flush=True)
for k, v in H.items():
    print(
        f"{k:13s}",
        np.round(
            hh.mass_weighted_freqs_modes(pmol, v, masses, rtproj="pyscf", strict=False)[0][-6:], 3
        ),
    )
for a in H:
    print(a, " ".join(f"{b}:{np.abs(H[a] - H[b]).max():.1e}" for b in H if b != a))
