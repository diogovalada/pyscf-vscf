"""Convergence and property extraction for mean fields built by the seam.

All arrays leave this module as NumPy arrays, whatever array library the
mean-field implementation uses internally. Dipoles are computed from the
converged density with CPU one-electron integrals, so they do not depend on
the implementation's own dipole routine.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from .._identity import immutable_json_mapping
from ..constants import ANG_TO_BOHR
from ._pyscf_factory import MeanFieldBuild

CONTINUITY_SUCCESS_SCOPE = "closed-shell-mulliken-and-meta-lowdin-ao-metrics"
CONTINUITY_REQUIRED_FIELDS = frozenset(
    {
        "mulliken_density_atom_populations",
        "meta_lowdin_density_atom_populations",
        "meta_lowdin_pre_orthogonalization",
        "mulliken_occupied_orbital_atom_populations",
        "occupied_orbital_energies_Eh",
        "reference_homo_lumo_gap_Eh",
    }
)


@dataclass(frozen=True)
class ConvergedMeanField:
    """A converged build plus its total energy, including any field term."""

    build: MeanFieldBuild
    total_energy_Eh: float
    nuclear_field_energy_Eh: float
    cycles: int | None

    @property
    def mean_field(self) -> Any:
        """The converged PySCF mean-field object."""

        return self.build.mean_field


def to_numpy(value: Any) -> np.ndarray:
    """Return ``value`` as a NumPy array, copying from device memory if needed."""

    getter = getattr(value, "get", None)
    if callable(getter) and type(value).__module__.split(".")[0] == "cupy":
        value = getter()
    return np.asarray(value)


def apply_electric_field(
    build: MeanFieldBuild,
    field_au: np.ndarray | None,
    origin_A: np.ndarray | None,
) -> float:
    """Add ``-F.mu`` to the core Hamiltonian; return the nuclear field energy (Eh).

    The Hamiltonian is ``H(F) = H(0) - F.mu`` with ``F`` in atomic units and the
    origin in angstrom. A zero or absent field leaves the mean field unchanged.
    """

    field_vector = np.zeros(3) if field_au is None else np.asarray(field_au, dtype=float)
    origin = np.zeros(3) if origin_A is None else np.asarray(origin_A, dtype=float)
    if field_vector.shape != (3,) or origin.shape != (3,):
        raise ValueError("Electric field and origin must be three-component vectors")
    if not np.all(np.isfinite(field_vector)) or not np.all(np.isfinite(origin)):
        raise ValueError("Electric field and origin must be finite")
    if not np.any(field_vector):
        return 0.0
    molecule = build.molecule
    mean_field = build.mean_field
    origin_bohr = origin * ANG_TO_BOHR
    with molecule.with_common_orig(origin_bohr):
        position_ao = molecule.intor_symmetric("int1e_r", comp=3)
    field_hcore = np.einsum("x,xij->ij", field_vector, position_ao, optimize=True)
    base_get_hcore = mean_field.get_hcore

    def get_hcore(mol: Any = None) -> Any:
        hcore = base_get_hcore(mol)
        return hcore + _like(field_hcore, hcore)

    mean_field.get_hcore = get_hcore
    nuclear_dipole = np.einsum(
        "i,ix->x",
        molecule.atom_charges(),
        molecule.atom_coords() - origin_bohr[None, :],
        optimize=True,
    )
    return -float(field_vector @ nuclear_dipole)


def converge(
    build: MeanFieldBuild,
    *,
    field_au: np.ndarray | None = None,
    field_origin_A: np.ndarray | None = None,
) -> ConvergedMeanField:
    """Apply an optional field, run the SCF, and require a converged finite energy."""

    nuclear_field_energy = apply_electric_field(build, field_au, field_origin_A)
    mean_field = build.mean_field
    mean_field.kernel()
    cycles = getattr(mean_field, "cycles", None)
    if not bool(getattr(mean_field, "converged", False)):
        cycle_text = "" if cycles is None else f", cycles run={int(cycles)}"
        method = build.method_identity.get("method", build.configuration_snapshot.get("class"))
        raise RuntimeError(
            f"SCF did not converge (method={method}, "
            f"max_cycle={getattr(mean_field, 'max_cycle', None)}, "
            f"conv_tol={float(getattr(mean_field, 'conv_tol', float('nan'))):g}{cycle_text})"
        )
    energy = float(to_numpy(mean_field.e_tot)) + nuclear_field_energy
    if not np.isfinite(energy):
        raise RuntimeError("SCF converged to a non-finite energy")
    return ConvergedMeanField(
        build=build,
        total_energy_Eh=energy,
        nuclear_field_energy_Eh=nuclear_field_energy,
        cycles=None if cycles is None else int(cycles),
    )


def density_matrix(converged: ConvergedMeanField) -> np.ndarray:
    """Return the total one-particle density matrix in the AO basis (NumPy)."""

    density = to_numpy(converged.mean_field.make_rdm1())
    if density.ndim == 3:
        density = density[0] + density[1]
    return np.asarray(density, dtype=float)


def dipole_au(converged: ConvergedMeanField) -> np.ndarray:
    """Return the dipole moment (atomic units) about the origin, input frame.

    ``mu = sum_A Z_A R_A - Tr(D r)``, the same convention as PySCF's
    ``dip_moment`` with its default zero origin.
    """

    molecule = converged.build.molecule
    density = density_matrix(converged)
    with molecule.with_common_orig((0.0, 0.0, 0.0)):
        position_ao = molecule.intor_symmetric("int1e_r", comp=3)
    electronic = np.einsum("xij,ji->x", position_ao, density, optimize=True).real
    nuclear = np.einsum("i,ix->x", molecule.atom_charges(), molecule.atom_coords(), optimize=True)
    return np.asarray(nuclear - electronic, dtype=float)


def nuclear_gradient(converged: ConvergedMeanField) -> np.ndarray:
    """Return the analytic nuclear gradient in Hartree/bohr, shape ``(n_atoms, 3)``."""

    mean_field = converged.mean_field
    factory = getattr(mean_field, "nuc_grad_method", None)
    gradients = factory() if callable(factory) else mean_field.Gradients()
    return np.asarray(to_numpy(gradients.kernel()), dtype=float)


def analytic_hessian(converged: ConvergedMeanField) -> np.ndarray | None:
    """Return the analytic Hessian, or ``None`` if the method offers none.

    Failures inside an available Hessian implementation are wrapped as
    ``RuntimeError("Analytic Hessian computation failed")``.
    """

    mean_field = converged.mean_field
    hessian_factory = getattr(mean_field, "Hessian", None)
    if not callable(hessian_factory):
        return None
    try:
        return np.asarray(to_numpy(hessian_factory().kernel()), dtype=float)
    except Exception as exc:
        raise RuntimeError("Analytic Hessian computation failed") from exc


def continuity_diagnostics(
    converged: ConvergedMeanField,
    *,
    mode: str,
    retain_occupied_mo_coefficients: bool,
) -> Mapping[str, object]:
    """Collect density/orbital continuity descriptors under ``mode`` (best-effort|strict)."""

    try:
        continuity = _validated_continuity_diagnostics(
            _mean_field_continuity_diagnostics(
                converged.mean_field,
                retain_occupied_mo_coefficients=retain_occupied_mo_coefficients,
            ),
            require_coefficients=retain_occupied_mo_coefficients,
        )
        if mode == "strict" and continuity.get("continuity_descriptor_scope") != (
            CONTINUITY_SUCCESS_SCOPE
        ):
            raise ValueError(
                f"strict continuity diagnostics require scope {CONTINUITY_SUCCESS_SCOPE!r}"
            )
        return continuity
    except Exception as exc:
        if mode == "strict":
            raise RuntimeError("Strict continuity diagnostics failed") from exc
        return immutable_json_mapping(
            {
                "continuity_descriptor_scope": "unavailable-after-error",
                "continuity_descriptor_error_type": (
                    f"{type(exc).__module__}.{type(exc).__qualname__}"
                ),
            }
        )


def max_rss_mb() -> float | None:
    """Return the peak resident memory of this process in MiB, if available."""

    try:
        import resource
    except ImportError:
        return None
    maximum = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return maximum / (1024.0 * 1024.0)
    return maximum / 1024.0


def current_pyscf_threads() -> int | None:
    """Return PySCF's current thread count without changing it."""

    try:
        from pyscf import lib
    except Exception:
        return None
    return int(lib.num_threads())


def _like(array: np.ndarray, reference: Any) -> Any:
    if type(reference).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy.asarray(array)
    return array


def _mean_field_continuity_diagnostics(
    mean_field: Any,
    *,
    retain_occupied_mo_coefficients: bool,
) -> dict[str, object]:
    """Return compact closed-shell density and occupied-orbital descriptors."""

    from pyscf.scf import hf

    required = ("mo_coeff", "mo_occ", "mo_energy", "make_rdm1", "get_ovlp", "mol")
    if any(not hasattr(mean_field, name) for name in required):
        return {"continuity_descriptor_scope": "unavailable"}
    coefficients = np.asarray(to_numpy(mean_field.mo_coeff), dtype=float)
    occupations = np.asarray(to_numpy(mean_field.mo_occ), dtype=float)
    orbital_energies = np.asarray(to_numpy(mean_field.mo_energy), dtype=float)
    density = np.asarray(to_numpy(mean_field.make_rdm1()), dtype=float)
    if coefficients.ndim != 2 or occupations.ndim != 1 or density.ndim != 2:
        return {"continuity_descriptor_scope": "unavailable-for-spin-resolved-reference"}
    occupied = np.flatnonzero(occupations > 0.0)
    virtual = np.flatnonzero(occupations == 0.0)
    if not np.all(np.isclose(occupations[occupied], 2.0)):
        return {"continuity_descriptor_scope": "unavailable-for-open-shell-reference"}
    overlap = np.asarray(to_numpy(mean_field.get_ovlp()), dtype=float)
    gross_ao = np.einsum("ij,ji->i", density, overlap, optimize=True)
    atom_slices = np.asarray(mean_field.mol.aoslice_by_atom(), dtype=int)[:, 2:4]
    mulliken_atom_populations = [
        float(np.sum(gross_ao[start:stop])) for start, stop in atom_slices
    ]
    _, meta_lowdin_charges = hf.mulliken_pop_meta_lowdin_ao(
        mean_field.mol,
        density,
        verbose=0,
        pre_orth_method="ANO",
        s=overlap,
    )
    meta_lowdin_atom_populations = (
        np.asarray(mean_field.mol.atom_charges(), dtype=float)
        - np.asarray(meta_lowdin_charges, dtype=float)
    ).tolist()
    occupied_characters = []
    for orbital in occupied:
        vector = coefficients[:, orbital]
        gross_orbital = vector * (overlap @ vector)
        occupied_characters.append(
            [float(np.sum(gross_orbital[start:stop])) for start, stop in atom_slices]
        )
    gap = None
    if occupied.size and virtual.size:
        gap = float(orbital_energies[virtual[0]] - orbital_energies[occupied[-1]])
    diagnostics = {
        "continuity_descriptor_scope": CONTINUITY_SUCCESS_SCOPE,
        "mulliken_density_atom_populations": mulliken_atom_populations,
        "meta_lowdin_density_atom_populations": meta_lowdin_atom_populations,
        "meta_lowdin_pre_orthogonalization": "ANO",
        "mulliken_occupied_orbital_atom_populations": occupied_characters,
        "occupied_orbital_energies_Eh": orbital_energies[occupied].tolist(),
        "reference_homo_lumo_gap_Eh": gap,
    }
    if retain_occupied_mo_coefficients:
        diagnostics["occupied_mo_coefficients_ao"] = coefficients[:, occupied].tolist()
    return diagnostics


def _validated_continuity_diagnostics(
    diagnostics: Mapping[str, object],
    *,
    require_coefficients: bool,
) -> Mapping[str, object]:
    """Freeze one descriptor mapping and validate a claimed successful scope."""

    frozen = immutable_json_mapping(diagnostics)
    scope = frozen.get("continuity_descriptor_scope")
    if isinstance(scope, str) and scope.startswith("unavailable"):
        return frozen
    if scope != CONTINUITY_SUCCESS_SCOPE:
        raise ValueError("Continuity diagnostics returned an invalid descriptor scope")
    required = set(CONTINUITY_REQUIRED_FIELDS)
    if require_coefficients:
        required.add("occupied_mo_coefficients_ao")
    missing = sorted(required.difference(frozen))
    if missing:
        raise ValueError(f"Successful continuity diagnostics are missing fields: {missing}")
    if frozen["meta_lowdin_pre_orthogonalization"] != "ANO":
        raise ValueError("Successful continuity diagnostics require ANO pre-orthogonalization")
    mulliken_density = _continuity_array(
        "mulliken_density_atom_populations",
        frozen["mulliken_density_atom_populations"],
        ndim=1,
    )
    meta_lowdin_density = _continuity_array(
        "meta_lowdin_density_atom_populations",
        frozen["meta_lowdin_density_atom_populations"],
        ndim=1,
    )
    occupied_populations = _continuity_array(
        "mulliken_occupied_orbital_atom_populations",
        frozen["mulliken_occupied_orbital_atom_populations"],
        ndim=2,
    )
    occupied_energies = _continuity_array(
        "occupied_orbital_energies_Eh",
        frozen["occupied_orbital_energies_Eh"],
        ndim=1,
    )
    atom_count = mulliken_density.size
    occupied_count = occupied_energies.size
    if atom_count == 0 or meta_lowdin_density.shape != (atom_count,):
        raise ValueError("Continuity atom-population arrays must have the same non-zero length")
    if occupied_count == 0 or occupied_populations.shape != (occupied_count, atom_count):
        raise ValueError("Occupied-orbital populations must have shape (occupied orbitals, atoms)")
    _continuity_scalar(
        "reference_homo_lumo_gap_Eh",
        frozen["reference_homo_lumo_gap_Eh"],
    )
    if require_coefficients:
        coefficients = _continuity_array(
            "occupied_mo_coefficients_ao",
            frozen["occupied_mo_coefficients_ao"],
            ndim=2,
        )
        if coefficients.shape[0] == 0 or coefficients.shape[1] != occupied_count:
            raise ValueError(
                "Occupied MO coefficients must have shape (AO functions, occupied orbitals)"
            )
    return frozen


def _continuity_array(name: str, value: object, *, ndim: int) -> np.ndarray:
    if not isinstance(value, tuple):
        raise ValueError(f"{name} must be a numeric array")
    try:
        raw = np.asarray(value)
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a rectangular numeric array") from exc
    if raw.dtype.kind not in "iuf" or values.ndim != ndim or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite {ndim}D numeric array")
    return values


def _continuity_scalar(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _normalized_continuity_mode(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("continuity_diagnostics must be a string")
    mode = value.strip().lower()
    if mode not in {"off", "best-effort", "strict"}:
        raise ValueError("continuity_diagnostics must be 'off', 'best-effort', or 'strict'")
    return mode


__all__ = [
    "CONTINUITY_REQUIRED_FIELDS",
    "CONTINUITY_SUCCESS_SCOPE",
    "ConvergedMeanField",
    "analytic_hessian",
    "apply_electric_field",
    "continuity_diagnostics",
    "converge",
    "current_pyscf_threads",
    "density_matrix",
    "dipole_au",
    "max_rss_mb",
    "nuclear_gradient",
    "to_numpy",
]
