"""Closed-shell PySCF CCSD(T) energies and finite-field dipoles."""

from __future__ import annotations

import operator
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from socket import gethostname
from typing import Any

import numpy as np

from .._identity import immutable_json_mapping, to_jsonable
from ..cache import runtime_provenance
from ..electronic import (
    ElectronicPointRequest,
    ElectronicProvider,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from ..molecule import Molecule
from .pyscf import (
    _apply_electric_field,
    _configure_pyscf_threads,
    _max_rss_mb,
    _optional_int,
    _require_pyscf,
    molecule_to_pyscf,
)


_FIELD_CAPABILITY = {
    "field_units": "atomic_unit",
    "origin_units": "angstrom",
    "hamiltonian": "H(F)=H(0)-F.mu",
    "nuclear_term": "included-in-total-energy",
}
_FINITE_FIELD_SIGN_CONVENTION = "mu=-dE/dF; H(F)=H(0)-F.mu"


@dataclass(frozen=True)
class CCSDPerturbativeTriplesSettings:
    """Scientific and numerical settings for closed-shell RHF-CCSD(T)."""

    basis: str
    density_fit: bool = False
    auxiliary_basis: str | None = None
    frozen_orbitals: tuple[int, ...] = ()
    scf_conv_tol: float = 1e-10
    scf_max_cycle: int = 80
    cc_conv_tol: float = 1e-9
    cc_max_cycle: int = 80
    direct: bool = False
    integral_policy: str = "auto"
    requested_diagnostics: tuple[str, ...] = ("t1", "d1")

    def __post_init__(self) -> None:
        basis = _nonempty("basis", self.basis)
        density_fit = bool(self.density_fit)
        auxiliary = self.auxiliary_basis
        if density_fit:
            auxiliary = _nonempty("auxiliary_basis", auxiliary)
        elif auxiliary is not None:
            raise ValueError("auxiliary_basis must be None when density_fit is disabled")

        frozen = tuple(operator.index(value) for value in self.frozen_orbitals)
        if frozen != tuple(sorted(set(frozen))) or any(value < 0 for value in frozen):
            raise ValueError("frozen_orbitals must be ordered unique non-negative indices")
        for name in ("scf_conv_tol", "cc_conv_tol"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        for name in ("scf_max_cycle", "cc_max_cycle"):
            value = operator.index(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

        policy = str(self.integral_policy).strip().lower()
        if policy not in {"auto", "incore", "outcore"}:
            raise ValueError("integral_policy must be auto, incore, or outcore")
        direct = bool(self.direct)
        if density_fit and direct:
            raise ValueError("PySCF DF-CCSD requires direct=False")
        if density_fit and policy != "auto":
            raise ValueError("DF-CCSD supports only integral_policy='auto'")
        if direct and policy == "incore":
            raise ValueError("AO-direct CCSD is incompatible with integral_policy='incore'")

        diagnostics = tuple(str(value).strip().lower() for value in self.requested_diagnostics)
        if (
            not diagnostics
            or "t1" not in diagnostics
            or len(set(diagnostics)) != len(diagnostics)
            or set(diagnostics) - {"t1", "d1"}
        ):
            raise ValueError("requested_diagnostics must be unique t1/d1 names and include t1")

        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "density_fit", density_fit)
        object.__setattr__(self, "auxiliary_basis", auxiliary)
        object.__setattr__(self, "frozen_orbitals", frozen)
        object.__setattr__(self, "direct", direct)
        object.__setattr__(self, "integral_policy", policy)
        object.__setattr__(self, "requested_diagnostics", diagnostics)


@dataclass(frozen=True)
class PySCFCCSDPerturbativeTriplesProvider:
    """Strict closed-shell RHF-CCSD(T) electronic energy provider."""

    settings: CCSDPerturbativeTriplesSettings
    threads: int = 1
    max_memory_mb: int = 4000
    user_annotations: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.settings, CCSDPerturbativeTriplesSettings):
            raise TypeError("settings must be CCSDPerturbativeTriplesSettings")
        threads = operator.index(self.threads)
        memory = operator.index(self.max_memory_mb)
        if threads < 1 or memory < 1:
            raise ValueError("threads and max_memory_mb must be positive")
        object.__setattr__(self, "threads", threads)
        object.__setattr__(self, "max_memory_mb", memory)
        object.__setattr__(
            self,
            "user_annotations",
            immutable_json_mapping(self.user_annotations),
        )

    def scientific_settings_payload(self) -> Mapping[str, object]:
        """Return every setting that can change the intended calculation."""

        effective_method = "RHF-DF-CCSD(T)" if self.settings.density_fit else "RHF-CCSD(T)"
        return {
            "schema": "pyscf-vscf-provider-settings",
            "schema_version": 1,
            "backend_family": "pyscf",
            "provider": "closed-shell-rhf-ccsd-perturbative-triples-energy",
            "effective_method": effective_method,
            "reference_policy": "restricted-closed-shell-only",
            "settings": asdict(self.settings),
            "orbital_policy": "self-consistent-canonical",
            "field_convention": dict(_FIELD_CAPABILITY),
        }

    def execution_provenance(self) -> Mapping[str, object]:
        """Return resources and software versions outside scientific identity."""

        return {
            "threads_requested": self.threads,
            "max_memory_mb": self.max_memory_mb,
            "host": gethostname(),
            "user_annotations": to_jsonable(self.user_annotations),
            "runtime": runtime_provenance(),
        }

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        """Evaluate one closed-shell CCSD(T) energy point."""

        if request.electronic_state != "ground":
            raise ValueError("CCSD(T) provider supports only electronic_state='ground'")
        if request.spin != 0:
            raise ValueError("CCSD(T) provider currently supports closed-shell spin=0 only")
        if request.requested_properties != ("energy",):
            raise ValueError("CCSD(T) provider returns energy only; use finite fields for dipoles")

        started = time.perf_counter()
        actual_threads = _configure_pyscf_threads(self.threads)
        _, scf, _, elements = _require_pyscf()
        from pyscf import cc

        symbols = [elements._symbol(charge) for charge in request.nuclear_charges]
        molecule = Molecule.from_arrays(
            symbols,
            request.coordinates_A,
            charge=request.charge,
            spin=request.spin,
            label="correlated-electronic-point",
        )
        pmol = molecule_to_pyscf(molecule, self.settings.basis)
        pmol.max_memory = self.max_memory_mb
        if pmol.nelectron % 2:
            raise ValueError("RHF-CCSD(T) requires an even electron count")

        mean_field = scf.RHF(pmol)
        mean_field.conv_tol = self.settings.scf_conv_tol
        mean_field.max_cycle = self.settings.scf_max_cycle
        field_vector = np.zeros(3) if request.field_au is None else request.field_au
        field_origin = np.zeros(3) if request.field_origin_A is None else request.field_origin_A
        nuclear_field_energy = _apply_electric_field(
            mean_field,
            pmol,
            field_vector,
            field_origin,
        )
        mean_field.kernel()
        if not mean_field.converged or not np.isfinite(mean_field.e_tot):
            raise RuntimeError("RHF did not converge to a finite energy")

        if not self.settings.density_fit:
            _configure_conventional_integrals(
                mean_field,
                pmol,
                policy=self.settings.integral_policy,
                max_memory_mb=self.max_memory_mb,
            )

        frozen = self.settings.frozen_orbitals or None
        coupled_cluster = cc.CCSD(mean_field, frozen=frozen)
        if self.settings.density_fit:
            coupled_cluster = coupled_cluster.density_fit(auxbasis=self.settings.auxiliary_basis)
        coupled_cluster.conv_tol = self.settings.cc_conv_tol
        coupled_cluster.max_cycle = self.settings.cc_max_cycle
        coupled_cluster.direct = self.settings.direct
        coupled_cluster.max_memory = self.max_memory_mb
        if self.settings.integral_policy == "incore":
            coupled_cluster.incore_complete = True

        eris = coupled_cluster.ao2mo()
        actual_integral_path = _actual_integral_path(coupled_cluster, eris)
        if self.settings.integral_policy == "incore" and actual_integral_path != "incore":
            raise RuntimeError("PySCF did not honor the requested in-core integral policy")
        if self.settings.integral_policy == "outcore" and not actual_integral_path.startswith(
            "outcore"
        ):
            raise RuntimeError("PySCF did not honor the requested out-of-core integral policy")

        correlation_ccsd, t1, t2 = coupled_cluster.kernel(eris=eris)
        if not coupled_cluster.converged or not np.isfinite(correlation_ccsd):
            raise RuntimeError("CCSD did not converge to a finite correlation energy")
        triples = float(coupled_cluster.ccsd_t(t1=t1, t2=t2, eris=eris))
        if not np.isfinite(triples):
            raise RuntimeError("The perturbative triples correction is not finite")

        t1_diagnostic = float(coupled_cluster.get_t1_diagnostic(t1))
        d1_diagnostic = (
            float(coupled_cluster.get_d1_diagnostic(t1))
            if "d1" in self.settings.requested_diagnostics
            else None
        )
        singular_values = np.linalg.svd(t1, compute_uv=False)
        total_energy = (
            float(mean_field.e_tot) + float(correlation_ccsd) + triples + nuclear_field_energy
        )
        if not np.isfinite(total_energy):
            raise RuntimeError("CCSD(T) total energy is not finite")

        provider_id = provider_scientific_fingerprint(self)
        effective_method = "RHF-DF-CCSD(T)" if self.settings.density_fit else "RHF-CCSD(T)"
        return ElectronicResult(
            total_energy_Eh=total_energy,
            dipole_au=None,
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
            scientific_diagnostics={
                "effective_method": effective_method,
                "reference": "RHF",
                "scf_converged": bool(mean_field.converged),
                "ccsd_converged": bool(coupled_cluster.converged),
                "triples_completed": True,
                "correlation_energy_ccsd_Eh": float(correlation_ccsd),
                "correlation_energy_triples_Eh": triples,
                "t1_diagnostic": t1_diagnostic,
                "t1_diagnostic_definition": "PySCF Lee-Taylor: norm(t1)/sqrt(2*nocc)",
                "t1_frobenius_norm": float(np.linalg.norm(t1)),
                "t1_nocc_spatial": int(t1.shape[0]),
                "t1_largest_singular_value": (
                    None if singular_values.size == 0 else float(singular_values[0])
                ),
                "d1_diagnostic": d1_diagnostic,
                "requested_diagnostics": list(self.settings.requested_diagnostics),
                "density_fitting": self.settings.density_fit,
                "scf_density_fitting": False,
                "effective_auxiliary_basis": self.settings.auxiliary_basis,
                "frozen_orbitals": list(self.settings.frozen_orbitals),
                "integral_policy": self.settings.integral_policy,
                "direct": self.settings.direct,
                "field_au": np.asarray(field_vector, dtype=float).tolist(),
                "field_origin_A": np.asarray(field_origin, dtype=float).tolist(),
                "field_hamiltonian": _FIELD_CAPABILITY["hamiltonian"],
                "nuclear_field_energy_Eh": nuclear_field_energy,
                "nuclear_field_term_included": True,
            },
            execution_diagnostics={
                "reference_class": type(mean_field).__name__,
                "ccsd_implementation": _qualified_type_name(coupled_cluster),
                "triples_implementation": _qualified_callable_name(coupled_cluster.ccsd_t),
                "actual_integral_path": actual_integral_path,
                "scf_cycles": _optional_int(getattr(mean_field, "cycles", None)),
                "ccsd_cycles": _optional_int(getattr(coupled_cluster, "cycles", None)),
                "runtime_seconds": time.perf_counter() - started,
                "max_rss_mb": _max_rss_mb(),
                "threads_actual": actual_threads,
                "max_memory_mb": self.max_memory_mb,
                "warnings": [],
            },
            provenance=self.execution_provenance(),
        )


@dataclass(frozen=True)
class FiniteFieldDipoleProvider:
    """Central finite-field dipoles assembled from a compatible energy provider."""

    energy_provider: ElectronicProvider
    field_magnitudes_au: tuple[float, ...] = (1e-4, 5e-5)
    sign_convention: str = _FINITE_FIELD_SIGN_CONVENTION

    def __post_init__(self) -> None:
        if not isinstance(self.energy_provider, ElectronicProvider):
            raise TypeError("energy_provider must implement ElectronicProvider")
        _validated_field_energy_capability(self.energy_provider)
        magnitudes = tuple(float(value) for value in self.field_magnitudes_au)
        if (
            len(magnitudes) < 2
            or any(not np.isfinite(value) or value <= 0.0 for value in magnitudes)
            or len(set(magnitudes)) != len(magnitudes)
        ):
            raise ValueError("At least two unique positive field magnitudes are required")
        magnitudes = tuple(sorted(magnitudes, reverse=True))
        design = _field_fit_design(magnitudes)
        condition_number = float(np.linalg.cond(design))
        if not np.isfinite(condition_number) or condition_number > 1e8:
            raise ValueError(
                "field_magnitudes_au are too closely spaced for stable F^2 extrapolation"
            )
        convention = _nonempty("sign_convention", self.sign_convention)
        if convention != _FINITE_FIELD_SIGN_CONVENTION:
            raise ValueError("The finite-field sign convention is frozen")
        object.__setattr__(self, "field_magnitudes_au", magnitudes)
        object.__setattr__(self, "sign_convention", convention)

    def scientific_settings_payload(self) -> Mapping[str, object]:
        """Return finite-field settings and the underlying energy model identity."""

        return {
            "schema": "pyscf-vscf-provider-settings",
            "schema_version": 1,
            "provider": "central-finite-field-dipole",
            "energy_provider_scientific_fingerprint": provider_scientific_fingerprint(
                self.energy_provider
            ),
            "field_magnitudes_au": list(self.field_magnitudes_au),
            "stencil": "two-point-central-per-axis-and-field-magnitude",
            "extrapolation": "linear-intercept-versus-field-squared",
            "sign_convention": self.sign_convention,
            "charged_system_policy": "reject",
            "field_origin_policy": "explicit-request-origin-default-zero-angstrom",
            "field_convention": dict(_FIELD_CAPABILITY),
        }

    def execution_provenance(self) -> Mapping[str, object]:
        """Return execution provenance inherited from the energy provider."""

        return {
            "energy_provider": to_jsonable(self.energy_provider.execution_provenance()),
            "runtime": runtime_provenance(),
        }

    def subrequests(self, request: ElectronicPointRequest) -> tuple[ElectronicPointRequest, ...]:
        """Return the exact zero and signed-field energy point identities."""

        self._validate_request(request)
        requests = [
            replace(
                request,
                requested_properties=("energy",),
                field_au=np.zeros(3),
            )
        ]
        for magnitude in self.field_magnitudes_au:
            for axis in range(3):
                for sign in (-1.0, 1.0):
                    vector = np.zeros(3)
                    vector[axis] = sign * magnitude
                    requests.append(
                        replace(
                            request,
                            requested_properties=("energy",),
                            field_au=vector,
                        )
                    )
        return tuple(requests)

    def assemble(
        self,
        request: ElectronicPointRequest,
        results: Mapping[str, ElectronicResult],
    ) -> ElectronicResult:
        """Assemble a dipole from a complete causal-fingerprint-keyed batch."""

        subrequests = self.subrequests(request)
        energy_provider_id = provider_scientific_fingerprint(self.energy_provider)
        expected = {item.causal_fingerprint(energy_provider_id): item for item in subrequests}
        if set(results) != set(expected):
            missing = sorted(set(expected) - set(results))
            extra = sorted(set(results) - set(expected))
            raise ValueError(f"Finite-field result set mismatch; missing={missing}, extra={extra}")
        for fingerprint, result in results.items():
            if result.provider_scientific_fingerprint != energy_provider_id:
                raise ValueError("A field result came from the wrong energy provider")
            if result.point_causal_fingerprint != fingerprint or not result.converged:
                raise ValueError("A field result is incompatible or unconverged")
            _validate_field_result(result, expected[fingerprint])

        zero_fingerprint = subrequests[0].causal_fingerprint(energy_provider_id)
        dipoles_by_step = np.empty((len(self.field_magnitudes_au), 3), dtype=float)
        field_points: list[dict[str, object]] = []
        cursor = 1
        for step_index, magnitude in enumerate(self.field_magnitudes_au):
            for axis in range(3):
                negative = subrequests[cursor]
                positive = subrequests[cursor + 1]
                cursor += 2
                negative_fp = negative.causal_fingerprint(energy_provider_id)
                positive_fp = positive.causal_fingerprint(energy_provider_id)
                dipoles_by_step[step_index, axis] = -(
                    results[positive_fp].total_energy_Eh - results[negative_fp].total_energy_Eh
                ) / (2.0 * magnitude)
                field_points.append(
                    {
                        "field_magnitude_au": magnitude,
                        "axis": axis,
                        "negative_point_causal_fingerprint": negative_fp,
                        "positive_point_causal_fingerprint": positive_fp,
                    }
                )

        design = _field_fit_design(self.field_magnitudes_au)
        coefficients = np.linalg.lstsq(design, dipoles_by_step, rcond=None)[0]
        dipole = coefficients[1]
        fitted = design @ coefficients
        provider_id = provider_scientific_fingerprint(self)
        zero_result = results[zero_fingerprint]
        runtime_values = [
            _optional_float(result.execution_diagnostics.get("runtime_seconds"))
            for result in results.values()
        ]
        rss_values = [
            _optional_float(result.execution_diagnostics.get("max_rss_mb"))
            for result in results.values()
        ]
        origin = np.zeros(3) if request.field_origin_A is None else request.field_origin_A
        return ElectronicResult(
            total_energy_Eh=zero_result.total_energy_Eh,
            dipole_au=dipole,
            dipole_unit="atomic_unit",
            dipole_frame="input_cartesian",
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
            scientific_diagnostics={
                "finite_field_completed": True,
                "field_magnitudes_au": list(self.field_magnitudes_au),
                "dipoles_by_step_au": dipoles_by_step.tolist(),
                "extrapolated_dipole_au": dipole.tolist(),
                "maximum_step_fit_residual_au": np.max(
                    np.abs(dipoles_by_step - fitted), axis=0
                ).tolist(),
                "maximum_step_spread_au": np.ptp(dipoles_by_step, axis=0).tolist(),
                "field_fit_condition_number": float(np.linalg.cond(design)),
                "zero_field_point_causal_fingerprint": zero_fingerprint,
                "field_points": field_points,
                "sign_convention": self.sign_convention,
                "field_origin_A": np.asarray(origin, dtype=float).tolist(),
                "field_convention": dict(_FIELD_CAPABILITY),
            },
            execution_diagnostics={
                "constituent_point_count": len(results),
                "runtime_seconds": sum(value for value in runtime_values if value is not None),
                "runtime_semantics": "sum-of-constituent-point-runtimes",
                "max_rss_mb": max(
                    (value for value in rss_values if value is not None),
                    default=None,
                ),
                "max_rss_semantics": "maximum-constituent-process-peak-rss",
                "warnings": [],
            },
            provenance=self.execution_provenance(),
        )

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        """Run the finite-field batch serially and assemble its dipole."""

        energy_provider_id = provider_scientific_fingerprint(self.energy_provider)
        results = {}
        for subrequest in self.subrequests(request):
            result = self.energy_provider.evaluate(subrequest)
            fingerprint = subrequest.causal_fingerprint(energy_provider_id)
            if fingerprint in results:
                raise RuntimeError("Finite-field subrequests produced a duplicate identity")
            results[fingerprint] = result
        return self.assemble(request, results)

    @staticmethod
    def _validate_request(request: ElectronicPointRequest) -> None:
        if request.charge != 0:
            raise ValueError("Finite-field dipoles currently reject charged systems")
        if request.field_au is not None and np.any(request.field_au != 0.0):
            raise ValueError("The outer finite-field request must have zero field")
        if "dipole" not in request.requested_properties:
            raise ValueError("Finite-field requests must include the dipole property")
        if request.electronic_state != "ground":
            raise ValueError("Finite-field provider supports only electronic_state='ground'")


def _configure_conventional_integrals(
    mean_field: Any,
    molecule: Any,
    *,
    policy: str,
    max_memory_mb: int,
) -> None:
    if policy == "incore":
        nao = int(molecule.nao_nr())
        nao_pair = nao * (nao + 1) // 2
        estimated_mb = nao_pair * (nao_pair + 1) // 2 * 8.0 / 1e6
        if estimated_mb > max_memory_mb:
            raise MemoryError("The requested in-core AO integral tensor exceeds max_memory_mb")
        mean_field._eri = molecule.intor("int2e", aosym="s8")
    elif policy == "outcore":
        mean_field._eri = None


def _actual_integral_path(coupled_cluster: object, eris: object) -> str:
    if type(coupled_cluster).__module__ == "pyscf.cc.dfccsd":
        return "density-fitted"
    if hasattr(eris, "feri1"):
        return (
            "outcore-ao-direct" if bool(getattr(coupled_cluster, "direct", False)) else "outcore"
        )
    return "incore"


def _validated_field_energy_capability(provider: ElectronicProvider) -> dict[str, str]:
    payload = dict(provider.scientific_settings_payload())
    capability = payload.get("field_convention")
    if not isinstance(capability, Mapping) or dict(capability) != _FIELD_CAPABILITY:
        raise ValueError(
            "The energy provider must declare the exact field Hamiltonian, units, "
            "origin units, and nuclear-term contract"
        )
    return dict(_FIELD_CAPABILITY)


def _validate_field_result(
    result: ElectronicResult,
    request: ElectronicPointRequest,
) -> None:
    diagnostics = result.scientific_diagnostics
    expected_field = np.zeros(3) if request.field_au is None else request.field_au
    expected_origin = np.zeros(3) if request.field_origin_A is None else request.field_origin_A
    try:
        actual_field = np.asarray(diagnostics["field_au"], dtype=float)
        actual_origin = np.asarray(diagnostics["field_origin_A"], dtype=float)
        nuclear_field_energy = float(diagnostics["nuclear_field_energy_Eh"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("A field result does not prove the required field convention") from exc
    if (
        diagnostics.get("field_hamiltonian") != _FIELD_CAPABILITY["hamiltonian"]
        or diagnostics.get("nuclear_field_term_included") is not True
        or actual_field.shape != (3,)
        or actual_origin.shape != (3,)
        or not np.isfinite(nuclear_field_energy)
        or not np.array_equal(actual_field, expected_field)
        or not np.array_equal(actual_origin, expected_origin)
    ):
        raise ValueError("A field result does not prove the required field convention")


def _field_fit_design(magnitudes: tuple[float, ...]) -> np.ndarray:
    squared = np.square(np.asarray(magnitudes, dtype=float))
    scaled_squared = squared / float(np.max(squared))
    return np.column_stack((scaled_squared, np.ones_like(scaled_squared)))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if np.isfinite(parsed) else None


def _qualified_type_name(value: object) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _qualified_callable_name(value: object) -> str:
    function = getattr(value, "__func__", value)
    return f"{function.__module__}.{function.__qualname__}"


def _nonempty(name: str, value: object) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name} must be non-empty")
    return text


__all__ = [
    "CCSDPerturbativeTriplesSettings",
    "FiniteFieldDipoleProvider",
    "PySCFCCSDPerturbativeTriplesProvider",
]
