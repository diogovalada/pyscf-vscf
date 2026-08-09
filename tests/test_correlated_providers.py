from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from pyscf_vscf.backends.pyscf import PySCFMeanFieldProvider
from pyscf_vscf.backends.pyscf_correlated import (
    CCSDPerturbativeTriplesSettings,
    FiniteFieldDipoleProvider,
    PySCFCCSDPerturbativeTriplesProvider,
)
from pyscf_vscf.constants import ANG_TO_BOHR
from pyscf_vscf.electronic import (
    ElectronicPointRequest,
    ElectronicResult,
    provider_scientific_fingerprint,
)
from pyscf_vscf.settings import ESSettings


@dataclass
class _PolynomialFieldProvider:
    dipole_au: tuple[float, float, float] = (0.7, -0.4, 0.2)
    cubic_coefficients: tuple[float, float, float] = (2.0, -1.5, 0.8)
    calls: int = 0

    def scientific_settings_payload(self) -> dict[str, object]:
        return {
            "provider": "analytic-polynomial-field-energy",
            "dipole_au": list(self.dipole_au),
            "cubic_coefficients": list(self.cubic_coefficients),
            "field_convention": {
                "field_units": "atomic_unit",
                "origin_units": "angstrom",
                "hamiltonian": "H(F)=H(0)-F.mu",
                "nuclear_term": "included-in-total-energy",
            },
        }

    def execution_provenance(self) -> dict[str, object]:
        return {"provider": "analytic-test-double"}

    def evaluate(self, request: ElectronicPointRequest) -> ElectronicResult:
        self.calls += 1
        field = np.zeros(3) if request.field_au is None else request.field_au
        origin = np.zeros(3) if request.field_origin_A is None else request.field_origin_A
        dipole = np.asarray(self.dipole_au)
        cubic = np.asarray(self.cubic_coefficients)
        energy = -10.0 - float(dipole @ field) - float(cubic @ np.power(field, 3))
        provider_id = provider_scientific_fingerprint(self)
        return ElectronicResult(
            total_energy_Eh=energy,
            dipole_au=None,
            converged=True,
            point_causal_fingerprint=request.causal_fingerprint(provider_id),
            provider_scientific_fingerprint=provider_id,
            scientific_diagnostics={
                "field_au": np.asarray(field).tolist(),
                "field_origin_A": np.asarray(origin).tolist(),
                "field_hamiltonian": "H(F)=H(0)-F.mu",
                "nuclear_field_energy_Eh": 0.0,
                "nuclear_field_term_included": True,
            },
            execution_diagnostics={"runtime_seconds": 0.01, "max_rss_mb": 1.0},
            provenance=self.execution_provenance(),
        )


def _water_request(*, origin_A: np.ndarray | None = None) -> ElectronicPointRequest:
    return ElectronicPointRequest(
        nuclear_charges=(8, 1, 1),
        coordinates_A=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.9572],
                [0.9266, 0.0, -0.2396],
            ]
        ),
        requested_properties=("energy", "dipole"),
        field_au=None if origin_A is None else np.zeros(3),
        field_origin_A=origin_A,
    )


def _h2_request(*, field_au: np.ndarray | None = None) -> ElectronicPointRequest:
    return ElectronicPointRequest(
        nuclear_charges=(1, 1),
        coordinates_A=np.array([[0.0, 0.0, -0.37], [0.0, 0.0, 0.37]]),
        requested_properties=("energy",),
        field_au=field_au,
    )


def test_finite_field_extrapolation_distinct_identities_and_origin() -> None:
    energy_provider = _PolynomialFieldProvider()
    finite_field = FiniteFieldDipoleProvider(
        energy_provider,
        field_magnitudes_au=(2e-3, 1e-3),
    )
    request = _water_request(origin_A=np.array([0.5, -0.2, 0.3]))
    energy_provider_id = provider_scientific_fingerprint(energy_provider)
    subrequests = finite_field.subrequests(request)
    result = finite_field.evaluate(request)

    assert len(subrequests) == 13
    assert len({item.causal_fingerprint(energy_provider_id) for item in subrequests}) == len(
        subrequests
    )
    np.testing.assert_allclose(result.dipole_au, energy_provider.dipole_au, atol=2e-10)
    assert result.total_energy_Eh == pytest.approx(-10.0)
    assert result.dipole_unit == "atomic_unit"
    assert result.dipole_frame == "input_cartesian"
    assert result.scientific_diagnostics["finite_field_completed"]
    assert len(result.scientific_diagnostics["dipoles_by_step_au"]) == 2
    assert len(result.scientific_diagnostics["field_points"]) == 6
    np.testing.assert_array_equal(
        result.scientific_diagnostics["field_origin_A"],
        request.field_origin_A,
    )
    assert energy_provider.calls == 13


def test_finite_field_assembly_fails_closed_on_missing_or_tampered_results() -> None:
    provider = _PolynomialFieldProvider()
    finite = FiniteFieldDipoleProvider(provider, field_magnitudes_au=(2e-3, 1e-3))
    request = _water_request()
    provider_id = provider_scientific_fingerprint(provider)
    results = {}
    for subrequest in finite.subrequests(request):
        fingerprint = subrequest.causal_fingerprint(provider_id)
        results[fingerprint] = provider.evaluate(subrequest)

    missing = dict(results)
    missing.pop(next(iter(missing)))
    with pytest.raises(ValueError, match="result set mismatch"):
        finite.assemble(request, missing)

    tampered = dict(results)
    field_fingerprint = finite.subrequests(request)[1].causal_fingerprint(provider_id)
    diagnostics = dict(tampered[field_fingerprint].scientific_diagnostics)
    diagnostics["nuclear_field_term_included"] = False
    tampered[field_fingerprint] = replace(
        tampered[field_fingerprint],
        scientific_diagnostics=diagnostics,
    )
    with pytest.raises(ValueError, match="does not prove"):
        finite.assemble(request, tampered)


@pytest.mark.parametrize(
    "defect",
    [
        "provider_identity",
        "point_identity",
        "property_identity",
        "unconverged",
        "field_vector",
        "field_origin",
    ],
)
def test_finite_field_assembly_rejects_incompatible_field_results(defect: str) -> None:
    provider = _PolynomialFieldProvider()
    finite = FiniteFieldDipoleProvider(provider, field_magnitudes_au=(2e-3, 1e-3))
    request = _water_request()
    provider_id = provider_scientific_fingerprint(provider)
    subrequests = finite.subrequests(request)
    results = {
        subrequest.causal_fingerprint(provider_id): provider.evaluate(subrequest)
        for subrequest in subrequests
    }
    target_request = subrequests[1]
    target_fingerprint = target_request.causal_fingerprint(provider_id)
    target = results[target_fingerprint]

    if defect == "provider_identity":
        replacement = replace(target, provider_scientific_fingerprint="wrong-provider")
    elif defect == "point_identity":
        replacement = replace(target, point_causal_fingerprint="wrong-point")
    elif defect == "property_identity":
        wrong_property_request = replace(
            target_request,
            requested_properties=("energy", "dipole"),
        )
        replacement = replace(
            target,
            point_causal_fingerprint=wrong_property_request.causal_fingerprint(provider_id),
        )
    elif defect == "unconverged":
        replacement = replace(target, converged=False)
    else:
        diagnostics = dict(target.scientific_diagnostics)
        diagnostics["field_au" if defect == "field_vector" else "field_origin_A"] = [
            9.0,
            8.0,
            7.0,
        ]
        replacement = replace(target, scientific_diagnostics=diagnostics)
    results[target_fingerprint] = replacement

    with pytest.raises(ValueError, match="wrong energy provider|incompatible|does not prove"):
        finite.assemble(request, results)


def test_finite_field_contract_and_outer_request_rejections() -> None:
    provider = _PolynomialFieldProvider()
    provider.scientific_settings_payload = lambda: {"provider": "missing-field-contract"}
    with pytest.raises(ValueError, match="must declare"):
        FiniteFieldDipoleProvider(provider)

    finite = FiniteFieldDipoleProvider(_PolynomialFieldProvider())
    with pytest.raises(ValueError, match="reject charged"):
        finite.evaluate(replace(_water_request(), charge=1))
    with pytest.raises(ValueError, match="must have zero field"):
        finite.evaluate(replace(_water_request(), field_au=np.array([1e-4, 0.0, 0.0])))
    with pytest.raises(ValueError, match="include the dipole"):
        finite.evaluate(replace(_water_request(), requested_properties=("energy",)))


def test_finite_field_rejects_ill_conditioned_steps_and_changed_convention() -> None:
    with pytest.raises(ValueError, match="too closely spaced"):
        FiniteFieldDipoleProvider(
            _PolynomialFieldProvider(),
            field_magnitudes_au=(1e-4, 1.000000001e-4),
        )
    with pytest.raises(ValueError, match="sign convention is frozen"):
        FiniteFieldDipoleProvider(
            _PolynomialFieldProvider(),
            sign_convention="mu=+dE/dF",
        )


def test_correlated_provider_identity_separates_science_from_execution() -> None:
    baseline = CCSDPerturbativeTriplesSettings(basis="sto-3g")
    provider = PySCFCCSDPerturbativeTriplesProvider(baseline)
    expected = provider_scientific_fingerprint(provider)
    changes = {
        "basis": "6-31g",
        "frozen_orbitals": (0,),
        "scf_conv_tol": 1e-9,
        "scf_max_cycle": 81,
        "cc_conv_tol": 1e-8,
        "cc_max_cycle": 81,
        "direct": True,
        "integral_policy": "incore",
        "requested_diagnostics": ("t1",),
    }
    for name, value in changes.items():
        changed = PySCFCCSDPerturbativeTriplesProvider(replace(baseline, **{name: value}))
        assert provider_scientific_fingerprint(changed) != expected, name

    density_fitted = PySCFCCSDPerturbativeTriplesProvider(
        CCSDPerturbativeTriplesSettings(
            basis="sto-3g",
            density_fit=True,
            auxiliary_basis="weigend",
        )
    )
    assert provider_scientific_fingerprint(density_fitted) != expected
    changed_resources = PySCFCCSDPerturbativeTriplesProvider(
        baseline,
        threads=4,
        max_memory_mb=2048,
        user_annotations={"scratch": "/different"},
    )
    assert provider_scientific_fingerprint(changed_resources) == expected
    assert changed_resources.execution_provenance() != provider.execution_provenance()

    finite = FiniteFieldDipoleProvider(provider)
    changed_steps = FiniteFieldDipoleProvider(
        provider,
        field_magnitudes_au=(2e-4, 1e-4),
    )
    assert provider_scientific_fingerprint(finite) != provider_scientific_fingerprint(
        changed_steps
    )


def test_correlated_settings_reject_incompatible_policies() -> None:
    with pytest.raises(ValueError, match="direct=False"):
        CCSDPerturbativeTriplesSettings(
            basis="sto-3g",
            density_fit=True,
            auxiliary_basis="weigend",
            direct=True,
        )
    with pytest.raises(ValueError, match="include t1"):
        CCSDPerturbativeTriplesSettings(
            basis="sto-3g",
            requested_diagnostics=("d1",),
        )
    with pytest.raises(ValueError, match="ordered unique"):
        CCSDPerturbativeTriplesSettings(
            basis="sto-3g",
            frozen_orbitals=(1, 0),
        )


@pytest.mark.pyscf
@pytest.mark.parametrize(
    ("method", "tolerance"),
    [("hf", 3e-7), ("lda,vwn", 8e-7)],
)
def test_mean_field_finite_field_matches_analytic_water_dipole(
    method: str,
    tolerance: float,
) -> None:
    provider = PySCFMeanFieldProvider(
        ESSettings(
            method=method,
            basis="sto-3g",
            use_density_fit=False,
            scf_conv_tol=1e-12,
            scf_max_cycle=100,
            dft_grid_level=3,
        ),
        threads=1,
    )
    request = _water_request()
    analytic = provider.evaluate(request)
    finite = FiniteFieldDipoleProvider(
        provider,
        field_magnitudes_au=(2e-4, 1e-4),
    ).evaluate(request)

    np.testing.assert_allclose(finite.dipole_au, analytic.dipole_au, atol=tolerance)


@pytest.mark.pyscf
def test_mean_field_field_energy_records_nuclear_term() -> None:
    provider = PySCFMeanFieldProvider(
        ESSettings(method="hf", basis="sto-3g", use_density_fit=False),
        threads=1,
    )
    field = np.array([1e-4, -2e-4, 0.5e-4])
    origin = np.array([0.3, -0.2, 0.1])
    request = replace(
        _water_request(),
        requested_properties=("energy",),
        field_au=field,
        field_origin_A=origin,
    )
    result = provider.evaluate(request)
    nuclear_dipole = np.einsum(
        "i,ix->x",
        np.asarray(request.nuclear_charges),
        (request.coordinates_A - origin) * ANG_TO_BOHR,
    )

    assert result.scientific_diagnostics["nuclear_field_energy_Eh"] == pytest.approx(
        -float(field @ nuclear_dipole),
        abs=1e-14,
    )
    assert result.scientific_diagnostics["nuclear_field_term_included"] is True


@pytest.mark.pyscf
def test_h2_ccsd_perturbative_triples_smoke() -> None:
    result = PySCFCCSDPerturbativeTriplesProvider(
        CCSDPerturbativeTriplesSettings(
            basis="sto-3g",
            scf_conv_tol=1e-11,
            cc_conv_tol=1e-10,
        ),
        threads=1,
    ).evaluate(_h2_request())

    diagnostics = result.scientific_diagnostics
    assert result.converged
    assert diagnostics["effective_method"] == "RHF-CCSD(T)"
    assert diagnostics["triples_completed"] is True
    assert diagnostics["t1_diagnostic"] >= 0.0
    assert diagnostics["d1_diagnostic"] >= 0.0
    assert result.execution_diagnostics["actual_integral_path"] in {"incore", "outcore"}
    assert result.execution_diagnostics["runtime_seconds"] >= 0.0
