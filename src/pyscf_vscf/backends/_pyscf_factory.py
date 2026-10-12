"""The single construction seam for PySCF mean-field calculations.

Every package workflow builds its PySCF molecule and configured, unconverged
mean-field object through ``build_mean_field``. A *factory* decides the
electronic-structure method; the seam validates the electronic system,
applies execution resources, records a canonical snapshot of the configured
object, and returns an immutable ``MeanFieldBuild`` envelope.
Convergence and property extraction live in
``pyscf_vscf.backends._pyscf_properties``.

Two factories exist:

- ``SettingsFactory`` reproduces the built-in method defined by
  ``ESSettings``.
- ``MethodRecipe`` wraps a user-supplied PySCF method (any functional,
  dispersion, solvent model, ECP, relativistic Hamiltonian, ...). Its
  scientific identity is a required, user-declared manifest; the snapshot of
  readable attributes is provenance and a consistency guard, never identity.
"""

from __future__ import annotations

import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .._arrays import immutable_array
from . import _gpu
from .._identity import FrozenJSONMapping, immutable_json_mapping, payload_fingerprint
from ..settings import (
    ESSettings,
    coerce_es_settings,
    default_auxbasis,
    resolve_scf_controls,
    runtime_max_memory_mb,
)

MANIFEST_SCHEMA = "pyscf-vscf-method-recipe"
MANIFEST_SCHEMA_VERSION = 1
SNAPSHOT_SCHEMA_VERSION = 1


class BackendUnavailableError(ImportError):
    """Raised when the required PySCF backend dependency is unavailable."""


@dataclass(frozen=True)
class ElectronicSystem:
    """Ordered nuclei, Cartesian geometry (angstrom), charge, spin, and state.

    This is everything the electronic Hamiltonian depends on. Isotope masses
    are deliberately absent: they belong to the vibrational analysis.
    """

    nuclear_charges: tuple[int, ...]
    coordinates_A: np.ndarray
    charge: int = 0
    spin: int = 0
    electronic_state: str = "ground"

    def __post_init__(self) -> None:
        charges = tuple(operator.index(value) for value in self.nuclear_charges)
        if not charges or any(value <= 0 for value in charges):
            raise ValueError("nuclear_charges must contain positive atomic numbers")
        coordinates = np.asarray(self.coordinates_A, dtype=float)
        if coordinates.shape != (len(charges), 3) or not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates_A must be a finite (n_atoms, 3) array")
        state = str(self.electronic_state).strip()
        if state != "ground":
            raise ValueError("Only electronic_state='ground' is supported")
        object.__setattr__(self, "nuclear_charges", charges)
        object.__setattr__(self, "coordinates_A", immutable_array(coordinates, dtype="<f8"))
        object.__setattr__(self, "charge", operator.index(self.charge))
        object.__setattr__(self, "spin", operator.index(self.spin))
        object.__setattr__(self, "electronic_state", state)

    @classmethod
    def from_request(cls, request: Any) -> ElectronicSystem:
        """Return the electronic system of an ``ElectronicPointRequest``."""

        return cls(
            nuclear_charges=request.nuclear_charges,
            coordinates_A=request.coordinates_A,
            charge=request.charge,
            spin=request.spin,
            electronic_state=request.electronic_state,
        )

    @classmethod
    def from_molecule(cls, molecule: Any) -> ElectronicSystem:
        """Return the electronic system of a package ``Molecule`` (D counts as H)."""

        _, _, _, elements = require_pyscf()
        symbols = [
            "H" if str(symbol).upper() == "D" else str(symbol) for symbol in molecule.symbols
        ]
        return cls(
            nuclear_charges=tuple(int(elements.charge(symbol)) for symbol in symbols),
            coordinates_A=np.asarray(molecule.coords, dtype=float),
            charge=int(getattr(molecule, "charge", 0)),
            spin=int(getattr(molecule, "spin", 0)),
        )

    def atom_list(self) -> list[tuple[str, tuple[float, float, float]]]:
        """Return PySCF ``mol.atom`` entries with element symbols and angstrom positions."""

        _, _, _, elements = require_pyscf()
        atoms = []
        for charge, xyz in zip(self.nuclear_charges, self.coordinates_A):
            x, y, z = (float(value) for value in xyz)
            atoms.append((str(elements._symbol(charge)), (x, y, z)))
        return atoms


@dataclass(frozen=True)
class PySCFResources:
    """Execution resources applied to a build; never part of scientific identity.

    ``max_memory_mb=None`` uses ``PYSCF_VSCF_MAX_MEMORY_MB`` when set, otherwise
    PySCF's own default.
    """

    max_memory_mb: int | None = None
    verbose: int = 0

    def effective_max_memory_mb(self) -> int | None:
        """Return the memory budget to apply, or ``None`` to keep PySCF's default."""

        if self.max_memory_mb is not None:
            return int(self.max_memory_mb)
        return runtime_max_memory_mb()


@dataclass(frozen=True)
class MeanFieldBuild:
    """Immutable envelope around one configured, unconverged mean-field object.

    ``mean_field`` and ``molecule`` are live, mutable PySCF objects owned by the
    caller. ``method_identity`` is the factory's scientific payload and
    ``configuration_snapshot`` the canonical record of readable attributes.
    """

    mean_field: Any
    molecule: Any
    method_identity: FrozenJSONMapping
    configuration_snapshot: FrozenJSONMapping

    @property
    def method_fingerprint(self) -> str:
        """Fingerprint of the method identity."""

        return payload_fingerprint(dict(self.method_identity))


@runtime_checkable
class MeanFieldFactory(Protocol):
    """Builds the PySCF molecule and configured mean field for one system."""

    def scientific_settings_payload(self) -> Mapping[str, object]:
        """Return every setting that can change the computed energy or dipole."""
        ...

    def build_molecule(self, system: ElectronicSystem) -> Any:
        """Return a built ``gto.Mole`` for ``system`` (basis, ECP, charge, spin)."""
        ...

    def build_mean_field(self, molecule: Any) -> Any:
        """Return a configured, unconverged mean-field object for ``molecule``."""
        ...


@dataclass(frozen=True)
class SettingsFactory:
    """The built-in method: HF or a DFT functional defined by ``ESSettings``.

    Unset SCF controls resolve to the package defaults, and the auxiliary
    basis resolves to its default for the main basis. Restricted references
    are used for closed shells and unrestricted ones otherwise.
    ``settings.device`` selects CPU (PySCF) or GPU (GPU4PySCF) execution.
    """

    settings: ESSettings

    def __post_init__(self) -> None:
        settings = coerce_es_settings(self.settings)
        method = str(settings.method).strip()
        basis = str(settings.basis).strip()
        if not method or not basis:
            raise ValueError("Electronic method and basis must be non-empty")
        controls = resolve_scf_controls(settings)
        auxbasis = None
        if bool(settings.use_density_fit):
            auxbasis = str(settings.auxbasis or default_auxbasis(basis)).strip()
            if not auxbasis:
                raise ValueError("Density fitting requires a non-empty auxiliary basis")
        object.__setattr__(
            self,
            "settings",
            ESSettings(
                method=method,
                basis=basis,
                use_density_fit=bool(settings.use_density_fit),
                auxbasis=auxbasis,
                scf_conv_tol=controls.conv_tol,
                scf_max_cycle=controls.max_cycle,
                dft_grid_level=controls.dft_grid_level,
                device=_gpu.normalize_device(settings.device),
            ),
        )

    @property
    def device(self) -> str:
        """``"cpu"`` or ``"gpu"``."""

        return self.settings.device

    def scientific_settings_payload(self) -> Mapping[str, object]:
        """Return the resolved settings that define the built-in method.

        ``device`` appears only for GPU methods, so CPU identities are those of
        earlier releases.
        """

        payload: dict[str, object] = {
            "method": self.settings.method,
            "basis": self.settings.basis,
            "use_density_fit": self.settings.use_density_fit,
            "auxbasis": self.settings.auxbasis,
            "scf_conv_tol": self.settings.scf_conv_tol,
            "scf_max_cycle": self.settings.scf_max_cycle,
            "dft_grid_level": self.settings.dft_grid_level,
        }
        if self.settings.device != "cpu":
            payload["device"] = self.settings.device
        return payload

    def build_molecule(self, system: ElectronicSystem) -> Any:
        """Build a ``gto.Mole`` in angstrom with the configured basis."""

        gto, _, _, _ = require_pyscf()
        molecule = gto.Mole()
        molecule.unit = "Angstrom"
        molecule.basis = self.settings.basis
        molecule.charge = system.charge
        molecule.spin = system.spin
        molecule.verbose = 0
        molecule.atom = system.atom_list()
        molecule.build()
        return molecule

    def build_mean_field(self, molecule: Any) -> Any:
        """Configure RHF/UHF or RKS/UKS with density fitting, grid, and SCF controls."""

        return configure_settings_mean_field(molecule, self.settings)


def configure_settings_mean_field(molecule: Any, settings: ESSettings) -> Any:
    """Configure (without running) the built-in mean field for ``molecule``."""

    _, scf, dft, _ = require_pyscf()
    controls = resolve_scf_controls(settings)
    method = str(settings.method)
    open_shell = int(getattr(molecule, "spin", 0)) != 0
    is_dft = method.lower() != "hf"
    if not is_dft:
        mean_field = scf.UHF(molecule) if open_shell else scf.RHF(molecule)
    else:
        mean_field = dft.UKS(molecule, xc=method) if open_shell else dft.RKS(molecule, xc=method)

    if bool(settings.use_density_fit):
        auxbasis = (
            settings.auxbasis if settings.auxbasis else default_auxbasis(str(settings.basis))
        )
        try:
            mean_field = mean_field.density_fit(auxbasis=auxbasis)
        except Exception as exc:
            raise RuntimeError(
                f"RI density_fit(auxbasis='{auxbasis}') failed. Refusing an implicit "
                "auxiliary-basis substitution because it would change the recorded "
                f"electronic-structure model ({exc})"
            ) from exc

    if is_dft and controls.dft_grid_level is not None:
        mean_field.grids.level = controls.dft_grid_level
    mean_field.conv_tol = controls.conv_tol
    mean_field.max_cycle = controls.max_cycle
    return mean_field


@dataclass(frozen=True)
class MethodRecipe:
    """A user-supplied PySCF method with a declared, stable identity.

    ``mean_field`` receives a built ``gto.Mole`` and returns a configured,
    unconverged mean-field object; it must not call ``kernel()``. To run in
    worker processes it must be an importable module-level function (or a
    picklable callable object). ``molecule`` optionally replaces the default
    molecule builder, for example to attach ECPs or per-element bases; it
    receives an ``ElectronicSystem`` and returns a built ``gto.Mole``.

    The identity is the manifest: ``recipe_id`` and ``recipe_version`` name
    the method, ``configuration`` must list every setting that can change a
    result (functional, basis, grids, dispersion, solvent, external files,
    ...), and ``implementation`` names the electronic-structure implementation.
    Bump ``recipe_version`` whenever the method changes. ``implementation_digest``
    is an optional user-declared digest of the implementation.

    ``variable_snapshot_fields`` declares snapshot fields that may
    legitimately change with geometry; every other snapshot field must agree
    between builds.

    ``device="gpu"`` converts the configured mean field with ``to_gpu()``
    (GPU4PySCF); ``mean_field`` itself still returns a CPU object. The
    ``implementation`` then defaults to ``"gpu4pyscf"``.
    """

    recipe_id: str
    recipe_version: str
    configuration: Mapping[str, Any]
    mean_field: Callable[[Any], Any]
    basis: Any = None
    ecp: Any = None
    molecule: Callable[[ElectronicSystem], Any] | None = None
    implementation: str = "pyscf-cpu"
    implementation_digest: str | None = None
    variable_snapshot_fields: tuple[str, ...] = ()
    device: str = "cpu"

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", _gpu.normalize_device(self.device))
        if self.device == "gpu" and str(self.implementation).strip() == "pyscf-cpu":
            object.__setattr__(self, "implementation", "gpu4pyscf")
        for name in ("recipe_id", "recipe_version", "implementation"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must be a non-empty string")
            object.__setattr__(self, name, value)
        if not callable(self.mean_field):
            raise TypeError("mean_field must be callable")
        if self.molecule is None and self.basis is None:
            raise ValueError("Provide basis (and optionally ecp) or a molecule builder")
        if self.molecule is not None and not callable(self.molecule):
            raise TypeError("molecule must be callable when given")
        configuration = immutable_json_mapping(self.configuration)
        if not configuration:
            raise ValueError(
                "configuration must list every setting that defines the method; "
                "an empty configuration cannot identify a calculation"
            )
        object.__setattr__(self, "configuration", configuration)
        object.__setattr__(
            self,
            "variable_snapshot_fields",
            tuple(sorted(str(name) for name in self.variable_snapshot_fields)),
        )
        if self.device == "cpu" and self.implementation == "gpu4pyscf":
            raise ValueError("implementation='gpu4pyscf' needs device='gpu'")
        if self.implementation_digest is not None:
            digest = str(self.implementation_digest).strip()
            object.__setattr__(self, "implementation_digest", digest or None)

    def on_device(self, device: str) -> MethodRecipe:
        """Return this recipe for ``device`` (``"cpu"`` or ``"gpu"``).

        The default implementation names follow the device; a custom
        ``implementation`` is kept.
        """

        device = _gpu.normalize_device(device)
        if device == self.device:
            return self
        implementation = self.implementation
        if implementation in {"pyscf-cpu", "gpu4pyscf"}:
            implementation = "gpu4pyscf" if device == "gpu" else "pyscf-cpu"
        return replace(self, device=device, implementation=implementation)

    def scientific_settings_payload(self) -> Mapping[str, object]:
        """Return the manifest that identifies the recipe."""

        return {
            "schema": MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "recipe_id": self.recipe_id,
            "recipe_version": self.recipe_version,
            "implementation": self.implementation,
            "implementation_digest": self.implementation_digest,
            "configuration": dict(self.configuration),
            "basis": _canonical_basis(self.basis),
            "ecp": _canonical_basis(self.ecp),
            "custom_molecule_builder": None
            if self.molecule is None
            else _callable_name(self.molecule),
            "variable_snapshot_fields": list(self.variable_snapshot_fields),
            **({"device": self.device} if self.device != "cpu" else {}),
        }

    def build_molecule(self, system: ElectronicSystem) -> Any:
        """Build the molecule with the recipe's builder, or from ``basis``/``ecp``."""

        if self.molecule is not None:
            return self.molecule(system)
        gto, _, _, _ = require_pyscf()
        molecule = gto.Mole()
        molecule.unit = "Angstrom"
        molecule.basis = self.basis
        if self.ecp is not None:
            molecule.ecp = self.ecp
        molecule.charge = system.charge
        molecule.spin = system.spin
        molecule.verbose = 0
        molecule.atom = system.atom_list()
        molecule.build()
        return molecule

    def build_mean_field(self, molecule: Any) -> Any:
        """Call the recipe's ``mean_field`` function."""

        return self.mean_field(molecule)


def build_mean_field(
    system: ElectronicSystem,
    factory: MeanFieldFactory,
    *,
    resources: PySCFResources | None = None,
) -> MeanFieldBuild:
    """Build the molecule and configured mean field for ``system``.

    The molecule must reproduce the system's nuclei, geometry, charge, and
    spin; the mean field must be unconverged and bound to that molecule. For a
    factory with ``device == "gpu"`` the configured mean field is converted
    with ``to_gpu()``; a factory that returns a GPU object itself must declare
    that device.
    """

    policy = resources or PySCFResources()
    molecule = factory.build_molecule(system)
    _validate_molecule(molecule, system)
    budget = policy.effective_max_memory_mb()
    if budget is not None:
        molecule.max_memory = budget
    molecule.verbose = int(policy.verbose)
    mean_field = factory.build_mean_field(molecule)
    if getattr(mean_field, "mol", None) is not molecule:
        raise ValueError("The method factory must build the mean field on the supplied molecule")
    if getattr(mean_field, "e_tot", 0) not in (0, None) and getattr(
        mean_field, "converged", False
    ):
        raise ValueError("The method factory must not run the SCF (do not call kernel())")
    device = _gpu.normalize_device(getattr(factory, "device", "cpu"))
    if _gpu.is_gpu_object(mean_field):
        raise ValueError(
            "The method factory returned a GPU4PySCF object; return the CPU object and "
            "set device='gpu' instead, so the method's identity records the device"
        )
    if device == "gpu":
        mean_field = _gpu.to_gpu(mean_field)
    if budget is not None and hasattr(mean_field, "max_memory"):
        mean_field.max_memory = budget
    mean_field.verbose = int(policy.verbose)
    return MeanFieldBuild(
        mean_field=mean_field,
        molecule=molecule,
        method_identity=immutable_json_mapping(factory.scientific_settings_payload()),
        configuration_snapshot=immutable_json_mapping(configuration_snapshot(mean_field)),
    )


def configuration_snapshot(mean_field: Any) -> dict[str, object]:
    """Return a canonical record of readable mean-field configuration.

    Values that cannot be represented stably are recorded as ``"<opaque>"``;
    callables are recorded by module-qualified name, never by address.
    Geometry-dependent arrays are excluded.
    """

    molecule = getattr(mean_field, "mol", None)
    with_df = getattr(mean_field, "with_df", None)
    grids = getattr(mean_field, "grids", None)
    nlcgrids = getattr(mean_field, "nlcgrids", None)
    solvent = getattr(mean_field, "with_solvent", None)
    snapshot = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "class": _class_name(mean_field),
        "xc": _stable(getattr(mean_field, "xc", None)),
        "nlc": _stable(getattr(mean_field, "nlc", None)),
        "disp": _stable(getattr(mean_field, "disp", None)),
        "conv_tol": _stable(getattr(mean_field, "conv_tol", None)),
        "conv_tol_grad": _stable(getattr(mean_field, "conv_tol_grad", None)),
        "max_cycle": _stable(getattr(mean_field, "max_cycle", None)),
        "level_shift": _stable(getattr(mean_field, "level_shift", None)),
        "init_guess": _stable(getattr(mean_field, "init_guess", None)),
        "small_rho_cutoff": _stable(getattr(mean_field, "small_rho_cutoff", None)),
        "omega": _stable(getattr(mean_field, "omega", None)),
        "density_fit_auxbasis": None
        if with_df is None
        else _canonical_basis(getattr(with_df, "auxbasis", None)),
        "grids_level": None if grids is None else _stable(getattr(grids, "level", None)),
        "grids_prune": None if grids is None else _stable(getattr(grids, "prune", None)),
        "grids_atom_grid": None if grids is None else _stable(getattr(grids, "atom_grid", None)),
        "nlcgrids_level": None if nlcgrids is None else _stable(getattr(nlcgrids, "level", None)),
        "solvent": None
        if solvent is None
        else {
            "class": _class_name(solvent),
            "eps": _stable(getattr(solvent, "eps", None)),
            "method": _stable(getattr(solvent, "method", None)),
        },
        "basis": None if molecule is None else _canonical_basis(getattr(molecule, "basis", None)),
        "ecp": None if molecule is None else _canonical_basis(getattr(molecule, "ecp", None)),
        "cart": None if molecule is None else _stable(getattr(molecule, "cart", None)),
    }
    return snapshot


def invariant_snapshot(snapshot: Mapping[str, object], variable_fields: tuple[str, ...]) -> dict:
    """Return the snapshot without fields declared to vary with geometry."""

    return {key: value for key, value in snapshot.items() if key not in set(variable_fields)}


class ConfigurationGuard:
    """Require one invariant method configuration across the builds of a calculation.

    The first checked snapshot becomes the reference; later snapshots must
    agree on every field except the factory's ``variable_snapshot_fields``.
    """

    def __init__(self, factory: Any) -> None:
        self._variable_fields = tuple(getattr(factory, "variable_snapshot_fields", ()) or ())
        self._reference: dict[str, object] | None = None
        self.reference_snapshot: Mapping[str, object] | None = None

    def check(self, snapshot: Mapping[str, object] | None) -> None:
        """Raise ``ValueError`` if ``snapshot`` differs from the reference."""

        if snapshot is None:
            return
        invariant = invariant_snapshot(snapshot, self._variable_fields)
        if self._reference is None:
            self._reference = invariant
            self.reference_snapshot = snapshot
            return
        if invariant != self._reference:
            reference = self._reference
            changed = sorted(
                key
                for key in set(invariant) | set(reference)
                if invariant.get(key) != reference.get(key)
            )
            raise ValueError(
                "The method configuration differs between geometries "
                f"(fields: {', '.join(changed)}); declare legitimately geometry-dependent "
                "fields in the recipe's variable_snapshot_fields"
            )


def require_pyscf() -> tuple[Any, Any, Any, Any]:
    """Import PySCF lazily; raise ``BackendUnavailableError`` if it is missing."""

    try:
        from pyscf import dft, gto, scf
        from pyscf.data import elements
    except Exception as exc:
        raise BackendUnavailableError(
            "PySCF is unavailable. Reinstall pyscf-vscf and its required dependencies."
        ) from exc
    return gto, scf, dft, elements


def _validate_molecule(molecule: Any, system: ElectronicSystem) -> None:
    if not getattr(molecule, "_built", True):
        raise ValueError("The molecule builder must return a built gto.Mole")
    # atom_charge() excludes ECP core electrons; add them back to compare nuclei.
    nuclear = tuple(
        int(molecule.atom_charge(index) + molecule.atom_nelec_core(index))
        for index in range(molecule.natm)
    )
    if nuclear != system.nuclear_charges:
        raise ValueError("The molecule builder changed the nuclei or their order")
    coordinates = np.asarray(molecule.atom_coords(unit="Angstrom"), dtype=float)
    if coordinates.shape != system.coordinates_A.shape or not np.allclose(
        coordinates, system.coordinates_A, rtol=0.0, atol=1e-10
    ):
        raise ValueError("The molecule builder changed the geometry")
    if int(molecule.charge) != system.charge or int(molecule.spin) != system.spin:
        raise ValueError("The molecule builder changed the charge or spin")


def _class_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", None) or type(value).__module__
    name = getattr(value, "__qualname__", None) or type(value).__qualname__
    return f"{module}.{name}"


def _stable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if callable(value):
        return _callable_name(value)
    if isinstance(value, (list, tuple)):
        return [_stable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _stable(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, np.ndarray) and value.size <= 16:
        return [_stable(item) for item in value.tolist()]
    return "<opaque>"


def _canonical_basis(value: Any) -> Any:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return {str(key): _canonical_basis(item) for key, item in sorted(value.items(), key=str)}
    if isinstance(value, (list, tuple)):
        return [_canonical_basis(item) for item in value]
    return _stable(value)


__all__ = [
    "BackendUnavailableError",
    "ElectronicSystem",
    "MeanFieldBuild",
    "MeanFieldFactory",
    "MethodRecipe",
    "PySCFResources",
    "SettingsFactory",
    "build_mean_field",
    "configuration_snapshot",
    "configure_settings_mean_field",
    "invariant_snapshot",
    "require_pyscf",
]
