"""Electronic-structure backend namespace."""

from importlib import import_module as _import_module

_pyscf_backend = _import_module("pyscf_vscf.backends.pyscf")

BackendUnavailableError = _pyscf_backend.BackendUnavailableError
ESSettings = _pyscf_backend.ESSettings
default_auxbasis = _pyscf_backend.default_auxbasis
electronic_symbol = _pyscf_backend.electronic_symbol
is_available = _pyscf_backend.is_available
make_mean_field = _pyscf_backend.make_mean_field
molecule_to_pyscf = _pyscf_backend.molecule_to_pyscf

del _import_module, _pyscf_backend

__all__ = [
    "BackendUnavailableError",
    "ESSettings",
    "default_auxbasis",
    "electronic_symbol",
    "is_available",
    "make_mean_field",
    "molecule_to_pyscf",
]
