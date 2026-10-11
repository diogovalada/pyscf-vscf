"""Warning categories and deprecation helpers."""

from __future__ import annotations

import warnings


class PyscfVscfDeprecationWarning(DeprecationWarning):
    """An API scheduled for removal in a later pyscf-vscf release."""


def deprecated(message: str, *, removal: str = "0.4", stacklevel: int = 3) -> None:
    """Emit a deprecation warning naming the release that removes the API."""

    warnings.warn(
        f"{message} It will be removed in pyscf-vscf {removal}.",
        PyscfVscfDeprecationWarning,
        stacklevel=stacklevel,
    )


__all__ = ["PyscfVscfDeprecationWarning", "deprecated"]
