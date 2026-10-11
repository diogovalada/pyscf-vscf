from __future__ import annotations

import logging

import pytest


@pytest.fixture(autouse=True)
def _restore_package_logger():
    """Undo logging changes (for example from CLI ``-v`` runs) after each test."""

    logger = logging.getLogger("pyscf_vscf")
    handlers, level = list(logger.handlers), logger.level
    yield
    for handler in list(logger.handlers):
        if handler not in handlers:
            logger.removeHandler(handler)
    logger.setLevel(level)
