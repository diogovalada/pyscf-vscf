"""CPU/GPU parity through the package (``pytest -m gpu`` on GPU hardware).

The checks and their tolerances live in ``scripts/gpu_parity.py``, which also
writes the archived report.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.pyscf, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[1]


def _gpu_available() -> bool:
    try:
        from pyscf_vscf.backends._gpu import require_gpu

        require_gpu()
    except Exception:
        return False
    return True


def test_gpu_matches_cpu_within_fixed_tolerances(tmp_path) -> None:
    if not _gpu_available():
        pytest.skip("GPU4PySCF or a CUDA device is unavailable")
    spec = importlib.util.spec_from_file_location("gpu_parity", ROOT / "scripts" / "gpu_parity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "parity.json"
    status = module.main(["--output", str(output), "--quick"])
    report = json.loads(output.read_text(encoding="utf-8"))
    failed = [check for check in report["checks"] if not check["passed"]]
    assert status == 0 and not failed, failed
