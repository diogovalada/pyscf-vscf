from __future__ import annotations

import json
import subprocess
import sys


def test_packaged_vscf_example_runs() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "pyscf_vscf.examples.vscf_two_mode"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["method"] == "state-specific VSCF"
    assert payload["ground_converged"] is True
    assert payload["ground_iterations"] > 1
    assert payload["transitions"]
