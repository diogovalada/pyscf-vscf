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


def test_packaged_nmode_vci_example_runs() -> None:
    process = subprocess.run(
        [sys.executable, "-m", "pyscf_vscf.examples.nmode_vci"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    payload = json.loads(process.stdout)
    assert payload["method"] == "VCI on converged VSCF modals"
    assert payload["vscf_converged"] is True
    assert payload["state_cutoff_margin_Eh"] > 0.0
    assert payload["maximum_residual_Eh"] < 1e-10
    assert len(payload["transition_frequencies_cm"]) == 5
