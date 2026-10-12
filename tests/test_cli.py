from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from pyscf_vscf import cli


def test_help_and_version_do_not_import_pyscf() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    blocker = """
import importlib.abc
import sys

class BlockExpensiveImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "pyscf" or fullname.startswith("pyscf."):
            raise RuntimeError(f"unexpected PySCF import: {fullname}")
        return None

sys.meta_path.insert(0, BlockExpensiveImports())
from pyscf_vscf.cli import main
raise SystemExit(main(ARGV))
"""
    import pyscf_vscf

    for argv, expected in ((["--help"], "scan1d"), (["--version"], pyscf_vscf.__version__)):
        code = blocker.replace("ARGV", repr(argv))
        proc = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            env=env,
            text=True,
            capture_output=True,
        )
        assert proc.returncode == 0, proc.stderr
        assert expected in proc.stdout


def test_scan1d_help_exposes_relaxed_normal_scans_and_no_dispersion_option() -> None:
    parser = cli._build_parser()
    commands = next(action for action in parser._actions if action.dest == "command")
    help_text = commands.choices["scan1d"].format_help()
    assert "normal-relaxed" in help_text
    assert "lbs-relaxed" not in help_text
    assert "--dispersion" not in help_text
