#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Sequence


def _sanitize_tag(tag: str) -> str:
    tag = tag.strip()
    if not tag:
        return "run"
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", tag)
    tag = tag.strip("._-")
    return tag or "run"


def _stream_process_to_log(proc: subprocess.Popen[str], log_fp) -> int:
    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fp.write(line)
            log_fp.flush()
    except KeyboardInterrupt:
        # Forward Ctrl-C to the child, then wait briefly.
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            pass
        try:
            return proc.wait(timeout=5.0)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
            return proc.wait()
    return proc.wait()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Run a command and tee stdout+stderr to logs/<timestamp>_<tag>.log (streamed live)."
    )
    ap.add_argument("--tag", default="run", help="Short label included in the log filename")
    ap.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory to write logs into (default: logs/)")
    ap.add_argument(
        "--pythonnousersite",
        action="store_true",
        help="If set, export PYTHONNOUSERSITE=1 for the command (recommended for reproducibility)",
    )
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="Command to run; prefix with -- to separate options")
    args = ap.parse_args(argv)

    cmd: List[str] = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("Provide a command to run after '--' (example: scripts/run_logged.py --tag harmonic -- python ...)")

    if args.pythonnousersite:
        os.environ["PYTHONNOUSERSITE"] = "1"

    log_dir: Path = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = _sanitize_tag(str(args.tag))
    log_path = log_dir / f"{ts}_{tag}.log"

    # Merge stderr into stdout so the log is coherent.
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"# start: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"# cwd: {Path.cwd()}\n")
        f.write(f"# cmd: {' '.join(cmd)}\n")
        for k in ["PYTHONNOUSERSITE", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
            if k in os.environ:
                f.write(f"# env {k}={os.environ[k]}\n")
        f.write("\n")
        f.flush()

        print(f"[run_logged] logging to: {log_path}", flush=True)
        proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        rc = _stream_process_to_log(proc, f)
        f.write(f"\n# end: {datetime.now().isoformat(timespec='seconds')} rc={rc}\n")
        f.flush()
        return int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
