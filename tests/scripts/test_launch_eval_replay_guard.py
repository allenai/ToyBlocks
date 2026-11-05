"""CLI guards on scripts/launch_eval.py."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_replay_rejects_init_after_hl_action() -> None:
    env = dict(__import__("os").environ)
    proc = subprocess.run(
        [
            sys.executable,
            str(_REPO_ROOT / "scripts" / "launch_eval.py"),
            "--dry-run",
            "--replay",
            "--smoke-test",
            "--init-after-hl-action",
            "flip",
        ],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    combined = proc.stdout + proc.stderr
    assert "init-after-hl-action" in combined.lower()
    assert "replay" in combined.lower()
