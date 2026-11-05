"""Isaac Sim launch for the env-randomization tests.

These tests require Isaac Lab / IsaacSim with GPU access. The app must be
launched before any isaac import happens, so it is done at collection time here
rather than in the rootdir conftest -- the rootdir conftest is also loaded for
the CPU-only test tiers, which must be collectable without Isaac Lab.

Run:
    pytest tests/env_randomization -v
    pytest tests/env_randomization -v --scene droid_with_box_table
"""

import os
import sys

import pytest

# The venv's Warp before the app starts, as toyblocks.cli.isaac_app.launch_app does: Isaac
# Sim's omni.warp.core otherwise puts its bundled (older) Warp first on sys.path, and every
# test collected after this one would import cuRobo against it.
import warp  # noqa: F401
from isaaclab.app import AppLauncher

simulation_app = AppLauncher(headless=True).app

# Set in pytest_sessionfinish; consumed in pytest_unconfigure (see below).
_pending_process_exitstatus: int | None = None


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
    """Remember pytest's exit status for ``pytest_unconfigure``.

    Kit's shutdown runs at normal interpreter exit and returns 0 regardless of
    test outcome, so CI could report success after failures. We ``os._exit``
    with pytest's status instead of calling ``simulation_app.close()`` (that
    hangs -- see ``scripts/generate_task_from_pddl.py``).

    Do not ``os._exit`` here: ``trylast`` sessionfinish hooks run *before* the
    terminal reporter's wrapping ``pytest_sessionfinish`` teardown, so an early
    exit suppresses collection errors, tracebacks, and the final summary.
    """
    global _pending_process_exitstatus
    _pending_process_exitstatus = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    """Exit with pytest's status after all reporting hooks have finished."""
    if _pending_process_exitstatus is None:
        return
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_pending_process_exitstatus)
