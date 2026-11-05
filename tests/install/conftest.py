"""The GPU tier's gate for the installation checks (``--install-gpu`` is defined in ``tests/conftest.py``)."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="session")
def gpu_tier(request: pytest.FixtureRequest) -> None:
    """Skip unless ``--install-gpu`` was passed; fail if the Isaac Sim EULA has not been accepted."""
    if not request.config.getoption("--install-gpu"):
        pytest.skip("GPU tier: pass --install-gpu to run it (starts Isaac Sim; several minutes)")
    if os.environ.get("OMNI_KIT_ACCEPT_EULA", "").strip().lower() not in ("yes", "y"):
        pytest.fail(
            "Isaac Sim needs its EULA accepted for headless runs: export OMNI_KIT_ACCEPT_EULA=YES "
            "(see the installation guide)"
        )
