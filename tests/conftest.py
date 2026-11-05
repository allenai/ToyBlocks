"""Shared pytest configuration for ToyBlocks tests.

`pytest_addoption` must live in the rootdir conftest, so the `--scene`,
`--rollout-case`, `--write-rollout-golden` and `--write-policy-input-golden`
options and the `scene_id` / `rollout_case` fixtures stay here even though each
is consumed by a single test directory. Isaac Sim is launched by
`tests/env_randomization/conftest.py`
instead of here, so that the CPU-only tiers (`tests/serving`, `tests/tamp`,
`tests/eval`) are collectable without Isaac Lab installed -- which is also why
the choices below are spelled out rather than imported from the test modules
that own them.

GPU tests:
    pytest tests/env_randomization -v
    pytest tests/env_randomization -v --scene droid_with_box_table
    pytest tests/env_randomization -v --scene sim2real
    pytest tests/env_randomization -v --rollout-case softer_pd_stiff

CPU tests:
    pytest tests/serving tests/tamp tests/eval -v

Installation checks (tests/install; the GPU tier is opt-in):
    pytest tests/install
    pytest tests/install --install-gpu
"""

import pytest

# Must match ROLLOUT_CASES in
# tests/env_randomization/test_rollout_determinism.py, which cannot be imported
# here: it pulls Isaac Lab, and this conftest is loaded for the CPU tiers too.
ROLLOUT_CASE_IDS = ["softer_config", "softer_pd_stiff", "soft_config"]


def pytest_addoption(parser):
    parser.addoption(
        "--scene", type=str, default=None,
        choices=["droid_with_box_table", "sim2real"],
        help="Scene preset to test (default: test both)",
    )
    parser.addoption(
        "--rollout-case", type=str, default=None,
        choices=ROLLOUT_CASE_IDS,
        help=(
            "Rollout-golden gain case to run (default: run all). Pair with "
            "--write-rollout-golden to re-record one case's golden at a time."
        ),
    )
    parser.addoption(
        "--write-rollout-golden", action="store_true", default=False,
        help=(
            "Re-baseline tests/env_randomization/rollout_determinism_golden_"
            "<case>.json from this run instead of comparing against it. Only "
            "for a deliberate Isaac / driver / physics change."
        ),
    )
    parser.addoption(
        "--write-policy-input-golden", action="store_true", default=False,
        help=(
            "Re-baseline tests/serving/goldens/policy_input_*.json from this run "
            "instead of comparing against it. Needs no GPU, no Isaac and no "
            "checkpoint -- it reruns the same pure functions the tests call -- so "
            "the diff is reviewable text. CI never passes it: a job that "
            "re-recorded its own baseline would report success against itself."
        ),
    )
    parser.addoption(
        "--install-gpu", action="store_true", default=False,
        help=(
            "Also run the GPU tier of tests/install: start Isaac Sim, run vecgen with and "
            "without videos, import the agentic skill servers. Takes several minutes and "
            "needs OMNI_KIT_ACCEPT_EULA=YES."
        ),
    )


def pytest_generate_tests(metafunc):
    """Parametrize scene_id / rollout_case from their flags, or over all values."""
    if "scene_id" in metafunc.fixturenames:
        scene = metafunc.config.getoption("--scene")
        if scene:
            metafunc.parametrize("scene_id", [scene], indirect=True)
        else:
            metafunc.parametrize("scene_id", ["droid_with_box_table", "sim2real"], indirect=True)
    if "rollout_case" in metafunc.fixturenames:
        case = metafunc.config.getoption("--rollout-case")
        cases = [case] if case else ROLLOUT_CASE_IDS
        metafunc.parametrize("rollout_case", cases, indirect=True)


@pytest.fixture
def scene_id(request):
    return request.param


@pytest.fixture
def rollout_case(request):
    return request.param
