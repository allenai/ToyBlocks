"""Installation checks, GPU tier: ``pytest tests/install --install-gpu`` (several minutes).

Each check runs in its own process, as the workflows do, since Isaac Sim cannot be restarted inside
one interpreter. Needs ``OMNI_KIT_ACCEPT_EULA=YES``. The first Isaac Sim start on a machine compiles
shaders and can take a few minutes on its own.

- PyTorch sees the GPU.
- Isaac Sim starts headless through ``toyblocks.cli.isaac_app.launch_app``, with the venv's Warp
  (the one cuRobo v2 needs) ahead of Isaac Sim's bundled copy.
- vecgen (with the ``vecgen`` extra): the installation guide's check, end to end. One env package
  is generated from ``flip.pddl``, then four episodes are recorded, and two more with camera videos
  that must decode.
- agentic robotics (with the ``agentic`` extra): its scripted and ManiFlow skill servers import
  once Isaac Sim is up (they reach ``toyblocks.eval.ll_oracle`` and ``toyblocks.serving``).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

from tests.install.helpers import extra_installed, run, run_python, tail

pytestmark = pytest.mark.usefixtures("gpu_tier")

PROBLEM = "src/toyblocks/tamp/blocks/problems/flip.pddl"
DOMAIN = "src/toyblocks/tamp/blocks/domain.pddl"
RESULT = re.compile(r"^RESULT episodes=(\d+) success=(\d+)", re.MULTILINE)
ISAAC_TIMEOUT_S = 1800  # a first start compiles shaders


def test_torch_sees_the_gpu() -> None:
    process = run_python(
        "import torch\nassert torch.cuda.is_available(), 'torch.cuda.is_available() is False'\n"
        "print('GPU', torch.cuda.get_device_name(0), torch.version.cuda)\n",
        timeout=300,
    )
    assert process.returncode == 0 and "GPU" in process.stdout, tail(process)


def test_isaac_sim_starts_with_the_venvs_warp() -> None:
    code = (
        "import importlib.metadata, os, sys\n"
        "sys.argv = [sys.argv[0], '--headless']\n"
        "from toyblocks.cli.isaac_app import launch_app\n"
        "app = launch_app()\n"
        "import warp\n"
        "assert warp.__file__.startswith(sys.prefix), f'warp from {warp.__file__}, not the venv'\n"
        "assert warp.__version__ == importlib.metadata.version('warp-lang'), warp.__version__\n"
        "print('ISAAC_OK warp', warp.__version__, flush=True)\n"
        "os._exit(0)  # closing Kit can hang; the check is done\n"
    )
    process = run_python(code, timeout=ISAAC_TIMEOUT_S, accept_eula=True)
    assert process.returncode == 0 and "ISAAC_OK" in process.stdout, tail(process)


@pytest.fixture(scope="module")
def flip_env(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The flip env package, generated as the installation guide does (8 reset states)."""
    if not extra_installed("vecgen"):
        pytest.skip("the vecgen extra is not installed")
    out = tmp_path_factory.mktemp("env")
    process = run(
        [
            sys.executable,
            "scripts/generate_task_from_pddl.py",
            "--domain_path",
            "src/toyblocks/tamp/blocks",
            "--pddl",
            PROBLEM,
            "--num_reset_states",
            "8",
            "--num_envs",
            "4",
            "--seed",
            "0",
            "--output_dir",
            str(out),
            "--headless",
        ],
        timeout=ISAAC_TIMEOUT_S,
        accept_eula=True,
    )
    package = out / "flip"
    assert process.returncode == 0 and (package / "reset_states.pkl").exists(), tail(process)
    return package


def _vecgen(env: Path, record: Path, envs: int, episodes: int, *extra: str) -> tuple[int, int]:
    process = run(
        [
            sys.executable,
            "scripts/vecgen/run.py",
            "--env-path",
            str(env),
            "--pddl",
            PROBLEM,
            "--domain",
            DOMAIN,
            "--num-envs",
            str(envs),
            "--episodes",
            str(episodes),
            "--record",
            str(record),
            "--mark-complete",
            "--headless",
            *extra,
        ],
        timeout=ISAAC_TIMEOUT_S,
        accept_eula=True,
    )
    match = RESULT.search(process.stdout)
    assert process.returncode == 0 and match is not None, tail(process)
    assert (record / ".execution_complete").exists(), (
        f"{record} was not marked complete\n{tail(process)}"
    )
    return int(match.group(1)), int(match.group(2))


def test_vecgen_records_episodes(flip_env: Path, tmp_path: Path) -> None:
    episodes, successes = _vecgen(flip_env, tmp_path / "flip", 4, 4)
    assert episodes == 4
    # Flips fail now and then; none of four succeeding points at the install, not at chance.
    assert successes >= 1, "0 of 4 flip episodes succeeded"


def test_vecgen_records_videos(flip_env: Path, tmp_path: Path) -> None:
    import av

    record = tmp_path / "flip_videos"
    episodes, _ = _vecgen(flip_env, record, 2, 2, "--record-videos", "--enable_cameras")
    assert episodes == 2
    streams = sorted({p.parent.parent.name for p in record.glob("*/videos/*/*/*.mp4")})
    expected = [
        f"observation.{kind}.camera:{camera}"
        for kind in ("depth", "images")
        for camera in ("front_camera", "gripper_camera")
    ]
    assert streams == expected, f"video streams {streams}, expected {expected}"
    for stream in expected:
        video = next(record.glob(f"*/videos/{stream}/*/*.mp4"))
        with av.open(str(video)) as container:
            frames = sum(1 for _ in container.decode(container.streams.video[0]))
        assert frames > 0, f"{video} has no frames"


def test_agentic_skill_servers_import() -> None:
    if not extra_installed("agentic"):
        pytest.skip("the agentic extra is not installed")
    code = (
        "import os\n"
        "import agentic_robotics.adapters.toyblocks.toyblocks_env  # starts Isaac Sim\n"
        "import agentic_robotics.adapters.toyblocks.skills.inline_oracle\n"
        "import agentic_robotics.adapters.toyblocks.maniflow_mcp_server\n"
        "print('AGENTIC_OK', flush=True)\n"
        "os._exit(0)\n"
    )
    process = run_python(code, timeout=ISAAC_TIMEOUT_S, accept_eula=True)
    assert process.returncode == 0 and "AGENTIC_OK" in process.stdout, tail(process)
