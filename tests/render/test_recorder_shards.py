"""Shard-directory selection for a problem's output.

``next_free_shard`` decides where the recorder writes. Shard 0 must stay the
problem dir itself even when a caller pre-created it, otherwise the dataset
lands in ``<dir>__s001`` while everything downstream (render.py's output
verifier, the orchestrator's completion oracle, --resume cleanup) keeps looking
at ``<dir>``. Runs without Isaac: the Omniverse/LeRobot imports are stubbed the
same way as in tests/vecgen/test_recorder_rows.py.
"""

import importlib.machinery
import importlib.util
import sys
import types
from pathlib import Path


def _install_sim_stubs() -> None:
    """Stub the Omniverse/LeRobot imports so the recorder module loads headless."""
    specs = {
        "isaaclab": (),
        "isaaclab.scene": (),
        "isaaclab.scene.interactive_scene": ("InteractiveScene",),
        "isaaclab.assets": ("Articulation", "RigidObject"),
        "isaaclab.sensors": ("ContactSensor",),
        "isaaclab.sensors.camera": ("Camera",),
    }
    for name, attrs in specs.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        sys.modules[name] = module
    # PyAV / LeRobot get stand-ins only when missing (the CPU lane has neither); nothing here
    # encodes or writes. Installed ones are left alone.
    optional = {
        "av": (),
        "lerobot": (),
        "lerobot.datasets": (),
        "lerobot.datasets.lerobot_dataset": ("LeRobotDataset", "embed_images"),
    }
    installed = {top: importlib.util.find_spec(top) is not None for top in ("av", "lerobot")}
    for name, attrs in optional.items():
        if name in sys.modules or installed[name.split(".")[0]]:
            continue
        module = types.ModuleType(name)
        # A spec, so a later importlib.util.find_spec(name) in the same session (other recorder
        # tests probe for the real package) reads it instead of raising on __spec__ None.
        module.__spec__ = importlib.machinery.ModuleSpec(name, None)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        sys.modules[name] = module


_install_sim_stubs()

from toyblocks.vecgen.recording.lerobot_recorder import (  # noqa: E402
    next_free_shard,
    shard_dir_for,
)


def test_missing_base_is_shard_zero(tmp_path: Path) -> None:
    """A problem dir that does not exist yet is shard 0."""
    base = tmp_path / "problem_x"
    assert next_free_shard(str(base)) == (base, 0)


def test_precreated_empty_base_is_still_shard_zero(tmp_path: Path) -> None:
    """The render.py / ``mkdir -p`` case: an empty dir is not a shard."""
    base = tmp_path / "problem_x"
    base.mkdir()
    assert next_free_shard(str(base)) == (base, 0)


def test_base_with_content_moves_to_next_sibling(tmp_path: Path) -> None:
    """A dir holding anything is a finalized-or-live shard and must be skipped."""
    base = tmp_path / "problem_x"
    (base / "blocks_success" / "meta").mkdir(parents=True)
    assert next_free_shard(str(base)) == (shard_dir_for(str(base), 1), 1)


def test_skips_occupied_siblings_but_reuses_empty_one(tmp_path: Path) -> None:
    """Occupied siblings are skipped; an empty pre-created one is taken; start is honoured."""
    base = tmp_path / "problem_x"
    (base / "blocks_success").mkdir(parents=True)
    (tmp_path / "problem_x__s001" / "blocks_success").mkdir(parents=True)
    (tmp_path / "problem_x__s002").mkdir()  # pre-created, empty
    assert next_free_shard(str(base)) == (tmp_path / "problem_x__s002", 2)
    assert next_free_shard(str(base), start=3) == (tmp_path / "problem_x__s003", 3)


def test_file_at_shard_path_is_not_free(tmp_path: Path) -> None:
    """A stray file at the shard path is never reused as a directory."""
    base = tmp_path / "problem_x"
    base.write_text("not a directory")
    assert next_free_shard(str(base)) == (shard_dir_for(str(base), 1), 1)
