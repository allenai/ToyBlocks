"""CPU tests for reading and applying a dataset's recorded scene provenance.

``robot_id`` / ``scene_preset_id`` decide which table baseline every recorded
``table_height_delta`` and block Z is relative to, so getting them wrong means
eval measures a different scene than datagen recorded while every other check
still passes. Two callers depend on this module and both now hard-fail on a
miss: ``generate_task_from_dataset.py`` refuses to emit a package with no
provenance, and ``scene_verification._compare_provenance`` treats a mismatch as
a hard failure.

The read path is the part with no coverage and the most ways to be quietly
wrong: datagen writes provenance as top-level *parquet columns*, older and
hand-built datasets put it in ``meta/info.json`` (top-level or under a nested
``blocksuite`` block), and ``scene_id`` is datagen's name for what eval calls
``scene_preset_id``. Each is exercised below against a tmp_path LeRobot
skeleton, including the case where nothing is recorded anywhere.

Runs without Isaac Lab; needs ``pyarrow`` to build the skeleton's data shard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from blocksuite.utils.dataset_provenance import (
    read_dataset_provenance,
    reconcile_env_cfg_with_dataset,
)

ROBOT = "fr3_robotiq_gripper_softer"
SCENE = "droid_with_box_table"


def make_dataset(
    root: Path,
    *,
    info: dict | None = None,
    columns: dict | None = None,
) -> Path:
    """A minimal LeRobot root: ``data.parquet`` plus an optional ``meta/info.json``.

    ``data.parquet`` at the root is one of the layouts ``_discover_lerobot_roots``
    recognizes, and is the only file needed for the directory to be seen as a
    dataset at all.
    """
    root.mkdir(parents=True, exist_ok=True)
    table = {"episode_index": [0, 0], **{k: [v, v] for k, v in (columns or {}).items()}}
    pq.write_table(pa.table(table), root / "data.parquet")
    if info is not None:
        (root / "meta").mkdir(exist_ok=True)
        (root / "meta" / "info.json").write_text(json.dumps(info))
    return root


class FakeEnvCfg:
    """Just the two attributes ``reconcile_env_cfg_with_dataset`` touches."""

    def __init__(self, robot_id: str, scene_preset_id: str) -> None:
        self.robot_id = robot_id
        self.scene_preset_id = scene_preset_id


# --------------------------------------------------------------------------
# read_dataset_provenance
# --------------------------------------------------------------------------


def test_reads_top_level_info_json_keys(tmp_path):
    root = make_dataset(
        tmp_path / "ds", info={"robot_id": ROBOT, "scene_preset_id": SCENE}
    )
    provenance = read_dataset_provenance(root)
    assert provenance["robot_id"] == ROBOT
    assert provenance["scene_preset_id"] == SCENE
    assert provenance["source"] == str(root)


def test_reads_the_nested_blocksuite_block(tmp_path):
    """Datagen may nest provenance rather than pollute LeRobot's own schema."""
    root = make_dataset(
        tmp_path / "ds",
        info={"robot_type": "franka", "blocksuite": {"robot_id": ROBOT, "scene_preset_id": SCENE}},
    )
    provenance = read_dataset_provenance(root)
    assert (provenance["robot_id"], provenance["scene_preset_id"]) == (ROBOT, SCENE)


def test_scene_id_is_an_alias_for_scene_preset_id(tmp_path):
    """``scene_id`` is the name datagen writes; eval calls the same thing
    ``scene_preset_id``. A dataset recording only the alias must still resolve,
    or the whole scene tier silently degrades to 'nothing recorded'."""
    root = make_dataset(tmp_path / "ds", info={"robot_id": ROBOT, "scene_id": SCENE})
    assert read_dataset_provenance(root)["scene_preset_id"] == SCENE


def test_falls_back_to_the_per_step_parquet_columns(tmp_path):
    """The path real datagen datasets take: info.json carries neither field."""
    root = make_dataset(
        tmp_path / "ds",
        info={"robot_type": "franka"},
        columns={"robot_id": ROBOT, "scene_id": SCENE},
    )
    provenance = read_dataset_provenance(root)
    assert (provenance["robot_id"], provenance["scene_preset_id"]) == (ROBOT, SCENE)


def test_info_json_wins_over_the_columns(tmp_path):
    """The column fallback only fills what info.json left unset."""
    root = make_dataset(
        tmp_path / "ds",
        info={"scene_preset_id": "sim2real"},
        columns={"robot_id": ROBOT, "scene_id": SCENE},
    )
    provenance = read_dataset_provenance(root)
    assert provenance["scene_preset_id"] == "sim2real"
    assert provenance["robot_id"] == ROBOT


def test_both_none_when_nothing_is_recorded(tmp_path):
    """Not an error here -- the callers decide what an unrecorded scene means."""
    root = make_dataset(tmp_path / "ds", info={"robot_type": "franka"})
    provenance = read_dataset_provenance(root)
    assert provenance["robot_id"] is None
    assert provenance["scene_preset_id"] is None


def test_a_column_with_two_distinct_values_raises(tmp_path):
    """Scene identity is constant per dataset, so picking one would be a guess."""
    root = tmp_path / "ds"
    root.mkdir()
    pq.write_table(
        pa.table({"episode_index": [0, 1], "scene_id": [SCENE, "sim2real"]}),
        root / "data.parquet",
    )
    with pytest.raises(ValueError, match="multiple distinct values"):
        read_dataset_provenance(root)


def test_a_path_with_no_lerobot_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no LeRobot dataset"):
        read_dataset_provenance(tmp_path)


# --------------------------------------------------------------------------
# reconcile_env_cfg_with_dataset
# --------------------------------------------------------------------------


def test_defaults_the_env_to_the_recorded_values(tmp_path):
    """No flag passed: the dataset decides, which is the healthy path."""
    root = make_dataset(
        tmp_path / "ds", info={"robot_id": ROBOT, "scene_preset_id": SCENE}
    )
    env_cfg = FakeEnvCfg(robot_id="franka_robotiq_gripper", scene_preset_id="sim2real")
    reconcile_env_cfg_with_dataset(env_cfg, root, argv=["evaluate.py"])
    assert env_cfg.robot_id == ROBOT
    assert env_cfg.scene_preset_id == SCENE


def test_an_explicit_override_is_kept_and_warned_about(tmp_path, caplog):
    """The override wins, but the run is measuring a different scene than the
    dataset records -- silently doing that is the bug this warning exists for."""
    root = make_dataset(
        tmp_path / "ds", info={"robot_id": ROBOT, "scene_preset_id": SCENE}
    )
    env_cfg = FakeEnvCfg(robot_id=ROBOT, scene_preset_id="sim2real")
    with caplog.at_level(logging.WARNING, logger="blocksuite.utils.dataset_provenance"):
        reconcile_env_cfg_with_dataset(
            env_cfg, root, argv=["evaluate.py", "--env-cfg.scene-preset-id", "sim2real"]
        )
    assert env_cfg.scene_preset_id == "sim2real"
    assert "OVERRIDE MISMATCH" in caplog.text


def test_the_equals_form_of_a_flag_counts_as_explicit(tmp_path):
    """``--flag=value`` and ``--flag value`` must not disagree about intent."""
    root = make_dataset(
        tmp_path / "ds", info={"robot_id": ROBOT, "scene_preset_id": SCENE}
    )
    env_cfg = FakeEnvCfg(robot_id=ROBOT, scene_preset_id="sim2real")
    reconcile_env_cfg_with_dataset(
        env_cfg, root, argv=["evaluate.py", "--env-cfg.scene-preset-id=sim2real"]
    )
    assert env_cfg.scene_preset_id == "sim2real"


def test_no_provenance_and_no_override_exits(tmp_path):
    """Running an unverifiable scene would report numbers for the wrong setup."""
    root = make_dataset(tmp_path / "ds", info={"robot_type": "franka"})
    env_cfg = FakeEnvCfg(robot_id=ROBOT, scene_preset_id=SCENE)
    with pytest.raises(SystemExit, match="records no robot_id"):
        reconcile_env_cfg_with_dataset(env_cfg, root, argv=["evaluate.py"])


def test_no_provenance_but_an_explicit_override_is_honored(tmp_path, caplog):
    root = make_dataset(tmp_path / "ds", info={"robot_type": "franka"})
    env_cfg = FakeEnvCfg(robot_id=ROBOT, scene_preset_id=SCENE)
    with caplog.at_level(logging.WARNING, logger="blocksuite.utils.dataset_provenance"):
        reconcile_env_cfg_with_dataset(
            env_cfg,
            root,
            argv=[
                "evaluate.py",
                "--env-cfg.robot-id", ROBOT,
                "--env-cfg.scene-preset-id", SCENE,
            ],
        )
    assert (env_cfg.robot_id, env_cfg.scene_preset_id) == (ROBOT, SCENE)
    assert "cannot verify" in caplog.text
