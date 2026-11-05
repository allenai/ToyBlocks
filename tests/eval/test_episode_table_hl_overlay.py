"""The episodes table's ``hl_overlay_prev_action`` column points at the LeRobot dataset's overlay video."""

from __future__ import annotations

import json
from pathlib import Path

from blocksuite.utils.wandb_logging import EpisodeResultsTable

_KEY = "observation.images.front_camera_hl_overlay"


def _table(tmp_path: Path, key: str | None = _KEY) -> EpisodeResultsTable:
    table = EpisodeResultsTable(tmp_path, save_sensor_obs_video=False, hl_overlay_video_key=key)
    for env_idx, episode_idx in ((0, 0), (1, 0), (0, 1)):
        table.add_episode(
            env_idx=env_idx, episode_idx=episode_idx, prompt=None, queue_setup=None, metrics={},
        )
    return table


def test_overlay_path_follows_the_lerobot_episode_index(tmp_path: Path) -> None:
    root = tmp_path / "lerobot_dataset" / "run_success"
    mp4 = root / "videos" / _KEY / "chunk-001" / "file-002.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.touch()
    table = _table(tmp_path)
    # (env 1, episode 0) was saved; (0, 1) was not; (0, 0) was saved but has no file yet.
    table.attach_lerobot_episode_indices({(1, 0): (str(root), 1002), (0, 0): (str(root), 3)})
    rows = json.loads(table.save_json_sidecar().read_text())
    assert [r["output_lerobot_episode_index"] for r in rows] == [3, 1002, None]
    assert [r["hl_overlay_prev_action_video_path"] for r in rows] == [None, str(mp4), None]


def test_no_overlay_column_without_a_video_key(tmp_path: Path) -> None:
    table = _table(tmp_path, key=None)
    table.attach_lerobot_episode_indices({(1, 0): (str(tmp_path), 0)})
    rows = json.loads(table.save_json_sidecar().read_text())
    assert all(r["hl_overlay_prev_action_video_path"] is None for r in rows)
    assert rows[1]["output_lerobot_episode_index"] == 0
