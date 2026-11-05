"""Stitching the per-writer datasets of a multi-writer recording (CPU only, no Isaac)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from toyblocks.vecgen.recording._stitch import stitch_datasets

# Needs the real LeRobot (the CPU CI lane has none); test_recorder_rows stands in a stub then.
if not hasattr(pytest.importorskip("lerobot.datasets.lerobot_dataset").LeRobotDataset, "create"):
    pytest.skip("needs LeRobot", allow_module_level=True)

FEATURES = {
    "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
    "action": {"dtype": "float32", "shape": (2,), "names": None},
}


def _dataset(root: Path, episodes: list[tuple[str, int, float]]) -> Path:
    """A LeRobot v3 dataset with one episode per ``(task, length, value)`` and a fake video
    file per episode in the recorder's one-file-per-episode layout."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.create(repo_id=root.name, root=str(root), fps=25, features=FEATURES)
    for ep, (task, length, value) in enumerate(episodes):
        for t in range(length):
            v = np.array([value, t], dtype=np.float32)
            ds.add_frame({"observation.state": v, "action": v, "task": task})
        ds.save_episode()
        video = root / "videos" / "observation.images.cam" / "chunk-000" / f"file-{ep:03d}.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_text(f"{root.name}:{ep}")
    ds.finalize()
    return root


def test_stitch_renumbers_episodes_frames_tasks_and_videos(tmp_path: Path) -> None:
    a = _dataset(tmp_path / "w000" / "blocks_success", [("pick", 3, 1.0), ("place", 2, 2.0)])
    b = _dataset(tmp_path / "w001" / "blocks_success", [("place", 4, 3.0)])
    dest = tmp_path / "blocks_success"

    offsets = stitch_datasets([a, b], dest)

    assert offsets == [0, 2]
    info = json.loads((dest / "meta" / "info.json").read_text())
    assert (info["total_episodes"], info["total_frames"], info["total_tasks"]) == (3, 9, 2)
    assert info["splits"] == {"train": "0:3"}

    frames = [pq.read_table(str(p)).to_pandas() for p in sorted((dest / "data").rglob("*.parquet"))]
    data = np.concatenate([f[["episode_index", "index", "task_index"]].to_numpy() for f in frames])
    assert data[:, 0].tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]
    assert data[:, 1].tolist() == list(range(9))
    # "place" keeps one global task index across both writers
    assert data[3, 2] == data[5, 2] != data[0, 2]

    eps = pq.read_table(
        str(dest / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    ).to_pylist()
    assert [(e["episode_index"], e["dataset_from_index"], e["dataset_to_index"]) for e in eps] == [
        (0, 0, 3),
        (1, 3, 5),
        (2, 5, 9),
    ]
    videos = sorted((dest / "videos" / "observation.images.cam" / "chunk-000").glob("*.mp4"))
    assert [v.read_text() for v in videos] == [
        "blocks_success:0",
        "blocks_success:1",
        "blocks_success:0",
    ]
    assert [v.name for v in videos] == ["file-000.mp4", "file-001.mp4", "file-002.mp4"]


def test_stitched_dataset_loads_in_lerobot(tmp_path: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    a = _dataset(tmp_path / "w000" / "x", [("pick", 3, 1.0)])
    b = _dataset(tmp_path / "w001" / "x", [("place", 2, 2.0), ("pick", 1, 3.0)])
    for root in (a, b):
        for v in (root / "videos").rglob("*.mp4"):
            v.unlink()
    dest = tmp_path / "x"
    stitch_datasets([a, b], dest)

    ds = LeRobotDataset(repo_id="x", root=str(dest))
    assert ds.num_episodes == 3 and ds.num_frames == 6
    assert [float(ds[i]["observation.state"][0]) for i in range(6)] == [1, 1, 1, 2, 2, 3]
    assert ds[5]["task"] == "pick" and ds[3]["task"] == "place"
