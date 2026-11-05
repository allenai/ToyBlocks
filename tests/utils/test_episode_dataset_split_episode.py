"""``load_setups`` measures an episode cut across two data files by its whole length."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from blocksuite.utils.episode_dataset import load_setups

# Episode 1 straddles the cut between the two data files (rows 5..11, cut at 8).
_LENGTHS = {0: 5, 1: 7, 2: 4}
_ROWS_PER_FILE = 8


def _write_dataset(root: Path) -> None:
    rows = [
        {
            "episode_index": ep,
            "frame_index": f,
            "is_first": f == 0,
            "observation.state.joint_position": [0.0] * 7,
        }
        for ep, n in _LENGTHS.items()
        for f in range(n)
    ]
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    for k, start in enumerate(range(0, len(rows), _ROWS_PER_FILE)):
        pq.write_table(
            pa.Table.from_pylist(rows[start : start + _ROWS_PER_FILE]),
            data / f"file-{k:03d}.parquet",
        )
    meta = root / "meta" / "episodes" / "chunk-000"
    meta.mkdir(parents=True)
    # Each episode's file is the one it starts in, as LeRobot records it.
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1, 2],
                "length": [_LENGTHS[0], _LENGTHS[1], _LENGTHS[2]],
                "data/chunk_index": [0, 0, 0],
                "data/file_index": [0, 0, 1],
            }
        ),
        meta / "file-000.parquet",
    )
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 25, "chunks_size": 1000}))


def test_split_episode_keeps_its_full_length(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    lengths = {int(s["episode_index"]): int(s["expert_episode_length"]) for s in load_setups(tmp_path)}
    assert lengths == _LENGTHS


def test_split_episode_full_length_when_filtered(tmp_path: Path) -> None:
    _write_dataset(tmp_path)
    (setup,) = load_setups(tmp_path, episode_indices=[1])
    assert int(setup["expert_episode_length"]) == _LENGTHS[1]
