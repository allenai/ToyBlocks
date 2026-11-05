"""load_episodes(keep=...): a member problem converts only its own episodes of the group dataset.

rerender passes ``keep`` for Level-1 fragments and every --split_groups member, so the records it
returns must be exactly the kept episodes (after the ``max_episodes`` cap), each with its own
frames -- the render then numbers them by position and maps them back via src_episode_index.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from blocksuite.domain_randomization.episodes import load_episodes


def _write_dataset(root: Path, lengths: list[int]) -> None:
    ep = np.concatenate([np.full(n, e) for e, n in enumerate(lengths)])
    fr = np.concatenate([np.arange(n) for n in lengths])
    joint = [[float(1000 * e + f)] * 7 for e, f in zip(ep, fr)]
    table = pa.table(
        {
            "episode_index": ep.astype(np.int64),
            "frame_index": fr.astype(np.int64),
            "observation.state.joint_position": joint,
            "observation.state.gripper_position": [[0.0]] * len(ep),
            "observation.state.object_pose.block_a": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]
            * len(ep),
            "observation.state.cube_scale": [[0.03 + 0.001 * e] for e in ep],
        }
    )
    dst = root / "data" / "chunk-000" / "file-000.parquet"
    dst.parent.mkdir(parents=True)
    pq.write_table(table, dst)


def test_keep_converts_only_the_kept_episodes(tmp_path: Path) -> None:
    lengths = [3, 5, 2, 4, 6]
    _write_dataset(tmp_path, lengths)
    eps = load_episodes(tmp_path, True, None, keep={1, 3})
    assert [e["episode_index"] for e in eps] == [1, 3]
    for e in eps:
        idx = e["episode_index"]
        assert e["length"] == lengths[idx]
        assert e["joint"][:, 0].tolist() == [1000.0 * idx + f for f in range(lengths[idx])]
        assert abs(e["cube_scale"] - (0.03 + 0.001 * idx)) < 1e-6


def test_keep_applies_after_the_max_episodes_cap(tmp_path: Path) -> None:
    _write_dataset(tmp_path, [3, 5, 2, 4])
    assert [e["episode_index"] for e in load_episodes(tmp_path, True, 2, keep={0, 3})] == [0]
    assert load_episodes(tmp_path, True, None, keep=set()) == []
