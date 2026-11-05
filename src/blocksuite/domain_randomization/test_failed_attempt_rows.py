"""Failed vecgen attempts (``is_failed_attempt``) are skipped by the rerender's loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

from blocksuite.domain_randomization.episodes import (
    FAILED_ATTEMPT_COL,
    drop_failed_attempt_rows,
    failed_attempt_rows,
    load_episodes,
)


def test_rows_after_a_flagged_run_take_the_snapshot_observation() -> None:
    flag = np.array([0, 0, 1, 1, 1, 0, 0, 1, 0], dtype=bool)
    kept, obs = failed_attempt_rows(flag)
    assert kept.tolist() == [0, 1, 5, 6, 8]
    assert obs.tolist() == [0, 1, 2, 6, 7]  # row 5 <- snapshot row 2, row 8 <- row 7


def test_nothing_flagged_keeps_every_row() -> None:
    kept, obs = failed_attempt_rows(np.zeros(4, dtype=bool))
    assert kept.tolist() == obs.tolist() == [0, 1, 2, 3]


def _episode_table(n: int, flagged: list[int]) -> pa.Table:
    flag = np.zeros(n, dtype=bool)
    flag[flagged] = True
    return pa.table({
        "observation.state.joint_position": [[float(i)] * 7 for i in range(n)],
        "action.joint_target": [[100.0 + i] * 7 for i in range(n)],
        "frame_index": pa.array(np.arange(n), pa.int64()),
        "timestamp": pa.array(np.arange(n) / 25.0, pa.float32()),
        "is_first": np.arange(n) == 0,
        "is_last": np.arange(n) == n - 1,
        FAILED_ATTEMPT_COL: flag,
    })


def test_drop_failed_attempt_rows_splices_and_renumbers() -> None:
    out = drop_failed_attempt_rows(_episode_table(6, [0, 1, 2]))
    assert out.num_rows == 3
    q = [row[0] for row in out["observation.state.joint_position"].to_pylist()]
    a = [row[0] for row in out["action.joint_target"].to_pylist()]
    assert q == [0.0, 4.0, 5.0]          # first kept row shows the snapshot (row 0) state
    assert a == [103.0, 104.0, 105.0]    # ... with the retry's actions
    assert out["frame_index"].to_pylist() == [0, 1, 2]
    assert np.allclose(out["timestamp"].to_pylist(), [0.0, 0.04, 0.08])
    assert out["is_first"].to_pylist() == [True, False, False]
    assert out["is_last"].to_pylist() == [False, False, True]
    assert not any(out[FAILED_ATTEMPT_COL].to_pylist())


def test_table_without_flags_is_unchanged() -> None:
    t = _episode_table(4, [])
    assert drop_failed_attempt_rows(t) is t
    assert drop_failed_attempt_rows(t.drop_columns([FAILED_ATTEMPT_COL])).num_rows == 4


def test_load_episodes_skips_flagged_rows(tmp_path: Path) -> None:
    n = 8
    flag = np.zeros(n, dtype=bool)
    flag[[2, 3, 4]] = True
    df = pd.DataFrame({
        "episode_index": [0] * n,
        "frame_index": np.arange(n),
        "observation.state.joint_position": [np.full(7, i, np.float32) for i in range(n)],
        "observation.state.gripper_position": [np.array([0.0], np.float32)] * n,
        FAILED_ATTEMPT_COL: flag,
    })
    (tmp_path / "data" / "chunk-000").mkdir(parents=True)
    df.to_parquet(tmp_path / "data" / "chunk-000" / "file-000.parquet")
    (ep,) = load_episodes(tmp_path, True, None)
    assert ep["length"] == 5
    assert ep["joint"][:, 0].tolist() == [0, 1, 2, 6, 7]  # row 5 replays the snapshot row 2
