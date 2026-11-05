"""toyblocks.vecgen.recording.merge on small synthetic LeRobot v3 datasets (no Isaac, no lerobot).

Each source is a problem_*/blocks_success dataset with its own episode/frame/task numbering, a
ragged object_pose column set, optional videos and meta/extra files. The merge must renumber
everything contiguously, re-point episodes at the merged data files, fill absent objects with NaN,
hardlink videos, and reduce stats equal to numpy over the merged frames.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from toyblocks.vecgen.recording import merge as M


def _features(blocks: list[str]) -> dict[str, Any]:
    feats: dict[str, Any] = {
        "observation.state.joint_position": {"dtype": "float32", "shape": [7], "names": None},
        "action": {"dtype": "float32", "shape": [3], "names": None},
        "is_valid": {"dtype": "bool", "shape": [1], "names": None},
        "observation.images.camera:front_camera": {"dtype": "video", "shape": [8, 8, 3], "names": None},
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    for b in blocks:
        feats[f"observation.state.object_pose.{b}"] = {"dtype": "float32", "shape": [7], "names": None}
    return feats


def make_dataset(
    root: Path,
    name: str,
    ep_lengths: list[int],
    tasks: list[str],
    blocks: list[str],
    seed: int,
    videos: bool = True,
    rows_per_file: int = 50,
) -> Path:
    """problem_<name>/blocks_success with local numbering starting at 0."""
    ds = root / f"problem_{name}" / "blocks_success"
    rng = np.random.default_rng(seed)
    n = sum(ep_lengths)
    ep = np.repeat(np.arange(len(ep_lengths)), ep_lengths)
    fi = np.concatenate([np.arange(k) for k in ep_lengths])
    # local task index i means tasks[i]; deliberately not the sorted order
    task_local = ep % len(tasks)
    cols: dict[str, pa.Array] = {
        "observation.state.joint_position": pa.array(list(rng.normal(size=(n, 7)).astype(np.float32)), pa.list_(pa.float32(), 7)),
        "action": pa.array(list(rng.normal(size=(n, 3)).astype(np.float32)), pa.list_(pa.float32(), 3)),
        "is_valid": pa.array(rng.random(n) > 0.3),
        "timestamp": pa.array((fi / 10).astype(np.float32)),
        "frame_index": pa.array(fi.astype(np.int64)),
        "episode_index": pa.array(ep.astype(np.int64)),
        "index": pa.array(np.arange(n, dtype=np.int64)),
        "task_index": pa.array(task_local.astype(np.int64)),
    }
    for b in blocks:
        cols[f"observation.state.object_pose.{b}"] = pa.array(list(rng.normal(size=(n, 7)).astype(np.float32)), pa.list_(pa.float32(), 7))
    table = pa.table(cols)
    for k, start in enumerate(range(0, n, rows_per_file)):
        f = ds / "data" / "chunk-000" / f"file-{k:03d}.parquet"
        f.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table.slice(start, rows_per_file), f, row_group_size=7)
    starts = np.concatenate([[0], np.cumsum(ep_lengths)[:-1]])
    meta = pa.table(
        {
            "episode_index": pa.array(np.arange(len(ep_lengths), dtype=np.int64)),
            "length": pa.array(np.asarray(ep_lengths, dtype=np.int64)),
            "dataset_from_index": pa.array(starts.astype(np.int64)),
            "dataset_to_index": pa.array((starts + np.asarray(ep_lengths)).astype(np.int64)),
            "tasks": pa.array([[tasks[e % len(tasks)]] for e in range(len(ep_lengths))]),
            **{
                f"stats/observation.state.object_pose.{b}/mean": pa.array([list(np.zeros(7))] * len(ep_lengths), pa.list_(pa.float64()))
                for b in blocks
            },
        }
    )
    (ds / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    pq.write_table(meta, ds / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    import pandas as pd

    pd.DataFrame({"task_index": range(len(tasks))}, index=tasks).to_parquet(ds / "meta" / "tasks.parquet")
    info = {"fps": 10, "robot_type": "fr3", "total_episodes": len(ep_lengths), "total_frames": n, "features": _features(blocks)}
    (ds / "meta" / "info.json").write_text(json.dumps(info))
    (ds / "meta" / "extra").mkdir()
    (ds / "meta" / "extra" / "domain.pddl").write_text("(domain)")
    (ds / "meta" / "extra" / f"{name}.pddl").write_text(f"(problem {name})")
    (ds / "meta" / "provenance.json").write_text(json.dumps({"git": {"sha": "abc"}, "files": [{"dest": f"meta/extra/{name}.pddl"}]}))
    if videos:
        for e in range(len(ep_lengths)):
            v = ds / "videos" / "observation.images.camera:front_camera" / "chunk-000" / f"file-{e:03d}.mp4"
            v.parent.mkdir(parents=True, exist_ok=True)
            v.write_bytes(f"{name}-{e}".encode())
    return ds


@pytest.fixture()
def sources(tmp_path: Path) -> Path:
    root = tmp_path / "execution"
    make_dataset(root, "a", [30, 25, 40], ["stack red", "flip blue"], ["block_red"], seed=1)
    make_dataset(root, "b", [10, 60], ["flip blue", "zz last"], ["block_red", "block_blue"], seed=2)
    make_dataset(root, "c", [33], ["stack red"], [], seed=3)
    return root


def _merge(sources: Path, out: Path, **kw: Any) -> None:
    paths = M.get_dataset_paths(sources)
    kw.setdefault("num_workers", 2)
    M.merge_datasets(paths, out, keep_cameras=["all"], union_schema=kw.pop("union_schema", True), **kw)


def _read(out: Path) -> pa.Table:
    return pa.concat_tables([pq.read_table(f) for f in sorted((out / "data").glob("*/*.parquet"))])


def test_indices_tasks_and_union(sources: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M, "MIN_ROWS_PER_FILE", 1)
    monkeypatch.setattr(M, "DATA_FILE_SIZE_IN_MB", 0.004)  # several output files
    out = tmp_path / "merged"
    _merge(sources, out)
    t = _read(out)
    n = 30 + 25 + 40 + 10 + 60 + 33
    assert t.num_rows == n
    assert len(list((out / "data").glob("*/*.parquet"))) > 1
    assert t.column("index").to_pylist() == list(range(n))
    ep = np.asarray(t.column("episode_index").to_pylist())
    assert np.all(np.diff(ep) >= 0) and ep[0] == 0 and ep[-1] == 5
    # tasks: global sorted table, every frame's index names its episode's task
    tasks = pq.read_table(out / "meta" / "tasks.parquet").to_pandas()
    assert tasks.index.tolist() == ["flip blue", "stack red", "zz last"]
    g = {"stack red": 1, "flip blue": 0, "zz last": 2}
    expected_task = [g[x] for x in ["stack red", "flip blue", "stack red", "flip blue", "zz last", "stack red"]]
    first = {int(e): int(ti) for e, ti in zip(ep, t.column("task_index").to_pylist())}
    assert [first[e] for e in range(6)] == expected_task
    # provenance columns
    sp = t.column("source_problem").to_pylist()
    assert sp[0] == "problem_a" and sp[-1] == "problem_c"
    assert set(t.column("instance_id").to_pylist()) == {-1}
    # ragged object poses: NaN vectors where the scene lacks the block, never null
    blue = t.column("observation.state.object_pose.block_blue")
    assert blue.null_count == 0
    vals = np.asarray(blue.to_pylist(), dtype=float)
    a_rows = slice(0, 95)
    assert np.isnan(vals[a_rows]).all() and np.isfinite(vals[95:165]).all() and np.isnan(vals[165:]).all()
    assert pa.types.is_list(blue.type)  # the TAMP merge's output type
    info = json.loads((out / "meta" / "info.json").read_text())
    assert info["total_episodes"] == 6 and info["total_frames"] == n and info["total_tasks"] == 3
    non_video = [k for k, v in info["features"].items() if v["dtype"] != "video"]
    assert non_video == [c for c in t.column_names if c in non_video]  # same order as the parquet


def test_episode_meta_points_at_first_frame_file(sources: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M, "MIN_ROWS_PER_FILE", 1)
    monkeypatch.setattr(M, "DATA_FILE_SIZE_IN_MB", 0.004)
    out = tmp_path / "merged"
    _merge(sources, out)
    meta = pq.read_table(out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").to_pandas()
    assert meta["episode_index"].tolist() == list(range(6))
    assert meta["dataset_from_index"].tolist() == [0, 30, 55, 95, 105, 165]
    for _, row in meta.iterrows():
        f = out / "data" / f"chunk-{row['data/chunk_index']:03d}" / f"file-{row['data/file_index']:03d}.parquet"
        idx = pq.read_table(f, columns=["index"]).column(0).to_pylist()
        assert idx[0] <= row["dataset_from_index"] <= idx[-1]
        assert row["videos/observation.images.camera:front_camera/file_index"] == row["episode_index"]
    # stats for a block an episode's scene lacks: NaN vectors, not null
    blue = meta["stats/observation.state.object_pose.block_blue/mean"]
    assert all(np.isnan(v).all() for v in blue[:3]) and all(np.isfinite(v).all() for v in blue[3:5])


def test_videos_hardlinked_and_collision_refused(sources: Path, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    _merge(sources, out)
    clips = sorted((out / "videos" / "observation.images.camera:front_camera").glob("*/*.mp4"))
    assert [c.name for c in clips] == [f"file-{i:03d}.mp4" for i in range(6)]
    src = sources / "problem_b" / "blocks_success" / "videos" / "observation.images.camera:front_camera" / "chunk-000" / "file-001.mp4"
    assert os.stat(clips[4]).st_ino == os.stat(src).st_ino
    dst = tmp_path / "x.mp4"
    dst.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        M.link_video((str(src), dst, "k", 0))
    assert src.read_bytes() == b"b-1"


def test_stats_match_numpy(sources: Path, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    _merge(sources, out)
    t = _read(out)
    stats = json.loads((out / "meta" / "stats.json").read_text())
    jp = np.asarray(t.column("observation.state.joint_position").to_pylist(), dtype=np.float64)
    s = stats["observation.state.joint_position"]
    np.testing.assert_allclose(s["mean"], jp.mean(0), rtol=1e-9)
    np.testing.assert_allclose(s["std"], jp.std(0), rtol=1e-6)
    np.testing.assert_allclose(s["min"], jp.min(0))
    np.testing.assert_allclose(s["q50"], np.percentile(jp, 50, axis=0))  # below the sample size: exact
    assert s["count"] == [t.num_rows]
    blue = np.asarray(t.column("observation.state.object_pose.block_blue").to_pylist(), dtype=np.float64)
    np.testing.assert_allclose(stats["observation.state.object_pose.block_blue"]["mean"], np.nanmean(blue, 0), rtol=1e-9)
    ts = stats["timestamp"]  # scalar column: scalar stats, as np.stack gave the TAMP reduction
    assert isinstance(ts["mean"], float)
    assert stats["is_valid"]["max"] == 1.0
    assert "source_problem" not in stats and "instance_id" not in stats
    assert stats["observation.images.camera:front_camera"]["mean"] == [[[0.5]], [[0.5]], [[0.5]]]


def test_corrupt_dataset_dropped_and_extra_copied(sources: Path, tmp_path: Path) -> None:
    bad = sources / "problem_b" / "blocks_success" / "data" / "chunk-000" / "file-000.parquet"
    bad.write_bytes(bad.read_bytes()[:100])
    out = tmp_path / "merged"
    _merge(sources, out)
    t = _read(out)
    assert t.num_rows == 30 + 25 + 40 + 33
    assert t.column("index").to_pylist() == list(range(t.num_rows))
    assert (out / "meta" / "extra" / "domain.pddl").read_text() == "(domain)"
    assert (out / "meta" / "extra" / "problem_c" / "c.pddl").exists()
    prov = json.loads((out / "meta" / "provenance.json").read_text())
    assert prov["git"] == {"sha": "abc"} and prov["merged_from_n_datasets"] == 3
    assert prov["sources"][0]["files"][0]["dest"] == "meta/extra/problem_a/a.pddl"


def test_recompute_stats_matches_merge(sources: Path, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    _merge(sources, out)
    first = json.loads((out / "meta" / "stats.json").read_text())
    (out / "meta" / "stats.json").unlink()
    M.main(["--recompute-stats", str(out), "--num_workers", "2"])
    again = json.loads((out / "meta" / "stats.json").read_text())
    assert first.keys() == again.keys()
    for k in first:
        for s in first[k]:
            np.testing.assert_allclose(np.asarray(first[k][s], float), np.asarray(again[k][s], float), rtol=1e-9)


def test_order_features_like_columns() -> None:
    feats = {
        "b": {"dtype": "float32"},
        "cam": {"dtype": "video"},
        "a": {"dtype": "float32"},
        "declared_only": {"dtype": "int64"},
    }
    out = M.order_features_like_columns(feats, ["a", "b"])
    assert list(out) == ["a", "cam", "b", "declared_only"]


def test_live_lock_refuses_second_merge(sources: Path, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    out.mkdir()
    (out / "keep.txt").write_text("written by the merge holding the lock")
    holder = M.OutputLock(out)
    holder.acquire()
    try:
        with pytest.raises(RuntimeError, match="being merged by another process"):
            _merge(sources, out)
        assert (out / "keep.txt").exists()  # refused before its rmtree
    finally:
        holder.release()
    assert not holder.path.exists()


def test_stale_lock_taken_over_and_released(sources: Path, tmp_path: Path) -> None:
    out = tmp_path / "merged"
    lock = tmp_path / "merged.merge.lock"
    lock.write_text(json.dumps({"token": "dead", "heartbeat": 0.0}))
    _merge(sources, out)
    assert (out / "meta" / "info.json").exists()
    assert not lock.exists()


def test_data_file_deleted_mid_merge_fails(sources: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(M, "MIN_ROWS_PER_FILE", 1)
    monkeypatch.setattr(M, "DATA_FILE_SIZE_IN_MB", 0.004)
    out = tmp_path / "merged"
    write_meta = M.write_episode_meta

    def _delete_then_write(*a: Any, **kw: Any) -> None:
        # what another process's rmtree did to the oct30 RG merge: data files gone after their write
        (out / "data" / "chunk-000" / "file-000.parquet").unlink()
        write_meta(*a, **kw)

    monkeypatch.setattr(M, "write_episode_meta", _delete_then_write)
    with pytest.raises(RuntimeError, match="1/.* data files missing"):
        _merge(sources, out)
    assert not (tmp_path / "merged.merge.lock").exists()


def test_failure_only_first_problem_merges_no_failures(tmp_path: Path) -> None:
    """A run whose first problem had no successful episode is still the vectorized layout."""
    base = tmp_path / "execution"
    for problem, splits in (("problem_a", ["blocks_failure"]), ("problem_b", ["blocks_failure", "blocks_success"])):
        for split in splits:
            meta = base / problem / split / "meta"
            meta.mkdir(parents=True)
            (meta / "info.json").write_text("{}")
    assert M.is_vectorized_layout(base)
    assert M.get_dataset_paths(base) == [base / "problem_b" / "blocks_success"]
