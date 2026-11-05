"""Stitch the per-writer LeRobot v3 datasets of one recording into a single dataset.

``IsaacDataRecorderLeRobot(num_writers=K)`` runs K encoder subprocesses, each writing its own
``<name>_success`` / ``<name>_failure`` pair. At finalize the pairs are stitched, so the published
layout is the one a single writer produces: data parquet files, ``meta/episodes``, one video file
per episode at ``videos/<key>/chunk-{ep // 1000:03d}/file-{ep % 1000:03d}.mp4`` and the
``videos/<key>/*`` episode columns, which the recorder backfills afterwards from the episode index.

Episodes are numbered writer by writer, each in its own saved order -- the numbering
``IsaacDataRecorderLeRobot.merged_episode_index_map`` reproduces. Files are moved, not copied.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np

_CHUNK = 1000  # LeRobot v3 files per chunk directory (``chunks_size``)


def _file_path(root: Path, kind: str, chunk: int, file: int, suffix: str) -> Path:
    return root / kind / f"chunk-{chunk:03d}" / f"file-{file:03d}{suffix}"


def _load_tasks(root: Path) -> dict[int, str]:
    """``{task_index: task}`` of one dataset."""
    import pandas as pd

    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        return {}
    df = pd.read_parquet(path)
    return {int(row["task_index"]): str(task) for task, row in df.iterrows()}


def _load_episode_rows(root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(str(path)).to_pylist())
    rows.sort(key=lambda r: int(r["episode_index"]))
    return rows


def _renumber_episode_row(
    row: dict[str, Any], episode: int, frame_offset: int, data_loc: tuple[int, int]
) -> dict[str, Any]:
    """``row`` moved to global ``episode`` and ``frame_offset``; video pointers are dropped
    (the recorder backfills them from the new episode index)."""
    out = {k: v for k, v in row.items() if not k.startswith("videos/")}
    from_idx = int(row["dataset_from_index"]) + frame_offset
    to_idx = int(row["dataset_to_index"]) + frame_offset
    out["episode_index"] = episode
    out["dataset_from_index"] = from_idx
    out["dataset_to_index"] = to_idx
    out["data/chunk_index"], out["data/file_index"] = data_loc
    out["meta/episodes/chunk_index"] = 0
    out["meta/episodes/file_index"] = 0
    for key, value in list(out.items()):
        if key.startswith("stats/episode_index/"):
            stat = key.rsplit("/", 1)[-1]
            out[key] = [0.0] if stat == "std" else ([float(episode)] if stat != "count" else value)
        elif key.startswith("stats/index/") and value is not None:
            stat = key.rsplit("/", 1)[-1]
            if stat not in ("std", "count"):
                out[key] = (np.asarray(value, dtype=np.float64) + frame_offset).tolist()
    return out


def _aggregate_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Dataset ``stats.json`` of the stitched dataset (LeRobot's count-weighted aggregate)."""
    from lerobot.datasets.compute_stats import aggregate_stats

    as_np = [
        {feat: {k: np.asarray(v) for k, v in fstats.items()} for feat, fstats in stats.items()}
        for stats in stats_list
    ]
    merged = aggregate_stats(as_np)
    return {
        feat: {k: np.asarray(v).tolist() for k, v in fstats.items()}
        for feat, fstats in merged.items()
    }


def stitch_datasets(sources: list[Path], dest: Path) -> list[int]:
    """Move the episodes of ``sources`` (LeRobot v3 datasets with one schema) into ``dest``.

    Returns each source's first global episode index. ``dest`` is replaced if it exists.
    """
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if dest.exists():
        shutil.rmtree(dest)
    (dest / "meta").mkdir(parents=True)

    info: dict[str, Any] = json.loads((sources[0] / "meta" / "info.json").read_text())

    global_tasks: dict[str, int] = {}
    episode_rows: list[dict[str, Any]] = []
    stats_list: list[dict[str, Any]] = []
    offsets: list[int] = []
    n_episodes = n_frames = n_data_files = 0

    for src in sources:
        offsets.append(n_episodes)
        task_remap: dict[int, int] = {}
        for old_idx, task in sorted(_load_tasks(src).items()):
            task_remap[old_idx] = global_tasks.setdefault(task, len(global_tasks))
        rows = _load_episode_rows(src)
        # The encoder publishes the mp4s itself, so the directories are the authority.
        video_dir = src / "videos"
        video_keys = (
            sorted(d.name for d in video_dir.iterdir() if d.is_dir()) if video_dir.is_dir() else []
        )
        episode_map = {int(r["episode_index"]): n_episodes + i for i, r in enumerate(rows)}

        # Data files keep their grouping; only the index columns are rewritten.
        data_locs: dict[tuple[int, int], tuple[int, int]] = {}
        for path in sorted((src / "data").glob("chunk-*/file-*.parquet")):
            chunk, file = int(path.parent.name[6:]), int(path.stem[5:])
            table = pq.read_table(str(path))
            cols = table.column_names
            if "episode_index" in cols:
                old = table.column("episode_index").to_numpy(zero_copy_only=False)
                new = np.array([episode_map[int(e)] for e in old], dtype=np.int64)
                table = table.set_column(
                    cols.index("episode_index"), "episode_index", pa.array(new)
                )
            if "index" in cols:
                table = table.set_column(
                    cols.index("index"), "index", pc.add(table.column("index"), n_frames)
                )
            if "task_index" in cols and task_remap:
                old = table.column("task_index").to_numpy(zero_copy_only=False)
                new = np.array([task_remap.get(int(t), int(t)) for t in old], dtype=np.int64)
                table = table.set_column(cols.index("task_index"), "task_index", pa.array(new))
            loc = (n_data_files // _CHUNK, n_data_files % _CHUNK)
            out = _file_path(dest, "data", loc[0], loc[1], ".parquet")
            out.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, str(out))
            data_locs[(chunk, file)] = loc
            n_data_files += 1

        for row in rows:
            episode = episode_map[int(row["episode_index"])]
            loc = data_locs[(int(row["data/chunk_index"]), int(row["data/file_index"]))]
            episode_rows.append(_renumber_episode_row(row, episode, n_frames, loc))
            for key in video_keys:
                src_ep = int(row["episode_index"])
                video = _file_path(
                    src / "videos" / key, "", src_ep // _CHUNK, src_ep % _CHUNK, ".mp4"
                )
                if video.is_file():
                    target = _file_path(
                        dest / "videos" / key, "", episode // _CHUNK, episode % _CHUNK, ".mp4"
                    )
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(video), str(target))

        stats_path = src / "meta" / "stats.json"
        if stats_path.is_file():
            stats_list.append(json.loads(stats_path.read_text()))
        n_episodes += len(rows)
        n_frames += sum(int(r["length"]) for r in rows)

    if episode_rows:
        out = _file_path(dest / "meta", "episodes", 0, 0, ".parquet")
        out.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(episode_rows), str(out))
    tasks_df = pd.DataFrame({"task_index": list(global_tasks.values())}, index=list(global_tasks))
    tasks_df.to_parquet(dest / "meta" / "tasks.parquet")
    if stats_list:
        (dest / "meta" / "stats.json").write_text(
            json.dumps(_aggregate_stats(stats_list), indent=4)
        )

    info["total_episodes"] = n_episodes
    info["total_frames"] = n_frames
    info["total_tasks"] = len(global_tasks)
    info["splits"] = {"train": f"0:{n_episodes}"}
    (dest / "meta" / "info.json").write_text(json.dumps(info, indent=4))
    return offsets
