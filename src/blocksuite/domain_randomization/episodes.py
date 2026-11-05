"""Loading per-episode replay records from a LeRobot dataset.

Moved verbatim out of scripts/rerender/rerender.py so that any consumer of the
domain-randomization package can build the episode records the package already
expects -- ``camera.py`` reads ``cam_pose``/``cam_K`` straight off these dicts. Pure
pandas/numpy: no Isaac, no sim, importable from a plain test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

# Parquet files the MOST RECENT load_episodes() call skipped as unreadable.
#
# A caller that ends up with zero episodes has to tell two very different things apart: the source
# is corrupt and there is genuinely nothing to render, versus its episode enumeration disagrees
# with the source, which is a bug that must fail loudly. Returning [] alone cannot express that,
# and treating a dead source as an enumeration bug wedges the fragment forever (the item is never
# acked, the lease expires, it is redelivered, and it dies the same way).
LAST_SKIPPED_PARQUET: List[Path] = []


#: Per-frame bool column marking a vecgen skill attempt that failed and was undone by a snapshot
#: restore: rows s..t, where the row after t equals the snapshot row s. Rendering skips them.
FAILED_ATTEMPT_COL = "is_failed_attempt"


def failed_attempt_rows(flag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(kept, obs)`` row indices of one episode with its failed-attempt rows removed.

    ``kept`` are the unflagged rows (their actions / labels); ``obs[i]`` is the row whose
    observation ``kept[i]`` takes: itself, except the first row after a flagged run ``[a, b]``,
    which takes row ``a`` -- the snapshot it was restored to (equal up to the restore, minus
    one-step quantities like Isaac's joint acceleration across the jump).
    """
    flag = np.asarray(flag, dtype=bool).reshape(-1)
    take = np.arange(len(flag))
    if flag.any():
        edges = np.diff(np.r_[0, flag.astype(np.int8), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if b < len(flag):  # b = first row after the run
                take[b] = take[a]
    kept = np.flatnonzero(~flag)
    return kept, take[kept]


def flag_values(values: Any) -> np.ndarray:
    """A ``FAILED_ATTEMPT_COL`` column (bool, [1]-lists or nulls) as a bool array."""
    return np.array([bool(np.asarray(v).reshape(-1)[0]) if v is not None and np.asarray(v).size else False
                     for v in values], dtype=bool)


def drop_failed_attempt_rows(table: Any) -> Any:
    """pyarrow ``table`` of one episode (in frame order) without its failed-attempt rows.

    Observation columns (``observation.*``, ``robot.poses.*``) follow :func:`failed_attempt_rows`'
    ``obs`` rows, all others its ``kept`` rows; ``frame_index`` / ``timestamp`` / ``is_first`` /
    ``is_last`` are renumbered. A table without the column, or with nothing flagged, is returned
    as-is.
    """
    import pyarrow as pa

    if FAILED_ATTEMPT_COL not in table.column_names:
        return table
    flag = flag_values(table.column(FAILED_ATTEMPT_COL).to_pylist())
    if not flag.any():
        return table
    kept, obs = failed_attempt_rows(flag)
    n = len(kept)
    cols = []
    for name in table.column_names:
        rows = obs if name.startswith(("observation.", "robot.poses.")) else kept
        cols.append(table.column(name).take(pa.array(rows)))
    out = pa.Table.from_arrays(cols, schema=table.schema)

    def setcol(name: str, values: np.ndarray) -> None:
        nonlocal out
        if name in out.column_names:
            out = out.set_column(out.column_names.index(name), out.schema.field(name),
                                 pa.array(values).cast(out.schema.field(name).type))

    fi = np.arange(n)
    setcol("frame_index", fi)
    if "timestamp" in table.column_names and table.num_rows > 1:
        ts = np.asarray(table.column("timestamp").to_numpy(), dtype=np.float64)
        dt = float(np.median(np.diff(ts)))
        setcol("timestamp", ts[0] + fi * dt)
    setcol("is_first", fi == 0)
    setcol("is_last", fi == n - 1)
    setcol(FAILED_ATTEMPT_COL, np.zeros(n, dtype=bool))
    return out


def load_episodes(dataset_path: Path, default_success: Optional[bool],
                  max_episodes: Optional[int],
                  keep: Optional[Set[int]] = None) -> List[Dict[str, Any]]:
    """Load per-episode replay data from a LeRobot dataset's data parquet(s).

    ``keep`` (episode indices) converts only those episodes, after the ``max_episodes`` cap: a
    group dataset holds every member problem's episodes, and converting all ~200 to render one
    member's ~24 took minutes per problem on Weka (rerender, 2026-11-02).

    Unreadable parquet files are skipped and recorded in `LAST_SKIPPED_PARQUET`, which is reset on
    every call -- read it immediately after, before the next load overwrites it.

    Rows flagged in ``FAILED_ATTEMPT_COL`` are dropped (see :func:`failed_attempt_rows`), so
    the episode replays straight from the snapshot into the retry that followed it.
    """
    LAST_SKIPPED_PARQUET.clear()
    data_files = sorted((dataset_path / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        return []
    frames = []
    for f in data_files:
        try:
            frames.append(pd.read_parquet(f))
        except Exception as e:
            LAST_SKIPPED_PARQUET.append(f)
            print(f"[rerender] WARNING skipping unreadable parquet {f}: {e}")
    if not frames:
        return []
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    obj_cols = [c for c in df.columns if c.startswith("observation.state.object_pose.")]
    campose_cols = [c for c in df.columns if c.startswith("observation.camera_pose.")]
    camK_cols = [c for c in df.columns if c.startswith("observation.camera_intrinsics.")]
    has = df.columns
    ep_ids = sorted(df["episode_index"].unique())
    if max_episodes is not None:
        ep_ids = ep_ids[:max_episodes]
    if keep is not None:
        ep_ids = [e for e in ep_ids if int(e) in keep]
        df = df[df["episode_index"].isin(ep_ids)]

    def col(ep, name):
        return np.stack([np.asarray(v, np.float32) for v in ep[name].values], axis=0)

    def scalar_col(ep, name):
        return ep[name].apply(lambda v: float(np.asarray(v).flatten()[0])).to_numpy(np.float32)

    episodes = []
    for eid in ep_ids:
        ep = df[df["episode_index"] == eid].sort_values("frame_index")
        if FAILED_ATTEMPT_COL in has:
            flag = flag_values(ep[FAILED_ATTEMPT_COL].values)
            if flag.any():
                ep = ep.iloc[failed_attempt_rows(flag)[1]]  # only observation columns are read
        rec = {
            "episode_index": int(eid),
            "length": len(ep),
            "joint": col(ep, "observation.state.joint_position"),       # (T,7)
            "gripper": scalar_col(ep, "observation.state.gripper_position"),  # (T,)
            "objects": {c[len("observation.state.object_pose."):]: col(ep, c)
                        for c in obj_cols},                              # name -> (T,7) wxyz
            # camera key (e.g. "camera:front_camera") -> recorded local pose / intrinsics
            "cam_pose": {c[len("observation.camera_pose."):]: col(ep, c) for c in campose_cols},
            "cam_K": {c[len("observation.camera_intrinsics."):]: col(ep, c) for c in camK_cols},
        }
        if "is_episode_successful" in has:
            rec["success"] = bool(np.asarray(ep["is_episode_successful"].iloc[0]).flatten()[0])
        else:
            rec["success"] = bool(default_success) if default_success is not None else True
        if default_success is not None:
            rec["success"] = default_success
        rec["cube_scale"] = (scalar_col(ep, "observation.state.cube_scale")[0]
                             if "observation.state.cube_scale" in has else 1.0)
        rec["table_height_delta"] = (scalar_col(ep, "observation.state.table_height_delta")[0]
                                     if "observation.state.table_height_delta" in has else 0.0)
        rec["objects"], rec["parked_objects"] = split_parked_objects(rec["objects"])
        episodes.append(rec)
    return episodes


def split_parked_objects(objects: Dict[str, np.ndarray]
                         ) -> "tuple[Dict[str, np.ndarray], List[str]]":
    """Separate objects that are not part of this episode from its replayable trajectories.

    A multi-problem env package spawns every member problem's blocks in every env; the recorder
    writes the blocks outside an episode's problem as all-NaN poses ("inactive objects"; eval's
    queue setup treats them as absent too). Those are returned by name, to be parked out of view,
    and never written to PhysX as NaN. A trajectory with only some NaN rows has the gaps filled
    from the nearest valid row.
    """
    kept: Dict[str, np.ndarray] = {}
    parked: List[str] = []
    for name, poses in objects.items():
        finite = np.isfinite(poses).all(axis=1)
        if not finite.any():
            parked.append(name)
            continue
        if not finite.all():
            idx = np.where(finite, np.arange(len(poses)), -1)
            idx = np.maximum.accumulate(idx)                       # forward fill
            first = int(np.argmax(finite))
            idx[idx < 0] = first                                   # leading gap: next valid row
            poses = poses[idx]
        kept[name] = poses
    return kept, parked


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
