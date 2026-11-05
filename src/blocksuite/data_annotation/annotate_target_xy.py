#!/usr/bin/env python3
"""
Lightweight annotation: writes ``goal.action`` and ``goal.position`` columns.

``goal.action`` is a string describing the current action (e.g. ``pick``,
``place``, ``flip right``, ``rotate cw``, ``push left``, ``align front``).

``goal.position`` is a float32 [2] column holding the (u, v) pixel coordinates
of the target object for the current action.  Frames with no target get NaN.

``goal.position_target`` is a second float32 [2] column, populated only for
push/align actions, holding the pixels of the block being pushed against — so a
push annotation reads ``push <direction>`` from ``(x, y)`` toward ``(x', y')``.

``goal.position_gripper`` and ``goal.position_target_gripper`` mirror the two
columns above in the gripper-camera image, using the wrist camera's per-frame
recorded pose and intrinsics.  Frames where the target does not project inside
the gripper image (or where the dataset carries no gripper-camera columns) are
NaN.

Projection uses per-frame intrinsics and camera pose from the dataset when
available (``observation.camera_intrinsics.camera:front_camera`` and
``observation.camera_pose.camera:front_camera``), via
:func:`resolve_frame_camera_model`.  Falls back to nominal front-camera
constants only when those columns are missing.

Usage::

    annotate-tamp-target-xy /path/to/dataset -j 64
    annotate-tamp-target-xy mar16_red --skip-existing
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from blocksuite.data_annotation.annotation_utils import (
    _PUSH_HL_ACTIONS,
    LIFT_LL_ACTIONS,
    SEGMENT_END_TARGET_LABELS,
    FrameCameraModel,
    build_action_goal_str,
    cardinal_direction,
    resolve_frame_camera_model,
    resolve_gripper_camera_model,
)

_ACTION_GOAL_STR_COL = "goal.action"
_SECONDARY_ACTION_COL = "goal.action_secondary"
_TARGET_XY_COL = "goal.position"
_TARGET2_XY_COL = "goal.position_target"
_TARGET_XY_GRIPPER_COL = "goal.position_gripper"
_TARGET2_XY_GRIPPER_COL = "goal.position_target_gripper"
_HL_PROMPT_COL = "goal.hl_prompt"
_PIXEL_ARROW_TYPE = pa.list_(pa.float32(), 2)

# The recorder writes the front-camera intrinsics under a feature key that carries
# the sensor prefix; older datasets used the bare name.  Checking only the bare
# name silently fell back to _DEFAULT_INTRINSICS for every modern dataset.
_hl_prompt_value: str | None = None

def _extract_target_xy(
    frame_idx: int,
    obj_name: str,
    pixel_arrays: dict[str, np.ndarray],
) -> tuple[float, float]:
    """Return (u, v) for *obj_name* at *frame_idx*, or (NaN, NaN)."""
    if obj_name in pixel_arrays:
        arr = pixel_arrays[obj_name]
        if frame_idx < len(arr):
            u, v = float(arr[frame_idx, 0]), float(arr[frame_idx, 1])
            if np.isfinite(u) and np.isfinite(v):
                return u, v
    return float("nan"), float("nan")


def compute_pixel_projections(
    df: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, np.ndarray]]:
    """Compute (u, v) pixel projections for every tracked object on every frame.

    Returns:
        ``(pixel_arrays, obj_names, world_positions)`` where *pixel_arrays* maps
        an object name to an (N, 2) float32 array of pixels and *world_positions*
        maps it to an (N, 3) float64 array of world-frame xyz (NaN where the pose
        is missing).  The world positions feed the push/align direction.
    """
    obj_pose_cols = sorted(
        c for c in df.columns if c.startswith("observation.state.object_pose.")
    )
    if not obj_pose_cols:
        return {}, [], {}

    obj_names = [c.split("observation.state.object_pose.")[-1] for c in obj_pose_cols]
    n = len(df)
    camera = resolve_frame_camera_model(df)

    pixel_arrays: dict[str, np.ndarray] = {}
    world_positions: dict[str, np.ndarray] = {}

    for name, pose_col_name in zip(obj_names, obj_pose_cols):
        raw = df[pose_col_name].tolist()
        poses = np.full((n, 7), np.nan, dtype=np.float64)
        valid_mask = np.zeros(n, dtype=bool)
        for idx, val in enumerate(raw):
            if val is not None and hasattr(val, "__len__") and len(val) == 7:
                try:
                    poses[idx] = val
                    valid_mask[idx] = True
                except (ValueError, TypeError):
                    pass

        result = np.full((n, 2), np.nan, dtype=np.float32)
        if valid_mask.any():
            frames = np.flatnonzero(valid_mask)
            result[valid_mask] = camera.project(poses[valid_mask, :3], frames)

        pixel_arrays[name] = result
        world_positions[name] = poses[:, :3]

    return pixel_arrays, obj_names, world_positions


def _cell_to_str(raw: object) -> str:
    """Normalise one parquet cell to a stripped string (empty for null/NaN)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return ""
    return str(raw).strip()


def _segment_push_direction(
    start: int,
    stop: int,
    obj_name: str,
    target_name: str,
    world_positions: dict[str, np.ndarray],
) -> str | None:
    """Cardinal direction of *obj_name* relative to *target_name* over one segment.

    Evaluated at the last frame in ``[start, stop)`` where both poses are finite,
    i.e. the placement the push actually achieved.  Resolving it once per segment
    keeps the label constant for the whole action — evaluating per frame would
    flip it mid-push as the block crosses the target's axis.
    """
    obj = world_positions.get(obj_name)
    tgt = world_positions.get(target_name)
    if obj is None or tgt is None:
        return None
    last = min(stop, len(obj), len(tgt))
    for i in range(last - 1, start - 1, -1):
        direction = cardinal_direction(obj[i], tgt[i])
        if direction is not None:
            return direction
    return None


def compute_action_goals(
    df: pd.DataFrame,
    pixel_arrays: dict[str, np.ndarray],
    world_positions: dict[str, np.ndarray] | None = None,
    gripper_camera: FrameCameraModel | None = None,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute the goal columns for every frame.

    Returns:
        ``(action_strings, secondary_strings, target_positions, target2_positions,
        gripper_target_positions, gripper_target2_positions)`` where all four
        position arrays are (N, 2) float32.  *target2_positions* holds the
        second point of a push/align annotation and is NaN elsewhere.  The
        gripper arrays hold the same target points projected into the
        gripper-camera image, NaN wherever the point is not visible there or
        *gripper_camera* is None.
    """
    n = len(df)
    result = [""] * n
    secondary_result = [""] * n
    target_xy = np.full((n, 2), np.nan, dtype=np.float32)
    target2_xy = np.full((n, 2), np.nan, dtype=np.float32)
    gripper_target_xy = np.full((n, 2), np.nan, dtype=np.float32)
    gripper_target2_xy = np.full((n, 2), np.nan, dtype=np.float32)
    # World-frame target points, gathered per frame and projected through the
    # gripper camera in one batch after the main loop.
    gripper_pts = np.full((n, 3), np.nan, dtype=np.float64)
    gripper_pts2 = np.full((n, 3), np.nan, dtype=np.float64)
    world_positions = world_positions or {}

    has_hl = "action.high_level_action" in df.columns
    has_ll = "action.low_level_action" in df.columns
    has_args = "action.high_level_args" in df.columns

    if not (has_hl and has_args):
        return result, secondary_result, target_xy, target2_xy, gripper_target_xy, gripper_target2_xy

    hl_values = df["action.high_level_action"].values
    ll_values = df["action.low_level_action"].values if has_ll else None
    args_values = df["action.high_level_args"].values

    hl_actions = [_cell_to_str(hl_values[i]) for i in range(n)]
    ll_actions = [_cell_to_str(ll_values[i]) if ll_values is not None else "" for i in range(n)]
    args_raw = [_cell_to_str(args_values[i]) for i in range(n)]

    parsed_args: list[list] = []
    for raw in args_raw:
        try:
            args = json.loads(raw) if raw else []
        except (json.JSONDecodeError, TypeError):
            args = []
        parsed_args.append(args if isinstance(args, list) else [])

    # Walk contiguous (hl_action, args) runs once: the push/align direction and
    # the "where did the block end up" place target are both properties of the
    # whole high-level action, not of the individual frame.
    push_dirs: list[str | None] = [None] * n
    seg_last: list[int] = list(range(n))
    seg_start = 0
    for i in range(1, n + 1):
        if i < n and (hl_actions[i], args_raw[i]) == (hl_actions[seg_start], args_raw[seg_start]):
            continue
        hl_action = hl_actions[seg_start]
        args = parsed_args[seg_start]
        direction = None
        if hl_action in _PUSH_HL_ACTIONS and len(args) > 2:
            direction = _segment_push_direction(
                seg_start, i, str(args[1]), str(args[2]), world_positions
            )
        for j in range(seg_start, i):
            push_dirs[j] = direction
            seg_last[j] = i - 1
        seg_start = i

    # The post-grasp lift of a pnp carries the block with the gripper, so its
    # per-frame pixel would drift upward through the segment. The goal is where
    # the block was grasped: resolve the whole lift at its first frame, i.e. the
    # block's pre-lift pose. Lift runs are contiguous (hl_action, args, ll_action).
    lift_seg_first: list[int] = list(range(n))
    seg_start = 0
    for i in range(1, n + 1):
        if i < n and (hl_actions[i], args_raw[i], ll_actions[i]) == (
            hl_actions[seg_start],
            args_raw[seg_start],
            ll_actions[seg_start],
        ):
            continue
        for j in range(seg_start, i):
            lift_seg_first[j] = seg_start
        seg_start = i

    for i in range(n):
        hl_action = hl_actions[i]
        if not hl_action:
            continue

        primary_action, secondary_action, target_obj, target_obj2 = build_action_goal_str(
            hl_action, ll_actions[i], parsed_args[i], push_direction=push_dirs[i]
        )

        result[i] = primary_action
        secondary_result[i] = secondary_action

        if target_obj is not None:
            # Placing at a table location has no destination object, so the goal
            # is where the held block comes to rest at the end of the action.
            if primary_action in SEGMENT_END_TARGET_LABELS:
                frame = seg_last[i]
            # The pnp lift targets the held block at its pre-lift (grasp) pose.
            elif hl_action.startswith("pnp_") and ll_actions[i] in LIFT_LL_ACTIONS:
                frame = lift_seg_first[i]
            else:
                frame = i
            target_xy[i] = _extract_target_xy(frame, target_obj, pixel_arrays)
            wp = world_positions.get(target_obj)
            if wp is not None and frame < len(wp):
                gripper_pts[i] = wp[frame]
        if target_obj2 is not None:
            target2_xy[i] = _extract_target_xy(i, target_obj2, pixel_arrays)
            wp = world_positions.get(target_obj2)
            if wp is not None and i < len(wp):
                gripper_pts2[i] = wp[i]

    # The gripper camera moves with the arm, so the target must be projected
    # through the *current* frame's camera pose even when the target point
    # itself comes from the segment-end frame — a lookup into segment-end
    # gripper pixels would pair a future position with a future camera pose.
    if gripper_camera is not None:
        for pts, out in ((gripper_pts, gripper_target_xy), (gripper_pts2, gripper_target2_xy)):
            valid = np.isfinite(pts).all(axis=1)
            if valid.any():
                frames = np.flatnonzero(valid)
                out[valid] = gripper_camera.project_visible(pts[valid], frames)

    return result, secondary_result, target_xy, target2_xy, gripper_target_xy, gripper_target2_xy


class SkippedParquet(Exception):
    """A parquet file could not be annotated.

    Raised rather than returned so the failure reaches the caller's error list
    instead of being indistinguishable from a successful write.
    """


def _process_single_parquet(pf: Path) -> tuple[Path, int] | None:
    """Process one parquet file: compute goal columns, write back.

    Raises:
        SkippedParquet: the file is unreadable or carries no object poses, so no
            goal columns could be written.
    """
    if pf.stat().st_size < 12:
        raise SkippedParquet("file is empty")
    try:
        table = pq.read_table(pf)
    except Exception as exc:
        raise SkippedParquet(f"unreadable parquet: {exc}") from exc
    df = table.to_pandas()
    pixel_arrays, obj_names, world_positions = compute_pixel_projections(df)
    if not obj_names:
        raise SkippedParquet("no observation.state.object_pose.* columns")

    gripper_camera = resolve_gripper_camera_model(df)
    (
        action_strs,
        secondary_strs,
        target_xy,
        target2_xy,
        gripper_target_xy,
        gripper_target2_xy,
    ) = compute_action_goals(
        df, pixel_arrays, world_positions, gripper_camera=gripper_camera
    )

    # build_action_goal_str drops frames whose hl args contradict their hl action.
    # That is a handful of boundary frames normally; a large share means the arg
    # counts in _HL_ARG_COUNTS do not match the domain this data came from.
    hl_col = df.get("action.high_level_action")
    if hl_col is not None:
        has_hl = hl_col.notna().to_numpy() & (hl_col.astype(str).str.strip() != "").to_numpy()
        labelled = np.fromiter((bool(s) for s in action_strs), dtype=bool, count=len(action_strs))
        with_action = int(has_hl.sum())
        dropped = int((has_hl & ~labelled).sum())
        if with_action and dropped > with_action * 0.2:
            tqdm.write(
                f"  WARNING: {pf.name}: {dropped}/{with_action} frames left unlabelled "
                f"(hl args contradict hl action) — check _HL_ARG_COUNTS against this domain"
            )

    for col in [
        _ACTION_GOAL_STR_COL, _SECONDARY_ACTION_COL, _TARGET_XY_COL, _TARGET2_XY_COL,
        _TARGET_XY_GRIPPER_COL, _TARGET2_XY_GRIPPER_COL,
        _HL_PROMPT_COL,
        "action_goal_str", "target_xy", "goal.action_secondary", "goal.position_secondary",
    ]:
        if col in table.column_names:
            table = table.remove_column(table.column_names.index(col))

    table = table.append_column(
        _ACTION_GOAL_STR_COL, pa.array(action_strs, type=pa.string())
    )
    table = table.append_column(
        _SECONDARY_ACTION_COL, pa.array(secondary_strs, type=pa.string())
    )
    for col_name, xy in (
        (_TARGET_XY_COL, target_xy),
        (_TARGET2_XY_COL, target2_xy),
        (_TARGET_XY_GRIPPER_COL, gripper_target_xy),
        (_TARGET2_XY_GRIPPER_COL, gripper_target2_xy),
    ):
        flat_values = pa.array(xy.flatten(), type=pa.float32())
        fsl_array = pa.FixedSizeListArray.from_arrays(flat_values, list_size=2)
        table = table.append_column(col_name, fsl_array)

    if _hl_prompt_value is not None:
        n = table.num_rows
        table = table.append_column(
            _HL_PROMPT_COL, pa.array([_hl_prompt_value] * n, type=pa.string())
        )

    pq.write_table(table, str(pf), compression="snappy")
    return (pf, len(df))


def _update_info_json(blocks_dir: Path) -> None:
    """Add goal columns to meta/info.json if missing."""
    info_path = blocks_dir / "meta" / "info.json"
    if not info_path.exists():
        return
    with open(info_path) as f:
        info = json.load(f)
    features = info.get("features", {})
    changed = False

    for old in ("action_goal_str", "target_xy", "goal.position_secondary"):
        if old in features:
            del features[old]
            changed = True

    if _ACTION_GOAL_STR_COL not in features:
        features[_ACTION_GOAL_STR_COL] = {"dtype": "string", "shape": [1], "names": None}
        changed = True
    if _SECONDARY_ACTION_COL not in features:
        features[_SECONDARY_ACTION_COL] = {"dtype": "string", "shape": [1], "names": None}
        changed = True
    for col in (_TARGET_XY_COL, _TARGET2_XY_COL, _TARGET_XY_GRIPPER_COL, _TARGET2_XY_GRIPPER_COL):
        if col not in features:
            features[col] = {
                "dtype": "float32",
                "shape": [2],
                "names": {"axes": ["x", "y"]},
            }
            changed = True
    if _hl_prompt_value is not None and _HL_PROMPT_COL not in features:
        features[_HL_PROMPT_COL] = {"dtype": "string", "shape": [1], "names": None}
        changed = True
    if changed:
        info["features"] = features
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
            f.write("\n")


def _blocks_dir(traj_dir: Path) -> Path:
    candidate = traj_dir / "blocks"
    return candidate if candidate.exists() else traj_dir


def _is_annotated(traj_dir: Path) -> bool:
    data_dir = _blocks_dir(traj_dir) / "data"
    first_pq = next(data_dir.glob("chunk-*/file-*.parquet"), None)
    if first_pq is None:
        return False
    schema = pq.read_schema(first_pq)
    return all(
        col in schema.names
        for col in (
            _TARGET_XY_COL, _TARGET2_XY_COL, _SECONDARY_ACTION_COL,
            _TARGET_XY_GRIPPER_COL, _TARGET2_XY_GRIPPER_COL,
        )
    )


# Set by main() before any worker forks, mirroring _hl_prompt_value.
_allow_nominal_camera = False


def _front_camera_columns_status(blocks_dir: Path, sample_parquet: Path) -> tuple[str, str]:
    """Can the front-camera projection be trusted for this dataset?

    goal.position is a PIXEL: an object's world pose projected through the front camera.
    resolve_frame_camera_model falls back to module constants when the parquet has no camera
    columns -- silently, and independently for intrinsics and pose, so a dataset can be half
    fallen back while still reporting `recorded`. Those constants describe a ~37.7 deg vertical
    FOV (fx 527.36); the scenes these datasets render use ~49 deg (fx ~394). The fallback
    therefore displaces the crosshair by tens of pixels, and four datasets shipped that way
    before anyone watched the overlay videos.

    Returns (status, detail):
      "ok"        recorded pose AND intrinsics are present for the front camera
      "no_video"  the dataset ships no camera video at all, so a pixel goal means nothing
      "missing"   there ARE video frames but no record of the camera that produced them
    """
    try:
        info = json.loads((blocks_dir / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        info = {}
    feats = info.get("features") or {}
    video_cams = sorted({k.split("camera:")[-1] for k, v in feats.items()
                         if isinstance(v, dict) and v.get("dtype") == "video"
                         and k.startswith("observation.images.camera:")})
    try:
        cols = set(pq.ParquetFile(sample_parquet).schema_arrow.names)
    except Exception as e:                                    # noqa: BLE001
        # A read error is not evidence of a camera problem; let the normal path report it.
        return "ok", f"could not read {sample_parquet.name}: {e}"
    have_pose = "observation.camera_pose.camera:front_camera" in cols
    have_k = "observation.camera_intrinsics.camera:front_camera" in cols
    if have_pose and have_k:
        return "ok", ""
    if not video_cams:
        return "no_video", ("the dataset declares no camera video, so a pixel goal.position "
                            "would point into a frame that does not exist")
    miss = " or ".join(n for n, ok in (("pose", have_pose), ("intrinsics", have_k)) if not ok)
    return "missing", (
        f"video is present for {video_cams} but the parquet carries no front-camera {miss} "
        f"column, so goal.position would be projected through the NOMINAL fallback camera "
        f"(fx 527.36 and one fixed pose) rather than the camera that rendered these frames. "
        f"An execution that recorded no cameras writes no camera columns; the re-render that "
        f"produced the video must supply them."
    )


def process_trajectory(traj_dir: str | Path, num_workers: int = 1) -> bool:
    blocks_dir = _blocks_dir(Path(traj_dir))
    data_dir = blocks_dir / "data"

    if not data_dir.exists():
        tqdm.write(f"  No data/ directory in {traj_dir}")
        return False

    parquet_files = sorted(data_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        tqdm.write(f"  No parquet files in {data_dir}")
        return False

    cam_status, cam_detail = _front_camera_columns_status(blocks_dir, parquet_files[0])
    if cam_status != "ok":
        if not _allow_nominal_camera:
            tqdm.write(f"  ERROR: {traj_dir}: {cam_detail}")
            tqdm.write("         Refusing to write pixel goals from a camera this dataset does "
                       "not record. Pass --allow-nominal-camera to annotate anyway (correct "
                       "ONLY for the legacy table / color_only_blocks_table presets).")
            return False
        tqdm.write(f"  WARNING: {traj_dir}: {cam_detail}"
                   " -- proceeding because --allow-nominal-camera was given")

    pbar = tqdm(total=len(parquet_files), desc="Parquet", unit="file", leave=False)
    failed: list[tuple[Path, str]] = []
    if num_workers > 1 and len(parquet_files) > 1:
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {
                pool.submit(_process_single_parquet, pf): pf for pf in parquet_files
            }
            for future in as_completed(futures):
                pf = futures[future]
                try:
                    future.result()
                except Exception as e:
                    failed.append((pf, str(e)))
                pbar.update()
    else:
        for pf in parquet_files:
            try:
                _process_single_parquet(pf)
            except Exception as e:
                failed.append((pf, str(e)))
            pbar.update()
    pbar.close()

    annotated = len(parquet_files) - len(failed)
    if failed:
        tqdm.write(f"  WARNING: {traj_dir}: {len(failed)}/{len(parquet_files)} files skipped:")
        for pf, err in failed[:5]:
            tqdm.write(f"    {pf.name}: {err[:120]}")

    if annotated == 0:
        # Declaring the goal features here would leave info.json advertising
        # columns that no parquet actually has.
        tqdm.write(f"  ERROR: {traj_dir}: no files annotated, leaving info.json untouched")
        return False

    _update_info_json(blocks_dir)
    return True


_DEFAULT_DATASETS_ROOT = Path(__file__).resolve().parents[3] / "datasets" / "unmerged"


def _resolve_path(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p.resolve()
    candidate = _DEFAULT_DATASETS_ROOT / raw
    if candidate.exists():
        return candidate
    print(f"Path not found: {raw}")
    print(f"  (also tried {candidate})")
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate action_goal_str and target_xy (lightweight, no video)"
    )
    parser.add_argument(
        "path",
        help="Path to dataset dir or single trajectory, or name under datasets/unmerged/",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip trajectories that already have target_xy column",
    )
    parser.add_argument(
        "--workers", "-j",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--hl-prompt",
        type=str,
        default=None,
        help="High-level prompt string to write as goal.hl_prompt (e.g. 'stack 3 blocks')",
    )
    parser.add_argument(
        "--allow-nominal-camera",
        action="store_true",
        help="Annotate even when the dataset records no front-camera pose/intrinsics. OFF by "
             "default: the nominal fallback describes a ~37.7 deg-FOV camera that these scenes "
             "do not use, so the pixel goals come out wrong by tens of pixels. Only correct for "
             "the legacy table / color_only_blocks_table presets.",
    )
    args = parser.parse_args()

    global _hl_prompt_value, _allow_nominal_camera
    _hl_prompt_value = args.hl_prompt
    _allow_nominal_camera = args.allow_nominal_camera

    num_workers = args.workers or os.cpu_count() or 1
    path = _resolve_path(args.path)

    def _is_dataset_root(p: Path) -> bool:
        return (p / "blocks" / "data").exists() or (p / "data").exists()

    if _is_dataset_root(path):
        print(f"Processing single dataset: {path.name}  (workers={num_workers})")
        if args.skip_existing and _is_annotated(path):
            print("  Already annotated, skipping (--skip-existing)")
        elif not process_trajectory(path, num_workers=num_workers):
            print(f"ERROR: {path.name} was not annotated")
            sys.exit(1)
    else:
        traj_dirs = sorted(
            d for d in path.iterdir() if d.is_dir() and _is_dataset_root(d)
        )
        if not traj_dirs:
            print(f"No trajectory folders found in {path}")
            sys.exit(1)

        to_process = []
        skipped = 0
        for traj_dir in traj_dirs:
            if args.skip_existing and _is_annotated(traj_dir):
                skipped += 1
            else:
                to_process.append(traj_dir)

        print(
            f"Found {len(traj_dirs)} trajectories in {path.name}"
            f" ({len(to_process)} to process, {skipped} skipped, workers={num_workers})"
        )
        t0 = time.monotonic()

        pbar = tqdm(total=len(to_process), desc="Trajectories", unit="traj")
        unannotated: list[str] = []
        if num_workers > 1 and len(to_process) > 1:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = {
                    pool.submit(process_trajectory, td, 1): td for td in to_process
                }
                for future in as_completed(futures):
                    td = futures[future]
                    try:
                        if not future.result():
                            unannotated.append(td.name)
                    except Exception as e:
                        tqdm.write(f"FAILED: {td.name}: {e}")
                        unannotated.append(td.name)
                    pbar.update()
        else:
            for traj_dir in to_process:
                if not process_trajectory(traj_dir, num_workers=num_workers):
                    unannotated.append(traj_dir.name)
                pbar.update()
        pbar.close()

        elapsed = time.monotonic() - t0
        ok = len(to_process) - len(unannotated)
        print(
            f"\nDone: {ok} annotated, {len(unannotated)} not annotated, "
            f"{skipped} skipped in {elapsed / 60:.1f}m"
        )
        if unannotated:
            # Exiting non-zero so callers do not treat a no-op run as success.
            print(f"ERROR: {len(unannotated)} trajectory/ies produced no goal columns:")
            for name in unannotated[:10]:
                print(f"  {name}")
            sys.exit(1)


if __name__ == "__main__":
    main()
