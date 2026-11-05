#!/usr/bin/env python3
"""Repair a wrong front-camera pose in recorded datasets, then re-annotate.

Datasets recorded before the camera-pose recorder fix (sim2real / fr3 cameras
using IsaacLab's default ``update_latest_camera_pose=False``) logged a stale
init-time ``observation.camera_pose.camera:front_camera`` that does not match the
camera which actually rendered the frames. That makes the projected
``goal.position`` / ``goal.position_target`` pixels systematically off (~8-13px
at table height, worse for elevated objects).

This tool overwrites the front-camera pose column with the correct calibrated
pose (from the sim2real scene's OffsetCfg, converted to the world-convention the
annotator expects) and re-runs ``annotate_target_xy.py`` so the goal pixels are
recomputed correctly.

The correct pose defaults to the current sim2real front camera calibration; pass
``--offset-pos`` / ``--offset-quat-ros`` to match a different camera OffsetCfg.

Usage:
    # a single split dir (has data/), or a dataset root (has blocks_success/ ...)
    python scripts/datagen/fix_camera_pose_annotations.py <path> [--splits blocks_success]
    python scripts/datagen/fix_camera_pose_annotations.py <path> --no-annotate
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

# Default = sim2real front camera OffsetCfg (src/toyblocks/scenes/sim2real.py).
# convention="ros" => the quaternion orients the OPTICAL frame in the base frame.
DEFAULT_OFFSET_POS = (0.944958, 0.000714, 0.48414)
DEFAULT_OFFSET_QUAT_ROS_WXYZ = (-0.3265056, 0.6272114, 0.6272114, -0.3265056)

FRONT_POSE_COL = "observation.camera_pose.camera:front_camera"


def corrected_world_pose(offset_pos, offset_quat_ros_wxyz) -> np.ndarray:
    """Convert a ros-convention camera OffsetCfg to the [x,y,z,qw,qx,qy,qz] pose
    (world convention: x fwd, y left, z up) that the annotator's
    ``extrinsics_from_camera_pose`` expects.

    The annotator computes ``R_cw = (Rq_world @ W2O).T`` and needs that to equal
    ``R_parent_from_optical.T``; solving gives ``Rq_world = R_parent_from_optical @ W2O.T``.
    """
    from toyblocks.data_annotation.annotation_utils import _WORLD_TO_OPTICAL

    q = np.asarray(offset_quat_ros_wxyz, dtype=float)
    R_parent_from_optical = Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()
    Rq_world = R_parent_from_optical @ _WORLD_TO_OPTICAL.T
    q_world_xyzw = Rotation.from_matrix(Rq_world).as_quat()
    q_world_wxyz = q_world_xyzw[[3, 0, 1, 2]]
    return np.concatenate([np.asarray(offset_pos, float), q_world_wxyz]).astype(np.float32)


def _iter_split_dirs(path: Path, splits: list[str] | None):
    """Yield split dirs (each containing data/) under *path*."""
    if (path / "data").is_dir():
        yield path
        return
    candidates = splits or ["blocks_success", "blocks_failure"]
    found = False
    for name in candidates:
        d = path / name
        if (d / "data").is_dir():
            found = True
            yield d
    if not found:
        # dataset-of-datasets: recurse one level (e.g. execution/<problem>/blocks_*)
        for sub in sorted(p for p in path.iterdir() if p.is_dir()):
            yield from _iter_split_dirs(sub, splits)


def fix_split(split_dir: Path, pose: np.ndarray, backup: bool) -> int:
    """Overwrite the front-cam pose column in every parquet of *split_dir*."""
    parquet_files = sorted((split_dir / "data").glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        print(f"  [skip] no parquet under {split_dir}/data")
        return 0
    n_fixed = 0
    for pf in parquet_files:
        t = pq.read_table(pf)
        if FRONT_POSE_COL not in t.column_names:
            print(f"  [skip] {pf.name}: no {FRONT_POSE_COL}")
            continue
        idx = t.column_names.index(FRONT_POSE_COL)
        old = np.asarray(t.column(idx)[0].as_py(), dtype=float)
        if backup:
            bak = pf.with_suffix(pf.suffix + ".orig_bak")
            if not bak.exists():
                shutil.copy2(pf, bak)
        flat = pa.array(np.tile(pose, t.num_rows), type=pa.float32())
        # Cast to the column's existing type so we match either a
        # fixed_size_list<..>[7] or a variable-length list<element: float>
        # (merged/stitched datasets store the pose as a plain list).
        newcol = pa.FixedSizeListArray.from_arrays(flat, 7).cast(t.schema.field(idx).type)
        t2 = t.set_column(idx, t.schema.field(idx), newcol)
        pq.write_table(t2, pf)
        print(f"  [fix]  {pf.name}: {np.round(old[:3],3)} -> {np.round(pose[:3],3)}")
        n_fixed += 1
    return n_fixed


def annotate(split_dir: Path, workers: int) -> None:
    import toyblocks.data_annotation.annotate_target_xy as m

    cmd = [sys.executable, m.__file__, str(split_dir), "-j", str(workers)]
    print(f"  [annotate] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("path", help="split dir (has data/) or dataset root (has blocks_success/ ...)")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="split subdir names to fix (default: blocks_success + blocks_failure if present)")
    ap.add_argument("--offset-pos", nargs=3, type=float, default=DEFAULT_OFFSET_POS,
                    metavar=("X", "Y", "Z"), help="camera OffsetCfg pos (default: sim2real front cam)")
    ap.add_argument("--offset-quat-ros", nargs=4, type=float, default=DEFAULT_OFFSET_QUAT_ROS_WXYZ,
                    metavar=("W", "X", "Y", "Z"), help="camera OffsetCfg rot, wxyz, convention=ros")
    ap.add_argument("--annotate", action=argparse.BooleanOptionalAction, default=True,
                    help="re-run annotate_target_xy.py after fixing the pose (default: on)")
    ap.add_argument("--workers", "-j", type=int, default=8, help="annotation workers")
    ap.add_argument("--no-backup", dest="backup", action="store_false",
                    help="do not write a .orig_bak copy before overwriting")
    args = ap.parse_args()

    pose = corrected_world_pose(args.offset_pos, args.offset_quat_ros)
    print(f"corrected front-cam pose [x,y,z,qw,qx,qy,qz] = {np.round(pose,5).tolist()}\n")

    split_dirs = list(_iter_split_dirs(Path(args.path), args.splits))
    if not split_dirs:
        print(f"No split dirs with data/ found under {args.path}")
        return 1

    for sd in split_dirs:
        print(f"== {sd} ==")
        if fix_split(sd, pose, args.backup) and args.annotate:
            annotate(sd, args.workers)
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
