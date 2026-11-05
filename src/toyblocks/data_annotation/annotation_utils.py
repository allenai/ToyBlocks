"""
Shared utilities for front-camera annotation and action-goal string encoding.

Used by annotate_blocks_all_frames.py, visualize_annotations.py, and inference.py.

Camera helpers
--------------
FRONT_CAMERA_POS, FRONT_CAMERA_QUAT_WXYZ, R_CW, _DEFAULT_INTRINSICS
project_world_to_pixel(point_world, cam_pos, R_cw, intrinsics) -> np.ndarray([u,v])

Action-goal encoding
--------------------
_parse_reorient_direction(direction_raw) -> (verb, direction)
cardinal_direction(obj_pos, target_pos) -> 'front' | 'back' | 'left' | 'right' | None
build_action_goal_str(hl_action, ll_action, hl_args, push_direction)
    -> (goal_action, secondary_goal_action, target_obj, target_obj2)
annotate_frame_with_uv(frame_rgb, uv) -> annotated RGB copy
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

# ---------------------------------------------------------------------------
# Front-camera constants (single source of truth for all annotation code)
# ---------------------------------------------------------------------------

FRONT_CAMERA_POS = np.array([0.944958, 0.000714, 0.48414])
FRONT_CAMERA_QUAT_WXYZ = np.array([-0.3265056, 0.6272114, 0.6272114, -0.3265056])

# Default (fx, fy, cx, cy) used when the dataset lacks a camera_intrinsics column.
_DEFAULT_INTRINSICS: tuple[float, float, float, float] = (527.36206055, 527.36206055, 320.0, 180.0)

# Column name variants across dataset formats (colon vs dot notation).
_FRONT_CAM_INTRINSICS_COLS = [
    "observation.camera_intrinsics.camera:front_camera",
    "observation.camera_intrinsics.front_camera",
]
_FRONT_CAM_POSE_COLS = [
    "observation.camera_pose.camera:front_camera",
    "observation.camera_pose.front_camera",
]


def find_front_camera_intrinsics_col(df_columns) -> str | None:
    """Return the front-camera intrinsics column name present in *df_columns*, or None."""
    col_set = set(df_columns)
    for col in _FRONT_CAM_INTRINSICS_COLS:
        if col in col_set:
            return col
    return None


def find_front_camera_pose_col(df_columns) -> str | None:
    """Return the front-camera pose column name present in *df_columns*, or None."""
    col_set = set(df_columns)
    for col in _FRONT_CAM_POSE_COLS:
        if col in col_set:
            return col
    return None


def prepare_dataframe_for_video_sync(df: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by global ``index`` so ``df.iloc[i]`` aligns with video frame *i*."""
    if "index" in df.columns:
        return df.sort_values("index").reset_index(drop=True)
    return df.reset_index(drop=True)


def slice_episode_rows(
    df: pd.DataFrame,
    from_index: int,
    to_index: int,
) -> pd.DataFrame:
    """Return one episode's rows from a parquet dataframe.

    ``dataset_from_index`` / ``dataset_to_index`` in ``meta/episodes`` refer to
    the global ``index`` column, not parquet iloc.  Merged datasets (e.g.
    ``bigblocks_flip_a``) can have ``df.iloc[0]["index"] != 0``; using iloc
    with meta offsets misaligns video by tens of frames.
    """
    if "index" in df.columns:
        mask = (df["index"] >= from_index) & (df["index"] <= to_index)
        return df.loc[mask].sort_values("index").reset_index(drop=True)
    return df.iloc[from_index : to_index + 1].reset_index(drop=True)


def load_lerobot_episode(
    dataset_root: Path,
    episode_index: int,
    *,
    front_cam_key: str = "observation.images.camera:front_camera",
) -> tuple[pd.DataFrame, Path, pd.Series]:
    """Load episode parquet rows and front-camera video path from a LeRobot dataset."""
    import pyarrow.parquet as pq

    dataset_root = Path(dataset_root)
    meta_dir = dataset_root / "meta" / "episodes"
    meta_files = sorted(meta_dir.glob("*/*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"No meta/episodes parquet under {dataset_root}")
    meta = pd.concat([pq.read_table(p).to_pandas() for p in meta_files], ignore_index=True)
    match = meta[meta["episode_index"] == episode_index]
    if match.empty:
        raise ValueError(f"Episode {episode_index} not found in {dataset_root}")

    ep_row = match.iloc[0]
    data_chunk = int(ep_row["data/chunk_index"])
    data_file = int(ep_row["data/file_index"])
    from_idx = int(ep_row["dataset_from_index"])
    to_idx = int(ep_row["dataset_to_index"])

    pf_path = dataset_root / "data" / f"chunk-{data_chunk:03d}" / f"file-{data_file:03d}.parquet"
    df = pq.read_table(pf_path).to_pandas()
    ep_data = slice_episode_rows(df, from_idx, to_idx)

    vchunk = int(ep_row[f"videos/{front_cam_key}/chunk_index"])
    vfile = int(ep_row[f"videos/{front_cam_key}/file_index"])
    video_path = (
        dataset_root / "videos" / front_cam_key
        / f"chunk-{vchunk:03d}" / f"file-{vfile:03d}.mp4"
    )
    if not video_path.exists():
        raise FileNotFoundError(f"Episode video not found: {video_path}")

    return ep_data, video_path, ep_row


def build_R_cw_from_quat_wxyz(quat_wxyz: np.ndarray) -> np.ndarray:
    """World-to-camera rotation matrix from a scalar-first wxyz quaternion."""
    q = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat_xyzw = np.array([q[1], q[2], q[3], q[0]])
    return Rotation.from_quat(quat_xyzw).as_matrix().T


def _build_R_cw() -> np.ndarray:
    """World-to-camera rotation matrix from the static front-camera quaternion."""
    return build_R_cw_from_quat_wxyz(FRONT_CAMERA_QUAT_WXYZ)


R_CW: np.ndarray = _build_R_cw()


# Isaac Lab records ``observation.camera_pose.*`` in the "world" convention
# (x forward, y left, z up).  The pinhole projection below needs the optical
# convention (x right, y down, z forward), so the camera axes get remapped:
# right = -left, down = -up, forward = forward.
_WORLD_TO_OPTICAL: np.ndarray = np.array(
    [[0.0, 0.0, 1.0],
     [-1.0, 0.0, 0.0],
     [0.0, -1.0, 0.0]]
)


# The recorder writes camera features under a key carrying the sensor prefix;
# older datasets used the bare name.  Readers must accept both — checking only
# the bare name silently fell back to the constants above for every modern
# dataset, which describe a different scene preset entirely.
INTRINSICS_COLUMNS: tuple[str, ...] = (
    "observation.camera_intrinsics.camera:front_camera",
    "observation.camera_intrinsics.front_camera",
)
CAMERA_POSE_COLUMNS: tuple[str, ...] = (
    "observation.camera_pose.camera:front_camera",
    "observation.camera_pose.front_camera",
)
GRIPPER_INTRINSICS_COLUMNS: tuple[str, ...] = (
    "observation.camera_intrinsics.camera:gripper_camera",
    "observation.camera_intrinsics.gripper_camera",
)
GRIPPER_CAMERA_POSE_COLUMNS: tuple[str, ...] = (
    "observation.camera_pose.camera:gripper_camera",
    "observation.camera_pose.gripper_camera",
)


def first_present_column(columns, candidates: tuple[str, ...]) -> str | None:
    """Return the first of *candidates* present in *columns*, or None."""
    available = set(columns)
    for col in candidates:
        if col in available:
            return col
    return None


class FrameCameraModel(NamedTuple):
    """Per-frame front-camera intrinsics and extrinsics for one episode file.

    All arrays are indexed by frame.  ``recorded`` is False when the dataset
    carried no camera columns and the legacy constants were used instead — those
    only describe the ``table`` / ``color_only_blocks_table`` scene presets.
    """

    fx: np.ndarray
    fy: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    cam_pos: np.ndarray
    R_cw: np.ndarray
    recorded: bool

    def project(self, points_world: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """Project world points to (u, v), one camera frame per point.

        Args:
            points_world: (M, 3) world-frame positions.
            frames:       (M,) frame indices selecting each point's camera.

        Returns:
            (M, 2) float32 pixels; NaN where the point is behind the camera.
        """
        points_world = np.asarray(points_world, dtype=np.float64).reshape(-1, 3)
        frames = np.asarray(frames, dtype=np.intp).reshape(-1)
        offset = points_world - self.cam_pos[frames]
        p_cam = np.einsum("nij,nj->ni", self.R_cw[frames], offset)
        z = p_cam[:, 2]
        behind = z <= 0
        z_safe = np.where(behind, 1.0, z)
        u = self.fx[frames] * p_cam[:, 0] / z_safe + self.cx[frames]
        v = self.fy[frames] * p_cam[:, 1] / z_safe + self.cy[frames]
        pixels = np.column_stack([u, v]).astype(np.float32)
        pixels[behind] = np.nan
        return pixels

    def project_visible(self, points_world: np.ndarray, frames: np.ndarray) -> np.ndarray:
        """Project like :meth:`project`, but NaN out pixels not visible in the image.

        A point counts as visible when it projects in front of the camera and
        inside the image bounds, on a frame with a usable calibration.  Image
        size is taken as ``(2*cx, 2*cy)`` — Isaac Lab cameras keep the principal
        point centered.  The recorder writes all-zero pose/intrinsics rows for
        frames where a camera sample was missing; those frames are treated as
        not visible.  Occlusion (a point hidden behind the gripper or another
        object) is not detected.
        """
        pixels = self.project(points_world, frames)
        frames = np.asarray(frames, dtype=np.intp).reshape(-1)
        width = 2.0 * self.cx[frames]
        height = 2.0 * self.cy[frames]
        invalid_frame = (self.fx[frames] <= 0) | np.all(self.cam_pos[frames] == 0.0, axis=1)
        with np.errstate(invalid="ignore"):
            outside = (
                (pixels[:, 0] < 0)
                | (pixels[:, 0] >= width)
                | (pixels[:, 1] < 0)
                | (pixels[:, 1] >= height)
            )
        pixels[invalid_frame | outside] = np.nan
        return pixels


def resolve_frame_camera_model(df) -> FrameCameraModel:
    """Build the per-frame front-camera model for one parquet file.

    Prefers the intrinsics and extrinsics the recorder stored per frame, and
    falls back to the module constants only when neither column exists.
    """
    n = len(df)
    intrinsics_col = first_present_column(df.columns, INTRINSICS_COLUMNS)
    if intrinsics_col is not None:
        K = np.array(df[intrinsics_col].tolist(), dtype=np.float64)
        fx, fy, cx, cy = K[:, 0], K[:, 4], K[:, 2], K[:, 5]
    else:
        fx_v, fy_v, cx_v, cy_v = _DEFAULT_INTRINSICS
        fx, fy = np.full(n, fx_v), np.full(n, fy_v)
        cx, cy = np.full(n, cx_v), np.full(n, cy_v)

    pose_col = first_present_column(df.columns, CAMERA_POSE_COLUMNS)
    cam_pos: np.ndarray | None = None
    R_cw: np.ndarray | None = None
    if pose_col is not None:
        try:
            cam_pos, R_cw = extrinsics_from_camera_pose(
                np.array(df[pose_col].tolist(), dtype=np.float64)
            )
        except (ValueError, TypeError):
            # Ragged or malformed pose column — fall back rather than fail.
            cam_pos, R_cw = None, None

    recorded = cam_pos is not None and R_cw is not None and len(cam_pos) == n
    if not recorded:
        cam_pos = np.broadcast_to(FRONT_CAMERA_POS, (n, 3))
        R_cw = np.broadcast_to(R_CW, (n, 3, 3))

    return FrameCameraModel(fx, fy, cx, cy, cam_pos, R_cw, recorded)


def resolve_gripper_camera_model(df) -> FrameCameraModel | None:
    """Build the per-frame gripper-camera model for one parquet file.

    The gripper camera rides the wrist, so unlike the front camera there is no
    nominal fallback pose: when the dataset lacks recorded gripper-camera pose
    or intrinsics columns this returns None and callers must leave
    gripper-frame annotations NaN.
    """
    n = len(df)
    intrinsics_col = first_present_column(df.columns, GRIPPER_INTRINSICS_COLUMNS)
    pose_col = first_present_column(df.columns, GRIPPER_CAMERA_POSE_COLUMNS)
    if intrinsics_col is None or pose_col is None:
        return None
    try:
        K = np.array(df[intrinsics_col].tolist(), dtype=np.float64).reshape(n, -1)
        cam_pos, R_cw = extrinsics_from_camera_pose(
            np.array(df[pose_col].tolist(), dtype=np.float64)
        )
    except (ValueError, TypeError):
        # Ragged or malformed columns — no usable model.
        return None
    if K.shape[1] < 6 or len(cam_pos) != n:
        return None
    fx, fy, cx, cy = K[:, 0], K[:, 4], K[:, 2], K[:, 5]
    return FrameCameraModel(fx, fy, cx, cy, cam_pos, R_cw, recorded=True)


def extrinsics_from_camera_pose(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert recorded camera poses into projection-ready extrinsics.

    Args:
        poses: (N, 7) array of ``[x, y, z, qw, qx, qy, qz]`` as written to
            ``observation.camera_pose.<cam>``, in Isaac Lab's world convention.

    Returns:
        ``(cam_pos, R_cw)`` with shapes (N, 3) and (N, 3, 3), where *R_cw*
        rotates a world-frame offset into the optical camera frame.
    """
    poses = np.asarray(poses, dtype=np.float64).reshape(-1, 7)
    cam_pos = poses[:, :3]
    quat_wxyz = poses[:, 3:7]
    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    # Guard against unnormalised / all-zero quaternions in short or corrupt rows.
    norms = np.linalg.norm(quat_xyzw, axis=1, keepdims=True)
    quat_xyzw = np.where(norms > 0, quat_xyzw / np.where(norms > 0, norms, 1.0), [0.0, 0.0, 0.0, 1.0])
    R_wc_world = Rotation.from_quat(quat_xyzw).as_matrix()
    R_cw = (R_wc_world @ _WORLD_TO_OPTICAL).transpose(0, 2, 1)
    return cam_pos, R_cw


# ---------------------------------------------------------------------------
# Projection helper
# ---------------------------------------------------------------------------

def project_world_to_pixel(
    point_world: np.ndarray,
    cam_pos: np.ndarray,
    R_cw: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> np.ndarray:
    """Project a 3-D world-frame point to (u, v) pixel coordinates.

    Args:
        point_world: (3,) position in world frame.
        cam_pos:     (3,) camera origin in world frame.
        R_cw:        (3, 3) world-to-camera rotation matrix.
        intrinsics:  (fx, fy, cx, cy).

    Returns:
        ``np.array([u, v], dtype=float32)``.  Returns ``[NaN, NaN]`` when
        the point is behind the camera (z ≤ 0).
    """
    p_cam = R_cw @ (point_world - cam_pos)
    if p_cam[2] <= 0:
        return np.array([np.nan, np.nan], dtype=np.float32)
    fx, fy, cx, cy = intrinsics
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return np.array([u, v], dtype=np.float32)


# ---------------------------------------------------------------------------
# Action-goal string encoding helpers
# ---------------------------------------------------------------------------

# The PDDL domain's direction constants, keyed by the action that consumes them.
# ``reorient`` takes a flip_dir, ``rotate`` takes a rot_dir.
_FLIP_DIRECTIONS: dict[str, str] = {
    "forward": "forward",
    "backward": "backward",
    "left_flip": "left",
    "right_flip": "right",
}
_ROTATE_DIRECTIONS: dict[str, str] = {"cw": "cw", "ccw": "ccw"}

_DIRECTION_VOCABULARY: dict[str, tuple[str, dict[str, str]]] = {
    "reorient": ("flip", _FLIP_DIRECTIONS),
    "rotate": ("rotate", _ROTATE_DIRECTIONS),
}


def _parse_reorient_direction(hl_action: str, direction_raw: str) -> tuple[str, str] | None:
    """Parse the direction argument of a reorient/rotate action.

    The verb comes from *hl_action*, never from the argument: deriving it from
    the string used to label a ``rotate`` whose direction read ``forward`` as a
    flip.  The direction itself must be one of the domain constants for that
    action's direction type, so a block name landing in the direction slot (from
    a desynced arg list) yields ``None`` instead of ``flip <block_name>``.

    ``'right_flip'`` → ``('flip', 'right')``;  ``'cw'`` → ``('rotate', 'cw')``.

    Returns:
        ``(verb, direction)``, or ``None`` if *hl_action* takes no direction or
        *direction_raw* is not a valid constant for it.
    """
    entry = _DIRECTION_VOCABULARY.get(hl_action)
    if entry is None:
        return None
    verb, vocabulary = entry
    direction = vocabulary.get(direction_raw.lower().strip())
    if direction is None:
        return None
    return verb, direction


# ll_actions where the target is the place object rather than the pick object
_PLACE_PHASE_LL_ACTIONS = frozenset(
    {"place_traj", "place_delta_traj", "lift_traj", "gripper_open_traj"}
)

# The post-grasp lift of a pnp: still part of the pick, targeting the held block
# at its pre-lift pose (annotate_target_xy resolves it at the lift's first frame,
# where the block was grasped, instead of the current frame).
LIFT_LL_ACTIONS = frozenset({"grasp_delta_traj"})

# Number of HL args each action takes, from the domain's action_config.yaml
# HL_ARGS.  ``action.high_level_action`` and ``action.high_level_args`` desync at
# action boundaries (they are read back from a plan-wide shared generated_values
# dict), which produced labels like ``flip block_c`` from a push action's args.
# A frame whose arg count contradicts its action name is left unlabelled.
_HL_ARG_COUNTS: dict[str, int] = {
    "isolate": 3,
    "move_to_side": 3,
    "move_to_center": 3,
    "pnp_table_table": 5,
    "pnp_block_table": 6,
    "pnp_table_block": 6,
    "pnp_block_block": 6,
    "push_to_right": 4,
    "push_next_to": 4,
    "rotate": 6,
    "reorient": 6,
}

# Index of the destination *block* in each pnp variant's HL args.  The place
# phase used to read args[2] for every variant, which is the block the object was
# lifted *off* for the pnp_block_* actions.  Variants absent here place onto the
# table, which has no meaningful object pixel — see build_action_goal_str.
_PNP_PLACE_TARGET_INDEX: dict[str, int] = {
    "pnp_table_block": 2,
    "pnp_block_block": 3,
}

# ll_actions that point at nothing: the robot is not yet addressing an object.
_NO_TARGET_LL_ACTIONS = frozenset({"noop_traj"})

# Actions that pick a block and set it down at a region of the table rather than
# onto another object.  They run the same phases as a pnp, so the pick half is
# labelled identically and only the place label names the destination region.
_MOVE_PLACE_LABELS: dict[str, str] = {
    "move_to_side": "place side",
    "move_to_center": "place center",
}

# Their place phase: the transfer trajectory plus the release.
_MOVE_PLACE_LL_ACTIONS = frozenset(
    {"move_to_side_traj", "move_to_center_traj", "gripper_open_traj"}
)

# Labels whose goal position is the held block's *final* pose rather than its
# pose on the current frame, because the destination is a location on the table
# and not an object that can be projected.  The caller resolves these at the last
# frame of the high-level action.
SEGMENT_END_TARGET_LABELS = frozenset(
    {"place on table", *_MOVE_PLACE_LABELS.values()}
)

# HL actions executed as a push RL skill followed by a nudge RL skill.
_PUSH_HL_ACTIONS = frozenset({"push_to_right", "push_next_to"})

# Within those, the nudge (align) phase; every other ll_action of a push HL action
# belongs to the push phase.  ``push_to_right`` runs pre_nudge -> nudge -> retract,
# ``push_next_to`` runs the same tail after two push stages.
_ALIGN_PHASE_LL_ACTIONS = frozenset({"pre_nudge_traj", "nudge_traj", "nudge_retract_traj"})

_GRIPPER_OPEN_LL_ACTIONS = frozenset({"gripper_open_traj", "init_gripper_open_traj"})

# Directions fixed by the PDDL effect rather than by where the block ends up:
# push_to_right's effect is (on_right ?b1 ?b2), so it is always 'right'.
_PINNED_PUSH_DIRECTIONS: dict[str, str] = {"push_to_right": "right"}


def cardinal_direction(obj_pos, target_pos) -> str | None:
    """Return where *obj_pos* sits relative to *target_pos* as a cardinal name.

    Uses the same table-frame convention as
    ``toyblocks.tamp.blocks.predicates.next_to._closest_cardinal``:
    +x is ``front``, -x is ``back``, +y is ``right``, -y is ``left``.  The
    dominant axis wins.

    Args:
        obj_pos:    (>=2,) world position of the pushed block.
        target_pos: (>=2,) world position of the block being pushed against.

    Returns:
        ``'front' | 'back' | 'left' | 'right'``, or ``None`` if either
        position is missing or non-finite.
    """
    if obj_pos is None or target_pos is None:
        return None
    dx = float(obj_pos[0]) - float(target_pos[0])
    dy = float(obj_pos[1]) - float(target_pos[1])
    if not (np.isfinite(dx) and np.isfinite(dy)):
        return None
    if abs(dx) >= abs(dy):
        return "front" if dx > 0 else "back"
    return "right" if dy > 0 else "left"


def build_action_goal_str(
    hl_action: str,
    ll_action: str,
    hl_args: list,
    push_direction: str | None = None,
) -> tuple[str, str, str | None, str | None]:
    """Build primary and secondary action-goal annotations for one frame.

    Primary encoding (high-level intent)::

        pick                     grasp_traj / grasp_delta_traj (lift) phase of pnp_*
        place                    place phase of pnp_table_block / pnp_block_block,
                                 targeting the destination block
        place on table           place phase of the pnp_*_table variants, whose
                                 destination is a surface rather than an object
        place side               place phase of move_to_side
        place center             place phase of move_to_center
        flip <direction>         reorient with *_flip direction
        rotate <direction>       reorient with *_rotate / cw / ccw direction
        push <direction>         push-skill phase of push_to_right / push_next_to
        align <direction>        nudge-skill phase of push_to_right / push_next_to

    Secondary encoding (immediate sub-task)::

        move_robot               grasp_traj, grasp_delta_traj, place_traj, place_delta_traj,
                                 lift_traj, pre_isolate_traj
        gripper_close            gripper_close_traj
        gripper_open             gripper_open_traj, init_gripper_open_traj

    Args:
        hl_action:      High-level action string (e.g. ``'pnp_table_block'``).
        ll_action:      Low-level phase string (e.g. ``'grasp_traj'``).
        hl_args:        High-level args list; ``args[1]`` = pick/target object,
                        ``args[2]`` = place target or reorient direction.
        push_direction: Cardinal direction the push/align is heading, derived
                        geometrically by the caller (see :func:`cardinal_direction`).
                        Ignored for actions whose direction is pinned by their
                        PDDL effect, and unused by non-push actions.

    Returns:
        ``(goal_action, secondary_goal_action, target_obj, target_obj2)`` where
        *target_obj* is the block whose pixel coordinates go in ``goal.position``
        and *target_obj2* is the block whose pixels go in ``goal.position_target``
        (push/align only).  ``None`` means no target applies.  A frame whose
        args contradict its action name comes back fully unlabelled.

        For the labels in :data:`SEGMENT_END_TARGET_LABELS` the caller must
        resolve *target_obj* at the last frame of the high-level action rather
        than the current one: the held block's final resting pixel is the goal,
        since no destination object exists.
    """
    if len(hl_args) < 2:
        if hl_action in ("done", "reach-goal"):
            return "done", "move_robot", None, None
        return "", "", None, None

    # An arg count that contradicts the action name means the two columns
    # desynced; nothing downstream can be trusted, so leave the frame unlabelled.
    expected_argc = _HL_ARG_COUNTS.get(hl_action)
    if expected_argc is not None and len(hl_args) != expected_argc:
        return "", "", None, None

    obj_name = str(hl_args[1])
    primary_str = ""
    secondary_str = ""
    target2: str | None = None

    # Default target is the object the action acts on; the place phase overrides
    # it below, and a few ll_actions address nothing at all.
    target: str | None = None if ll_action in _NO_TARGET_LL_ACTIONS else obj_name

    if hl_action.startswith("pnp_"):
        # pick
        if ll_action == "grasp_traj":
            primary_str = "pick"
            secondary_str = "move_robot"
        elif ll_action == "init_gripper_open_traj":
            primary_str = "pick"
            secondary_str = "gripper_open"
        elif ll_action == "gripper_close_traj":
            primary_str = "pick"
            secondary_str = "gripper_close"
        elif ll_action in LIFT_LL_ACTIONS:
            primary_str = "pick"
            secondary_str = "move_robot"
        # place
        elif ll_action in _PLACE_PHASE_LL_ACTIONS:
            place_index = _PNP_PLACE_TARGET_INDEX.get(hl_action)
            if place_index is not None and len(hl_args) > place_index:
                primary_str = "place"
                target = str(hl_args[place_index])
            else:
                # Placing onto the table: the destination is a surface, not an
                # object, so the target stays the held block and the caller
                # resolves its pixels at the end of the action — where it lands.
                primary_str = "place on table"
                target = obj_name
            secondary_str = (
                "gripper_open" if ll_action == "gripper_open_traj" else "move_robot"
            )

    elif hl_action in _MOVE_PLACE_LABELS:
        # Same phases as a pnp, so the pick half reads the same; the place half
        # names the destination region and keeps the held block as its target.
        if ll_action in _MOVE_PLACE_LL_ACTIONS:
            primary_str = _MOVE_PLACE_LABELS[hl_action]
            secondary_str = (
                "gripper_open" if ll_action == "gripper_open_traj" else "move_robot"
            )
        elif ll_action == "grasp_traj":
            primary_str = "pick"
            secondary_str = "move_robot"
        elif ll_action == "init_gripper_open_traj":
            primary_str = "pick"
            secondary_str = "gripper_open"
        elif ll_action == "gripper_close_traj":
            primary_str = "pick"
            secondary_str = "gripper_close"

    elif hl_action in ("done", "reach-goal"):
        primary_str = "done"
        secondary_str = "move_robot"

    elif hl_action in _DIRECTION_VOCABULARY and len(hl_args) > 2:
        parsed = _parse_reorient_direction(hl_action, str(hl_args[2]))
        if parsed is not None:
            verb, direction = parsed
            primary_str = f"{verb} {direction}"
            secondary_str = (
                "gripper_open" if ll_action in _GRIPPER_OPEN_LL_ACTIONS else "move_robot"
            )
        else:
            target = None

    elif hl_action in _PUSH_HL_ACTIONS and len(hl_args) > 2:
        # The HL action is executed as two RL skills back to back: the push skill
        # shoves the block into range, the nudge skill aligns it against the
        # target.  The verb tracks which skill is running, not the HL action.
        verb = "align" if ll_action in _ALIGN_PHASE_LL_ACTIONS else "push"
        direction = _PINNED_PUSH_DIRECTIONS.get(hl_action) or push_direction
        primary_str = f"{verb} {direction}" if direction else verb
        secondary_str = (
            "gripper_open" if ll_action in _GRIPPER_OPEN_LL_ACTIONS else "move_robot"
        )
        # goal.position tracks the block being pushed, goal.position_target the
        # block it is pushed against — both for the whole HL action.
        target = obj_name
        target2 = str(hl_args[2])

    elif hl_action == "isolate":
        primary_str = "isolate"
        secondary_str = (
            "gripper_open" if ll_action in _GRIPPER_OPEN_LL_ACTIONS else "move_robot"
        )

    if not primary_str:
        # No branch claimed this frame — do not leave a stray target behind.
        return "", "", None, None

    return primary_str, secondary_str, target, target2


def camera_from_sensor(pos_w, quat_w_world, intrinsic_matrix):
    """Projection inputs from a live Isaac Lab camera sensor.

    Mirrors what the recorder stores per frame (``pos_w`` + ``quat_w_world`` +
    ``intrinsic_matrices``), so annotations made at inference time land on the
    same pixels as the training labels instead of on a hardcoded scene preset.

    Args:
        pos_w:            (3,) camera position in world frame.
        quat_w_world:     (4,) camera orientation, wxyz, Isaac's world convention.
        intrinsic_matrix: (3, 3) or (9,) pinhole intrinsics.

    Returns:
        ``(cam_pos, R_cw, intrinsics)`` ready for :func:`project_world_to_pixel`.
    """
    pose = np.concatenate([
        np.asarray(pos_w, dtype=np.float64).reshape(3),
        np.asarray(quat_w_world, dtype=np.float64).reshape(4),
    ])
    cam_pos, R_cw = extrinsics_from_camera_pose(pose[None, :])
    K = np.asarray(intrinsic_matrix, dtype=np.float64).reshape(-1)
    return cam_pos[0], R_cw[0], (float(K[0]), float(K[4]), float(K[2]), float(K[5]))


def format_goal_prompt(goal_action: str, uv) -> str:
    """Render one frame's prompt exactly as the training data spells it.

    Matches ``RobotTrainDataset._row_to_task_ll_str``: ``"<action> <u, v>"``,
    degrading to the bare action when there is no finite pixel.
    """
    if not goal_action:
        return ""
    if uv is None:
        return goal_action
    u, v = float(uv[0]), float(uv[1])
    if not (np.isfinite(u) and np.isfinite(v)):
        return goal_action
    return f"{goal_action} <{u:.0f}, {v:.0f}>"


def build_frame_annotation(
    hl_action: str,
    ll_action: str,
    hl_args: list,
    get_uv=None,
    push_direction: str | None = None,
) -> tuple[str, tuple[int, int] | None]:
    """Annotation string and marker pixel for one live frame.

    The inference-time counterpart of :func:`build_action_goal_str`, returning
    the prompt string rather than the raw label so callers do not each reinvent
    the formatting.

    Args:
        get_uv: ``obj_name -> (u, v) | None``, projecting with the live camera.

    Returns:
        ``(annotation_str, marker_uv)``; ``marker_uv`` is None when there is no
        target or it does not project.
    """
    goal_action, _secondary, target_obj, _target2 = build_action_goal_str(
        hl_action, ll_action, hl_args, push_direction=push_direction
    )
    if not goal_action:
        return "", None

    uv = get_uv(target_obj) if (target_obj is not None and get_uv is not None) else None
    text = format_goal_prompt(goal_action, uv)
    marker: tuple[int, int] | None = None
    if uv is not None and np.isfinite(uv[0]) and np.isfinite(uv[1]):
        marker = (int(round(float(uv[0]))), int(round(float(uv[1]))))
    return text, marker


def annotate_frame_with_uv(frame_rgb: np.ndarray, uv) -> np.ndarray:
    """Return an annotated RGB copy with a circle+cross marker at *uv*.

    *uv* is a ``(u, v)`` integer pixel pair, or ``None`` (returns unchanged copy).
    Matches the marker style of :func:`draw_overlay_on_frame_rgb`.
    """
    frame = frame_rgb.copy()
    if uv is None:
        return frame
    h, w = frame.shape[:2]
    u, v = int(uv[0]), int(uv[1])
    if not (0 <= u < w and 0 <= v < h):
        return frame
    vis_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    color = (0, 200, 255)  # BGR
    cv2.circle(vis_bgr, (u, v), MARKER_RADIUS, color, MARKER_THICKNESS)
    cv2.drawMarker(vis_bgr, (u, v), color, cv2.MARKER_CROSS, MARKER_CROSS_LEN, MARKER_THICKNESS)
    frame[...] = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return frame

# BGR palette; marker size is 0.5× the previous defaults (7→4, 2→1, 14→7)
MARKER_PALETTE = [
    (0, 200, 255),
    (255, 100, 100),
    (180, 255, 100),
    (255, 180, 255),
    (100, 200, 255),
    (200, 130, 255),
]
MARKER_RADIUS = 8
MARKER_THICKNESS = 2
MARKER_CROSS_LEN = 14
FALLBACK_MARKER_BGR = (180, 180, 180)


def object_pixel_columns(df):
    cols = sorted(c for c in df.columns if c.startswith("observation.pos_in_camera_frame.") and c.endswith(".front_camera"))
    if not cols:
        cols = sorted(c for c in df.columns if c.startswith("observation.state.object_pixel."))
    return cols


def _pixel_col_to_name(col):
    if col.startswith("observation.pos_in_camera_frame."):
        return col.removeprefix("observation.pos_in_camera_frame.").removesuffix(".front_camera")
    return col.split("observation.state.object_pixel.")[-1]


def trackable_object_names(df):
    pix = object_pixel_columns(df)
    if pix:
        return {_pixel_col_to_name(c) for c in pix}
    prefix = "observation.state.object_pose."
    return {c[len(prefix) :] for c in df.columns if c.startswith(prefix)}


def parse_hl_args(raw):
    if raw is None or (isinstance(raw, str) and not str(raw).strip()):
        return []
    try:
        args = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    return args if isinstance(args, list) else []


def _pnp_phase_filter(names, ll_action, hl_args_list):
    """For pnp_table_block, restrict markers to the phase-relevant object.

    HL args layout: [robot, pick_obj, place_target, table].
    - grasp_traj / grasp_delta_traj  -> show only pick_obj   (args[1])
    - place_traj / place_delta_traj  -> show only place_target (args[2])
    - lift_traj   -> show only place_target (args[2])
    Other phases (start_traj, etc.) show all.
    """
    if ll_action == "grasp_traj" or ll_action in LIFT_LL_ACTIONS:
        pick_obj = hl_args_list[1] if len(hl_args_list) > 1 else None
        return [n for n in names if n == pick_obj] if pick_obj else names
    if ll_action in ("place_traj", "place_delta_traj", "lift_traj"):
        place_obj = hl_args_list[2] if len(hl_args_list) > 2 else None
        return [n for n in names if n == place_obj] if place_obj else names
    return names


def object_names_from_row(df, frame_idx):
    """Return object names whose markers should be drawn for this frame.

    Returns:
        list[str] — specific objects to mark.
        ``"all"`` — fall back to drawing all tracked objects.
        ``None``  — explicitly draw nothing.
    """
    if frame_idx < 0 or frame_idx >= len(df):
        return "all"
    known = trackable_object_names(df)
    row = df.iloc[frame_idx]
    raw = row.get("action.high_level_args")
    if raw is None or pd.isna(raw):
        raw = ""
    else:
        raw = str(raw)
    args = parse_hl_args(raw)
    out, seen = [], set()
    for a in args:
        if isinstance(a, str) and a in known and a not in seen:
            seen.add(a)
            out.append(a)

    if not out:
        return "all"

    hl_action = row.get("action.high_level_action")
    hl_action = "" if hl_action is None or pd.isna(hl_action) else str(hl_action)
    if hl_action.startswith("pnp_"):
        ll_action = row.get("action.low_level_action")
        ll_action = "" if ll_action is None or pd.isna(ll_action) else str(ll_action)
        if hl_action == "pnp_table_block":
            out = _pnp_phase_filter(out, ll_action, args)
        elif ll_action in ("place_traj", "place_delta_traj", "lift_traj"):
            return None

    return out


def pixel_from_parquet(df, frame_idx, block_name):
    if not block_name:
        return None
    col = f"observation.pos_in_camera_frame.{block_name}.front_camera"
    if col not in df.columns:
        col = f"observation.state.object_pixel.{block_name}"
    if col not in df.columns or frame_idx < 0 or frame_idx >= len(df):
        return None
    arr = np.asarray(df.iloc[frame_idx][col], dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[0]) or not np.isfinite(arr[1]):
        return None
    return float(arr[0]), float(arr[1])


def goal_position_from_row(df, frame_idx):
    """Return (u, v) from ``goal.position``, or None."""
    if "goal.position" not in df.columns or frame_idx < 0 or frame_idx >= len(df):
        return None
    gp = df.iloc[frame_idx].get("goal.position")
    if gp is None or (isinstance(gp, float) and pd.isna(gp)):
        return None
    arr = np.asarray(gp, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[0]) or not np.isfinite(arr[1]):
        return None
    return float(arr[0]), float(arr[1])


def _append_goal_position_marker(
    df,
    frame_idx,
    h: int,
    w: int,
    markers: list,
    uv_parts: list,
    info: dict,
) -> None:
    """Draw a single marker from ``goal.position`` when per-object pixels are unavailable."""
    gp = goal_position_from_row(df, frame_idx)
    if gp is None:
        return
    u, v = int(round(gp[0])), int(round(gp[1]))
    if not (0 <= u < w and 0 <= v < h):
        return
    ga = df.iloc[frame_idx].get("goal.action")
    label = "" if ga is None or pd.isna(ga) else str(ga)
    markers.append((u, v, (0, 200, 255)))
    uv_txt = f"goal ({gp[0]:.1f}, {gp[1]:.1f})"
    if label:
        uv_txt += f" — {label}"
    uv_parts.append(uv_txt)
    info["object_uvs_str"] = "; ".join(uv_parts)


def build_hl_segments(df):
    segments = []
    prev_key = None
    known = trackable_object_names(df)
    has_ll = "action.low_level_action" in df.columns
    for i in range(len(df)):
        row = df.iloc[i]
        hl_raw = row.get("action.high_level_action")
        hl = "" if hl_raw is None or pd.isna(hl_raw) else str(hl_raw)
        raw_args_cell = row.get("action.high_level_args")
        raw_s = "" if raw_args_cell is None or pd.isna(raw_args_cell) else str(raw_args_cell)
        ll_raw = row.get("action.low_level_action") if has_ll else None
        ll = "" if ll_raw is None or (isinstance(ll_raw, float) and pd.isna(ll_raw)) else str(ll_raw)
        key = (hl, raw_s, ll)
        if key == prev_key:
            continue
        prev_key = key
        if not hl:
            continue
        args = parse_hl_args(raw_s)
        object_names = []
        seen = set()
        for a in args:
            if isinstance(a, str) and a in known and a not in seen:
                seen.add(a)
                object_names.append(a)
        if hl.startswith("pnp_"):
            if hl == "pnp_table_block":
                object_names = _pnp_phase_filter(object_names, ll, args)
            elif ll in ("place_traj", "place_delta_traj", "lift_traj"):
                object_names = []
        object_refs = []
        goal_pt = goal_position_from_row(df, i)
        for n in object_names:
            pix = pixel_from_parquet(df, i, n)
            if pix is None and goal_pt is not None and len(object_names) == 1:
                pix = goal_pt
            pt = [round(pix[0], 1), round(pix[1], 1)] if pix else None
            object_refs.append({"name": n, "pixel_2d": pt})
        if not object_refs and goal_pt is not None:
            ga = row.get("goal.action")
            label = "" if ga is None or pd.isna(ga) else str(ga)
            name = label or "goal"
            object_refs.append(
                {"name": name, "pixel_2d": [round(goal_pt[0], 1), round(goal_pt[1], 1)]}
            )
        action_label = f"{hl} / {ll}" if ll else hl
        segments.append(
            {
                "frame_index": int(i),
                "action": action_label,
                "args": args,
                "robot": args[0] if args else None,
                "object_refs": object_refs,
            }
        )
    return segments


def _hl_row_strings(df, frame_idx):
    n_rows = len(df)
    hl_action, hl_args = "", ""
    if frame_idx < n_rows:
        row = df.iloc[frame_idx]
        ha = row.get("action.high_level_action")
        hl_action = "" if ha is None or pd.isna(ha) else str(ha)
        harg = row.get("action.high_level_args")
        hl_args = "" if harg is None or pd.isna(harg) else str(harg)
    return hl_action, hl_args


def _prepare_overlay(df, frame_idx, h: int, w: int):
    """Returns (info dict, list of (u, v, color_bgr)) for drawing."""
    hl_action, hl_args = _hl_row_strings(df, frame_idx)
    info = {
        "hl_action": hl_action,
        "hl_args": hl_args,
        "object_uvs_str": "-",
    }

    names = object_names_from_row(df, frame_idx)

    markers = []
    uv_parts = []

    if names is None:
        _append_goal_position_marker(df, frame_idx, h, w, markers, uv_parts, info)
        return info, markers

    use_fallback = False
    if names == "all":
        pix_cols = object_pixel_columns(df)
        if pix_cols:
            use_fallback = True
            refs = [
                {"name": _pixel_col_to_name(c), "pixel_2d": None}
                for c in pix_cols
            ]
        else:
            refs = []
    else:
        refs = [{"name": n, "pixel_2d": None} for n in names]

    for idx, ref in enumerate(refs):
        name = ref.get("name")
        pt = pixel_from_parquet(df, frame_idx, name)
        if pt is None and ref.get("pixel_2d"):
            p2 = ref["pixel_2d"]
            pt = (float(p2[0]), float(p2[1]))
        if pt is None:
            continue
        u, v = int(round(pt[0])), int(round(pt[1]))
        if not (0 <= u < w and 0 <= v < h):
            continue

        color = FALLBACK_MARKER_BGR if use_fallback else MARKER_PALETTE[idx % len(MARKER_PALETTE)]
        markers.append((u, v, color))
        label_name = name if name is not None else "?"
        uv_parts.append(f"{label_name} ({pt[0]:.1f}, {pt[1]:.1f})")

    if not markers:
        _append_goal_position_marker(df, frame_idx, h, w, markers, uv_parts, info)
    elif uv_parts:
        info["object_uvs_str"] = "; ".join(uv_parts)

    return info, markers


def draw_overlay_on_frame_rgb(frame_rgb: np.ndarray, df: pd.DataFrame, frame_idx: int):
    """RGB uint8 (H,W,3) -> annotated RGB copy + sidebar info dict."""
    frame = frame_rgb.copy()
    h, w = frame.shape[:2]
    info, markers = _prepare_overlay(df, frame_idx, h, w)
    if not markers:
        return frame, info

    vis_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    for u, v, color in markers:
        cv2.circle(vis_bgr, (u, v), MARKER_RADIUS, color, MARKER_THICKNESS)
        cv2.drawMarker(
            vis_bgr,
            (u, v),
            color,
            cv2.MARKER_CROSS,
            MARKER_CROSS_LEN,
            MARKER_THICKNESS,
        )
    frame[...] = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    return frame, info


def overlay_sidebar_info(df: pd.DataFrame, frame_idx: int, image_hw: tuple[int, int]):
    """HL + object UV text for a frame (no drawing). image_hw = (height, width)."""
    h, w = image_hw[0], image_hw[1]
    info, _ = _prepare_overlay(df, frame_idx, h, w)
    return info


ANNOTATED_FEATURE_KEY = "observation.images.front_camera_annotated"


def annotated_video_path(source_video: Path) -> Path:
    """Return destination path inside its own LeRobot feature folder.

    source_video: .../blocks/videos/observation.images.front_camera/chunk-NNN/file-NNN.mp4
    returns:      .../blocks/videos/observation.images.front_camera_annotated/chunk-NNN/file-NNN.mp4
    """
    videos_root = source_video.parent.parent.parent  # .../blocks/videos
    return videos_root / ANNOTATED_FEATURE_KEY / source_video.parent.name / source_video.name


def write_annotated_mp4(src: Path, dst: Path, df: pd.DataFrame) -> None:
    """Decode src with PyAV, overlay each frame, encode AV1 (yuv420p) via libsvtav1.

    SVT-AV1 library output is suppressed by redirecting stderr at the
    file-descriptor level during encoding.
    """
    import os
    from fractions import Fraction

    import av as _av

    in_c = _av.open(str(src))
    in_stream = in_c.streams.video[0]
    in_stream.thread_type = "AUTO"

    fps = 30.0
    if in_stream.average_rate:
        fps = float(in_stream.average_rate)
    fps = max(1.0, min(fps, 120.0))

    out_c = None
    out_stream = None
    n_decoded = 0

    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_stderr_fd = os.dup(2)
    os.dup2(devnull_fd, 2)

    try:
        for i, frame in enumerate(in_c.decode(in_stream)):
            rgb = frame.to_ndarray(format="rgb24")
            if i < len(df):
                rgb, _ = draw_overlay_on_frame_rgb(rgb, df, i)
            n_decoded += 1

            if out_c is None:
                h, w = rgb.shape[:2]
                out_c = _av.open(str(dst), mode="w")
                out_stream = out_c.add_stream("libsvtav1", rate=Fraction(fps).limit_denominator(10000))
                out_stream.width = w
                out_stream.height = h
                out_stream.pix_fmt = "yuv420p"
                out_stream.options = {"preset": "8", "crf": "10"}

            out_frame = _av.VideoFrame.from_ndarray(rgb, format="rgb24")
            out_frame.pts = i
            for packet in out_stream.encode(out_frame):
                out_c.mux(packet)

        if out_stream is not None:
            for packet in out_stream.encode():
                out_c.mux(packet)
    finally:
        in_c.close()
        if out_c is not None:
            out_c.close()
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
        os.close(devnull_fd)

    if n_decoded == 0:
        raise ValueError("no frames decoded from source video")

    if len(df) != n_decoded:
        import warnings

        warnings.warn(
            f"parquet rows ({len(df)}) != video frames ({n_decoded}); "
            "extra video frames left without overlay, or missing tail frames",
            stacklevel=2,
        )
