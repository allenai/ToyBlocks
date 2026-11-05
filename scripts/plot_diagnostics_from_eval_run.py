"""Render an eval run's control-diagnostic plots offline from its recorded data.

Eval once rendered ``<eval_dir>/control_diagnostics/episode_{N}/`` *during* the
rollout from in-memory per-step buffers filled straight off the Isaac
articulation; this script is now the only path to those figures. It builds the
same three after the fact, plus the per-camera first-frame comparison the eval
loop writes to ``<eval_dir>/first_frames/``, reading only what the run left on
disk:

- ``<eval_dir>/lerobot_dataset/<name>_success/`` and ``<name>_failure/``: joint positions and velocities,
  gripper observation, EE pose, PD gains, commanded pd_mode, the raw policy
  action (``action.policy``), and the per-episode camera mp4s.
- ``<eval_dir>/eval_episodes.json``: maps each recorded LeRobot episode to the
  source-dataset episode index the online plots are named after, to the sub-env
  that ran it, and to the source-dataset videos it is compared against.
- ``<eval_dir>/launch_params.json``: the ``dataset_path`` used for the
  ground-truth overlay.

The first-frame figures land in ``<eval_dir>/first_frames_offline/`` as
``episode_{N}_env_{M}_{cam}.png``, where ``N`` is the source-dataset episode
index that also names the ``control_diagnostics_offline/episode_{N}/`` folder
and ``M`` is the sub-env that ran it. The online run only writes these for
episodes whose queue setup carried its source videos; here every recorded
episode gets one per camera. They are built by the same renderer the eval loop
uses (:func:`~blocksuite.utils.wandb_logging.render_first_frame_comparison`),
but the eval side comes from the recorded video rather than the reset
observation. On runs recorded with the row-layout marker below, video frame 0
is the reset state; on older runs it is one env step later.

Plot step ``j`` is ground-truth row ``j``: the state at ``j``, the command
taken from it, and the PD gains that command ran under. Where a run's rows land
depends on how it was recorded:

- Runs whose ``meta/info.json`` has a ``blocksuite_row_layout`` key (or, if
  recorded before the key moved there, a ``meta/blocksuite_row_layout.json``
  file) already use that layout (row 0 is the reset state, gains travel with the action), so rows stay
  at their own step. The terminal row (``action.is_valid == False``) keeps its
  state and gains but plots no command.
- Older policy-eval runs never recorded the reset state, so states and
  commands sit at ``t + 1`` (step 0 is a gap). Their gains describe the step
  that produced the row's state, i.e. the command on the row before, so gains
  stay at ``t``.
- Older replay runs (``replay`` in launch_params.json, else the per-episode
  replay metrics in eval_episodes.json) wrote each row's
  state *after* its command: states go to ``t + 1``, commands and gains stay
  at ``t``. Their last commanded row replays the source's terminal row and
  samples the next episode's reset frame, so it is dropped.

Each episode folder also gets ``goal_progress.png``, drawn from
``eval_episodes.json`` alone (no LeRobot data needed): the number of goal
predicates satisfied over the episode, and below it one on/off row per
``goal_progress`` entry, from the change points ``RecordMetrics`` traced over the
env's episode clock, followed by rows for the other ``goal_success`` terms (goal
blocks settled, gripper open) and ``goal_success`` itself when the record traced
them. Runs recorded with ``--save_step_metrics`` also get panels from
``<eval_dir>/step_metrics/env{i}_episode_{n}.csv`` with what the settle gate read:
each goal block's linear and angular speed and the finger joint, against their
thresholds. Steps are that clock (hold steps are not counted); seconds are steps
x the record's ``step_dt_s``. ``--goal-progress-only`` draws just these. Runs
recorded before the trace existed have no goal plot.

Some series cannot be reproduced exactly from disk:

- ``observation.state.joint_velocity`` only exists in runs recorded after the
  recorder learned to write it unconditionally; for older runs the joint
  velocity plot is omitted.
- ``observation.state.object_mass`` is a live PhysX mass query, not tied to any
  randomization (mass is not currently randomized), and only exists in runs
  recorded after the recorder learned to query it; older runs omit that row of
  the ``<block_name>_pose_comparison.png`` physics-parameter panel. The GT side
  is only populated if a future dataset happens to record the same column.
- PD gains (``observation.state.current_*``) and ``action.pd_mode`` only exist in
  runs recorded after the recorder learned to write them; for older runs the gain
  plots keep their ground-truth curves alone, and the pd_mode falls back to the
  ninth channel of ``action.policy`` when the action is 9-dim (the eval loop
  strips pd_mode before stepping, so usually it is not). ``action.pd_mode`` also
  holds the binary mode the gains settled on rather than the policy's raw
  prediction, so it lags that prediction wherever the gain controller's dead
  zone or debounce held a switch back.
- ``observation.state.ee_position`` needs no correction once a run was recorded
  with the robot config's ``ee_body_name`` in the robot-base frame, which for
  these robots (base at the env origin, unrotated) is the env frame the eval-time
  EE observation uses. Runs recorded before that sampled the last articulation
  body in world coordinates, leaving their curves a gripper length off in
  position and rotated by a constant in orientation (the divergence
  ``BaseEnvCfg.legacy_ee_pos`` documents); subtracting the env's grid origin
  brings those positions back, and is a no-op for the newer runs.

Pure pyarrow + pandas + matplotlib; no Isaac Lab imports.

Example::

    python scripts/plot_control_diagnostics_from_eval_run.py \
        --eval-dir runs/aug12_flip_.../seed_1/local_20260813_225257 \
        --num-workers 8
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import pyarrow.dataset as pa_ds
import tyro

from blocksuite.eval.plots import save_eval_plots
from blocksuite.utils.wandb_logging import (
    _IMAGE_VIDEO_KEY_PREFIX,
    _extract_expert_first_frame,
    render_first_frame_comparison,
    short_camera_name,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes

_N_JOINTS = 7
_OBJECT_POSE_PREFIX = "observation.state.object_pose."
_CAMERA_POSE_PREFIX = "observation.camera_pose."
_CAMERA_INTRINSICS_PREFIX = "observation.camera_intrinsics."
_ROBOT_POSE_PREFIX = "robot.poses."
_LEFT_FINGER_LINK = "left_inner_finger"
_RIGHT_FINGER_LINK = "right_inner_finger"
# Buffer name -> dataset column, for the PD gains. Absent from runs recorded
# before the recorder gained these features; those plots keep GT curves only.
_GAIN_COLUMNS: dict[str, str] = {
    "joint_stiffness": "observation.state.current_stiffness",
    "joint_damping": "observation.state.current_damping",
    "joint_effort_limits": "observation.state.current_effort_limit",
}
# BaseEnvCfg.scene.env_spacing, i.e. what a run that did not pass --env-spacing used.
_DEFAULT_ENV_SPACING = 5.0
# Mirrors blocksuite.vecgen.recording.lerobot_recorder.ROW_LAYOUT_INFO_KEY (not imported: that
# module needs Isaac).
_ROW_LAYOUT_INFO_KEY = "blocksuite_row_layout"
# Marker file used by recordings made before the key moved into meta/info.json.
_ROW_LAYOUT_FILENAME = "blocksuite_row_layout.json"
GOAL_PROGRESS_FILENAME = "goal_progress.png"
# Legend labels longer than this are cut; the full text stays in eval_episodes.json.
_GOAL_LABEL_MAX_CHARS = 70
# Mirror the settle gate in blocksuite.tamp.blocks.pddl_goal_tensor_conversions (not
# imported: that module needs torch); tests check they still match.
_SETTLED_LIN_VEL = 0.1
_SETTLED_ANG_VEL = 0.5
_GRIPPER_OPEN_BELOW = 0.25
# Speeds are drawn on a log axis; exact zeros are lifted to this.
_SPEED_LOG_FLOOR = 1e-4
_STEP_METRICS_DIRNAME = "step_metrics"
# (record key prefix, row label, color) of the goal_success limbs below the predicate rows.
_GATE_ROWS: tuple[tuple[str, str, str], ...] = (
    ("goal_blocks_settled", "blocks settled", "tab:olive"),
    ("goal_gripper_open", "gripper open", "tab:cyan"),
    ("goal_success", "goal_success", "tab:green"),
)


@dataclasses.dataclass
class Args:
    eval_dir: str
    """Eval run directory (the one holding ``lerobot_dataset/`` and ``eval_episodes.json``)."""
    output_subdir: str = "control_diagnostics_offline"
    """Subdirectory of --eval-dir the figures are written to, one folder per episode."""
    first_frames_subdir: str = "first_frames_offline"
    """Subdirectory of --eval-dir the per-camera first-frame comparisons are written to."""
    episodes: tuple[int, ...] | None = None
    """Restrict to these episode labels (source-dataset episode indices). None plots all."""
    gt_dataset_path: str | None = None
    """Ground-truth dataset for the overlay. Defaults to ``dataset_path`` from launch_params.json."""
    overlay_ground_truth: bool = True
    """If False, plot the policy rollout alone (no ground-truth curves)."""
    env_spacing: float | None = None
    """Grid spacing (m) the run used, for recovering env-local EE positions.
    Defaults to ``env_spacing`` from launch_params.json, else the env config default."""
    subtract_env_origin: bool = True
    """If False, EE positions stay in the world frame exactly as recorded."""
    num_workers: int = 1
    """Episodes rendered in parallel processes. Each figure is large, so >1 helps a lot."""
    goal_progress_only: bool = False
    """Only draw each episode's goal_progress.png (from eval_episodes.json); skip the
    LeRobot-based control diagnostics and first-frame comparisons."""


@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    """One recorded episode and where its pieces live."""

    label: int
    """Source-dataset episode index; names the output folder and selects the GT episode."""
    output_lerobot_episode_index: int
    """``episode_index`` inside the run's recorded LeRobot dataset."""
    sim_env_idx: int | None
    """Sub-env that ran the episode; part of the first-frame comparison filename."""
    expert_videos: tuple[tuple[str, str, float | None], ...] = ()
    """``(video_key, path, from_timestamp)`` of the source-dataset RGB videos this
    episode was evaluated against; empty when the run recorded no queue setup."""
    lerobot_root: str | None = None
    """Dataset holding the episode (``output_lerobot_dataset_dir``: the run's success or
    failure dataset); None for runs that recorded a single dataset."""


@dataclasses.dataclass(frozen=True)
class PlotJob:
    """Everything one worker process needs to render a single episode."""

    spec: EpisodeSpec
    lerobot_root: str
    output_dir: str
    first_frames_dir: str
    """Flat directory collecting every episode's ``episode_{N}_env_{M}_{cam}.png``."""
    gt_dataset_path: str | None
    env_spacing: float | None
    """Grid spacing used to recover env-local EE positions; None keeps the world frame."""
    row_layout_marked: bool = False
    """Whether the dataset carries the recorder's row-layout marker (see the module docstring)."""
    replay_recording: bool = False
    """Whether the run replayed recorded actions (``replay`` in launch_params.json)."""


def _has_row_layout_marker(lerobot_root: Path) -> bool:
    """Whether ``lerobot_root`` was recorded with rows that match the TAMP source row for row."""
    meta = lerobot_root / "meta"
    try:
        info = json.loads((meta / "info.json").read_text())
    except (OSError, json.JSONDecodeError):
        info = {}
    if isinstance(info, dict) and _ROW_LAYOUT_INFO_KEY in info:
        return True
    return (meta / _ROW_LAYOUT_FILENAME).is_file()


def _find_lerobot_root(eval_dir: Path) -> Path:
    """The run's default LeRobot dataset under ``<eval_dir>/lerobot_dataset``.

    Runs recorded with the datagen recorder write a ``<name>_success`` and a
    ``<name>_failure`` dataset and name each episode's in ``eval_episodes.json``; this
    one is only used for episodes without that (older runs, replay mappings): the
    success dataset when present.
    """
    base = eval_dir / "lerobot_dataset"
    if not base.is_dir():
        raise FileNotFoundError(f"No lerobot_dataset/ directory under {eval_dir}")
    if (base / "data").is_dir():
        return base
    candidates = sorted(p for p in base.iterdir() if (p / "data").is_dir())
    if not candidates:
        raise FileNotFoundError(f"No LeRobot dataset with a data/ directory under {base}")
    success = [p for p in candidates if p.name.endswith("_success")]
    if len(candidates) > 1 and len(success) != 1:
        raise RuntimeError(
            f"Multiple LeRobot datasets under {base}: {[p.name for p in candidates]}. "
            "Point --eval-dir at a run with a single recorded dataset."
        )
    return (success or candidates)[0]


def _expert_video_entries(rec: dict) -> tuple[tuple[str, str, float | None], ...]:
    """Pull one episode's RGB source videos out of its ``eval_episodes.json`` record.

    Depth feeds are dropped: only ``observation.images.*`` has a frame worth
    comparing against the rendered eval view.
    """
    videos = rec.get("expert_videos") or {}
    return tuple(
        (key, str(meta["path"]), meta.get("from_timestamp"))
        for key, meta in sorted(videos.items())
        if key.startswith(_IMAGE_VIDEO_KEY_PREFIX) and meta.get("path")
    )


def _load_episode_specs(eval_dir: Path, lerobot_root: Path) -> list[EpisodeSpec]:
    """Build the episode list from evaluation or replay episode mappings."""
    records_path = eval_dir / "eval_episodes.json"
    if records_path.exists():
        records = json.loads(records_path.read_text())
        specs = [
            EpisodeSpec(
                label=(
                    int(rec["source_dataset_episode_idx"])
                    if rec.get("source_dataset_episode_idx") is not None
                    else int(rec["output_lerobot_episode_index"])
                ),
                output_lerobot_episode_index=int(rec["output_lerobot_episode_index"]),
                sim_env_idx=(
                    int(rec["sim_env_idx"]) if rec.get("sim_env_idx") is not None else None
                ),
                expert_videos=_expert_video_entries(rec),
                lerobot_root=rec.get("output_lerobot_dataset_dir"),
            )
            for rec in records
            if rec.get("output_lerobot_episode_index") is not None
        ]
        if specs:
            return sorted(specs, key=lambda s: s.label)
        print(f"[offline-diag] {records_path.name} has no recorded episodes; falling back to dataset order")

    replay_results_path = eval_dir / "replay_results.json"
    if replay_results_path.exists():
        replay_results = json.loads(replay_results_path.read_text())
        specs = [
            EpisodeSpec(
                label=int(result["gt_dataset_episode_idx"]),
                output_lerobot_episode_index=lerobot_index,
                sim_env_idx=0,
            )
            for lerobot_index, result in enumerate(replay_results)
            if result.get("gt_dataset_episode_idx") is not None
        ]
        if specs:
            return sorted(specs, key=lambda s: s.label)
        print(
            f"[offline-diag] {replay_results_path.name} has no ground-truth episode "
            "indices; falling back to dataset order"
        )

    info = json.loads((lerobot_root / "meta" / "info.json").read_text())
    total = int(info["total_episodes"])
    print(
        "[offline-diag] no episode mapping available: labelling folders by LeRobot "
        "episode index and using it for the ground-truth overlay too"
    )
    return [
        EpisodeSpec(label=i, output_lerobot_episode_index=i, sim_env_idx=None)
        for i in range(total)
    ]


def _enrich_replay_specs_with_expert_videos(
    specs: list[EpisodeSpec], gt_dataset_path: str | None
) -> list[EpisodeSpec]:
    """Attach GT video clips to replay specs so first-frame comparisons work.

    Replay specs built from ``replay_results.json`` have no expert videos
    because that information lives only in ``eval_episodes.json``.  When a
    ``--gt_dataset_path`` is available, resolve the per-camera video clips
    from the GT dataset and attach them.
    """
    if not gt_dataset_path:
        return specs

    needs_enrichment = any(not s.expert_videos for s in specs)
    if not needs_enrichment:
        return specs

    from blocksuite.utils.tamp_dataset import EpisodeSpec as TampEpisodeSpec
    from blocksuite.utils.tamp_dataset import iter_episode_video_clips

    gt_root = Path(gt_dataset_path)
    enriched: list[EpisodeSpec] = []
    for spec in specs:
        if spec.expert_videos:
            enriched.append(spec)
            continue
        gt_spec = TampEpisodeSpec(
            name=f"gt_ep{spec.label:05d}",
            source=str(gt_root),
            merged_root=gt_root,
            episode_index=spec.label,
        )
        try:
            clips = iter_episode_video_clips(gt_spec)
        except Exception as exc:
            print(
                f"[offline-diag] episode {spec.label}: could not resolve GT videos: {exc}"
            )
            enriched.append(spec)
            continue
        expert_videos = tuple(
            (clip.video_key, str(clip.video_path), clip.from_timestamp)
            for clip in clips
        )
        enriched.append(
            EpisodeSpec(
                label=spec.label,
                output_lerobot_episode_index=spec.output_lerobot_episode_index,
                sim_env_idx=spec.sim_env_idx,
                expert_videos=expert_videos,
            )
        )
    return enriched


def _load_episode_frames(lerobot_root: Path, episode_index: int) -> pd.DataFrame:
    """Read one episode's rows, using predicate pushdown to skip other row-groups."""
    dataset = pa_ds.dataset(str(lerobot_root / "data"), format="parquet")
    table = dataset.to_table(filter=pa_ds.field("episode_index") == episode_index)
    if table.num_rows == 0:
        raise RuntimeError(
            f"No rows for episode_index={episode_index} in {lerobot_root / 'data'}"
        )
    return table.to_pandas().sort_values("frame_index").reset_index(drop=True)


def _stack_column(df: pd.DataFrame, column: str, width: int) -> np.ndarray:
    """Stack a column into a ``(len(df), width)`` float array.

    Handles both the list-valued columns and the ones pandas hands back as bare
    scalars (any feature of shape ``(1,)``).
    """
    return np.stack(
        [
            np.atleast_1d(np.asarray(row, dtype=np.float32)).ravel()[:width]
            for row in df[column]
        ]
    )


def _num_commanded_rows(df: pd.DataFrame) -> int:
    """Number of leading rows that carry an action (the terminal row does not)."""
    valid = np.array(
        [bool(np.asarray(v).flat[0]) for v in df["action.is_valid"]], dtype=bool
    )
    n_valid = int(valid.sum())
    if not (valid[:n_valid].all() and not valid[n_valid:].any()):
        raise RuntimeError(
            "action.is_valid is not a leading run of True; the episode rows are out of "
            "order or the recording is corrupt."
        )
    return n_valid


def _infer_env_origin(ee_world: np.ndarray, env_spacing: float, label: int) -> np.ndarray:
    """Recover the grid origin of the env this episode ran in.

    Isaac Lab spaces parallel envs on a grid whose origins sit at multiples of
    ``env_spacing / 2`` with ``z = 0``, and the arm never leaves a quarter
    spacing of its own origin, so rounding the mean world EE position onto that
    grid recovers the offset without assuming a row/column ordering.
    """
    quantum = env_spacing / 2.0
    origin = np.round(ee_world[:, :2].mean(axis=0) / quantum) * quantum
    local = ee_world[:, :2] - origin
    if np.abs(local).max() > quantum / 2.0:
        print(
            f"[offline-diag] episode {label}: EE strays {np.abs(local).max():.2f} m from the "
            f"inferred env origin {origin.tolist()} (env_spacing={env_spacing}). Pass "
            "--env-spacing to match the run, or --no-subtract-env-origin."
        )
    return np.array([origin[0], origin[1], 0.0])


@dataclasses.dataclass(frozen=True)
class _Placement:
    """Which dataset rows a series uses and the plot step its first row lands on."""

    rows: slice
    offset: int


def _row_placements(
    n_rows: int, n_cmd: int, *, row_layout_marked: bool, replay_recording: bool
) -> tuple[int, _Placement, _Placement, _Placement]:
    """``(n_steps, state, command, gain)`` placements; plot step ``j`` is GT row ``j``.

    See the module docstring's alignment note for the three layouts.
    """
    if row_layout_marked:
        return (
            n_rows,
            _Placement(slice(0, n_rows), 0),
            _Placement(slice(0, n_cmd), 0),
            _Placement(slice(0, n_rows), 0),
        )
    if replay_recording:
        # The last commanded row replays the source's terminal row and samples
        # the next episode's reset frame; neither belongs on the plot.
        used = slice(0, max(0, n_cmd - 1))
        return n_cmd, _Placement(used, 1), _Placement(used, 0), _Placement(used, 0)
    used = slice(0, n_cmd)
    return n_cmd + 1, _Placement(used, 1), _Placement(used, 1), _Placement(used, 0)


def _placed(values: np.ndarray, placement: _Placement, n_steps: int) -> np.ndarray:
    """Lay ``values`` (one entry per dataset row) out on the plot steps, NaN elsewhere."""
    picked = values[placement.rows].astype(np.float64, copy=False)
    out = np.full((n_steps, *values.shape[1:]), np.nan, dtype=np.float64)
    out[placement.offset : placement.offset + len(picked)] = picked
    return out


def _rollout_from_frames(
    df: pd.DataFrame,
    env_spacing: float | None,
    label: int,
    row_layout_marked: bool = False,
    replay_recording: bool = False,
) -> dict[str, np.ndarray]:
    """Rebuild the online per-step buffers from one episode's dataset rows.

    ``row_layout_marked`` says the dataset carries the recorder's layout marker
    (rows already match the TAMP source row for row). Without it,
    ``replay_recording`` picks between the legacy replay and policy-eval layouts.
    """
    n_cmd = _num_commanded_rows(df)
    if n_cmd == 0:
        raise RuntimeError(
            "Episode has no rows carrying an action; nothing to plot for this rollout."
        )
    n_steps, state_at, command_at, gain_at = _row_placements(
        len(df),
        n_cmd,
        row_layout_marked=row_layout_marked,
        replay_recording=replay_recording,
    )

    if "action.policy" not in df.columns:
        raise RuntimeError(
            "Recorded dataset has no action.policy column, so the action the policy "
            "actually emitted is unavailable; re-record with the recorder's action_dim set."
        )
    actions = _stack_column(df, "action.policy", width=9)
    action_width = int(np.asarray(df["action.policy"].iloc[0]).size)

    def _state(values: np.ndarray) -> np.ndarray:
        return _placed(values, state_at, n_steps)

    def _command(values: np.ndarray) -> np.ndarray:
        return _placed(values, command_at, n_steps)

    def _gain(values: np.ndarray) -> np.ndarray:
        return _placed(values, gain_at, n_steps)

    ee_poses = _stack_column(df, "observation.state.ee_position", 6)
    if env_spacing is not None:
        ee_poses[:, :3] -= _infer_env_origin(
            ee_poses[state_at.rows, :3], env_spacing, label
        )

    if "action.pd_mode" in df.columns:
        pd_modes = _command(_stack_column(df, "action.pd_mode", 1))[:, 0]
    elif action_width >= 9:
        pd_modes = _command(actions[:, 8:9])[:, 0]
    else:
        pd_modes = np.full(n_steps, np.nan)

    # An empty array (rather than NaNs) keeps the policy curve off the figure
    # entirely for runs that predate gain recording.
    gains = {
        key: (
            _gain(_stack_column(df, column, _N_JOINTS))
            if column in df.columns
            else np.full((0, _N_JOINTS), np.nan)
        )
        for key, column in _GAIN_COLUMNS.items()
    }

    # Gain vectors are padded with one slot per articulation joint; index
    # _N_JOINTS is the gripper's finger_joint, present only when the vector
    # is wider than the 7 arm joints (older runs recorded arm-only gains).
    gripper_gains = {
        f"gripper_{quantity}": (
            _gain(_stack_column(df, column, _N_JOINTS + 1))[:, _N_JOINTS]
            if column in df.columns and int(np.asarray(df[column].iloc[0]).size) > _N_JOINTS
            else np.full(0, np.nan)
        )
        for quantity, column in (
            ("stiffness", "observation.state.current_stiffness"),
            ("damping", "observation.state.current_damping"),
            ("effort_limit", "observation.state.current_effort_limit"),
        )
    }

    # An empty array (rather than NaNs) keeps the policy curve off the figure
    # entirely for runs that predate unconditional joint-velocity recording.
    joint_velocities = (
        _state(_stack_column(df, "observation.state.joint_velocity", _N_JOINTS))
        if "observation.state.joint_velocity" in df.columns
        else np.full((0, _N_JOINTS), np.nan)
    )

    block_poses = {
        col[len(_OBJECT_POSE_PREFIX):]: _state(_stack_column(df, col, 7))
        for col in df.columns
        if col.startswith(_OBJECT_POSE_PREFIX) and "table" not in col.lower()
    }
    camera_poses = {
        col[len(_CAMERA_POSE_PREFIX):]: _state(_stack_column(df, col, 7))
        for col in df.columns
        if col.startswith(_CAMERA_POSE_PREFIX)
    }
    camera_intrinsics = {
        col[len(_CAMERA_INTRINSICS_PREFIX):]: _state(_stack_column(df, col, 9))
        for col in df.columns
        if col.startswith(_CAMERA_INTRINSICS_PREFIX)
    }
    cube_side_length = (
        float(np.asarray(df["observation.state.cube_scale"].iloc[0]).flat[0])
        if "observation.state.cube_scale" in df.columns
        else None
    )
    friction_properties = (
        np.asarray(df["observation.state.friction_properties"].iloc[0], dtype=np.float32).ravel()
        if "observation.state.friction_properties" in df.columns
        else None
    )
    table_friction = (
        float(np.asarray(df["observation.state.table_friction"].iloc[0]).flat[0])
        if "observation.state.table_friction" in df.columns
        else None
    )
    table_height_delta = (
        float(np.asarray(df["observation.state.table_height_delta"].iloc[0]).flat[0])
        if "observation.state.table_height_delta" in df.columns
        else None
    )
    object_mass = (
        float(np.asarray(df["observation.state.object_mass"].iloc[0]).flat[0])
        if "observation.state.object_mass" in df.columns
        else None
    )

    left_finger_col = f"{_ROBOT_POSE_PREFIX}{_LEFT_FINGER_LINK}"
    right_finger_col = f"{_ROBOT_POSE_PREFIX}{_RIGHT_FINGER_LINK}"
    left_finger_pose = (
        _state(_stack_column(df, left_finger_col, 7)) if left_finger_col in df.columns else None
    )
    right_finger_pose = (
        _state(_stack_column(df, right_finger_col, 7)) if right_finger_col in df.columns else None
    )

    return {
        **gains,
        **gripper_gains,
        "joint_positions": _state(
            _stack_column(df, "observation.state.joint_position", _N_JOINTS)
        ),
        "joint_velocities": joint_velocities,
        "action_targets": _command(actions[:, :_N_JOINTS]),
        "gripper_actions": _command(actions[:, 7:8])[:, 0],
        "gripper_obs": _state(
            _stack_column(df, "observation.state.gripper_position", 1)
        )[:, 0],
        "pd_modes": pd_modes,
        "ee_poses": _state(ee_poses),
        "block_poses": block_poses,
        "cube_side_length": cube_side_length,
        "friction_properties": friction_properties,
        "table_friction": table_friction,
        "table_height_delta": table_height_delta,
        "object_mass": object_mass,
        "left_finger_pose": left_finger_pose,
        "right_finger_pose": right_finger_pose,
        "camera_poses": camera_poses,
        "camera_intrinsics": camera_intrinsics,
    }


def _camera_video_dirs(lerobot_root: Path) -> dict[str, Path]:
    """Short camera name -> the directory holding that camera's recorded mp4s.

    Covers both video layouts the recorder and the stitcher produce
    (``videos/chunk-000/<key>/`` and ``videos/<key>/chunk-000/``) by keying off
    the feature-name component wherever it sits.
    """
    dirs: dict[str, Path] = {}
    for path in (lerobot_root / "videos").rglob("*"):
        if path.is_dir() and path.name.startswith(_IMAGE_VIDEO_KEY_PREFIX):
            dirs[short_camera_name(path.name)] = path
    return dirs


def _find_episode_video(camera_dir: Path, episode_index: int) -> Path | None:
    """Locate one episode's mp4 under a camera directory, chunked or not.

    The datagen recorder writes one file per episode as
    ``chunk-{i // 1000:03d}/file-{i % 1000:03d}.mp4``; older eval runs ``episode_{i:06d}.mp4``.
    """
    v3 = camera_dir / f"chunk-{episode_index // 1000:03d}" / f"file-{episode_index % 1000:03d}.mp4"
    if v3.is_file():
        return v3
    name = f"episode_{episode_index:06d}.mp4"
    direct = camera_dir / name
    if direct.is_file():
        return direct
    return next(iter(sorted(camera_dir.glob(f"**/{name}"))), None)


def _render_first_frames(job: PlotJob) -> int:
    """Save this episode's ``[eval | dataset | diff]`` PNG for every camera.

    The eval side is frame 0 of the run's own recorded video: the reset state
    for runs carrying the row-layout marker, one env step past it (the arm a
    single control step further along) for older runs. Returns how many cameras
    were written.
    """
    spec = job.spec
    if not spec.expert_videos:
        print(
            f"[offline-diag] episode {spec.label}: eval_episodes.json lists no source "
            "videos, first-frame comparison skipped"
        )
        return 0
    if spec.sim_env_idx is None:
        print(
            f"[offline-diag] episode {spec.label}: no sim_env_idx recorded, first-frame "
            "comparison skipped (its filename needs the env index)"
        )
        return 0

    camera_dirs = _camera_video_dirs(Path(job.lerobot_root))
    out_dir = Path(job.first_frames_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for video_key, expert_path, from_timestamp in spec.expert_videos:
        camera = short_camera_name(video_key)
        camera_dir = camera_dirs.get(camera)
        if camera_dir is None:
            print(
                f"[offline-diag] episode {spec.label}: the recorded dataset has no "
                f"{camera} camera video, skipping that comparison"
            )
            continue
        eval_video = _find_episode_video(camera_dir, spec.output_lerobot_episode_index)
        if eval_video is None:
            print(
                f"[offline-diag] episode {spec.label}: no {camera} mp4 for LeRobot episode "
                f"{spec.output_lerobot_episode_index} under {camera_dir}"
            )
            continue

        eval_frame = _extract_expert_first_frame(str(eval_video))
        expert_frame = _extract_expert_first_frame(expert_path, from_timestamp)
        if eval_frame is None or expert_frame is None:
            print(
                f"[offline-diag] episode {spec.label}: missing {camera} frame "
                f"(eval={eval_frame is not None}, dataset={expert_frame is not None})"
            )
            continue

        canvas = render_first_frame_comparison(eval_frame, expert_frame)
        out_path = out_dir / f"episode_{spec.label}_env_{spec.sim_env_idx}_{camera}.png"
        imageio.imwrite(str(out_path), canvas)
        written += 1
    return written


def _render_episode(job: PlotJob) -> str:
    """Render one episode's three figures; runs in a worker process."""
    frames = _load_episode_frames(Path(job.lerobot_root), job.spec.output_lerobot_episode_index)
    rollout = _rollout_from_frames(
        frames,
        job.env_spacing,
        job.spec.label,
        row_layout_marked=job.row_layout_marked,
        replay_recording=job.replay_recording,
    )
    n_steps = len(rollout["gripper_obs"])

    save_eval_plots(
        output_dir=job.output_dir,
        all_joint_positions=list(rollout["joint_positions"]),
        all_joint_velocities=list(rollout["joint_velocities"]),
        all_action_targets=list(rollout["action_targets"]),
        all_gripper_actions=list(rollout["gripper_actions"]),
        all_gripper_obs=list(rollout["gripper_obs"]),
        all_pd_modes=list(rollout["pd_modes"]),
        all_ee_poses=list(rollout["ee_poses"]),
        all_joint_stiffness=list(rollout["joint_stiffness"]),
        all_joint_damping=list(rollout["joint_damping"]),
        all_joint_effort_limits=list(rollout["joint_effort_limits"]),
        gt_dataset_path=job.gt_dataset_path,
        gt_episode_index=job.spec.label,
        block_poses=rollout["block_poses"],
        cube_side_length=rollout["cube_side_length"],
        left_finger_pose=rollout["left_finger_pose"],
        right_finger_pose=rollout["right_finger_pose"],
        gripper_stiffness=rollout["gripper_stiffness"],
        gripper_damping=rollout["gripper_damping"],
        gripper_effort_limit=rollout["gripper_effort_limit"],
        friction_properties=rollout["friction_properties"],
        table_friction=rollout["table_friction"],
        table_height_delta=rollout["table_height_delta"],
        object_mass=rollout["object_mass"],
        camera_poses=rollout["camera_poses"],
        camera_intrinsics=rollout["camera_intrinsics"],
    )
    n_cameras = _render_first_frames(job)
    return (
        f"episode {job.spec.label}: {n_steps} steps -> {job.output_dir} "
        f"(+{n_cameras} first-frame comparison(s))"
    )


def _try_render_episode(job: PlotJob) -> tuple[int, str | None, str]:
    """``(label, error, message)`` for one episode; one bad episode must not abort the rest."""
    try:
        return job.spec.label, None, _render_episode(job)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return job.spec.label, error, f"episode {job.spec.label}: FAILED ({error})"


@dataclasses.dataclass(frozen=True)
class GoalPlotJob:
    """One episode's goal-progress figure; needs nothing but its eval_episodes.json record."""

    label: int
    """Names the ``episode_{label}/`` folder, like the control-diagnostic plots."""
    record: dict
    """The episode's ``eval_episodes.json`` record (``goal_progress`` with traces)."""
    output_dir: str
    fallback_fps: float | None = None
    """LeRobot fps, for runs whose records predate ``step_dt_s``; None omits seconds."""
    step_metrics_csv: str | None = None
    """The episode's ``step_metrics/`` CSV (``--save_step_metrics`` runs); None skips the speed panels."""


def goal_plot_label(rec: dict, position: int) -> int:
    """Folder label of a record: source episode, else LeRobot episode, else its position."""
    for key in ("source_dataset_episode_idx", "output_lerobot_episode_index"):
        if rec.get(key) is not None:
            return int(rec[key])
    return position


def _has_goal_traces(rec: dict) -> bool:
    progress = rec.get("goal_progress") or []
    return bool(progress) and all(entry.get("trace") for entry in progress)


def _goal_episode_len(rec: dict) -> int:
    """Steps in the episode: ``episode_len``, else the last change point."""
    if rec.get("episode_len") is not None:
        return max(1, int(round(float(rec["episode_len"]))))
    steps = [step for entry in rec["goal_progress"] for step, _ in entry["trace"]]
    return max(steps, default=1)


def expand_trace(trace: list[list[int]], n_steps: int) -> np.ndarray:
    """Dense 0/1 array over steps ``1..n_steps`` (index ``s - 1``) from change points."""
    out = np.zeros(n_steps, dtype=np.uint8)
    for j, (step, value) in enumerate(trace):
        end = trace[j + 1][0] if j + 1 < len(trace) else n_steps + 1
        out[max(0, int(step) - 1) : max(0, min(int(end), n_steps + 1) - 1)] = int(value)
    return out


def trace_true_spans(trace: list[list[int]], n_steps: int) -> list[tuple[int, int]]:
    """``(start_step, n_steps_true)`` runs where the trace is true, for ``broken_barh``."""
    spans: list[tuple[int, int]] = []
    for j, (step, value) in enumerate(trace):
        if not value or step > n_steps:
            continue
        end = trace[j + 1][0] if j + 1 < len(trace) else n_steps + 1
        spans.append((int(step), int(min(end, n_steps + 1) - step)))
    return spans


def _finite(value: float | int | None) -> float | None:
    """``value`` as a float, or None when missing or NaN (how JSON stores "never")."""
    if value is None or math.isnan(float(value)):
        return None
    return float(value)


def goal_final_count(progress: list[dict], dense: np.ndarray) -> int:
    """Predicates satisfied at the end, from the stored ``final`` summaries.

    The trace OR-s hold steps, so its last step can read true where ``final``
    (the value on the episode's very last sim step) is false; the summary wins.
    Falls back to the trace's last step when a summary is missing.
    """
    finals = [_finite(entry.get("final")) for entry in progress]
    if any(v is None for v in finals):
        return int(dense[:, -1].sum())
    return sum(1 for v in finals if v is not None and v > 0.5)


def _short_label(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= _GOAL_LABEL_MAX_CHARS:
        return text
    return text[: _GOAL_LABEL_MAX_CHARS - 3] + "..."


def goal_step_dt(rec: dict, fallback_fps: float | None) -> float | None:
    """Seconds per episode step: the record's ``step_dt_s``, else ``1 / fallback_fps``, else None."""
    step_dt = _finite(rec.get("step_dt_s"))
    if step_dt is None and fallback_fps:
        step_dt = 1.0 / fallback_fps
    return step_dt


def _format_step(step: float, step_dt: float | None) -> str:
    if not step_dt:
        return f"{step:g}"
    seconds = step * step_dt
    return f"{step:g} ({seconds:.1f} s)" if seconds >= 10 else f"{step:g} ({seconds:.2f} s)"


@dataclasses.dataclass(frozen=True)
class SettleTermSeries:
    """Per-row settle-gate inputs from one episode's step-metrics CSV."""

    steps: np.ndarray
    """``(rows,)`` episode-clock step of each row; hold steps repeat a step."""
    lin: np.ndarray
    """``(rows, k)`` linear speed (m/s) of goal-object slot ``k``."""
    ang: np.ndarray
    """``(rows, k)`` angular speed (rad/s) of goal-object slot ``k``."""
    finger: np.ndarray
    """``(rows,)`` finger joint position; NaN where the gate skipped the gripper."""


def load_settle_terms(
    csv_path: str | Path, n_steps: int, n_objects: int | None = None
) -> SettleTermSeries | None:
    """Settle-gate series of one episode, or None when the CSV can't be placed on its clock.

    Rows are placed by the ``episode_step`` column, dropping rows parked before
    the episode's first counted step; the last one must equal ``n_steps``.
    CSVs written before that column existed are placed at ``step + 1``, which
    is only right without hold steps, so they need exactly ``n_steps`` rows.

    ``n_objects`` is the number of goal-object slots to read (the record's
    ``goal_objects``); None reads every slot present and drops trailing
    all-zero ones, which are padding past the end of a shorter goal.
    """
    df = pd.read_csv(csv_path)
    if "goal_finger_pos" not in df.columns:
        return None
    if "episode_step" in df.columns:
        df = df[df["episode_step"] >= 1]
        if df.empty or int(df["episode_step"].max()) != n_steps:
            return None
        steps = df["episode_step"].to_numpy(dtype=np.int64)
    else:
        if len(df) != n_steps:
            return None
        steps = np.arange(1, n_steps + 1, dtype=np.int64)
    n_present = 0
    while f"goal_obj_{n_present}_lin_vel" in df.columns and f"goal_obj_{n_present}_ang_vel" in df.columns:
        n_present += 1
    if n_objects is None:
        n_objects = n_present
        while n_objects > 0 and not (
            df[f"goal_obj_{n_objects - 1}_lin_vel"].to_numpy().any()
            or df[f"goal_obj_{n_objects - 1}_ang_vel"].to_numpy().any()
        ):
            n_objects -= 1
    if n_objects < 1 or n_objects > n_present:
        return None
    lin = np.stack([df[f"goal_obj_{k}_lin_vel"].to_numpy(dtype=float) for k in range(n_objects)], axis=1)
    ang = np.stack([df[f"goal_obj_{k}_ang_vel"].to_numpy(dtype=float) for k in range(n_objects)], axis=1)
    return SettleTermSeries(
        steps=steps, lin=lin, ang=ang, finger=df["goal_finger_pos"].to_numpy(dtype=float)
    )


def goal_slot_names(rec: dict, n_slots: int) -> list[str]:
    """Block name of each goal-object slot: the record's ``goal_objects``, else a guess.

    Older records lack ``goal_objects``; the slots are the goal's blocks sorted by
    name, so the sorted ``block_*`` tokens of the predicate labels are used when
    their count matches. Otherwise the slots stay ``slot k``.
    """
    names = rec.get("goal_objects")
    if names and len(names) == n_slots:
        return [str(name) for name in names]
    tokens = sorted(
        {
            token
            for entry in rec.get("goal_progress") or []
            for token in re.findall(r"\bblock_\w+", str(entry.get("predicate", "")))
        }
    )
    if len(tokens) == n_slots:
        return tokens
    return [f"slot {k}" for k in range(n_slots)]


def _first_true_step(trace: list[list[int]]) -> int | None:
    return next((int(step) for step, value in trace if value), None)


def _draw_settle_panels(
    ax_lin: Axes, ax_ang: Axes, ax_finger: Axes, series: SettleTermSeries, slot_names: list[str]
) -> None:
    """Per-block speeds and the finger joint, each against its settle-gate threshold."""
    for ax, values, limit, unit, name in (
        (ax_lin, series.lin, _SETTLED_LIN_VEL, "m/s", "linear speed"),
        (ax_ang, series.ang, _SETTLED_ANG_VEL, "rad/s", "angular speed"),
    ):
        for k, slot in enumerate(slot_names):
            ax.plot(series.steps, np.maximum(values[:, k], _SPEED_LOG_FLOOR), linewidth=1, label=slot)
        ax.axhline(limit, color="red", linestyle="--", linewidth=1, label=f"settled <= {limit:g} {unit}")
        ax.set_yscale("log")
        ax.set_ylabel(f"{name} ({unit})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    if np.isnan(series.finger).all():
        ax_finger.text(
            0.5, 0.5, "finger joint not read (gripper check skipped)",
            transform=ax_finger.transAxes, ha="center", va="center", color="gray",
        )
    else:
        ax_finger.plot(series.steps, series.finger, color="black", linewidth=1, label="finger_joint")
        ax_finger.axhline(
            _GRIPPER_OPEN_BELOW, color="red", linestyle="--", linewidth=1,
            label=f"open below {_GRIPPER_OPEN_BELOW:g}",
        )
        ax_finger.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
    ax_finger.set_ylabel("finger joint")
    ax_finger.grid(True, alpha=0.3)


def _render_goal_progress(job: GoalPlotJob) -> str:
    """Write ``episode_{label}/goal_progress.png``.

    Satisfied count on top, then one row per goal predicate and per
    ``goal_success`` limb (blocks settled, gripper open, goal_success). With a
    step-metrics CSV, panels below show what the settle gate read: each goal
    block's linear and angular speed and the finger joint, against its thresholds.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    rec = job.record
    progress: list[dict] = rec["goal_progress"]
    n_steps = _goal_episode_len(rec)
    total = len(progress)
    dense = np.stack([expand_trace(entry["trace"], n_steps) for entry in progress])
    counts = dense.sum(axis=0)
    final_count = goal_final_count(progress, dense)
    step_dt = goal_step_dt(rec, job.fallback_fps)
    gates = [
        (label, color, rec[f"{key}_trace"])
        for key, label, color in _GATE_ROWS
        if rec.get(f"{key}_trace")
    ]
    n_gates = len(gates)
    series: SettleTermSeries | None = None
    if job.step_metrics_csv is not None:
        goal_objects = rec.get("goal_objects")
        series = load_settle_terms(
            job.step_metrics_csv, n_steps, len(goal_objects) if goal_objects else None
        )

    colors = plt.get_cmap("tab10")
    # Predicate rows sit above the gate rows, with a gap between the two groups.
    pred_offset = n_gates + 0.6 if n_gates else 0.0
    rows_top = pred_offset + total - 0.4
    rows_height = max(1.6, 0.45 * (total + n_gates + (0.6 if n_gates else 0.0)))
    height_ratios = [3.2, rows_height]
    if series is not None:
        height_ratios += [2.2, 2.2, 1.6]
    fig, axes = plt.subplots(
        len(height_ratios),
        1,
        figsize=(14, sum(height_ratios) + 1.4),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
        layout="constrained",
    )
    ax_count, ax_rows = axes[0], axes[1]
    steps = np.arange(1, n_steps + 1)
    x_max = n_steps + 1

    ax_count.step(np.append(steps, x_max), np.append(counts, counts[-1]), where="post", color="tab:blue")
    ax_count.axhline(total, color="gray", linestyle="--", linewidth=1, label=f"total = {total}")
    success_trace = rec.get("goal_success_trace")
    if success_trace:
        for start, width in trace_true_spans(success_trace, n_steps):
            ax_count.axvspan(start, start + width, color="tab:green", alpha=0.2, linewidth=0)
        if any(v for _, v in success_trace):
            ax_count.plot([], [], color="tab:green", alpha=0.4, linewidth=8, label="goal_success")
    ax_count.set_ylim(-0.3, total + 0.5)
    ax_count.set_yticks(range(total + 1))
    ax_count.set_ylabel("goal predicates satisfied")
    ax_count.grid(True, alpha=0.3)
    ax_count.legend(loc="upper left", fontsize=8)
    if step_dt:
        top = ax_count.secondary_xaxis(
            "top", functions=(lambda s: s * step_dt, lambda t: t / step_dt)
        )
        top.set_xlabel("episode time (s)")

    # (y, color, trace, legend text, first true step) of every row, predicates first.
    rows: list[tuple[float, str | tuple, list[list[int]], str, float | None]] = [
        (
            pred_offset + total - 1 - c,
            colors(c % 10),
            entry["trace"],
            f"{c}: {_short_label(str(entry.get('predicate', '')))}",
            _finite(entry.get("first_true_step")),
        )
        for c, entry in enumerate(progress)
    ]
    for j, (label, color, trace) in enumerate(gates):
        first_true = _first_true_step(trace)
        rows.append(
            (n_gates - 1 - j, color, trace, label, None if first_true is None else float(first_true))
        )
    handles: list[Patch] = []
    for y, color, trace, text, first_true in rows:
        spans = trace_true_spans(trace, n_steps)
        ax_rows.broken_barh(spans, (y - 0.4, 0.8), facecolors=color)
        change_steps = [step for step, _ in trace[1:] if step <= n_steps]
        if change_steps:
            ax_rows.vlines(change_steps, y - 0.45, y + 0.45, colors="black", linewidth=0.6)
        n_switches = len(trace) - 1
        detail = f"{n_switches} switch{'es' if n_switches != 1 else ''}"
        if first_true is not None:
            detail += f", first true @ {_format_step(first_true, step_dt)}"
        handles.append(Patch(color=color, label=f"{text} ({detail})"))
    if n_gates:
        ax_rows.axhline(n_gates - 0.2, color="gray", linewidth=0.8)
    ax_rows.set_yticks([row[0] for row in rows])
    ax_rows.set_yticklabels(
        [str(entry.get("index", c)) for c, entry in enumerate(progress)]
        + [label for label, _, _ in gates]
    )
    ax_rows.set_ylim(-0.6, rows_top)
    ax_rows.set_ylabel("goal_progress index")
    ax_rows.set_xlim(1, x_max)
    ax_rows.grid(True, axis="x", alpha=0.3)
    ax_rows.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False
    )

    if series is not None:
        _draw_settle_panels(axes[2], axes[3], axes[4], series, goal_slot_names(rec, series.lin.shape[1]))
    axes[-1].set_xlabel("episode step (env clock)")

    length = _format_step(n_steps, step_dt)
    fig.suptitle(
        f"episode {job.label}: {final_count}/{total} goal predicates satisfied at the end, "
        f"{length} steps"
    )
    footnote = (
        "Steps are the env episode clock: hold steps (PD-switch settles, replay gap holds, "
        "chunk-sync waits) are not counted.\n"
        "goal_success = all goal predicates & blocks settled & gripper open. Blocks settled: "
        f"every goal block has linear speed <= {_SETTLED_LIN_VEL:g} m/s and angular speed <= "
        f"{_SETTLED_ANG_VEL:g} rad/s (PhysX root velocity, read once per control step). "
        f"Gripper open: finger_joint < {_GRIPPER_OPEN_BELOW:g}."
    )
    if job.step_metrics_csv is not None and series is None:
        footnote += (
            f"\nSpeed panels omitted: {Path(job.step_metrics_csv).name} does not line up with "
            f"this episode's {n_steps} steps."
        )
    fig.text(0.01, -0.01, footnote, fontsize=7, color="gray", va="top")
    out_dir = Path(job.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / GOAL_PROGRESS_FILENAME
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return f"episode {job.label}: goal progress {final_count}/{total} -> {out_path}"


def _try_render_goal_progress(job: GoalPlotJob) -> tuple[int, str | None, str]:
    """``(label, error, message)``; one bad episode must not abort the rest."""
    try:
        return job.label, None, _render_goal_progress(job)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        return job.label, error, f"episode {job.label}: goal progress FAILED ({error})"


def _lerobot_fps(eval_dir: Path) -> float | None:
    """The run's LeRobot fps, if it recorded a dataset (older runs lack ``step_dt_s``)."""
    try:
        info = json.loads((_find_lerobot_root(eval_dir) / "meta" / "info.json").read_text())
    except (OSError, RuntimeError, json.JSONDecodeError):
        return None
    fps = info.get("fps") if isinstance(info, dict) else None
    return float(fps) if fps else None


def _goal_plot_jobs(
    eval_dir: Path, output_root: Path, episodes: tuple[int, ...] | None
) -> tuple[list[GoalPlotJob], int]:
    """``(jobs, n_without_traces)`` for every record carrying goal-predicate traces."""
    records_path = eval_dir / "eval_episodes.json"
    if not records_path.exists():
        return [], 0
    records = json.loads(records_path.read_text())
    wanted = set(episodes) if episodes is not None else None
    fallback_fps: float | None = None
    if any(rec.get("step_dt_s") is None for rec in records if _has_goal_traces(rec)):
        fallback_fps = _lerobot_fps(eval_dir)
    jobs: list[GoalPlotJob] = []
    without = 0
    for position, rec in enumerate(records):
        label = goal_plot_label(rec, position)
        if wanted is not None and label not in wanted:
            continue
        if not _has_goal_traces(rec):
            without += 1
            continue
        step_metrics_csv: str | None = None
        if rec.get("sim_env_idx") is not None and rec.get("episode_idx_for_sim_env") is not None:
            csv_path = (
                eval_dir
                / _STEP_METRICS_DIRNAME
                / f"env{int(rec['sim_env_idx'])}_episode_{int(rec['episode_idx_for_sim_env']):04d}.csv"
            )
            if csv_path.exists():
                step_metrics_csv = str(csv_path)
        jobs.append(
            GoalPlotJob(
                label=label,
                record=rec,
                output_dir=str(output_root / f"episode_{label}"),
                fallback_fps=fallback_fps,
                step_metrics_csv=step_metrics_csv,
            )
        )
    return sorted(jobs, key=lambda j: j.label), without


def _render_goal_progress_plots(
    eval_dir: Path, output_root: Path, args: Args
) -> tuple[int, dict[int, str]]:
    """Draw every episode's goal_progress.png; ``(n_jobs, failures)``."""
    jobs, without = _goal_plot_jobs(eval_dir, output_root, args.episodes)
    if without:
        print(
            f"[offline-diag] {without} episode(s) have no goal-predicate trace (run predates "
            "it, or the task has no PDDL goal); no goal_progress.png for them"
        )
    if not jobs:
        return 0, {}
    print(f"[offline-diag] goal progress for {len(jobs)} episode(s) into {output_root}")
    failures: dict[int, str] = {}
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.num_workers, len(jobs))) as pool:
            results = list(pool.map(_try_render_goal_progress, jobs))
    else:
        results = [_try_render_goal_progress(job) for job in jobs]
    for label, error, message in results:
        print(f"[offline-diag] {message}")
        if error is not None:
            failures[label] = error
    return len(jobs), failures


def _load_launch_params(eval_dir: Path) -> dict[str, object]:
    params_path = eval_dir / "launch_params.json"
    if not params_path.exists():
        return {}
    return json.loads(params_path.read_text())


def _is_replay_run(eval_dir: Path, launch_params: dict[str, object]) -> bool:
    """Whether the run replayed recorded actions; it decides the legacy row layout.

    Runs without launch_params.json are recognised by the per-episode replay
    metrics (or the older ``replay_results.json``) only replay writes.
    """
    if "replay" in launch_params:
        return bool(launch_params["replay"])
    if launch_params.get("policy") is not None:
        return launch_params["policy"] == "replay"
    if (eval_dir / "replay_results.json").exists():
        return True
    episodes_path = eval_dir / "eval_episodes.json"
    if not episodes_path.exists():
        return False
    records = json.loads(episodes_path.read_text())
    return any(isinstance(rec, dict) and "replay_num_frames" in rec for rec in records)


def _resolve_gt_dataset_path(
    args: Args, launch_params: dict[str, object]
) -> str | None:
    if not args.overlay_ground_truth:
        return None
    if args.gt_dataset_path:
        return args.gt_dataset_path
    dataset_path = launch_params.get("dataset_path")
    if not dataset_path:
        print("[offline-diag] no dataset_path in launch_params.json; no ground-truth overlay")
        return None
    return str(dataset_path)


def _resolve_env_spacing(args: Args, launch_params: dict[str, object]) -> float | None:
    """Grid spacing used to put recorded EE positions back in the env frame."""
    if not args.subtract_env_origin:
        return None
    if args.env_spacing is not None:
        return args.env_spacing
    spacing = launch_params.get("env_spacing")
    return float(spacing) if spacing else _DEFAULT_ENV_SPACING


def main(args: Args) -> None:
    eval_dir = Path(args.eval_dir).expanduser().resolve()
    if not eval_dir.is_dir():
        raise FileNotFoundError(f"--eval-dir does not exist: {eval_dir}")

    # First and on its own: it needs only eval_episodes.json, so neither a missing
    # LeRobot dataset nor a control-plot failure costs an episode its goal plot.
    output_root = eval_dir / args.output_subdir
    n_goal, goal_failures = _render_goal_progress_plots(eval_dir, output_root, args)
    if goal_failures:
        print(
            f"[offline-diag] {len(goal_failures)}/{n_goal} goal-progress plot(s) failed: "
            f"{sorted(goal_failures)}"
        )
    all_goal_failed = n_goal > 0 and len(goal_failures) == n_goal
    if args.goal_progress_only:
        if all_goal_failed:
            raise RuntimeError(
                "Every goal-progress plot failed; first error: "
                f"{goal_failures[min(goal_failures)]}"
            )
        print(f"[offline-diag] done: {output_root}")
        return

    try:
        lerobot_root = _find_lerobot_root(eval_dir)
    except FileNotFoundError:
        if n_goal == 0 or all_goal_failed:
            raise
        print(
            f"[offline-diag] no LeRobot dataset under {eval_dir} (run without "
            "--save-lerobot): control-diagnostic plots and first-frame comparisons skipped"
        )
        print(f"[offline-diag] done: {output_root}")
        return
    specs = _load_episode_specs(eval_dir, lerobot_root)
    if args.episodes is not None:
        wanted = set(args.episodes)
        specs = [s for s in specs if s.label in wanted]
        missing = wanted - {s.label for s in specs}
        if missing:
            raise ValueError(f"--episodes not found in this run: {sorted(missing)}")
    if not specs:
        raise RuntimeError(f"No recorded episodes found under {lerobot_root}")

    launch_params = _load_launch_params(eval_dir)
    gt_dataset_path = _resolve_gt_dataset_path(args, launch_params)
    env_spacing = _resolve_env_spacing(args, launch_params)

    specs = _enrich_replay_specs_with_expert_videos(specs, gt_dataset_path)

    output_root.mkdir(parents=True, exist_ok=True)
    first_frames_root = eval_dir / args.first_frames_subdir
    first_frames_root.mkdir(parents=True, exist_ok=True)

    row_layout_marked = _has_row_layout_marker(lerobot_root)
    replay_recording = _is_replay_run(eval_dir, launch_params)

    print(f"[offline-diag] dataset: {lerobot_root}")
    print(
        "[offline-diag] row layout: "
        + (
            "matches the source dataset"
            if row_layout_marked
            else f"legacy {'replay' if replay_recording else 'policy-eval'} (realigned here)"
        )
    )
    print(f"[offline-diag] ground truth: {gt_dataset_path or 'none'}")
    print(
        "[offline-diag] EE frame: "
        + (f"env-local (env_spacing={env_spacing})" if env_spacing else "world (as recorded)")
    )
    print(f"[offline-diag] rendering {len(specs)} episode(s) into {output_root}")
    print(f"[offline-diag] first-frame comparisons into {first_frames_root}")

    jobs = [
        PlotJob(
            spec=spec,
            lerobot_root=spec.lerobot_root or str(lerobot_root),
            output_dir=str(output_root / f"episode_{spec.label}"),
            first_frames_dir=str(first_frames_root),
            gt_dataset_path=gt_dataset_path,
            env_spacing=env_spacing,
            row_layout_marked=(
                _has_row_layout_marker(Path(spec.lerobot_root))
                if spec.lerobot_root else row_layout_marked
            ),
            replay_recording=replay_recording,
        )
        for spec in specs
    ]

    failures: dict[int, str] = {}
    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.num_workers, len(jobs))) as pool:
            results = pool.map(_try_render_episode, jobs)
            for label, error, message in results:
                print(f"[offline-diag] {message}")
                if error is not None:
                    failures[label] = error
    else:
        for job in jobs:
            label, error, message = _try_render_episode(job)
            print(f"[offline-diag] {message}")
            if error is not None:
                failures[label] = error

    if failures:
        print(
            f"[offline-diag] {len(failures)}/{len(jobs)} episode(s) failed: "
            f"{sorted(failures)}"
        )
        if len(failures) == len(jobs):
            raise RuntimeError(
                "Every episode failed to render; first error: "
                f"{failures[min(failures)]}"
            )

    print(f"[offline-diag] done: {output_root}, {first_frames_root}")


if __name__ == "__main__":
    main(tyro.cli(Args))
