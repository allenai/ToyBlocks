"""Dataset replay as an action source for the queue-driven eval loop.

:class:`ReplayPolicy` plays the recorded actions of each queue episode back
through ``env.step()`` inside :func:`blocksuite.eval.run_eval._run_eval`, in
place of a policy. The queue setup (``apply_setup`` in ``TemplateEnv._reset_idx``)
restores the recorded frame-0 scene; this class only supplies the per-step
action and measures how far the simulated joints/objects drift from the
recorded ground truth.

The emitted action is 9-dimensional: 7 absolute joint targets, the binary
gripper command, and the recorded ``action.pd_mode`` as the 9th channel. The
eval loop hands that channel to :class:`~blocksuite.eval.pd_mode_controller.PDModeController`
exactly as it would a policy's prediction, so gain switching (including the
TAMP-style settle substeps on a switch) lives in one place.
"""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from blocksuite.utils.episode_dataset import _episode_data_coords
from blocksuite.utils.tamp_dataset import (
    _load_merged_info,
    _shard_path_for,
    _table_to_dict,
    robot_base_pose_np,
    tamp_action_to_env_action,
)
from blocksuite.utils.tf import world_pose_to_local

logger = logging.getLogger(__name__)

# Number of arm joints the recorded ``joint_position`` / ``joint_target`` cover.
_N_ARM = 7
# Data shards kept in memory. Queue episodes are served in index order, so
# consecutive episodes almost always share a shard; two shards cover the
# boundary between neighbours without holding a whole dataset resident.
_SHARD_CACHE_SIZE = 2
# Unrecorded-tick detection (see _detect_unrecorded_ticks): the smallest one-tick
# arm motion (rad, from the recorded velocities) the motion ratio is trusted at,
# the ratio of recorded to one-tick motion that flags at least one missing tick,
# the largest change (rad) between consecutive arm targets that counts as held,
# and the most ticks one gap can stand for (the RL hand-off skips two).
_GAP_MIN_TICK_MOTION = 1e-3
_GAP_MIN_RATIO = 1.6
_GAP_TARGET_TOL = 1e-5
_GAP_MAX_TICKS = 2


def _recorded_pd_modes(episode_data: dict) -> np.ndarray | None:
    """The episode's commanded pd_mode series, or ``None`` if it commanded none.

    The recorder writes ``action.pd_mode`` for every run, filling it with NaN
    where nothing drives the channel, so the column being present says nothing
    about whether a mode was ever commanded.
    """
    values = episode_data.get("action.pd_mode")
    if values is None:
        return None
    modes = np.asarray(values, dtype=np.float32).reshape(-1)
    return None if bool(np.isnan(modes).all()) else modes


def _recorded_object_pose(data: dict, obj_name: str, frame: int) -> np.ndarray | None:
    """Recorded env-local ``(7,)`` pose for *obj_name*, or ``None`` if not recorded.

    Datasets carry an ``object_pose`` column for every block in the PDDL domain,
    not only the ones a given problem spawns; the unused ones are written as
    scalar ``NaN``. The column being present therefore says nothing about whether
    that object was in the recorded scene, so this checks the value rather than
    the key — a 0-d NaN read as a pose raises ``IndexError`` at ``pose[:3]``.
    """
    col = f"observation.state.object_pose.{obj_name}"
    if col not in data:
        return None
    pose = np.asarray(data[col][frame], dtype=np.float32)
    if pose.ndim != 1 or pose.shape[0] < 7 or not np.isfinite(pose[:7]).all():
        return None
    return pose


def _load_joint_target_clip_bounds(
    stats_path: str | Path, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(q01, q99)`` of ``action.joint_target`` from a ``stats.json``, as ``(7,)`` tensors.

    These are the bands a q01/q99 action normalizer with a ``[-1, 1]`` clip
    (hvla's ``FrankaDataset``) squeezes every training label into, so clamping
    the replayed targets to them reproduces the labels such a policy learned.
    """
    with open(stats_path) as f:
        stats = json.load(f)
    entry = stats.get("action.joint_target")
    if entry is None or "q01" not in entry or "q99" not in entry:
        raise ValueError(
            f"{stats_path} has no 'action.joint_target' q01/q99 entry to clip replay targets to."
        )
    lo = torch.as_tensor(entry["q01"], dtype=torch.float32, device=device).reshape(-1)[:_N_ARM]
    hi = torch.as_tensor(entry["q99"], dtype=torch.float32, device=device).reshape(-1)[:_N_ARM]
    if lo.numel() != _N_ARM or hi.numel() != _N_ARM:
        raise ValueError(
            f"{stats_path}: 'action.joint_target' q01/q99 must cover {_N_ARM} arm joints "
            f"(got {lo.numel()} / {hi.numel()})."
        )
    return lo, hi


def _num_action_rows(episode_data: dict) -> int:
    """Number of leading rows that carry a recorded action.

    The recorders blank the terminal row's action block (zero joint targets,
    open gripper, ``pd_mode`` 0) and mark it ``action.is_valid`` False, so it
    is not an action TAMP took. Recordings without the column count every row.
    """
    n_rows = len(episode_data["observation.state.joint_position"])
    valid = episode_data.get("action.is_valid")
    if valid is None:
        return n_rows
    valid_rows = np.flatnonzero(np.asarray(valid, dtype=bool).reshape(-1))
    return int(valid_rows[-1]) + 1 if valid_rows.size else 0


def _detect_profile_switch_frame(episode_data: dict) -> int | None:
    """Return the first recorded frame where the PD profile changes.

    Only rows that carry an action are compared, so the blanked terminal row's
    ``pd_mode`` does not read as a switch.
    """
    n_actions = _num_action_rows(episode_data)
    pd = _recorded_pd_modes(episode_data)
    if pd is not None:
        changes = np.where(np.diff(pd[:n_actions]) != 0)[0]
        if len(changes):
            return int(changes[0] + 1)
    if "observation.state.current_stiffness" in episode_data:
        arm_stiff = np.asarray(episode_data["observation.state.current_stiffness"])[:n_actions, 0]
        changes = np.where(np.diff(arm_stiff) != 0)[0]
        if len(changes):
            return int(changes[0] + 1)
    return None


def _detect_unrecorded_ticks(episode_data: dict) -> dict[int, int]:
    """``{row: k}`` for rows recorded ``k`` ticks later than one step after the previous row.

    TAMP datagen can step the sim without recording the tick: with
    ``TMP3_MOTION_UNIFORM_STIFFNESS=1`` the hand-off to an RL policy still passes
    through the unrecorded stiffness-settle state. Replaying one action per row
    would leave the sim ``k`` ticks behind the recording from such a row on.

    A row is flagged when the arm moved at least ``_GAP_MIN_RATIO`` times as far
    as its recorded joint velocities allow in one tick (``0.5 (v[r-1] + v[r]) dt``,
    arm joints only) while ``pd_mode`` stayed unchanged and the arm target was
    held over both ``r-2 -> r-1`` and ``r-1 -> r``; ``k = round(ratio) - 1``,
    capped at ``_GAP_MAX_TICKS``. The first tick after a target jump is left out
    because the arm accelerates non-linearly through it, so the velocity
    average undercounts its travel. Rows that switch ``pd_mode`` are left out
    because their unrecorded settle is replayed by the PD controller
    (``DATASET_SWITCH_SETTLE_TICKS``). Recordings without timestamps or joint
    velocities report no gaps.
    """
    columns = (
        "timestamp",
        "observation.state.joint_position",
        "observation.state.joint_velocity",
        "action.joint_target",
    )
    if any(col not in episode_data for col in columns):
        return {}
    timestamps = np.asarray(episode_data["timestamp"], dtype=np.float64).reshape(-1)
    if timestamps.size < 2:
        return {}
    dt = float(np.median(np.diff(timestamps)))
    joint_pos = np.asarray(episode_data["observation.state.joint_position"], dtype=np.float64)
    joint_vel = np.asarray(episode_data["observation.state.joint_velocity"], dtype=np.float64)
    target = np.asarray(episode_data["action.joint_target"], dtype=np.float64)[:, :_N_ARM]
    moved = np.linalg.norm(np.diff(joint_pos[:, :_N_ARM], axis=0), axis=1)
    one_tick = np.linalg.norm(0.5 * (joint_vel[1:, :_N_ARM] + joint_vel[:-1, :_N_ARM]) * dt, axis=1)
    held = np.abs(np.diff(target, axis=0)).max(axis=1) <= _GAP_TARGET_TOL
    held_before = np.concatenate(([False], held[:-1]))
    candidate = (one_tick >= _GAP_MIN_TICK_MOTION) & held & held_before
    pd = _recorded_pd_modes(episode_data)
    if pd is not None:
        candidate &= (pd[1:] == pd[:-1]) | (np.isnan(pd[1:]) & np.isnan(pd[:-1]))
    ratio = np.where(candidate, moved / np.maximum(one_tick, _GAP_MIN_TICK_MOTION), 0.0)
    return {
        int(idx) + 1: min(int(round(float(ratio[idx]))) - 1, _GAP_MAX_TICKS)
        for idx in np.flatnonzero(ratio >= _GAP_MIN_RATIO)
    }


def _phase_slices(switch_frame: int, n_errors: int) -> tuple[slice, slice]:
    """Map a profile switch at recorded frame *switch_frame* to error-array slices.

    ``errors[i]`` scores the step that applied action row ``i`` (against recorded
    state row ``i + 1``), so the split falls on the first action under the new
    profile.
    """
    boundary = max(0, switch_frame)
    return slice(0, boundary), slice(boundary, n_errors)


def _phase_metrics(prefix: str, values: np.ndarray) -> dict[str, float]:
    """``{prefix}_{mean,max,final}`` of *values*, NaN when there are none."""
    if values.size == 0:
        nan = float("nan")
        return {f"{prefix}_mean": nan, f"{prefix}_max": nan, f"{prefix}_final": nan}
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_final": float(values[-1]),
    }


class _ShardCache:
    """Tiny LRU of decoded parquet shards keyed by path."""

    def __init__(self, capacity: int = _SHARD_CACHE_SIZE) -> None:
        self._capacity = capacity
        self._tables: OrderedDict[Path, pa.Table] = OrderedDict()

    def get(self, shard: Path) -> pa.Table:
        table = self._tables.get(shard)
        if table is not None:
            self._tables.move_to_end(shard)
            return table
        table = pq.read_table(str(shard))
        self._tables[shard] = table
        while len(self._tables) > self._capacity:
            self._tables.popitem(last=False)
        return table


def _resolve_episode_shard(root: Path, episode_index: int) -> Path:
    """Data shard holding *episode_index*'s rows.

    Prefers the per-episode ``data/chunk_index`` + ``data/file_index`` that
    LeRobot v3 records in ``meta/episodes`` (shards are split by size, not by
    episode count); falls back to :func:`tamp_dataset._shard_path_for`'s
    ``episode_index // chunks_size`` arithmetic for datasets without it.
    """
    coords = _episode_data_coords(root, {episode_index})
    if coords and episode_index in coords:
        chunk, file_ = coords[episode_index]
        for data_dir in (root / "data", root / "blocks" / "data"):
            shard = data_dir / f"chunk-{chunk:03d}" / f"file-{file_:03d}.parquet"
            if shard.is_file():
                return shard
    return _shard_path_for(root, _load_merged_info(root), episode_index)


class ReplayPolicy:
    """Replays each queue episode's recorded actions as a 9-dim action source.

    Per-env state is kept for every slot even though :func:`evaluate` restricts
    ``--replay`` to ``num_envs == 1`` (:meth:`get_action` advances every env's
    recording, so one env cannot hold through a PD-switch settle while the
    others step), so nothing here is hard-coded to slot 0.
    """

    def __init__(
        self,
        base_env,
        device: torch.device,
        num_envs: int,
        *,
        use_delta_actions: bool = False,
        clip_stats_path: str | None = None,
    ) -> None:
        """
        Args:
            clip_stats_path: Optional ``stats.json``; when given, every replayed
                arm joint target is clamped to its ``action.joint_target``
                ``[q01, q99]`` band, and each episode reports how many frames
                that clamp touched (``clipped_frames``) and by how much at most
                (``clip_max_rad``).
        """
        self._base_env = base_env
        self._device: torch.device = device
        self._num_envs: int = int(num_envs)
        self._use_delta_actions: bool = use_delta_actions
        self._shards = _ShardCache()
        self._clip_bounds: tuple[torch.Tensor, torch.Tensor] | None = (
            _load_joint_target_clip_bounds(clip_stats_path, device)
            if clip_stats_path
            else None
        )
        if self._clip_bounds is not None:
            logger.info(
                "clipping replayed joint targets to action.joint_target [q01, q99] from %s: "
                "q01=%s q99=%s",
                clip_stats_path,
                np.round(self._clip_bounds[0].cpu().numpy(), 4).tolist(),
                np.round(self._clip_bounds[1].cpu().numpy(), 4).tolist(),
            )
        self._clipped_frames: list[int] = [0] * self._num_envs
        self._clip_max: list[float] = [0.0] * self._num_envs

        self._episode_data: list[dict | None] = [None] * self._num_envs
        self._episode_ids: list[str] = ["<none>"] * self._num_envs
        self._step: list[int] = [0] * self._num_envs
        self._n_frames: list[int] = [0] * self._num_envs
        self._n_actions: list[int] = [0] * self._num_envs
        self._profile_switch_frame: list[int | None] = [None] * self._num_envs
        self._joint_pos_errors: list[list[float]] = [[] for _ in range(self._num_envs)]
        self._object_pos_errors: list[dict[str, list[float]]] = [
            {} for _ in range(self._num_envs)
        ]
        self._object_quat_errors: list[dict[str, list[float]]] = [
            {} for _ in range(self._num_envs)
        ]
        # Unrecorded ticks are replayed as holds of the previous row's action
        # (see pop_gap_holds). A delta replay resolves each row's target from the
        # sim state when the row is emitted, which for the row after a gap is
        # before its hold has run, so delta replays leave gaps unfilled.
        self._fill_gaps: bool = not use_delta_actions
        self._gap_ticks: list[dict[int, int]] = [{} for _ in range(self._num_envs)]
        self._last_action8: list[torch.Tensor | None] = [None] * self._num_envs
        self._pending_gap: list[tuple[int, torch.Tensor] | None] = [None] * self._num_envs
        self._gap_score_row: list[int | None] = [None] * self._num_envs
        self._gap_hold_left: list[int] = [0] * self._num_envs
        self._unrecorded_ticks: list[int] = [0] * self._num_envs

    # ------------------------------------------------------------------ loading

    def _load_episode(self, setup: dict) -> dict:
        root = Path(setup["source_dataset_path"])
        episode_index = int(setup["episode_index"])
        shard = _resolve_episode_shard(root, episode_index)
        table = self._shards.get(shard)
        ep_col = table.column("episode_index").to_numpy(zero_copy_only=False)
        mask = ep_col == episode_index
        if not bool(mask.any()):
            raise RuntimeError(
                f"No rows with episode_index={episode_index} in {shard} "
                f"(setup episode_id={setup.get('episode_id', '<unknown>')!r})."
            )
        return _table_to_dict(table.filter(pa.array(mask)))

    def on_reset(self, env_ids: Iterable[int], setups: Sequence[dict | None]) -> None:
        """Load the recorded episode each env in *env_ids* was just reset into.

        ``setups[k]`` is the queue setup for ``env_ids[k]``; ``None`` (drained
        queue) leaves that env idle -- it emits a hold action and is never
        scored, matching the phantom-episode filter in the eval loop.

        Raises ``ValueError`` when the recording has no usable
        ``action.pd_mode``: without it there is no mode to drive the PD
        controller with, and silently replaying under the wrong gains would
        report divergence that has nothing to do with the recorded physics.
        """
        for env_idx, setup in zip(env_ids, setups):
            i = int(env_idx)
            self._step[i] = 0
            self._clipped_frames[i] = 0
            self._clip_max[i] = 0.0
            self._joint_pos_errors[i] = []
            self._object_pos_errors[i] = {}
            self._object_quat_errors[i] = {}
            self._last_action8[i] = None
            self._pending_gap[i] = None
            self._gap_score_row[i] = None
            self._gap_hold_left[i] = 0
            self._unrecorded_ticks[i] = 0
            if setup is None:
                self._episode_data[i] = None
                self._episode_ids[i] = "<none>"
                self._n_frames[i] = 0
                self._n_actions[i] = 0
                self._profile_switch_frame[i] = None
                self._gap_ticks[i] = {}
                continue

            data = self._load_episode(setup)
            ep_id = str(setup.get("episode_id", setup.get("episode_index", "<unknown>")))
            if _recorded_pd_modes(data) is None:
                raise ValueError(
                    f"episode {ep_id!r} from {setup['source_dataset_path']} has no "
                    "usable 'action.pd_mode' column (missing or all-NaN). Replay "
                    "drives the PD controller from the recorded mode, so the "
                    "dataset must record it."
                )
            self._episode_data[i] = data
            self._episode_ids[i] = ep_id
            self._n_frames[i] = int(len(data["observation.state.joint_position"]))
            self._n_actions[i] = _num_action_rows(data)
            self._profile_switch_frame[i] = _detect_profile_switch_frame(data)
            gaps = _detect_unrecorded_ticks(data)
            self._gap_ticks[i] = gaps if self._fill_gaps else {}
            logger.info(
                "env %d: replaying episode %s (%d frames%s%s)",
                i,
                ep_id,
                self._n_frames[i],
                (
                    f", PD switch at frame {self._profile_switch_frame[i]}"
                    if self._profile_switch_frame[i] is not None
                    else ""
                ),
                (
                    f", unrecorded ticks (row: count) {gaps}"
                    + ("" if self._fill_gaps else " not replayed (delta actions)")
                    if gaps
                    else ""
                ),
            )

    # ------------------------------------------------------------------ acting

    def _hold_action(self, env_idx: int) -> torch.Tensor:
        """9-dim action for an idle env: hold the arm, gripper open, no PD request.

        A NaN pd_mode lands in the controller's dead zone, so the gains of an
        idle env are left alone.
        """
        joint_pos = self._base_env.robot.articulation.data.joint_pos[env_idx, :_N_ARM]
        return torch.cat(
            [
                joint_pos.to(dtype=torch.float32),
                torch.tensor([1.0, float("nan")], device=self._device, dtype=torch.float32),
            ]
        )

    def get_action(self, obs: dict[str, Any]) -> torch.Tensor:
        """Return the ``(num_envs, 9)`` action for the current replay step.

        Mirrors a policy's ``get_action(obs)`` interface; *obs* is unused. Row
        ``t`` of each env's recording becomes 7 joint targets + the binary
        gripper command (:func:`tamp_action_to_env_action`) plus the recorded
        ``action.pd_mode[t]`` as the 9th channel, then the env's counter
        advances so the next call returns the following frame. Only rows that
        carry a recorded action are replayed (:func:`_num_action_rows`); the
        blanked terminal row is not. A row recorded after ticks the recording
        skipped also stages a hold of the previous row's action for those ticks
        (:meth:`pop_gap_holds`).
        """
        rows: list[torch.Tensor] = []
        for i in range(self._num_envs):
            data = self._episode_data[i]
            if data is None or self._n_actions[i] == 0:
                rows.append(self._hold_action(i))
                continue
            t = min(self._step[i], self._n_actions[i] - 1)
            current_joint_pos = None
            if self._use_delta_actions:
                current_joint_pos = self._base_env.robot.articulation.data.joint_pos[i, :_N_ARM]
            action8 = tamp_action_to_env_action(
                data,
                t,
                self._device,
                use_delta_actions=self._use_delta_actions,
                current_joint_pos=current_joint_pos,
            )[0]
            if self._clip_bounds is not None:
                lo, hi = self._clip_bounds
                clipped = torch.maximum(torch.minimum(action8[:_N_ARM], hi), lo)
                excess = float((action8[:_N_ARM] - clipped).abs().max().item())
                if excess > 0.0:
                    self._clipped_frames[i] += 1
                    self._clip_max[i] = max(self._clip_max[i], excess)
                action8 = torch.cat([clipped, action8[_N_ARM:]])
            gap = self._gap_ticks[i].get(t)
            previous_action8 = self._last_action8[i]
            if gap is not None and t == self._step[i] and previous_action8 is not None:
                self._pending_gap[i] = (gap, previous_action8)
            self._last_action8[i] = action8
            pd_mode = float(np.asarray(data["action.pd_mode"][t], dtype=np.float32).reshape(-1)[0])
            rows.append(
                torch.cat(
                    [action8, torch.tensor([pd_mode], device=self._device, dtype=torch.float32)]
                )
            )
            self._step[i] += 1
        return torch.stack(rows, dim=0)

    def done_indices(self) -> list[int]:
        """Envs whose recording is exhausted after the action just emitted.

        Call after :meth:`get_action` and before ``env.step`` so the eval loop
        can ``request_episode_end`` on that same step: the last valid recorded
        action is still applied, the step consumes the truncation, and the
        episode is finalized with full metrics.
        """
        return [
            i
            for i in range(self._num_envs)
            if self._episode_data[i] is not None and self._step[i] >= self._n_actions[i]
        ]

    def current_targets(self, env_idx: int) -> set[str] | None:
        """The block the recorded step moves (``obj``, ``action.high_level_args[1]``
        in every TAMP ActionConfig layout) for the row last emitted; ``None``
        when the row carries none (e.g. ``done``)."""
        data = self._episode_data[env_idx]
        if data is None or "action.high_level_args" not in data or self._step[env_idx] == 0:
            return None
        row = min(self._step[env_idx], self._n_frames[env_idx]) - 1
        raw = data["action.high_level_args"][row]
        hl_args = json.loads(raw) if raw else []
        return {hl_args[1]} if len(hl_args) > 1 else None

    def pop_gap_holds(
        self, available: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Holds that replay the ticks the recording skipped before the rows just emitted.

        TAMP kept commanding the previous row's targets through the skipped
        ticks (see :func:`_detect_unrecorded_ticks`), so an env that
        :meth:`get_action` just moved onto such a row holds the previous row's
        8-dim action for that many ticks before the row runs. Call once per
        step after :meth:`get_action`; the eval loop runs the holds as
        deferred-settle hold steps.

        Args:
            available: ``(num_envs,)`` bool, the envs not already holding. A
                staged gap on an env that started a PD-switch settle on the
                same step is dropped (with a warning) and its ticks are not
                replayed.

        Returns:
            ``(env_mask, ticks, hold_action)`` in the layout
            ``DeferredSettle.start`` takes, or ``None`` when no env starts a
            gap hold this step.
        """
        if all(pending is None for pending in self._pending_gap):
            return None
        env_mask = torch.zeros(self._num_envs, dtype=torch.bool, device=self._device)
        ticks = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        hold_action = torch.zeros(
            self._num_envs, _N_ARM + 1, dtype=torch.float32, device=self._device
        )
        for i, pending in enumerate(self._pending_gap):
            if pending is None:
                continue
            self._pending_gap[i] = None
            n_ticks, action8 = pending
            if not bool(available[i]):
                logger.warning(
                    "env %d: episode %s settles a PD switch on row %d, so the %d "
                    "unrecorded tick(s) before that row are not replayed",
                    i,
                    self._episode_ids[i],
                    self._step[i] - 1,
                    n_ticks,
                )
                self._gap_score_row[i] = None
                continue
            env_mask[i] = True
            ticks[i] = n_ticks
            hold_action[i] = action8
            self._gap_hold_left[i] = n_ticks
            self._unrecorded_ticks[i] += n_ticks
        if not bool(env_mask.any()):
            return None
        return env_mask, ticks, hold_action

    # ------------------------------------------------------------- divergence

    def record_divergence(
        self, skip_env_ids: Iterable[int] = (), holding_env_ids: Iterable[int] = ()
    ) -> None:
        """Compare each env's sim state to the recorded state the applied row led to.

        TAMP rows pair a state with the action taken *from* it, so after action
        row ``t`` (selected by the preceding :meth:`get_action`) the sim should
        match recorded state row ``t + 1``; the terminal row has no successor
        and is not scored. *skip_env_ids* are the envs that auto-reset inside
        this ``env.step`` -- their sim state is already the next episode's
        frame 0 and must not be scored against the old episode.

        *holding_env_ids* spent this step on a deferred-settle hold and applied
        no row, so they are not scored either, except on the last step of a gap
        hold (:meth:`pop_gap_holds`): the row before the skipped ticks reaches
        its recorded successor only once they have run, so its score waits
        until then.
        """
        skip = set(int(i) for i in skip_env_ids)
        holding = set(int(i) for i in holding_env_ids)
        for i in range(self._num_envs):
            data = self._episode_data[i]
            if data is None or i in skip:
                continue
            if i in holding:
                if self._gap_hold_left[i] > 0:
                    self._gap_hold_left[i] -= 1
                    deferred_row = self._gap_score_row[i]
                    if self._gap_hold_left[i] == 0 and deferred_row is not None:
                        self._gap_score_row[i] = None
                        self._score_row(i, data, deferred_row)
                continue
            # get_action already advanced the counter past the applied row.
            next_row = self._step[i]
            if next_row < 1 or next_row >= self._n_frames[i]:
                continue
            if next_row in self._gap_ticks[i]:
                self._gap_score_row[i] = next_row
                continue
            self._score_row(i, data, next_row)

    def _score_row(self, env_idx: int, data: dict, row: int) -> None:
        """Append env *env_idx*'s joint and object errors against recorded state *row* of *data*."""
        i = env_idx
        env = self._base_env
        robot = env.robot.articulation
        gt_joint_pos = np.asarray(data["observation.state.joint_position"][row], dtype=np.float32)[
            :_N_ARM
        ]
        sim_joint_pos = robot.data.joint_pos[i, :_N_ARM].cpu().numpy()
        self._joint_pos_errors[i].append(float(np.linalg.norm(sim_joint_pos - gt_joint_pos)))

        base_pos, base_quat = robot_base_pose_np(robot, env_idx=i)
        for obj_name, obj in env.scene.rigid_objects.items():
            gt_pose = _recorded_object_pose(data, obj_name, row)
            if gt_pose is None:
                continue
            sim_world = np.concatenate(
                [
                    obj.data.root_pos_w[i].cpu().numpy(),
                    obj.data.root_quat_w[i].cpu().numpy(),
                ]
            ).astype(np.float32)
            sim_pose = world_pose_to_local(sim_world, base_pos, base_quat)
            pos_err = float(np.linalg.norm(sim_pose[:3] - gt_pose[:3]))
            dot = abs(float(np.dot(sim_pose[3:7], gt_pose[3:7])))
            self._object_pos_errors[i].setdefault(obj_name, []).append(pos_err)
            self._object_quat_errors[i].setdefault(obj_name, []).append(1.0 - min(dot, 1.0))

    def episode_metrics(self, env_idx: int) -> dict[str, float]:
        """Scalar divergence metrics for the episode env *env_idx* just finished.

        Every value is a float (``profile_switch_frame`` and ``num_frames`` are
        cast) and every metric that has no data is emitted as ``NaN``, so the
        eval loop can fold them into ``eval_metrics`` as 1-element tensors and
        the NaN-safe aggregation treats them as missing.
        """
        i = int(env_idx)
        nan = float("nan")
        errs = np.asarray(self._joint_pos_errors[i], dtype=np.float64)
        switch = self._profile_switch_frame[i]
        metrics: dict[str, float] = {
            "num_frames": float(self._n_frames[i]) if self._episode_data[i] is not None else nan,
            "profile_switch_frame": float(switch) if switch is not None else nan,
        }
        if self._clip_bounds is not None:
            metrics["clipped_frames"] = float(self._clipped_frames[i])
            metrics["clip_max_rad"] = float(self._clip_max[i])
        if self._fill_gaps:
            metrics["unrecorded_ticks"] = float(self._unrecorded_ticks[i])
        metrics.update(_phase_metrics("joint_pos_error", errs))
        if switch is not None:
            pre, post = _phase_slices(switch, len(errs))
            metrics.update(_phase_metrics("pre_switch_joint_pos_error", errs[pre]))
            metrics.update(_phase_metrics("post_switch_joint_pos_error", errs[post]))
        else:
            metrics.update(_phase_metrics("pre_switch_joint_pos_error", np.empty(0)))
            metrics.update(_phase_metrics("post_switch_joint_pos_error", np.empty(0)))

        for obj_name in sorted(self._object_pos_errors[i]):
            pos_errs = np.asarray(self._object_pos_errors[i][obj_name], dtype=np.float64)
            quat_errs = np.asarray(self._object_quat_errors[i][obj_name], dtype=np.float64)
            metrics.update(_phase_metrics(f"object_{obj_name}_pos_error", pos_errs))
            metrics.update(_phase_metrics(f"object_{obj_name}_quat_error", quat_errs))
            if switch is not None:
                pre, post = _phase_slices(switch, len(pos_errs))
                metrics.update(
                    _phase_metrics(f"pre_switch_object_{obj_name}_pos_error", pos_errs[pre])
                )
                metrics.update(
                    _phase_metrics(f"post_switch_object_{obj_name}_pos_error", pos_errs[post])
                )
                metrics.update(
                    _phase_metrics(f"pre_switch_object_{obj_name}_quat_error", quat_errs[pre])
                )
                metrics.update(
                    _phase_metrics(f"post_switch_object_{obj_name}_quat_error", quat_errs[post])
                )
        return metrics
