"""Core evaluation loop and top-level ``evaluate()`` entry point."""

from __future__ import annotations

import itertools
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import tqdm

from toyblocks.eval.pd_mode_controller import (
    DATASET_SWITCH_SETTLE_TICKS,
    DeferredSettle,
    PDModeController,
    load_pd_gains_from_dataset,
    read_pd_gains_from_robot,
    settle_hold_action,
)
from toyblocks.eval.render_replay import WaveGate
from toyblocks.eval.replay import ReplayPolicy
from toyblocks.serving.depth_monitoring import (
    build_depth_stats_payload,
    log_depth_stats_summary,
)
from toyblocks.serving.obs_transforms import find_obs_transform, find_obs_transform_policy
from toyblocks.serving.obs_transforms.maniflow import ManiflowObsTransform
from toyblocks.utils.common_utils import per_env_hwc_uint8, to_numpy
from toyblocks.utils.dataset_recording_utils import (
    _aggregate_metrics,
    _build_run_name,
    _collect_per_episode_metrics,
    _connect_policy_server,
    _make_env,
    _make_policy_client,
    _print_results,
    _reconcile_obs_mode_with_server,
    _save_results,
    hl_latency_video_enabled,
    hl_overlay_enabled,
    hl_overlay_video_key,
    server_reads_prev_front_frame,
    validate_hl_latency_video_args,
)

if TYPE_CHECKING:
    from toyblocks.eval.args import Args


_ORACLE_SUCCESS_BACKSTOP_STEPS = 10
# --episode_end ll_done + oracle HL: max steps held after the LL's -1 on the final command.
_LL_DONE_SETTLE_STEPS = 25
# --go_home: skills it follows, joint speed (rad/s), arrival tolerance (rad) / step cap.
_GO_HOME_SKILLS = ("place", "flip", "rotate")
_GO_HOME_JOINT_SPEED = 0.5
_GO_HOME_TOL = 0.02
_GO_HOME_SETTLE_STEPS = 25
# --post_place_lift: end-effector rise speed (m/s; 0.035 m in ~11 steps at 25 Hz, as
# vecgen's lift_traj), damped-least-squares lambda, and steps to wait at the top.
_LIFT_SPEED = 0.08
_LIFT_DLS_LAMBDA = 0.01
_LIFT_SETTLE_STEPS = 10



#: HL oracle: fallback front-camera size for pixel -> heatmap mapping (used only if the
#: live camera resolution can't be read) and the per-episode cap on re-plans from the
#: live state when the world drifts from the plan.
ORACLE_CAM_W = 640
ORACLE_CAM_H = 360
ORACLE_MAX_REPLANS = 3
#: A "place on table" sub-goal counts only with the block within this many front-camera
#: pixels of the HL's table point (goal success is the PDDL predicates alone).
PLACE_ON_TABLE_PX_TOL = 25.0

def _log_step0_depth_stats(adapter: Any) -> None:
    """Client-side depth stats on the model-input grid (control step 0, ManiFlow depth evals)."""
    wrapper = find_obs_transform_policy(adapter.policy)
    if wrapper is None or wrapper.last_transformed_obs is None:
        return
    transform = wrapper.transform
    if not isinstance(transform, ManiflowObsTransform) or not transform.depth_keys:
        return
    payload = build_depth_stats_payload(
        wrapper.last_transformed_obs,
        depth_keys=transform.depth_keys,
        per_camera_ptmap_bounds=(
            transform._per_camera_ptmap_bounds_for_masking() if transform.depth_keys else None
        ),
    )
    log_depth_stats_summary(payload)


def _hl_latency_enabled(args: Args, needs_prev_front_frame: bool) -> bool:
    """Whether the client-side HL-latency buffer is active for this run.

    Active when any latency knob is set, the side-by-side latency video is
    requested, or the served checkpoint reads ``prev_front_rgb`` -- the buffer is
    the only thing that caches that frame, so it stays alive even at zero latency
    (where it hands back the current frame). Otherwise the loop keeps the exact
    fresh-every-step behavior.
    """
    return bool(
        args.hl_update_interval_ms > 0
        or args.estimated_hl_inference_time_ms > 0
        or args.save_hl_latency_video
        or needs_prev_front_frame
    )


def _aux_hl_heatmaps(x, num_envs: int) -> np.ndarray:
    """(num_envs, h, w) per-env HL heatmap batch from a policy-server aux array.

    The unbatched client stacks the per-env responses, so the aux already
    carries a leading env axis; the batched client (single-env only, since the
    HL-LL server asserts B==1) hands back the server's bare 2D map.
    """
    a = np.asarray(x)
    if a.ndim == 2:
        if num_envs != 1:
            raise ValueError(
                f"HL aux heatmap is a single {a.shape} map but num_envs={num_envs}; "
                "the per-env batch is missing (run the client unbatched)."
            )
        return a[None, ...]
    if a.ndim == 3 and a.shape[0] == num_envs:
        return a
    raise ValueError(f"unexpected HL aux heatmap shape {a.shape} for num_envs={num_envs}")


def _hl_heatmaps_2d(x) -> np.ndarray | None:
    """(B, H, W) CPU heatmaps from an env-batched (B,1,3,H,W) / (B,H,W) HL heatmap."""
    if x is None:
        return None
    h, w = x.shape[-2], x.shape[-1]
    return to_numpy(x.reshape(x.shape[0], -1, h, w)[:, 0])


def _aux_hl_texts(x, num_envs: int) -> list[str]:
    """Length-``num_envs`` list of per-env HL strings from a policy-server aux array."""
    flat = np.asarray(x).reshape(-1)
    if flat.size != num_envs:
        raise ValueError(
            f"expected {num_envs} HL aux texts, got shape {np.asarray(x).shape}"
        )
    return [str(v) for v in flat]


# An HL command: (verb, block it moves, place target, flip/rotate direction, oracle
# plan index). ``None`` place target = on the table; the plan index is ``None`` for
# Molmo, so two identical Molmo commands in a row are one sub-goal.
_HLCommand = tuple[str, str | None, str | None, str | None, int | None]


class _CollisionMonitor:
    """``--collision``: flags envs where a block that is not a target of the
    current HL step moved from its reference pose (its pose at episode start or
    when it last stopped being a target).

    Given HL commands (oracle / Molmo), it also scores each sub-goal (pick, place,
    flip, rotate): one starts whenever the command changes and succeeds when its
    completion check passed and no block was newly knocked while it was current.
    """

    _POS_TOL = 0.02  # m
    _ROT_TOL = math.radians(15.0)
    _HL_PX_TOL = 25.0  # --hl_from_model: HL point to projected block center
    _PARKED_BELOW_Z = -0.5  # m, env-local: under the table = parked (unused)
    # The oracle's grasp test (OracleHighLevelPolicy._is_object_grasped).
    _GRASP_LIFT_DELTA = 0.02  # m above the block's z when the pick started
    _GRASP_EE_DIST = 0.10  # m block to TCP
    _TABLE = "workspace_table"

    def __init__(
        self,
        base_env,
        num_envs: int,
        domain_file: str | None,
        count_collisions: bool = True,
        place_px_tol: float = 0.0,
    ) -> None:
        import re

        import toyblocks
        from toyblocks.utils.episode_dataset import SCENE_FURNITURE_PREFIXES

        self._env = base_env
        # --collision off: a collision does not fail a sub-goal.
        self._count_collisions = count_collisions
        self._names: list[str] = [
            n for n in base_env.scene.rigid_objects if not n.startswith(SCENE_FURNITURE_PREFIXES)
        ]
        self._ref_pos: torch.Tensor | None = None
        self._ref_quat: torch.Tensor | None = None
        self._fresh = torch.ones(num_envs, dtype=torch.bool, device=base_env.device)
        self.collided = torch.zeros(num_envs, dtype=torch.bool, device=base_env.device)
        self._targets: list[set[str]] = [set() for _ in range(num_envs)]
        # --hl_from_model: the last pick's block and the HL text last resolved.
        self._carried: list[str | None] = [None] * num_envs
        self._hl_text: list[str | None] = [None] * num_envs
        # The HL's latest "place on table" point (Molmo's, or the oracle's target);
        # the placed block must land within place_px_tol of it (0 = no check).
        self._place_px: list[np.ndarray | None] = [None] * num_envs
        self._place_px_tol = place_px_tol
        # Per episode: every block flagged as moved.
        self._moved: list[set[str]] = [set() for _ in range(num_envs)]
        # Sub-goals: the open one's command, flip/rotate goal face, pick start z,
        # completion latch and len(_moved) at its start; closed ones per episode.
        self._has_commands = False
        self._cmd: list[_HLCommand | None] = [None] * num_envs
        self._ori_goal: list[str | None] = [None] * num_envs
        self._pick_z: list[float] = [0.0] * num_envs
        self._complete: list[bool] = [False] * num_envs
        self._moved_at: list[int] = [0] * num_envs
        self._sub_goals: list[list[str]] = [[] for _ in range(num_envs)]
        self._sub_goal_ok: list[list[bool]] = [[] for _ in range(num_envs)]
        # The completion check alone, ignoring collisions (--collision penalize).
        self._sub_goal_done: list[list[bool]] = [[] for _ in range(num_envs)]
        # (direction, face) -> face after it, from the domain's flip_next / yaw_next.
        text = Path(
            domain_file or Path(toyblocks.__file__).resolve().parent / "tamp" / "blocks" / "domain.pddl"
        ).read_text()
        self._next_face: dict[tuple[str, str], str] = {
            (d, o): o2
            for d, o, o2 in re.findall(r"\((?:flip_next|yaw_next)\s+(\w+)\s+(\w+)\s+(\w+)\s*\)", text)
        }
        self._dirs: set[str] = {d for d, _ in self._next_face}

    def reset(self, mask: torch.Tensor) -> None:
        mask = mask.to(self.collided.device)
        self._fresh |= mask
        self.collided &= ~mask
        for i in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            self._targets[i] = set()
            self._carried[i] = None
            self._place_px[i] = None
            # _hl_text is kept: the finished episode's HL text stays in the
            # adapter's aux until the new episode's first query, and is not new.
            self._moved[i] = set()
            self._cmd[i] = None
            self._sub_goals[i] = []
            self._sub_goal_ok[i] = []
            self._sub_goal_done[i] = []

    def collision_counts(self) -> torch.Tensor:
        """Per env: the number of distinct blocks knocked this episode."""
        return torch.tensor(
            [float(len(m)) for m in self._moved], device=self.collided.device
        )

    def finish_sub_goals(
        self, mask: torch.Tensor
    ) -> dict[int, tuple[list[str], list[bool], list[bool]]]:
        """Close the open sub-goal of each env in ``mask``; return its episode's
        ``(sub-goal names, successes, successes ignoring collisions)``. Empty
        when no HL commands are tracked."""
        if not self._has_commands:
            return {}
        out: dict[int, tuple[list[str], list[bool], list[bool]]] = {}
        for i in mask.nonzero(as_tuple=False).squeeze(-1).tolist():
            self._close_sub_goal(i)
            out[i] = (self._sub_goals[i], self._sub_goal_ok[i], self._sub_goal_done[i])
        return out

    def _poses(self) -> tuple[torch.Tensor, torch.Tensor]:
        objs = [self._env.scene.rigid_objects[n].data for n in self._names]
        return (
            torch.stack([o.root_pos_w for o in objs], dim=1),
            torch.stack([o.root_quat_w for o in objs], dim=1),
        )

    def hl_commands(self, hl_texts: list[str]) -> list[_HLCommand | None]:
        """Commands from Molmo HL text: the block within ``_HL_PX_TOL`` of the
        point (pick / flip / rotate; for place, the block placed on), with the
        carried block for place."""
        from toyblocks.serving.maniflow_serving_utils import _parse_target_xy, _strip_coords
        from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import front_camera_pixels

        pos, _ = self._poses()
        uv = torch.stack([front_camera_pixels(self._env, pos[:, b]) for b in range(len(self._names))], dim=1)

        def pointed(i: int, xy: np.ndarray) -> str | None:
            d = torch.linalg.norm(uv[i] - torch.as_tensor(xy, dtype=uv.dtype, device=uv.device), dim=-1)
            b = int(torch.argmin(d))
            return self._names[b] if float(d[b]) <= self._HL_PX_TOL else None

        out: list[_HLCommand | None] = []
        for i, text in enumerate(hl_texts):
            lang, xy = _strip_coords(text).lower(), _parse_target_xy(text)
            # Resolve on a new HL output only: a lifted block leaves its pick point.
            if text == self._hl_text[i] or lang == "done":
                out.append(None)
                continue
            self._hl_text[i] = text
            words = lang.split()
            if lang.startswith("place"):
                # The block placed on is not a target: it must stay put.
                on = pointed(i, xy) if lang == "place" and xy is not None else None
                if on is not None:
                    on = self._stack_top(i, on)
                if lang == "place" and on is None:
                    out.append(None)  # points at no block: not a feasible command, no sub-goal
                    continue
                self._place_px[i] = xy if on is None else None
                out.append(("place", self._carried[i], on, None, None))
            elif xy is not None and words:
                block = pointed(i, xy)
                if block is None:
                    out.append(None)  # points at no block: not a feasible command, no sub-goal
                    continue
                if words[0] == "pick":
                    self._carried[i] = block
                out.append((words[0], block, None, words[-1] if len(words) > 1 else None, None))
            else:
                out.append(None)
        return out

    def _stack_top(self, i: int, block: str) -> str:
        """The top of the stack ``block`` is in: a place on a covered block lands on top."""
        from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import goal_on

        for _ in self._names:
            above = next(
                (b for b in self._names if b not in (block, self._carried[i]) and bool(goal_on(self._env, (b, block))[i])),
                None,
            )
            if above is None:
                return block
            block = above
        return block

    def set_oracle_place_px(
        self, target_xy: np.ndarray, commands: list[_HLCommand | None]
    ) -> None:
        """Oracle runs: a table placement's point is the oracle's current target pixel."""
        for i, c in enumerate(commands):
            if c is not None and c[0] == "place" and c[2] is None:
                xy = np.asarray(target_xy[i], dtype=np.float32)
                self._place_px[i] = xy if bool(np.isfinite(xy).all()) else None

    def _close_sub_goal(self, i: int) -> None:
        cmd = self._cmd[i]
        if cmd is None or cmd[0] not in ("pick", "place", "flip", "rotate"):
            return
        verb, block, on, direction, _ = cmd
        name = " ".join(
            str(p) for p in (verb, block, f"on {on or self._TABLE}" if verb == "place" else direction) if p
        )
        self._sub_goals[i].append(name)
        self._sub_goal_ok[i].append(
            self._complete[i]
            and (not self._count_collisions or len(self._moved[i]) == self._moved_at[i])
        )
        self._sub_goal_done[i].append(self._complete[i])
        self._cmd[i] = None

    def _open_sub_goal(self, i: int, cmd: _HLCommand) -> None:
        from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import (
            ORI_QUATERNIONS,
            goal_ori_of,
            resolve_root_pose_w,
        )

        self._close_sub_goal(i)
        self._cmd[i] = cmd
        self._complete[i] = False
        self._moved_at[i] = len(self._moved[i])
        verb, block, _, direction, _ = cmd
        self._ori_goal[i] = None
        if block is None:
            return
        self._pick_z[i] = float(resolve_root_pose_w(self._env, block)[i, 2])
        if verb in ("flip", "rotate") and direction:
            d = direction if direction in self._dirs else f"{direction}_flip"
            face = next(
                (f for f in ORI_QUATERNIONS if bool(goal_ori_of(self._env, (block, f))[i])), None
            )
            self._ori_goal[i] = self._next_face.get((d, face)) if face else None

    def _sub_goal_reached(self, i: int) -> bool:
        from toyblocks.eval.oracle.sim_adapter import _EE_TCP_SENSOR
        from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import (
            _GRIPPER_JOINT_NAME,
            _GRIPPER_OPEN_BELOW,
            front_camera_pixels,
            goal_on,
            goal_on_table,
            goal_ori_of,
            resolve_root_pose_w,
            scene_settled_and_gripper_open,
        )

        verb, block, on, _, _ = self._cmd[i]
        if block is None:
            return False
        env = self._env
        if verb == "pick":
            art = env.robot.articulation
            ids, _ = art.find_joints([_GRIPPER_JOINT_NAME])
            closed = float(art.data.joint_pos[i, ids[0]]) >= _GRIPPER_OPEN_BELOW
            p = resolve_root_pose_w(env, block)[i, :3]
            tcp = env.scene.sensors[_EE_TCP_SENSOR].data.target_pos_w[i, 0]
            return (
                closed
                and float(p[2]) > self._pick_z[i] + self._GRASP_LIFT_DELTA
                and float(torch.linalg.norm(p - tcp)) < self._GRASP_EE_DIST
            )
        if verb == "place":
            placed = goal_on(env, (block, on)) if on else goal_on_table(env, (block, self._TABLE))
            if not bool((placed & scene_settled_and_gripper_open(env, [block]))[i]):
                return False
            if on is not None or self._place_px[i] is None or self._place_px_tol <= 0:
                return True
            # A table placement counts where the HL pointed.
            uv = front_camera_pixels(env, resolve_root_pose_w(env, block)[:, :3])[i]
            px = torch.as_tensor(self._place_px[i], dtype=uv.dtype, device=uv.device)
            return float(torch.linalg.norm(uv - px)) <= self._place_px_tol
        if self._ori_goal[i] is None:
            return False
        # Released and settled, as for place: mid-flip the held block can pass
        # through the goal face.
        turned = goal_ori_of(env, (block, self._ori_goal[i]))
        return bool((turned & scene_settled_and_gripper_open(env, [block]))[i])

    def update(
        self,
        targets: list[set[str] | None],
        commands: list[_HLCommand | None] | None = None,
    ) -> list[tuple[int, list[str]]]:
        """Check this step's poses (and sub-goals, given ``commands``); return
        ``(env, moved blocks)`` for envs newly collided. A command's block is its
        env's target."""
        if commands is not None:
            targets = [None if c is None else ({c[1]} if c[1] else set()) for c in commands]
        for i, t in enumerate(targets):
            if t is not None:
                self._targets[i] = t
        pos, quat = self._poses()
        if self._ref_pos is None:
            self._ref_pos, self._ref_quat = pos.clone(), quat.clone()
        is_target = torch.tensor(
            [[n in t for n in self._names] for t in self._targets],
            dtype=torch.bool, device=pos.device,
        )
        # Targets and fresh episodes take the current pose as their reference.
        follow = (is_target | self._fresh[:, None]).unsqueeze(-1)
        self._ref_pos = torch.where(follow, pos, self._ref_pos)
        self._ref_quat = torch.where(follow, quat, self._ref_quat)
        self._fresh[:] = False
        d_pos = torch.linalg.norm(pos - self._ref_pos, dim=-1)
        d_rot = 2 * torch.acos((quat * self._ref_quat).sum(-1).abs().clamp(max=1.0))
        # Blocks an episode does not use are parked in the graveyard (z = -5 m);
        # PhysX pushes them up to rest under the floor (z ~ -1 m), so a block
        # under the table now or at its reference pose is parked.
        origin_z = self._env.scene.env_origins[:, 2:3]
        parked = (pos[..., 2] - origin_z < self._PARKED_BELOW_Z) | (
            self._ref_pos[..., 2] - origin_z < self._PARKED_BELOW_Z
        )
        moved = ((d_pos > self._POS_TOL) | (d_rot > self._ROT_TOL)) & ~follow[..., 0] & ~parked
        moved_names = {
            i: [self._names[b] for b in moved[i].nonzero(as_tuple=False).squeeze(-1).tolist()]
            for i in moved.any(dim=1).nonzero(as_tuple=False).squeeze(-1).tolist()
        }
        for i, names in moved_names.items():
            self._moved[i].update(names)
        if commands is not None:
            self._has_commands = True
            for i, c in enumerate(commands):
                if c is not None and c != self._cmd[i]:
                    self._open_sub_goal(i, c)
        if self._has_commands:
            for i, c in enumerate(self._cmd):
                if c is not None and not self._complete[i]:
                    self._complete[i] = self._sub_goal_reached(i)
        new = moved.any(dim=1) & ~self.collided
        self.collided |= new
        return [(i, moved_names[i]) for i in new.nonzero(as_tuple=False).squeeze(-1).tolist()]


def _episode_seed(base_seed: int, episode_index: int) -> int:
    """Deterministic per-episode seed so each episode is reproducible
    regardless of which other episodes ran before it."""
    return (base_seed + episode_index * 1000003) % (2**31 - 1)


def _find_pd_mode_setter(adapter):
    """Find ``set_pd_mode`` on the ManiFlow obs transform inside ``adapter``.

    The client stack is either ``ObsTransformPolicy`` directly on the adapter,
    or wrapped in an ``ActionChunkBroker`` / ``PerEnvActionChunkBroker``. Walk
    ``._policy`` looking for a transform that exposes ``set_pd_mode``. Returns
    the bound method, or ``None`` when no such transform is in the chain
    (e.g. non-maniflow obs_transform, or the ``--random_actions`` / zero-action
    paths that have no adapter at all).

    The presence of this setter is also what gates pre-applying the stiff PD
    profile at reset: transforms without a pd_mode column mean the policy
    doesn't drive the PD controller, so overwriting the default robot gains
    would be wrong.
    """
    if adapter is None:
        return None
    node = getattr(adapter, "policy", None)
    for _ in range(8):  # bounded walk; the real chain is at most 2 deep
        if node is None:
            return None
        transform = getattr(node, "_transform", None)
        if transform is not None and hasattr(transform, "set_pd_mode"):
            return transform.set_pd_mode
        node = getattr(node, "_policy", None)
    return None


def _initial_pd_modes_from_setups(
    setups: list | None,
    num_envs: int,
    device: torch.device,
    default_mode: int = 0,
) -> torch.Tensor:
    """Per-env initial PD-mode tensor read from queue setups.

    Reads ``setup["initial_pd_mode"]`` for each env (recorded first-frame
    ``action.pd_mode`` from the source dataset; see
    :func:`toyblocks.utils.episode_dataset._row_to_setup`). Envs with no
    setup or no key fall back to ``default_mode``, so runs without a queue
    (``queue_client is None``) keep the pre-existing hardcoded behavior.

    The returned tensor is passed straight to
    :meth:`PDModeController.initialize_after_reset` so different envs can
    start under different impedances when the dataset recorded different
    initial modes.
    """
    out = torch.full((num_envs,), int(default_mode), dtype=torch.long, device=device)
    if setups is None:
        return out
    for i in range(min(num_envs, len(setups))):
        setup = setups[i]
        if setup is None:
            continue
        v = setup.get("initial_pd_mode")
        if v is None:
            continue
        out[i] = int(v)
    return out


def _fmt_vec(t: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(x):.6f}" for x in t.detach().cpu().reshape(-1).tolist()) + "]"


def _log_episode_start_robot(
    *,
    env_idx: int,
    step: int,
    episode_idx: int,
    base_env,
    action: torch.Tensor,
    pd_controller: PDModeController,
    setup: dict | None,
) -> None:
    """Print the live robot + first action for one env's episode-start step."""
    robot = base_env.robot.articulation
    i = env_idx
    act = action[i] if action.ndim > 1 else action
    agent = base_env._get_obs_robot()
    pd_mode = int(pd_controller.current_pd_mode()[i].item())
    ep_id = setup.get("episode_id") if setup is not None else None
    lines = [
        f"[INFO]: episode-start dump env={i} global_step={step} "
        f"episode_idx={episode_idx} episode_id={ep_id!r} "
        f"control_mode={base_env.robot.control_mode!r} pd_mode={pd_mode}",
        f"[INFO]:   joint_names={list(robot.joint_names)}",
        f"[INFO]:   joint_pos={_fmt_vec(agent['joint_pos'][i])}",
        f"[INFO]:   joint_vel={_fmt_vec(agent['joint_vel'][i])}",
        f"[INFO]:   joint_pos_target={_fmt_vec(agent['joint_targets'][i])}",
        f"[INFO]:   joint_vel_target={_fmt_vec(robot.data.joint_vel_target[i])}",
        f"[INFO]:   ee_pos={_fmt_vec(agent['ee_pos'][i])}",
        f"[INFO]:   stiffness={_fmt_vec(robot.data.joint_stiffness[i])}",
        f"[INFO]:   damping={_fmt_vec(robot.data.joint_damping[i])}",
        f"[INFO]:   effort_limit={_fmt_vec(robot.data.joint_effort_limits[i])}",
        f"[INFO]:   action={_fmt_vec(act)}",
    ]
    if act.numel() >= 9:
        lines.append(
            f"[INFO]:   action_arm={_fmt_vec(act[:7])} gripper={float(act[7]):.6f} "
            f"action_pd_mode={float(act[8]):.6f}"
        )
    elif act.numel() >= 8:
        lines.append(
            f"[INFO]:   action_arm={_fmt_vec(act[:7])} gripper={float(act[7]):.6f}"
        )
    lines.append(f"[INFO]:   applied_torque={_fmt_vec(robot.data.applied_torque[i])}")
    if setup is not None and setup.get("robot_joint_positions") is not None:
        expected = torch.tensor(
            setup["robot_joint_positions"],
            device=agent["joint_pos"].device,
            dtype=agent["joint_pos"].dtype,
        )
        live_arm = agent["joint_pos"][i, : expected.numel()]
        lines.append(
            f"[INFO]:   setup_joint_pos={_fmt_vec(expected)} "
            f"max_abs_diff={float((live_arm - expected).abs().max()):.6f}"
        )
    if setup is not None and setup.get("robot_gripper_position") is not None:
        lines.append(
            f"[INFO]:   setup_gripper={float(setup['robot_gripper_position']):.6f}"
        )
    print("\n".join(lines), flush=True)


def _refresh_recorded_renders(
    env: Any, base_env: Any, env_ids: list[int], *, warmup_frames: int, obs: Any
) -> Any:
    """Apply the staged render-replay records of ``env_ids`` and re-read the observations.

    The scene-global lighting comes from the lowest of ``env_ids`` with a record. The
    recorder's s_0 for these envs is re-sampled so row 0 shows the applied look. ``obs``
    is returned unchanged when it is not the camera obs dict (state obs is flattened).
    """
    base_env.render_replay.apply_staged(env_ids, apply_globals=True)
    if base_env.sim.has_rtx_sensors():
        for _ in range(max(1, int(warmup_frames))):
            base_env.sim.render()
    base_env._update_sensor_buffers(env_ids)
    if isinstance(obs, dict):
        obs = base_env._get_observations()
    recorder = getattr(env, "data_recorder", None)
    if hasattr(recorder, "capture_reset_frames"):
        recorder.capture_reset_frames(env_ids)
    return obs


def _reseed_eval(seed: int, adapter=None) -> None:
    """Re-seed all eval-side RNGs and optionally tell the policy server."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if adapter is not None and hasattr(adapter, "set_episode_seed"):
        adapter.set_episode_seed(seed)


def evaluate(args: Args) -> None:
    """Top-level evaluation entry point.

    Builds the env and runs :func:`_run_eval`, which serves every action source
    (policy server / LL oracle / random / zero / ``--replay`` dataset replay).
    """
    run_name = _build_run_name(args)
    output_dir = args.output_dir or f"runs/{run_name}"
    device = torch.device(args.device)

    from toyblocks.utils.wandb_logging import setup_wandb

    wandb_run, on_video_saved, wandb_state = setup_wandb(
        args,
        run_name=run_name,
        output_dir=output_dir,
    )

    if args.random_actions and args.policy_host is not None:
        raise ValueError("--random_actions cannot be used with --policy_host.")
    if args.replay_clip_stats and not args.replay:
        raise ValueError("--replay_clip_stats only applies to --replay.")
    if args.center_crop_ratio is not None and args.obs_transform not in (
        "maniflow",
        "maniflow_blocks_table",
    ):
        raise ValueError(
            f"--center_crop_ratio only applies to the maniflow obs transforms, "
            f"not --obs_transform {args.obs_transform!r}."
        )
    if args.replay:
        if args.policy_host is not None or args.random_actions:
            raise ValueError("--replay cannot be combined with --policy_host or --random_actions.")
        if args.queue_url is None or not args.dataset_path:
            raise ValueError(
                "--replay plays back the recorded actions of each queue episode, so it "
                f"requires both --queue_url (got {args.queue_url!r}) and --dataset_path "
                f"(got {args.dataset_path!r})."
            )
        if args.num_envs != 1:
            raise ValueError(
                f"--replay requires --num_envs 1 (got {args.num_envs}). ReplayPolicy "
                "advances every env's recording on each step and cannot hold one env "
                "through a PD-switch settle. If you passed 1, reconcile_num_envs_with_task "
                "raised it to cover the task package's cube-scale slots: restrict "
                "--episode-indices to episodes of a single cube scale, or use a task "
                "whose package bakes one slot."
            )
        if args.episode_end != "horizon":
            print(
                "[INFO]: --replay forces episode_end='horizon' so every recording "
                "plays to its last frame."
            )
            args.episode_end = "horizon"
        # Match the Robotiq close target used by TAMP datagen exactly.
        if args.env_cfg.replay_gripper_close_target is None:
            args.env_cfg.replay_gripper_close_target = 0.72
        # ReplayPolicy.done_indices ends each episode at its last recorded frame;
        # the global timeout only has to be long enough never to cut one short.
        # Computed from the eval control period rather than the dataset fps so
        # the bound holds even when the two disagree.
        from toyblocks.utils.episode_dataset import max_episode_frames

        max_frames = max_episode_frames(args.dataset_path)
        step_dt = float(args.env_cfg.sim.dt) * float(args.env_cfg.decimation)
        needed_s = max_frames * step_dt * 1.1 + 1.0
        if float(args.env_cfg.episode_length_s) < needed_s:
            print(
                f"[INFO]: --replay raising episode_length_s "
                f"{float(args.env_cfg.episode_length_s):g} -> {needed_s:g} to cover the "
                f"longest recording ({max_frames} frames x {step_dt:g}s)."
            )
            args.env_cfg.episode_length_s = needed_s

    if args.enable_crop and args.obs_transform not in (
        "maniflow",
        "maniflow_blocks_table",
    ):
        raise ValueError(
            f"--enable_crop only applies to the maniflow obs transforms, "
            f"not --obs_transform {args.obs_transform!r}."
        )
    if args.use_calibrated_cameras:
        args.ingest_camera_intrinsics = False
        args.use_render_params = False
        args.env_cfg.use_calibrated_cameras = True
    if args.use_render_params and args.dataset_path:
        if args.replay and args.num_envs != 1:
            raise ValueError(
                f"--replay with --use_render_params needs --num_envs 1 (got "
                f"{args.num_envs}): the recorded sky, dome, sun and exposure are one "
                "value for the whole scene. Pass --no-use-render-params to replay "
                "with the scene's own look."
            )
    if args.wave_resets:
        if args.num_envs <= 1:
            raise ValueError("--wave_resets needs --num_envs > 1.")
        if not (args.use_render_params and args.dataset_path and args.queue_url):
            raise ValueError(
                "--wave_resets only exists to share recorded scene lighting across a "
                "wave, so it needs --use_render_params, --dataset_path and --queue_url."
            )

    if args.use_oracle:
        if args.policy_host is None:
            raise ValueError("--use_oracle requires --policy_host (the maniflow LL server).")
        if args.obs_transform not in ("maniflow", "maniflow_blocks_table"):
            raise ValueError(
                "--use_oracle requires --obs_transform maniflow or "
                f"maniflow_blocks_table; got {args.obs_transform!r}."
            )
        if args.queue_url is None and args.pddl_problem_file is None:
            raise ValueError(
                "--use_oracle needs a PDDL problem source: pass --pddl_problem_file "
                "(single problem for all envs), or --queue_url (per-env setups). "
                "With a queue, each setup's 'pddl_problem_file' is used when "
                "present, otherwise a minimal problem is reconstructed from the "
                "setup's 'pddl_goal'."
            )

    # Seeding
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
        print(f"[INFO]: Evaluation client seeded with seed={args.seed}")
    if args.deterministic_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[INFO]: CUDA deterministic mode enabled")

    # Connect before the env is built: the checkpoint's advertised obs keys
    # decide the env's obs_mode (depth), and the env's obs_mode is frozen at
    # construction. The same connection is reused for inference.
    policy_client = None
    if args.policy_host is not None:
        policy_client = _connect_policy_server(args)
        _reconcile_obs_mode_with_server(args, policy_client.metadata)

    t0_env = time.perf_counter()
    env, base_env, video_fps = _make_env(
        args,
        output_dir=output_dir,
        on_video_saved=on_video_saved,
    )

    env_setup_time = time.perf_counter() - t0_env
    print(f"[INFO]: Environment setup took {env_setup_time:.2f}s")

    if wandb_run is not None:
        import wandb

        wandb.run.summary["env_setup_seconds"] = env_setup_time

    wandb_state["video_fps"] = video_fps
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # Viewer (optional)
    if args.enable_viewer:
        from toyblocks.viewer.viewer import Viewer, ViewerCfg

        viewer = Viewer(cfg=ViewerCfg(env_ids=list(range(args.num_envs))))
        viewer.hook_to_env(env)
        viewer.track_env = True

    log_episode_tables = _run_eval(
        args,
        env,
        base_env,
        output_dir,
        device,
        policy_client=policy_client,
        wandb_state=wandb_state,
    )

    env.close()

    # Nothing in the output LeRobot dataset records which sub-env produced an episode,
    # so eval_episodes.json maps each episode to its LeRobot dataset episode index, which
    # also locates its HL overlay video for the wandb table. Only possible after close():
    # the stitched episode indices are not final until every episode is saved.
    if log_episode_tables is not None:
        data_recorder = getattr(env, "data_recorder", None)
        log_episode_tables(
            data_recorder.merged_episode_index_map()
            if data_recorder is not None and hasattr(data_recorder, "merged_episode_index_map")
            else {}
        )

    if wandb_run is not None:
        import wandb

        wandb.finish()

    if args.enable_viewer:
        print(f"Scripted Finished. Viewer is still running at {viewer.get_server_url()}")
        while True:
            time.sleep(0.1)


def _run_eval(
    args: Args,
    env,
    base_env,
    output_dir: str,
    device: torch.device,
    policy_client,
    wandb_state: dict | None = None,
) -> Callable[[dict[tuple[int, int], tuple[str, int]]], None] | None:
    """Evaluation loop (policy server / LL oracle / random / zero / dataset replay).

    ``policy_client`` is the connection :func:`evaluate` already opened to read
    the served checkpoint's metadata (``None`` for the random / zero paths); it is wrapped here rather than re-dialed.
    """
    from toyblocks.domain_randomization.block_variables import BlockIdentity, rendered_text
    from toyblocks.utils.prompt_resolver import PromptResolver, policy_goal

    prompt_resolver = PromptResolver(
        num_envs=args.num_envs, replay_block_identity=args.replay_block_identity
    )

    # Validate HL-latency flags up front (fail loud, no silent fallbacks).
    validate_hl_latency_video_args(args)
    if args.oracle_comparison and not args.hl_from_model:
        raise ValueError(
            "--oracle_comparison requires --hl_from_model (the learned HL model "
            "served by molmo2_maniflow); it renders a model | oracle side-by-side "
            "overlay panel and is meaningless without a model HL to compare."
        )
    if args.oracle_comparison and not hl_overlay_enabled(args):
        raise ValueError(
            "--oracle_comparison only renders into the HL overlay video, which the "
            "LeRobot recorder writes: pass --save_lerobot and "
            "--save_hl_overlay_video."
        )
    # An `action_goal` checkpoint's spatial goal lives in its language, so its
    # `task_name` has to come from a per-step HL (the oracle, or the server's
    # Molmo model). With neither, the only source is the episode's english
    # `pddl_goal` prompt, which carries no action and no pixels -- the run would
    # look healthy and report a success rate for a policy that was never told
    # where to go.
    if (
        policy_client is not None
        and policy_client.metadata.get("language_mode") == "action_goal"
        and not (args.use_oracle or args.hl_from_model)
    ):
        raise ValueError(
            "the served checkpoint was trained with language_mode='action_goal', "
            "i.e. on strings like 'pick 409,210', but this run has no high-level "
            "policy to produce one: task_name would be the episode's english goal "
            "prompt. Pass --use_oracle (TAMP oracle HL) or serve with --hl model "
            "and pass --hl_from_model."
        )

    # --- Oracle high-level policy (client-side, replaces the learned HL model) ---
    def _served_language_mode() -> str:
        """The served LL checkpoint's ``language_mode``, i.e. its text contract.

        The oracle builds the ``task_name`` the LL is conditioned on, so it has to
        build it in the format training used. Getting this wrong is silent: an
        ``action_goal`` checkpoint fed a bare verb still runs, but its spatial
        goal was in the language and is now gone (its ``target_heatmap`` is
        dropped unless the checkpoint sets ``target_injection_mode``).

        A server that reports the key as ``None`` is serving a checkpoint with no
        language conditioning at all (a heatmap-only LL, whose ``task_name`` the
        server drops before inference). ``action_only`` is then the definition
        rather than a guess: the string only reaches the video burn-in.
        """
        if policy_client is None:
            raise ValueError(
                "an oracle HL controller needs the served checkpoint's "
                "language_mode to format task_name, but this run has no policy "
                "server connection (--policy_host). The oracle HL path only "
                "applies to a served ManiFlow LL."
            )
        meta = policy_client.metadata
        if "language_mode" not in meta:
            raise ValueError(
                "the policy server does not report 'language_mode' at all, so the "
                "format the LL's task_name must take is unknown -- and a mismatch "
                "is silent. Restart the server from this revision (it advertises "
                f"the mode in its metadata). Reported keys: {sorted(meta)}."
            )
        return str(meta["language_mode"] or "action_only")

    def _build_oracle_hl_controller():
        """Construct an ``OracleHLController`` bound to the live ``base_env``.

        Shared by the LL-driving oracle (``--use_oracle``) and the
        comparison-only oracle (``--oracle_comparison``, model HL path).
        Returns ``(controller, domain_file)``.
        """
        from pathlib import Path as _Path
        import toyblocks as _toyblocks
        from toyblocks.eval.oracle import OracleHLController
        from toyblocks.serving.maniflow_serving_utils import DEFAULT_TARGET_SIGMA

        domain_file = str(
            _Path(_toyblocks.__file__).resolve().parent / "tamp" / "blocks" / "domain.pddl"
        )
        controller = OracleHLController(
            base_env,
            domain_file=domain_file,
            default_problem_file=args.pddl_problem_file,
            language_mode=_served_language_mode(),
            action_config_path=None,
            target_sigma=DEFAULT_TARGET_SIGMA,
            orig_cam_w=ORACLE_CAM_W,
            orig_cam_h=ORACLE_CAM_H,
            reconstruct_pddl=True,
            problem_dump_dir=None,
            replan=True,
            max_replans=ORACLE_MAX_REPLANS,
        )
        return controller, domain_file

    oracle_controller = None
    if args.use_oracle:
        oracle_controller, domain_file = _build_oracle_hl_controller()
        print(f"[INFO]: Oracle HL policy enabled (domain={domain_file}).")

    # Comparison-only oracle for the learned HL model path: ticked every step for
    # the side-by-side overlay video (model | oracle) but never fed to the LL, so
    # ``hl_controller`` stays None and the LL keeps acting on the server HL.
    comparison_oracle = None
    if args.oracle_comparison:
        comparison_oracle, domain_file = _build_oracle_hl_controller()
        print(f"[INFO]: Oracle comparison enabled (domain={domain_file}).")

    # The client-side controller drives the injected task_name + target_heatmap.
    # (The other HL option is a learned model served via
    # serve_maniflow_policy --hl model, which produces conditioning server-side
    # and needs no client controller.)
    hl_controller = oracle_controller

    # --- Action source ---
    adapter = None
    replay_policy: ReplayPolicy | None = None
    # Keeps the HL-latency buffer alive for a checkpoint trained with
    # use_previous_front_frame even when no latency is configured, since that
    # buffer is what caches the frame the LL reads as prev_front_rgb.
    needs_prev_front_frame = server_reads_prev_front_frame(policy_client)
    if args.replay:
        # Recorded 9-dim actions (7 joints + gripper + pd_mode) per queue
        # episode; the pd_mode channel goes through the same PD controller
        # path a policy's prediction would (see the has_pd_mode branch below).
        replay_policy = ReplayPolicy(
            base_env,
            device,
            args.num_envs,
            use_delta_actions=False,
            clip_stats_path=args.replay_clip_stats,
        )
        print(f"[INFO]: Replaying recorded actions from dataset: {args.dataset_path}")
    elif args.policy_host is not None:
        adapter = _make_policy_client(
            args, prompt_resolver=prompt_resolver, policy=policy_client
        )
        print(
            f"Connected to policy server at {args.policy_host}:{args.policy_port}  "
            f"(action_key={args.action_key!r})"
        )
    elif args.random_actions:
        print("[INFO]: Using uniform random actions in [-1, 1].")
    else:
        print(
            "[INFO]: No --policy_host, --replay or --random_actions; sending zero actions."
        )

    # The one handle on the live obs transform, resolved from the client stack
    # here so the pd_mode setter, the policy-input panels and the staleness
    # recorder all read the same negotiated transform (None on the
    # checkpoint / random / zero paths, which have no adapter).
    obs_transform = find_obs_transform(adapter)
    # Gated on the transform emitting a pd_mode column: transforms without one and
    # ``--random_actions`` / zero actions don't use the PD mode
    # controller at all, so we must not stomp on their default gains.
    pd_mode_setter = getattr(obs_transform, "set_pd_mode", None)

    # --- What the served checkpoint means by action / state dim 8 ---
    # pd_mode (the actuator gain profile) or, for a checkpoint trained with
    # hvla's pause_on_attempt, the sim2real play mode: 1 play, 0 pause, -1 skill
    # finished. Under the play mode the gains never switch, the state column
    # carries the last executed mode, and a -1 makes that env ignore the rest of
    # its chunk (see the action handling in the loop).
    server_meta: dict = dict(getattr(policy_client, "metadata", None) or {})
    play_mode = adapter is not None and server_meta.get("action_dim8") == "play_mode"
    if play_mode:
        if base_env.robot.control_mode != "arm_joint_gripper_binary":
            raise ValueError(
                "a play-mode checkpoint stops an env on -1 by holding it with absolute "
                "joint-position targets, which needs control_mode "
                f"'arm_joint_gripper_binary'; got {base_env.robot.control_mode!r}."
            )
        print("[INFO]: action dim 8 is the play mode (1 play, 0 pause, -1 skill done).")
    # --episode_end ll_done: the play-mode LL owns termination.
    ll_ends_episode = args.episode_end == "ll_done"
    if ll_ends_episode:
        if not play_mode:
            raise ValueError(
                "--episode_end ll_done needs a play-mode LL checkpoint (hvla "
                "pause_on_attempt, served with action_dim8='play_mode'); the server "
                f"advertises action_dim8={server_meta.get('action_dim8')!r}. Use "
                "--episode_end goal or horizon for this checkpoint."
            )
        if args.hl_from_model:
            raise ValueError(
                "--episode_end ll_done is not supported with --hl_from_model: there "
                "play mode -1 ends the current skill and the HL's 'done' ends the episode."
            )
        print("[INFO]: --episode_end ll_done: play mode -1 ends the episode.")
    # Goal success / oracle plan completion end the episode only under 'goal'.
    end_on_goal = args.episode_end == "goal"
    # Built under --collision off too: it also scores the sub-goals.
    collision_monitor = _CollisionMonitor(
        base_env,
        args.num_envs,
        None,
        args.collision != "off",
        PLACE_ON_TABLE_PX_TOL,
    )
    play_state = torch.ones(args.num_envs, device=device)
    # Envs whose LL ended its skill (-1) in the current chunk: they hold the pose
    # the skill ended in until their next chunk, which comes on the usual schedule.
    skill_done = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    skill_done_hold: torch.Tensor | None = None
    play_broker = None
    if play_mode:
        from toyblocks.serving.action_chunk_broker import find_action_chunk_broker

        play_broker = find_action_chunk_broker(adapter)
    # A Molmo2 HL server (--hl model) takes each env's goal in the phrasing its
    # checkpoint was trained on (goal.hl_prompt, else e.g. "Stack A B C").
    hl_goal_contract = None
    if adapter is not None and server_meta.get("hl_native_frames"):
        from toyblocks.serving.molmo2_hl import HLContract

        hl_goal_contract = HLContract(
            prompt_mode=server_meta.get("hl_prompt_mode"),
            html_v2_variant=server_meta.get("hl_html_v2_variant"),
        )

    # --- Exact LL policy-input trace (one env, one episode) ---
    # Rendered *inside the policy server*, right before predict_action, so the
    # panels are the tensors the model receives rather than what the client sent
    # (see toyblocks.serving.policy_input_trace). Only the maniflow servers
    # consume the request key, so it is not sent to a pi server that would carry
    # it into its input transform as an unknown obs.
    ll_trace_dir: Path | None = None
    hl_trace_dir: Path | None = None
    ll_trace_env = int(args.video_env_ids[0])
    if (
        args.save_diagnostic_plots
        and adapter is not None
        and args.obs_transform in ("maniflow", "maniflow_blocks_table")
    ):
        ll_trace_dir = Path(output_dir) / "ll_policy_input_trace"
        hl_trace_dir = (
            Path(output_dir) / "hl_policy_input_trace" if args.hl_from_model else None
        )
        adapter.set_policy_input_trace(
            str(ll_trace_dir), ll_trace_env,
            hl_dir=str(hl_trace_dir) if hl_trace_dir is not None else None,
        )
        print(
            f"[INFO]: LL policy-input trace requested for env {ll_trace_env}'s first "
            f"episode; the policy server writes panels to {ll_trace_dir}"
        )
        if hl_trace_dir is not None:
            print(
                f"[INFO]: HL policy-input trace requested for the same env; "
                f"the policy server writes Molmo2 panels to {hl_trace_dir}"
            )

    def get_action(obs):
        with torch.no_grad():
            if replay_policy is not None:
                return replay_policy.get_action(obs)
            if adapter is not None:
                return adapter.act(obs)
            if args.random_actions:
                return 2 * torch.rand(base_env.action_space.shape, device=device) - 1
            return torch.zeros(base_env.action_space.shape, device=device)

    # --- PD mode controller ---
    pd_gains_kwargs: dict = {}
    if args.dataset_path:
        stiff_gains, compliant_gains = load_pd_gains_from_dataset(args.dataset_path)
        pd_gains_kwargs["stiff_gains"] = stiff_gains
        pd_gains_kwargs["compliant_gains"] = compliant_gains
        print(f"[INFO]: PD gains loaded from dataset: {args.dataset_path}")
        print(
            f"[INFO]:   stiff:     stiffness={stiff_gains['stiffness']} "
            f"damping={stiff_gains['damping']}"
        )
        print(
            f"[INFO]:   compliant: stiffness={compliant_gains['stiffness']} "
            f"damping={compliant_gains['damping']}"
        )
    else:
        robot_gains = read_pd_gains_from_robot(base_env)
        pd_gains_kwargs["stiff_gains"] = robot_gains
        pd_gains_kwargs["compliant_gains"] = robot_gains
        print("[INFO]: No --dataset_path; PD gains from robot config.")
        print(
            f"[INFO]:   both modes: stiffness={robot_gains['stiffness']} "
            f"damping={robot_gains['damping']}"
        )
    # A ground-truth dataset also fixes the settle behavior on a mode switch:
    # the TAMP datagen held the arm for a profile-dependent number of control
    # ticks after every switch (see DATASET_SWITCH_SETTLE_TICKS), so evals
    # driven by a dataset -- replay and policy evals alike -- reproduce that.
    # Without a dataset the switch stays settle-free.
    # The settle runs as per-env hold steps (DeferredSettle), hidden from the
    # policy, the recording and the episode clock, for every env count: raw sim
    # substeps would advance every env in the scene, and one mechanism keeps
    # single- and multi-env runs comparable.
    use_deferred_settle = bool(args.dataset_path)
    pd_controller = PDModeController(
        base_env,
        device,
        num_envs=args.num_envs,
        initial_mode=1,
        switch_settle_ticks=DATASET_SWITCH_SETTLE_TICKS if args.dataset_path else None,
        defer_settle=use_deferred_settle,
        **pd_gains_kwargs,
    )
    deferred_settle: DeferredSettle | None = None
    settle_broker = None
    skip_recording_next_step = None
    if use_deferred_settle:
        if base_env.robot.control_mode != "arm_joint_gripper_binary":
            raise ValueError(
                "the deferred PD-switch settle (enabled by --dataset_path) holds the "
                "arm with absolute joint-position targets, which needs control_mode "
                f"'arm_joint_gripper_binary'; got {base_env.robot.control_mode!r}."
            )
        from toyblocks.serving.action_chunk_broker import find_action_chunk_broker

        deferred_settle = DeferredSettle(args.num_envs, device)
        settle_broker = find_action_chunk_broker(adapter) if adapter is not None else None
        skip_recording_next_step = getattr(env, "skip_recording_next_step", None)
        print(
            "[INFO]: PD mode switches settle for "
            f"{DATASET_SWITCH_SETTLE_TICKS[0]}/{DATASET_SWITCH_SETTLE_TICKS[1]} "
            "(compliant/stiff) control ticks with the arm frozen (dataset-given default), "
            "as per-env hold steps (deferred settle)."
        )

    # --- Shared chunk boundary (--sync_action_chunks) ---
    # An env whose episode ends mid-chunk has no chunk until the next boundary
    # (the broker lists it in waiting_envs); until then it holds the pose it was
    # reset to, as hold steps hidden like the settle's, so each episode starts
    # at a boundary and the server is queried once per horizon.
    sync_broker = None
    park_hold_action: torch.Tensor | None = None
    if adapter is not None and args.sync_action_chunks and not args.unbatched:
        from toyblocks.serving.action_chunk_broker import (
            ActionChunkBroker,
            find_action_chunk_broker,
        )

        broker = find_action_chunk_broker(adapter)
        if isinstance(broker, ActionChunkBroker):
            if base_env.robot.control_mode != "arm_joint_gripper_binary":
                raise ValueError(
                    "--sync_action_chunks holds an env reset mid-chunk with absolute "
                    "joint-position targets until the next chunk boundary, which needs "
                    "control_mode 'arm_joint_gripper_binary'; got "
                    f"{base_env.robot.control_mode!r}. Drop --sync_action_chunks."
                )
            sync_broker = broker
            # Rows are refreshed for each env as it resets; this is only the shape.
            park_hold_action = settle_hold_action(base_env)
            if skip_recording_next_step is None:
                skip_recording_next_step = getattr(env, "skip_recording_next_step", None)
            print(
                f"[INFO]: Shared action-chunk boundary every {args.action_horizon} steps; "
                "envs reset mid-chunk hold their reset pose until it."
            )

    # --- Wave resets (--wave_resets) ---
    # A finished env is reset as usual, then parked like a chunk-boundary waiter
    # until every env with a live setup has finished. The wave then gets its
    # recorded renders and starts together, so envs that share the scene-global
    # lighting (one sky/dome/sun/exposure for the whole stage) start together.
    wave_gate: WaveGate | None = None
    if args.wave_resets:
        if getattr(base_env, "render_replay", None) is None:
            print("[WARN]: --wave_resets ignored: the dataset has no render_params to replay.")
        else:
            if base_env.robot.control_mode != "arm_joint_gripper_binary":
                raise ValueError(
                    "--wave_resets holds parked envs with absolute joint-position targets, "
                    "which needs control_mode 'arm_joint_gripper_binary'; got "
                    f"{base_env.robot.control_mode!r}."
                )
            wave_gate = WaveGate(args.num_envs)
            if park_hold_action is None:
                park_hold_action = settle_hold_action(base_env)
            if skip_recording_next_step is None:
                skip_recording_next_step = getattr(env, "skip_recording_next_step", None)
            print(
                "[INFO]: --wave_resets: finished envs hold their reset pose until every "
                "active env has finished; each wave starts with its recorded renders."
            )

    # A batched Molmo2 HL server runs the HL for every env of a request and feeds
    # each env's output back as its next "Previous action". With per-env chunk
    # phases most requests serve one env and the others discard the result, so
    # their HL history would advance on outputs they never acted on; a shared
    # boundary makes every env consume every request.
    if args.async_hl and sync_broker is None:
        raise ValueError("--async_hl needs the batched client with --sync_action_chunks.")
    if args.async_hl and not (hl_goal_contract is not None and play_mode):
        raise ValueError(
            "--async_hl refreshes the Molmo2 HL after the LL ends a skill (play mode "
            "-1), so it needs the --hl model server and a play-mode LL checkpoint "
            f"(server metadata: hl_native_frames={server_meta.get('hl_native_frames')!r}, "
            f"action_dim8={server_meta.get('action_dim8')!r})."
        )
    if args.go_home and not (args.async_hl or args.ll_oracle):
        raise ValueError("--go_home needs --async_hl or --ll_oracle: it delays the HL refresh until the arm is home.")
    if args.ll_oracle and not (
        args.hl_from_model
        and args.action_horizon == 0
        and base_env.robot.control_mode == "arm_joint_gripper_binary"
    ):
        raise ValueError(
            "--ll_oracle runs Molmo2 commands with the vecgen low-level oracle, so it needs the "
            "Molmo2 HL server (--hl_from_model), --action_horizon 0 (each query reaches the "
            "server) and --control_mode arm_joint_gripper_binary (absolute joint targets)."
        )
    # Envs whose LL finished a skill since their last server request (--async_hl).
    hl_refresh = np.zeros(args.num_envs, dtype=bool)
    # --go_home: envs driving their (skill_done) hold target to the home pose, and
    # steps each has spent with that target at home; -1 while still moving.
    going_home = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    home_settle = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)
    home_arm: torch.Tensor | None = None
    if args.go_home:
        arm_ids = base_env.robot.controller["arm"]._joint_ids
        home_arm = base_env.robot.articulation.data.default_joint_pos[:, arm_ids].clone().to(device)
        home_max_step = _GO_HOME_JOINT_SPEED * float(base_env.step_dt)
    # --post_place_lift: envs raising their (skill_done) hold target after a release,
    # the rise each still has to command (m), and steps spent at the top (-1 = rising).
    lifting = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
    lift_left = torch.zeros(args.num_envs, dtype=torch.float32, device=device)
    lift_settle = torch.full((args.num_envs,), -1, dtype=torch.long, device=device)
    if args.post_place_lift > 0:
        if args.go_home:
            raise ValueError("--post_place_lift and --go_home both move the held arm; pick one.")
        if not play_mode:
            raise ValueError("--post_place_lift needs a play-mode LL checkpoint (its -1 ends a place).")
        lift_art = base_env.robot.articulation
        lift_arm_ids = base_env.robot.controller["arm"]._joint_ids
        lift_body = lift_art.find_bodies(base_env.robot.cfg.ee_body_name)[0][0]
        # PhysX Jacobians (world frame) skip the root body and, on a floating
        # base, put the 6 root DoFs first (as Isaac Lab's differential IK).
        lift_jac_body = lift_body - 1 if lift_art.is_fixed_base else lift_body
        lift_jac_joints = (
            list(lift_arm_ids) if lift_art.is_fixed_base else [j + 6 for j in lift_arm_ids]
        )
        lift_max_step = _LIFT_SPEED * float(base_env.step_dt)
    # --ll_oracle sends hl_refresh, so a query advances only the envs it is for.
    if hl_goal_contract is not None and not args.unbatched and sync_broker is None and not args.ll_oracle:
        raise ValueError(
            "the Molmo2 HL + LL server (--hl model) needs --sync_action_chunks when the "
            "client is batched, so every env consumes every HL call (see above)."
        )

    # --- Throughput / episode tracking ---
    from toyblocks.utils.wandb_logging import EpisodeResultsTable, EvalThroughputTracker

    tracker = (
        EvalThroughputTracker(
            num_envs=args.num_envs,
            log_every_n_seconds=args.wandb_log_interval_seconds,
            log_to_wandb=args.wandb,
        )
        if args.wandb or args.queue_url is not None
        else None
    )

    episode_table = EpisodeResultsTable(
        output_dir=output_dir,
        save_sensor_obs_video=args.save_sensor_obs_video,
        save_expert_video=args.save_expert_video or args.wandb,
        save_hl_latency_video=hl_latency_video_enabled(args),
        hl_overlay_video_key=hl_overlay_video_key(args),
    )
    per_env_episode_count: dict[int, int] = defaultdict(int)
    # Whether the HL model emitted "done" during the in-progress episode
    # (--hl_from_model only); reported as the ``hl_done`` episode metric.
    hl_done_flags: list[bool] = [False] * args.num_envs
    # Oracle termination is reported as two exclusive booleans so plan completion
    # is distinguishable from the sustained-success backstop.
    oracle_done_flags: list[bool] = [False] * args.num_envs
    oracle_success_backstop_flags: list[bool] = [False] * args.num_envs
    # Recovering oracle hit the place->pick cap. Distinct from plan-complete
    # and from a timeout so a stuck comparison overlay cannot look like an
    # ordinary failure (and must not abort evaluate.py).
    oracle_stuck_flags: list[bool] = [False] * args.num_envs
    # Play-mode LL (sim2real): skills it ended (-1) and steps it ran paused (0),
    # per episode, reported as ll_skill_stops / ll_pause_steps.
    ll_skill_stops: list[int] = [0] * args.num_envs
    ll_pause_steps: list[int] = [0] * args.num_envs
    # --episode_end ll_done + oracle HL: steps each env has held since the LL's -1
    # on the final command (-1 = not holding). Held envs take no policy actions.
    ll_final_settle: list[int] = [-1] * args.num_envs
    # Whether the oracle was on its final command when each env's current LL skill
    # began (None = not yet sampled; set at episode start and after each -1).
    ll_skill_started_final: list[bool | None] = [None] * args.num_envs

    # --- Queue setup ---
    from toyblocks.utils.eval_helpers import setup_queue

    from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import set_goal_ori_of_yaw_tol_deg

    set_goal_ori_of_yaw_tol_deg(base_env, args.goal_ori_yaw_tol_deg)
    # H, I and O read the same upside down: their ori_of goals also pass at 180 degrees.
    from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import set_symmetric_glyph_ori_of

    set_symmetric_glyph_ori_of(base_env)

    # Goal success is the PDDL predicates alone: on_table names no spot. Where a
    # table placement lands is judged per sub-goal, at the HL's own point.
    queue_client = setup_queue(args, env, base_env, prompt_resolver)

    def _hl_goals() -> list[str]:
        """Per-env HL goal prompts for the current queue setups ("" for none)."""
        setups = list(queue_client.current_setups) if queue_client is not None else []
        setups += [None] * (args.num_envs - len(setups))
        return [
            hl_goal_contract.goal_prompt(  # type: ignore[union-attr]
                s.get("hl_prompt"), policy_goal(s, args.replay_block_identity)
            )
            if s is not None
            else ""
            for s in setups
        ]

    # --- Per-episode expert-scaled timeout (--expert_timeout_scale) ---
    # Precomputed control steps per recorded expert frame, so the per-env budget
    # below is one multiply: expert frames -> seconds (dataset fps) -> control
    # steps (env step_dt). Zero when the feature is off.
    expert_timeout_steps_per_frame = 0.0
    if args.expert_timeout_scale > 0:
        if queue_client is None or not args.dataset_path:
            raise ValueError(
                "--expert_timeout_scale scales each episode's own expert "
                "demonstration, which is only known from the queue setup's recorded "
                "expert_episode_length and the dataset's fps, so it requires both "
                f"--queue_url (got {args.queue_url!r}) and --dataset_path (got "
                f"{args.dataset_path!r})."
            )
        from toyblocks.utils.episode_dataset import dataset_fps

        expert_fps = dataset_fps(args.dataset_path)
        expert_timeout_steps_per_frame = args.expert_timeout_scale / (
            expert_fps * float(base_env.step_dt)
        )
        print(
            f"[INFO]: per-episode timeout = {args.expert_timeout_scale:g}x the expert "
            f"duration ({expert_fps:g} fps expert frames -> {base_env.step_dt:g}s control "
            f"steps, so {expert_timeout_steps_per_frame:.4g} steps per expert frame). "
            f"The global timeout of {base_env.max_episode_length} steps still caps it."
        )

    # --- TAMPER failure capture ---
    # Attached before the first step so the pre-auto-reset hook is in place for
    # every episode boundary; the terminal state only exists inside that hook.
    tamper_collector = None
    if args.tamper_output_dir:
        from toyblocks.eval.tamper import TamperCollector

        tamper_collector = TamperCollector(
            base_env,
            args.tamper_output_dir,
            source_env_path=args.tamper_source_env_path,
            capture_successes=args.tamper_capture_successes,
            max_failures=args.tamper_max_states,
        )
        tamper_collector.attach()

    # TODO: We will need to try to move this in the offline plotting of diagnostics
    # --- First-frame capture for eval-vs-expert comparison ---
    def _capture_eval_first_frames(obs_dict, env_idx: int) -> dict[str, np.ndarray] | None:
        if not isinstance(obs_dict, dict) or "images" not in obs_dict:
            return None
        from toyblocks.utils.dataset_recording_utils import _extract_frame

        frames: dict[str, np.ndarray] = {}
        for cam_name, cam_data in obs_dict["images"].items():
            env_obs_slice = {"images": {cam_name: {dk: dv[env_idx] for dk, dv in cam_data.items()}}}
            f = _extract_frame(env_obs_slice, cam_name)
            if f is not None:
                frames[cam_name] = f
        return frames or None

    eval_first_frames: dict[int, dict[str, np.ndarray] | None] = {}

    # --- Policy-input verification panels (gated diagnostic) ---
    # Built from the live obs transform (not a fresh one) so the panels show the
    # arrays inference actually sends; see toyblocks.eval.policy_input_debug.
    # The panels are one artifact of the umbrella --save_diagnostic_plots, so a
    # run that cannot produce them says so and keeps going rather than aborting
    # the artifacts it can produce.
    policy_input_debug = None
    if args.save_diagnostic_plots:
        from toyblocks.eval.policy_input_debug import PolicyInputDebugRecorder
        from toyblocks.serving.obs_transforms.maniflow import ManiflowObsTransform

        server_state_norm = policy_client.metadata.get("state_norm") if policy_client else None
        if adapter is None:
            print(
                "[INFO]: policy-input panels skipped: they describe what a policy "
                "server receives, and this run has no --policy_host (the "
                "random / zero paths have no obs transform)."
            )
        elif not isinstance(obs_transform, ManiflowObsTransform):
            print(
                "[INFO]: policy-input panels skipped: they render ManiFlow policy "
                "inputs (model-input depth, agent_pos bands) and "
                f"--obs_transform={args.obs_transform!r} is not a maniflow transform."
            )
        elif not server_state_norm:
            print(
                "[INFO]: policy-input panels skipped: the server advertises no "
                "'state_norm' (serve with a build of serve_maniflow_policy.py that "
                "publishes it); the state panel has no band to compare against."
            )
        else:
            policy_input_debug = PolicyInputDebugRecorder(
                output_dir=output_dir,
                transform=obs_transform,
                server_meta=policy_client.metadata if policy_client is not None else {},
            )
    # {env_idx: {wandb column: png path}} for the episode in progress.
    policy_input_debug_paths: dict[int, dict[str, str]] = {}

    # --ll_oracle: the vecgen skills on this env (it measures the spawned block colliders
    # and builds its cuRobo planner pool here, before the initial reset).
    ll_oracle = None
    if args.ll_oracle:
        from toyblocks.eval.ll_oracle import LLOracle

        ll_oracle = LLOracle(
            env, args.num_envs, _GO_HOME_JOINT_SPEED * float(base_env.step_dt), _GO_HOME_TOL, args.go_home
        )
        hl_refresh[:] = True
    # Per episode: Molmo2 calls (--async_hl starts at the server's first call), else new skill + XY HL outputs.
    hl_calls = np.full(args.num_envs, int(args.async_hl))
    hl_last_text: list[str | None] = [None] * args.num_envs

    # --- Initial reset ---
    obs, _ = env.reset()

    if queue_client is not None and queue_client.done:
        print("[INFO] Queue is empty; nothing to evaluate.")
        return

    # Per-episode re-seeding for the initial episode.
    if args.seed is not None:
        ep_idx = 0
        if queue_client is not None:
            setup = queue_client.current_setups[0]
            if setup is not None:
                ep_idx = setup.get("episode_index", 0)
        ep_seed = _episode_seed(args.seed, ep_idx)
        _reseed_eval(ep_seed, adapter)
        if args.debug:
            print(f"[INFO]: Per-episode seed for initial episode (ep_idx={ep_idx}): {ep_seed}")

    # --- Ingest recorded camera intrinsics (after sensors initialize) ---
    # Each env gets that episode's first-frame K. Randomized-camera renders
    # vary K across episodes, so a later reset re-applies via apply_setup.
    if args.ingest_camera_intrinsics and args.dataset_path:
        from toyblocks.utils.dataset_provenance import ingest_dataset_camera_intrinsics
        ingest_dataset_camera_intrinsics(
            base_env,
            args.dataset_path,
            setups=(
                list(queue_client.current_setups) if queue_client is not None else None
            ),
        )

    # --- Recorded renders for the first batch (--use_render_params) ---
    # With one env apply_setup already applied everything inside env.reset(). With
    # several, the scene-global lighting is set here, once (or per wave), from the
    # lowest env, and wave mode's per-env records were only staged.
    if getattr(base_env, "render_replay", None) is not None and args.num_envs > 1:
        obs = _refresh_recorded_renders(
            env,
            base_env,
            list(range(args.num_envs)),
            warmup_frames=args.render_replay_warmup_frames,
            obs=obs,
        )

    # --- Scene verification (gated diagnostic) ---
    if args.verify_scene or args.verify_scene_strict:
        if queue_client is None:
            raise ValueError("--verify_scene requires --queue_url (per-env setups).")
        from toyblocks.eval.scene_verification import verify_setups_against_sim
        verify_setups_against_sim(
            base_env,
            obs,
            list(queue_client.current_setups),
            output_dir,
            compute_first_frames="rgb" in args.obs_mode,
            strict=args.verify_scene_strict,
        )

    # Pre-apply the training-time PD profile before the first policy step so
    # the initial dynamics match data collection. Isaac's ``env.reset()`` leaves
    # the robot in whatever its actuator cfg specifies (e.g. the softer
    # ``fr3_robotiq_gripper_softer`` gains); the target mode is read per env
    # from ``setup["initial_pd_mode"]`` (recorded first-frame ``action.pd_mode``)
    # so the very first policy step runs under the exact impedance each
    # episode was recorded under. Without this the pd_mode column in the
    # observation also lies at step 0. Also sync the transform's pd_mode
    # column to the value actually written so ``agent_pos`` stops lying.
    #
    # Before the panel capture below, so its pd_mode row shows the gains that
    # were actually written rather than the transform's default.
    #
    # Replay has no transform but drives the controller from the recorded
    # pd_mode channel, so it starts every episode under the recorded
    # first-frame mode too (no settle, like the initial write datagen made).
    pd_init_enabled = pd_mode_setter is not None or replay_policy is not None
    if pd_init_enabled:
        target_modes = _initial_pd_modes_from_setups(
            list(queue_client.current_setups) if queue_client is not None else None,
            args.num_envs,
            device,
        )
        pd_controller.initialize_after_reset(target_mode=target_modes)
        if pd_mode_setter is not None:
            pd_mode_setter(pd_controller.current_pd_mode())

    # Load the recorded episodes the first reset placed in each env.
    if replay_policy is not None:
        replay_policy.on_reset(range(args.num_envs), list(queue_client.current_setups))

    # Rendered here rather than in the reset loop above so the panels describe
    # the obs that inference will see: --ingest_camera_intrinsics rewrites the
    # intrinsics the pointmap uses.
    if policy_input_debug is not None:
        policy_input_debug_paths = policy_input_debug.capture(
            obs,
            list(range(args.num_envs)),
            {i: 0 for i in range(args.num_envs)},
            list(queue_client.current_setups) if queue_client is not None else [None] * args.num_envs,
        )

    # Build the per-env HL controller(s) from the just-reset state / first setups.
    if hl_controller is not None or comparison_oracle is not None:
        init_setups = (
            list(queue_client.current_setups)
            if queue_client is not None
            else [None] * args.num_envs
        )
        if hl_controller is not None:
            hl_controller.on_reset(range(args.num_envs), init_setups)
        if comparison_oracle is not None:
            comparison_oracle.on_reset(range(args.num_envs), init_setups)

    # --- HL-latency buffer (client-side oracle) ---
    # Decouples the oracle HL (invoked every step to preserve its state machine)
    # from what the LL is conditioned on: commit the fresh payload every ~K ms,
    # then feed the LL the entry that is >= M (estimated HL inference time) ms
    # old. See toyblocks.serving.hl_latency.HLLatencyBuffer.
    hl_latency_buffer = None
    if hl_controller is not None and _hl_latency_enabled(args, needs_prev_front_frame):
        from toyblocks.serving.hl_latency import HLLatencyBuffer

        hl_latency_buffer = HLLatencyBuffer(
            update_interval_ms=args.hl_update_interval_ms,
            estimated_inference_time_ms=args.estimated_hl_inference_time_ms,
            num_envs=args.num_envs,
            update_interval_jitter_ms=args.hl_update_interval_jitter_ms,
            seed=args.seed,
        )
        print(
            f"[INFO]: HL latency enabled (K={args.hl_update_interval_ms}ms "
            f"jitter={args.hl_update_interval_jitter_ms}ms "
            f"M={args.estimated_hl_inference_time_ms}ms "
            f"prev_front_rgb={needs_prev_front_frame})."
        )
        # A commit cadence that is an integer multiple of the action-chunk
        # period pins the selected entry to the same phase at every inference,
        # so the age the LL is conditioned on is a constant rather than the
        # sawtooth over [M, M + K) the checkpoint was trained on. Only the
        # chunk boundaries matter, so this is invisible in the per-step
        # staleness plot.
        chunk_period_ms = args.action_horizon * float(base_env.step_dt) * 1000.0
        if needs_prev_front_frame and args.hl_update_interval_ms > 0 and chunk_period_ms > 0:
            chunks_per_update = args.hl_update_interval_ms / chunk_period_ms
            if (
                not args.hl_update_interval_jitter_ms
                and abs(chunks_per_update - round(chunks_per_update)) < 1e-6
            ):
                raise ValueError(
                    f"--hl_update_interval_ms={args.hl_update_interval_ms} is an "
                    f"integer multiple of the action-chunk period "
                    f"({args.action_horizon} steps * {float(base_env.step_dt) * 1000.0:g}ms "
                    f"= {chunk_period_ms:g}ms), so the HL commits phase-lock with the "
                    "inferences and the checkpoint's 'prev_front_rgb' is aged by the "
                    "same constant every chunk instead of sweeping [M, M + K). Offset "
                    "--hl_update_interval_ms from the chunk period, or pass "
                    "--hl-update-interval-jitter-ms to break the lock."
                )

    # --- HL conditioning staleness (same gate as the policy-input panels) ---
    # Both broker handles are bound here rather than inside the branches below:
    # each branch binds only one of them, and neither is reached when the
    # diagnostics are off or the latency buffer is not live, while
    # ``_at_inference_now`` resolves both names on every call.
    staleness = None
    staleness_broker = None
    staleness_per_env_broker = None
    if args.save_diagnostic_plots:
        if hl_latency_buffer is None:
            latency_requested = bool(
                args.hl_update_interval_ms or args.estimated_hl_inference_time_ms
            )
            if latency_requested and args.hl_from_model:
                # Learned HL: the K/M gating runs in the policy server's own
                # HLLatencyBuffer, so the requested staleness *is* happening --
                # the client just has no buffer to record it from. Skip the plot
                # rather than reject the run, like the other capability-gated
                # diagnostics. The HL policy-input trace covers the same
                # question from the server side, one panel per Molmo2 forward.
                print(
                    "[INFO]: skipping the HL staleness plot: with --hl_from_model "
                    f"the K={args.hl_update_interval_ms}ms / "
                    f"M={args.estimated_hl_inference_time_ms}ms gating lives in the "
                    "policy server, which the client cannot instrument. See "
                    "hl_policy_input_trace/ for the per-forward HL inputs."
                )
            elif latency_requested:
                raise ValueError(
                    "--save_diagnostic_plots was given with "
                    f"--hl_update_interval_ms={args.hl_update_interval_ms} / "
                    f"--estimated_hl_inference_time_ms="
                    f"{args.estimated_hl_inference_time_ms}, but no HL-latency "
                    "buffer is live: the client-side buffer needs --use_oracle "
                    "and the server-side one needs --hl_from_model. The requested "
                    "staleness is silently not happening and there is nothing to "
                    "plot."
                )
        else:
            from toyblocks.eval.hl_staleness import HLStalenessRecorder
            from toyblocks.serving.action_chunk_broker import (
                PerEnvActionChunkBroker,
                find_action_chunk_broker,
            )

            staleness = HLStalenessRecorder(
                output_dir,
                env_idx=int(args.video_env_ids[0]),
                camera=args.hl_overlay_camera,
                transform=obs_transform,
                update_interval_ms=args.hl_update_interval_ms,
                estimated_inference_time_ms=args.estimated_hl_inference_time_ms,
                chunk_period_ms=args.action_horizon * float(base_env.step_dt) * 1000.0,
            )
            # The LL-call marks come off the broker rather than
            # ``step % action_horizon``: adapter.reset() fires whenever *any* env
            # finishes an episode and re-phases the shared broker's boundaries,
            # which at --num-envs > 1 happens well inside the followed env's
            # first episode.
            broker = find_action_chunk_broker(adapter)
            if isinstance(broker, PerEnvActionChunkBroker):
                staleness_per_env_broker = broker
            else:
                staleness_broker = broker

    def _at_inference_now() -> bool:
        """Whether this step's obs reaches the policy server for the followed env.

        Queried before ``get_action``, so it means exactly "this step will hit the
        server". With ``--action_horizon 0`` no broker is built and every step is
        an inference.
        """
        if staleness_per_env_broker is not None:
            return staleness_per_env_broker.needs_inference_for_env(int(args.video_env_ids[0]))
        if staleness_broker is not None:
            return staleness_broker.needs_inference
        return True

    # Pre-apply the training-time PD profile before the first policy step so
    # the initial dynamics match data collection. Isaac's ``env.reset()`` leaves
    # the robot in whatever its actuator cfg specifies (e.g. the softer
    # ``fr3_robotiq_gripper_softer`` gains); the target mode is read per env
    # from ``setup["initial_pd_mode"]`` (recorded first-frame ``action.pd_mode``)
    # so the very first policy step runs under the exact impedance each
    # episode was recorded under. Without this the pd_mode column in the
    # observation also lies at step 0. Also sync the transform's pd_mode
    # column to the value actually written so ``agent_pos`` stops lying.
    #
    # Gated on the transform emitting a pd_mode column: transforms without one and
    # ``--random_actions`` / zero actions don't use the PD mode
    # controller at all, so we must not stomp on their default gains here.
    # Replay drives the controller itself (see pd_init_enabled above).
    pd_mode_setter = _find_pd_mode_setter(adapter)

    def sync_state_dim8() -> None:
        """Write the state's dim-8 column: the committed PD mode, or the play mode."""
        if pd_mode_setter is not None:
            pd_mode_setter(play_state if play_mode else pd_controller.current_pd_mode())

    if pd_init_enabled:
        target_modes = _initial_pd_modes_from_setups(
            list(queue_client.current_setups) if queue_client is not None else None,
            args.num_envs,
            device,
        )
        pd_controller.initialize_after_reset(target_mode=target_modes)
        sync_state_dim8()

    eval_metrics: dict[str, list[torch.Tensor]] = defaultdict(list)
    # Per finished episode, in eval_metrics order: PDDL text of each goal_conj_<c>.
    per_episode_goal_conjuncts: list[list[str] | None] = []
    # Generated PDDL envs carry every compiled goal's conjunct labels and the goal
    # each env is on, so labels need no queue setup. Packages generated before
    # the labels existed lack _GOAL_CONJUNCT_LABELS and keep None labels.
    goal_labels_by_idx: tuple | None = (
        getattr(base_env, "_GOAL_CONJUNCT_LABELS", None)
        if hasattr(base_env, "_per_env_goal_list")
        else None
    )
    # Per compiled goal: the blocks behind the goal_obj_<k>_* stats, in slot order.
    goal_objects_by_idx: tuple | None = (
        getattr(base_env, "_GOAL_OBJECTS", None)
        if hasattr(base_env, "_per_env_goal_list")
        else None
    )
    goal_idx_before_step: list[int] = []
    step_dt_s = float(base_env.step_dt)
    num_episodes = 0

    start_time = time.perf_counter()
    step_iter = itertools.count() if queue_client is not None else range(args.num_eval_steps)
    # mininterval throttles refreshes: when stderr is a log file (not a TTY)
    # tqdm writes a fresh line per refresh, so a large interval keeps the run
    # log from being flooded with per-step progress lines.
    pbar = tqdm.tqdm(step_iter, desc="Evaluating", file=sys.stderr, mininterval=10.0)

    np.set_printoptions(suppress=True, precision=6)

    # The pd_mode is the one recorded field the recorder cannot read off the
    # scene: it is stripped from the action below, before the step it drives.
    data_recorder = getattr(env, "data_recorder", None)
    stage_pd_modes = getattr(data_recorder, "set_pending_pd_modes_batched", None)
    # The learned HL's per-step output, recorded beside the action it produced.
    stage_hl_outputs = getattr(data_recorder, "set_hl_model_outputs_batched", None)
    # The HL drawn on the overlay video stream, staged like the HL output.
    stage_hl_overlay = (
        getattr(data_recorder, "set_hl_overlay_batched", None)
        if hl_overlay_enabled(args)
        else None
    )
    # Handed back to a step on which every env is mid-settle and the policy is
    # not queried; the settle overrides each row of it.
    last_policy_action: torch.Tensor | None = None

    for step in pbar:
        oracle_done_ids: list[int] = []
        # Envs mid-settle neither advance nor re-plan their action chunk.
        settle_busy: torch.Tensor | None = None
        if deferred_settle is not None:
            settle_busy = deferred_settle.busy
            if settle_broker is not None:
                settle_broker.hold_envs(
                    settle_busy.nonzero(as_tuple=False).squeeze(-1).tolist()
                )
        # Stamp HL timing on obs for the server-side buffer / episode reset
        # (the --hl model path, where hl_controller is None). Only when a policy
        # server is in use — the random/zero paths never read
        # it and injecting extra obs keys there is unnecessary. Uses ndarrays so
        # to_numpy(obs) in PolicyClientAdapter handles them (bare scalars raise).
        sim_time_ms = step * float(base_env.step_dt) * 1000.0
        if adapter is not None:
            # Per-env array (not 0-d): the unbatched client's _split_dict indexes
            # every obs leaf per env (np_val[i]), which would IndexError on a 0-d
            # scalar. The server reads reshape(-1)[0], so this is safe batched too.
            obs["hl_sim_time_ms"] = np.full(args.num_envs, sim_time_ms, dtype=np.float32)
            obs["hl_episode_id"] = np.asarray(
                [per_env_episode_count[i] for i in range(args.num_envs)], dtype=np.int64
            )
            obs["hl_env_index"] = np.arange(args.num_envs, dtype=np.int64)
            # Rebuilt every step from the live setups: the env's reset fills them
            # in, so anything cached before the first step would be empty.
            if hl_goal_contract is not None:
                obs["hl_goal"] = _hl_goals()
            if args.async_hl or args.ll_oracle:
                obs["hl_refresh"] = hl_refresh.copy()

        # Tick the HL controller every control step and inject task_name +
        # target_heatmap into obs. ManiflowObsTransform forwards these to the LL
        # server. (The oracle advances grasp/predicate state per call; the
        # data-HL controller is static per episode.)
        if hl_controller is not None:
            task_names, heatmaps = hl_controller.compute()
            display_texts = hl_controller.display_texts
            if oracle_controller is not None:
                oracle_done_ids = oracle_controller.done_indices()
                for i in oracle_controller.stuck_indices():
                    if not oracle_stuck_flags[i]:
                        print(f"[INFO]: env {i}: oracle plan stuck -> ending episode")
                    oracle_stuck_flags[i] = True
            if hl_latency_buffer is not None:
                # Buffer the *fresh* oracle output (payload carries the plan-done
                # flag) and feed the LL the *latent* (M-aged) entry instead.
                done_set = (
                    set(oracle_done_ids)
                    if oracle_controller is not None
                    else set()
                )
                # The frame the HL was invoked on, cached alongside its output.
                # It has two independent consumers -- the latency video's
                # LL-view panel and the ``prev_front_rgb`` of a checkpoint
                # trained with a previous front frame -- and only the second is
                # unknown here (it is negotiated from the policy server's
                # advertised inputs), so cache it whenever the buffer is live.
                # Missing is fatal rather than skipped: the LL would then be
                # rejected by the server for a missing 'prev_front_rgb' with no
                # hint that the overlay camera is the cause.
                # The copy costs one device->host transfer per HL update, which
                # is small next to the per-step heatmap batch this loop already
                # builds.
                imgs = obs.get("images")
                if imgs is None or args.hl_overlay_camera not in imgs:
                    available = sorted(imgs) if imgs is not None else []
                    raise ValueError(
                        f"hl_overlay_camera={args.hl_overlay_camera!r} is not in the "
                        f"observation images (available: {available}), so the "
                        "HL-latency buffer has no frame to age. Pass "
                        "--hl-overlay-camera to name a camera the env renders (and "
                        "an rgb --obs_mode)."
                    )
                cond_rgb = imgs[args.hl_overlay_camera]["rgb"]

                latent_task_names: list[str] = []
                latent_heatmaps: list[np.ndarray] = []
                latent_texts: list[str] = []
                latent_frames: list = []
                # Captured in the loop for the staleness plot's commit series:
                # commit() overwrites the buffer's _last_update, so re-asking
                # should_update() afterwards answers False on exactly the steps
                # the series marks.
                followed_env_committed = False
                for i in range(args.num_envs):
                    if hl_latency_buffer.should_update(i, sim_time_ms):
                        if i == int(args.video_env_ids[0]):
                            followed_env_committed = True
                        frame = per_env_hwc_uint8(cond_rgb, i)
                        hl_latency_buffer.commit(
                            i,
                            sim_time_ms,
                            {
                                "task_name": task_names[i],
                                "heatmap": heatmaps[i],  # (1, 3, 224, 224)
                                "done": i in done_set,
                                "text": display_texts[i]
                                if i < len(display_texts)
                                else task_names[i],
                            },
                            frame,
                        )
                    payload, frame = hl_latency_buffer.select(i, sim_time_ms)
                    latent_task_names.append(payload["task_name"])
                    latent_heatmaps.append(payload["heatmap"])
                    latent_texts.append(payload["text"])
                    latent_frames.append(frame)

                latent_batch = np.stack(latent_heatmaps, axis=0).astype(np.float32)
                obs["target_heatmap"] = torch.from_numpy(latent_batch).to(device)
                obs["oracle_task_name"] = latent_task_names
                # hl_overlay_text is the LL-view (latent) text; the fresh text
                # goes under a separate viz key below.
                obs["hl_overlay_text"] = latent_texts

                if any(f is not None for f in latent_frames):
                    H, W = None, None
                    for f in latent_frames:
                        if f is not None:
                            H, W = f.shape[:2]
                            break
                    obs["hl_conditioning_image"] = np.stack(
                        [
                            f if f is not None else np.zeros((H, W, 3), np.uint8)
                            for f in latent_frames
                        ],
                        axis=0,
                    )

                if staleness is not None:
                    # Recorded before get_action, so the inference mark means
                    # "this step's obs reaches the server" rather than "a chunk
                    # boundary was crossed".
                    staleness.record(
                        obs,
                        sim_time_ms,
                        hl_latency_buffer,
                        at_inference=_at_inference_now(),
                        at_hl_update=followed_env_committed,
                    )

                if args.save_hl_latency_video:
                    obs["hl_fresh_heatmap"] = heatmaps  # (B, 1, 3, 224, 224) numpy
                    obs["hl_fresh_text"] = list(display_texts)

                # The latent done flag remains a video-overlay property. Episode
                # termination below follows the fresh oracle plan pointer so HL
                # latency cannot postpone completion after the oracle is done.
            else:
                obs["target_heatmap"] = torch.from_numpy(heatmaps).to(device)
                obs["oracle_task_name"] = task_names
                # Raw HL string for the video burn-in, separate from the
                # language_mode-formatted task_name the LL policy consumes.
                obs["hl_overlay_text"] = display_texts
                # The oracle keeps repeating its last instruction for this final
                # control step; the fresh plan-complete signal terminates below.

        # On step 0, tell the server to dump the model-ready batch.
        if args.save_first_batch_data and step == 0 and adapter is not None:
            first_batch_dir = str(Path(output_dir) / "first_batch_data")
            adapter._pending_first_batch_dir = first_batch_dir

        t_policy = time.perf_counter()
        parked_ids: list[int] = []
        if (
            settle_busy is not None
            and last_policy_action is not None
            and bool(settle_busy.all())
        ):
            action = last_policy_action
        else:
            # A shared-boundary request carries every env, so it consumes every flag.
            reaches_server = sync_broker is not None and sync_broker.needs_inference
            # An env that ended its skill resumes with the next chunk it is handed;
            # one settling after its final command holds until its episode ends.
            if bool(skill_done.any()):
                for i in skill_done.nonzero(as_tuple=False).squeeze(-1).tolist():
                    if ll_final_settle[i] >= 0 or bool(going_home[i]) or bool(lifting[i]):
                        continue
                    if play_broker is None or play_broker.needs_inference_for_env(i):
                        skill_done[i] = False
                        ll_skill_started_final[i] = None
            if ll_oracle is not None:
                if hl_refresh.any():
                    # Molmo2 only, for the flagged envs: the server skips the LL.
                    obs["hl_only"] = np.ones(args.num_envs, dtype=bool)
                    get_action(obs)
                    del obs["hl_only"]
                    # Aux leaves are (T, B, ...) chunks, normally sliced per step by
                    # the chunk broker; --action_horizon 0 has none, so take step 0.
                    adapter.last_aux = {k: v[0] for k, v in adapter.last_aux.items()}
                    hl_texts = _aux_hl_texts(adapter.last_aux["hl_model_text"], args.num_envs)
                    over_budget: list[int] = []
                    for i in np.flatnonzero(hl_refresh).tolist():
                        setup = queue_client.current_setups[i] if queue_client is not None else None
                        budget = math.ceil(args.ll_oracle_hl_call_scale * setup["expert_hl_calls"]) if setup else 0
                        hl_calls[i] += 1
                        if budget and hl_calls[i] > budget:
                            over_budget.append(i)
                        else:
                            ll_oracle.command(i, hl_texts[i])
                    hl_refresh[:] = False
                    if over_budget:
                        print(
                            f"[INFO]: envs {over_budget}: Molmo2 called more than "
                            f"{args.ll_oracle_hl_call_scale:g}x its expert's HL calls -> ending episode"
                        )
                        base_env.request_episode_end(over_budget)
                action = ll_oracle.act()
            else:
                action = get_action(obs)
            if step == 0 and adapter is not None:
                _log_step0_depth_stats(adapter)
            if reaches_server:
                hl_calls[hl_refresh] += 1
                if args.async_hl and args.async_hl_call_scale > 0 and queue_client is not None:
                    over_budget = []
                    for i in np.flatnonzero(hl_refresh).tolist():
                        setup = queue_client.current_setups[i]
                        budget = math.ceil(args.async_hl_call_scale * setup["expert_hl_calls"]) if setup else 0
                        if budget and hl_calls[i] > budget:
                            over_budget.append(i)
                    if over_budget:
                        print(
                            f"[INFO]: envs {over_budget}: Molmo2 called more than "
                            f"{args.async_hl_call_scale:g}x its expert's HL calls -> ending episode"
                        )
                        base_env.request_episode_end(over_budget)
                hl_refresh[:] = False
            if sync_broker is not None:
                parked_ids = sync_broker.waiting_envs
                # Envs the queue has run out for (unscored filler) hold still until it is done.
                if queue_client is not None:
                    setups = queue_client.current_setups
                    parked_ids += [i for i, s in enumerate(setups) if s is None and i not in parked_ids]
        if wave_gate is not None and wave_gate.parked:
            parked_ids = sorted(set(parked_ids) | set(wave_gate.parked))
        last_policy_action = action
        policy_dt = time.perf_counter() - t_policy
        parked: torch.Tensor | None = None
        if parked_ids:
            parked = torch.zeros(args.num_envs, dtype=torch.bool, device=device)
            parked[parked_ids] = True

        # First control step of each episode (length is incremented after
        # env.step). Dumps the reset-pose robot and the command about to land.
        start_env_ids = [
            int(i)
            for i in (base_env.episode_length_buf == 0)
            .nonzero(as_tuple=False)
            .flatten()
            .tolist()
            if int(i) not in parked_ids
        ]
        if start_env_ids and args.debug:
            setups = (
                list(queue_client.current_setups)
                if queue_client is not None
                else [None] * args.num_envs
            )
            for i in start_env_ids:
                _log_episode_start_robot(
                    env_idx=i,
                    step=step,
                    episode_idx=per_env_episode_count[i],
                    base_env=base_env,
                    action=action,
                    pd_controller=pd_controller,
                    setup=setups[i],
                )

        # Model HL path (--hl model / --hl_from_model, hl_controller is None):
        # pull the server's HL aux (captured on adapter.last_aux) into the obs the
        # recorder reads. The latency video (fresh/latent side-by-side) takes
        # precedence when enabled; otherwise fall back to the HL the server
        # conditioned on (hl_model_*), which drives the HL overlay video. Both set
        # target_heatmap, so this is intentionally one branch (no double write).
        # The aux is normalized to a leading per-env axis so the recorder can
        # index every video env, not just env 0.
        if hl_controller is None and adapter is not None and adapter.last_aux:
            aux = adapter.last_aux
            if args.save_hl_latency_video and "hl_fresh_heatmap" in aux:
                obs["hl_fresh_heatmap"] = _aux_hl_heatmaps(aux["hl_fresh_heatmap"], args.num_envs)
                obs["target_heatmap"] = _aux_hl_heatmaps(
                    aux["hl_latent_heatmap"], args.num_envs
                )
                obs["hl_fresh_text"] = _aux_hl_texts(aux["hl_fresh_text"], args.num_envs)
                obs["hl_overlay_text"] = _aux_hl_texts(aux["hl_latent_text"], args.num_envs)
            elif args.hl_from_model and "hl_model_heatmap" in aux:
                obs["target_heatmap"] = _aux_hl_heatmaps(aux["hl_model_heatmap"], args.num_envs)
                obs["hl_overlay_text"] = _aux_hl_texts(aux["hl_model_text"], args.num_envs)
            if args.hl_from_model:
                has_prev = "hl_prev_action" in aux
                has_prompt = "hl_input_prompt" in aux
                if has_prev != has_prompt:
                    raise ValueError(
                        "--hl_from_model run received only one of "
                        f"'hl_prev_action'/'hl_input_prompt' (got keys {sorted(aux)}). "
                        "The HL server emits both together so the previous-action "
                        "debug video can show the exact Molmo2 question."
                    )
                if has_prev:
                    obs["hl_prev_action"] = _aux_hl_texts(
                        aux["hl_prev_action"], args.num_envs
                    )
                    obs["hl_input_prompt"] = _aux_hl_texts(
                        aux["hl_input_prompt"], args.num_envs
                    )
                if not (args.async_hl or args.ll_oracle) and "hl_model_text" in aux:
                    for i, text in enumerate(_aux_hl_texts(aux["hl_model_text"], args.num_envs)):
                        if text and text != hl_last_text[i]:
                            hl_calls[i] += 1
                        hl_last_text[i] = text

        # The client-side oracle owns oracle-run termination. Its plan-complete
        # signal is primary; a goal held for several consecutive control steps is
        # a gated backstop for plans whose final symbolic transition never fires.
        # Requests happen before env.step so that step consumes the truncation and
        # still finalizes the environment's complete episode metrics.
        if oracle_controller is not None:
            # Outside 'goal' the plan-complete flag is still recorded
            # (oracle_done) but only a stuck plan ends the episode here.
            done_set = set(oracle_done_ids) if end_on_goal else set()
            backstop_ids: list[int] = []
            if end_on_goal:
                streak_ids = (
                    (base_env._success_streak >= _ORACLE_SUCCESS_BACKSTOP_STEPS)
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                    .tolist()
                )
                backstop_ids = [i for i in streak_ids if i not in done_set]
            for i in oracle_done_ids:
                if not oracle_done_flags[i]:
                    action_taken = "ending episode" if end_on_goal else "continuing"
                    print(f"[INFO]: env {i}: oracle plan complete -> {action_taken}")
                oracle_done_flags[i] = True
            for i in backstop_ids:
                if not oracle_success_backstop_flags[i]:
                    print(
                        f"[INFO]: env {i}: success held for "
                        f"{_ORACLE_SUCCESS_BACKSTOP_STEPS} steps -> ending oracle episode"
                    )
                oracle_success_backstop_flags[i] = True
            stuck_ids = [i for i in range(args.num_envs) if oracle_stuck_flags[i]]
            end_ids = sorted(done_set | set(backstop_ids) | set(stuck_ids))
            if end_ids:
                base_env.request_episode_end(end_ids)

        # The HL model owns episode termination: it emits "done" when it judges
        # the task complete. Requesting the end *before* env.step means
        # _get_dones consumes it on this step, so the "done" step is the last
        # step of the episode and evaluate() still finalizes full metrics.
        if args.hl_from_model:
            aux = adapter.last_aux if adapter is not None else {}
            if "hl_done" not in aux:
                raise ValueError(
                    "--hl_from_model run received no 'hl_done' aux from the policy "
                    f"server (got keys {sorted(aux)}). The HL model path always "
                    "emits it; without it the HL 'done' token cannot end the "
                    "episode. Check that the server runs HLLLInferAdapter with "
                    "--hl model."
                )
            done_now = np.asarray(aux["hl_done"]).reshape(-1)
            if done_now.size != args.num_envs:
                raise ValueError(
                    f"expected {args.num_envs} hl_done flags, got shape "
                    f"{np.asarray(aux['hl_done']).shape}"
                )
            if stage_hl_outputs is not None and "hl_model_text" in aux:
                blank = [""] * args.num_envs
                stage_hl_outputs(
                    _aux_hl_texts(aux["hl_model_text"], args.num_envs),
                    _aux_hl_texts(aux["hl_prev_action"], args.num_envs)
                    if "hl_prev_action" in aux
                    else blank,
                    _aux_hl_texts(aux["hl_input_prompt"], args.num_envs)
                    if "hl_input_prompt" in aux
                    else blank,
                    [bool(v) for v in done_now],
                )
            done_ids = [i for i in range(args.num_envs) if bool(done_now[i])]
            if done_ids:
                for i in done_ids:
                    if not hl_done_flags[i]:
                        print(f"[INFO]: env {i}: HL model emitted 'done' -> ending episode")
                    hl_done_flags[i] = True
                base_env.request_episode_end(done_ids)

        # Per-episode expert-scaled timeout: end an episode that has already run
        # --expert_timeout_scale times its own expert demonstration's duration.
        # Requested before env.step for the same reason as the blocks above --
        # _get_dones consumes it on this step, so it lands as a truncation and
        # evaluate() still finalizes full episode metrics.
        #
        # Budgets are recomputed from current_setups every step rather than
        # cached at reset: _reset_idx refreshes that list on every auto-reset, so
        # there is no per-env state that can go stale. episode_length_buf is
        # incremented *after* the step, so it holds the number of steps already
        # taken here and `>= budget` makes the episode exactly budget steps long.
        if expert_timeout_steps_per_frame > 0.0:
            lengths = base_env.episode_length_buf.tolist()
            over_budget: list[int] = []
            for i, setup in enumerate(queue_client.current_setups):
                # Phantom episodes after the queue drains have no expert to scale.
                if setup is None:
                    continue
                frames = int(setup["expert_episode_length"])
                budget = max(1, math.ceil(frames * expert_timeout_steps_per_frame))
                if lengths[i] >= budget:
                    over_budget.append(i)
                    print(
                        f"[INFO]: env {i}: ran {lengths[i]} steps, "
                        f"{args.expert_timeout_scale:g}x its {frames}-frame expert is "
                        f"{budget} -> ending episode"
                    )
            if over_budget:
                base_env.request_episode_end(over_budget)

        # Comparison-only oracle: tick the client-side TAMP oracle and inject its
        # HL under separate viz keys so the recorder draws a model | oracle
        # side-by-side panel. Never fed to the LL (target_heatmap/oracle_task_name
        # from the model path above are untouched).
        if comparison_oracle is not None:
            _, oracle_heatmaps = comparison_oracle.compute()
            obs["hl_oracle_heatmap"] = oracle_heatmaps
            obs["hl_oracle_text"] = comparison_oracle.display_texts
            for i in comparison_oracle.stuck_indices():
                if not oracle_stuck_flags[i]:
                    print(
                        f"[INFO]: env {i}: comparison oracle plan stuck "
                        "(overlay only; episode continues)"
                    )
                oracle_stuck_flags[i] = True

        # HL overlay video: the HL injected into obs above (oracle or model, plus
        # the comparison oracle), drawn by the recorder on this step's row.
        if stage_hl_overlay is not None:
            hl_texts = obs.get("hl_overlay_text", obs.get("oracle_task_name"))
            if hl_texts is not None:
                stage_hl_overlay(
                    hl_texts,
                    _hl_heatmaps_2d(obs.get("target_heatmap")),
                    obs.get("hl_prev_action"),
                    obs.get("hl_input_prompt"),
                    obs.get("hl_oracle_text"),
                    _hl_heatmaps_2d(obs.get("hl_oracle_heatmap")),
                )

        if args.debug:
            raw_np = action.detach().cpu().numpy().flatten()
            print(f"step {step}: {raw_np}", flush=True)

        # Handle 9-dim actions: extract pd_mode and switch gains
        # 9-dim = 7 joints + 1 gripper + 1 pd_mode action
        has_pd_mode = action.shape[-1] >= 9
        stop_ids: list[int] = []
        if has_pd_mode and play_mode:
            # sim2real play mode: 1 / 0 are fed back as the next state's dim 8;
            # -1 ends the skill, so that env does not run the step and ignores
            # the rest of its chunk, holding where the skill ended until its next
            # HL + LL call, which keeps its usual schedule. -1 itself is never fed
            # back. The held steps count on the episode clock and are recorded,
            # so an LL stuck on -1 times out.
            vals = action[:, 8].to(play_state)
            live = ~skill_done.to(vals.device)
            if parked is not None:
                live &= ~parked.to(live.device)
            if settle_busy is not None:
                live &= ~settle_busy.to(live.device)
            stop = live & (vals < -0.5)
            play_state = torch.where(live & ~stop, vals, play_state)
            play_state[stop] = 1.0
            stop_ids = stop.nonzero(as_tuple=False).squeeze(-1).tolist()
            for i in stop_ids:
                ll_skill_stops[i] += 1
            for i in (live & (vals.abs() < 0.5)).nonzero(as_tuple=False).squeeze(-1).tolist():
                ll_pause_steps[i] += 1
            sync_state_dim8()
            if stage_pd_modes is not None:
                stage_pd_modes(pd_controller.applied_modes)
            action = action[..., :8]
            if stop_ids:
                hold = settle_hold_action(base_env).to(device)
                if skill_done_hold is None:
                    skill_done_hold = hold.clone()
                skill_done_hold[stop] = hold[stop]
                skill_done |= stop
                hl_refresh[stop_ids] = True
                if args.debug:
                    print(f"[INFO]: step {step}: play mode -1 (skill done) for envs {stop_ids}")
            if ll_ends_episode:
                # Without the oracle HL any -1 ends the episode. With it, -1 on the
                # final command holds the robot until the success check passes or
                # _LL_DONE_SETTLE_STEPS pass (plan confirmation ignores the gripper).
                end_ids: list[int] = []
                if oracle_controller is None:
                    end_ids = stop_ids
                    for i in end_ids:
                        print(f"[INFO]: env {i}: play mode -1 -> ending episode")
                else:
                    # "Final" is the oracle's state when the LL's skill began: the
                    # oracle turns pick -> place on grasp, before the LL's -1.
                    final_ids = set(oracle_controller.final_command_indices())
                    for i, started_final in enumerate(ll_skill_started_final):
                        if started_final is None:
                            ll_skill_started_final[i] = i in final_ids
                    for i in stop_ids:
                        if ll_skill_started_final[i] or oracle_done_flags[i]:
                            ll_final_settle[i] = 0
                            print(f"[INFO]: env {i}: play mode -1 on the final command -> settling")
                    for i, held in enumerate(ll_final_settle):
                        if held < 0:
                            continue
                        goal_met = bool(base_env._success_streak[i] > 0)
                        if goal_met or held >= _LL_DONE_SETTLE_STEPS:
                            status = "met" if goal_met else "not met"
                            print(
                                f"[INFO]: env {i}: goal {status} after {held} settle steps "
                                f"(oracle plan complete: {oracle_done_flags[i]}) -> ending episode"
                            )
                            end_ids.append(i)
                        else:
                            ll_final_settle[i] = held + 1
                if end_ids:
                    base_env.request_episode_end(end_ids)
            if home_arm is not None and skill_done_hold is not None:
                # --go_home: the held env's target walks home; the HL refreshes once there.
                if stop_ids:
                    hl_texts = _aux_hl_texts(adapter.last_aux["hl_model_text"], args.num_envs)
                    for i in stop_ids:
                        words = hl_texts[i].strip().lower().split()
                        if words and words[0] in _GO_HOME_SKILLS and ll_final_settle[i] < 0:
                            going_home[i] = True
                            home_settle[i] = -1
                            hl_refresh[i] = False
                if bool(going_home.any()):
                    n_arm = home_arm.shape[-1]
                    cmd = skill_done_hold[:, :n_arm]
                    # Scale all joints by one factor so they arrive together: a straight
                    # joint-space line (as agentic_robotics' return_home), gripper open.
                    delta = home_arm - cmd
                    scale = (home_max_step / delta.abs().amax(-1, keepdim=True).clamp_min(1e-9)).clamp(max=1.0)
                    stepped = cmd + delta * scale
                    skill_done_hold[going_home, :n_arm] = stepped[going_home]
                    skill_done_hold[going_home, n_arm] = 1.0
                    at_cmd = going_home & ((skill_done_hold[:, :n_arm] - home_arm).abs().amax(-1) < 1e-6)
                    home_settle[at_cmd] += 1
                    arm_pos = base_env.robot.articulation.data.joint_pos[:, arm_ids].to(device)
                    arrived = at_cmd & (
                        ((arm_pos - home_arm).abs().amax(-1) < _GO_HOME_TOL)
                        | (home_settle >= _GO_HOME_SETTLE_STEPS)
                    )
                    arrived_ids = arrived.nonzero(as_tuple=False).squeeze(-1).tolist()
                    if arrived_ids:
                        going_home[arrived] = False
                        home_settle[arrived] = -1
                        hl_refresh[arrived_ids] = True
                        if args.debug:
                            print(f"[INFO]: step {step}: --go_home reached home for envs {arrived_ids}")
            if args.post_place_lift > 0 and skill_done_hold is not None:
                # --post_place_lift: a -1 with the gripper open (a place's release,
                # not a flip / rotate) raises the hold target before the next query.
                n_arm = len(lift_arm_ids)
                if stop_ids:
                    commands = oracle_controller.current_commands() if oracle_controller is not None else None
                    for i in stop_ids:
                        verb = commands[i][0] if commands is not None and commands[i] is not None else None
                        if (
                            float(skill_done_hold[i, n_arm]) > 0
                            and ll_final_settle[i] < 0
                            and verb not in ("flip", "rotate")
                        ):
                            lifting[i] = True
                            lift_left[i] = args.post_place_lift
                            lift_settle[i] = -1
                if bool(lifting.any()):
                    jac = lift_art.root_physx_view.get_jacobians()[:, lift_jac_body][..., lift_jac_joints]
                    jac = jac.to(device=device, dtype=torch.float32)  # (num_envs, 6, n_arm)
                    rise = torch.where(lifting, lift_left.clamp(max=lift_max_step), torch.zeros_like(lift_left))
                    twist = torch.zeros(args.num_envs, 6, device=device)
                    twist[:, 2] = rise
                    # Damped least squares: straight up, orientation held.
                    jjt = jac @ jac.transpose(1, 2) + _LIFT_DLS_LAMBDA**2 * torch.eye(6, device=device)
                    dq = (jac.transpose(1, 2) @ torch.linalg.solve(jjt, twist.unsqueeze(-1))).squeeze(-1)
                    skill_done_hold[lifting, :n_arm] += dq[lifting].to(skill_done_hold)
                    lift_left -= rise
                    at_top = lifting & (lift_left <= 1e-6)
                    lift_settle[at_top] += 1
                    arm_pos = lift_art.data.joint_pos[:, lift_arm_ids].to(device)
                    arrived = at_top & (
                        ((arm_pos - skill_done_hold[:, :n_arm].to(device)).abs().amax(-1) < _GO_HOME_TOL)
                        | (lift_settle >= _LIFT_SETTLE_STEPS)
                    )
                    if bool(arrived.any()):
                        lifting[arrived] = False
                        lift_settle[arrived] = -1
                        if args.debug:
                            arrived_ids = arrived.nonzero(as_tuple=False).squeeze(-1).tolist()
                            print(f"[INFO]: step {step}: --post_place_lift done for envs {arrived_ids}")
            if skill_done_hold is not None and bool(skill_done.any()):
                action = action.clone()
                rows = skill_done.to(action.device)
                action[rows] = skill_done_hold[skill_done].to(action)
        elif has_pd_mode:
            pd_mode_vals = action[:, 8] if action.ndim > 1 else action[8].unsqueeze(0)
            # Parked envs carry a placeholder row, not a pd_mode prediction.
            pd_mask = None if settle_busy is None else ~settle_busy
            if parked is not None:
                pd_mask = ~parked if pd_mask is None else pd_mask & ~parked
            switched = pd_controller.apply(pd_mode_vals, env_mask=pd_mask)
            # A switching env holds for its settle ticks, then runs the action
            # that triggered the switch.
            if deferred_settle is not None and bool(switched.any()):
                deferred_settle.start(
                    switched,
                    pd_controller.settle_ticks(),
                    settle_hold_action(base_env),
                    action[..., :8],
                )
            # Training's agent_pos pd_mode is the gain profile in effect when the
            # frame was captured. The transform only runs at chunk boundaries and
            # would otherwise carry the first step of the previous chunk (for
            # every env, including ones mid-chunk), so resync it from the
            # committed modes every step.
            sync_state_dim8()
            # Record the mode the gains ended up in, which is what datagen
            # records; the controller's dead zone and debounce mean it can lag
            # the raw prediction, and that prediction is stripped below anyway.
            if stage_pd_modes is not None:
                stage_pd_modes(pd_controller.applied_modes)
            action = action[..., :8]

        # Control ticks the recording skipped before the row just replayed run
        # as hold steps of the previous row's action, so the sim stays in step
        # with the recording (see ReplayPolicy.pop_gap_holds).
        if replay_policy is not None and deferred_settle is not None:
            gap_holds = replay_policy.pop_gap_holds(available=~deferred_settle.busy)
            if gap_holds is not None:
                gap_mask, gap_ticks, gap_hold_action = gap_holds
                deferred_settle.start(gap_mask, gap_ticks, gap_hold_action, action[..., :8])

        if parked is not None:
            action = action.clone()
            action[parked] = park_hold_action[parked].to(action)

        settle_holding: torch.Tensor | None = None
        settle_holding_ids: list[int] = []
        if deferred_settle is not None:
            action, settle_holding = deferred_settle.compose(action)
            settle_holding_ids = settle_holding.nonzero(as_tuple=False).squeeze(-1).tolist()

        # Replay owns its own termination: the action just built is the last
        # recorded frame for these envs. Requested before env.step so that step
        # still applies it, consumes the truncation, and finalizes full metrics.
        # An env holding through a PD-switch settle has not applied that frame
        # yet, so its request waits for the step that runs the deferred action.
        if replay_policy is not None:
            replay_done_ids = [
                i for i in replay_policy.done_indices() if i not in settle_holding_ids
            ]
            if replay_done_ids:
                for i in replay_done_ids:
                    print(f"[INFO]: env {i}: recording exhausted -> ending episode")
                base_env.request_episode_end(replay_done_ids)

        # Hold steps are invisible to the episode clock (the step's increment
        # cancels the decrement) and to the recording, like the raw substeps of
        # the legacy settle. A hold step that ends its episode still counts: in
        # the legacy settle that step ran its action, so it was a counted step.
        if settle_holding is not None and settle_holding_ids:
            uncounted = settle_holding.to(base_env.episode_length_buf.device)
            uncounted &= ~base_env._episode_end_requests
            base_env.episode_length_buf[uncounted] -= 1
            if skip_recording_next_step is not None:
                skip_recording_next_step(settle_holding_ids)
        # Parked steps are hidden the same way: the episode starts at the boundary.
        if parked is not None:
            uncounted = parked.to(base_env.episode_length_buf.device)
            uncounted &= ~base_env._episode_end_requests
            base_env.episode_length_buf[uncounted] -= 1
            if skip_recording_next_step is not None:
                skip_recording_next_step(parked_ids)

        # The goal each env is graded against on this step; auto-reset inside
        # env.step can already move a finished env to its next queue goal.
        if goal_labels_by_idx is not None:
            goal_idx_before_step = list(base_env._per_env_goal_list)
        obs, reward, terminated, truncated, infos = env.step(action)

        # --ll_oracle: an env asks Molmo2 again once its command ends or its episode resets.
        if ll_oracle is not None:
            hl_refresh[ll_oracle.pop_finished()] = True
            if "final_info" in infos:
                reset_ids = infos["_final_info"].nonzero(as_tuple=False).view(-1).tolist()
                ll_oracle.reset_envs(reset_ids)
                hl_refresh[reset_ids] = True

        # Score the sim state against the recorded state the applied row led to. Envs that
        # finished (and auto-reset) inside this step already hold the next
        # episode's frame 0, so they are excluded rather than compared, and envs
        # that spent the step holding have not applied a new row (the last hold
        # step of an unrecorded-tick gap scores the row before the gap).
        if replay_policy is not None:
            reset_ids = (
                infos["_final_info"].nonzero(as_tuple=False).squeeze(-1).tolist()
                if "final_info" in infos
                else []
            )
            replay_policy.record_divergence(
                skip_env_ids=reset_ids, holding_env_ids=settle_holding_ids
            )

        # End episodes on genuine goal-success. base_env._success_once now
        # requires the scene settled and the gripper released (see the goal gate
        # in pddl_goal_tensor_conversions.py), so this is meaningful. Requesting
        # the end here truncates on the next step, finalizing with full metrics
        # exactly like the old oracle-done path; the episode_length_s timeout is
        # the fallback for failed episodes. Envs that reset this step already had
        # _success_once cleared in _reset_idx, so they are not re-flagged.
        # Skipped on HL-controller runs: the oracle uses plan completion plus its
        # sustained-success backstop above, while the learned HL model owns
        # termination
        # (its "done" ends the episode above), so an episode whose goal is
        # reached without a "done" runs out episode_length_s and is visible as
        # success_once=True with hl_done=False.
        if (
            not args.hl_from_model
            and hl_controller is None
            and end_on_goal
        ):
            success_ids = base_env._success_once.nonzero(as_tuple=False).squeeze(-1).tolist()
            if success_ids:
                base_env.request_episode_end(success_ids)

        if queue_client is not None:
            queue_client.verify_after_step()

        extra_log: dict = {}
        if "final_info" in infos:
            if deferred_settle is not None:
                deferred_settle.clear(infos["_final_info"])
            mask = infos["_final_info"].clone()
            if queue_client is not None:
                for i in range(args.num_envs):
                    if queue_client.previous_setups[i] is None:
                        mask[i] = False

            # A collision fails the episode whatever the goal check says.
            hit = collision_monitor.collided & infos["_final_info"].to(
                collision_monitor.collided.device
            )
            ep = infos["final_info"].get("episode")
            if ep is not None and "success_once" in ep and args.collision != "off":
                if args.collision == "penalize":
                    # penalize keeps the success that ignores collisions too.
                    ep["success_once_ignore_collision"] = ep["success_once"].clone()
                    ep["success_at_end_ignore_collision"] = ep["success_at_end"].clone()
                ep["success_once"] = ep["success_once"] & ~hit
                ep["success_at_end"] = ep["success_at_end"] & ~hit
                ep["collision"] = hit
                ep["collision_count"] = collision_monitor.collision_counts()
            finished_sub_goals = collision_monitor.finish_sub_goals(infos["_final_info"])
            collision_monitor.reset(infos["_final_info"])

            ep_metrics_dict = infos["final_info"].get("episode", {})
            for i in range(args.num_envs):
                if not bool(mask[i].item()):
                    continue
                prev_setup = queue_client.previous_setups[i] if queue_client is not None else None
                ep_metrics = {k: v[i] for k, v in ep_metrics_dict.items()}
                # Set by render_replay.stage when the episode's setup was applied.
                if prev_setup is not None and "render_replay_status" in prev_setup:
                    ep_metrics["render_replay_status"] = prev_setup["render_replay_status"]
                # Set by episode_queue.apply_setup under --replay_block_identity.
                if prev_setup is not None and "block_identity_status" in prev_setup:
                    ep_metrics["block_identity_status"] = prev_setup["block_identity_status"]
                if args.hl_from_model:
                    # Only meaningful when the HL model owns termination; pairs
                    # with success_once to spot a reached goal the HL missed.
                    ep_metrics["hl_done"] = hl_done_flags[i]
                if oracle_controller is not None:
                    ep_metrics["oracle_done"] = oracle_done_flags[i]
                    ep_metrics["oracle_success_backstop"] = (
                        oracle_success_backstop_flags[i]
                    )
                if replay_policy is not None:
                    # Scalar divergence vs the recording, folded into the
                    # per-episode table and the aggregate results under a
                    # ``replay_`` prefix (an underscore, not ``/``: keys with a
                    # slash are dropped from the wandb summary and hidden in
                    # the console table). Missing values are NaN, which the
                    # aggregation treats as absent.
                    for k, v in replay_policy.episode_metrics(i).items():
                        ep_metrics[f"replay_{k}"] = v
                        eval_metrics[f"replay_{k}"].append(
                            torch.tensor([v], dtype=torch.float32)
                        )
                if play_mode:
                    for key, counts in (("ll_skill_stops", ll_skill_stops), ("ll_pause_steps", ll_pause_steps)):
                        ep_metrics[key] = counts[i]
                        eval_metrics[key].append(torch.tensor([float(counts[i])]))
                if oracle_controller is not None or comparison_oracle is not None:
                    ep_metrics["oracle_stuck"] = oracle_stuck_flags[i]
                if ll_oracle is not None:
                    for key, counts in (("ll_oracle_failures", ll_oracle.failures),):
                        ep_metrics[key] = int(counts[i])
                        eval_metrics[key].append(torch.tensor([float(counts[i])]))
                    ll_oracle.failures[i] = 0
                if args.hl_from_model:
                    ep_metrics["hl_calls"] = int(hl_calls[i])
                    eval_metrics["hl_calls"].append(torch.tensor([float(hl_calls[i])]))
                    hl_calls[i] = int(args.async_hl)
                goal_labels: list[str] | None = (
                    list(goal_labels_by_idx[goal_idx_before_step[i]])
                    if goal_labels_by_idx is not None
                    else None
                )
                if goal_labels is not None and prev_setup is not None and prev_setup.get("block_identity"):
                    # Graded in source block names; show the rendered names alongside.
                    identity = BlockIdentity.from_dict(prev_setup["block_identity"])
                    goal_labels = [
                        f"{lab} [{rendered_text(lab, identity)}]"
                        if rendered_text(lab, identity) != lab
                        else lab
                        for lab in goal_labels
                    ]
                per_episode_goal_conjuncts.append(goal_labels)
                goal_objects: list[str] | None = (
                    list(goal_objects_by_idx[goal_idx_before_step[i]])
                    if goal_objects_by_idx is not None
                    else None
                )
                if i in finished_sub_goals:
                    # Per sub-goal, in order; the aggregate gets a rate per skill
                    # (sub_goal_success_pick, _place, _flip, _rotate).
                    names, oks, dones = finished_sub_goals[i]
                    ep_metrics["sub_goals"] = names
                    scored = {"sub_goal_success": oks}
                    if args.collision == "penalize":
                        scored["sub_goal_success_ignore_collision"] = dones
                    for key, flags in scored.items():
                        ep_metrics[key] = flags
                        for name, ok in zip(names, flags):
                            eval_metrics[f"{key}_{name.split()[0]}"].append(torch.tensor([float(ok)]))
                episode_table.add_episode(
                    env_idx=i,
                    episode_idx=per_env_episode_count[i],
                    prompt=(
                        prompt_resolver.prompt_for_setup(prev_setup)
                        if prev_setup is not None
                        else None
                    ),
                    queue_setup=prev_setup,
                    metrics=ep_metrics,
                    eval_first_frames=eval_first_frames.get(i),
                    policy_input_debug=policy_input_debug_paths.get(i),
                    goal_conjunct_labels=goal_labels,
                    goal_objects=goal_objects,
                    goal_trace=infos["final_info"].get("goal_trace", {}).get(i),
                    step_dt_s=step_dt_s,
                )
                if goal_labels is not None:
                    # (CT - IT) / (GT - IT) over the goal conjuncts, in [-1, 1];
                    # None (NaN in the aggregate, which drops it) when the goal
                    # already held at the start or a conjunct summary is missing.
                    rec = episode_table.records[-1]
                    gt = rec["num_goal_conjuncts"]
                    it = rec["num_goal_conjuncts_initial"]
                    ct = rec["num_goal_conjuncts_final"]
                    partial_progress = (
                        (ct - it) / (gt - it)
                        if it is not None and ct is not None and gt > it
                        else None
                    )
                    rec["partial_progress"] = partial_progress
                    eval_metrics["partial_progress"].append(
                        torch.tensor(
                            [float("nan") if partial_progress is None else partial_progress]
                        )
                    )
                per_env_episode_count[i] += 1
                hl_done_flags[i] = False
                oracle_done_flags[i] = False
                oracle_success_backstop_flags[i] = False
                oracle_stuck_flags[i] = False
                ll_skill_stops[i] = 0
                ll_pause_steps[i] = 0
                ll_final_settle[i] = -1
                ll_skill_started_final[i] = None
                # One rollout only; the buffers are per-step.
                if staleness is not None and i == int(args.video_env_ids[0]):
                    staleness.close()
                # Same one-episode scope for the server-side input trace.
                if adapter is not None and ll_trace_dir is not None and i == ll_trace_env:
                    adapter.set_policy_input_trace(None)
                eval_first_frames[i] = _capture_eval_first_frames(obs, i)
                # Only the followed env past the initial reset, matching the
                # staleness recorder and the LL trace above: every env at every
                # episode start would be num_envs x episodes matplotlib figures,
                # each also decoding a dataset depth clip per camera.
                if policy_input_debug is not None and i == int(args.video_env_ids[0]):
                    next_setups = (
                        list(queue_client.current_setups)
                        if queue_client is not None
                        else [None] * args.num_envs
                    )
                    # A drained queue auto-resets the env into a phantom episode
                    # that is never scored or recorded (the mask above drops it),
                    # and the prompt resolver has no live setup left, so there is
                    # no episode for the panels to describe.
                    if queue_client is None or next_setups[i] is not None:
                        policy_input_debug_paths.update(
                            policy_input_debug.capture(
                                obs,
                                [i],
                                {i: per_env_episode_count[i]},
                                next_setups,
                            )
                        )

            newly_done = mask.sum().item()
            num_episodes += newly_done

            for k, v in infos["final_info"]["episode"].items():
                eval_metrics[k].append(v[mask])

            if "stats" in infos["final_info"]:
                for k, v in infos["final_info"]["stats"].items():
                    eval_metrics[f"stats/{k}"].append(v[mask])

            done_indices = [i for i in range(args.num_envs) if bool(mask[i].item())]

            # Per-env resets
            if adapter is not None and newly_done > 0:
                adapter.reset_envs(done_indices)
            skill_done[mask.to(skill_done.device)] = False
            going_home[mask.to(going_home.device)] = False
            home_settle[mask.to(home_settle.device)] = -1
            lifting[mask.to(lifting.device)] = False
            lift_settle[mask.to(lift_settle.device)] = -1
            # The pose these envs park at until the next chunk boundary: the one
            # the auto-reset just wrote.
            if sync_broker is not None and newly_done > 0:
                park_hold_action[done_indices] = settle_hold_action(base_env)[done_indices]
            # Wave mode: an env that got a next setup waits for the rest of its wave
            # (a drained one runs an unscored episode and must not hold the wave).
            if wave_gate is not None and newly_done > 0:
                wave_ids = [i for i in done_indices if queue_client.current_setups[i] is not None]
                if wave_ids:
                    park_hold_action[wave_ids] = settle_hold_action(base_env)[wave_ids]
                    wave_gate.park(wave_ids)

            # Reset PD mode state for environments that just finished, then
            # immediately re-apply the training-time PD profile so the next
            # step starts under the impedance the policy was trained under.
            # ``current_setups`` was refreshed by the env's auto-reset just
            # above, so the per-env ``initial_pd_mode`` now reflects the newly
            # fetched episode for each done env; envs outside ``mask`` are
            # ignored by ``initialize_after_reset``. Without a per-episode
            # initial mode, the done envs return to the constructor's stiff
            # ``initial_mode`` so every episode starts like the first: no
            # settle, and the debounce stays active.
            if newly_done > 0:
                pd_controller.reset_envs(mask)
                if pd_init_enabled:
                    target_modes = _initial_pd_modes_from_setups(
                        list(queue_client.current_setups) if queue_client is not None else None,
                        args.num_envs,
                        device,
                    )
                    pd_controller.initialize_after_reset(mask, target_mode=target_modes)
                    if play_mode:
                        play_state[mask.to(play_state.device)] = 1.0
                    sync_state_dim8()
                else:
                    pd_controller.initialize_after_reset(mask, target_mode=1)

            # Load the recordings the auto-reset placed in the finished envs;
            # current_setups already reflects the newly fetched episodes.
            if replay_policy is not None and newly_done > 0:
                replay_policy.on_reset(
                    done_indices, [queue_client.current_setups[i] for i in done_indices]
                )

            # Per-episode re-seeding at episode boundaries.
            if args.seed is not None and newly_done > 0:
                ep_idx = 0
                if queue_client is not None:
                    setup = queue_client.current_setups[0]
                    if setup is not None:
                        ep_idx = setup.get("episode_index", 0)
                    else:
                        ep_idx = per_env_episode_count[0]
                else:
                    ep_idx = per_env_episode_count[0]
                ep_seed = _episode_seed(args.seed, ep_idx)
                _reseed_eval(ep_seed, adapter)

            # Rebuild the HL controller for envs that just auto-reset;
            # current_setups now reflects the new episode and the sim holds the
            # freshly-reset state.
            if (hl_controller is not None or comparison_oracle is not None) and newly_done > 0:
                new_setups = [
                    queue_client.current_setups[i] if queue_client is not None else None
                    for i in done_indices
                ]
                if hl_controller is not None:
                    hl_controller.on_reset(done_indices, new_setups)
                    # Clear buffered HL history so the next episode does not select
                    # a stale entry from the finished one.
                    if hl_latency_buffer is not None:
                        for i in done_indices:
                            hl_latency_buffer.reset(i)
                if comparison_oracle is not None:
                    comparison_oracle.on_reset(done_indices, new_setups)

            desc_parts = []
            for key in ("success_once", "return"):
                if key in eval_metrics:
                    vals = torch.cat([v.float().flatten() for v in eval_metrics[key]])
                    desc_parts.append(f"{key}: {vals.mean():.3f}")
                    if tracker is not None:
                        extra_log[f"eval/{key}"] = float(vals.mean())
            desc_parts.append(f"episodes: {num_episodes}")
            # refresh=False: let the throttled mininterval redraw pick this up
            # instead of forcing an extra line on every episode completion.
            pbar.set_description(" | ".join(desc_parts), refresh=False)

            if tracker is not None:
                extra_log["eval/episodes_completed"] = num_episodes
                tracker.record_episodes(int(newly_done))

        # After the finalize block so just-reset envs (and their HL / replay
        # state) start the new episode's reference poses.
        targets: list[set[str] | None] = [None] * args.num_envs
        commands: list[_HLCommand | None] | None = None
        if hl_controller is not None:
            commands = hl_controller.current_commands()
            collision_monitor.set_oracle_place_px(hl_controller.target_xy_batch, commands)
        elif replay_policy is not None:
            targets = [replay_policy.current_targets(i) for i in range(args.num_envs)]
        elif args.hl_from_model and adapter is not None and "hl_model_text" in (adapter.last_aux or {}):
            commands = collision_monitor.hl_commands(
                _aux_hl_texts(adapter.last_aux["hl_model_text"], args.num_envs)
            )
            if ll_oracle is not None:
                # Commands the oracle skipped as infeasible open no sub-goal.
                for i in ll_oracle.skipped:
                    commands[i] = None
                ll_oracle.skipped.clear()
        collided = collision_monitor.update(targets, commands)
        if args.collision == "off":
            collided = []
        for i, moved in collided:
            print(f"[INFO]: env {i}: collision, non-target block(s) {moved} moved")
            # Save the episode to the LeRobot failure split, as success_once scores it.
            if hasattr(data_recorder, "success_vetoed"):
                data_recorder.success_vetoed[i] = True
                data_recorder.success_flags[i] = False
        # Replay always plays its recording to the end.
        if args.collision == "end" and replay_policy is None and collided:
            base_env.request_episode_end([i for i, _ in collided])
        # Release a complete wave before the next policy query: its recorded renders,
        # a fresh obs, and no chunk/history computed from the parked (stale) scene.
        if wave_gate is not None and wave_gate.ready(
            i for i in range(args.num_envs) if queue_client.current_setups[i] is not None
        ):
            wave = wave_gate.release()
            obs = _refresh_recorded_renders(
                env, base_env, wave, warmup_frames=args.render_replay_warmup_frames, obs=obs
            )
            if adapter is not None:
                adapter.reset_envs(wave)
            for i in wave:
                eval_first_frames[i] = _capture_eval_first_frames(obs, i)

        if tracker is not None:
            transfer_kwargs: dict = {}
            if adapter is not None:
                t = adapter.transfer_timings
                transfer_kwargs["transfer_obs_to_cpu_ms"] = t["obs_to_cpu_ms"]
                transfer_kwargs["transfer_action_to_gpu_ms"] = t["action_to_gpu_ms"]
            tracker.record_step(policy_call_seconds=policy_dt, **transfer_kwargs)
            if wandb_state is not None:
                wandb_state["env_steps"] = tracker.total_env_steps

            if data_recorder is not None and hasattr(data_recorder, "recent_save_times"):
                times = []
                try:
                    while True:
                        times.append(data_recorder.recent_save_times.pop())
                except IndexError:
                    pass
                if times:
                    avg_save_time = sum(times) / len(times)
                    extra_log["throughput/episode_save_seconds"] = avg_save_time

            tracker.maybe_log(extra_log if extra_log else None)

        if queue_client is not None and queue_client.done:
            print("Queue empty. Terminating evaluation.")
            break

        if args.num_episodes is not None and num_episodes >= args.num_episodes:
            break

    elapsed = time.perf_counter() - start_time

    if staleness is not None:
        png = staleness.save()
        if png is not None:
            print(f"[INFO]: HL conditioning staleness plot: {png}")

    if ll_trace_dir is not None:
        panels = sorted(ll_trace_dir.glob("ll_input_*.png")) if ll_trace_dir.is_dir() else []
        if not panels:
            # The server renders with its own filesystem view. That is the same
            # node under scripts/eval/run_eval_pipeline.sh (server on GPU 0,
            # workers on GPUs 1..N), but not for a genuinely remote
            # --policy_host -- in which case the panels exist over there and this
            # directory is empty, which is worth failing on rather than
            # reporting an empty wandb table.
            raise RuntimeError(
                f"a policy-input trace was requested for env {ll_trace_env} but "
                f"{ll_trace_dir} holds no panels. The policy server renders them "
                f"with its own filesystem view, so this fails when the server does "
                f"not share a filesystem with this worker (--policy_host "
                f"{args.policy_host!r}), or when the server is not a maniflow one."
            )
        print(f"[INFO]: LL policy-input trace: {len(panels)} panel(s) in {ll_trace_dir}")

    if hl_trace_dir is not None:
        hl_panels = (
            sorted(hl_trace_dir.glob("hl_input_*.png")) if hl_trace_dir.is_dir() else []
        )
        if not hl_panels:
            raise RuntimeError(
                f"an HL policy-input trace was requested for env {ll_trace_env} but "
                f"{hl_trace_dir} holds no panels. The policy server renders them "
                f"with its own filesystem view, so this fails when the server does "
                f"not share a filesystem with this worker (--policy_host "
                f"{args.policy_host!r}), or when Molmo2 never ran (no "
                f"--hl_from_model / --hl model)."
            )
        print(f"[INFO]: HL policy-input trace: {len(hl_panels)} panel(s) in {hl_trace_dir}")

    # --- TAMPER seeds ---
    # Written before the metrics report so a crash in plotting/aggregation does
    # not lose states that took a full eval run to collect.
    if tamper_collector is not None:
        summary = tamper_collector.save()
        print(
            f"[TAMPER]: captured {summary['num_captured']} state(s) across "
            f"{len(summary['problems'])} problem(s) -> {args.tamper_output_dir}"
        )
        for slug, info in summary["problems"].items():
            print(
                f"  {slug}: {info['num_instances']} instance(s) "
                f"(--env_path {info['dir']} --instance-ids 0..{info['num_instances'] - 1})"
            )

    # --- Aggregate and report ---
    if eval_metrics:
        per_episode = _collect_per_episode_metrics(eval_metrics)
        aggregated = _aggregate_metrics(eval_metrics)
        _print_results(aggregated, num_episodes, elapsed)
        success_efficiency: float | None = None
        if args.hl_from_model:
            from toyblocks.utils.wandb_logging import _success_efficiency

            success_efficiency = _success_efficiency(episode_table.records)
        _save_results(
            output_dir,
            aggregated,
            args,
            num_episodes,
            elapsed,
            per_episode=per_episode,
            per_episode_goal_conjuncts=per_episode_goal_conjuncts,
            success_efficiency=success_efficiency,
        )

        if tracker is not None:
            tracker.final_log({"eval/episodes_completed": num_episodes})
            if args.wandb:
                import wandb

                from toyblocks.utils.wandb_logging import log_aggregates_table

                log_aggregates_table(aggregated, step=tracker.total_env_steps)
                wandb.run.summary["episodes_completed"] = num_episodes
                wandb.run.summary["wall_seconds"] = elapsed
                wandb.run.summary["env_steps_per_sec"] = tracker.total_env_steps / max(
                    elapsed, 1e-6
                )
                for k, stats in aggregated.items():
                    if "/" in k:
                        continue
                    for stat in ("mean", "std", "min", "max"):
                        wandb.run.summary[f"{k}_{stat}"] = stats[stat]
    else:
        if queue_client is not None:
            print(
                "\nNo episodes completed before the queue drained. "
                "The queue might be smaller than --num_envs, or every "
                "queued episode timed out before reaching a terminal."
            )
        else:
            print(
                f"\nNo episodes completed in {args.num_eval_steps} steps. "
                "Try increasing --num_eval_steps or --num_envs."
            )
        if tracker is not None:
            tracker.final_log({"eval/episodes_completed": 0})
            if args.wandb:
                import wandb

                wandb.run.summary["episodes_completed"] = 0

    if tracker is not None:
        tracker.save_json_sidecar(output_dir)

    # Written now so the results survive a failing env.close(); rewritten, and the
    # episode tables logged, by the returned callback once the LeRobot recorder
    # has closed and its episode indices (and HL overlay videos) are final.
    episode_table.save_json_sidecar()

    def log_episode_tables(episode_index_map: dict[tuple[int, int], tuple[str, int]]) -> None:
        episode_table.attach_lerobot_episode_indices(episode_index_map)
        if not args.wandb or episode_table.num_records == 0:
            episode_table.save_json_sidecar()
            return
        step = tracker.total_env_steps if tracker is not None else None
        episode_table.log(
            fps=int(wandb_state["video_fps"]) if wandb_state is not None else 25,
            step=step,
        )
        from toyblocks.utils.wandb_logging import log_run_stats_table

        log_run_stats_table(
            episode_table.records,
            run_scalars={
                "policy_name": args.wandb_name,
                "task": args.task,
                "dataset_path": args.dataset_path,
                "wall_seconds": elapsed,
                "env_steps_per_sec": (
                    tracker.total_env_steps / max(elapsed, 1e-6) if tracker is not None else None
                ),
                "num_workers": 1,
            },
            step=step,
        )

    return log_episode_tables
