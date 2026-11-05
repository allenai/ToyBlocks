"""Molmo2's low-level oracle: the vecgen skills running one Molmo2 command per env.

Each command becomes a one-skill vecgen program (``pick``, ``place_on``, a table place at
Molmo2's pixel, ``flip``, ``rotate``) that :class:`_OracleEngine` runs with vecgen's
batched IK / cuRobo planning (``Geometry``) and segment table. The engine never steps the
sim: run_eval's ``env.step`` does, with the arm and gripper targets :meth:`LLOracle.act`
returns. A skill runs once: no retries, no snapshot restores, no re-planning.

Oracle failures (``ll_oracle_failures``; Molmo2 is asked again):
- no feasible parameters (grasp, place pose, flip / rotate start), or a flip / rotate of a
  block on no known face;
- place with nothing held, flip / rotate while holding;
- an IK / motion plan fails, or an effect does not hold (e.g. a flip lands wrong).

Not oracle failures (nothing runs, no sub-goal; Molmo2 is asked again):
- the point is on no block (25 px) or, for a table place, off the table;
- a pick of the held block; a place on a covered stack that still cannot run.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from isaaclab.utils import math as math_utils
from torch import Tensor

from blocksuite.serving.maniflow_serving_utils import _parse_hl_action
from blocksuite.tamp.blocks.camera_view import blocks_in_front_view
from blocksuite.tamp.blocks.pddl_goal_tensor_conversions import (
    _FRONT_CAMERA_SENSOR,
    ORI_QUATERNIONS,
    _quat_close_batched,
    front_camera_pixels,
    goal_on,
    resolve_asset,
    resolve_root_pose_w,
)
from blocksuite.vecgen.engine import Engine, EngineConfig, EngineContext
from blocksuite.vecgen.geometry import Geometry
from blocksuite.vecgen.math import pose_compose, pose_inverse, world_extents
from blocksuite.vecgen.program import Phase, Program, SkillCall
from blocksuite.vecgen.segments import Segment
from blocksuite.vecgen.sim import VecSim
from blocksuite.vecgen.skills.base import Skill
from blocksuite.vecgen.skills.flip import Flip
from blocksuite.vecgen.skills.pnp import (
    MAX_REACH_RADIUS,
    PLACE_FINGER_CLEARANCE,
    PLACE_Z_GAP,
    PRE_PLACE_STANDOFF,
    Pick,
    Place,
)
from blocksuite.vecgen.skills.rotate import Rotate
from blocksuite.vecgen.symbolic.predicates import ORI_QUATERNIONS as ORI_QUAT_TENSOR
from blocksuite.vecgen.symbolic.predicates import classify_orientation, ori_index

logger = logging.getLogger(__name__)

DOMAIN_PDDL = Path(__file__).resolve().parents[1] / "tamp" / "blocks" / "domain.pddl"

# Pixel rings tried around a "place on table" point, nearest first. All lie within the
# 25 px a Molmo table placement is scored by (run_eval's place_px_tol).
_TABLE_RING_PX = (0.0, 8.0, 16.0, 24.0)
# A Molmo2 point is on a block within this many px of its projected center (as run_eval's
# sub-goal tracker resolves it); otherwise the command points at no block.
_POINT_PX = 25.0
# Molmo2's flip directions that the domain's flip_dir names differently.
_FLIP_DIRS = {"left": "left_flip", "right": "right_flip"}
# Control steps (~1 s) an env holds still after letting go of or moving a block.
_SETTLE_STEPS = 25
_TABLE = "workspace_table"
# The datagen stiffness profile the skills' feed-forward and settle times are tuned for.
_PROFILE = "medium"
# A finger target above this is a closed gripper (vecgen closes to 0.7, flips to 0.9).
_CLOSED_TARGET = 0.05


def table_points(base_env: Any, env_idx: int, uv: Sequence[float], z_local: float) -> list[Tensor]:
    """Env-local points at height ``z_local`` that the front camera sees at, or around, pixel ``uv``.

    The inverse of ``front_camera_pixels`` for each pixel in :data:`_TABLE_RING_PX`, nearest first.
    """
    cam = base_env.scene.sensors[_FRONT_CAMERA_SENSOR].data
    k = cam.intrinsic_matrices[env_idx]
    origin = base_env.scene.env_origins[env_idx]
    points = []
    for radius in _TABLE_RING_PX:
        for i in range(8 if radius else 1):
            u = uv[0] + radius * math.cos(i * math.pi / 4)
            v = uv[1] + radius * math.sin(i * math.pi / 4)
            # Camera frame is x forward, z up: u grows with -y and v with -z.
            ray_cam = torch.tensor(
                [1.0, -(u - k[0, 2]) / k[0, 0], -(v - k[1, 2]) / k[1, 1]], device=k.device
            )
            ray = math_utils.quat_apply(cam.quat_w_world[env_idx], ray_cam)
            t = (z_local + origin[2] - cam.pos_w[env_idx, 2]) / ray[2]
            points.append(cam.pos_w[env_idx] + t * ray - origin)
    return points


class _PixelTablePlace(Place):
    """``place_table`` at the table points around Molmo2's pixel (``ctx.params[e]["table_uv"]``),
    nearest first; the first in reach, in the front camera's view and clear of the other
    blocks wins. With none, the pixel's own point is attempted (as the TAMP oracle did)."""

    def __init__(self) -> None:
        super().__init__("table")

    def begin(self, ctx: EngineContext, env_ids: Tensor, args: list[tuple[str, ...]]) -> Tensor:
        b = int(env_ids.numel())
        dev = ctx.device
        ar = torch.arange(b, device=dev)
        j = torch.tensor([ctx.block_index[a[0]] for a in args], dtype=torch.long, device=dev)
        poses = ctx.block_poses()[env_ids]
        sizes = ctx.sizes[env_ids]
        blk = poses[ar, j]
        ee = ctx.sim.ee_pose()[env_ids]
        in_hand = pose_compose(pose_inverse(blk), ee)  # T_obj^-1 T_ee
        goal_idx = torch.tensor(
            [
                ori_index(a[-1]) if a[-1] else int(classify_orientation(blk[i : i + 1, 3:7]))
                for i, a in enumerate(args)
            ],
            dtype=torch.long,
            device=dev,
        )
        q_put = ORI_QUAT_TENSOR.to(dev)[goal_idx]
        ext_held = world_extents(q_put, sizes[ar, j])
        r_held = 0.5 * ext_held[:, :2].norm(dim=1)
        r_all = 0.5 * world_extents(poses[..., 3:7], sizes)[..., :2].norm(dim=2)  # [B, K]
        top = ctx.table_pose()[env_ids][:, 2] + ctx.geo.table_top[env_ids]
        z = top + ext_held[:, 2] / 2 + PLACE_Z_GAP
        put = torch.zeros(b, 7, device=dev)
        put[:, 3:7] = q_put
        for i, e in enumerate(env_ids.tolist()):
            cands = torch.stack(
                table_points(ctx.sim.base_env, e, ctx.params[e]["table_uv"], float(z[i]))
            )
            cands[:, 2] = z[i]
            reach = torch.hypot(cands[:, 0], cands[:, 1]) <= MAX_REACH_RADIUS
            others = torch.ones(len(ctx.block_names), dtype=torch.bool, device=dev)
            others[j[i]] = False
            dxy = (poses[i, None, :, :2] - cands[:, None, :2]).abs()  # [C, K, 2]
            clr = (r_held[i] + r_all[i] + PLACE_FINGER_CLEARANCE)[None, :, None]
            free = ~((dxy < clr).all(dim=2) & others[None, :]).any(dim=1)
            view = blocks_in_front_view(cands, float(ext_held[i, 0])).to(dev)
            feasible = (reach & free & view).nonzero(as_tuple=False).view(-1)
            put[i, :3] = cands[int(feasible[0]) if feasible.numel() else 0]
        ok = ctx.held[env_ids] == j
        ee_place = pose_compose(put, in_hand)
        ee_pre = ee_place.clone()
        ee_pre[:, 2] += PRE_PLACE_STANDOFF
        for i, e in enumerate(env_ids.tolist()):
            ctx.params[e].update(
                {"block": int(j[i]), "ee_place": ee_place[i].clone(), "ee_pre": ee_pre[i].clone()}
            )
        return ok


class _EvalSim(VecSim):
    """:class:`VecSim` over run_eval's env: the episode queue sets each episode up, so the
    engine hands out no reset states."""

    def num_reset_states(self) -> int:
        return 0

    def env_slots(self) -> list[int]:
        return [0] * self.num_envs

    def reset_states_by_slot(self) -> dict[int, list[int]]:
        return {0: []}


class _OracleEngine(Engine):
    """The vecgen engine split around run_eval's ``env.step``: :meth:`control` writes this
    step's arm targets, :meth:`after_step` advances phases, verifies and dispatches. Each
    command is a one-skill program started by :meth:`start`; its outcome lands in
    :attr:`results` (``env -> (success, reason)``) instead of an episode record."""

    def __init__(
        self, sim: VecSim, geo: Geometry, skills: dict[str, Skill], cfg: EngineConfig
    ) -> None:
        super().__init__(sim, geo, skills, program_fn=lambda _env, _rid: None, cfg=cfg)
        self.results: dict[int, tuple[bool, str]] = {}
        self._finished: Tensor | None = None

    def start(self, env: int, call: SkillCall, params: dict[str, Any]) -> None:
        st = self.states[env]
        st.assign(Program([call], "oracle", -1), self.next_episode)
        self.next_episode += 1
        self.phase_idx[env] = 0
        self._begun.discard(env)
        self._wait_since.pop(env, None)
        self.ctx.params[env] = dict(params)
        self.episode_ticks[env] = 0

    def cancel(self, env: int) -> None:
        """Drop ``env``'s command (its episode reset): no segment, no outcome."""
        self.table.cancel([env])
        self.states[env].clear()
        self.snapshots.pop(env, None)
        self._begun.discard(env)
        self._wait_since.pop(env, None)
        self.results.pop(env, None)
        self.ctx.params[env] = {}

    def busy(self, env: int) -> bool:
        return self.states[env].phase is not Phase.IDLE

    def control(self) -> None:
        """This step's arm targets for every env with a segment (``tick`` up to the sim step)."""
        pos, vel, acc, commanded, finished = self.table.gather()
        ids = self._ids[commanded]
        if ids.numel():
            ff = self.table.ff[ids]
            fc = self.cfg.friction_comp
            if self.cfg.target_ff and bool(ff.any()):
                self.sim.command_arm(
                    pos[ids][ff], ids[ff], vel[ids][ff], acc[ids][ff], True, friction_comp=fc
                )
                if bool((~ff).any()):
                    self.sim.command_arm(pos[ids][~ff], ids[~ff], target_ff=False, friction_comp=fc)
            else:
                self.sim.command_arm(pos[ids], ids, target_ff=False, friction_comp=fc)
            self.sim.flush()
        self.episode_ticks += commanded.long()
        self._finished = finished

    def after_step(self) -> None:
        """``tick`` after the sim step: finished segments advance; READY envs dispatch."""
        if self._finished is not None:
            done_ids = self._ids[self._finished]
            self._finished = None
            if done_ids.numel():
                self._on_segments_finished(done_ids)
        self._dispatch_ready()
        self.ticks += 1

    # -- the engine's episode handling, replaced by one outcome per command ---------------

    def _assign_idle(self) -> None:
        return  # commands come from Molmo2 through start()

    def _end_episode(self, e: int, success: bool, reason: str) -> None:
        if self.states[e].program is None:
            return
        self.results[e] = (success, reason)
        self.table.cancel([e])
        self.snapshots.pop(e, None)
        self._begun.discard(e)
        self._wait_since.pop(e, None)
        self.states[e].clear()

    def _retry_or_fail(self, e: int, reason: str) -> None:
        # One attempt; the held block stays attached (the oracle decides what follows).
        self._end_episode(e, False, reason)

    def _install(self, segs: dict[int, Segment], owner: dict[int, Any]) -> None:
        super()._install(
            {e: s for e, s in segs.items() if self.states[e].program is not None}, owner
        )


class LLOracle:
    """run_eval steps the env: :meth:`command` starts an env's Molmo2 command, :meth:`act`
    advances every env one control step and returns the action, :meth:`pop_finished` lists
    envs ready for their next command. Flip and rotate run vecgen's scripted skills."""

    def __init__(
        self, env: Any, num_envs: int, home_max_step: float, home_tol: float, go_home: bool = False
    ) -> None:
        self._base_env = env.unwrapped
        dev = torch.device(self._base_env.device)
        self.sim = _EvalSim(env, dev)
        sizes, _centre = self.sim.measure_block_dims()
        self.geo = Geometry(num_envs, self.sim.block_names, sizes, dev)
        self.geo.table_top = self.sim.measure_table_top(_TABLE)
        skills: dict[str, Skill] = {
            "pick": Pick(),
            "place_on": Place("on"),
            "place_table_px": _PixelTablePlace(),
            "flip": Flip(profile=_PROFILE),
            "rotate": Rotate(),
        }
        # Synchronous: plans and skill builds resolve inside after_step, so no result of an
        # env's earlier command can land after its episode reset.
        cfg = EngineConfig(
            max_attempts=1,
            max_replans=0,
            profile=_PROFILE,
            async_workers=0,
            async_plans=False,
            async_programs=False,
            plan_gather_ticks=0,
        )
        self.engine = _OracleEngine(self.sim, self.geo, skills, cfg)
        self._ctx = self.engine.ctx
        self._blocks = list(self.sim.block_names)
        self._next_face = {
            (d, o): o2
            for d, o, o2 in re.findall(
                r"\((?:flip_next|yaw_next)\s+(\w+)\s+(\w+)\s+(\w+)\s*\)", DOMAIN_PDDL.read_text()
            )
        }
        self._home = self.sim.art.data.default_joint_pos[:, self.sim.arm_ids].clone()
        self._goal = self._home.clone()  # where a walking arm is headed: home, or a lift
        self._home_max_step = home_max_step
        self._go_home_after = go_home  # walk home after each place / flip / rotate (--go_home)
        self._home_tol = home_tol
        self._calls: dict[int, SkillCall] = {}  # env -> its running command's skill call
        self._lift_first: set[int] = set()  # failed with its plan starting in collision
        self._queued: dict[int, str] = {}  # env -> pick to run once a dropped block settles
        self._homing = torch.zeros(num_envs, dtype=torch.bool)
        self._finished: list[int] = []
        self.failures = [0] * num_envs  # this episode's oracle failures; run_eval clears it
        self.skipped: list[int] = []  # infeasible commands this step: no sub-goal
        self._settling = [0] * num_envs  # steps left holding still after letting go
        self._profile_due: set[int] = set(range(num_envs))  # gains to (re)apply: start, resets
        self._stepped = False  # act() follows an env.step (not on the very first call)

    # -- run_eval interface --------------------------------------------------------------

    def command(self, env_idx: int, text: str) -> None:
        """Start ``env_idx`` on Molmo2's HL output ``text``.

        ``"done"`` starts nothing, and neither does ``""``: the server sends it for envs
        the queue has run out for (unscored filler), which hold still until the end.
        """
        lang, uv = _parse_hl_action(text)
        if lang in ("done", ""):
            return
        held = self._held(env_idx)
        if not self._points_at_target(env_idx, lang, uv):
            # Not a feasible command: nothing for the oracle to run (not its failure), ask again.
            logger.warning(f"env {env_idx}: {text!r} points at no block or table; asking again")
            self._finished.append(env_idx)
            return
        if lang == "pick" and held is not None and uv is not None:
            if self._nearest_block(env_idx, uv, None) == held:
                # Already holding the block Molmo2 points at: nothing to do, ask again.
                self._finished.append(env_idx)
            else:
                # Pointing at another block: drop the held one, let it settle, then pick.
                self._release(env_idx)
                self._queued[env_idx] = text
                self._settling[env_idx] = _SETTLE_STEPS
            return
        call, params = self._call_for(env_idx, lang, uv)
        if call is None and lang == "place" and held is not None:
            on = self._nearest_block(env_idx, uv, held)
            if self._top_of(env_idx, on, held) != on:
                # Pointed at a covered block whose stack takes no more: not the oracle's failure.
                logger.warning(
                    f"env {env_idx}: cannot run {text!r} on the covered {on}; asking again"
                )
                self.skipped.append(env_idx)
                self._finished.append(env_idx)
                return
        if call is None:
            logger.warning(f"env {env_idx}: cannot run {text!r}; going home")
            self._go_home(env_idx)
            return
        self._calls[env_idx] = call
        self.engine.start(env_idx, call, params)

    def act(self) -> Tensor:
        """One control step of every env's command; the joint and gripper targets as an action."""
        if self._stepped:
            self.engine.after_step()
        self._stepped = True
        self._apply_due_profiles()
        self._take_results()
        self._walk_home()
        for i, steps in enumerate(self._settling):
            if steps:
                self._settling[i] = steps - 1
                if steps == 1:
                    queued = self._queued.pop(i, None)
                    if queued is None:
                        self._finished.append(i)
                    else:
                        self.command(i, queued)
        self.engine.control()
        art = self.sim.art
        action = torch.empty(self._base_env.action_space.shape, device=self._base_env.device)
        action[:, :-1] = art.data.joint_pos_target[:, self.sim.arm_ids]
        closed = self.sim._gripper_target[:, 0] > _CLOSED_TARGET
        action[:, -1] = torch.where(closed.to(action.device), -1.0, 1.0)
        return action

    def pop_finished(self) -> list[int]:
        """Envs whose command ended (or which reached home after a failure) since the last call."""
        finished, self._finished = self._finished, []
        return finished

    def reset_envs(self, env_ids: Sequence[int]) -> None:
        """Forget the commands of envs whose episode just reset."""
        ids = list(env_ids)
        for i in ids:
            self.engine.cancel(i)
            self._calls.pop(i, None)
            self._lift_first.discard(i)
            self._goal[i] = self._home[i]
            self._queued.pop(i, None)
            self._homing[i] = False
            self._settling[i] = 0
            self._profile_due.add(i)
        if ids:
            t = torch.tensor(ids, dtype=torch.long, device=self._ctx.device)
            self.geo.detach(t)
            self._ctx.held[t] = -1
            self.sim._gripper_target[t] = 0.0
        self._finished = [i for i in self._finished if i not in ids]

    # -- commands ------------------------------------------------------------------------

    def _call_for(
        self, env_idx: int, lang: str, uv: Any
    ) -> tuple[SkillCall | None, dict[str, Any]]:
        held = self._held(env_idx)
        if uv is None:
            return None, {}
        words = lang.split()
        if lang == "pick" and held is None:
            obj = self._nearest_block(env_idx, uv, exclude=None)
            return SkillCall("pick", (obj,), hl_name="pick"), {}
        if lang == "place on table" and held is not None:
            return SkillCall("place_table_px", (held, "")), {
                "table_uv": [float(uv[0]), float(uv[1])]
            }
        if lang == "place" and held is not None:
            target = self._top_of(env_idx, self._nearest_block(env_idx, uv, exclude=held), held)
            return SkillCall("place_on", (held, target, "")), {}
        if words[0] in ("rotate", "flip") and len(words) == 2 and held is None:
            direction = words[1] if words[0] == "rotate" else _FLIP_DIRS.get(words[1], words[1])
            obj = self._nearest_block(env_idx, uv, exclude=None)
            ori = self._face(env_idx, obj)
            if ori is None:
                # Resting on no known face (e.g. tipped by a failed flip): no target orientation.
                logger.warning(
                    f"env {env_idx}: {obj} is in no known orientation; cannot {words[0]}"
                )
                return None, {}
            ori2 = self._next_face.get((direction, ori))
            if ori2 is None:
                logger.warning(f"env {env_idx}: no {direction} {words[0]} from {ori}")
                return None, {}
            return SkillCall(words[0], (obj, direction, ori, ori2)), {}
        return None, {}

    def _held(self, env_idx: int) -> str | None:
        j = int(self._ctx.held[env_idx])
        return self._blocks[j] if j >= 0 else None

    def _release(self, env_idx: int) -> None:
        """Open the gripper and drop whatever it holds from the planning world."""
        t = torch.tensor([env_idx], dtype=torch.long, device=self._ctx.device)
        self.sim.command_gripper(torch.zeros(1, dtype=torch.bool, device=self._ctx.device), t)
        self.sim.flush()
        self.geo.detach(t)
        self._ctx.held[env_idx] = -1

    def _nearest_block(self, env_idx: int, uv: Any, exclude: str | None) -> str | None:
        """The block whose projected center is nearest pixel ``uv`` (ignoring ``exclude``),
        if within :data:`_POINT_PX` of it."""
        names = [n for n in self._blocks if n != exclude]
        target = torch.as_tensor(uv, dtype=torch.float32, device=self._base_env.device)
        dists = []
        for name in names:
            pos = self._base_env.scene.rigid_objects[name].data.root_pos_w
            dists.append(
                float(torch.linalg.norm(front_camera_pixels(self._base_env, pos)[env_idx] - target))
            )
        best = min(range(len(names)), key=dists.__getitem__)
        return names[best] if dists[best] <= _POINT_PX else None

    def _points_at_target(self, env_idx: int, lang: str, uv: Any) -> bool:
        """Whether Molmo2's point is on what its command acts on: the table for a table
        place, else a block (for a place, one other than the held block)."""
        if uv is None:
            return False
        if lang == "place on table":
            from blocksuite.utils.obs_utils import cube_corner_points

            corners = cube_corner_points(resolve_asset(self._base_env, _TABLE))[env_idx]
            origin = self._base_env.scene.env_origins[env_idx]
            point = (
                table_points(self._base_env, env_idx, uv, float(corners[:, 2].max() - origin[2]))[0]
                + origin
            )
            lo, hi = corners.amin(dim=0), corners.amax(dim=0)
            return bool((point[:2] >= lo[:2]).all() and (point[:2] <= hi[:2]).all())
        return (
            self._nearest_block(env_idx, uv, self._held(env_idx) if lang == "place" else None)
            is not None
        )

    def _top_of(self, env_idx: int, block: str, held: str) -> str:
        """The top of the stack ``block`` is in: a place on a covered block goes on top of it."""
        for _ in self._blocks:
            above = next(
                (
                    b
                    for b in self._blocks
                    if b not in (block, held) and bool(goal_on(self._base_env, (b, block))[env_idx])
                ),
                None,
            )
            if above is None:
                return block
            block = above
        return block

    def _face(self, env_idx: int, obj: str) -> str | None:
        """The face ``obj`` rests on: its side up, at the nearest of that side's four yaws.

        ``ori_of``'s yaw tolerance leaves gaps between them, but a block turned into one still
        has a face to flip or rotate from (the skills correct for its actual yaw).
        """
        q = resolve_root_pose_w(self._base_env, obj)[env_idx : env_idx + 1, 3:7]
        for face, q_face in ORI_QUATERNIONS.items():
            q_face = torch.tensor([q_face], device=q.device, dtype=q.dtype)
            if bool(_quat_close_batched(q, q_face, yaw_tol_deg=45.0)[0]):
                return face
        return None

    # -- outcomes, homing, gains ---------------------------------------------------------

    def _take_results(self) -> None:
        for env_idx, (success, reason) in list(self.engine.results.items()):
            call = self._calls.pop(env_idx, None)
            if success:
                if call is not None and call.name == "flip":
                    # vecgen's flip pushes with a closed gripper and ends closed, holding nothing.
                    # Open it: run_eval counts a flip (and the episode goal) only once released.
                    self._release(env_idx)
                if call is not None and call.name == "pick":
                    self._finished.append(env_idx)  # holding it: Molmo2 says where it goes
                elif self._go_home_after:
                    self._homing[env_idx] = True  # asks Molmo2 again once home
                else:
                    self._settling[env_idx] = _SETTLE_STEPS
            else:
                logger.warning(f"env {env_idx}: command {call} failed: {reason}")
                if "INVALID_START_STATE" in reason:
                    self._lift_first.add(env_idx)
                self._go_home(env_idx)
        self.engine.results.clear()

    def _go_home(self, env_idx: int) -> None:
        """After a failed command: walk the arm home with an empty hand before the next one.

        An arm still holding a block stays where it is, block in hand, and Molmo2 is asked
        again from there, after lifting it 5 cm if its plan started in collision (replanning
        from the same state fails the same way). A gripper that is open (or holds nothing
        the oracle knows of) is opened and cleared.
        """
        self.failures[env_idx] += 1
        if (
            self._held(env_idx) is not None
            and float(self.sim._gripper_target[env_idx, 0]) > _CLOSED_TARGET
        ):
            lift = self._lifted_joints(env_idx) if env_idx in self._lift_first else None
            self._lift_first.discard(env_idx)
            if lift is None:
                self._finished.append(env_idx)
            else:
                self._goal[env_idx] = lift
                self._homing[env_idx] = True
            return
        self._release(env_idx)
        self._homing[env_idx] = True

    def _lifted_joints(self, env_idx: int) -> Tensor | None:
        """Arm joints with the gripper 5 cm above where it is, or None without an IK solution."""
        ids = torch.tensor([env_idx], dtype=torch.long, device=self._ctx.device)
        goal = self.sim.ee_pose()[ids].clone()
        goal[:, 2] += 0.05
        self.geo.update_worlds(self._ctx.block_poses(), self._ctx.table_pose(), env_ids=ids)
        q_now = self.sim.joint_pos()[ids]
        ok, q_goal, _perr, _rerr = self.geo.ik(ids, goal, seeds=q_now)
        return q_goal[0].to(self._home.device) if bool(ok[0]) else None

    def _walk_home(self) -> None:
        if not bool(self._homing.any()):
            return
        art = self.sim.art
        ids = self._homing.nonzero(as_tuple=False).view(-1).tolist()
        env_ids = torch.tensor(ids, device=self._base_env.device)
        target = art.data.joint_pos_target[env_ids][:, self.sim.arm_ids]
        step = (self._goal[env_ids] - target).clamp(-self._home_max_step, self._home_max_step)
        art.set_joint_position_target(target + step, joint_ids=self.sim.arm_ids, env_ids=env_ids)
        arm_pos = art.data.joint_pos[env_ids][:, self.sim.arm_ids]
        arrived = (arm_pos - self._goal[env_ids]).abs().amax(-1) < self._home_tol
        for i, done in zip(ids, arrived.tolist(), strict=True):
            if done:
                self._homing[i] = False
                self._goal[i] = self._home[i]
                self._finished.append(i)

    def _apply_due_profiles(self) -> None:
        """The skills' stiffness profile on envs that just started or reset (the reset writes
        the episode's recorded gains; run_eval sends the oracle no pd_mode)."""
        due = [i for i in self._profile_due if not self.engine.busy(i)]
        if due:
            self.sim.apply_profile(
                _PROFILE, torch.tensor(due, dtype=torch.long, device=self._ctx.device)
            )
            self._profile_due.difference_update(due)
