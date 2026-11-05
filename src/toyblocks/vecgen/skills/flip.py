"""Reorient (scripted flip) skill, reusing the TAMP executor's flip geometry.

``ReOrientTrajectory`` (ported from the TAMP executor) holds the whole scripted flip
(edge/contact/pull waypoints, tip-offset measurement, IK anchor search with yaw and
seed candidates, drag resampling, retry escalation). Those helpers only need a robot
object with IK/FK and a simulator with scene access, so this skill feeds them thin
adapters over :class:`VecSim` / :class:`Geometry` and turns the returned joint-target
list into one segment per env. The IK calls are per env (not batched); flips are a
small fraction of an episode's ticks, so this is the reuse-first path.

Sequence, as in the old pipeline: plan to the pre-reorient start pose (0.2111 m above
the block, gripper yawed by the block's residual yaw), fully close the gripper, run the
scripted press-and-pull, retract, settle, then check ``ori_of``.
"""

from __future__ import annotations

import logging
import math

import torch
from torch import Tensor

from toyblocks.vecgen.engine import EngineContext
from toyblocks.vecgen.math import quat_mul, quat_yaw
from toyblocks.vecgen.skills._tamp_adapters import SlotRobot, SlotSim
from toyblocks.vecgen.skills.base import PhaseRequest, Skill
from toyblocks.vecgen.skills.flip_batched import build_flip_rows_batched
from toyblocks.vecgen.skills.reorient_trajectory import ReOrientTrajectory
from toyblocks.vecgen.symbolic.predicates import (
    ORI_NAMES,
    classify_orientation,
    ori_index,
    ori_of,
)

log = logging.getLogger("vecgen.flip")
START_HEIGHT = 0.2111  # PreReorientRLPoseGenerator
START_OFFSETS = [(x, y) for x in (-0.01, 0.0, 0.01) for y in (-0.01, 0.0, 0.01)]
START_BASE_QUAT = (0.0, -0.7071, 0.7071, 0.0)
FULL_CLOSE_TICKS = ReOrientTrajectory.FLIP_GRIPPER_FULL_CLOSE_WAIT_STEPS // 4  # 10
SETTLE_TICKS = 3  # NoOpMarker(decimation=12)


class Flip(Skill):
    """``flip(block, direction, ori, ori2)``."""

    name = "flip"
    phases = ["start", "close_full", "flip", "retract", "settle"]
    retry_from_live = True  # the block has moved; rebuild the waypoints from where it is
    async_phases = frozenset({"flip"})

    def __init__(self, profile: str = "medium") -> None:
        """``profile``: the stiffness profile the arm runs at; sizes the flip's press."""
        self.executor = ReOrientTrajectory("reorient_traj")
        self.executor._flip_press_depth = self.executor._press_depth_for_profile(profile)
        self.executor._retreat_start = {}

    def begin(self, ctx: EngineContext, env_ids: Tensor, args: list[tuple[str, ...]]) -> Tensor:
        b = int(env_ids.numel())
        dev = ctx.device
        ar = torch.arange(b, device=dev)
        j = torch.tensor([ctx.block_index[a[0]] for a in args], dtype=torch.long, device=dev)
        blk = ctx.block_poses()[env_ids][ar, j]
        q_now = ctx.sim.joint_pos()[env_ids]
        qz, qw = blk[:, 6], blk[:, 3]
        world_yaw = 2.0 * torch.atan2(qz, qw)
        delta = world_yaw - torch.round(world_yaw / (math.pi / 2)) * (math.pi / 2)
        base = torch.tensor(START_BASE_QUAT, device=dev).expand(b, 4)
        start_q = quat_mul(quat_yaw(delta), base)
        ctx.geo.update_worlds(ctx.block_poses(), ctx.table_pose(), env_ids=env_ids)
        ok = torch.zeros(b, dtype=torch.bool, device=dev)
        chosen = torch.zeros(b, 7, device=dev)
        for dx, dy in START_OFFSETS:
            todo = ~ok
            if not bool(todo.any()):
                break
            sub = ar[todo]
            goal = torch.zeros(int(sub.numel()), 7, device=dev)
            goal[:, 0] = blk[sub, 0] + dx
            goal[:, 1] = blk[sub, 1] + dy
            goal[:, 2] = blk[sub, 2] + START_HEIGHT
            goal[:, 3:7] = start_q[sub]
            ok_c, _q, perr, rerr = ctx.geo.ik(env_ids[sub], goal, seeds=q_now[sub])
            acc = ok_c | ((perr <= 0.005) & (rerr <= 0.05))
            chosen[sub[acc]] = goal[acc]
            ok[sub[acc]] = True
        for i, e in enumerate(env_ids.tolist()):
            name, d, _o, o2 = args[i]
            ctx.params[e].update(
                {
                    "block": int(j[i]),
                    "obj": name,
                    "direction": d,
                    "ori2": o2,
                    "start_pose": chosen[i].clone(),
                }
            )
        return ok

    def request(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> PhaseRequest:
        envs = env_ids.tolist()
        if phase == "start":
            goals = torch.stack([ctx.params[e]["start_pose"] for e in envs])
            return PhaseRequest("plan", "reorient_start_traj", goals=goals)
        if phase == "close_full":
            return PhaseRequest(
                "gripper",
                "reorient_start_traj",
                closed=True,
                gripper_target=ReOrientTrajectory.FLIP_ROBOTIQ_FULL_CLOSE_TARGET,
                ticks=FULL_CLOSE_TICKS,
            )
        if phase == "flip":
            rows, ok = self._build_rows(ctx, env_ids)
            return PhaseRequest("rows", "reorient_traj", rows=rows, ok=ok)
        if phase == "retract":
            rows = [ctx.params[e].pop("_retract_rows", None) for e in envs]
            ok = torch.tensor([r is not None and r.shape[0] > 0 for r in rows], device=ctx.device)
            rows = [r if r is not None else torch.zeros(0, ctx.geo.dof) for r in rows]
            return PhaseRequest("rows", "reorient_retract_traj", rows=rows, ok=ok)
        if phase == "settle":
            return PhaseRequest("hold", "reorient_retract_traj", ticks=SETTLE_TICKS)
        raise ValueError(phase)

    def _build_rows(self, ctx: EngineContext, env_ids: Tensor) -> tuple[list[Tensor], Tensor]:
        envs = env_ids.tolist()
        k = len(ctx.block_names)
        excl = torch.zeros(ctx.sim.num_envs, k, dtype=torch.bool, device=ctx.device)
        for e in envs:
            excl[e, ctx.params[e]["block"]] = True
        ctx.geo.update_worlds(ctx.block_poses(), ctx.table_pose(), exclude=excl, env_ids=env_ids)
        fsim = SlotSim(ctx.sim)
        ex = self.executor
        tips: list[Tensor] = []
        for e in envs:
            robot = SlotRobot(ctx.sim, ctx.geo, e)
            tip = ex._measure_ee_to_tip_offset(robot, e)
            tips.append(
                tip if tip is not None else ex._constant_ee_to_tip_offset(robot, ctx.device)
            )
        built, starts = build_flip_rows_batched(
            ctx,
            ex,
            fsim,
            env_ids,
            [ctx.params[e]["obj"] for e in envs],
            [ctx.params[e]["direction"] for e in envs],
            [int(ctx.params[e].get("_attempt", 0)) for e in envs],
            tips,
        )
        rows: list[Tensor] = []
        ok: list[bool] = []
        for i, e in enumerate(envs):
            r = built[i]
            if r is None:
                rows.append(torch.zeros(0, ctx.geo.dof))
                ok.append(False)
                continue
            start = starts[i]
            if start is None or start <= 0 or start >= r.shape[0]:
                rows.append(r)
                ctx.params[e]["_retract_rows"] = r[-1:].clone()
            else:
                rows.append(r[:start])
                ctx.params[e]["_retract_rows"] = r[start:]
            ok.append(True)
        return rows, torch.tensor(ok, dtype=torch.bool, device=ctx.device)

    def verify(self, ctx: EngineContext, env_ids: Tensor) -> Tensor:
        envs = env_ids.tolist()
        j = torch.tensor(
            [ctx.params[e]["block"] for e in envs], dtype=torch.long, device=ctx.device
        )
        idx = torch.tensor([ori_index(ctx.params[e]["ori2"]) for e in envs], device=ctx.device)
        poses = ctx.block_poses()[env_ids]
        q = poses[torch.arange(len(envs), device=ctx.device), j, 3:7]
        ok = ori_of(q, idx)
        cls = classify_orientation(q)
        for i, e in enumerate(envs):
            log.info(
                "flip verify env %d: target %s, nearest %s, ok=%s",
                e,
                ctx.params[e]["ori2"],
                ORI_NAMES[int(cls[i])],
                bool(ok[i]),
            )
        return ok

    def hl_name(self) -> str:
        return "reorient"
