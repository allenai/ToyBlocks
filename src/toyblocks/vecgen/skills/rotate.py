"""Rotate (scripted grasp-and-twist) skill, reusing the RotateTrajectory helpers (ported from TAMP).

Sequence (per sub-step; a rotate splits into an align correction plus the 90 deg step
when the block's spawn yaw pushes the total past 95 deg): descend onto the block, close
and attach, lift / twist joint 7 / lower, release, retreat (plus an inter-stage
clearance lift). The start pose and its IK check follow PreRLPoseGenerator.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from toyblocks.vecgen.engine import EngineContext
from toyblocks.vecgen.math import quat_mul, quat_yaw
from toyblocks.vecgen.skills._tamp_adapters import SlotRobot, SlotSim, run_per_env
from toyblocks.vecgen.skills.base import PhaseRequest, Skill
from toyblocks.vecgen.skills.rotate_trajectory import RotateTrajectory
from toyblocks.vecgen.symbolic.predicates import (
    AXIS_TOL_DEG,
    ORI_QUATERNIONS,
    check_quat,
    ori_index,
)

START_HEIGHT = 0.2048
START_OFFSETS = [(x, y) for x in (-0.01, 0.0, 0.01) for y in (-0.01, 0.0, 0.01)]
START_BASE_QUAT = (0.0, -0.7071, 0.7071, 0.0)
GRIP_TICKS = RotateTrajectory.ROTATE_GRIPPER_CLOSE_WAIT_STEPS // 4  # 5
OPEN_TICKS = RotateTrajectory.ROTATE_GRIPPER_OPEN_WAIT_STEPS // 4  # 5
SETTLE_TICKS = 3
MAX_SUBSTEPS = 2
# The wrist (joint 7: kp 50, 12 Nm in the medium profile) trails large joint-7 moves by up to
# ~45 deg, and every gripper/hold phase and next segment re-bases the arm target on the MEASURED
# joints, so lag left at a phase end is lost. Rotate therefore holds (a) the pre-grasp start pose
# and (b) the twisted pose until every joint is within WRIST_SETTLE_TOL (at most
# WRIST_SETTLE_MAX_TICKS): (a) skewed grasps, (b) released blocks 20-40 deg short of the target.
WRIST_SETTLE_TOL = math.radians(2.0)
WRIST_SETTLE_MAX_TICKS = 75  # 3 s
# A rotate counts as done only within this of the target yaw (the shared ori_of tolerance, 40 deg,
# accepted those short rotates); anything wider fails the effect check and is retried. 15, not
# tighter: joint 7 parks ~7.5 deg short of its target (friction deadband), and 10 deg failed
# rotates that could never get closer.
VERIFY_YAW_TOL_DEG = 15.0


def _substep_phases(s: int) -> list[str]:
    return [
        f"descend_{s}",
        f"close_{s}",
        f"twist_{s}",
        f"lower_{s}",
        f"release_{s}",
        f"retreat_{s}",
        f"settle_{s}",
    ]


class Rotate(Skill):
    """``rotate(block, direction, ori, ori2)``."""

    name = "rotate"
    phases = ["open", "start", *_substep_phases(0), *_substep_phases(1)]
    async_phases = frozenset({"descend_0", "descend_1"})

    def __init__(self) -> None:
        self.executor = RotateTrajectory("rotate_traj")

    # ------------------------------------------------------------------ begin
    def begin(self, ctx: EngineContext, env_ids: Tensor, args: list[tuple[str, ...]]) -> Tensor:
        b = int(env_ids.numel())
        dev = ctx.device
        ar = torch.arange(b, device=dev)
        j = torch.tensor([ctx.block_index[a[0]] for a in args], dtype=torch.long, device=dev)
        poses = ctx.block_poses()[env_ids]
        blk = poses[ar, j]
        q_now = ctx.sim.joint_pos()[env_ids]
        # start pose: 0.2048 above the block, gripper yawed by the block's residual yaw
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
        fsim = SlotSim(ctx.sim)
        for i, e in enumerate(env_ids.tolist()):
            blk_name, d, o, o2 = args[i]
            target = ORI_QUATERNIONS.to(dev)[ori_index(o2)]
            subs = self.executor._rotate_sub_targets(fsim, {"obj": blk_name, "ori": o}, e, target)
            ctx.params[e].update(
                {
                    "block": int(j[i]),
                    "obj": blk_name,
                    "direction": d,
                    "ori2": o2,
                    "start_pose": chosen[i].clone(),
                    "subs": [s.to(dev) for s in subs][:MAX_SUBSTEPS],
                }
            )
        return ok

    def skips(self, ctx: EngineContext, phase: str, env: int) -> bool:
        """Sub-step 1's phases are no-ops for a rotate that was not split (one sub-target):
        skipped, they record no idle frames after the retreat."""
        if phase in ("open", "start"):
            return False
        return len(ctx.params[env]["subs"]) <= int(phase.rsplit("_", 1)[1])

    # ---------------------------------------------------------------- request
    def request(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> PhaseRequest:
        envs = env_ids.tolist()
        if phase == "open":
            return PhaseRequest("gripper", "init_gripper_open_traj", closed=False, ticks=6)
        if phase == "start":
            # The descend anchors take the gripper's yaw from the CURRENT pose: arriving with the
            # wrist still lagging (~15-40 deg) grasps the block skewed and the fingers twist it.
            goals = torch.stack([ctx.params[e]["start_pose"] for e in envs])
            return PhaseRequest(
                "plan",
                "rotate_start_traj",
                goals=goals,
                hold=WRIST_SETTLE_MAX_TICKS,
                settle_tol=WRIST_SETTLE_TOL,
            )
        kind, s = phase.rsplit("_", 1)
        s = int(s)
        if kind == "descend":
            rows, ok = self._build_substep(ctx, env_ids, s)
            return PhaseRequest("rows", "rotate_traj", rows=rows, ok=ok)
        if kind == "close":
            return PhaseRequest("gripper", "rotate_traj", closed=True, ticks=GRIP_TICKS)
        if kind in ("twist", "lower", "retreat"):
            key = f"_{kind}_rows"
            rows = [ctx.params[e].pop(key, None) for e in envs]
            ok = torch.tensor([r is not None and r.shape[0] > 0 for r in rows], device=ctx.device)
            rows = [r if r is not None else torch.zeros(0, ctx.geo.dof) for r in rows]
            label = "rotate_retract_traj" if kind == "retreat" else "rotate_traj"
            if kind == "twist":  # hold the twisted pose until the wrist has caught up
                return PhaseRequest(
                    "rows",
                    label,
                    rows=rows,
                    ok=ok,
                    hold=WRIST_SETTLE_MAX_TICKS,
                    settle_tol=WRIST_SETTLE_TOL,
                )
            return PhaseRequest("rows", label, rows=rows, ok=ok)
        if kind == "release":
            return PhaseRequest("gripper", "rotate_traj", closed=False, ticks=OPEN_TICKS)
        if kind == "settle":
            return PhaseRequest("hold", "rotate_retract_traj", ticks=SETTLE_TICKS)
        raise ValueError(phase)

    def _build_substep(
        self, ctx: EngineContext, env_ids: Tensor, s: int
    ) -> tuple[list[Tensor], Tensor]:
        envs = env_ids.tolist()
        k = len(ctx.block_names)
        excl = torch.zeros(ctx.sim.num_envs, k, dtype=torch.bool, device=ctx.device)
        for e in envs:
            excl[e, ctx.params[e]["block"]] = True
        ctx.geo.update_worlds(ctx.block_poses(), ctx.table_pose(), exclude=excl, env_ids=env_ids)
        fsim = SlotSim(ctx.sim)
        q_all = ctx.sim.joint_pos()

        def build(e: int) -> Tensor | None:
            ex = RotateTrajectory("rotate_traj")
            p = ctx.params[e]
            subs = p["subs"]
            if len(subs) <= s:  # this env has no sub-step s: a 1-row no-op
                return q_all[e : e + 1].clone()
            is_final = s == len(subs) - 1
            robot = SlotRobot(ctx.sim, ctx.geo, e)
            gv = {"obj": p["obj"], "direction": p["direction"]}
            current_q = q_all[e].detach().clone()
            anchors = ex._build_scripted_rotate_anchors(
                robot,
                fsim,
                gv,
                e,
                subs[s],
                current_q,
                fallback_direction=p["direction"] if is_final else None,
                need_clearance=not is_final,
            )
            if anchors is None:
                return None
            descend: list[Tensor] = []
            ex._append_segment(
                descend, current_q, anchors["q_grasp"], ex.ROTATE_DESCENT_DURATION, robot
            )
            twist: list[Tensor] = []
            ex._append_segment(
                twist, anchors["q_grasp"], anchors["q_lift"], ex.ROTATE_LIFT_DURATION, robot
            )
            ex._append_segment(
                twist, anchors["q_lift"], anchors["q_lift_rot"], ex.ROTATE_TWIST_DURATION, robot
            )
            ex._append_hold(twist, anchors["q_lift_rot"], ex.ROTATE_TWIST_HOLD_DURATION)
            # lowering is its own phase so it starts only once the wrist has converged
            lower: list[Tensor] = []
            ex._append_segment(
                lower,
                anchors["q_lift_rot"],
                anchors["q_grasp_rot"],
                ex.ROTATE_LOWER_DURATION,
                robot,
            )
            retreat: list[Tensor] = []
            ex._append_segment(
                retreat,
                anchors["q_grasp_rot"],
                anchors["q_lift_rot"],
                ex.ROTATE_RETREAT_DURATION,
                robot,
            )
            q_clear = anchors.get("q_clear_rot")
            if not is_final and q_clear is not None:
                ex._append_segment(
                    retreat,
                    anchors["q_lift_rot"],
                    q_clear,
                    ex.ROTATE_INTERSTAGE_LIFT_DURATION,
                    robot,
                )
                ex._append_hold(retreat, q_clear, ex.ROTATE_INTERSTAGE_HOLD_DURATION)

            def to_rows(ts: list[Tensor]) -> Tensor:
                return torch.stack([x.reshape(-1)[: ctx.geo.dof] for x in ts]).to(ctx.device)

            p["_twist_rows"] = to_rows(twist)
            p["_lower_rows"] = to_rows(lower)
            p["_retreat_rows"] = to_rows(retreat)
            return to_rows(descend) if descend else q_all[e : e + 1].clone()

        built = run_per_env(ctx.geo, envs, build)
        rows: list[Tensor] = []
        ok: list[bool] = []
        for r in built:
            rows.append(r if r is not None else torch.zeros(0, ctx.geo.dof))
            ok.append(r is not None)
        return rows, torch.tensor(ok, dtype=torch.bool, device=ctx.device)

    def after_phase(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> None:
        kind = phase.rsplit("_", 1)[0]
        if kind == "close":
            envs = [e for e in env_ids.tolist() if ctx.params[e].get("_twist_rows") is not None]
            if not envs:
                return
            ids = torch.tensor(envs, dtype=torch.long, device=ctx.device)
            j = torch.tensor(
                [ctx.params[e]["block"] for e in envs], dtype=torch.long, device=ctx.device
            )
            poses = ctx.block_poses()[ids]
            blk = poses[torch.arange(len(envs), device=ctx.device), j]
            ctx.geo.attach(ids, j, ctx.sim.ee_pose()[ids], blk)
            ctx.held[ids] = j
        elif kind == "release":
            ctx.geo.detach(env_ids)
            ctx.held[env_ids] = -1

    def verify(self, ctx: EngineContext, env_ids: Tensor) -> Tensor:
        envs = env_ids.tolist()
        j = torch.tensor(
            [ctx.params[e]["block"] for e in envs], dtype=torch.long, device=ctx.device
        )
        idx = torch.tensor([ori_index(ctx.params[e]["ori2"]) for e in envs], device=ctx.device)
        poses = ctx.block_poses()[env_ids]
        q = poses[torch.arange(len(envs), device=ctx.device), j, 3:7]
        target = ORI_QUATERNIONS.to(ctx.device)[idx]
        return check_quat(q, target, VERIFY_YAW_TOL_DEG, AXIS_TOL_DEG)

    def hl_name(self) -> str:
        return "rotate"
