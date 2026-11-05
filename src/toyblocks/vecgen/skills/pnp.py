"""Pick and place as batched keyframe generators.

Geometry follows the TAMP generators/executors (see docs/datagen_revamp.md and the
executor spec): top-down grasps with the EE 0.14 m above the top face, a 5 cm post-grasp
lift by IK interpolation, transport to a 3 cm pre-place standoff with XY-padded obstacles,
an IK-interpolated descent to 1 mm above the support, release, and a short retreat.
Block sizes are the spawned colliders' edge lengths (``VecSim.measure_block_dims``); heights
and footprints are their world extents in the block's current or placed orientation.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from toyblocks.tamp.blocks.camera_view import blocks_in_front_view
from toyblocks.vecgen.engine import EngineContext
from toyblocks.vecgen.math import (
    pose_compose,
    pose_inverse,
    quat_apply,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    quat_yaw,
    world_extents,
)
from toyblocks.vecgen.skills.base import PhaseRequest, Skill
from toyblocks.vecgen.symbolic.derived import (
    FLIP_DIRECTION_XY,
    FLIP_DIRS,
    FLIP_LANDING_SHIFT,
    SKILL_AWARE_SAMPLE_FACTOR,
    relocation_scores,
)
from toyblocks.vecgen.symbolic.predicates import (
    ORI_QUATERNIONS,
    classify_orientation,
    on,
    on_table,
    ori_index,
)

GRASP_CLEARANCE = 0.14  # EE (robotiq base link) above the top face
GRASP_NEIGHBOR_CLEARANCE = 0.03  # preferred robot-to-neighbour-block gap at the grasp pose
POST_GRASP_LIFT = 0.05
PLACE_Z_GAP = 0.001  # release the block 1 mm above the support (block top or table top)
PRE_PLACE_STANDOFF = 0.03
PLACE_FINGER_CLEARANCE = 0.03
TABLE_PLACE_X = (-0.2, 0.13)
TABLE_PLACE_Y = (-0.25, 0.25)
TABLE_PLACE_SAMPLES = 50
SIDE_X_RANGES = ((0.15, 0.25), (0.55, 0.63))  # MoveToSidePlacePoseGenerator
SIDE_BAND_WIDTH = 0.05
SIDE_CLEARANCE = 0.13
TABLE_HALF_WIDTH_Y = 0.4
MAX_REACH_RADIUS = 0.65
IK_ROWS = 20  # 0.8 s descents / lifts
PLACE_SETTLE = 10
GRIP_TICKS = 6
RETREAT_ROWS = 10
RETREAT_HEIGHT = 0.05
LIFT_VERIFY_MIN = 0.02


def _grasp_heading(q_block: Tensor) -> Tensor:
    """Yaw of the block's dominant horizontal axis (GraspPoseGenerator heading rule)."""
    x = quat_apply(
        q_block, torch.tensor([1.0, 0, 0], device=q_block.device).expand_as(q_block[:, :3])
    )
    y = quat_apply(
        q_block, torch.tensor([0, 1.0, 0], device=q_block.device).expand_as(q_block[:, :3])
    )
    use_x = x[:, :2].norm(dim=1) >= y[:, :2].norm(dim=1)
    hx = torch.atan2(x[:, 1], x[:, 0])
    hy = torch.atan2(y[:, 1], y[:, 0]) - math.pi / 2
    return torch.where(use_x, hx, hy)


def grasp_candidates(block_pose: Tensor, block_size: Tensor, ee_pose: Tensor) -> Tensor:
    """Four top-down grasps ``[B, 4, 7]`` ordered by rotation distance from the current EE."""
    b = block_pose.shape[0]
    dev = block_pose.device
    heading = _grasp_heading(block_pose[:, 3:7])
    delta = quat_yaw(heading)  # [B, 4]
    cands = []
    for i in range(4):
        off = quat_from_euler_xyz(
            torch.zeros(b, device=dev),
            torch.full((b,), math.pi, device=dev),
            torch.full((b,), i * math.pi / 2, device=dev),
        )
        q = quat_mul(delta, off)
        pos = block_pose[:, :3].clone()
        pos[:, 2] += world_extents(block_pose[:, 3:7], block_size)[:, 2] / 2 + GRASP_CLEARANCE
        cands.append(torch.cat([pos, q], dim=1))
    c = torch.stack(cands, dim=1)  # [B, 4, 7]
    dot = (c[:, :, 3:7] * ee_pose[:, None, 3:7]).sum(-1).abs().clamp(max=1.0)
    order = torch.argsort(2.0 * torch.acos(dot), dim=1)
    return torch.gather(c, 1, order[:, :, None].expand(-1, -1, 7))


class Pick(Skill):
    name = "pick"
    phases = ["open", "grasp", "close", "lift"]

    def begin(self, ctx: EngineContext, env_ids: Tensor, args: list[tuple[str, ...]]) -> Tensor:
        b = int(env_ids.numel())
        dev = ctx.device
        j = torch.tensor([ctx.block_index[a[0]] for a in args], dtype=torch.long, device=dev)
        poses = ctx.block_poses()[env_ids]  # [B, K, 7]
        ar = torch.arange(b, device=dev)
        blk = poses[ar, j]
        size = ctx.sizes[env_ids, j]
        ee = ctx.sim.ee_pose()[env_ids]
        q_now = ctx.sim.joint_pos()[env_ids]
        cands = grasp_candidates(blk, size, ee)  # [B, 4, 7]
        ctx.geo.update_worlds(ctx.block_poses(), ctx.table_pose(), env_ids=env_ids)
        # the grasped block and the blocks of its stack (below it) are not neighbours
        ignore = (poses[:, :, :2] - blk[:, None, :2]).norm(dim=-1) < 0.6 * size[:, None, 0]
        # IK + robot-to-neighbour clearance of every candidate; take the first in rotation
        # order with GRASP_NEIGHBOR_CLEARANCE, else the clearest (the IK collision term is a
        # soft cost, so grasps with the fingers against a neighbour block pass it)
        clear = torch.full((b, 4), float("-inf"), device=dev)
        for c in range(4):
            ok_c, q_c, perr, rerr = ctx.geo.ik(env_ids, cands[:, c], seeds=q_now)
            acc = ok_c | ((perr <= 0.005) & (rerr <= 0.05))
            gap = ctx.geo.block_clearance(env_ids, q_c, ignore)
            clear[:, c] = torch.where(acc, gap, clear[:, c])
        ok = (clear > float("-inf")).any(dim=1)
        enough = clear >= GRASP_NEIGHBOR_CLEARANCE
        first_enough = torch.argmax(enough.int(), dim=1)  # first True in rotation order
        pick = torch.where(enough.any(dim=1), first_enough, clear.argmax(dim=1))
        chosen = cands[ar, pick]
        for i, e in enumerate(env_ids.tolist()):
            ctx.params[e].update(
                {
                    "block": int(j[i]),
                    "grasp_pose": chosen[i].clone(),
                    "z0": float(blk[i, 2]),
                }
            )
        return ok

    def request(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> PhaseRequest:
        envs = env_ids.tolist()
        k = len(ctx.block_names)
        if phase == "open":
            return PhaseRequest("gripper", "init_gripper_open_traj", closed=False, ticks=GRIP_TICKS)
        if phase == "grasp":
            goals = torch.stack([ctx.params[e]["grasp_pose"] for e in envs])
            return PhaseRequest("plan", "grasp_traj", goals=goals)
        if phase == "close":
            return PhaseRequest("gripper", "gripper_close_traj", closed=True, ticks=GRIP_TICKS)
        if phase == "lift":
            goals = torch.stack([ctx.params[e]["grasp_pose"] for e in envs]).clone()
            goals[:, 2] += POST_GRASP_LIFT
            excl = torch.zeros(len(envs), k, dtype=torch.bool, device=ctx.device)
            for i, e in enumerate(envs):
                excl[i, ctx.params[e]["block"]] = True
            return PhaseRequest(
                "ik_interp", "grasp_delta_traj", goals=goals, ticks=IK_ROWS, exclude=excl
            )
        raise ValueError(phase)

    def after_phase(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> None:
        if phase == "close":
            envs = env_ids.tolist()
            j = torch.tensor(
                [ctx.params[e]["block"] for e in envs], dtype=torch.long, device=ctx.device
            )
            poses = ctx.block_poses()[env_ids]
            blk = poses[torch.arange(len(envs), device=ctx.device), j]
            ee = ctx.sim.ee_pose()[env_ids]
            ctx.geo.attach(env_ids, j, ee, blk)
            ctx.held[env_ids] = j

    def verify(self, ctx: EngineContext, env_ids: Tensor) -> Tensor:
        envs = env_ids.tolist()
        j = torch.tensor(
            [ctx.params[e]["block"] for e in envs], dtype=torch.long, device=ctx.device
        )
        poses = ctx.block_poses()[env_ids]
        blk = poses[torch.arange(len(envs), device=ctx.device), j]
        ee = ctx.sim.ee_pose()[env_ids]
        z0 = torch.tensor([ctx.params[e]["z0"] for e in envs], device=ctx.device)
        lifted = blk[:, 2] - z0 >= LIFT_VERIFY_MIN
        near = (blk[:, :2] - ee[:, :2]).norm(dim=1) <= 0.04
        return lifted & near


class Place(Skill):
    """``place_on(obj, target, ori)`` and ``place_table(obj, ori)``; ``mode`` selects."""

    phases = ["transport", "lower", "release", "retreat"]

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.name = {"on": "place_on", "table": "place_table", "side": "place_side"}[mode]
        self.stats: dict[str, int] = {
            "places": 0,
            "ranked": 0,
            "all_satisfied": 0,
            "plain_would_satisfy": 0,
        }

    def begin(self, ctx: EngineContext, env_ids: Tensor, args: list[tuple[str, ...]]) -> Tensor:
        b = int(env_ids.numel())
        dev = ctx.device
        ar = torch.arange(b, device=dev)
        j = torch.tensor([ctx.block_index[a[0]] for a in args], dtype=torch.long, device=dev)
        poses = ctx.block_poses()[env_ids]
        sizes = ctx.sizes[env_ids]
        blk = poses[ar, j]
        ee = ctx.sim.ee_pose()[env_ids]
        held_ok = ctx.held[env_ids] == j
        in_hand = pose_compose(pose_inverse(blk), ee)  # T_obj^-1 T_ee
        ori_names = [a[-1] for a in args]
        if self.mode == "on":
            k = torch.tensor([ctx.block_index[a[1]] for a in args], dtype=torch.long, device=dev)
            tgt = poses[ar, k]
            tgt_cls = classify_orientation(tgt[:, 3:7])
            pert = quat_mul(tgt[:, 3:7], quat_inverse(ORI_QUATERNIONS.to(dev)[tgt_cls]))
            goal_idx = torch.tensor(
                [
                    ori_index(o) if o else int(classify_orientation(blk[i : i + 1, 3:7]))
                    for i, o in enumerate(ori_names)
                ],
                dtype=torch.long,
                device=dev,
            )
            q_put = quat_mul(pert, ORI_QUATERNIONS.to(dev)[goal_idx])
            p_put = tgt[:, :3].clone()
            h_support = world_extents(tgt[:, 3:7], sizes[ar, k])[:, 2]
            h_held = world_extents(q_put, sizes[ar, j])[:, 2]
            p_put[:, 2] += h_support / 2 + h_held / 2 + PLACE_Z_GAP
            put = torch.cat([p_put, q_put], dim=1)
            ok = held_ok
            for i, e in enumerate(env_ids.tolist()):
                ctx.params[e]["target"] = int(k[i])
        else:
            table = ctx.table_pose()[env_ids]
            goal_idx = torch.tensor(
                [
                    ori_index(o) if o else int(classify_orientation(blk[i : i + 1, 3:7]))
                    for i, o in enumerate(ori_names)
                ],
                dtype=torch.long,
                device=dev,
            )
            q_put = ORI_QUATERNIONS.to(dev)[goal_idx]
            put = torch.zeros(b, 7, device=dev)
            put[:, 3:7] = q_put
            ext_held = world_extents(q_put, sizes[ar, j])  # [B, 3] as placed
            r_held = 0.5 * ext_held[:, :2].norm(dim=1)
            r_all = 0.5 * world_extents(poses[..., 3:7], sizes)[..., :2].norm(dim=2)  # [B, K]
            pending = self._pending_skills(ctx, env_ids, j, args, poses)
            ranked = pending["ranked"]
            n_c = TABLE_PLACE_SAMPLES * (SKILL_AWARE_SAMPLE_FACTOR if bool(ranked.any()) else 1)
            u = torch.rand(b, n_c, 2, generator=ctx.rng).to(dev)
            table_xy = table[:, None, :2]
            if self.mode == "side":
                # out of the way: the outermost 5 cm strip on either side of the table,
                # x in one of two bands (MoveToSidePlacePoseGenerator)
                pick = torch.rand(b, n_c, 2, generator=ctx.rng).to(dev)
                first = pick[..., 0] < 0.5
                lo = torch.where(first, SIDE_X_RANGES[0][0], SIDE_X_RANGES[1][0])
                hi = torch.where(first, SIDE_X_RANGES[0][1], SIDE_X_RANGES[1][1])
                x = lo + u[..., 0] * (hi - lo)
                side_hi = (TABLE_HALF_WIDTH_Y - r_held)[:, None]
                side_lo = side_hi - SIDE_BAND_WIDTH
                sign = torch.where(pick[..., 1] < 0.5, -1.0, 1.0)
                y = table_xy[..., 1] + sign * (side_lo + u[..., 1] * (side_hi - side_lo))
            else:
                x = (
                    table_xy[..., 0]
                    + TABLE_PLACE_X[0]
                    + u[..., 0] * (TABLE_PLACE_X[1] - TABLE_PLACE_X[0])
                )
                y = (
                    table_xy[..., 1]
                    + TABLE_PLACE_Y[0]
                    + u[..., 1] * (TABLE_PLACE_Y[1] - TABLE_PLACE_Y[0])
                )
            # block centre: table collider top (Geometry.table_top) + h/2 + the release gap
            top = table[:, 2] + ctx.geo.table_top[env_ids]
            z = (top + ext_held[:, 2] / 2 + PLACE_Z_GAP)[:, None].expand(b, n_c)
            pos = torch.stack([x, y, z], dim=2)  # [B, C, 3]
            reach = torch.hypot(x, y) <= MAX_REACH_RADIUS
            dxy = (poses[:, None, :, :2] - pos[:, :, None, :2]).abs()  # [B, C, K, 2]
            if self.mode == "side":
                clr = torch.full_like(dxy, SIDE_CLEARANCE)
            else:
                clr = (r_held[:, None] + r_all + PLACE_FINGER_CLEARANCE)[:, None, :, None]
            hit = (dxy < clr).all(dim=3) & pending["others"][:, None, :]
            free = ~hit.any(dim=2)
            view = (
                blocks_in_front_view(pos.reshape(-1, 3), ext_held[:, 0].repeat_interleave(n_c))
                .to(dev)
                .view(b, n_c)
            )
            feasible = reach & free & view & held_ok[:, None]
            # envs with nothing pending keep the plain sampler (first feasible of 50)
            feasible[~ranked, TABLE_PLACE_SAMPLES:] = False
            score = torch.zeros(b, n_c, device=dev)
            if bool(ranked.any()):
                score = relocation_scores(
                    pos,
                    ext_held[:, 2],
                    poses[..., :3],
                    pending["others"],
                    pending["own_flip"],
                    pending["own_rotate"],
                    pending["oth_flip_landing"],
                    pending["oth_flip_dir"],
                    pending["oth_flip_valid"],
                    pending["oth_rot_pos"],
                    pending["oth_rot_valid"],
                )
                score = torch.where(ranked[:, None], score, torch.zeros_like(score))
            # highest score first, sample order breaks ties (ranked, never filtered)
            idx = torch.arange(n_c, device=dev, dtype=score.dtype)
            key = torch.where(feasible, score * (n_c + 1) - idx, torch.full_like(score, -1e9))
            best = key.argmax(dim=1)
            ok = feasible.any(dim=1)
            put[ok, :3] = pos[ar, best][ok]
            # what the plain sampler would have taken: the first feasible of the first 50
            plain = feasible[:, :TABLE_PLACE_SAMPLES]
            first = torch.where(plain.any(dim=1), plain.float().argmax(dim=1), best)
            self._record_stats(pending, score[ar, best], ok, score[ar, first])
        ee_place = pose_compose(put, in_hand)
        ee_pre = ee_place.clone()
        ee_pre[:, 2] += PRE_PLACE_STANDOFF - (PLACE_Z_GAP if self.mode == "on" else 0.0)
        for i, e in enumerate(env_ids.tolist()):
            ctx.params[e].update(
                {"block": int(j[i]), "ee_place": ee_place[i].clone(), "ee_pre": ee_pre[i].clone()}
            )
        return ok

    def _pending_skills(
        self,
        ctx: EngineContext,
        env_ids: Tensor,
        j: Tensor,
        args: list[tuple[str, ...]],
        poses: Tensor,
    ) -> dict[str, Tensor]:
        """What the rest of each env's program needs from this placement.

        Scans the skills after this place until the held block is picked again: its own
        flips (by direction) and rotate, which the new spot should allow (naman/dev
        02b9a693 reads the same requirement from the HL facts that made FF move the
        block), and the flips and rotates of other blocks that have not moved by then,
        which the new spot should not obstruct.
        """
        b, k = int(env_ids.numel()), len(ctx.block_names)
        dev = ctx.device
        own_flip = torch.zeros(b, len(FLIP_DIRS), dtype=torch.bool, device=dev)
        own_rotate = torch.zeros(b, dtype=torch.bool, device=dev)
        oth_flips: list[list[tuple[int, int]]] = [[] for _ in range(b)]
        oth_rots: list[list[int]] = [[] for _ in range(b)]
        enabled = bool(ctx.states)
        for i, e in enumerate(env_ids.tolist()):
            if not enabled:
                break
            st = ctx.states[e]
            if st.program is None:
                continue
            held = args[i][0]
            moved: set[str] = set()
            for sc in st.program.skills[st.pc + 1 :]:
                obj = sc.args[0] if sc.args else ""
                if sc.name == "pick":
                    if obj == held:
                        break
                    moved.add(obj)
                elif sc.name == "flip" and len(sc.args) >= 2 and sc.args[1] in FLIP_DIRS:
                    di = FLIP_DIRS.index(sc.args[1])
                    if obj == held:
                        own_flip[i, di] = True
                    elif obj not in moved and obj in ctx.block_index:
                        oth_flips[i].append((ctx.block_index[obj], di))
                elif sc.name == "rotate":
                    if obj == held:
                        own_rotate[i] = True
                    elif obj not in moved and obj in ctx.block_index:
                        oth_rots[i].append(ctx.block_index[obj])
        m = max([1] + [len(x) for x in oth_flips])
        r = max([1] + [len(x) for x in oth_rots])
        oth_flip_landing = torch.zeros(b, m, 3, device=dev)
        oth_flip_dir = torch.zeros(b, m, dtype=torch.long, device=dev)
        oth_flip_valid = torch.zeros(b, m, dtype=torch.bool, device=dev)
        oth_rot_pos = torch.zeros(b, r, 3, device=dev)
        oth_rot_valid = torch.zeros(b, r, dtype=torch.bool, device=dev)
        for i in range(b):
            for n, (bk, di) in enumerate(oth_flips[i]):
                dx, dy = FLIP_DIRECTION_XY[FLIP_DIRS[di]]
                oth_flip_landing[i, n] = poses[i, bk, :3] + FLIP_LANDING_SHIFT * torch.tensor(
                    [dx, dy, 0.0], device=dev
                )
                oth_flip_dir[i, n] = di
                oth_flip_valid[i, n] = True
            for n, bk in enumerate(oth_rots[i]):
                oth_rot_pos[i, n] = poses[i, bk, :3]
                oth_rot_valid[i, n] = True
        others = torch.ones(b, k, dtype=torch.bool, device=dev)
        others[torch.arange(b, device=dev), j] = False
        ranked = (
            own_flip.any(dim=1) | own_rotate | oth_flip_valid.any(dim=1) | oth_rot_valid.any(dim=1)
        )
        return {
            "ranked": ranked,
            "others": others,
            "own_flip": own_flip,
            "own_rotate": own_rotate,
            "oth_flip_landing": oth_flip_landing,
            "oth_flip_dir": oth_flip_dir,
            "oth_flip_valid": oth_flip_valid,
            "oth_rot_pos": oth_rot_pos,
            "oth_rot_valid": oth_rot_valid,
        }

    def _record_stats(
        self, pending: dict[str, Tensor], best: Tensor, ok: Tensor, plain: Tensor
    ) -> None:
        """Counters for the run summary: how often a place was ranked, whether the chosen
        spot allows every own flip/rotate and obstructs nothing that comes next, and whether
        the plain sampler's first feasible spot would have (same candidates, same run)."""
        ranked = pending["ranked"] & ok
        need = pending["own_flip"].sum(dim=1).float() + pending["own_rotate"].float()
        self.stats["places"] += int(ok.sum())
        self.stats["ranked"] += int(ranked.sum())
        self.stats["all_satisfied"] += int((ranked & (best >= need)).sum())
        self.stats["plain_would_satisfy"] += int((ranked & (plain >= need)).sum())

    def request(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> PhaseRequest:
        envs = env_ids.tolist()
        k = len(ctx.block_names)
        if phase == "transport":
            goals = torch.stack([ctx.params[e]["ee_pre"] for e in envs])
            excl = torch.zeros(len(envs), k, dtype=torch.bool, device=ctx.device)
            for i, e in enumerate(envs):
                excl[i, ctx.params[e]["block"]] = True
            return PhaseRequest("plan", "place_traj", goals=goals, exclude=excl, pad_fraction=0.2)
        if phase == "lower":
            goals = torch.stack([ctx.params[e]["ee_place"] for e in envs])
            excl = torch.ones(len(envs), k, dtype=torch.bool, device=ctx.device)
            return PhaseRequest(
                "ik_interp",
                "place_delta_traj",
                goals=goals,
                ticks=IK_ROWS,
                hold=PLACE_SETTLE,
                exclude=excl,
            )
        if phase == "release":
            return PhaseRequest("gripper", "gripper_open_traj", closed=False, ticks=GRIP_TICKS)
        if phase == "retreat":
            ee = ctx.sim.ee_pose()[env_ids].clone()
            ee[:, 2] += RETREAT_HEIGHT
            excl = torch.ones(len(envs), k, dtype=torch.bool, device=ctx.device)
            return PhaseRequest(
                "ik_interp", "lift_traj", goals=ee, ticks=RETREAT_ROWS, exclude=excl
            )
        raise ValueError(phase)

    def after_phase(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> None:
        if phase == "release":
            ctx.geo.detach(env_ids)
            ctx.held[env_ids] = -1

    def verify(self, ctx: EngineContext, env_ids: Tensor) -> Tensor:
        envs = env_ids.tolist()
        dev = ctx.device
        ar = torch.arange(len(envs), device=dev)
        j = torch.tensor([ctx.params[e]["block"] for e in envs], dtype=torch.long, device=dev)
        poses = ctx.block_poses()[env_ids]
        blk = poses[ar, j]
        if self.mode == "on":
            k = torch.tensor([ctx.params[e]["target"] for e in envs], dtype=torch.long, device=dev)
            return on(blk, poses[ar, k])
        return on_table(blk, ctx.sizes[env_ids, j], ctx.table_pose()[env_ids])
