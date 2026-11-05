"""Batched anchor search for the scripted flip (same search as ReOrientTrajectory).

``ReOrientTrajectory._scripted_joint_targets_from_ik`` solves, per env and in sequence,
the IK anchors edge-align / contact / 12-step drag / retreat for every candidate
(pull scale x gripper yaw x edge seed) and keeps the cheapest: ~15 IK calls per
candidate, ~48 per flip. Here all candidates of all envs are solved in three padded pool
rounds per pull scale (edge, contact, then drag + retreat seeded by the contact solution),
each round packed to the pool's slot count, so a flip costs a handful of IK rounds. The
waypoint geometry, costs, timing rule and row assembly reuse the executor's methods and
constants.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import torch
from torch import Tensor

from blocksuite.vecgen.math import quat_apply
from blocksuite.vecgen.skills.reorient_trajectory import ReOrientTrajectory as Ex

log = logging.getLogger("vecgen.flip")


def _partial_cost(executor: Ex, q_start: Tensor, q_edge: Tensor, q_contact: Tensor) -> float:
    """The executor's anchor cost terms known before the drag: travel, limit margin, and the
    edge -> contact jump (the drag's own jumps are added once it is solved)."""
    return executor._flip_anchor_cost([q_start, q_edge, q_contact, q_contact, q_contact], None)


def _ik_rows(
    geo: Any, env_ids: Tensor, goals_pos: Tensor, quat: Tensor, seeds: Tensor
) -> tuple[Tensor, Tensor]:
    """Strict-success batched IK (``ok``, clamped ``q``) for EE goals ``goals_pos`` / ``quat``."""
    goals = torch.cat([goals_pos, quat], dim=1)
    ok, q, _perr, _rerr = geo.ik(env_ids, goals, seeds=seeds)
    return ok, Ex._clamp_franka_joints(q)


def _segment_steps(
    geo: Any, q0: Tensor, q1: Tensor, duration_s: float, min_time_s: Tensor
) -> Tensor:
    """Per-env step count for ``q0 -> q1`` (``[B, 7]``): the executor's duration-to-steps,
    stretched so the gripper rotation stays under FLIP_MAX_GRIPPER_ANGULAR_SPEED."""
    b = q0.shape[0]
    n = Ex.FLIP_ROTATION_SAMPLES
    alphas = torch.linspace(0.0, 1.0, n, device=q0.device).view(1, n, 1)
    path = (1.0 - alphas) * q0[:, None, :] + alphas * q1[:, None, :]  # [B, n, 7]
    quats = geo.fk(path.reshape(-1, q0.shape[1]))[:, 3:7].view(b, n, 4)
    dots = (quats[:, 1:] * quats[:, :-1]).sum(-1).abs().clamp(max=1.0)
    rate = (2.0 * torch.acos(dots)).max(dim=1).values * (n - 1)  # rad per unit alpha
    min_time = torch.maximum(min_time_s, rate / Ex.FLIP_MAX_GRIPPER_ANGULAR_SPEED)
    intervals = torch.clamp(torch.ceil(min_time / Ex.FLIP_TARGET_PERIOD - 1e-6), min=1)
    base = max(2, int(round(duration_s / Ex.FLIP_ACTION_DT)))
    return torch.maximum(torch.full_like(intervals, base), intervals + 1).long()


def _interp(q0: Tensor, q1: Tensor, steps: int, skip_first: bool) -> list[Tensor]:
    out = []
    for step in range(steps):
        if skip_first and step == 0:
            continue
        a = step / max(1, steps - 1)
        out.append((1.0 - a) * q0 + a * q1)
    return out


def build_flip_rows_batched(
    ctx: Any,
    executor: Ex,
    fsim: Any,
    env_ids: Tensor,
    objs: list[str],
    directions: list[str],
    attempts: list[int],
    tip_offsets: list[Tensor],
) -> tuple[list[Tensor | None], list[int | None]]:
    """Joint-target rows per env (``None`` when no reachable flip) and the retreat start index."""
    geo = ctx.geo
    dev = ctx.device
    b = int(env_ids.numel())
    envs = env_ids.tolist()
    q_now = ctx.sim.joint_pos()[env_ids]
    poses_all = ctx.block_poses()
    obj_pose = torch.stack(
        [poses_all[e, ctx.block_index[o]] for e, o in zip(envs, objs, strict=True)]
    )
    yaw_select = Ex.FLIP_YAW_SELECT_DEFAULT
    no_blind = Ex.FLIP_NO_BLIND_FALLBACK_DEFAULT
    cartesian = Ex.FLIP_CARTESIAN_DRAG_DEFAULT
    pull_scales = Ex.FLIP_PULL_SCALES if no_blind else (1.0,)
    edge_seeds: list[Tensor | None] = [None]
    if cartesian:
        edge_seeds += [torch.tensor(q, device=dev, dtype=torch.float32) for q in Ex.FLIP_EDGE_SEEDS]
    tips = torch.stack([t.to(dev, torch.float32) for t in tip_offsets])  # [B, 3]

    best: list[dict[str, Any] | None] = [None] * b
    stats = {"rounds": 0, "rows": 0, "fallback": 0}
    # Candidates (gripper yaw x edge seed) of one pull scale are solved together: one IK
    # round for every candidate's edge pose, one for the contact pose (seeded by the edge
    # solution), then ONE round for the 12 drag points and the retreat, all seeded by the
    # contact solution. The executor chains those 15 solves per candidate; here the chain
    # is 3 rounds for all candidates of all envs, each round packed up to the pool's slot
    # count (the same env may occupy several slots). Drag continuity is checked afterwards
    # as before; a candidate whose batched drag jumps is re-solved step by step, seeded by
    # its previous step, only when the env has no clean candidate.
    n_drag = Ex.FLIP_DRAG_STEPS if cartesian else 1

    def solve(
        env_rows: list[int], pos: Tensor, quat: Tensor, seeds: Tensor
    ) -> tuple[Tensor, Tensor]:
        ids = env_ids[torch.tensor(env_rows, dtype=torch.long, device=dev)]
        stats["rounds"] += -(-len(env_rows) // max(1, len(geo.region_slots())))
        stats["rows"] += len(env_rows)
        return _ik_rows(geo, ids, pos, quat, seeds)

    for pull_scale in pull_scales:
        todo = [i for i in range(b) if best[i] is None]
        if not todo:
            break
        wps: dict[int, list[Tensor]] = {}
        yaws: dict[int, list[float]] = {}
        for i in todo:
            positions, flip_yaw = executor._build_scripted_flip_waypoint_positions(
                fsim,
                envs[i],
                objs[i],
                obj_pose[i],
                directions[i],
                attempt=attempts[i],
                pull_scale=pull_scale,
            )
            wps[i] = [p.to(dev) for p in positions]
            yaws[i] = [flip_yaw]
            if yaw_select:
                yaws[i].append(flip_yaw - math.pi if flip_yaw > 0.0 else flip_yaw + math.pi)
        # candidate table: one row per (env, yaw, seed)
        cand: list[tuple[int, int, int]] = [
            (i, yaw_k, si)
            for i in todo
            for yaw_k in range(2 if yaw_select else 1)
            for si in range(len(edge_seeds))
        ]
        n = len(cand)
        rows_env = [c[0] for c in cand]
        quat = torch.stack(
            [executor._yawed_flip_gripper_quat(dev, yaws[i][yk]) for i, yk, _ in cand]
        )
        tip_w = quat_apply(quat, tips[torch.tensor(rows_env, device=dev)])
        wp = torch.stack([torch.stack(wps[i]) for i, _, _ in cand])  # [n, 4, 3]
        seed_q = torch.stack(
            [q_now[i] if edge_seeds[si] is None else edge_seeds[si] for i, _, si in cand]  # type: ignore[misc]
        )
        alive = torch.ones(n, dtype=torch.bool, device=dev)
        ok, q_edge = solve(rows_env, wp[:, 0] - tip_w, quat, seed_q)
        alive &= ok
        live = alive.nonzero().flatten().tolist()
        q_contact = torch.zeros_like(q_edge)
        if live:
            sel = torch.tensor(live, device=dev)
            ok, q_c = solve(
                [rows_env[r] for r in live], (wp[:, 1] - tip_w)[sel], quat[sel], q_edge[sel]
            )
            alive[sel] &= ok
            q_contact[sel] = q_c
        # drag + retreat for each env's cheapest candidate after the contact solve (edge
        # travel, joint-limit margin, edge->contact jump); the next-ranked candidates only
        # for envs still without a smooth one, so a flip usually costs 3 IK rounds
        partial = torch.full((n,), float("inf"), device=dev)
        for r in alive.nonzero().flatten().tolist():
            partial[r] = _partial_cost(executor, q_now[cand[r][0]], q_edge[r], q_contact[r])
        order = torch.argsort(partial).tolist()
        rank: dict[int, int] = {}
        seen: dict[int, int] = {}
        for r in order:
            i = cand[r][0]
            rank[r] = seen.get(i, 0)
            seen[i] = rank[r] + 1
        drag_q = torch.zeros(n, n_drag, q_edge.shape[1], device=dev)
        q_ret = torch.zeros_like(q_edge)
        solved_env: dict[int, bool] = {i: False for i in todo}
        n_rank = max(seen.values(), default=0)
        waves = [[r for r in order if bool(alive[r]) and rank[r] == k] for k in range(n_rank)]
        for wave in waves:
            live = [r for r in wave if not solved_env[rows_env[r]]]
            if not live:
                continue
            # drag points k = 1..n_drag then the retreat, all seeded by the contact solution
            goals: list[Tensor] = []
            for k in range(1, n_drag + 1):
                goals.append(
                    wp[:, 1] + (wp[:, 2] - wp[:, 1]) * (k / n_drag) if cartesian else wp[:, 2]
                )
            goals.append(wp[:, 3])
            g = torch.stack(goals, dim=1) - tip_w[:, None, :]  # [n, n_drag + 1, 3]
            sel = torch.tensor(live, device=dev)
            m = len(live)
            flat_env = [rows_env[r] for r in live for _ in range(n_drag + 1)]
            ok, q_all = solve(
                flat_env,
                g[sel].reshape(-1, 3),
                quat[sel].repeat_interleave(n_drag + 1, dim=0),
                q_contact[sel].repeat_interleave(n_drag + 1, dim=0),
            )
            ok = ok.view(m, n_drag + 1)
            q_all = q_all.view(m, n_drag + 1, -1)
            alive[sel] &= ok.all(dim=1)
            drag_q[sel] = q_all[:, :n_drag]
            q_ret[sel] = q_all[:, n_drag]
            if cartesian:
                chain = torch.cat([q_contact[sel][:, None], q_all[:, :n_drag]], dim=1)
                jumps = (chain[:, 1:] - chain[:, :-1]).abs().amax(dim=2).amax(dim=1)
                smooth = jumps <= Ex.FLIP_DRAG_MAX_STEP_RAD
                # envs with no smooth candidate: re-solve the drag step by step (the
                # executor's chain) for their candidates whose anchors solved
                env_smooth = {i: False for i in todo}
                for r, sm in zip(live, smooth.tolist(), strict=True):
                    if sm and bool(alive[r]):
                        env_smooth[rows_env[r]] = True
                redo = [
                    r
                    for r, sm in zip(live, smooth.tolist(), strict=True)
                    if not sm and bool(alive[r]) and not env_smooth[rows_env[r]]
                ]
                alive[sel[~smooth]] = False
                if redo:
                    stats["fallback"] += len(redo)
                    rsel = torch.tensor(redo, device=dev)
                    prev = q_contact[rsel]
                    good = torch.ones(len(redo), dtype=torch.bool, device=dev)
                    for k in range(1, n_drag + 1):
                        pos_k = (wp[:, 1] + (wp[:, 2] - wp[:, 1]) * (k / n_drag) - tip_w)[rsel]
                        ok_k, q_k = solve([rows_env[r] for r in redo], pos_k, quat[rsel], prev)
                        good &= ok_k & (
                            (q_k - prev).abs().max(dim=1).values <= Ex.FLIP_DRAG_MAX_STEP_RAD
                        )
                        drag_q[rsel, k - 1] = q_k
                        prev = q_k
                    ok_r, q_r = solve(
                        [rows_env[r] for r in redo], (wp[:, 3] - tip_w)[rsel], quat[rsel], prev
                    )
                    good &= ok_r
                    q_ret[rsel] = q_r
                    alive[rsel] = good
            for r in live:
                if bool(alive[r]):
                    solved_env[rows_env[r]] = True
        reached = {r for wave in waves for r in wave}
        for r, (i, yaw_k, si) in enumerate(cand):
            if r not in reached or not bool(alive[r]) or not bool(drag_q[r].abs().sum() > 0):
                continue
            path = [drag_q[r, k] for k in range(n_drag)] if cartesian else None
            pull = drag_q[r, n_drag - 1]
            anc = [q_now[i], q_edge[r], q_contact[r], pull, q_ret[r]]
            cost = executor._flip_anchor_cost(anc, path)
            cur = best[i]
            if cur is None or cost < cur["cost"]:
                best[i] = {
                    "cost": cost,
                    "anchors": anc,
                    "drag": path,
                    "wps": wps[i],
                    "yaw": yaws[i][yaw_k],
                    "seed": si,
                    "pull_scale": pull_scale,
                }
    log.info(
        "flip search: %d envs, %d IK rounds (%d rows), %d step-by-step fallbacks, %d solved",
        b,
        stats["rounds"],
        stats["rows"],
        stats["fallback"],
        sum(x is not None for x in best),
    )

    # ---- rows (the executor's segment timing, batched FK for the rotation-rate stretch)
    solved = [i for i in range(b) if best[i] is not None]
    rows: list[Tensor | None] = [None] * b
    retreat_start: list[int | None] = [None] * b
    if not solved:
        return rows, retreat_start
    anc = [
        torch.stack([best[i]["anchors"][k] for i in solved]) for k in range(5)
    ]  # start, edge, contact, pull, retreat
    side = (
        torch.tensor(
            [
                1.0
                if attempts[i] == 0
                and directions[i] in ("left_flip", "left", "right_flip", "right")
                else 0.0
                for i in solved
            ],
            device=dev,
        )
        * Ex.FLIP_SIDE_ROTATE_MIN_TIME
    )
    zero = torch.zeros(len(solved), device=dev)
    n_free = _segment_steps(geo, anc[0], anc[1], Ex.FLIP_FREE_MOTION_DURATION, side)
    n_desc = _segment_steps(geo, anc[1], anc[2], Ex.FLIP_CONTACT_DESCENT_DURATION, zero)
    n_pull = _segment_steps(geo, anc[2], anc[3], Ex.FLIP_PULL_DURATION, zero)
    n_ret = _segment_steps(geo, anc[3], anc[4], Ex.FLIP_RETREAT_DURATION, zero)
    hover = max(
        2,
        int(
            round(
                Ex.FLIP_HOVER_SETTLE_DURATION
                / Ex.FLIP_ACTION_DT
            )
        ),
    )
    pull_hold = max(2, int(round(Ex.FLIP_PULL_HOLD_DURATION / Ex.FLIP_ACTION_DT)))
    n_drag = max(2, int(round(Ex.FLIP_PULL_DURATION / Ex.FLIP_ACTION_DT)))
    for r, i in enumerate(solved):
        bi = best[i]
        q_start, q_edge, q_contact, q_pull, q_ret = bi["anchors"]
        settle = (
            Ex.FLIP_CONTACT_SETTLE_DURATION
            if attempts[i] == 0
            else Ex.FLIP_RETRY_CONTACT_SETTLE_DURATION
        )
        n_settle = max(2, int(round(settle / Ex.FLIP_ACTION_DT)))
        targets: list[Tensor] = []
        targets += _interp(q_start, q_edge, int(n_free[r]), skip_first=False)
        targets += [q_edge] * hover
        targets += _interp(q_edge, q_contact, int(n_desc[r]), skip_first=True)
        targets += [q_contact] * n_settle
        if bi["drag"]:
            pts = [q_contact, *bi["drag"]]
            for k in range(1, n_drag):
                u = (len(pts) - 1) * k / (n_drag - 1)
                j = min(int(u), len(pts) - 2)
                a = u - j
                targets.append((1.0 - a) * pts[j] + a * pts[j + 1])
        else:
            targets += _interp(q_contact, q_pull, int(n_pull[r]), skip_first=True)
        targets += [q_pull] * pull_hold
        retreat_start[i] = len(targets)
        targets += _interp(q_pull, q_ret, int(n_ret[r]), skip_first=True)
        rows[i] = torch.stack(targets)
    return rows, retreat_start
