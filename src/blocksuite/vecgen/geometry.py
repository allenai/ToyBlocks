"""Batched geometry on the cuRobo v2 planner pool: per-env worlds, IK, plans, attachments.

All poses are ``[.., 7]`` = xyz + wxyz in the robot-base frame (= env-local frame).
Blocks are cuboids; the table is a thick slab whose top is 1 cm above the table root
(the sim2real 2 cm slab is centred on its root).

The pool has ``pool_slots`` planner slots (each with its own collision world and
attached-object spheres). Envs are mapped onto slots per call: a request for ``B`` envs
is solved in chunks of ``pool_slots`` rows, loading each env's world and attachment into
the slot it is assigned for that call. With ``pool_slots == num_envs`` the mapping is the
identity; with fewer slots the pool's memory stays bounded while the env count grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from curobo._src.types.pose import Pose
from torch import Tensor

from blocksuite.vecgen.math import quat_apply, quat_inverse
from blocksuite.vecgen.planner import CuroboPlannerPool

ATTACH_MARGIN = 0.004  # cuRobo-side shrink of the held cuboid (CuroboRobot.ATTACHED_OBJECT_MARGIN)
ATTACH_Z_OFFSET = 0.015  # held spheres lifted along world z (robot.py attach pose_offset)
TABLE_TOP_ABOVE_ROOT = 0.01  # default; run.py sets Geometry.table_top from the table collider
# The sim2real slab is 0.6 (x) by 0.8 (y) after its -90 deg yaw; a wider cuboid would reach
# back under the robot base and flag every start state as colliding.
TABLE_DIMS = (0.6, 0.8, 0.2)
DEFAULT_POOL_SLOTS = 32


@dataclass
class PlanResult:
    """One env's transport plan at the pool's interpolation dt (0.04 s = one tick)."""

    ok: bool
    position: Tensor | None  # [T, 7]
    velocity: Tensor | None
    acceleration: Tensor | None
    status: str = ""


class _SlotMap:
    """Env -> slot assignment for one pool: which env's world / attachment each slot holds.

    Threads own slot *regions* (contiguous ranges) so concurrent callers do not evict each
    other's worlds (``set_world`` costs ~3 ms per slot): the main thread gets region 0 and
    keeps half the slots when there are several regions; other threads are assigned a
    region round-robin on first use.
    """

    def __init__(self, pool: CuroboPlannerPool, n_slots: int, regions: int) -> None:
        self.pool = pool
        self.n_slots = n_slots
        self.regions = max(1, min(regions, n_slots))
        self._region_of: dict[int, int] = {}
        self.slot_world: list[int] = [-1] * n_slots
        self.slot_attach: list[int] = [-1] * n_slots

    def invalidate_worlds(self, envs: list[int]) -> None:
        stale = set(envs)
        for s in range(self.n_slots):
            if self.slot_world[s] in stale:
                self.slot_world[s] = -1

    def invalidate_attach(self, env: int) -> None:
        for s in range(self.n_slots):
            if self.slot_attach[s] == env:
                self.slot_attach[s] = -1

    def region_slots(self) -> list[int]:
        import threading

        if self.regions == 1:
            return list(range(self.n_slots))
        tid = threading.get_ident()
        if tid == threading.main_thread().ident:
            r = 0
        else:
            r = self._region_of.get(tid)
            if r is None:
                r = 1 + len(self._region_of) % (self.regions - 1)
                self._region_of[tid] = r
        main = max(1, self.n_slots // 2)
        if r == 0:
            return list(range(main))
        width = max(1, (self.n_slots - main) // (self.regions - 1))
        lo = main + (r - 1) * width
        hi = self.n_slots if r == self.regions - 1 else lo + width
        return list(range(lo, min(hi, self.n_slots)))

    def chunks(self, envs: list[int]) -> list[list[int]]:
        w = len(self.region_slots())
        return [envs[i : i + w] for i in range(0, len(envs), w)]

    def load(self, envs: list[int], geo: Geometry) -> list[int]:
        """Assign ``envs`` (one chunk) to this thread's slots and load their worlds (one
        tensor write for the chunk) and attachments; an env repeated in ``envs`` occupies
        several slots."""
        free = self.region_slots()
        if len(envs) > len(free):
            raise ValueError(f"{len(envs)} rows for {len(free)} slots")
        slots: list[int] = []
        # keep an env on a slot that already holds its world when possible
        for e in envs:
            s = next((s for s in free if self.slot_world[s] == e), None)
            if s is None:
                s = next(s for s in free if s not in slots)
            free.remove(s)
            slots.append(s)
        stale = [(e, s) for e, s in zip(envs, slots, strict=True) if self.slot_world[s] != e]
        if stale:
            env_t = torch.tensor([e for e, _ in stale], device=geo.device)
            if not bool(geo._w_set[env_t].all()):
                raise RuntimeError("an env has no collision world set (call update_worlds)")
            self.pool.set_worlds_tensor(
                [s for _, s in stale], geo._w_dims[env_t], geo._w_inv[env_t], geo._w_enable[env_t]
            )
            for e, s in stale:
                self.slot_world[s] = e
        stale = [(e, s) for e, s in zip(envs, slots, strict=True) if self.slot_attach[s] != e]
        if stale:
            self.pool.set_attachments_tensor(
                [s for _, s in stale], [geo._attach[e] for e, _ in stale]
            )
            for e, s in stale:
                self.slot_attach[s] = e
        return slots


class Geometry:
    """One planner pool; worlds and attachments kept per env, loaded into slots per call."""

    def __init__(
        self,
        num_envs: int,
        block_names: list[str],
        block_sizes: Tensor,
        device: torch.device,
        robotiq: bool = True,
        random_seed: int | None = None,
        pool_slots: int | None = None,
        regions: int = 1,
        build_slots: int = 0,
        build_regions: int = 1,
        ik_seeds: int | None = None,
        trajopt_seeds: int | None = None,
        build_ik_seeds: int | None = None,
    ) -> None:
        self.num_envs = num_envs
        self.block_names = block_names
        self.sizes = block_sizes.to(device)  # [N, K, 3]
        self.device = device
        # table collider top above the table body origin [N] (VecSim.measure_table_top)
        self.table_top = torch.full((num_envs,), TABLE_TOP_ABOVE_ROOT, device=device)
        # every pool call solves all slots (IK ~4 ms at 16 slots, 8 ms at 64; trajopt 95 ms
        # at 16, 285 ms at 64), so the pool stays no wider than the env count
        self.pool_slots = min(num_envs, pool_slots or DEFAULT_POOL_SLOTS)

        def make_pool(n: int, n_ik_seeds: int | None) -> CuroboPlannerPool:
            return CuroboPlannerPool(
                num_envs=n,
                collision_cache={"cuboid": max(30, len(block_names) + 1), "mesh": 10},
                num_ik_seeds=n_ik_seeds,
                num_trajopt_seeds=trajopt_seeds,
                robotiq=robotiq,
                device=str(device),
                batch_wait_s=0.0,
                plan_wait_s=0.0,
                batch_attempts=2,
                single_fallback=True,
                warmup=True,
                random_seed=random_seed,
            )

        # the main pool serves the tick thread (IK, batched requests) and the planner
        # thread (transport plans); skill builds on worker threads get their own pool so
        # their many-row IK searches neither widen the main pool nor evict its worlds
        self.pool = make_pool(self.pool_slots, ik_seeds)
        self._main = _SlotMap(self.pool, self.pool_slots, regions)
        self.build_pool: CuroboPlannerPool | None = None
        self._build: _SlotMap | None = None
        if build_slots > 0:
            self.build_pool = make_pool(build_slots, build_ik_seeds)
            self._build = _SlotMap(self.build_pool, build_slots, build_regions)
        self._main_threads: set[int] = set()
        self.dof = self.pool.dof
        self.interpolation_dt = self.pool.interpolation_dt
        k = len(block_names)
        self._all_ids = torch.arange(num_envs, device=device)
        # per-env cuboid worlds: K blocks + the table slab (last row)
        self._w_dims = torch.zeros(num_envs, k + 1, 3, device=device)
        self._w_inv = torch.zeros(num_envs, k + 1, 7, device=device)
        self._w_enable = torch.zeros(num_envs, k + 1, dtype=torch.bool, device=device)
        self._w_set = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self._attach: list[Tensor | None] = [None] * num_envs
        self._last_q: list[Tensor | None] = [None] * num_envs

    def mark_main_thread(self) -> None:
        """Route the calling thread to the main pool (the engine's planner thread)."""
        import threading

        self._main_threads.add(threading.get_ident())

    def _map(self) -> _SlotMap:
        import threading

        if self._build is None:
            return self._main
        tid = threading.get_ident()
        if tid == threading.main_thread().ident or tid in self._main_threads:
            return self._main
        return self._build

    def region_slots(self) -> list[int]:
        """Slots the calling thread may fill in one call."""
        return self._map().region_slots()

    # ---------------------------------------------------------------- worlds
    def update_worlds(
        self,
        block_poses: Tensor,
        table_pose: Tensor,
        exclude: Tensor | None = None,
        pad_fraction: float = 0.0,
        no_pad: Tensor | None = None,
        env_ids: Tensor | None = None,
    ) -> None:
        """Rebuild the collision worlds of ``env_ids`` (default: every env) from current poses.

        ``block_poses`` ``[N, K, 7]``, ``table_pose`` ``[N, 7]``. ``exclude`` ``[N, K]`` bool
        drops a block from its env's world (held or intended-contact objects);
        ``pad_fraction`` scales the XY dims of every block by ``1 + 2 * pad`` except those
        flagged in ``no_pad`` ``[N, K]``. The table is never padded. Worlds are kept as
        tensors (dims, inverse poses, enable flags) and written into pool slots in one
        tensor copy when the env is next solved.
        """
        ids = self._all_ids if env_ids is None else env_ids
        n = int(ids.numel())
        k = block_poses.shape[1]
        poses = block_poses[ids]  # [n, K, 7]
        dims = self.sizes[ids].clone()  # [n, K, 3]
        scale = 1.0 + 2.0 * float(pad_fraction)
        if scale != 1.0:
            keep = (
                no_pad[ids]
                if no_pad is not None
                else torch.zeros(n, k, dtype=torch.bool, device=self.device)
            )
            dims[..., :2] = torch.where(keep[..., None], dims[..., :2], dims[..., :2] * scale)
        enable = (
            ~exclude[ids]
            if exclude is not None
            else torch.ones(n, k, dtype=torch.bool, device=self.device)
        )
        table = table_pose[ids]
        t_pose = torch.zeros(n, 1, 7, device=self.device)
        t_pose[:, 0, :2] = table[:, :2]
        t_pose[:, 0, 2] = table[:, 2] + self.table_top[ids] - TABLE_DIMS[2] / 2
        t_pose[:, 0, 3] = 1.0
        all_pose = torch.cat([poses, t_pose], dim=1)  # [n, K+1, 7]
        all_dims = torch.cat(
            [dims, torch.tensor(TABLE_DIMS, device=self.device).expand(n, 1, 3)], dim=1
        )
        all_enable = torch.cat(
            [enable, torch.ones(n, 1, dtype=torch.bool, device=self.device)], dim=1
        )
        q_inv = quat_inverse(all_pose[..., 3:7].reshape(-1, 4))
        p_inv = -quat_apply(q_inv, all_pose[..., :3].reshape(-1, 3))
        inv = torch.cat([p_inv, q_inv], dim=1).view(n, k + 1, 7)
        self._w_dims[ids] = all_dims
        self._w_inv[ids] = inv
        self._w_enable[ids] = all_enable
        self._w_set[ids] = True
        for m in self._maps():  # the slots' copies are stale now
            m.invalidate_worlds(ids.tolist())

    # ----------------------------------------------------------- attachments
    def attach(
        self, env_ids: Tensor, block_idx: Tensor, ee_pose: Tensor, block_pose: Tensor
    ) -> None:
        """Attach block ``block_idx[i]`` to env ``env_ids[i]`` given the grasp-time poses.

        Spheres are fitted in the block frame (dims shrunk by ``ATTACH_MARGIN``), lifted by
        ``ATTACH_Z_OFFSET`` in the base frame and expressed in the end-effector frame.
        """
        for i, e in enumerate(env_ids.tolist()):
            dims = (self.sizes[e, int(block_idx[i])] - ATTACH_MARGIN).clamp(min=0.005)
            spheres = self.pool.fit_cuboid_spheres(tuple(dims.tolist()))  # [n, 4] block frame
            centers = spheres[:, :3]
            p_base = (
                quat_apply(block_pose[i, 3:7].expand(len(centers), 4), centers) + block_pose[i, :3]
            )
            p_base = p_base + torch.tensor([0.0, 0.0, ATTACH_Z_OFFSET], device=p_base.device)
            q_inv = quat_inverse(ee_pose[i, 3:7]).expand(len(centers), 4)
            p_link = quat_apply(q_inv, p_base - ee_pose[i, :3])
            self.set_attachment_of(e, torch.cat([p_link, spheres[:, 3:4]], dim=1))

    def detach(self, env_ids: Tensor) -> None:
        for e in env_ids.tolist():
            self.set_attachment_of(e, None)

    def attachment_of(self, env: int) -> Tensor | None:
        """The env's attached-object spheres (link frame) or None."""
        return self._attach[env]

    def set_attachment_of(self, env: int, spheres: Tensor | None) -> None:
        self._attach[env] = None if spheres is None else spheres.detach().to(self.device)
        for m in self._maps():
            m.invalidate_attach(env)

    def _maps(self) -> list[_SlotMap]:
        return [self._main] if self._build is None else [self._main, self._build]

    # ------------------------------------------------------------------- IK
    def ik_result(self, env: int, pose: Pose, seed: Tensor | None) -> Any:
        """Pool IK result object for one env (used by the scripted-executor adapters)."""
        m = self._map()
        (slot,) = m.load([env], self)
        seed_q = seed if seed is not None else self._last_q[env]
        return m.pool.solve_ik_rows([(slot, pose, seed_q)])[slot]

    def ik(
        self, env_ids: Tensor, goal_poses: Tensor, seeds: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Batched IK: returns ``(ok [B], q [B, dof], pos_err [B], rot_err [B])``."""
        envs = env_ids.tolist()
        b = len(envs)
        ok = torch.zeros(b, dtype=torch.bool, device=self.device)
        q = torch.zeros(b, self.dof, device=self.device)
        perr = torch.full((b,), float("inf"), device=self.device)
        rerr = torch.full((b,), float("inf"), device=self.device)
        pos = 0
        m = self._map()
        for chunk in m.chunks(envs):
            slots = m.load(chunk, self)
            rows = []
            for i, (e, s) in enumerate(zip(chunk, slots, strict=True)):
                k = pos + i
                pose = Pose(
                    position=goal_poses[k : k + 1, :3], quaternion=goal_poses[k : k + 1, 3:7]
                )
                seed = seeds[k] if seeds is not None else self._last_q[e]
                rows.append((s, pose, seed))
            res = m.pool.solve_ik_rows(rows)
            for i, s in enumerate(slots):
                r = res[s]
                k = pos + i
                ok[k] = bool(r.success.view(-1)[0])
                q[k] = r.js_solution.position.view(-1)[: self.dof]
                if r.position_error is not None:
                    perr[k] = r.position_error.view(-1)[0]
                if r.rotation_error is not None:
                    rerr[k] = r.rotation_error.view(-1)[0]
            pos += len(chunk)
        return ok, q, perr, rerr

    def states_valid(self, env_ids: Tensor, q: Tensor, check_world: bool = True) -> Tensor:
        envs = env_ids.tolist()
        out: list[bool] = []
        pos = 0
        m = self._map()
        for chunk in m.chunks(envs):
            slots = m.load(chunk, self)
            out += m.pool.states_valid(slots, q[pos : pos + len(chunk)], check_world)
            pos += len(chunk)
        return torch.tensor(out, dtype=torch.bool, device=self.device)

    # ---------------------------------------------------------------- plans
    def plan(self, env_ids: Tensor, start_q: Tensor, goal_poses: Tensor) -> list[PlanResult]:
        """Batched transport plans from ``start_q`` to ``goal_poses`` in each env's world."""
        envs = env_ids.tolist()
        out: list[PlanResult] = []
        pos = 0
        m = self._map()
        for chunk in m.chunks(envs):
            slots = m.load(chunk, self)
            rows = []
            for i, s in enumerate(slots):
                k = pos + i
                pose = Pose(
                    position=goal_poses[k : k + 1, :3], quaternion=goal_poses[k : k + 1, 3:7]
                )
                rows.append((s, start_q[k], pose, None))
            res = m.pool.plan_pose_rows(rows)
            for s in slots:
                r = res[s]
                ok = bool(r.success.view(-1)[0]) and r.interpolated_plan is not None
                if not ok:
                    out.append(PlanResult(False, None, None, None, str(getattr(r, "status", ""))))
                    continue
                plan = r.interpolated_plan
                out.append(
                    PlanResult(
                        True,
                        plan.position[..., : self.dof].reshape(-1, self.dof),
                        None
                        if plan.velocity is None
                        else plan.velocity[..., : self.dof].reshape(-1, self.dof),
                        None
                        if plan.acceleration is None
                        else plan.acceleration[..., : self.dof].reshape(-1, self.dof),
                        "SUCCESS",
                    )
                )
            pos += len(chunk)
        return out

    # ------------------------------------------------------------------- FK
    def fk(self, q: Tensor, env_idx: int = 0) -> Tensor:
        """End-effector poses ``[B, 7]`` for joint positions ``[B, dof]`` (no attachments)."""
        pose, _ = self._map().pool.kinematics_state(q, env_idx=0)
        return torch.cat([pose.position.view(-1, 3), pose.quaternion.view(-1, 4)], dim=1)

    def block_clearance(self, env_ids: Tensor, q: Tensor, ignore: Tensor) -> Tensor:
        """Smallest gap ``[B]`` (m) between the robot's collision spheres at ``q`` ``[B, dof]``
        and the blocks of each env's current world (:meth:`update_worlds`), skipping blocks
        flagged in ``ignore`` ``[B, K]`` and the table; negative when a sphere penetrates."""
        pool = self._map().pool
        _, spheres = pool.kinematics_state(q, env_idx=0)
        spheres = spheres.reshape(q.shape[0], -1, 4)
        robot = torch.ones(spheres.shape[1], dtype=torch.bool, device=spheres.device)
        robot[pool._attached_idx.to(spheres.device).flatten()] = False  # slot 0's held object
        sph = spheres[:, robot & (spheres[0, :, 3] > 0)].to(self.device)  # [B, S, 4]
        k = ignore.shape[1]
        dims = self._w_dims[env_ids, :k]  # [B, K, 3]
        inv = self._w_inv[env_ids, :k]  # [B, K, 7]
        b, s = sph.shape[:2]
        centers = sph[:, None, :, :3].expand(b, k, s, 3).reshape(-1, 3)
        q_inv = inv[:, :, None, 3:7].expand(b, k, s, 4).reshape(-1, 4)
        p_inv = inv[:, :, None, :3].expand(b, k, s, 3).reshape(-1, 3)
        local = (quat_apply(q_inv, centers) + p_inv).view(b, k, s, 3)
        d = local.abs() - dims[:, :, None, :] / 2
        dist = d.clamp(min=0).norm(dim=-1) + d.max(dim=-1).values.clamp(max=0)
        dist = dist - sph[:, None, :, 3]
        skip = ignore | ~self._w_enable[env_ids, :k]
        dist = dist.masked_fill(skip[:, :, None], float("inf"))
        return dist.flatten(1).min(dim=1).values

    def remember_q(self, env_ids: Tensor, q: Tensor) -> None:
        """Seed future unseeded solves of these envs from ``q``."""
        for i, e in enumerate(env_ids.tolist()):
            self._last_q[e] = q[i].detach().clone()
