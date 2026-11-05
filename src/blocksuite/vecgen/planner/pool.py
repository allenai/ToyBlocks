"""Batched cuRobo v2 planner pool for the vecgen engine (ported from TAMP's
``tmp3.Backends.Curobo.coordinator.CuroboPlannerPool``).

The pool owns one ``BatchMotionPlanner`` configured with ``multi_env=True`` and
``max_batch_size=num_envs``: slot ``k`` has its own collision world (cuboid tensors written
by :meth:`CuroboPlannerPool.set_worlds_tensor`) and its own row in the robot's
attached-object sphere table (``link_spheres[k]``). Callers solve IK and pose plans for
several slots as one padded GPU batch (:meth:`CuroboPlannerPool.solve_ik_rows`,
:meth:`CuroboPlannerPool.plan_pose_rows`) and check states in batch
(:meth:`CuroboPlannerPool.states_valid`).

Rows that fail in the batch (which plans with few seeds and no graph search) are retried
on a single-problem, graph-enabled ``MotionPlanner`` loaded with that slot's world and
attachment, mirroring the v1 ``plan_single`` retry/graph behaviour.

TAMP's request loop (instance threads submitting single requests that a coordinator
thread gathers), ``SceneCfg`` world loads and c-space plans are not ported: vecgen batches
its own requests and writes worlds as tensors.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import torch
from curobo._src.collision.collision_robot_scene import RobotSceneCollision
from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_batch import BatchMotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.state.state_joint_trajectory_ops import trim_joint_state_trajectory
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.robot import RobotCfg
from curobo._src.types.tool_pose import GoalToolPose

from blocksuite.vecgen.planner.robot_config import build_robot_cfg_dict
from blocksuite.vecgen.planner.solver_configs import apply_v1_world_collision_configs
from blocksuite.vecgen.planner.types import (
    IKResult,
    MotionGenPlanConfig,
    MotionGenResult,
    MotionGenStatus,
)

ATTACHED_OBJECT_LINK = "attached_object"
#: Radius written into unused attached-object sphere slots (cuRobo ignores negative radii).
DISABLED_SPHERE_RADIUS = -100.0


@dataclass
class _Request:
    kind: str
    env_idx: int
    goal_pose: Pose | None = None
    start_q: torch.Tensor | None = None
    seed_q: torch.Tensor | None = None
    plan_config: MotionGenPlanConfig | None = None
    result: Any = None


@dataclass
class _EnvSlot:
    world: SceneCfg | None = None
    # cuboid world written as tensors (dims [m, 3], inverse pose [m, 7], enable [m]);
    # the SceneCfg is only materialised when the single-planner fallback needs it
    world_tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
    attached_spheres: torch.Tensor | None = None
    last_q: torch.Tensor | None = None


class CuroboPlannerPool:
    """One cuRobo v2 batch planner, ``num_envs`` environment slots, request batching."""

    def __init__(
        self,
        num_envs: int,
        robotiq: bool = True,
        device: str = "cuda:0",
        interpolation_dt: float = 0.04,
        interpolation_buffer_size: int = 1000,
        collision_cache: dict[str, int] | None = None,
        num_attached_spheres: int = 32,
        batch_wait_s: float = 0.025,
        plan_wait_s: float = 0.15,
        batch_attempts: int = 2,
        single_fallback: bool = True,
        single_fallback_max_attempts: int = 10,
        warmup: bool = True,
        robot_cfg_dict: dict[str, Any] | None = None,
        random_seed: int | None = None,
        num_ik_seeds: int | None = None,
        num_trajopt_seeds: int | None = None,
    ) -> None:
        """``robot_cfg_dict`` overrides the robot model built from ``robotiq`` /
        ``num_attached_spheres`` (the Isaac backend passes the dict its single-slot
        ``MotionGen`` was built from, so both planners share one kinematic model).
        ``num_ik_seeds`` / ``num_trajopt_seeds`` override cuRobo's defaults (32 / 4); every
        batch call solves all slots x seeds, so fewer seeds make each call cheaper."""
        self.num_envs = int(num_envs)
        self.robotiq = robotiq
        self.device_str = device
        self.tensor_args = DeviceCfg(device=torch.device(device))
        self.interpolation_dt = interpolation_dt
        self.batch_wait_s = batch_wait_s
        self.plan_wait_s = plan_wait_s
        self.batch_attempts = batch_attempts
        self.single_fallback_max_attempts = single_fallback_max_attempts
        cache = dict(collision_cache or {"cuboid": 30, "mesh": 10})

        robot_dict = (
            robot_cfg_dict
            if robot_cfg_dict is not None
            else build_robot_cfg_dict(robotiq=robotiq, num_attached_spheres=num_attached_spheres)
        )
        self.robot_cfg: RobotCfg = RobotCfg.create(
            robot_dict, self.tensor_args, num_envs=self.num_envs
        )
        seed_kwargs: dict[str, int] = (
            {} if random_seed is None else {"random_seed": int(random_seed)}
        )
        batch_cfg = MotionPlannerCfg.create(
            robot=self.robot_cfg,
            collision_cache=cache,
            device_cfg=self.tensor_args,
            max_batch_size=self.num_envs,
            multi_env=True,
            interpolation_dt=interpolation_dt,
            interpolation_buffer_size=interpolation_buffer_size,
            num_ik_seeds=num_ik_seeds,
            num_trajopt_seeds=num_trajopt_seeds,
            **seed_kwargs,
        )
        if batch_cfg.scene_collision_cfg is not None:
            batch_cfg.scene_collision_cfg.num_envs = self.num_envs
        # Same v1-style soft world-collision IK / trajopt configs as MotionGen (a hard
        # constraint fails contact-adjacent goals that v1 planned and gates batch success).
        apply_v1_world_collision_configs(
            batch_cfg,
            robot=self.robot_cfg,
            device_cfg=self.tensor_args,
            self_collision_check=True,
            random_seed=random_seed,
        )
        self.planner = BatchMotionPlanner(batch_cfg)
        self.joint_names: list[str] = list(self.planner.joint_names)
        self.dof: int = len(self.joint_names)
        self.tool_frame: str = self.planner.tool_frames[0]
        self.default_q: torch.Tensor = self.planner.default_joint_state.position.view(-1)[
            : self.dof
        ].clone()

        self.single_planner: MotionPlanner | None = None
        self._single_cfg: RobotCfg | None = None
        if single_fallback:
            self._single_cfg = RobotCfg.create(robot_dict, self.tensor_args, num_envs=1)
            single_cfg = MotionPlannerCfg.create(
                robot=self._single_cfg,
                collision_cache=cache,
                device_cfg=self.tensor_args,
                interpolation_dt=interpolation_dt,
                interpolation_buffer_size=interpolation_buffer_size,
                **seed_kwargs,
            )
            apply_v1_world_collision_configs(
                single_cfg,
                robot=self._single_cfg,
                device_cfg=self.tensor_args,
                self_collision_check=True,
                random_seed=random_seed,
            )
            self.single_planner = MotionPlanner(single_cfg)

        self._validator = RobotSceneCollision(
            RobotSceneCollisionCfg.load_from_config(
                robot_config=self.robot_cfg,
                device_cfg=self.tensor_args,
                num_envs=self.num_envs,
                scene_collision_checker=self.planner.scene_collision_checker,
            ),
        )
        self._kin_params = self.planner.kinematics.config.kinematics_config
        self._attached_idx: torch.Tensor = self._kin_params.get_sphere_index_from_link_name(
            ATTACHED_OBJECT_LINK
        )
        self._sphere_fit_cache: dict[tuple[float, float, float], torch.Tensor] = {}

        self.slots: list[_EnvSlot] = [
            _EnvSlot(last_q=self.default_q.clone()) for _ in range(self.num_envs)
        ]
        self._gpu_lock = threading.RLock()

        #: Per request kind: [rounds, rows, seconds]; plus fallback counters.
        self.stats: dict[str, list[float]] = {
            "ik": [0, 0, 0.0],
            "plan_pose": [0, 0, 0.0],
        }
        self.fallback_calls = 0
        self.fallback_successes = 0

        if warmup:
            with self._gpu_lock:
                self.planner.warmup()
                if self.single_planner is not None:
                    self.single_planner.warmup()
                # torch loads its linear-algebra backend lazily and that load is not
                # thread safe ("lazy wrapper should be called at most once"); trigger it
                # here before worker threads call torch.inverse concurrently.
                torch.inverse(torch.eye(4, device=self.tensor_args.device))
                self._warm_pose_kernels()
                torch.cuda.synchronize(self.tensor_args.device)
        self.clear_all_attachments()

    def _warm_pose_kernels(self) -> None:
        """Load the Warp kernels behind ``Pose`` math once, from this thread.

        Generators call ``Pose.multiply`` / ``inverse`` / ``from_matrix`` / ``transform_points``
        from their own threads (outside the pool lock); in cuRobo v2 these are Warp kernels
        whose first use compiles/loads a module, which is not safe to race from several
        threads (observed as a segfault in ``warp-clang.so``).
        """
        dev = self.tensor_args.device
        pos = torch.zeros((2, 3), device=dev, dtype=self.tensor_args.dtype)
        quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2, device=dev, dtype=self.tensor_args.dtype)
        a = Pose(position=pos, quaternion=quat)
        b = a.clone()
        a.multiply(b)
        a.inverse()
        mat = a.get_matrix()
        Pose.from_matrix(mat)
        Pose.from_matrix(mat[0])
        a.transform_points(torch.zeros((4, 3), device=dev, dtype=self.tensor_args.dtype))
        a.distance(b)
        a.angular_distance(b)
        a.linear_distance(b)
        a.get_rotation()

    def stats_summary(self) -> str:
        parts = []
        for kind, (rounds, rows, secs) in self.stats.items():
            if rounds:
                parts.append(
                    f"{kind}: {int(rounds)} rounds, {rows / rounds:.1f} rows/round, {1e3 * secs / rounds:.1f} ms/round"
                )
        parts.append(
            f"single-planner fallbacks: {self.fallback_successes}/{self.fallback_calls} succeeded"
        )
        return "; ".join(parts)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def close(self) -> None:
        with self._gpu_lock:
            if self.single_planner is not None:
                self.single_planner.destroy()
                self.single_planner = None
            self.planner.destroy()

    # ------------------------------------------------------------------
    # Per-slot state
    # ------------------------------------------------------------------
    def set_worlds_tensor(
        self,
        env_ids: list[int],
        dims: torch.Tensor,
        inv_pose: torch.Tensor,
        enable: torch.Tensor,
    ) -> None:
        """Write cuboid-only worlds for several slots in one tensor write.

        ``dims`` ``[n, m, 3]`` (full extents), ``inv_pose`` ``[n, m, 7]`` (the obstacle's
        world->object inverse pose as x, y, z, qw, qx, qy, qz) and ``enable`` ``[n, m]``
        (bool) for slots ``env_ids``. ``load_collision_model`` costs ~8 ms of per-cuboid
        Python for a 28-obstacle world; this writes the checker's cuboid tensors directly
        (the Warp kernels read them in place) with no per-obstacle work.
        """
        data = self.planner.scene_collision_checker.data.cuboids
        if data is None:
            raise RuntimeError("the collision checker has no cuboid cache")
        m = int(dims.shape[1])
        if m > data.max_n:
            raise ValueError(f"{m} cuboids per world exceed the cache of {data.max_n}")
        dev, dtype = data.dims.device, data.dims.dtype
        idx = torch.tensor(env_ids, device=dev, dtype=torch.long)
        with self._gpu_lock:
            data.dims[idx, :m, :3] = dims.to(dev, dtype)
            data.inv_pose[idx, :m, :7] = inv_pose.to(dev, dtype)
            data.enable[idx, :m] = enable.to(dev, torch.uint8)
            if m < data.max_n:
                data.enable[idx, m:] = 0
            data.count[idx] = m
        for i, k in enumerate(env_ids):
            data.names[k] = [f"cuboid_{j}" for j in range(m)] + [None] * (data.max_n - m)
            slot = self.slots[k]
            slot.world = None
            slot.world_tensors = (dims[i], inv_pose[i], enable[i])

    @staticmethod
    def _world_from_tensors(
        tensors: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> SceneCfg:
        """SceneCfg for a tensor world (enabled cuboids only), for the single planner."""
        dims, inv_pose, enable = (t.detach().cpu() for t in tensors)
        cuboids: list[Cuboid] = []
        for j in range(int(dims.shape[0])):
            if not bool(enable[j]):
                continue
            p_inv = inv_pose[j, :3]
            w, x, y, z = (float(v) for v in inv_pose[j, 3:7])
            # pose = inverse of (p_inv, q_inv): q = conj(q_inv), p = -rotate(q, p_inv)
            qw, qx, qy, qz = w, -x, -y, -z
            px, py, pz = (float(v) for v in p_inv)
            # rotate p_inv by q (wxyz)
            t_x = 2.0 * (qy * pz - qz * py)
            t_y = 2.0 * (qz * px - qx * pz)
            t_z = 2.0 * (qx * py - qy * px)
            rx = px + qw * t_x + (qy * t_z - qz * t_y)
            ry = py + qw * t_y + (qz * t_x - qx * t_z)
            rz = pz + qw * t_z + (qx * t_y - qy * t_x)
            cuboids.append(
                Cuboid(
                    name=f"cuboid_{j}",
                    pose=[-rx, -ry, -rz, qw, qx, qy, qz],
                    dims=[float(v) for v in dims[j, :3]],
                )
            )
        return SceneCfg(cuboid=cuboids)

    def fit_cuboid_spheres(self, dims: tuple[float, float, float]) -> torch.Tensor:
        """Spheres (object frame, ``[n, 4]``) approximating a cuboid of ``dims``; cached."""
        key = tuple(round(float(d), 5) for d in dims)
        spheres = self._sphere_fit_cache.get(key)
        if spheres is None:
            n_slots = int(self._attached_idx.shape[0])
            with self._gpu_lock:
                spheres = self.planner.attachment_manager.fit_spheres(
                    [Cuboid(name="attached", dims=list(key), pose=[0, 0, 0, 1, 0, 0, 0])],
                    num_spheres=n_slots,
                    sphere_fit_type=SphereFitType.VOXEL,
                )
            self._sphere_fit_cache[key] = spheres.to(self.tensor_args.device)
            spheres = self._sphere_fit_cache[key]
        return spheres

    def set_attachments_tensor(
        self, env_ids: list[int], spheres: list[torch.Tensor | None]
    ) -> None:
        """Set the attached-object spheres of several slots in one tensor write
        (``spheres[i]`` is ``[n, 4]`` in the link frame, or None to clear slot ``env_ids[i]``)."""
        link_spheres = self._kin_params.link_spheres
        n_slots = int(self._attached_idx.shape[0])
        block = torch.zeros(
            (len(env_ids), n_slots, 4), device=link_spheres.device, dtype=link_spheres.dtype
        )
        block[:, :, 3] = DISABLED_SPHERE_RADIUS
        for i, (k, sp) in enumerate(zip(env_ids, spheres)):
            self.slots[k].attached_spheres = (
                None if sp is None else sp.detach().to(self.tensor_args.device)
            )
            if sp is not None:
                n = min(int(sp.shape[0]), n_slots)
                block[i, :n] = sp[:n].to(device=link_spheres.device, dtype=link_spheres.dtype)
        idx = torch.tensor(env_ids, device=link_spheres.device, dtype=torch.long)
        with self._gpu_lock:
            if link_spheres.dim() == 3:
                link_spheres[idx[:, None], self._attached_idx[None, :], :] = block
            else:
                link_spheres[self._attached_idx, :] = block[-1]

    def clear_all_attachments(self) -> None:
        with self._gpu_lock:
            for k in range(self.num_envs):
                self.slots[k].attached_spheres = None
                self._write_attachment_row(self._kin_params.link_spheres, k, None)

    def _write_attachment_row(
        self,
        link_spheres: torch.Tensor,
        row: int,
        spheres: torch.Tensor | None,
    ) -> None:
        n_slots = int(self._attached_idx.shape[0])
        block = torch.zeros((n_slots, 4), device=link_spheres.device, dtype=link_spheres.dtype)
        block[:, 3] = DISABLED_SPHERE_RADIUS
        if spheres is not None:
            n = min(int(spheres.shape[0]), n_slots)
            block[:n] = spheres[:n].to(device=link_spheres.device, dtype=link_spheres.dtype)
        if link_spheres.dim() == 3:
            link_spheres[row, self._attached_idx, :] = block
        else:
            link_spheres[self._attached_idx, :] = block

    # ------------------------------------------------------------------
    # Direct (non-batched) queries
    # ------------------------------------------------------------------
    def _js(self, q: torch.Tensor) -> JointState:
        return JointState.from_position(q.view(-1, self.dof), joint_names=self.joint_names)

    def _env_idx_tensor(self, env_idx: int, n: int = 1) -> torch.Tensor:
        return torch.full((n,), int(env_idx), device=self.tensor_args.device, dtype=torch.int32)

    def kinematics_state(self, q: torch.Tensor, env_idx: int = 0) -> tuple[Pose, torch.Tensor]:
        """``(ee_pose, robot_spheres)`` for joint positions ``q`` (``[dof]`` or ``[b, dof]``).

        ``robot_spheres`` include slot ``env_idx``'s attached-object spheres.
        """
        with self._gpu_lock:
            js = self._js(q)
            state = self.planner.kinematics.compute_kinematics(
                js, idxs_env=self._env_idx_tensor(env_idx, js.position.shape[0])
            )
            pose = state.tool_poses.get_link_pose(self.tool_frame).clone()
            spheres = state.robot_spheres.clone() if state.robot_spheres is not None else None
            return pose, spheres

    def _states_valid(
        self, env_ids: list[int], q: torch.Tensor, check_world: bool = True
    ) -> list[bool]:
        """Row ``i`` of ``q`` (``[n, dof]``) in slot ``env_ids[i]``; ``check_world=False`` checks
        joint limits and self collision only.  (Not ``RobotSceneCollision.validate``: that
        runs kinematics with env 0's spheres.)"""
        n = len(env_ids)
        with self._gpu_lock:
            qq = q.reshape(n, 1, self.dof).to(self.tensor_args.device, self.tensor_args.dtype)
            env = torch.tensor(env_ids, device=self.tensor_args.device, dtype=torch.int32)
            v = self._validator
            v.setup_batch_tensors(n, 1)
            kin_state = v.kinematics.compute_kinematics(
                JointState.from_position(qq, joint_names=self.joint_names), idxs_env=env
            )
            total = v.get_bound(qq).sum(dim=-1)
            if v.self_collision_cost is not None:
                total = total + v.get_self_collision(kin_state.robot_spheres).sum(dim=-1)
            if check_world and v.collision_constraint is not None:
                total = total + v.get_collision_constraint(kin_state, env).sum(dim=-1)
            return [bool(x) for x in (total.view(n, -1).sum(dim=-1) == 0.0).tolist()]

    def obstacle_names(self, env_idx: int) -> list[str]:
        with self._gpu_lock:
            return list(self.planner.scene_collision_checker.get_obstacle_names(env_idx))

    # ------------------------------------------------------------------
    # Batched requests
    # ------------------------------------------------------------------
    def solve_ik_rows(
        self,
        rows: list[tuple[int, Pose, torch.Tensor | None]],
    ) -> dict[int, IKResult]:
        """Solve several IK goals synchronously as ONE padded GPU batch.

        ``rows`` holds ``(env_slot, goal_pose, seed_q)`` with at most one row per slot
        (``seed_q=None`` seeds from the slot's last configuration).  Each slot's world and
        attachment must already be loaded.  Returns ``{env_slot: result}``.
        """
        reqs = [
            _Request(kind="ik", env_idx=int(k), goal_pose=goal, seed_q=seed)
            for k, goal, seed in rows
        ]
        if not reqs:
            return {}
        t0 = time.monotonic()
        with self._gpu_lock:
            self._run_ik(reqs)
        st = self.stats["ik"]
        st[0] += 1
        st[1] += len(reqs)
        st[2] += time.monotonic() - t0
        return {req.env_idx: req.result for req in reqs}

    def states_valid(
        self, env_ids: list[int], q: torch.Tensor, check_world: bool = True
    ) -> list[bool]:
        """True per row when row ``i`` of ``q`` (``[n, dof]``) is within limits and collision
        free in slot ``env_ids[i]``'s world, with that slot's attached-object spheres."""
        return self._states_valid(env_ids, q, check_world)

    def plan_pose_rows(
        self,
        rows: list[tuple[int, torch.Tensor, Pose, MotionGenPlanConfig | None]],
    ) -> dict[int, MotionGenResult]:
        """Plan several pose goals synchronously as ONE padded GPU batch.

        ``rows`` holds ``(env_slot, start_q, goal_pose, plan_config)`` with at most one
        row per slot; each slot's world and attachment must already be loaded with
        :meth:`set_worlds_tensor` / :meth:`set_attachments_tensor`.  Rows that fail in the
        batch retry on the single graph-enabled planner.  Returns ``{env_slot: result}``.
        """
        reqs = [
            _Request(kind="plan_pose", env_idx=int(k), goal_pose=goal, start_q=q, plan_config=cfg)
            for k, q, goal, cfg in rows
        ]
        if not reqs:
            return {}
        t0 = time.monotonic()
        with self._gpu_lock:
            self._run_plan(reqs)
        st = self.stats["plan_pose"]
        st[0] += 1
        st[1] += len(reqs)
        st[2] += time.monotonic() - t0
        return {req.env_idx: req.result for req in reqs}

    # ------------------------------------------------------------------
    # Batch assembly
    # ------------------------------------------------------------------
    def _rows(self, reqs: list[_Request]) -> dict[int, _Request]:
        """Map env slot -> request; a slot can only carry one request per round."""
        rows: dict[int, _Request] = {}
        for req in reqs:
            if req.env_idx in rows:
                raise RuntimeError(f"two concurrent requests for env slot {req.env_idx}")
            rows[req.env_idx] = req
        return rows

    def _goal_batch(self, rows: dict[int, _Request]) -> GoalToolPose:
        B = self.num_envs
        any_req = next(iter(rows.values()))
        pos = torch.zeros((B, 3), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        quat = torch.zeros((B, 4), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        quat[:, 0] = 1.0
        filler = any_req.goal_pose
        for k in range(B):
            req = rows.get(k)
            pose = req.goal_pose if req is not None else filler
            assert pose is not None
            pos[k] = pose.position.view(-1, 3)[0]
            quat[k] = pose.quaternion.view(-1, 4)[0]
        return GoalToolPose.from_poses({self.tool_frame: Pose(position=pos, quaternion=quat)})

    def _start_batch(self, rows: dict[int, _Request], use_seed: bool) -> JointState:
        B = self.num_envs
        q = torch.zeros((B, self.dof), device=self.tensor_args.device, dtype=self.tensor_args.dtype)
        for k in range(B):
            req = rows.get(k)
            src: torch.Tensor | None = None
            if req is not None:
                src = req.seed_q if use_seed else req.start_q
            if src is None:
                src = self.slots[k].last_q if self.slots[k].last_q is not None else self.default_q
            q[k] = src.reshape(-1)[: self.dof].to(q.device, q.dtype)
        return JointState.from_position(q, joint_names=self.joint_names)

    def _run_ik(self, reqs: list[_Request]) -> None:
        rows = self._rows(reqs)
        goal = self._goal_batch(rows)
        current = self._start_batch(rows, use_seed=True)
        t0 = time.monotonic()
        ik = self.planner.ik_solver.solve_pose(goal, current_state=current, return_seeds=1)
        dt = time.monotonic() - t0
        success = ik.success.view(self.num_envs, -1)
        js = ik.js_solution
        for k, req in rows.items():
            req.result = IKResult(
                success=success[k, :1].clone(),
                js_solution=js[k : k + 1].clone(),
                solution=None if ik.solution is None else ik.solution[k : k + 1].clone(),
                position_error=None
                if ik.position_error is None
                else ik.position_error[k : k + 1].clone(),
                rotation_error=None
                if ik.rotation_error is None
                else ik.rotation_error[k : k + 1].clone(),
                solve_time=dt,
            )

    def _run_plan(self, reqs: list[_Request]) -> None:
        rows = self._rows(reqs)
        start = self._start_batch(rows, use_seed=False)
        # Start-state validity (one batched check), as v1 plan_single reports it.
        pending: dict[int, _Request] = dict(rows)
        check = [
            k
            for k, req in rows.items()
            if req.plan_config is None or req.plan_config.check_start_validity
        ]
        if check:
            valid = self._states_valid(check, start.position[check])
            for k, ok in zip(check, valid):
                if not ok:
                    rows[k].result = MotionGenResult(
                        success=torch.zeros(1, dtype=torch.bool, device=self.tensor_args.device),
                        status=MotionGenStatus.INVALID_START_STATE_WORLD_COLLISION,
                        interpolation_dt=self.interpolation_dt,
                    )
                    del pending[k]
        if not pending:
            return
        # Padded rows duplicate a real request, so requiring every row to succeed
        # (success_ratio=1.0) retries exactly when a real row failed.
        t0 = time.monotonic()
        goal = self._goal_batch(pending)
        result = self.planner.plan_pose(
            goal,
            start,
            max_attempts=self.batch_attempts,
            success_ratio=1.0,
        )
        dt = time.monotonic() - t0
        for k, req in pending.items():
            ok = False
            if result is not None and result.success is not None:
                ok = bool(result.success.view(self.num_envs, -1)[k].any().item())
            if ok:
                req.result = self._row_result(result, k, dt)
            else:
                req.result = self._fallback_single(k, req, start.position[k], dt)

    def _row_result(self, result: Any, k: int, dt: float) -> MotionGenResult:
        """Extract row ``k`` (best seed) of a batch trajopt result as a v1-shaped result."""
        interp = result.interpolated_trajectory[k][0]
        last = int(result.interpolated_last_tstep.view(self.num_envs, -1)[k, 0].item())
        plan = trim_joint_state_trajectory(interp, 0, last)
        if plan.joint_names is not None and list(plan.joint_names) != self.joint_names:
            plan = plan.reorder(self.joint_names)
        optimized = result.js_solution[k][0]
        if optimized.joint_names is not None and list(optimized.joint_names) != self.joint_names:
            optimized = optimized.reorder(self.joint_names)
        return MotionGenResult(
            success=torch.ones(1, dtype=torch.bool, device=self.tensor_args.device),
            status=MotionGenStatus.SUCCESS,
            optimized_plan=optimized,
            optimized_dt=None if optimized.dt is None else optimized.dt,
            interpolated_plan=plan,
            interpolation_dt=self.interpolation_dt,
            path_buffer_last_tstep=[last],
            solve_time=dt,
            total_time=dt,
        )

    def _single_result(self, result: Any, dt: float, used_graph: bool) -> MotionGenResult:
        plan = result.get_interpolated_plan()
        if plan.ndim == 3:
            plan = plan[0]
        if plan.joint_names is not None and list(plan.joint_names) != self.joint_names:
            plan = plan.reorder(self.joint_names)
        optimized = result.js_solution[0][0] if result.js_solution.ndim == 4 else result.js_solution
        if optimized.joint_names is not None and list(optimized.joint_names) != self.joint_names:
            optimized = optimized.reorder(self.joint_names)
        return MotionGenResult(
            success=torch.ones(1, dtype=torch.bool, device=self.tensor_args.device),
            status=MotionGenStatus.SUCCESS,
            optimized_plan=optimized,
            optimized_dt=None if optimized.dt is None else optimized.dt,
            interpolated_plan=plan,
            interpolation_dt=self.interpolation_dt,
            path_buffer_last_tstep=[int(plan.position.shape[-2])],
            solve_time=dt,
            total_time=dt,
            used_graph=used_graph,
        )

    def _fallback_single(
        self,
        k: int,
        req: _Request,
        start_q: torch.Tensor,
        batch_dt: float,
    ) -> MotionGenResult:
        """Retry a failed batch row on the graph-enabled single planner."""
        failed = MotionGenResult(
            success=torch.zeros(1, dtype=torch.bool, device=self.tensor_args.device),
            status=MotionGenStatus.TRAJOPT_FAIL,
            interpolation_dt=self.interpolation_dt,
            solve_time=batch_dt,
            total_time=batch_dt,
        )
        sp = self.single_planner
        if sp is None:
            return failed
        self.fallback_calls += 1
        cfg = req.plan_config or MotionGenPlanConfig()
        max_attempts = max(1, min(int(cfg.max_attempts), self.single_fallback_max_attempts))
        graph_attempt = cfg.enable_graph_attempt if cfg.enable_graph else None
        enable_graph_attempt = (
            max_attempts + 1 if graph_attempt is None else max(0, int(graph_attempt))
        )
        slot = self.slots[k]
        if slot.world is None and slot.world_tensors is not None:
            slot.world = self._world_from_tensors(slot.world_tensors)
        if slot.world is not None:
            sp.update_world(slot.world)
        self._write_attachment_row(
            sp.kinematics.config.kinematics_config.link_spheres, 0, slot.attached_spheres
        )
        start = JointState.from_position(start_q.view(1, self.dof), joint_names=self.joint_names)
        t0 = time.monotonic()
        assert req.goal_pose is not None
        goal = GoalToolPose.from_poses(
            {
                self.tool_frame: Pose(
                    position=req.goal_pose.position.view(-1, 3)[:1],
                    quaternion=req.goal_pose.quaternion.view(-1, 4)[:1],
                )
            },
        )
        result = sp.plan_pose(
            goal, start, max_attempts=max_attempts, enable_graph_attempt=enable_graph_attempt
        )
        dt = time.monotonic() - t0
        if (
            result is None
            or result.success is None
            or not bool(result.success.view(-1).any().item())
        ):
            failed.status = (
                MotionGenStatus.IK_FAIL if result is None else MotionGenStatus.TRAJOPT_FAIL
            )
            failed.total_time = batch_dt + dt
            return failed
        self.fallback_successes += 1
        return self._single_result(
            result, batch_dt + dt, used_graph=enable_graph_attempt < max_attempts
        )
