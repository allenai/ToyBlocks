"""Collision-free random robot start configurations for reset-state sampling.

Used by ``scripts/generate_task_from_pddl.py`` when ``--randomize_robot_init`` is set.
The *end-effector pose* is what gets randomized: its position is drawn uniformly from a
workspace box whose top is the end-effector height at the robot's home configuration, and
its orientation is held at the home configuration's end-effector orientation (the same
gripper orientation as a non-randomized start). CuRobo IK (seeded at the home
configuration, with self-collision and per-env world collision enabled) turns each pose
into a joint configuration; poses without a valid solution are redrawn. The gripper is
independently set open or closed per environment.

The robot model and the collision world are the ones the vecgen engine plans with: the
FR3 + Robotiq 2F-85 cuRobo model of :mod:`toyblocks.vecgen.planner.robot_config` and, per env,
every block's collider as a cuboid plus the table slab (``Geometry.update_worlds``), in the
robot base frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
from curobo.scene import Cuboid
from curobo.scene import Scene as WorldConfig
from curobo.types import DeviceCfg, GoalToolPose, JointState, Pose
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from torch import Tensor

from toyblocks.tasks.base_env import BaseEnv
from toyblocks.vecgen.geometry import TABLE_DIMS
from toyblocks.vecgen.planner.robot_config import build_robot_cfg_dict
from toyblocks.vecgen.sim import VecSim

#: Attached-object sphere slots of the cuRobo model (as the planner's robot model was built).
NUM_ATTACHED_SPHERES = 100
TABLE_NAME = "workspace_table"


@dataclass
class RobotInitSamplingConfig:
    ee_x_range: tuple[float, float] = (0.25, 0.65)
    """End-effector x range (meters, robot base frame) to sample from."""
    ee_y_range: tuple[float, float] = (-0.35, 0.35)
    """End-effector y range (meters, robot base frame) to sample from."""
    ee_z_min: float = 0.20
    """Lowest end-effector height (meters, robot base frame). The upper bound is the
    end-effector height at the robot's home configuration."""
    num_candidates: int = 32
    """End-effector poses tried per environment before giving up and keeping the home pose."""
    gripper_closed_prob: float = 0.5
    """Probability that an environment starts with the gripper closed."""
    settle_steps: int = 24
    """Physics steps after writing the new configuration, so the gripper reaches its target."""
    ik_num_seeds: int = 20
    """CuRobo IK seeds per pose."""
    joint_limit_margin: float = 0.15
    """Reject IK solutions closer than this (radians) to any joint bound, so the start is not on a
    limit the planner's trajectory optimization refuses to start from."""


class CollisionFreeRobotInitSampler:
    """Samples end-effector poses and solves collision-checked IK against each env's live scene."""

    def __init__(
        self,
        base_env: BaseEnv,
        sim: VecSim,
        cfg: RobotInitSamplingConfig,
        rng: np.random.Generator,
    ) -> None:
        self._base_env = base_env
        self._sim = sim
        self._cfg = cfg
        self._rng = rng
        self._articulation = sim.art
        self._arm_joint_ids: list[int] = list(sim.arm_ids)
        self._robot_cfg: dict[str, Any] = build_robot_cfg_dict(
            robotiq=True, num_attached_spheres=NUM_ATTACHED_SPHERES
        )
        # Collider edge lengths [N, K, 3] and table collider top above its origin [N]; the
        # blocks keep their size across reset batches, so both are measured once.
        self._block_dims: Tensor | None = None
        self._table_top: Tensor | None = None
        self._device = torch.device(base_env.device)
        self._tensor_args = DeviceCfg(device=self._device)
        self._gripper_joint_ids, self._gripper_open, self._gripper_close = self._gripper_targets()
        # Home configuration from the articulation's default joint state (the template's init pose).
        self._home_q: torch.Tensor = (
            self._articulation.data.default_joint_pos[0, self._arm_joint_ids]
            .detach()
            .clone()
            .float()
        )
        self._home_ee_z: float | None = None
        self._home_ee_quat: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _gripper_targets(self) -> tuple[list[int], torch.Tensor, torch.Tensor]:
        """Gripper joint ids with their open / closed position targets from the robot's binary gripper action."""
        for controller in self._base_env._robot_cfg.controllers.values():
            gripper_cfg = controller.get("gripper")
            if isinstance(gripper_cfg, BinaryJointPositionActionCfg):
                joint_names = list(gripper_cfg.open_command_expr.keys())
                joint_ids = list(self._articulation.find_joints(joint_names)[0])
                open_t = torch.tensor(
                    [gripper_cfg.open_command_expr[n] for n in joint_names], device=self._device
                )
                close_t = torch.tensor(
                    [gripper_cfg.close_command_expr[n] for n in joint_names], device=self._device
                )
                return joint_ids, open_t, close_t
        raise ValueError("Robot config has no BinaryJointPositionActionCfg gripper controller")

    def collision_worlds(self) -> list[WorldConfig]:
        """Per-env collision worlds in the robot base frame from the current scene: every
        block's collider as a cuboid at its root pose plus the table slab (top at the table
        collider's top), as ``Geometry.update_worlds`` builds them for planning."""
        if self._block_dims is None or self._table_top is None:
            self._block_dims, _ = self._sim.measure_block_dims()
            self._table_top = self._sim.measure_table_top(TABLE_NAME)
        poses = self._sim.object_poses().detach().cpu().numpy()  # [N, K, 7]
        dims = self._block_dims.detach().cpu().numpy()
        table = self._sim.object_pose(TABLE_NAME).detach().cpu().numpy()  # [N, 7]
        table_top = self._table_top.detach().cpu().numpy()
        worlds: list[WorldConfig] = []
        for e in range(self._sim.num_envs):
            cuboids = [
                Cuboid(name=name, pose=poses[e, k].tolist(), dims=dims[e, k].tolist())
                for k, name in enumerate(self._sim.block_names)
            ]
            table_pose = [
                float(table[e, 0]),
                float(table[e, 1]),
                float(table[e, 2] + table_top[e] - TABLE_DIMS[2] / 2),
                1.0,
                0.0,
                0.0,
                0.0,
            ]
            cuboids.append(Cuboid(name=TABLE_NAME, pose=table_pose, dims=list(TABLE_DIMS)))
            worlds.append(WorldConfig(cuboid=cuboids))
        return worlds

    def _build_ik_solver(self, world_cfgs: list[WorldConfig]) -> InverseKinematics:
        """Batched cuRobo IK with one collision world per environment."""
        n_cuboids = max(30, max(len(w.cuboid or []) for w in world_cfgs))
        n_meshes = max(30, max(len(w.mesh or []) for w in world_cfgs))
        config = InverseKinematicsCfg.create(
            robot=self._robot_cfg,
            device_cfg=self._tensor_args,
            num_seeds=self._cfg.ik_num_seeds,
            position_tolerance=0.005,
            orientation_tolerance=0.05,
            use_cuda_graph=False,
            self_collision_check=True,
            collision_cache={"cuboid": n_cuboids, "mesh": n_meshes},
            max_batch_size=len(world_cfgs),
            multi_env=True,
            # cuRobo v2 builds the scene collision checker only when given a scene model; env 0's
            # world seeds it and every env's own world is loaded below.
            scene_model={
                "cuboid": {
                    c.name: {"dims": list(c.dims), "pose": list(c.pose)}
                    for c in world_cfgs[0].cuboid or []
                }
            },
        )
        if config.scene_collision_cfg is None:
            raise RuntimeError("cuRobo IK was built without a scene collision checker")
        config.scene_collision_cfg.num_envs = len(world_cfgs)
        ik = InverseKinematics(config)
        for env_idx, world_cfg in enumerate(world_cfgs):
            ik.scene_collision_checker.load_collision_model(world_cfg, env_idx=env_idx)
        return ik

    def _ee_poses(self, ik: InverseKinematics, q: torch.Tensor) -> Pose:
        """End-effector poses for arm configurations ``q`` (``[n, dof]``)."""
        js = JointState.from_position(q.reshape(-1, q.shape[-1]), joint_names=list(ik.joint_names))
        return ik.compute_kinematics(js).tool_poses.get_link_pose(ik.tool_frames[0])

    def _random_positions(self, n: int, z_max: float) -> torch.Tensor:
        """``(n, 3)`` positions uniform in the workspace box, robot base frame."""
        lo = np.array([self._cfg.ee_x_range[0], self._cfg.ee_y_range[0], self._cfg.ee_z_min])
        hi = np.array([self._cfg.ee_x_range[1], self._cfg.ee_y_range[1], z_max])
        p = lo + self._rng.random((n, 3)) * (hi - lo)
        return torch.from_numpy(p.astype(np.float32)).to(self._device)

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample_and_apply(self) -> int:
        """Write a random collision-free arm pose and a random gripper state to every env.

        Envs for which no sampled end-effector pose has a valid IK solution keep their current
        arm configuration. Returns the number of envs that received a new arm configuration.
        """
        num_envs = self._articulation.num_instances
        world_cfgs = self.collision_worlds()
        ik = self._build_ik_solver(world_cfgs)
        dof = len(ik.joint_names)
        assert dof == len(self._arm_joint_ids), (
            f"CuRobo IK has {dof} active joints but the arm has {len(self._arm_joint_ids)}"
        )
        if self._home_ee_z is None or self._home_ee_quat is None:
            home_fk = self._ee_poses(ik, self._home_q.view(1, -1))
            self._home_ee_z = float(home_fk.position.reshape(-1)[2].item())
            self._home_ee_quat = home_fk.quaternion.reshape(4).detach().clone().float()
            print(
                f"[robot_init_sampling] home EE height {self._home_ee_z:.3f} m caps the sampled z "
                f"(box x{self._cfg.ee_x_range} y{self._cfg.ee_y_range} z[{self._cfg.ee_z_min}, home]); "
                f"EE orientation fixed at home quat (wxyz) {self._home_ee_quat.tolist()}"
            )
        home_quat_batch: torch.Tensor = (
            self._home_ee_quat.unsqueeze(0).expand(num_envs, -1).contiguous()
        )
        z_max = max(self._home_ee_z, self._cfg.ee_z_min)
        limits = ik.kinematics.get_joint_limits().position.to(
            self._device
        )  # [2, dof], ik.joint_names order

        home_batch = self._home_q.unsqueeze(0).expand(num_envs, -1).contiguous()
        q_new = home_batch.clone()
        pending = torch.ones(num_envs, dtype=torch.bool, device=self._device)
        for _ in range(self._cfg.num_candidates):
            goal = GoalToolPose.from_poses(
                {
                    ik.tool_frames[0]: Pose(
                        position=self._random_positions(num_envs, z_max),
                        quaternion=home_quat_batch,
                    )
                }
            )
            result = ik.solve_pose(
                goal,
                current_state=JointState.from_position(home_batch, joint_names=list(ik.joint_names)),
                seed_config=home_batch.unsqueeze(1),
                return_seeds=1,
            )
            success = result.success.reshape(-1).to(torch.bool)
            solution = (
                result.js_solution.reorder(list(ik.joint_names))
                .position.detach()
                .reshape(num_envs, -1)[:, :dof]
            )
            margin = torch.minimum(solution - limits[0], limits[1] - solution).min(dim=1).values
            success &= margin >= self._cfg.joint_limit_margin
            newly = success & pending
            q_new[newly] = solution[newly]
            pending &= ~newly
            if not pending.any():
                break

        env_ids = torch.arange(num_envs, device=self._device)
        self._articulation.write_joint_position_to_sim(
            q_new, joint_ids=self._arm_joint_ids, env_ids=env_ids
        )
        self._articulation.write_joint_velocity_to_sim(
            torch.zeros_like(q_new), joint_ids=self._arm_joint_ids, env_ids=env_ids
        )
        self._articulation.set_joint_position_target(
            q_new, joint_ids=self._arm_joint_ids, env_ids=env_ids
        )
        self._articulation.set_joint_velocity_target(
            torch.zeros_like(q_new), joint_ids=self._arm_joint_ids, env_ids=env_ids
        )

        closed = torch.from_numpy(self._rng.random(num_envs) < self._cfg.gripper_closed_prob).to(
            self._device
        )
        gripper_target = torch.where(
            closed.unsqueeze(1),
            self._gripper_close.unsqueeze(0).expand(num_envs, -1),
            self._gripper_open.unsqueeze(0).expand(num_envs, -1),
        )
        self._articulation.set_joint_position_target(
            gripper_target, joint_ids=self._gripper_joint_ids, env_ids=env_ids
        )
        self._base_env.scene.write_data_to_sim()
        for _ in range(self._cfg.settle_steps):
            self._base_env.step(None)

        num_failed = int(pending.sum().item())
        if num_failed:
            print(
                f"[robot_init_sampling] {num_failed}/{num_envs} env(s) had no IK-feasible pose in "
                f"{self._cfg.num_candidates} attempts; keeping their current arm configuration"
            )
        return num_envs - num_failed
