from collections.abc import Sequence

import numpy as np
import torch
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.utils import math

from blocksuite.robots.kinematics import Kinematics
from blocksuite.tasks.base_env import BaseEnv
from blocksuite.utils import obs_utils, randomization_utils, rigid_objects

from .reach_env_cfg import ReachEnvCfg

_ARM_COLLISION_LINKS = [
    "panda_link3",
    "panda_link4",
    "panda_link5",
    "panda_link6",
    "panda_link7",
    "robotiq_arg2f_base_link",
    "left_inner_finger",
    "right_inner_finger",
]
"""Links checked for contact. Excludes link0-2, which cannot reach the table or an
obstacle without the rest of the arm getting there first."""


class ReachEnv(BaseEnv):
    """Reach a commanded EE pose without touching anything.

    Built on the push_stacked structure: preallocated obstacle slots with an active
    mask, inactive slots parked underground, per-env slot permutation, relaxation-based
    non-overlap, and IK-randomized start configurations.

    What differs from push:

    - The goal is an EE pose, not a block pose. It is produced by IK-ing the arm to a
      sampled pose and keeping the *achieved* pose, so the goal is reachable by
      construction and no rejection loop is needed for reachability itself.
    - Obstacles are only ever obstacles; there is no manipulated block.
    - The policy is shown ``num_attended_obstacles`` of ``num_obstacles``, chosen by
      proximity to the [tcp, goal] corridor.
    - Contact sensors cover the whole arm rather than just the fingers, because a graze
      that displaces nothing still matters here.
    """

    cfg: ReachEnvCfg

    obstacles: list[RigidObject]
    goal: RigidObject
    kinematics: Kinematics

    # intermediate values
    tcp_pos: torch.Tensor
    tcp_quat: torch.Tensor
    pos_error: torch.Tensor
    axis_angle_error: torch.Tensor
    reached_pose: torch.Tensor
    acheived_goal: torch.Tensor

    # obstacle state
    obstacles_init_pos: torch.Tensor  # (num_envs, num_obstacles, 3)
    obstacles_init_quat: torch.Tensor  # (num_envs, num_obstacles, 4)
    obstacles_moved: torch.Tensor  # (num_envs,)
    obstacles_corner_pts_rel_ee: torch.Tensor
    active_obstacle_mask: torch.Tensor  # (num_envs, num_obstacles) physically present
    attended_obstacle_mask: torch.Tensor  # (num_envs, num_obstacles) shown to the policy
    clearance_penalty: torch.Tensor  # (num_envs,) barrier in [0, 1]
    pruned_min_clearance: torch.Tensor  # (num_envs,) nearest obstacle we are NOT showing

    # collision
    arm_collision_force: torch.Tensor  # (num_envs,)
    table_penetration: torch.Tensor  # (num_envs,)
    collided: torch.Tensor  # (num_envs,)

    # goal info
    goal_pos: torch.Tensor
    goal_quat: torch.Tensor

    # ------------------------------------------------------------------
    # Scene
    # ------------------------------------------------------------------

    def _setup_scene_objects(self):
        block_sizes = torch.zeros(self.num_envs, 3, device=self.device)
        block_sizes[:, :] = (
            torch.rand(self.num_envs, 1, device=self.device)
            * (self.cfg.block_size_range[1] - self.cfg.block_size_range[0])
            + self.cfg.block_size_range[0]
        )
        if self.cfg.randomize_block_size:
            from blocksuite.utils.rigid_objects.spawn_indexed import bucket_isotropic_cube_sizes

            sides = block_sizes[:, 0].cpu().tolist()
            indices, bucket_sizes = bucket_isotropic_cube_sizes(
                sides,
                self.cfg.block_size_num_buckets,
                self.cfg.block_size_range,
            )
            quantized_sides = torch.tensor(
                [bucket_sizes[i][0] for i in indices],
                device=self.device,
                dtype=torch.float32,
            )
            block_sizes[:, :] = quantized_sides.unsqueeze(1)
        self.block_sizes = (
            block_sizes
            if self.cfg.randomize_block_size
            else torch.tensor(self.cfg.block_size, device=self.device, dtype=torch.float32)
            .unsqueeze(0)
            .expand(self.num_envs, -1)
            .clone()
        )
        block_sizes_list = self.block_sizes.cpu().tolist()
        bucket_kw = (
            dict(
                max_unique_assets=self.cfg.block_size_num_buckets,
                size_range=self.cfg.block_size_range,
            )
            if self.cfg.randomize_block_size
            else {}
        )

        obstacles = []
        underground_z = -1.05 + self.cfg.block_size[2] / 2  # base_env ground plane sits at z=-1.05
        for i in range(self.cfg.num_obstacles):
            obstacle = rigid_objects.create_cuboid(
                prim_path=f"/World/envs/env_.*/Obstacle_{i}",
                size=block_sizes_list if self.cfg.randomize_block_size else self.cfg.block_size,
                diffuse_color=(1.0, 0.0, 0.0),
                init_pos=(1.0, 0.0, underground_z),
                **bucket_kw,
            )
            self.scene.rigid_objects[f"obstacle_{i}"] = obstacle
            obstacles.append(obstacle)
        self.obstacles = obstacles

        # Visual-only marker for the commanded pose.
        self.goal = rigid_objects.create_toy_block(
            prim_path="/World/envs/env_.*/Goal",
            size=block_sizes_list if self.cfg.randomize_block_size else self.cfg.block_size,
            init_pos=(0.0, 0.0, self.cfg.block_size[2] / 2),
            kinematic=True,
            collision=False,
            **bucket_kw,
        )
        self.scene.rigid_objects["goal"] = self.goal

        # Whole-arm contact, filtered against the table and every obstacle.
        env_ns = self.scene.env_regex_ns
        collision_filter = [f"{env_ns}/workspace_table"] + [
            f"{env_ns}/Obstacle_{i}" for i in range(self.cfg.num_obstacles)
        ]
        for link in _ARM_COLLISION_LINKS:
            cfg = ContactSensorCfg(
                prim_path=f"{env_ns}/Robot/panda/{link}",
                filter_prim_paths_expr=collision_filter,
                track_air_time=False,
                history_length=1,
            )
            self.scene.sensors[f"contact:{link}"] = ContactSensor(cfg)

    def _after_scene_setup(self):
        super()._after_scene_setup()
        n, d = self.num_envs, self.cfg.num_obstacles
        self.obstacles_init_pos = torch.zeros(n, d, 3, device=self.device)
        self.obstacles_init_quat = torch.zeros(n, d, 4, device=self.device)
        self.active_obstacle_mask = torch.zeros(n, d, dtype=torch.bool, device=self.device)
        self.attended_obstacle_mask = torch.zeros(n, d, dtype=torch.bool, device=self.device)
        self._attend_ttl = torch.zeros(n, d, dtype=torch.int32, device=self.device)
        self.collided = torch.zeros(n, dtype=torch.bool, device=self.device)

        self._arm_contact_sensors = [
            self.scene.sensors[f"contact:{link}"] for link in _ARM_COLLISION_LINKS
        ]

    @property
    def ee_frame(self) -> FrameTransformer:
        return self.scene.sensors["frame_transformer:ee"]

    def _transform_corners_to_frame(
        self,
        frame: FrameTransformer,
        corner_points: torch.Tensor,
    ) -> torch.Tensor:
        """Express ``corner_points`` (N, K, 3) in ``frame``, returned as (N, K * 3)."""
        N, K = corner_points.shape[:2]
        corner_points_flat = corner_points.reshape(-1, 3)
        unit_quat = torch.zeros(N * K, 4, device=corner_points.device)
        unit_quat[:, 0] = 1.0

        target_pos_w_expanded = frame.data.target_pos_w.expand(N, K, -1).reshape(-1, 3)
        target_quat_w_expanded = frame.data.target_quat_w.expand(N, K, -1).reshape(-1, 4)

        corner_points_rel_flat, _ = math.subtract_frame_transforms(
            target_pos_w_expanded,
            target_quat_w_expanded,
            corner_points_flat,
            unit_quat,
        )
        return corner_points_rel_flat.reshape(N, K * 3)

    # ------------------------------------------------------------------
    # Per-step quantities
    # ------------------------------------------------------------------

    def _compute_goals(self):
        self.goal_pos = self.goal.data.root_pos_w.clone() - self.scene.env_origins
        self.goal_quat = self.goal.data.root_quat_w

    def _refresh_and_read_tcp(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Current TCP pose in env-local frame, forced fresh.

        Must go through the ee frame transformer rather than the articulation body
        state, for two measured reasons:

        - ``Kinematics`` drives ``panda_link7`` and adds ``tcp_offset`` in the *world*
          frame, while the frame transformer applies that same 0.2418 m offset
          *body-locally*. The two agree only when the gripper happens to be axis
          aligned; otherwise they differ by up to the offset length.
        - ``_resolve_ee_body_idx()`` returns ``robotiq_arg2f_base_link``, which sits a
          further fixed 0.1348 m from the TCP.

        Reading the frame transformer keeps the goal in exactly the frame
        ``_compute_intermediate_values`` uses for ``tcp_pos``, so ``pos_error`` is
        self-consistent. It is stale immediately after ``ik_set_joint_pos`` writes
        joint state -- ``update()`` is what makes it valid without a physics step.
        """
        self.ee_frame.update(self.sim.get_physics_dt())
        pos_w = self.ee_frame.data.target_pos_w[:, 0]
        quat_w = self.ee_frame.data.target_quat_w[:, 0]
        return pos_w - self.scene.env_origins, quat_w

    @staticmethod
    def _point_segment_distance(
        pts: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
    ) -> torch.Tensor:
        """Distance from each point in ``pts`` (N, D, 3) to segment [a, b] (N, 3)."""
        ab = (b - a).unsqueeze(1)  # (N, 1, 3)
        ap = pts - a.unsqueeze(1)  # (N, D, 3)
        denom = (ab * ab).sum(-1, keepdim=True).clamp(min=1e-9)  # (N, 1, 1)
        t = ((ap * ab).sum(-1, keepdim=True) / denom).clamp(0.0, 1.0)  # (N, D, 1)
        closest = a.unsqueeze(1) + t * ab  # (N, D, 3)
        return (pts - closest).norm(dim=-1)  # (N, D)

    def _select_attended(self, obstacle_pos: torch.Tensor) -> torch.Tensor:
        """Pick which obstacle slots the policy sees this step.

        Scored by distance to the [tcp, goal] corridor rather than to the TCP, so an
        obstacle sitting in the path is selected even when it is far from the gripper.
        Hysteresis keeps a slot in the set for ``prune_hysteresis_steps`` once chosen,
        so tokens do not flicker as the EE moves.
        """
        k = min(self.cfg.num_attended_obstacles, self.cfg.num_obstacles)
        d = self._point_segment_distance(obstacle_pos, self.tcp_pos, self.goal_pos)
        # Inactive slots must never win a top-k slot.
        d = torch.where(self.active_obstacle_mask, d, torch.full_like(d, float("inf")))
        idx = d.topk(k, dim=-1, largest=False).indices  # (N, k)
        chosen = torch.zeros_like(self.active_obstacle_mask)
        chosen.scatter_(1, idx, True)
        chosen &= self.active_obstacle_mask  # drop inf picks

        if self.cfg.prune_hysteresis_steps > 0:
            self._attend_ttl = torch.where(
                chosen,
                torch.full_like(self._attend_ttl, self.cfg.prune_hysteresis_steps),
                (self._attend_ttl - 1).clamp(min=0),
            )
            chosen = (chosen | (self._attend_ttl > 0)) & self.active_obstacle_mask
        return chosen

    def _compute_intermediate_values(self):
        self.tcp_pos = self.ee_frame.data.target_pos_w[:, 0] - self.scene.env_origins
        self.tcp_quat = self.ee_frame.data.target_quat_w[:, 0]

        self.pos_error, self.axis_angle_error = math.compute_pose_error(
            self.tcp_pos,
            self.tcp_quat,
            self.goal_pos,
            self.goal_quat,
        )
        self.reached_pose = (self.pos_error.norm(dim=-1) < self.cfg.distance_threshold) & (
            self.axis_angle_error.norm(dim=-1) < self.cfg.orientation_threshold
        )

        num_obs = len(self.obstacles)
        obs_pos_w = torch.stack([o.data.root_pos_w for o in self.obstacles], dim=1)  # (N, D, 3)
        obs_quat_w = torch.stack([o.data.root_quat_w for o in self.obstacles], dim=1)  # (N, D, 4)
        obstacle_pos = obs_pos_w - self.scene.env_origins.unsqueeze(1)  # env-local

        self.attended_obstacle_mask = self._select_attended(obstacle_pos)

        half_sizes = torch.stack(
            [
                (obs_utils.get_bounding_box(o)[:, 1] - obs_utils.get_bounding_box(o)[:, 0]) / 2.0
                for o in self.obstacles
            ],
            dim=1,
        )  # (N, D, 3)
        signs = obs_utils.ENDPOINT_SIGNS.to(self.device)  # (8, 3)
        K = 8
        endpoints_local = signs[None, None] * half_sizes[:, :, None, :]  # (N, D, 8, 3)
        quat_expanded = obs_quat_w.unsqueeze(2).expand(-1, -1, K, -1).reshape(-1, 4)
        pos_expanded = obs_pos_w.unsqueeze(2).expand(-1, -1, K, -1).reshape(-1, 3)
        all_corners_w = (
            pos_expanded + math.quat_apply(quat_expanded, endpoints_local.reshape(-1, 3))
        ).reshape(self.num_envs, num_obs * K, 3)
        corners_rel_ee = self._transform_corners_to_frame(self.ee_frame, all_corners_w)

        # Zero unattended slots. AttentionMLPAgent infers active tokens from
        # ``tokens.abs().sum(-1) > 0``, so zeroing is exactly how pruning is expressed.
        attended_corners_mask = self.attended_obstacle_mask.repeat_interleave(K * 3, dim=1)
        self.obstacles_corner_pts_rel_ee = corners_rel_ee * attended_corners_mask

        # Displacement from spawn, over physically present slots.
        flat_pos = obs_pos_w.reshape(-1, 3)
        flat_quat = obs_quat_w.reshape(-1, 4)
        pos_delta, rot_delta = math.compute_pose_error(
            self.obstacles_init_pos.reshape(-1, 3),
            self.obstacles_init_quat.reshape(-1, 4),
            flat_pos,
            flat_quat,
        )
        obstacles_pos_error = pos_delta.reshape(self.num_envs, -1, 3).norm(dim=-1)
        self.obstacles_moved = (
            (obstacles_pos_error > self.cfg.distance_threshold) & self.active_obstacle_mask
        ).any(dim=-1)

        # Clearance barrier, following push's formulation but EE-only (no carried block).
        obstacle_r = self.block_sizes[:, 0:1] / 2.0
        margin = self.cfg.clearance_safety_margin
        ee_clear = (self.tcp_pos[:, None, :] - obstacle_pos).norm(dim=-1) - obstacle_r  # (N, D)
        barrier = (1.0 - ee_clear / margin).clamp(0.0, 1.0) * self.active_obstacle_mask.float()
        self.clearance_penalty = barrier.max(dim=-1).values

        # How close is the nearest obstacle we are NOT showing the policy? Fed to the
        # critic and logged, so pruning cannot silently hide a collision cause.
        unattended = self.active_obstacle_mask & ~self.attended_obstacle_mask
        ee_clear_unattended = torch.where(
            unattended,
            ee_clear,
            torch.full_like(ee_clear, float("inf")),
        )
        self.pruned_min_clearance = ee_clear_unattended.min(dim=-1).values
        self.pruned_min_clearance = torch.where(
            torch.isinf(self.pruned_min_clearance),
            torch.ones_like(self.pruned_min_clearance),
            self.pruned_min_clearance,
        )

        # Contact. force_matrix_w is (N, bodies, filters, 3); reduce over filters.
        forces = [
            s.data.force_matrix_w.norm(dim=-1).amax(dim=(1, 2)) for s in self._arm_contact_sensors
        ]
        self.arm_collision_force = torch.stack(forces, dim=0).amax(dim=0)  # (N,)
        self.table_penetration = self.arm_collision_force  # table is one of the filters
        self.collided |= self.arm_collision_force > self.cfg.collision_force_threshold

        self.acheived_goal = self.reached_pose & ~self.collided & ~self.obstacles_moved

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs_extra(self) -> dict:
        return dict(
            # WARNING: obstacles_corner_pts_rel_ee must stay FIRST -- AttentionMLPAgent
            # slices the token block by offset. Don't reorder.
            obstacles_corner_pts_rel_ee=self.obstacles_corner_pts_rel_ee.reshape(self.num_envs, -1),
            tcp_pos=self.tcp_pos,
            tcp_quat=self.tcp_quat,
            pruned_min_clearance=self.pruned_min_clearance.unsqueeze(-1),
        )

    def _get_obs_goal(self) -> dict:
        return dict(
            pos_error=self.pos_error,
            axis_angle_error=self.axis_angle_error,
        )

    def _get_obs_privileged(self) -> dict:
        return dict(
            table_height_offset=self.table_height_offset.unsqueeze(-1),
            table_friction=self.table_friction,
            block_side_length=self.block_sizes[:, 0:1],
            num_active_obstacles=self.active_obstacle_mask.float().sum(-1, keepdim=True),
            clearance_penalty=self.clearance_penalty.unsqueeze(-1),
            pruned_min_clearance=self.pruned_min_clearance.unsqueeze(-1),
        )

    # ------------------------------------------------------------------
    # Reward / evaluation
    # ------------------------------------------------------------------

    def _get_rewards(self) -> torch.Tensor:
        pos_dist = self.pos_error.norm(dim=-1)
        reach_reward = 1 - torch.tanh(pos_dist * 5)
        reach_fine_reward = torch.exp(-((pos_dist / 0.03) ** 2))
        ang = self.axis_angle_error.norm(dim=-1)
        orientation_reward = (torch.exp(-((ang / 0.1) ** 2)) + torch.exp(-((ang / 0.5) ** 2))) / 2
        success_reward = self.acheived_goal.float()
        action_magnitude_penalty = -self.actions.norm(dim=-1)

        denom = (
            self.cfg.reach_reward_coef
            + self.cfg.reach_fine_reward_coef
            + self.cfg.orientation_reward_coef
            + self.cfg.success_reward_coef
        )
        reward = (
            self.cfg.reach_reward_coef * reach_reward
            + self.cfg.reach_fine_reward_coef * reach_fine_reward
            + self.cfg.orientation_reward_coef * orientation_reward
            + self.cfg.success_reward_coef * success_reward
            - self.cfg.collision_penalty_coef * self.collided.float()
            - self.cfg.obstacle_moved_penalty_coef * self.obstacles_moved.float()
            - self.cfg.clearance_penalty_coef * self.clearance_penalty
            + self.cfg.action_magnitude_reward_coef * action_magnitude_penalty
        ) / denom
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated, truncated = super()._get_dones()
        if self.cfg.terminate_on_collision:
            terminated = terminated | self.collided
        return terminated, truncated

    def evaluate(self) -> dict:
        result = super().evaluate()
        result["success"] = self.acheived_goal
        stats = result.setdefault("stats", {})
        stats["pos_error"] = self.pos_error.norm(dim=-1)
        stats["axis_angle_error"] = self.axis_angle_error.norm(dim=-1)
        stats["reached_pose"] = self.reached_pose
        stats["collided"] = self.collided.float()
        stats["obstacles_moved"] = self.obstacles_moved.float()
        stats["clearance_penalty"] = self.clearance_penalty
        stats["num_active_obstacles"] = self.active_obstacle_mask.float().sum(-1)
        stats["num_attended_obstacles"] = self.attended_obstacle_mask.float().sum(-1)
        stats["pruned_min_clearance"] = self.pruned_min_clearance
        # Is the straight line actually blocked? If this trends to zero the policy is
        # learning to travel straight, not to avoid.
        stats["corridor_blocked"] = self._corridor_blocked.float()
        stats["action_magnitude"] = self.actions.norm(dim=-1)
        return result

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def _sample_downward_quat(self, b: int) -> torch.Tensor:
        """Orientation about a downward-pointing gripper, as push does for its start."""
        yaw = randomization_utils.random_quaterion(
            b,
            (0.0, 0.0),
            (self.cfg.goal_pitch_range[0], self.cfg.goal_pitch_range[1]),
            (self.cfg.goal_yaw_range[0], self.cfg.goal_yaw_range[1]),
            self.device,
        )
        down = torch.tensor([0.0, 1.0, 0.0, 0.0], device=self.device).repeat(b, 1)
        return math.quat_mul(yaw, down)

    def _sample_ee_pos(self, b: int, x_range, y_range, z_range) -> torch.Tensor:
        # z is relative to the table surface; callers add table_height_offset.
        pos = torch.zeros(b, 3, device=self.device)
        for i, (lo, hi) in enumerate((x_range, y_range, z_range)):
            pos[:, i] = torch.empty(b, device=self.device).uniform_(lo, hi)
        return pos

    def _ik_with_resample(
        self,
        env_ids: torch.Tensor,
        pos: torch.Tensor,
        quat: torch.Tensor,
        x_range,
        y_range,
        z_range,
        seed: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """IK to ``pos``/``quat``, resampling the request where it did not converge.

        ``ik_set_joint_pos`` performs no reachability or collision check, so a converged
        solution is the only evidence the pose is attainable. Returns the *achieved*
        pose, which is what callers should treat as the commanded target.
        """
        for _ in range(self.cfg.ik_max_resamples):
            self.kinematics.ik_set_joint_pos(
                env_ids,
                pos + self.scene.env_origins[env_ids],
                quat,
                seed_joint_pos=seed,
            )
            achieved_pos, achieved_quat = self._refresh_and_read_tcp()
            residual = (achieved_pos[env_ids] - pos).norm(dim=-1)
            bad = residual > self.cfg.ik_residual_tolerance
            if not bad.any():
                break
            n_bad = int(bad.sum())
            resampled = self._sample_ee_pos(n_bad, x_range, y_range, z_range)
            pos = pos.clone()
            pos[bad] = resampled
            quat = quat.clone()
            quat[bad] = self._sample_downward_quat(n_bad)
        achieved_pos, achieved_quat = self._refresh_and_read_tcp()
        return achieved_pos[env_ids], achieved_quat[env_ids]

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self.collided[env_ids] = False
        self._attend_ttl[env_ids] = 0
        with torch.device(self.device):
            b = len(env_ids)
            table_z = self.table_height_offset[env_ids]

            # --- Goal: IK to a sampled pose, keep what was achieved -------------
            # Using the achieved pose makes the goal reachable by construction, so no
            # separate reachability filter (or motion planner) is needed for it.
            goal_req_pos = self._sample_ee_pos(
                b,
                self.cfg.goal_ee_x_range,
                self.cfg.goal_ee_y_range,
                self.cfg.goal_ee_z_range,
            )
            goal_req_pos[:, 2] += table_z
            goal_req_quat = self._sample_downward_quat(b)
            goal_pos, goal_quat = self._ik_with_resample(
                env_ids,
                goal_req_pos,
                goal_req_quat,
                self.cfg.goal_ee_x_range,
                self.cfg.goal_ee_y_range,
                self.cfg.goal_ee_z_range,
                seed=self.kinematics.articulation.data.default_joint_pos[env_ids],
            )
            self.goal.write_root_pose_to_sim(
                torch.cat([goal_pos + self.scene.env_origins[env_ids], goal_quat], dim=-1),
                env_ids=env_ids,
            )

            # --- Start: some at home, the rest spread over the workspace --------
            start_req_pos = self._sample_ee_pos(
                b,
                self.cfg.start_ee_x_range,
                self.cfg.start_ee_y_range,
                self.cfg.start_ee_z_range,
            )
            start_req_pos[:, 2] += table_z
            # Keep start and goal apart, else the initial pose already solves the task.
            away = (start_req_pos - goal_pos).norm(dim=-1) < self.cfg.min_start_goal_distance
            if away.any():
                push_dir = start_req_pos - goal_pos
                push_dir = push_dir / push_dir.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                start_req_pos = torch.where(
                    away.unsqueeze(-1),
                    goal_pos + push_dir * self.cfg.min_start_goal_distance,
                    start_req_pos,
                )
            start_req_quat = self._sample_downward_quat(b)
            start_pos, _ = self._ik_with_resample(
                env_ids,
                start_req_pos,
                start_req_quat,
                self.cfg.start_ee_x_range,
                self.cfg.start_ee_y_range,
                self.cfg.start_ee_z_range,
                seed=self.kinematics.articulation.data.default_joint_pos[env_ids],
            )
            at_home = torch.rand(b, device=self.device) < self.cfg.start_at_home_fraction
            if at_home.any():
                home = self.kinematics.articulation.data.default_joint_pos[env_ids]
                home_ids = (
                    env_ids[at_home]
                    if torch.is_tensor(env_ids)
                    else torch.tensor(env_ids, device=self.device)[at_home]
                )
                self.kinematics.articulation.write_joint_state_to_sim(
                    position=home[at_home][:, self.kinematics.joint_ids],
                    velocity=torch.zeros(int(at_home.sum()), self.kinematics.n_joints),
                    joint_ids=self.kinematics.joint_ids,
                    env_ids=home_ids,
                )
                start_pos = torch.where(
                    at_home.unsqueeze(-1),
                    self._refresh_and_read_tcp()[0][env_ids],
                    start_pos,
                )

            # --- Obstacles ------------------------------------------------------
            num_active = torch.randint(
                self.cfg.min_num_obstacles,
                self.cfg.num_obstacles + 1,
                (b,),
                device=self.device,
            )
            active_mask = (
                torch.arange(self.cfg.num_obstacles, device=self.device)[None, :]
                < num_active[:, None]
            )
            stack_height = torch.randint(1, self.cfg.max_stack_height + 1, (b,), device=self.device)

            n_slots = self.cfg.num_obstacles
            pos_w = torch.zeros(b, n_slots, 3, device=self.device)
            pos_w[:, :, 0] = torch.empty(b, n_slots, device=self.device).uniform_(
                self.cfg.obstacle_spawn_x_range[0],
                self.cfg.obstacle_spawn_x_range[1],
            )
            pos_w[:, :, 1] = torch.empty(b, n_slots, device=self.device).uniform_(
                self.cfg.obstacle_spawn_y_range[0],
                self.cfg.obstacle_spawn_y_range[1],
            )

            # Seed some obstacles onto the [start, goal] corridor. Without this the
            # straight line is usually clear and no avoidance behaviour is required.
            obstructed = torch.rand(b, device=self.device) < self.cfg.obstructed_fraction
            n_corridor = (
                torch.randint(
                    self.cfg.obstructed_count_range[0],
                    self.cfg.obstructed_count_range[1] + 1,
                    (b,),
                    device=self.device,
                )
                * obstructed.long()
            )
            self._corridor_blocked = torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._corridor_blocked[env_ids] = obstructed & (num_active > 0)
            slot_idx = torch.arange(n_slots, device=self.device)[None, :]
            in_corridor = (slot_idx < n_corridor[:, None]) & active_mask  # (b, n_slots)
            t = torch.rand(b, n_slots, device=self.device) * 0.7 + 0.15  # avoid the endpoints
            line = start_pos[:, None, :2] + t.unsqueeze(-1) * (
                goal_pos[:, None, :2] - start_pos[:, None, :2]
            )
            perp = goal_pos[:, None, :2] - start_pos[:, None, :2]
            perp = torch.stack([-perp[..., 1], perp[..., 0]], dim=-1)
            perp = perp / perp.norm(dim=-1, keepdim=True).clamp(min=1e-6)
            jitter = torch.empty(b, n_slots, 1, device=self.device).uniform_(
                -self.cfg.corridor_perp_offset,
                self.cfg.corridor_perp_offset,
            )
            corridor_xy = line + perp * jitter
            pos_w[:, :, :2] = torch.where(in_corridor.unsqueeze(-1), corridor_xy, pos_w[:, :, :2])

            pos_w[:, :, 2] = (self.block_sizes[env_ids, 2] / 2 + 5e-4 + table_z).unsqueeze(1)
            pos_w += self.scene.env_origins[env_ids, None, :]

            # Relaxation-based non-overlap (push's approach; rejection sampling with
            # per-env retry counts does not vectorize). Third reference is the arm's
            # start position -- push does not need it because its arm starts above the
            # block by construction, but wide start sampling will spawn blocks into it.
            min_clearance = self.block_sizes[env_ids, 0].unsqueeze(-1).unsqueeze(-1) * 2.0
            for ref_xy in [
                (goal_pos + self.scene.env_origins[env_ids])[:, :2],
                (start_pos + self.scene.env_origins[env_ids])[:, :2],
            ]:
                diff = pos_w[:, :, :2] - ref_xy[:, None, :]
                dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                push = (min_clearance / dist - 1.0).clamp(min=0.0) * diff
                pos_w[:, :, :2] += push

            no_self = ~torch.eye(n_slots, device=self.device, dtype=torch.bool)
            pair_active = active_mask[:, :, None] * active_mask[:, None, :] * no_self[None]
            for _ in range(10):
                xy = pos_w[:, :, :2]
                diff = xy[:, :, None, :] - xy[:, None, :, :]
                dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                push = (
                    (min_clearance.unsqueeze(1) / dist - 1.0).clamp(min=0.0)
                    * pair_active.unsqueeze(-1)
                    * diff
                )
                pos_w[:, :, :2] += push.sum(dim=2)

            # Inactive slots rest on the ground plane, spaced in y so they don't overlap.
            underground_base = self.scene.env_origins[env_ids].clone()
            underground_base[:, 0] += 1.0
            underground_base[:, 2] = -1.05 + self.block_sizes[env_ids, 2] / 2

            # Per-env permutation so a given physical object lands in a different
            # logical slot each episode; without it the policy keys off slot order.
            perm = torch.stack([torch.randperm(n_slots, device=self.device) for _ in range(b)])
            arange_b = torch.arange(b, device=self.device)
            self.active_obstacle_mask[env_ids] = torch.gather(active_mask, 1, perm)

            yaw_quat = randomization_utils.random_quaterion(
                b,
                (0.0, 0.0),
                (0.0, 0.0),
                (-np.pi, np.pi),
                self.device,
            )
            for i, obstacle_i in enumerate(self.obstacles):
                pi = perm[:, i]
                is_active = active_mask[arange_b, pi]
                is_stacked = pi < stack_height
                stacked_pos = pos_w[:, 0].clone()
                stacked_pos[:, 2] = (
                    pos_w[:, 0, 2] + (self.block_sizes[env_ids, 2] + 2e-3) * pi.float() + 5e-4
                )
                slot = (pi - stack_height + 1).clamp(0, n_slots - 1)
                free_pos = pos_w[arange_b, slot]
                active_pos = torch.where(is_stacked[:, None], stacked_pos, free_pos)
                underground_pos = underground_base.clone()
                underground_pos[:, 1] += i * 0.1
                pos_i = torch.where(is_active[:, None], active_pos, underground_pos)
                pose_i = torch.cat([pos_i, yaw_quat], dim=-1)
                obstacle_i.write_root_pose_to_sim(pose_i, env_ids=env_ids)
                obstacle_i.write_root_velocity_to_sim(torch.zeros(b, 6), env_ids=env_ids)
                self.obstacles_init_pos[env_ids, i] = pose_i[:, :3]
                self.obstacles_init_quat[env_ids, i] = pose_i[:, 3:]
