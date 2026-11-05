from torch._tensor import Tensor
import torch
from collections.abc import Sequence
from toyblocks.robots.kinematics import Kinematics, KinematicsCfg
from toyblocks.tasks.base_env import BaseEnv
from toyblocks.utils import obs_utils, rigid_objects
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from .rotate_env_cfg import RotateEnvCfg
from isaaclab.utils import math
from toyblocks.utils import randomization_utils
import numpy as np
class RotateEnv(BaseEnv):
    cfg: RotateEnvCfg

    def _setup_scene_objects(self):
        block_sizes = torch.zeros(self.num_envs, 3, device=self.device)
        block_sizes[:, :] = torch.rand(self.num_envs, 1, device=self.device) * (self.cfg.block_size_range[1] - self.cfg.block_size_range[0]) + self.cfg.block_size_range[0]
        if self.cfg.randomize_block_size:
            from toyblocks.utils.rigid_objects.spawn_indexed import bucket_isotropic_cube_sizes
            sides = block_sizes[:, 0].cpu().tolist()
            indices, bucket_sizes = bucket_isotropic_cube_sizes(
                sides, self.cfg.block_size_num_buckets, self.cfg.block_size_range,
            )
            quantized_sides = torch.tensor(
                [bucket_sizes[i][0] for i in indices],
                device=self.device, dtype=torch.float32,
            )
            block_sizes[:, :] = quantized_sides.unsqueeze(1)
        self.block_sizes = block_sizes if self.cfg.randomize_block_size else torch.tensor(self.cfg.block_size, device=self.device, dtype=torch.float32).unsqueeze(0).expand(self.num_envs, -1).clone()
        block_sizes_list = self.block_sizes.cpu().tolist()
        bucket_kw = dict(
            max_unique_assets=self.cfg.block_size_num_buckets,
            size_range=self.cfg.block_size_range,
        ) if self.cfg.randomize_block_size else {}
        self.block = rigid_objects.create_toy_block(
            prim_path="/World/envs/env_.*/Block",
            size=block_sizes_list if self.cfg.randomize_block_size else self.cfg.block_size,
            init_pos=(0.0, 0.0, self.cfg.block_size[2] / 2),
            **bucket_kw,
        )
        self.scene.rigid_objects["block"] = self.block

        self.goal = rigid_objects.create_toy_block(
            prim_path="/World/envs/env_.*/Goal",
            size=block_sizes_list if self.cfg.randomize_block_size else self.cfg.block_size,
            init_pos=(0.0, 0.0, 0.0),
            kinematic=True,
            collision=False,
            **bucket_kw,
        )
        self.scene.rigid_objects["goal"] = self.goal

        # Contact sensors: each finger vs the table surface.
        env_ns = self.scene.env_regex_ns
        table_filter = [f"{env_ns}/workspace_table"]
        for finger in ["left_inner_finger", "right_inner_finger"]:
            cfg = ContactSensorCfg(
                prim_path=f"{env_ns}/Robot/panda/{finger}",
                filter_prim_paths_expr=table_filter,
                track_air_time=False,
                history_length=1,
            )
            self.scene.sensors[f"contact:finger_{finger}"] = ContactSensor(cfg)
        
    def _after_scene_setup(self):
        super()._after_scene_setup()
        self._finger_contact_sensors = [
            self.scene.sensors["contact:finger_left_inner_finger"],
            self.scene.sensors["contact:finger_right_inner_finger"],
        ]

        # reach-curriculum latch: True once the tcp has reached the block this episode (see _get_rewards)
        self._reach_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def ee_frame(self) -> FrameTransformer:
        return self.scene.sensors["frame_transformer:ee"]

    def _update_visualization_markers(self):
        self.block_arrow.visualize(self.block.data.root_pos_w, self.block.data.root_quat_w)
        self.goal_arrow.visualize(self.goal.data.root_pos_w, self.goal.data.root_quat_w)
        corner_translations = (self.block_corner_pts + self.scene.env_origins.unsqueeze(1)).reshape(self.num_envs * 8, 3)
        corner_indices = list(range(8)) * self.num_envs
        self.cube_corner_visuals.visualize(translations=corner_translations, marker_indices=corner_indices)
        goal_corner_translations = (self.goal_corner_pts + self.scene.env_origins.unsqueeze(1)).reshape(self.num_envs * 8, 3)
        goal_corner_indices = list(range(8)) * self.num_envs
        self.goal_corner_visuals.visualize(translations=goal_corner_translations, marker_indices=goal_corner_indices)

    @property
    def visualization_markers(self) -> dict:
        return {
            "block_arrow": self.block_arrow,
            "goal_arrow": self.goal_arrow,
            "cube_corners": self.cube_corner_visuals,
            "goal_cube_corners": self.goal_corner_visuals
          }
          
    def _compute_goals(self):
        self.goal_pos = self.goal.data.root_pos_w
        self.goal_quat = self.goal.data.root_quat_w
        self.goal_pos[:, 2] = self.block_sizes[:, 2] / 2 + self.table_height_offset


    def _compute_intermediate_values(self):
        self.object_pos = self.block.data.root_pos_w - self.scene.env_origins
        self.object_quat = self.block.data.root_quat_w
        self.tcp_pos = self.ee_frame.data.target_pos_w[:, 0] - self.scene.env_origins
        self.tcp_quat = self.ee_frame.data.target_quat_w[:, 0]
        self.object_to_tcp = self.tcp_pos - self.object_pos
        
        self.pos_error, self.axis_angle_error = math.compute_pose_error(self.object_pos, self.object_quat, self.goal_pos - self.scene.env_origins, self.goal_quat)
        self.block_corner_pts = obs_utils.cube_corner_points(self.block) - self.scene.env_origins.unsqueeze(1)
        self.goal_corner_pts = obs_utils.cube_corner_points(self.block, self.goal_pos, self.goal_quat) - self.scene.env_origins.unsqueeze(1)

        self.block_corner_pts_to_goal_corner_pts_dist = (self.block_corner_pts - self.goal_corner_pts).norm(dim=-1).mean(dim=-1)
        self.reached_pose = (self.pos_error.norm(dim=-1) < self.cfg.distance_threshold) & (self.axis_angle_error.norm(dim=-1) < self.cfg.orientation_threshold)

        # Block corners expressed in the goal frame — invariant to absolute world-frame
        # orientation, so the policy sees the same representation across all 24 block symmetries.
        q_inv = math.quat_inv(self.goal_quat)  # (N, 4)
        delta = self.block_corner_pts - self.object_pos.unsqueeze(1)  # (N, 8, 3)
        q_inv_exp = q_inv.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)  # (N*8, 4)
        self.block_corners_in_goal_frame = math.quat_apply(q_inv_exp, delta.reshape(-1, 3)).reshape(self.num_envs, -1)  # (N, 24)

        # Table contact (recomputed every step, like align/push). Computing this once in _after_scene_setup
        # left table_penetration frozen at its startup value (no contact), so table_collision was always 0.
        left_force  = self._finger_contact_sensors[0].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)  # (num_envs,)
        right_force = self._finger_contact_sensors[1].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)  # (num_envs,)
        self.table_penetration = left_force + right_force  # (num_envs,), >0 when fingers touch table

    def _get_obs_extra(self) -> dict:
        return dict(
            object_to_tcp=self.object_to_tcp,
            tcp_pos=self.tcp_pos,
            tcp_quat=self.tcp_quat,
        )

    def _get_obs_goal(self) -> dict:
        return dict(
            # block_corner_pts_to_goal_corner_pts=(self.block_corner_pts - self.goal_corner_pts).reshape(self.num_envs, -1),
            block_corners_in_goal_frame=self.block_corners_in_goal_frame,
            pos_error=self.pos_error,
            axis_angle_error=self.axis_angle_error,
        )

    def _get_obs_privileged(self) -> dict:
        # Privileged domain-randomization params seen only by the critic (asymmetric actor-critic).
        return dict(
            table_height_offset=self.table_height_offset.unsqueeze(-1),
            table_friction=self.table_friction,
            block_friction=self.block_friction,
            block_side_length=self.block_sizes[:, 0:1],
        )

    def _get_rewards(self) -> torch.Tensor:
        surface_to_tcp = self.object_to_tcp.clone()
        # target a point 2cm below the block's top face
        surface_to_tcp[:, 2] -= self.block_sizes[:, 2] / 2 - 0.01
        dist_to_block = surface_to_tcp.norm(dim=-1)
        reach_reward = (1 - torch.tanh(dist_to_block * 5))
        # reach reward is only active for the first part of the episode, then it's disabled.
        self._reach_achieved |= dist_to_block < self.cfg.reach_success_threshold
        reach_reward = reach_reward * (~self._reach_achieved).float()

        corner_reward = torch.exp(-(self.block_corner_pts_to_goal_corner_pts_dist / 0.1)**2)
        corner_reward2 = torch.exp(-(self.block_corner_pts_to_goal_corner_pts_dist / 0.04)**2)
        axis_angle_reward1 = torch.exp(-(self.axis_angle_error.norm(dim=-1) / 0.04)**2)
        axis_angle_reward2 = torch.exp(-(self.axis_angle_error.norm(dim=-1))**2)
        axis_angle_reward = (axis_angle_reward1 + axis_angle_reward2) / 2.0
        success_reward = self.reached_pose.float()
        action_magnitude_penalty = -self.actions.norm(dim=-1)
        positive_reward_sum = self.cfg.reach_reward_coef + self.cfg.corner_reward_coef + self.cfg.corner_reward2_coef + self.cfg.axis_angle_reward_coef + self.cfg.success_reward_coef
        # Once reach is latched off, drop its coefficient from the (per-env) denominator
        denom = positive_reward_sum - self.cfg.reach_reward_coef * self._reach_achieved.float()
        aligning = self._reach_achieved.float()
        reward = (
            self.cfg.reach_reward_coef * reach_reward
            + aligning * (
            self.cfg.corner_reward_coef * corner_reward + \
            self.cfg.corner_reward2_coef * corner_reward2 + \
            self.cfg.axis_angle_reward_coef * axis_angle_reward + \
            self.cfg.success_reward_coef * success_reward
            - self.cfg.table_collision_penalty_coef * (self.table_penetration > 0).float()
            )
            + self.cfg.action_magnitude_reward_coef * action_magnitude_penalty
        ) / denom
        return reward


    def evaluate(self) -> dict:
        result = super().evaluate()
        result["success"] = self.reached_pose
        stats = result.setdefault("stats", {})
        stats["object_to_tcp_dist"] = self.object_to_tcp.norm(dim=-1)
        stats["pos_error"] = self.pos_error[:, :2].norm(dim=-1)
        stats["axis_angle_error"] = self.axis_angle_error.norm(dim=-1)
        stats["block_corner_pts_to_goal_corner_pts_dist"] = self.block_corner_pts_to_goal_corner_pts_dist
        stats["table_collision"] = (self.table_penetration > 0).float()
        stats["reach_achieved"] = self._reach_achieved.float()
        stats["action_magnitude"] = self.actions.norm(dim=-1)
        return result

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._reach_achieved[env_ids] = False
        with torch.device(self.device):
            b = len(env_ids)
            block_pose = torch.zeros(b, 7)
            block_pose[:, 3:] = randomization_utils.random_cube_orientation_symmetry(b, self.device)
            rand_yaw_quat = randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (self.cfg.block_spawn_yaw_range[0], self.cfg.block_spawn_yaw_range[1]), self.device)
            block_pose[:, 3:] = math.quat_mul(rand_yaw_quat, block_pose[:, 3:])
            block_pose[:, 0] = torch.rand(b) * (self.cfg.block_spawn_x_range[1] - self.cfg.block_spawn_x_range[0]) + self.cfg.block_spawn_x_range[0]
            block_pose[:, 1] = torch.rand(b) * (self.cfg.block_spawn_y_range[1] - self.cfg.block_spawn_y_range[0]) + self.cfg.block_spawn_y_range[0]
            block_pose[:, 2] = self.block_sizes[env_ids, 2] / 2 + 5e-4 + self.table_height_offset[env_ids]
            block_pose[:, :3] += self.scene.env_origins[env_ids]
            self.block.write_root_pose_to_sim(block_pose, env_ids=env_ids)
            self.block.write_root_velocity_to_sim(torch.zeros(b, 6), env_ids=env_ids)

            goal_pose = block_pose.clone()
            goal_pose[:, :2] += torch.randn(b, 2) * 0.01 - 0.005
            relative_quat = randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (-torch.pi, torch.pi), self.device)
            goal_pose[:, 3:] = math.quat_mul(relative_quat, block_pose[:, 3:])
            self.goal.write_root_pose_to_sim(goal_pose, env_ids=env_ids)

            ee_pose = block_pose.clone()
            ee_xy_offset = torch.empty(b, 2, device=self.device).uniform_(-0.04, 0.04)
            ee_pose[:, :2] += ee_xy_offset
            ee_pose[:, 2] += 0.07 + torch.empty(b, device=self.device).uniform_(-0.01, 0.01)
            ee_pose[:, 3:] = math.quat_mul(
                randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (-torch.pi, torch.pi), self.device),
                torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(b, 1))
            self.kinematics.ik_set_joint_pos(env_ids, ee_pose[:, :3], ee_pose[:, 3:])
            