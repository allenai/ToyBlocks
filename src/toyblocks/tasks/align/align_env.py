import torch
import numpy as np
import isaaclab.sim as sim_utils

from collections.abc import Sequence
from typing import List
from toyblocks.tasks.base_env import BaseEnv
from toyblocks.utils import obs_utils, rigid_objects
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from .align_env_cfg import AlignEnvCfg
from isaaclab.utils import math
from toyblocks.utils import randomization_utils   

class AlignEnv(BaseEnv):
    cfg: AlignEnvCfg

    block: RigidObject
    distractors: List[RigidObject]
    goal: RigidObject

    # intermediate values
    object_pos: torch.Tensor
    object_quat: torch.Tensor
    tcp_pos: torch.Tensor
    tcp_quat: torch.Tensor
    object_to_tcp: torch.Tensor
    pos_error: torch.Tensor
    block_corner_pts_to_goal_corner_pts_dist: torch.Tensor
    orientation_corner_dist: torch.Tensor

    # distractor intermediate values
    distractors_init_pos: torch.Tensor   # (num_envs, num_distractors, 3)
    distractors_init_quat: torch.Tensor  # (num_envs, num_distractors, 4)
    distractors_moved: torch.Tensor
    distractors_corner_pts_rel_ee: torch.Tensor
    distractors_pos_error: torch.Tensor  # (num_envs, num_distractors)
    distractors_rot_error: torch.Tensor  # (num_envs, num_distractors)
    active_distractor_mask: torch.Tensor # (num_envs, num_distractors)
    _reach_dist: torch.Tensor

    # evaluation / visualization
    reached_pose: torch.Tensor
    acheived_goal: torch.Tensor
    block_corner_pts: torch.Tensor
    goal_corner_pts: torch.Tensor
    block_corner_pts_to_goal_corner_pts: torch.Tensor
    block_corners_in_goal_frame: torch.Tensor
    block_corner_visuals: VisualizationMarkers
    goal_corner_visuals: VisualizationMarkers
    block_arrow: VisualizationMarkers
    goal_arrow: VisualizationMarkers

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

        distractors = []
        underground_z = -1.05 + self.cfg.block_size[2] / 2  # matches base_env.py ground plane at z=-1.05
        for i in range(self.cfg.num_distractors):
            distractor = rigid_objects.create_cuboid(
                prim_path=f"/World/envs/env_.*/Distractor_{i}",
                size=block_sizes_list if self.cfg.randomize_block_size else self.cfg.block_size,
                diffuse_color=(1.0, 0.0, 0.0),
                init_pos=(1.0, 0.0, underground_z),
                **bucket_kw,
            )
            self.scene.rigid_objects[f"distractor_{i}"] = distractor
            distractors.append(distractor)
        self.distractors = distractors

        self.goal = rigid_objects.create_toy_block(
            prim_path="/World/envs/env_.*/Goal",
            size=block_sizes_list if self.cfg.randomize_block_size else self.cfg.block_size,
            init_pos=(0.0, 0.0, self.cfg.block_size[2] / 2),
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

        colors = [(1.0, 0.0, 0.0), (1.0, 0.5, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 1.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.5)]
        
        cube_corner_cfg = VisualizationMarkersCfg(
            prim_path="/Visuals/CubeCornerVisuals",
            markers={
                f'corner_pt_visual_{i}': sim_utils.SphereCfg(
                    radius=0.0025,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(colors[i])),
                )
                for i in range(8)
            }
        )
        self.cube_corner_visuals = VisualizationMarkers(cfg=cube_corner_cfg)
        self.cube_corner_visuals.set_visibility(True)
        self.goal_corner_visuals = VisualizationMarkers(cfg=cube_corner_cfg.replace(prim_path="/Visuals/GoalCornerVisuals"))
        self.goal_corner_visuals.set_visibility(True)
        self.block_arrow = VisualizationMarkers(cfg=GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/BlockArrow"))
        self.goal_arrow = VisualizationMarkers(cfg=BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/GoalArrow"))
        self.block_arrow.set_visibility(True)
        self.goal_arrow.set_visibility(True)

    def _after_scene_setup(self):
        super()._after_scene_setup()
        self.distractors_init_pos = torch.zeros(self.num_envs, self.cfg.num_distractors, 3, device=self.device)
        self.distractors_init_quat = torch.zeros(self.num_envs, self.cfg.num_distractors, 4, device=self.device)
        self.active_distractor_mask = torch.zeros(self.num_envs, self.cfg.num_distractors, dtype=torch.bool, device=self.device)

        # Per-slot movement thresholds: slot 0 is tight, slots 1+ are loose
        self._distractor_pos_threshold = torch.full(
            (self.num_envs, self.cfg.num_distractors), self.cfg.extra_distractor_distance_threshold, device=self.device
        )
        self._distractor_pos_threshold[:, 0] = self.cfg.distractor_distance_threshold
        self._distractor_rot_threshold = torch.full(
            (self.num_envs, self.cfg.num_distractors), self.cfg.extra_distractor_orientation_threshold, device=self.device
        )
        self._distractor_rot_threshold[:, 0] = self.cfg.distractor_orientation_threshold

        self._finger_contact_sensors = [
            self.scene.sensors["contact:finger_left_inner_finger"],
            self.scene.sensors["contact:finger_right_inner_finger"],
        ]

        # reach-curriculum latch: True once the tcp has reached the block this episode (see _get_rewards)
        self._reach_achieved = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Hold the gripper closed to `gripper_close_fraction` of its travel 
        self._finger_joint_ids, _ = self.robot.articulation.find_joints(["finger_joint"])
        finger_upper = self.robot.articulation.data.joint_pos_limits[0, self._finger_joint_ids, 1]  # (n_finger,)
        self._gripper_close_target = self.cfg.gripper_close_fraction * finger_upper
        self.robot.articulation.data.default_joint_pos[:, self._finger_joint_ids] = self._gripper_close_target.unsqueeze(0)

        # Track the true gripper contact point (fingertip midpoint) for the reach metric.
        self._inner_finger_body_ids, _ = self.robot.articulation.find_bodies(["left_inner_finger", "right_inner_finger"])
        self._finger_tip_local = torch.tensor([0.0433, 0.0, 0.1388], device=self.device)

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

    @property
    def ee_frame(self) -> FrameTransformer:
        return self.scene.sensors["frame_transformer:ee"]


    def _transform_corners_to_frame(self, frame: FrameTransformer, corner_points: torch.Tensor) -> torch.Tensor:
        """
        Transforms the corner points into the frame of the given frame transformer.
        Args:
            frame: FrameTransformer to transform the data into.
            corner_points: Corner points to transform. (N, K, 3)

        Returns:
            Transformed corner points. (N, K * 3)
        """
        N, K = corner_points.shape[:2]
        corner_points_flat = corner_points.reshape(-1, 3)
        unit_quat = torch.zeros(N * K, 4, device=corner_points.device)
        unit_quat[:, 0] = 1.0

        target_pos_w_expanded = frame.data.target_pos_w.expand(N, K, -1).reshape(-1, 3)
        target_quat_w_expanded = frame.data.target_quat_w.expand(N, K, -1).reshape(-1, 4)

        corner_points_rel_flat, _ = math.subtract_frame_transforms(
        target_pos_w_expanded, target_quat_w_expanded, corner_points_flat, unit_quat
        )

        return corner_points_rel_flat.reshape(N, K * 3)

    def _compute_goals(self):
        # TODO: rename this. This fn actually computes everything rl-env specific that tamp won't 
        # need to compute or will get from somewhere else.
        self.goal_pos = self.goal.data.root_pos_w.clone() - self.scene.env_origins
        self.goal_pos[:, 2] = self.block_sizes[:,2] / 2 + self.table_height_offset
        self.goal_quat = self.goal.data.root_quat_w
        # Table contact
        left_force  = self._finger_contact_sensors[0].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)  # (num_envs,)
        right_force = self._finger_contact_sensors[1].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)  # (num_envs,)
        self.table_penetration = left_force + right_force  # (num_envs,), >0 when fingers touch table

    def _compute_intermediate_values(self):
        self.object_pos = self.block.data.root_pos_w - self.scene.env_origins
        self.object_quat = self.block.data.root_quat_w
        self.tcp_pos = self.ee_frame.data.target_pos_w[:, 0] - self.scene.env_origins
        self.tcp_quat = self.ee_frame.data.target_quat_w[:, 0] 
        self.object_to_tcp = self.tcp_pos - self.object_pos

        # Reach target = center of the block's top face, measured from the true gripper contact point (fingertip
        # midpoint). Can't use TCP because it shifts when the gripper is closed.
        finger_body_pos = self.robot.articulation.data.body_pos_w[:, self._inner_finger_body_ids]    # (N, 2, 3)
        finger_body_quat = self.robot.articulation.data.body_quat_w[:, self._inner_finger_body_ids]  # (N, 2, 4)
        tip_local = self._finger_tip_local.view(1, 1, 3).expand(self.num_envs, 2, 3).reshape(-1, 3)
        tip_world = finger_body_pos.reshape(-1, 3) + math.quat_apply(finger_body_quat.reshape(-1, 4), tip_local)
        self.fingertip_pos = tip_world.reshape(self.num_envs, 2, 3).mean(dim=1) - self.scene.env_origins  # env-local
        top_face_center = self.object_pos.clone()
        top_face_center[:, 2] += self.block_sizes[:, 2] / 2
        reach_vec = self.fingertip_pos - top_face_center
        self._reach_dist = reach_vec.norm(dim=-1)

        self.pos_error, self.axis_angle_error = math.compute_pose_error(self.object_pos, self.object_quat, self.goal_pos, self.goal_quat)
        self.reached_pose = (self.pos_error.norm(dim=-1) < self.cfg.distance_threshold) & (self.axis_angle_error.norm(dim=-1) < self.cfg.orientation_threshold)

        self.block_corner_pts = obs_utils.cube_corner_points(self.block) - self.scene.env_origins.unsqueeze(1)

        # get all distractors' corner points in tensor (num_envs, 24 * num_distractors)
        num_dist = len(self.distractors)
        dist_pos_w = torch.stack([d.data.root_pos_w for d in self.distractors], dim=1)   # (N, num_dist, 3)
        dist_quat_w = torch.stack([d.data.root_quat_w for d in self.distractors], dim=1) # (N, num_dist, 4)
        half_sizes = torch.stack(
            [(obs_utils.get_bounding_box(d)[:, 1] - obs_utils.get_bounding_box(d)[:, 0]) / 2.0
             for d in self.distractors], dim=1
        )  # (N, num_dist, 3)
        signs = obs_utils.ENDPOINT_SIGNS.to('cuda:0' if torch.cuda.is_available() else 'cpu')  # (8, 3)
        endpoints_local = signs[None, None] * half_sizes[:, :, None, :]  # (N, num_dist, 8, 3)
        K = 8
        endpoints_flat = endpoints_local.reshape(-1, 3)
        quat_expanded = dist_quat_w.unsqueeze(2).expand(-1, -1, K, -1).reshape(-1, 4)
        pos_expanded  = dist_pos_w.unsqueeze(2).expand(-1, -1, K, -1).reshape(-1, 3)
        all_corners_w = (pos_expanded + math.quat_apply(quat_expanded, endpoints_flat)).reshape(self.num_envs, num_dist * K, 3)
        self.distractors_corner_pts_rel_ee = self._transform_corners_to_frame(self.ee_frame, all_corners_w)
        # Zero out corner observations for inactive distractor slots
        active_corners_mask = self.active_distractor_mask.repeat_interleave(K * 3, dim=1)  # (N, num_dist*K*3)
        self.distractors_corner_pts_rel_ee = self.distractors_corner_pts_rel_ee * active_corners_mask

        # see if the distractors have moved (only active slots count)
        distractors_pos, distractors_quat = obs_utils.distractors_pos_and_quat(self.distractors)
        distractor_pos_error, distractor_rot_error = math.compute_pose_error(self.distractors_init_pos.reshape(-1, 3), self.distractors_init_quat.reshape(-1, 4), distractors_pos, distractors_quat)
        self.distractors_pos_delta = distractor_pos_error.reshape(self.num_envs, -1, 3)   # (num_envs, num_distractors, 3)
        self.distractors_rot_delta = distractor_rot_error.reshape(self.num_envs, -1, 3)   # (num_envs, num_distractors, 3) axis-angle
        self.distractors_pos_error = self.distractors_pos_delta.norm(dim=-1)  # (num_envs, num_distractors)
        self.distractors_rot_error = self.distractors_rot_delta.norm(dim=-1)  # (num_envs, num_distractors)
        self.distractors_moved = (
            (self.distractors_pos_error > self._distractor_pos_threshold) & self.active_distractor_mask
        ).any(dim=-1) | (
            (self.distractors_rot_error > self._distractor_rot_threshold) & self.active_distractor_mask
        ).any(dim=-1)

        # if the task was solved or not
        self.acheived_goal = self.reached_pose & ~self.distractors_moved

        self.goal_corner_pts = obs_utils.cube_corner_points(self.block, self.goal_pos, self.goal_quat)
        self.block_corner_pts_to_goal_corner_pts_dist = (self.block_corner_pts - self.goal_corner_pts).norm(dim=-1).mean(dim=-1)

        # Block corners expressed in the goal frame — invariant to absolute world-frame
        # orientation, so the policy sees the same representation across all 24 block symmetries.
        q_inv = math.quat_inv(self.goal_quat)  # (N, 4)
        delta = self.block_corner_pts - self.object_pos.unsqueeze(1)  # (N, 8, 3)
        q_inv_exp = q_inv.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)  # (N*8, 4)
        self.block_corners_in_goal_frame = math.quat_apply(q_inv_exp, delta.reshape(-1, 3)).reshape(self.num_envs, -1)  # (N, 24)

    def _get_obs_extra(self) -> dict:
        masked_pos_delta = self.distractors_pos_delta * self.active_distractor_mask.unsqueeze(-1)
        masked_rot_delta = self.distractors_rot_delta * self.active_distractor_mask.unsqueeze(-1)
        return dict(
            # WARNING: distractors_corner_pts_rel_ee must be the first obs here for attention agent to work. Don't change the order!
            distractors_corner_pts_rel_ee=self.distractors_corner_pts_rel_ee.reshape(self.num_envs, -1),
            object_pos=self.object_pos,
            # object_quat=self.object_quat,
            object_to_tcp=self.object_to_tcp,
            tcp_pos=self.tcp_pos,
            tcp_quat=self.tcp_quat,
            # block_corner_pts=self.block_corner_pts.reshape(self.num_envs, -1),
            distractors_pos_delta=masked_pos_delta.reshape(self.num_envs, -1),
            distractors_rot_delta=masked_rot_delta.reshape(self.num_envs, -1),
        )

    def _get_obs_goal(self) -> dict:
        return dict(
            # goal_corner_pts=self.goal_corner_pts.reshape(self.num_envs, -1),
            # block_corner_pts_to_goal_corner_pts=(self.block_corner_pts - self.goal_corner_pts).reshape(self.num_envs, -1),
            block_corners_in_goal_frame=self.block_corners_in_goal_frame,
            pos_error=self.pos_error,
            axis_angle_error=self.axis_angle_error,
        )

    def _get_obs_privileged(self) -> dict:
        # Privileged params seen only by the critic (asymmetric actor-critic).
        return dict(
            table_height_offset=self.table_height_offset.unsqueeze(-1),
            table_friction=self.table_friction,
            block_friction=self.block_friction,
            block_side_length=self.block_sizes[:, 0:1],
            reach_achieved=self._reach_achieved.float().unsqueeze(-1),
        )

    def _get_rewards(self) -> torch.Tensor:
        # reach target = center of the block's top face (computed in _compute_intermediate_values).
        dist_to_block = self._reach_dist
        reach_reward = (1 - torch.tanh(dist_to_block * 5))
        # reach reward is only active for the first part of the episode, then it's disabled.
        self._reach_achieved |= dist_to_block < self.cfg.reach_success_threshold
        reach_reward = reach_reward * (~self._reach_achieved).float()

        corner_dist = self.block_corner_pts_to_goal_corner_pts_dist
        corner_reward = torch.exp(-(corner_dist / 0.15)**2)
        corner_reward2 = torch.exp(-(corner_dist / 0.03)**2)
        # Dedicated orientation term to sharpen final yaw alignment near the goal
        axis_angle_reward = (torch.exp(-(self.axis_angle_error.norm(dim=-1) / 0.1)**2) + torch.exp(-(self.axis_angle_error.norm(dim=-1) / 0.5)**2)) / 2
        # Only reward based on slot 0 (cardinal distractor next to goal) using tight sigma.
        slot0_pos_error = self.distractors_pos_error[:, 0]
        slot0_rot_error = self.distractors_rot_error[:, 0]
        distractor_reward1 = torch.exp(-(slot0_pos_error / 0.04)**2)
        distractor_reward2 = (torch.exp(-(slot0_rot_error / 0.04)**2) + torch.exp(-(slot0_rot_error / 0.5)**2)) / 2.0
        distractor_reward = (distractor_reward1 + distractor_reward2) / 2.0
        # Extra distractors (slots 1+): reward for keeping them near their spawn pose.
        extra_active = self.active_distractor_mask[:, 1:].float()          # (N, D)
        extra_count = extra_active.sum(dim=-1).clamp(min=1.0)              # (N,)
        extra_pos_rew = torch.exp(-(self.distractors_pos_error[:, 1:] / 0.05)**2)
        extra_rot_rew = torch.exp(-(self.distractors_rot_error[:, 1:] / 0.5)**2)
        extra_distractor_reward = ((extra_pos_rew + extra_rot_rew) / 2.0 * extra_active).sum(dim=-1) / extra_count
        # binary success reward
        success_reward = self.acheived_goal.float()
        action_magnitude_penalty = -self.actions.norm(dim=-1)
        positive_reward_sum = self.cfg.reach_reward_coef + self.cfg.corner_reward_coef + self.cfg.corner_reward2_coef + self.cfg.success_reward_coef + self.cfg.axis_angle_reward_coef + self.cfg.distractor_reward_coef + self.cfg.extra_distractor_reward_coef
        # Once reach is latched off, drop its coefficient from the (per-env) denominator so the
        # remaining align/success terms are not diluted by the now-dead reach slot.
        denom = positive_reward_sum - self.cfg.reach_reward_coef * self._reach_achieved.float()
        # Two-phase curriculum: before the block is reached, ONLY the reach reward is active
        aligning = self._reach_achieved.float()
        reward = (
            self.cfg.reach_reward_coef * reach_reward
            + aligning * (
                self.cfg.corner_reward_coef * corner_reward +
                self.cfg.corner_reward2_coef * corner_reward2 +
                self.cfg.success_reward_coef * success_reward +
                self.cfg.axis_angle_reward_coef * axis_angle_reward +
                self.cfg.distractor_reward_coef * distractor_reward +
                self.cfg.extra_distractor_reward_coef * extra_distractor_reward
                - self.cfg.table_collision_penalty_coef * (self.table_penetration > 0).float()
            )
            + self.cfg.action_magnitude_reward_coef * action_magnitude_penalty
        ) / denom
        return reward

    def evaluate(self) -> dict:
        result = super().evaluate()
        result["success"] = self.acheived_goal
        stats = result.setdefault("stats", {})
        stats["object_to_tcp_dist"] = self.object_to_tcp.norm(dim=-1)
        stats["reach_dist"] = self._reach_dist
        stats["pos_error"] = self.pos_error[:, :2].norm(dim=-1)
        stats["axis_angle_error"] = self.axis_angle_error.norm(dim=-1)
        stats["block_corner_pts_to_goal_corner_pts_dist"] = self.block_corner_pts_to_goal_corner_pts_dist
        stats["reached_pose"] = self.reached_pose
        stats["distractors_moved"] = self.distractors_moved
        stats["table_collision"] = (self.table_penetration > 0).float()
        stats["reach_achieved"] = self._reach_achieved.float()
        stats["action_magnitude"] = self.actions.norm(dim=-1)
        return result

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        # clear the reach-curriculum latch for the envs being reset
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

            # sample goal xy offset in a radius around the block center with some yaw randomization
            goal_pose = block_pose.clone()
            relative_quat = randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (self.cfg.goal_spawn_yaw_range[0], self.cfg.goal_spawn_yaw_range[1]), self.device)
            # Apply goal yaw in world frame (left-multiply) so the goal stays face-flat on the
            # table regardless of which 24-symmetry orientation the block was spawned in.
            goal_pose[:, 3:] = math.quat_mul(relative_quat, block_pose[:, 3:])
            # sample goal xy offset in a torus around the block center
            radius = torch.empty(b, device=self.device).uniform_(self.cfg.goal_xy_offset_min_radius, self.cfg.goal_xy_offset_max_radius)
            theta = torch.empty(b, device=self.device).uniform_(0.0, 2 * np.pi)
            goal_offset_x = radius * torch.cos(theta)
            goal_offset_y = radius * torch.sin(theta)
            goal_pose[:, 0] += goal_offset_x
            goal_pose[:, 1] += goal_offset_y
            self.goal.write_root_pose_to_sim(goal_pose, env_ids=env_ids)

            # Using this for distractors makes their spawn position/orientation invariant to A.
            goal_yaw_quat = math.quat_mul(rand_yaw_quat, relative_quat)

            # --- Distractor 0: cardinal direction next to goal (always active) ---
            cardinal_angles = torch.tensor([0.0, torch.pi/2, torch.pi, 3*torch.pi/2], device=self.device)
            angle_inds = torch.randint(0, 4, (b,), device=self.device)
            angles = cardinal_angles[angle_inds]
            distractor_offset_local = torch.zeros(b, 3, device=self.device)
            distractor_offset_local[:, 0] = torch.cos(angles) * self.cfg.distractor_offset_from_goal
            distractor_offset_local[:, 1] = torch.sin(angles) * self.cfg.distractor_offset_from_goal
            distractor_offset_w = math.quat_apply(goal_yaw_quat, distractor_offset_local)
            distractor0_pos_w = goal_pose[:, :3] + distractor_offset_w
            # Distractor 0 shares the per-env randomized cube size, so seat it with that half-height.
            distractor0_pos_w[:, 2] = self.scene.env_origins[env_ids, 2] + self.block_sizes[env_ids, 2] / 2 + 5e-4 + self.table_height_offset[env_ids]
            distractor0_pose_w = torch.cat([distractor0_pos_w, goal_yaw_quat], dim=-1)
            self.distractors[0].write_root_pose_to_sim(distractor0_pose_w, env_ids=env_ids)
            self.distractors[0].write_root_velocity_to_sim(torch.zeros(b, 6, device=self.device), env_ids=env_ids)
            self.distractors_init_pos[env_ids, 0] = distractor0_pose_w[:, :3]
            self.distractors_init_quat[env_ids, 0] = distractor0_pose_w[:, 3:]

            # resolve block/distractor-0 overlap
            xy_diff = block_pose[:, :2] - distractor0_pos_w[:, :2]
            xy_dist = xy_diff.norm(dim=-1)
            min_sep = self.block_sizes[env_ids, 0]
            overlap = xy_dist < min_sep
            if overlap.any():
                push_dir = xy_diff / xy_dist.clamp(min=1e-6).unsqueeze(-1)
                correction = (min_sep - xy_dist).clamp(min=0.0)
                block_pose[overlap, :2] += push_dir[overlap] * correction[overlap].unsqueeze(-1)
                self.block.write_root_pose_to_sim(block_pose, env_ids=env_ids)

            ee_pose = block_pose.clone()
            # EE xy randomization wrt toy block
            ee_pose[:, 0] += torch.empty(b, device=self.device).uniform_(-0.04, 0.04)
            ee_pose[:, 1] += torch.empty(b, device=self.device).uniform_(-0.04, 0.04)
            ee_pose[:, 2] += 0.07 + torch.empty(b, device=self.device).uniform_(-0.01, 0.01)
            ee_pose[:, 3:] = math.quat_mul(
                randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (-torch.pi, torch.pi), self.device),
                torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(b, 1))
            self.kinematics.ik_set_joint_pos(env_ids, ee_pose[:, :3], ee_pose[:, 3:])

            # --- Distractors 1-8: additional scattered + stacked distractors ---
            num_additional = self.cfg.num_distractors - 1  # slots 1..num_distractors-1
            stack_height = torch.randint(1, self.cfg.max_stack_height + 1, (b,), device=self.device)

            # Sample how many additional distractors are active (1 to num_additional)
            # num_active_additional = torch.zeros(b, device=self.device, dtype=torch.int32)
            num_active_additional = torch.randint(1, num_additional + 1, (b,), device=self.device)
            additional_active = torch.arange(num_additional, device=self.device)[None, :] < num_active_additional[:, None]  # (b, num_additional)

            # Sample unique xy positions for the additional distractors
            unique_pos_w = torch.zeros(b, num_additional, 3, device=self.device)
            unique_pos_w[:, :, 0] = torch.empty(b, num_additional, device=self.device).uniform_(
                self.cfg.distractor_spawn_x_range[0], self.cfg.distractor_spawn_x_range[1]
            )
            unique_pos_w[:, :, 1] = torch.empty(b, num_additional, device=self.device).uniform_(
                self.cfg.distractor_spawn_y_range[0], self.cfg.distractor_spawn_y_range[1]
            )
            # Distractors share the per-env randomized cube size, so seat them on the table
            # using that half-height (matches how the block itself is seated).
            unique_pos_w[:, :, 2] = (self.block_sizes[env_ids, 2] / 2 + 5e-4 + self.table_height_offset[env_ids]).unsqueeze(1)
            unique_pos_w += self.scene.env_origins[env_ids, None, :]

            min_clearance = self.block_sizes[env_ids, 0].unsqueeze(-1).unsqueeze(-1) * 2.0  # 2x block width (~8 cm)
            # Push additional distractors away from block, goal, and distractor 0
            for ref_xy in [block_pose[:, :2], goal_pose[:, :2], distractor0_pos_w[:, :2]]:
                diff = unique_pos_w[:, :, :2] - ref_xy[:, None, :]
                dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                push = (min_clearance / dist - 1.0).clamp(min=0.0) * diff
                unique_pos_w[:, :, :2] += push

            # Push active additional distractors away from each other (5 passes)
            no_self = ~torch.eye(num_additional, device=self.device, dtype=torch.bool)
            pair_active = additional_active[:, :, None] * additional_active[:, None, :] * no_self[None]
            for _ in range(5):
                xy = unique_pos_w[:, :, :2]
                diff = xy[:, :, None, :] - xy[:, None, :, :]
                dist = diff.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                push = (min_clearance.unsqueeze(1) / dist - 1.0).clamp(min=0.0) * pair_active.unsqueeze(-1) * diff
                unique_pos_w[:, :, :2] += push.sum(dim=2)

            # Place inactive distractors on top of the ground plane (z=-1.05) so they
            # rest there stably. Space them out in y so they don't collide with each other.
            underground_base = self.scene.env_origins[env_ids].clone()
            underground_base[:, 0] += 1.0
            underground_base[:, 2] = -1.05 + self.block_sizes[env_ids, 2] / 2

            # Per-env random permutation so each physical object gets a different logical slot each episode
            perm = torch.stack([torch.randperm(num_additional, device=self.device) for _ in range(b)])  # (b, num_additional)
            arange_b = torch.arange(b, device=self.device)

            # Store the active mask aligned to PHYSICAL objects (dist_pos_w / distractors_pos_error / corner
            # obs are all physical-indexed). Physical object 0 is the always-active cardinal distractor;
            # physical object j+1 is active iff its assigned logical slot perm[:, j] is active.
            self.active_distractor_mask[env_ids] = torch.cat([
                torch.ones(b, 1, dtype=torch.bool, device=self.device),
                torch.gather(additional_active, 1, perm),
            ], dim=1)

            for j, distractor_j in enumerate(self.distractors[1:], start=0):
                pj = perm[:, j]  # (b,) logical slot index for this physical object
                is_active = additional_active[arange_b, pj]  # (b,) bool
                is_stacked = (pj < stack_height)              # (b,) bool, per-env
                # Stacked: all at slot 0's xy, z increases with level
                stacked_pos = unique_pos_w[:, 0].clone()
                stacked_pos[:, 2] = unique_pos_w[:, 0, 2] + (self.block_sizes[env_ids, 2] + 2e-3) * pj.float() + 5e-4
                # Free: offset slot by stack_height
                slot = (pj - stack_height + 1).clamp(0, num_additional - 1)
                free_pos = unique_pos_w[arange_b, slot]
                active_pos = torch.where(is_stacked[:, None], stacked_pos, free_pos)
                # Inactive distractors go underground; each gets a unique y offset so they don't overlap
                underground_pos = underground_base.clone()
                underground_pos[:, 1] += j * 0.1
                pos_j = torch.where(is_active[:, None], active_pos, underground_pos)
                pose_j = torch.cat([pos_j, rand_yaw_quat], dim=-1)
                distractor_j.write_root_pose_to_sim(pose_j, env_ids=env_ids)
                distractor_j.write_root_velocity_to_sim(torch.zeros(b, 6, device=self.device), env_ids=env_ids)
                self.distractors_init_pos[env_ids, j + 1] = pose_j[:, :3]
                self.distractors_init_quat[env_ids, j + 1] = pose_j[:, 3:]

