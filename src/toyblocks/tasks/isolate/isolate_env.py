import torch
import numpy as np
import math as pmath
import isaaclab.sim as sim_utils

from collections.abc import Sequence
from typing import List
from toyblocks.robots.kinematics import Kinematics, KinematicsCfg
from toyblocks.tasks.base_env import BaseEnv
from toyblocks.utils import obs_utils, rigid_objects
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from .isolate_env_cfg import IsolateEnvCfg
from isaaclab.utils import math
from toyblocks.utils import randomization_utils 

class IsolateEnv(BaseEnv):
    cfg: IsolateEnvCfg

    block: RigidObject
    distractors: List[RigidObject]
    goal: RigidObject

    kinematics: Kinematics

    # intermediate values
    object_pos: torch.Tensor 
    object_quat: torch.Tensor 
    tcp_pos: torch.Tensor 
    tcp_quat: torch.Tensor 
    object_to_tcp: torch.Tensor 
    pos_error: torch.Tensor 
    axis_angle_error: torch.Tensor 
    block_corner_pts_to_goal_corner_pts_dist: torch.Tensor 
    _reach_dist: torch.Tensor

    # distractor intermediate values
    distractors_init_pos: torch.Tensor # (num_envs, num_distractors, 3)
    distractors_init_quat: torch.Tensor # (num_envs, num_distractors, 4)
    distractors_moved: torch.Tensor 
    distractors_corner_pts_rel_ee: torch.Tensor

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
        for i in range(self.cfg.num_distractors):
            distractor = rigid_objects.create_cuboid(
                prim_path=f"/World/envs/env_.*/Distractor_{i}",
                size=self.cfg.block_size,
                diffuse_color=(1.0, 0.0, 0.0),
                init_pos=(0.0, 0.0, self.cfg.block_size[2] / 2),
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

        # Contact sensors: each finger vs the table surface.
        env_ns = self.scene.env_regex_ns
        table_filter = [f"{env_ns}/BoxTable"]
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
        self.block_corner_visuals = VisualizationMarkers(cfg=cube_corner_cfg)
        self.block_corner_visuals.set_visibility(True)
        self.goal_corner_visuals = VisualizationMarkers(cfg=cube_corner_cfg.replace(prim_path="/Visuals/GoalCornerVisuals"))
        self.goal_corner_visuals.set_visibility(True)
        self.block_arrow = VisualizationMarkers(cfg=GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/BlockArrow"))
        self.goal_arrow = VisualizationMarkers(cfg=BLUE_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/GoalArrow"))
        self.block_arrow.set_visibility(True)
        self.goal_arrow.set_visibility(True)

    def _after_scene_setup(self):
        super()._after_scene_setup()
        self._finger_contact_sensors = [
            self.scene.sensors["contact:finger_left_inner_finger"],
            self.scene.sensors["contact:finger_right_inner_finger"],
        ]

        self._finger_contact_sensors = [
            self.scene.sensors["contact:finger_left_inner_finger"],
            self.scene.sensors["contact:finger_right_inner_finger"],
        ]

        self.distractors_init_pos = torch.zeros(self.num_envs, self.cfg.num_distractors, 3, device=self.device)
        self.distractors_init_quat = torch.zeros(self.num_envs, self.cfg.num_distractors, 4, device=self.device)
        self.active_distractor_mask = torch.zeros(self.num_envs, self.cfg.num_distractors, dtype=torch.bool, device=self.device)

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
        corner_translations = (self.block_corner_pts + self.scene.env_origins.unsqueeze(1)).reshape(self.num_envs * 8, 3)
        corner_indices = list(range(8)) * self.num_envs
        self.block_corner_visuals.visualize(translations=corner_translations, marker_indices=corner_indices)
        goal_corner_translations = (self.goal_corner_pts + self.scene.env_origins.unsqueeze(1)).reshape(self.num_envs * 8, 3)
        goal_corner_translations = (self.goal_corner_pts + self.scene.env_origins.unsqueeze(1)).reshape(self.num_envs * 8, 3)
        goal_corner_indices = list(range(8)) * self.num_envs
        self.goal_corner_visuals.visualize(translations=goal_corner_translations, marker_indices=goal_corner_indices)

    @property
    def visualization_markers(self) -> dict:
        return {
            "block_arrow": self.block_arrow,
            "goal_arrow": self.goal_arrow,
            "cube_corners": self.block_corner_visuals,
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
        self.goal_pos = self.goal.data.root_pos_w - self.scene.env_origins
        self.goal_pos[:, 2] = self.block_sizes[:, 2] / 2 + self.table_height_offset
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
        distractors_corner_pts_rel_ee = []
        for distractor in self.distractors:
            distractor_corner_pts = obs_utils.cube_corner_points(distractor)
            distractors_corner_pts_rel_ee.append(self._transform_corners_to_frame(self.ee_frame, distractor_corner_pts))
        all_dist_corners = torch.stack(distractors_corner_pts_rel_ee, dim=1)  # (N, num_dist, 24)
        active_corners_mask = self.active_distractor_mask.unsqueeze(-1).expand_as(all_dist_corners)
        self.distractors_corner_pts_rel_ee = (all_dist_corners * active_corners_mask).reshape(self.num_envs, -1)

        # see if the distractors have moved (only active slots count)
        distractors_pos, distractors_quat = obs_utils.distractors_pos_and_quat(self.distractors)
        distractor_pos_error, distractor_rot_error = math.compute_pose_error(self.distractors_init_pos.reshape(-1, 3), self.distractors_init_quat.reshape(-1, 4), distractors_pos, distractors_quat)
        self.distractors_pos_error = distractor_pos_error.reshape(self.num_envs, -1, 3).norm(dim=-1)  # (num_envs, num_distractors)
        self.distractors_rot_error = distractor_rot_error.reshape(self.num_envs, -1, 3).norm(dim=-1)  # (num_envs, num_distractors)
        self.distractors_moved = (
            (self.distractors_pos_error > self.cfg.distractor_distance_threshold) & self.active_distractor_mask
        ).any(dim=-1) | (
            (self.distractors_rot_error > self.cfg.distractor_orientation_threshold) & self.active_distractor_mask
        ).any(dim=-1)

        self.acheived_goal = self.reached_pose & ~self.distractors_moved
        self.goal_corner_pts = obs_utils.cube_corner_points(self.block, self.goal_pos, self.goal_quat)
        self.block_corner_pts_to_goal_corner_pts_dist = (self.block_corner_pts - self.goal_corner_pts).norm(dim=-1).mean(dim=-1)

        # Orientation invariant block corners in the goal frame
        q_inv = math.quat_inv(self.goal_quat)  # (N, 4)
        delta = self.block_corner_pts - self.object_pos.unsqueeze(1)  # (N, 8, 3)
        q_inv_exp = q_inv.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)  # (N*8, 4)
        self.block_corners_in_goal_frame = math.quat_apply(q_inv_exp, delta.reshape(-1, 3)).reshape(self.num_envs, -1)  # (N, 24)

    def _get_obs_extra(self) -> dict:
        return dict(
            object_pos=self.object_pos,
            # object_quat=self.object_quat,
            # object_quat=self.object_quat,
            object_to_tcp=self.object_to_tcp,
            tcp_pos=self.tcp_pos,
            # tcp_quat=self.tcp_quat,
            # tcp_quat=self.tcp_quat,
            # block_corner_pts=self.block_corner_pts.reshape(self.num_envs, -1),
            distractors_corner_pts_rel_ee=self.distractors_corner_pts_rel_ee.reshape(self.num_envs, -1),
        )

    def _get_obs_goal(self) -> dict:
        return dict(
            # goal_corner_pts=self.goal_corner_pts.reshape(self.num_envs, -1),
            # block_corner_pts_to_goal_corner_pts=(self.block_corner_pts - self.goal_corner_pts).reshape(self.num_envs, -1),
            block_corners_in_goal_frame=self.block_corners_in_goal_frame,
            pos_error=self.pos_error,
            # axis_angle_error=self.axis_angle_error,
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
        # reach target = center of the block's top face (computed in _compute_intermediate_values).
        dist_to_block = self._reach_dist
        reach_reward = (1 - torch.tanh(dist_to_block * 5))
        # reach reward is only active for the first part of the episode, then it's disabled.
        self._reach_achieved |= dist_to_block < self.cfg.reach_success_threshold
        reach_reward = reach_reward * (~self._reach_achieved).float()

        corner_reward = torch.exp(-(self.block_corner_pts_to_goal_corner_pts_dist / 0.1)**2)
        corner_reward2 = torch.exp(-(self.block_corner_pts_to_goal_corner_pts_dist / 0.04)**2)
        active_count = self.active_distractor_mask.float().sum(dim=-1).clamp(min=1)
        per_d_pos_rew = torch.exp(-(self.distractors_pos_error / 0.02)**2)   # (N, num_distractors)
        per_d_rot_rew = torch.exp(-(self.distractors_rot_error / 0.02)**2)
        distractor_reward1 = (per_d_pos_rew * self.active_distractor_mask).sum(dim=-1) / active_count
        distractor_reward2 = (per_d_rot_rew * self.active_distractor_mask).sum(dim=-1) / active_count
        distractor_reward = (distractor_reward1 + distractor_reward2) / 2.0
        success_reward = self.acheived_goal.float()
        positive_reward_sum = self.cfg.reach_reward_coef + self.cfg.corner_reward_coef + self.cfg.corner_reward2_coef + self.cfg.success_reward_coef + self.cfg.distractor_reward_coef
        # Once reach is latched off, drop its coefficient from the (per-env) denominator
        denom = positive_reward_sum - self.cfg.reach_reward_coef * self._reach_achieved.float()
        # Two-phase curriculum: before the block is reached, ONLY the reach reward is active
        aligning = self._reach_achieved.float()
        reward = (
            self.cfg.reach_reward_coef * reach_reward
            + aligning * (
            self.cfg.corner_reward_coef * corner_reward + \
            self.cfg.corner_reward2_coef * corner_reward2 + \
            self.cfg.success_reward_coef * success_reward + \
            self.cfg.distractor_reward_coef * distractor_reward 
            - self.cfg.table_collision_penalty_coef * (self.table_penetration > 0).float()
            )
        ) / denom 
        return reward

    def evaluate(self) -> dict:
        result = super().evaluate()
        result["success"] = self.acheived_goal
        stats = result.setdefault("stats", {})
        stats["object_to_tcp_dist"] = self.object_to_tcp.norm(dim=-1)
        # latch distance vs cfg.reach_success_threshold; if this never drops below the threshold, reach can't latch
        stats["reach_dist"] = self._reach_dist
        stats["pos_error"] = self.pos_error[:, :2].norm(dim=-1)
        stats["axis_angle_error"] = self.axis_angle_error.norm(dim=-1)
        stats["block_corner_pts_to_goal_corner_pts_dist"] = self.block_corner_pts_to_goal_corner_pts_dist
        stats["reached_pose"] = self.reached_pose
        stats["distractors_moved"] = self.distractors_moved
        stats["table_collision"] = (self.table_penetration > 0).float()
        stats["reach_achieved"] = self._reach_achieved.float()
        return result

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        self._reach_achieved[env_ids] = False
        with torch.device(self.device):
            b = len(env_ids)
            n_total = 1 + self.cfg.num_distractors  # block + distractors

            # Build a compact grid of local offsets centered at origin.
            # Spacing ensures no overlap even when cubes have random yaw:
            #   min separation = block_size * sqrt(2) + clutter_grid_gap
            spacing = self.cfg.block_size[0] * pmath.sqrt(2) + self.cfg.clutter_grid_gap
            n_cols = int(np.ceil(np.sqrt(n_total)))
            n_rows = int(np.ceil(n_total / n_cols))
            col_idx = torch.arange(n_cols, dtype=torch.float32, device=self.device)
            row_idx = torch.arange(n_rows, dtype=torch.float32, device=self.device)
            grid_yy, grid_xx = torch.meshgrid(row_idx, col_idx, indexing='ij')
            grid_pos = torch.stack([grid_xx.flatten(), grid_yy.flatten()], dim=-1) * spacing  # (n_cols*n_rows, 2)
            grid_pos = grid_pos - grid_pos.mean(dim=0)  # center at origin

            # Per-env: shuffle grid slots and pick the first n_total
            n_grid = grid_pos.shape[0]
            perms = torch.rand(b, n_grid, device=self.device).argsort(dim=-1)[:, :n_total]  # (b, n_total)
            local_offsets = grid_pos[perms]  # (b, n_total, 2)

            # Add small independent jitter to each cube
            jitter = (torch.rand(b, n_total, 2, device=self.device) - 0.5) * 2 * self.cfg.clutter_jitter
            local_offsets = local_offsets + jitter

            # Rotate the entire cluster by a random global angle per env
            global_angle = torch.rand(b, device=self.device) * 2 * np.pi
            cos_a, sin_a = torch.cos(global_angle), torch.sin(global_angle)  # (b,)
            rot_x = cos_a[:, None] * local_offsets[:, :, 0] - sin_a[:, None] * local_offsets[:, :, 1]
            rot_y = sin_a[:, None] * local_offsets[:, :, 0] + cos_a[:, None] * local_offsets[:, :, 1]
            local_offsets = torch.stack([rot_x, rot_y], dim=-1)  # (b, n_total, 2)

            # Sample clutter center in local (env-relative) coords then convert to world
            clutter_local_x = torch.rand(b, device=self.device) * (self.cfg.block_spawn_x_range[1] - self.cfg.block_spawn_x_range[0]) + self.cfg.block_spawn_x_range[0]
            clutter_local_y = torch.rand(b, device=self.device) * (self.cfg.block_spawn_y_range[1] - self.cfg.block_spawn_y_range[0]) + self.cfg.block_spawn_y_range[0]
            clutter_center_w = torch.stack([
                clutter_local_x + self.scene.env_origins[env_ids, 0],
                clutter_local_y + self.scene.env_origins[env_ids, 1],
                self.scene.env_origins[env_ids, 2] + self.block_sizes[env_ids,2] / 2 + 5e-4 + self.table_height_offset[env_ids],
            ], dim=-1)  # (b, 3); z is the block's center height

            # World positions for all n_total objects
            all_pos_w = torch.zeros(b, n_total, 3, device=self.device)
            all_pos_w[:, :, :2] = clutter_center_w[:, None, :2] + local_offsets
            # Block (slot 0) has randomized size, captured in clutter_center_w
            all_pos_w[:, 0, 2] = clutter_center_w[:, 2]
            # Distractors (slots 1+) have fixed cfg.block_size, so use the env's base height
            all_pos_w[:, 1:, 2] = (self.scene.env_origins[env_ids, 2] + self.cfg.block_size[2] / 2 + 5e-4 + self.table_height_offset[env_ids]).unsqueeze(-1)

            # Sort distractor slots by distance to the block so active distractors
            # (indices 0..num_active-1) always occupy the grid positions nearest the block.
            block_pos_2d = all_pos_w[:, 0:1, :2]  # (b, 1, 2)
            dist_to_block = (all_pos_w[:, 1:, :2] - block_pos_2d).norm(dim=-1)  # (b, n_total-1)
            sort_idx = dist_to_block.argsort(dim=-1)  # (b, n_total-1)
            all_pos_w[:, 1:] = all_pos_w[:, 1:][torch.arange(b, device=self.device)[:, None], sort_idx]

            # Place block (slot 0) with random yaw
            block_pose = torch.zeros(b, 7, device=self.device)
            block_pose[:, 3:] = randomization_utils.random_cube_orientation_symmetry(b, self.device)
            rand_yaw_quat = randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (self.cfg.block_spawn_yaw_range[0], self.cfg.block_spawn_yaw_range[1]), self.device)
            block_pose[:, 3:] = math.quat_mul(rand_yaw_quat, block_pose[:, 3:])
            block_pose[:, :3] = all_pos_w[:, 0]
            self.block.write_root_pose_to_sim(block_pose, env_ids=env_ids)
            self.block.write_root_velocity_to_sim(torch.zeros(b, 6, device=self.device), env_ids=env_ids)

            # Place goal at a sampled distance from the block, on the same side as the block relative to clutter
            block_dir = all_pos_w[:, 0, :2] - clutter_center_w[:, :2]
            block_theta = torch.atan2(block_dir[:, 1], block_dir[:, 0])
            goal_theta = block_theta + (torch.rand(b, device=self.device) - 0.5) * pmath.radians(30)  # ±15° around block direction
            goal_pose = torch.zeros(b, 7, device=self.device)
            goal_distance = torch.rand(b, device=self.device) * (self.cfg.goal_distance_max - self.cfg.goal_distance_min) + self.cfg.goal_distance_min
            goal_pose[:, 0] = all_pos_w[:, 0, 0] + goal_distance * torch.cos(goal_theta)
            goal_pose[:, 1] = all_pos_w[:, 0, 1] + goal_distance * torch.sin(goal_theta)
            goal_pose[:, 2] = clutter_center_w[:, 2]
            goal_pose[:, 3:] = block_pose[:, 3:]  # match block spawn orientation so axis_angle_error measures rotation-from-spawn, not distance-from-identity
            self.goal.write_root_pose_to_sim(goal_pose, env_ids=env_ids)

            # Sample how many distractors are active this episode (per env)
            num_active = torch.randint(self.cfg.min_num_distractors, self.cfg.num_distractors + 1, (b,), device=self.device)
            active_mask = torch.arange(self.cfg.num_distractors, device=self.device)[None, :] < num_active[:, None]  # (b, num_distractors)
            self.active_distractor_mask[env_ids] = active_mask

            underground_base = self.scene.env_origins[env_ids].clone()
            underground_base[:, 0] += 1.0
            underground_base[:, 2] = -1.05 + self.cfg.block_size[2] / 2

            # Per-env random permutation so each physical object gets a different logical slot each episode
            perm = torch.stack([torch.randperm(self.cfg.num_distractors, device=self.device) for _ in range(b)])  # (b, num_distractors)
            arange_b = torch.arange(b, device=self.device)

            # Place distractors (slots 1..n_total-1) with random yaw; inactive go underground
            for i, distractor in enumerate(self.distractors):
                pi = perm[:, i]  # (b,) logical slot for this physical object
                dist_quat = randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), self.cfg.block_spawn_yaw_range, self.device)
                active_pos = all_pos_w[arange_b, pi + 1]
                underground_pos = underground_base.clone()
                underground_pos[:, 1] += i * 0.1
                pos_i = torch.where(active_mask[arange_b, pi, None], active_pos, underground_pos)
                dist_pose = torch.cat([pos_i, dist_quat], dim=-1)
                distractor.write_root_pose_to_sim(dist_pose, env_ids=env_ids)
                distractor.write_root_velocity_to_sim(torch.zeros(b, 6, device=self.device), env_ids=env_ids)
                self.distractors_init_pos[env_ids, i] = dist_pose[:, :3]
                self.distractors_init_quat[env_ids, i] = dist_pose[:, 3:]

            # Move robot EE to 1 cm from the block in the direction away from the clutter center
            ee_target_pos = block_pose[:, :3].clone()
            ee_target_pos[:, 0] += 0.04 * torch.cos(block_theta)
            ee_target_pos[:, 1] += 0.04 * torch.sin(block_theta)
            ee_target_pos[:, 2] += 0.07 + torch.empty(b, device=self.device).uniform_(-0.01, 0.01)
            # Yaw aligned with block-to-goal direction so grippers face the pull direction
            goal_yaw_quat = torch.zeros(b, 4, device=self.device)
            goal_yaw_quat[:, 0] = torch.cos((goal_theta + pmath.pi / 2) / 2)
            goal_yaw_quat[:, 3] = torch.sin((goal_theta + pmath.pi / 2) / 2)
            pointing_down = torch.zeros(b, 4, device=self.device)
            pointing_down[:, 1] = 1.0  # 180° around x-axis = pointing down
            # Original code for resetting the EE to yaw-align with the block-goal dir
            # ee_target_quat = math.quat_mul(goal_yaw_quat, pointing_down)
            ee_target_quat = math.quat_mul(
                randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (-torch.pi, torch.pi), self.device),
                torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(b, 1))
            seed_joint_pos = self.kinematics.articulation.data.default_joint_pos[env_ids]
            self.kinematics.ik_set_joint_pos(env_ids, ee_target_pos, ee_target_quat, seed_joint_pos=seed_joint_pos)