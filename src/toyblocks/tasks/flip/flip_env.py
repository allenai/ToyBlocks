import torch
import numpy as np
import isaaclab.sim as sim_utils

from collections.abc import Sequence
from toyblocks.robots.kinematics import Kinematics, KinematicsCfg
from toyblocks.tasks.base_env import BaseEnv
from toyblocks.utils import obs_utils, rigid_objects
from isaaclab.assets.rigid_object.rigid_object import RigidObject
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from isaaclab.sensors.contact_sensor import ContactSensor, ContactSensorCfg
from isaaclab.markers.visualization_markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from .flip_env_cfg import FlipEnvCfg
from isaaclab.utils import math
from toyblocks.utils import randomization_utils    


class FlipEnv(BaseEnv):
    """
    Flip task where robot instantiated in a pose above a block and aligned with the block, must flip the block either left or right 90 degrees.

    Goal Conditions: Flip one hot encoding ([0, 1] or [1, 0]), goal corner points
    """
    cfg: FlipEnvCfg

    # initialize class attributes 
    block: RigidObject 
    goal: RigidObject

    kinematics: Kinematics 

    # intermediate values
    object_pos: torch.Tensor 
    object_quat: torch.Tensor 
    tcp_pos: torch.Tensor 
    tcp_quat: torch.Tensor 
    object_to_tcp: torch.Tensor 
    pos_error: torch.Tensor 
    block_corner_pts_to_goal_corner_pts_dist: torch.Tensor 

    # evaluation / visualization
    reached_goal: torch.Tensor 
    block_corner_pts: torch.Tensor 
    goal_corner_pts: torch.Tensor 
    block_corner_pts_to_goal_corner_pts: torch.Tensor
    block_corners_in_goal_frame: torch.Tensor 
    block_arrow: VisualizationMarkers
    goal_arrow: VisualizationMarkers
    block_corner_visuals: VisualizationMarkers
    goal_corner_visuals: VisualizationMarkers

    # goal orientation is sampled from one of four cardinal directions wrt current cube pose 
    discrete_orientation_options: torch.Tensor
    
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
            init_pos=(0.0, 0.0, self.cfg.block_size[2] / 2),
            kinematic=True,
            collision=False,
            **bucket_kw,
        )
        self.scene.rigid_objects["goal"] = self.goal

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
        self.discrete_orientation_options = torch.tensor([
            (1.5708, 0.0, 0.0),   # Roll right 90° (left face to top)
            (-1.5708, 0.0, 0.0),  # Roll left 90° (right face to top)
            (0.0, 1.5708, 0.0),   # Pitch forward 90° (back face to top)
            (0.0, -1.5708, 0.0),  # Pitch backward 90° (front face to top)
        ], device=self.device)
        self.flip_axis = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.goal_orientation_one_hot = torch.zeros(self.num_envs, len(self.discrete_orientation_options), device=self.device)
        # (local_x, local_y) offset to the push edge on the top face, per env and option
        # shape: (num_envs, 4, 2)
        hy = self.block_sizes[:, 1] / 2   # (num_envs,)
        hx = self.block_sizes[:, 0] / 2   # (num_envs,)
        zeros = torch.zeros(self.num_envs, device=self.device)
        self._push_offset_table = torch.stack([
            torch.stack([zeros,   -hy], dim=-1),  # option 0: roll  +90° → press -Y edge
            torch.stack([zeros,    hy], dim=-1),  # option 1: roll  -90° → press +Y edge
            torch.stack([ hx,  zeros], dim=-1),   # option 2: pitch +90° → press +X edge
            torch.stack([-hx,  zeros], dim=-1),   # option 3: pitch -90° → press -X edge
        ], dim=1)  # (num_envs, 4, 2)
        self.flip_push_offset = torch.zeros(self.num_envs, 2, device=self.device)
        # Flip push direction in world frame: where the current top face goes after the flip.
        self.flip_push_dir_world = torch.zeros(self.num_envs, 3, device=self.device)

        self._finger_contact_sensors = [
            self.scene.sensors["contact:finger_left_inner_finger"],
            self.scene.sensors["contact:finger_right_inner_finger"],
        ]

        # Hold the gripper closed to `gripper_close_fraction` of its travel. Setting the
        # finger_joint default target means _reset_robots keeps it closed every episode,
        # while the arm-only control mode leaves the action space unchanged.
        self._finger_joint_ids, _ = self.robot.articulation.find_joints(["finger_joint"])
        finger_upper = self.robot.articulation.data.joint_pos_limits[0, self._finger_joint_ids, 1]  # (n_finger,)
        self._gripper_close_target = self.cfg.gripper_close_fraction * finger_upper
        self.robot.articulation.data.default_joint_pos[:, self._finger_joint_ids] = self._gripper_close_target.unsqueeze(0)

        # Track the true gripper contact point (fingertip midpoint) 
        self._inner_finger_body_ids, _ = self.robot.articulation.find_bodies(["left_inner_finger", "right_inner_finger"])
        self._finger_tip_local = torch.tensor([0.0433, 0.0, 0.1388], device=self.device)

    def _update_visualization_markers(self):
        self.block_arrow.visualize(self.block.data.root_pos_w, self.block.data.root_quat_w)
        self.goal_arrow.visualize(self.goal.data.root_pos_w, self.goal.data.root_quat_w)
        corner_translations = (self.block_corner_pts + self.scene.env_origins.unsqueeze(1)).reshape(self.num_envs * 8, 3)
        corner_translations = (self.block_corner_pts + self.scene.env_origins.unsqueeze(1)).reshape(self.num_envs * 8, 3)
        corner_indices = list(range(8)) * self.num_envs
        self.cube_corner_visuals.visualize(translations=corner_translations, marker_indices=corner_indices)
        goal_corner_translations = (self.goal_corner_pts + self.scene.env_origins.unsqueeze(1)).reshape(self.num_envs * 8, 3)
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

    def _compute_goals(self):
        self.goal_pos = self.goal.data.root_pos_w
        self.goal_quat = self.goal.data.root_quat_w
        self.goal_pos[:, 2] = self.table_height_offset + self.block_sizes[:, 2] / 2

    def _compute_intermediate_values(self):
        self.object_pos = self.block.data.root_pos_w - self.scene.env_origins
        self.object_quat = self.block.data.root_quat_w
        self.tcp_pos = self.ee_frame.data.target_pos_w[:, 0] - self.scene.env_origins
        self.tcp_quat = self.ee_frame.data.target_quat_w[:, 0] 
        self.object_to_tcp = self.tcp_pos - self.object_pos

        self.pos_error, self.axis_angle_error = math.compute_pose_error(self.object_pos, self.object_quat, self.goal_pos - self.scene.env_origins, self.goal_quat)
        self.axis_angle_error_norm = self.axis_angle_error.norm(dim=-1)
        self.reached_orientation = self.axis_angle_error_norm < self.cfg.orientation_threshold
        self.reached_pose = (self.pos_error.norm(dim=-1) < self.cfg.distance_threshold) & self.reached_orientation

        self.block_corner_pts = obs_utils.cube_corner_points(self.block) - self.scene.env_origins.unsqueeze(1)
        self.goal_corner_pts = obs_utils.cube_corner_points(self.block, self.goal_pos, self.goal_quat) - self.scene.env_origins.unsqueeze(1)
        self.block_corner_pts_to_goal_corner_pts_dist = (self.block_corner_pts - self.goal_corner_pts).norm(dim=-1).mean(dim=-1)

        # Block corners expressed in the goal frame — invariant to absolute world-frame
        # orientation, so the policy sees the same representation across all 24 block symmetries.
        q_inv = math.quat_inv(self.goal_quat)  # (N, 4)
        delta = self.block_corner_pts - self.object_pos.unsqueeze(1)  # (N, 8, 3)
        q_inv_exp = q_inv.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)  # (N*8, 4)
        self.block_corners_in_goal_frame = math.quat_apply(q_inv_exp, delta.reshape(-1, 3)).reshape(self.num_envs, -1)  # (N, 24)

        self.block_to_goal_quat = math.quat_mul(math.quat_inv(self.object_quat), self.goal_quat)

        # Reach target: top-face edge in flip direction, computed fully in world frame.
        # block_pos + h*Z + h*push_dir is equivalent to the old local-frame formulation because
        # (a) the physical top is always at block_pos+[0,0,h] regardless of orientation, and
        # (b) for a 90° flip, flip_push_dir_world is perpendicular to Z so no projection needed.
        h = self.block_sizes[:, 0] / 2  # half-side; valid for a cube
        self.reach_target = self.object_pos.clone()
        self.reach_target[:, 2] += h
        self.reach_target = self.reach_target + h.unsqueeze(1) * self.flip_push_dir_world
        self.tcp_to_reach_target = self.reach_target - self.tcp_pos

        # Gripper contact point = midpoint of the two (closed) fingertips, measured from the
        # actual finger body poses. Because the gripper is held closed, the fingers meet at the
        # center and their tips shift in the TCP z-direction; reading the live body transforms
        # captures that shift so the reach target is computed against the true contact point.
        finger_body_pos = self.robot.articulation.data.body_pos_w[:, self._inner_finger_body_ids]    # (N, 2, 3)
        finger_body_quat = self.robot.articulation.data.body_quat_w[:, self._inner_finger_body_ids]  # (N, 2, 4)
        tip_local = self._finger_tip_local.view(1, 1, 3).expand(self.num_envs, 2, 3).reshape(-1, 3)
        tip_world = finger_body_pos.reshape(-1, 3) + math.quat_apply(finger_body_quat.reshape(-1, 4), tip_local)
        self.gripper_tip_pos = tip_world.reshape(self.num_envs, 2, 3).mean(dim=1) - self.scene.env_origins  # env-local
        # XY+Z distance: bias toward horizontal alignment first, then descent straight down
        _xy = (self.gripper_tip_pos[:, :2] - self.reach_target[:, :2]).norm(dim=-1)
        _z  = (self.gripper_tip_pos[:, 2]  - self.reach_target[:, 2]).abs()
        self.gripper_tip_xy_dist = _xy + 0.5 * _z

        left_force  = self._finger_contact_sensors[0].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)  # (num_envs,)
        right_force = self._finger_contact_sensors[1].data.force_matrix_w[:, 0, 0, :].norm(dim=-1)  # (num_envs,)
        self.table_penetration = left_force + right_force  # (num_envs,), >0 when fingers touch table


    def _get_obs_extra(self) -> dict:
        return dict(
            object_pos=self.object_pos,
            # object_quat=self.object_quat,
            object_to_tcp=self.object_to_tcp,
            tcp_pos=self.tcp_pos,
            tcp_quat=self.tcp_quat,
            tcp_to_reach_target=self.tcp_to_reach_target,
            # block_corner_pts=self.block_corner_pts.reshape(self.num_envs, -1),
        )

    def _get_obs_goal(self) -> dict:
        return dict(
            # TODO: might not need this one hot thing anyway, and if we do, it should be part of intermediate values (computed from some given goal condition)
            # goal_orientation_one_hot=self.goal_orientation_one_hot,
            # TODO: maybe add these back after merging with TAMP?
            # goal_corner_pts=self.goal_corner_pts.reshape(self.num_envs, -1),
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
        reach_reward = 1 - torch.tanh(self.gripper_tip_xy_dist * 5)
        axis_angle_reward1 = torch.exp(-(self.axis_angle_error_norm)**2)
        axis_angle_reward2 = torch.exp(-(self.axis_angle_error_norm / 0.04)**2)
        axis_angle_reward = (axis_angle_reward1 + axis_angle_reward2) / 2.0
        success_reward = self.reached_pose.float()
        corner_reward = torch.exp(-(self.block_corner_pts_to_goal_corner_pts_dist / 0.1)**2)
        reward = (
            self.cfg.reach_reward_coef * reach_reward + \
            self.cfg.axis_angle_reward_coef * axis_angle_reward + \
            self.cfg.success_reward_coef * success_reward + \
            self.cfg.corner_reward_coef * corner_reward - 
            self.cfg.table_collision_penalty_coef * (self.table_penetration > 0).float()
        )
        return reward / (self.cfg.reach_reward_coef + self.cfg.axis_angle_reward_coef + self.cfg.success_reward_coef + self.cfg.corner_reward_coef)      
    
    def evaluate(self) -> dict:
        result = super().evaluate()
        result["success"] = self.reached_pose
        stats = result.setdefault("stats", {})
        stats["object_to_tcp_dist"] = self.object_to_tcp.norm(dim=-1)
        stats["tcp_to_reach_target_dist"] = self.tcp_to_reach_target.norm(dim=-1)
        stats["gripper_tip_xy_dist"] = self.gripper_tip_xy_dist
        stats["pos_error"] = self.pos_error[:, :2].norm(dim=-1)
        stats["axis_angle_error"] = self.axis_angle_error.norm(dim=-1)
        stats["flip_axis_error"] = self.axis_angle_error[torch.arange(self.num_envs), :2].norm(dim=-1)
        stats["flip_success"] = self.axis_angle_error[torch.arange(self.num_envs), :2].norm(dim=-1) < 0.1
        stats["block_corner_pts_to_goal_corner_pts_dist"] = self.block_corner_pts_to_goal_corner_pts_dist
        stats["reached_orientation"] = self.reached_orientation.float()
        stats["table_collision"] = (self.table_penetration > 0).float()
        return result

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        with torch.device(self.device):
            b = len(env_ids)
            block_pose = torch.zeros(b, 7)
            sym_quat = randomization_utils.random_cube_orientation_symmetry(b, self.device)
            # sym_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(b, 1)
            rand_yaw_quat = randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (self.cfg.block_spawn_yaw_range[0], self.cfg.block_spawn_yaw_range[1]), self.device)
            block_pose[:, 3:] = math.quat_mul(rand_yaw_quat, sym_quat)
            block_pose[:, 0] = torch.rand(b) * (self.cfg.block_spawn_x_range[1] - self.cfg.block_spawn_x_range[0]) + self.cfg.block_spawn_x_range[0]
            block_pose[:, 1] = torch.rand(b) * (self.cfg.block_spawn_y_range[1] - self.cfg.block_spawn_y_range[0]) + self.cfg.block_spawn_y_range[0]

            block_pose[:, 2] = self.block_sizes[env_ids, 2] / 2 + 5e-4 + self.table_height_offset[env_ids]
            block_pose[:, :3] += self.scene.env_origins[env_ids]
            self.block.write_root_pose_to_sim(block_pose, env_ids=env_ids)
            self.block.write_root_velocity_to_sim(torch.zeros(b, 6), env_ids=env_ids)

            # TODO: xy offset and discrete orientation are sampled independently. Maybe they should be sampled in sync? 
            # e.g. flip front should translate +x, flip right should translate -y etc. 

            # sample goal xy offset in a radius around the block center 
            radius = torch.empty(b, device=self.device).uniform_(self.cfg.goal_xy_offset_min_radius, self.cfg.goal_xy_offset_max_radius)
            theta = torch.empty(b, device=self.device).uniform_(0.0, 2 * np.pi)
            goal_offset_x = radius * torch.cos(theta)
            goal_offset_y = radius * torch.sin(theta)

            # sample goal orientation from one of four cardinal directions
            num_options = len(self.discrete_orientation_options)
            option_inds = torch.randint(0, num_options, (b,), device=self.device)
            self.goal_orientation_one_hot[env_ids] = torch.nn.functional.one_hot(option_inds, num_classes=num_options).float()
            self.flip_push_offset[env_ids] = self._push_offset_table[env_ids][torch.arange(b, device=self.device), option_inds]
            
            euler_angles = self.discrete_orientation_options[option_inds]
            relative_quat = math.quat_from_euler_xyz(euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2])

            # Apply flip in world frame: chain relative_quat with sym_quat first, then rand_yaw.
            # This ensures the flip axis is always horizontal (world X or Y) regardless of which
            # face sym_quat placed on top, avoiding degenerate yaw-only goal orientations.
            goal_quat = math.quat_mul(rand_yaw_quat, math.quat_mul(relative_quat, sym_quat)).to(torch.float32)

            # Compute the flip push direction in world frame: where the current top face (world_up)
            # goes after the flip. 
            world_up = torch.zeros(b, 3, device=self.device)
            world_up[:, 2] = 1.0
            world_up_after = math.quat_apply(relative_quat, world_up)
            self.flip_push_dir_world[env_ids] = world_up_after

            # update goal pose 
            goal_pose = block_pose.clone() 
            goal_pose[:, 0] += goal_offset_x
            goal_pose[:, 1] += goal_offset_y
            goal_pose[:, 3:] = goal_quat
            self.goal.write_root_pose_to_sim(goal_pose, env_ids=env_ids)

            # set the IK target to above the block 
            # TODO: we might want to standardize the initial robot state joint object distributions for TAMP parts later
            ee_pose = block_pose.clone()
            ee_xy_offset = torch.empty(b, 2, device=self.device).uniform_(-0.04, 0.04)
            ee_pose[:, :2] += ee_xy_offset
            ee_pose[:, 2] += 0.07 + torch.empty(b, device=self.device).uniform_(-0.01, 0.01)
            ee_pose[:, 3:] = math.quat_mul(
                randomization_utils.random_quaterion(b, (0.0, 0.0), (0.0, 0.0), (-torch.pi, torch.pi), self.device),
                torch.tensor([0.0, 1.0, 0.0, 0.0]).repeat(b, 1))
            self.kinematics.ik_set_joint_pos(env_ids, ee_pose[:, :3], ee_pose[:, 3:])






        
        