"""
Kinematic related utilities and classes
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from toyblocks.tasks.base_env import BaseEnv
from isaaclab.assets import Articulation
import torch
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg
from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
import isaaclab.utils.math as math_utils

@dataclass
class KinematicsCfg:
    articulation_cfg: SceneEntityCfg
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)
    num_ik_iterations: int = 25

class Kinematics:
    def __init__(self, env: BaseEnv, cfg: KinematicsCfg):
        self.cfg = cfg
        self.cfg.articulation_cfg.resolve(env.scene)
        self.articulation: Articulation = env.scene[self.cfg.articulation_cfg.name]

        self.joint_ids: list[int] | slice = self.cfg.articulation_cfg.joint_ids
        self.n_joints: int = self.articulation.num_joints if isinstance(self.joint_ids, slice) else len(self.joint_ids)
        self.tcp_offset = torch.tensor(self.cfg.tcp_offset, device=env.device, dtype=torch.float32)
        
        # Create IK solver
        robot_ik_solver_cfg = DifferentialInverseKinematicsActionCfg(
            asset_name=self.cfg.articulation_cfg.name,
            joint_names=self.cfg.articulation_cfg.joint_names,
            body_name=self.cfg.articulation_cfg.body_names,
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            scale=1.0,
        )
        self.solver: DifferentialInverseKinematicsAction = robot_ik_solver_cfg.class_type(robot_ik_solver_cfg, env)

    def ik_set_joint_pos(self, env_ids: torch.Tensor, target_pos_w: torch.Tensor, target_quat_w: torch.Tensor, seed_joint_pos: torch.Tensor | None = None) -> None:
        """
        Sets the joint position on the articulation that will reach the target pose via IK.

        Args
            env_ids: The environment indices to compute IK for
            target_pos_w: The target position in world frame
            target_quat_w: The target orientation in world frame
            seed_joint_pos: Optional seed joint positions to initialize IK from. If None, starts from the current joint positions.
        """

        if seed_joint_pos is not None:
            self.articulation.write_joint_state_to_sim(
                position=seed_joint_pos[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=self.articulation.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,
            )

        target_pos_w = target_pos_w.clone()
        target_pos_w += self.tcp_offset.unsqueeze(0)

        pos_b, quat_b = self.solver._compute_frame_pose()
        pos_b[env_ids], quat_b[env_ids] = math_utils.subtract_frame_transforms(
            self.articulation.data.root_link_pos_w[env_ids],
            self.articulation.data.root_link_quat_w[env_ids],
            target_pos_w,
            target_quat_w,
        )
        
        # Set target pose for IK solver
        self.solver.process_actions(torch.cat([pos_b, quat_b], dim=1))

        # Solve IK iteratively for better convergence
        for iteration in range(self.cfg.num_ik_iterations):
            self.solver.apply_actions()
            
            # Compute delta and apply smoothly
            delta_joint_pos = 0.25 * (
                self.articulation.data.joint_pos_target[env_ids] - self.articulation.data.joint_pos[env_ids]
            )
            
            self.articulation.write_joint_state_to_sim(
                position=(delta_joint_pos + self.articulation.data.joint_pos[env_ids])[:, self.joint_ids],
                velocity=torch.zeros((len(env_ids), self.n_joints), device=self.articulation.device),
                joint_ids=self.joint_ids,
                env_ids=env_ids,
            )