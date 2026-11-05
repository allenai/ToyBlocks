from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import numpy as np
from blocksuite.tasks.base_env import BaseEnv
from isaaclab.assets.rigid_object import RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCollection, RigidObjectCollectionCfg
from isaaclab.utils.math import quat_from_euler_xyz

from .blocksuite_env_cfg import BlocksuiteEnvCfg


class BlocksuiteEnv(BaseEnv):
    cfg: BlocksuiteEnvCfg

    def __init__(self, cfg: BlocksuiteEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

    def _setup_scene_objects(self):
        # add some blocks
        blocks_cfg = {}
        for i in range(self.cfg.num_blocks):
            blocks_cfg[f"block_{i}"] = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/Block_{i}",
                spawn=sim_utils.CuboidCfg(
                    size=self.cfg.block_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(density=1000.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.1, self.cfg.block_size[1] * 2 * i, self.cfg.block_size[2])),
            )

        self.blocks = RigidObjectCollection(RigidObjectCollectionCfg(rigid_objects=blocks_cfg))
        self.scene.rigid_object_collections["blocks"] = self.blocks

    def _get_observations(self) -> dict:
        # obs = torch.cat(
        #     (
        #         self.joint_pos[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._pole_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_pos[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #         self.joint_vel[:, self._cart_dof_idx[0]].unsqueeze(dim=1),
        #     ),
        #     dim=-1,
        # )
        # observations = {"policy": obs}
        observations = {"policy": torch.zeros(self.num_envs, 21)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self.reset_terminated.float()

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return time_out, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        b = len(env_ids)
        # reset the robot


        # reset blocks
        if self.cfg.block_spawn_mode == "debug":
            block_poses = self.blocks.data.object_pose_w[env_ids].clone()
            block_poses[:, 0, :3] = torch.tensor([0.765, 0.0, 0.0], device=self.device)
            block_poses[:, :, :3] += self.scene.env_origins[env_ids].unsqueeze(1)
            self.blocks.write_object_pose_to_sim(block_poses, env_ids)
        if self.cfg.block_spawn_mode == "cuboid":
            n = self.cfg.num_blocks
            size = int(np.ceil(np.power(n, 1/3)))
            x_offset = 0.4
            y_offset = - (self.cfg.block_size[1] * size) / 2 # symmetric around 0
            block_poses = self.blocks.data.object_pose_w[env_ids].clone()
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        pack_idx = i * size * size + j * size + k
                        if pack_idx < n:
                            block_poses[:, pack_idx, :3] = torch.tensor([x_offset + i * self.cfg.block_size[0], y_offset + j * self.cfg.block_size[1], k * self.cfg.block_size[2] + self.cfg.block_size[2] / 2])
            block_poses[:, :, :3] += self.scene.env_origins[env_ids].unsqueeze(1)
            self.blocks.write_object_pose_to_sim(block_poses, env_ids)
        elif self.cfg.block_spawn_mode == "pile_1":
            n = self.cfg.num_blocks
            size = int(np.ceil(np.power(n, 1/3)))
            x_offset = 0.4
            block_spacing_factor = 2
            wiggle_room = torch.tensor(self.cfg.block_size, device=self.device) * (block_spacing_factor - 1) / 2
            y_offset = - (self.cfg.block_size[1] * block_spacing_factor * size) / 2 # symmetric around 0
            block_poses = self.blocks.data.object_pose_w[env_ids].clone()
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        pack_idx = i * size * size + j * size + k
                        if pack_idx < n:
                            block_poses[:, pack_idx, :3] = torch.tensor([x_offset + i * self.cfg.block_size[0] * block_spacing_factor, y_offset + j * self.cfg.block_size[1] * block_spacing_factor, k * self.cfg.block_size[2] * block_spacing_factor + self.cfg.block_size[2] / 2])

                            quat = quat_from_euler_xyz(torch.rand(b, device=self.device) * 2 * math.pi - math.pi, torch.rand(b, device=self.device) * 2 * math.pi - math.pi, torch.rand(b, device=self.device) * 2 * math.pi - math.pi)
                            block_poses[:, pack_idx, 3:7] = quat
                            block_poses[:, pack_idx, :3] += torch.rand((b, 3), device=self.device) * wiggle_room * 2 - wiggle_room
            block_poses[:, :, :3] += self.scene.env_origins[env_ids].unsqueeze(1)
            self.blocks.write_object_pose_to_sim(block_poses, env_ids)