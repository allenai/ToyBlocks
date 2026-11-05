import torch
from collections.abc import Sequence
from toyblocks.tasks.base_env import BaseEnv
from toyblocks.utils import rigid_objects
from isaaclab.sensors.frame_transformer.frame_transformer import FrameTransformer
from .lift_env_cfg import LiftEnvCfg


class LiftEnv(BaseEnv):
    cfg: LiftEnvCfg

    def _setup_scene_objects(self):
        self.block = rigid_objects.create_toy_block(
            prim_path="/World/envs/env_.*/Block",
            size=self.cfg.block_size,
            init_pos=(0.0, 0.0, self.cfg.block_size[2] / 2),
        )
        self.scene.rigid_objects["block"] = self.block

        self.goal = rigid_objects.create_cuboid(
            prim_path="/World/envs/env_.*/Goal",
            size=self.cfg.block_size,
            diffuse_color=(0.0, 1.0, 0.0),
            init_pos=(0.0, 0.0, self.cfg.block_size[2] / 2),
            kinematic=True,
            collision=False,
        )
        self.scene.rigid_objects["goal"] = self.goal
        
    @property
    def ee_frame(self) -> FrameTransformer:
        return self.scene.sensors["frame_transformer:ee"]

    def _get_obs_extra(self) -> dict:
        object_pos = self.block.data.root_pos_w - self.scene.env_origins
        object_quat = self.block.data.root_quat_w
        tcp_pos = self.ee_frame.data.target_pos_w[:, 0] - self.scene.env_origins
        tcp_quat = self.ee_frame.data.target_quat_w[:, 0]
        object_to_tcp = tcp_pos - object_pos
        object_to_goal = self.goal.data.root_pos_w - self.block.data.root_pos_w
        goal_pos = self.goal.data.root_pos_w - self.scene.env_origins
        return {"object_pos": object_pos, "object_quat": object_quat, "goal_pos": goal_pos, "object_to_tcp": object_to_tcp, "object_to_goal": object_to_goal, "tcp_pos": tcp_pos, "tcp_quat": tcp_quat}

    def _get_rewards(self) -> torch.Tensor:
        # TODO: isaac lab uses a _compute_intermediate_values method to avoid recomputing lines like below. not sure if we should adopt that as well
        # for ref, maniskill does the same but instead passes the output of get_observations as a dictionary to functions like _get_rewards
        object_to_tcp = self.ee_frame.data.target_pos_w[:, 0] - self.block.data.root_pos_w
        reach_reward = 1 - torch.tanh(object_to_tcp.norm(dim=-1) * 5)
        lift_reward = torch.zeros_like(reach_reward)
        lifted = self.block.data.root_pos_w[:, 2] > self.cfg.block_size[2] / 2 + 5e-4
        lift_reward[lifted] = 1.0
        object_to_goal = self.goal.data.root_pos_w - self.block.data.root_pos_w
        goal_reward = 1 - torch.tanh(object_to_goal.norm(dim=-1) * 5)
        goal_reward[~lifted] = 0.0
        
        reward = reach_reward + lift_reward + goal_reward
        reward[object_to_goal.norm(dim=-1) < self.cfg.block_size[2]] += 1.0

        return reward / 4.0


    def evaluate(self) -> dict:
        result = super().evaluate()
        object_to_goal = self.goal.data.root_pos_w - self.block.data.root_pos_w
        object_to_tcp = self.ee_frame.data.target_pos_w[:, 0] - self.block.data.root_pos_w
        result["success"] = torch.linalg.norm(object_to_goal, dim=-1) < self.cfg.block_size[2]
        stats = result.setdefault("stats", {})
        stats["object_to_goal_dist"] = torch.linalg.norm(object_to_goal, dim=-1)
        stats["object_to_tcp_dist"] = object_to_tcp.norm(dim=-1)
        return result

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        with torch.device(self.device):
            b = len(env_ids)
            block_pos = torch.zeros(b, 7)
            block_pos[:, 3:] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            block_pos[:, 0] = 0.615
            block_pos[:, 2] = self.cfg.block_size[2] / 2 + 5e-4
            block_pos[:, :2] += torch.randn(b, 2) * 0.1 - 0.05
            block_pos[:, :3] += self.scene.env_origins[env_ids]
            self.block.write_root_pose_to_sim(block_pos, env_ids=env_ids)
            self.block.write_root_velocity_to_sim(torch.zeros(b, 6), env_ids=env_ids)

            goal_pos = block_pos.clone()
            goal_pos[:, 2] = 0.25
            self.goal.write_root_pose_to_sim(goal_pos, env_ids=env_ids)
        