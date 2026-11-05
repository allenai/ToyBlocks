"""
Minimal template environment. If making a new environment, copy the _template folder this file is in
"""

import torch
from collections.abc import Sequence
from blocksuite.tasks.base_env import BaseEnv
from ._template_env_cfg import TemplateEnvCfg


class TemplateEnv(BaseEnv):
    cfg: TemplateEnvCfg

    def _setup_scene_objects(self):
        # add objects other than the robot
        pass

    def _after_scene_setup(self):
        # before the first environment is reset, we call this function once
        # useful for initializing anything that may need access to instantiated objects e.g. kinematics
        pass

    def _compute_intermediate_values(self):
        # compute any intermediate values needed for other functions such as observations or rewards
        pass

    def _get_obs_robot(self) -> dict:
        # By default this returns agent observerables, {"joint_pos", "joint_targets"}
        # Override this to modify that part of the observation
        return super()._get_obs_robot()

    def _get_obs_extra(self) -> dict:
        # By default observations only include the robot's information returned by _get_obs_robot
        # to add more information like ground truth object poses return them here as a dictionary
        # Whatever is returned here will also be used to automatically create the observation space
        observations = {"random_pose": torch.rand(self.num_envs, 7)}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        return self.reset_terminated.float()

    def evaluate(self) -> dict:
        result = super().evaluate()
        result["success"] = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        return result

    def _reset_idx(self, env_ids: Sequence[int] | None):
        super()._reset_idx(env_ids)
        with torch.device(self.device):
            b = len(env_ids) # batch size
            # implement reset logic below
            torch.zeros(b, 7, device=self.device)
        