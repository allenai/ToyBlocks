# Copyright (c) 2024-2025, The Octi Lab Project Developers. (https://github.com/zoctipus/OctiLab/blob/main/CONTRIBUTORS.md).
# Proprietary and Confidential - All Rights Reserved.
#
# Unauthorized copying of this file, via any medium is strictly prohibited

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

class DefaultJointPositionStaticAction(JointAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    cfg: "DefaultJointPositionStaticActionCfg"
    """The configuration of the action term."""

    def __init__(self, cfg: "DefaultJointPositionStaticActionCfg", env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        self._default_actions = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return 0

    def process_actions(self, actions: torch.Tensor):
        pass

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(self._default_actions, joint_ids=self._joint_ids)

@configclass
class DefaultJointPositionStaticActionCfg(JointActionCfg):
    """Configuration for the joint position action term.

    See :class:`JointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = DefaultJointPositionStaticAction

    use_default_offset: bool = True