from typing import Sequence

import torch
from isaaclab.envs.manager_based_env import ManagerBasedEnv
from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

class RelativeJointPositionAction(JointAction):
    r"""Joint action term that applies the processed actions to the articulation's joints as relative position commands.

    Unlike :class:`JointPositionAction`, this action term applies the processed actions as relative position commands.
    This means that the processed actions are added to the current joint positions of the articulation's joints
    before being sent as position commands.

    This means that the action applied at every step is:

    .. math::

         \text{applied action} = \text{current joint positions} + \text{processed actions}

    where :math:`\text{current joint positions}` are the current joint positions of the articulation's joints.
    """

    cfg: "RelativeJointPositionActionCfg"
    """The configuration of the action term."""

    def __init__(self, cfg: "RelativeJointPositionActionCfg", env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use zero offset for relative position
        if cfg.use_zero_offset:
            self._offset = 0.0
        if cfg.use_target:
            self._target = self._asset.data.joint_pos[:, self._joint_ids].clone()
            if self.cfg.max_target_diff_norm is not None:
                self._max_target_diff_norm = torch.tensor(self.cfg.max_target_diff_norm, device=self.device)
                self._apply_max_target_per_joint = not isinstance(self.cfg.max_target_diff_norm, float)
                if self._apply_max_target_per_joint:
                    assert len(self.cfg.max_target_diff_norm) == len(self._joint_ids)
    def process_actions(self, actions: torch.Tensor):
        # NOTE: default isaac lab action processing might be backwards, they scale + shift then clip.
        # store the raw actions
        self._raw_actions[:] = actions
        # clip actions
        if self.cfg.clip is not None:
            self._raw_actions = torch.clamp(self._raw_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        # apply the affine transformations
        self._processed_actions = self._raw_actions * self._scale + self._offset

    def apply_actions(self):
        # add current joint positions to the processed actions
        if self.cfg.use_target:
            current_actions = self.processed_actions + self._target
            if self.cfg.max_target_diff_norm is not None:
                current_robot_pos = self._asset.data.joint_pos[:, self._joint_ids]
                target_diff = current_actions - current_robot_pos
                if self._apply_max_target_per_joint:
                    clip_mask = target_diff.abs() > self._max_target_diff_norm
                    current_actions[clip_mask] = current_robot_pos[clip_mask] + self._max_target_diff_norm.repeat(self.num_envs, 1)[clip_mask] * torch.sign(target_diff[clip_mask])
                else:
                    target_diff_norm = target_diff.norm(dim=1)
                    clip_mask = target_diff_norm > self._max_target_diff_norm
                    current_actions[clip_mask] = current_robot_pos[clip_mask] + (target_diff[clip_mask] / target_diff_norm[clip_mask].unsqueeze(1)) * self.cfg.max_target_diff_norm
            self._target = current_actions
        else:
            current_actions = self.processed_actions + self._asset.data.joint_pos[:, self._joint_ids]
        # set position targets
        self._asset.set_joint_position_target(current_actions, joint_ids=self._joint_ids)
    
    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if self.cfg.use_target:
            self._target[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids].clone()

@configclass
class RelativeJointPositionActionCfg(JointActionCfg):
    """Configuration for the relative joint position action term.

    See :class:`RelativeJointPositionAction` for more details.
    """

    class_type: type[ActionTerm] = RelativeJointPositionAction

    use_zero_offset: bool = True
    """Whether to ignore the offset defined in articulation asset. Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to zero.
    """

    use_target: bool = False
    """Whether actions are applied relative to a virtual target position or the current robot 
    position. Defaults to False meaning applied to current robot position."""

    max_target_diff_norm: Sequence[float] | float | None = None
    """Sets a maximum norm for the difference between the virtual target position and the current robot position. If a sequence of floats
    are provided, the maximum norm is applied to each joint separately.
    
    Let target_diff = virtual_target_position - current_robot_position. Normalize target_diff into target_diff_norm.

    For all robots where `target_diff_norm > max_target_diff_norm`, set the target position to
    `current_robot_position + (target_diff / target_diff_norm) * max_target_diff_norm`.

    If None, no clipping is applied.

    Setting this value helps ensure the robot does not move too fast or set the virtual target position so far to the point it is difficult to recover.
    
    This may also be useful for ensuring better exploration for RL?
    """