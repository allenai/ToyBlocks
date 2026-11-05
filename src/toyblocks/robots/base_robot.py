from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from toyblocks.tasks.base_env import BaseEnv
from dataclasses import dataclass, field

import torch
from isaaclab.assets.articulation.articulation import Articulation
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers.manager_term_cfg import ActionTermCfg
from isaaclab.sensors.camera.camera_cfg import CameraCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg


@dataclass
class BaseRobotCfg:
    articulation: ArticulationCfg
    """the configuration for the articulated asset"""
    frame_transformers: dict[str, FrameTransformerCfg]
    """configuration for any useful frame transformations for the robot"""
    controllers: dict[str, dict[str, ActionTermCfg]]
    """dictionary mapping control mode IDs to controller configurations.
    A single controller configuration is a dictionary mapping sub-controller names to their action term configs."""
    cameras: dict[str, CameraCfg]
    """dictionary mapping camera names to their camera configurations. These are cameras specific to this robot/embodiment, typically mounted."""
    contact_sensors: dict[str, ContactSensorCfg] = field(default_factory=dict)
    """Optional per-link contact sensors (e.g. fingertips, wrist). Spawned in :meth:`toyblocks.tasks.base_env.BaseEnv._setup_scene` as ``contact:<name>``."""
    ee_body_name: str = "panda_hand"
    """End effector base body name and must match the body link from which ``observation.state.ee_position`` is recorded during datagen."""
    arm_joint_names: tuple[str, ...] = tuple(f"panda_joint{i}" for i in range(1, 8))
    """Arm joints in kinematic-chain order, base to tip. The recorder's FK of ``action.ee_target`` (:class:`toyblocks.vecgen.recording.articulation_fk.ArticulationEEFK`) walks this chain."""
    arm_link_names: tuple[str, ...] = tuple(f"panda_link{i}" for i in range(1, 8))
    """Child link of each arm joint, same order as :attr:`arm_joint_names`. The base link is not listed: the FK chain starts at the articulation root."""


class BaseRobot:
    id: str
    """a unique identifier/shorthand for the robot class"""
    urdf_path: str | None = None
    """the path to the URDF file for the robot if it exists"""
    cfg: BaseRobotCfg
    """the configuration for the robot"""
    articulation: Articulation
    """The spawned articulation asset."""
    controller: dict[str, JointAction] = dict()
    """The active controller dict for the robot."""
    _control_mode: str

    def __init__(self, cfg: BaseRobotCfg, env: BaseEnv):
        self._env = env
        self.cfg = cfg
        self.articulation = Articulation(cfg.articulation)

    @property
    def control_mode(self):
        return self._control_mode

    @control_mode.setter
    def control_mode(self, value):
        assert value in self.cfg.controllers, (
            f"Control mode {value} not found. Available control modes are {list(self.cfg.controllers.keys())}"
        )
        self._control_mode = value
        # change controller part internally
        controller_cfg = self.cfg.controllers[value]
        self.controller = dict[str, JointAction]()
        for k, v in controller_cfg.items():
            self.controller[k] = v.class_type(cfg=v, env=self._env)
        self._actuator_indices_dict = {
            actuator_name: self.articulation.actuators[actuator_name].joint_indices
            for actuator_name in self.articulation.actuators
        }
        self._actuator_indices = torch.cat(
            [
                self._actuator_indices_dict[actuator_name]
                for actuator_name in self.articulation.actuators
            ],
            dim=0,
        )

    @property
    def actuated_indices(self):
        return self._actuator_indices

    @property
    def data(self):
        return self.articulation.data

    def write_data_to_sim(self):
        self.articulation.write_data_to_sim()
