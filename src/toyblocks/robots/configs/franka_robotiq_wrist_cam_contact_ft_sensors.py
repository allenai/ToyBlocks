"""Franka + Robotiq with wrist camera and fingertip / wrist contact sensors.

Uses the same articulation, actuators, frame transformer, and controllers as
:class:`~toyblocks.robots.configs.franka.FrankaRobotiqGripper`, and adds the
gripper-mounted pinhole camera plus contact sensors from
``apr15_pnp/envs/problem_color-only-*/init.py``.
"""

from toyblocks import FRANKA_ROBOTIQ_URDF_FILEPATH
from toyblocks.robots.base_robot import BaseRobot
from toyblocks.robots.configs.franka import FrankaRobotiqGripperCfg
from toyblocks.robots.registration import register_robot
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.contact_sensor import ContactSensorCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils


@configclass
class FrankaRobotiqWristCamContactFtSensorsCfg(FrankaRobotiqGripperCfg):
    contact_sensors = {
        "left_finger_contact": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/left_inner_finger",
            update_period=0.0,
            history_length=1,
            force_threshold=0.1,
            debug_vis=False,
        ),
        "right_finger_contact": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/right_inner_finger",
            update_period=0.0,
            history_length=1,
            force_threshold=0.1,
            debug_vis=False,
        ),
        "wrist_ft_sensor": ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/panda_link7",
            update_period=0.0,
            history_length=1,
            force_threshold=0.1,
            debug_vis=False,
        ),
    }
    cameras = {
        "gripper_camera": TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/robotiq_arg2f_base_link/GripperCamera",
            update_period=0.04,
            height=360,
            width=640,
            data_types=["rgb", "distance_to_image_plane", "instance_segmentation_fast", "semantic_segmentation"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8,
                focus_distance=400.0,
                horizontal_aperture=6.916,
                clipping_range=(0.001, 2.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.03915, 0.0, 0.077545),
                rot=(0.701560, -0.088392, -0.088392, 0.701560),
                convention="ros",
            ),
        ),
    }


@register_robot()
class FrankaRobotiqWristCamContactFtSensors(BaseRobot):
    cfg: FrankaRobotiqWristCamContactFtSensorsCfg
    id = "franka_robotiq_wrist_cam_contact_ft_sensors"
    urdf_path = FRANKA_ROBOTIQ_URDF_FILEPATH.as_posix()

    def open_gripper(self):
        raise NotImplementedError("Open gripper not implemented for FrankaRobotiqWristCamContactFtSensors")

    def close_gripper(self):
        raise NotImplementedError("Close gripper not implemented for FrankaRobotiqWristCamContactFtSensors")
