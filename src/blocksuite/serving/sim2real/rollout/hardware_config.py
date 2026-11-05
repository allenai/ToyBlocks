"""Site-specific hardware constants for the vlagent RCS client."""

from __future__ import annotations

import numpy as np
import rcs
from rcs._core.common import RobotPlatform
from rcs.envs.base import ControlMode, RelativeTo

ROBOT2IP = {
    "right": "192.168.1.12",
}

#ROBOT_INSTANCE = RobotPlatform.SIMULATION
ROBOT_INSTANCE = RobotPlatform.HARDWARE

REALSENSE_CAMERA_DICT = None
ZED_CAMERA_DICT = {
    "gripper_rgb": "14943057",
    "front_rgb": "35115330",
}
HOME_POSE = rcs.HOME_POSITIONS["FR3_DROID"]
ROBOTIQ_SERIAL = {
    "right": "DAANTG8W",
}
INCLUDE_DEPTH = False
INCLUDE_RIGHT = True

CONTROL_MODE = ControlMode.JOINTS
RELATIVETO = RelativeTo.NONE

MAX_REL_MOV_JOINTS = np.deg2rad(0.5)
MAX_REL_MOV_CART = (0.5, np.deg2rad(90))
