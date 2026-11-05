import math
from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.envs.mdp.actions.actions_cfg import (
    BinaryJointPositionActionCfg,
    JointPositionActionCfg,
)
from isaaclab.markers.visualization_markers import VisualizationMarkersCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets.robots.franka import (
    FRANKA_ROBOTIQ_GRIPPER_CFG as ISAAC_FRANKA_ROBOTIQ_GRIPPER_CFG,
)

from blocksuite import FRANKA_ROBOTIQ_NEW_USD_FILEPATH, FRANKA_ROBOTIQ_URDF_FILEPATH
from blocksuite.robots.actions.relative_joint_position import RelativeJointPositionActionCfg
from blocksuite.robots.base_robot import BaseRobot, BaseRobotCfg
from blocksuite.robots.registration import register_robot

# USD prim structure: {ENV_REGEX_NS}/Robot/Geometry/panda_link0 ... panda_link7
# Gripper links under: {ENV_REGEX_NS}/Robot/Geometry/panda_gripper_left_driver, etc.
FRANKA_DROID_ARTICULATION_CFG: ArticulationCfg = deepcopy(ISAAC_FRANKA_ROBOTIQ_GRIPPER_CFG)
FRANKA_DROID_ARTICULATION_CFG.spawn.usd_path = FRANKA_ROBOTIQ_NEW_USD_FILEPATH.as_posix()
# Pre-assembled USD already includes Robotiq; no Isaac Nucleus "Gripper" variant set.
FRANKA_DROID_ARTICULATION_CFG.spawn.variants = None
FRANKA_DROID_ARTICULATION_CFG.spawn.disable_gravity = True
FRANKA_DROID_ARTICULATION_CFG.spawn.activate_contact_sensors = True
FRANKA_DROID_ARTICULATION_CFG.init_state.pos = (0.0, 0.0, 0.0)
FRANKA_DROID_ARTICULATION_CFG.init_state.joint_pos = {
    "panda_joint1": 0.0,
    "panda_joint2": -math.pi / 4,
    "panda_joint3": 0.0,
    "panda_joint4": -math.pi * 3 / 4,
    "panda_joint5": 0.0,
    "panda_joint6": math.pi / 2,
    "panda_joint7": 0,
    "finger_joint": 0.0,
    ".*_inner_finger_joint": 0.0,
    ".*_inner_knuckle_joint": 0.0,
    ".*_outer_.*_joint": 0.0,
}
FRANKA_DROID_ARTICULATION_CFG.actuators = {
    "panda_shoulder": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[1-4]"],
        effort_limit=87.0,
        velocity_limit=2.175,
        stiffness=80.0,
        damping=18.0,
    ),
    "panda_forearm": ImplicitActuatorCfg(
        joint_names_expr=["panda_joint[5-7]"],
        effort_limit=12.0,
        velocity_limit=2.61,
        stiffness=80.0,
        damping=18.0,
    ),
    "gripper": ImplicitActuatorCfg(
        joint_names_expr=["finger_joint"],
        stiffness=50.0,
        damping=20.0,
        effort_limit=0.2,
        velocity_limit=1.0,
    ),
}

### Transformation frame definitions ###
FRAME_MARKER_CFG = VisualizationMarkersCfg(
    markers={
        "frame": sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
            scale=(0.05, 0.05, 0.05),
        ),
    }
)
EE_FRAME_TRANSFORMER_CFG = FrameTransformerCfg(
    prim_path="{ENV_REGEX_NS}/Robot/panda/base_link",
    debug_vis=False,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/panda_link7",
            name="end_effector",
            offset=OffsetCfg(
                pos=[0.0, 0.0, 0.2418],
            ),
        ),
    ],
    visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/franka_droid_ee_frame"),
)

### Controller definitions ###

ARM_RELATIVE_JOINT_TARGET_ACTION = RelativeJointPositionActionCfg(
    joint_names=["panda_joint.*"],
    scale=0.2,
    clip={".*": (-1.0, 1.0)},
    use_target=True,
    max_target_diff_norm=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
)
ROBOTIQ_GRIPPER_BINARY_ACTION = BinaryJointPositionActionCfg(
    joint_names=["finger_joint"],
    open_command_expr={"finger_joint": 0.0},
    close_command_expr={"finger_joint": 0.72},
)

ArmRelativeJointTarget_GripperBinary = dict(
    arm=ARM_RELATIVE_JOINT_TARGET_ACTION,
    gripper=ROBOTIQ_GRIPPER_BINARY_ACTION,
)

ARM_RELATIVE_JOINT_ACTION = RelativeJointPositionActionCfg(
    joint_names=["panda_joint.*"],
    scale=0.2,
    clip={".*": (-1.0, 1.0)},
    use_target=False,
)
ArmRelativeJoint_GripperBinary = dict(
    arm=ARM_RELATIVE_JOINT_ACTION,
    gripper=ROBOTIQ_GRIPPER_BINARY_ACTION,
)
ArmRelativeJoint_NoGripper = dict(
    arm=ARM_RELATIVE_JOINT_ACTION,
)

ARM_JOINT_ACTION = JointPositionActionCfg(
    joint_names=["panda_joint.*"],
    use_default_offset=False,
)
ArmJoint_GripperBinary = dict(
    arm=ARM_JOINT_ACTION,
    gripper=ROBOTIQ_GRIPPER_BINARY_ACTION,
)

ARM_RAW_RELATIVE_JOINT_ACTION = RelativeJointPositionActionCfg(
    joint_names=["panda_joint.*"],
    scale=1.0,
    use_target=True,
    use_zero_offset=True,
)
ArmRawRelativeJoint_GripperBinary = dict(
    arm=ARM_RAW_RELATIVE_JOINT_ACTION,
    gripper=ROBOTIQ_GRIPPER_BINARY_ACTION,
)


@configclass
class FR3RobotiqGripperSoftCfg(BaseRobotCfg):
    articulation: ArticulationCfg = FRANKA_DROID_ARTICULATION_CFG
    ee_body_name: str = "robotiq_arg2f_base_link"
    frame_transformers = {
        "ee": EE_FRAME_TRANSFORMER_CFG,
    }
    controllers = {
        "arm_relative_joint_target_gripper_binary": ArmRelativeJointTarget_GripperBinary,
        "arm_relative_joint_gripper_binary": ArmRelativeJoint_GripperBinary,
        "arm_relative_joint_no_gripper": ArmRelativeJoint_NoGripper,
        "arm_joint_gripper_binary": ArmJoint_GripperBinary,
        "arm_raw_relative_joint_gripper_binary": ArmRawRelativeJoint_GripperBinary,
    }
    cameras = {
        "gripper_camera": TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda/robotiq_arg2f_base_link/GripperCamera",
            update_period=0.04,
            height=360,
            width=640,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "instance_segmentation_fast",
                "semantic_segmentation",
            ],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=666.64,
                focus_distance=400.0,
                horizontal_aperture=1280,
                clipping_range=(0.01, 2.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.031, 0.074, 0.022),
                rot=(-0.000966, 0.005634, -0.169037, 0.985593),
                convention="ros",
            ),
        ),
    }
    contact_sensors = {}


@register_robot()
class FR3RobotiqGripperSoft(BaseRobot):
    cfg: FR3RobotiqGripperSoftCfg
    id = "fr3_robotiq_gripper_soft"
    urdf_path = FRANKA_ROBOTIQ_URDF_FILEPATH.as_posix()

    def open_gripper(self):
        raise NotImplementedError("Open gripper not implemented for FrankaDroid")

    def close_gripper(self):
        raise NotImplementedError("Close gripper not implemented for FrankaDroid")
