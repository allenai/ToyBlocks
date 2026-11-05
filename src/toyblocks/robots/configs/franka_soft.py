import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from toyblocks import FRANKA_ROBOTIQ_URDF_FILEPATH
from toyblocks.robots.actions.relative_joint_position import RelativeJointPositionActionCfg
from toyblocks.robots.base_robot import BaseRobot, BaseRobotCfg
from toyblocks.robots.configs.franka import (
    EE_FRAME_TRANSFORMER_CFG,
    FRANKA_ROBOTIQ_GRIPPER_ARTICULATION_CFG,
    ArmJoint_GripperBinary,
    ArmRelativeJoint_GripperBinary,
    ArmRelativeJointTarget_GripperBinary,
)
from toyblocks.robots.registration import register_robot

FRANKA_ROBOTIQ_GRIPPER_SOFT_ARTICULATION_CFG: ArticulationCfg = (
    FRANKA_ROBOTIQ_GRIPPER_ARTICULATION_CFG.replace(
        actuators={
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
                stiffness=200.0,
                damping=20.0,
                effort_limit=1.0,
                velocity_limit=1.0,
            ),
        }
    )
)

ARM_RELATIVE_JOINT_ACTION = RelativeJointPositionActionCfg(
    joint_names=["panda_joint.*"],
    scale=0.2,
    clip={".*": (-1.0, 1.0)},
    use_target=False,
)

ArmRelativeJoint_NoGripper = dict(
    arm=ARM_RELATIVE_JOINT_ACTION,
)


@configclass
class FrankaRobotiqGripperSoftCfg(BaseRobotCfg):
    articulation: ArticulationCfg = FRANKA_ROBOTIQ_GRIPPER_SOFT_ARTICULATION_CFG
    ee_body_name: str = "robotiq_arg2f_base_link"
    frame_transformers = {"ee": EE_FRAME_TRANSFORMER_CFG}
    controllers = {
        "arm_relative_joint_target_gripper_binary": ArmRelativeJointTarget_GripperBinary,
        "arm_relative_joint_gripper_binary": ArmRelativeJoint_GripperBinary,
        "arm_relative_joint_no_gripper": ArmRelativeJoint_NoGripper,
        "arm_joint_gripper_binary": ArmJoint_GripperBinary,
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
class FrankaRobotiqGripper(BaseRobot):
    cfg: FrankaRobotiqGripperSoftCfg
    id = "franka_robotiq_gripper_soft"
    urdf_path = FRANKA_ROBOTIQ_URDF_FILEPATH.as_posix()

    ### Various pre-programmed behaviors ###
    def open_gripper(self):
        raise NotImplementedError("Open gripper not implemented for FrankaRobotiqGripper")

    def close_gripper(self):
        raise NotImplementedError("Close gripper not implemented for FrankaRobotiqGripper")
