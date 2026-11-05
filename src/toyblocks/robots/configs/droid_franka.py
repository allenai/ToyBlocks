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
from isaaclab.sensors import CameraCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg, OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from toyblocks.robots.actions.relative_joint_position import RelativeJointPositionActionCfg
from toyblocks.robots.base_robot import BaseRobot, BaseRobotCfg
from toyblocks.robots.registration import register_robot

# NOTE: this config uses an usd we don't have anywhere, maybe it came from nucleus. Be
# aware that this config doesn't work anymore, as it seems nobody is using it

DROID_FRANKA = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        # usd_path=str(ROBOT_ASSET_DIR / "droid_franka/usd/noninstanceable.usd"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            # NOTE: Original codebase https://github.com/arhanjain/polaris/ uses iteration count of 64
            # From my testing it is possible to massively reduce that (to e.g. 8) with minor stability issues
            # once the other PD parameters are tuned
            # I believe 64 was used originally to try and get the mimic joints to converge better
            # while maintaining a lower PD setting for the robot arm. I am not sure why that was done however.
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0, 0, 0),
        rot=(1, 0, 0, 0),
        joint_pos={
            "panda_joint1": 0.0,
            "panda_joint2": -math.pi / 4,
            "panda_joint3": 0.0,
            "panda_joint4": -math.pi * 3 / 4,
            "panda_joint5": 0.0,
            "panda_joint6": math.pi / 2,
            "panda_joint7": 0,
            "finger_joint": 0.0,
            "right_outer.*": 0.0,
            "left_inner.*": 0.0,
            "right_inner.*": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=1,
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=87.0,
            velocity_limit_sim=2.175,
            stiffness=2000.0,
            damping=200.0,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=12.0,
            velocity_limit_sim=2.61,
            stiffness=2000.0,
            damping=200.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            # TODO: this probably needs tuning
            # I did also tune these lower to avoid some sim explosions caused by the gripper forcing the rest of the arm to move
            stiffness=1,
            damping=0.1,
            # stiffness=None,
            # damping=None,
            effort_limit_sim=200.0,
            velocity_limit_sim=5.0,
        ),
    },
)
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
    prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link",
    debug_vis=False,
    target_frames=[
        FrameTransformerCfg.FrameCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link7",
            name="end_effector",
            offset=OffsetCfg(
                pos=[0.0, 0.0, 0.2418],
            ),
        ),
    ],
    visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/franka_ee_frame"),
)

### Controller definitions ###

ARM_RELATIVE_JOINT_TARGET_ACTION = RelativeJointPositionActionCfg(
    # TODO: this fixes gripper drifting issues but may or may not be harder for RL. Might need some tuning
    joint_names=["panda_joint.*"],
    scale=0.2,
    clip={".*": (-1.0, 1.0)},
    use_target=True,
    max_target_diff_norm=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1),
    preserve_order=True,
)
ROBOTIQ_GRIPPER_BINARY_ACTION = BinaryJointPositionActionCfg(
    joint_names=["finger_joint"],
    open_command_expr={"finger_joint": 0.0},
    close_command_expr={"finger_joint": math.pi / 4},
)

ArmRelativeJointTarget_GripperBinary = dict(
    arm=ARM_RELATIVE_JOINT_TARGET_ACTION,
    gripper=ROBOTIQ_GRIPPER_BINARY_ACTION,
)

ARM_RELATIVE_JOINT_ACTION = RelativeJointPositionActionCfg(
    joint_names=["panda_joint.*"],
    scale=0.1,
    clip={".*": (-1.0, 1.0)},
    use_target=False,
    preserve_order=True,
)
ARM_RELATIVE_JOINT_ACTION_SLOW = deepcopy(ARM_RELATIVE_JOINT_ACTION)
ARM_RELATIVE_JOINT_ACTION_SLOW.scale = 0.02

ArmRelativeJoint = dict[str, RelativeJointPositionActionCfg](
    arm=ARM_RELATIVE_JOINT_ACTION,
)

ArmRelativeJoint_GripperBinary = dict(
    arm=ARM_RELATIVE_JOINT_ACTION,
    gripper=ROBOTIQ_GRIPPER_BINARY_ACTION,
)

ArmRelativeJointSlow = dict[str, RelativeJointPositionActionCfg](
    arm=ARM_RELATIVE_JOINT_ACTION_SLOW,
)

ArmRelativeJointSlow_GripperBinary = dict(
    arm=ARM_RELATIVE_JOINT_ACTION_SLOW,
    gripper=ROBOTIQ_GRIPPER_BINARY_ACTION,
)

ARM_JOINT_ACTION = JointPositionActionCfg(
    joint_names=["panda_joint.*"],
    preserve_order=True,
    use_default_offset=False,
)
ArmJoint_GripperBinary = dict(
    arm=ARM_JOINT_ACTION,
    gripper=ROBOTIQ_GRIPPER_BINARY_ACTION,
)


@configclass
class DroidFrankaCfg(BaseRobotCfg):
    articulation = DROID_FRANKA
    frame_transformers = {"ee": EE_FRAME_TRANSFORMER_CFG}
    controllers = {
        "arm_relative_joint_target_gripper_binary": ArmRelativeJointTarget_GripperBinary,
        "arm_relative_joint": ArmRelativeJoint,
        "arm_relative_joint_gripper_binary": ArmRelativeJoint_GripperBinary,
        "arm_relative_joint_slow": ArmRelativeJointSlow,
        "arm_relative_joint_slow_gripper_binary": ArmRelativeJointSlow_GripperBinary,
        "arm_joint_gripper_binary": ArmJoint_GripperBinary,
    }
    cameras = {
        "wrist_camera": CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/Gripper/Robotiq_2F_85/base_link/wrist_cam",
            height=720,
            width=1280,
            data_types=["rgb", "semantic_segmentation"],
            colorize_semantic_segmentation=False,
            update_latest_camera_pose=True,
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=2.8,
                focus_distance=28.0,
                horizontal_aperture=5.376,
                vertical_aperture=3.024,
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.011, -0.031, -0.074),
                rot=(-0.420, 0.570, 0.576, -0.409),
                convention="opengl",
            ),
        )
    }
    contact_sensors = {}


@register_robot()
class DroidFranka(BaseRobot):
    cfg: DroidFrankaCfg
    id = "droid_franka"

    def open_gripper(self):
        raise NotImplementedError("Open gripper not implemented for DroidFranka")

    def close_gripper(self):
        raise NotImplementedError("Close gripper not implemented for DroidFranka")
