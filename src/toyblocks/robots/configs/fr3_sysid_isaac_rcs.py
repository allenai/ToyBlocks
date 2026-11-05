from copy import deepcopy

from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets.articulation.articulation_cfg import ArticulationCfg
from isaaclab.utils import configclass

from toyblocks.robots.configs.fr3_softer import (
    FRANKA_DROID_ARTICULATION_CFG,
    FR3RobotiqGripperSofter,
    FR3RobotiqGripperSofterCfg,
)
from toyblocks.robots.registration import register_robot

FR3_SYSID_ISAAC_RCS_ARTICULATION_CFG: ArticulationCfg = deepcopy(FRANKA_DROID_ARTICULATION_CFG)

FR3_SYSID_ISAAC_RCS_ARTICULATION_CFG.actuators["panda_shoulder"] = ImplicitActuatorCfg(
    joint_names_expr=["panda_joint[1-4]"],
    effort_limit=87.0,
    velocity_limit=2.175,
    stiffness=300.0,
    damping=35.0,
    armature={
        "panda_joint1": 0.48452226811378707,
        "panda_joint2": 0.4946407448253728,
        "panda_joint3": 0.4992308742073909,
        "panda_joint4": 0.4740688224432851,
    },
    friction={
        "panda_joint1": 3.291819635492132,
        "panda_joint2": 2.4417331863722342,
        "panda_joint3": 2.956594244011692,
        "panda_joint4": 3.00410180950234,
    },
    dynamic_friction={
        "panda_joint1": 1.9412537263674545,
        "panda_joint2": 1.4442367408597734,
        "panda_joint3": 1.235344723669846,
        "panda_joint4": 1.7368094785526735,
    },
    viscous_friction={
        "panda_joint1": 3.599891233261096,
        "panda_joint2": 3.1261484722465553,
        "panda_joint3": 3.912238871956422,
        "panda_joint4": 3.848929115657661,
    },
)

FR3_SYSID_ISAAC_RCS_ARTICULATION_CFG.actuators["panda_forearm_1"] = ImplicitActuatorCfg(
    joint_names_expr=["panda_joint5"],
    effort_limit=12.0,
    velocity_limit=2.61,
    stiffness=150.0,
    damping=20.0,
    armature=0.3132401575601743,
    friction=0.8691236135856975,
    dynamic_friction=0.8041114788884856,
    viscous_friction=2.616276779805788,
)
FR3_SYSID_ISAAC_RCS_ARTICULATION_CFG.actuators["panda_forearm_2"] = ImplicitActuatorCfg(
    joint_names_expr=["panda_joint6"],
    effort_limit=12.0,
    velocity_limit=2.61,
    stiffness=100.0,
    damping=15.0,
    armature=0.31370087488109333,
    friction=0.8065852891156902,
    dynamic_friction=0.5700099115425594,
    viscous_friction=1.9663741476851488,
)
FR3_SYSID_ISAAC_RCS_ARTICULATION_CFG.actuators["panda_forearm_3"] = ImplicitActuatorCfg(
    joint_names_expr=["panda_joint7"],
    effort_limit=12.0,
    velocity_limit=2.61,
    stiffness=50.0,
    damping=10.0,
    armature=0.17293726460929226,
    friction=0.45900134860600783,
    dynamic_friction=0.4517284125513586,
    viscous_friction=1.1728972853762565,
)


@configclass
class FR3RobotiqGripperSysIdIsaacRcsCfg(FR3RobotiqGripperSofterCfg):
    articulation: ArticulationCfg = FR3_SYSID_ISAAC_RCS_ARTICULATION_CFG


@register_robot()
class FR3RobotiqGripperSysIdIsaacRcs(FR3RobotiqGripperSofter):
    cfg: FR3RobotiqGripperSysIdIsaacRcsCfg
    id = "fr3_robotiq_gripper_sysid"
