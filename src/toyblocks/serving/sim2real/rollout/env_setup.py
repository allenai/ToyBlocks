"""RCS environment setup for vlagent sim-to-real inference."""

from __future__ import annotations

import numpy as np
import rcs
from toyblocks.serving.sim2real.rollout.camera_utils import build_hardware_camera_cfgs
from toyblocks.serving.sim2real.rollout.hardware_config import (
    CONTROL_MODE,
    HOME_POSE,
    INCLUDE_DEPTH,
    RELATIVETO,
    ROBOT2IP,
    ROBOT_INSTANCE,
    ROBOTIQ_SERIAL,
)
from toyblocks.serving.sim2real.rollout.inference_config import InferenceConfig
from toyblocks.serving.sim2real.rollout.tam_utils import tam_ideal_model_xml

from duobench.tasks.flipping import FlippingEnvConfig
from rcs._core.common import Pose, RobotPlatform
from rcs._core.sim import SimConfig
from rcs.envs.base import ControlMode
from rcs.envs.storage_wrapper import StorageWrapper

# ``rcs.ROBOTS[FR3].joint_limits`` is a conservative envelope, not the arm's
# travel: it stops joint 4 at -2.7478 while the FR3 MJCF (assets/robots/fr3/fr3.xml)
# reaches -3.0421, so sim trajectories that go to -2.894 leave joint 4 pinned at
# its floor. Joint 4's lower bound is the ONLY limit we override; every other
# joint keeps the rcs default.
JOINT4_MJCF_LIMITS: dict[int, str] = {3: "low"}
JOINT_LIMIT_MARGIN = 0.05  # rad, kept clear of the MJCF hard stops

# --- joint PD gains: ONE source of truth for hardware and sim -----------------
# Both branches of get_env() read these, so a sim run and a hardware run are
# driven by the same controller and any sim-vs-real difference is dynamics, not
# mismatched gains. System identification depends on this (see
# sysid/ISAAC_SYSID_PLAN.md S1: stiffness/damping are FIXED, not fit) -- if the
# branches drift apart again, the fit silently absorbs the gain error into
# armature/friction. Sim names the damping term ``kv``; it is the same quantity
# as the hardware's ``kd``.
#
# Previous profiles, kept for reference:
#   stiff      kp [600,600,600,600,250,150, 50]  kd [50,50,50,50,30,25,15]
#   softer     kp [ 40, 40, 40, 40, 40, 40, 40]  kd [13,13,13,13,13,13,13]
#   flip_soft  kp [200,200,200,200, 50, 50, 30]  kd [40,40,40,40,10,10, 8]
#   soft       kp [100,100,100,100, 75,150, 50]  kd [20,20,20,20,7.5,15, 5]
PD_PROFILE = "kp300"
PD_KP = np.array([300.0, 300.0, 300.0, 300.0, 150.0, 100.0, 50.0])
PD_KD = np.array([35.0, 35.0, 35.0, 35.0, 20.0, 15.0, 10.0])
PD_TORQUE_LIMIT = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0])


def override_joint_limits(
    robot_cfg,
    overrides: dict[int, str] = JOINT4_MJCF_LIMITS,
    margin: float = JOINT_LIMIT_MARGIN,
) -> None:
    """Replace configured joint bounds with the arm's MJCF travel, less ``margin``.

    ``overrides`` maps a joint index to the side to take from the model: "low",
    "high" or "both". Bounds are derived from the kinematic model, so no override
    can reach past the hard stops whatever is asked for.
    """
    import mujoco

    mjcf = rcs.ROBOTS[robot_cfg.robot_type].mjcf_model_path
    travel = np.array(mujoco.MjModel.from_xml_path(mjcf).jnt_range[: int(robot_cfg.dof)]).T
    limits = np.array(robot_cfg.joint_limits, dtype=np.float64)
    for j, side in overrides.items():
        if side not in ("low", "high", "both"):
            raise ValueError(f"joint {j + 1}: side must be low/high/both, got {side!r}")
        if side in ("low", "both"):
            limits[0][j] = travel[0][j] + margin
        if side in ("high", "both"):
            limits[1][j] = travel[1][j] - margin
    robot_cfg.joint_limits = limits


def get_env(cfg: InferenceConfig) -> StorageWrapper:
    if ROBOT_INSTANCE == RobotPlatform.HARDWARE:
        from rcs_fr3.configs import DROIDEnv

        env_creator = DROIDEnv()
        env_creator.robot_ip = ROBOT2IP["right"]
        env_creator.gripper_serial_number = ROBOTIQ_SERIAL["right"]
        hw_cfg = env_creator.config()
        hw_cfg.robot_cfgs["right"].q_home = HOME_POSE
        hw_cfg.robot_cfgs["right"].tcp_offset = Pose()
        override_joint_limits(hw_cfg.robot_cfgs["right"])
        camera_cfgs = build_hardware_camera_cfgs()
        hw_cfg.camera_cfgs = camera_cfgs or None
        hw_cfg.control_mode = CONTROL_MODE
        hw_cfg.gripper_cfgs["right"].async_control = False
        hw_cfg.wrapper_cfg.include_depth = INCLUDE_DEPTH
        hw_cfg.wrapper_cfg.binary_gripper = True
        hw_cfg.wrapper_cfg.home_on_reset = False
        hw_cfg.max_relative_movement = (
            cfg.max_rel_mov_joints
            if CONTROL_MODE == ControlMode.JOINTS
            else cfg.max_rel_mov_cart
        )
        hw_cfg.relative_to = RELATIVETO
        hw_cfg.robot_to_shared_base_frame = {
            "right": rcs.common.Pose(
                translation=np.array([0, 0, 0]), rpy_vector=np.array([0, 0, 0])
            ),
        }
        hw_cfg.robot_cfgs["right"].ignore_realtime = True
        hw_cfg.robot_cfgs["right"].tam_enabled = cfg.tam
        if cfg.tam:
            # TAM's training convention: the plant sees the ideal model's
            # gravity compensation (see FrankaConfig.tam_ideal_model_path).
            hw_cfg.robot_cfgs["right"].tam_ideal_model_path = tam_ideal_model_xml()
        hw_cfg.robot_cfgs["right"].speed_factor = 0.4
        hw_cfg.robot_cfgs["right"].policy_rate = cfg.fps
        hw_cfg.robot_cfgs["right"].allow_high_collision = True
        hw_cfg.frequency = cfg.fps
        env_rel = env_creator.create_env(hw_cfg)
        robot = env_rel.get_wrapper_attr("robot")["right"]
        rcfg = robot.get_config()
        # Gains come from the module-level PD_* constants (shared with the sim
        # branch below) -- edit them there, not here.
        rcfg.kp = PD_KP.copy()
        rcfg.kd = PD_KD.copy()
        rcfg.torque_limit = PD_TORQUE_LIMIT.copy()
        if cfg.tam:
            # ---- TAM ---- #
            # kp/kd/torque_limit stay whatever the block above selected; TAM
            # only adds its residual on top of that PD controller.
            rcfg.rate_limit = cfg.rate_limit
            rcfg.tam_history_pre_ratelimit = cfg.tam_history_pre_ratelimit
            if cfg.tam_residual_clip is not None:
                rcfg.tam_residual_clip = np.array(
                    cfg.tam_residual_clip, dtype=np.float64
                )

        robot.set_config(rcfg)
        stiffness = {
            "profile": PD_PROFILE,
            "kp": np.asarray(rcfg.kp).tolist(),
            "kd": np.asarray(rcfg.kd).tolist(),
            "torque_limit": np.asarray(rcfg.torque_limit).tolist(),
            "tam": cfg.tam,
        }
    else:
        scene = FlippingEnvConfig()
        sim_cfg_data = scene.config()
        # Same PD gains as the hardware branch (sim calls the damping term kv).
        sim_cfg_data.robot_cfgs["right"].kp = PD_KP.copy()
        sim_cfg_data.robot_cfgs["right"].kv = PD_KD.copy()
        sim_cfg_data.robot_cfgs["right"].q_home = HOME_POSE
        sim_cfg_data.sim_cfg = SimConfig(
            async_control=True,
            realtime=True,
            frequency=cfg.fps,
            max_convergence_steps=500,
        )
        sim_cfg_data.wrapper_cfg.include_depth = INCLUDE_DEPTH
        sim_cfg_data.control_mode = ControlMode.JOINTS
        sim_cfg_data.relative_to = RELATIVETO
        sim_cfg_data.wrapper_cfg.binary_gripper = True
        sim_cfg_data.headless = True
        sim_cfg_data.max_relative_movement = (
            cfg.max_rel_mov_joints
            if CONTROL_MODE == ControlMode.JOINTS
            else cfg.max_rel_mov_cart
        )
        env_rel = scene.create_env(sim_cfg_data)
        stiffness = {
            "profile": PD_PROFILE,
            "kp": np.asarray(sim_cfg_data.robot_cfgs["right"].kp).tolist(),
            "kv": np.asarray(sim_cfg_data.robot_cfgs["right"].kv).tolist(),
            "tam": False,
        }

    env = StorageWrapper(
        env_rel,
        cfg.record_path,
        cfg.instruction,
        batch_size=32,
        max_rows_per_group=2,
        max_rows_per_file=10,
        allow_wrapper_instruction=False,
    )
    # Which of the commented-out PD blocks above was actually active. Read back
    # by eval_log.stiffness_profile for the eval log's run header.
    env.stiffness_profile = stiffness
    return env
