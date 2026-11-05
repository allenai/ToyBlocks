"""Config for the "R on Red" task.

Matches the scene/physics of the old TAMP ``BlocksSceneCfg`` used to collect
the ``mar16_red_high`` dataset (``/weka/jasonl/TAMP/datasets/unmerged/``).
The goal is to place a red R-shaped block on a red target cube in the
presence of three blue distractor cubes.

This task is primarily used for replaying TAMP-recorded trajectories through
``scripts/evaluate.py``.
"""

from isaaclab.utils import configclass

from toyblocks.tasks.base_env_cfg import BaseEnvCfg


@configclass
class ROnRedEnvCfg(BaseEnvCfg):
    episode_length_s: float = 30.0
    robot_id: str = "franka_robotiq_gripper"
    control_mode: str = "arm_joint_gripper_binary"
    scene_preset_id: str = "droid_with_table"
    obs_mode: str = "state"
    auto_reset: bool = False
    ignore_terminations: bool = True
    actuator_profile: str = "tamp"

    block_size: tuple = (0.045, 0.045, 0.045)
    """Size of the cuboid distractor / target objects (matches mar16_red_high)."""

    table_pos: tuple = (0.5, 0.0, 0.0)
    """Table position override — old TAMP used [0.5, 0, 0]; ToyBlocks default
    is [0.55, 0, 0.003]."""
