from blocksuite.tasks.base_env_cfg import BaseEnvCfg
from isaaclab.utils import configclass
@configclass
class RotateEnvCfg(BaseEnvCfg):
    episode_length_s = 4 # episode horizon = 100

    randomize_block_size = False
    block_size_range = (0.03, 0.05)
    block_size_num_buckets: int = 256
    """Number of discrete USD prototypes when randomize_block_size is on.
    Per-env sizes are quantized to this many buckets to avoid loading one
    USD per environment (which stalls scene creation at high num_envs)."""
    # use this when we don't want to randomize the block size
    block_size = (0.04, 0.04, 0.04)

    distance_threshold = 0.01
    orientation_threshold = 0.0173

    # reach curriculum
    reach_success_threshold = 0.02

    block_spawn_x_range = (0.25, 0.75)
    block_spawn_y_range = (-0.3, 0.3)

    block_spawn_yaw_range = (-1.57, 1.57)

    # reward coefficients
    reach_reward_coef = 1.0 
    corner_reward_coef = 1.0 
    corner_reward2_coef = 1.0 
    axis_angle_reward_coef = 2.0 
    success_reward_coef = 3.0 
    table_collision_penalty_coef = 0.5
    action_magnitude_reward_coef = 0.01

    robot_id: str = "fr3_robotiq_gripper_sysid"
    control_mode: str = "arm_relative_joint_no_gripper"