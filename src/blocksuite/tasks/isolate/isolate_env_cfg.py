from blocksuite.tasks.base_env_cfg import BaseEnvCfg
from isaaclab.utils import configclass
@configclass
class IsolateEnvCfg(BaseEnvCfg):
    episode_length_s = 2.56 # episode horizon = 64

    # use this when we don't want to randomize the block size
    block_size = (0.04, 0.04, 0.04)
    randomize_block_size = False
    block_size_range = (0.03, 0.05)
    block_size_num_buckets: int = 256
    """Number of discrete USD prototypes when randomize_block_size is on.
    Per-env sizes are quantized to this many buckets to avoid loading one
    USD per environment (which stalls scene creation at high num_envs)."""

    # block spawn params
    block_spawn_x_range = (0.30, 0.75)
    block_spawn_y_range = (-0.3, 0.3)
    block_spawn_yaw_range = (-0.785, 0.785)

    # success criteria 
    distance_threshold = 0.01
    orientation_threshold = 1.5
    distractor_distance_threshold = 0.05
    distractor_orientation_threshold = 0.6

    # reach curriculum
    reach_success_threshold = 0.02

    # Fraction of the gripper's travel to close to while dragging. 
    gripper_close_fraction: float = 0.8

    # distractor params
    num_distractors = 5
    min_num_distractors = 1

    # clutter spawn params
    # spacing between cube centers = block_size * sqrt(2) + clutter_grid_gap to safely allow random yaw
    clutter_grid_gap = 0.0         # gap added on top of diagonal block extent 
    clutter_jitter = 0.005        # max xy jitter added to each cube's grid position (must be < clutter_grid_gap / 2)
    goal_distance_min = 0.10
    goal_distance_max = 0.15

    # reward coefficients
    reach_reward_coef = 1.0 
    corner_reward_coef = 1.0 
    corner_reward2_coef = 1.0 
    distractor_reward_coef = 0.5
    success_reward_coef = 3.0
    table_collision_penalty_coef = 0.5
    
    robot_id: str = "fr3_robotiq_gripper_sysid"
    control_mode: str = "arm_relative_joint_no_gripper"
    
    