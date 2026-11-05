from blocksuite.tasks.base_env_cfg import BaseEnvCfg
from isaaclab.utils import configclass
@configclass
class AlignEnvCfg(BaseEnvCfg):
    episode_length_s = 4 # episode horizon = 100

    # use this when we don't want to randomize the block size
    block_size = (0.04, 0.04, 0.04)
    randomize_block_size = False
    block_size_range = (0.03, 0.05)
    block_size_num_buckets: int = 256
    """Number of discrete USD prototypes when randomize_block_size is on.
    Per-env sizes are quantized to this many buckets to avoid loading one
    USD per environment (which stalls scene creation at high num_envs)."""

    # success criteria
    distance_threshold = 0.01
    orientation_threshold = 0.2

    # reach curriculum
    reach_success_threshold = 0.01

    # Fraction of the gripper's travel to close to while dragging. 
    gripper_close_fraction: float = 0.8

    block_spawn_x_range = (0.30, 0.75)
    block_spawn_y_range = (-0.3, 0.3)
    block_spawn_yaw_range = (-0.785, 0.785)

    goal_xy_offset_min_radius = 0.05
    goal_xy_offset_max_radius = 0.15
    goal_spawn_yaw_range = (-1.57, 1.57)

    # distractor spawn params — slot 0 (cardinal, next to goal) uses tight thresholds;
    # additional distractors (slots 1+) use loose thresholds
    distractor_distance_threshold = 0.02
    distractor_orientation_threshold = 0.1
    extra_distractor_distance_threshold = 0.03
    extra_distractor_orientation_threshold = 0.52
    distractor_offset_from_goal = 0.0475  # distance of distractor 0 from goal center (cardinal placement)
    num_distractors = 4        # maximum (preallocated); actual count sampled each episode
    min_num_distractors = 1    # minimum active distractors per episode
    max_stack_height = 4

    distractor_spawn_x_range = (0.25, 0.75)
    distractor_spawn_y_range = (-0.3, 0.3)

    # reward coefficients
    reach_reward_coef = 1.0
    corner_reward_coef = 1.0 
    corner_reward2_coef = 1.0 
    success_reward_coef = 3.0 
    axis_angle_reward_coef = 0.5
    distractor_reward_coef = 1.0
    extra_distractor_reward_coef = 1.0
    table_collision_penalty_coef = 0.5
    action_magnitude_reward_coef = 0.01

    robot_id: str = "fr3_robotiq_gripper_sysid"
    control_mode: str = "arm_relative_joint_no_gripper"