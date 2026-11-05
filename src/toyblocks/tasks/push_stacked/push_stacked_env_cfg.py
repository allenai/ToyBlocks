from toyblocks.tasks.base_env_cfg import BaseEnvCfg
from isaaclab.utils import configclass
@configclass
class PushStackedEnvCfg(BaseEnvCfg):
    episode_length_s = 4 # episode horizon = 100

    # use this when we don't want to randomize the block size
    block_size = (0.04, 0.04, 0.04)
    randomize_block_size = False
    block_size_range = (0.03, 0.05)
    block_size_num_buckets: int = 256
    """Number of discrete USD prototypes when randomize_block_size is on.
    Per-env sizes are quantized to this many buckets to avoid loading one
    USD per environment (which stalls scene creation at high num_envs)."""

    block_spawn_x_range = (0.25, 0.75) 
    block_spawn_y_range = (-0.3, 0.3)
    block_spawn_yaw_range = (-1.571, 1.571)

    goal_xy_offset_min_radius = 0.15
    goal_xy_offset_max_radius = 0.40

    # Goal is clipped to stay on the table surface 
    goal_spawn_x_range = (0.25, 0.75)
    goal_spawn_y_range = (-0.3, 0.3)

    # success criteria 
    distance_threshold = 0.03
    orientation_threshold = 1.5 
    distractor_distance_threshold = 0.03
    distractor_orientation_threshold = 0.3

    # reach curriculum
    reach_success_threshold = 0.02

    # Fraction of the gripper's travel to close to while dragging.
    gripper_close_fraction: float = 0.8

    # distractor spawn params
    num_distractors = 9        # maximum (preallocated); actual count sampled each episode
    min_num_distractors = 0    # minimum active distractors per episode
    max_stack_height = 5

    # Uniform spawn area for distractors: 60cm x 60cm region on the table
    distractor_spawn_x_range = (0.25, 0.75)  # 60 cm span in x
    distractor_spawn_y_range = (-0.3, 0.3)     # 60 cm span in y

    # reward coefficients
    reach_reward_coef = 1.0
    corner_reward_coef = 1.0
    corner_reward2_coef = 1.0
    axis_angle_reward_coef = 0.0
    action_smoothness_reward_coef = 0.0
    action_magnitude_reward_coef = 0.01
    distractor_reward_coef = 1.0
    success_reward_coef = 3.0
    table_collision_penalty_coef = 0.5

    # Proactive collision-avoidance clearance penalty. 
    clearance_penalty_coef = 0.0
    clearance_safety_margin = 0.05

    robot_id: str = "fr3_robotiq_gripper_sysid"
    control_mode: str = "arm_relative_joint_no_gripper"
