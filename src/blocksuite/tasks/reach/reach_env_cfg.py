from isaaclab.utils import configclass

from blocksuite.tasks.base_env_cfg import BaseEnvCfg


@configclass
class ReachEnvCfg(BaseEnvCfg):
    """Collision-free reaching to a commanded end-effector pose.

    Derived from ``PushStackedEnvCfg`` rather than align: push already randomizes the
    arm's start configuration through IK and allows zero active obstacles, both of
    which reaching needs. The differences from push are the obstacle budget (20 vs 9),
    the goal being an EE pose rather than a block pose, and ``obstructed_fraction``
    below.
    """

    episode_length_s = 6  # longer than push's 4 -- traversing the workspace takes more steps

    # Obstacles share one per-env cube size, as in push. Sizes are fixed per env at
    # construction because Isaac cannot rescale at runtime.
    block_size = (0.04, 0.04, 0.04)
    randomize_block_size = False
    block_size_range = (0.03, 0.05)
    block_size_num_buckets: int = 256
    """Discrete USD prototypes when randomize_block_size is on; per-env sizes are
    quantized to this many buckets so scene creation does not stall at high num_envs."""

    # ------------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------------
    num_obstacles = 20  # maximum (preallocated); actual count sampled each episode
    min_num_obstacles = 0  # covers the whole 0..20 range, as push does with 0..9
    max_stack_height = 5

    obstacle_spawn_x_range = (0.25, 0.75)
    obstacle_spawn_y_range = (-0.3, 0.3)

    obstructed_fraction = 0.5
    """Fraction of episodes in which some obstacles are deliberately seeded onto the
    straight line from the start EE position to the goal.

    Uniform placement over the table mostly leaves that line clear, so a policy trained
    on it learns to travel straight and never learns avoidance. Log
    ``stats["corridor_blocked"]`` to see what this is actually buying."""

    obstructed_count_range = (1, 3)
    """How many obstacles get seeded into the corridor when an episode is obstructed."""

    corridor_perp_offset = 0.03
    """Perpendicular jitter applied to corridor-seeded obstacles, so they sit *near*
    the line rather than exactly on it."""

    # ------------------------------------------------------------------
    # Attention-set pruning
    # ------------------------------------------------------------------
    num_attended_obstacles = 8
    """Obstacles fed to the policy per step, selected by proximity to the swept
    corridor. Unselected slots are zeroed, which is exactly how ``AttentionMLPAgent``
    detects inactive tokens -- so this needs no change to the agent.

    Relevance is distance to the segment [tcp, goal], not distance to the TCP: an
    obstacle far from the current gripper can be the one sitting in the path."""

    prune_hysteresis_steps = 5
    """Once selected, keep an obstacle in the attention set for this many steps. Without
    it, obstacles pop in and out as the EE moves, which is a step change in the
    observation. Set to 0 to disable."""

    # ------------------------------------------------------------------
    # Start / goal sampling
    # ------------------------------------------------------------------
    start_ee_x_range = (0.25, 0.70)
    start_ee_y_range = (-0.35, 0.35)
    start_ee_z_range = (0.06, 0.45)
    """Workspace box for the start EE position. Much wider than push, which samples a
    thin shell above the block it is about to push. A reach policy asked to start from a
    post-manipulation pose will fail if it only ever trained from near-home starts."""

    start_at_home_fraction = 0.2
    """Fraction of episodes that start at the robot's default joint configuration, so
    the common case stays well covered."""

    goal_ee_x_range = (0.25, 0.70)
    goal_ee_y_range = (-0.35, 0.35)
    goal_ee_z_range = (0.06, 0.45)

    goal_yaw_range = (-3.1416, 3.1416)
    goal_pitch_range = (-0.6, 0.6)
    """Goal orientation is sampled about a downward-pointing gripper, as push does for
    its start pose."""

    min_start_goal_distance = 0.15
    """Resample the goal if it lands too close to the start -- otherwise a large share
    of episodes are solved by the initial pose and teach nothing."""

    ik_residual_tolerance = 0.035
    """An IK solution is accepted when the achieved TCP is within this of the request.
    ``ik_set_joint_pos`` writes joint state with no reachability or collision check, so
    the residual is the only evidence a pose was attainable -- unlike obstacle placement
    this genuinely needs a rejection loop.

    Measured on a 5090: ``KinematicsCfg.num_ik_iterations = 25`` at a 0.25 damped step
    converges to ~2 cm on a reachable target, while an unreachable one sits at >13 cm.
    0.035 separates those two cases. Tightening this without also raising
    ``num_ik_iterations`` just makes every reset exhaust its resamples."""

    ik_max_resamples = 4

    # ------------------------------------------------------------------
    # Success / collision
    # ------------------------------------------------------------------
    distance_threshold = 0.03
    orientation_threshold = 0.2
    """Tighter than push's 1.5: reaching is a pose task, so orientation matters."""

    collision_force_threshold = 1.0
    """Contact force (N) above which an arm link counts as having collided."""

    # ------------------------------------------------------------------
    # Reward coefficients
    # ------------------------------------------------------------------
    reach_reward_coef = 1.0
    reach_fine_reward_coef = 1.0
    orientation_reward_coef = 0.5
    success_reward_coef = 3.0
    action_magnitude_reward_coef = 0.01

    collision_penalty_coef = 2.0
    """Penalty for an actual contact. Dominant term -- a collided episode is a failure
    regardless of how close to the goal it got."""

    obstacle_moved_penalty_coef = 1.0
    """Displacement-from-spawn backstop, in case a contact slips past the sensors."""

    table_collision_penalty_coef = 0.5

    clearance_penalty_coef = 0.5
    clearance_safety_margin = 0.05
    """Proactive barrier, reusing push's formulation. Non-zero by default here (push
    defaults it off) because avoidance is the whole task."""

    terminate_on_collision = True
    """End the episode on contact. A collided episode teaches nothing further, and
    letting it run dilutes the collision signal across the remaining steps."""

    robot_id: str = "fr3_robotiq_gripper_sysid"
    control_mode: str = "arm_relative_joint_no_gripper"
