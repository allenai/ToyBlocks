from dataclasses import MISSING

from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.direct_rl_env_cfg import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.simulation_cfg import PhysxCfg
from isaaclab.utils import configclass


@configclass
class BaseEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 4
    gripper_close_decimation: int = 0
    """Extra physics substeps to run when the gripper transitions to close.
    When > 0, if the gripper action changes from open to close on a given step,
    the environment runs this many additional substeps (on top of the normal
    decimation) with the arm holding its current position target and only the
    gripper target active. 0 disables (default)."""
    gripper_settle_steps: int = 0
    """When > 0, on any gripper open/close transition the arm is frozen at its
    current joint positions and only the gripper command is applied for this many
    full env steps *before* the normal action (with arm motion) executes.
    This guarantees the gripper fully closes/opens before the arm moves."""
    episode_length_s = 1.0
    """Duration of each episode in seconds. Equals to (episode_length_s / sim.dt / decimation) in discrete steps."""
    auto_reset: bool = True
    """Whether to automatically reset the environment when the episode is done. If False, the environment will need to be reset manually."""
    ignore_terminations: bool = True
    """Whether termination condition (success/fail) should be ignored when determining if the episode is done."""
    num_renders_on_reset: int = 1
    """Extra ``sim.render()`` calls after every reset so RTX cameras reflect
    the post-reset state before ``_get_observations``. IsaacLab's upstream
    default is ``0`` (stale RGB until the next render-decimation step). Set to
    ``0`` for state-only pipelines (existing ``rotate``/``push`` configs do)."""
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100, render_interval=decimation, physx=PhysxCfg(gpu_max_rigid_patch_count=2**22)
    )

    # robot configuration
    robot_id: str | tuple[str] | None = "franka_robotiq_gripper"
    """The id(s) of the robot(s) to use. Must be one of the pre-defined robot configurations in `toyblocks.robots`. If none, `robot_cfg` should be provided instead.
    The robot_id provided may also add any number of mounted cameras to the environment (e.g. wrist cameras)."""
    control_mode: str | None = "arm_relative_joint_gripper_binary"
    """The control mode to use. Must be one of the pre-defined control modes for the given robot_id. If none, `control_cfgs` should be provided instead."""
    scene_preset_id: str | None = "droid_with_box_table"
    """The scene preset to use which pre-fills the environment with static assets and cameras (not mounted).
    These are based on typically used reproducible real-world robot workstations/setups.
    Must be one of the pre-defined scene presets in `toyblocks.scenes`.
    If none, user must add objects like the ground plane/table etc., and must add cameras to `camera_cfgs` themselves"""
    obs_mode: str = "state"
    """The observation mode to use. It can be one of "state" or "rgb" or "rgb+state". Observations are always a dictionary

    - "state": observations includes ground truth information from the environment like object poses (defined by `_get_obs_extra`)
    - "rgb": observations include all rgb images from all configured cameras.

    in all cases, agent information (defined by `_get_obs_agent`) is included in the observations.
    """
    always_spawn_cameras: bool = False
    """Cameras are spawned only when obs_mode contains any image based modality like "rgb". If this is set to True,
    cameras are spawned regardless of obs_mode."""

    spawn_only_cameras: tuple[str, ...] | None = None
    """If set, spawn only these keys of :attr:`camera_cfgs` and skip the rest.

    Rendering is usually the largest per-step cost, and it scales with the total
    pixels across every spawned camera and env. Runs that only need one view
    (e.g. a preview video) can drop the others instead of paying to render them.
    ``None`` spawns every configured camera."""

    actuator_profile: str | None = None
    """Optional actuator profile name (e.g. ``'tamp'``) that overrides the
    default robot actuator configs at env creation time.  Profiles are defined
    in :mod:`toyblocks.robots.actuator_profiles`.  CLI usage::

        --env-cfg.actuator-profile tamp
    """

    actuator_gains_ckpt_config: str | None = None
    """Optional path to an RL run's ``config.json`` (or its directory, or a ``.pt``
    checkpoint inside it). The PD gains recorded there under ``actuator_gains`` by
    ``scripts/ppo_fast.py`` override the robot config's own gains, so a checkpoint is
    rolled out under the physics it was trained with.  ``scripts/evaluate.py`` wires
    this automatically from ``--checkpoint``.  Applied after ``actuator_profile``.
    """

    actuator_gains_json: str | None = None
    """Optional JSON object mapping **actuator group name** (keys of
    ``ArticulationCfg.actuators``, e.g. ``panda_shoulder``) to partial gain updates.
    Same schema as the recorded ``actuator_gains``; fields may be a scalar or a
    ``{joint regex: value}`` object.  Applied after ``actuator_gains_ckpt_config``::

        --env-cfg.actuator-gains-json '{"panda_shoulder": {"stiffness": 60.0, "damping": 15.0}}'
    """

    actuator_stiffness: dict[str, float] = dict()
    """Per-actuator-group stiffness overrides, applied last (they win over
    ``actuator_profile`` / ``actuator_gains_json`` / recorded checkpoint gains).
    The key ``'*'`` targets every group::

        --env-cfg.actuator-stiffness panda_shoulder 60.0 panda_forearm 60.0
        --env-cfg.actuator-stiffness '*' 60.0
    """

    actuator_damping: dict[str, float] = dict()
    """Per-actuator-group damping overrides. Same semantics as
    :attr:`actuator_stiffness`."""

    # events
    # add isaac lab event configs to perform randomizations

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=5.0, replicate_physics=False
    )
    viewer: ViewerCfg = ViewerCfg(
        eye=(1.8, 1.2, 1.2),
        lookat=(0.5, -0.2, 0.0),
        origin_type="env",
        env_index=0,
        asset_name="robot",
    )

    # metrics
    record_metrics: bool = True
    """Whether to record standard metrics to infos object. Metrics include return, episode length, reward, success (at end and once), fail (at end and once)

    success/fail at end refer to success/fail at the end of an episode. once refers to success/fail occuring at least once during an episode.s
    """

    track_joint_metrics: bool = False
    """Track per-step joint position, velocity, and acceleration norms via :class:`~toyblocks.utils.metrics.MetricsTracker`."""

    track_ee_metrics: bool = False
    """Track per-step end-effector position, velocity, and acceleration via :class:`~toyblocks.utils.metrics.MetricsTracker`.
    Requires at least one ``FrameTransformer`` (typically the ``ee`` frame) to be configured."""

    track_collisions: bool = False
    """Track per-step collision forces and cumulative collision counts via an Isaac Lab ``ContactSensor``.
    When enabled, a contact sensor is automatically added to the robot during scene setup."""

    # table / friction randomization (used by :meth:`~toyblocks.tasks.base_env.BaseEnv._reset_idx`)
    table_height_range: float = 0.0
    """If > 0, randomize workspace table height by ±this many metres on each reset."""
    randomize_friction: bool = False
    table_static_friction_range: tuple[float, float] = (0.1, 0.4)
    table_dynamic_friction_range: tuple[float, float] = (0.1, 0.4)
    table_restitution_range: tuple[float, float] = (0.0, 0.0)
    block_static_friction_range: tuple[float, float] = (0.1, 0.4)
    block_dynamic_friction_range: tuple[float, float] = (0.1, 0.4)
    block_restitution_range: tuple[float, float] = (0.0, 0.0)
    make_consistent: bool = True

    rl_agent_obs_keys: list[str] = ["joint_pos", "joint_targets"]
    """Which of the ``agent`` observation keys the *flat* RL observation vector
    consumes.  ``_get_obs_robot`` always emits every key (see
    :attr:`~toyblocks.tasks.base_env.BaseEnv.AGENT_OBS_KEYS`) so dict-observation
    consumers -- the ManiFlow / DROID obs transforms -- can read any of them; only the
    flat path selects a subset, because its width is baked into RL checkpoints.

    Order is ignored: :meth:`~toyblocks.tasks.base_env.BaseEnv.rl_agent_obs` always
    concatenates in canonical emission order, so reordering this list cannot silently
    misalign an existing checkpoint.

    Must be a ``list``, not a ``tuple``: ``ppo_fast.py`` records
    ``dataclasses.asdict(env_cfg)`` through ``strip_invalid_json``, which recurses into
    dicts and lists only -- a tuple would be dropped from every checkpoint's
    ``config.json`` and the value could never be recovered at eval time."""

    legacy_ee_pos: bool = False
    """Compute the ``ee_pos`` observation with the pre-``ee_body_name`` convention:
    the *last* articulation body, expressed in the robot base frame.

    Temporary A/B switch. Eval resolves the EE link via the robot config's
    ``ee_body_name`` (``robotiq_arg2f_base_link``), while
    :mod:`toyblocks.vecgen.recording.lerobot_recorder` still records
    ``observation.state.ee_position`` -- the distribution the ManiFlow state
    normalizer is fit to -- from ``len(body_names) - 1``. The two links differ by
    roughly a gripper length, so checkpoints trained on that data see ``ee_pos``
    off-distribution. This flag exists to measure that gap; remove it once the
    recorder and the env agree on one link."""

    randomize_init_joint_vel: bool = False
    """If enabled, add uniform noise to the initial joint velocities (see _reset_idx in BaseEnv)"""

    asymmetric_critic: bool = False
    """If True (and state is included in observations), an additional ``privileged`` observation group
    (defined per-task by :meth:`~toyblocks.tasks.base_env.BaseEnv._get_obs_privileged`) is appended as the
    last obs group. The critic consumes the full observation while the actor slices off this privileged tail,
    enabling asymmetric actor-critic training."""

    collision_filter_prim_paths: list[str] = list()
    """Optional list of prim path expressions to filter collision reporting against (e.g. ``["{ENV_REGEX_NS}/Object"]``).
    Used to populate the ``ContactSensorCfg.filter_prim_paths_expr`` when ``track_collisions`` is True."""

    collision_filter_body_names: list[str] = list()
    """Robot body names to create per-body filter sensors for (e.g. ``["panda_link7", "panda_link8"]``).
    When non-empty and ``collision_filter_prim_paths`` is set, lightweight single-body contact
    sensors are created for each listed body so that per-body-per-filter pair forces
    (``contact_pair/`` metrics) are reported correctly.  If empty, only aggregate contact
    metrics are tracked (no ``contact_pair/`` breakdown)."""

    # robot_cfg: ArticulationCfg = MISSING
    # """The robot configuration to use. Filled depending on the robot_id property, but can be customized here as well to override robot_id"""
    # control_cfgs: dict[str, JointActionCfg] = MISSING
    # """The control configurations to use. Filled depending on the control_mode property, but can be customized here as well to override control_mode"""
    camera_cfgs: dict[str, SensorBaseCfg] = MISSING
    """The camera configurations to use. Filled depending on the scene_preset_id property, but can be customized here as well to override scene_preset_id"""

    env_slots: tuple[int, ...] = ()
    """Spawn slot (cube scale / randomization profile) of each env, one entry per
    env, for generated block tasks. Empty (default) means round-robin
    ``i % MAX_NON_SIM_STATE_RANDOMIZATIONS``. Queue-driven eval fills it so each
    slot gets envs in proportion to its queued workload."""

    drop_cameras: tuple[str, ...] = ()
    """Camera names removed from ``camera_cfgs`` once the scene preset and robot
    cameras are merged, so they are never spawned or rendered (e.g. an eval whose
    policy never reads ``left_camera``). Names the scene does not have are ignored."""

    instance_layout_json_path: str | None = None
    """Optional path to a per-instance JSON (e.g. ``apr15_pnp/envs/.../1000.json``) with
    ``block_positions``, ``block_quaternions``, and optional ``gripper_camera_*`` fields.
    Parsed in generated PDDL envs via :meth:`~toyblocks.tasks.base_env.BaseEnv._merge_instance_layout_json_into_cfg`
    before ``camera_overrides_json`` is applied."""

    camera_overrides_json: str | None = None
    """Optional JSON object mapping **camera name** (keys in ``camera_cfgs``) to partial
    updates. Applied after the scene preset and robot cameras are merged (before
    ``replace_env_regex_ns``). Examples::

        --env-cfg.camera-overrides-json '{"gripper_camera": {"pos": [0.04, 0, 0.08], "rot": [0.7, -0.09, -0.09, 0.7]}}'
        --env-cfg.camera-overrides-json '{"external_camera": {"pos": [1.3, 0.0, 0.6], "rot": [0.5, 0.5, 0.5, 0.5]}}'
        --env-cfg.camera-overrides-json '{"front_camera": {"spawn": {"focal_length": 30.0, "clipping_range": [0.5, 2.4]}}}'

    Each camera value is an object with any of:

    - ``pos`` / ``rot`` — same as :attr:`CameraCfg.offset` (3 and 4 floats; convention matches that camera).
    - ``spawn`` — partial :class:`~isaaclab.sim.spawners.sensors.sensors_cfg.PinholeCameraCfg` (or subclass)
      fields, e.g. ``focal_length``, ``focus_distance``, ``horizontal_aperture``, ``clipping_range`` (2 floats),
      ``vertical_aperture`` (number or JSON ``null``). Only for cameras whose ``spawn`` is not ``None``.

    Unknown camera names are skipped. """

    replay_block_side_length: float | None = None
    """TAMP replay only: when set with ``num_envs=1``, spawn every block at this
    isotropic side length (metres), matching ``observation.state.cube_scale`` from
    the dataset instead of randomization slot 0."""

    replay_gripper_close_target: float | None = None
    """TAMP replay only: override ``BinaryJointPositionAction`` close target for
    ``finger_joint`` (radians). TAMP datagen commands ~0.7; robot configs vary
    (``fr3_robotiq_gripper_sysid`` / ``fr3_robotiq_gripper_softer`` close to 0.72,
    ``droid_franka`` to pi/4), so set this when they differ from the dataset."""

    eval_table_z_offset: float = 0.0
    """Queue eval only: extra world-frame Z offset (metres) applied to
    ``workspace_table`` and rigid blocks after the dataset table height is
    reconstructed. Negative values lower the table (e.g. ``-0.02`` for 2 cm). Block
    local Z is shifted by the same amount so blocks stay on the table surface."""

    use_calibrated_cameras: bool = False
    """Queue eval only: apply fixed sim2real+robot preset camera K instead of
    per-episode recorded ``camera_intrinsics`` from the queue setup."""

    frame_transformers: dict[str, FrameTransformerCfg] = MISSING
    """The frame transformations to use. Filled depending on the robot_id property as each robot has pre-defined frame transforms
    (such as end-effector frames). Can otherwise be customized here to override the pre-defined frame transforms."""

    enable_visual_markers: bool = False
    """Whether to enable visual markers for the environment. Used primarily for debugging and visualization."""

    # placeholders, auto-filled upon environment creation
    observation_space = 1
    action_space = 1


def rl_agent_obs_keys_from_recorded_env_cfg(recorded_env_cfg: dict) -> list[str]:
    """Derive :attr:`BaseEnvCfg.rl_agent_obs_keys` from a checkpoint's recorded ``env_cfg``.

    Takes the ``env_cfg`` dict out of an RL run's ``config.json`` (not a live config
    object) so it can read fields that no longer exist on :class:`BaseEnvCfg`.
    Checkpoints trained before ``rl_agent_obs_keys`` recorded up to three booleans
    instead; ``joint_targets`` was unconditional before ``include_joint_targets_in_obs``
    existed, hence its ``True`` default here.
    """
    recorded = recorded_env_cfg.get("rl_agent_obs_keys")
    if recorded is not None:
        return list(recorded)
    extended = bool(recorded_env_cfg.get("include_extended_state_in_obs", False))
    keys = ["joint_pos"]
    if recorded_env_cfg.get("include_joint_targets_in_obs", True):
        keys.append("joint_targets")
    if extended or recorded_env_cfg.get("include_joint_vel_obs", False):
        keys.append("joint_vel")
    if extended:
        keys.append("ee_pos")
    return keys
