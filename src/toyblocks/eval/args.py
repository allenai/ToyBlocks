"""CLI arguments for the unified evaluation script."""

import dataclasses
from typing import Literal, Optional

from toyblocks.cli.base_args import BaseArgs, add_env_cfg_args


@add_env_cfg_args
@dataclasses.dataclass
class Args(BaseArgs):
    """Evaluation script arguments."""

    # Experiment
    exp_name: str | None = None
    """Name for this evaluation run. Auto-generated if not provided."""
    seed: int | None = None
    """Random seed. When ``None`` (default), no fixed seed is set."""
    deterministic_cuda: bool = False
    """Force deterministic CUDA operations (sets ``cudnn.deterministic=True``
    and ``cudnn.benchmark=False``). Slower but bitwise reproducible."""

    # Environment
    task: str = "ToyBlocks-Lift-v0"
    """Gymnasium task id."""
    num_envs: int = 4
    """Number of parallel environments."""
    env_spacing: float | None = None
    """Grid spacing (meters) between parallel envs. When ``None``, uses the env
    config default. Increase it to spread envs apart in the rendering video."""
    num_eval_steps: int = 200
    """Maximum number of environment steps to run. Ignored under
    ``--queue_url``: queue exhaustion (``queue_client.done``) terminates."""
    num_episodes: int | None = None
    """Stop after this many episodes complete (across all envs). If set
    together with --num_eval_steps, whichever limit is hit first wins."""
    obs_mode: str = "state"
    """'+'-joined observation components: 'state', 'rgb', 'rgb+state'. A 'depth'
    component (metric depth + native camera intrinsics per
    ``obs["images"][cam]``) is appended automatically when the served checkpoint
    declares ``*_depth`` inputs, so it is not set by hand."""
    control_mode: str | None = None
    """Control mode for the environment. If None, uses the env config default
    (e.g. 'arm_relative_joint_gripper_binary' for droid_franka)."""
    # Policy server
    policy_host: str | None = None
    """Host of the policy WebSocket server. When not set, no server connection
    is made (zero-action mode, unless --random_actions or --replay)."""
    policy_port: int = 8000
    """Port of the policy WebSocket server."""
    random_actions: bool = False
    """If True, sample actions uniformly in [-1, 1] each step. Mutually
    exclusive with ``--policy_host`` and ``--replay``."""
    action_key: str = "actions"
    """Key in the server response dict that holds the action array."""
    unbatched: bool = False
    """If True, call the policy server once per env (``PolicyClientAdapter``
    with ``batched=False``), for servers that add their own batch dimension and
    expect unbatched observation leaves. If False, one
    WebSocket ``infer`` per step with a leading ``num_envs`` axis on every array
    (requires a server and obs transform that support true batching)."""

    # Observation / action transforms (remote policy server only)
    center_crop_ratio: float | None = None
    """Maniflow obs transforms only: after resize to 224, centre-crop RGB,
    depth, and ``target_heatmap`` to the same ``int(224 * ratio)`` window and
    resize back (inference counterpart of ``random_crop_ratio``, e.g. 0.95).
    ``camera_intrinsics`` stay nominal (224 grid scale only). ``None`` feeds
    the full frame."""
    depth_aug: str | None = None
    """Maniflow obs transforms only: live depth augmentation before the policy
    sees depth (``ejbl`` matches HVLA ``datasets.vla_data.augment_depth``).
    ``None`` sends clean sim depth."""
    enable_crop: bool = False
    """Maniflow policy evals only: build ``target_front_crop_rgb`` on the client
    from native front camera + native target pixels (training ``_crop_centered``
    parity). Requires a checkpoint with ``use_target_front_crop >= 0``."""
    obs_transform: str = "none"
    """Observation transform to apply before sending to the policy server.
    Use 'none' for no transform, or a registered name (e.g. 'maniflow',
    'maniflow_blocks_table' for table + Franka wrist cam)."""
    action_horizon: int = 0
    """Action chunk horizon. When > 0, the policy is expected to return action
    chunks and only one action per step is dispensed. 0 disables chunking."""
    async_hl: bool = False
    """Molmo2 HL server (``--policy molmo2_maniflow``) with a play-mode LL:
    query the HL only at episode start and after the LL ends a skill (play mode
    -1), reusing each env's last HL output in between -- sim2real's
    ``async_molmo``. Off (default): the HL runs at every chunk boundary."""
    async_hl_call_scale: float = 0.0
    """``--async_hl``: an episode ends once Molmo2 has been called more than this many
    times its expert's HL calls (as ``--ll_oracle_hl_call_scale``). 0 (default): no cap."""
    go_home: bool = False
    """``--async_hl`` or ``--ll_oracle``: after the LL (or oracle) ends a place, flip or
    rotate, drive the arm home before that env's next Molmo2 query."""
    post_place_lift: float = 0.0
    """Play-mode LL: when it ends a skill with the gripper open (a place's
    release), raise the held end effector this many meters before that env's
    next LL query; 0 disables. ``pause_on_attempt`` training labels the
    demonstration's post-release lift (vecgen ``lift_traj``, 0.035 m) as -1, so
    without it every next pick starts from the release pose."""
    ll_oracle: bool = False
    """Molmo2 HL server (``--hl_from_model``) with the vecgen skills (cuRobo v2) as a low-level oracle
    in place of the served LL (``eval/ll_oracle.py``): each Molmo2 command runs on
    the block nearest its pixel, or on the table within 25 px of it, and Molmo2 is
    queried again only once that env's command ends (asynchronously per env). A
    command that cannot run or fails returns the arm home first. Needs
    ``--action_horizon 0`` and ``--control_mode arm_joint_gripper_binary``."""
    ll_oracle_hl_call_scale: float = 3.0
    """``--ll_oracle``: an episode ends once Molmo2 has been called more than this many
    times its expert's HL calls (2 per pnp, 1 per flip / rotate, 1 for ``done``)."""
    sync_action_chunks: bool = False
    """Batched ``--action_horizon`` evals: give every env the same chunk boundary,
    so the server is queried once per horizon instead of whenever any env's chunk
    runs out (which, once episodes end at different steps, is most steps). An env
    whose episode ends mid-chunk holds its reset pose until the next boundary,
    hidden from the episode clock and the recording like a PD-switch settle. Needs
    ``--control_mode arm_joint_gripper_binary`` (the hold is an absolute joint
    target); ignored with ``--unbatched``."""

    # Oracle high-level policy (client-side; replaces the learned HL model)
    use_oracle: bool = False
    """Drive ``task_name`` + ``target_heatmap`` from the symbolic
    ``OracleHighLevelPolicy`` (run client-side next to the live sim) instead of a
    learned HL model. Requires ``--policy_host`` + a maniflow ``--obs_transform``,
    and a per-env PDDL problem source: ``--pddl_problem_file``, or ``--queue_url``
    (each setup's ``pddl_problem_file`` field when present, else a minimal problem
    reconstructed from its ``pddl_goal`` and recorded ``high_level_state`` at reset)."""
    pddl_problem_file: Optional[str] = None
    """Single PDDL problem file used for all envs. A per-env queue setup's
    ``pddl_problem_file`` field overrides this when present."""

    # Output
    output_dir: str | None = None
    """Directory to write results. Defaults to runs/<run_name>."""
    save_sensor_obs_video: bool = False
    """Save videos constructed from RGB camera observations in
    ``obs["images"]`` for the env indices listed in ``video_env_ids`` (one
    video per finished episode per env). Requires ``--obs_mode`` to include
    ``rgb``. Videos are written to ``<output_dir>/sensor_obs_videos/``."""
    video_env_ids: tuple[int, ...] = (0,)
    """Which sub-environments to record RGB/camera video for. Ignored unless
    ``--save_sensor_obs_video`` is set. Each
    index must satisfy ``0 <= id < num_envs``. Overridden to the full
    ``range(num_envs)`` when ``--save_video_all_envs`` is set."""
    save_video_all_envs: bool = False
    """Override ``video_env_ids`` to record video for every sub-env. Convenient
    for eval yamls that don't know ``num_envs`` until launch."""
    save_hl_overlay_video: bool = False
    """With ``--save_lerobot`` and an HL (``--use_oracle`` / ``--hl_from_model``),
    record ``observation.images.<hl_overlay_camera>_hl_overlay`` in the LeRobot
    dataset: the frame with the HL heatmap and text (and Molmo2 prompt) drawn on it."""
    hl_overlay_camera: str = "front_camera"
    """Camera the HL overlay video draws on — the view the ``target_heatmap`` is
    aligned to (see ``OracleHLController._resolve_cam_resolution``)."""

    # High-level latency (update interval K + estimated HL inference time M),
    # both in milliseconds (``sim_time_ms = step * base_env.step_dt * 1000``).
    # See ``toyblocks.serving.hl_latency.HLLatencyBuffer``. All default to
    # 0/False so existing runs are unchanged (fresh HL every step, no delay, no
    # extra video).
    hl_update_interval_ms: float = 0.0
    """K (ms): HL invocation interval; see ``toyblocks.serving.hl_latency``.
    ``0`` = every step."""
    hl_update_interval_jitter_ms: float = 0.0
    """Resampled ``U(-jitter, +jitter)`` ms added to each interval; see
    ``toyblocks.serving.hl_latency``. ``0`` = fixed K."""
    estimated_hl_inference_time_ms: float = 0.0
    """M (ms): assumed HL inference time; see ``toyblocks.serving.hl_latency``.
    ``0`` = instantaneous.

    For a checkpoint trained with ``use_previous_front_frame`` this also sets how
    stale the LL's ``prev_front_rgb`` is, so it has to match training. The
    dataloader samples the previous frame ``n_action_steps * U{1,2,3,4}`` steps
    back (``n_action_steps: 16`` in the Franka YAML), while the buffer's selected
    entry has age in ``[M, M+K]``. So set ``M ≈ n_action_steps * step_dt * 1000``
    and size ``K`` so the age reaches ~4x that. A mismatched M/K puts
    ``prev_front_rgb`` out of distribution even though every shape still
    validates, so ``M = 0`` on such a checkpoint prints a loud startup
    ``[WARNING]`` naming the resulting age (it is not fatal: ``M = 0`` is also
    how a deliberately synchronous HL is configured). At episode start the buffer
    holds only its first entry, so ``prev_front_rgb == front_rgb`` — matching
    training's ``max(0, ...)`` clamp."""
    save_hl_latency_video: bool = False
    """Alongside ``--save_sensor_obs_video``, write a per-episode side-by-side
    ``hl_latency_videos/`` MP4: left = real-time (fresh HL) panel, right = the
    (aged) HL view the LL actually acted on. No-op without HL conditioning."""
    hl_from_model: bool = False
    """Internal, launcher-derived (NOT a user-facing launcher flag): signals
    that the policy server produces the HL conditioning (the learned Molmo2 HL
    model, i.e. ``--policy molmo2_maniflow``). The eval client cannot infer this
    -- ``pi``, ``maniflow``, and ``molmo2_maniflow`` all just connect to
    ``--policy_host``, and ``pi`` also runs without ``--use_oracle`` -- so the
    launcher derives it, mirroring ``unbatched``. Enables the model HL overlay
    video (``--save_hl_overlay_video``) by pulling the server's ``hl_model_*`` aux
    into the obs each step."""
    oracle_comparison: bool = False
    """Also run a client-side TAMP oracle HL in parallel (comparison-only; NOT
    fed to the LL) and render a side-by-side ``model | oracle`` panel in the HL
    overlay video. Valid only with ``--hl_from_model`` and ``--save_lerobot``
    (raises otherwise)."""
    save_lerobot: bool = False
    """Save rollout trajectories as a LeRobot-format dataset (includes rich
    scene data and raw policy actions)."""
    lerobot_dataset_name: str | None = None
    """Name for the LeRobot dataset.  Defaults to the run name."""
    num_shards: int = 8
    """Number of parallel LeRobot writer (encoder) subprocesses. Env ``i`` is
    routed to writer ``i // ceil(num_envs / num_shards)``; their datasets are
    stitched into one ``<name>_success`` / ``<name>_failure`` pair at the end.
    Clamped to ``num_envs``."""
    video_codec: str = "libx264"
    """RGB codec of the LeRobot videos: ``'libx264'`` or ``'libsvtav1'`` (datagen's;
    about 4x the native memory per open stream, which adds up at one stream per
    env and camera)."""
    video_camera: str | None = None
    """Camera name to use for video. If None, tiles all available cameras."""
    drop_cameras: tuple[str, ...] = ("left_camera",)
    """Scene cameras not spawned for a policy eval, so they are never rendered.
    The ManiFlow and DROID obs transforms read only their ``external_cam`` /
    ``wrist_cam`` (``front_camera`` / ``gripper_camera`` on the blocks-table
    scenes), so ``left_camera`` is dead weight there (about 40% of the rendered
    pixels on sim2real). Ignored under ``--replay``. Pass ``--drop_cameras``
    with no names to keep every camera."""
    video_fps: int | None = None
    """Frames per second for saved videos and LeRobot datasets. If omitted,
    uses the task control rate ``1 / (decimation × sim.dt)`` (e.g. 25 for
    default lift: dt=0.01, decimation=4)."""
    max_steps_per_video: int | None = None
    """Flush video every N steps. Auto-set to episode length for multi-env."""

    # Metrics tracking
    track_metrics: bool = True
    """Enable joint and EE metric tracking (no extra GPU cost). Equivalent to
    --track_joint_metrics --track_ee_metrics."""
    track_joint_metrics: bool = True
    """Track per-step joint position, velocity, and acceleration norms."""
    track_ee_metrics: bool = True
    """Track per-step end-effector position, velocity, and acceleration."""
    save_step_metrics: bool = False
    """Also write the tracked stats of every control step to
    ``<output_dir>/step_metrics/env{idx}_episode_NNNN.csv`` (one file per
    finished episode). Off by default: they cost a host copy of every stat on
    every step, and the episode summaries in ``eval_results.json`` do not
    depend on them."""

    dataset_path: Optional[str] = None
    """Path to a LeRobot dataset root. When provided: (1) ground-truth
    ``action.joint_target`` / EE poses are overlaid on eval plots, and (2) the
    eval ``robot_id`` / ``scene_preset_id`` default to the dataset's recorded
    provenance (``meta/info.json``), warning loudly on override mismatch or
    when provenance is missing. For queue evals, pass the dataset the queue
    serves so this reconciliation runs."""

    # Dataset replay (queue-driven)
    replay: bool = False
    """Replay each queue episode's recorded actions instead of querying a
    policy. The recorded 7 joint targets + gripper command + ``action.pd_mode``
    are fed through the same eval loop a policy uses (gain switches go through
    the PD controller with the TAMP-style settle), the recorded frame-0 joint
    velocity is restored at reset, and per-episode divergence from the
    recording is reported as ``replay_*`` metrics. Requires ``--queue_url`` +
    ``--dataset_path`` and ``--num_envs 1``; mutually exclusive with
    ``--policy_host`` / ``--random_actions``. Forces
    ``--episode_end horizon`` so every recording plays to its last frame."""
    replay_clip_stats: str | None = None
    """When replaying, a ``stats.json`` whose ``action.joint_target`` q01/q99
    bands every replayed arm joint target is clamped to (e.g. a maniflow run's
    ``stats.json``). Tests whether a policy whose labels were clipped to those
    bands could still succeed. Adds ``replay_clipped_frames`` /
    ``replay_clip_max_rad`` per-episode metrics. Requires ``--replay``."""
    # Viewer
    enable_viewer: bool = False
    """Launch the viser-based 3D viewer."""

    # Debug
    debug: bool = False
    """Enable debug diagnostics: save t=0 visual comparison (sim vs training
    data), per-episode seed logs, episode-start robot/action dumps, and
    per-step action prints (``--debug`` in the main loop)."""

    # Wandb logging
    wandb: bool = False
    """If True, mirror per-step throughput, per-episode metrics, and any
    sensor-obs videos to a wandb run. Requires ``WANDB_API_KEY`` to be set."""
    wandb_project: str = "toyblocks-eval"
    """wandb project name."""
    wandb_entity: str | None = None
    """wandb entity (team or user). Falls back to ``WANDB_ENTITY`` env var."""
    wandb_group: str | None = None
    """wandb group (none by default)."""
    wandb_name: str | None = None
    """wandb run name. Defaults to ``{task}__seed{seed}``."""
    wandb_job_type: str = "eval"
    """wandb job_type label (used to distinguish eval vs serve runs)."""
    wandb_tags: tuple[str, ...] = ()
    """Extra wandb tags."""
    wandb_log_interval_seconds: float = 5.0
    """Min seconds between throughput log flushes."""
    save_expert_video: bool = False
    """Extract the source dataset's expert demonstration (primary front camera)
    into ``<output_dir>/source_episode_videos/`` and surface it in the wandb
    episodes table. Enabled by ``launch_eval.py`` when ``--wandb`` is set."""

    queue_url: str | None = None
    """URL to fetch episode setups from (e.g., http://localhost:8001/next)."""

    expert_timeout_scale: float = 0.0
    """End each episode once it has run this multiple of its own expert
    demonstration's duration (``1.5`` = 150%), instead of letting every episode
    share the single ``episode_length_s`` horizon. The per-episode budget comes
    from the queue setup's recorded ``expert_episode_length`` (frames) over the
    dataset's fps, so this requires both ``--queue_url`` and ``--dataset_path``.

    ``episode_length_s`` still applies and is the *shorter* of the two horizons
    whenever it fires first, so it has to be raised to cover the longest expert
    in the queue for this to bind (``run_eval_pipeline.sh`` does that
    automatically from the queue's ``/size`` ``expert_seconds_max``). ``0``
    disables, leaving today's single global timeout."""

    ingest_camera_intrinsics: bool = True
    """After the initial reset, override each live camera's intrinsics with the
    recorded ``observation.camera_intrinsics`` (via ``set_intrinsic_matrices``)
    so eval renders with the recorded calibration rather than the preset
    ``camera_cfgs``. Each env gets that episode's first-frame K; later resets
    re-apply via ``apply_setup`` (randomized-camera datasets vary K per
    episode). Requires ``--dataset_path``. Camera extrinsics are replayed by
    ``--use_render_params`` when the dataset is a rerender output, and are
    otherwise verify-only. Mutually exclusive with ``--use_calibrated_cameras``."""

    use_calibrated_cameras: bool = True
    """Apply fixed sim2real + fr3 gripper preset intrinsics at native render
    resolution (aug26-style) instead of dataset/queue ``camera_intrinsics``.
    Disables dataset intrinsics ingestion and ``--use_render_params``
    automatically. On by default so eval runs the one sim2real robot scene;
    ``--no-use-calibrated-cameras`` uses the dataset's K instead."""

    use_render_params: bool = False
    """Opt-in; needs ``--no-use-calibrated-cameras``. When ``--dataset_path`` is a rerender output (its ``meta/render_params.json``,
    or for a merged dataset each source's, reached through ``meta/provenance.json``),
    reproduce each queue episode's recorded camera pose and appearance: the env-mounted
    cameras sit on the recorded ``observation.camera_pose``, the arm camera gets its
    recorded mount offset, and the HDRI, table material, lights, exposure and backdrop
    come from the episode's record (see ``toyblocks.eval.render_replay``). Episodes whose
    record fails the lookup checks get nothing applied and are flagged in
    ``render_replay_status``. A no-op for datasets without ``render_params.json``.
    With ``--replay`` it needs ``--num_envs 1``. With ``--num_envs > 1`` the sky, dome,
    sun and exposure are one value for the whole scene: they are set once at the
    first reset, or per wave with ``--wave_resets``."""

    replay_block_identity: bool = True
    """On glyph/colour-rerendered datasets (rows carry ``block_variable_map`` /
    ``glyph_map``), paint each episode's recorded block letters and colours at reset
    and prompt the policies with the recorded (rendered) goal. Off
    (``--no-replay_block_identity``): blocks keep their authored look and the prompts
    use the source block names. Success is graded on the source block names either
    way. A no-op for datasets without the maps."""

    render_replay_assets_paths: tuple[str, ...] = ()
    """Extra HDRI roots searched (in order, before the repo's ``assets/environments``) for the
    skybox / backdrop textures a ``--use_render_params`` record names.
    ``asset_manifest.json`` does not install the HDRI pool (see docs/rerender/README_local.md)."""

    render_replay_warmup_frames: int = 16
    """``sim.render()`` calls after a reset that applies a recorded render, so the
    path tracer converges on the new look before the first observation (the
    rerender's ``rtx_warmup_frames``). Raises ``env_cfg.num_rerenders_on_reset`` to
    this value when ``--use_render_params`` is active."""

    wave_resets: bool = False
    """With ``--use_render_params`` and ``--num_envs > 1``: start episodes in waves.
    An env whose episode ends is reset as usual, then holds its reset pose, hidden from
    the episode clock and the recording, until every env with a live setup has finished;
    the whole wave then gets its recorded renders (scene-global lighting from the lowest
    env index) and starts together. Needs ``--control_mode arm_joint_gripper_binary``
    (the hold is an absolute joint target), like ``--sync_action_chunks``."""

    verify_scene: bool = False
    """After the initial queue reset, run scene verification (table height,
    blocks present, block/table poses, first-frame image MAE/MSE/SSIM) on the
    first batch of setups and dump ``scene_verification.json`` + comparison PNGs
    to ``output_dir``. Advisory by default: mismatches are logged at ERROR and the
    eval continues (use ``--verify_scene_strict`` to make them fatal).
    Requires ``--queue_url``; first-frame images require an rgb ``--obs_mode``."""

    verify_scene_strict: bool = False
    """Make ``--verify-scene`` hard failures abort the run instead of only being
    logged. The report is still written first. Use this when the numbers matter:
    a hard mismatch means the measured scene is not the recorded one, so the
    metrics describe something other than the dataset being replayed. Implies
    ``--verify-scene``."""

    save_diagnostic_plots: bool = False
    """Emit the diagnostic artifacts that need the live rollout (things a policy
    server or the client obs transform sees mid-episode). Off by default to keep
    ``_run_eval`` cheap; enable for closed-loop debugging of a single short
    rollout (the one env ``video_env_ids`` names is what every artifact below
    traces). Four artifacts:

    * per-episode policy-input verification panels in
      ``output_dir/policy_input_debug/``, surfaced as the wandb
      ``debug/policy_inputs_table``: raw sim depth -> the depth that goes on the
      wire -> the dataset's own depth and their difference, plus an
      ``agent_pos`` panel showing each dim against the ``[q01, q99]`` band the
      server normalizes it into;
    * ``hl_conditioning_staleness.png``, when HL conditioning frames are being
      injected by the *client* oracle. Under ``--hl_from_model`` the K/M gating
      lives in the policy server, so this plot is skipped and the HL
      policy-input trace below is the equivalent record;
    * the server-side LL policy-input trace, rendered inside the policy server
      right before ``predict_action`` (the same question as the client panels,
      answered from the other end of the wire);
    * the server-side HL policy-input trace (``--hl_from_model`` only), one PNG
      per Molmo2 forward showing the front frame, the previous-action string
      that was prepended, the exact question, and the HL output.

    All four are capability-gated, not required: a run without a policy
    server, without a maniflow obs transform, or against a server that
    advertises no state normalization bands prints why it skipped them and
    keeps going. What the encoder does with the depth is logged by the server
    itself; see ``instrument_ptmap_saturation``.

    The per-step joint-position / velocity / EE / fingertip / camera comparison
    plots this flag used to render online do not exist anymore (moved offline
    in commit 96ddd7641) and do not need this flag at all — only
    ``--save_lerobot``. Run ``scripts/plot_diagnostics_from_eval_run.py
    --eval-dir <output_dir>`` by hand for every recorded episode, or rely on
    ``wandb_consolidate.py``, which runs it automatically for one
    representative episode per worker (see its ``_render_offline_diagnostics``)
    before uploading ``diagnostic_plots/*``."""
    
    save_first_batch_data: bool = False
    """On step 0, ask the policy server to dump the model-ready batch (post
    normalization / validation) to ``<output_dir>/first_batch_data`` in the same
    format training writes, for comparing eval inputs against training inputs.
    Requires a remote policy server; only the ManiFlow server implements it.
    Depth consumption stats on step 0 are always logged to stdout for ManiFlow
    depth evals (independent of this flag)."""


    goal_ori_yaw_tol_deg: float = 40.0
    """Yaw tolerance (deg) of the PDDL goal's ``ori_of`` atoms when judging goal
    success (episode end); raise it to relax orientation. The 5 deg tilt
    tolerance is unchanged. Flip / rotate sub-goal scoring keeps the default."""

    episode_end: Literal["goal", "horizon", "ll_done"] = "goal"
    """What ends an episode before the ``episode_length_s`` timeout:
    ``goal`` = the goal predicate (oracle runs: plan completion);
    ``horizon`` = nothing; ``ll_done`` = the LL's play mode -1 (play-mode
    checkpoints, not ``--hl_from_model``). With ``--use_oracle``, -1 on the
    final command holds the robot until the goal is met or 25 steps pass.
    A stuck oracle plan and a learned HL's "done" end it in every mode."""

    collision: Literal["end", "penalize", "off"] = "end"
    """A non-target block moving more than 2 cm or 15 deg is a collision and
    fails the episode (``collision``, ``collision_count``). ``end`` also ends
    the episode (never under ``--replay``); ``penalize`` lets it run on and also
    records the ``*_ignore_collision`` successes; ``off`` does not score them.
    Oracle and Molmo runs also record per sub-goal ``sub_goals`` /
    ``sub_goal_success`` and per-skill rates (``sub_goal_success_pick``, ...)."""

    tamper_output_dir: str | None = None
    """Capture the terminal sim state of every unsuccessful episode and write it
    under this directory, for TAMPER (TAMP Expert Relabeling): vecgen replans
    expert demonstrations starting from the states the policy actually got wrong.

    One subdirectory per source PDDL problem, each an env package vecgen loads
    directly (``reset_states.pkl``, ``problems.json``) plus a ``tamper_failures.json``
    mapping each instance id back to its episode, outcome and goal; see
    ``scripts/datagen/run_tamper_pipeline.py`` and ``docs/tamper.md``.

    Off unless set. Episodes are tagged ``failure`` (terminated) or ``timeout``."""

    tamper_capture_successes: bool = False
    """Also snapshot successful episodes into the TAMPER output, tagged
    ``success``, so the failure filter can be applied when selecting
    ``--instance-ids`` for planning rather than during eval. No-op without
    ``--tamper_output_dir``."""

    tamper_max_states: int | None = None
    """Stop capturing after this many TAMPER snapshots. Bounds the output of a
    long eval whose policy fails often. No-op without ``--tamper_output_dir``."""

    tamper_source_env_path: str | None = None
    """Env package the eval task was generated from. Its
    ``non_sim_state_randomizations.json`` / ``block_config.json`` are copied into
    each TAMPER directory so vecgen can run against it directly.
    No-op without ``--tamper_output_dir``."""

    # Override defaults from BaseArgs for headless evaluation
    headless: bool = True
