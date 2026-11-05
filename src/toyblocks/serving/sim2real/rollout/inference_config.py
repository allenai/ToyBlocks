"""Runtime JSON configuration for vlagent_rcs_client."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path

from toyblocks.serving.sim2real.rollout.hardware_config import (
    INCLUDE_RIGHT,
    MAX_REL_MOV_CART,
    MAX_REL_MOV_JOINTS,
)

logger = logging.getLogger(__name__)

DEFAULT_VLAGENTS_HOST = "localhost"
DEFAULT_VLAGENTS_PORT = 8080
CONFIG_PATH = Path(__file__).with_name("vlagent_rcs_client.json")
DEFAULT_SAVE_FIRST_FRAME_DIR = "first_frames"

POINT_GOAL_FORMATS = ("action_goal", "action_only")

# Checkpoint training configs record the prompt format the policy was trained
# with in ``framework.language_mode``, using these same two names (see
# hvla's franka_dataset: "action_goal" appends the target pixel to the action
# string, "action_only" does not). That field is the ground truth for
# ``InferenceConfig.point_goal_format`` -- send the other format and the policy
# sees a prompt distribution it never saw in training.
CHECKPOINT_CONFIG_NAMES = ("config.yaml", "config.yml", "config.json")


def parse_target_front_crop(value: object) -> int:
    """Validate ``use_target_front_crop``: ``-1`` (off) or ``0-99``.

    Same contract as hvla's ``franka_dataset.parse_target_front_crop``: the
    value is the percent of the crop that sits *below* the target (``25`` =
    75% top / 25% bottom, ``50`` = zero-centered).
    """
    if value is None or value == "":
        return -1
    crop = int(float(value))
    if crop != -1 and not 0 <= crop < 100:
        raise ValueError(
            f"use_target_front_crop must be -1 (off) or 0-99 (bottom-share percent), got {crop}"
        )
    return crop


@dataclass
class InferenceConfig:
    vlagents_host: str = DEFAULT_VLAGENTS_HOST
    vlagents_port: int = DEFAULT_VLAGENTS_PORT
    vlagents_model: str = "maniflow"
    instruction: str = "pick up the green cube"
    robot_keys: list[str] = field(default_factory=lambda: ["right"])
    jpeg_encoding: bool = True
    on_same_machine: bool = False
    fps: int = 30
    # Recordings and the eval log land in ``<checkpoint dir>/<record_path>``
    # once the vlagents server names its checkpoint; until then (or when that
    # folder is not writable here) this stays a plain relative directory.
    record_path: str = "inference_recordings"
    # Record every episode by default: the GUI's "Record episode" box starts
    # checked and the sim client auto-starts a recorded episode. The terminal
    # 'e' command still starts an unrecorded one.
    record: bool = True
    show_camera_windows: bool = False
    save_first_frame: bool = False
    save_first_frame_dir: str = DEFAULT_SAVE_FIRST_FRAME_DIR
    interactive_gui: bool = False
    dry_run: bool = False
    point_goal_format: str = "action_only"
    # Checkpoint whose training config decides ``point_goal_format``: the
    # training ``config.yaml``, the checkpoint directory, or the
    # ``*.safetensors`` file sitting next to it. ``null`` falls back to the
    # path the vlagents server reports for the checkpoint it loaded, which
    # resolves whenever the server runs on this machine.
    model_config_path: str | None = None
    # Checkpoint the vlagents server was launched with. Superseded for display
    # purposes by ``InferenceController.policy_checkpoint``, which asks the
    # server directly; kept because ``vlagent_rcs_client.json`` still carries
    # the key and the config is loaded with ``InferenceConfig(**json)``, which
    # rejects unknown keys.
    policy_checkpoint: str = ""
    include_heatmap: bool = True
    target_sigma: float = 0.8
    # Send the front frame frozen at the goal click as an extra
    # ``prev_front_rgb`` camera (``--prev-frame``). Only checkpoints trained
    # with ``framework.use_previous_front_frame`` consume it; off by default so
    # every other policy is spared the extra frame on the wire.
    include_prev_frame: bool = False
    # Square ``target_front_crop_rgb`` camera cropped from the front frame
    # around the goal point (``--use-target-front-crop``); the value is the
    # percent of the crop below the target, ``-1`` is off. Set from the
    # checkpoint's framework.use_target_front_crop by sync_frame_conditioning.
    use_target_front_crop: int = -1
    include_depth: bool = False
    # Fast-FoundationStereo depth from the left/right pair (takes precedence over include_depth,
    # which means ZED depth; needs INCLUDE_RIGHT=True and INCLUDE_DEPTH=False in hardware_config.py)
    use_ffstereo: bool = False
    ffstereo_clip_m: float = 1.0  # FFS depth beyond this becomes invalid (0); 0 disables the clip
    ffstereo_clip_min_m: float = 0.0  # FFS depth below this becomes invalid (0); 0 disables the clip
    ffstereo_host: str = "localhost"
    ffstereo_port: int = 8081  # 8080 is the ManiFlow vlagents server
    # Server-side "max trusted depth" (metres). Forwarded to the ffstereo depth
    # server as ``--z-far-m`` by ``launch_ffstereo_server`` so tiny disparities
    # -> very-far / noisy depth get marked invalid at the source. Distinct
    # from ``ffstereo_clip_m``: this bounds the raw geometry (fxB / z_far_m ->
    # d_min disparity threshold); the clip is a per-request cosmetic cap.
    max_depth_m: float = 30.0
    # Molmo2 HL (vision-language) server: turns a natural-language high-level
    # instruction + front_rgb frame into a low-level action phrase and a target
    # pixel coordinate that this client turns into ManiFlow's ``target_heatmap``
    # conditioning + LL ``instruction``. When ``use_molmo=true`` the GUI
    # exposes a text box for the HL instruction and disables point-clicking
    # (Molmo2 picks the point).
    use_molmo: bool = False
    molmo_host: str = "localhost"
    molmo_port: int = 8082  # 8080=ManiFlow, 8081=ffstereo, 8082=molmo2
    molmo_checkpoint: str = ""  # required when use_molmo=true; path to Molmo2 HL checkpoint dir
    # Natural-language HL instruction fed to Molmo2 (mirrors --hl_prompt on the
    # origin/main HLLLInferAdapter). The interactive GUI overrides this from a
    # text box; a run without the GUI uses this value directly.
    molmo_prompt: str = "put A on red"
    # When True (default), the Molmo2 server puts the previous HL action in
    # front of the goal, phrased as the checkpoint was trained
    # (``Previous action: <last>. `` or html-v2's ``previous action: <last>. ``),
    # so the HL sees its own rolling history. Set to False to drop that
    # prefix entirely for every inference call; the server still tracks
    # the previous action internally so toggling this at runtime (or re-
    # enabling it later) stays coherent, but it is never used in the prompt.
    molmo_use_prev_action: bool = True
    # The next three describe what the Molmo2 HL checkpoint was trained to see.
    # The checkpoint's own training config is the source of truth: the server
    # reads crop / front-only / frame size from it and raises if these
    # settings disagree, so a run cannot silently feed the wrong view. Set
    # them to match the checkpoint (JSON or the ``--molmo-*`` flags).
    #
    # ``front_crop`` of the training dataloader, as ``(x0, y0, x1, y1)``
    # pixels of the ``molmo_input_size`` frame, or None for no crop. The
    # server takes the crop; the emitted ``<u,v>`` comes back in full-frame
    # coords either way, so the ManiFlow heatmap / GUI marker / action_goal
    # language need no (x0, y0) shift.
    molmo_front_crop: tuple[int, int, int, int] | None = None
    # ``front_view_only`` of the training dataloader: when True,
    # ``gripper_rgb`` is not sent, matching a front-only checkpoint's
    # single-image prompt. Independent of ``molmo_front_crop``.
    molmo_front_view_only: bool = False
    # Frame size (width, height) the Molmo2 HL checkpoint was trained on.
    # ``front_rgb`` is always resized to it before being sent, and the
    # emitted ``<u,v>`` is lifted out of it back to live-frame pixels.
    # Independent of the 640x360 ManiFlow policy space (``POLICY_GOAL_*``):
    # the ManiFlow heatmap / ``action_goal`` language are always rebuilt from
    # live-frame coords.
    molmo_input_size: tuple[int, int] = (640, 360)
    # Query the Molmo2 HL only at low-level skill boundaries instead of on
    # every observation build. With ``async_molmo=false`` (the default) every
    # policy query first blocks on a Molmo2 call; with it true the cached
    # ``instruction`` + ``target_heatmap`` from the last HL call are reused
    # until the LL skill ends (execution status -1), the episode resets, or
    # the HL prompt changes. Only meaningful with ``use_molmo=true``.
    async_molmo: bool = False
    n_action_steps: int | None = None
    stop_after_action_chunk: bool = False
    # Temporal action ensembling (ACT-style)
    temporal_agg: bool = False
    # Exponential decay constant for temporal aggregation. Larger -> older
    # chunks dominate more sharply; smaller -> flatter weighting.
    temporal_agg_k: float = 0.01
    # Query the policy every N executed steps when temporal_agg is on.
    n_steps_per_prediction: int | None = None
    max_rel_mov_joints: float = MAX_REL_MOV_JOINTS
    max_rel_mov_cart: tuple[float, float] = MAX_REL_MOV_CART
    # TAM (Torque Adaptation Module, see tam_utils.py): master switch. Hardware
    # only. When enabled, everything else resolves automatically: the default
    # checkpoint is downloaded once into ~/.cache/simadaptor and the ideal-model
    # MJCF is installed with the torque-adaptation-module package. PD gains are
    # NOT changed: TAM adds its residual on top of whatever gains env_setup.py
    # selects. All tam_* settings are baked in at env creation -> restart.
    tam: bool = False
    # Before the robot moves, drive synthetic history through the encoder until
    # push latency is consistently fast, so JAX has finished compiling every
    # shape the live loop will hit (the runtime only warms the decode step at
    # init; the full push_window path compiles lazily on first real calls).
    tam_warmup: bool = True
    tam_warmup_target_ms: float = 60.0  # a push is "fast" below this
    tam_warmup_streak: int = 8  # consecutive fast pushes required
    tam_warmup_timeout_s: float = 90.0  # give up (and warn) after this
    # Apply franka::limitRate to the final torque. Keep True: FCI trips its own
    # torque-rate reflex otherwise, because PD+residual can sum past ~1 Nm/ms
    # even when neither term alone does. Only applied when tam=true.
    rate_limit: bool = True
    # Record the pre-rate-limit intended torque (tau_pd + residual) into TAM's
    # history instead of the post-limiter command, so the rate limiter (kept on
    # for FCI) does not distort the applied-torque history TAM conditions on.
    tam_history_pre_ratelimit: bool = True
    # Per-joint |clip| [Nm] on the TAM residual. The wrist (J7, and to a lesser
    # extent J5/J6) is low-inertia/soft and goes bang-bang at the clip, driving
    # a limit cycle -> choke the wrist authority. None keeps the C++ default
    # [10,10,10,10,2,2,2]. (0.5 on J5-J7 was stable; 2.0 on J7 diverged to a
    # velocity fault.)
    tam_residual_clip: list[float] | None = field(
        default_factory=lambda: [10.0, 10.0, 10.0, 10.0, 0.5, 0.5, 0.5]
    )


def load_inference_config() -> InferenceConfig:
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(json.dumps(asdict(InferenceConfig()), indent=2) + "\n")
        return InferenceConfig()
    cfg = InferenceConfig(**json.loads(CONFIG_PATH.read_text()))
    cfg.use_target_front_crop = parse_target_front_crop(cfg.use_target_front_crop)
    # JSON round-trips tuples as lists; keep the declared (w, h) tuple type.
    cfg.molmo_input_size = tuple(int(v) for v in cfg.molmo_input_size)  # type: ignore[assignment]
    return cfg


def warn_remote_transport_config(cfg: InferenceConfig) -> None:
    if not cfg.on_same_machine:
        return
    logger.info(
        "on_same_machine=true: client and vlagents server must run as the same UNIX user. "
        "If you see PermissionError on /psm_*, set on_same_machine=false and "
        "jpeg_encoding=true in %s.",
        CONFIG_PATH,
    )


def find_checkpoint_config(path: str | Path | None) -> Path | None:
    """Locate a checkpoint's training config from whatever the user points at.

    Accepts the config file itself, the checkpoint directory, or a file inside
    it (``steps_70000_model.safetensors``), which is what the vlagents server
    reports as its checkpoint path. Returns ``None`` when nothing readable is
    there -- a remote server's path simply does not exist on this machine.
    """
    if not path:
        return None
    candidate = Path(path).expanduser()
    if candidate.is_file() and candidate.suffix in (".yaml", ".yml", ".json"):
        return candidate
    directory = candidate if candidate.is_dir() else candidate.parent
    if not directory.is_dir():
        return None
    for name in CHECKPOINT_CONFIG_NAMES:
        if (directory / name).is_file():
            return directory / name
    # The serving loader also accepts a separate training YAML next to the
    # checkpoint (see vlagents_server --train-config); mirror that fallback.
    for train_yaml in sorted(directory.glob("train_*.yaml")):
        return train_yaml
    return None


def checkpoint_config_source(
    cfg: InferenceConfig, checkpoint_hint: str | None = None
) -> str | None:
    """Where the training config is looked for: the explicit override, then the
    path the server reports, then the checkpoint named in the JSON."""
    return cfg.model_config_path or checkpoint_hint or cfg.policy_checkpoint or None


def checkpoint_framework(config_path: Path) -> dict | None:
    """The ``framework`` block of a checkpoint config, or ``None`` if unreadable.

    Every training switch this client has to mirror (``language_mode``,
    ``use_previous_front_frame``, ``use_target_front_crop``) lives in there.
    """
    try:
        text = config_path.read_text()
    except OSError:
        logger.warning("cannot read checkpoint config %s", config_path, exc_info=True)
        return None
    try:
        if config_path.suffix == ".json":
            data = json.loads(text)
        else:
            import yaml  # lazy: only the checkpoint-config path needs PyYAML

            data = yaml.safe_load(text)
    except Exception:
        logger.warning("cannot parse checkpoint config %s", config_path, exc_info=True)
        return None
    return (data or {}).get("framework") or {}


def checkpoint_language_mode(config_path: Path) -> str | None:
    """``framework.language_mode`` from a checkpoint config, or ``None``.

    ``None`` means "cannot tell" (unreadable file, pre-``language_mode``
    checkpoint, unknown value) and leaves ``point_goal_format`` alone.
    """
    framework = checkpoint_framework(config_path)
    if framework is None:
        return None
    mode = framework.get("language_mode")
    if mode is None:
        logger.info(
            "%s has no framework.language_mode (checkpoint predates it); "
            "keeping the configured point_goal_format",
            config_path,
        )
        return None
    mode = str(mode)
    if mode not in POINT_GOAL_FORMATS:
        logger.warning(
            "%s has framework.language_mode=%r, which is not one of %s; "
            "keeping the configured point_goal_format",
            config_path,
            mode,
            POINT_GOAL_FORMATS,
        )
        return None
    if framework.get("language_conditioned") is False:
        # The server drops task_name entirely for such a checkpoint, so the
        # prompt format is moot -- report it, but say why it does not matter.
        logger.info(
            "%s was trained with language_conditioned=false; the policy ignores "
            "the prompt, so point_goal_format is cosmetic",
            config_path,
        )
    return mode


def sync_frame_conditioning(
    cfg: InferenceConfig,
    checkpoint_hint: str | None = None,
    *,
    explicit_prev_frame: bool = False,
    explicit_target_crop: bool = False,
    sync_depth: bool = True,
) -> None:
    """Align the extra policy inputs with what the checkpoint was trained on.

    ``framework.use_previous_front_frame``, ``framework.use_target_front_crop``
    and the ``*_depth`` keys in ``framework.shape_meta`` decide whether the
    encoder expects a ``prev_front_rgb`` / ``target_front_crop_rgb`` / depth
    input. Getting them wrong is not cosmetic -- a checkpoint that wants an
    input and does not get it fails in the server -- so each is switched to
    match, loudly; a value the operator passed explicitly is kept, also loudly.
    A key the config does not carry leaves the setting alone.

    Depth is only ever switched *on* (to ffstereo, the source that needs no
    hardware_config change), and only from the startup path:
    ``launch_ffstereo_server`` has to run before the first inference, so the
    controller's later re-sync passes ``sync_depth=False``.
    """
    config_path = find_checkpoint_config(checkpoint_config_source(cfg, checkpoint_hint))
    framework = None if config_path is None else checkpoint_framework(config_path)
    if not framework:
        return

    def _sync(key: str, attr: str, wanted: object, explicit: bool) -> None:
        current = getattr(cfg, attr)
        if wanted is None:
            logger.info("%s has no framework.%s; keeping %s=%r", config_path, key, attr, current)
        elif wanted == current:
            pass
        elif explicit:
            logger.warning(
                "%s=%r was requested explicitly but the checkpoint was trained with "
                "framework.%s=%r (%s); using the requested value anyway",
                attr, current, key, wanted, config_path,
            )
        else:
            logger.warning(
                "%s %r -> %r to match framework.%s in %s", attr, current, wanted, key, config_path
            )
            setattr(cfg, attr, wanted)

    prev = framework.get("use_previous_front_frame")
    _sync(
        "use_previous_front_frame",
        "include_prev_frame",
        None if prev is None else bool(prev),
        explicit_prev_frame,
    )
    crop = framework.get("use_target_front_crop")
    try:
        crop = None if crop is None else parse_target_front_crop(crop)
    except ValueError:
        logger.warning("%s has an unusable framework.use_target_front_crop=%r", config_path, crop)
        crop = None
    _sync("use_target_front_crop", "use_target_front_crop", crop, explicit_target_crop)

    if not sync_depth:
        return
    # Depth is used when the encoder back-projects pointmaps AND shape_meta
    # declares depth keys -- the same pair the vlagents server checks.
    depth_keys = sorted(
        k
        for k, v in ((framework.get("shape_meta") or {}).get("obs") or {}).items()
        if k.endswith("_depth") and v is not None
    )
    wants_depth = bool(framework.get("use_ptmap", True)) and bool(depth_keys)
    if wants_depth and not (cfg.use_ffstereo or cfg.include_depth):
        logger.warning(
            "the checkpoint consumes depth (%s in %s) but no depth source was configured; "
            "turning on use_ffstereo. Pass --ffstereo to silence this, or set "
            "include_depth=true in %s for ZED depth instead",
            ", ".join(depth_keys), config_path, CONFIG_PATH.name,
        )
        cfg.use_ffstereo = True
    elif not wants_depth and (cfg.use_ffstereo or cfg.include_depth):
        logger.warning(
            "%s declares no depth inputs but %s is on; the server will drop the depth "
            "this client sends",
            config_path, "use_ffstereo" if cfg.use_ffstereo else "include_depth",
        )
    if cfg.use_ffstereo and not INCLUDE_RIGHT:
        logger.warning(
            "use_ffstereo needs the right stereo frames: set INCLUDE_RIGHT = True "
            "(and INCLUDE_DEPTH = False) in hardware_config.py"
        )


def sync_point_goal_format(
    cfg: InferenceConfig,
    checkpoint_hint: str | None = None,
    *,
    explicit: bool = False,
) -> str | None:
    """Align ``cfg.point_goal_format`` with the checkpoint's ``language_mode``.

    ``checkpoint_hint`` is the path the vlagents server reports for the
    checkpoint it loaded; ``cfg.model_config_path`` wins over it. An explicit
    ``--point-goal-format`` is never overridden, only warned about. Returns the
    mode read from the checkpoint, or ``None`` when it could not be read.
    """
    source = checkpoint_config_source(cfg, checkpoint_hint)
    config_path = find_checkpoint_config(source)
    if config_path is None:
        logger.info(
            "no checkpoint config found for %r; point_goal_format=%s is "
            "unverified (set model_config_path in %s to check it)",
            source or "<unset>",
            cfg.point_goal_format,
            CONFIG_PATH.name,
        )
        return None

    mode = checkpoint_language_mode(config_path)
    if mode is None:
        return None
    if mode == cfg.point_goal_format:
        logger.info(
            "point_goal_format=%s matches framework.language_mode in %s",
            cfg.point_goal_format,
            config_path,
        )
        return mode
    if explicit:
        logger.warning(
            "point_goal_format=%s was requested explicitly but the checkpoint "
            "was trained with framework.language_mode=%s (%s); sending the "
            "requested format anyway",
            cfg.point_goal_format,
            mode,
            config_path,
        )
        return mode
    logger.warning(
        "point_goal_format %s -> %s to match framework.language_mode in %s",
        cfg.point_goal_format,
        mode,
        config_path,
    )
    cfg.point_goal_format = mode
    return mode
