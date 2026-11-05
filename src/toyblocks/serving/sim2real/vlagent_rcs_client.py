"""Drive maniflow vlagents policy against an rcs env.

The action's ``pd_mode`` channel (index 8) is ignored: only joints ``[0:7]``
and gripper ``[7:8]`` are forwarded to the env.

All runtime-tunable properties (instruction, n_action_steps/horizon, fps, ...)
live in ``rollout/vlagent_rcs_client.json`` (auto-created on first run).
Edit that file and send the ``o`` command at the prompt to hot-reload it
without restarting the client. Note that env-level settings baked in at
``get_env()`` time (fps, max_rel_mov_joints/cart, control mode) require a
restart to take effect; only settings read per-loop-iteration (instruction,
n_action_steps, robot_keys, vlagents_host/port/model, record_path) are truly
live-reloadable.

Example:

    python -m toyblocks.serving.sim2real.vlagent_rcs_client

Evaluation logging: episodes are recorded by default (``record`` in the JSON) into
``<checkpoint dir>/<record_path>/``, next to the weights the vlagents server loaded.
The GUI's "Submit (success)/(failed)" and "Stop (success)/(failed)" buttons grade the
command that just ran, appending a line to ``eval_log_<date>.json`` in that same folder
naming the command, the verdict, and the parquet fragments holding it. The first line of
each log records the ToyBlocks and hvla commits, the policy checkpoint, the stereo
source and the PD stiffness the run used. See ``eval_log.py``.

Interactive goal GUI (browser at http://localhost:8765 — click a point + submit):

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --interactive-gui
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --interactive-gui --point-goal-format action_only
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --interactive-gui --point-goal-format action_goal
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --interactive-gui --stop-after-chunk

Previous front frame (only for checkpoints trained with ``framework.use_previous_front_frame``;
off by default, ``include_prev_frame`` in the JSON): clicking a goal point freezes the plain
front frame at that moment (no marker, no heatmap) and sends it to the ManiFlow server as a
``prev_front_rgb`` camera on every later inference. Before the first click the live front
frame stands in, so the key is never missing.

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --interactive-gui --prev-frame

Target-centered front crop (``use_target_front_crop`` in the JSON): a 224x224 window cropped
out of the front frame around the goal point, sent as an extra ``target_front_crop_rgb`` camera.
The number is the percent of the crop *below* the point (``25`` = 75% above / 25% below):

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --interactive-gui --use-target-front-crop 25

None of these extra inputs has to be passed by hand: the checkpoint's ``config.yaml`` (next to
the weights -- ``model_config_path``, else the path the server reports, else ``policy_checkpoint``
in the JSON) decides them. ``framework.use_previous_front_frame``, ``use_target_front_crop`` and
the depth keys in ``shape_meta`` switch ``--prev-frame``, ``--use-target-front-crop`` and
``--ffstereo`` on (or off) with a warning; a value passed on the command line is kept, also with
a warning.

Temporal action ensembling (ACT-style; requires n_action_steps in the JSON):

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --temporal-agg
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --temporal-agg --n-steps-per-prediction 12 
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --temporal-agg --temporal-agg-k 0.05

Dry-run (robot off; uses hardware cameras when connected, else mock frames):

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --dry-run

Save the first camera frame(s) to disk (respects INCLUDE_DEPTH / INCLUDE_RIGHT):

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --save-first-frame
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --save-first-frame --save-first-frame-dir /tmp/cam_check

Fast-FoundationStereo depth instead of ZED depth (set INCLUDE_RIGHT=True, INCLUDE_DEPTH=False in
hardware_config.py; the depth server is started automatically, settings ``ffstereo_*`` in the JSON):

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --ffstereo
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --ffstereo --ffstereo-clip-m 1.5
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --ffstereo --ffstereo-clip-min-m 0.2


TAM (Torque Adaptation Module; hardware only, see ``tam_utils.py``): a learned residual torque
on top of the PD controller. The JAX history encoder runs in this process (GPU recommended) and
feeds latents to the C++ controller at ~5 Hz; the PD gains switch to TAM's in-distribution range.
Set ``tam`` (and the other ``tam_*`` keys) in the JSON, or override from the CLI. Requires the
``torque-adaptation-module`` (``simadaptor``) package. Baked in at env creation -> restart to change.

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --tam
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --tam --no-tam-warmup
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --no-tam
    
Molmo2 HL (vision-language) instead of a manual click + action goal: the front frame + a natural
language prompt (from the GUI text box, seeded by ``molmo_prompt`` in the JSON) are sent to the
molmo2 server every obs-build step; the parsed language + target pixel become the ManiFlow
``instruction`` + ``target_heatmap``. ``--async-molmo`` instead queries the HL only when the
low-level skill ends (execution status -1), at episode start, and on HL-prompt changes, reusing
the cached instruction + heatmap in between. ``--use-molmo`` implies ``--interactive-gui`` (the HL prompt
is entered from the browser text box) and disables point-clicking on ``front_rgb``.

The Molmo2 server runs ``serving/molmo2_hl.py`` (the sim eval's Molmo2 code) and reads what the
checkpoint was trained on -- crop, front-only, frame size, preprocessor style, question format,
legacy ``pick <812, 705>`` or html-v2 answers -- from the checkpoint's own config. Type only the
goal (e.g. ``stack G V L``); the server adds the previous action and, for html-v2, the
``point to target. predict action.`` instruction. An html-v2 point (0-1000 of the crop the model
saw) comes back in full-frame pixels.

Works with both action-chunking modes:

- without ``--temporal-agg``: Molmo2 -> ffstereo -> ManiFlow run in sequence on the main thread
  at every chunk refill.
- with ``--temporal-agg``: Molmo2 + ffstereo + ManiFlow run in the background inference worker
  in parallel with the main controller loop, which keeps consuming ensembled actions at loop rate.

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo --molmo-checkpoint /path/to/molmo2
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo --async-molmo
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo --temporal-agg --ffstereo

``--molmo-front-view-only``, ``--molmo-front-crop`` and ``--molmo-input-size`` must match the
checkpoint's training config (front_view_only / front_crop / image_size in
``olmo/data/robot_datasets.py``); the server checks them and raises on a mismatch. The crop box
is in Molmo-space (640x360), the same frame the training dataloader cropped:

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo --molmo-front-view-only
    python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo --molmo-front-view-only --molmo-front-crop 30 90 610 360

Drop the ``Previous action: ...`` prefix from every Molmo2 prompt (checkpoint trained without
a rolling action history):

    python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo --no-molmo-prev-action

Remote SSH: forward the web GUI from your laptop::

    ssh -L 8765:localhost:8765 user@robots-workstation1
    # then open http://localhost:8765
"""

from __future__ import annotations

import os

# The TAM history encoder (JAX) shares the GPU with everything else on this
# machine; never let JAX preallocate ~75% of device memory. Must be set before
# JAX initializes its backend (first use inside ModelInference._init_tam), so
# it goes before every other import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import argparse  # noqa: E402
import logging  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402
from time import sleep  # noqa: E402

from rcs._core.common import RobotPlatform  # noqa: E402

from toyblocks.serving.sim2real.rollout.env_setup import get_env  # noqa: E402
from toyblocks.serving.sim2real.rollout.ffstereo_server import launch_ffstereo_server
from toyblocks.serving.sim2real.rollout.gui_utils import WEB_GUI_PORT, launch_goal_gui
from toyblocks.serving.sim2real.rollout.hardware_config import ROBOT_INSTANCE
from toyblocks.serving.sim2real.rollout.molmo2_server import launch_molmo2_server
from toyblocks.serving.sim2real.rollout.inference import (
    DryRunInference,
    ModelInference,
    command_loop,
)
from toyblocks.serving.sim2real.rollout.inference_config import (
    CONFIG_PATH,
    DEFAULT_SAVE_FIRST_FRAME_DIR,
    POINT_GOAL_FORMATS,
    load_inference_config,
    parse_target_front_crop,
    sync_frame_conditioning,
    sync_point_goal_format,
    warn_remote_transport_config,
)

logger = logging.getLogger(__name__)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


def _install_sigterm_handler() -> None:
    """Make ``kill`` unwind the same way Ctrl-C does.

    SIGTERM's default action kills the process outright, so ``with env_rel:``
    never exits and StorageWrapper's buffered rows -- up to ``batch_size``
    steps, about a second at 30 Hz -- never reach disk. Raising
    KeyboardInterrupt instead reuses every shutdown path the client already has
    for Ctrl-C, in all three of its main-thread shapes (GUI wait loop, terminal
    command_loop, sim inference loop). The eval log needs no such protection:
    each line is written and closed as it is graded.
    """

    def _raise(signum: int, _frame: object) -> None:
        logger.info("received %s; shutting down", signal.Signals(signum).name)
        raise KeyboardInterrupt

    try:
        signal.signal(signal.SIGTERM, _raise)
    except ValueError:
        # Only the main thread can install handlers; leave the default action.
        logger.debug("could not install a SIGTERM handler", exc_info=True)


def main() -> None:
    _install_sigterm_handler()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--interactive-gui",
        action="store_true",
        help="Open browser GUI at http://localhost:<gui-port> to pick XY and submit goals.",
    )
    parser.add_argument(
        "--point-goal-format",
        choices=POINT_GOAL_FORMATS,
        default=None,
        help=(
            "action_goal appends ' u,v'; action_only sends the action string only. "
            "Unset, this is taken from the checkpoint's framework.language_mode."
        ),
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help=(
            "Checkpoint whose framework.language_mode decides --point-goal-format: "
            "its config.yaml, its directory, or the *.safetensors in it (sets "
            "model_config_path). Unset, the path the server reports is used."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without robot motion (cameras if connected).",
    )
    parser.add_argument(
        "--gui-port",
        type=int,
        default=WEB_GUI_PORT,
        help=f"Port for browser GUI (default: {WEB_GUI_PORT}).",
    )
    parser.add_argument(
        "--stop-after-chunk",
        action="store_true",
        help="Execute one policy chunk (n_action_steps) per start/submit, then wait for the next command.",
    )
    parser.add_argument(
        "--temporal-agg",
        action="store_true",
        help=(
            "ACT-style temporal action ensembling. Requires n_action_steps."
        ),
    )
    parser.add_argument(
        "--temporal-agg-k",
        type=float,
        default=None,
        help="Decay constant for --temporal-agg (larger = older chunks dominate more sharply).",
    )
    parser.add_argument(
        "--n-steps-per-prediction",
        type=int,
        default=None,
        help=(
            "Query the policy every N executed steps when --temporal-agg is on. "
        ),
    )
    parser.add_argument(
        "--save-first-frame",
        action="store_true",
        help=(
            "Save the first available camera frame(s) to disk once at startup. "
            "Writes left RGB always; adds right RGB / depth when INCLUDE_RIGHT / "
            "INCLUDE_DEPTH are enabled in hardware_config.py."
        ),
    )
    parser.add_argument(
        "--save-first-frame-dir",
        default=DEFAULT_SAVE_FIRST_FRAME_DIR,
        help=f"Output directory for --save-first-frame (default: {DEFAULT_SAVE_FIRST_FRAME_DIR}).",
    )
    parser.add_argument(
        "--prev-frame",
        action="store_true",
        help=(
            "Send the front frame captured when the goal point was clicked as an "
            "extra prev_front_rgb camera (sets include_prev_frame=true). Only for "
            "checkpoints trained with framework.use_previous_front_frame."
        ),
    )
    parser.add_argument(
        "--use-target-front-crop",
        type=int,
        default=None,
        metavar="BOTTOM_PCT",
        help=(
            "Send a 224x224 front-frame crop centered on the goal point as an extra "
            "target_front_crop_rgb camera. The value is the percent of the crop below "
            "the point (0-99; 25 = 75%% top / 25%% bottom), -1 disables. Taken from the "
            "checkpoint's framework.use_target_front_crop when not passed."
        ),
    )
    parser.add_argument(
        "--ffstereo",
        action="store_true",
        help="Use Fast-FoundationStereo depth from the left/right pair (sets use_ffstereo=true).",
    )
    parser.add_argument(
        "--ffstereo-clip-m",
        type=float,
        default=None,
        help="Override ffstereo_clip_m from the JSON: FFS depth beyond this is invalid (0 disables).",
    )
    parser.add_argument(
        "--ffstereo-clip-min-m",
        type=float,
        default=None,
        help="Override ffstereo_clip_min_m from the JSON: FFS depth below this is invalid (0 disables).",
    )
    tam_group = parser.add_mutually_exclusive_group()
    tam_group.add_argument(
        "--tam",
        dest="tam",
        action="store_true",
        default=None,
        help=(
            "Enable the Torque Adaptation Module (sets tam=true): learned residual "
            "torque + TAM PD gains on the Franka controller, JAX history encoder in "
            "this process. Hardware only; needs the simadaptor package."
        ),
    )
    tam_group.add_argument(
        "--no-tam",
        dest="tam",
        action="store_false",
        help="Disable TAM even if tam=true in the JSON.",
    )
    parser.add_argument(
        "--no-tam-warmup",
        action="store_true",
        help="Skip the blocking JAX warmup of the TAM history encoder (sets tam_warmup=false).",
    )
    parser.add_argument(
        "--use-molmo",
        action="store_true",
        help=(
            "Use the Molmo2 HL server to produce the LL instruction + target "
            "heatmap from a natural-language prompt (GUI text box or "
            "molmo_prompt in the JSON) applied to the front_rgb frame. "
            "Disables point clicking in the interactive GUI. Compatible "
            "with --temporal-agg."
        ),
    )
    parser.add_argument(
        "--molmo-checkpoint",
        default=None,
        help="Override molmo_checkpoint from the JSON: path to the Molmo2 HL checkpoint directory.",
    )
    parser.add_argument(
        "--async-molmo",
        action="store_true",
        help=(
            "Only query the Molmo2 HL at low-level skill boundaries "
            "(execution status -1), at episode start, and on HL-prompt "
            "changes, reusing the cached instruction + target heatmap in "
            "between, instead of calling it on every obs build. Requires "
            "--use-molmo."
        ),
    )
    parser.add_argument(
        "--no-molmo-prev-action",
        action="store_true",
        help=(
            "Drop the ``Previous action: <last raw>. `` prefix from every "
            "Molmo2 prompt (sets molmo_use_prev_action=false). The server "
            "still tracks the rolling prev_action_str internally so toggling "
            "this at runtime stays coherent, but it is never used in the "
            "prompt."
        ),
    )
    parser.add_argument(
        "--molmo-front-view-only",
        action="store_true",
        help=(
            "Mirror the training dataloader's front_view_only=True: skip "
            "sending gripper_rgb to Molmo2 so a front-only checkpoint sees "
            "the same single-image prompt it was trained on (sets "
            "molmo_front_view_only=true)."
        ),
    )
    parser.add_argument(
        "--molmo-front-crop",
        type=int,
        nargs=4,
        default=None,
        metavar=("X0", "Y0", "X1", "Y1"),
        help=(
            "Crop front_rgb to the pixel box (x0, y0, x1, y1) before sending "
            "it to Molmo2, mirroring front_crop in the training dataloader. "
            "Pass this only for a checkpoint actually trained with that crop: "
            "cropping a full-frame checkpoint's input offsets every predicted "
            "<u,v>. The returned <u,v> is in full 640x360 front-frame coords "
            "either way. Sets molmo_front_crop; omitted, the JSON value is "
            "used (no crop by default)."
        ),
    )
    parser.add_argument(
        "--molmo-input-size",
        type=int,
        nargs=2,
        default=None,
        metavar=("WIDTH", "HEIGHT"),
        help=(
            "Resolution the Molmo2 HL checkpoint was trained on (default "
            "640 360). front_rgb is resized to it before being sent, "
            "--molmo-front-crop is interpreted in it, and the predicted "
            "<u,v> is read out of it. Set 1280 720 for a checkpoint trained "
            "at full ZED resolution. Independent of the 640x360 ManiFlow "
            "policy space. Sets molmo_input_size."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    cfg = load_inference_config()
    if args.interactive_gui or args.dry_run:
        cfg.interactive_gui = True
    if args.dry_run:
        cfg.dry_run = True
    if args.model_config is not None:
        cfg.model_config_path = args.model_config
    if args.point_goal_format is not None:
        cfg.point_goal_format = args.point_goal_format
    # Read framework.language_mode off the checkpoint now when a path is
    # configured; otherwise the controller retries with the checkpoint path the
    # server reports, once it is connected.
    sync_point_goal_format(cfg, explicit=args.point_goal_format is not None)
    if args.stop_after_chunk:
        cfg.stop_after_action_chunk = True
    if args.save_first_frame:
        cfg.save_first_frame = True
        cfg.save_first_frame_dir = args.save_first_frame_dir
    if args.prev_frame:
        cfg.include_prev_frame = True
    if args.use_target_front_crop is not None:
        try:
            cfg.use_target_front_crop = parse_target_front_crop(
                args.use_target_front_crop
            )
        except ValueError as exc:
            parser.error(str(exc))
    # Both extra cameras are decided by the checkpoint, not by taste: mirror
    # framework.use_previous_front_frame / use_target_front_crop unless the
    # operator asked for a specific value on the command line.
    sync_frame_conditioning(
        cfg,
        explicit_prev_frame=args.prev_frame,
        explicit_target_crop=args.use_target_front_crop is not None,
    )
    if args.ffstereo:
        cfg.use_ffstereo = True
    if args.ffstereo_clip_m is not None:
        cfg.ffstereo_clip_m = args.ffstereo_clip_m
    if args.ffstereo_clip_min_m is not None:
        cfg.ffstereo_clip_min_m = args.ffstereo_clip_min_m
    if args.tam is not None:
        cfg.tam = args.tam
    if args.no_tam_warmup:
        cfg.tam_warmup = False
    if cfg.tam and cfg.dry_run:
        logger.info("dry-run: TAM has no robot to attach to, ignoring tam=true")
        cfg.tam = False
    if cfg.tam and ROBOT_INSTANCE != RobotPlatform.HARDWARE:
        logger.warning(
            "tam=true but ROBOT_INSTANCE is not HARDWARE; TAM only exists in the "
            "Franka hardware controller, ignoring"
        )
        cfg.tam = False
    if args.use_molmo:
        cfg.use_molmo = True
    if args.molmo_checkpoint is not None:
        cfg.molmo_checkpoint = args.molmo_checkpoint
    if args.async_molmo:
        cfg.async_molmo = True
    if args.no_molmo_prev_action:
        cfg.molmo_use_prev_action = False
    if args.molmo_front_view_only:
        cfg.molmo_front_view_only = True
    if args.molmo_input_size is not None:
        width, height = args.molmo_input_size
        if width <= 0 or height <= 0:
            parser.error(
                f"--molmo-input-size requires positive dimensions; got "
                f"({width}, {height})."
            )
        cfg.molmo_input_size = (width, height)
    if args.molmo_front_crop is not None:
        x0, y0, x1, y1 = args.molmo_front_crop
        if not (x0 < x1 and y0 < y1):
            parser.error(
                f"--molmo-front-crop requires x0 < x1 and y0 < y1; got "
                f"({x0}, {y0}, {x1}, {y1})."
            )
        cfg.molmo_front_crop = (x0, y0, x1, y1)
    if cfg.async_molmo and not cfg.use_molmo:
        logger.warning("async_molmo=true but use_molmo=false; ignoring")
        cfg.async_molmo = False
    # Molmo2 HL is prompted through the interactive GUI's text box: there is
    # no CLI text-entry path for a natural-language HL instruction, so
    # ``use_molmo`` unconditionally implies the interactive GUI (mirrors the
    # ``--dry-run`` behaviour above).
    if cfg.use_molmo:
        cfg.interactive_gui = True
        if not cfg.molmo_checkpoint:
            parser.error(
                "--use-molmo requires molmo_checkpoint (set it in "
                f"{CONFIG_PATH.name} or pass --molmo-checkpoint)."
            )
    if args.temporal_agg:
        cfg.temporal_agg = True
    if args.temporal_agg_k is not None:
        cfg.temporal_agg_k = args.temporal_agg_k
    if args.n_steps_per_prediction is not None:
        cfg.n_steps_per_prediction = args.n_steps_per_prediction
    if cfg.temporal_agg:
        from rcs.envs.base import RelativeTo

        from toyblocks.serving.sim2real.rollout.hardware_config import RELATIVETO

        if cfg.n_action_steps is None or cfg.n_action_steps <= 0:
            parser.error(
                "--temporal-agg requires n_action_steps > 0 in "
                f"{CONFIG_PATH.name} (that's the chunk horizon H)."
            )
        if cfg.stop_after_action_chunk:
            parser.error("--stop-after-chunk is incompatible with --temporal-agg.")
        if RELATIVETO != RelativeTo.NONE:
            parser.error(
                "--temporal-agg currently only supports "
                "RELATIVETO=RelativeTo.NONE (absolute joint/pose targets); "
            )
        stride = cfg.n_steps_per_prediction if cfg.n_steps_per_prediction is not None else 1
        if stride < 1 or stride > cfg.n_action_steps:
            parser.error(
                f"--n-steps-per-prediction must be in [1, n_action_steps]="
                f"[1, {cfg.n_action_steps}]; got {stride}."
            )
        if cfg.n_steps_per_prediction is None:
            cfg.n_steps_per_prediction = 1
    warn_remote_transport_config(cfg)

    if cfg.use_ffstereo:
        # Same interpreter, child process; it exits by itself once this client is gone.
        launch_ffstereo_server(cfg)

    if cfg.use_molmo:
        # Same interpreter, child process; it exits by itself once this client is gone.
        launch_molmo2_server(cfg)

    if cfg.dry_run:
        logger.info("Dry-run mode: skipping robot env")
        controller = DryRunInference(cfg)
        worker = threading.Thread(
            target=controller.loop, name="dry-run", daemon=True
        )
        worker.start()
        launch_goal_gui(controller, gui_port=args.gui_port)
        controller.request_shutdown()
        controller.close()
        worker.join()
        return

    logger.info("Building env")
    env_rel = get_env(cfg)
    env_rel.reset(seed=2)
    controller = ModelInference(env_rel, cfg)

    if cfg.tam:
        # Finish JAX compilation before the robot moves so the first live
        # latents are already fast (no startup stalls -> no stale-latent jumps).
        controller.warmup_history_encoder()

    with env_rel:
        if cfg.tam:
            # The controller applies zero residual until the MLP weights and
            # the first latent arrive, then ramps in over 1 s. The thread runs
            # across episodes/resets; the runtime bridges controller restarts.
            controller.start_history_encoder()

        if ROBOT_INSTANCE == RobotPlatform.SIMULATION and cfg.interactive_gui:
            # Sim env step/reset must run on the main thread, so swap the
            # roles: serve the GUI on a worker and run the loop on main.
            worker = threading.Thread(
                target=launch_goal_gui,
                args=(controller,),
                kwargs={"gui_port": args.gui_port},
                name="web-gui",
                daemon=True,
            )
            worker.start()
            controller.loop()
        elif ROBOT_INSTANCE == RobotPlatform.HARDWARE and cfg.interactive_gui:
            worker = threading.Thread(
                target=controller.loop, name="model-inference", daemon=True
            )
            worker.start()
            launch_goal_gui(controller, gui_port=args.gui_port)
            worker.join()
        elif ROBOT_INSTANCE == RobotPlatform.SIMULATION:
            sleep(5)
            # cfg.record (default true) decides whether the auto-started sim
            # episode is written to the parquet dataset.
            controller.submit_command("r" if cfg.record else "e")
            controller.loop()
        else:
            worker = threading.Thread(
                target=controller.loop, name="model-inference", daemon=True
            )
            worker.start()
            command_loop(controller)
            worker.join()


if __name__ == "__main__":
    main()
