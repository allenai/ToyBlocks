"""Launch the queue-driven eval on this machine.

The pipeline itself lives once in ``scripts/eval/run_eval_pipeline.sh`` (policy server -> task
codegen -> episode queue -> right-size num_envs -> optional wandb coordinator -> N eval workers ->
teardown), driven entirely by environment variables. This launcher builds that environment and
runs the pipeline as a subprocess; ``--dry-run`` prints the ``env VAR=... bash
run_eval_pipeline.sh`` invocation instead. GPU 0 serves the policy and GPUs 1..N run the eval
workers (0..N-1 under ``--replay``); with fewer GPUs than that, workers share them round-robin.

Supports Molmo2 + ManiFlow (``--policy molmo2_maniflow``, the default; ``--ll-oracle`` swaps the
ManiFlow LL for the vecgen skills) and ManiFlow with the HL oracle (``--policy maniflow``), and
dataset replay via ``--replay`` (no policy server; each eval worker plays back the queue episodes'
recorded actions with ``evaluate.py --replay``). ``--smoke-test`` turns the defaults (1 eval GPU,
full queue, INFO logs, ``evals/`` output dir) into a small validation run (queue capped at 10,
DEBUG logs, short episodes, ``queue_test/`` output dir). Individual knobs (``--queue-limit``,
``--queue-log-level``, ``--num-eval-gpus``, ``--out-subdir``) override either profile.

Maniflow evals require ``--model-path`` (the safetensors file, inside the run's ``checkpoints/``
dir); ``--config-path`` and ``--train-config-path`` have sensible defaults, and ``--stats-path`` is
only needed to override the normalization bands the checkpoint records for itself.

``--dataset-path`` (required) is the LeRobot root the episodes come from. The gym task slug (and
class names of the autogen env package) is the dataset directory basename with hyphens replaced by
underscores; pass ``--task-name`` to override. ``scripts/generate_task_from_dataset.py`` reads
``pddl_goal`` and the ``observation.state.object_pose.*`` columns from that dataset to emit the env
package.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROBLEM_ID = "problem_000000"
# Keep in sync with toyblocks.tamp.blocks.glyph_blocks.EVAL_DEFAULT_GLYPH_LIBRARY.
DEFAULT_USD_LIBRARY = "train_blocks_large"

# -- Per-policy defaults --

_MANIFLOW_DEFAULTS = dict(
    obs_transform="maniflow_blocks_table",
    # Empty → defer to the dataset's recorded provenance (scene_id/robot_id).
    # reconcile_env_cfg_with_dataset fills these; pass --robot-id/--scene-preset
    # to override (and skip provenance).
    robot_id="",
    scene_preset="",
    policy_name="maniflow",
    action_mode="",  # empty → server auto-detects from config.yaml
    model_path="",  # required from CLI
    config_path="",  # derived from model_path
    train_config_path="",  # uses H_VLA_PATH default in template
    stats_path="",  # derived from dataset_path
    action_horizon=25,
)

_MOLMO2_MANIFLOW_DEFAULTS = dict(
    obs_transform="maniflow_blocks_table",
    # Empty → defer to dataset provenance (see _MANIFLOW_DEFAULTS). Pass
    # --scene-preset droid_with_box_table to match the interactive flow
    # (scripts/eval/serve_local_client.sh) when the dataset records no scene.
    robot_id="",
    scene_preset="",
    policy_name="molmo2_maniflow",
    action_mode="",  # empty → server auto-detects from config.yaml
    model_path="",  # used for LL
    hl_model_path="",  # required from CLI
    config_path="",  # derived from model_path
    train_config_path="",
    stats_path="",
    action_horizon=25,
)

# --replay: no policy server; the eval workers play back each queue episode's
# recorded actions (evaluate.py --replay). Model / HL / transform fields are
# empty because nothing reads them on this path.
_REPLAY_DEFAULTS = dict(
    obs_transform="",
    robot_id="",
    scene_preset="",
    policy_name="replay",
    action_mode="",
    model_path="",
    config_path="",
    train_config_path="",
    stats_path="",
    action_horizon=0,
)

_POLICY_DEFAULTS = {
    "maniflow": _MANIFLOW_DEFAULTS,
    "molmo2_maniflow": _MOLMO2_MANIFLOW_DEFAULTS,
}


def task_slug_from_dataset_path(dataset_path: str) -> str:
    """LeRobot dataset directory basename with ``-`` replaced by ``_``.

    Keep in sync with :func:`toyblocks.tamp.blocks.template.codegen.task_slug_from_dataset_path`
    (can't import directly — codegen pulls Isaac via the pddl_goal_tensor_conversions chain).
    """
    name = os.path.basename(os.path.normpath(dataset_path))
    return name.replace("-", "_")


_MANIFLOW_OBS_TRANSFORMS = frozenset({"maniflow", "maniflow_blocks_table"})


def _require_maniflow_obs_transform(obs_transform: str, flag: str) -> None:
    if obs_transform not in _MANIFLOW_OBS_TRANSFORMS:
        raise SystemExit(
            f"{flag} only applies to policy evals with a maniflow obs "
            f"transform (got {obs_transform!r})"
        )


# Default profile: the full queue on one eval GPU (pass --num-eval-gpus for more).
_PROD_PROFILE = dict(
    queue_limit_arg="",
    queue_log_level="INFO",
    num_envs=50,
    num_eval_gpus=1,
    episode_length_s=30.0,
    out_subdir="evals",
)

# Smoke-test overrides: small queue, single eval GPU, debug logs, short episodes.
_SMOKE_PROFILE = dict(
    queue_limit_arg="--limit 10",
    queue_log_level="DEBUG",
    num_envs=2,
    num_eval_gpus=1,
    episode_length_s=3.0,
    out_subdir="queue_test",
)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # -- Policy selection --
    p.add_argument("--policy", choices=["maniflow", "molmo2_maniflow"], default="molmo2_maniflow",
                   help="Policy type to evaluate (default: molmo2_maniflow).")
    p.add_argument(
        "--replay",
        action="store_true",
        default=False,
        help="Replay the dataset's recorded actions instead of evaluating a "
        "policy: no policy server is started and each eval worker runs "
        "evaluate.py --replay against the queue. Overrides --policy; "
        "--num-envs defaults to 1 and must stay 1 (recorded-gain settling "
        "advances the global sim). --model-path/--hl-model-path/"
        "--oracle-comparison/--replicate-noise are rejected; "
        "--use-oracle is ignored. Output goes to runs/<task>-v0/replay/...",
    )
    p.add_argument(
        "--replay-clip-stats",
        default="",
        help="With --replay: stats.json (e.g. a maniflow run's) whose "
        "action.joint_target q01/q99 bands every replayed arm joint target is "
        "clamped to, to test whether a policy trained on labels clipped to those "
        "bands could still succeed. Adds replay_clipped_frames / "
        "replay_clip_max_rad per-episode metrics.",
    )
    p.add_argument(
        "--center-crop-ratio",
        type=float,
        default=None,
        help="Maniflow policies only: centre-crop each resized 224x224 RGB input "
        "to int(224 * ratio) and resize it back, matching the average view of a "
        "checkpoint trained with random_crop_ratio=ratio (e.g. 0.95). Default: "
        "0.95 for maniflow obs transforms, unset otherwise. Pass 1.0 to feed the "
        "full frame.",
    )
    # -- Profile --
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Use the queue-eval smoke-test profile "
        "(queue capped at 10, 1 eval GPU, DEBUG logs, "
        "short episodes, queue_test/ output). "
        "Per-knob flags below override the profile values.",
    )

    # -- Common args --
    p.add_argument(
        "--dataset-path",
        required=True,
        help="LeRobot dataset directory the queue server reads "
        "setups from AND the autogen task is built from.",
    )
    p.add_argument(
        "--task-name",
        default=None,
        help="Override the gym task slug derived from the dataset "
        "basename (default: <basename with - -> _>; gym id is "
        "<slug>-v0). Must be a valid Python identifier.",
    )
    p.add_argument(
        "--policy-name",
        default=None,
        help="Policy label baked into wandb names + output paths (default: derived from --policy).",
    )
    p.add_argument(
        "--problem-id",
        default=DEFAULT_PROBLEM_ID,
        help=f"Output-path PROBLEM_ID segment (default: {DEFAULT_PROBLEM_ID}).",
    )
    p.add_argument(
        "--obs-transform",
        default=None,
        help="--obs_transform passed to evaluate.py (default: policy-specific).",
    )
    p.add_argument(
        "--robot-id", default=None, help="--env-cfg.robot-id (default: policy-specific)."
    )
    p.add_argument(
        "--scene-preset", default=None, help="--env-cfg.scene-preset-id (default: policy-specific)."
    )
    p.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="--action_horizon passed to evaluate.py: actions dispensed per "
        "policy chunk (default: 25 for maniflow / molmo2_maniflow).",
    )
    p.add_argument(
        "--use-oracle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drive task_name + target_heatmap from the client-side TAMP oracle "
        "(forwards --use_oracle to evaluate.py). On by default for '--policy "
        "maniflow'; use --no-use-oracle to disable. The oracle needs a PDDL "
        "problem per episode (queue setup 'pddl_problem_file' or "
        "--pddl-problem-file). The other HL option is a learned model served via "
        "'--policy molmo2_maniflow'.",
    )
    p.add_argument(
        "--episode-indices",
        default="",
        help="Optional --episode-indices spec for serve_episode_queue.py (e.g. '0,3,5,10-19').",
    )
    p.add_argument(
        "--init-after-hl-action",
        default="",
        help="Start each queued episode at the first frame after every "
        "instance of any of these high-level actions (comma or '|', e.g. "
        "'flip,rotate'). 'flip' matches both 'flip' and 'reorient'. "
        "Episodes with no match start at frame 0; episodes whose last "
        "match is the last frame are skipped. Reset poses, expert frames, "
        "and --expert-timeout-scale remaining time come from the cut. "
        "Empty (default) starts at frame 0. Codegen is unchanged.",
    )
    p.add_argument(
        "--usd-library",
        default=DEFAULT_USD_LIBRARY,
        help="Glyph asset library for bare block names in autogen codegen "
        f"(default: {DEFAULT_USD_LIBRARY}). Passed to "
        "generate_task_from_dataset.py; pass train_blocks for legacy "
        "small-block datasets.",
    )
    p.add_argument(
        "--allow-duplicate-episodes",
        action="store_true",
        default=False,
        help="Allow duplicate entries in --episode-indices (e.g. "
        "'0,0,0,0,0' with --num-envs 5 runs episode 0 five "
        "times in parallel). Videos are named by env_idx so "
        "filenames remain unique.",
    )

    # -- Maniflow-specific args --
    maniflow_group = p.add_argument_group("maniflow", "Maniflow policy arguments")
    maniflow_group.add_argument(
        "--model-path",
        default=None,
        help="Maniflow safetensors checkpoint path (required for --policy maniflow).",
    )
    maniflow_group.add_argument(
        "--stats-path",
        default=None,
        help=(
            "Maniflow stats.json path (default: unset, letting the server use the "
            "checkpoint's own framework.train_data_stats, else <model-path>/../stats.json)."
        ),
    )
    maniflow_group.add_argument(
        "--config-path",
        default=None,
        help="Maniflow config.yaml path (default: <model-path>/../config.yaml).",
    )
    maniflow_group.add_argument(
        "--train-config-path",
        default=None,
        help="Maniflow train config path "
        "(default: $H_VLA_PATH/examples/Franka/train_maniflow_multimodal.yaml).",
    )
    maniflow_group.add_argument(
        "--action-mode",
        default=None,
        help="Maniflow action mode: relative or absolute (default: auto-detected from the checkpoint).",
    )

    # -- Policy server seed --
    p.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Random seed for the policy server (controls action "
        "sampling stochasticity). Default: 0.",
    )

    # -- Deterministic CUDA --
    p.add_argument(
        "--deterministic-cuda",
        action="store_true",
        help="Set cudnn.deterministic=True and cudnn.benchmark=False "
        "on both the policy server and eval workers. Slower but "
        "bitwise reproducible.",
    )
    p.add_argument(
        "--async-hl",
        action="store_true",
        default=False,
        help="--policy molmo2_maniflow with a play-mode LL: query the Molmo2 HL only at "
        "episode start and when the LL ends a skill (play mode -1), as sim2real's "
        "async_molmo; default queries it at every chunk boundary.",
    )
    p.add_argument(
        "--go-home",
        action="store_true",
        default=False,
        help="--async-hl or --ll-oracle: after the LL (or oracle) ends a place, flip or "
        "rotate, drive the arm to the home pose before the next Molmo2 query.",
    )
    p.add_argument(
        "--ll-oracle",
        action="store_true",
        default=False,
        help="--policy molmo2_maniflow with the vecgen low-level oracle (cuRobo v2) in place of "
        "the served LL: each Molmo2 command runs on the block nearest its pixel (or the "
        "table within 25 px), and Molmo2 is queried again per env once it ends: always "
        "async HL (--async-hl is implied; its play-mode LL path is not used).",
    )
    p.add_argument(
        "--async-hl-call-scale",
        type=float,
        default=None,
        help="--async-hl: end an episode once Molmo2 is called more than this many times "
        "its expert's HL calls (default: no cap).",
    )
    p.add_argument(
        "--ll-oracle-hl-call-scale",
        type=float,
        default=3.0,
        help="--ll-oracle: end an episode once Molmo2 is called more than this many times "
        "its expert's HL calls.",
    )
    p.add_argument(
        "--no-hl-prev-action",
        dest="hl_no_prev_action",
        action="store_true",
        default=False,
        help="--policy molmo2_maniflow: never put the previous action in the Molmo2 "
        "question (serve_maniflow_policy --no-hl-prev-action); default feeds it back.",
    )
    p.add_argument(
        "--hl-max-new-tokens",
        type=int,
        default=None,
        help="--policy molmo2_maniflow: Molmo2 decode budget per answer "
        "(serve_maniflow_policy --hl_max_new_tokens); default: the server's.",
    )
    p.add_argument(
        "--replicate-noise",
        action="store_true",
        default=False,
        help="Start every sample in a batched maniflow inference call from the "
        "same flow-matching noise, so identical observations produce identical "
        "actions regardless of batch position (forwards --replicate-noise to "
        "serve_maniflow_policy). Requires --policy maniflow (the molmo2_maniflow "
        "server launch does not forward it).",
    )
    p.add_argument(
        "--legacy-ee-pos",
        action="store_true",
        default=False,
        help="Phased out and rejected: kept only so an old command line gets an "
        "explanation instead of 'unrecognized arguments'. ee_pos now comes from "
        "the robot config's ee_body_name in the env frame; see the guard in "
        "BaseEnv.__init__.",
    )

    # -- LeRobot recording --
    lerobot_group = p.add_argument_group("lerobot", "LeRobot dataset recording")
    lerobot_group.add_argument(
        "--save-lerobot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record every env's rollout as a LeRobot dataset in the eval workers "
        "(default on; --no-save-lerobot to disable). Frames go to sharded writer "
        "processes that encode as they arrive, unlike --save-sensor-obs-video, "
        "which buffers each episode's frames in RAM.",
    )
    lerobot_group.add_argument(
        "--lerobot-dataset-name",
        default=None,
        help="Base dataset name for the merged LeRobot dataset (default: the run name).",
    )
    lerobot_group.add_argument(
        "--num-shards",
        type=int,
        default=8,
        help="Number of parallel LeRobot writer (encoder) subprocesses. "
        "Env i is routed to writer i // ceil(num_envs / num_shards); their datasets "
        "are stitched into one success / failure pair at the end (default: 8).",
    )
    lerobot_group.add_argument(
        "--video-codec",
        default="libx264",
        choices=["libx264", "libsvtav1"],
        help="RGB codec of the LeRobot videos (default: libx264; libsvtav1, datagen's, "
        "takes ~4x the native memory per open stream: 57 vs 14 MiB at 640x480).",
    )

    # -- Molmo2-specific args --
    molmo2_group = p.add_argument_group("molmo2", "Molmo2 HL policy arguments")
    molmo2_group.add_argument("--hl-model-path", default=None,
                   help="Molmo2 HL checkpoint path "
                        "(required for --policy molmo2_maniflow).")
    molmo2_group.add_argument(
        "--hl-prompt",
        default="",
        help="Manual override for the language goal fed to the Molmo2 HL model "
        "(--policy molmo2_maniflow). Empty (default) uses the per-episode "
        "prompt (parquet goal.hl_prompt when present, else english from "
        "pddl_goal); set e.g. 'stack 2 blocks' to force a fixed goal.",
    )
    molmo2_group.add_argument(
        "--oracle-comparison",
        action="store_true",
        default=False,
        help="Also tick a client-side comparison oracle and draw a "
        "model | oracle side-by-side HL overlay panel (forwards "
        "--oracle_comparison to evaluate.py). Requires "
        "--policy molmo2_maniflow and --save-hl-overlay-video.",
    )

    # -- First-batch dump --
    p.add_argument(
        "--save-first-batch-data",
        action="store_true",
        help="Dump the model-ready batch the policy server saw on step 0 to "
        "first_batch_data/ in each eval worker's output dir, in the same format "
        "training writes. --policy maniflow only.",
    )
    p.add_argument(
        "--save-diagnostic-plots",
        action="store_true",
        help="Save the diagnostics that need the live rollout in each eval "
        "worker's output dir: per-episode policy-input verification panels "
        "(raw depth -> model-input depth -> pointmap -> dataset depth, plus "
        "the agent_pos normalization bands), the HL conditioning staleness "
        "plot, the server-side LL policy-input trace, and (for "
        "--policy molmo2_maniflow) the server-side HL policy-input trace. The "
        "wandb coordinator uploads these as debug/policy_inputs_table, "
        "debug/ll_policy_inputs_table, debug/hl_policy_inputs_table, and "
        "(the HL staleness plot) under diagnostic_plots/. Everything here is "
        "capability-gated: a run with "
        "no policy server, no maniflow obs transform, or a server that "
        "advertises no state normalization bands prints why it skipped them "
        "and continues. Implies --save-first-batch-data and --save-lerobot. "
        "The per-step joint-position / velocity / EE / fingertip / camera "
        "comparison plots this flag used to produce no longer come from it at "
        "all -- only from --save-lerobot, via "
        "scripts/plot_diagnostics_from_eval_run.py. This flag additionally "
        "runs that script at the end of the pipeline, "
        "rendering every episode's control-diagnostic plots to disk under "
        "control_diagnostics_offline/ and first_frames_offline/ in each eval "
        "worker's output dir. With --wandb enabled, the coordinator "
        "separately renders + uploads one representative episode's copy of "
        "those plots under diagnostic_plots/ regardless of this flag.",
    )
    p.add_argument(
        "--save-sensor-obs-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --save_sensor_obs_video to evaluate.py ("
        "use --no-save-sensor-obs-video to disable).",
    )
    p.add_argument(
        "--save-hl-overlay-video",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass --save_hl_overlay_video to evaluate.py: record the HL overlay "
        "as a video stream in the LeRobot dataset (needs --save-lerobot and an HL; "
        "independent of --save-sensor-obs-video).",
    )
    p.add_argument(
        "--save-step-metrics",
        action="store_true",
        help="Pass --save_step_metrics to evaluate.py: write every control step's "
        "tracked stats (goal predicates, settle-gate block speeds, finger position, "
        "...) to step_metrics/env{idx}_episode_NNNN.csv in each eval worker's "
        "output dir. Costs a host copy of every stat per step; meant for debugging.",
    )

    # -- Episode termination --
    p.add_argument(
        "--episode-end",
        choices=["goal", "horizon", "ll_done"],
        default="goal",
        help="What ends an episode before the --episode-length-s / "
        "--expert-timeout-scale timeout (which applies in every mode): "
        "'goal' = the goal predicate (oracle runs: plan completion); "
        "'horizon' = nothing, run to the timeout; 'll_done' = the LL's play "
        "mode -1 (pause_on_attempt checkpoints only, --policy maniflow), "
        "then success_at_end says whether the goal still holds. Replay always "
        "uses 'horizon'.",
    )
    p.add_argument(
        "--collision",
        choices=["end", "penalize", "off"],
        default="end",
        help="evaluate.py --collision: 'end' fails and ends the episode on a "
        "collision; 'penalize' fails it but lets it run on, and also records "
        "the success ignoring collisions; 'off' does not score collisions.",
    )
    p.add_argument(
        "--goal-ori-yaw-tol-deg",
        type=float,
        default=40.0,
        help="evaluate.py --goal_ori_yaw_tol_deg: yaw tolerance (deg) of the PDDL "
        "goal's ori_of atoms when judging goal success; raise it to relax orientation.",
    )
    p.add_argument(
        "--post-place-lift",
        type=float,
        default=0.0,
        help="evaluate.py --post_place_lift: after a play-mode LL releases a block "
        "(-1 with the gripper open), raise the held arm this many meters before the "
        "next LL query (0.035 = vecgen's lift_traj); 0 disables.",
    )

    # -- High-level latency (update interval K + age M) --
    hl_latency_group = p.add_argument_group(
        "hl-latency", "Decouple HL conditioning from per-step LL inference"
    )
    hl_latency_group.add_argument(
        "--hl-update-interval-ms",
        type=float,
        default=0.0,
        help="K (ms): HL invocation interval; see 'HL latency' in docs/source/user_guide/workflows/evals/launch_eval.md. 0 = every step.",
    )
    hl_latency_group.add_argument(
        "--hl-update-interval-jitter-ms",
        type=float,
        default=0.0,
        help="Resampled U(-j, +j) ms added to each HL interval; see 'HL latency' in docs/source/user_guide/workflows/evals/launch_eval.md. 0 = fixed K.",
    )
    hl_latency_group.add_argument(
        "--estimated-hl-inference-time-ms",
        type=float,
        default=0.0,
        help="M (ms): assumed HL inference time; see 'HL latency' in docs/source/user_guide/workflows/evals/launch_eval.md. 0 = latest.",
    )
    hl_latency_group.add_argument(
        "--save-hl-latency-video",
        action="store_true",
        default=False,
        help="Write per-episode side-by-side latency MP4; requires --save-sensor-obs-video.",
    )

    # -- Sizing / queue / profile overrides --
    p.add_argument(
        "--queue-limit",
        type=int,
        default=None,
        help="Cap the queue at N episodes (default: unlimited in "
        "production, 10 under --smoke-test).",
    )
    p.add_argument(
        "--queue-log-level",
        default=None,
        help="serve_episode_queue.py --log-level (default: INFO; DEBUG under --smoke-test).",
    )
    p.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help="Max parallel sub-envs per eval GPU (default: 50; 2 under --smoke-test).",
    )
    p.add_argument(
        "--num-eval-gpus",
        type=int,
        default=None,
        help="Eval clients. The policy server runs on GPU 0 and client i on GPU i; with "
        "fewer visible GPUs the clients share them round-robin (client 1 with the server). "
        "Under --replay there is no server. Default: 1.",
    )
    p.add_argument(
        "--episode-length-s",
        type=float,
        default=None,
        help="Per-episode timeout in sim seconds (default: 30.0; 3.0 under --smoke-test).",
    )
    p.add_argument(
        "--expert-timeout-scale",
        type=float,
        default=0.0,
        help="Give each episode its own timeout, set to this multiple of that "
        "episode's expert demonstration (150%% of the expert episode = 1.5). "
        "The pipeline raises --episode-length-s to cover the queue's longest "
        "expert so the per-episode budget is what binds. Default 0 = off, one "
        "shared --episode-length-s for every episode.",
    )
    p.add_argument(
        "--replay-gripper-close-target",
        type=float,
        default=None,
        help="--env-cfg.replay-gripper-close-target passed to evaluate.py: "
        "override binary gripper close target (radians) for finger_joint.",
    )
    p.add_argument(
        "--eval-table-z-offset",
        type=float,
        default=None,
        help="--env-cfg.eval-table-z-offset passed to evaluate.py: extra world-frame "
        "Z offset (metres) for workspace_table and blocks after queue setup "
        "(e.g. -0.02 to lower the table by 2 cm).",
    )
    p.add_argument(
        "--depth-aug",
        default=None,
        metavar="PROFILE",
        help="Maniflow policies only: live depth augmentation on the client obs "
        "transform (ejbl). Pipeline installs opencv-contrib from HVLA requirements when set.",
    )
    p.add_argument(
        "--enable-crop",
        action="store_true",
        help="Maniflow policies only: client-side target_front_crop_rgb (training parity).",
    )
    p.add_argument(
        "--use-calibrated-cameras",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use fixed sim2real preset intrinsics (default); --no-use-calibrated-cameras uses per-episode queue K.",
    )
    p.add_argument(
        "--use-render-params",
        "--use_render_params",
        action="store_true",
        help="Replay each episode's recorded camera pose and look (HDRI, table "
        "material, lights, exposure, backdrop) from a rendered --dataset-path "
        "(evaluate.py --use_render_params). Block glyphs/colours are replayed by "
        "--replay-block-identity. Needs --no-use-calibrated-cameras; with --num-envs > 1 "
        "add --wave-resets so each wave shares one episode's scene-global lighting.",
    )
    p.add_argument(
        "--replay-block-identity",
        "--replay_block_identity",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="On a glyph/colour-rendered --dataset-path, paint each episode's recorded "
        "block letters and colours and prompt the policies with the recorded goal "
        "(evaluate.py --replay_block_identity). Success is graded on the source block "
        "names either way. On by default; --no-replay-block-identity keeps the "
        "authored look and the source-name prompts.",
    )
    p.add_argument(
        "--wave-resets",
        "--wave_resets",
        action="store_true",
        help="Start episodes in waves (evaluate.py --wave_resets): a finished env holds "
        "its reset pose until every active env has finished, so each wave shares the "
        "dataset's recorded scene lighting. Needs a rendered --dataset-path and "
        "--num-envs > 1; not with --replay or --use-calibrated-cameras.",
    )
    p.add_argument(
        "--out-subdir",
        default=None,
        help="Output sub-dir under runs/<task>/<policy>/ (default: "
        "evals; queue_test under --smoke-test).",
    )
    p.add_argument(
        "--wandb-project", default="toyblocks-eval", help="Wandb project for serve + eval runs."
    )
    p.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="Enable wandb logging (disabled by default). Also extracts each "
        "queued episode's source-dataset expert MP4 for the "
        "videos/episodes_with_expert_table ``source_episode`` column.",
    )

    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the pipeline invocation (env VAR=... bash run_eval_pipeline.sh) without running it.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="With --dry-run: write the invocation here instead of stdout.",
    )
    args = p.parse_args()

    if args.save_diagnostic_plots:
        args.save_first_batch_data = True
        args.save_lerobot = True

    # Fail here rather than after rendering a yaml whose worker would reject it
    # anyway (see the guard in BaseEnv.__init__).
    if args.legacy_ee_pos:
        raise SystemExit(
            "--legacy-ee-pos is phased out and rejected by BaseEnv.__init__. ee_pos "
            "now comes from the robot config's ee_body_name in the env frame, "
            "matching what the LeRobot recorder writes."
        )

    # -- Resolve policy defaults --
    if args.replay:
        pol = _REPLAY_DEFAULTS
        for flag, value in (
            ("--model-path", args.model_path),
            ("--hl-model-path", args.hl_model_path),
        ):
            if value is not None:
                raise SystemExit(f"{flag} has no effect under --replay (no policy server runs)")
        if args.oracle_comparison:
            raise SystemExit("--oracle-comparison needs a learned HL; not applicable to --replay")
        if args.replicate_noise:
            raise SystemExit("--replicate-noise is a maniflow server option; not applicable to --replay")
        if args.num_envs is None:
            args.num_envs = 1
        elif args.num_envs != 1:
            raise SystemExit(
                f"--replay requires --num-envs 1 (got {args.num_envs}): a recorded PD gain "
                "switch settles with raw global sim substeps, which would also advance "
                "every other env. Use --num-eval-gpus to run episodes in parallel instead."
            )
        if (args.init_after_hl_action or "").strip():
            raise SystemExit(
                "--init-after-hl-action is not supported with --replay: replay starts "
                "each episode at frame 0 and does not honor queue init_frame_index "
                "cuts yet. Drop --init-after-hl-action or run without --replay."
            )
    else:
        pol = _POLICY_DEFAULTS[args.policy]
    if args.replay_clip_stats:
        if not args.replay:
            raise SystemExit("--replay-clip-stats only applies to --replay")
        if not Path(args.replay_clip_stats).is_file():
            raise SystemExit(f"--replay-clip-stats: no such file {args.replay_clip_stats!r}")

    if not args.replay and args.policy == "maniflow" and args.model_path is None:
        raise SystemExit("--model-path is required for --policy maniflow")

    if args.save_hl_overlay_video and not args.save_lerobot:
        raise SystemExit("--save-hl-overlay-video records into the LeRobot dataset: needs --save-lerobot")
    if args.oracle_comparison and not args.save_hl_overlay_video:
        raise SystemExit(
            "--oracle-comparison only renders into the HL overlay video: needs --save-hl-overlay-video"
        )

    if args.episode_end == "ll_done" and (args.replay or args.policy != "maniflow"):
        raise SystemExit(
            "--episode-end ll_done needs a play-mode ManiFlow LL with no learned HL "
            "(--policy maniflow, not --replay)"
        )

    if not args.replay and args.policy == "molmo2_maniflow":
        if args.model_path is None:
            raise SystemExit("--model-path is required for --policy molmo2_maniflow")
        if args.hl_model_path is None:
            raise SystemExit("--hl-model-path is required for --policy molmo2_maniflow")
    elif args.oracle_comparison:
        raise SystemExit(
            "--oracle-comparison compares the learned server-side HL against the "
            "client-side oracle, so it requires --policy molmo2_maniflow"
        )

    if args.replicate_noise and args.policy != "maniflow":
        raise SystemExit("--replicate-noise requires --policy maniflow")

    if args.go_home and not (args.async_hl or args.ll_oracle):
        raise SystemExit("--go-home is only allowed with --async-hl or --ll-oracle")
    if args.async_hl_call_scale is not None and (not args.async_hl or args.ll_oracle):
        raise SystemExit("--async-hl-call-scale needs --async-hl (use --ll-oracle-hl-call-scale with --ll-oracle)")
    if args.ll_oracle and args.policy != "molmo2_maniflow":
        raise SystemExit("--ll-oracle needs --policy molmo2_maniflow")

    policy_name = args.policy_name or pol["policy_name"]
    obs_transform = args.obs_transform or pol["obs_transform"]
    if (
        args.center_crop_ratio is None
        and not args.replay
        and obs_transform in ("maniflow", "maniflow_blocks_table")
    ):
        args.center_crop_ratio = 0.95
    if args.center_crop_ratio is not None:
        if args.replay:
            raise SystemExit(
                "--center-crop-ratio only applies to policy evals with a maniflow obs "
                "transform (got --replay)"
            )
        _require_maniflow_obs_transform(obs_transform, "--center-crop-ratio")
        if not 0.0 < args.center_crop_ratio <= 1.0:
            raise SystemExit(f"--center-crop-ratio must be in (0, 1], got {args.center_crop_ratio}")
        if args.center_crop_ratio == 1.0:
            args.center_crop_ratio = None
    if args.depth_aug is not None:
        _require_maniflow_obs_transform(obs_transform, "--depth-aug")
        if str(args.depth_aug).strip().lower() not in ("ejbl",):
            raise SystemExit(f"--depth-aug must be 'ejbl' when set, got {args.depth_aug!r}")
        from toyblocks.serving.maniflow_serving_utils import require_depth_aug_supported

        require_depth_aug_supported(str(args.depth_aug).strip().lower())
    if args.enable_crop:
        _require_maniflow_obs_transform(obs_transform, "--enable-crop")
    robot_id = args.robot_id or pol["robot_id"]
    scene_preset = args.scene_preset or pol["scene_preset"]
    action_mode = args.action_mode or pol["action_mode"]
    action_horizon = (
        args.action_horizon if args.action_horizon is not None else pol["action_horizon"]
    )
    if args.ll_oracle:
        action_horizon = 0  # every Molmo2 query reaches the server (no chunk broker)

    # Maniflow model paths: derive defaults from --model-path and --dataset-path
    model_path = args.model_path or pol["model_path"]
    hl_model_path = args.hl_model_path or pol.get("hl_model_path", "")
    if not args.replay and args.policy in ("maniflow", "molmo2_maniflow") and model_path:
        # .safetensors files live in checkpoints/; config.yaml is in the run
        # root one level up.  Mirror the server's two-level search.
        mp = Path(model_path)
        if mp.suffix == ".safetensors":
            config_path = args.config_path or str(mp.parent.parent / "config.yaml")
        else:
            config_path = args.config_path or str(mp.parent / "config.yaml")
        # Empty → run_eval_pipeline.sh defaults it under $H_VLA_PATH (the job's checkout).
        train_config_path = args.train_config_path or ""
        # Only forward an explicit override.  The server resolves the bands from
        # the checkpoint's own config (framework.train_data_stats, else the
        # stats.json beside it), and a path derived here would take precedence
        # over the normalizer the run actually trained with.  Using the
        # *dataset's* meta/stats.json is likewise wrong whenever the eval
        # dataset differs from the training mix.
        stats_path = args.stats_path or ""
    else:
        config_path = args.config_path or pol["config_path"]
        train_config_path = args.train_config_path or pol["train_config_path"]
        stats_path = args.stats_path or pol["stats_path"]

    # -- Resolve profile --
    profile = _SMOKE_PROFILE if args.smoke_test else _PROD_PROFILE

    queue_limit_arg = (
        profile["queue_limit_arg"] if args.queue_limit is None else f"--limit {args.queue_limit}"
    )

    task_slug = args.task_name or task_slug_from_dataset_path(args.dataset_path)
    if not task_slug.isidentifier():
        raise SystemExit(
            f"task slug {task_slug!r} is not a valid Python identifier "
            "(letters, digits, underscore; cannot start with a digit). "
            "Pass --task-name to override."
        )

    if args.replay_block_identity is None:
        args.replay_block_identity = True
    # Provenance for the run dir: the checkout this eval runs from.
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False).stdout.strip()
    out_subdir = args.out_subdir if args.out_subdir is not None else profile["out_subdir"]
    queue_log_level = (
        args.queue_log_level if args.queue_log_level is not None else profile["queue_log_level"]
    )
    num_envs = args.num_envs if args.num_envs is not None else profile["num_envs"]
    if args.use_render_params and args.use_calibrated_cameras:
        raise SystemExit(
            "--use-render-params needs --no-use-calibrated-cameras: calibrated cameras "
            "turn render params off in evaluate.py"
        )
    if args.wave_resets and not args.use_render_params:
        raise SystemExit(
            "--wave-resets only shares recorded scene lighting across a wave, so it "
            "needs --use-render-params (evaluate.py rejects it otherwise)"
        )
    if args.wave_resets:
        if args.replay:
            raise SystemExit("--wave-resets needs --num-envs > 1, which --replay does not allow")
        if num_envs <= 1:
            raise SystemExit(f"--wave-resets needs --num-envs > 1 (got {num_envs})")
        if args.use_calibrated_cameras:
            raise SystemExit(
                "--wave-resets shares the dataset's recorded renders across a wave, "
                "which --use-calibrated-cameras disables"
            )
    episode_length_s = (
        args.episode_length_s if args.episode_length_s is not None else profile["episode_length_s"]
    )
    num_eval_gpus = (
        args.num_eval_gpus if args.num_eval_gpus is not None else profile["num_eval_gpus"]
    )
    if args.policy == "molmo2_maniflow" and num_eval_gpus > 1:
        raise SystemExit(
            "--policy molmo2_maniflow requires num_eval_gpus=1: "
            "HLLLInferAdapter.exclusive allows only one eval client connection "
            "to the HL-LL server."
        )
    if args.policy == "molmo2_maniflow" and num_envs > 128:
        raise SystemExit(
            "--policy molmo2_maniflow requires num_envs <= 128: "
            "HLLLInferAdapter pre-registers 128 HL-latency slots "
            "(_HL_SERVER_MAX_ENVS)."
        )
    # Every policy server batches envs: the '--hl model' HLLLInferAdapter runs
    # Molmo2 and the LL once per request over all envs, so molmo2_maniflow runs
    # the client batched (with the shared chunk boundary) like maniflow.
    unbatched = False
    launch_command = " ".join(shlex.quote(a) for a in sys.argv)
    # Replay forces episode_end=horizon worker-side (evaluate.py) so every
    # recording plays to its last frame; pass the flag explicitly for clarity.
    # (Still carried in the TERMINATE_ON_SUCCESS_ARG pipeline variable, as is
    # --collision.)
    episode_end = "horizon" if args.replay else args.episode_end
    terminate_on_success_arg = (
        f"--episode_end {episode_end} --collision {args.collision}"
        f" --goal_ori_yaw_tol_deg {args.goal_ori_yaw_tol_deg}"
    )
    if args.post_place_lift > 0:
        terminate_on_success_arg += f" --post_place_lift {args.post_place_lift}"

    # Client-side HL source forwarded to evaluate.py. The oracle is the default
    # for the batchable maniflow LL path; it doesn't apply to molmo2_maniflow
    # (learned server-side HL model) or replay
    # (no policy to condition).
    use_oracle = bool(args.use_oracle) and args.policy == "maniflow" and not args.replay
    hl_flag = "--use_oracle" if use_oracle else ""
    # The learned server-side HL: tells evaluate.py to pull the server's
    # hl_model_* aux back into the obs so the HL overlay video can be drawn.
    hl_from_model = args.policy == "molmo2_maniflow" and not args.replay

    eval_params = {
        "policy": "replay" if args.replay else args.policy,
        "replay": args.replay,
        "commit": head,
        "dataset_path": args.dataset_path,
        "task_slug": task_slug,
        "policy_name": policy_name,
        "problem_id": args.problem_id,
        "obs_transform": obs_transform,
        "robot_id": robot_id,
        "scene_preset": scene_preset,
        "action_mode": action_mode,
        "action_horizon": action_horizon,
        "use_oracle": use_oracle,
        "unbatched": unbatched,
        "hl_from_model": hl_from_model,
        "oracle_comparison": args.oracle_comparison,
        "replicate_noise": args.replicate_noise,
        "legacy_ee_pos": args.legacy_ee_pos,
        "hl_update_interval_ms": args.hl_update_interval_ms,
        "hl_update_interval_jitter_ms": args.hl_update_interval_jitter_ms,
        "estimated_hl_inference_time_ms": args.estimated_hl_inference_time_ms,
        "save_hl_latency_video": args.save_hl_latency_video,
        "model_path": model_path,
        "hl_model_path": hl_model_path,
        "hl_prompt": args.hl_prompt,
        "config_path": config_path,
        "train_config_path": train_config_path,
        "stats_path": stats_path,
        "out_subdir": out_subdir,
        "queue_limit_arg": queue_limit_arg,
        "queue_log_level": queue_log_level,
        "episode_indices": args.episode_indices,
        "init_after_hl_action": args.init_after_hl_action,
        "usd_library": args.usd_library,
        "num_envs": num_envs,
        "episode_length_s": episode_length_s,
        "expert_timeout_scale": args.expert_timeout_scale,
        "replay_clip_stats": args.replay_clip_stats,
        "center_crop_ratio": args.center_crop_ratio,
        "replay_gripper_close_target": args.replay_gripper_close_target,
        "depth_aug": args.depth_aug,
        "enable_crop": args.enable_crop,
        "use_calibrated_cameras": args.use_calibrated_cameras,
        "use_render_params": args.use_render_params,
        "replay_block_identity": args.replay_block_identity,
        "wave_resets": args.wave_resets,
        "eval_table_z_offset": args.eval_table_z_offset,
        "num_eval_gpus": num_eval_gpus,
        "wandb_project": args.wandb_project,
        "save_expert_video": args.wandb,
        "save_first_batch_data": args.save_first_batch_data,
        "save_sensor_obs_video": args.save_sensor_obs_video,
        "save_hl_overlay_video": args.save_hl_overlay_video,
        "episode_end": episode_end,
        "collision": args.collision,
        "goal_ori_yaw_tol_deg": args.goal_ori_yaw_tol_deg,
        "post_place_lift": args.post_place_lift,
        "deterministic_cuda": args.deterministic_cuda,
        "policy_seed": args.policy_seed,
        "allow_duplicate_episodes": args.allow_duplicate_episodes,
        "save_lerobot": args.save_lerobot,
        "lerobot_dataset_name": args.lerobot_dataset_name,
        "num_shards": args.num_shards,
        "video_codec": args.video_codec,
        "smoke_test": args.smoke_test,
        "launch_argv": sys.argv[1:],
        "launch_command": launch_command,
    }
    eval_params_json = json.dumps(eval_params, separators=(",", ":"))

    # -- Run the pipeline --
    pipeline = REPO_ROOT / "scripts" / "eval" / "run_eval_pipeline.sh"
    # Every value must be a str: this is spliced into os.environ below and
    # printed as an `env VAR=...` prefix under --dry-run.
    eval_env = {
        "ENABLE_WANDB": "1" if args.wandb else "",
        "SAVE_EXPERT_VIDEO": "1" if args.wandb else "",
        "WANDB_PROJECT": args.wandb_project,
        "POLICY_NAME": policy_name,
        "MODEL_PATH": model_path,
        "HL_MODEL_PATH": hl_model_path,
        "HL_PROMPT": args.hl_prompt,
        "CONFIG_PATH": config_path,
        "TRAIN_CONFIG_PATH": train_config_path,
        "STATS_PATH": stats_path,
        "ACTION_MODE": action_mode,
        "POLICY_SEED": str(args.policy_seed),
        "DETERMINISTIC_CUDA": "1" if args.deterministic_cuda else "",
        "REPLICATE_NOISE": "1" if args.replicate_noise else "",
        "ROBOT_ID": robot_id,
        "SCENE_PRESET": scene_preset,
        "TASK_SLUG": task_slug,
        "PROBLEM_ID": args.problem_id,
        "OBS_TRANSFORM": obs_transform,
        "LEGACY_EE_POS": "1" if args.legacy_ee_pos else "",
        "DATASET_PATH": args.dataset_path,
        "OUT_SUBDIR": out_subdir,
        "QUEUE_LIMIT_ARG": queue_limit_arg,
        "QUEUE_LOG_LEVEL": queue_log_level,
        "EPISODE_INDICES": args.episode_indices,
        "INIT_AFTER_HL_ACTION": args.init_after_hl_action,
        "USD_LIBRARY": args.usd_library,
        "ALLOW_DUPLICATE_EPISODES": "1" if args.allow_duplicate_episodes else "",
        "ACTION_HORIZON": str(action_horizon),
        "HL_FLAG": hl_flag,
        "UNBATCHED": "1" if unbatched else "",
        "HL_FROM_MODEL": "1" if hl_from_model else "",
        "ORACLE_COMPARISON": "1" if args.oracle_comparison else "",
        "ASYNC_HL": "1" if args.async_hl and not args.ll_oracle else "",
        "ASYNC_HL_CALL_SCALE": "" if args.async_hl_call_scale is None else str(args.async_hl_call_scale),
        "GO_HOME": "1" if args.go_home else "",
        "LL_ORACLE": str(args.ll_oracle_hl_call_scale) if args.ll_oracle else "",
        "HL_NO_PREV_ACTION": "1" if args.hl_no_prev_action else "",
        "HL_MAX_NEW_TOKENS": "" if args.hl_max_new_tokens is None else str(args.hl_max_new_tokens),
        "HL_UPDATE_INTERVAL_MS": str(args.hl_update_interval_ms),
        "HL_UPDATE_INTERVAL_JITTER_MS": str(args.hl_update_interval_jitter_ms),
        "ESTIMATED_HL_INFERENCE_TIME_MS": str(args.estimated_hl_inference_time_ms),
        "SAVE_HL_LATENCY_VIDEO": "1" if args.save_hl_latency_video else "",
        "NUM_EVAL_GPUS": str(num_eval_gpus),
        "MAX_ENVS_PER_GPU": str(num_envs),
        "EPISODE_LENGTH_S": str(episode_length_s),
        "EXPERT_TIMEOUT_SCALE": str(args.expert_timeout_scale),
        "REPLAY_CLIP_STATS": args.replay_clip_stats,
        "CENTER_CROP_RATIO": (
            "" if args.center_crop_ratio is None else str(args.center_crop_ratio)
        ),
        "REPLAY_GRIPPER_CLOSE_TARGET": (
            str(args.replay_gripper_close_target)
            if args.replay_gripper_close_target is not None
            else ""
        ),
        "DEPTH_AUG": "" if args.depth_aug is None else str(args.depth_aug).strip().lower(),
        "ENABLE_CROP": "1" if args.enable_crop else "",
        "USE_CALIBRATED_CAMERAS": "1" if args.use_calibrated_cameras else "",
        "USE_RENDER_PARAMS": "1" if args.use_render_params else "",
        "REPLAY_BLOCK_IDENTITY": "1" if args.replay_block_identity else "",
        "WAVE_RESETS": "1" if args.wave_resets else "",
        "EVAL_TABLE_Z_OFFSET": (
            str(args.eval_table_z_offset) if args.eval_table_z_offset is not None else ""
        ),
        "FIRST_BATCH_ARG": (
            "--save_first_batch_data" if args.save_first_batch_data else ""
        ),
        "DIAGNOSTIC_PLOTS_ARG": (
            "--save_diagnostic_plots" if args.save_diagnostic_plots else ""
        ),
        "SENSOR_OBS_VIDEO_ARG": (
            "--save_sensor_obs_video" if args.save_sensor_obs_video else ""
        ),
        "HL_OVERLAY_VIDEO_ARG": (
            "--save_hl_overlay_video" if args.save_hl_overlay_video else ""
        ),
        "STEP_METRICS_ARG": "--save_step_metrics" if args.save_step_metrics else "",
        "TERMINATE_ON_SUCCESS_ARG": terminate_on_success_arg,
        "SAVE_LEROBOT": "1" if args.save_lerobot else "",
        "LEROBOT_DATASET_NAME": args.lerobot_dataset_name or "",
        "LEROBOT_NUM_SHARDS": str(args.num_shards),
        "LEROBOT_VIDEO_CODEC": args.video_codec,
        "EVAL_PARAMS": eval_params_json,
        "LAUNCH_COMMAND": launch_command,
    }
    if args.dry_run:
        env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in eval_env.items())
        invocation = f"env {env_prefix} bash {shlex.quote(str(pipeline))}\n"
        if args.output is None:
            sys.stdout.write(invocation)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(invocation)
            print(f"[dry-run] Wrote {args.output}", file=sys.stderr)
        raise SystemExit(0)
    rc = subprocess.run(["bash", str(pipeline)], env={**os.environ, **eval_env}).returncode
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
