"""Unified evaluation script for ToyBlocks Isaac Lab environments.

Supports several action sources, selected by CLI flags:

1. **Policy server** (``--policy_host`` / ``--policy_port``):
   Connect to a WebSocket + msgpack policy server (``toyblocks.serving``), e.g.
   ``python -m toyblocks.serving.serve_maniflow_policy`` serving ManiFlow (``--hl oracle``)
   or Molmo2 HL + ManiFlow LL (``--hl model``, with ``--hl_from_model`` here). Supports
   action chunking and the registered obs transforms (``maniflow``,
   ``maniflow_blocks_table``).

2. **LL oracle** (``--ll_oracle``, with a Molmo2 HL server via ``--policy_host`` and
   ``--hl_from_model``): Molmo2 picks each command and the vecgen skills (cuRobo v2) execute
   it in place of the served low-level policy. Needs ``--action_horizon 0`` and
   ``--control_mode arm_joint_gripper_binary``.

3. **Dataset replay** (``--replay``, with ``--queue_url`` + ``--dataset_path``):
   Play back each queue episode's recorded actions (7 joint targets + gripper
   + ``action.pd_mode``) through the same eval loop a policy uses. The queue
   setup restores the recorded frame-0 scene and joint velocity, gain switches
   settle through the PD controller, and per-episode divergence from the
   recording lands in ``eval_episodes.json`` / ``eval_results.json`` as
   ``replay_*`` metrics. Requires ``--num_envs 1``.

4. **Random actions** (``--random_actions``):
   Sample each action component uniformly from ``[-1, 1]`` every step (same
   convention as ``scripts/random_actions.py``).

5. **Zero actions** (none of the above):
   Send zero actions every step -- useful for sanity-checking the sim.

``scripts/launch_eval.py`` is the usual entry point: it starts the policy server, the
episode queue and the eval workers (see docs/source/user_guide/workflows/evaluations.md).

Example usage::

    # Policy server (ManiFlow LL, oracle HL), started separately on :8000
    python scripts/evaluate.py --task <slug>-v0 --policy_host localhost --policy_port 8000 \\
        --queue_url http://localhost:8001/next --dataset_path /path/to/lerobot_root \\
        --obs_mode rgb+state --obs_transform maniflow --control_mode arm_joint_gripper_binary

    # Override env config (tyro nested args), e.g. track collision forces:
    # python scripts/evaluate.py --task <slug>-v0 --env-cfg.track-collisions ...

    # Random-action rollout (no policy)
    python scripts/evaluate.py --task ToyBlocks-Lift-v0 --random_actions --num_envs 4 --num_eval_steps 200

    # Zero-action sanity check
    python scripts/evaluate.py --task ToyBlocks-Lift-v0 --num_episodes 3

    # Dataset replay via the launcher (starts the queue, no policy server)
    python scripts/launch_eval.py --replay --dataset-path /path/to/lerobot_root \\
        --episode-indices 0 --save-diagnostic-plots

    # Dataset replay directly (queue already running on :8001)
    python scripts/evaluate.py --task <slug>-v0 --replay \\
        --queue_url http://localhost:8001/next --dataset_path /path/to/lerobot_root \\
        --control_mode arm_joint_gripper_binary --obs_mode rgb+state --num_envs 1
"""

import os
import sys

os.environ["TORCHDYNAMO_INLINE_INBUILT_NN_MODULES"] = "1"

# ---------------------------------------------------------------------------
# Isaac Lab bootstrap — SimulationApp MUST be instantiated before any
# isaaclab / omni imports.
# ---------------------------------------------------------------------------

if "--headless" not in sys.argv and "--no-headless" not in sys.argv:
    sys.argv.append("--headless")


def _has_rgb_obs():
    for i, arg in enumerate(sys.argv):
        if arg in ("--obs_mode", "--obs-mode") and i + 1 < len(sys.argv):
            return "rgb" in sys.argv[i + 1]
        if arg.startswith("--obs_mode=") or arg.startswith("--obs-mode="):
            return "rgb" in arg.split("=", 1)[1]
    return False


if _has_rgb_obs():
    if "--enable_cameras" not in sys.argv and "--enable-cameras" not in sys.argv:
        sys.argv.append("--enable_cameras")


def _argv_option(*names: str) -> tuple[int | None, str | None]:
    """``(index, value)`` of the first ``--name value`` / ``--name=value`` in sys.argv."""
    for i, arg in enumerate(sys.argv):
        if arg in names and i + 1 < len(sys.argv):
            return i, sys.argv[i + 1]
        for name in names:
            if arg.startswith(name + "="):
                return i, arg.split("=", 1)[1]
    return None, None


def _merge_render_replay_kit_args() -> None:
    # A rendered dataset's carb_settings (e.g. the RealTimePathTracing render mode)
    # only take effect as launch-time kit args, so --use_render_params has to add
    # them here; merged into a user --kit_args rather than passed as a second flag.
    # Both are opt-in: --use_render_params, and --no-use-calibrated-cameras (calibrated
    # cameras, on by default, turn render params off).
    argv = set(sys.argv)
    if not (
        {"--use-render-params", "--use_render_params"} & argv
        and {"--no-use-calibrated-cameras", "--no_use_calibrated_cameras"} & argv
    ):
        return
    _, dataset_path = _argv_option("--dataset_path", "--dataset-path")
    if not dataset_path:
        return
    from toyblocks.eval.render_replay import RecordedRenderReplay, merge_kit_args

    replay = RecordedRenderReplay.from_dataset(dataset_path)
    extra = replay.kit_args() if replay is not None else ""
    if not extra:
        return
    idx, existing = _argv_option("--kit_args", "--kit-args")
    merged = merge_kit_args(existing or "", extra)
    if idx is None:
        sys.argv += ["--kit_args", merged]
    elif "=" in sys.argv[idx]:
        sys.argv[idx] = sys.argv[idx].split("=", 1)[0] + "=" + merged
    else:
        sys.argv[idx + 1] = merged
    print(f"[INFO]: render replay kit_args (launch-time carb settings): {merged}", flush=True)


_merge_render_replay_kit_args()

# Import tyro and its lazy dependency typeguard NOW, before launch_app(). 
import tyro
try:
    import typeguard  # noqa: F401  (force the lazy dependency to load from the conda env)
except ImportError:
    pass

from toyblocks.cli.isaac_app import launch_app
simulation_app = launch_app()

# Now safe to import everything that touches isaaclab / omni.
import tyro

import isaaclab_tasks  # noqa: F401
import toyblocks.tasks  # noqa: F401
import isaaclab.sim as sim_utils  # noqa: F401

from toyblocks.eval.args import Args
from toyblocks.cli.base_args import parse_cli
from toyblocks.eval import evaluate


def main() -> None:
    args = parse_cli(Args)

    # Snap num_envs to a value the task's generated env accepts. Autogen block envs
    # bake a fixed number of randomization slots at codegen time because block size
    # is a spawn-time property Isaac cannot change at runtime: a queue-driven eval
    # sizes each slot's env count to its queued workload (env_cfg.env_slots),
    # otherwise env i spawns at slot i % N and num_envs has to be a multiple of N.
    # Must run before video_env_ids is derived from num_envs below.
    from toyblocks.utils.eval_helpers import reconcile_num_envs_with_task
    reconcile_num_envs_with_task(args)

    if args.save_video_all_envs:
        args.video_env_ids = tuple(range(args.num_envs))
    if "rgb" in args.obs_mode:
        args.enable_cameras = True
    # Default the eval robot/scene to the dataset's recorded provenance (warns
    # loudly on override mismatch; fails loudly when provenance is missing and
    # no --env-cfg.robot-id/scene-preset-id override was given). Requires
    # --dataset_path; for queue evals pass the same dataset the queue serves.
    if args.dataset_path:
        from toyblocks.utils.dataset_provenance import reconcile_env_cfg_with_dataset
        reconcile_env_cfg_with_dataset(args.env_cfg, args.dataset_path)

    evaluate(args)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        # Kit's shutdown turns an uncaught exception's interpreter exit into status 0, which
        # would leave a crashed eval worker looking green to run_eval_pipeline.sh.
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    simulation_app.close()
