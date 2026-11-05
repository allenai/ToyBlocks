"""Standalone scene-loading diagnostic for queue eval (Approach A).

Launches Isaac, builds the generated task for a dataset, applies each requested
episode's recorded setup exactly the way the live queue eval does (via
``apply_setup`` inside ``_reset_idx``), then runs the shared
``toyblocks.eval.scene_verification`` checks: structured scene diff
(table height, blocks present, block/table poses, block sizes, camera
intrinsics and pose) plus first-frame image comparison (MAE / MSE / SSIM +
side-by-side PNGs). Writes a report JSON and the comparison PNGs, and exits
non-zero on any hard mismatch (fail loud).

Every hard check here is against the dataset's recorded ground truth, so there
is nothing to re-baseline -- with one narrow exception noted below.

**The policy input is verified too** (``--verify-policy-input``, on by default,
needs ``depth`` in ``--obs-mode``). The reset observation is fed through the
ManiFlow obs transform at its widest layout and the result is checked against the
dataset: the arm block of ``agent_pos`` equals the episode's recorded joints, the
emitted ``camera_intrinsics`` equal the recorded per-camera K rescaled to the
model's 224x224 grid, depth survives the metre cutoff, ``ee_pos`` is env-local
and RGB is a real frame. It lives here rather than in its own script because this
is the only place a live observation, the dataset's ground truth and the
``target_env`` matching this episode's ``cube_scale`` exist at once. The one
committed golden is the *observation contract* -- keys, dtypes, ranks, native
resolution, emitted wire layout -- keyed by scene preset and carrying no
episode-specific numbers, which is what keeps it stable across datagen runs. See
``toyblocks.eval.policy_input_verification``.

**First-frame SSIM carries a hard floor** (``--min-first-frame-ssim``, default
0.90) per verified camera, alongside the reported MAE / MSE / SSIM family. What
the floor catches is a wrong scene, a wrong camera or a setup that never got
applied -- not calibration drift, which is the intrinsics/pose job above. It
sits far below the observed spread (worst seen: front 0.9864, gripper 0.9881),
so lighting and driver variation cannot reach it and no environment bump
demands a re-baseline. The overlay PNGs remain what makes a breach actionable.

One deliberate exclusion:

- **Dynamics are not checked here.** A seeded free-space rollout needs no
  dataset, episode or profile, so it lives in
  ``tests/env_randomization/test_rollout_determinism.py`` where its identity is
  just ``(task, seed, action_scale, num_steps)`` and cannot go stale because a
  datagen run's success pattern shifted.

Every episode is verified inside ONE Isaac session: the env already spans every
cube-scale spawn slot, so a second process would only pay another cold start to
build the same scene. Each episode gets its own reset (a fresh queue client and
a fresh ``apply_setup``) and its own report under ``<output-dir>/ep<N>/``, and
the exit code is non-zero if any of them hard-fails -- one bad episode must not
hide the rest.

This mirrors the ``generate_task_from_dataset.py`` bootstrap (``launch_app()``
before isaaclab imports). The task package must already be generated:

    python scripts/generate_task_from_dataset.py --dataset-path <DS> --headless
    python scripts/eval/verify_dataset_loading.py \
        --dataset-path <DS> --task <slug>-v0 --episode-indices 0 4 \
        --obs-mode rgb+state --output-dir runs/verify --headless --enable_cameras

``tests/eval/scene_load.sh`` drives this one subprocess per dataset; run
standalone it is meant for a GPU box with the weka datasets mounted.

Every comparison also reports a *margin* (fraction of its allowance consumed),
and the console prints the tightest few -- that is what flags a field about to
start flaking before it does.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

os.environ.setdefault("TORCHDYNAMO_INLINE_INBUILT_NN_MODULES", "1")

if "--headless" not in sys.argv and "--no-headless" not in sys.argv:
    sys.argv.append("--headless")
if "--enable_cameras" not in sys.argv and "--enable-cameras" not in sys.argv:
    sys.argv.append("--enable_cameras")

from toyblocks.cli.isaac_app import launch_app

simulation_app = launch_app()

import importlib.util
import json
from pathlib import Path

import numpy as np
import tyro

import isaaclab_tasks  # noqa: F401
import toyblocks.tasks  # noqa: F401
import gymnasium as gym

from toyblocks.cli.base_args import BaseArgs, add_env_cfg_args
from toyblocks.utils.episode_dataset import (
    CUBE_SCALE_DECIMALS,
    load_setups,
)
from toyblocks.eval.policy_input_verification import append_policy_input_diffs
from toyblocks.eval.scene_verification import (
    FieldDiff,
    SceneMismatchReport,
    compare_first_frames,
    compare_scene,
    extract_dataset_scene_spec,
    format_diff_line,
    read_sim_scene_state,
)

# Cameras the first-frame comparison covers. left_camera is excluded because its
# first-frame SSIM varied 0.866-0.926 across three repeats of the same scene,
# which is noise rather than a reading of the scene; the two below are stable
# enough for the overlay to be worth looking at. Every camera's intrinsics and
# pose are checked by the scene tier regardless.
VERIFIED_CAMERAS = ("front_camera", "gripper_camera")


def _cube_scale_slots_for_task(task_id: str) -> list[float]:
    """Read the generated task package's baked ``cube_scale_slots``.

    Raises rather than defaulting to an empty list: ``write_env_package``
    always emits ``non_sim_state_randomizations.json``, so its absence means the
    package is broken or not an autogen package at all. Returning ``[]`` would
    silently fall back to verifying sub-env 0, which for a task with per-episode
    cube-scale slots is a different block size than the episode was recorded at.
    """
    spec = gym.spec(task_id)
    module_path = str(spec.entry_point).split(":")[0]  # e.g. pkg.<slug>.env
    found = importlib.util.find_spec(module_path)
    if found is None or found.origin is None:
        raise ImportError(
            f"cannot locate the module {module_path!r} backing task {task_id!r}, so "
            "its cube-scale spawn slots are unreadable. Regenerate the task package "
            "with scripts/generate_task_from_dataset.py."
        )
    rand_json = Path(found.origin).parent / "non_sim_state_randomizations.json"
    if not rand_json.is_file():
        raise FileNotFoundError(
            f"{rand_json} is missing. write_env_package always writes it, so this is "
            f"a broken package for task {task_id!r}; regenerate it with "
            "scripts/generate_task_from_dataset.py rather than verifying an "
            "arbitrary sub-env."
        )
    with open(rand_json) as f:
        return list(json.load(f).get("cube_scale_slots") or [])


def _target_env_for_scale(slots: list[float], cube_scale: float | None) -> int:
    """Index of the env spawned at ``cube_scale`` (raises if no slot matches)."""
    key = round(float(cube_scale), CUBE_SCALE_DECIMALS)
    rounded = [round(float(s), CUBE_SCALE_DECIMALS) for s in slots]
    if key not in rounded:
        raise ValueError(
            f"episode cube_scale={cube_scale} (->{key}) is not among the task's "
            f"spawned cube-scale slots {rounded}. The task was generated from a "
            "different episode set than this episode; regenerate it including this "
            "episode (so its size has a spawn slot)."
        )
    return rounded.index(key)


@add_env_cfg_args
@dataclass
class Args(BaseArgs):
    """Diagnostic args. ``--task`` selects the generated env package."""

    dataset_path: str = ""
    """LeRobot dataset root the task was generated from."""
    task: str = ""
    """Gym id of the generated task (e.g. ``apr15_pnp-v0``)."""
    episode_indices: tuple[int, ...] = (0,)
    """Episodes to load and verify, all inside one Isaac session."""
    init_after_hl_action: str = ""
    """Same cut as ``launch_eval.py --init-after-hl-action`` (e.g. ``flip,rotate``).
    Empty starts each episode at frame 0."""
    profile_ranks: tuple[int, ...] = ()
    """Rank of each episode's randomization profile within the dataset, as
    emitted by ``tests/eval/pick_profile_episodes.py`` (0 is the smallest
    cube_scale), positionally matched to ``--episode-indices``. Reported only,
    so a failing case in a multi-profile sweep is traceable back to the profile
    it covered. Empty outside that driver, which records every rank as -1."""
    num_envs: int = 1
    """Parallel envs. Ignored when the task bakes per-episode cube-scale slots:
    num_envs is forced to the slot count and, per episode, the slot matching its
    cube_scale is the one verified (block size is non-sim-state)."""
    obs_mode: str = "rgb+state"
    """Observation mode; must include ``rgb`` for first-frame image comparison."""
    output_dir: str = "runs/verify_dataset_loading"
    """Parent directory; each episode's report JSON + comparison PNGs land in
    ``<output_dir>/ep<N>/`` (the PNG names are per-camera, so episodes sharing
    one directory would overwrite each other's overlays)."""
    skip_image_compare: bool = False
    """Skip first-frame image comparison (scene diff only)."""
    min_first_frame_ssim: float = 0.90
    """Hard floor on each verified camera's first-frame colour SSIM. Set well
    below the observed spread (worst seen: front 0.9864, gripper 0.9881) so it
    reads as "this is the recorded scene at all", not as a calibration bound."""
    cameras: tuple[str, ...] = VERIFIED_CAMERAS
    """Cameras the first-frame comparison covers; see VERIFIED_CAMERAS for why
    left_camera is not among them."""
    ingest_camera_intrinsics: bool = True
    """Override live camera intrinsics with the dataset's recorded values before
    verifying (mirrors the eval default). Disable to inspect the raw preset-vs-
    recorded intrinsics gap (the gripper camera then hard-fails by design)."""
    verify_policy_input: bool = True
    """Also feed the reset observation through the ManiFlow obs transform and
    check what the policy would actually receive (see
    ``toyblocks.eval.policy_input_verification``). Needs ``depth`` in
    ``--obs-mode``, since the transform is driven at its widest layout."""
    policy_input_golden_dir: str = "tests/eval/goldens"
    """Where the per-scene-preset observation-contract golden lives. The CPU
    fixture-parity test reads the same file, so moving this decouples the two."""
    write_policy_input_golden: bool = False
    """Re-record the observation-contract golden for this run's scene preset
    instead of comparing against it. Refuses to write if the dataset-derived
    policy-input checks failed, and skips the contract comparison for the run."""


class _OneShotQueueClient:
    """Minimal queue-client stand-in so ``_reset_idx`` applies our setup.

    Implements only what :meth:`TemplateEnv._reset_idx` and the verification
    path touch, so the diagnostic exercises the exact same ``apply_setup`` code
    the live queue eval runs (table + blocks + joints + goal) and the reset obs
    reflects it (cameras render during reset).

    ``setups`` is per env: an env whose entry is ``None`` keeps its default
    reset. Only envs spawned at a setup's cube_scale may receive it, since
    ``apply_setup`` rejects a setup on wrong-sized blocks.
    """

    def __init__(self, setups: list[dict | None]) -> None:
        self.current_setups: list[dict | None] = list(setups)
        self.previous_setups: list[dict | None] = [None] * len(setups)
        # Scene-load verification is not replay; keep joints at rest after reset.
        self.apply_recorded_joint_velocity: bool = False

    def fetch_reset_states(self, env_ids) -> list[dict | None]:
        ids = env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)
        return [self.current_setups[i] for i in ids]

    def verify_after_step(self) -> None:
        return

    @property
    def done(self) -> bool:
        return False


def print_tightest_margins(report: SceneMismatchReport, count: int = 5) -> None:
    """Print the fields closest to their tolerance -- the next things to flake."""
    ranked = sorted(
        (d for d in report.diffs if d.margin is not None and np.isfinite(d.margin)),
        key=lambda d: d.margin,
        reverse=True,
    )[:count]
    if not ranked:
        return
    print(f"Tightest margins (fraction of tolerance consumed, top {len(ranked)}):", flush=True)
    for d in ranked:
        print(f"  {d.margin:6.1%}  {d.name}", flush=True)


def _resolve_profile_ranks(args: Args) -> tuple[int, ...]:
    """Ranks aligned to ``--episode-indices``; all -1 when none were passed."""
    if not args.profile_ranks:
        return (-1,) * len(args.episode_indices)
    if len(args.profile_ranks) != len(args.episode_indices):
        raise ValueError(
            f"--profile-ranks {list(args.profile_ranks)} has "
            f"{len(args.profile_ranks)} entries but --episode-indices "
            f"{list(args.episode_indices)} has {len(args.episode_indices)}; they "
            "are matched positionally, so a report would be filed under another "
            "episode's profile."
        )
    return args.profile_ranks


def _append_ssim_floor_diffs(
    report: SceneMismatchReport, image_result: dict, floor: float
) -> None:
    """Turn each verified camera's first-frame SSIM into a hard :class:`FieldDiff`.

    The floor is what makes a wrong scene, a wrong camera or an unapplied setup
    fail instead of merely printing a number; it is not a calibration bound, and
    it is set far enough below the observed spread that lighting or driver
    variation cannot reach it. Reported margin is the share of the allowance
    ``1 - floor`` that the shortfall consumed.
    """
    for cam, m in image_result["cameras"].items():
        ssim = float(m["ssim"])
        report.diffs.append(
            FieldDiff(
                name=f"first_frame_ssim[{cam}]",
                passed=ssim >= floor,
                hard=True,
                dataset_value=f">= {floor}",
                sim_value=ssim,
                diff=ssim,
                tol=floor,
                margin=(1.0 - ssim) / (1.0 - floor) if floor < 1.0 else float("inf"),
            )
        )


def verify_episode(
    env,
    args: Args,
    spec,
    setup: dict,
    target_env: int,
    num_envs: int,
    profile_rank: int,
) -> bool:
    """Reset into one episode's recorded setup and diff the resulting scene.

    A fresh :class:`_OneShotQueueClient` per episode, so ``_reset_idx`` applies
    this episode's setup the same way the first one was applied -- the env
    itself is reused, only the scene state is swapped.
    """
    episode_index = spec.episode_index
    base_env = env.unwrapped
    per_env: list[dict | None] = [None] * num_envs
    per_env[target_env] = setup
    base_env.queue_client = _OneShotQueueClient(per_env)
    obs, _ = env.reset()

    out_dir = os.path.join(args.output_dir, f"ep{episode_index}")
    os.makedirs(out_dir, exist_ok=True)

    # --- Scene diff ---
    sim_state = read_sim_scene_state(base_env, target_env)
    report = compare_scene(spec, sim_state)
    print(report.format_console(), flush=True)

    # --- First-frame image comparison (reported, plus a hard SSIM floor) ---
    image_result = None
    if not args.skip_image_compare:
        image_result = compare_first_frames(
            spec, obs, target_env, out_dir, cameras=args.cameras
        )
        print("First-frame image metrics:", flush=True)
        for cam, m in image_result["cameras"].items():
            ssim_txt = (
                f" ssim={m['ssim']:.4f} gray={m['ssim_gray']:.4f} "
                f"edge={m['ssim_edge']:.4f} photonorm={m['ssim_photonorm']:.4f}"
                if "ssim" in m else ""
            )
            print(f"  {cam}: mae={m['mae']:.3f} mse={m['mse']:.3f}{ssim_txt} -> {m['png']}", flush=True)
        if image_result.get("caveat"):
            print(f"  note: {image_result['caveat']}", flush=True)
        _append_ssim_floor_diffs(report, image_result, args.min_first_frame_ssim)

    # --- Policy input (what the model would actually receive) ---
    # Appended to the same report on purpose: this is the only place a live
    # observation, the dataset's ground truth and the ``target_env`` matching this
    # episode's cube_scale all exist at once, so the checks that need all three
    # ride the existing console output, margin ranking, report JSON and exit code
    # rather than paying a second Isaac cold start.
    if args.verify_policy_input:
        tier_start = len(report.diffs)
        append_policy_input_diffs(
            obs,
            report,
            spec=spec,
            sim_state=sim_state,
            target_env=target_env,
            obs_mode=args.obs_mode,
            golden_dir=args.policy_input_golden_dir,
            write_golden=args.write_policy_input_golden,
        )
        print("Policy input checks:", flush=True)
        for d in report.diffs[tier_start:]:
            print(format_diff_line(d), flush=True)

    print_tightest_margins(report)

    report_dict = report.to_dict()
    report_dict["profile_rank"] = int(profile_rank)
    if image_result is not None:
        report_dict["first_frames"] = image_result

    report_path = os.path.join(out_dir, f"scene_verification_ep{episode_index}.json")
    with open(report_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Wrote report: {report_path}", flush=True)

    if not report.ok:
        print(
            f"\nHARD MISMATCH ({len(report.hard_failures)} field(s)): "
            f"{[d.name for d in report.hard_failures]}. The eval scene does not match "
            f"the dataset for episode {episode_index}.",
            flush=True,
        )
        return False
    print(f"\nEpisode {episode_index} passed (no hard mismatch).", flush=True)
    return True


def run(args: Args) -> int:
    if not args.dataset_path:
        raise ValueError("--dataset-path is required.")
    if not args.task:
        raise ValueError("--task is required (the generated gym id).")
    if not args.episode_indices:
        raise ValueError("--episode-indices is required (at least one episode).")
    if "rgb" not in args.obs_mode and not args.skip_image_compare:
        raise ValueError(
            "first-frame image comparison needs an rgb obs_mode; pass "
            "--obs-mode rgb+state or --skip-image-compare."
        )
    if args.verify_policy_input and "depth" not in args.obs_mode:
        raise ValueError(
            "policy-input verification drives the ManiFlow transform at its widest "
            "layout, which reads depth and camera intrinsics; pass "
            "--obs-mode rgb+state+depth or --no-verify-policy-input. Refusing to "
            "silently skip it, since a skipped check reads as a passing one."
        )
    profile_ranks = _resolve_profile_ranks(args)

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Dataset ground truth (same load path as the queue server) ---
    # Read for every episode before Isaac starts: these are cheap parquet reads,
    # and a missing episode or a cube_scale with no spawn slot is then a failure
    # that costs seconds instead of one that surfaces after a cold start.
    cut = args.init_after_hl_action.strip() or None
    specs = [
        extract_dataset_scene_spec(
            args.dataset_path, i, init_after_hl_action=cut
        )
        for i in args.episode_indices
    ]
    setups = [
        load_setups(
            args.dataset_path, episode_indices=[i], init_after_hl_action=cut
        )[0]
        for i in args.episode_indices
    ]

    # --- Pick the env to verify per episode ---
    # When the task bakes per-episode cube-scale slots, block size is non-sim-state:
    # an episode can only be reproduced in the env spawned at its scale. Size
    # num_envs to cover all slots -- which is also why every episode fits in one
    # Isaac session -- and read the slot matching each episode.
    slots = _cube_scale_slots_for_task(args.task)
    if slots:
        num_envs = len(slots)
        target_envs = [_target_env_for_scale(slots, s.get("cube_scale")) for s in setups]
        if args.num_envs != num_envs:
            print(
                f"[cube scale] task has {num_envs} spawn slot(s); overriding "
                f"--num-envs {args.num_envs} -> {num_envs}. Verifying env(s) "
                f"{target_envs} for episode(s) {list(args.episode_indices)}.",
                flush=True,
            )
    else:
        num_envs = args.num_envs
        target_envs = [0] * len(setups)

    # --- Build the env once, then reset into each episode in turn ---
    env_cfg = args.env_cfg
    env_cfg.sim.device = args.device
    env_cfg.scene.num_envs = num_envs
    env_cfg.obs_mode = args.obs_mode

    # evaluate.py always reconciles robot/scene against the dataset before building
    # the env. Skipping it here would let this diagnostic verify a different
    # scene_preset_id than the live eval uses, which makes its table_height_delta
    # comparison meaningless (the delta is relative to the preset baseline).
    from toyblocks.utils.dataset_provenance import reconcile_env_cfg_with_dataset

    reconcile_env_cfg_with_dataset(env_cfg, args.dataset_path)

    env = gym.make(id=args.task, cfg=env_cfg)
    base_env = env.unwrapped

    # --- Ingest recorded camera intrinsics (mirrors the eval default) ---
    # Intrinsics can only be set once the sensors exist, i.e. after a reset, and
    # they change what the cameras render -- so this warm-up reset happens before
    # the loop rather than inside it, or the first episode's first-frame overlay
    # would be the only one taken through the preset lens.
    # Each target env gets the first episode routed to it, so every env that is
    # verified below has its recorded intrinsics ingested.
    warmup: list[dict | None] = [None] * num_envs
    for setup, target_env in zip(setups, target_envs):
        if warmup[target_env] is None:
            warmup[target_env] = setup
    base_env.queue_client = _OneShotQueueClient(warmup)
    env.reset()
    if args.ingest_camera_intrinsics:
        from toyblocks.utils.dataset_provenance import ingest_dataset_camera_intrinsics
        ingest_dataset_camera_intrinsics(
            base_env,
            args.dataset_path,
            setups=list(base_env.queue_client.current_setups),
        )

    failed: list[int] = []
    for spec, setup, target_env, profile_rank in zip(
        specs, setups, target_envs, profile_ranks
    ):
        print(
            f"\n=== episode {spec.episode_index} (profile rank {profile_rank}, "
            f"env {target_env}, cube_scale={setup.get('cube_scale')}) ===",
            flush=True,
        )
        if not verify_episode(
            env, args, spec, setup, target_env, num_envs, profile_rank
        ):
            failed.append(spec.episode_index)

    if failed:
        print(
            f"\nFAILED {len(failed)}/{len(specs)} episode(s): {failed}. "
            f"See {args.output_dir}/ep<N>/scene_verification_ep<N>.json.",
            flush=True,
        )
        return 1
    print(
        f"\nScene verification passed for all {len(specs)} episode(s).", flush=True
    )
    return 0


def main(args: Args) -> None:
    exit_code = 1
    try:
        exit_code = run(args)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        exit_code = 130
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        exit_code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(exit_code)


if __name__ == "__main__":
    main(tyro.cli(Args))
