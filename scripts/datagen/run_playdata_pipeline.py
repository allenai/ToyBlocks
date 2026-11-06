#!/usr/bin/env python3
"""End-to-end pipeline for play-data experiments.

Pipeline:
  task-gen   Generate PDDL scenes                     (task_gen/play_scene.py)
  env        Build env packages + reset states         (generate_task_from_pddl.py)
  play       Generate play data                        (vecgen/run.py --play)
  render     Render each play problem                   (opt-in: --render-pass)
  merge      Merge goal-reached episodes into one LeRobot dataset (scripts/datagen/merge_datasets.py)
  annotate   Annotate the merged dataset               (annotate_target_xy.py)

Play data records exactly like the datagen pipeline's execution stage (same
LeRobot recorder, Parquet only; images come from --render-pass). Episodes whose goal was
reached go to ``blocks_success`` and missed ones to ``blocks_failure``, so the merge keeps
only goal-reached episodes. The merge hardlinks every rendered video instead of copying it.

Each reset state is one play "env": a sim slot runs a chain
of plans from it until ``--max-plans-per-env`` / ``--min-successful-plans-per-env``
is met, then moves to the next reset state; ``--play-num-envs`` chains run at once.

Goals are chosen for a roughly even skill mix by default (``--no-balanced-skills``
for random goals): about 1/3 pick-and-place, 1/3 flip (even over the 4
directions) and 1/3 rotate (even over cw / ccw), restricted to the allowed
high-level actions.

The play stage runs on the vecgen engine (``scripts/vecgen/run.py --play``): no TAMP and
no planner workers. Each goal is one FF call from the symbolic state read off the sim
(with the derived geometric facts), and the plan runs as batched vecgen skills.
--render-pass FLAGS.yaml[:N] renders each problem inside its play job, right after play, with
the datagen pipeline's render defaults. Repeat the flag for more looks.

Every stage runs locally, one problem per GPU (--local-gpus, default every visible GPU).
Datasets land under --output-root (default data/).

Usage:
  python scripts/datagen/run_playdata_pipeline.py --experiment-id sep25_play \\
      --num-pddls 8 --env-per-pddl 16 --play-num-envs 8 \\
      --max-plans-per-env 20 --min-successful-plans-per-env 10 --local-gpus 0 1 2 3
  python scripts/datagen/run_playdata_pipeline.py --experiment-id sep25_play \\
      --render-pass src/toyblocks/domain_randomization/flags/render_flags_plain.yaml \\
      --num-pddls 2 --usd-library train_blocks_large
  python scripts/datagen/run_playdata_pipeline.py --experiment-id sep25_play --resume

With --resume the pipeline continues an earlier run of the same experiment id:
completed stages (<base>/.pipeline_state/<stage>.done) are skipped, and within the
play stage every problem that already finished is skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_datagen_pipeline as dg  # noqa: E402  (shared helpers: run, local GPUs, resume)

log = dg.log

PLAY_STEPS: list[tuple[str, str]] = [
    ("task_gen", "Generate PDDL scenes"),
    ("env", "Build env packages + reset states"),
    ("play", "Generate play data"),
    ("render", "Render each play problem"),
    ("merge", "Merge goal-reached episodes (rendered if render ran)"),
    ("annotate", "Annotate merged dataset (target xy)"),
]

#: Push / align need RL skills, so play data leaves them out unless asked.
DEFAULT_DISALLOWED_ACTIONS: list[str] = ["push_to_right", "push_next_to"]

PLAY_COMPLETE_MARKER = ".play_complete"


PARAM_GROUPS: list[tuple[str, str | None, list[tuple[str, str, str]]]] = [
    ("General", None, [
        ("experiment_id", "experiment-id", "str"),
        ("output_root", "output-root", "str"),
        ("dataset_folder", "dataset-folder", "str"),
    ]),
    ("Task generation", "task_gen", [
        ("task_gen_script", "task-gen-script", "str"),
        ("num_pddls", "num-pddls", "int"),
        ("task_gen_seed", "task-gen-seed", "int"),
    ]),
    ("Env", "env", [
        ("env_per_pddl", "env-per-pddl", "int"),
        ("num_envs", "num-envs", "int"),
        ("num_scene_profiles", "num-scene-profiles", "int"),
        ("usd_library", "usd-library", "str"),
        ("randomize_robot_init", "randomize-robot-init", "bool"),
    ]),
    ("Play", "play", [
        ("play_num_envs", "play-num-envs", "int"),
        ("max_plans_per_env", "max-plans-per-env", "int"),
        ("min_successful_plans_per_env", "min-successful-plans-per-env", "int"),
        ("allowed_actions", "allowed-actions", "[str]"),
        ("disallowed_actions", "disallowed-actions", "[str]"),
        ("balanced_skills", "balanced-skills", "bool"),
        ("abandon_plan_percent", "abandon-plan-percent", "float"),
    ]),
    ("Render", "render", [
        ("render_pass", "render-pass", "[str]"),
        ("render_skip_cameras", "render-skip-cameras", "[str]"),
        ("render_skip_failure_episodes", "render-skip-failure-episodes", "bool"),
        ("render_extra", "render-extra", "[str]"),
    ]),
    ("Merge", "merge", [
        ("merge_cameras", "merge-cameras", "[str]"),
        ("merge_num_workers", "merge-num-workers", "int"),
        ("merge_output_dir", "merge-output-dir", "str"),
    ]),
    ("Annotate", "annotate", [
        ("annotate_num_workers", "annotate-num-workers", "int"),
    ]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--experiment-id", required=True, help="Experiment name")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Interactively choose which steps to run and edit parameters")
    parser.add_argument("--local-gpus", nargs="+", type=int, default=None, metavar="ID",
                        help="GPU ids the stages may use (default: every visible GPU); one problem per GPU")
    parser.add_argument("--resume", action="store_true",
                        help="Continue a previous run of this experiment id: skip completed stages "
                             "and rerun the first unfinished one")
    parser.add_argument("--skip-task-gen", action="store_true", help="Skip PDDL generation")
    parser.add_argument("--skip-env", action="store_true", help="Skip env package generation")
    parser.add_argument("--skip-play", action="store_true", help="Skip play-data generation")
    parser.add_argument("--skip-merge", action="store_true", help="Skip dataset merging")
    parser.add_argument("--skip-annotate", action="store_true",
                        help="Skip annotation of the merged dataset (annotate_target_xy.py)")
    dg.add_render_args(parser)
    # Missed goals are recorded into blocks_failure, which the merge drops, so by
    # default the renderer skips them too.
    parser.add_argument("--render-failure-episodes", dest="render_skip_failure_episodes",
                        action="store_false",
                        help="Render: also render blocks_failure (missed goals; the merge drops them)")
    parser.set_defaults(render_skip_failure_episodes=True)
    dg.add_merge_args(parser)

    parser.add_argument("--output-root", default="data",
                        help="Root for every dataset the pipeline writes (default: data)")
    parser.add_argument("--dataset-folder", default="unmerged")
    parser.add_argument("--domain-path", default="src/toyblocks/tamp/blocks")
    parser.add_argument("--seed", type=int, default=42, help="Play-data seed")

    # Stage 1: task generation
    parser.add_argument("--task-gen-script", default="scripts/task_gen/play_scene.py",
                        help="Scene generator plus its own flags, e.g. "
                             "\"scripts/task_gen/play_scene.py --blocks A I 2 E\"")
    parser.add_argument("--num-pddls", type=int, default=4, help="How many PDDL problems to generate")
    parser.add_argument("--task-gen-seed", type=int, default=42)

    # Stage 2: env packages
    parser.add_argument("--env-per-pddl", type=int, default=16,
                        help="Reset states per PDDL; each is one play env (chain of plans)")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Parallel sim envs while sampling reset states (env-per-pddl must be a multiple)")
    parser.add_argument("--num-scene-profiles", type=int, default=8,
                        help="Block-size / table-height / friction profiles baked into each env package")
    parser.add_argument("--usd-library", default="",
                        help="Glyph asset library, e.g. train_blocks_large (default: train_blocks_large)")
    parser.add_argument("--randomize-robot-init", action="store_true", default=False,
                        help="Give each reset state a random collision-free arm configuration")
    parser.add_argument("--robot-init-ee-x-range", type=float, nargs=2, default=[0.25, 0.65],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--robot-init-ee-y-range", type=float, nargs=2, default=[-0.35, 0.35],
                        metavar=("MIN", "MAX"))
    parser.add_argument("--robot-init-ee-z-min", type=float, default=0.20)
    parser.add_argument("--robot-init-candidates", type=int, default=32)
    parser.add_argument("--robot-init-gripper-closed-prob", type=float, default=0.5)

    # Stage 3: play data
    parser.add_argument("--play-num-envs", type=int, default=8,
                        help="Parallel play envs (sim slots) per problem")
    parser.add_argument("--max-plans-per-env", type=int, default=20,
                        help="Max executed plans per env (reset state)")
    parser.add_argument("--min-successful-plans-per-env", type=int, default=None,
                        help="Stop an env (reset state) once this many of its plans reached their goal")
    parser.add_argument("--allowed-actions", nargs="*", default=None,
                        help="Only these high-level actions may be planned with (default: all)")
    parser.add_argument("--disallowed-actions", nargs="*", default=list(DEFAULT_DISALLOWED_ACTIONS),
                        help=f"High-level actions the planner must not use "
                             f"(default: {' '.join(DEFAULT_DISALLOWED_ACTIONS)}; pass the flag "
                             "with no names to allow them)")
    parser.add_argument("--no-balanced-skills", dest="balanced_skills", action="store_false",
                        help="Random goals instead of the balanced pnp / flip / rotate mix")
    parser.set_defaults(balanced_skills=True)
    parser.add_argument("--max-stack-height", type=int, default=3,
                        help="Tallest tower a balanced stacking goal may build")
    parser.add_argument("--balance-by-success", action="store_true",
                        help="Balanced skills: count a plan's skills only once its episode is kept, "
                             "so often-failing skills are requested until they reach their share")
    parser.add_argument("--flip-min-x", nargs="+", default=None, metavar="DIR=M",
                        help="Balanced skills: min block x (m from robot) per flip direction, "
                             "e.g. forward=0.45 left_flip=0.35")
    parser.add_argument("--abandon-plan-percent", type=float, default=0.0,
                        help="Percent of pick-and-place plans abandoned mid-carry (recorded as a "
                             "success); the next episode drops the held block and picks another")
    parser.add_argument("--play-record-videos", action="store_true",
                        help="Record the front / gripper camera videos while playing (run.py "
                             "--record-videos); without it play records Parquet only and images "
                             "come from --render-pass")
    parser.add_argument("--dfa-spec", default=None,
                        help="DFA strategy choosing the goals (e.g. "
                             "src/toyblocks/playdata/dfa/library/flip_search.yaml) instead of "
                             "balanced / random goals; on ACCEPT / REJECT the env moves on")
    parser.add_argument("--dfa-param", nargs="+", default=None, metavar="KEY=VALUE",
                        help="Override the DFA spec's params (e.g. letter=Q flips_per_block=3)")
    parser.add_argument("--play-env", nargs="+", default=None, metavar="KEY=VALUE",
                        help="Extra env vars for the play jobs")
    return parser.parse_args()


def play_env(args: argparse.Namespace) -> dict[str, str]:
    """Extra environment variables for the play jobs (--play-env)."""
    env: dict[str, str] = {}
    for kv in args.play_env or []:
        if "=" not in kv:
            raise SystemExit(f"--play-env expects KEY=VALUE, got {kv!r}")
        key, value = kv.split("=", 1)
        env[key] = value
    return env


def vecgen_play_cli_flags(args: argparse.Namespace) -> list[str]:
    """``scripts/vecgen/run.py --play`` flags for the local play stage."""
    flags: list[str] = [
        "--num-envs", str(args.play_num_envs),
        "--max-plans-per-env", str(args.max_plans_per_env),
        "--one-chain-per-reset-state",
        # Goal-reached episodes to blocks_success, missed ones to blocks_failure.
        "--missed-goals-to-failure",
    ]
    if args.min_successful_plans_per_env is not None:
        flags += ["--min-successful-plans-per-env", str(args.min_successful_plans_per_env)]
    if args.allowed_actions:
        flags += ["--allowed-actions", *args.allowed_actions]
    if args.disallowed_actions:
        flags += ["--disallowed-actions", *args.disallowed_actions]
    if args.dfa_spec:
        flags += ["--dfa-spec", args.dfa_spec]
        if args.dfa_param:
            flags += ["--dfa-param", *args.dfa_param]
    elif args.balanced_skills:
        flags += ["--balanced-skills", "--max-stack-height", str(args.max_stack_height)]
        if args.balance_by_success:
            flags.append("--balance-by-success")
        if args.flip_min_x:
            flags += ["--flip-min-x", *args.flip_min_x]
    if args.abandon_plan_percent:
        flags += ["--abandon-plan-percent", str(args.abandon_plan_percent)]
    if args.play_record_videos:
        flags += ["--record-videos", "--enable_cameras"]
    return flags


def robot_init_flags(args: argparse.Namespace, style: str) -> list[str]:
    """Robot-init flags for ``generate_task_from_pddl.py`` (``underscore``) or stage 1 (``dash``)."""
    if not args.randomize_robot_init:
        return []
    sep = "_" if style == "underscore" else "-"
    name = lambda n: "--" + n.replace("_", sep)  # noqa: E731
    return [
        name("randomize_robot_init"),
        name("robot_init_ee_x_range"), *(str(v) for v in args.robot_init_ee_x_range),
        name("robot_init_ee_y_range"), *(str(v) for v in args.robot_init_ee_y_range),
        name("robot_init_ee_z_min"), str(args.robot_init_ee_z_min),
        name("robot_init_candidates"), str(args.robot_init_candidates),
        name("robot_init_gripper_closed_prob"), str(args.robot_init_gripper_closed_prob),
    ]


def stage_merge_source(roots: list[Path], staging_dir: Path) -> list[str]:
    """Symlink every ``problem_*`` dir under *roots* that holds a ``blocks_success`` dataset.

    merge_datasets.py decides between its layouts from the first ``problem_*`` dir it
    sees, and in the fallback layout takes whichever dataset subdir it lists first,
    which may be ``blocks_failure``. A problem dir with no ``blocks_success`` (a base
    dir whose data went to a ``__sNNN`` shard, or one whose episodes all missed) would
    trip that, so the merge reads a staging dir holding only dirs with a
    ``blocks_success``. Hardlinking videos through the symlinks still works.
    Returns (staging dirs to merge, linked problem names).
    Each root is staged into its OWN subdirectory and handed to merge_datasets.py as its own
    --input_dirs entry: every render pass holds the same ``problem_*`` names, so one flat
    staging dir made a second --render-pass abort on a name collision.
    """
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged: list[Path] = []
    linked: list[str] = []
    for root in roots:
        sub_dir = staging_dir / root.name
        sub_dir.mkdir(parents=True, exist_ok=True)
        found = False
        for src in sorted(root.glob("problem_*")):
            if not (src / "blocks_success" / "meta" / "info.json").exists():
                continue
            os.symlink(str(src.resolve()), sub_dir / src.name)
            linked.append(src.name)
            found = True
        if found:
            staged.append(sub_dir)
        else:
            sub_dir.rmdir()
    return staged, linked


def summarize_play(play_dir: Path, slugs: list[str]) -> dict[str, Any]:
    """Aggregate the per-problem ``summary.json`` files of the play stage."""
    skills: Counter[str] = Counter()
    totals: Counter[str] = Counter()
    chains_below_min = 0
    missing: list[str] = []
    for slug in slugs:
        path = play_dir / slug / "summary.json"
        if not path.exists():
            missing.append(slug)
            continue
        play = json.loads(path.read_text()).get("play_data", {})
        skills.update(play.get("executed_skills", {}))
        for key in ("plans_installed", "plans_failed", "goals_reached", "goals_missed"):
            totals[key] += int(play.get(key, 0) or 0)
        chains_below_min += int(play.get("chains_below_min_success", 0) or 0)
    total_skills = sum(skills.values())
    return {
        "problems": len(slugs),
        "problems_missing_summary": missing,
        **dict(totals),
        "chains_below_min_success": chains_below_min,
        "executed_skills": dict(sorted(skills.items())),
        "executed_skill_shares": {
            k: round(v / total_skills, 4) for k, v in sorted(skills.items())
        } if total_skills else {},
    }


def main() -> int:
    args = parse_args()
    exp = args.experiment_id

    enabled = {
        "task_gen": not args.skip_task_gen,
        "env": not args.skip_env,
        "play": not args.skip_play,
        "render": bool(dg.resolve_render_passes(args)),
        "merge": not args.skip_merge,
        "annotate": not args.skip_annotate,
    }
    if args.interactive:
        enabled = dg.prompt_steps(enabled, steps=PLAY_STEPS)
        dg.prompt_params(args, enabled, param_groups=PARAM_GROUPS)
        exp = args.experiment_id
        # The render runs inside each play job from resolve_render_passes, not from
        # enabled["render"], so unticking the stage has to drop the passes too. Ticking it with
        # no --render-pass would otherwise mark the stage complete having rendered nothing.
        if not enabled["render"]:
            args.render_pass = []
        elif not dg.resolve_render_passes(args):
            log.error("Render: the stage is ticked but no --render-pass was given. Pass "
                      "--render-pass FLAGS.yaml[:N]")
            return 1

    # -- validate -------------------------------------------------------------
    # As in the datagen pipeline: render.py spawns env i at block-size slot i % profiles, so fewer
    # envs than profiles leaves slots without an env and every problem is skipped.
    if enabled["render"]:
        # Checked before the play stage burns its GPU hours: an unreadable YAML would otherwise
        # fail inside the render, one Isaac startup per problem later.
        for _flags, _copies in dg.resolve_render_passes(args):
            try:
                readable = Path(_flags).is_file()
            except OSError:
                readable = False
            if not readable:
                log.error("the render needs a flags YAML readable on THIS machine; %s is not",
                          _flags)
                return 1
    if enabled["render"] and args.render_num_envs < args.num_scene_profiles:
        log.error("--render-num-envs (%d) must be at least --num-scene-profiles (%d)",
                  args.render_num_envs, args.num_scene_profiles)
        return 1
    if enabled["env"] and args.env_per_pddl % args.num_envs != 0:
        log.error("env-per-pddl (%d) must be divisible by num-envs (%d)", args.env_per_pddl, args.num_envs)
        return 1
    for stage, flag, envs in (("env", "num-envs", args.num_envs), ("play", "play-num-envs", args.play_num_envs)):
        if enabled[stage] and envs >= args.num_scene_profiles and envs % args.num_scene_profiles != 0:
            log.error("%s (%d) must be a multiple of num-scene-profiles (%d)", flag, envs, args.num_scene_profiles)
            return 1
    if args.min_successful_plans_per_env is not None and args.min_successful_plans_per_env > args.max_plans_per_env:
        log.error("min-successful-plans-per-env (%d) cannot exceed max-plans-per-env (%d)",
                  args.min_successful_plans_per_env, args.max_plans_per_env)
        return 1
    env_vars = play_env(args)

    # -- pin commits ------------------------------------------------------------
    toyblocks_sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, check=True).stdout.strip()
    log.info("Pinned commit: ToyBlocks=%s", toyblocks_sha)

    # -- paths ----------------------------------------------------------------------
    # Absolute: the render staging symlinks and the per-stage subprocesses must not depend
    # on the working directory a relative --output-root was given in.
    output_root = os.path.abspath(args.output_root)
    base = f"{output_root}/datasets/{args.dataset_folder}/{exp}"
    tasks_dir = f"{base}/tasks"
    envs_dir = f"{base}/envs"
    play_dir = Path(base) / "play"
    local_logs_dir = Path(base) / "local_logs"
    render_dir = Path(base) / "render"
    merge_source_dir = Path(base) / "merge_src"
    merge_output_dir = os.path.abspath(args.merge_output_dir or f"{output_root}/datasets/merged/train/{exp}")
    log.info("Experiment: %s", exp)
    log.info("Base path:  %s", base)

    if args.resume:
        dg.apply_resume(
            base,
            enabled,
            {
                "task_gen": [tasks_dir],
                "env": [envs_dir],
                # Finished problems are skipped by their .play_complete marker, so a
                # resumed play stage keeps them and only redoes the rest.
                "play": [],
                "render": [],
                "merge": [merge_output_dir],
                "annotate": [],
            },
            steps=PLAY_STEPS,
        )

    local_slots: list[str | None] = [None]
    gpus = [str(g) for g in args.local_gpus] if args.local_gpus else dg.detect_local_gpus()
    if gpus:
        local_slots = list(gpus)
    log.info("GPUs: %s", f"{len(gpus)} [{', '.join(gpus)}]" if gpus else "none detected")

    # -- config -------------------------------------------------------------------
    Path(base).mkdir(parents=True, exist_ok=True)
    config = {
        **{k: v for k, v in vars(args).items() if isinstance(v, (str, int, float, bool, list, type(None)))},
        "experiment_id": exp,
        "toyblocks_commit_sha": toyblocks_sha,
        "play_env": env_vars,
        "enabled_steps": enabled,
    }
    config_path = Path(base) / f"run_playdata_pipeline_config_{datetime.now():%Y%m%d_%H%M%S}.json"
    config_path.write_text(json.dumps(config, indent=2))
    log.info("Saved pipeline config to %s", config_path)

    # -- stage 1: PDDL scenes -------------------------------------------------------
    if enabled["task_gen"]:
        log.info("=" * 60)
        log.info("Task generation: %d PDDL scene(s)", args.num_pddls)
        log.info("=" * 60)
        dg.run([
            sys.executable, *shlex.split(args.task_gen_script),
            "--count", str(args.num_pddls),
            "--out-dir", tasks_dir,
            "--seed", str(args.task_gen_seed),
        ])
        dg.mark_stage_complete(base, "task_gen")

    # -- stage 2: env packages ------------------------------------------------------
    if enabled["env"]:
        log.info("=" * 60)
        log.info("Env: build env packages + %d reset states per PDDL locally", args.env_per_pddl)
        log.info("=" * 60)
        Path(envs_dir).mkdir(parents=True, exist_ok=True)
        env_jobs: list[tuple[str, list[str]]] = []
        for pddl in dg.list_pddls(tasks_dir):
            slug = dg.task_slug(pddl)
            if (Path(envs_dir) / slug / "reset_states.pkl").exists():
                log.info("Skipping %s (reset_states.pkl exists)", slug)
                continue
            cmd = [
                sys.executable, "scripts/generate_task_from_pddl.py",
                "--pddl", str(pddl),
                "--domain_path", args.domain_path,
                "--output_dir", envs_dir,
                "--num_reset_states", str(args.env_per_pddl),
                "--num_envs", str(args.num_envs),
                "--num_scene_profiles", str(args.num_scene_profiles),
                "--seed", str(dg.ENV_GEN_SEED),
                "--headless",
            ]
            if args.usd_library:
                cmd += ["--usd_library", args.usd_library]
            cmd += robot_init_flags(args, "underscore")
            env_jobs.append((slug, cmd))
        dg.run_local_stage("Env", env_jobs, local_slots, local_logs_dir / "env")
    if enabled["env"]:
        dg.mark_stage_complete(base, "env")

    # -- stage 3: play data ---------------------------------------------------------
    slugs = [dg.task_slug(p) for p in dg.list_pddls(tasks_dir)]
    if enabled["play"]:
        log.info("=" * 60)
        log.info("Play: generate play data locally (%s)", " ".join(f"{k}={v}" for k, v in env_vars.items()))
        log.info("=" * 60)
        play_dir.mkdir(parents=True, exist_ok=True)
        play_jobs: list[tuple[str, list[str]]] = []
        for pddl in dg.list_pddls(tasks_dir):
            slug = dg.task_slug(pddl)
            out_dir = play_dir / slug
            if (out_dir / PLAY_COMPLETE_MARKER).exists():
                log.info("Skipping %s (already collected)", slug)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            play_cmd = [
                sys.executable, "scripts/vecgen/run.py", "--play",
                "--env-path", str(Path(envs_dir) / slug),
                "--pddl", str(pddl),
                "--domain", f"{args.domain_path}/domain.pddl",
                "--record", str(out_dir),
                "--summary", str(out_dir / "summary.json"),
                "--mark-complete", "--headless",
                "--seed", str(args.seed),
                *vecgen_play_cli_flags(args),
            ]
            if enabled["render"]:
                # Coupled: the render runs in the play job's own slot once play has finished.
                render = dg.local_render_steps(args, base, envs_dir, slug, source_dir=out_dir)
                play_jobs.append((slug, ["bash", "-c", " && ".join([shlex.join(play_cmd), *render])]))
            else:
                play_jobs.append((slug, play_cmd))
        # run_local_stage passes this process's environment on to every job.
        os.environ.update(env_vars)
        dg.run_local_stage("Play", play_jobs, local_slots, local_logs_dir / "play")
        for slug, _ in play_jobs:
            # The summary is written only after a clean finish.
            if (play_dir / slug / "summary.json").exists():
                (play_dir / slug / PLAY_COMPLETE_MARKER).write_text("")
    if enabled["play"]:
        summary = summarize_play(play_dir, slugs)
        (Path(base) / "play_summary.json").write_text(json.dumps(summary, indent=2))
        log.info("Play summary (%s):\n%s", Path(base) / "play_summary.json", json.dumps(summary, indent=2))
        if summary["problems_missing_summary"]:
            log.warning("Play: %d problem(s) have no summary (failed or unfinished): %s",
                        len(summary["problems_missing_summary"]), " ".join(summary["problems_missing_summary"]))
        else:
            dg.mark_stage_complete(base, "play")

    # -- stage 4: render ----------------------------------------------------------
    # The renders ran inside the local play jobs; ground truth is the output on disk.
    if enabled["render"]:
        out_roots = [render_dir / "out" / job
                     for job, _ in dg.local_render_passes(args, str(render_dir.parent))]
        incomplete = [s for s in slugs
                      if not all(dg.render_problem_complete(r, s) for r in out_roots)]
        if incomplete:
            log.error("Render: %d problem(s) did not finish rendering and will be missing "
                      "from the merge: %s", len(incomplete), " ".join(incomplete[:20]))
        if len(incomplete) < len(slugs):
            dg.mark_stage_complete(base, "render")

    # -- stage 5: merge goal-reached episodes -------------------------------------------
    render_ran = enabled["render"] or dg.stage_marker(base, "render").exists()
    if enabled["merge"]:
        log.info("=" * 60)
        log.info("Merge: merge goal-reached episodes (%s)", "rendered" if render_ran else "play")
        log.info("=" * 60)
        roots = sorted(d for d in (render_dir / "out").glob("job_*") if d.is_dir()) if render_ran else [play_dir]
        staged, linked = stage_merge_source(roots, merge_source_dir)
        if not linked:
            log.error("Merge: no blocks_success dataset under %s", " ".join(str(r) for r in roots))
            return 1
        log.info("Merge: %d problem dir(s) staged in %s: %s", len(linked), merge_source_dir,
                 " ".join(d.name for d in staged))
        dg.run([
            sys.executable, "scripts/datagen/merge_datasets.py",
            "--input_dirs", *(str(d) for d in staged),
            "--output_dir", merge_output_dir,
            "--cameras", *args.merge_cameras,
            "--num_workers", str(args.merge_num_workers),
        ])
        shutil.copy(config_path, Path(merge_output_dir) / config_path.name)
        log.info("Copied pipeline config to %s", Path(merge_output_dir) / config_path.name)
        dg.mark_stage_complete(base, "merge")

    # -- stage 6: annotate --------------------------------------------------------------
    # The annotator projects goals into the front camera: needs recorded (--play-record-videos)
    # or rendered episodes.
    if enabled["annotate"] and not render_ran and not args.play_record_videos:
        log.warning("Annotate: skipped, the dataset records no camera (pixel goals need "
                    "--render-pass or --play-record-videos)")
        enabled["annotate"] = False
    if enabled["annotate"]:
        log.info("=" * 60)
        log.info("Annotate: run annotate_target_xy.py on the merged dataset")
        log.info("=" * 60)
        if not Path(merge_output_dir).is_dir():
            log.error("Annotate: merged dataset not found at %s", merge_output_dir)
            return 1
        # The annotator exits non-zero when it wrote no goal columns; treat that as a failure.
        result = dg.run([
            sys.executable, "src/toyblocks/data_annotation/annotate_target_xy.py",
            merge_output_dir, "-j", str(args.annotate_num_workers),
        ], check=False)
        if result.returncode != 0:
            log.error("Annotation FAILED (exit %d) for %s", result.returncode, merge_output_dir)
            return 1
        dg.mark_stage_complete(base, "annotate")

    # -- plot a sample episode ----------------------------------------------------------
    if Path(merge_output_dir).exists():
        plot_output = str(Path(merge_output_dir) / "episode_0.png")
        dg.run([
            sys.executable, "scripts/datagen/plot_episode.py",
            "--dataset", merge_output_dir, "--episode", "0", "--output", plot_output,
        ])
        log.info("Plot saved to %s", plot_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
