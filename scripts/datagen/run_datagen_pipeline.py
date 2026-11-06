#!/usr/bin/env python3
"""End-to-end pipeline for ToyBlocks datagen experiments.

Pipeline:
  task-gen            Generate PDDL tasks              (task_gen/rotate.py)
  env                 Build the env packages + reset states (generate_task_from_pddl.py)
  execution           Run vecgen per problem (scripts/vecgen/run.py)
  render              Render finished execution (opt-in): one --render-pass per look,
                      scheduled by --render-schedule
  merge               Merge successful datasets        (scripts/datagen/merge_datasets.py)
  annotate            Annotate the merged dataset      (annotate_target_xy.py)

The execution stage runs the vecgen engine (scripts/vecgen/run.py):
one Isaac process per problem plans with FF per env, executes and verifies the skills in batch
and records the LeRobot dataset (Parquet only unless --vecgen-record-videos) into
execution/<slug>/. --exec-num-envs and --seed apply; episodes
per problem default to --env-per-problem (the reset states). The flow needs no TAMP checkout.

With --render-pass FLAGS.yaml[:N] each finished problem is rendered
(scripts/render/render.py) and the merge consumes the rendered episodes instead of the
execution data. Repeat the flag for more looks: each pass renders every episode into its own
job_<name> directory and the merge takes them all. The defaults around a pass are one copy per
episode, the 224 RGB and rg8_55_224 depth sidecars, left_camera skipped and failure episodes
skipped.

Every stage runs locally, one problem per GPU slot (--local-gpus, default every visible
GPU, x --local-gpu-workers); each step sees a single GPU. Datasets land under --output-root
(default data/): <root>/datasets/unmerged/<exp>/ and <root>/datasets/merged/train/<exp>/.

Usage:
  python run_datagen_pipeline.py --experiment-id jun23_rotate
  python run_datagen_pipeline.py --experiment-id jun23_rotate --skip-env
  python run_datagen_pipeline.py --experiment-id jun23_rotate --interactive
  python run_datagen_pipeline.py --experiment-id jun23_rotate --local-gpus 0 1
  python run_datagen_pipeline.py --experiment-id jun23_rotate --resume
  python run_datagen_pipeline.py --experiment-id jun23_rotate \
      --render-pass src/toyblocks/domain_randomization/flags/render_flags_plain.yaml
  python run_datagen_pipeline.py --experiment-id jun23_rotate \
      --render-pass src/toyblocks/domain_randomization/flags/render_flags_plain.yaml \
      --render-extra="--extra_depth g12_224"

The video sidecars have no dedicated --render-* flag. --extra_rgb 224 and --extra_depth
rg8_55_224 are on by default and --render-extra overrides them. --extra_rgb 224 writes videos/observation.images.<cam>_224/ (the HVLA training
sidecar, Pillow BILINEAR, byte-compatible with reencode_to_224_batch.py). --extra_depth
g12_224 writes the depth equivalent as lossless gray12le HEVC with a short per-camera GOP
-- ~0.16x the bytes of the older rg8_224 and far cheaper to read cold; both decode to
identical depth values, so prefer g12_224 for new datasets. Both emit ONE sidecar per RENDERED camera, so
--render-skip-cameras reduces their cost too.

The defaults are the settings the Nov 2026 stack-3 training set (RI) was generated with:
stack_three.py --train (1000 problems), randomized scenes, 32 reset states per problem,
8 problems per vecgen env package at 128 envs, push actions excluded.
With --vecgen-problems-per-env G > 1 there is no env stage: each vecgen job builds one
multi-problem env package per G PDDLs (envs/problem_grp_<first slug>/, scripts/vecgen/make_superset_env.py:
union of the blocks, the others parked, --env-per-problem reset states per problem) and streams
all of the group's episodes through one process with --exec-num-envs envs (e.g. 128-256), so
the Isaac start-up and the episode tail are paid once per group; data lands in
execution/problem_grp_<first slug>/.

With --resume the pipeline continues an earlier run of the same experiment id:
completed stages (marked by <base>/.pipeline_state/<stage>.done) are skipped and
the first unfinished stage restarts from scratch with its partially written outputs deleted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def _read_template_env_cfg(domain_path: str) -> dict[str, str | None]:
    """Parse ``robot_id`` and ``scene_preset_id`` from the template env cfg.

    Reads the file textually (no isaaclab import needed) and returns a dict
    with keys ``robot_id`` and ``scene_preset_id`` (None if not found).
    """
    cfg_path = Path(domain_path) / "template" / "template_env_cfg.py"
    result: dict[str, str | None] = {"robot_id": None, "scene_preset_id": None}
    if not cfg_path.exists():
        log.warning("Template env cfg not found at %s", cfg_path)
        return result
    text = cfg_path.read_text()
    for key in result:
        m = re.search(rf'^\s*{key}\s*(?::\s*\w+)?\s*=\s*["\'](.+?)["\']', text, re.MULTILINE)
        if m:
            result[key] = m.group(1)
    return result


def run(cmd: list[str] | str, *, check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a command, logging it first."""
    cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
    log.info("Running: %s", cmd_str)
    result = subprocess.run(cmd, check=check, **kwargs)
    return result


# ── Rendering ──────────────────────────────────────────────────────
# render.py fingerprints its whole source dir and indexes episodes into it, so it must never
# point at execution/ while sibling jobs are still writing there: each render instead gets a
# frozen directory of symlinks to its finished problem dirs.
RENDER_SPLITS = ("blocks_success", "blocks_failure")


def stage_render_source(execution_dir: Path, staging_dir: Path, slugs: list[str]) -> list[str]:
    """Symlink the finalized problem dirs for *slugs* into *staging_dir*.

    Includes the ``<slug>__sNNN`` shard siblings written by preempted-and-resumed
    execution jobs; the renderer resolves their env from the base problem.
    Returns the directory names linked.
    The render planner only enumerates ``problem_*`` dirs holding a
    ``blocks_success``/``blocks_failure`` dataset.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    linked: list[str] = []
    for slug in slugs:
        candidates = [
            execution_dir / slug,
            *sorted(execution_dir.glob(f"{slug}__s[0-9][0-9][0-9]")),
        ]
        for src in candidates:
            finalized = any(
                (src / split / "meta" / "info.json").exists() for split in RENDER_SPLITS
            )
            if not finalized:
                continue
            if not src.name.startswith("problem_"):
                log.warning(
                    "Render: skipping %s (the renderer only picks up problem_* dirs)", src
                )
                continue
            link = staging_dir / src.name
            # Absolute target: a relative one resolves against the link's own directory. A link
            # left by an older run pointing anywhere else is stale, so re-point it.
            target = str(src.resolve())
            if link.is_symlink() and os.readlink(link) != target:
                link.unlink()
            if not link.is_symlink():
                os.symlink(target, link)
            linked.append(src.name)
    return linked


#: ``problem_x__s001`` is a preemption shard of ``problem_x``, not a problem of its own.
SHARD_SUFFIX = re.compile(r"__s\d{3}$")


def base_problem_slugs(slugs: Iterable[str]) -> list[str]:
    """Collapse ``<slug>__sNNN`` preemption shards onto their base slug, order preserved.

    ``stage_render_source`` already pulls a slug's shard siblings into that slug's staging
    dir, so feeding it both ``problem_x`` and ``problem_x__s001`` stages -- and renders --
    ``problem_x__s001`` TWICE, into one output dir. With two slots both run at once and
    ``render.py --resume`` ``shutil.rmtree``s the one that has not finalized yet, destroying
    a live writer's output. A disk glob for ``problem_*`` matches shards; a job list never did,
    which is why only the local sweep needs this.
    """
    out, seen = [], set()
    for slug in slugs:
        base = SHARD_SUFFIX.sub("", slug)
        if base not in seen:
            seen.add(base)
            out.append(base)
    return out


#: Where each render pass writes (see render_job_name). The basename matters twice: merge globs
#: ``out/job_*`` (so it must start with ``job_``), and the per-problem output directory name is
#: the DR salt (``seeds._stable_seed_salt``), so it must not change or a rerun of the same
#: problem draws different randomization. Each pass gets its OWN job dir because render.py
#: --resume skips a problem whose render_params says completed, so a second pass into the same
#: dir would render nothing and exit 0.

#: Prefix that marks a render item in the shared local slot pool, so the completion
#: callback can tell "an execution just finished, queue its render" from "a render finished".
RENDER_ITEM_PREFIX = "render:"

#: render.py prints "all jobs done" and then hangs in Isaac teardown, so every local render
#: goes through this wrapper, which keys completion on the marker and SIGKILLs the group.
RUN_LOCAL_RENDER = Path(__file__).resolve().parents[2] / "scripts" / "render" / "run_local_render.py"
#: The video sidecars and GOPs every pass gets; --render-extra (appended after) overrides them.
RENDER_SIDECARS = ("--extra_rgb", "224", "--extra_depth", "rg8_55_224",
                        "--extra_rgb_GOP", "2", "--rgb_GOP", "2")


def local_env_argv(args, pddl, envs_dir: str) -> list[str]:
    """argv for one problem's LOCAL env-generation job (scripts/generate_task_from_pddl.py).

    A module-level builder so the composed argv is testable without running the stage, like
    :func:`local_render_argv` and :func:`robot_init_flags`.

    Spellings are UNDERSCORED here because the target is generate_task_from_pddl.py. It is a
    tyro CLI and accepts either form, but every other flag in this argv is underscored. Do NOT
    copy these spellings to the
    multi-problem superset call: make_superset_env.py is argparse with hyphen-only names AND
    parse_known_args, so an underscore there is silently swallowed back to the default.
    """
    cmd = [
        sys.executable,
        "scripts/generate_task_from_pddl.py",
        "--pddl",
        str(pddl),
        "--domain_path",
        args.domain_path,
        "--output_dir",
        envs_dir,
        "--num_reset_states",
        str(args.env_per_problem),
        "--num_envs",
        str(args.num_envs),
        # Baked into the package at codegen time as MAX_NON_SIM_STATE_RANDOMIZATIONS, and used
        # as the SIZE of the block-size / table-height / friction draws. Omitting it let the
        # generator fall back to its own default of 8 while the pre-flight check validated
        # --num-envs against the REQUESTED count, so a run could pass validation for 4 profiles
        # and then build 8-profile packages. make_superset_env always forwarded it.
        "--num_scene_profiles",
        str(args.num_scene_profiles),
        "--seed",
        str(ENV_GEN_SEED),
        "--headless",
    ]
    if args.usd_library:
        cmd += ["--usd_library", args.usd_library]
    if args.block_x_range:
        cmd += ["--block_x_range", args.block_x_range]
    if args.block_y_range:
        cmd += ["--block_y_range", args.block_y_range]
    if args.randomize_robot_init:
        cmd += [
            "--randomize_robot_init",
            "--robot_init_ee_x_range",
            *(str(v) for v in args.robot_init_ee_x_range),
            "--robot_init_ee_y_range",
            *(str(v) for v in args.robot_init_ee_y_range),
            "--robot_init_ee_z_min",
            str(args.robot_init_ee_z_min),
            "--robot_init_candidates",
            str(args.robot_init_candidates),
            "--robot_init_gripper_closed_prob",
            str(args.robot_init_gripper_closed_prob),
        ]
    return cmd


def local_render_argv(args, src_dir: Path, out_dir: Path, envs_dir: str, *,
                        flags_path: Path, renders_per_episode: int) -> list[str]:
    """The render.py command for ONE PASS of one problem.

    Four of these differ from ``render.py``'s OWN argparse defaults and silently change the
    pixels when omitted:

      --env_spacing 60        render.py default None -> 5.0; at 5.0 neighbouring envs are
                              close enough to appear in each other's cameras. render_params.json
                              records it under args.env_spacing, but only when a run passes it
                              explicitly.
      --camera_base scene     render.py default "data" -- a different pose source.
      --rtx_warmup_frames 16  render.py default 1 -- fewer warmup frames, different exposure.
      --num_envs              A FLOOR, not taste: plan_block_sizes
                              errors and SKIPS the problem unless num_envs covers every size slot
                              in the env package, and the generated table has
                              MAX_NON_SIM_STATE_RANDOMIZATIONS (8) of them.

    The video sidecars (RENDER_SIDECARS) come before ``--render-extra``, which can
    override them (render.py is last-wins).

    render.py is run BY ABSOLUTE PATH with -u, never ``-m`` and never imported: it sibling-imports
    ``scene_swap`` and ``render_sources`` from its own directory.
    """
    rr = Path(__file__).resolve().parents[2] / "scripts" / "render" / "render.py"
    argv = [
        sys.executable, "-u", str(rr),
        str(src_dir),
        "--output", str(out_dir),
        "--envs_dir", envs_dir,
        "-N", str(renders_per_episode),
        "--num_envs", str(args.render_num_envs),
        "--skip_initial_step",
        "--resume",
        "--rtx_warmup_frames", "16",
        "--depth_codec", "hevc",
        "--depth_format", "12bit",
        "--depth_gop", "1000",
        "--dr_seeding", "hash",
        "--camera_base", "scene",
    ]
    # ONE flag with every camera after it. render.py declares --skip_cameras as a plain
    # nargs="*" store, so a repeated flag OVERWRITES: passing it twice keeps only the last camera
    # and silently renders the rest.
    argv += ["--flags", str(flags_path.resolve())]
    if args.render_skip_cameras:
        argv += ["--skip_cameras", *args.render_skip_cameras]
    # ALWAYS after --skip_cameras, and never the last thing before --render-extra: nargs="*"
    # would absorb a stray bare token from the extra into the camera list instead of failing
    # with "unrecognized arguments", silently re-enabling a camera the run meant to skip.
    argv += ["--env_spacing", "60"]
    if args.render_skip_failure_episodes:
        argv.append("--skip_failure_episodes")
    if args.render_fail_on_video_defects:
        argv.append("--fail_on_video_defects")
    argv += list(RENDER_SIDECARS)
    argv += render_extra_tokens(args)
    # A pass's IDENTITY goes after the extra. render.py is last-wins, so a shared
    # --render-extra carrying -N or --flags would otherwise rewrite one pass into another:
    # its copies would land in a directory named after a YAML it was not rendered with, and
    # every count would still look right.
    argv += ["--flags", str(flags_path.resolve()), "-N", str(renders_per_episode)]
    return argv


#: Flags that --render-extra must not set locally: each one silently breaks the pipeline's own
#: accounting rather than the render. --output steers a problem out of its render/out/job_* dir, so
#: merge never ships it and the gate reports it unrendered; the episode selectors make the
#: rendered episode set disagree with what the sweep and the parity checks expect.
RENDER_EXTRA_DENIED = ("--output", "--envs_dir", "--episodes_file", "--episode_offset",
                         "--max_episodes")


def render_extra_tokens(args) -> list[str]:
    """--render-extra, shlex-split, for the LOCAL render.py argv.

    Appended last so a user override beats the hardcodes above: render.py declares no append,
    extend or count actions, so every repeated option is last-wins.

    Denied options are dropped with a warning rather than failing the run -- dropping them
    restores the pipeline's own value, which is the correct one, so the render still produces
    the right data in the right place.

    """
    if not getattr(args, "render_extra", None):
        return []
    out: list[str] = []
    for chunk in args.render_extra:
        toks = shlex.split(chunk)
        i = 0
        while i < len(toks):
            tok = toks[i]
            name = tok.split("=", 1)[0]
            # --camera_base data means "use the pose the DATASET records". The local execution
            # stage is parquet-only -- it spawns no cameras, so its output carries no
            # observation.camera_pose.* at all -- and recorded_static_bases then collects nothing
            # and silently falls back to the scene mount. Asking for it locally is never what the
            # caller wanted: either they expect the recorded extrinsics (there are none) or they
            # expect the flags `cameras:` block (which camera_base=data is what CANCELS, on a
            # source that does have poses). Drop it and say so; `scene` is already what the
            # pipeline passes.
            if name == "--camera_base":
                val = tok.split("=", 1)[1] if "=" in tok else (
                    toks[i + 1] if i + 1 < len(toks) else "")
                if val == "data":
                    log.warning(
                        "Render: ignoring --camera_base data from --render-extra. "
                        "The local execution stage records no observation.camera_pose.* (it "
                        "spawns no cameras), so there is no recorded pose to base on; the "
                        "pipeline's --camera_base scene is used",
                    )
                    i += 1 if "=" in tok else 2
                    continue
            if name in RENDER_EXTRA_DENIED:
                skip = 1 if "=" in tok else 2
                log.warning(
                    "Render: ignoring %s from --render-extra; the pipeline owns it "
                    "(a different value silently breaks merge or the episode accounting)", name,
                )
                i += skip
                continue
            out.append(tok)
            i += 1
    return out


def render_job_name(flags: str | Path) -> str:
    """Output directory for one pass, named after its flags YAML.

    Merge globs ``render/out/job_*``, so each pass lands in its own sibling directory and the
    directory name says which look it holds.
    """
    stem = Path(flags).stem
    for prefix in ("render_flags_", "rerender_flags_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    return f"job_{stem}"


def parse_render_pass(spec: str) -> tuple[str, int]:
    """``FLAGS.yaml`` or ``FLAGS.yaml:N`` -> (path, copies per source episode)."""
    path, sep, count = spec.rpartition(":")
    return (path, int(count)) if sep and count.isdigit() else (spec, 1)


def resolve_render_passes(args) -> list[tuple[str, int]]:
    """Every pass this run should produce, as (flags YAML, copies per source episode).

    ``--render-pass`` is the only source: one entry per flag, in the order given.
    """
    passes = [parse_render_pass(s) for s in (getattr(args, "render_pass", None) or [])]
    seen, unique = {}, []
    for flags, copies in passes:
        name = render_job_name(flags)
        if name in seen:
            log.warning("Render: %s and %s would both write to %s; keeping the first",
                        seen[name], flags, name)
            continue
        seen[name] = flags
        unique.append((flags, copies))
    return unique


def local_render_passes(args, base: str) -> list[tuple[str, dict]]:
    """The render passes one problem needs: [(job_dir_name, argv_kwargs), ...].

    Each pass is a SEPARATE render.py invocation into its own sibling job dir rather than extra
    -N iterations, because one process applies one flags dict before the sim is built, and the
    per-wave levers (dome, skybox, exposure) are scene-global: a differently-flagged unit
    sharing a wave would still be lit by that wave's draw.
    """
    return [
        (render_job_name(flags), {"flags_path": Path(flags), "renders_per_episode": copies})
        for flags, copies in resolve_render_passes(args)
    ]


def local_render_steps(args, base: str, envs_dir: str, slug: str,
                         source_dir: Path | None = None) -> list[str]:
    """Shell steps that stage *slug* and render it, appended to its local execution job.

    *source_dir* is the problem's recorded dataset dir (default ``<base>/execution/<slug>``; the
    play pipeline passes ``<base>/play/<slug>``).

    Staging is a one-entry directory of symlinks because ``discover_jobs`` enumerates the
    ``problem_*`` CHILDREN of the path it is given -- it cannot be pointed at a single problem.
    """
    src_root = Path(base) / "render" / "src" / slug
    passes = local_render_passes(args, base)
    out_dirs = [Path(base) / "render" / "out" / job for job, _ in passes]
    link = src_root / slug
    return [
        shlex.join(["mkdir", "-p", str(src_root), *(str(d) for d in out_dirs)]),
        shlex.join(["ln", "-sfn", str(source_dir or Path(base) / "execution" / slug), str(link)]),
        # Through run_local_render.py, never render.py directly: render.py prints
        # "all jobs done" and then hangs in Isaac teardown. Inside a `bash -c a && b` chain
        # that hang blocks run_local_stage forever and the pipeline never reaches merge
        # (observed: 16/16 episodes finalized, then 53 minutes of nothing).
        *(shlex.join([
            sys.executable, "-u", str(RUN_LOCAL_RENDER),
            "--stall-timeout", str(args.render_stall_timeout), "--",
            *local_render_argv(args, src_root, d, envs_dir, **kw),
        ]) for d, (_job, kw) in zip(out_dirs, passes)),
    ]


def render_problem_complete(out_dir: Path, problem: str) -> bool:
    """True iff the renderer finalized *problem* in *out_dir* (as completion.problem_complete)."""
    for split in RENDER_SPLITS:
        rp = out_dir / problem / split / "meta" / "render_params.json"
        if not rp.exists():
            continue
        try:
            with open(rp) as f:
                if json.load(f).get("completed") is True:
                    return True
        except (OSError, ValueError):
            return False
    return False


#: What the MERGE will do with a problem, which is not what the renderer says about it.
RENDER_SHIPPED, RENDER_UNRENDERABLE, RENDER_ABSENT = "shipped", "unrenderable", "absent"


def local_render_state(out_dir: Path, problem: str) -> str:
    """Classify *problem* the way merge sees it, not the way the renderer reports it.

    ``render_problem_complete`` asks render_params for ``completed``, but completed means
    TERMINAL, not HAS DATA. render.py finalizes a problem with ``episodes_recorded: 0``
    (the payload writes the ``total_episodes`` argument out under the key ``episodes_total``;
    there is no ``total_episodes`` key on disk) and an
    ``empty_reason`` when the recorded block size is not in the problem's size table
    (render.py plan_block_sizes -> "bad_size") or when every source parquet is unreadable,
    so that a re-run does not retry it forever. Both write
    ``blocks_success/meta/render_params.json`` and no ``info.json`` -- so the problem reports
    complete while ``merge.get_dataset_paths`` skips it, which is the silent drop every gate
    in this file exists to prevent.

    Three states, because two is not enough: an unrenderable problem is a permanent,
    acknowledged outcome that re-running will never change, so failing the run on it would
    wedge the pipeline on a condition that is routine for a large render.
    """
    completed = None
    completed_split = None
    for split in RENDER_SPLITS:
        rp = out_dir / problem / split / "meta" / "render_params.json"
        if not rp.exists():
            continue
        try:
            with open(rp) as f:
                params = json.load(f)
        except (OSError, ValueError):
            return RENDER_ABSENT
        if params.get("completed") is True:
            completed = params
            completed_split = split
            break
    if completed is None:
        return RENDER_ABSENT
    if (out_dir / problem / "blocks_success" / "meta" / "info.json").exists():
        return RENDER_SHIPPED
    # Finalized, but the merge will not see it. Only an output that declares itself empty is
    # treated as permanent; anything else is ABSENT so the sweep retries it.
    if completed.get("empty_reason") or completed.get("episodes_recorded") == 0:
        return RENDER_UNRENDERABLE
    # A problem that rendered only failures finalizes with blocks_failure alone -- render.py
    # builds job["sources"] from the splits that exist, so no blocks_success/ is created and
    # render_params lands under blocks_failure. merge reads blocks_success only, so this can
    # never ship, and re-running cannot change it: the SOURCE has no successful episodes.
    # Permanent, like the empty cases above -- not ABSENT, which would ask the sweep to retry a
    # problem that can never satisfy it. (Failure-only problems are routine, not exotic.)
    if completed_split != "blocks_success":
        return RENDER_UNRENDERABLE
    return RENDER_ABSENT


def merge_shipped_problems(base_dir: Path) -> list[str]:
    """Problem names a merge of *base_dir* would actually ship, via merge's OWN discovery.

    Asking merge_datasets.py rather than globbing ``problem_*`` keeps the pipeline's idea of
    "in the dataset" from drifting from what the merge does, and it is what makes the gates
    below safe to fail on. A glob is not: the execution stage ``mkdir``s each problem's output
    dir BEFORE running its job, so every crashed execution leaves an empty ``problem_*`` dir
    behind for good, and a gate quantified over the glob would refuse to merge the other
    forty-nine problems forever over a directory the merge would never have read.
    """
    if not base_dir.is_dir():
        return []
    # Same path dance as scripts/datagen/merge_datasets.py, for runs where ToyBlocks is not
    # installed in the interpreter.
    src = str(Path(__file__).resolve().parents[2] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from toyblocks.vecgen.recording.merge import (  # noqa: PLC0415
        get_dataset_paths,
        problem_name_from_dataset_path,
    )
    return [problem_name_from_dataset_path(p) for p in get_dataset_paths(base_dir)]


class LocalRenderScheduler:
    """Queues each problem's local render the moment its execution finalizes.

    The coupled schedule appends the render to the execution job's own
    shell chain, so the render holds the slot its execution just used. This pipelined mode
    (``--render-schedule pipelined``) makes the render an independent item in the SAME slot pool,
    eligible as soon as execution N finalizes, so it can start while execution N+1 is still
    running whenever a slot is free.

    Dispatch is exactly-once in-process: ``_seen`` means a problem is enqueued at most once,
    so two renders can never share an output dir. That is load-bearing, not tidiness --
    ``render.py --resume`` ``shutil.rmtree``s a non-finalized ``<out>/<problem>``, so a
    double dispatch would delete a live writer's output instead of failing.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        envs_dir: str,
        execution_dir: Path,
        render_dir: Path,
    ) -> None:
        self._args = args
        self._envs_dir = envs_dir
        self._execution = execution_dir
        self._src_root = render_dir / "src"
        self._passes = local_render_passes(args, str(render_dir.parent))
        self._out_dirs = [render_dir / "out" / job for job, _ in self._passes]
        #: The DR pass. Kept as a scalar because the rest of the class, and the tests, key on it.
        self.out_dir = self._out_dirs[0]
        self.feed = JobFeed()
        self._lock = threading.Lock()
        self._pending = 0             # queued-or-running items that still owe an on_done
        self._seen: set[str] = set()  # slugs already dealt with (queued, or already rendered)
        self._staged: list[str] = []  # staged problem dir names == render.py output names
        self._skipped: list[str] = []  # deliberately not rendered; still must be REPORTED

    def render_cmd(self, slug: str) -> list[str]:
        """One problem's command: the hang guard wrapping the shared render argv.

        The single seam the tests override, so they can exercise the scheduler without Isaac.
        """
        one = [[
            sys.executable, "-u", str(RUN_LOCAL_RENDER),
            "--stall-timeout", str(self._args.render_stall_timeout), "--",
            *local_render_argv(self._args, self._src_root / slug, d, self._envs_dir, **kw),
        ] for d, (_job, kw) in zip(self._out_dirs, self._passes)]
        if len(one) == 1:
            return one[0]
        # Both passes as ONE item: the scheduler's exactly-once guarantee is per problem, and a
        # problem is only finished when every pass it owes has landed.
        return ["bash", "-c", " && ".join(shlex.join(c) for c in one)]

    def _enqueue(self, slug: str, *, first: bool) -> bool:
        """Stage and queue *slug*. The caller holds the lock. False if there is nothing to do."""
        if slug in self._seen:
            return False
        self._seen.add(slug)
        linked = stage_render_source(self._execution, self._src_root / slug, [slug])
        if not linked:
            log.warning("Render: %s has no finalized dataset; not rendering", slug)
            self._skipped.append(slug)
            return False
        self._staged += linked
        if all(self._state_of(d) != RENDER_ABSENT for d in linked):
            log.info("Render: %s is already rendered; skipping", slug)
            return False
        self._pending += 1
        self.feed.add(f"{RENDER_ITEM_PREFIX}{slug}", self.render_cmd(slug), first=first)
        return True

    def arm(self, exec_jobs: list[tuple[str, Any]], slugs: list[str]) -> JobFeed:
        """Load the execution items, plus renders for problems an EARLIER run executed."""
        with self._lock:
            for name, build in exec_jobs:
                self._pending += 1
                self.feed.add(name, build)
            running = {name for name, _ in exec_jobs}
            # Problems the execution stage skipped because .execution_complete already exists
            # would otherwise never be rendered -- and once render_ran is set, merge reads
            # ONLY render/out/job_*, so they would vanish from the dataset with no error.
            for slug in slugs:
                if slug not in running and (self._execution / slug).is_dir():
                    self._enqueue(slug, first=False)
            if self._pending == 0:
                self.feed.close()
        return self.feed

    def on_done(self, name: str, ok: bool) -> None:
        """``run_local_stage`` worker-thread callback. Enqueues only; never renders inline."""
        try:
            with self._lock:
                if not name.startswith(RENDER_ITEM_PREFIX):
                    if ok:
                        # Increment happens inside the same lock hold as the decrement below,
                        # so _pending cannot transiently hit zero while a render is still owed.
                        self._enqueue(name, first=True)
                    else:
                        # Not rendered on purpose -- but it must still be REPORTED, or a problem
                        # whose execution finalized its dataset and then exited non-zero (the
                        # routine Isaac teardown failure) is absent from both `done` and
                        # `missing`, report() claims success, the stage marker is set, and merge
                        # -- which reads ONLY render/out -- silently drops it.
                        log.warning(
                            "Render: execution of %s failed; not rendering it", name
                        )
                        self._seen.add(name)
                        self._skipped.append(name)
                self._pending -= 1
                drained = self._pending == 0
        except Exception:
            log.exception("Render: scheduling failed after %s", name)
            with self._lock:
                self._pending -= 1
                drained = self._pending == 0
        if drained:
            self.feed.close()

    def sweep(self, slugs: list[str], slots: list[str | None], log_dir: Path) -> None:
        """Render anything the execution stage did not hand over (resume, or exec disabled)."""
        self.feed = JobFeed()
        with self._lock:
            self._pending = 0
            for slug in base_problem_slugs(slugs):
                # Shards are collapsed for DISPATCH, so the base dir may not exist at all:
                # an execution preempted on its first attempt leaves only problem_x__s001.
                # stage_render_source globs the siblings, so staging still works.
                if (self._execution / slug).is_dir() or any(
                    self._execution.glob(f"{slug}__s[0-9][0-9][0-9]")
                ):
                    try:
                        self._enqueue(slug, first=False)
                    except OSError:
                        # Staging touches the filesystem; one bad symlink must not abort the
                        # run after the execution stage has already been marked complete.
                        log.exception("Render: could not stage %s", slug)
                        self._skipped.append(slug)
            if self._pending == 0:
                self.feed.close()
                return
        run_local_stage("Render", self.feed, slots, log_dir, on_done=self.on_done)

    def _state_of(self, problem: str) -> str:
        """Worst state across every pass this run owes for *problem*.

        With more than one --render-pass a problem is only SHIPPED when EVERY pass is on disk:
        merge globs render/out/job_*, so a half-finished problem would otherwise contribute
        fewer copies than were promised and nothing would report it.
        ABSENT beats UNRENDERABLE beats SHIPPED.
        """
        states = [local_render_state(d, problem) for d in self._out_dirs]
        for worst in (RENDER_ABSENT, RENDER_UNRENDERABLE):
            if worst in states:
                return worst
        return RENDER_SHIPPED

    def report(self) -> tuple[list[str], list[str], list[str]]:
        """(shipped, unrenderable, absent) over every problem this scheduler saw.

        Shards are reported individually -- they are collapsed for DISPATCH only, since
        ``problem_x`` rendering says nothing about whether ``problem_x__s001`` did.

        `absent` includes problems deliberately skipped (failed execution, no finalized
        dataset), not just renders that ran and failed: from the merge's point of view the
        two are the same event -- episodes that will not be in the dataset. `unrenderable`
        is NOT a failure; see local_render_state.
        """
        staged = list(dict.fromkeys(self._staged))
        state = {d: self._state_of(d) for d in staged}
        shipped = [d for d in staged if state[d] == RENDER_SHIPPED]
        unrenderable = [d for d in staged if state[d] == RENDER_UNRENDERABLE]
        absent = [d for d in staged if state[d] == RENDER_ABSENT]
        absent += [d for d in dict.fromkeys(self._skipped) if d not in state]
        return shipped, unrenderable, absent


# Env-generation seed (the reset-state sampler's), fixed so reruns rebuild the same packages.
ENV_GEN_SEED = 0


def list_pddls(tasks_dir: str) -> list[Path]:
    """Return the PDDL problems in *tasks_dir*, sorted by file name."""
    pddls = sorted(Path(tasks_dir).glob("*.pddl"))
    if not pddls:
        log.warning("No PDDL problems found in %s", tasks_dir)
    return pddls


def task_slug(pddl: Path) -> str:
    """Slug used for the per-problem output folders (matches ``task_slug_from_pddl_path``)."""
    return pddl.stem.replace("-", "_")


def robot_init_flags(args: argparse.Namespace) -> list[str]:
    """--randomize-robot-init and its options for vecgen_jobs.py / make_superset_env.py ([] when off)."""
    if not args.randomize_robot_init:
        return []
    return [
        "--randomize-robot-init",
        "--robot-init-ee-x-range",
        *(str(v) for v in args.robot_init_ee_x_range),
        "--robot-init-ee-y-range",
        *(str(v) for v in args.robot_init_ee_y_range),
        "--robot-init-ee-z-min",
        str(args.robot_init_ee_z_min),
        "--robot-init-candidates",
        str(args.robot_init_candidates),
        "--robot-init-gripper-closed-prob",
        str(args.robot_init_gripper_closed_prob),
    ]


def detect_local_gpus() -> list[str]:
    """GPU ids usable by this process, honouring an existing CUDA_VISIBLE_DEVICES."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None:
        return [tok.strip() for tok in visible.split(",") if tok.strip()]
    try:
        listing = subprocess.run(
            ["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        log.warning("Could not query GPUs via nvidia-smi; running one problem at a time")
        return []
    return [str(i) for i, line in enumerate(listing.splitlines()) if line.strip()]


class JobFeed:
    """A job list :func:`run_local_stage` may keep pulling from while the stage is running.

    A plain list is still accepted everywhere; this exists for the one case where finishing a
    job creates another one -- the local render of a problem whose execution just finalized.

    Render items are served BEFORE queued execution items. At one slot that reproduces the
    coupled chain's exec/render interleaving, so at most one un-rendered execution dataset sits
    on disk at a time; at two slots the render simply takes the free slot while the next
    execution keeps running.
    """

    _RENDER, _EXECUTION, _SENTINEL = 0, 1, 2

    def __init__(self, jobs: Iterable[tuple[str, Any]] = ()) -> None:
        self._q: queue.PriorityQueue = queue.PriorityQueue()
        self._lock = threading.Lock()
        self._seq = 0
        self._total = 0
        self._closed = False
        for name, build in jobs:
            self.add(name, build)

    def add(
        self,
        name: str,
        build: list[str] | Callable[[int], list[str]],
        *,
        first: bool = False,
    ) -> bool:
        """Queue one job. ``first`` puts it ahead of every execution item still waiting."""
        with self._lock:
            if self._closed:
                # The caller has already counted this job as pending; a silent drop here
                # would leave the ledger non-zero and the feeder loop waiting forever.
                log.error("JobFeed: %s was added after close and will not run", name)
                return False
            self._seq += 1
            self._total += 1
            prio = (self._RENDER if first else self._EXECUTION, self._seq)
        self._q.put((*prio, name, build))
        return True

    def close(self) -> None:
        """Stop the feed once everything already queued has been handed out."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._seq += 1
            seq = self._seq
        self._q.put((self._SENTINEL, seq, None, None))

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def __iter__(self):
        """Yield jobs until the feed is closed and drained. Safe for several consumers.

        The sentinel is put BACK before returning, so every worker sees it and stops; it is
        the lowest priority, so it is only ever reached once the real work is gone.
        """
        while True:
            item = self._q.get()
            if item[0] == self._SENTINEL:
                self._q.put(item)
                return
            yield item[2], item[3]


def run_local_stage(
    label: str,
    jobs: list[tuple[str, list[str] | Callable[[int], list[str]]]] | JobFeed,
    slots: list[str | None],
    log_dir: Path,
    *,
    on_done: Callable[[str, bool], None] | None = None,
) -> None:
    """Run per-problem jobs across GPU slots, continuing past failures.

    Every slot owns one ``CUDA_VISIBLE_DEVICES`` value, so ``len(slots)`` problems
    run at a time. A job is a command, or a ``build_cmd(slot)`` callable for jobs
    that bind a port and need one their concurrent neighbours are not using.
    Output is streamed when there is a single slot and written per problem under
    *log_dir* otherwise, since concurrent Isaac Sim logs are unreadable interleaved.

    A single failing problem does not stop the stage; the failure count is reported at the
    end.

    *jobs* may be a :class:`JobFeed`, in which case the stage keeps running until the feed is
    closed and every job it yielded has finished. ``on_done(name, ok)`` is then called on the
    worker thread once a job has finished AND its slot has been returned; it must only
    ENQUEUE, never run anything and never wait for a slot, or it deadlocks the pool. A
    callback that raises is logged, not propagated.
    """
    if not isinstance(jobs, JobFeed) and not jobs:
        log.warning("%s: nothing to run", label)
        return

    # True when two items can actually be live at once, which is what per-slot log files and
    # per-slot env isolation are for. A JobFeed's length is not knowable up front (it grows
    # while the stage runs), so any feed with >1 slot counts; a plain list keeps the old
    # behaviour exactly, so a multi-GPU box running a single problem still streams to the
    # console instead of silently redirecting to a log file.
    parallel = len(slots) > 1 and (isinstance(jobs, JobFeed) or len(jobs) > 1)
    if parallel:
        log_dir.mkdir(parents=True, exist_ok=True)
    available = [str(s) for s in slots if s is not None]
    log.info(
        "%s: %s problem(s) on %s%s",
        label,
        # A feed's total grows while the stage runs, so announce what is queued so far.
        f"{jobs.total}+" if isinstance(jobs, JobFeed) else len(jobs),
        f"GPUs [{', '.join(available)}]" if available else "default GPU",
        f"; logs in {log_dir}" if parallel else "",
    )

    failed: list[str] = []
    failed_lock = threading.Lock()
    index_lock = threading.Lock()
    next_index = 0

    def run_job(slot: int, name: str, build: list[str] | Callable[[int], list[str]]) -> None:
        nonlocal next_index
        with index_lock:
            next_index += 1
            index = next_index
        gpu_id = slots[slot]
        # Bound before the try: build(slot) or open() raising would otherwise leave `result`
        # unbound and surface as UnboundLocalError out of future.result(), aborting the whole
        # stage -- the opposite of this function's continue-past-failures contract.
        rc = 1
        try:
            cmd = build(slot) if callable(build) else build
            env = os.environ.copy()
            if gpu_id is not None:
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
            if parallel:
                # NOTE: these three are NOT read by the installed Isaac Sim (grepping the whole
                # isaacsim tree finds zero references, while a real one like CARB_APP_PATH finds
                # 14). They are kept because they cost nothing and other Kit builds do read them,
                # but they must NOT be treated as isolation: concurrent slots still share
                # ~/.cache/ov, ~/.local/share/ov and ~/.nvidia-omniverse.
                env["KIT_CACHE_ROOT"] = f"/tmp/kit-cache-{slot}"
                env["KIT_LOG_DIR"] = f"/tmp/kit-logs-{slot}"
                env["OMNI_USER_CONFIG_DIR"] = f"/tmp/omni-config-{slot}"
                # This one IS read (_video_encoder.resolve_encode_threads). Without it each
                # Isaac process sizes its encode pool from the whole CPU set, so N concurrent
                # slots spin N x ncpu encoder threads over the same cores. Use sched_getaffinity,
                # exactly as the encoder's own default does -- os.cpu_count() ignores taskset and
                # cgroup limits, and would hand out a cap LARGER than the default it replaces.
                try:
                    _ncpu = len(os.sched_getaffinity(0))
                except (AttributeError, OSError):
                    _ncpu = os.cpu_count() or 8
                env.setdefault("TMP3_ENCODER_THREADS", str(max(1, _ncpu // len(slots))))
            total = jobs.total if isinstance(jobs, JobFeed) else len(jobs)
            log.info("[%d/%d] %s: %s on GPU %s", index, total, label, name, gpu_id or "default")
            log.info("[sched] %s start %s slot=%d t=%.3f", label, name, slot, time.time())
            if parallel:
                with open(log_dir / f"{name}.log", "w") as log_file:
                    result = run(
                        cmd, check=False, env=env, stdout=log_file, stderr=subprocess.STDOUT
                    )
            else:
                result = run(cmd, check=False, env=env)
            rc = result.returncode
        except Exception:
            log.exception("FAILED %s: %s (the job raised before it could run)", label, name)

        log.info("[sched] %s end %s slot=%d rc=%d t=%.3f", label, name, slot, rc, time.time())
        if rc != 0:
            detail = f" (see {log_dir / f'{name}.log'})" if parallel else ""
            log.warning("FAILED %s: %s%s", label, name, detail)
            with failed_lock:
                failed.append(name)
        if on_done is not None:
            # Must not propagate: a raising callback would abort the stage through
            # future.result() even though the job itself succeeded.
            try:
                on_done(name, rc == 0)
            except Exception:
                log.exception("%s: completion callback raised for %s", label, name)

    feed = jobs if isinstance(jobs, JobFeed) else JobFeed(jobs)
    if not isinstance(jobs, JobFeed):
        feed.close()

    def worker(slot: int) -> None:
        """One slot, pulling the next job only when it is free.

        Deliberately NOT `for job in feed: pool.submit(...)`: submitting eagerly drains the
        whole feed into the pool's own FIFO queue at once, which both destroys the feed's
        priority ordering (a render queued later lands behind every execution already
        submitted) and removes all back-pressure. Pulling per-slot is what makes `first=True`
        mean anything.
        """
        for name, build in feed:
            run_job(slot, name, build)

    with ThreadPoolExecutor(max_workers=len(slots)) as pool:
        futures: list[Future[None]] = [pool.submit(worker, i) for i in range(len(slots))]
        for future in futures:
            future.result()

    total = feed.total
    if failed:
        log.warning(
            "%s: %d/%d problem(s) failed: %s", label, len(failed), total, " ".join(failed)
        )
    else:
        log.info("%s: all %d problem(s) succeeded", label, total)


STEPS = [
    ("task_gen", "Generate PDDL tasks"),
    ("env", "Generate env packages + reset states"),
    ("exec", "Run vecgen execution"),
    ("render", "Render finished execution"),
    ("merge", "Merge successful datasets (rendered if render ran)"),
    ("annotate", "Annotate merged dataset (target xy)"),
]

# --resume bookkeeping: a stage that finished writes a marker here, and anything
# without a marker is considered partial, so its outputs are wiped before rerun.
STAGE_STATE_DIR = ".pipeline_state"


def stage_marker(base: str, stage: str) -> Path:
    return Path(base) / STAGE_STATE_DIR / f"{stage}.done"


def mark_stage_complete(base: str, stage: str) -> None:
    marker = stage_marker(base, stage)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"stage": stage, "completed_at": datetime.now().isoformat()}, indent=2)
    )
    log.info("Stage '%s' complete (marker: %s)", stage, marker)


def _rmtree_retry(target: Path, retries: int = 5, delay: float = 2.0) -> None:
    """rmtree with retries to handle network-filesystem metadata propagation delays."""
    for attempt in range(retries):
        try:
            shutil.rmtree(target)
            return
        except OSError as e:
            if e.errno == 39 and attempt < retries - 1:  # ENOTEMPTY
                import time

                log.warning(
                    "rmtree failed (attempt %d/%d, ENOTEMPTY), retrying in %.1fs...",
                    attempt + 1,
                    retries,
                    delay,
                )
                time.sleep(delay)
            else:
                raise


def clear_stage(base: str, stage: str, outputs: list[str]) -> None:
    """Delete a stage's outputs and marker so it reruns from scratch."""
    for path in outputs:
        target = Path(path)
        if target.exists():
            log.info("Resume: deleting partial output %s", target)
            _rmtree_retry(target)
    stage_marker(base, stage).unlink(missing_ok=True)


def apply_resume(
    base: str,
    enabled: dict[str, bool],
    stage_outputs: dict[str, list[str]],
    steps: list[tuple[str, str]] | None = None,
) -> None:
    """Disable stages already completed in *base* and clean the ones that rerun.

    A no-op when *base* does not exist. Otherwise the first ENABLED stage without
    a completion marker is where the pipeline picks up: everything before it is
    skipped, and every stage from there on has its outputs deleted first, since
    a stage interrupted halfway leaves partial data that must not be reused.

    Stages this run disables never write a marker, so they must not decide where
    the run resumes: --vecgen-problems-per-env turns env off, and counting such missing markers resumed
    every vecgen run at 'env' and deleted a finished execution/
    (oct26_stack3_ori_vg_1000x32, 2026-10-27).
    """
    if not Path(base).is_dir():
        log.info("Resume: %s does not exist yet; running the full pipeline", base)
        return

    steps = STEPS if steps is None else steps
    pending = next(
        (i for i, (key, _) in enumerate(steps) if enabled[key] and not stage_marker(base, key).exists()),
        len(steps),
    )
    if pending == len(steps):
        log.info("Resume: all stages already complete in %s", base)

    for i, (key, label) in enumerate(steps):
        if i < pending:
            if enabled[key]:
                log.info("Resume: skipping completed stage '%s' (%s)", key, label)
            enabled[key] = False
        elif enabled[key]:
            clear_stage(base, key, stage_outputs[key])


# Supported type tags for interactive editing:
#   str   – plain string
#   int   – integer (empty → None)
#   float – float (empty → None)
#   [str] – space-separated list of strings
#   [float] – space-separated list of floats
PARAM_GROUPS: list[tuple[str, str | None, list[tuple[str, str, str]]]] = [
    (
        "General",
        None,
        [
            ("experiment_id", "experiment-id", "str"),
            ("output_root", "output-root", "str"),
            ("dataset_folder", "dataset-folder", "str"),
        ],
    ),
    (
        "Task generation",
        "task_gen",
        [
            ("task_gen_script", "task-gen-script", "str"),
            ("task_gen_count", "task-gen-count", "int"),
            ("task_gen_seed", "task-gen-seed", "int"),
            ("block_size_range", "block-size-range", "[float]"),
            ("table_height_range", "table-height-range", "[float]"),
            ("friction_range", "friction-range", "[float]"),
            ("num_distractors", "num-distractors", "int"),
        ],
    ),
    (
        "Env",
        "env",
        [
            ("num_reset_states", "num-reset-states", "int"),
            ("num_envs", "num-envs", "int"),
            ("block_x_range", "block-x-range", "str"),
            ("block_y_range", "block-y-range", "str"),
            ("num_scene_profiles", "num-scene-profiles", "int"),
            ("env_per_problem", "env-per-problem", "int"),
            ("randomize_robot_init", "randomize-robot-init", "bool"),
            ("robot_init_ee_x_range", "robot-init-ee-x-range", "[float]"),
            ("robot_init_ee_y_range", "robot-init-ee-y-range", "[float]"),
            ("robot_init_ee_z_min", "robot-init-ee-z-min", "float"),
            ("robot_init_candidates", "robot-init-candidates", "int"),
            ("robot_init_gripper_closed_prob", "robot-init-gripper-closed-prob", "float"),
        ],
    ),
    (
        "Execution",
        "exec",
        [
            ("exec_num_envs", "exec-num-envs", "int"),
            ("seed", "seed", "int"),
            ("vecgen_episodes", "vecgen-episodes", "int"),
            ("vecgen_problems_per_env", "vecgen-problems-per-env", "int"),
            ("vecgen_record_videos", "vecgen-record-videos", "bool"),
        ],
    ),
    (
        "Render",
        "render",
        [
            ("render_pass", "render-pass", "[str]"),
            ("render_skip_cameras", "render-skip-cameras", "[str]"),
            ("render_skip_failure_episodes", "render-skip-failure-episodes", "bool"),
            ("render_extra", "render-extra", "[str]"),
        ],
    ),
    (
        "Merge",
        "merge",
        [
            ("merge_cameras", "merge-cameras", "[str]"),
            ("merge_num_workers", "merge-num-workers", "int"),
            ("merge_output_dir", "merge-output-dir", "str"),
        ],
    ),
    (
        "Annotate",
        "annotate",
        [
            ("annotate_num_workers", "annotate-num-workers", "int"),
        ],
    ),
]


def _format_value(val: object) -> str:
    if val is None:
        return "(none)"
    if isinstance(val, list):
        return " ".join(str(v) for v in val)
    return str(val)


def _parse_value(raw: str, type_tag: str) -> object:
    """Convert a raw string to the appropriate Python type."""
    if type_tag == "bool":
        return raw.lower() in ("true", "1", "yes") if raw else False
    if type_tag == "int":
        return int(raw) if raw else None
    if type_tag == "float":
        return float(raw) if raw else None
    if type_tag == "[float]":
        return [float(x) for x in raw.split()]
    if type_tag == "[str]":
        return raw.split()
    return raw or None


def prompt_steps(
    enabled: dict[str, bool], steps: list[tuple[str, str]] | None = None
) -> dict[str, bool]:
    """Show step checklist and let the user toggle entries."""
    steps = STEPS if steps is None else steps
    print()
    print(
        "Pipeline steps (enter numbers to toggle, 'a' for all, 'n' for none, or Enter to confirm):"
    )
    print("-" * 60)
    keys = [k for k, _ in steps]
    while True:
        for i, (key, label) in enumerate(steps, 1):
            mark = "x" if enabled[key] else " "
            print(f"  [{mark}] {i}. {label}")
        print()
        ans = input("Toggle> ").strip().lower()
        if ans == "":
            break
        if ans == "a":
            for k in keys:
                enabled[k] = True
            continue
        if ans == "n":
            for k in keys:
                enabled[k] = False
            continue
        for token in ans.replace(",", " ").split():
            try:
                idx = int(token) - 1
                if 0 <= idx < len(keys):
                    enabled[keys[idx]] = not enabled[keys[idx]]
                else:
                    print(f"  (ignored out-of-range: {token})")
            except ValueError:
                print(f"  (ignored: {token})")
    return enabled


def prompt_params(
    args: argparse.Namespace,
    enabled: dict[str, bool],
    param_groups: list[tuple[str, str | None, list[tuple[str, str, str]]]] | None = None,
) -> None:
    """Show parameters for enabled steps and let the user edit them."""
    param_groups = PARAM_GROUPS if param_groups is None else param_groups
    visible: list[tuple[str, list[tuple[str, str, str]]]] = []
    for group_label, step_key, params in param_groups:
        if step_key is None or enabled.get(step_key, False):
            visible.append((group_label, params))

    if not visible:
        return

    attr_index: dict[str, tuple[str, str]] = {}
    for _, params in visible:
        for attr, display, type_tag in params:
            attr_index[display] = (attr, type_tag)

    print()
    print("Parameters (type 'name=value' to modify, or Enter to confirm):")
    print("-" * 60)
    while True:
        for group_label, params in visible:
            print(f"\n  {group_label}:")
            for attr, display, _ in params:
                val = getattr(args, attr)
                print(f"    {display:24s} = {_format_value(val)}")
        print()
        ans = input("Edit> ").strip()
        if ans == "":
            break
        eq = ans.find("=")
        if eq == -1:
            print("  (use 'name=value' format)")
            continue
        name = ans[:eq].strip()
        raw = ans[eq + 1 :].strip()
        if name not in attr_index:
            print(f"  (unknown parameter: {name})")
            continue
        attr, type_tag = attr_index[name]
        try:
            val = _parse_value(raw, type_tag)
            setattr(args, attr, val)
            print(f"  {name} -> {_format_value(val)}")
        except ValueError as exc:
            print(f"  (invalid value: {exc})")


class _RemovedFlag(argparse.Action):
    """A removed option kept only so it fails LOUDLY with a migration message.

    Without an exact option, argparse abbreviates: ``--render-n 16`` would match the only
    option starting with that prefix, ``--render-num-envs``, and the run would render one copy
    per episode with 16 envs per wave instead of 16 copies. That looks like a successful run
    with a silently wrong dataset size.
    """

    def __init__(self, option_strings, dest, message="", **kwargs):
        super().__init__(option_strings, dest, nargs="?", default=None,
                         help=argparse.SUPPRESS, **kwargs)
        self._message = message

    def __call__(self, parser, namespace, values, option_string=None):
        parser.error(f"{option_string} has been removed: {self._message}")


def add_render_args(parser: argparse.ArgumentParser) -> None:
    """The ``--render*`` flags (shared with run_playdata_pipeline.py).

    ``--render-pass`` is the only flag that turns the stage on. Every other flag here applies to
    EVERY pass the run produces, so a run cannot give two passes different cameras or schedules.
    """
    parser.add_argument(
        "--render-pass",
        action="append",
        metavar="FLAGS.yaml[:N]",
        help="Render one pass with this flags YAML, optionally N copies of every source "
        "episode. Repeat the flag for more looks: each pass renders the same episodes and "
        "lands in its own job_<name> directory, and the merge takes them all. This is the only "
        "flag that turns the render stage on. src/toyblocks/domain_randomization/flags/ ships "
        "render_flags_plain.yaml for a plain render and the allrand files for visual domain "
        "randomization",
    )
    parser.add_argument(
        "--render-n",
        action=_RemovedFlag,
        message="copies per source episode are the ':N' suffix on --render-pass, as in "
                "--render-pass FLAGS.yaml:3",
    )
    parser.add_argument(
        "--render-schedule",
        choices=["pipelined", "coupled"],
        default=None,
        help="How each render is scheduled against the executions. Applies to every pass. Both "
        "produce identical pixels, so this is a throughput and observability choice. "
        "'coupled' (default) appends the render to that problem's execution job, holding one "
        "slot across both phases: faster with more than one worker. 'pipelined' queues the "
        "render as its own item in the shared GPU-slot pool as soon as that problem's execution "
        "finalizes, so it can overlap the next execution and gets its own log, timing and exit "
        "status, at some throughput cost",
    )
    parser.add_argument(
        "--render-skip-cameras",
        nargs="*",
        default=["left_camera"],
        help="Camera names not to render, in every pass (default: left_camera; pass the flag "
             "with no names to render every camera)",
    )
    parser.add_argument(
        "--render-skip-failure-episodes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Render only blocks_success, in every pass, as the merge keeps only that split "
             "anyway (default: on)",
    )
    parser.add_argument(
        "--render-extra",
        nargs="*",
        default=[],
        metavar="ARG",
        help="Extra arguments appended to every pass's render.py command line; they override "
        "the pipeline's defaults (--extra_rgb 224 --extra_depth rg8_55_224 --extra_rgb_GOP 2 "
        "--rgb_GOP 2). Each pass re-applies its own --flags and -N afterwards, so an extra "
        "cannot rewrite one pass into another. "
        "Use the = FORM, "
        "quoted as one string: --render-extra=\"--extra_rgb 224 --extra_depth g12_224 "
        "--depth_gop 1000\". The '=' is not optional for a BARE flag -- argparse reads a "
        "space-separated value starting with '-' as an option of its own, so "
        "--render-extra \"--skip_failure_episodes\" is REJECTED while "
        "--render-extra=\"--skip_failure_episodes\" works. "
        "This is how the video sidecars are requested: --extra_rgb 224 writes a 224x224 RGB "
        "copy per camera (the training sidecar), --extra_depth g12_224 the depth one "
        "(lossless gray12le HEVC, short per-camera GOP; rg8_224 is the older, ~6x larger "
        "variant -- both decode to identical depth). "
        "Neither has a dedicated --render-* flag.",
    )
    parser.add_argument(
        "--render-num-envs",
        type=int,
        default=8,
        help="Envs per render wave, in every pass. This is a FLOOR, not a preference: "
        "plan_block_sizes skips the problem unless this covers every block-size slot in the env "
        "package (8 by default). 8 keeps the encoder's RSS survivable beside Isaac",
    )
    parser.add_argument(
        "--render-stall-timeout",
        type=int,
        default=3600,
        help="SIGKILL a render that prints nothing for this long. Success is "
        "decided by the 'all jobs done' marker and the on-disk result, never by exit status, "
        "because render.py hangs in Isaac teardown after finishing its work",
    )
    parser.add_argument(
        "--render-fail-on-video-defects",
        action="store_true",
        help="Fail a problem whose encoder could not publish every video "
             "(default: off). Without it the problem finalizes as completed with "
             "status=incomplete_videos and merges with parquet rows whose mp4 is absent, which a "
             "path-resolving dataloader reads as zeros. Off by default because a failed problem "
             "renders whole (scripts/render/fast_patch.py fills video holes afterwards)",
    )

def add_merge_args(parser: argparse.ArgumentParser) -> None:
    """The ``--merge-*`` / ``--annotate-*`` flags (shared with run_playdata_pipeline.py)."""
    parser.add_argument(
        "--merge-cameras",
        nargs="+",
        default=["all"],
        help="Cameras to keep during merge; 'all' keeps every video feature any problem declares. "
             "The old default was an explicit six-key list, which SILENTLY DROPPED the sidecars "
             "--render-extra had just produced: --extra_rgb 224 / --extra_depth rg8_224 write "
             "observation.images.camera:<cam>_224 and observation.depth.camera:<cam>_rg8_224, "
             "neither of which was on the list, so the merged dataset lost exactly the streams "
             "HVLA trains on. merge.wants_all_cameras carries the same warning. The list "
             "also named left_camera, which --render-skip-cameras left_camera never renders. "
             "Name keys explicitly only when you intend to drop everything else.",
    )
    parser.add_argument(
        "--merge-num-workers", type=int, default=16, help="Parallel workers for merge (default: 16)"
    )
    parser.add_argument(
        "--merge-allow-missing-problems",
        action="store_true",
        help="Merge: ship the dataset even though problems that executed are absent from it "
             "(default: off -- the run fails instead). Their episodes are gone from the merged "
             "dataset for good, because once the render stage is complete merge reads only "
             "render/out/job_*",
    )
    parser.add_argument(
        "--merge-union-schema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="merge_datasets.py --union_schema: declare the union of the problems' columns (e.g. "
        "object poses of blocks only some scenes have) so LeRobotDataset can load a merge whose "
        "problems differ in their blocks. --no-merge-union-schema merges on the intersection.",
    )
    parser.add_argument(
        "--merge-output-dir",
        default=None,
        help="Output dir for merged dataset (default: <output-root>/datasets/merged/train/<experiment-id>)",
    )
    parser.add_argument(
        "--merge-depth-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Extract depth videos to .npy during merge for fast dataloader access (default: enabled)",
    )
    parser.add_argument(
        "--annotate-num-workers",
        type=int,
        default=64,
        help="Parallel workers for annotation (default: 64)",
    )


def build_parser() -> argparse.ArgumentParser:
    """The pipeline's CLI, factored out of main() so tests can read the real defaults."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiment-id", required=True, help="Experiment name (e.g. jun23_rotate)"
    )
    parser.add_argument(
        "--interactive", "-i", action="store_true", help="Interactively choose which steps to run"
    )
    parser.add_argument(
        "--local-gpus",
        nargs="+",
        type=int,
        default=None,
        metavar="ID",
        help="GPU ids the stages may use, e.g. --local-gpus 0 1 2 3 (default: every GPU "
        "visible to this process). Each step runs one problem per GPU",
    )
    parser.add_argument(
        "--local-gpu-workers",
        type=int,
        default=1,
        metavar="N",
        help="Concurrent problems per GPU (default 1, a strict no-op). Applies to "
        "every local stage. Measured on one RTX 5090 at --render-num-envs 8: a "
        "render peaks at 7.7 GiB of VRAM and 76%% GPU utilisation, so 2 fits a 32 GiB card "
        "and leaves compute headroom. Host RAM, not VRAM, is the binding constraint, and an "
        "OOM mid-encode corrupts videos behind a completed=true marker -- so raise this only "
        "with the output validated afterwards",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue a previous run of this experiment id: skip stages already marked "
        "complete and rerun the first unfinished stage from scratch (deleting its partially written "
        "outputs). A no-op if the experiment directory does not exist",
    )
    parser.add_argument("--skip-task-gen", action="store_true", help="Skip PDDL task generation")
    parser.add_argument(
        "--skip-env", action="store_true", help="Skip env package generation"
    )
    parser.add_argument(
        "--skip-execution", action="store_true", help="Skip vecgen execution"
    )
    parser.add_argument("--skip-merge", action="store_true", help="Skip dataset merging")
    parser.add_argument(
        "--skip-annotate",
        "--skip-annotate-unmerged",
        dest="skip_annotate",
        action="store_true",
        help="Skip annotation of the merged dataset (annotate_target_xy.py)",
    )
    add_render_args(parser)
    parser.add_argument(
        "--output-root",
        default="data",
        help="Root for every dataset the pipeline writes: <root>/datasets/<dataset-folder>/<exp> "
        "and the merged <root>/datasets/merged/train/<exp> (default: data, relative to the "
        "working directory)",
    )
    parser.add_argument("--dataset-folder", default="unmerged")
    parser.add_argument(
        "--task-gen-script",
        default="scripts/task_gen/stack_three.py --train",
        help="Task generation script, optionally with its own flags, e.g. "
        '"scripts/task_gen/stack_three.py --num-blocks 4 6" '
        "(default: scripts/task_gen/stack_three.py --train)",
    )
    parser.add_argument("--task-gen-count", type=int, default=1000)
    parser.add_argument("--task-gen-seed", type=int, default=6303)
    parser.add_argument(
        "--block-size-range",
        type=float,
        nargs=2,
        default=[0.03, 0.05],
        metavar=("MIN", "MAX"),
        help="Block size range (default: 0.03 0.05)",
    )
    parser.add_argument(
        "--table-height-range",
        type=float,
        nargs=2,
        default=[-0.05, 0.05],
        metavar=("MIN", "MAX"),
        help="Table height range (default: -0.05 0.05)",
    )
    parser.add_argument(
        "--friction-range",
        type=float,
        nargs=2,
        default=[0.5, 0.5],
        metavar=("MIN", "MAX"),
        help="Friction range (default: 0.5 0.5)",
    )
    parser.add_argument(
        "--num-distractors", type=int, default=0, help="Number of distractors per problem"
    )
    # Removed: use --env-per-problem. Kept registered so passing it reports that.
    parser.add_argument("--num-reset-states", type=int, default=None, help=argparse.SUPPRESS)
    # Removed: the pipeline only runs locally. Registered as an exact option so a bare
    # --local reports that instead of matching --local-gpus by prefix.
    parser.add_argument("--local", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--block-x-range",
        type=str,
        default="",
        metavar="SPEC",
        help="Table-frame x window for on-table block spawns. One interval "
        '("[-0.2, 0.58]") or disjoint intervals ("[[-0.2,-0.1],[0.3,0.5]]"), which '
        "spawn nothing in the gap. Overrides ;block_x_range in the PDDL. Empty keeps "
        "the PDDL value, else the sampler default [-0.1, 0.25]. NOTE: offsets are from "
        "the table origin, whose robot-frame position depends on the scene preset; the "
        "usable window is capped by MAX_REACH_RADIUS (0.65 m from the robot base), so "
        "large positive x is mostly unreachable. Stage 1 prints the usable fraction.",
    )
    parser.add_argument(
        "--block-y-range",
        type=str,
        default="",
        metavar="SPEC",
        help="Table-frame y window for on-table block spawns. Same format as "
        "--block-x-range. Empty keeps the PDDL value, else the default [-0.30, 0.30].",
    )
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument(
        "--num-scene-profiles",
        type=int,
        default=8,
        help="Block-size / table-height / friction profiles baked into each env package. "
        "num-envs and exec-num-envs must each be a multiple of this (default: 8)",
    )
    parser.add_argument("--env-per-problem", type=int, default=32)
    parser.add_argument(
        "--randomize-robot-init",
        action="store_true",
        default=False,
        help="Give each reset state a random collision-free arm configuration and gripper state",
    )
    parser.add_argument(
        "--robot-init-ee-x-range",
        type=float,
        nargs=2,
        default=[0.25, 0.65],
        metavar=("MIN", "MAX"),
        help="Env: end-effector x range (m, robot base frame) for robot init poses (default: 0.25 0.65)",
    )
    parser.add_argument(
        "--robot-init-ee-y-range",
        type=float,
        nargs=2,
        default=[-0.35, 0.35],
        metavar=("MIN", "MAX"),
        help="Env: end-effector y range (m, robot base frame) for robot init poses (default: -0.35 0.35)",
    )
    parser.add_argument(
        "--robot-init-ee-z-min",
        type=float,
        default=0.20,
        help="Env: lowest end-effector height (m); the upper bound is the home pose height (default: 0.2)",
    )
    parser.add_argument(
        "--robot-init-candidates",
        type=int,
        default=32,
        help="Env: end-effector poses tried per env before keeping the home pose (default: 32)",
    )
    parser.add_argument(
        "--robot-init-gripper-closed-prob",
        type=float,
        default=0.5,
        help="Env: probability a reset state starts with the gripper closed (default: 0.5)",
    )
    parser.add_argument(
        "--vecgen-episodes",
        type=int,
        default=None,
        help="vecgen: episodes per problem (default: --env-per-problem, one per reset state)",
    )
    parser.add_argument(
        "--vecgen-exclude-actions",
        default="push_to_right,push_next_to",
        help="vecgen: comma list of domain actions FF may never plan, also when re-planning "
        "(e.g. reorient,rotate for pick-and-place-only data)",
    )
    parser.add_argument(
        "--vecgen-problems-per-env",
        type=int,
        default=8,
        help="vecgen: PDDLs sharing one multi-problem env package / process, built inside the "
        "vecgen jobs (no env stage); 1 = one "
        "stage-1 env package per problem (default: 8)",
    )
    parser.add_argument(
        "--vecgen-record-videos",
        action="store_true",
        help="vecgen: also record the scene cameras as LeRobot videos (slower: the physics loop renders)",
    )
    parser.add_argument(
        "--vecgen-pool-slots",
        type=int,
        default=32,
        help="vecgen: cuRobo planner pool slots per process (default: 32)",
    )
    parser.add_argument(
        "--exec-num-envs",
        type=int,
        default=128,
        help="Execution: parallel Isaac Sim environments (default: 128)",
    )
    parser.add_argument("--seed", type=int, default=42)
    add_merge_args(parser)
    parser.add_argument(
        "--randomize-scene",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Scene randomization (block size, table height, friction) over the ranges above",
    )
    parser.add_argument(
        "--domain-path",
        default="src/toyblocks/tamp/blocks",
        help="Path to the TAMP domain (default: src/toyblocks/tamp/blocks)",
    )
    parser.add_argument(
        "--usd-library",
        default="train_blocks_large",
        help="Glyph asset library for USD block resolution at codegen time "
        "(e.g. train_blocks_large). Empty uses the default (train_blocks_large). Forwarded to the env stage.",
    )

    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override params for a fast smoke test: one problem, no distractors, "
        "fixed block size/table height/friction, 16 envs",
    )
    parser.add_argument(
        "--single-task-gen",
        action="store_true",
        help="Generate a single task with 8960 envs and no distractors",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.smoke_test:
        args.num_distractors = 0
        args.num_envs = 16
        args.env_per_problem = 16
        args.exec_num_envs = 16
        args.task_gen_count = 1

    if args.single_task_gen:
        args.task_gen_count = 1
        args.env_per_problem = 8960
        args.num_distractors = 0

    if args.smoke_test or not args.randomize_scene:
        args.block_size_range = [0.04, 0.04]
        args.table_height_range = [0.0, 0.0]
        args.friction_range = [0.4, 0.4]

    exp = args.experiment_id

    # ── Resolve which steps are enabled ────────────────────────────────
    enabled = {
        "task_gen": not args.skip_task_gen,
        "env": not args.skip_env,
        "exec": not args.skip_execution,
        "render": bool(resolve_render_passes(args)),
        "merge": not args.skip_merge,
        "annotate": not args.skip_annotate,
    }

    if args.interactive:
        enabled = prompt_steps(enabled)
        prompt_params(args, enabled)
        exp = args.experiment_id
        # Unticking the stage here must also drop the passes. The coupled schedule appends the
        # render to each execution job from resolve_render_passes, NOT from enabled["render"],
        # so leaving the passes in place would burn the GPU rendering and then merge the
        # un-rendered execution tree at exit 0.
        if not enabled["render"]:
            args.render_pass = []
        elif not resolve_render_passes(args):
            log.error("Render: the stage is ticked but no --render-pass was given, so there is "
                      "nothing to render. Pass --render-pass FLAGS.yaml[:N]")
            return 1

    # ── Validate parameters ──────────────────────────────────────────
    if args.local:
        log.error(
            "--local has been removed: this pipeline only runs locally, so there is no "
            "backend to select. Drop the flag. (--local-gpus and --local-gpu-workers still "
            "choose which GPUs to use and how many problems to run on each.)"
        )
        return 1
    if args.num_reset_states is not None:
        log.error(
            "--num-reset-states has never had an effect and has been removed: the env stage "
            "passes --env-per-problem to the generator as its reset-state count. Use "
            "--env-per-problem (default %d) instead.", parser.get_default("env_per_problem"),
        )
        return 1
    if args.render_schedule is None:
        args.render_schedule = "coupled"
    # The one thing the rest of the run branches on.
    args.render_coupled = args.render_schedule == "coupled"

    if enabled["render"]:
        # Recording in-physics video and then overwriting it with a render pays the slow
        # physics-render cost for frames nothing reads.
        if args.vecgen_record_videos:
            log.error(
                "the render supplies the videos; drop --vecgen-record-videos "
                "(vecgen stays parquet-only and the render renders the cameras)"
            )
            return 1
        # A --render-pass YAML that is not readable here would otherwise fail inside the
        # render, after the Kit startup.
        # is_file() RAISES PermissionError (not False) for a path under an unreadable dir.
        for flags, _copies in resolve_render_passes(args):
            try:
                readable = Path(flags).is_file()
            except OSError:
                readable = False
            if not readable:
                log.error(
                    "the render needs a flags YAML readable on THIS machine; %s is not",
                    flags,
                )
                return 1
        # --render-num-envs is a FLOOR: render maps env i to block-size slot
        # i % len(table), so fewer envs than the package has profiles leaves the high slots
        # without an env and plan_block_sizes SKIPS the problem -- every problem, silently
        # enough that the run just produces no videos. Refuse up front instead.
        if args.render_num_envs < args.num_scene_profiles:
            log.error(
                "--render-num-envs (%d) must be at least --num-scene-profiles (%d): "
                "the render spawns env i at block-size slot i %% %d, so %d envs leave the "
                "remaining slots without an env and every problem is skipped",
                args.render_num_envs,
                args.num_scene_profiles,
                args.num_scene_profiles,
                args.render_num_envs,
            )
            return 1

    if enabled["env"] and args.env_per_problem % args.num_envs != 0:
        log.error(
            "env-per-problem (%d) must be divisible by num-envs (%d)",
            args.env_per_problem,
            args.num_envs,
        )
        return 1

    # The generated env package asserts this at every instantiation, so execution
    # inherits the constraint from the profile count baked in at codegen time.
    for stage, flag, envs in (
        ("env", "num-envs", args.num_envs),
        ("exec", "exec-num-envs", args.exec_num_envs),
    ):
        if (
            enabled[stage]
            and envs >= args.num_scene_profiles
            and envs % args.num_scene_profiles != 0
        ):
            log.error(
                "%s (%d) must be a multiple of num-scene-profiles (%d)",
                flag,
                envs,
                args.num_scene_profiles,
            )
            return 1

    # Recorded before the default is filled in, so the grouped path below can tell an
    # explicit count from an unset one.
    args.vecgen_episodes_explicit = args.vecgen_episodes is not None
    if args.vecgen_episodes is None:
        args.vecgen_episodes = args.env_per_problem
    if args.vecgen_problems_per_env > 1:
        if args.env_per_problem % args.num_envs or args.exec_num_envs % args.num_scene_profiles:
            log.error("multi-problem packages: --num-envs must divide --env-per-problem and "
                      "--num-scene-profiles must divide --exec-num-envs")
            return 1
        if enabled["env"]:
            log.info("--vecgen-problems-per-env: no env stage (the vecgen jobs build the "
                     "multi-problem packages)")
        enabled["env"] = False

    # ── Pin git commit for reproducibility across stages ─────────────
    # Passed as --branch to the stage scripts (which accept branch names or SHAs).
    toyblocks_commit_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    log.info("Pinned commit: ToyBlocks=%s", toyblocks_commit_sha)

    # ── Paths ──────────────────────────────────────────────────────────
    # Absolute: the render staging symlinks and the per-stage subprocesses must not depend
    # on the working directory a relative --output-root was given in.
    output_root = os.path.abspath(args.output_root)
    base = f"{output_root}/datasets/{args.dataset_folder}/{exp}"
    tasks_dir = f"{base}/tasks"
    envs_dir = f"{base}/envs"
    execution_data_dir = f"{base}/execution"
    local_logs_dir = Path(base) / "local_logs"
    render_dir = Path(base) / "render"
    merge_output_dir = os.path.abspath(args.merge_output_dir or f"{output_root}/datasets/merged/train/{exp}")

    log.info("Experiment: %s", exp)
    log.info("Base path:  %s", base)

    # ── Resume: skip completed stages, wipe the partial ones ─────────
    if args.resume:
        apply_resume(
            base,
            enabled,
            {
                "task_gen": [tasks_dir],
                "env": [envs_dir],
                "exec": [execution_data_dir, str(render_dir)],
                # render.py --resume skips every problem whose render_params.json says
                # completed, so a rerun of this stage renders only what is missing.
                "render": [],
                "merge": [merge_output_dir],
                # Annotation edits the merged dataset in place; rerunning it
                # re-annotates.
                "annotate": [],
            },
        )

    # ── GPU fan-out ───────────────────────────────────────────────────
    # One problem per GPU slot (--local-gpus x --local-gpu-workers); every stage runs single-GPU
    # jobs and scales out by spreading problems over more slots.
    local_gpus: list[str] = []
    local_slots: list[str | None] = [None]
    local_gpus = (
        [str(gpu) for gpu in args.local_gpus] if args.local_gpus else detect_local_gpus()
    )
    # A slot is one concurrent problem, so repeating a GPU id oversubscribes that card.
    # Per-slot separation is weaker than it looks: the KIT_*/OMNI_* vars run_local_stage sets
    # are not read by the installed Isaac, so concurrent slots DO share the Omniverse caches
    # and user config. Only TMP3_ENCODER_THREADS actually takes effect. Validate the output
    # of any run above 1 worker.
    workers = max(1, args.local_gpu_workers)
    if local_gpus:
        local_slots = [g for g in local_gpus for _ in range(workers)]
    elif workers > 1:
        local_slots = [None] * workers
    log.info(
        "Local mode: %s%s",
        f"{len(local_gpus)} GPU(s) [{', '.join(local_gpus)}]"
        if local_gpus
        else "no GPUs detected",
        f", {workers} concurrent problem(s) per GPU ({len(local_slots)} slots)"
        if workers > 1
        else "",
    )

    # ── Read robot_id / scene_preset_id from the env template ─────────
    template_cfg = _read_template_env_cfg(args.domain_path)
    log.info(
        "Template env cfg: robot_id=%s, scene_preset_id=%s",
        template_cfg["robot_id"],
        template_cfg["scene_preset_id"],
    )

    # ── Save pipeline config for reproducibility ─────────────────────
    config = {
        **{
            k: v
            for k, v in vars(args).items()
            if isinstance(v, (str, int, float, bool, list, type(None)))
        },
        "experiment_id": exp,
        "toyblocks_commit_sha": toyblocks_commit_sha,
        "scene_preset_id": template_cfg["scene_preset_id"],
        "robot_id": template_cfg["robot_id"],
        "enabled_steps": enabled,
    }
    Path(base).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config_path = Path(base) / f"run_datagen_pipeline_config_{timestamp}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Saved pipeline config to %s", config_path)

    # ── Task generation ───────────────────────────────────────────────
    if enabled["task_gen"]:
        log.info("=" * 60)
        log.info("Task generation: Generate PDDL tasks")
        log.info("=" * 60)
        # Split so the script can carry its own flags, e.g.
        # --task-gen-script "scripts/task_gen/stack_three.py --num-blocks 4 6".
        # A bare path splits to a single element, so existing values are unaffected.
        run(
            [
                sys.executable,
                *shlex.split(args.task_gen_script),
                "--count",
                str(args.task_gen_count),
                "--out-dir",
                tasks_dir,
                "--seed",
                str(args.task_gen_seed),
                "--block-size-range",
                str(args.block_size_range[0]),
                str(args.block_size_range[1]),
                "--table-height-range",
                str(args.table_height_range[0]),
                str(args.table_height_range[1]),
                "--friction-range",
                str(args.friction_range[0]),
                str(args.friction_range[1]),
                "--num_distractors",
                str(args.num_distractors),
            ]
        )
        mark_stage_complete(base, "task_gen")

    # ── Env: build the env packages + reset states ───────────────────
    if enabled["env"]:
        log.info("=" * 60)
        log.info("Env: Generate envs + reset states locally")
        log.info("=" * 60)
        Path(envs_dir).mkdir(parents=True, exist_ok=True)
        env_jobs: list[tuple[str, list[str] | Callable[[int], list[str]]]] = []
        for pddl in list_pddls(tasks_dir):
            slug = task_slug(pddl)
            if (Path(envs_dir) / slug / "reset_states.pkl").exists():
                log.info("Skipping %s (reset_states.pkl exists)", slug)
                continue
            env_jobs.append((slug, local_env_argv(args, pddl, envs_dir)))
        run_local_stage("Env", env_jobs, local_slots, local_logs_dir / "env")

    if enabled["env"]:
        mark_stage_complete(base, "env")

    # ── Render submitter (shared by the execution + render stages) ─
    local_render: LocalRenderScheduler | None = None
    if enabled["render"]:
        local_render = LocalRenderScheduler(
            args, envs_dir, Path(execution_data_dir), render_dir
        )

    # ── Execution ──────────────────────────────────────────────────
    if enabled["exec"]:
        log.info("=" * 60)
        log.info("Execution: Run vecgen locally (Isaac Sim)")
        log.info("=" * 60)
        exec_jobs: list[tuple[str, list[str] | Callable[[int], list[str]]]] = []
        group = args.vecgen_problems_per_env
        vecgen_video_flags = ["--record-videos", "--enable_cameras"] if args.vecgen_record_videos else []
        pddls_all = list_pddls(tasks_dir)
        for g0 in range(0, len(pddls_all) if group > 1 else 0, group):
            # multi-problem package per group: build it (unless present), then stream it
            members = pddls_all[g0 : g0 + group]
            # "problem_" prefix: merge_datasets.py takes problem_*/blocks_success as the layout
            slug = f"problem_grp_{task_slug(members[0])}"
            env_path = Path(envs_dir) / slug
            out_dir = Path(execution_data_dir) / slug
            if (out_dir / ".execution_complete").exists():
                log.info("Skipping %s (already collected)", slug)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            build = [
                sys.executable,
                "scripts/vecgen/make_superset_env.py",
                "--pddls",
                *(str(m) for m in members),
                "--out-dir",
                envs_dir,
                "--name",
                slug,
                "--domain-path",
                args.domain_path,
                "--num-reset-states",
                str(args.env_per_problem),
                "--num-envs",
                str(args.num_envs),
                "--num-scene-profiles",
                str(args.num_scene_profiles),
                "--seed",
                str(args.seed + g0),
                "--headless",
            ] + (["--usd-library", args.usd_library] if args.usd_library else []) + robot_init_flags(args)
            if args.block_x_range:
                build += ["--block-x-range", args.block_x_range]
            if args.block_y_range:
                build += ["--block-y-range", args.block_y_range]
            stream = [
                sys.executable,
                "scripts/vecgen/run.py",
                "--env-path",
                str(env_path),
                "--domain",
                f"{args.domain_path}/domain.pddl",
                "--num-envs",
                str(args.exec_num_envs),
                "--episodes",
                # One episode per reset state across the group, unless a count was given.
                str(args.vecgen_episodes if args.vecgen_episodes_explicit
                    else len(members) * args.env_per_problem),
                "--pool-slots",
                str(args.vecgen_pool_slots),
                "--record",
                str(out_dir),
                "--mark-complete",
                "--seed",
                str(args.seed),
                "--exclude-actions",
                args.vecgen_exclude_actions,
                "--headless",
            ] + vecgen_video_flags
            steps = ([] if (env_path / "problems.json").exists() else [shlex.join(build)])
            rr = local_render_steps(args, base, envs_dir, slug) if args.render_coupled else []
            exec_jobs.append(
                (slug, ["bash", "-c", " && ".join([*steps, shlex.join(stream), *rr])])
            )
        for pddl in pddls_all if group == 1 else []:
            slug = task_slug(pddl)
            out_dir = Path(execution_data_dir) / slug
            env_path = Path(envs_dir) / slug
            if not env_path.is_dir():
                log.info("Skipping %s (no env at %s)", slug, env_path)
                continue
            if (out_dir / ".execution_complete").exists():
                log.info("Skipping %s (already collected)", slug)
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                "scripts/vecgen/run.py",
                "--env-path",
                str(env_path),
                "--pddl",
                str(pddl),
                "--domain",
                f"{args.domain_path}/domain.pddl",
                "--num-envs",
                str(args.exec_num_envs),
                "--episodes",
                str(args.vecgen_episodes),
                "--pool-slots",
                str(args.vecgen_pool_slots),
                "--record",
                str(out_dir),
                "--mark-complete",
                "--seed",
                str(args.seed),
                "--exclude-actions",
                args.vecgen_exclude_actions,
                "--headless",
            ] + vecgen_video_flags
            if args.render_coupled:
                # Same shape as the grouped path so the render runs in the SAME job, holding
                # the GPU slot across both phases instead of queueing behind every execution.
                rr = local_render_steps(args, base, envs_dir, slug)
                exec_jobs.append((slug, ["bash", "-c", " && ".join([shlex.join(cmd), *rr])]))
            else:
                exec_jobs.append((slug, cmd))
        if local_render is not None and not args.render_coupled:
            # Coupled mode is excluded deliberately: its render already runs inside each job's
            # shell chain (local_render_steps), so arming the scheduler as well would queue a
            # SECOND render.py against the same render/out/job_<name>/<slug>. --resume
            # resolves that by shutil.rmtree-ing whichever has not finalized -- i.e. by deleting
            # a live writer's output, the hazard _seen exists for. Coupled takes the plain-list
            # branch below and meets the scheduler again only in the post-stage sweep.
            # Pipelined: the execution items and their renders share one slot pool, and a
            # problem's render is queued by on_done the moment its execution finalizes.
            # For the UNGROUPED path slugs_all is every task slug, which also recovers problems
            # an earlier invocation already executed (the stage skips those on .execution_complete)
            # so they are pipelined rather than left to the post-stage sweep. For the GROUPED path
            # it is just the job list, so there is nothing extra to recover here and the sweep
            # picks up any already-executed group from disk instead.
            slugs_all = (
                [s_ for s_, _ in exec_jobs] if group > 1 else [task_slug(p_) for p_ in pddls_all]
            )
            run_local_stage(
                "Execution",
                local_render.arm(exec_jobs, slugs_all),
                local_slots,
                local_logs_dir / "execution",
                on_done=local_render.on_done,
            )
        else:
            run_local_stage("Execution", exec_jobs, local_slots, local_logs_dir / "execution")

        # No render report here. It used to live in this block and quantify over exec_jobs,
        # which is this invocation's job list: a problem the loops above skipped on
        # .execution_complete, or on a missing env dir, was in neither `done` nor `missing`, and
        # the marker was latched by `if done:` -- ANY problem rendering was enough. Merge then
        # read only render/out and dropped the rest with a warning and exit 0. Both local modes
        # now report from the post-stage sweep below, whose universe is the disk.

    if enabled["exec"]:
        mark_stage_complete(base, "exec")

    # ── Render: render anything the execution stage did not hand over ───
    if local_render is not None:
        log.info("=" * 60)
        log.info("Render: Render finished execution locally")
        log.info("=" * 60)
        # From disk, not from this invocation's exec_jobs: apply_resume DISABLES the exec
        # stage when it resumes at 'render', so on a resumed run the block above never ran,
        # and in coupled mode the execution loops skip every problem that already finished.
        # The DISPATCH universe stays the raw glob so nothing that is rendered today stops
        # being rendered; what may FAIL the run is narrowed to `expected` below.
        slugs_all = [
            d.name for d in sorted(Path(execution_data_dir).glob("problem_*")) if d.is_dir()
        ]
        local_render.sweep(slugs_all, local_slots, local_logs_dir / "render")
        shipped, unrenderable, absent = local_render.report()

        # Only a problem the merge would have shipped from execution/ can be LOST by merging
        # render/out instead, so only those may fail the run. Asking merge's own discovery
        # (rather than globbing problem_*) is what keeps this gate from dead-ending on the
        # empty problem dirs the execution stage mkdirs before every job: a crashed execution
        # leaves one behind for good, and it holds nothing the merge would ever have read.
        expected = set(merge_shipped_problems(Path(execution_data_dir)))
        lost = [d for d in absent if d in expected]
        no_loss = [d for d in absent if d not in expected]
        log.info(
            "Render: %d problem(s) rendered, %d unrenderable, %d missing",
            len(shipped), len(unrenderable), len(lost),
        )
        if unrenderable:
            # Terminal and acknowledged -- render.py finalized these empty on purpose and a
            # re-run will produce the same result, so failing on them would wedge the pipeline.
            log.warning(
                "Render: %d problem(s) can never render (empty source or block size "
                "not in the size table) and carry no episodes: %s",
                len(unrenderable), " ".join(unrenderable[:20]),
            )
        if no_loss:
            log.info(
                "Render: %d problem dir(s) had nothing the merge would ship; ignoring:"
                " %s", len(no_loss), " ".join(no_loss[:20]),
            )
        if not shipped:
            if not expected:
                # Distinguish "nothing to render" from "the renders did not work": the first
                # means the execution stage produced no mergeable dataset at all, and telling
                # the operator to re-run the render would send them in the wrong direction.
                log.error(
                    "Render: %s holds no finalized blocks_success dataset, so there "
                    "is nothing to render or merge. The execution stage produced no usable "
                    "episodes; check its logs under %s",
                    execution_data_dir, local_logs_dir / "execution",
                )
            else:
                log.error(
                    "Render: nothing was rendered, though %d problem(s) executed. "
                    "Merge reads only render/out once this stage is marked complete, so "
                    "there is no dataset to ship. Re-run to render them; to merge the raw "
                    "un-rendered execution tree on purpose, re-run without "
                    "--render-pass.", len(expected),
                )
            return 1
        if lost:
            # Not a warning: once this stage is marked complete, merge reads ONLY
            # render/out/job_*, so every problem listed here is silently absent from the
            # dataset. Exiting non-zero and leaving the marker unset makes a later --resume
            # render them instead of shipping a quietly truncated dataset.
            log.error(
                "Render: %d of %d problem(s) have a finalized execution dataset but "
                "no render, and would be MISSING from the merge: %s",
                len(lost), len(expected), " ".join(lost[:20]),
            )
            if not args.merge_allow_missing_problems:
                log.error(
                    "Render: refusing to mark the stage complete; re-run to finish "
                    "them, or pass --merge-allow-missing-problems to merge what rendered"
                )
                return 1
            log.warning(
                "Render: --merge-allow-missing-problems is set, merging %d problem(s) "
                "and dropping %d", len(shipped), len(lost)
            )
        mark_stage_complete(base, "render")

    # ── Merge successful datasets ─────────────────────────────────
    # Rendered episodes replace the execution data whenever the render
    # stage ran, in this invocation or an earlier (resumed) one.
    render_ran = enabled["render"] or stage_marker(base, "render").exists()
    if enabled["merge"]:
        log.info("=" * 60)
        log.info(
            "Merge: Merge successful datasets (%s)", "rendered" if render_ran else "execution"
        )
        log.info("=" * 60)
        if render_ran:
            merge_input_dirs = sorted(
                str(d) for d in (render_dir / "out").glob("job_*") if d.is_dir()
            )
            if not merge_input_dirs:
                log.error("Merge: render ran but no outputs found under %s", render_dir / "out")
                return 1
            # Name them. The glob is deliberately wide (every job_* dir), but
            # that means a job dir an earlier run wrote is merged again whether or not this
            # run asked for it -- most easily by dropping a --render-pass from an otherwise
            # identical command, which silently keeps shipping that pass's copies and leaves
            # the dataset larger than the command says.
            log.info("Merge: input job dirs: %s",
                     " ".join(Path(d).name for d in merge_input_dirs))
            owed = {render_job_name(f) for f, _ in resolve_render_passes(args)}
            # A pass this run asked for that is not on disk must never be merged around.
            # --resume disables every stage before the first unfinished one, so adding a
            # --render-pass to a run whose render marker already exists skips the render and
            # would ship only the older passes, at a size the command does not describe.
            present = {Path(d).name for d in merge_input_dirs}
            if owed - present:
                log.error(
                    "Merge: %s was asked for but never rendered, so the merged dataset would be "
                    "smaller than this command describes. Re-run without --resume, or delete "
                    "%s to re-run the render stage.",
                    " ".join(sorted(owed - present)), stage_marker(base, "render"),
                )
                return 1
            # `owed` is empty on a merge-only or resumed run, where every job dir is legitimately
            # from an earlier invocation; warning there would cry wolf on the normal case.
            stale = sorted(
                Path(d).name for d in merge_input_dirs
                if owed and Path(d).name not in owed
            )
            if stale:
                log.warning(
                    "Merge: %s left by an earlier run and NOT produced by this one; it is "
                    "being merged anyway because merge reads every job_* dir. Delete it to "
                    "ship only what this command rendered: %s",
                    " ".join(stale), " ".join(str(render_dir / "out" / n) for n in stale),
                )
        else:
            merge_input_dirs = [f"{base}/execution"]
        merge_cmd = [
            sys.executable,
            "scripts/datagen/merge_datasets.py",
            "--input_dirs",
            *merge_input_dirs,
            "--output_dir",
            merge_output_dir,
            "--cameras",
            *args.merge_cameras,
            "--num_workers",
            str(args.merge_num_workers),
            # Problems differing in their blocks need the union schema to stay loadable.
            *(["--union_schema"] if args.merge_union_schema else []),
        ]

        run(merge_cmd)
        shutil.copy(config_path, Path(merge_output_dir) / config_path.name)
        log.info("Copied pipeline config to %s", Path(merge_output_dir) / config_path.name)
        mark_stage_complete(base, "merge")

    # ── Annotate the merged dataset ───────────────────────────────
    # The annotator projects goals into the front camera and refuses a dataset that records
    # none, which is what vecgen writes unless it records videos or the episodes were
    # rendered.
    if enabled["annotate"] and not render_ran and not args.vecgen_record_videos:
        log.warning("Annotate: skipped, the dataset records no camera (pixel goals need "
                    "--vecgen-record-videos or --render-pass)")
        enabled["annotate"] = False
    if enabled["annotate"]:
        log.info("=" * 60)
        log.info("Annotate: Run annotate_target_xy.py on the merged dataset")
        log.info("=" * 60)
        if not Path(merge_output_dir).is_dir():
            log.error("Annotate: merged dataset not found at %s", merge_output_dir)
            return 1
        # The annotator exits non-zero when it wrote no goal columns; without
        # them the dataset trains unconditioned, so treat that as a failure.
        result = run(
            [
                sys.executable,
                "src/toyblocks/data_annotation/annotate_target_xy.py",
                merge_output_dir,
                "-j",
                str(args.annotate_num_workers),
            ],
            check=False,
        )
        if result.returncode != 0:
            log.error("Annotation FAILED (exit %d) for %s", result.returncode, merge_output_dir)
            return 1
        mark_stage_complete(base, "annotate")

    # ── Plot sample episode (always runs after merge) ──────────────
    plot_dataset = merge_output_dir
    if Path(plot_dataset).exists():
        log.info("=" * 60)
        log.info("Plot: Plot sample episode from merged dataset")
        log.info("=" * 60)
        plot_output = str(Path(plot_dataset) / "episode_0.png")
        run(
            [
                sys.executable,
                "scripts/datagen/plot_episode.py",
                "--dataset",
                plot_dataset,
                "--episode",
                "0",
                "--output",
                plot_output,
            ]
        )
        log.info("Plot saved to %s", plot_output)

    log.info("=" * 60)
    log.info("Pipeline complete!")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
