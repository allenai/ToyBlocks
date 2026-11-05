#!/usr/bin/env python3
"""Run the TAMPER datagen pipeline over states captured during evaluation.

Picks up where ``scripts/evaluate.py --tamper_output_dir`` leaves off: takes the
directory of captured failure states and drives the vecgen expert (planning +
execution + recording), annotation and merging on them.  The eval itself is
*not* run here — it needs a policy and a live sim, so it stays a separate step
(see ``docs/tamper.md``).  Nothing here needs the TAMP submodule.

Each step runs on this machine via subprocess, one problem at a time.

Each TAMPER problem directory is a multi-problem vecgen env package (a clone of
the eval's env package, the captured ``reset_states.pkl`` and a ``problems.json``
naming each state's goal and blocks), so ``scripts/vecgen/run.py`` points straight
at it: ``--init-from-scene`` plans every episode with FF from the captured scene
toward its recorded goal, ``--reset-state-ids`` runs the selected states once each.

Usage::

    # Local, everything
    python scripts/datagen/run_tamper_pipeline.py --tamper-dir outputs/tamper/round1

    # Only failures (skip timeouts), no merge
    python scripts/datagen/run_tamper_pipeline.py --tamper-dir outputs/tamper/round1 \\
        --reasons failure --skip-merge
"""

from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

SUMMARY_FILENAME = "tamper_summary.json"
FAILURES_FILENAME = "tamper_failures.json"
PROBLEMS_INDEX_FILENAME = "problems.json"


class Problem:
    """One TAMPER problem directory and the instances selected from it.

    ``pddl_problem_file`` is set only when the directory has no vecgen problem index
    (``problems.json``) and ``--pddl-problem-file`` supplies the one problem instead.
    """

    def __init__(self, slug: str, path: Path, pddl_problem_file: str | None,
                 instance_ids: list[int], reasons: dict[str, int],
                 num_scene_profiles: int) -> None:
        self.slug = slug
        self.path = path
        self.pddl_problem_file = pddl_problem_file
        self.instance_ids = instance_ids
        self.reasons = reasons
        self.num_scene_profiles = num_scene_profiles

    @property
    def data_name(self) -> str:
        """Recording subdir: merge_datasets.py only collects ``problem_*/blocks_success``."""
        return self.slug if self.slug.startswith("problem_") else f"problem_{self.slug}"


def run(cmd: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    """Run a command, logging it first."""
    log.info("Running: %s", " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=check)


def load_problems(
    tamper_dir: Path,
    *,
    reasons: list[str] | None,
    max_instances: int | None,
    pddl_override: str | None,
) -> list[Problem]:
    """Read the TAMPER summary and select instances from each problem.

    Raises:
        SystemExit: the directory is not a TAMPER output, or a problem has no
            PDDL problem file and none was supplied.
    """
    summary_path = tamper_dir / SUMMARY_FILENAME
    if not summary_path.exists():
        log.error("%s not found — is %s a TAMPER output directory?", SUMMARY_FILENAME, tamper_dir)
        sys.exit(1)

    summary = json.loads(summary_path.read_text())
    problems: list[Problem] = []

    for slug, info in sorted(summary.get("problems", {}).items()):
        problem_dir = Path(info["dir"])
        if not problem_dir.is_absolute():
            problem_dir = tamper_dir / slug
        # Prefer the on-disk directory: a TAMPER tree copied elsewhere has stale
        # absolute paths baked into the summary from the machine that wrote it.
        local_dir = tamper_dir / slug
        if local_dir.exists():
            problem_dir = local_dir

        failures_path = problem_dir / FAILURES_FILENAME
        if not failures_path.exists():
            log.warning("%s: no %s, skipping", slug, FAILURES_FILENAME)
            continue
        meta = json.loads(failures_path.read_text())

        selected = [
            entry for entry in meta.get("failures", [])
            if reasons is None or entry.get("reason") in reasons
        ]
        has_index = (problem_dir / PROBLEMS_INDEX_FILENAME).exists()
        if pddl_override is None:
            if not has_index:
                log.error(
                    "%s: no %s (no PDDL problem or goal was recorded: eval ran without "
                    "--queue_url, or captured before the vecgen port) and no "
                    "--pddl-problem-file given; cannot plan for it",
                    slug, PROBLEMS_INDEX_FILENAME,
                )
                sys.exit(1)
            # instances whose episode recorded no problem have no problems.json entry
            plannable = [e for e in selected if e.get("vecgen_problem")]
            if len(plannable) < len(selected):
                log.warning("%s: %d instance(s) have no recorded problem, skipping them",
                            slug, len(selected) - len(plannable))
            selected = plannable
        if max_instances is not None:
            selected = selected[:max_instances]
        if not selected:
            log.warning("%s: no instances match reasons=%s, skipping", slug, reasons)
            continue

        counts: dict[str, int] = {}
        for entry in selected:
            counts[entry.get("reason", "?")] = counts.get(entry.get("reason", "?"), 0) + 1

        problems.append(Problem(
            slug=slug,
            path=problem_dir,
            pddl_problem_file=pddl_override,
            instance_ids=[int(e["instance_id"]) for e in selected],
            reasons=counts,
            num_scene_profiles=int(meta.get("num_scene_profiles") or 1),
        ))

    if not problems:
        log.error("No usable problems found under %s", tamper_dir)
        sys.exit(1)
    return problems


# ----------------------------------------------------------------------------
# vecgen command
# ----------------------------------------------------------------------------
def num_envs_for(problem: Problem, requested: int) -> int:
    """``requested`` (at most one env per selected state) rounded up to a multiple of the
    package's scene-profile count, so the round-robin spawn has an env in every slot a
    captured state may need."""
    k = max(1, problem.num_scene_profiles)
    n = min(requested, len(problem.instance_ids))
    return max(k, -(-n // k) * k)


def vecgen_command(
    args: argparse.Namespace, problem: Problem, data_dir: str, python: str
) -> list[str]:
    """``scripts/vecgen/run.py`` over one TAMPER dir: FF from each captured scene toward
    its recorded goal, the selected states once each, recorded to ``data_dir``."""
    cmd = [
        python, "scripts/vecgen/run.py",
        "--env-path", str(problem.path),
        "--domain", f"{args.domain_path}/domain.pddl",
        "--init-from-scene",
        "--reset-state-ids", *[str(i) for i in problem.instance_ids],
        "--episodes", str(len(problem.instance_ids)),
        "--num-envs", str(num_envs_for(problem, args.exec_num_envs)),
        "--seed", str(args.seed),
        "--record", data_dir,
        "--mark-complete",
        "--headless",
    ]
    if problem.pddl_problem_file:
        cmd += ["--pddl", problem.pddl_problem_file]
    if args.record_videos:
        cmd += ["--record-videos", "--enable_cameras"]
    return cmd + shlex.split(args.vecgen_args)


# ----------------------------------------------------------------------------
# Local mode
# ----------------------------------------------------------------------------
def run_local(args: argparse.Namespace, problems: list[Problem]) -> int:
    data_root = Path(args.data_dir)
    failures: list[str] = []

    for i, problem in enumerate(problems, 1):
        log.info("=" * 70)
        log.info("[%d/%d] %s — %d instance(s) %s",
                 i, len(problems), problem.slug, len(problem.instance_ids), problem.reasons)
        log.info("=" * 70)

        problem_data_dir = data_root / problem.data_name
        if not args.skip_execution:
            result = run(vecgen_command(args, problem, str(problem_data_dir), sys.executable))
            if result.returncode != 0:
                log.error("%s: vecgen failed (exit %d)", problem.slug, result.returncode)
                failures.append(f"{problem.slug}:vecgen")
                continue

        if not args.skip_annotate and args.record_videos:
            result = run([
                sys.executable, "src/toyblocks/data_annotation/annotate_target_xy.py",
                problem_data_dir, "-j", args.annotate_workers,
            ])
            if result.returncode != 0:
                # The annotator exits non-zero when it wrote no goal columns.
                log.error("%s: annotation failed (exit %d)", problem.slug, result.returncode)
                failures.append(f"{problem.slug}:annotate")
        elif not args.skip_annotate:
            log.info("%s: annotation skipped, nothing recorded a camera (pixel goals need "
                     "--record-videos, or render the episodes first)", problem.slug)

    if not args.skip_merge:
        log.info("=" * 70)
        log.info("Merging %s -> %s", data_root, args.merge_output_dir)
        log.info("=" * 70)
        result = run([
            sys.executable, "scripts/datagen/merge_datasets.py",
            "--input_dirs", data_root,
            "--output_dir", args.merge_output_dir,
            # 'all' keeps every video feature any problem declares. Omitting --cameras used to
            # inherit merge_success's named-list default, which silently dropped both depth
            # streams and the --extra_rgb/--extra_depth 224 sidecars HVLA trains on. The default
            # is 'all' upstream now; passing it explicitly keeps this correct against an older
            # pinned TAMP too.
            "--cameras", "all",
            "--num_workers", args.merge_workers,
        ])
        if result.returncode != 0:
            log.error("Merge failed (exit %d)", result.returncode)
            failures.append("merge")

    log.info("=" * 70)
    if failures:
        log.error("TAMPER pipeline finished with %d failed step(s): %s",
                  len(failures), ", ".join(failures))
        return 1
    log.info("TAMPER pipeline complete: %d problem(s)", len(problems))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tamper-dir", required=True, type=Path,
                        help="Directory written by evaluate.py --tamper_output_dir")

    # Instance selection
    parser.add_argument("--reasons", nargs="+", default=None,
                        choices=["failure", "timeout", "success"],
                        help="Only plan for instances with these outcomes (default: all captured)")
    parser.add_argument("--max-instances", type=int, default=None,
                        help="Cap instances per problem")
    parser.add_argument("--pddl-problem-file", default=None,
                        help="Plan every instance toward this one problem's goal instead of "
                             "the recorded ones. Required when eval ran without --queue_url, "
                             "since no problem or goal is recorded then.")

    # Steps
    parser.add_argument("--skip-execution", action="store_true",
                        help="Skip the vecgen run (planning + execution + recording)")
    parser.add_argument("--skip-annotate", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")

    # Pipeline knobs
    parser.add_argument("--domain_path", "--domain-path", dest="domain_path",
                        default="src/toyblocks/tamp/blocks")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exec-num-envs", type=int, default=32,
                        help="Parallel vecgen envs; rounded up to a multiple of the package's "
                             "scene-profile count (default: 32)")
    parser.add_argument("--record-videos", action="store_true",
                        help="Record the scene cameras too (needed to annotate here; without "
                             "it, render the episodes before annotating)")
    parser.add_argument("--vecgen-args", default="",
                        help="Extra scripts/vecgen/run.py flags, one quoted string "
                             "(e.g. '--max-attempts 2 --pool-slots 16')")
    parser.add_argument("--annotate-workers", type=int, default=8)
    parser.add_argument("--merge-workers", type=int, default=8)
    parser.add_argument("--data-dir", default=None,
                        help="Recorded episodes root (default: <tamper-dir>/datasets)")
    parser.add_argument("--merge-output-dir", default=None,
                        help="Merged dataset (default: <tamper-dir>/merged)")

    args = parser.parse_args()

    args.tamper_dir = args.tamper_dir.resolve()
    if args.data_dir is None:
        args.data_dir = str(Path(args.tamper_dir) / "datasets")
    if args.merge_output_dir is None:
        args.merge_output_dir = str(Path(args.tamper_dir) / "merged")

    problems = load_problems(
        Path(args.tamper_dir),
        reasons=args.reasons,
        max_instances=args.max_instances,
        pddl_override=args.pddl_problem_file,
    )

    total = sum(len(p.instance_ids) for p in problems)
    log.info("TAMPER dir: %s", args.tamper_dir)
    log.info("Problems:   %d (%d instance(s) total)", len(problems), total)
    for p in problems:
        log.info("  %-30s %3d instance(s) %s", p.slug, len(p.instance_ids), p.reasons)

    return run_local(args, problems)


if __name__ == "__main__":
    raise SystemExit(main())
