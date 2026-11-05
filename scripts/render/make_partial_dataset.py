#!/usr/bin/env python3
"""Merge a SUBSET of an in-progress render into a usable LeRobot dataset.

    # ~110 episodes to smoke-test training
    python make_partial_dataset.py --src <out_root> --output <dir> --target_episodes 110

    # everything finished so far
    python make_partial_dataset.py --src <out_root> --output <dir> --all_completed

    # see what it would pick, touch nothing
    python make_partial_dataset.py --src <out_root> --target_episodes 110 --dry_run

WHY THIS EXISTS, rather than pointing merge_success.py at the output root:

merge_success.get_dataset_paths() accepts ANY problem_*/blocks_success that has meta/info.json.
A problem that is still rendering has one -- the recorder writes it incrementally, per wave -- so
merging the live root silently ingests half-rendered problems and produces a dataset whose episode
count nobody can explain. Completion is recorded elsewhere: meta/render_params.json carries
meta.completed=true only when render.py finalised that problem.

So: select on render_params.json, stage the chosen problems into a fresh dir as a HARDLINK TREE
(`cp -al`), and point merge_success at that.

Hardlinks, not copies, and not symlinks:
  * a real copy of this data measured 85-234 MB/s => hours; `cp -al` recreates the directory tree
    for real but links the files, so it moves no bytes and finishes in seconds.
  * hardlinking a problem that is STILL RENDERING would be unsafe -- merge_success itself
    hardlinks videos, and the render rmtree()s and renders an unfinished problem, so the link
    would pin a stale inode and the merged dataset would hold truncated video. That hazard is
    exactly why the completed=true filter is not optional: render.py's resume does
    `if _job_completed(out_dir): continue` BEFORE the branch that rmtree()s a partial problem, so
    a finalized problem's bytes are immutable for the rest of the run.
  * symlinks would leave the merge reading through to a live tree; hardlinks give a stable
    snapshot of the inodes as they were at staging time.

The selection is deterministic (problems sorted by name) so re-running with the same target picks
the same problems, and a later call with a bigger target is a superset.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGE = HERE.parent.parent / "scripts" / "datagen" / "merge_datasets.py"


def completed_problems(src: Path) -> list[tuple[str, int, Path]]:
    """(problem_name, episodes, blocks_success_dir) for every FINISHED problem, name-sorted.

    Finished means render_params.json says meta.completed is true. A problem that is mid-render
    has an info.json and a growing data/ tree, and is deliberately excluded: merging it yields a
    dataset that disagrees with itself about how many episodes it has.
    """
    out = []
    for prob in sorted(p for p in src.iterdir() if p.is_dir() and p.name.startswith("problem_")):
        bs = prob / "blocks_success"
        rp, info = bs / "meta" / "render_params.json", bs / "meta" / "info.json"
        if not (rp.exists() and info.exists()):
            continue
        try:
            d = json.loads(rp.read_text()) or {}
            # The fields are TOP-LEVEL in render_params.json. The codebase writes
            # "meta/render_params.json (meta.completed=true)" where meta/ is the DIRECTORY -- read
            # that as a nested key and every problem looks unfinished, which is exactly what
            # happened on 2026-09-15: two partial merges exited "0 completed problem(s)" against
            # an output with 6 finished problems. Accept a nested shape too, in case it changes.
            meta = d.get("meta") if isinstance(d.get("meta"), dict) else {}
            completed = d.get("completed", meta.get("completed"))
            if not completed:
                continue
            n = int(d.get("episodes_recorded", meta.get("episodes_recorded"))
                    or (json.loads(info.read_text()) or {}).get("total_episodes") or 0)
        except Exception:                                    # noqa: BLE001 - a torn write is "not done"
            continue
        if n > 0:
            out.append((prob.name, n, bs))
    return out


def select(cands, target: int | None):
    """Fewest whole problems that reach `target` episodes. Never splits a problem.

    Biased to OVERSHOOT on purpose: the caller asked for "about N, more rather than less", and a
    problem is the smallest unit that is internally consistent.
    """
    if target is None:
        return cands, sum(n for _, n, _ in cands)
    picked, tot = [], 0
    for name, n, d in cands:
        picked.append((name, n, d))
        tot += n
        if tot >= target:
            break
    return picked, tot


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="render output root (problem_* dirs)")
    ap.add_argument("--output", type=Path, help="merged dataset dir (not needed for --dry_run)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--target_episodes", type=int, help="stop once this many episodes are covered")
    g.add_argument("--all_completed", action="store_true", help="every finished problem")
    ap.add_argument("--staging", type=Path, default=None,
                    help="where to build the hardlink tree; MUST be on the same filesystem as --src "
                         "(default: <output>_staging)")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--cameras", nargs="*", default=["all"])
    ap.add_argument("--union_schema", action=argparse.BooleanOptionalAction, default=False,
                    help="OFF by default. It exists to make the merged dataset openable by "
                         "LeRobotDataset, by re-deriving features from arrow when the parquet "
                         "carries columns info.json does not declare (this data: ~88 vs ~78). "
                         "Consumers here read parquet DIRECTLY -- HVLA's FrankaMultiModalDataset "
                         "uses pyarrow and never constructs a LeRobotDataset -- so the "
                         "reconciliation buys nothing and costs the memory that made a partial "
                         "merge OOM-prone. Pass --union_schema if something really does need "
                         "LeRobot loadability.")
    ap.add_argument("--validate_timeout", type=int, default=900)
    ap.add_argument("--validate", action="store_true",
                    help="Run merge_success's LeRobot loadability check. Off by default: it only "
                         "answers a question these consumers never ask (hvla reads parquet via "
                         "pyarrow) and has been timing out at 900s on recent merges. Passing "
                         "--union_schema turns it on automatically, since that flag exists to "
                         "make the output LeRobot-loadable.")
    ap.add_argument("--dry_run", action="store_true", help="report the selection and exit")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"[partial] --src not found: {args.src}")
    cands = completed_problems(args.src)
    total_eps = sum(n for _, n, _ in cands)
    print(f"[partial] {len(cands)} completed problem(s) under {args.src}, {total_eps} episode(s)")
    if not cands:
        raise SystemExit("[partial] nothing finished yet -- nothing to merge")

    picked, tot = select(cands, None if args.all_completed else args.target_episodes)
    print(f"[partial] selected {len(picked)} problem(s) = {tot} episode(s)"
          + ("" if args.all_completed else f" (target {args.target_episodes})"))
    for name, n, _ in picked[:10]:
        print(f"           {n:>4d}  {name}")
    if len(picked) > 10:
        print(f"           ... and {len(picked)-10} more")
    if args.target_episodes and tot < args.target_episodes:
        print(f"[partial] NOTE only {tot} episode(s) are finished, short of "
              f"{args.target_episodes}. Merging what exists; re-run later for more.")
    if args.dry_run:
        return 0
    if not args.output:
        raise SystemExit("[partial] --output is required unless --dry_run")

    staging = args.staging or Path(str(args.output) + "_staging")
    staging.mkdir(parents=True, exist_ok=True)
    staged = 0
    for name, _, bs in picked:
        dst = staging / name
        if dst.exists():                     # resumable: a previous call already linked it
            staged += 1
            continue
        # cp -al: recreate the tree, HARDLINK the files. Safe only because every problem here is
        # finalized (see the module docstring) and therefore immutable for the rest of the run.
        r = subprocess.run(["cp", "-al", str(bs.parent), str(dst)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            # A cross-device staging dir cannot hardlink; say so rather than silently falling back
            # to a multi-hour copy.
            raise SystemExit(f"[partial] cp -al failed for {name}: {r.stderr.strip()[:200]}\n"
                             f"          --staging must be on the SAME filesystem as --src "
                             f"(hardlinks cannot cross devices).")
        staged += 1
    print(f"[partial] staged {staged} problem(s) as a hardlink tree in {staging}")

    cmd = [args.python, "-u", str(MERGE),
           "--input_dirs", str(staging), "--output_dir", str(args.output),
           "--num_workers", str(args.num_workers),
           "--validate_timeout", str(args.validate_timeout)]
    if args.cameras:
        cmd += ["--cameras", *args.cameras]
    if args.union_schema:
        cmd += ["--union_schema"]
    if args.validate:
        cmd += ["--validate"]
    print("[partial] " + " ".join(shlex.quote(c) for c in cmd), flush=True)
    rc = subprocess.run(cmd).returncode
    print(f"[partial] merge exit {rc}")
    if rc == 0:
        info = args.output / "meta" / "info.json"
        if info.exists():
            d = json.loads(info.read_text())
            print(f"[partial] DONE {args.output}: {d.get('total_episodes')} episodes, "
                  f"{d.get('total_frames')} frames")
    return rc


if __name__ == "__main__":
    sys.exit(main())
