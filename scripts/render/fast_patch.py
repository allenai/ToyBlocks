#!/usr/bin/env python3
"""Close a dataset's video holes in one command: scan -> render only the holes -> apply -> re-scan.

This is the completion phase without the fleet. It is what you want when a dataset is finished
except for a handful of missing videos and you have a GPU in front of you, rather than a reason to re-run the whole render.

Fast by construction:

  * `--episodes_file` renders ONLY the listed (problem, split, episode) rows.
  * `--no-fragment_invariant` skips the DR-alignment prefix, so the render jumps straight to the
    missing episodes instead of replaying every episode before them. The patch's domain
    randomization then need not match what a full replay would have drawn -- and it does not have
    to: what must agree is the video with the camera pose recorded beside it, which it does,
    because each episode's camera draw is derived from its own index. --verify_camera_pose checks
    exactly that, per patched episode, before anything is written into the dataset.
  * `--num_envs` defaults to whatever the dataset itself was rendered with (its own
    meta/render_params.json), which is both the fastest batch that fits and the faithful one.

Isaac hangs after finishing (it skips its own teardown) and a hung process keeps ~20GB of VRAM, so
the render is reaped on its LOG (`all jobs done`), never on process exit, and a render that writes
no new episode for --stall_timeout seconds is killed rather than left to sit.

Usage:
  python fast_patch.py --dataset data/datasets/unmerged/DS \\
      --src data/datasets/unmerged/bigblocks_flip_a/execution \\
      --envs_dir data/datasets/unmerged/bigblocks_flip_a/envs \\
      --flags src/toyblocks/domain_randomization/flags/render_flags_..._nocamrand.yaml

  --dry_run stops after the scan and prints the patch list.
"""
import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import completion                                                      # noqa: E402

DONE_MARKER = "all jobs done"


def log(msg: str) -> None:
    print(f"[fast_patch] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def items_for(dataset: str, problems=None):
    """One item per problem present in the dataset, which is all the scans need."""
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(dataset, "problem_*"))
                   if os.path.isdir(p))
    if problems:
        want = set(problems)
        names = [n for n in names if n in want]
    return [{"fragment_id": f"frag_{i:03d}", "problems": [n], "episode_offset": 0,
             "max_episodes": 0} for i, n in enumerate(names)]


def dataset_render_args(dataset: str) -> dict:
    """The arguments this dataset was rendered with, from any problem's render_params.json.

    Matching them is not cosmetic: --num_envs, --depth_format/--depth_gop/--extra_depth/
    --extra_rgb and the flags yaml decide what the patched videos ARE. Guessing them produces
    videos that differ from their neighbours in ways no scan can see -- a dropped sidecar flag
    leaves the patched episodes as the only ones in the dataset missing that stream.
    """
    for rp in sorted(glob.glob(os.path.join(dataset, "problem_*", "*", "meta",
                                            "render_params.json"))):
        try:
            with open(rp) as fh:
                a = (json.load(fh) or {}).get("args") or {}
            if a:
                return a
        except (OSError, ValueError):
            continue
    return {}


def reap_on_log(cmd, logfile: str, stall_timeout: int, poll: int = 15) -> str:
    """Run a render, watching its LOG. Returns 'done' | 'stalled' | 'exited'.

    Never waits on exit: a finished Isaac render and a wedged one look identical from the outside.
    """
    with open(logfile, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT)
    log(f"render pid {proc.pid} -> {logfile}")
    seen, last_change = 0, time.time()
    try:
        while True:
            time.sleep(poll)
            try:
                with open(logfile, errors="replace") as fh:
                    text = fh.read()
            except OSError:
                text = ""
            n = text.count("Episode saved")
            if n != seen:
                seen, last_change = n, time.time()
                prog = re.findall(r"^\[progress\].*$", text, re.M)
                log(f"{n} episode(s) rendered | {prog[-1][:110] if prog else ''}")
            if DONE_MARKER in text:
                log("log says done; letting the encoder settle, then reaping")
                time.sleep(25)
                proc.kill()
                return "done"
            if proc.poll() is not None:
                return "exited"
            if time.time() - last_change > stall_timeout:
                log(f"STALL: no new episode for {stall_timeout}s; killing pid {proc.pid}")
                proc.kill()
                return "stalled"
    finally:
        if proc.poll() is None:
            proc.kill()


def camera_pose_matches(dataset: str, patch_root: str) -> list:
    """Every patched episode whose camera pose disagrees with the row already in the dataset.

    A patch replaces pixels but not the parquet, so the videos must have been rendered from the
    camera the dataset already claims -- otherwise the patched episodes quietly stop matching their
    own observation.camera_pose.* and no completeness check would ever notice. Empty list = safe.
    """
    import numpy as np
    import pandas as pd

    bad = []
    for pdir in sorted(glob.glob(os.path.join(patch_root, "problem_*"))):
        problem = os.path.basename(pdir)
        for split in completion.SPLITS:
            rp = os.path.join(pdir, split, "meta", "render_params.json")
            if not os.path.exists(rp):
                continue
            try:
                with open(rp) as fh:
                    eps = (json.load(fh) or {}).get("episodes") or []
                pf = sorted(glob.glob(os.path.join(pdir, split, "data", "*", "*.parquet")))
                tf = sorted(glob.glob(os.path.join(dataset, problem, split, "data", "*",
                                                   "*.parquet")))
                if not pf or not tf:
                    continue
                cols = [c for c in pd.read_parquet(pf[0]).columns if "camera_pose" in c]
                if not cols:
                    continue
                dp = pd.concat([pd.read_parquet(f, columns=["episode_index"] + cols) for f in pf])
                dt = pd.concat([pd.read_parquet(f, columns=["episode_index"] + cols) for f in tf])
            except Exception as e:                                     # noqa: BLE001
                bad.append(f"{problem}/{split}: could not compare ({type(e).__name__}: {e})")
                continue
            for ep in eps:
                src = ep.get("src_episode_index")
                if src is None:
                    continue
                a = dt[dt.episode_index == int(src)]
                b = dp[dp.episode_index == int(ep["episode_index"])]
                if a.empty or b.empty:
                    continue
                for c in cols:
                    va, vb = np.stack(a[c].values), np.stack(b[c].values)
                    n = min(len(va), len(vb))
                    d = float(np.abs(va[:n] - vb[:n]).max()) if n else 0.0
                    if d > 1e-5:
                        bad.append(f"{problem}/{split} ep{src}: {c} differs by {d:.6g}")
                        break
    return bad


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="dataset root holding problem_*/")
    p.add_argument("--src", required=True, help="source execution dataset")
    p.add_argument("--envs_dir", default=None)
    p.add_argument("--flags", default=None,
                   help="DR yaml. Defaults to the one recorded in the dataset's render_params.")
    p.add_argument("--problems", nargs="*", default=None, help="limit the scan to these problems")
    p.add_argument("--num_envs", type=int, default=None,
                   help="default: whatever the dataset was rendered with")
    p.add_argument("--rounds", type=int, default=2, help="patch attempts before giving up")
    p.add_argument("--work_dir", default=None, help="default: <dataset>/_patch")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--dr_seeding", default=None, choices=["stream", "indexed", "hash"],
                   help="default: the mode recorded in the dataset's own render_params.json. A "
                        "patch drawn with a different mode redraws the episode, and under camera "
                        "randomization the new video then disagrees with the pose in the parquet.")
    p.add_argument("--stall_timeout", type=int, default=900,
                   help="kill a render that writes no new episode for this long")
    p.add_argument("--verify_camera_pose", default=True, action=argparse.BooleanOptionalAction,
                   help="refuse to apply a patch whose camera poses disagree with the parquet")
    p.add_argument("--dry_run", action="store_true", help="scan and print the patch list, then stop")
    args = p.parse_args()

    dataset = os.path.abspath(args.dataset)
    work = os.path.abspath(args.work_dir or os.path.join(dataset, "_patch"))
    os.makedirs(work, exist_ok=True)
    items = items_for(dataset, args.problems)
    if not items:
        log(f"no problem_* directories under {dataset}")
        return 2

    rargs = dataset_render_args(dataset)
    num_envs = args.num_envs or int(rargs.get("num_envs") or 32)
    flags = args.flags or rargs.get("flags_file")
    if not flags or not os.path.exists(flags):
        # the recorded path is the one INSIDE the render container; fall back to this checkout
        cand = os.path.join(HERE, os.path.basename(flags or ""))
        flags = cand if flags and os.path.exists(cand) else flags
    if not flags or not os.path.exists(flags):
        log(f"ERROR: no usable --flags (dataset recorded {rargs.get('flags_file')!r})")
        return 2
    seeding = args.dr_seeding or rargs.get("dr_seeding")
    log(f"{len(items)} problem(s) | num_envs={num_envs} | flags={os.path.basename(flags)} "
        f"| dr_seeding={seeding or '(render default)'}")

    absent = completion.scan_missing_problems(dataset, items)
    if absent:
        log(f"WARNING {len(absent)} problem(s) produced NOTHING and cannot be patched (there are "
            f"no episodes to enumerate); they need a full render: {absent[:5]}")

    for rnd in range(1, max(1, args.rounds) + 1):
        holes = completion.scan_missing_videos(dataset, items)
        if not holes:
            log("no video holes; the dataset is whole")
            return 0
        probs = sorted({h["problem"] for h in holes})
        log(f"round {rnd}/{args.rounds}: {len(holes)} hole(s) across {len(probs)} problem(s)")
        tsv = os.path.join(work, f"fast_patch-r{rnd}.tsv")
        completion.write_patch_tsv(tsv, holes)
        log(f"patch list -> {tsv}")
        if args.dry_run:
            for h in holes[:20]:
                print(f"  {h['problem']} {h['split']} {h['episode']}")
            if len(holes) > 20:
                print(f"  ... and {len(holes) - 20} more")
            return 0

        patch_root = os.path.join(work, f"render-r{rnd}")
        cmd = completion.render_command(
            {"fragment_id": f"fast_patch_r{rnd}", "problems": probs,
             "episode_offset": 0, "max_episodes": 0},
            python=args.python, script=os.path.join(HERE, "render.py"),
            src=args.src, envs_dir=args.envs_dir, out_root=patch_root, flags=flags,
            n=1, num_envs=num_envs, assets_paths=[],
            env_spacing=rargs.get("env_spacing", 60),
            rtx_warmup_frames=int(rargs.get("rtx_warmup_frames") or 16),
            depth_codec=rargs.get("depth_codec", "hevc"),
            depth_format=rargs.get("depth_format", "12bit"),
            extra_depth=rargs.get("extra_depth"),
            extra_rgb=rargs.get("extra_rgb"),
            depth_gop=int(rargs.get("depth_gop") or 1),
            # Absent in render_params written before these flags existed: keep render.py's
            # defaults (encoder-chosen sidecar GOP, main RGB GOP 2), which is what those used.
            extra_rgb_gop=int(rargs.get("extra_rgb_GOP") or 0),
            rgb_gop=(int(rargs["rgb_GOP"]) if rargs.get("rgb_GOP") is not None else None),
            # Recorded only by runs that set them; older render_params keep the old argv.
            render_scale=(float(rargs["render_scale"]) if rargs.get("render_scale") else None),
            skip_depth=bool(rargs.get("skip_depth")),
            skip_cameras=tuple(rargs.get("skip_cameras") or ()),
            dr_seeding=args.dr_seeding or rargs.get("dr_seeding"),
            episodes_file=tsv, extra=("--no-fragment_invariant",))
        log("render: " + " ".join(shlex.quote(c) for c in cmd[:6]) + " ...")
        state = reap_on_log(cmd, os.path.join(work, f"render-r{rnd}.log"),
                            args.stall_timeout)
        log(f"render {state}")
        if state == "stalled":
            return 4

        if args.verify_camera_pose:
            bad = camera_pose_matches(dataset, patch_root)
            if bad:
                log(f"REFUSING to apply: {len(bad)} patched episode(s) do not match the camera "
                    f"pose recorded in the dataset")
                for b in bad[:8]:
                    log(f"   {b}")
                return 5
            log("camera poses match the dataset's parquet for every patched episode")


        res = completion.apply_video_patch(dataset, patch_root)
        log(f"applied {res['videos']} video(s) for {res['episodes']} episode(s) across "
            f"{res['problems']} problem(s)")
        for why in res["skipped"][:8]:
            log(f"  skipped: {why}")

    left = completion.scan_missing_videos(dataset, items)
    if left:
        log(f"{len(left)} hole(s) remain after {args.rounds} round(s)")
        return 3
    log("all video holes closed; the dataset is whole")
    return 0


if __name__ == "__main__":
    sys.exit(main())
