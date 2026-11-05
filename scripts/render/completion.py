"""Completion helpers for a rendered dataset: find missing problems / videos, render only the
holes, and apply the patch.

Used by fast_patch.py (and render.py's command builder); importable without Isaac.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Sequence
from typing import Any

SPLITS = ("blocks_success", "blocks_failure")


# --------------------------------------------------------------------------------------------
# completion oracle (ground truth == the output files, never exit codes, never isdir)
# --------------------------------------------------------------------------------------------
def problem_complete(out_root: str, problem: str) -> bool:
    """True iff this problem finalized in the output: some split's meta/render_params.json says
    completed=true (mirrors render.py's own _job_completed)."""
    for split in SPLITS:
        rp = os.path.join(out_root, problem, split, "meta", "render_params.json")
        if not os.path.exists(rp):
            continue
        try:
            with open(rp) as fh:
                if json.load(fh).get("completed") is True:
                    return True
        except (OSError, ValueError):
            return False                          # partial/corrupt write -> not complete
    return False


# --------------------------------------------------------------------------------------------
# render command construction (identical across fragments -> level-2 byte-identity)
# --------------------------------------------------------------------------------------------
def render_command(item: dict[str, Any], *, python: str, script: str, src: str,
                     envs_dir: str | None,
                     out_root: str, flags: str, n: int, num_envs: int, assets_paths: Sequence[str],
                     env_spacing: float | None = 60, rtx_warmup_frames: int = 16,
                     depth_codec: str = "hevc", depth_format: str = "12bit",
                     extra_depth: str | None = None,
                     extra_rgb: str | None = None, depth_gop: int = 1,
                     extra_rgb_gop: int = 0, rgb_gop: int | None = None,
                     render_scale: float | None = None, skip_depth: bool = False,
                     episodes_file: str | None = None,
                     skip_cameras: Sequence[str] = (),
                     skip_failures: bool = False,
                     split_groups: bool = False,
                     dr_seeding: str | None = None,
                     camera_base: str | None = None,
                     extra: Sequence[str] = ()) -> list[str]:
    """Exact argv for one item.

    Every non-slice argument MUST be identical across items or the output stops being
    byte-comparable to a single-machine run (level-2 invariance also requires identical --num_envs,
    --flags, -N, --skip_initial_step, --rtx_warmup_frames, --depth_codec and --env_spacing).
    --resume makes a retry skip problems this item already finalized.

    Every item shares ONE --output: the partition is problem-aligned, so two items never write the
    same <problem>/ subtree and they cannot collide.
    """
    cmd = [
        python, "-u", script, src,
        "--output", out_root,
        "--flags", flags,
        "-N", str(n),
        "--num_envs", str(num_envs),
        "--skip_initial_step",
        "--rtx_warmup_frames", str(rtx_warmup_frames),
        "--depth_codec", depth_codec,
        "--depth_format", depth_format,
        "--resume",
    ]
    cmd += ["--depth_gop", str(depth_gop)]
    # Emit the seeding mode EXPLICITLY rather than inheriting render.py's default. A patch pass
    # must draw what the dataset was rendered with, and the default has changed over time (these
    # datasets record stream, indexed and hash); inheriting it silently redraws every patched
    # episode -- which for a camera-randomizing run means the new video no longer matches the
    # camera pose already in the parquet.
    if dr_seeding:
        cmd += ["--dr_seeding", str(dr_seeding)]
    # Same reason as dr_seeding: rendering INTO an existing dataset must use the base that
    # dataset was rendered with, and the default changed the day --camera_base landed.
    if camera_base:
        cmd += ["--camera_base", str(camera_base)]
    # Episode selection is EITHER the fragment slice OR the patch list -- never both.
    # render.py documents "--episode_offset ... Overrides --episodes_file", so emitting the slice
    # alongside a patch list silently discards the list: the filter loads (it even logs "rendering
    # only N episode(s)") and is then replaced by the offset slice, which for a whole-problem item
    # is EVERY episode. That turned a 35-episode patch into a full render of 15 problems before
    # it was caught.
    if episodes_file:
        # Targeted PATCH mode: render only the listed (problem, split, episode) rows. render.py
        # keeps the non-rendered prefix so per-episode DR sampling and wave grouping still match a
        # full run, and stamps src_episode_index so the patch can be mapped back.
        cmd += ["--episodes_file", episodes_file]
    else:
        cmd += ["--episode_offset", str(item["episode_offset"]),
                "--max_episodes", str(item["max_episodes"])]
    if extra_rgb:
        cmd += ["--extra_rgb", extra_rgb]
    if extra_depth:
        cmd += ["--extra_depth", extra_depth]
    # Emitted only when set, so a run that never asked for them keeps its exact argv (and
    # render.py's own defaults: 0 = encoder-chosen sidecar GOP, main RGB GOP 2).
    if extra_rgb_gop:
        cmd += ["--extra_rgb_GOP", str(int(extra_rgb_gop))]
    if rgb_gop is not None:
        cmd += ["--rgb_GOP", str(int(rgb_gop))]
    if render_scale is not None:
        cmd += ["--render_scale", str(float(render_scale))]
    if skip_depth:
        cmd += ["--skip_depth"]
    # Optional: envs may live under a separate root (the S3 layout keeps them in
    # basics/<exp>/envs/). When omitted, render.py falls back to <src>/envs/<problem>/env.py.
    if envs_dir:
        cmd += ["--envs_dir", envs_dir]
    if env_spacing is not None:
        cmd += ["--env_spacing", str(env_spacing)]
    if skip_cameras:
        cmd += ["--skip_cameras"] + list(skip_cameras)
    if skip_failures:
        # MUST mirror the planner: --episode_offset indexes the enumeration, so the worker has to
        # filter splits exactly as build_plan did or every item renders the wrong episodes.
        cmd += ["--skip_failure_episodes"]
    if split_groups:
        # Same lockstep as skip_failures: the offsets index the member-expanded enumeration.
        cmd += ["--split_groups"]
    cmd += list(extra)
    if assets_paths:
        cmd += ["--assets_paths"] + list(assets_paths)   # last by convention
    return cmd


def expected_video_features(info: dict[str, Any], vroot: str) -> list[str]:
    """The video features a split is SUPPOSED to have.

    Taken from the split's own meta/info.json, because a feature directory that was never created
    cannot be discovered by listing the ones that were. Falls back to what is on disk when
    info.json declares no features, so a hand-made or pre-features dataset still scans.
    """
    feats = info.get("features") or {}
    declared = sorted(k for k, v in feats.items()
                      if isinstance(v, dict) and v.get("dtype") == "video")
    if declared:
        return declared
    try:
        return sorted(f for f in os.listdir(vroot)
                      if os.path.isdir(os.path.join(vroot, f)))
    except OSError:
        return []


def scan_missing_problems(out_root: str, items) -> list[str]:
    """Planned problems that produced NOTHING -- no split, no info.json, no episodes.

    scan_missing_videos deliberately stays silent about these: with no info.json there is no
    episode count, so there are no episode-level holes to enumerate and nothing a patch pass could
    render. That silence is safe only if somebody else is asking the question, which is what this
    is for -- allrand shipped 49 of the 50 problems its plan named, and every episode-level check
    on the 49 that did land came back clean.
    """
    absent = []
    for it in items:
        for prob in it["problems"]:
            for split in SPLITS:
                try:
                    with open(os.path.join(out_root, prob, split,
                                           "meta", "info.json")) as fh:
                        if int(json.load(fh).get("total_episodes") or 0) > 0:
                            break
                except (OSError, ValueError):
                    continue
            else:
                # Zero episodes is LEGITIMATE when every source parquet was unreadable: the worker
                # finalizes the problem with completed=true and an empty_reason instead of
                # rendering it. Without this the merge is blocked forever on problems that can
                # never produce an episode. The check stays strict otherwise -- it still requires a
                # worker to have positively finished the problem, so the case this was written for
                # (a planned problem no worker ever rendered) is still caught.
                if not problem_complete(out_root, prob):
                    absent.append(prob)
    return sorted(set(absent))


def scan_missing_videos(out_root: str, items) -> list[dict[str, Any]]:
    """Every (problem, split, episode) whose video is absent ON DISK.

    This is the authoritative completeness check, and deliberately not the same thing as
    collect_video_defects(): that reports what the ENCODER knew it failed to publish, so it is
    blind to anything lost before the defect bookkeeping existed, or to a file removed afterwards.
    Here we simply compare each video feature's file count against the split's declared
    total_episodes and, when short, work out exactly which indices are missing.

    A merge is only meaningful on a whole dataset, so this runs before it and gates it.
    """
    rows = []
    for it in items:
        for prob in it["problems"]:
            for split in SPLITS:
                meta = os.path.join(out_root, prob, split, "meta", "info.json")
                try:
                    with open(meta) as fh:
                        info = json.load(fh)
                    total = int(info.get("total_episodes") or 0)
                except (OSError, ValueError):
                    continue
                if total <= 0:
                    continue
                vroot = os.path.join(out_root, prob, split, "videos")
                if not os.path.isdir(vroot):
                    rows += [{"problem": prob, "split": split, "episode": e}
                             for e in range(total)]
                    continue
                missing = set()
                for feat in expected_video_features(info, vroot):
                    have = set()
                    fdir = os.path.join(vroot, feat)
                    if not os.path.isdir(fdir):
                        # A feature whose directory never appeared is missing for EVERY episode.
                        # Unioning over os.listdir() alone could not see this: one allrand split
                        # holds 3 of the 9 features its own info.json declares, and a scan that
                        # only looks at what is there calls the other 6 complete.
                        missing |= set(range(total))
                        continue
                    for chunk in os.listdir(fdir):
                        cdir = os.path.join(fdir, chunk)
                        if not os.path.isdir(cdir):
                            continue
                        cm = re.match(r"chunk-(\d+)$", chunk)
                        base = int(cm.group(1)) * 1000 if cm else 0
                        for fn in os.listdir(cdir):
                            m = re.match(r"file-(\d+)\.mp4$", fn)
                            if m:
                                # chunk-NNN/file-MMM is episode NNN*1000+MMM
                                have.add(base + int(m.group(1)))
                    missing |= (set(range(total)) - have)
                rows += [{"problem": prob, "split": split, "episode": e} for e in sorted(missing)]
    # one render fixes every missing feature of an episode
    seen, uniq = set(), []
    for r in rows:
        k = (r["problem"], r["split"], r["episode"])
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return uniq


def video_path(root: str, problem: str, split: str, feature: str, episode: int) -> str:
    """The one canonical place an episode's video lives: chunk-{ep//1000}/file-{ep%1000}.mp4."""
    return os.path.join(root, problem, split, "videos", feature,
                        f"chunk-{episode // 1000:03d}", f"file-{episode % 1000:03d}.mp4")


def has_moov(path: str) -> bool:
    """True if this mp4 carries a 'moov' atom, i.e. the encoder finished writing it.

    A file killed mid-flush is the right size and completely undecodable ("moov atom not found"),
    so size alone cannot tell a good patch from a truncated one -- and a truncated patch copied
    over a merely-missing video turns a hole into corruption.
    """
    try:
        with open(path, "rb") as fh:
            size = os.fstat(fh.fileno()).st_size
            off = 0
            while off < size:
                fh.seek(off)
                hdr = fh.read(8)
                if len(hdr) < 8:
                    return False
                box = int.from_bytes(hdr[:4], "big")
                kind = hdr[4:8]
                if kind == b"moov":
                    return True
                if box == 1:                                  # 64-bit extended size follows
                    ext = fh.read(8)
                    if len(ext) < 8:
                        return False
                    box = int.from_bytes(ext, "big")
                elif box == 0:                                # this box runs to EOF
                    return False
                if box < 8:
                    return False
                off += box
    except OSError:
        return False
    return False


def merge_patched_render_params(out_root: str, problem: str, split: str,
                                patched: dict[int, dict[str, Any]]) -> None:
    """Fold the patch's episode records into the target's render_params and clear those defects.

    The recorded DR params must describe the pixels actually on disk, so a patched episode replaces
    its original record. Dropping the matching video_defects rows and recomputing `status` is what
    lets the file say "complete" truthfully -- the whole point of marking success only once the
    videos are really there.
    """
    path = os.path.join(out_root, problem, split, "meta", "render_params.json")
    try:
        with open(path) as fh:
            rp = json.load(fh)
    except (OSError, ValueError):
        return
    eps = rp.get("episodes") or []
    for i, ep in enumerate(eps):
        idx = ep.get("episode_index")
        if idx is not None and int(idx) in patched:
            eps[i] = patched[int(idx)]
    rp["episodes"] = eps
    rp["video_defects"] = [d for d in (rp.get("video_defects") or [])
                           if int(d.get("episode", -1)) not in patched]
    rp["status"] = ("complete" if rp.get("completed") and not rp["video_defects"]
                    else "incomplete_videos" if rp.get("completed") else "failed")
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(rp, fh, indent=1)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def apply_video_patch(out_root: str, patch_root: str) -> dict[str, Any]:
    """Copy freshly rendered videos out of a patch render and over the holes in the dataset.

    A patch CANNOT be rendered into the dataset it repairs. The recorder numbers episodes by save
    order into a fresh dataset -- there is no "write this one at index 250" -- so a patch aimed at
    the real output would write episode 250's pixels to file-000 and destroy a good episode. It
    therefore renders into its own directory, and the intermediate's render_params carries
    src_episode_index for the remap back. Same mapping patch_remote_from_render.py does over S3,
    done locally because the output directory is already mounted.

    Every file is checked for a moov before it is copied and again at the destination, and it lands
    via a temp file + os.replace, so a failure here can only leave the original hole -- never a
    half-written video where a whole one used to be.
    """
    import glob as _glob
    import shutil

    out: dict[str, Any] = {"episodes": 0, "videos": 0, "problems": 0, "skipped": []}
    for pdir in sorted(_glob.glob(os.path.join(patch_root, "problem_*"))):
        problem = os.path.basename(pdir)
        touched = False
        for split in SPLITS:
            sdir = os.path.join(pdir, split)
            rp_path = os.path.join(sdir, "meta", "render_params.json")
            if not os.path.isdir(sdir) or not os.path.exists(rp_path):
                continue
            try:
                with open(rp_path) as fh:
                    rp = json.load(fh)
            except (OSError, ValueError) as e:
                out["skipped"].append(f"{problem}/{split}: unreadable render_params ({e})")
                continue
            feats = [os.path.basename(d)
                     for d in _glob.glob(os.path.join(sdir, "videos", "observation.*"))]
            patched: dict[int, dict[str, Any]] = {}
            for ep in rp.get("episodes", []):
                if ep.get("split", split) != split:
                    continue
                src_ep = ep.get("src_episode_index")
                if src_ep is None:
                    out["skipped"].append(f"{problem}/{split}: episode without src_episode_index")
                    continue
                out_ep, src_ep = int(ep["episode_index"]), int(src_ep)
                copied = 0
                for feat in feats:
                    src = video_path(patch_root, problem, split, feat, out_ep)
                    if not os.path.exists(src):
                        out["skipped"].append(f"{problem}/{split}/{feat}: no patch video for src "
                                              f"episode {src_ep}")
                        continue
                    if not has_moov(src):
                        out["skipped"].append(f"{problem}/{split}/{feat}: patch video for src "
                                              f"episode {src_ep} has no moov (truncated); "
                                              f"kept the hole")
                        continue
                    dst = video_path(out_root, problem, split, feat, src_ep)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    tmp = dst + ".patchtmp"
                    try:
                        shutil.copyfile(src, tmp)
                        if os.path.getsize(tmp) != os.path.getsize(src) or not has_moov(tmp):
                            os.unlink(tmp)
                            out["skipped"].append(f"{problem}/{split}/{feat}: copy of src episode "
                                                  f"{src_ep} did not verify; kept the hole")
                            continue
                        os.replace(tmp, dst)
                    except OSError as e:
                        out["skipped"].append(f"{problem}/{split}/{feat}: {type(e).__name__}: {e}")
                        try:
                            os.unlink(tmp)
                        except OSError:
                            pass
                        continue
                    copied += 1
                if copied:
                    out["videos"] += copied
                    out["episodes"] += 1
                    touched = True
                    rec = dict(ep)
                    rec["episode_index"] = src_ep             # renumber into the target's space
                    rec["patched"] = True
                    patched[src_ep] = rec
            if patched:
                merge_patched_render_params(out_root, problem, split, patched)
        if touched:
            out["problems"] += 1
    return out


def write_patch_tsv(path: str, rows) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("problem\tsplit\tepisode\n")
        for r in rows:
            fh.write(f"{r['problem']}\t{r['split']}\t{r['episode']}\n")
    return len(rows)
