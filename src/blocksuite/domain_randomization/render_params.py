"""render_params.json assembly + atomic, lock-guarded write -- the DR provenance format rerender
writes and eval will reuse (so an eval render carries the same per-episode camera/skybox/materials/
lights/exposure/env_sphere record). No module globals: the run-`args` block is passed in, so eval
supplies its own args dict instead of reading rerender's argparse namespace.

Moved verbatim from rerender.py except `render_params_payload` now takes `args` + `script` as
parameters (was reading the `args_cli` global).
"""
from __future__ import annotations

import fcntl
import glob
import hashlib
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional


def split_meta_dirs(out_dir: str, job: Dict[str, Any]) -> List[str]:
    """Expected output split meta dirs for a job (blocks_success/meta, blocks_failure/meta, ...),
    plus whatever the recorder already created."""
    names = []
    for _, succ in job["sources"]:
        names.append("blocks_success" if succ else ("blocks_failure" if succ is False else "blocks"))
    existing = [os.path.dirname(m) for m in glob.glob(os.path.join(out_dir, "*", "meta"))]
    dirs = {os.path.join(out_dir, n, "meta") for n in names}
    dirs.update(os.path.join(e, "meta") for e in existing)
    return sorted(dirs)


def _split_name(meta_dir: str) -> str:
    """'.../blocks_success/meta' -> 'blocks_success'."""
    return os.path.basename(os.path.dirname(meta_dir))


def episodes_for_split(episode_records: list, split: str) -> list:
    """This split's episodes, renumbered 0..k-1 in render order.

    `episode_index` in the live list is the PROBLEM-wide counter, so success and failure episodes
    are interleaved (0=success, 1=failure, 2=success, ...). That index matches neither split's
    LeRobot episode_index, and consumers which read the per-split file -- patch_tasks_from_source,
    validate_N, split_by_iter_plan -- all assume per-split numbering. Filter on the `split` each
    record already carries, then renumber.

    Each record is COPIED: the caller reuses one list across every incremental checkpoint, so
    renumbering in place would compound over successive writes.
    """
    if not any("split" in e for e in episode_records):
        return list(episode_records)          # records predating `split`: nothing to filter on
    out = []
    for e in episode_records:
        if e.get("split") == split:
            rec = dict(e)
            rec["episode_index"] = len(out)
            out.append(rec)
    return out


def _flags_file_digest(flags_file: Optional[str]) -> Optional[str]:
    """sha256 (16 hex) of the flags YAML as written on disk.

    Identifies the literal file a shard was rendered from; None when the file is unavailable
    (already-parsed flags, or a moved path).
    """
    if not flags_file:
        return None
    try:
        with open(flags_file, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()[:16]
    except OSError:
        return None


def _resolved_flags_digest(flags: Optional[dict]) -> Optional[str]:
    """sha256 (16 hex) of the RESOLVED flags dict. This is the one to group shards by: two shards
    with the same digest were rendered under identical settings even if the yaml was reformatted
    or lives at a different path, and differ if a code-side default changed under the same file.
    Datasets get sharded and re-sharded over weeks; the file digest alone cannot answer that."""
    if not flags:
        return None
    try:
        canon = json.dumps(flags, sort_keys=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        return None
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]


def render_params_payload(job: Dict[str, Any], flags: dict, fps: int, use_physics: bool,
                          initial: dict, episode_records: list, completed: bool, args: dict,
                          total_episodes: Optional[int] = None,
                          script: str = "scripts/rerender/rerender.py",
                          empty_reason: Optional[str] = None) -> dict:
    """Assemble the render_params.json payload. `args` (the run's CLI/config args) and `script` are
    passed in so this is consumer-agnostic (rerender fills them from argparse; eval from its config)."""
    # completed=true with episodes_recorded=0 is a legitimate terminal state (every source parquet
    # unreadable), but indistinguishable from a bug without a reason on disk.
    payload = {
        "rendered_at": datetime.now().isoformat(timespec="seconds"),
        "rerender_script": script,
        "completed": bool(completed),
        "empty_reason": empty_reason,
        "episodes_recorded": len(episode_records),
        "episodes_total": total_episodes,
        "mode": "physics-step" if use_physics else "render-only",
        "fps": fps,
        "env_file": job.get("env_file"),
        "source_datasets": [str(s[0]) for s in job["sources"]],
        "args": dict(args or {}),
        "flags": flags or {},
        "flags_sha256": _flags_file_digest((args or {}).get("flags_file")),
        "flags_resolved_sha256": _resolved_flags_digest(flags),
        # Which DR seeding path produced these values. Top-level (not just inside args) because a
        # consumer must branch on it: 'stream' episodes are only reproducible by re-rendering the
        # same episode LIST, while 'indexed' episodes reproduce standalone from (seed, index).
        # Mixing the two inside one dataset would silently mix two different randomizations.
        "dr_seeding": (args or {}).get("dr_seeding", "stream"),
        "initial": initial,
        "episodes": episode_records,
    }
    # --split_groups: this output is ONE member problem of a vecgen group; source_datasets name the
    # group's splits, so record which member it is. Absent otherwise, so other outputs are unchanged.
    if job.get("member"):
        payload["split_group"] = {"group": job.get("group"), "problem": job["member"]}
    return payload


def write_json_locked(path: str, payload: dict) -> None:
    """Atomic + flock-guarded JSON write (write temp, os.replace, under an exclusive .lock)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock_path = path + ".lock"
    with open(lock_path, "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(payload, f, indent=2)
            os.replace(tmp, path)
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def write_render_params(out_dir: str, job: Dict[str, Any], flags: dict, fps: int, use_physics: bool,
                        initial: dict, episode_records: list, args: dict, completed: bool = True,
                        total_episodes: Optional[int] = None, verbose: bool = True,
                        script: str = "scripts/rerender/rerender.py",
                        video_defects: Optional[list] = None,
                        empty_reason: Optional[str] = None) -> None:
    """Write meta/render_params.json into each output split (incrementally, locked+atomic).

    Each split gets ONLY its own episodes, renumbered to match that split's LeRobot episode_index
    (see episodes_for_split). Previously every split received an identical copy of the whole
    problem's episodes[], which both duplicated the payload and left episode_index misaligned with
    the split's data for any problem that had failures.

    `episodes_recorded` is therefore this split's count, while `episodes_total` stays the
    problem-wide expected total (all splits) -- the per-split total is not knowable during the
    incremental checkpoints written mid-run.
    """
    for meta_dir in split_meta_dirs(out_dir, job):
        eps = episodes_for_split(episode_records, _split_name(meta_dir))
        payload = render_params_payload(job, flags, fps, use_physics, initial, eps,
                                        completed, args, total_episodes, script, empty_reason)
        # Tri-state, so a hole is neither silently "fine" nor a total failure:
        #   completed=true,  video_defects=[]   -> whole and usable
        #   completed=true,  video_defects=[..] -> USABLE BUT INCOMPLETE; the listed (episode,
        #                                          feature) videos are absent and need a patch pass
        #   completed=false                     -> the problem must be re-rendered
        # Carrying the exact list here is what lets the orchestrator patch automatically instead of
        # someone re-deriving it from a 115k-object listing later.
        defects = [d for d in (video_defects or [])
                   if _split_name(meta_dir) in (d.get("split") or _split_name(meta_dir))]
        payload["video_defects"] = defects
        payload["status"] = ("complete" if completed and not defects else
                             "incomplete_videos" if completed else "failed")
        path = os.path.join(meta_dir, "render_params.json")
        write_json_locked(path, payload)
        if verbose:
            print(f"[dr] wrote {path} (completed={completed}, status={payload['status']}, "
                  f"episodes={len(eps)}, video_defects={len(defects)})")


def job_completed(out_dir: str) -> bool:
    """True if a prior run finished this problem (any split render_params.completed=true)."""
    for meta in glob.glob(os.path.join(out_dir, "*", "meta", "render_params.json")):
        try:
            if json.load(open(meta)).get("completed") is True:
                return True
        except Exception:
            pass
    return False
