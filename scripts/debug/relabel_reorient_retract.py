#!/usr/bin/env python3
"""Relabel the scripted-flip retreat inside ``reorient_traj`` as ``reorient_retract_traj``.

The scripted contact flip (ReOrientTrajectory, non-RL mode) bakes its retreat into the
``reorient_traj`` segment and the action's own ``reorient_retract_traj`` is skipped
(LiftTrajectory._skipped_for_scripted_flip). Every attempt, retries included, runs

    free move -> hover hold -> descent -> contact hold -> pull -> pull hold -> RETREAT -> settle hold

where the retreat is the one joint-space move whose commanded EE rises (FLIP_RETREAT_HEIGHT, 0.12 m).
This marks every such move run, plus the hold that follows it (up to the next attempt's first move
or the segment end), as ``reorient_retract_traj``. Only
``action.low_level_action`` changes; goal.* annotations label every reorient frame
``flip <dir>`` / ``move_robot`` regardless, so they are unaffected.

Usage:
    relabel_reorient_retract.py SRC_DATASET [--dst DST_DATASET | --in-place] [--workers N] [--limit N]
Without --dst / --in-place it is a dry run that only reports what would change. Existing
reorient_retract_traj labels are folded back into reorient_traj before segmenting, so reruns and
in-place runs give identical labels. Writing is two-phase: every file is planned (read-only, with
its neighbours' edge rows as context) before any file is rewritten, and each rewrite reads only its
own file -- so no file is read while another worker renames its new version into place.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

LL_COL = "action.low_level_action"
REORIENT_LL = "reorient_traj"
RETRACT_LL = "reorient_retract_traj"
MOVE_EPS = 1e-6  # joint-target change (rad) that counts as a move; holds repeat the target exactly
# Commanded EE rise (m) of a retreat; the flip retreats FLIP_RETREAT_HEIGHT = 0.12 m. No other move of
# the flip rises: approach and descent go down, the pull stays level. The joint-space interpolation can
# drift the commanded EE a few cm in xy on the way up (forward flips), so xy is not constrained.
RETREAT_MIN_DZ = 0.06
PULL_MIN_DXY = 0.03  # commanded EE xy change (m) of a pull move (one per attempt), for the report only
# Rows read from each neighbouring file: episodes (and so reorient segments, < ~500 frames) can straddle
# data-file boundaries, so each file is relabeled with its neighbours' edge rows as context.
CONTEXT_ROWS = 2000
READ_COLS = ["episode_index", "action.high_level_args", LL_COL, "action.joint_target", "action.ee_target"]


def _runs(mask: np.ndarray) -> list[tuple[int, int, bool]]:
    """(start, end_exclusive, value) runs of a boolean array."""
    if len(mask) == 0:
        return []
    edges = np.flatnonzero(mask[1:] != mask[:-1]) + 1
    starts = np.r_[0, edges]
    ends = np.r_[edges, len(mask)]
    return [(int(s), int(e), bool(mask[s])) for s, e in zip(starts, ends)]


def _arrays(table: pa.Table) -> dict[str, np.ndarray]:
    return {
        "ll": table.column(LL_COL).to_numpy(zero_copy_only=False).astype(object),
        "hl": table.column("action.high_level_args").to_numpy(zero_copy_only=False).astype(object),
        "ep": table.column("episode_index").to_numpy(),
        "jt": np.stack(table.column("action.joint_target").to_numpy(zero_copy_only=False)),
        "et": np.stack(table.column("action.ee_target").to_numpy(zero_copy_only=False)),
    }


def relabel(
    arr: dict[str, np.ndarray], lo: int, hi: int, window_cut_lo: bool, window_cut_hi: bool
) -> tuple[np.ndarray, collections.Counter, list[int], list[float]]:
    """New low_level_action values for rows [lo, hi) of a context window, plus stats.

    window_cut_lo / window_cut_hi: the window starts / ends inside the dataset (context rows were
    truncated there), so a segment touching that edge may be incomplete.
    """
    ll, hl_args, ep, jt, et = arr["ll"], arr["hl"], arr["ep"], arr["jt"], arr["et"]
    new = ll[lo:hi].copy()
    stats: collections.Counter = collections.Counter()
    retreat_lens: list[int] = []
    retreat_dz: list[float] = []

    # Segment on labels with any existing retract folded back into reorient_traj, so a neighbouring
    # file that was already relabeled (in-place runs, reruns) gives the same segments as the original.
    seg_ll = np.where(ll == RETRACT_LL, REORIENT_LL, ll)
    key_change = np.r_[True, (seg_ll[1:] != seg_ll[:-1]) | (hl_args[1:] != hl_args[:-1]) | (ep[1:] != ep[:-1])]
    starts = np.flatnonzero(key_change)
    ends = np.r_[starts[1:], len(ll)]
    stats["existing_retract_frames"] = int((ll[lo:hi] == RETRACT_LL).sum())
    for s, e in zip(starts, ends):
        if seg_ll[s] != REORIENT_LL or e <= lo or s >= hi:
            continue
        owned = lo <= s < hi  # count each segment once: in the file holding its first row
        if (s == 0 and window_cut_lo) or (e == len(ll) and window_cut_hi):
            stats["segments_cut_by_context_window"] += 1
        if owned:
            stats["segments"] += 1
            if hi < e:
                stats["segments_spanning_files"] += 1
        # Change of each frame's target from the previous frame (same episode).
        prev = s - 1 if s > 0 and ep[s - 1] == ep[s] else s
        d = np.linalg.norm(np.diff(jt[prev:e], axis=0), axis=1)
        if prev == s:
            d = np.r_[0.0, d]
        move_runs = [(a, b) for a, b, mv in _runs(d > MOVE_EPS) if mv]
        n_retreat = n_pull = 0
        retreat_starts = []
        for a, b in move_runs:
            before = s + a - 1 if s + a - 1 >= prev else s + a  # last held target before the move
            dz = float(et[s + b - 1, 2] - et[before, 2])
            dxy = float(np.linalg.norm(et[s + b - 1, :2] - et[before, :2]))
            if dz > RETREAT_MIN_DZ:
                n_retreat += 1
                retreat_starts.append(a)
                if owned:
                    retreat_lens.append(b - a)
                    retreat_dz.append(dz)
            elif dxy > PULL_MIN_DXY and abs(dz) < 0.03:
                n_pull += 1
        if owned:
            stats[f"retreats_per_segment={n_retreat}"] += 1
            if n_retreat != n_pull:
                stats["segments_retreats!=pulls"] += 1
        if not retreat_starts:
            continue
        # Label each retreat through the hold after it: up to the next move run or the segment end.
        for a in retreat_starts:
            b = next(bb for aa, bb in move_runs if aa == a)
            nxt = next((aa for aa, _ in move_runs if aa >= b), e - s)
            r0, r1 = max(s + a, lo), min(s + nxt, hi)
            if r1 > r0:
                new[r0 - lo : r1 - lo] = RETRACT_LL
                stats["relabeled_frames"] += int(r1 - r0)
        last_end = next(bb for aa, bb in move_runs if aa == retreat_starts[-1])
        if owned and any(aa >= last_end for aa, _ in move_runs):
            stats["segments_moves_after_last_retreat"] += 1
    stats["frames"] = hi - lo
    stats["changed_frames"] = int((new != ll[lo:hi]).sum())
    return new, stats, retreat_lens, retreat_dz


def plan(job: tuple[str, str | None, str | None]) -> tuple[str, np.ndarray, collections.Counter, list[int], list[float]]:
    """Phase 1 (read-only): rows of ``src`` whose label becomes RETRACT_LL, plus stats.

    Reads only READ_COLS of the file and its neighbours' edge rows (context for segments that
    straddle data files). Every file is planned before any is written, so no neighbour is read while
    it is being replaced.
    """
    src, prev_f, next_f = job
    parts = []
    if prev_f is not None:
        t = pq.read_table(prev_f, columns=READ_COLS)
        parts.append(t.slice(max(0, t.num_rows - CONTEXT_ROWS)))
    lo = sum(p.num_rows for p in parts)
    own = pq.read_table(src, columns=READ_COLS)
    parts.append(own)
    hi = lo + own.num_rows
    next_rows = 0
    if next_f is not None:
        t = pq.read_table(next_f, columns=READ_COLS)
        parts.append(t.slice(0, CONTEXT_ROWS))
        next_rows = t.num_rows
    window = pa.concat_tables(parts)
    cut_lo = prev_f is not None and lo == CONTEXT_ROWS
    cut_hi = next_f is not None and next_rows > CONTEXT_ROWS
    arrays = _arrays(window)
    new, stats, lens, dzs = relabel(arrays, lo, hi, cut_lo, cut_hi)
    rows = np.flatnonzero(new != arrays["ll"][lo:hi]).astype(np.int64)
    return src, rows, stats, lens, dzs


def write(job: tuple[str, str, np.ndarray]) -> int:
    """Phase 2: rewrite one file with ``rows`` relabeled; reads only that file. Returns rows changed."""
    src, dst, rows = job
    if dst == src and len(rows) == 0:
        return 0
    pf = pq.ParquetFile(src)
    table = pf.read()
    idx = table.schema.get_field_index(LL_COL)
    field = table.schema.field(idx)
    ll = table.column(idx).to_numpy(zero_copy_only=False).astype(object)
    if len(rows) and not (ll[rows] == REORIENT_LL).all():
        raise RuntimeError(f"{src}: planned rows are not all {REORIENT_LL} (file changed since planning?)")
    ll[rows] = RETRACT_LL
    table = table.set_column(idx, field, pa.array(ll.tolist(), type=field.type))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    pq.write_table(table, tmp, compression="snappy", row_group_size=max(1, pf.metadata.row_group(0).num_rows))
    os.replace(tmp, dst)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("--dst", default=None, help="Output dataset root; data/ files are written here (dry run if unset)")
    ap.add_argument("--in-place", action="store_true",
                    help="Rewrite SRC's data files in place (each via a temp file + atomic rename)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="Only the first N data files (testing)")
    args = ap.parse_args()

    if args.in_place:
        if args.dst is not None:
            ap.error("--in-place and --dst are exclusive")
        args.dst = args.src
    files = sorted(glob.glob(os.path.join(args.src, "data", "*", "*.parquet")))
    if args.limit:
        files = files[: args.limit]
    all_files = sorted(glob.glob(os.path.join(args.src, "data", "*", "*.parquet")))
    pos = {f: i for i, f in enumerate(all_files)}
    jobs = [
        (
            f,
            all_files[pos[f] - 1] if pos[f] > 0 else None,
            all_files[pos[f] + 1] if pos[f] + 1 < len(all_files) else None,
            None if args.dst is None else os.path.join(args.dst, os.path.relpath(f, args.src)),
        )
        for f in files
    ]
    total: collections.Counter = collections.Counter()
    lens: list[int] = []
    dzs: list[float] = []
    planned: dict[str, np.ndarray] = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (src, rows, st, ln, dz) in enumerate(ex.map(plan, [j[:3] for j in jobs]), 1):
            planned[src] = rows
            total.update(st)
            lens += ln
            dzs += dz
            if i % 50 == 0 or i == len(jobs):
                print(f"[plan {i}/{len(jobs)}] changed_frames={total['changed_frames']}", flush=True)
        if args.dst is not None:
            writes = [(src, dst, planned[src]) for src, _, _, dst in jobs]
            written = 0
            for i, n in enumerate(ex.map(write, writes), 1):
                written += n
                if i % 50 == 0 or i == len(writes):
                    print(f"[write {i}/{len(writes)}] written_frames={written}", flush=True)
            total["written_frames"] = written
    print(json.dumps({k: int(v) for k, v in sorted(total.items())}, indent=1))
    if lens:
        print("retreat move length (frames):", dict(sorted(collections.Counter(lens).items())))
        p = np.percentile(dzs, [0, 1, 50, 99, 100])
        print(f"retreat commanded dz (m): p0 {p[0]:.3f} p1 {p[1]:.3f} p50 {p[2]:.3f} p99 {p[3]:.3f} p100 {p[4]:.3f}")


if __name__ == "__main__":
    main()
