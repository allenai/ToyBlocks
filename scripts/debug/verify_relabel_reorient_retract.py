#!/usr/bin/env python3
"""Check a relabel_reorient_retract.py output against its source dataset.

Per data file: same row count and schema (incl. metadata), every column except
action.low_level_action identical, and that column changed only reorient_traj -> reorient_retract_traj.
Also checks every reorient segment in the output starts with reorient_traj and ends in
reorient_retract_traj, and reports the label counts.

Usage: verify_relabel_reorient_retract.py SRC_DATASET DST_DATASET [--workers N]
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
import pyarrow.compute as pc
import pyarrow.parquet as pq

LL_COL = "action.low_level_action"
REORIENT_LL = "reorient_traj"
RETRACT_LL = "reorient_retract_traj"


def _same_column(x: pa.ChunkedArray, y: pa.ChunkedArray) -> bool:
    """Column equality with NaN == NaN (object poses of blocks absent from a scene are NaN, which
    ChunkedArray.equals counts as different)."""
    if x.equals(y):
        return True
    x, y = x.combine_chunks(), y.combine_chunks()
    if x.type != y.type or x.null_count != y.null_count:
        return False
    if pa.types.is_list(x.type) or pa.types.is_large_list(x.type) or pa.types.is_fixed_size_list(x.type):
        if not np.array_equal(pc.list_value_length(x).to_numpy(zero_copy_only=False),
                              pc.list_value_length(y).to_numpy(zero_copy_only=False)):
            return False
        x, y = pc.list_flatten(x), pc.list_flatten(y)
    if not pa.types.is_floating(x.type):
        return False
    return bool(np.array_equal(x.to_numpy(zero_copy_only=False), y.to_numpy(zero_copy_only=False), equal_nan=True))


def check(pair: tuple[str, str]) -> tuple[str, list[str], collections.Counter]:
    src, dst = pair
    errors: list[str] = []
    counts: collections.Counter = collections.Counter()
    if not os.path.exists(dst):
        return src, ["missing output file"], counts
    a, b = pq.read_table(src), pq.read_table(dst)
    if a.num_rows != b.num_rows:
        errors.append(f"rows {a.num_rows} != {b.num_rows}")
    if not a.schema.equals(b.schema, check_metadata=True):
        errors.append("schema differs")
    if errors:
        return src, errors, counts
    for name in a.column_names:
        if name != LL_COL and not _same_column(a.column(name), b.column(name)):
            errors.append(f"column {name} differs")
    la = a.column(LL_COL).to_numpy(zero_copy_only=False).astype(object)
    lb = b.column(LL_COL).to_numpy(zero_copy_only=False).astype(object)
    diff = la != lb
    bad = diff & ~((la == REORIENT_LL) & (lb == RETRACT_LL))
    if bad.any():
        errors.append(f"{int(bad.sum())} label changes other than {REORIENT_LL}->{RETRACT_LL}")
    counts["changed"] += int(diff.sum())
    counts[REORIENT_LL] += int((lb == REORIENT_LL).sum())
    counts[RETRACT_LL] += int((lb == RETRACT_LL).sum())
    # A retract run must directly follow reorient_traj (never open an action / episode).
    ep = b.column("episode_index").to_numpy()
    starts = np.flatnonzero((lb == RETRACT_LL) & np.r_[True, (lb[1:] != lb[:-1]) | (ep[1:] != ep[:-1])])
    for s in starts:
        if s == 0:
            counts["retract_runs_at_file_start"] += 1  # continuation from the previous file
        elif lb[s - 1] != REORIENT_LL or ep[s - 1] != ep[s]:
            errors.append(f"retract run at row {s} follows {lb[s - 1]!r}")
    counts["retract_runs"] += len(starts)
    return src, errors, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    files = sorted(glob.glob(os.path.join(args.src, "data", "*", "*.parquet")))
    pairs = [(f, os.path.join(args.dst, os.path.relpath(f, args.src))) for f in files]
    extra = set(glob.glob(os.path.join(args.dst, "data", "*", "*"))) - {d for _, d in pairs}
    total: collections.Counter = collections.Counter()
    failed = 0
    with ProcessPoolExecutor(args.workers) as ex:
        for src, errors, counts in ex.map(check, pairs):
            total.update(counts)
            if errors:
                failed += 1
                print(f"FAIL {os.path.relpath(src, args.src)}: {'; '.join(errors[:5])}")
    print(json.dumps({"files": len(pairs), "failed": failed, "extra_files_in_dst": sorted(extra),
                      **{k: int(v) for k, v in sorted(total.items())}}, indent=1))


if __name__ == "__main__":
    main()
