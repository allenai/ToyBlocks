#!/usr/bin/env python3
"""Check a MERGED glyph-swapped dataset: the per-problem rewrites must survive the merge.

merge_success builds one global tasks.parquet from the union of every problem's strings and
remaps each problem's LOCAL task_index to the global one. With randomize_task_letters, each
problem's table was itself rebuilt per episode, so this remap is doing real work -- and if it
mismatches, episodes silently resolve to ANOTHER episode's instruction text. The dataset still
loads; only the words are wrong.

Checks, per episode:
  * glyph_map and pddl_goal survived the merge at all (a column-filtering merge could drop them);
  * every task string the episode's task_index resolves to names that episode's NEW glyphs and
    none of its ORIGINAL block letters;
  * pddl_goal likewise carries no original letters.

Metadata only. It cannot tell you the pixels are right -- use glyph_grid.py and look.

    python scripts/datagen/verify_merged_glyph_dataset.py --dataset <merged root> [--episodes 40]
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

TOKEN = re.compile(r"\bblock_([A-Za-z0-9]+(?:_\d+)?)\b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", type=int, default=0, help="0 = every episode")
    ap.add_argument("--max-report", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.dataset)

    files = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not files:
        print(f"FAIL: no data parquet under {root}")
        return 1
    names = pq.ParquetFile(files[0]).schema_arrow.names
    for col in ("glyph_map", "pddl_goal"):
        if col not in names:
            print(f"FAIL: the merge dropped `{col}` -- the per-episode rewrite is unrecoverable")
            return 1
    tp = root / "meta" / "tasks.parquet"
    if not tp.is_file():
        print(f"FAIL: no meta/tasks.parquet under {root}")
        return 1
    tasks = pd.read_parquet(tp)
    idx2str = {int(v): str(s) for s, v in tasks["task_index"].items()}
    print(f"  merged task table rows : {len(tasks)}")

    cols = [c for c in ("episode_index", "glyph_map", "pddl_goal", "task_index") if c in names]
    # Stream per FILE and keep one row per episode plus that episode's task_index set. A merged
    # 5000-episode dataset is ~10.8M rows and glyph_map/pddl_goal repeat the same string on every
    # row, so concatenating everything costs several GB to answer a per-EPISODE question.
    first: dict = {}
    tidx: dict = {}
    for f in files:
        t = pq.read_table(f, columns=cols).to_pandas()
        for ep, grp in t.groupby("episode_index", sort=False):
            e = int(ep)
            if e not in first:
                r = grp.iloc[0]
                first[e] = (r.get("glyph_map"), r.get("pddl_goal"))
            if "task_index" in grp:
                tidx.setdefault(e, set()).update(int(v) for v in grp["task_index"].unique())
        del t
    eps = sorted(first)
    if a.episodes:
        eps = eps[:a.episodes]
    print(f"  episodes checked       : {len(eps)}")

    fails, swapped = [], 0
    for ep in eps:
        raw, goal_txt = first[ep]
        if not raw:
            continue
        swapped += 1
        gm = json.loads(raw)
        new = {v.lower() for v in gm.values()}
        orig = {k.split("_", 1)[1].lower() for k in gm}

        used = sorted(tidx.get(ep, ()))
        strs = {idx2str.get(i, "") for i in used}
        missing_idx = [i for i in used if i not in idx2str]
        if missing_idx:
            fails.append(f"ep{int(ep)}: task_index {missing_idx[:3]} absent from the merged table")
            continue
        toks = {m.group(1).lower() for s in strs for m in TOKEN.finditer(s)}
        stale = (orig & toks) - new
        if stale:
            fails.append(f"ep{int(ep)}: task strings still name original block(s) {sorted(stale)}")
            continue
        goal = goal_txt or ""
        gtoks = {m.group(1).lower() for m in TOKEN.finditer(goal)}
        gstale = (orig & gtoks) - new
        if gstale:
            fails.append(f"ep{int(ep)}: pddl_goal still names original block(s) {sorted(gstale)}")

    print(f"  episodes with a swap   : {swapped}")
    print(f"  disagreements          : {len(fails)}")
    for f in fails[:a.max_report]:
        print(f"    {f}")
    if not swapped:
        print("\n  NOTHING CHECKED -- no episode carries a glyph_map (was the flag on?)")
        return 1
    if fails:
        print("\n  FAIL: the merge did not preserve the per-episode rewrite.")
        return 1
    print("\n  PASS: every episode resolves to task strings naming its OWN new glyphs.\n"
          "  (metadata only -- now LOOK at the frames with glyph_grid.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
