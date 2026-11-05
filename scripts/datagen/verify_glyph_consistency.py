#!/usr/bin/env python3
"""Four-way consistency check for a glyph-swapped dataset.

The assignment is written through three independent paths, and a bug in any one of them yields a
dataset that still validates:

  1. render_params.json episodes[].glyphs   <- render's wave loop (what was sampled/applied)
  2. the glyph_map column                   <- appended during passthrough
  3. the block_* tokens inside pddl_goal    <- rewritten during passthrough
  4. the task strings meta/tasks.parquet resolves task_index to  <- table rebuilt in passthrough

If all three agree, the sampling, the recording and the rewrite are consistent end to end. They
have already disagreed twice in this feature's history: render_params carried the assignment while
the parquet carried neither the column nor the rewrite (passthrough restored the source); and the
whole task table was replaced by the recorder's one-row placeholder ["manipulation"], silently
discarding the instruction text.

NOTE none of these four reads a PIXEL. They all descend from one sampling call, so they can agree
perfectly while the render is untouched -- which is exactly how this feature shipped broken. Pair
this with scripts/datagen/verify_glyph_pixels.py and with looking at the frames.

    python scripts/datagen/verify_glyph_consistency.py --dataset <rendered dataset>
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

# `_\d+` tail = the copy index (block_a_2 is the second "a" block), same as
# glyphs._TOKEN_RE. Without it this checker mis-parses second-copy names.
TOKEN = re.compile(r"\bblock_([A-Za-z0-9]+(?:_\d+)?)\b")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--max-report", type=int, default=8)
    a = ap.parse_args()
    root = Path(a.dataset)

    rp_path = root / "meta" / "render_params.json"
    if not rp_path.is_file():
        sys.exit(f"no render_params.json under {root}")
    rp = json.loads(rp_path.read_text())
    rp_glyphs = {int(e["episode_index"]): dict(e.get("glyphs") or {})
                 for e in (rp.get("episodes") or [])}

    dfile = sorted(root.glob("data/chunk-*/file-*.parquet"))
    if not dfile:
        sys.exit(f"no data parquet under {root}")
    names = pq.ParquetFile(dfile[0]).schema_arrow.names
    cols = [c for c in ("episode_index", "pddl_goal", "glyph_map", "task_index") if c in names]
    t = pq.read_table(dfile[0], columns=cols).to_pandas()

    # 4. the task strings, resolved through task_index
    task_txt, ep_tasks = {}, {}
    tp = root / "meta" / "tasks.parquet"
    if tp.is_file():
        d = pq.read_table(tp).to_pydict()
        idx = d.get("task_index") or []
        txt = next((v for k, v in d.items() if k != "task_index"), [])
        task_txt = {int(i): str(x) for i, x in zip(idx, txt)}
    if "task_index" in t.columns and task_txt:
        for _, rr in t.iterrows():
            ep_tasks.setdefault(int(rr["episode_index"]), set()).add(int(rr["task_index"]))

    have_col = "glyph_map" in t.columns
    print(f"  episodes in render_params: {len(rp_glyphs)}")
    print(f"  glyph_map column present : {have_col}")

    fails, checked = [], 0
    rows = {}
    for _, r in t.iterrows():
        rows.setdefault(int(r["episode_index"]), r)

    for ep, r in sorted(rows.items()):
        want = rp_glyphs.get(ep)
        if not want:
            continue
        checked += 1
        # 2. the column
        got = {}
        if have_col and r.get("glyph_map"):
            try:
                got = json.loads(r["glyph_map"])
            except Exception as e:
                fails.append(f"ep{ep}: glyph_map is not JSON ({e})")
                continue
        if got != want:
            fails.append(f"ep{ep}: glyph_map {got or '{}'} != render_params {want}")
            continue
        # 3. the goal text: no ORIGINAL block token may survive the rewrite
        goal = r.get("pddl_goal") or ""
        present = {m.group(1).lower() for m in TOKEN.finditer(goal)}
        expected = {g.lower() for g in want.values()}
        originals = {b.split("_", 1)[1].lower() for b in want}
        # A goal names a SUBSET of the scene: a block that is present but not mentioned keeps its
        # new glyph out of the text entirely. Requiring every new glyph to appear was wrong -- it
        # passed only on problems whose goal happens to mention every block, and fired on all four
        # episodes of a correctly-rewritten real one. The invariant is one-directional: every
        # token must be accounted for, not every glyph must be used.
        stale = (originals & present) - expected
        unknown = present - expected - originals
        if stale:
            fails.append(f"ep{ep}: pddl_goal still names original block(s) {sorted(stale)} "
                         f"-- the rewrite missed them")
        elif unknown:
            # tokens that are neither a new glyph nor a known block: a colour cuboid or a block
            # the sampler skipped is fine, so only report when it is none of those.
            leftover = sorted(t for t in unknown if len(t) <= 2)
            if leftover:
                fails.append(f"ep{ep}: pddl_goal names {leftover}, which is neither a new glyph "
                             f"nor an original block of this episode")
        # 4. the task strings this episode resolves to
        for ti in sorted(ep_tasks.get(ep, ())):
            s_txt = task_txt.get(ti)
            if not s_txt:
                fails.append(f"ep{ep}: task_index {ti} resolves to nothing in tasks.parquet")
                continue
            if s_txt.strip().lower() == "manipulation":
                fails.append(f"ep{ep}: task is the recorder placeholder 'manipulation' -- the "
                             f"real instruction text was discarded, not rewritten")
                continue
            seen = {m.group(1).lower() for m in TOKEN.finditer(s_txt)}
            bad = ({b.split("_", 1)[1].lower() for b in want} & seen) - expected
            if bad:
                fails.append(f"ep{ep}: task still names original block(s) {sorted(bad)}: {s_txt[:80]}")

    print(f"  episodes checked         : {checked}")
    print(f"  disagreements            : {len(fails)}")
    for f in fails[:a.max_report]:
        print(f"    {f}")
    if not checked:
        print("\n  NOTHING CHECKED -- render_params has no glyphs (was the flag on?)")
        return 1
    if fails:
        print("\n  FAIL: the write paths disagree.")
        return 1
    n_tasks = len(task_txt)
    print(f"  task table rows          : {n_tasks}"
          + ("" if n_tasks else "  (no tasks.parquet -- task text NOT checked)"))
    print("\n  PASS: render_params, glyph_map, pddl_goal and the task table all agree.\n"
          "  (metadata only -- none of this reads a pixel; see verify_glyph_pixels.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
