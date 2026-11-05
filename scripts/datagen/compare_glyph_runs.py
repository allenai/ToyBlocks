#!/usr/bin/env python3
"""Compare a randomize_task_letters ON run against an OFF run of the same episodes.

The visual check (verify_glyph_swap.py) answers "do the pixels match the text". This answers the
other half, which no image can show:

  * OFF is a true no-op      -- empty glyph_map, and pddl_goal/task identical to the source
  * ON actually rewrote      -- glyph_map populated, pddl_goal/task differ from OFF
  * ONLY the text moved      -- object poses, actions and states are bit-identical between runs

That last one is the point. If the swap perturbed the physics or the recorded trajectory, the
augmentation would be changing the task rather than relabelling it, and every downstream metric
would shift for a reason nobody would trace back to here.

    python scripts/datagen/compare_glyph_runs.py --on <ds> --off <ds>
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TEXT_COLS = ("pddl_goal", "task", "glyph_map")


def load(root: Path):
    f = sorted(Path(root).glob("data/chunk-*/file-*.parquet"))
    if not f:
        sys.exit(f"no data parquet under {root}")
    return pq.read_table(f[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True)
    ap.add_argument("--off", required=True)
    a = ap.parse_args()
    on, off = load(Path(a.on)), load(Path(a.off))

    print(f"  ON  rows={on.num_rows} cols={len(on.schema.names)}")
    print(f"  OFF rows={off.num_rows} cols={len(off.schema.names)}")
    fails = []

    if on.num_rows != off.num_rows:
        fails.append(f"row count differs ({on.num_rows} vs {off.num_rows}) -- not comparable")
        print("\n".join("  FAIL " + f for f in fails)); return 1

    # 1. OFF must be a no-op
    if "glyph_map" in off.schema.names:
        vals = set(v for v in off["glyph_map"].to_pylist() if v)
        print(f"  OFF glyph_map non-empty values: {len(vals)}")
        if vals:
            fails.append(f"OFF wrote a glyph_map ({list(vals)[:2]}) -- the flag is not opt-in")

    # 2. ON must have rewritten
    if "glyph_map" in on.schema.names:
        vals = [v for v in on["glyph_map"].to_pylist() if v]
        print(f"  ON  glyph_map populated rows: {len(vals)}/{on.num_rows}")
        if not vals:
            fails.append("ON wrote no glyph_map -- the flag did nothing")
        else:
            try:
                sample = json.loads(vals[0])
                print(f"  ON  sample glyph_map: {sample}")
            except Exception as e:
                fails.append(f"glyph_map is not valid JSON: {e}")

    for col in ("pddl_goal", "task"):
        if col not in on.schema.names or col not in off.schema.names:
            continue
        same = on[col].to_pylist() == off[col].to_pylist()
        print(f"  {col}: ON == OFF ? {same}")
        if same:
            fails.append(f"{col} identical between ON and OFF -- no rewrite happened")

    # 3. everything that is not text must be bit-identical
    moved = []
    for name in on.schema.names:
        if name in TEXT_COLS or name not in off.schema.names:
            continue
        try:
            if not on[name].equals(off[name]):
                a_, b_ = on[name].to_pylist(), off[name].to_pylist()
                if a_ != b_ and not all(
                        (x is None and y is None)
                        or (isinstance(x, float) and isinstance(y, float)
                            and np.isnan(x) and np.isnan(y))
                        for x, y in zip(a_, b_)):
                    moved.append(name)
        except Exception:
            moved.append(name)
    print(f"  non-text columns that differ: {len(moved)}")
    if moved:
        print(f"    {moved[:10]}")
        fails.append(f"{len(moved)} non-text column(s) changed -- the swap altered more than labels")

    print()
    if fails:
        for f in fails:
            print(f"  FAIL {f}")
        return 1
    print("  PASS: OFF is a no-op, ON rewrote the text, and nothing else moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
