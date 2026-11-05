#!/usr/bin/env python3
"""Differential pixel check: with randomize_task_letters ON the blocks must LOOK different.

Every metadata check on this feature passed while the renders were untouched, because
glyph_map, pddl_goal and render_params all derive from the same sampling call -- they agree with
each other and none of them reads a pixel. The applier is the only code that warns, so when it
was never called the log showed zero warnings and a clean run and a broken run were identical.

This compares two renders of the SAME episode, one with the flag on and one off. The replay is
state-driven, so block poses are identical frame for frame and any difference in the block
regions is the decal swap and nothing else.

What it covers: the swap reached the image at all (the failure that shipped). It does NOT read
the letters -- that the offset lands on the intended atlas tile is pinned exactly, and far more
cheaply, by test_glyphs.py::test_delta_lands_exactly_on_the_target_tile.

    python scripts/datagen/verify_glyph_pixels.py --on <ds> --off <ds> [--frames 6]
"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

RGBK = "observation.images.camera:front_camera"
# Fraction of block-region pixels that must change before we call the swap visible. The decal is
# one face of a small cube, so even a correct swap moves only part of the region.
MIN_CHANGED = 0.02


def _episode_row(root: Path, ep: int):
    t = pq.read_table(sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))[0],
                      columns=["episode_index", "length", f"videos/{RGBK}/chunk_index",
                               f"videos/{RGBK}/file_index", f"videos/{RGBK}/from_timestamp"]).to_pandas()
    sel = t[t.episode_index == ep]
    return sel.iloc[0] if len(sel) else None


def frames(root: Path, ep: int, n: int):
    """`n` frames spread across the episode, as RGB arrays."""
    fps = json.loads((root / "meta" / "info.json").read_text()).get("fps", 25)
    r = _episode_row(root, ep)
    if r is None:
        return []
    mp4 = (root / "videos" / RGBK / f"chunk-{int(r[f'videos/{RGBK}/chunk_index']):03d}"
           / f"file-{int(r[f'videos/{RGBK}/file_index']):03d}.mp4")
    base = int(round(float(r[f"videos/{RGBK}/from_timestamp"]) * fps))
    L = int(r["length"])
    want = sorted({base + min(L - 1, int(L * f)) for f in np.linspace(0.02, 0.97, n)})
    out, wset = [], set(want)
    with av.open(str(mp4)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i in wset:
                out.append(fr.to_ndarray(format="rgb24").copy())
            if i > max(want):
                break
    return out


def block_mask(img):
    """Saturated, non-wood pixels: the printed block faces."""
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = ((s > 110) & (v > 60)) & ~((h > 5) & (h < 28) & (s < 170))
    return cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--on", required=True, help="dataset rendered with randomize_task_letters: true")
    ap.add_argument("--off", required=True, help="same episode with the flag false")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--frames", type=int, default=6)
    a = ap.parse_args()

    on, off = frames(Path(a.on), a.episode, a.frames), frames(Path(a.off), a.episode, a.frames)
    if not on or not off:
        return int(bool(print("FAIL: could not decode front-camera frames from both datasets")))
    if len(on) != len(off):
        print(f"note: decoded {len(on)} ON vs {len(off)} OFF frames; comparing the first "
              f"{min(len(on), len(off))}")

    gm = pq.read_table(sorted(Path(a.on).glob("data/chunk-*/file-*.parquet"))[0],
                       columns=["glyph_map"]).to_pandas()["glyph_map"].iloc[0]
    print(f"glyph_map: {gm}")

    worst, rows = 1.0, []
    for i, (x, y) in enumerate(zip(on, off)):
        m = (block_mask(x) | block_mask(y)).astype(bool)
        if m.sum() == 0:
            continue
        diff = (np.abs(x.astype(int) - y.astype(int)).max(axis=2) > 28) & m
        frac = diff.sum() / m.sum()
        worst = min(worst, frac)
        rows.append((i, int(m.sum()), frac))
        print(f"  frame {i}: block pixels={m.sum():>6}  changed={frac * 100:5.1f}%")

    if not rows:
        print("FAIL: no block pixels detected in either render")
        return 1
    if worst < MIN_CHANGED:
        print(f"\nFAIL: the weakest frame changed only {worst * 100:.1f}% of its block pixels "
              f"(need >{MIN_CHANGED * 100:.0f}%). The metadata claims a swap the image does not "
              f"show -- exactly the state this feature shipped in once.")
        return 1
    print(f"\nPASS: every frame changed >{MIN_CHANGED * 100:.0f}% of its block pixels "
          f"(weakest {worst * 100:.1f}%); the swap reached the image.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
