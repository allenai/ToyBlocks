#!/usr/bin/env python3
"""Grid of near-final front-camera frames, each annotated with that episode's glyph swap.

One panel per episode, taken near the LAST frame so the finished stack is visible, captioned
with `original -> new` for every block and the rewritten pddl_goal. Reading the caption against
the letters on the blocks is the only check that the augmentation is coherent: the pixels and
the labels are written by different code paths, and a disagreement is invisible to every
schema- or value-level test.

    python scripts/datagen/glyph_grid.py --dataset <rendered> --out grid.png
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

RGBK = "observation.images.camera:front_camera"


def frame_near_end(root: Path, ep: int, fps: int, frac: float):
    """Decode one frame `frac` of the way through the episode (1.0 = last)."""
    em = pq.read_table(sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))[0],
                       columns=["episode_index", "length",
                                f"videos/{RGBK}/chunk_index", f"videos/{RGBK}/file_index",
                                f"videos/{RGBK}/from_timestamp"]).to_pandas()
    sel = em[em.episode_index == ep]
    if not len(sel):
        return None
    r = sel.iloc[0]
    mp4 = (root / "videos" / RGBK / f"chunk-{int(r[f'videos/{RGBK}/chunk_index']):03d}"
           / f"file-{int(r[f'videos/{RGBK}/file_index']):03d}.mp4")
    if not mp4.exists():
        return None
    base = int(round(float(r[f"videos/{RGBK}/from_timestamp"]) * fps))
    # clamp one frame in from the true end: the final frame is often mid-fade or duplicated
    want = base + max(0, int(int(r["length"]) * frac) - 1)
    img = None
    with av.open(str(mp4)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i == want:
                img = fr.to_ndarray(format="rgb24").copy()
                break
            if i > want:
                break
    return img


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default="glyph_grid.png")
    ap.add_argument("--episodes", type=int, default=48)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--panel-w", type=int, default=0,
                    help="downscale each panel to this width; 0 = native. A 8x6 grid of\n"
                         "640px frames is 5kpx wide, which most viewers downsample badly.")
    ap.add_argument("--frac", type=float, default=0.97, help="1.0 = last frame")
    a = ap.parse_args()
    root = Path(a.dataset)

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info.get("fps", 25)
    # render_params.json records each episode's glyphs as the wave is rendered, so it is the
    # only mapping available while a problem is still in flight (the data parquet has no footer
    # until the problem is finalized). Used as a fallback below, never in preference.
    rp_glyphs = {}
    rp_path = root / "meta" / "render_params.json"
    if rp_path.exists():
        try:
            for e in json.loads(rp_path.read_text()).get("episodes") or []:
                if e.get("glyphs"):
                    rp_glyphs[int(e["episode_index"])] = e["glyphs"]
        except (OSError, ValueError, KeyError, TypeError):
            pass

    dfile = sorted(root.glob("data/chunk-*/file-*.parquet"))[0]
    try:
        names = pq.ParquetFile(dfile).schema_arrow.names
        cols = [c for c in ("episode_index", "pddl_goal", "task", "glyph_map") if c in names]
        df = pq.read_table(dfile, columns=cols).to_pandas()
        eps = df["episode_index"].to_numpy()
    except pa.ArrowInvalid:
        if not rp_glyphs:
            raise
        # The episode->video index is in the SAME unfinalized state, and there is no safe way to
        # guess it: pairing a frame with the wrong episode's glyphs would produce a grid that
        # looks fine and says the wrong thing, which is the one error this tool exists to catch.
        try:
            pq.ParquetFile(sorted((root / "meta" / "episodes")
                                  .glob("chunk-*/file-*.parquet"))[0])
        except pa.ArrowInvalid:
            sys.exit("[grid] this render is still in flight: neither the data nor the episode "
                     "parquet has a footer yet, so frames cannot be matched to episodes. "
                     f"render_params.json already lists {len(rp_glyphs)} episode(s) -- rerun "
                     "once the problem is finalized.")
        print(f"[grid] data parquet has no footer yet; using render_params.json for "
              f"{len(rp_glyphs)} episode(s). pddl_goal unavailable.")
        df, eps = None, np.array(sorted(rp_glyphs))

    panels = []
    for ep in list(dict.fromkeys(eps.tolist()))[:a.episodes]:
        img = frame_near_end(root, int(ep), fps, a.frac)
        if img is None:
            continue
        gm, goal = {}, ""
        if df is not None:
            row = int(np.flatnonzero(eps == ep)[0])
            if "glyph_map" in df.columns and df["glyph_map"].iloc[row]:
                try:
                    gm = json.loads(df["glyph_map"].iloc[row])
                except ValueError:
                    pass
            if "pddl_goal" in df:
                goal = " ".join((df["pddl_goal"].iloc[row] or "").split())
        if not gm:
            gm = rp_glyphs.get(int(ep), {})
        panels.append((int(ep), img, gm, goal))

    if not panels:
        sys.exit("no frames decoded -- does the dataset have front-camera videos?")

    if a.panel_w:
        panels = [(ep, cv2.resize(img, (a.panel_w,
                                        int(img.shape[0] * a.panel_w / img.shape[1])),
                                  interpolation=cv2.INTER_AREA), gm, goal)
                  for ep, img, gm, goal in panels]
    H, W = panels[0][1].shape[:2]
    cap_h = 46 if W < 420 else 92
    cols_n = min(a.cols, len(panels))
    rows_n = (len(panels) + cols_n - 1) // cols_n
    sheet = np.full((rows_n * (H + cap_h) + 8, cols_n * (W + 8) + 8, 3), 18, np.uint8)
    for k, (ep, img, gm, goal) in enumerate(panels):
        r, c = divmod(k, cols_n)
        y0, x0 = 8 + r * (H + cap_h), 8 + c * (W + 8)
        sheet[y0:y0 + H, x0:x0 + W] = img
        swap = "  ".join(f"{b.replace('block_', '')}->{g}" for b, g in sorted(gm.items())) \
            or "(no glyph_map: swap OFF or not applied)"
        small = W < 420
        fs_ = 0.34 if small else 0.46
        lines = [f"ep{ep}  {swap}"]
        if not small:
            lines += textwrap.wrap(goal, width=max(40, W // 7))[:2]
        for j, line in enumerate(lines[:3]):
            cv2.putText(sheet, line[:160], (x0 + 3, y0 + H + (16 if small else 22) + j * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, fs_,
                        (90, 230, 255) if j == 0 else (215, 215, 215), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, sheet[:, :, ::-1])
    print(f"[grid] {len(panels)} episode(s) -> {Path(a.out).resolve()}  {sheet.shape}")
    for ep, _, gm, _ in panels:
        print(f"   ep{ep}: {gm or '{}'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
