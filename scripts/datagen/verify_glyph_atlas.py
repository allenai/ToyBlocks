#!/usr/bin/env python3
"""Crop the real decal atlas at each glyph's computed anchor and build a labelled contact sheet.

Checks `domain_randomization.glyphs.anchor_for` against the SHIPPED texture rather than against
dataset_description.json -- the json and the png could in principle disagree, and it is the png
the renderer samples. If a tile does not show the glyph it is labelled with, the atlas math is
wrong and every swapped block would carry the wrong letter.

Validates mesh-UV space only. It does NOT settle whether OmniPBR's `texture_translate` adds or
subtracts the offset; that needs an actual render.

    python scripts/datagen/verify_glyph_atlas.py [--out out.png]
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from toyblocks.domain_randomization import glyphs as G

ATLAS = (Path(__file__).resolve().parents[2] / "assets" / "objects" / "train_blocks_large"
         / "usd" / "Textures" / "alphabet_block_large_decals_albedotransparency.png")
FRONT_CELL = (1, 1)      # front face within the 3x2 quadrant (proc_blocks/simplified/atlas.py)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", default=str(ATLAS))
    ap.add_argument("--out", default="glyph_atlas_contact_sheet.png")
    ap.add_argument("--cell", type=int, default=150)
    ap.add_argument("--cols", type=int, default=6)
    a = ap.parse_args()

    im = cv2.imread(a.atlas, cv2.IMREAD_UNCHANGED)
    if im is None:
        sys.exit(f"could not read atlas: {a.atlas}")
    H, W = im.shape[:2]
    tw, th = W // G.GRID_W, H // G.GRID_H
    if im.shape[2] == 4:                      # composite over white: the glyphs are cut out
        rgb = im[:, :, :3].astype(np.float32)
        al = im[:, :, 3:4].astype(np.float32) / 255.0
        flat = (rgb * al + 255.0 * (1 - al)).astype(np.uint8)
    else:
        flat = im

    rows = (len(G.GLYPHS) + a.cols - 1) // a.cols
    sheet = np.full((rows * (a.cell + 26) + 10, a.cols * (a.cell + 10) + 10, 3), 245, np.uint8)
    for i, g in enumerate(G.GLYPHS):
        ax, ay = G.anchor_for(g)
        cx, cy = ax + FRONT_CELL[0], ay + FRONT_CELL[1]
        tile = flat[cy * th:(cy + 1) * th, cx * tw:(cx + 1) * tw]
        if tile.size == 0:
            sys.exit(f"glyph {g}: anchor {(ax, ay)} lands outside a {G.GRID_W}x{G.GRID_H} atlas")
        tile = cv2.resize(tile, (a.cell, a.cell), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, a.cols)
        y0, x0 = 10 + r * (a.cell + 26), 10 + c * (a.cell + 10)
        sheet[y0:y0 + a.cell, x0:x0 + a.cell] = tile
        cv2.rectangle(sheet, (x0 - 1, y0 - 1), (x0 + a.cell, y0 + a.cell), (180, 180, 180), 1)
        cv2.putText(sheet, f"{g}  anchor{ax},{ay}", (x0, y0 + a.cell + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.imwrite(a.out, sheet)
    print(f"[atlas] {W}x{H}, tile {tw}x{th}, {len(G.GLYPHS)} glyphs -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
