#!/usr/bin/env python3
"""Check a glyph-swapped dataset: do the rendered letters match the rewritten goal?

The failure this exists to catch is invisible to every other check. If `pddl_goal` says
"stack X on Q" while the blocks render A and I, the parquet is internally consistent, the
schema validates, and a language-conditioned policy trains on a contradiction. Only reading
the letters off the pixels catches it.

For each sampled episode this renders one frame with every block circled and labelled with the
glyph the dataset CLAIMS it carries, and prints the rewritten pddl_goal / task beside it. A
human (or a vision model) then confirms the labels match what is drawn on the blocks.

    python scripts/datagen/verify_glyph_swap.py --dataset <rerendered dataset> --out-dir out/
"""
import argparse
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pyarrow.parquet as pq

from blocksuite.data_annotation.annotation_utils import resolve_frame_camera_model

RGBK = "observation.images.camera:front_camera"


def qmat(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", default="glyph_check")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--frac", type=float, default=0.5)
    a = ap.parse_args()
    root, out = Path(a.dataset), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    info = json.loads((root / "meta" / "info.json").read_text())
    fps = info.get("fps", 25)
    if "glyph_map" not in (info.get("features") or {}):
        print("[glyph] WARNING: dataset has no glyph_map column -- was randomize_task_letters on?")

    dfile = sorted(root.glob("data/chunk-*/file-*.parquet"))[0]
    names = pq.ParquetFile(dfile).schema_arrow.names
    obj = [c for c in names if c.startswith("observation.state.object_pose.")
           and "workspace_table" not in c]
    keep = [c for c in (["episode_index", "frame_index", "pddl_goal", "task", "glyph_map",
                         "observation.state.cube_scale"] + obj
                        + [n for n in names if "camera_pose.camera:front" in n
                           or "camera_intrinsics.camera:front" in n]) if c in names]
    df = pq.read_table(dfile, columns=keep).to_pandas()
    cam = resolve_frame_camera_model(df)
    eps, fidx = df["episode_index"].to_numpy(), df["frame_index"].to_numpy()

    em = pq.read_table(sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))[0],
                       columns=["episode_index", f"videos/{RGBK}/chunk_index",
                                f"videos/{RGBK}/file_index",
                                f"videos/{RGBK}/from_timestamp"]).to_pandas()

    whole = [int(e) for e in dict.fromkeys(eps.tolist())][:a.episodes]
    written = []
    for e in whole:
        sel = em[em.episode_index == e]
        if not len(sel):
            continue
        row = sel.iloc[0]
        mp4 = (root / "videos" / RGBK
               / f"chunk-{int(row[f'videos/{RGBK}/chunk_index']):03d}"
               / f"file-{int(row[f'videos/{RGBK}/file_index']):03d}.mp4")
        if not mp4.exists():
            continue
        rows = np.flatnonzero(eps == e)
        r = int(rows[int(a.frac * (len(rows) - 1))])
        idx = int(round(float(row[f"videos/{RGBK}/from_timestamp"]) * fps)) + int(fidx[r])
        img = None
        with av.open(str(mp4)) as c:
            for i, fr in enumerate(c.decode(video=0)):
                if i == idx:
                    img = fr.to_ndarray(format="rgb24").copy(); break
        if img is None:
            continue
        H, W, _ = img.shape
        sx, sy = W / (2 * cam.cx[r]), H / (2 * cam.cy[r])
        gm = {}
        raw = df["glyph_map"].iloc[r] if "glyph_map" in df.columns else ""
        if raw:
            try:
                gm = json.loads(raw)
            except Exception:
                pass
        half = float(df["observation.state.cube_scale"].iloc[r]) / 2.0
        for c2 in obj:
            v = df[c2].iloc[r]
            if v is None or len(v) != 7:
                continue
            top = np.asarray(v[:3], float) + qmat(np.asarray(v[3:7], float)) @ np.array([0, 0, half])
            uv = cam.project(top.reshape(1, 3), np.array([r]))[0]
            if not np.isfinite(uv).all():
                continue
            x, y = int(round(uv[0] * sx)), int(round(uv[1] * sy))
            block = c2.split(".")[-1]
            claim = gm.get(block, "?")
            cv2.circle(img, (x, y), 16, (0, 230, 230), 2)
            cv2.putText(img, claim, (x - 10, y - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (255, 255, 0), 2, cv2.LINE_AA)
        pad = np.full((120, W, 3), 25, np.uint8)
        goal = " ".join((df["pddl_goal"].iloc[r] or "").split())
        task = (df["task"].iloc[r] or "") if "task" in df.columns else ""
        for k, line in enumerate((f"ep{e}  glyph_map={json.dumps(gm)}"[:130],
                                  f"goal: {goal}"[:130], f"task: {task}"[:130])):
            cv2.putText(pad, line, (8, 30 + k * 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (235, 235, 235), 1, cv2.LINE_AA)
        p = out / f"glyph_ep{e}.png"
        cv2.imwrite(str(p), np.vstack([img, pad])[:, :, ::-1])
        written.append(str(p))
        print(f"[glyph] ep{e}: {gm} -> {p}")
    print(f"[glyph] {len(written)} frame(s) written to {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
