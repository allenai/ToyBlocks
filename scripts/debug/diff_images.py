#!/usr/bin/env python3
"""Compare recorded camera frames A (blocksuite) vs B (classic), per instance.

Only meaningful once both datasets share a matched scene + cameras. Reads
front_camera / gripper_camera frames from each LeRobot dataset's mp4s, aligns by
reset-state instance, and produces side-by-side (A | B | |diff|) panels plus a
per-camera mean-pixel-difference summary.

    RG_OUT=scratchpad/red_green_2block/test1 .venv/bin/python scripts/debug/diff_images.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import av
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("RG_OUT", "scratchpad/red_green_2block/test1"))
if not OUT.is_absolute():
    OUT = REPO / OUT
A_ROOT = REPO / "scratchpad/red_green_2block/red_on_green_2block/dataset/blocks_success"
B_ROOT = REPO / "scratchpad/red_green_2block/classic/data_matched/blocks_success"
CLASSIC = REPO / "scratchpad/red_green_2block/classic/red_on_green_2block"
RESET_RED_XY = np.array([
    json.loads((CLASSIC / f"{i+1}.json").read_text())["block_positions"]["block_red1"][:2]
    for i in range(8)
])
CAMS = ["front_camera", "gripper_camera"]


def read_frame(path, frame_idx):
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for i, fr in enumerate(c.decode(s)):
            if i == frame_idx:
                return fr.to_ndarray(format="rgb24")
    return None


def inst_to_episode(root):
    """map reset-state instance -> (episode_index, length) using parquet."""
    pfs = sorted(root.glob("data/chunk-*/file-*.parquet"))
    fail = root.parent / "blocks_failure"
    if fail.exists():
        pfs += sorted(fail.glob("data/chunk-*/file-*.parquet"))
    m = {}
    for pf in pfs:
        d = pq.read_table(pf).to_pydict()
        ep = np.asarray(d["episode_index"]).reshape(-1)
        for e in sorted(set(ep.tolist())):
            idx = np.flatnonzero(ep == e)
            if "instance_id" in d:
                inst = int(np.asarray(d["instance_id"][idx[0]]).reshape(-1)[0])
            else:
                r = np.asarray(d["observation.state.object_pose.block_red1"][idx[0]])[:2]
                inst = int(np.argmin(np.linalg.norm(RESET_RED_XY - r, axis=1)))
            m.setdefault(inst, (int(e), len(idx)))
    return m


def vid_path(root, cam, episode):
    info = json.loads((root / "meta/info.json").read_text())
    key = next(k for k in info["features"] if k.endswith(cam) and "images" in k)
    tmpl = info["video_path"]
    return root / tmpl.format(video_key=key, chunk_index=episode // 1000, file_index=episode % 1000)


def main():
    Am, Bm = inst_to_episode(A_ROOT), inst_to_episode(B_ROOT)
    insts = sorted(set(Am) & set(Bm))
    print(f"common instances: {insts}")
    stats = {c: [] for c in CAMS}

    for cam in CAMS:
        fig, axes = plt.subplots(len(insts), 3, figsize=(9, 2.4 * len(insts)))
        axes = np.atleast_2d(axes)
        for r, inst in enumerate(insts):
            ea, la = Am[inst]; eb, lb = Bm[inst]
            fidx = min(la, lb) // 2  # a mid frame (manipulation in progress)
            a = read_frame(vid_path(A_ROOT, cam, ea), fidx)
            b = read_frame(vid_path(B_ROOT, cam, eb), fidx)
            if a is None or b is None:
                continue
            h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
            a, b = a[:h, :w], b[:h, :w]
            diff = np.abs(a.astype(int) - b.astype(int)).astype(np.uint8)
            stats[cam].append(float(diff.mean()))
            for cidx, (img, lab) in enumerate([(a, "A blocksuite"), (b, "B classic"), (diff, "|diff|")]):
                axes[r, cidx].imshow(img)
                axes[r, cidx].set_title(f"inst {inst} {lab}" if r == 0 or cidx else f"inst {inst} {lab}",
                                        fontsize=7)
                axes[r, cidx].axis("off")
        fig.suptitle(f"{cam}: A (blocksuite) vs B (classic), mid-episode frame  "
                     f"[mean |diff| = {np.mean(stats[cam]):.1f}/255]", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        op = OUT / f"image_diff_{cam}.png"
        fig.savefig(op, dpi=110); plt.close(fig)
        print(f"wrote {op}  mean|diff|={np.mean(stats[cam]):.2f}/255")


if __name__ == "__main__":
    main()
