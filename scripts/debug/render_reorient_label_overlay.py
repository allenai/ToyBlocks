#!/usr/bin/env python3
"""Overlay low-level labels on the camera videos around each reorient action, to eyeball
the reorient_traj / reorient_retract_traj split written by relabel_reorient_retract.py.

Picks episodes from DATASET whose reorient segments have 1, 2 and 3 flip attempts (and one whose
reorient segment crosses a data-file boundary), then writes one mp4 per episode with a clip per
reorient action: front + gripper camera, label panel and a label timeline with a cursor.

Usage: render_reorient_label_overlay.py DATASET OUT_DIR [--max-files N] [--file-range FIRST END]
"""
from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

LL_COL = "action.low_level_action"
CAMS = ["observation.images.camera:front_camera", "observation.images.camera:gripper_camera"]
PAD_BEFORE, PAD_AFTER = 10, 25  # frames of context around each reorient action
PANEL_H, TIMELINE_H, CAM_H = 150, 36, 360
COLORS: dict[str, tuple[int, int, int]] = {  # RGB
    "reorient_start_traj": (90, 160, 255),
    "reorient_traj": (60, 200, 90),
    "reorient_retract_traj": (255, 120, 30),
}
OTHER_COLOR = (150, 150, 150)
COLS = ["index", "episode_index", "frame_index", "action.high_level_action", "action.high_level_args", LL_COL,
        "goal.action", "goal.action_secondary", "action.ee_target", "observation.state.gripper_position"]


@dataclass
class Clip:
    episode: int
    start: int  # episode frame index, inclusive
    end: int  # exclusive
    attempts: int


def _segments(ll: np.ndarray, hl: np.ndarray) -> list[tuple[int, int]]:
    change = np.r_[True, (ll[1:] != ll[:-1]) | (hl[1:] != hl[:-1])]
    starts = np.flatnonzero(change)
    return list(zip(starts, np.r_[starts[1:], len(ll)]))


def episode_rows(dataset: str, files: list[str], episode: int) -> pd.DataFrame:
    parts = []
    for f in files:
        t = pq.read_table(f, columns=COLS, filters=[("episode_index", "=", episode)]).to_pandas()
        if len(t):
            parts.append(t)
        elif parts:
            break
    return pd.concat(parts).sort_values("frame_index").reset_index(drop=True)


def reorient_clips(df: pd.DataFrame) -> list[Clip]:
    """One clip per reorient action: reorient_start_traj through the last retract frame."""
    ll = df[LL_COL].to_numpy(object)
    hl = df["action.high_level_args"].to_numpy(object)
    clips = []
    for s, e in _segments(ll, hl):
        if ll[s] != "reorient_start_traj":
            continue
        end = e
        while end < len(ll) and ll[end] in ("reorient_traj", "reorient_retract_traj") and hl[end] == hl[s]:
            end += 1
        seg = ll[s:end]
        attempts = int(np.sum((seg[1:] == "reorient_retract_traj") & (seg[:-1] != "reorient_retract_traj")))
        clips.append(Clip(int(df.episode_index[0]), max(0, s - PAD_BEFORE), min(len(ll), end + PAD_AFTER), attempts))
    return clips


def find_episodes(dataset: str, files: list[str]) -> dict[str, int]:
    """Episode per category: 1/2/3 attempts and a reorient that crosses a data-file boundary.

    Only episodes lying wholly inside ``files`` qualify: one that starts before the first file or
    runs past the last would be rendered from a partial row set.
    """
    tables = [pq.read_table(f, columns=["episode_index", "frame_index", "action.high_level_args", LL_COL]).to_pandas()
              for f in files]
    partial: set[int] = set()
    if tables and len(tables[0]) and int(tables[0].frame_index.iloc[0]) != 0:
        partial.add(int(tables[0].episode_index.iloc[0]))
    if tables and len(tables[-1]):
        partial.add(int(tables[-1].episode_index.iloc[-1]))
    picks: dict[str, int] = {}
    prev_tail: tuple[int, str] | None = None
    for t in tables:
        ll, hl, ep = t[LL_COL].to_numpy(object), t["action.high_level_args"].to_numpy(object), t.episode_index.to_numpy()
        if prev_tail is not None and len(t) and ep[0] == prev_tail[0] and int(ep[0]) not in partial \
                and prev_tail[1] in ("reorient_traj", "reorient_retract_traj") \
                and ll[0] in ("reorient_traj", "reorient_retract_traj"):
            picks.setdefault("spans_files", int(ep[0]))
        for s, e in _segments(ll, hl):
            if ll[s] != "reorient_traj" or e == len(ll) or ep[s] != ep[e - 1] or int(ep[s]) in partial:
                continue
            n = 0
            i = s
            while i < len(ll) and ll[i] in ("reorient_traj", "reorient_retract_traj") and hl[i] == hl[s] and ep[i] == ep[s]:
                n += ll[i] == "reorient_retract_traj" and ll[i - 1] != "reorient_retract_traj"
                i += 1
            if 1 <= n <= 3:
                picks.setdefault(f"{n}_attempt{'s' if n > 1 else ''}", int(ep[s]))
        prev_tail = (int(ep[-1]), str(ll[-1])) if len(t) else None
        if len(picks) == 4:
            break
    return picks


def decode_frames(path: str, t0: float, t1: float, fps: float) -> list[np.ndarray]:
    """RGB frames with timestamps in [t0, t1)."""
    out = []
    with av.open(path) as container:
        stream = container.streams.video[0]
        container.seek(int(max(0.0, t0 - 1.0) / stream.time_base), stream=stream, backward=True)
        for frame in container.decode(stream):
            t = float(frame.pts * stream.time_base)
            if t < t0 - 0.5 / fps:
                continue
            if t >= t1 - 0.5 / fps:
                break
            out.append(frame.to_ndarray(format="rgb24"))
    return out


def text(img: np.ndarray, s: str, xy: tuple[int, int], scale: float = 0.6, color=(255, 255, 255), thick: int = 1) -> None:
    cv2.putText(img, s, xy, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render(dataset: str, files: list[str], episode: int, label: str, out_path: str) -> int:
    df = episode_rows(dataset, files, episode)
    clips = reorient_clips(df)
    meta_files = sorted(glob.glob(os.path.join(dataset, "meta", "episodes", "*", "*.parquet")))
    ep_meta = pd.concat(pq.read_table(f, filters=[("episode_index", "=", episode)]).to_pandas() for f in meta_files).iloc[0]
    fps = 25.0
    ee_z = np.stack(df["action.ee_target"].to_numpy())[:, 2]

    writer = None
    n_out = 0
    for ci, clip in enumerate(clips):
        cam_frames = []
        for cam in CAMS:
            chunk, fidx = int(ep_meta[f"videos/{cam}/chunk_index"]), int(ep_meta[f"videos/{cam}/file_index"])
            base = float(ep_meta[f"videos/{cam}/from_timestamp"])
            path = os.path.join(dataset, "videos", cam, f"chunk-{chunk:03d}", f"file-{fidx:03d}.mp4")
            # Seek by the rows' own frame_index: row positions in df need not start at frame 0.
            f0 = int(df.frame_index.iloc[clip.start])
            f1 = int(df.frame_index.iloc[clip.end - 1]) + 1
            cam_frames.append(decode_frames(path, base + f0 / fps, base + f1 / fps, fps))
        n = min(len(c) for c in cam_frames)
        cams = [[cv2.resize(f, (int(f.shape[1] * CAM_H / f.shape[0]), CAM_H)) for f in c[:n]] for c in cam_frames]
        width = sum(c[0].shape[1] for c in cams)
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, CAM_H + PANEL_H + TIMELINE_H))
        rows = df.iloc[clip.start : clip.start + n]
        labels = rows[LL_COL].to_numpy(object)
        for k in range(n):
            canvas = np.zeros((CAM_H + PANEL_H + TIMELINE_H, width, 3), np.uint8)
            canvas[:CAM_H] = np.concatenate([c[k] for c in cams], axis=1)
            r = rows.iloc[k]
            ll = str(r[LL_COL])
            col = COLORS.get(ll, OTHER_COLOR)
            y0 = CAM_H
            text(canvas, f"{label} | episode {episode} | reorient {ci + 1}/{len(clips)} ({clip.attempts} attempt"
                         f"{'s' if clip.attempts != 1 else ''}) | frame {int(r.frame_index)}", (10, y0 + 24), 0.55)
            text(canvas, f"HL: {r['action.high_level_action']} {r['action.high_level_args']}", (10, y0 + 52), 0.55)
            cv2.rectangle(canvas, (10, y0 + 64), (26, y0 + 80), col, -1)
            text(canvas, f"LL: {ll}", (34, y0 + 80), 0.8, col, 2)
            text(canvas, f"goal: {r['goal.action']} / {r['goal.action_secondary']}", (10, y0 + 108), 0.55)
            grip = float(np.ravel(r["observation.state.gripper_position"])[0])
            text(canvas, f"commanded EE z: {ee_z[clip.start + k]:.3f} m   gripper position: {grip:.2f}", (10, y0 + 136), 0.55)
            # Timeline of the clip's labels with a cursor.
            ty = CAM_H + PANEL_H
            for j in range(n):
                x0, x1 = int(j * width / n), int((j + 1) * width / n)
                canvas[ty + 6 : ty + TIMELINE_H - 6, x0 : max(x1, x0 + 1)] = COLORS.get(str(labels[j]), OTHER_COLOR)
            cx = int((k + 0.5) * width / n)
            canvas[ty : ty + TIMELINE_H, max(0, cx - 1) : cx + 2] = (255, 255, 255)
            writer.write(cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
            n_out += 1
        for _ in range(int(fps)):  # 1 s black separator between clips
            writer.write(np.zeros((CAM_H + PANEL_H + TIMELINE_H, width, 3), np.uint8))
    if writer is not None:
        writer.release()
    return n_out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("out_dir")
    ap.add_argument("--max-files", type=int, default=40, help="Data files to search for example episodes")
    ap.add_argument("--file-range", type=int, nargs=2, default=None, metavar=("FIRST", "END"),
                    help="Only data files numbered FIRST..END-1 (must all exist and be contiguous)")
    args = ap.parse_args()
    files = sorted(f for f in glob.glob(os.path.join(args.dataset, "data", "*", "*.parquet")))
    if args.file_range:
        first, end = args.file_range
        files = [f for f in files if first <= int(os.path.basename(f)[5:8]) < end]
    files = files[: args.max_files]
    os.makedirs(args.out_dir, exist_ok=True)
    picks = find_episodes(args.dataset, files)
    print("picked:", picks, flush=True)
    for label, ep in picks.items():
        out = os.path.join(args.out_dir, f"reorient_labels_{label}_ep{ep}.mp4")
        n = render(args.dataset, files, ep, label, out)
        print(f"wrote {out} ({n} frames)", flush=True)


if __name__ == "__main__":
    main()
