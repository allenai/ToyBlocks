# ruff: noqa  # verbatim port from TAMP tmp3 (kept diffable against upstream)
"""Backfill the per-episode video-location columns into a LeRobot v3 split's meta/episodes.

Our forkserver video encoder writes the mp4s itself (and moves them into videos/<key>/chunk-XXX/
file-NNN.mp4), which BYPASSES lerobot's own video-encode path — so lerobot never writes the
`videos/<key>/{chunk_index,file_index,from_timestamp,to_timestamp}` columns it later needs in
`get_video_file_path()` (lerobot_dataset.py). Without them a modern lerobot raises
`KeyError: videos/<key>/chunk_index` on load. This adds them, matching our file layout:

  chunk_index    = episode_index // 1000     (encoder: chunk-{ep//1000:03d})
  file_index     = episode_index %  1000     (encoder: file-{ep%1000:03d}.mp4 == save order)
  from_timestamp = 0.0                        (one file per episode -> starts at 0)
  to_timestamp   = length / fps               (== the episode's video duration)

Idempotent: if the columns already exist it leaves them untouched. Works for -N>1 augmented datasets
unchanged — each output episode has its own file-{output_index}.mp4, so the same per-episode_index
formula is correct.
"""
import glob
import json
import os


def backfill_video_meta(split_root: str) -> dict:
    """Add videos/<key>/* columns to every meta/episodes parquet under `split_root`. Returns a summary
    dict. No-op (skipped=True) if there are no video features or the columns are already present."""
    import pandas as pd

    info_path = os.path.join(split_root, "meta", "info.json")
    if not os.path.exists(info_path):
        return {"ok": False, "reason": "no info.json", "root": split_root}
    info = json.load(open(info_path))
    fps = info.get("fps")
    video_keys = [k for k, f in info.get("features", {}).items() if f.get("dtype") == "video"]
    if not video_keys or not fps:
        return {"ok": True, "skipped": True, "reason": "no video features/fps", "root": split_root}

    ep_parquets = sorted(glob.glob(os.path.join(split_root, "meta", "episodes", "**", "*.parquet"),
                                   recursive=True))
    if not ep_parquets:
        return {"ok": False, "reason": "no meta/episodes parquet", "root": split_root}

    total_rows, patched_files, already = 0, 0, 0
    for pq in ep_parquets:
        df = pd.read_parquet(pq)
        if f"videos/{video_keys[0]}/chunk_index" in df.columns:
            already += 1
            total_rows += len(df)
            continue
        ep = df["episode_index"].astype("int64")
        length = df["length"].astype("int64")
        for k in video_keys:
            df[f"videos/{k}/chunk_index"] = (ep // 1000).astype("int64")
            df[f"videos/{k}/file_index"] = (ep % 1000).astype("int64")
            df[f"videos/{k}/from_timestamp"] = 0.0
            df[f"videos/{k}/to_timestamp"] = (length / float(fps)).astype("float64")
        df.to_parquet(pq, index=False)
        patched_files += 1
        total_rows += len(df)

    return {"ok": True, "skipped": patched_files == 0, "root": split_root,
            "video_keys": len(video_keys), "episodes": total_rows,
            "patched_files": patched_files, "already_present_files": already,
            "added_columns": 0 if patched_files == 0 else len(video_keys) * 4}
