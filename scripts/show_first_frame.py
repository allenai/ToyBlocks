# make a video of inital frames
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    import pandas


def _load_gt_episode(gt_source: str, episode_index: int = 0) -> pandas.DataFrame:
    """Load a single episode from a LeRobot dataset, avoiding full-dataset reads.

    Uses pyarrow predicate pushdown so only matching row-groups are scanned.
    """
    import pyarrow.dataset as ds

    gt_root = Path(gt_source)
    data_dir = gt_root / "data"
    assert data_dir.exists(), f"No data/ directory found at {data_dir}"
    dataset = ds.dataset(str(data_dir), format="parquet")
    table = dataset.to_table(filter=ds.field("episode_index") == episode_index)
    assert table.num_rows > 0, f"No rows found for episode_index={episode_index} in {data_dir}"
    return table.to_pandas()


def _read_video_frame(video_path, frame_idx: int) -> np.ndarray:
    import av

    container = av.open(str(video_path))
    frame_arr = None
    for i, frame in enumerate(container.decode(video=0)):
        frame_arr = frame.to_ndarray(format="rgb24")
        if i >= frame_idx:
            break
    container.close()
    return frame_arr


def get_obs_gt(gt_path, gt_df, step, device, episode_index: int = 0):
    row = gt_df.iloc[min(step, len(gt_df) - 1)]

    state = np.array(row["observation.state"], dtype=np.float32)
    joint_pos = np.zeros(13, dtype=np.float32)
    joint_pos[: len(state)] = state

    obs = dict(
        agent={"joint_pos": torch.tensor(joint_pos, device=device).unsqueeze(0)},
        goal={},
        images={},
    )

    gt_path = Path(gt_path)
    chunk = f"chunk-{episode_index // 1000:03d}"
    episode_mod = episode_index % 1000
    for camera_key in ["front_camera", "left_camera", "gripper_camera"]:
        video_path = (
            gt_path
            / "videos"
            / f"observation.images.camera:{camera_key}"
            / chunk
            / f"file-{episode_mod:03d}.mp4"
        )
        frame = _read_video_frame(video_path, step)
        obs["images"][camera_key] = {
            "rgb": torch.tensor(frame, dtype=torch.uint8, device=device).unsqueeze(0)
        }

    return obs


def show_first_frames(gt_path, save_location=None):
    # make a folder and save all video first frames
    gt_path = Path(gt_path)
    os.makedirs(save_location, exist_ok=True)
    for i in range(100):
        gt_df = _load_gt_episode(gt_path, episode_index=i)
        obs = get_obs_gt(gt_path, gt_df, step=0, device="cpu", episode_index=i)
        for camera_key in ["front_camera"]:
            frame = obs["images"][camera_key]["rgb"].squeeze(0).cpu().numpy()
            if save_location:
                save_path = Path(save_location) / f"first_frame_{i}_{camera_key}.png"
            else:
                save_path = gt_path / f"first_frame_{i}_{camera_key}.png"
            from PIL import Image

            Image.fromarray(frame).save(save_path)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Save the first front-camera frame of each episode.")
    ap.add_argument("dataset", help="LeRobot dataset root")
    ap.add_argument("--out", default=None, help="Output dir for the PNGs (default: the dataset dir)")
    a = ap.parse_args()
    show_first_frames(a.dataset, save_location=a.out)
