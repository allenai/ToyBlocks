"""Shared front-camera video helper for the replay scripts (RG_VIDEO=1)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

VIDEO = os.environ.get("RG_VIDEO", "0") == "1"


def front_rgb(sim, name="front_camera"):
    """(num_envs, H, W, 3) uint8 from the front_camera, across sim variants, or None."""
    out = None
    for getter in (
        lambda: sim.scene[name].data.output["rgb"],
        lambda: sim.scene.sensors[name].data.output["rgb"],
        lambda: sim.base_env.cameras[name].data.output["rgb"],
        lambda: sim.cameras[name].data.output["rgb"],
    ):
        try:
            out = getter()
            if out is not None:
                break
        except Exception:
            out = None
    if out is None:
        return None
    a = out[..., :3].detach().cpu().numpy()
    return np.clip(a, 0, 255).astype(np.uint8)


def tile8(imgs):
    """(N,H,W,3) -> 2x4 grid; pads with black to 8 tiles."""
    n, h, w, _ = imgs.shape
    tiles = [imgs[i] if i < n else np.zeros((h, w, 3), np.uint8) for i in range(8)]
    rows = [np.concatenate(tiles[r * 4:(r + 1) * 4], axis=1) for r in range(2)]
    return np.concatenate(rows, axis=0)


def save(frames, path, fps=20):
    if not frames:
        print(f"[video] no frames for {path}")
        return
    import imageio.v2 as imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path, frames, fps=fps, macro_block_size=1)
    print(f"[video] wrote {path} ({len(frames)} frames)")
