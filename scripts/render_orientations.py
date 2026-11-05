#!/usr/bin/env python3
"""Render one block from the front camera in all 24 canonical orientations.

Spawns the search env with 24 parallel environments, holds ``block_r`` (the
R.usd glyph block) at orientation ``o1``..``o24`` — one per env — and grabs the
front-camera RGB for each. Distractor blocks are pushed out of frame. No physics
is stepped, so each block stays exactly at its orientation while rendering.

Usage:
    python scripts/render_orientations.py --env_path outputs/search_r/<task> \
        --headless --enable_cameras
"""

import argparse
import sys
from pathlib import Path

from blocksuite.cli.isaac_app import launch_app

simulation_app = launch_app()  # consumes --headless / --enable_cameras from argv

import importlib.util

import cv2
import numpy as np
import gymnasium as gym
import torch
from isaaclab_tasks.utils import parse_env_cfg
from blocksuite.tamp.blocks.pddl_goal_tensor_conversions import ORI_QUATERNIONS

# Camera-verified orientations where the letter face is visible (registries.FOUND_ORIS).
ACCEPTING = {"o2", "o6", "o10", "o14", "o17", "o18", "o19", "o20"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", required=True)
    parser.add_argument("--block", default="block_r")
    parser.add_argument("--camera", default="front_camera")
    parser.add_argument("--out_dir", default="outputs/ori_views")
    parser.add_argument("--device", default="cuda:0")
    args, _ = parser.parse_known_args()

    n = len(ORI_QUATERNIONS)  # 24
    env_path = Path(args.env_path)
    spec = importlib.util.spec_from_file_location(env_path.name, env_path / "__init__.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[env_path.name] = module
    spec.loader.exec_module(module)

    task_id = f"{env_path.name}-v0"
    env_cfg = parse_env_cfg(task_id, device=args.device, num_envs=n)
    env_cfg.always_spawn_cameras = True
    env = gym.make(task_id, cfg=env_cfg)
    base = env.unwrapped
    base.reset()

    dev = base.device
    origins = base.scene.env_origins  # (n, 3)
    block = base.scene.rigid_objects[args.block]

    # Same in-view spot in every env (env 0's block position), one orientation each.
    state0 = block.data.root_state_w[:, :7].clone()
    local0 = state0[0, :3] - origins[0]
    pose = torch.zeros(n, 7, device=dev, dtype=state0.dtype)
    pose[:, :3] = origins + local0.unsqueeze(0)
    quats = torch.tensor(
        [ORI_QUATERNIONS[f"o{i + 1}"] for i in range(n)], device=dev, dtype=state0.dtype
    )
    pose[:, 3:7] = quats

    # Push distractor blocks far below the table so only `block` is in frame.
    zero_vel = torch.zeros(n, 6, device=dev)
    for name, obj in base.scene.rigid_objects.items():
        if name == args.block or "table" in name.lower():
            continue
        ds = obj.data.root_state_w[:, :7].clone()
        ds[:, 2] -= 5.0
        obj.write_root_pose_to_sim(ds)
        obj.write_root_velocity_to_sim(zero_vel)

    env_ids = torch.arange(n, device=dev)
    for _ in range(12):  # hold the pose and render; no physics step
        block.write_root_pose_to_sim(pose)
        block.write_root_velocity_to_sim(zero_vel)
        base.scene.write_data_to_sim()
        base.sim.forward()
        base.sim.render()
    for sensor in base.scene.sensors.values():
        try:
            sensor._update_buffers_impl(env_ids)
        except Exception:
            pass

    cam = base.cameras.get(args.camera) or base.scene.sensors[f"camera:{args.camera}"]
    rgb = cam.data.output["rgb"]  # (n, H, W, 3|4)
    rgb = rgb[..., :3]
    if rgb.dtype == torch.uint8:
        rgb = rgb.detach().cpu().numpy()
    else:
        arr = rgb.detach().float().cpu().numpy()
        arr = arr * 255.0 if arr.max() <= 1.0 else arr
        rgb = arr.astype(np.uint8)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for i in range(n):
        name = f"o{i + 1}"
        bgr = cv2.cvtColor(rgb[i], cv2.COLOR_RGB2BGR)
        color = (60, 200, 60) if name in ACCEPTING else (240, 240, 240)
        tag = name + (" READABLE" if name == "o2" else (" (found)" if name in ACCEPTING else ""))
        cv2.rectangle(bgr, (0, 0), (bgr.shape[1] - 1, bgr.shape[0] - 1), color, 2)
        cv2.putText(bgr, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(bgr, tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imwrite(str(out / f"{name}.png"), bgr)
        tiles.append(bgr)

    # 6x4 contact sheet
    th, tw = 240, 320
    small = [cv2.resize(t, (tw, th)) for t in tiles]
    cols, rows = 6, 4
    grid = np.full((rows * th, cols * tw, 3), 20, dtype=np.uint8)
    for i, t in enumerate(small):
        r, c = divmod(i, cols)
        grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = t
    cv2.imwrite(str(out / "grid_all24.png"), grid)
    print(f"[render_orientations] wrote {n} views + grid to {out.resolve()}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
