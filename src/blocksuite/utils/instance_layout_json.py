"""Load per-instance layout JSON (e.g. ``apr15_pnp/envs/.../*.json``) into env cfg and sim."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
from isaaclab.utils import math as math_isaac

from blocksuite.tasks.base_env_cfg import BaseEnvCfg


'''
def _gripper_look_dir_to_ros_quat_wxyz(look_dir: tuple[float, ...]) -> tuple[float, float, float, float]:
    """Build ROS-offset quaternion (w,x,y,z) for a camera whose +Z looks along ``look_dir`` (parent frame).

    Uses the same reference up as ``apr15_pnp/.../init.py`` comments: ``(1, 0, 0)``, with a Z-up fallback
    if ``look_dir`` is parallel to that axis.
    """
    f = torch.tensor(look_dir[:3], dtype=torch.float64)
    f = f / (torch.norm(f) + 1e-12)
    up = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    x = torch.cross(up, f)
    n = torch.norm(x)
    if float(n) < 1e-6:
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
        x = torch.cross(up, f)
        n = torch.norm(x) + 1e-12
    x = x / n
    y = torch.cross(f, x)
    r = torch.stack([x, y, f], dim=1).to(dtype=torch.float32)
    q = math_isaac.quat_from_matrix(r.unsqueeze(0))[0]
    return tuple(float(t) for t in q.tolist())
'''

def merge_instance_json_into_env_cfg(
    cfg: BaseEnvCfg,
    *,
    block_names: frozenset[str],
) -> dict[str, dict[str, tuple[float, ...]]] | None:
    """If ``cfg.instance_layout_json_path`` is set, load JSON and merge into ``cfg``.

    - ``block_positions`` / ``block_quaternions`` for names in ``block_names`` → returned dict.
    - ``gripper_camera_pos``, ``gripper_camera_look_at_pos``, ``gripper_horizontal_aperture`` →
      mutates ``cfg.camera_cfgs["gripper_camera"]`` when that camera exists.

    Returns ``None`` when no path is configured. Returns ``{}`` when the file has no matching blocks.
    """
    raw = getattr(cfg, "instance_layout_json_path", None)
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"instance_layout_json_path not found: {path}")
    data: dict[str, Any] = json.loads(path.read_text())

    out: dict[str, dict[str, tuple[float, ...]]] = {}
    pos_all = data.get("block_positions") or {}
    quat_all = data.get("block_quaternions") or {}
    for name in block_names:
        bp = pos_all.get(name)
        bq = quat_all.get(name)
        if bp is not None and bq is not None:
            out[str(name)] = {
                "pos": tuple(float(x) for x in bp[:3]),
                "quat": tuple(float(x) for x in bq[:4]),
            }

    '''
    cam = cfg.camera_cfgs.get("gripper_camera")
    if cam is not None:
        gc = data.get("gripper_camera_pos")
        gl = data.get("gripper_camera_look_at_pos")
        gap = data.get("gripper_horizontal_aperture")
        if gc is not None:
            pos_t = tuple(float(x) for x in gc[:3])
            if gl is not None:
                rot_t = _gripper_look_dir_to_ros_quat_wxyz(tuple(float(x) for x in gl[:3]))
            else:
                rot_t = tuple(float(x) for x in cam.offset.rot)
            cam.offset = cam.offset.replace(pos=pos_t, rot=rot_t)
        if gap is not None and getattr(cam, "spawn", None) is not None:
            cam.spawn = cam.spawn.replace(horizontal_aperture=float(gap))
    '''

    return out


def write_block_layout_to_sim(
    env: Any,
    layout: dict[str, dict[str, tuple[float, ...]]],
    env_ids: torch.Tensor | Sequence[int],
) -> None:
    """Write fixed poses for rigid objects in ``layout`` (world XY + table height; add env origins)."""
    if not layout:
        return
    if env_ids is None:
        return
    if isinstance(env_ids, torch.Tensor):
        ids = env_ids.to(device=env.device, dtype=torch.long).reshape(-1)
    else:
        ids = torch.tensor(list(env_ids), device=env.device, dtype=torch.long).reshape(-1)
    n = int(ids.shape[0])
    origins = env.scene.env_origins[ids]
    for name, pq in layout.items():
        obj = env.scene.rigid_objects.get(name)
        if obj is None:
            continue
        pos = torch.tensor(pq["pos"], device=env.device, dtype=torch.float32).unsqueeze(0).expand(n, -1) + origins
        quat = torch.tensor(pq["quat"], device=env.device, dtype=torch.float32).unsqueeze(0).expand(n, -1)
        pose = torch.cat([pos, quat], dim=-1)
        obj.write_root_pose_to_sim(pose, env_ids=ids)
        obj.write_root_velocity_to_sim(torch.zeros(n, 6, device=env.device), env_ids=ids)
