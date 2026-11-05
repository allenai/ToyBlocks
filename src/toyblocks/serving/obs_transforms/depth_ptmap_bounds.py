"""Per-camera depth Z validity for ManiFlow eval (vendored from HVLA ptmap_depth_bounds).

The pinned ``maniflow-tb`` does not ship this module; eval depth masking uses the same
constants as training at the pointmap-Z-bounds commit so depth-conditioned checkpoints stay
aligned without importing hvla for it.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

# Matches ``maniflow_train_config`` dotlist when ``--depth`` + ``per_cam_norm``.
DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS: dict[str, dict[str, list[float]]] = {
    "front_depth": {"min": [-0.6, -0.4, 0.4], "max": [0.6, 0.3, 1.0]},
    "gripper_depth": {"min": [-0.5, -0.4, 0.1], "max": [0.5, 0.4, 0.8]},
}

_CAM_TO_DEPTH_KEY = {
    "front_camera": "front_depth",
    "gripper_camera": "gripper_depth",
}


def depth_key_for_cam(cam_name: str) -> str:
    if cam_name not in _CAM_TO_DEPTH_KEY:
        raise KeyError(f"unknown camera {cam_name!r}; expected one of {sorted(_CAM_TO_DEPTH_KEY)}")
    return _CAM_TO_DEPTH_KEY[cam_name]


def z_bounds_for_depth_key(
    depth_key: str,
    bounds_dict: Mapping[str, Mapping[str, list[float]]],
    *,
    ptmap_min: list[float] | None = None,
    ptmap_max: list[float] | None = None,
) -> tuple[float, float]:
    if depth_key in bounds_dict:
        b = bounds_dict[depth_key]
        return float(b["min"][2]), float(b["max"][2])
    if ptmap_min is not None and ptmap_max is not None:
        return float(ptmap_min[2]), float(ptmap_max[2])
    raise KeyError(
        f"no ptmap bounds for depth key {depth_key!r} and no global ptmap_min/ptmap_max fallback"
    )


def mask_depth_m_by_z_bounds(
    depth_m: np.ndarray,
    z_min: float,
    z_max: float,
) -> np.ndarray:
    """Invalidate (zero) depths outside ``[z_min, z_max]``; never clamp in-range values."""
    out = depth_m.astype(np.float32, copy=True)
    invalid = (out <= 0) | (out < z_min) | (out > z_max)
    out[invalid] = 0.0
    return out
