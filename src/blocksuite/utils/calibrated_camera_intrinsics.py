"""Fixed sim2real + fr3_robotiq_gripper_softer camera K at native render resolution.

Used for eval ``--use_calibrated_cameras`` (aug26-style constant intrinsics) and
for comparing datasets to codebase presets. Front K uses a centered principal
point at 640x360 (see ``blocksuite.tamp.blocks.camera_view``), not naive
scaling of ``K_front`` in ``sim2real.py``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FRONT_CAMERA_FX: float = 393.07184
FRONT_CAMERA_FY: float = 393.07184
FRONT_CAMERA_WIDTH: int = 640
FRONT_CAMERA_HEIGHT: int = 360

# sim2real left_camera PinholeCameraCfg(fl=2.12, ha=5.139) @ 640x360 (same ZED module as front)
_LEFT_FL = 2.12
_LEFT_HA = 5.139
_LEFT_W = 640
_LEFT_H = 360

# fr3_robotiq_gripper_softer gripper_camera PinholeCameraCfg(fl=666.64, ha=1280) @ 640x360
_GRIPPER_FL = 666.64
_GRIPPER_HA = 1280.0
_GRIPPER_W = 640
_GRIPPER_H = 360


def _pinhole_k(w: int, h: int, focal_length: float, horizontal_aperture: float) -> np.ndarray:
    fx = w * focal_length / horizontal_aperture
    return np.array(
        [[fx, 0.0, w / 2.0], [0.0, fx, h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def calibrated_camera_k3x3() -> dict[str, np.ndarray]:
    """3x3 pinhole K per live scene camera name at native render resolution."""
    return {
        "front_camera": np.array(
            [
                [FRONT_CAMERA_FX, 0.0, FRONT_CAMERA_WIDTH / 2.0],
                [0.0, FRONT_CAMERA_FY, FRONT_CAMERA_HEIGHT / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        "left_camera": _pinhole_k(_LEFT_W, _LEFT_H, _LEFT_FL, _LEFT_HA),
        "gripper_camera": _pinhole_k(_GRIPPER_W, _GRIPPER_H, _GRIPPER_FL, _GRIPPER_HA),
    }


def calibrated_camera_k9_by_name() -> dict[str, list[float]]:
    """Row-major 9-vectors for :func:`apply_camera_intrinsics_for_env`."""
    return {
        name: k.reshape(-1).astype(float).tolist()
        for name, k in calibrated_camera_k3x3().items()
    }


def preset_k_for_camera(cam: str) -> np.ndarray:
    """Single-camera preset K (for tooling / tests)."""
    k = calibrated_camera_k3x3().get(cam)
    if k is None:
        raise KeyError(cam)
    return k


def apply_calibrated_camera_intrinsics_for_env(base_env: Any, env_idx: int) -> list[str]:
    """Write preset K into one env row of each matching live camera."""
    from blocksuite.utils.dataset_provenance import apply_camera_intrinsics_for_env

    return apply_camera_intrinsics_for_env(
        base_env, env_idx, calibrated_camera_k9_by_name()
    )


def apply_calibrated_camera_intrinsics(base_env: Any) -> list[str]:
    """Broadcast preset K to every env (initial eval reset)."""
    n_envs = int(base_env.num_envs)
    set_names: set[str] = set()
    for env_idx in range(n_envs):
        set_names.update(apply_calibrated_camera_intrinsics_for_env(base_env, env_idx))
    if not set_names:
        raise ValueError(
            "use_calibrated_cameras is set but no live scene camera matched the "
            f"preset names {sorted(calibrated_camera_k3x3())}."
        )
    k = calibrated_camera_k3x3()
    for name in sorted(set_names):
        mat = k[name]
        logger.info(
            "calibrated intrinsics for %r fx=%.6f fy=%.6f cx=%.1f cy=%.1f",
            name,
            float(mat[0, 0]),
            float(mat[1, 1]),
            float(mat[0, 2]),
            float(mat[1, 2]),
        )
    logger.info(
        "applied calibrated camera intrinsics to %d env(s), cameras %s",
        n_envs,
        sorted(set_names),
    )
    return sorted(set_names)
