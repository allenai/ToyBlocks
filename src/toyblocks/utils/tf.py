"""Frame transformation helpers (numpy, wxyz quaternions)."""

from __future__ import annotations

import numpy as np


def quat_rotate_vec_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate 3-D vectors by unit quaternion(s) in ``[qw, qx, qy, qz]`` format."""
    q = np.asarray(quat, dtype=np.float32)
    v = np.asarray(vec, dtype=np.float32)
    q_xyz = q[..., 1:4]
    qw = q[..., 0:1]
    t = 2.0 * np.cross(q_xyz, v)
    return (v + qw * t + np.cross(q_xyz, t)).astype(np.float32)


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions in ``[qw, qx, qy, qz]`` format."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1).astype(np.float32)


def local_pose_to_world(
    pose_local: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> np.ndarray:
    """Convert a robot-base pose to world frame."""
    pose_local = np.asarray(pose_local, dtype=np.float32)
    base_pos = np.asarray(base_pos, dtype=np.float32)
    base_quat = np.asarray(base_quat, dtype=np.float32)
    world_pos = base_pos + quat_rotate_vec_wxyz(base_quat, pose_local[:3])
    world_quat = quat_mul_wxyz(base_quat, pose_local[3:7])
    return np.concatenate([world_pos, world_quat]).astype(np.float32)


def world_pose_to_local(
    pose_world: np.ndarray,
    base_pos: np.ndarray,
    base_quat: np.ndarray,
) -> np.ndarray:
    """Convert a world pose to the robot-base frame.

    Takes either a single ``(7,)`` pose with a ``(3,)`` base position and
    ``(4,)`` base quaternion, or a batch of them (``(N, 7)``, ``(N, 3)``,
    ``(N, 4)``).
    """
    pose_world = np.asarray(pose_world, dtype=np.float32)
    base_pos = np.asarray(base_pos, dtype=np.float32)
    base_quat = np.asarray(base_quat, dtype=np.float32)
    base_quat_inv = base_quat.copy()
    base_quat_inv[..., 1:] *= -1.0
    local_pos = quat_rotate_vec_wxyz(base_quat_inv, pose_world[..., :3] - base_pos)
    local_quat = quat_mul_wxyz(base_quat_inv, pose_world[..., 3:7])
    return np.concatenate([local_pos, local_quat], axis=-1).astype(np.float32)
