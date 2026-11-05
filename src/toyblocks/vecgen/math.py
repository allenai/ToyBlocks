"""Batched quaternion / pose helpers (wxyz, torch)."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def quat_normalize(q: Tensor) -> Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def quat_mul(a: Tensor, b: Tensor) -> Tensor:
    """Hamilton product ``a ⊗ b`` for ``[.., 4]`` wxyz tensors."""
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quat_inverse(q: Tensor) -> Tensor:
    """Conjugate of a unit quaternion."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_apply(q: Tensor, v: Tensor) -> Tensor:
    """Rotate vectors ``v`` ``[.., 3]`` by unit quaternions ``q`` ``[.., 4]``."""
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def world_extents(q: Tensor, dims: Tensor) -> Tensor:
    """World axis-aligned extents ``[.., 3]`` of a box with body-frame edge lengths ``dims``
    ``[.., 3]`` rotated by ``q`` ``[.., 4]`` (wxyz): ``|R| @ dims``."""
    eye = torch.eye(3, device=q.device, dtype=q.dtype)
    lead = q.shape[:-1]
    axes = quat_apply(q[..., None, :].expand(*lead, 3, 4), eye.expand(*lead, 3, 3))
    return (axes.abs() * dims[..., :, None]).sum(dim=-2)


def quat_from_axis_angle(axis: Tensor, angle: Tensor) -> Tensor:
    """``axis`` ``[.., 3]`` unit, ``angle`` ``[..]`` rad -> wxyz."""
    half = angle.unsqueeze(-1) * 0.5
    return torch.cat([torch.cos(half), axis * torch.sin(half)], dim=-1)


def quat_yaw(yaw: Tensor) -> Tensor:
    """Rotation about world z by ``yaw`` ``[..]``."""
    zeros = torch.zeros_like(yaw)
    return torch.stack([torch.cos(yaw / 2), zeros, zeros, torch.sin(yaw / 2)], dim=-1)


def quat_from_euler_xyz(roll: Tensor, pitch: Tensor, yaw: Tensor) -> Tensor:
    """Intrinsic XYZ euler (same convention as ``isaaclab.utils.math.quat_from_euler_xyz``)."""
    cy, sy = torch.cos(yaw * 0.5), torch.sin(yaw * 0.5)
    cr, sr = torch.cos(roll * 0.5), torch.sin(roll * 0.5)
    cp, sp = torch.cos(pitch * 0.5), torch.sin(pitch * 0.5)
    return torch.stack(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ],
        dim=-1,
    )


def quat_angle(q: Tensor) -> Tensor:
    """Rotation angle of unit quaternions ``[..]`` in [0, pi]."""
    w = q[..., 0].abs().clamp(max=1.0)
    return 2.0 * torch.acos(w)


def quat_z_axis(q: Tensor) -> Tensor:
    """Body z axis expressed in the parent frame."""
    z = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
    z[..., 2] = 1.0
    return quat_apply(q, z)


def pose_compose(a: Tensor, b: Tensor) -> Tensor:
    """``T_a @ T_b`` for ``[.., 7]`` poses (xyz, wxyz)."""
    pos = a[..., :3] + quat_apply(a[..., 3:7], b[..., :3])
    return torch.cat([pos, quat_mul(a[..., 3:7], b[..., 3:7])], dim=-1)


def pose_inverse(a: Tensor) -> Tensor:
    q_inv = quat_inverse(a[..., 3:7])
    return torch.cat([-quat_apply(q_inv, a[..., :3]), q_inv], dim=-1)


def yaw_of(q: Tensor) -> Tensor:
    """World-z yaw of the body x axis (used for grasp heading)."""
    x = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
    x[..., 0] = 1.0
    xw = quat_apply(q, x)
    return torch.atan2(xw[..., 1], xw[..., 0])


def wrap_angle(a: Tensor) -> Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


PI = math.pi
