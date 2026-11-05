"""Front-camera visibility of blocks on the table.

Blocks the robot works on must stay inside the front camera's image, so reset-state
sampling, table placements, rotates and flips all gate block positions through
:func:`blocks_in_front_view`.

The camera constants are the sim2real ``front_camera`` as rendered (recorded in
``observation.camera_pose`` / ``observation.camera_intrinsics``): the pose from
``blocksuite/scenes/sim2real.py`` in Isaac's "world" camera convention (+X forward,
+Y left, +Z up), and the 640x360 render intrinsics. The renderer centres the
principal point, so ``cx, cy`` are the image centre rather than the calibrated
offsets in ``K_front``. Positions are in the env-local / robot base frame (the robot
base sits at the env origin).
"""

from __future__ import annotations

import math

import torch

from blocksuite.utils.calibrated_camera_intrinsics import (
    FRONT_CAMERA_FX,
    FRONT_CAMERA_FY,
    FRONT_CAMERA_HEIGHT,
    FRONT_CAMERA_WIDTH,
)

FRONT_CAMERA_POS: tuple[float, float, float] = (0.944958, 0.000714, 0.484140)
#: (w, x, y, z), world convention.
FRONT_CAMERA_QUAT: tuple[float, float, float, float] = (0.0, -0.3007058, 0.0, 0.9537169)
#: Every projected block corner must be at least this many pixels inside the image.
FRONT_CAMERA_MARGIN_PX: float = 12.0


def _camera_rotation(device: torch.device) -> torch.Tensor:
    """Camera-to-world rotation matrix (columns are the camera's X/Y/Z axes)."""
    w, x, y, z = FRONT_CAMERA_QUAT
    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        device=device,
        dtype=torch.float32,
    )


def blocks_in_front_view(
    centers: torch.Tensor,
    edges: torch.Tensor | float,
    margin_px: float = FRONT_CAMERA_MARGIN_PX,
) -> torch.Tensor:
    """Whether each (cube) block lies entirely inside the front camera image.

    The block is approximated by a box that contains it at any yaw: half-width
    ``edge / sqrt(2)`` in x and y, half-height ``edge / 2`` in z. All eight box
    corners must project inside the image with ``margin_px`` to spare. Occlusion by
    the arm is not considered.

    Callers pass the z extent from ``get_size()`` as the edge: the x/y extents it
    reports for the letter-block assets are not the block's footprint.

    Args:
        centers: ``(N, 3)`` or ``(3,)`` block centres, env-local / robot base frame.
        edges: ``(N,)`` cube edge lengths in metres, or a scalar edge.

    Returns:
        ``(N,)`` bool tensor (``(1,)`` for a single block).
    """
    centers = torch.as_tensor(centers, dtype=torch.float32).reshape(-1, 3)
    device = centers.device
    edge = (
        torch.as_tensor(edges, dtype=torch.float32, device=device)
        .reshape(-1)
        .expand(centers.shape[0])
    )

    half_xy = edge / math.sqrt(2.0)
    half_z = edge / 2.0
    signs = torch.tensor(
        [[sx, sy, sz] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
        device=device,
    )  # (8, 3)
    half = torch.stack([half_xy, half_xy, half_z], dim=-1)  # (N, 3)
    corners = centers[:, None, :] + signs[None, :, :] * half[:, None, :]  # (N, 8, 3)

    cam_pos = torch.tensor(FRONT_CAMERA_POS, device=device, dtype=torch.float32)
    rel = (corners - cam_pos) @ _camera_rotation(device)  # world -> camera axes
    depth = rel[..., 0]
    safe_depth = depth.clamp(min=1e-6)
    u = FRONT_CAMERA_WIDTH / 2 - FRONT_CAMERA_FX * rel[..., 1] / safe_depth
    v = FRONT_CAMERA_HEIGHT / 2 - FRONT_CAMERA_FY * rel[..., 2] / safe_depth
    inside = (
        (depth > 0)
        & (u >= margin_px)
        & (u <= FRONT_CAMERA_WIDTH - margin_px)
        & (v >= margin_px)
        & (v <= FRONT_CAMERA_HEIGHT - margin_px)
    )
    return inside.all(dim=-1)


def block_in_front_view(center: torch.Tensor, edge: torch.Tensor | float) -> bool:
    """Scalar convenience wrapper around :func:`blocks_in_front_view` for one block."""
    return bool(blocks_in_front_view(center, edge)[0].item())
