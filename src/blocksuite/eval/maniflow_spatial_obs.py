"""Inject HL spatial fields the Maniflow obs transform forwards to the LL server."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def inject_maniflow_hl_spatial_obs(
    obs: dict[str, Any],
    *,
    task_names: Sequence[str],
    heatmaps: np.ndarray,
    target_xy_batch: np.ndarray | None,
    device: torch.device | str | None = None,
) -> None:
    """Write ``oracle_task_name``, ``target_heatmap``, and optional ``target_xy``.

    Shapes match the eval control loop: heatmaps ``(B, 1, 3, 224, 224)``,
    ``target_xy_batch`` ``(B, 2)`` in native front-camera pixels.
    """
    hm = np.asarray(heatmaps, dtype=np.float32)
    if device is not None:
        obs["target_heatmap"] = torch.from_numpy(hm).to(device)
        if target_xy_batch is not None:
            obs["target_xy"] = torch.from_numpy(
                np.asarray(target_xy_batch, dtype=np.float32)
            ).to(device)
    else:
        obs["target_heatmap"] = hm
        if target_xy_batch is not None:
            obs["target_xy"] = np.asarray(target_xy_batch, dtype=np.float32)
    obs["oracle_task_name"] = list(task_names)
