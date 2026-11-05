"""Maniflow eval ptmap Z depth masking (after resize and optional EJBL)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from blocksuite.serving.obs_transforms.depth_ptmap_bounds import (
    DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS,
    z_bounds_for_depth_key,
)
from blocksuite.serving.obs_transforms.maniflow import ManiflowObsTransform


def _depth_inputs_with_patches(
    transform: ManiflowObsTransform,
    depth: np.ndarray,
) -> dict[str, np.ndarray]:
    obs = {
        "images": {"external_camera": {"depth": depth[0, ..., None]}},
        "camera_intrinsics": {"external_camera": np.eye(3, dtype=np.float32)},
    }
    with patch.object(
        ManiflowObsTransform,
        "get_depth",
        return_value=depth,
    ), patch.object(
        ManiflowObsTransform,
        "get_intrinsics",
        return_value=np.tile(np.eye(3, dtype=np.float32), (1, 1, 1)),
    ), patch.object(
        ManiflowObsTransform,
        "process_depth",
        side_effect=lambda d, size, **kw: d.astype(np.float32),
    ):
        return transform._depth_inputs(obs)


def test_ptmap_z_mask_after_ejbl_order() -> None:
    z_min, z_max = z_bounds_for_depth_key(
        "front_depth", DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS
    )
    transform = ManiflowObsTransform(depth_keys=("front_depth",))
    depth = np.array([[[z_min - 0.05, z_min + 0.05, z_max + 0.5]]], dtype=np.float32)
    with patch.object(
        transform,
        "_apply_ejbl_depth_nhw",
        return_value=np.array([[[z_min - 0.02, z_min + 0.05, z_max + 0.5]]], dtype=np.float32),
    ) as mock_ejbl:
        transform.augment_depth = "ejbl"
        out = _depth_inputs_with_patches(transform, depth)
        mock_ejbl.assert_called_once()
    front = out["front_depth"][0, 0, 0]
    assert float(front[0, 0]) == 0.0
    assert float(front[0, 1]) == pytest.approx(z_min + 0.05)
    assert float(front[0, 2]) == 0.0


def test_per_camera_ptmap_bounds_override() -> None:
    custom = {
        "front_depth": {"min": [-1.0, -1.0, 0.2], "max": [1.0, 1.0, 0.8]},
    }
    transform = ManiflowObsTransform(
        depth_keys=("front_depth",),
        per_camera_ptmap_bounds=custom,
    )
    depth = np.array([[[0.15, 0.5, 0.9]]], dtype=np.float32)
    out = _depth_inputs_with_patches(transform, depth)
    front = out["front_depth"][0, 0, 0]
    assert float(front[0, 0]) == 0.0
    assert float(front[0, 1]) == pytest.approx(0.5)
    assert float(front[0, 2]) == 0.0


def test_transform_kwargs_from_server_metadata_ptmap_bounds() -> None:
    from blocksuite.serving.obs_transforms.maniflow import transform_kwargs_from_server_metadata

    bounds = {
        "front_depth": {"min": [-0.6, -0.4, 0.4], "max": [0.6, 0.3, 1.0]},
    }
    kwargs = transform_kwargs_from_server_metadata(
        {
            "obs_keys": ["front_depth", "front_rgb", "agent_pos"],
            "per_camera_ptmap_bounds": bounds,
        }
    )
    assert kwargs["per_camera_ptmap_bounds"] == bounds
