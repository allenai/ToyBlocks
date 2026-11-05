"""Tests for eval client ``target_front_crop_rgb`` (--enable_crop)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from blocksuite.serving.maniflow_serving_utils import (
    crop_centered_on_target,
    inject_target_front_crop_rgb,
)
from blocksuite.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)
from tests.serving.policy_input_fixture import build_obs


def test_crop_centered_on_target_matches_size() -> None:
    img = np.full((10, 12, 3), 200, dtype=np.uint8)
    crop = crop_centered_on_target(img, 6.0, 5.0, size=4, bottom_pct=50)
    assert crop.shape == (4, 4, 3)
    assert int(crop.mean()) == 200


def test_enable_crop_emits_target_front_crop_rgb() -> None:
    obs = build_obs(batch=2)
    obs["target_xy"] = np.array([[320.0, 180.0], [100.0, 50.0]], dtype=np.float32)
    transform = ManiflowBlocksTableObsTransform(
        depth_keys=(),
        include_ptmap=False,
        enable_target_front_crop=True,
        target_front_crop_bottom_pct=50,
        orig_cam_w=640.0,
        orig_cam_h=360.0,
        image_interpolate_device="cpu",
    )
    wire = transform(obs)
    assert "target_front_crop_rgb" in wire
    crop = np.asarray(wire["target_front_crop_rgb"])
    front = np.asarray(wire["front_rgb"])
    assert crop.shape == front.shape
    assert not np.allclose(crop, front)


def test_inject_target_front_crop_skips_client_tensor() -> None:
    sentinel = torch.zeros(1, 1, 3, 224, 224)
    torch_obs = {
        "target_front_crop_rgb": sentinel,
        "front_rgb": torch.ones(1, 1, 3, 224, 224),
        "target_heatmap": torch.zeros(1, 1, 3, 224, 224),
    }
    inject_target_front_crop_rgb(torch_obs, 50)
    assert torch_obs["target_front_crop_rgb"] is sentinel


def test_enable_crop_requires_bottom_pct() -> None:
    with pytest.raises(ValueError, match="target_front_crop_bottom_pct"):
        ManiflowBlocksTableObsTransform(
            enable_target_front_crop=True,
            target_front_crop_bottom_pct=-1,
        )
