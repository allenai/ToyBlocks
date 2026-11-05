"""HL spatial obs injection for crop-trained Maniflow eval."""

from __future__ import annotations

import numpy as np

from tests.serving.policy_input_fixture import build_obs
from toyblocks.eval.maniflow_spatial_obs import inject_maniflow_hl_spatial_obs
from toyblocks.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)


def test_inject_maniflow_hl_spatial_obs_enables_crop_transform() -> None:
    obs = build_obs(batch=2)
    heatmaps = np.zeros((2, 1, 3, 224, 224), dtype=np.float32)
    heatmaps[0, 0, 0, 112, 112] = 1.0
    heatmaps[1, 0, 0, 56, 56] = 1.0
    target_xy = np.array([[320.0, 180.0], [160.0, 90.0]], dtype=np.float32)
    inject_maniflow_hl_spatial_obs(
        obs,
        task_names=["pick left", "place right"],
        heatmaps=heatmaps,
        target_xy_batch=target_xy,
    )
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
    assert wire["task_name"] == ["pick left", "place right"]
