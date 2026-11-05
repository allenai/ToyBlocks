"""Joint center-crop on RGB, depth, and target_heatmap in Maniflow eval."""

from __future__ import annotations

import numpy as np
import pytest

from tests.serving.policy_input_fixture import (
    FIXTURE_CAM_HW,
    FIXTURE_INTRINSICS,
    build_obs,
    build_target_heatmap,
)
from toyblocks.serving.obs_transforms.maniflow import (
    ManiflowObsTransform,
    _center_crop_box,
    _center_crop_resize_nearest_nhw,
)
from toyblocks.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)


def test_center_crop_box_square_224_ratio_095() -> None:
    top, left, crop = _center_crop_box(224, 224, 0.95)
    assert crop == 212
    assert top == 6
    assert left == 6


def test_depth_center_crop_preserves_center_spike() -> None:
    h, w = 224, 224
    depth = np.zeros((1, h, w), dtype=np.float32)
    cy, cx = h // 2, w // 2
    depth[0, cy, cx] = 1.25
    out = _center_crop_resize_nearest_nhw(depth, 0.95)
    assert out.shape == depth.shape
    assert float(out[0, cy, cx]) == pytest.approx(1.25)


def test_depth_crop_matches_rgb_crop_window() -> None:
    """Depth inside the RGB crop window maps to the same output pixel (nearest)."""
    ratio = 0.95
    h, w = 224, 224
    top, left, crop = _center_crop_box(h, w, ratio)
    depth = np.zeros((1, h, w), dtype=np.float32)
    # A pixel on the crop box center maps to output center after resize.
    src_y = top + crop // 2
    src_x = left + crop // 2
    depth[0, src_y, src_x] = 0.88
    out = _center_crop_resize_nearest_nhw(depth, ratio)
    out_y, out_x = h // 2, w // 2
    assert float(out[0, out_y, out_x]) == pytest.approx(0.88)


def test_transform_applies_depth_crop_after_resize() -> None:
    transform = ManiflowBlocksTableObsTransform(
        depth_keys=("front_depth", "gripper_depth"),
        include_ptmap=False,
        center_crop_ratio=0.95,
        image_interpolate_device="cpu",
    )
    obs = build_obs(batch=1)
    native_h, native_w = FIXTURE_CAM_HW
    obs["images"]["front_camera"]["depth"] = np.full(
        (1, native_h, native_w, 1), 0.42, dtype=np.float32
    )
    out = transform(obs)
    front = out["front_depth"][0, 0, 0]
    assert front.shape == (224, 224)
    # Uniform depth survives crop+nearest resize unchanged.
    assert float(front.min()) == pytest.approx(0.42)
    assert float(front.max()) == pytest.approx(0.42)


def test_target_heatmap_center_peak_unchanged_under_crop() -> None:
    transform = ManiflowBlocksTableObsTransform(
        depth_keys=(),
        include_ptmap=False,
        center_crop_ratio=0.95,
        image_interpolate_device="cpu",
    )
    peak = (112, 112)
    obs = build_obs(batch=1)
    obs["target_heatmap"] = build_target_heatmap(peak_xy=peak, batch=1)
    out = transform(obs)
    hm = out["target_heatmap"]
    ch = hm[0, 0, 0]
    y, x = divmod(int(ch.argmax()), ch.shape[1])
    assert (x, y) == peak


def test_camera_intrinsics_independent_of_center_crop() -> None:
    kwargs = dict(
        depth_keys=("front_depth", "gripper_depth"),
        include_ptmap=False,
        image_interpolate_device="cpu",
    )
    obs = build_obs(batch=1)
    no_crop = ManiflowBlocksTableObsTransform(center_crop_ratio=None, **kwargs)(obs)
    with_crop = ManiflowBlocksTableObsTransform(center_crop_ratio=0.95, **kwargs)(obs)
    assert np.allclose(
        no_crop["camera_intrinsics"],
        with_crop["camera_intrinsics"],
    )
    rows = no_crop["camera_intrinsics"][0, 0]
    assert rows.shape == (2, 4)
    native_h, native_w = FIXTURE_CAM_HW
    sx = 224.0 / native_w
    sy = 224.0 / native_h
    k_front = FIXTURE_INTRINSICS["front_camera"]
    k_grip = FIXTURE_INTRINSICS["gripper_camera"]
    assert rows[0, 0] == pytest.approx(k_front[0][0] * sx)
    assert rows[0, 1] == pytest.approx(k_front[1][1] * sy)
    assert rows[1, 0] == pytest.approx(k_grip[0][0] * sx)
    assert rows[1, 1] == pytest.approx(k_grip[1][1] * sy)
    assert rows[0, 0] != rows[1, 0]


def test_ptmap_z_mask_before_center_crop() -> None:
    """Ptmap Z mask runs before center crop."""
    transform = ManiflowObsTransform(
        depth_keys=("front_depth",),
        center_crop_ratio=0.95,
        external_cam="front_camera",
        image_interpolate_device="cpu",
    )
    depth = np.full((1, 224, 224), 2.0, dtype=np.float32)
    depth[0, 112, 112] = 0.5
    obs = {
        "images": {
            "front_camera": {
                "depth": depth[..., None],
                "intrinsics": np.eye(3, dtype=np.float32)[None, ...],
                "rgb": np.zeros((1, 224, 224, 3), dtype=np.uint8),
            },
        },
        "agent": {"joint_pos": np.zeros((1, 8), dtype=np.float32)},
    }
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            ManiflowObsTransform,
            "get_depth",
            lambda self, o, cam: depth,
        )
        mp.setattr(
            ManiflowObsTransform,
            "process_depth",
            lambda self, d, size, **kw: d.astype(np.float32),
        )
        out = transform._depth_inputs(obs)
    front = out["front_depth"][0, 0, 0]
    assert float(front[112, 112]) == pytest.approx(0.5)
    assert float(front[0, 0]) == 0.0
