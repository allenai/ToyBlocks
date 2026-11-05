"""Depth transform + ptmap Z masking tests (requires maniflow-tb)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from tests.serving.policy_input_fixture import load_depth_fixture, to_torch
from toyblocks.serving.maniflow_serving_utils import (
    TRAIN_MAX_DEPTH_M,
    derive_model_input_spec,
    select_and_validate_obs,
)
from toyblocks.serving.obs_transforms.maniflow import ManiflowObsTransform

DUMMY_CAM_HW = (180, 320)


def dummy_obs(batch: int = 2, *, include_depth: bool = True) -> dict[str, Any]:
    arm = np.arange(1.0, 8.0, dtype=np.float32)
    gripper = np.full(1, 100.0, dtype=np.float32)
    joint_pos = np.tile(np.concatenate([arm, gripper]), (batch, 1))
    rgb = np.zeros((batch, *DUMMY_CAM_HW, 3), dtype=np.uint8)
    obs: dict[str, Any] = {
        "agent": {
            "joint_pos": joint_pos,
            "joint_vel": np.tile(np.arange(11.0, 18.0, dtype=np.float32), (batch, 1)),
            "ee_pos": np.tile(np.array([21.0, 22.0, 23.0], dtype=np.float32), (batch, 1)),
        },
        "images": {
            "front_camera": {"rgb": rgb},
            "gripper_camera": {"rgb": rgb.copy()},
        },
    }
    if include_depth:
        front_depth = np.full((batch, *DUMMY_CAM_HW), 0.9, dtype=np.float32)
        front_depth[:, :30] = np.inf
        front_depth[:, 30:60] = -1.0
        front_depth[:, 60:90] = TRAIN_MAX_DEPTH_M + 1.0
        gripper_depth = np.full((batch, *DUMMY_CAM_HW), 0.5, dtype=np.float32)
        front_K = np.array(
            [[100.0, 0.0, 30.0], [0.0, 200.0, 40.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        gripper_K = np.array(
            [[300.0, 0.0, 50.0], [0.0, 400.0, 60.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        obs["images"]["front_camera"].update(
            {"depth": front_depth, "intrinsics": np.tile(front_K, (batch, 1, 1))}
        )
        obs["images"]["gripper_camera"].update(
            {"depth": gripper_depth, "intrinsics": np.tile(gripper_K, (batch, 1, 1))}
        )
    return obs


def _depth_transform(
    *,
    image_size: tuple[int, int] = (224, 224),
    depth_keys: tuple[str, ...] = ("gripper_depth", "front_depth"),
) -> ManiflowObsTransform:
    return ManiflowObsTransform(
        external_cam="front_camera",
        wrist_cam="gripper_camera",
        image_size=image_size,
        image_interpolate_device="cpu",
        depth_keys=depth_keys,
        include_pd_mode=True,
        state_dim=9,
        prompt="move the block",
    )


def test_depth_and_intrinsics_kept_with_ptmaps() -> None:
    spec = derive_model_input_spec(*load_depth_fixture())
    assert spec["obs_shapes"]["front_depth"] == (1, 224, 224)
    assert spec["obs_shapes"]["gripper_depth"] == (1, 224, 224)
    assert spec["obs_shapes"]["camera_intrinsics"] == (2, 4)


def test_depth_roundtrip_shapes() -> None:
    spec = derive_model_input_spec(*load_depth_fixture())
    torch_obs = to_torch(_depth_transform()(dummy_obs(include_depth=True)))
    filtered, dropped = select_and_validate_obs(
        torch_obs, spec["obs_shapes"], spec["conditioning_keys"]
    )
    assert filtered["front_depth"].shape == (2, 1, 1, 224, 224)
    assert filtered["gripper_depth"].shape == (2, 1, 1, 224, 224)
    assert filtered["camera_intrinsics"].shape == (2, 1, 2, 4)
    assert dropped == []


def test_intrinsics_rescaled_per_axis() -> None:
    rows = _depth_transform()(dummy_obs(batch=1, include_depth=True))[
        "camera_intrinsics"
    ][0, 0]
    sx = 224.0 / DUMMY_CAM_HW[1]
    sy = 224.0 / DUMMY_CAM_HW[0]
    np.testing.assert_allclose(
        rows[0],
        [100.0 * sx, 200.0 * sy, 30.0 * sx, 40.0 * sy],
    )
    np.testing.assert_allclose(
        rows[1],
        [300.0 * sx, 400.0 * sy, 50.0 * sx, 60.0 * sy],
    )


def test_intrinsics_rows_follow_sorted_depth_keys() -> None:
    transform = _depth_transform(depth_keys=("gripper_depth", "front_depth"))
    assert transform.depth_keys == ["front_depth", "gripper_depth"]
    rows = transform(dummy_obs(batch=1, include_depth=True))["camera_intrinsics"][0, 0]
    assert rows[0, 0] < rows[1, 0]
    np.testing.assert_allclose(rows[:, 0], [70.0, 210.0])


def test_depth_invalid_samples_zeroed() -> None:
    front = _depth_transform()(dummy_obs(batch=1, include_depth=True))[
        "front_depth"
    ][0, 0, 0]
    assert np.isfinite(front).all()
    uniq = np.unique(front)
    assert np.any(uniq == 0.0)
    assert np.any(np.isclose(uniq, 0.9, rtol=0, atol=1e-5))


def test_depth_resize_is_nearest() -> None:
    obs = dummy_obs(batch=1, include_depth=True)
    edge = np.full((1, *DUMMY_CAM_HW), 0.5, dtype=np.float32)
    edge[:, :, DUMMY_CAM_HW[1] // 2 :] = 0.55
    obs["images"]["front_camera"]["depth"] = edge
    front = _depth_transform(image_size=(2, 2), depth_keys=("front_depth",))(obs)[
        "front_depth"
    ][0, 0, 0]
    np.testing.assert_allclose(
        np.sort(np.unique(front)),
        [0.5, 0.55],
        rtol=1e-5,
        atol=1e-5,
    )


def test_training_max_depth_is_pinned() -> None:
    assert TRAIN_MAX_DEPTH_M == 2.5


def test_training_max_depth_matches_hvla_when_installed() -> None:
    import importlib

    for mod_name in (
        "hvla.dataloader.franka_dataset",
        "hvla.dataloader.constants",
        "hvla.dataloader",
    ):
        try:
            mod = importlib.import_module(mod_name)
        except ImportError:
            continue
        value = getattr(mod, "MAX_DEPTH_M", None)
        if value is not None:
            assert value == TRAIN_MAX_DEPTH_M
            return
    pytest.skip("HVLA build exports no MAX_DEPTH_M")
