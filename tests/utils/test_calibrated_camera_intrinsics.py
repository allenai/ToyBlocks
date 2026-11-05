"""Tests for sim2real preset camera intrinsics."""

from __future__ import annotations

import numpy as np

from toyblocks.utils.calibrated_camera_intrinsics import (
    calibrated_camera_k3x3,
    calibrated_camera_k9_by_name,
    preset_k_for_camera,
)


def test_calibrated_k9_length_and_shape() -> None:
    k9 = calibrated_camera_k9_by_name()
    assert set(k9) == {"front_camera", "gripper_camera", "left_camera"}
    for name, flat in k9.items():
        assert len(flat) == 9
        mat = np.asarray(flat, dtype=np.float64).reshape(3, 3)
        np.testing.assert_allclose(mat, calibrated_camera_k3x3()[name])


def test_aug26_front_gripper_values() -> None:
    front = preset_k_for_camera("front_camera")
    np.testing.assert_allclose(front[0, 0], 393.07184, rtol=1e-5)
    np.testing.assert_allclose(front[1, 1], 393.07184, rtol=1e-5)
    assert front[0, 2] == 320.0
    assert front[1, 2] == 180.0

    gripper = preset_k_for_camera("gripper_camera")
    np.testing.assert_allclose(gripper[0, 0], 333.32, rtol=1e-4)
    assert gripper[0, 2] == 320.0
    assert gripper[1, 2] == 180.0


def test_dropped_camera_is_skipped_without_a_warning(caplog) -> None:
    from types import SimpleNamespace

    from toyblocks.utils.dataset_provenance import apply_camera_intrinsics_for_env

    env = SimpleNamespace(cameras={}, cfg=SimpleNamespace(drop_cameras=("left_camera",)))
    with caplog.at_level("WARNING"):
        written = apply_camera_intrinsics_for_env(env, 0, calibrated_camera_k9_by_name())
    assert written == []
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "'left_camera'" not in messages
    assert "'front_camera'" in messages
