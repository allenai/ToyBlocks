"""Vendored ptmap Z bounds (no maniflow-tb import)."""

from __future__ import annotations

import numpy as np
import pytest

from toyblocks.serving.obs_transforms.depth_ptmap_bounds import (
    DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS,
    mask_depth_m_by_z_bounds,
    z_bounds_for_depth_key,
)


def test_z_bounds_front_depth_matches_training_defaults() -> None:
    z_min, z_max = z_bounds_for_depth_key(
        "front_depth", DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS
    )
    assert z_min == pytest.approx(0.4)
    assert z_max == pytest.approx(1.0)


def test_mask_depth_m_by_z_bounds_zeros_out_of_band() -> None:
    depth = np.array([[0.3, 0.5, 1.5]], dtype=np.float32)
    out = mask_depth_m_by_z_bounds(depth, 0.4, 1.0)
    assert out.tolist() == [[0.0, 0.5, 0.0]]
