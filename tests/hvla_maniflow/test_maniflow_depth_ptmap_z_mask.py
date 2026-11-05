"""Maniflow eval ptmap Z depth masking vs vendored bounds."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from blocksuite.serving.obs_transforms.depth_ptmap_bounds import (
    DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS,
    z_bounds_for_depth_key,
)
from blocksuite.serving.obs_transforms.maniflow import (
    ManiflowObsTransform,
    _default_per_camera_ptmap_bounds,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_default_ptmap_z_bounds_mask_after_resize() -> None:
    """Depth checkpoints use DEFAULT_FRANKA ptmap Z bounds on front_depth."""
    z_min, z_max = z_bounds_for_depth_key(
        "front_depth", DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS
    )
    transform = ManiflowObsTransform(depth_keys=("front_depth",))
    depth = np.array([[[z_min - 0.05, z_min + 0.05, z_max + 0.5]]], dtype=np.float32)
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
        out = transform._depth_inputs({"images": {}, "camera_intrinsics": {}})
    front = out["front_depth"][0, 0, 0]
    assert float(front[0, 0]) == 0.0
    assert float(front[0, 1]) == pytest.approx(z_min + 0.05)
    assert float(front[0, 2]) == 0.0


def test_default_z_mask_loads_ptmap_bounds_lazily() -> None:
    with patch(
        "blocksuite.serving.obs_transforms.maniflow._default_per_camera_ptmap_bounds",
        wraps=_default_per_camera_ptmap_bounds,
    ) as loader:
        transform = ManiflowObsTransform(depth_keys=("front_depth",))
        assert transform._per_camera_ptmap_bounds is None
        loader.assert_not_called()
        z_min, _z_max = z_bounds_for_depth_key(
            "front_depth", _default_per_camera_ptmap_bounds()
        )
        depth = np.array([[[z_min + 0.05]]], dtype=np.float32)
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
            transform._depth_inputs({"images": {}, "camera_intrinsics": {}})
        loader.assert_called_once()


def test_hvla_submodule_checkout_matches_gitlink() -> None:
    """blocksuite pin and maniflow-tb checkout stay aligned."""
    line = subprocess.check_output(
        ["git", "ls-tree", "HEAD", "maniflow-tb"],
        cwd=_REPO_ROOT,
        text=True,
    ).strip()
    assert line, "maniflow-tb gitlink missing at HEAD"
    pinned = line.split()[2]
    checked_out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPO_ROOT / "maniflow-tb",
        text=True,
    ).strip()
    assert checked_out == pinned, (
        f"maniflow-tb checkout {checked_out[:12]} != blocksuite pin {pinned[:12]}; "
        "run git submodule update --init maniflow-tb"
    )
