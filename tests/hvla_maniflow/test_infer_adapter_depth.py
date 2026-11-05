"""Infer adapter tests that run depth transforms (requires maniflow-tb)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from blocksuite.serving.maniflow_serving_utils import (
    inject_target_front_crop_from_native,
    select_and_validate_obs,
)
from blocksuite.serving.obs_transforms.maniflow import transform_kwargs_from_server_metadata
from blocksuite.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)
from blocksuite.serving.serve_maniflow_policy import HLLLInferAdapter
from tests.serving.infer_adapter_fixtures import (
    _StubPolicy,
    _adapter,
    _client_request,
    _spec_and_transform,
)
from tests.serving.policy_input_fixture import (
    FIXTURE_BATCH,
    build_obs,
    load_stats,
    stats_action_mode,
)


def test_metadata_rederives_the_client_transform_layout_state9_depth() -> None:
    adapter, _, spec, _ = _adapter(9, depth=True)
    kwargs = transform_kwargs_from_server_metadata(adapter.metadata)
    rebuilt = ManiflowBlocksTableObsTransform(
        prompt="pick up the red block", image_interpolate_device="cpu", **kwargs
    )
    assert rebuilt.depth_keys == sorted(
        k for k in spec["obs_shapes"] if k.endswith("_depth")
    )
    torch_obs = {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in rebuilt(build_obs()).items()
    }
    filtered, dropped = select_and_validate_obs(
        torch_obs, spec["obs_shapes"], spec["conditioning_keys"]
    )
    assert filtered["agent_pos"].shape[-1] == 9
    assert dropped == []


def test_depth_and_intrinsics_reach_the_model_for_a_depth_checkpoint() -> None:
    adapter, stub, _, transform = _adapter(9, depth=True)
    adapter.infer(_client_request(transform))
    assert stub.seen_obs["front_depth"].shape == (FIXTURE_BATCH, 1, 1, 224, 224)
    assert stub.seen_obs["camera_intrinsics"].shape == (FIXTURE_BATCH, 1, 2, 4)


def test_hlll_native_crop_after_molmo_hl() -> None:
    """Server builds crop from native_front_rgb + HL target_xy (needs torchvision)."""
    spec, _ = _spec_and_transform(19)
    spec = dict(spec)
    spec["use_target_front_crop"] = 50
    obs_shapes = {
        k: v for k, v in spec["obs_shapes"].items() if not k.endswith("_ptmap")
    }
    obs_shapes["target_front_crop_rgb"] = obs_shapes["front_rgb"]
    spec["obs_shapes"] = obs_shapes

    transform = ManiflowBlocksTableObsTransform(
        prompt="pick up the red block",
        state_dim=19,
        depth_keys=(),
        include_ptmap=False,
        enable_target_front_crop=True,
        target_front_crop_bottom_pct=50,
        orig_cam_w=640.0,
        orig_cam_h=360.0,
        pass_native_front_for_server_crop=True,
        image_interpolate_device="cpu",
    )
    wire = transform(build_obs(batch=1))
    assert "native_front_rgb" in wire
    assert "target_front_crop_rgb" not in wire
    wire["hl_env_index"] = np.array([0], dtype=np.int64)
    wire["hl_episode_id"] = np.array([1], dtype=np.int64)
    native_front = np.asarray(wire["native_front_rgb"])

    class _FixedXYHL:
        def get_action(self, obs: dict[str, Any], lang: str = "") -> str:
            return "pick <160, 90>"

    ll_stub = _StubPolicy()
    adapter = HLLLInferAdapter(
        ll_policy=ll_stub,  # type: ignore[arg-type]
        hl_policy=_FixedXYHL(),  # type: ignore[arg-type]
        stats=load_stats(19),
        action_mode=stats_action_mode(19),
        model_input_spec=spec,
    )
    adapter.infer(wire)
    assert ll_stub.seen_obs is not None

    expected: dict[str, Any] = {
        "front_rgb": torch.from_numpy(np.asarray(wire["front_rgb"], dtype=np.float32)),
    }
    inject_target_front_crop_from_native(
        expected,
        50,
        native_front,
        np.array([160.0, 90.0], dtype=np.float32),
    )
    seen_crop = ll_stub.seen_obs["target_front_crop_rgb"].detach().cpu().numpy()
    golden = expected["target_front_crop_rgb"].detach().cpu().numpy()
    np.testing.assert_allclose(seen_crop, golden, rtol=1e-5, atol=1e-5)
