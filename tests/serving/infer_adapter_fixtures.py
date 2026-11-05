"""Shared stubs/helpers for ManiFlow infer adapter tests (unit + hvla_maniflow)."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from tests.serving.policy_input_fixture import (
    build_obs,
    load_depth_fixture,
    load_fixture,
    load_stats,
    stats_action_mode,
)
from toyblocks.serving.maniflow_serving_utils import derive_model_input_spec
from toyblocks.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)
from toyblocks.serving.serve_maniflow_policy import HLLLInferAdapter, ManiFlowInferAdapter

ACTION_HORIZON = 4


class _StubPolicy:
    def __init__(self, action_dim: int = 9, horizon: int = ACTION_HORIZON) -> None:
        self._action_dim = action_dim
        self._horizon = horizon
        self._param = torch.zeros(1)
        self.eval_calls = 0
        self.to_calls: list[str] = []
        self.seen_obs: dict[str, Any] | None = None
        self.next_action: torch.Tensor | None = None

    def eval(self) -> _StubPolicy:
        self.eval_calls += 1
        return self

    def to(self, device: Any) -> _StubPolicy:
        self.to_calls.append(str(device))
        return self

    def parameters(self):
        yield self._param

    def predict_action(self, obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        self.seen_obs = dict(obs)
        if self.next_action is not None:
            return {"action": self.next_action}
        batch = int(obs["agent_pos"].shape[0])
        return {
            "action": torch.zeros(batch, self._horizon, self._action_dim, dtype=torch.float32)
        }


class _StubHLPolicy:
    def __init__(self) -> None:
        self.seen_langs: list[str] = []

    def get_action(self, obs: dict[str, Any], lang: str = "") -> str:
        self.seen_langs.append(lang)
        return f"pick <100, 200> for {lang}"


def _hlll_adapter() -> tuple[HLLLInferAdapter, _StubPolicy, _StubHLPolicy, Any]:
    spec, transform = _spec_and_transform(19)
    ll_stub = _StubPolicy()
    hl_stub = _StubHLPolicy()
    adapter = HLLLInferAdapter(
        ll_policy=ll_stub,  # type: ignore[arg-type]
        hl_policy=hl_stub,  # type: ignore[arg-type]
        stats=load_stats(19),
        action_mode=stats_action_mode(19),
        model_input_spec=spec,
    )
    return adapter, ll_stub, hl_stub, transform


def _hlll_request(transform: Any, env_idx: int, episode_id: int) -> dict[str, Any]:
    wire = transform(build_obs(batch=1))
    wire["hl_env_index"] = np.array([env_idx], dtype=np.int64)
    wire["hl_episode_id"] = np.array([episode_id], dtype=np.int64)
    wire["task_name"] = [f"goal for env {env_idx}"]
    return wire


def _spec_and_transform(state_dim: int, *, depth: bool = False):
    cfg, fw = load_depth_fixture() if depth else load_fixture(state_dim)
    spec = derive_model_input_spec(cfg, fw)
    transform = ManiflowBlocksTableObsTransform(
        prompt="pick up the red block",
        state_dim=spec["state_dim"],
        depth_keys=sorted(k for k in spec["obs_shapes"] if k.endswith("_depth")),
        image_interpolate_device="cpu",
    )
    return spec, transform


def _adapter(
    state_dim: int, *, depth: bool = False, with_stats: bool = True
) -> tuple[ManiFlowInferAdapter, _StubPolicy, dict[str, Any], Any]:
    spec, transform = _spec_and_transform(state_dim, depth=depth)
    stub = _StubPolicy()
    adapter = ManiFlowInferAdapter(
        policy=stub,  # type: ignore[arg-type]
        stats=load_stats(state_dim) if with_stats else None,
        action_mode=stats_action_mode(state_dim),
        model_input_spec=spec,
    )
    return adapter, stub, spec, transform


def _client_request(transform, **extra: Any) -> dict[str, Any]:
    wire = transform(build_obs())
    wire.update(extra)
    return wire
