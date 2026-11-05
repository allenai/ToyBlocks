"""``PolicyClientAdapter`` copies observations to the host only for steps that reach the server."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from toyblocks.serving.action_chunk_broker import ActionChunkBroker, PerEnvActionChunkBroker
from toyblocks.serving.policy_client import PolicyClientAdapter


class _RecordingPolicy:
    """Returns ``(T, B, 2)`` chunks and remembers every observation it was sent."""

    def __init__(self, num_envs: int | None, chunk_len: int) -> None:
        self.num_envs = num_envs
        self.chunk_len = chunk_len
        self.seen: list[dict[str, Any]] = []

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.seen.append(obs)
        batch = obs["state"].shape[0] if self.num_envs is None else self.num_envs
        a = np.zeros((self.chunk_len, batch, 2), dtype=np.float32)
        a[:, :, 0] = len(self.seen)
        a[:, :, 1] = np.arange(self.chunk_len)[:, None]
        if self.num_envs is None:  # unbatched: one env per request, (T, D)
            a = a[:, 0]
        return {"actions": a}

    def reset(self) -> None:
        pass


def _obs(num_envs: int) -> dict[str, torch.Tensor]:
    return {"state": torch.arange(num_envs * 3, dtype=torch.float32).reshape(num_envs, 3)}


def test_batched_copies_only_at_the_chunk_boundary() -> None:
    policy = _RecordingPolicy(num_envs=4, chunk_len=8)
    broker = ActionChunkBroker(policy, action_horizon=3, sync=True)
    adapter = PolicyClientAdapter(policy=broker, action_key="actions", device="cpu")
    copied: list[bool] = []
    for _ in range(7):
        action = adapter.act(_obs(4))
        copied.append(adapter.transfer_timings["obs_to_cpu_ms"] is not None)
        assert action.shape == (4, 2)
    assert copied == [True, False, False, True, False, False, True]
    assert len(policy.seen) == 3
    assert all(isinstance(o["state"], np.ndarray) for o in policy.seen)


def test_batched_episode_seed_waits_for_the_next_request() -> None:
    policy = _RecordingPolicy(num_envs=2, chunk_len=8)
    broker = ActionChunkBroker(policy, action_horizon=2, sync=True)
    adapter = PolicyClientAdapter(policy=broker, action_key="actions", device="cpu")
    adapter.act(_obs(2))
    adapter.set_episode_seed(7)
    adapter.act(_obs(2))  # served from the chunk: no request carries the seed
    adapter.act(_obs(2))  # boundary
    assert "__episode_seed__" not in policy.seen[0]
    assert int(policy.seen[1]["__episode_seed__"]) == 7


def test_unbatched_copies_only_envs_whose_chunk_ran_out() -> None:
    policy = _RecordingPolicy(num_envs=None, chunk_len=8)
    broker = PerEnvActionChunkBroker(policy, action_horizon=4)
    adapter = PolicyClientAdapter(policy=broker, action_key="actions", device="cpu", batched=False)
    adapter.act(_obs(3))
    assert len(policy.seen) == 3
    adapter.act(_obs(3))
    assert len(policy.seen) == 3
    assert adapter.transfer_timings["obs_to_cpu_ms"] is None
    broker.reset_envs([1])  # env 1 alone needs a new chunk
    adapter.act(_obs(3))
    assert len(policy.seen) == 4
    np.testing.assert_array_equal(policy.seen[-1]["state"], _obs(3)["state"][1].numpy())
