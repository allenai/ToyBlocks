"""``ActionChunkBroker(sync=True)``: one shared chunk boundary for every env."""

from __future__ import annotations

from typing import Any

import numpy as np

from toyblocks.serving.action_chunk_broker import ActionChunkBroker


class _CountingPolicy:
    """Returns ``(T, B, 2)`` chunks whose rows encode ``(call id, step index)``."""

    def __init__(self, num_envs: int, chunk_len: int) -> None:
        self.num_envs = num_envs
        self.chunk_len = chunk_len
        self.calls = 0

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        a = np.zeros((self.chunk_len, self.num_envs, 2), dtype=np.float32)
        a[:, :, 0] = self.calls
        a[:, :, 1] = np.arange(self.chunk_len)[:, None]
        return {"actions": a}

    def reset(self) -> None:
        pass


def test_sync_infers_once_per_horizon_and_parks_reset_envs():
    policy = _CountingPolicy(num_envs=3, chunk_len=8)
    broker = ActionChunkBroker(policy, action_horizon=4, sync=True)
    broker.infer({})
    broker.infer({})
    broker.reset_envs([1])  # env 1's episode ended mid-chunk
    out = broker.infer({})
    assert broker.waiting_envs == [1]
    assert np.all(out["actions"][1] == 0)  # placeholder row
    broker.infer({})
    assert policy.calls == 1
    out = broker.infer({})  # boundary: every env, including the parked one, takes a fresh chunk
    assert policy.calls == 2
    assert broker.waiting_envs == []
    assert np.all(out["actions"][:, 1] == 0)  # every env at step 0 of the new chunk


def test_sync_infers_early_once_every_env_is_waiting():
    policy = _CountingPolicy(num_envs=2, chunk_len=8)
    broker = ActionChunkBroker(policy, action_horizon=4, sync=True)
    broker.infer({})
    broker.reset_envs([0, 1])
    assert broker.needs_inference
    broker.infer({})
    assert policy.calls == 2


def test_unsynced_reinfers_whenever_one_env_needs_a_chunk():
    policy = _CountingPolicy(num_envs=2, chunk_len=8)
    broker = ActionChunkBroker(policy, action_horizon=4)
    broker.infer({})
    broker.reset_envs([0])
    broker.infer({})
    assert policy.calls == 2
    assert broker.waiting_envs == []


def test_needs_inference_for_env_marks_each_envs_next_chunk():
    policy = _CountingPolicy(num_envs=2, chunk_len=8)
    synced = ActionChunkBroker(policy, action_horizon=4, sync=True)
    synced.infer({})
    assert not synced.needs_inference_for_env(0)
    for _ in range(3):
        synced.infer({})
    assert synced.needs_inference_for_env(0) and synced.needs_inference_for_env(1)

    unsynced = ActionChunkBroker(_CountingPolicy(num_envs=2, chunk_len=8), action_horizon=4)
    unsynced.infer({})
    unsynced.reset_envs([1])
    assert not unsynced.needs_inference_for_env(0)
    assert unsynced.needs_inference_for_env(1)
