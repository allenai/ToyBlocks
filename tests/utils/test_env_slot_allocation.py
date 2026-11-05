"""Cube-scale slot allocation: envs per slot sized to the queue's per-slot workload."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from blocksuite.utils.eval_helpers import allocate_env_slots

_REPO = Path(__file__).resolve().parents[2]


def _rounds(counts: list[int], env_slots: list[int]) -> int:
    return max(-(-n // env_slots.count(s)) for s, n in enumerate(counts) if n)


def test_uneven_slots_finish_in_one_round_when_envs_match_episodes() -> None:
    counts = [9, 8, 8, 8, 8, 8, 8, 7]  # 64 episodes, uneven over 8 slots
    slots = allocate_env_slots(counts, [float(n) for n in counts], 64)
    assert len(slots) == 64
    assert [slots.count(s) for s in range(8)] == counts
    assert _rounds(counts, slots) == 1
    # Round-robin (env i -> slot i % 8) needs a second round for the 9-episode slot.
    assert _rounds(counts, [i % 8 for i in range(64)]) == 2


def test_more_work_gets_more_envs_and_empty_slots_get_none() -> None:
    counts = [10, 0, 10]
    work = [10 * 1000.0, 0.0, 10 * 200.0]  # slot 0's episodes are 5x longer
    slots = allocate_env_slots(counts, work, 6)
    assert slots.count(1) == 0
    assert slots.count(0) > slots.count(2) >= 1


def test_never_more_envs_than_episodes_until_all_are_covered() -> None:
    slots = allocate_env_slots([2, 3], [2.0, 3.0], 7)
    assert len(slots) == 7
    assert slots.count(0) >= 2 and slots.count(1) >= 3  # every episode has an env


def test_too_few_envs_for_the_active_slots_raises() -> None:
    with pytest.raises(ValueError, match="cannot cover"):
        allocate_env_slots([1, 1, 1], [1.0, 1.0, 1.0], 2)


def _queue_module():
    spec = importlib.util.spec_from_file_location(
        "serve_episode_queue", _REPO / "scripts" / "serve_episode_queue.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("serve_episode_queue", module)
    spec.loader.exec_module(module)
    return module


def test_queue_reports_slot_workload_and_serves_longest_first() -> None:
    queue = _queue_module()
    setups = [
        {"episode_id": "a", "cube_scale": 0.04, "expert_episode_length": 100},
        {"episode_id": "b", "cube_scale": 0.05, "expert_episode_length": 300},
        {"episode_id": "c", "cube_scale": 0.04, "expert_episode_length": 400},
    ]
    server = queue.QueueServer(("127.0.0.1", 0), queue.QueueHandler, setups, expert_seconds_max=16.0)
    try:
        status = server.size_status()
        assert status["slot_counts"] == [2, 1]
        assert status["slot_expert_frames"] == [500, 300]
        scales = [0.04, 0.05]
        assert server.get_next(0, scales)["episode_id"] == "c"
        assert server.get_next(0, scales)["episode_id"] == "a"
    finally:
        server.server_close()


def test_queue_client_asks_for_each_env_s_spawn_slot(monkeypatch) -> None:
    from blocksuite.utils import episode_queue

    asked: list[int] = []

    def fake_fetch(url, timeout=30.0, slot=None, slot_scales=None):
        asked.append(slot)
        return None

    monkeypatch.setattr(episode_queue, "fetch_next_setup", fake_fetch)
    client = episode_queue.EpisodeQueueClient(
        base_env=None, queue_url="http://q/next", num_envs=3, num_slots=4,
        env_slots=[3, 0, 3], verify=False,
    )
    client.fetch_reset_states([0, 1, 2])
    assert asked == [3, 0, 3]
