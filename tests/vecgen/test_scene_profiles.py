"""Reset states and dataset episodes are matched to envs by profile *value*.

A scene profile (per-block sizes, friction) is fixed per env at spawn, so a
reset state or recorded episode is only valid in an env spawned with the same
values. These tests pin the matching to values rather than to ``i % N`` index
conventions: equal ids with different values must not match, different ids
with equal values must.
"""

from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import pytest
import torch

from toyblocks.utils.reset_states_io import (
    load_reset_states,
    load_reset_states_with_profiles,
    pack_reset_states,
    reset_state_slot_compat,
    sample_compatible_reset_states,
    spawn_profiles_match,
)


def _profile(pid: int, side: float, friction: float = 0.5, table_height: float = 0.0) -> dict:
    return {
        "profile_id": pid,
        "block_sizes": {"block_a": [side] * 3, "block_b": [side] * 3},
        "table_height": table_height,
        "static_friction": friction,
        "dynamic_friction": friction,
        "restitution": 0.0,
    }


SLOTS = [_profile(0, 0.040), _profile(1, 0.045), _profile(2, 0.050)]


def test_profile_values_round_trip_through_reset_states_pkl(tmp_path) -> None:
    states: dict[str, Any] = {"rigid_object": {"block_a": {"root_pose": np.zeros((3, 7))}}}
    path = tmp_path / "reset_states.pkl"
    with open(path, "wb") as f:
        pickle.dump(pack_reset_states(states, np.array([0, 2, 2]), 3, profiles=SLOTS), f)

    _, ids, num, profiles = load_reset_states_with_profiles(path)
    assert ids.tolist() == [0, 2, 2] and num == 3
    assert profiles == SLOTS
    # The 3-tuple reader keeps working for existing consumers.
    assert load_reset_states(path)[1].tolist() == [0, 2, 2]


def test_ids_only_and_legacy_payloads_report_no_profile_values(tmp_path) -> None:
    ids_only = tmp_path / "ids_only.pkl"
    with open(ids_only, "wb") as f:
        pickle.dump(pack_reset_states({}, np.array([0]), 1), f)
    legacy = tmp_path / "legacy.pkl"
    with open(legacy, "wb") as f:
        pickle.dump({"rigid_object": {}}, f)

    assert load_reset_states_with_profiles(ids_only)[3] is None
    assert load_reset_states_with_profiles(legacy)[1:] == (None, None, None)


def test_spawn_profiles_match_compares_sizes_and_friction_not_ids_or_table() -> None:
    assert spawn_profiles_match(_profile(0, 0.04), _profile(7, 0.04, table_height=0.03))
    assert not spawn_profiles_match(_profile(0, 0.04), _profile(0, 0.0401))
    assert not spawn_profiles_match(_profile(0, 0.04), _profile(0, 0.04, friction=0.3))
    other_blocks = _profile(0, 0.04)
    other_blocks["block_sizes"] = {"block_a": [0.04] * 3}
    assert not spawn_profiles_match(_profile(0, 0.04), other_blocks)
    # Float32 round trips of the same value still match.
    assert spawn_profiles_match(_profile(0, 0.04), _profile(0, float(np.float32(0.04))))


def test_compat_matches_recorded_values_not_recorded_ids() -> None:
    # Recorded profile 0 has the values of the loading env's slot 2, and the
    # recorded profile 1 exists in no slot at all.
    recorded = [_profile(0, 0.050), _profile(1, 0.060)]
    compat = reset_state_slot_compat(np.array([0, 1, 0]), recorded, SLOTS)
    assert compat.tolist() == [
        [False, False, True],
        [False, False, False],
        [False, False, True],
    ]


def test_compat_without_values_uses_ids_as_this_package_slots() -> None:
    compat = reset_state_slot_compat(np.array([2, 0, 5]), None, SLOTS)
    assert compat.tolist() == [
        [False, False, True],
        [True, False, False],
        [False, False, False],
    ]


def test_compat_rejects_ids_with_no_recorded_values() -> None:
    with pytest.raises(ValueError, match=r"profile\(s\) \[4\]"):
        reset_state_slot_compat(np.array([0, 4]), [_profile(0, 0.04)], SLOTS)


def test_random_reset_only_draws_states_of_the_env_profile() -> None:
    compat = torch.as_tensor(reset_state_slot_compat(np.array([0, 1, 2, 0, 1, 2]), SLOTS, SLOTS))
    env_slots = torch.tensor([2, 0, 1, 2, 0, 1] * 50)
    torch.manual_seed(0)
    picks = sample_compatible_reset_states(compat, env_slots)
    assert compat[picks, env_slots].all()
    # Both candidates of each slot get drawn.
    assert set(picks[env_slots == 2].tolist()) == {2, 5}


def test_random_reset_raises_when_a_slot_has_no_reset_state() -> None:
    compat = torch.as_tensor(reset_state_slot_compat(np.array([0, 0]), SLOTS, SLOTS))
    with pytest.raises(ValueError, match="spawn slot 1"):
        sample_compatible_reset_states(compat, torch.tensor([0, 1]))


# --- vecgen: envs draw reset states by the profile they were spawned with ---


class _FakeTemplateEnv:
    """The TemplateEnv surface VecSim reads: per-env spawn slots and slot values."""

    MAX_NON_SIM_STATE_RANDOMIZATIONS = 3

    def __init__(
        self,
        env_slots: list[int],
        slot_profiles: list[dict],
        rs_ids: list[int] | None,
        rs_profiles: list[dict] | None,
    ) -> None:
        self.env_slots = env_slots
        self.slot_profiles = slot_profiles
        self.rs_ids = rs_ids
        self.rs_profiles = rs_profiles
        self.num_reset_states = len(rs_ids) if rs_ids is not None else 6

    def env_scene_profile(self, e: int) -> dict:
        return self.slot_profiles[self.env_slots[e]]

    def _reset_state_slot_compat(self) -> torch.Tensor | None:
        if self.rs_ids is None:
            return None
        return torch.as_tensor(
            reset_state_slot_compat(np.array(self.rs_ids), self.rs_profiles, self.slot_profiles)
        )


def _vec_sim(base_env: _FakeTemplateEnv) -> Any:
    from toyblocks.vecgen.sim import VecSim

    sim = VecSim.__new__(VecSim)
    sim.base_env = base_env
    sim.num_envs = len(base_env.env_slots)
    sim.block_names = ["block_a", "block_b"]
    return sim


def test_vecgen_matches_another_packages_profile_by_value_not_id() -> None:
    # Reset states recorded by package A (its profile 1 = 4.5 cm) run in package B,
    # whose profile 1 is 4 cm and whose 4.5 cm profile is id 2: B's slot 1 gets A's
    # profile-0 states and slot 2 gets A's profile-1 states; B's 5 cm slot gets none.
    recorded = [_profile(0, 0.040), _profile(1, 0.045)]
    package_b = [_profile(0, 0.050), _profile(1, 0.040), _profile(2, 0.045)]
    sim = _vec_sim(_FakeTemplateEnv([0, 1, 2, 1], package_b, [0, 1, 0, 1], recorded))
    assert sim.reset_states_by_slot() == {0: [], 1: [0, 2], 2: [1, 3]}


def test_vecgen_reset_states_follow_explicit_env_slots() -> None:
    sim = _vec_sim(_FakeTemplateEnv([2, 2, 0], SLOTS, [0, 1, 2, 2, 0], SLOTS))
    assert sim.env_slots() == [2, 2, 0]
    assert sim.reset_states_by_slot() == {0: [0, 4], 2: [2, 3]}


def test_vecgen_block_sizes_are_the_spawned_profile_per_env() -> None:
    sim = _vec_sim(_FakeTemplateEnv([2, 0, 1], SLOTS, [0, 1, 2], SLOTS))
    sizes = sim.block_sizes()
    assert sizes.shape == (3, 2, 3)
    assert torch.allclose(sizes[:, 0, 0], torch.tensor([0.050, 0.040, 0.045]))


def test_vecgen_engine_names_the_profile_with_no_reset_state() -> None:
    from collections import deque

    from toyblocks.vecgen.engine import Engine, EngineConfig

    recorded = [_profile(0, 0.040)]
    package = [_profile(0, 0.050), _profile(1, 0.040)]
    sim = _vec_sim(_FakeTemplateEnv([0, 1], package, [0, 0], recorded))
    engine = Engine.__new__(Engine)
    engine.sim = sim
    engine.cfg = EngineConfig()
    engine._env_slot = sim.env_slots()
    engine._rs_by_slot = sim.reset_states_by_slot()
    engine._rs_queue = {s: deque(q) for s, q in engine._rs_by_slot.items()}
    assert [engine._next_reset_state(1) for _ in range(3)] == [0, 1, 0]
    with pytest.raises(RuntimeError, match="0.05"):
        engine._next_reset_state(0)

    # reset_states_once: each state is handed out once, then None instead of cycling
    engine.cfg = EngineConfig(reset_states_once=True)
    engine._started = set()
    engine._rs_queue = {s: deque(q) for s, q in engine._rs_by_slot.items()}
    assert [engine._next_reset_state(1) for _ in range(3)] == [0, 1, None]

    # a state that matches two slots (equal profile values) still starts only once
    engine._started = set()
    engine._env_slot = [0, 1]
    engine._rs_by_slot = {0: [0, 1, 2], 1: [0, 1, 3]}
    engine._rs_queue = {s: deque(q) for s, q in engine._rs_by_slot.items()}
    got = [engine._next_reset_state(0), engine._next_reset_state(1)]
    got += [engine._next_reset_state(1), engine._next_reset_state(0), engine._next_reset_state(0)]
    assert got == [0, 1, 3, 2, None]
    assert not engine._unstarted(0) and not engine._unstarted(1)


def test_vecgen_rejects_a_package_without_per_env_profiles() -> None:
    sim = _vec_sim(_FakeTemplateEnv([0], SLOTS, [0], SLOTS))
    sim.base_env = object()
    with pytest.raises(RuntimeError, match="regenerate"):
        sim.env_slots()
