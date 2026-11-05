"""CPU tests for the trace request channel and the wire, with no server process.

The policy-input trace request crosses four hops that all drop silently:

    PolicyClientAdapter.set_policy_input_trace
      -> ``__ll_input_trace__`` on the observation dict
      -> forwarded by ObsTransformPolicy, which must *re-add* it because the
         transform builds its output dict from scratch
      -> msgpack to the server
      -> popped by ManiFlowInferAdapter.infer

Lose any one and the symptom is an empty trace directory, discovered by the
post-run ``RuntimeError`` in ``run_eval`` -- after a full eval has been paid for.
Each hop is cheap to check and none of them was.

The wire is covered by a round-trip through the real ``msgpack_numpy`` codec
rather than by standing up ``PolicyServer``: ``serve_forever`` is
``asyncio.run(self._run())`` with the bound socket never exposed, so a test needs
a pre-reserved port and an unkillable daemon thread. That is flakiness for value
the codec round-trip plus ``test_infer_adapter.py`` already deliver.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from blocksuite.serving import msgpack_numpy
from blocksuite.serving.action_chunk_broker import (
    ActionChunkBroker,
    PerEnvActionChunkBroker,
    _split_envs,
)
from blocksuite.serving.base_policy import BasePolicy
from blocksuite.serving.obs_transforms import ObsTransformPolicy
from blocksuite.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)
from blocksuite.serving.policy_client import PolicyClientAdapter
from tests.serving.policy_input_fixture import (
    FIXTURE_BATCH,
    build_obs,
    build_target_heatmap,
)

TRACE_KEY = "__ll_input_trace__"
ACTION_HORIZON = 3


#: What a batched server returns: leading time axis, then env, then action dim.
BATCHED_ACTION_SHAPE = (ACTION_HORIZON, FIXTURE_BATCH, 9)
#: What the unbatched client sees per env, which it then stacks itself.
UNBATCHED_ACTION_SHAPE = (ACTION_HORIZON, 9)


class _RecordingPolicy(BasePolicy):
    """Stands in for the websocket client: records every request it is sent.

    The response shape is fixed at construction rather than derived from the
    observation, because these tests put the recorder at several depths in the
    client stack -- sometimes behind the obs transform (so it sees ``agent_pos``)
    and sometimes directly under the adapter (so it sees the raw env dict).
    """

    def __init__(self, action_shape: tuple[int, ...] = BATCHED_ACTION_SHAPE) -> None:
        self._action_shape = action_shape
        self.requests: list[dict[str, Any]] = []
        self.resets = 0

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self.requests.append(dict(obs))
        return {"actions": np.zeros(self._action_shape, dtype=np.float32)}

    def reset(self) -> None:
        self.resets += 1

    @property
    def traced(self) -> list[dict | None]:
        """The trace payload seen on each request, ``None`` where absent."""
        return [r.get(TRACE_KEY) for r in self.requests]


def _transform() -> ManiflowBlocksTableObsTransform:
    return ManiflowBlocksTableObsTransform(
        prompt="pick up the red block",
        state_dim=19,
        image_interpolate_device="cpu",
    )


def _torch_obs() -> dict[str, torch.Tensor]:
    """The batched torch observation an Isaac env hands the adapter."""
    obs = build_obs()
    return {
        "agent": {k: torch.from_numpy(v) for k, v in obs["agent"].items()},
        "images": {
            cam: {k: torch.from_numpy(v) for k, v in entry.items()}
            for cam, entry in obs["images"].items()
        },
    }


# --------------------------------------------------------------------------
# set_policy_input_trace
# --------------------------------------------------------------------------


def test_batched_trace_is_injected_on_every_request_until_cleared():
    """Not one-shot, unlike ``set_episode_seed``.

    Only the calls that survive the chunk broker reach the server, and that set
    is exactly the handovers worth dumping -- so the request stays on until the
    eval clears it at the traced env's episode end.
    """
    recorder = _RecordingPolicy()
    adapter = PolicyClientAdapter(recorder, device="cpu")
    obs = _torch_obs()

    adapter.set_policy_input_trace("/tmp/trace", env_idx=1)
    adapter.act(obs)
    adapter.act(obs)
    adapter.set_policy_input_trace(None)
    adapter.act(obs)

    assert recorder.traced == [
        {"dir": "/tmp/trace", "env_idx": 1},
        {"dir": "/tmp/trace", "env_idx": 1},
        None,
    ]


def test_no_trace_key_when_none_was_requested():
    recorder = _RecordingPolicy()
    PolicyClientAdapter(recorder, device="cpu").act(_torch_obs())
    assert recorder.traced == [None]


def test_episode_seed_is_one_shot_and_the_trace_is_not():
    """Pinned together because they share the injection site and differ here.

    A seed that stuck would re-seed the server every step; a trace that did not
    would dump one panel per episode instead of one per handover.
    """
    recorder = _RecordingPolicy()
    adapter = PolicyClientAdapter(recorder, device="cpu")
    adapter.set_episode_seed(11)
    adapter.set_policy_input_trace("/tmp/trace")
    adapter.act(_torch_obs())
    adapter.act(_torch_obs())

    assert "__episode_seed__" in recorder.requests[0]
    assert "__episode_seed__" not in recorder.requests[1]
    assert recorder.traced == [{"dir": "/tmp/trace", "env_idx": 0}] * 2


def test_unbatched_trace_goes_only_to_the_followed_env_and_reindexes():
    """Each unbatched request is already ``B == 1``, so the env is selected here.

    The server then slices batch element 0, which is why the injected payload
    carries ``env_idx: 0`` rather than the client's env index -- forwarding the
    client's index would have the server slice past the end of a 1-env batch.
    """
    recorder = _RecordingPolicy(UNBATCHED_ACTION_SHAPE)
    adapter = PolicyClientAdapter(recorder, device="cpu", batched=False)
    adapter.set_policy_input_trace("/tmp/trace", env_idx=1)
    adapter.act(_torch_obs())

    assert len(recorder.requests) == FIXTURE_BATCH
    assert recorder.traced == [None, {"dir": "/tmp/trace", "env_idx": 0}]


def test_unbatched_trace_for_an_env_outside_the_batch_injects_nothing():
    recorder = _RecordingPolicy(UNBATCHED_ACTION_SHAPE)
    adapter = PolicyClientAdapter(recorder, device="cpu", batched=False)
    adapter.set_policy_input_trace("/tmp/trace", env_idx=99)
    adapter.act(_torch_obs())
    assert recorder.traced == [None] * FIXTURE_BATCH


# --------------------------------------------------------------------------
# The hop through ObsTransformPolicy
# --------------------------------------------------------------------------


def test_obs_transform_policy_forwards_the_trace_across_the_transform():
    """The transform builds its dict from scratch, so the key must be re-added.

    This is the hop most likely to be lost by accident: nothing in the transform
    references the key, so it survives only because ``ObsTransformPolicy.infer``
    copies it across -- and the loss looks exactly like "the trace was never
    requested".
    """
    recorder = _RecordingPolicy()
    policy = ObsTransformPolicy(recorder, _transform())
    obs = build_obs()
    obs[TRACE_KEY] = {"dir": "/tmp/trace", "env_idx": 1}

    policy.infer(obs)

    assert recorder.requests[0][TRACE_KEY] == {"dir": "/tmp/trace", "env_idx": 1}
    # ... and the transform's own output is otherwise unchanged.
    assert "agent_pos" in recorder.requests[0]
    assert "joint_pos" not in recorder.requests[0]


def test_obs_transform_policy_adds_nothing_when_no_trace_was_requested():
    recorder = _RecordingPolicy()
    ObsTransformPolicy(recorder, _transform()).infer(build_obs())
    assert TRACE_KEY not in recorder.requests[0]


# --------------------------------------------------------------------------
# The whole client stack
# --------------------------------------------------------------------------


def test_only_chunk_boundaries_reach_the_server():
    """``PolicyClientAdapter -> ActionChunkBroker -> ObsTransformPolicy``.

    ``set_policy_input_trace``'s docstring claims the traced set is exactly the
    handovers, which is a claim about the broker, not about the trace: with a
    horizon of 3, steps 0 and 3 hit the server and steps 1, 2, 4, 5 are served
    from the cached chunk. Pinning it is what makes "one PNG per handover" a
    checked statement rather than a comment.
    """
    recorder = _RecordingPolicy()
    stack = ObsTransformPolicy(recorder, _transform())
    stack = ActionChunkBroker(stack, action_horizon=ACTION_HORIZON)
    adapter = PolicyClientAdapter(stack, device="cpu")
    adapter.set_policy_input_trace("/tmp/trace", env_idx=0)

    obs = _torch_obs()
    for _ in range(2 * ACTION_HORIZON):
        adapter.act(obs)

    assert len(recorder.requests) == 2
    assert recorder.traced == [{"dir": "/tmp/trace", "env_idx": 0}] * 2


def test_reset_rephases_the_chunk_boundaries():
    """Any env finishing an episode resets the broker, so boundaries move.

    Which is why ``needs_inference`` is asked rather than ``step %
    action_horizon`` recomputed: after a reset the two disagree, and a diagnostic
    keyed to the arithmetic would mark the wrong steps.
    """
    recorder = _RecordingPolicy()
    broker = ActionChunkBroker(
        ObsTransformPolicy(recorder, _transform()), action_horizon=ACTION_HORIZON
    )
    adapter = PolicyClientAdapter(broker, device="cpu")
    obs = _torch_obs()

    assert broker.needs_inference is True
    adapter.act(obs)
    assert broker.needs_inference is False
    adapter.reset()
    assert broker.needs_inference is True
    adapter.act(obs)
    assert len(recorder.requests) == 2
    assert recorder.resets == 1


def test_action_postprocess_feeds_pd_mode_into_the_next_agent_pos():
    """The pd_mode column is closed-loop: the last action decides the next state.

    ``(T, B, 9)`` from the batched server. Per-env values, because a scalar would
    let a bug that collapsed the batch pass.
    """
    transform = _transform()
    actions = np.zeros((ACTION_HORIZON, FIXTURE_BATCH, 9), dtype=np.float32)
    actions[0, 0, 8] = 1.0
    actions[0, 1, 8] = 0.0

    transform.action_postprocess({"actions": actions})
    agent_pos = transform(build_obs())["agent_pos"]
    assert agent_pos[0, 0, 18] == pytest.approx(1.0)
    assert agent_pos[1, 0, 18] == pytest.approx(0.0)


def test_action_postprocess_handles_the_unbatched_hlll_response():
    """``(T, 9)``: the HLLL adapter drops the singleton batch axis.

    So the leading axis is time, not batch -- reading it as batch would take
    timestep 0 as env 0 and then broadcast it, which for ``B == 1`` looks right
    and for anything else is wrong. The obs is ``B == 1`` so this actually
    checks the time axis rather than a second env's default pd_mode.
    """
    transform = _transform()
    actions = np.zeros((ACTION_HORIZON, 9), dtype=np.float32)
    actions[0, 8] = 0.0
    actions[1, 8] = 1.0  # a later timestep, which must not be read

    transform.action_postprocess({"actions": actions})
    agent_pos = transform(build_obs(batch=1))["agent_pos"]
    np.testing.assert_allclose(agent_pos[:, 0, 18], 0.0)


def test_pd_mode_column_unknown_env_reads_the_default():
    """An env with no dict entry uses ``_pd_mode_default``, not a raise.

    After a 2-env postprocess, env 0/1 are in the dict at 0.0. A larger batch
    includes env 2, which has never been written -- that slot is the default.
    """
    transform = _transform()
    actions = np.zeros((ACTION_HORIZON, FIXTURE_BATCH, 9), dtype=np.float32)
    actions[0, :, 8] = 0.0
    transform.action_postprocess({"actions": actions})
    agent_pos = transform(build_obs(batch=FIXTURE_BATCH + 1))["agent_pos"]
    assert agent_pos[0, 0, 18] == pytest.approx(0.0)
    assert agent_pos[1, 0, 18] == pytest.approx(0.0)
    assert agent_pos[2, 0, 18] == pytest.approx(1.0)


def test_reset_restores_the_scalar_pd_mode_default():
    transform = _transform()
    transform.action_postprocess(
        {"actions": np.zeros((ACTION_HORIZON, FIXTURE_BATCH, 9), dtype=np.float32)}
    )
    transform.reset()
    # Training episodes always start stiff, so the pre-first-action default is 1.0
    # and it broadcasts to any batch size.
    agent_pos = transform(build_obs(batch=FIXTURE_BATCH + 1))["agent_pos"]
    np.testing.assert_allclose(agent_pos[:, 0, 18], 1.0)


# --------------------------------------------------------------------------
# The wire
# --------------------------------------------------------------------------


def _roundtrip(payload: dict[str, Any]) -> dict[str, Any]:
    return msgpack_numpy.unpackb(msgpack_numpy.packb(payload))


def test_msgpack_roundtrip_preserves_every_array_leaf():
    """dtype, shape and C-contiguity, per leaf.

    C-contiguity is not pedantry: the codec reconstructs arrays from a raw buffer,
    and the model's first op on a non-contiguous view would copy silently or, for
    a wrong-strided view, read the wrong pixels.
    """
    wire = _transform()(build_obs())
    wire["target_heatmap"] = build_target_heatmap()
    got = _roundtrip(wire)

    assert sorted(got) == sorted(wire)
    for key, value in wire.items():
        if not isinstance(value, np.ndarray):
            continue
        out = got[key]
        assert out.dtype == value.dtype, key
        assert out.shape == value.shape, key
        assert out.flags["C_CONTIGUOUS"], key
        np.testing.assert_array_equal(out, value, err_msg=key)


def test_msgpack_roundtrip_preserves_list_str_task_name():
    """``task_name`` is the one input the server type-checks rather than reshapes.

    A codec that turned it into a numpy string array would trip the adapter's
    conditioning-key ``TypeError`` on every request.
    """
    wire = _transform()(build_obs())
    got = _roundtrip(wire)
    assert got["task_name"] == ["pick up the red block"] * FIXTURE_BATCH
    assert all(isinstance(v, str) for v in got["task_name"])


def test_msgpack_roundtrip_preserves_the_nested_trace_request():
    """The trace payload is a nested dict of a str and an int, not an array.

    It is also the only key on the request the server *pops* instead of
    validating, so a codec change that stringified the int would surface as an
    ``env_idx`` slice failure inside the renderer.
    """
    wire = _transform()(build_obs())
    wire[TRACE_KEY] = {"dir": "/tmp/trace", "env_idx": 1}
    got = _roundtrip(wire)
    assert got[TRACE_KEY] == {"dir": "/tmp/trace", "env_idx": 1}
    assert isinstance(got[TRACE_KEY]["env_idx"], int)


def test_msgpack_roundtrip_preserves_the_hl_stamps():
    """The trace's row keys travel as arrays and must keep their dtypes.

    ``hl_episode_id`` is int64 and names the panel; a float round-trip would name
    it ``ep0002`` by luck and something else after a rounding change.
    """
    wire = _transform()(build_obs())
    wire["hl_sim_time_ms"] = np.full(FIXTURE_BATCH, 1234.5, dtype=np.float32)
    wire["hl_episode_id"] = np.full(FIXTURE_BATCH, 2, dtype=np.int64)
    wire["hl_env_index"] = np.arange(FIXTURE_BATCH, dtype=np.int64)
    got = _roundtrip(wire)
    assert got["hl_sim_time_ms"].dtype == np.float32
    assert got["hl_episode_id"].dtype == np.int64
    assert got["hl_env_index"].dtype == np.int64
    np.testing.assert_array_equal(got["hl_episode_id"], 2)
    np.testing.assert_array_equal(got["hl_env_index"], np.arange(FIXTURE_BATCH))


# --------------------------------------------------------------------------
# Chunk broker: holding envs during a deferred PD-switch settle
# --------------------------------------------------------------------------


class _FrozenActionChunkBroker:
    """``ActionChunkBroker`` as it was before ``hold_envs``, kept as the reference.

    The hold support restructured ``_dispense`` around per-env rows; every
    batched eval, the 1-env path included, runs through it with no holds, so
    with no holds the rewrite must dispense exactly what this did.
    """

    def __init__(self, policy: BasePolicy, action_horizon: int) -> None:
        self._policy = policy
        self._action_horizon = action_horizon
        self._chunks: list[dict[str, Any] | None] = []
        self._cur_step: list[int] = []
        self._shared: dict[str, Any] = {}

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        if not self._chunks:
            self._absorb(self._policy.infer(obs), None)
        else:
            stale = [i for i, chunk in enumerate(self._chunks) if chunk is None]
            if stale:
                self._absorb(self._policy.infer(obs), stale)
        return self._dispense()

    def _absorb(self, results: dict[str, Any], env_ids) -> None:
        per_env, shared, num_envs = _split_envs(results)
        self._shared = shared
        if not self._chunks:
            self._chunks = [None] * num_envs
            self._cur_step = [0] * num_envs
            env_ids = range(num_envs)
        for i in env_ids:
            self._chunks[i] = per_env[i]
            self._cur_step[i] = 0

    def _dispense(self) -> dict[str, Any]:
        chunked_keys = list(self._chunks[0])
        out: dict[str, Any] = {
            k: np.stack(
                [self._chunks[i][k][self._cur_step[i]] for i in range(len(self._chunks))],
                axis=0,
            )
            for k in chunked_keys
        }
        for i, chunk in enumerate(self._chunks):
            self._cur_step[i] += 1
            chunk_len = self._chunk_length(chunk)
            if self._cur_step[i] >= self._action_horizon or self._cur_step[i] >= chunk_len:
                self._chunks[i] = None
        out.update(self._shared)
        return out

    @staticmethod
    def _chunk_length(d: dict[str, Any]) -> int:
        for v in d.values():
            if isinstance(v, np.ndarray) and v.ndim >= 1:
                return v.shape[0]
            if isinstance(v, dict):
                n = _FrozenActionChunkBroker._chunk_length(v)
                if n > 0:
                    return n
        return 0

    @property
    def needs_inference(self) -> bool:
        return (not self._chunks) or any(chunk is None for chunk in self._chunks)

    def reset(self) -> None:
        self._policy.reset()
        self._chunks = []
        self._cur_step = []
        self._shared = {}

    def reset_envs(self, env_ids) -> None:
        for i in env_ids:
            if 0 <= i < len(self._chunks):
                self._chunks[i] = None
                self._cur_step[i] = 0


class _ScriptedChunkPolicy(BasePolicy):
    """Answers call ``n`` with values that name the call, timestep and env.

    The response carries every leaf shape the broker distinguishes: the
    ``(T, B, D)`` action chunk, wider per-env aux, a ``(T, B)`` leaf whose
    per-env rows are scalars, and a ``(B, D')`` leaf passed through as shared.
    """

    def __init__(self, num_envs: int, chunk_len: int, *, batched: bool = True) -> None:
        self._num_envs = num_envs
        self._chunk_len = chunk_len
        self._batched = batched
        self.call_steps: list[int] = []
        self.resets = 0

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        n = len(self.call_steps)
        self.call_steps.append(int(obs["step"]))
        t = np.arange(self._chunk_len, dtype=np.float32)
        if not self._batched:
            actions = 1000.0 * n + 10.0 * t[:, None] + np.arange(3, dtype=np.float32) / 10
            return {"actions": actions.astype(np.float32)}
        b = np.arange(self._num_envs, dtype=np.float32)
        actions = (
            1000.0 * n
            + 10.0 * t[:, None, None]
            + b[None, :, None]
            + np.arange(3, dtype=np.float32)[None, None, :] / 10
        )
        shape = (self._chunk_len, self._num_envs)
        return {
            "actions": actions.astype(np.float32),
            "aux": np.full((*shape, 2, 2), n, dtype=np.int64),
            "flag": np.full(shape, n, dtype=np.int32),
            "state": np.full((self._num_envs, self._num_envs + 2), n, dtype=np.float64),
        }

    def reset(self) -> None:
        self.resets += 1


NUM_ENVS = 3


def _action_ids(out: dict[str, Any]) -> list[float]:
    """``1000 * call + 10 * t + env`` per env, read off action dim 0."""
    return np.round(out["actions"][:, 0], 3).tolist()


@pytest.mark.parametrize(
    "action_horizon, chunk_len",
    [(1, 4), (3, 5), (5, 5), (8, 5), (4, 16)],
)
def test_broker_without_holds_dispenses_exactly_what_it_did_before(action_horizon, chunk_len):
    """Every batched eval, the 1-env path included, runs the ``_dispense`` rewrite."""
    ref_policy = _ScriptedChunkPolicy(NUM_ENVS, chunk_len)
    new_policy = _ScriptedChunkPolicy(NUM_ENVS, chunk_len)
    ref = _FrozenActionChunkBroker(ref_policy, action_horizon)
    new = ActionChunkBroker(new_policy, action_horizon)
    rng = np.random.default_rng(action_horizon * 100 + chunk_len)

    for step in range(40):
        assert new.needs_inference == ref.needs_inference, step
        obs = {"step": step}
        got, want = new.infer(obs), ref.infer(obs)
        assert sorted(got) == sorted(want), step
        for k in want:
            assert got[k].dtype == want[k].dtype, (step, k)
            np.testing.assert_array_equal(got[k], want[k], err_msg=f"step {step} key {k}")
        draw = rng.random()
        if draw < 0.2:
            envs = sorted(rng.choice(NUM_ENVS, size=rng.integers(1, NUM_ENVS + 1), replace=False))
            ref.reset_envs(envs)
            new.reset_envs(envs)
        elif draw < 0.25:
            ref.reset()
            new.reset()

    assert new_policy.call_steps == ref_policy.call_steps
    assert new_policy.resets == ref_policy.resets


def test_a_held_env_neither_advances_nor_triggers_inference():
    """And resumes at the chunk index it paused at; the other envs keep their phase."""
    policy = _ScriptedChunkPolicy(2, chunk_len=3)
    broker = ActionChunkBroker(policy, action_horizon=3)

    assert _action_ids(broker.infer({"step": 0})) == [0.0, 1.0]
    broker.hold_envs([0])
    assert _action_ids(broker.infer({"step": 1})) == [0.0, 11.0]
    assert _action_ids(broker.infer({"step": 2})) == [0.0, 21.0]
    # Env 1's chunk is exhausted: one call serves it and leaves held env 0 alone.
    assert _action_ids(broker.infer({"step": 3})) == [0.0, 1001.0]
    broker.hold_envs([])
    assert _action_ids(broker.infer({"step": 4})) == [10.0, 1011.0]
    assert _action_ids(broker.infer({"step": 5})) == [20.0, 1021.0]
    assert policy.call_steps == [0, 3]


def test_a_held_env_with_an_exhausted_chunk_does_not_reinfer():
    policy = _ScriptedChunkPolicy(2, chunk_len=2)
    broker = ActionChunkBroker(policy, action_horizon=2)
    broker.infer({"step": 0})
    broker.infer({"step": 1})  # both chunks exhausted

    broker.hold_envs([0, 1])
    assert broker.needs_inference is False
    assert _action_ids(broker.infer({"step": 2})) == [10.0, 11.0]
    assert policy.call_steps == [0]

    broker.hold_envs([0])
    assert broker.needs_inference is True
    assert _action_ids(broker.infer({"step": 3})) == [10.0, 1001.0]
    assert policy.call_steps == [0, 3]

    broker.hold_envs([])
    assert broker.needs_inference is True
    assert _action_ids(broker.infer({"step": 4})) == [2000.0, 1011.0]


def test_held_rows_keep_every_chunked_key():
    policy = _ScriptedChunkPolicy(2, chunk_len=3)
    broker = ActionChunkBroker(policy, action_horizon=3)
    first = broker.infer({"step": 0})
    broker.hold_envs([0])
    held = broker.infer({"step": 1})
    assert sorted(held) == sorted(first)
    for k in ("actions", "aux", "flag"):
        np.testing.assert_array_equal(held[k][0], first[k][0])
        assert held[k].dtype == first[k].dtype


def test_reset_envs_clears_the_hold():
    policy = _ScriptedChunkPolicy(2, chunk_len=4)
    broker = ActionChunkBroker(policy, action_horizon=4)
    broker.infer({"step": 0})
    broker.hold_envs([0])
    broker.reset_envs([0])

    assert broker.needs_inference is True
    assert _action_ids(broker.infer({"step": 1})) == [1000.0, 11.0]


def test_reset_clears_the_hold():
    policy = _ScriptedChunkPolicy(2, chunk_len=4)
    broker = ActionChunkBroker(policy, action_horizon=4)
    broker.infer({"step": 0})
    broker.hold_envs([0, 1])
    broker.reset()
    assert _action_ids(broker.infer({"step": 1})) == [1000.0, 1001.0]
    assert _action_ids(broker.infer({"step": 2})) == [1010.0, 1011.0]


def test_holding_an_env_that_has_nothing_to_hand_back_raises():
    policy = _ScriptedChunkPolicy(2, chunk_len=2)
    broker = ActionChunkBroker(policy, action_horizon=2)
    broker.infer({"step": 0})
    broker.reset_envs([0])
    broker.hold_envs([0])
    with pytest.raises(RuntimeError, match="held with no chunk"):
        broker.infer({"step": 1})


def _per_env_ids(broker: PerEnvActionChunkBroker, env: int, step: int) -> float:
    return round(float(broker.infer_for_env(env, {"step": step})["actions"][0]), 3)


def test_per_env_broker_holds_the_same_way():
    policy = _ScriptedChunkPolicy(2, chunk_len=2, batched=False)
    broker = PerEnvActionChunkBroker(policy, action_horizon=2)

    assert _per_env_ids(broker, 0, 0) == 0.0  # call 0
    assert _per_env_ids(broker, 1, 0) == 1000.0  # call 1
    assert _per_env_ids(broker, 0, 1) == 10.0  # env 0 exhausted
    broker.hold_envs([0])
    assert broker.needs_inference_for_env(0) is False
    assert _per_env_ids(broker, 0, 2) == 10.0
    assert _per_env_ids(broker, 1, 2) == 1010.0
    assert policy.call_steps == [0, 0]

    broker.hold_envs([])
    assert broker.needs_inference_for_env(0) is True
    assert _per_env_ids(broker, 0, 3) == 2000.0
    assert policy.call_steps == [0, 0, 3]


def test_per_env_broker_resumes_mid_chunk():
    policy = _ScriptedChunkPolicy(1, chunk_len=4, batched=False)
    broker = PerEnvActionChunkBroker(policy, action_horizon=4)
    assert _per_env_ids(broker, 0, 0) == 0.0
    broker.hold_envs([0])
    assert _per_env_ids(broker, 0, 1) == 0.0
    broker.hold_envs([])
    assert _per_env_ids(broker, 0, 2) == 10.0
    assert policy.call_steps == [0]


def test_per_env_broker_reset_envs_clears_the_hold():
    policy = _ScriptedChunkPolicy(1, chunk_len=4, batched=False)
    broker = PerEnvActionChunkBroker(policy, action_horizon=4)
    broker.infer_for_env(0, {"step": 0})
    broker.hold_envs([0])
    broker.reset_envs([0])
    assert broker.needs_inference_for_env(0) is True
    assert _per_env_ids(broker, 0, 1) == 1000.0
