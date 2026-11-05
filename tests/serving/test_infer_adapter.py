"""CPU tests for ``ManiFlowInferAdapter.infer`` against a stub model.

``infer`` is ~140 lines of the real serving path -- pop the control keys,
numpy -> torch, normalize the state, log, select and validate, render a trace,
call the model, denormalize, postprocess -- and none of it needed weights to
exercise. A stub exposing ``eval()`` / ``to()`` / ``parameters()`` /
``predict_action`` covers all of it, and it can capture the exact ``torch_obs``
the model was handed, which is the thing worth asserting: that dict is what a
checkpoint sees in production, and every transformation on the way to it is
silent when wrong.

The two assertions that matter most and cannot be made anywhere else:

* the ``agent_pos`` reaching ``predict_action`` equals an **independently
  computed** numpy normalization of the fixture's raw state, so a change to
  ``_normalize_state`` cannot agree with itself;
* the adapter's advertised ``metadata`` is sufficient to rebuild the client's
  input layout, closing the negotiation loop the eval depends on -- the server
  publishes ``state_dim`` / ``obs_keys`` and the client derives its whole input
  layout from them, through the same
  ``transform_kwargs_from_server_metadata`` ``_make_policy_client`` calls, so
  drift on either side fails here.

Runs without hvla: ``serve_maniflow_policy`` imports it under ``TYPE_CHECKING``
only, because these adapters are plain numpy/torch over an already-built policy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

from blocksuite.serving.maniflow_serving_utils import (
    client_advertised_obs_keys,
    derive_model_input_spec,
    postprocess_actions,
    select_and_validate_obs,
)
from blocksuite.serving.molmo2_hl import HLContract, hl_contract_from_config, stack_label
from blocksuite.serving.obs_transforms.maniflow import (
    STATE_DIM_LAYOUTS,
    transform_kwargs_from_server_metadata,
)
from blocksuite.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)
from blocksuite.serving.serve_maniflow_policy import (
    HLLLInferAdapter,
    ManiFlowInferAdapter,
    _maybe_render_input_trace,
    hl_question_with_prev_action,
)
from tests.serving.policy_input_fixture import (
    FIXTURE_BATCH,
    build_obs,
    load_depth_fixture,
    load_fixture,
    load_stats,
    stats_action_mode,
)

ACTION_HORIZON = 4


class _StubPolicy:
    """The minimum surface ``ManiFlowInferAdapter`` touches on a real policy.

    ``parameters()`` yields one tensor because the adapter reads
    ``next(self._policy.parameters()).device`` to decide where to put the obs --
    a real model has thousands and this needs exactly one.
    """

    def __init__(self, action_dim: int = 9, horizon: int = ACTION_HORIZON) -> None:
        self._action_dim = action_dim
        self._horizon = horizon
        self._param = torch.zeros(1)
        self.eval_calls = 0
        self.to_calls: list[str] = []
        #: The dict handed to ``predict_action``, verbatim. The point of the stub.
        self.seen_obs: dict[str, Any] | None = None
        #: Raw normalized-space actions the adapter will denormalize.
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
    """The batched Molmo2 surface: ``images_for`` + one ``generate`` per call."""

    contract = HLContract(front_view_only=False, prompt_mode="stack_extract")

    def __init__(self) -> None:
        self.seen_langs: list[str] = []
        self.batch_sizes: list[int] = []
        self.seen_images: list[list[np.ndarray]] = []

    def images_for(self, front: Any, gripper: Any) -> list[np.ndarray]:
        return [np.asarray(front), np.asarray(gripper)]

    def generate(self, images: list[list[np.ndarray]], questions: list[str]) -> list[str]:
        self.batch_sizes.append(len(questions))
        self.seen_langs += list(questions)
        self.seen_images += list(images)
        return [f"pick <100, 200> for {q}" for q in questions]


def _hlll_adapter() -> tuple[HLLLInferAdapter, _StubPolicy, _StubHLPolicy, Any]:
    spec, transform = _spec_and_transform(19, hl_frames=True)
    ll_stub = _StubPolicy()
    hl_stub = _StubHLPolicy()
    ll_adapter = ManiFlowInferAdapter(
        policy=ll_stub,  # type: ignore[arg-type]
        stats=load_stats(19),
        action_mode=stats_action_mode(19),
        model_input_spec=spec,
    )
    adapter = HLLLInferAdapter(ll_adapter=ll_adapter, hl_policy=hl_stub)  # type: ignore[arg-type]
    return adapter, ll_stub, hl_stub, transform


def _hlll_request(transform: Any, env_idx: int, episode_id: int) -> dict[str, Any]:
    wire = transform(build_obs(batch=1))
    wire["hl_env_index"] = np.array([env_idx], dtype=np.int64)
    wire["hl_episode_id"] = np.array([episode_id], dtype=np.int64)
    wire["task_name"] = [f"goal for env {env_idx}"]
    return wire


def _spec_and_transform(state_dim: int, *, depth: bool = False, hl_frames: bool = False):
    """A model input spec and the client transform that feeds it."""
    cfg, fw = load_depth_fixture() if depth else load_fixture(state_dim)
    spec = derive_model_input_spec(cfg, fw)
    transform = ManiflowBlocksTableObsTransform(
        prompt="pick up the red block",
        state_dim=spec["state_dim"],
        depth_keys=sorted(k for k in spec["obs_shapes"] if k.endswith("_depth")),
        image_interpolate_device="cpu",
        include_hl_frames=hl_frames,
    )
    return spec, transform


def _adapter(
    state_dim: int, *, depth: bool = False, with_stats: bool = True
) -> tuple[ManiFlowInferAdapter, _StubPolicy, dict[str, Any], Any]:
    """``(adapter, stub, spec, transform)`` -- the whole server/client pair."""
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
    """What the client sends: the transform's wire dict plus control keys."""
    wire = transform(build_obs())
    wire.update(extra)
    return wire


# --------------------------------------------------------------------------
# Construction and metadata
# --------------------------------------------------------------------------


def test_adapter_puts_the_policy_in_eval_mode_on_a_device():
    _, stub, _, _ = _adapter(19)
    assert stub.eval_calls == 1
    assert stub.to_calls  # cuda when present, cpu otherwise -- either is fine


def test_metadata_advertises_the_full_client_contract():
    adapter, _, spec, _ = _adapter(19)
    meta = adapter.metadata
    assert meta["state_dim"] == 19
    assert meta["obs_keys"] == client_advertised_obs_keys(spec["obs_shapes"])
    assert meta["language_mode"] == spec["language_mode"]
    assert meta["action_mode"] == "relative"
    assert meta["has_action_stats"] is True
    assert meta["supports_batched_obs"] is True
    assert meta["policy_class"] == "_StubPolicy"
    # The bands the client's debug panel draws; omitting them silently turns the
    # state panel into a plot with no reference.
    assert set(meta["state_norm"]) == {
        "observation.state.joint_position",
        "observation.state.joint_velocity",
        "observation.state.ee_position",
        "observation.state.gripper_position",
    }


def test_metadata_works_without_stats():
    """Deliberate, per the comment at the adapter's stats binding.

    ``metadata`` is read on every client connect, so a stats-less adapter that
    raised there would fail the handshake rather than serve un-normalized actions
    -- which is a legitimate configuration.
    """
    adapter, _, _, _ = _adapter(19, with_stats=False)
    meta = adapter.metadata
    assert meta["has_action_stats"] is False
    assert meta["state_norm"] == {}
    assert meta["state_dim"] == 19


def test_transform_kwargs_carry_target_front_crop_bottom_pct():
    adapter, _, _, _ = _adapter(9)
    from blocksuite.serving.obs_transforms.maniflow import transform_kwargs_from_server_metadata

    meta = dict(adapter.metadata)
    meta["use_target_front_crop"] = 40
    kwargs = transform_kwargs_from_server_metadata(meta)
    assert kwargs["target_front_crop_bottom_pct"] == 40
    assert kwargs["enable_target_front_crop"] is True


@pytest.mark.parametrize(
    "state_dim,depth",
    [(8, False), (9, False), (19, False), (9, True)],
    ids=["state8", "state9", "state19", "state9_depth"],
)
def test_metadata_rederives_the_client_transform_layout(state_dim: int, depth: bool):
    """The negotiation loop: server metadata -> client transform -> valid obs.

    This is the link that is silent both ways. An 8-dim checkpoint fed a 19-dim
    state gets a plausible-looking wrong slice, and a depth checkpoint whose
    ``obs_keys`` stopped listing depth would have the eval quietly stop rendering
    it. Rebuilding the transform from nothing but ``metadata`` and then validating
    against the same server proves the advertisement is sufficient.

    ``transform_kwargs_from_server_metadata`` is the **live** derivation
    ``_make_policy_client`` calls, not a copy of it: that function additionally
    needs an eval ``args`` object, a prompt resolver and a queue URL, none of
    which this negotiation depends on.
    """
    adapter, _, spec, _ = _adapter(state_dim, depth=depth)

    kwargs = transform_kwargs_from_server_metadata(adapter.metadata)
    rebuilt = ManiflowBlocksTableObsTransform(
        prompt="pick up the red block", image_interpolate_device="cpu", **kwargs
    )
    assert rebuilt.depth_keys == sorted(
        k for k in spec["obs_shapes"] if k.endswith("_depth")
    )
    assert rebuilt.include_ptmap == any(
        k.endswith("_ptmap") for k in spec["obs_shapes"]
    )

    torch_obs = {
        k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
        for k, v in rebuilt(build_obs()).items()
    }
    filtered, dropped = select_and_validate_obs(
        torch_obs, spec["obs_shapes"], spec["conditioning_keys"]
    )
    assert filtered["agent_pos"].shape[-1] == state_dim
    # Negotiated kwargs mean the client sends nothing the model will not read.
    assert dropped == []


# --------------------------------------------------------------------------
# What reaches predict_action
# --------------------------------------------------------------------------


def _numpy_normalize(raw: np.ndarray, stats: dict, state_dim: int) -> np.ndarray:
    """An independent normalization of ``raw``, written from the stats directly.

    Deliberately not a call into ``_normalize_state``: comparing that function to
    itself would pass through any change to it. The layout encoded here is the
    same one ``test_policy_input_trace.py`` declares, so a block-order change has
    to break both files.
    """
    out = raw.astype(np.float64).copy()

    def band(key: str) -> tuple[np.ndarray, np.ndarray]:
        entry = stats[key]
        lo = np.atleast_1d(np.asarray(entry["q01"], dtype=np.float64))
        hi = np.atleast_1d(np.asarray(entry["q99"], dtype=np.float64))
        return lo, np.maximum(hi - lo, 1e-6)

    lo, den = band("observation.state.joint_position")
    out[..., :7] = 2.0 * (out[..., :7] - lo) / den - 1.0
    if state_dim == 19:
        lo, den = band("observation.state.joint_velocity")
        out[..., 7:14] = 2.0 * (out[..., 7:14] - lo) / den - 1.0
        lo, den = band("observation.state.ee_position")
        out[..., 14:17] = 2.0 * (out[..., 14:17] - lo[:3]) / den[:3] - 1.0
    return out


@pytest.mark.parametrize("state_dim", sorted(STATE_DIM_LAYOUTS))
def test_agent_pos_reaching_the_model_is_normalized(state_dim: int):
    adapter, stub, _, transform = _adapter(state_dim)
    request = _client_request(transform)
    raw = np.asarray(request["agent_pos"]).copy()

    adapter.infer(request)

    seen = stub.seen_obs["agent_pos"].numpy()
    expected = _numpy_normalize(raw, load_stats(state_dim), state_dim)
    np.testing.assert_allclose(seen, expected, atol=1e-5)
    # The tail (gripper, and pd_mode when present) is passed through, so a
    # normalizer that started scaling the gripper shows up here and nowhere else.
    tail = 7 if state_dim <= 9 else 17
    np.testing.assert_allclose(seen[..., tail:], raw[..., tail:], atol=1e-6)


def test_an_adapter_without_stats_forwards_the_raw_state():
    """No stats means no normalization, not a silently identity-normalized state."""
    adapter, stub, _, transform = _adapter(19, with_stats=False)
    request = _client_request(transform)
    raw = np.asarray(request["agent_pos"]).copy()
    adapter.infer(request)
    np.testing.assert_allclose(stub.seen_obs["agent_pos"].numpy(), raw, atol=0.0)


def test_control_keys_are_popped_and_never_reach_the_model():
    """HL bookkeeping and the trace request are the server's, not the model's.

    Left in, the ndarray branch would coerce them to tensors and hand them to
    ``predict_action``, where ``dict_apply`` indexes every leaf -- so this is the
    difference between a working trace and a crash inside the model.
    """
    adapter, stub, _, transform = _adapter(19)
    request = _client_request(
        transform,
        hl_sim_time_ms=np.full(FIXTURE_BATCH, 1234.0, dtype=np.float32),
        hl_episode_id=np.zeros(FIXTURE_BATCH, dtype=np.int64),
        __episode_seed__=np.array(7, dtype=np.int64),
    )
    adapter.infer(request)
    for key in ("hl_sim_time_ms", "hl_episode_id", "hl_env_index", "__episode_seed__", "__ll_input_trace__"):
        assert key not in stub.seen_obs


def test_undeclared_ptmaps_are_dropped_rather_than_fed():
    """The client always sends zeroed ptmaps; an 8-dim checkpoint declares none.

    Feeding them would build a pointmap branch the served weights have no tensors
    for, so the server -- not the client -- is authoritative here.
    """
    adapter, stub, spec, transform = _adapter(8)
    assert not [k for k in spec["obs_shapes"] if k.endswith("_ptmap")]
    request = _client_request(transform)
    assert "front_ptmap" in request

    adapter.infer(request)
    assert "front_ptmap" not in stub.seen_obs
    assert "gripper_ptmap" not in stub.seen_obs
    assert sorted(k for k in stub.seen_obs if k != "task_name") == sorted(spec["obs_shapes"])


def test_depth_and_intrinsics_reach_the_model_for_a_depth_checkpoint():
    adapter, stub, _, transform = _adapter(9, depth=True)
    adapter.infer(_client_request(transform))
    assert stub.seen_obs["front_depth"].shape == (FIXTURE_BATCH, 1, 1, 224, 224)
    assert stub.seen_obs["camera_intrinsics"].shape == (FIXTURE_BATCH, 1, 2, 4)


def test_task_name_is_forwarded_as_a_list_of_strings():
    adapter, stub, _, transform = _adapter(19)
    adapter.infer(_client_request(transform))
    assert stub.seen_obs["task_name"] == ["pick up the red block"] * FIXTURE_BATCH


def test_a_bare_string_conditioning_key_raises():
    """A legacy client's single ``str`` is ambiguous for ``B > 1``.

    Broadcasting the guess would run the model on one env's prompt for all of
    them and report the result as conditioned, so it raises instead.
    """
    adapter, _, _, transform = _adapter(19)
    request = _client_request(transform)
    request["task_name"] = "pick up the red block"
    with pytest.raises(TypeError, match="list\\[str\\] of length B"):
        adapter.infer(request)


# --------------------------------------------------------------------------
# What comes back
# --------------------------------------------------------------------------


def test_actions_are_denormalized_and_postprocessed():
    """``(T, B, D)`` with a binary gripper and an integral pd_mode.

    The transpose matters: ``ActionChunkBroker`` slices the leading axis as time,
    so a ``(B, T, D)`` response would hand env 1's chunk to timestep 1.
    """
    adapter, stub, _, transform = _adapter(19)
    action = torch.zeros(FIXTURE_BATCH, ACTION_HORIZON, 9)
    action[..., 7] = 0.9  # close
    action[0, :, 7] = 0.1  # env 0 opens
    action[..., 8] = 0.6  # stiff
    action[1, :, 8] = 0.2  # env 1 soft
    stub.next_action = action

    actions = adapter.infer(_client_request(transform))["actions"]
    assert actions.shape == (ACTION_HORIZON, FIXTURE_BATCH, 9)
    # BinaryJointAction convention: negative closes, positive opens.
    assert set(np.unique(actions[:, 0, 7]).tolist()) == {1.0}
    assert set(np.unique(actions[:, 1, 7]).tolist()) == {-1.0}
    assert set(np.unique(actions[..., 8]).tolist()) <= {0.0, 1.0}
    assert set(np.unique(actions[:, 0, 8]).tolist()) == {1.0}
    assert set(np.unique(actions[:, 1, 8]).tolist()) == {0.0}


def test_relative_actions_accumulate_onto_the_raw_state():
    """Cumsum is applied to the **raw** arm, not the normalized one.

    Accumulating onto the normalized state would produce joint targets in
    ``[-1, 1]`` -- physically plausible-looking numbers in the wrong units, which
    is precisely the failure that only shows up as a robot moving wrongly.
    """
    adapter, stub, _, transform = _adapter(19)
    request = _client_request(transform)
    raw_arm = np.asarray(request["agent_pos"])[:, :, :7].copy()

    # +1 in normalized action space maps to the band's q99 under _denormalize.
    stub.next_action = torch.ones(FIXTURE_BATCH, ACTION_HORIZON, 9)
    actions = adapter.infer(request)["actions"]

    step = np.asarray(load_stats(19)["action.relative_joint_target"]["q99"], dtype=np.float64)
    for t in range(ACTION_HORIZON):
        expected = raw_arm[:, 0, :] + step * (t + 1)
        np.testing.assert_allclose(actions[t, :, :7], expected, atol=1e-4)


def test_absolute_actions_leave_the_arm_alone():
    """No cumsum and no state term: +1 denormalizes to the band's q99, per env.

    The expectation is broadcast over the batch on purpose -- absolute mode does
    not read the state, so both envs land on the same target even though their
    raw arms differ, which is the contrast with the relative case above.
    """
    adapter, stub, _, transform = _adapter(8)
    stub.next_action = torch.ones(FIXTURE_BATCH, ACTION_HORIZON, 9)
    actions = adapter.infer(_client_request(transform))["actions"]
    q99 = np.asarray(load_stats(8)["action.joint_target"]["q99"], dtype=np.float64)
    expected = np.broadcast_to(q99, (FIXTURE_BATCH, q99.size))
    for t in range(ACTION_HORIZON):
        np.testing.assert_allclose(actions[t, :, :7], expected, atol=1e-4)


# --------------------------------------------------------------------------
# The trace hook
# --------------------------------------------------------------------------


def test_trace_request_renders_a_panel_through_infer(tmp_path):
    """End to end: a client trace request on the obs produces a server-side PNG.

    The full hop, not the renderer in isolation: ``infer`` has to pop the key,
    keep the stamps it also pops, and call the renderer *after*
    ``select_and_validate_obs`` so the panel shows the model's dict rather than
    the client's.
    """
    adapter, _, _, transform = _adapter(19)
    trace_dir = tmp_path / "ll_policy_input_trace"
    adapter.infer(
        _client_request(
            transform,
            hl_sim_time_ms=np.full(FIXTURE_BATCH, 500.0, dtype=np.float32),
            hl_episode_id=np.full(FIXTURE_BATCH, 2, dtype=np.int64),
            __ll_input_trace__={"dir": str(trace_dir), "env_idx": 1},
        )
    )
    assert [p.name for p in sorted(trace_dir.glob("ll_input_*.png"))] == [
        "ll_input_ep0002_t0000500ms.png"
    ]


def test_trace_without_stamps_raises():
    """A trace request from a client that is not the eval harness has no row keys.

    Guessing an index would produce panels that silently overwrite each other,
    which reads as "the eval made one handover".
    """
    with pytest.raises(ValueError, match="hl_sim_time_ms"):
        _maybe_render_input_trace(
            {"dir": "/tmp/does-not-matter", "env_idx": 0},
            {},
            sim_time_ms=None,
            episode_id=None,
            raw_state=None,
            dropped=[],
        )


def test_hlll_diverging_episode_ids_do_not_cross_clear():
    """Two envs with different ``hl_episode_id`` values must not wipe each other's HL cache."""
    adapter, _, _, transform = _hlll_adapter()
    adapter.infer(_hlll_request(transform, 0, 1))
    adapter.infer(_hlll_request(transform, 1, 5))
    assert adapter._last_lang[0]
    assert adapter._last_lang[1]


def test_hl_question_with_prev_action_matches_tamp():
    assert hl_question_with_prev_action("stack 2 red blocks", "") == "stack 2 red blocks"
    assert hl_question_with_prev_action("stack 2 red blocks", "pick <409, 210>") == (
        "Previous action: pick <409, 210>. stack 2 red blocks"
    )


def test_hlll_feeds_previous_raw_output_on_the_next_call():
    """Second infer of the same env must prepend the first call's raw output."""
    adapter, _, hl_stub, transform = _hlll_adapter()
    adapter.infer(_hlll_request(transform, 0, 1))
    adapter.infer(_hlll_request(transform, 0, 1))
    first_out = "pick <100, 200> for goal for env 0"
    assert hl_stub.seen_langs[0] == "goal for env 0"
    assert hl_stub.seen_langs[1] == hl_question_with_prev_action("goal for env 0", first_out)


def test_hlll_episode_change_clears_previous_action():
    adapter, _, hl_stub, transform = _hlll_adapter()
    adapter.infer(_hlll_request(transform, 0, 1))
    adapter.infer(_hlll_request(transform, 0, 2))
    assert hl_stub.seen_langs[1] == "goal for env 0"


def test_hlll_episode_change_clears_only_that_env():
    adapter, _, _, transform = _hlll_adapter()
    adapter.infer(_hlll_request(transform, 0, 1))
    adapter.infer(_hlll_request(transform, 1, 1))
    lang_env1 = adapter._last_lang[1]
    adapter.infer(_hlll_request(transform, 0, 2))
    assert adapter._last_lang[1] == lang_env1


def test_no_trace_request_renders_nothing(monkeypatch):
    """``None`` short-circuits before the stamp check and before the renderer.

    The ordering is the assertion: the stamps are missing here too, so the test
    above proves a non-None request raises at this point. Returning quietly means
    the ``None`` branch was taken first, and the stub proves nothing was drawn --
    which a check on an empty temp dir the renderer was never given cannot.
    """
    import blocksuite.serving.policy_input_trace as trace_module

    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        trace_module,
        "render_ll_input_panel",
        lambda *args, **kwargs: calls.append(kwargs),
    )
    _maybe_render_input_trace(
        None, {}, sim_time_ms=None, episode_id=None, raw_state=None, dropped=[]
    )
    assert calls == []


# --------------------------------------------------------------------------
# Batched HL + LL, the Molmo2 contract, and the play mode
# --------------------------------------------------------------------------


def test_hlll_batches_every_env_into_one_hl_and_one_ll_call():
    """B envs in one request: one Molmo2 generate over all B, one LL forward, (T, B, ...) leaves."""
    adapter, ll_stub, hl_stub, transform = _hlll_adapter()
    batch = 3

    def request() -> dict[str, Any]:
        wire = transform(build_obs(batch=batch))
        wire["hl_env_index"] = np.array([4, 7, 9], dtype=np.int64)
        wire["hl_episode_id"] = np.zeros(batch, dtype=np.int64)
        wire["hl_goal"] = ["Stack A B", "Stack C D", "Stack E F"]
        return wire

    out = adapter.infer(request())

    assert hl_stub.batch_sizes == [batch]
    assert hl_stub.seen_langs == ["Stack A B", "Stack C D", "Stack E F"]
    # The HL saw native frames, not the LL's 224x224 squares.
    assert hl_stub.seen_images[0][0].shape == wire_native_shape(transform)
    assert ll_stub.seen_obs is not None
    assert ll_stub.seen_obs["target_heatmap"].shape[:2] == (batch, 1)
    assert len(ll_stub.seen_obs["task_name"]) == batch
    steps = out["actions"].shape[0]
    assert out["actions"].shape == (steps, batch, 9)
    for key in ("hl_model_heatmap", "hl_model_text", "hl_done", "hl_prev_action", "hl_input_prompt"):
        assert out[key].shape[:2] == (steps, batch), key
    # The next call prepends each env's own previous output.
    adapter.infer(request())
    assert hl_stub.seen_langs[batch] == hl_question_with_prev_action(
        "Stack A B", "pick <100, 200> for Stack A B"
    )


def wire_native_shape(transform: Any) -> tuple[int, ...]:
    native = transform.get_image(build_obs(batch=1), transform.external_cam)
    return tuple(np.asarray(native).shape[1:3]) + (3,)


def test_hlll_unbatched_request_returns_a_per_env_chunk():
    """A 0-d hl_env_index (PolicyClientAdapter(batched=False)) gets (T, D) back."""
    adapter, _, _, transform = _hlll_adapter()
    wire = _hlll_request(transform, 2, 1)
    wire["hl_env_index"] = np.asarray(2, dtype=np.int64)
    out = adapter.infer(wire)
    assert out["actions"].ndim == 2
    assert out["hl_done"].ndim == 1


def test_hlll_without_native_frames_raises():
    adapter, _, _, _ = _hlll_adapter()
    _, plain_transform = _spec_and_transform(19)
    with pytest.raises(ValueError, match="hl_front_rgb"):
        adapter.infer(_hlll_request(plain_transform, 0, 1))


def test_hlll_metadata_asks_the_client_for_native_frames():
    adapter, _, _, _ = _hlll_adapter()
    meta = adapter.metadata
    assert meta["supports_batched_obs"] is True
    kwargs = transform_kwargs_from_server_metadata(meta)
    assert kwargs["include_hl_frames"] is True
    assert kwargs["hl_front_view_only"] is False
    assert meta["hl_prompt_mode"] == "stack_extract"


def test_stack_label_matches_the_hl_training_prompt():
    goal = """(and (on_table block_i workspace_table) (on block_t block_i)
        (on block_gray block_t) (clear block_gray) (on_table block_red2 workspace_table))"""
    assert stack_label(goal) == "Stack GRAY T I\nStack RED"
    contract = HLContract(prompt_mode="stack_extract")
    assert contract.goal_prompt(None, goal) == "Stack GRAY T I\nStack RED"
    assert contract.goal_prompt("put A on red", goal) == "put A on red"
    assert HLContract(prompt_mode="stack_ai2").goal_prompt(None, goal) == "Stack Ai2"


def test_hl_contract_is_read_from_either_training_config_layout():
    kwargs_layout = {"data": {"kwargs_mixture": [{"datasets": [{
        "dataset_name": "robot_train/blocks",
        "kwargs": {"front_view_only": True, "front_crop": None, "hl_prompt_mode": "stack_extract",
                   "image_size": [640, 360]},
    }]}]}}
    multi_layout = {"data": {"multi_level_mixture": [{"datasets": [{
        "dataset": {"_name": "dataset/RobotTrain", "front_view_only": True,
                    "front_crop": [30, 90, 610, 360], "hl_prompt_mode": "stack_extract"},
    }]}]}}
    a = hl_contract_from_config(kwargs_layout)
    assert (a.front_view_only, a.front_crop, a.image_size, a.prompt_mode) == (
        True, None, (640, 360), "stack_extract")
    b = hl_contract_from_config(multi_layout)
    assert b.front_crop == (30, 90, 610, 360)


def test_hl_contract_html_v2_question_matches_training():
    """The html-v2 question for each stack_point_prob / point_blocks / plan_prob combination, as
    molmo_flow ``RobotTrainDataset`` phrases it."""

    def contract(**kwargs: Any) -> HLContract:
        return hl_contract_from_config({"data": {"dataset_kwargs": {"target_format": "html-v2", **kwargs}}})

    tail = "point to target. predict action."
    cases = {
        (): f"stack Q Y. {tail}",
        (("plan_prob", 1.0),): f"stack Q Y. plan. {tail}",
        (("stack_point_prob", 1.0),): f"stack Q Y. point to stack blocks. {tail}",
        (("stack_point_prob", 1.0), ("point_blocks", "all")): f"stack Q Y. point to all blocks. {tail}",
        (("stack_point_prob", 1.0), ("point_blocks", "all"), ("plan_prob", 1.0)):
            f"stack Q Y. point to all blocks. plan. {tail}",
    }
    for kwargs, question in cases.items():
        assert contract(**dict(kwargs)).question("stack Q Y", "") == question
    assert contract().question("stack Q Y", "pick") == f"previous action: pick. stack Q Y. {tail}"


def test_play_mode_keeps_the_stop_signal():
    """pd_mode rounds into {0, 1}; a pause_on_attempt checkpoint's -1 must survive."""
    raw = np.zeros((2, 3, 9), dtype=np.float32)
    raw[..., 8] = [-0.9, 0.2, 0.8]
    pd = postprocess_actions(raw.copy(), "absolute", None, "pd_mode")
    play = postprocess_actions(raw.copy(), "absolute", None, "play_mode")
    assert sorted(set(pd[..., 8].ravel().tolist())) == [0.0, 1.0]
    assert sorted(set(play[..., 8].ravel().tolist())) == [-1.0, 0.0, 1.0]


def test_hlll_async_refresh_runs_the_hl_only_where_asked():
    """With ``hl_refresh`` the HL runs for rows that ask (or have no output yet); the rest reuse theirs."""
    adapter, ll_stub, hl_stub, transform = _hlll_adapter()

    def request(refresh: list[bool]) -> dict[str, Any]:
        wire = transform(build_obs(batch=2))
        wire["hl_env_index"] = np.array([0, 1], dtype=np.int64)
        wire["hl_episode_id"] = np.zeros(2, dtype=np.int64)
        wire["hl_goal"] = ["Stack A B", "Stack C D"]
        wire["hl_refresh"] = np.array(refresh)
        return wire

    adapter.infer(request([False, False]))  # first call of the episode: both run
    assert hl_stub.batch_sizes == [2]
    adapter.infer(request([False, False]))  # nothing asked: HL skipped, cached outputs reused
    assert hl_stub.batch_sizes == [2]
    assert ll_stub.seen_obs is not None and len(ll_stub.seen_obs["task_name"]) == 2
    adapter.infer(request([False, True]))  # env 1 finished a skill
    assert hl_stub.batch_sizes == [2, 1]
    assert hl_stub.seen_langs[-1].startswith("Previous action: ")


def test_hlll_malformed_hl_output_keeps_the_last_instruction():
    """A garbled generation repeats the env's last valid instruction instead of failing the eval."""
    adapter, ll_stub, hl_stub, transform = _hlll_adapter()
    adapter.infer(_hlll_request(transform, 0, 1))
    first_heatmap = np.asarray(ll_stub.seen_obs["target_heatmap"]).copy()  # type: ignore[index]
    hl_stub.generate = lambda images, questions: ["pick <25 “217, 194>"]  # type: ignore[method-assign]
    adapter.infer(_hlll_request(transform, 0, 1))
    np.testing.assert_array_equal(np.asarray(ll_stub.seen_obs["target_heatmap"]), first_heatmap)  # type: ignore[index]
