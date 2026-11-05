"""CPU tests for the model-config -> state-layout -> obs-validation contract.

The chain under test, which nothing else covers without a live GPU eval:

    checkpoint config.yaml
      -> derive_model_input_spec  (shape_meta, obs_shapes, conditioning_keys,
                                   state_dim, language_mode)
      -> server advertises state_dim as metadata
      -> eval client builds ManiflowBlocksTableObsTransform(state_dim=...)
      -> transform(obs) produces agent_pos of that width
      -> select_and_validate_obs checks the width back against obs_shapes

Every link is silent when it breaks: an 8-dim checkpoint fed a 19-dim state gets
a plausible-looking but wrong slice, and ``_normalize_state`` indexes by fixed
position so a reordered state normalizes the wrong columns. All three supported
state widths (8, 9, 19) are covered end to end, plus the normalization and
action-postprocess paths that read the same layout.

Runs without Isaac Lab or hvla (see the note in ``maniflow_serving_utils``).

The config / stats fixture loaders and the normalizer host live in
``policy_input_fixture.py`` so this file and the golden tests share one copy.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from blocksuite.serving.maniflow_serving_utils import (
    derive_model_input_spec,
    postprocess_actions,
    select_and_validate_obs,
    stats_from_train_data_stats,
)
from blocksuite.serving.obs_transforms.maniflow import ManiflowObsTransform
from blocksuite.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)
from tests.serving.policy_input_fixture import (
    _Normalizer,
    load_fixture,
    load_stats,
    to_torch,
)

# Camera resolution the dummy obs are rendered at: deliberately NOT 224 so the
# transform's resize is actually exercised on the way to the model's input grid.
DUMMY_CAM_HW = (180, 320)


def dummy_obs(
    batch: int = 2,
    gripper_joints: int = 1,
) -> dict[str, Any]:
    """A blocksuite-shaped observation with distinguishable values per block.

    Each state block gets its own decade (arm 1-7, joint_vel 11-17, ee 21-23,
    gripper 100) so a concat-order mistake shows up as a wrong number rather than
    a wrong shape -- ``_normalize_state`` slices by fixed index, so a reordered
    state is silent.
    """
    arm = np.arange(1.0, 8.0, dtype=np.float32)
    gripper = np.full(gripper_joints, 100.0, dtype=np.float32)
    joint_pos = np.tile(np.concatenate([arm, gripper]), (batch, 1))
    rgb = np.zeros((batch, *DUMMY_CAM_HW, 3), dtype=np.uint8)
    obs: dict[str, Any] = {
        "agent": {
            "joint_pos": joint_pos,
            "joint_vel": np.tile(np.arange(11.0, 18.0, dtype=np.float32), (batch, 1)),
            "ee_pos": np.tile(np.array([21.0, 22.0, 23.0], dtype=np.float32), (batch, 1)),
        },
        "images": {
            "front_camera": {"rgb": rgb},
            "gripper_camera": {"rgb": rgb.copy()},
        },
    }
    return obs


# --------------------------------------------------------------------------
# config -> input spec
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state_dim", [8, 9, 19])
def test_derive_model_input_spec_state_dim(state_dim):
    cfg, fw = load_fixture(state_dim)
    spec = derive_model_input_spec(cfg, fw)
    assert spec["state_dim"] == state_dim
    assert spec["obs_shapes"]["agent_pos"] == (state_dim,)
    assert spec["obs_shapes"]["front_rgb"] == (3, 224, 224)


def test_include_ptmap_false_strips_ptmaps_and_use_ptmap():
    cfg, fw = load_fixture(8)
    # Pinned so the fixture cannot drift into having nothing to strip, which
    # would leave the assertions below passing over an empty set.
    assert [k for k in fw.shape_meta.obs if k.endswith("_ptmap")] == [
        "front_ptmap",
        "gripper_ptmap",
    ]
    assert fw.use_ptmap is True

    spec = derive_model_input_spec(cfg, fw)
    assert not [k for k in spec["obs_shapes"] if k.endswith("_ptmap")]
    # fw is mutated in place: the caller builds the policy from the same object,
    # so the two must not disagree about whether ptmaps exist.
    assert not [k for k in fw.shape_meta.obs if k.endswith("_ptmap")]
    assert fw.use_ptmap is False
    # The returned shape_meta is what the caller hands to the policy ctor, so a
    # ptmap left in it builds a backbone the served weights have no tensors for.
    assert not [k for k in spec["shape_meta"]["obs"] if k.endswith("_ptmap")]


def test_ptmaps_kept_when_include_ptmap_unset():
    cfg, fw = load_fixture(19)
    assert fw.use_ptmap is True
    spec = derive_model_input_spec(cfg, fw)
    assert spec["obs_shapes"]["front_ptmap"] == (3, 224, 224)
    assert spec["obs_shapes"]["gripper_ptmap"] == (3, 224, 224)
    # The mutation is the exception, not the rule: the caller builds the policy
    # from this same fw, so clearing use_ptmap here would drop the ptmap
    # backbone from a model whose weights expect it.
    assert fw.use_ptmap is True


@pytest.mark.parametrize(
    "state_dim,expected_keys",
    [
        (8, {"task_name"}),
        (9, {"task_name", "target_heatmap"}),
        (19, {"task_name", "target_heatmap"}),
    ],
)
def test_conditioning_keys_follow_model_flags(state_dim, expected_keys):
    """Conditioning keys are model inputs absent from shape_meta.obs.

    Filtering must never drop them, and the set is derived from the model flags
    rather than hardcoded, so it self-corrects for future conditioning changes.
    """
    cfg, fw = load_fixture(state_dim)
    # Pinned so the fixtures keep covering both sides of the target flag; with
    # them all on one side a hardcoded key set would pass every row.
    assert bool(fw.target_injection_mode) == ("target_heatmap" in expected_keys)
    assert derive_model_input_spec(cfg, fw)["conditioning_keys"] == frozenset(
        expected_keys
    )


def test_no_conditioning_keys_when_both_flags_are_off():
    """Every checked-in fixture is language-conditioned, so the empty case needs
    an explicit flip: a bug that hardcoded ``task_name`` would pass every test
    above."""
    cfg, fw = load_fixture(8)
    fw.language_conditioned = False
    assert derive_model_input_spec(cfg, fw)["conditioning_keys"] == frozenset()


def test_wrong_image_size_raises():
    cfg, fw = load_fixture(8)
    fw.shape_meta.obs.front_rgb.shape = [3, 256, 256]
    with pytest.raises(ValueError, match="MANIFLOW_IMAGE_SIZE"):
        derive_model_input_spec(cfg, fw)


def test_missing_agent_pos_raises():
    cfg, fw = load_fixture(8)
    del fw.shape_meta.obs["agent_pos"]
    with pytest.raises(ValueError, match="agent_pos"):
        derive_model_input_spec(cfg, fw)


# --------------------------------------------------------------------------
# The depth contract
#
# The dedicated fixture carries the complete GPU-backprojection contract. A
# separate mutation test retains coverage for legacy 3-channel declarations.
# --------------------------------------------------------------------------


def _declare(fw: DictConfig, key: str, shape: list[int]) -> None:
    """Add a ``shape_meta.obs`` entry the fixture does not carry."""
    OmegaConf.update(fw, f"shape_meta.obs.{key}", {"shape": shape}, force_add=True)


def test_declared_3_channel_depth_is_contracted_to_1():
    """Some training configs declare depth as 3 channels (a precomputed-pointmap
    leftover) while the encoder reshapes it to one, so the validation contract is
    fixed up and the policy's own shape_meta deliberately is not.

    Uses the 19-dim fixture because it is the only one with pointmaps left on:
    with them off the entry is stripped before the rewrite could run, and this
    test would pass over the wrong code path.
    """
    cfg, fw = load_fixture(19)
    _declare(fw, "front_depth", [3, 224, 224])
    spec = derive_model_input_spec(cfg, fw)
    assert spec["obs_shapes"]["front_depth"] == (1, 224, 224)
    assert spec["shape_meta"]["obs"]["front_depth"]["shape"] == [3, 224, 224]


@pytest.mark.parametrize("state_dim", [8, 9])
def test_depth_and_intrinsics_dropped_without_ptmaps(state_dim):
    """Depth and intrinsics exist only to feed the encoder's back-projection, so
    a checkpoint with pointmaps off reads neither -- otherwise the eval renders
    depth the model discards. Both fixtures reach ``use_ptmap`` False by a
    different route (dataset ``include_ptmap`` false vs the framework flag)."""
    cfg, fw = load_fixture(state_dim)
    _declare(fw, "front_depth", [1, 224, 224])
    _declare(fw, "camera_intrinsics", [1, 4])
    spec = derive_model_input_spec(cfg, fw)
    assert "front_depth" not in spec["obs_shapes"]
    assert "camera_intrinsics" not in spec["obs_shapes"]
    assert "front_rgb" in spec["obs_shapes"]


# Depth transform + ptmap masking tests live in tests/hvla_maniflow/test_model_config_depth.py

# --------------------------------------------------------------------------
# prev_front_rgb and language_mode
# --------------------------------------------------------------------------


def test_use_previous_front_frame_mirrors_front_rgb():
    """The flag adds ``prev_front_rgb`` to the model's rgb_keys without a
    shape_meta entry, so the contract has to complete it or the key is dropped as
    undeclared and the encoder KeyErrors on it."""
    cfg, fw = load_fixture(9)
    OmegaConf.update(fw, "use_previous_front_frame", True, force_add=True)
    spec = derive_model_input_spec(cfg, fw)
    assert spec["obs_shapes"]["prev_front_rgb"] == spec["obs_shapes"]["front_rgb"]


def test_use_previous_front_frame_without_front_rgb_raises():
    cfg, fw = load_fixture(9)
    OmegaConf.update(fw, "use_previous_front_frame", True, force_add=True)
    del fw.shape_meta.obs["front_rgb"]
    with pytest.raises(ValueError, match="prev_front_rgb has no layout"):
        derive_model_input_spec(cfg, fw)


def test_use_target_front_crop_mirrors_front_rgb_but_not_client_obs_keys():
    cfg, fw = load_fixture(9)
    OmegaConf.update(fw, "use_target_front_crop", 50, force_add=True)
    spec = derive_model_input_spec(cfg, fw)
    assert spec["obs_shapes"]["target_front_crop_rgb"] == spec["obs_shapes"]["front_rgb"]
    assert spec["use_target_front_crop"] == 50
    from blocksuite.serving.maniflow_serving_utils import client_advertised_obs_keys
    from blocksuite.serving.serve_maniflow_policy import _maniflow_server_metadata

    assert "target_front_crop_rgb" not in client_advertised_obs_keys(spec["obs_shapes"])
    meta = _maniflow_server_metadata(
        action_mode="relative",
        obs_shapes=spec["obs_shapes"],
        model_input_spec=spec,
        stats=None,
    )
    assert meta["use_target_front_crop"] == 50


@pytest.mark.parametrize(
    "state_dim,expected_mode",
    [(8, "action_only"), (9, "action_goal"), (19, "action_only")],
)
def test_language_mode_is_advertised(state_dim, expected_mode):
    """Whoever builds ``task_name`` formats it to this, so it travels with the
    spec instead of being guessed: an ``action_goal`` checkpoint fed a bare verb
    has no spatial goal at all."""
    assert derive_model_input_spec(*load_fixture(state_dim))["language_mode"] == (
        expected_mode
    )


def test_unknown_language_mode_raises():
    cfg, fw = load_fixture(9)
    fw.language_mode = "action_and_pixels"
    with pytest.raises(ValueError, match="language_mode"):
        derive_model_input_spec(cfg, fw)


def test_language_mode_unread_when_not_language_conditioned():
    """The raise is gated on the flag: a model that reads no text has no text
    contract to validate."""
    cfg, fw = load_fixture(9)
    fw.language_conditioned = False
    fw.language_mode = "action_and_pixels"
    assert "language_mode" not in derive_model_input_spec(cfg, fw)


# --------------------------------------------------------------------------
# The end-to-end contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize("state_dim", [8, 9, 19])
def test_config_to_transform_to_validation_roundtrip(state_dim):
    """A transform configured from the config's state_dim validates against it."""
    spec = derive_model_input_spec(*load_fixture(state_dim))
    transform = ManiflowBlocksTableObsTransform(
        prompt="do something",
        state_dim=spec["state_dim"],
        image_interpolate_device="cpu",
    )
    torch_obs = to_torch(transform(dummy_obs()))
    filtered, dropped = select_and_validate_obs(
        torch_obs, spec["obs_shapes"], spec["conditioning_keys"]
    )
    assert filtered["agent_pos"].shape[-1] == state_dim
    assert "task_name" in filtered
    # The client always sends zeroed ptmaps; only the 19-dim fixture's contract
    # keeps them (the 8-dim one has them stripped, the 9-dim one declares none).
    assert ("front_ptmap" in dropped) == (state_dim != 19)


@pytest.mark.parametrize(
    "state_dim,expected_extended,expected_pd_mode,expected_width",
    [
        (8, False, False, 8),
        (9, False, True, 9),
        (19, True, True, 19),
        (None, True, True, 19),
    ],
)
def test_state_dim_configures_include_flags(
    state_dim, expected_extended, expected_pd_mode, expected_width
):
    """pd_mode is what separates the 8- and 9-dim layouts, so it is asserted
    apart from the two extended-state blocks rather than sharing a column."""
    transform = ManiflowBlocksTableObsTransform(
        state_dim=state_dim, image_interpolate_device="cpu"
    )
    assert transform.include_joint_vel is expected_extended
    assert transform.include_ee_pos is expected_extended
    assert transform.include_pd_mode is expected_pd_mode
    assert transform(dummy_obs())["agent_pos"].shape[-1] == expected_width


def test_unsupported_state_dim_raises():
    with pytest.raises(ValueError, match="state_dim must be one of"):
        ManiflowBlocksTableObsTransform(state_dim=12)


@pytest.mark.parametrize("state_dim", [9, 19])
def test_state_dim_without_matching_flags_raises(state_dim):
    """The base transform defaults every optional block off, so a width needing
    one must raise here, naming the missing block, rather than assemble a
    narrower state that only fails at the first inference as a shape mismatch."""
    with pytest.raises(ValueError, match=f"state_dim={state_dim} requires"):
        ManiflowObsTransform(state_dim=state_dim)


def test_state_dim_8_needs_no_optional_blocks():
    """8-dim is the base transform's default layout: arm + gripper, no pd_mode."""
    transform = ManiflowObsTransform(
        external_cam="front_camera",
        wrist_cam="gripper_camera",
        state_dim=8,
        image_interpolate_device="cpu",
    )
    assert transform(dummy_obs())["agent_pos"].shape[-1] == 8


def test_19dim_concat_order():
    """[7 arm, 7 joint_vel, 3 ee_xyz, 1 gripper, 1 pd_mode].

    Order is not observable from shapes and ``_normalize_state`` slices by fixed
    index, so a swap silently normalizes the wrong columns.
    """
    transform = ManiflowBlocksTableObsTransform(
        state_dim=19, image_interpolate_device="cpu"
    )
    agent_pos = transform(dummy_obs(batch=1))["agent_pos"][0, 0]
    np.testing.assert_allclose(agent_pos[:7], np.arange(1.0, 8.0))
    np.testing.assert_allclose(agent_pos[7:14], np.arange(11.0, 18.0))
    np.testing.assert_allclose(agent_pos[14:17], [21.0, 22.0, 23.0])
    assert agent_pos[17] == 100.0
    # pd_mode starts stiff; training episodes always start there.
    assert agent_pos[18] == 1.0


def test_9dim_concat_order():
    """[7 arm, 1 gripper, 1 pd_mode] -- pd_mode last, as in the 19-dim layout.

    Same silence as above, and here a swap of the two tail scalars would also
    survive ``_normalize_state``, which passes both through raw.
    """
    transform = ManiflowBlocksTableObsTransform(
        state_dim=9, image_interpolate_device="cpu"
    )
    agent_pos = transform(dummy_obs(batch=1))["agent_pos"][0, 0]
    np.testing.assert_allclose(agent_pos[:7], np.arange(1.0, 8.0))
    assert agent_pos[7] == 100.0
    assert agent_pos[8] == 1.0


def test_pd_mode_is_per_env_via_hl_env_index():
    """``set_pd_mode`` with a full-batch column, then unbatched ``B == 1`` reads."""
    transform = ManiflowBlocksTableObsTransform(
        state_dim=9, image_interpolate_device="cpu"
    )
    transform.set_pd_mode(np.arange(7, dtype=np.float32).reshape(7, 1))

    for env_idx in range(7):
        obs = dummy_obs(batch=1)
        obs["hl_env_index"] = np.array([env_idx], dtype=np.int64)
        agent_pos = transform.transform(obs)["agent_pos"][0, 0]
        assert agent_pos[8] == pytest.approx(float(env_idx))

    # PolicyInputDebug's batched call between unbatched inferences must stay valid.
    batched = transform.transform(
        {
            **dummy_obs(batch=7),
            "hl_env_index": np.arange(7, dtype=np.int64),
        }
    )
    np.testing.assert_allclose(batched["agent_pos"][:, 0, 8], np.arange(7, dtype=np.float32))

    actions = np.zeros((1, 7, 9), dtype=np.float32)
    for env_idx in range(7):
        actions[0, env_idx, 8] = float(env_idx) + 0.5
    transform.action_postprocess({"actions": actions})
    obs = dummy_obs(batch=1)
    obs["hl_env_index"] = np.array([3], dtype=np.int64)
    assert transform.transform(obs)["agent_pos"][0, 0, 8] == pytest.approx(3.5)


# --------------------------------------------------------------------------
# select_and_validate_obs
# --------------------------------------------------------------------------


def test_wrong_state_width_raises():
    spec = derive_model_input_spec(*load_fixture(19))
    transform = ManiflowBlocksTableObsTransform(
        state_dim=8, image_interpolate_device="cpu"
    )
    torch_obs = to_torch(transform(dummy_obs()))
    with pytest.raises(ValueError, match="agent_pos.*mismatch"):
        select_and_validate_obs(torch_obs, spec["obs_shapes"], spec["conditioning_keys"])


def test_wrong_image_shape_raises():
    """Images are shape-checked too, not just ``agent_pos``.

    The validator compares the trailing dims, so a transform that resized to the
    wrong grid -- or emitted HWC where the model wants CHW -- must be caught
    here rather than reaching ``predict_action`` as a plausible tensor.
    """
    spec = derive_model_input_spec(*load_fixture(8))
    transform = ManiflowBlocksTableObsTransform(
        state_dim=8, image_interpolate_device="cpu"
    )
    torch_obs = to_torch(transform(dummy_obs()))
    torch_obs["front_rgb"] = torch.zeros(2, 1, 3, 256, 256)
    with pytest.raises(ValueError, match=r"front_rgb.*trailing-shape mismatch"):
        select_and_validate_obs(torch_obs, spec["obs_shapes"], spec["conditioning_keys"])


def test_missing_required_key_raises():
    spec = derive_model_input_spec(*load_fixture(8))
    transform = ManiflowBlocksTableObsTransform(
        state_dim=8, image_interpolate_device="cpu"
    )
    torch_obs = to_torch(transform(dummy_obs()))
    del torch_obs["gripper_rgb"]
    with pytest.raises(ValueError, match="Missing required obs key"):
        select_and_validate_obs(torch_obs, spec["obs_shapes"], spec["conditioning_keys"])


def test_conditioning_keys_pass_through_unchecked():
    spec = derive_model_input_spec(*load_fixture(19))
    transform = ManiflowBlocksTableObsTransform(
        state_dim=19, image_interpolate_device="cpu"
    )
    torch_obs = to_torch(transform(dummy_obs()))
    # Neither is a tensor of the model's declared shapes, and neither may be
    # dropped or shape-checked.
    torch_obs["task_name"] = ["pick up the block"] * 2
    torch_obs["target_heatmap"] = torch.zeros(2, 1, 3, 224, 224)
    filtered, dropped = select_and_validate_obs(
        torch_obs, spec["obs_shapes"], spec["conditioning_keys"]
    )
    assert filtered["task_name"] == ["pick up the block"] * 2
    assert "target_heatmap" in filtered
    assert dropped == []


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------


def _train_data_stats() -> dict[str, Any]:
    """Complete short-state normalizer snapshot as stored in config.yaml."""
    return {
        "actions_min": [0.1, 0.2],
        "actions_denominator": [0.4, 0.5],
        "state_min": [0.0, 1.0],
        "state_denominator": [2.0, 3.0],
        "gripper_min": 0.01,
        "gripper_denominator": 0.03,
        "actions_gripper_min": 0.0,
        "actions_gripper_denominator": 1.0,
        "actions_pd_mode_min": 0.0,
        "actions_pd_mode_denominator": 1.0,
    }


def test_stats_from_train_data_stats():
    stats = stats_from_train_data_stats(_train_data_stats(), "relative")
    np.testing.assert_allclose(
        stats["action.relative_joint_target"]["q01"], [0.1, 0.2]
    )
    np.testing.assert_allclose(
        stats["action.relative_joint_target"]["q99"], [0.5, 0.7]
    )
    np.testing.assert_allclose(
        stats["observation.state.joint_position"]["q99"], [2.0, 4.0]
    )
    assert stats["observation.state.gripper_position"]["q99"] == pytest.approx(0.04)


@pytest.mark.parametrize("malformation", ["partial", "mismatched"])
def test_stats_from_train_data_stats_rejects_malformed_pairs(malformation: str):
    train_stats = _train_data_stats()
    if malformation == "partial":
        del train_stats["actions_denominator"]
        match = "without its pair"
    else:
        train_stats["state_denominator"] = [2.0]
        match = "shape"
    with pytest.raises(ValueError, match=match):
        stats_from_train_data_stats(train_stats, "absolute")


def test_normalize_19dim_state_against_short_stats_raises():
    """A 19-dim state normalized with short-layout stats would silently skip the
    joint_velocity / ee bands, so the mismatch has to raise."""
    normalizer = _Normalizer()
    normalizer._init_stats(load_stats(9), "absolute")
    with pytest.raises(ValueError, match="joint_velocity"):
        normalizer._normalize_state(np.zeros((1, 19), dtype=np.float32))


def test_normalize_19dim_state_passes_gripper_and_pd_mode_raw():
    stats = load_stats(19)
    normalizer = _Normalizer()
    normalizer._init_stats(stats, "relative")

    state = np.zeros((1, 19), dtype=np.float32)
    state[0, :7] = np.array(stats["observation.state.joint_position"]["q99"])
    state[0, 7:14] = np.array(stats["observation.state.joint_velocity"]["q01"])
    state[0, 14:17] = np.array(stats["observation.state.ee_position"]["q99"])
    state[0, 17] = 0.037
    state[0, 18] = 1.0

    out = normalizer._normalize_state(state)
    np.testing.assert_allclose(out[0, :7], 1.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 7:14], -1.0, atol=1e-6)
    np.testing.assert_allclose(out[0, 14:17], 1.0, atol=1e-6)
    assert out[0, 17] == pytest.approx(0.037)
    assert out[0, 18] == pytest.approx(1.0)


def test_normalize_8dim_state_passes_gripper_raw():
    stats = load_stats(8)
    normalizer = _Normalizer()
    normalizer._init_stats(stats, "absolute")

    state = np.zeros((1, 8), dtype=np.float32)
    state[0, :7] = np.array(stats["observation.state.joint_position"]["q01"])
    state[0, 7] = 0.021

    out = normalizer._normalize_state(state)
    np.testing.assert_allclose(out[0, :7], -1.0, atol=1e-6)
    assert out[0, 7] == pytest.approx(0.021)


def test_normalize_9dim_state_passes_gripper_and_pd_mode_raw():
    """The 9-dim tail is two raw scalars, not one: the same short-state code path
    as 8-dim, which is only correct because nothing after the arm is normalized."""
    stats = load_stats(9)
    normalizer = _Normalizer()
    normalizer._init_stats(stats, "absolute")

    state = np.zeros((1, 9), dtype=np.float32)
    state[0, :7] = np.array(stats["observation.state.joint_position"]["q99"])
    state[0, 7] = 0.021
    state[0, 8] = 1.0

    out = normalizer._normalize_state(state)
    assert out.shape == (1, 9)
    np.testing.assert_allclose(out[0, :7], 1.0, atol=1e-6)
    assert out[0, 7] == pytest.approx(0.021)
    assert out[0, 8] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# postprocess_actions
# --------------------------------------------------------------------------


def test_postprocess_relative_accumulates_onto_raw_state():
    # (B=1, T=3, D=9), constant +0.1 arm deltas.
    actions = np.zeros((1, 3, 9), dtype=np.float32)
    actions[..., :7] = 0.1
    raw_state = np.zeros((1, 1, 19), dtype=np.float32)
    raw_state[..., :7] = 5.0

    out = postprocess_actions(actions, "relative", raw_state)
    # Transposed to (T, B, D).
    assert out.shape == (3, 1, 9)
    np.testing.assert_allclose(out[0, 0, :7], 5.1, atol=1e-6)
    np.testing.assert_allclose(out[1, 0, :7], 5.2, atol=1e-6)
    np.testing.assert_allclose(out[2, 0, :7], 5.3, atol=1e-6)


def test_postprocess_absolute_leaves_arm_untouched():
    actions = np.zeros((1, 2, 9), dtype=np.float32)
    actions[..., :7] = 0.1
    raw_state = np.full((1, 1, 19), 5.0, dtype=np.float32)

    out = postprocess_actions(actions, "absolute", raw_state)
    np.testing.assert_allclose(out[..., :7], 0.1, atol=1e-6)


def test_postprocess_binarizes_gripper_and_rounds_pd_mode():
    actions = np.zeros((2, 9), dtype=np.float32)
    actions[0, 7], actions[0, 8] = 0.9, 0.6
    actions[1, 7], actions[1, 8] = 0.2, 0.4

    out = postprocess_actions(actions, "absolute", None)
    # BinaryJointAction convention: negative closes, positive opens.
    assert out[0, 7] == -1.0
    assert out[1, 7] == 1.0
    assert out[0, 8] == 1.0
    assert out[1, 8] == 0.0
    # Unbatched (T, D) keeps its axis order.
    assert out.shape == (2, 9)


