"""Tests for live EJBL depth augmentation in :class:`ManiflowObsTransform`."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from tests.serving.policy_input_fixture import build_obs
from toyblocks.serving.obs_transforms.maniflow import ManiflowObsTransform
from toyblocks.serving.obs_transforms.maniflow_blocks_table import (
    ManiflowBlocksTableObsTransform,
)

_EJBL_PATCH = (
    "hvla.training.trainer_utils.depth_aug.elastic_jbl_depth.add_ejbl_depth_noise"
)


def _ensure_hvla_depth_aug_stubs() -> None:
    """Minimal ``hvla`` tree so ``@patch`` can resolve without the submodule."""
    chain = [
        "hvla",
        "hvla.training",
        "hvla.training.trainer_utils",
        "hvla.training.trainer_utils.depth_aug",
        "hvla.training.trainer_utils.depth_aug.elastic_jbl_depth",
        "hvla.training.trainer_utils.depth_aug.disparity_common",
        "hvla.training.trainer_utils.depth_aug.elastic_disparity_depth",
        "hvla.training.trainer_utils.depth_bake_profiles",
    ]
    for name in chain:
        if name in sys.modules:
            continue
        mod = types.ModuleType(name)
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            parent = sys.modules[parent_name]
            setattr(parent, attr, mod)
        sys.modules[name] = mod

    disparity = sys.modules["hvla.training.trainer_utils.depth_aug.disparity_common"]
    if not hasattr(disparity, "baseline_for_camera"):
        disparity.baseline_for_camera = lambda _cam: 0.12

    elastic = sys.modules["hvla.training.trainer_utils.depth_aug.elastic_disparity_depth"]
    if not hasattr(elastic, "ELASTIC_DEPTH_AUG_PRESET"):
        elastic.ELASTIC_DEPTH_AUG_PRESET = {}

    profiles = sys.modules["hvla.training.trainer_utils.depth_bake_profiles"]
    if not hasattr(profiles, "resolve_profile"):
        profiles.resolve_profile = lambda _name: SimpleNamespace(
            elastic_preset=None,
            noise_kwargs={},
            random_ideal_mix=False,
            ideal_mix_min=0.0,
            ideal_mix_max=1.0,
        )

    ejbl = sys.modules["hvla.training.trainer_utils.depth_aug.elastic_jbl_depth"]
    if not hasattr(ejbl, "add_ejbl_depth_noise"):
        ejbl.add_ejbl_depth_noise = lambda depth_m, *_args, **_kwargs: depth_m


_ensure_hvla_depth_aug_stubs()


def _depth_transform(augment_depth: str | None, seed: int | None) -> ManiflowBlocksTableObsTransform:
    return ManiflowBlocksTableObsTransform(
        depth_keys=["front_depth", "gripper_depth"],
        include_ptmap=False,
        augment_depth=augment_depth,
        depth_aug_seed=seed,
    )


def _bump_valid_depth(depth_m: np.ndarray, *_args: object, **_kwargs: object) -> np.ndarray:
    out = np.array(depth_m, copy=True)
    valid = out > 0
    out[valid] = out[valid] + 0.05
    return out


@patch(_EJBL_PATCH, side_effect=_bump_valid_depth)
def test_ejbl_changes_model_depth(_mock_add) -> None:
    obs = build_obs(batch=1)
    clean = _depth_transform(None, None).transform(obs)
    noisy = _depth_transform("ejbl", 42).transform(obs)
    assert _mock_add.call_count >= 1
    for key in ("front_depth", "gripper_depth"):
        d_clean = np.asarray(clean[key])[0, 0, 0]
        d_noisy = np.asarray(noisy[key])[0, 0, 0]
        assert not np.allclose(d_clean, d_noisy)


@patch(_EJBL_PATCH, side_effect=_bump_valid_depth)
def test_ejbl_invokes_once_per_batch_env(_mock_add) -> None:
    obs = build_obs(batch=2)
    _depth_transform("ejbl", 7).transform(obs)
    # Two depth keys × two envs in the batch.
    assert _mock_add.call_count == 4


def test_augment_depth_none_skips_ejbl() -> None:
    obs = build_obs(batch=1)
    with patch(_EJBL_PATCH) as mock_add:
        _depth_transform(None, None).transform(obs)
        mock_add.assert_not_called()


def test_augment_depth_none_unchanged_vs_default() -> None:
    obs = build_obs(batch=1)
    explicit = _depth_transform(None, None).transform(obs)["gripper_depth"]
    default = ManiflowBlocksTableObsTransform(
        depth_keys=["gripper_depth"],
        include_ptmap=False,
    ).transform(obs)["gripper_depth"]
    np.testing.assert_allclose(explicit, default)


def test_unknown_augment_depth_raises() -> None:
    with pytest.raises(ValueError, match="augment_depth"):
        ManiflowObsTransform(augment_depth="elastic")


def test_require_depth_aug_raises_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    import toyblocks.serving.maniflow_serving_utils as msu

    monkeypatch.setattr(msu, "depth_aug_supported", lambda _profile: False)
    with pytest.raises(RuntimeError, match="--depth-aug"):
        msu.require_depth_aug_supported("ejbl")


def test_ejbl_transform_init_fails_when_unsupported(monkeypatch: pytest.MonkeyPatch) -> None:
    import toyblocks.serving.maniflow_serving_utils as msu

    monkeypatch.setattr(msu, "depth_aug_supported", lambda _profile: False)
    with pytest.raises(RuntimeError, match="--depth-aug"):
        ManiflowObsTransform(augment_depth="ejbl", depth_keys=("front_depth",))
