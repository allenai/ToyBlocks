"""Test block size randomization.

Run:
    pytest tests/env_randomization/test_block_size.py -v
    pytest tests/env_randomization/test_block_size.py -v --scene sim2real
"""

import pytest
import torch

from toyblocks.tasks.push_stacked.push_stacked_env_cfg import PushStackedEnvCfg
from toyblocks.tasks.push_stacked.push_stacked_env import PushStackedEnv
from tests.helpers import SCENE_ROBOT_MAP, teardown_sim


@pytest.fixture
def block_size_env(scene_id):
    """PushStackedEnv with randomize_block_size=True."""
    import omni.usd
    omni.usd.get_context().new_stage()

    cfg = PushStackedEnvCfg()
    # One env per bucket: test_block_sizes_cover_range asks for a span over 30%
    # of the range, which 4 draws from 16 buckets miss often enough to flake.
    cfg.scene.num_envs = 16
    cfg.sim.device = "cuda:0"
    cfg.scene_preset_id = scene_id
    cfg.randomize_block_size = True
    cfg.block_size_range = (0.03, 0.05)
    cfg.block_size_num_buckets = 16
    cfg.randomize_friction = False
    cfg.robot_id = SCENE_ROBOT_MAP[scene_id]

    env = PushStackedEnv(cfg=cfg)
    try:
        env.reset()
        yield env
    finally:
        env.close()
        teardown_sim()


def test_block_sizes_within_range(block_size_env):
    """block_sizes tensor should be filled with values in the configured range."""
    sizes = block_size_env.block_sizes
    assert sizes.shape == (block_size_env.num_envs, 3)
    assert (sizes >= 0.03 - 1e-4).all(), f"Block size below min: {sizes.min().item()}"
    assert (sizes <= 0.05 + 1e-4).all(), f"Block size above max: {sizes.max().item()}"


def test_block_sizes_isotropic(block_size_env):
    """Isotropic cubes should have equal x, y, z dimensions."""
    sizes = block_size_env.block_sizes
    assert torch.allclose(sizes[:, 0], sizes[:, 1], atol=1e-5)
    assert torch.allclose(sizes[:, 0], sizes[:, 2], atol=1e-5)


def test_block_sizes_vary_across_envs(block_size_env):
    """With 16 envs and randomization, sizes should not all be identical."""
    sizes = block_size_env.block_sizes[:, 0]
    assert not torch.all(sizes == sizes[0]), (
        f"All block sizes identical: {sizes.tolist()}"
    )


def test_block_sizes_quantized(block_size_env):
    """Sizes should be quantized to num_buckets discrete values."""
    sizes = block_size_env.block_sizes[:, 0]
    unique = sizes.unique()
    assert len(unique) <= block_size_env.cfg.block_size_num_buckets, (
        f"More unique sizes ({len(unique)}) than buckets ({block_size_env.cfg.block_size_num_buckets})"
    )


def test_block_sizes_cover_range(block_size_env):
    """Sizes should span a meaningful portion of the configured range."""
    sizes = block_size_env.block_sizes[:, 0]
    lo, hi = block_size_env.cfg.block_size_range
    span = hi - lo

    actual_span = sizes.max() - sizes.min()
    assert actual_span > 0.3 * span, (
        f"Block size span {actual_span.item():.4f} covers less than 30% "
        f"of range [{lo}, {hi}] (span={span})"
    )
