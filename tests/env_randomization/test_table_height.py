"""Test table height randomization.

Run:
    pytest tests/env_randomization/test_table_height.py -v
    pytest tests/env_randomization/test_table_height.py -v --scene sim2real
"""

import pytest
import torch

from toyblocks.tasks._template._template_env_cfg import TemplateEnvCfg
from toyblocks.tasks._template._template_env import TemplateEnv
from tests.helpers import SCENE_ROBOT_MAP, teardown_sim


@pytest.fixture
def height_env(scene_id):
    """Env with table_height_range > 0."""
    import omni.usd
    omni.usd.get_context().new_stage()

    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 4
    cfg.sim.device = "cuda:0"
    cfg.scene_preset_id = scene_id
    cfg.table_height_range = 0.02
    cfg.randomize_friction = False
    cfg.robot_id = SCENE_ROBOT_MAP[scene_id]

    env = TemplateEnv(cfg=cfg)
    try:
        env.reset()
        yield env
    finally:
        env.close()
        teardown_sim()


def test_table_height_offset_within_range(height_env):
    """All offsets must lie within [-table_height_range, +table_height_range]."""
    offsets = height_env.table_height_offset
    assert offsets.shape == (height_env.num_envs,)
    assert offsets.abs().max() <= height_env.cfg.table_height_range + 1e-6


def test_table_height_varies_across_envs(height_env):
    """With 4 envs, offsets should not all be identical (probabilistically)."""
    offsets = height_env.table_height_offset
    assert not torch.all(offsets == offsets[0]), (
        f"All offsets identical: {offsets.tolist()} — height randomization may not be working"
    )


def test_table_height_distribution_covers_range(height_env):
    """Collect offsets over many resets and verify the distribution covers the full range."""
    all_offsets = []
    for _ in range(20):
        height_env.reset()
        all_offsets.append(height_env.table_height_offset.clone())

    offsets = torch.cat(all_offsets)  # (20 * num_envs,)
    r = height_env.cfg.table_height_range

    assert offsets.min() < 0, f"No negative offsets seen: min={offsets.min().item():.4f}"
    assert offsets.max() > 0, f"No positive offsets seen: max={offsets.max().item():.4f}"

    assert offsets.min() < -0.6 * r, (
        f"Min offset {offsets.min().item():.4f} doesn't reach -60% of range ({-0.6*r:.4f})"
    )
    assert offsets.max() > 0.6 * r, (
        f"Max offset {offsets.max().item():.4f} doesn't reach +60% of range ({0.6*r:.4f})"
    )

    assert offsets.std() > 0.2 * r, (
        f"Std {offsets.std().item():.4f} too low — distribution may be degenerate"
    )


def test_table_height_changes_on_reset(height_env):
    """Resetting should produce new height offsets."""
    offsets_before = height_env.table_height_offset.clone()
    height_env.reset()
    offsets_after = height_env.table_height_offset.clone()
    assert not torch.allclose(offsets_before, offsets_after), (
        "Table height offsets unchanged after reset"
    )


def test_table_pos_matches_offset(height_env):
    """workspace_table Z position should reflect the applied offset."""
    table = height_env.scene.rigid_objects["workspace_table"]
    table_z = table.data.root_pos_w[:, 2]
    env_origin_z = height_env.scene.env_origins[:, 2]
    relative_z = table_z - env_origin_z

    offsets = height_env.table_height_offset
    z_spread = relative_z - relative_z.mean()
    offset_spread = offsets - offsets.mean()
    assert torch.allclose(z_spread, offset_spread, atol=1e-3), (
        f"Table Z spread {z_spread.tolist()} doesn't match offset spread {offset_spread.tolist()}"
    )
