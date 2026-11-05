"""Test friction material property read/write on the merged workspace_table.

Run:
    pytest tests/env_randomization/test_friction_api.py -v
    pytest tests/env_randomization/test_friction_api.py -v --scene sim2real
"""

import pytest
import torch

from toyblocks.tasks._template._template_env_cfg import TemplateEnvCfg
from toyblocks.tasks._template._template_env import TemplateEnv
from toyblocks.utils.randomization_utils import randomize_rigid_body_friction, set_rigid_body_friction
from tests.helpers import SCENE_ROBOT_MAP, teardown_sim


@pytest.fixture
def env(scene_id):
    """Create a TemplateEnv, yield it, then fully tear down."""
    import omni.usd
    omni.usd.get_context().new_stage()

    cfg = TemplateEnvCfg()
    cfg.scene.num_envs = 2
    cfg.sim.device = "cuda:0"
    cfg.scene_preset_id = scene_id
    cfg.randomize_friction = False
    cfg.robot_id = SCENE_ROBOT_MAP[scene_id]

    base_env = TemplateEnv(cfg=cfg)
    try:
        base_env.reset()
        yield base_env
    finally:
        base_env.close()
        teardown_sim()


def test_workspace_table_in_rigid_objects(env):
    assert "workspace_table" in env.scene.rigid_objects, (
        f"Available: {list(env.scene.rigid_objects.keys())}"
    )


def test_root_physx_view_accessible(env):
    table = env.scene.rigid_objects["workspace_table"]
    assert table.root_physx_view is not None
    assert table.root_physx_view.count == env.num_envs


def test_initial_material_shape(env):
    table = env.scene.rigid_objects["workspace_table"]
    mat = table.root_physx_view.get_material_properties()
    assert mat.shape == (env.num_envs, table.root_physx_view.max_shapes, 3)


def test_set_friction_readback(env):
    table = env.scene.rigid_objects["workspace_table"]
    env_ids = torch.arange(env.num_envs, device=env.device)

    set_rigid_body_friction(table, 0.7, 0.5, 0.1, env_ids)

    mat = table.root_physx_view.get_material_properties()
    for i in range(env.num_envs):
        assert abs(mat[i, 0, 0].item() - 0.7) < 1e-4
        assert abs(mat[i, 0, 1].item() - 0.5) < 1e-4
        assert abs(mat[i, 0, 2].item() - 0.1) < 1e-4


def test_per_env_friction(env):
    table = env.scene.rigid_objects["workspace_table"]
    physx_view = table.root_physx_view
    total_shapes = physx_view.max_shapes
    all_env_ids = torch.arange(env.num_envs, device=env.device)

    mat_custom = torch.zeros(env.num_envs, total_shapes, 3, device="cpu")
    mat_custom[0, :, 0] = 0.9
    mat_custom[0, :, 1] = 0.8
    mat_custom[1, :, 0] = 0.1
    mat_custom[1, :, 1] = 0.05
    physx_view.set_material_properties(mat_custom, all_env_ids.cpu())

    mat = physx_view.get_material_properties()
    assert abs(mat[0, 0, 0].item() - 0.9) < 1e-4
    assert abs(mat[0, 0, 1].item() - 0.8) < 1e-4
    assert abs(mat[1, 0, 0].item() - 0.1) < 1e-4
    assert abs(mat[1, 0, 1].item() - 0.05) < 1e-4


def test_randomize_friction_within_range(env):
    table = env.scene.rigid_objects["workspace_table"]
    env_ids = torch.arange(env.num_envs, device=env.device)

    randomize_rigid_body_friction(
        asset=table,
        static_friction_range=(0.2, 0.3),
        dynamic_friction_range=(0.1, 0.2),
        restitution_range=(0.0, 0.0),
        env_ids=env_ids,
    )

    mat = table.root_physx_view.get_material_properties()
    for i in range(env.num_envs):
        s = mat[i, 0, 0].item()
        d = mat[i, 0, 1].item()
        r = mat[i, 0, 2].item()
        assert 0.2 - 1e-4 <= s <= 0.3 + 1e-4
        assert 0.1 - 1e-4 <= d <= 0.2 + 1e-4
        assert abs(r) < 1e-4
