"""Test that friction physically affects block sliding on workspace_table.

Run:
    pytest tests/env_randomization/test_friction_slide.py -v
    pytest tests/env_randomization/test_friction_slide.py -v --scene sim2real
"""

import pytest
import torch

from blocksuite.tasks._template._template_env_cfg import TemplateEnvCfg
from blocksuite.tasks._template._template_env import TemplateEnv
from blocksuite.utils.randomization_utils import set_rigid_body_friction
from blocksuite.utils import rigid_objects
from isaaclab.utils import configclass
from tests.helpers import SCENE_ROBOT_MAP, teardown_sim

NUM_STEPS = 100
VELOCITY = 0.5


@configclass
class SlideTestEnvCfg(TemplateEnvCfg):
    episode_length_s: float = 99999.0


class SlideTestEnv(TemplateEnv):
    cfg: SlideTestEnvCfg

    def _setup_scene_objects(self):
        block_cfg = rigid_objects.create_cuboid(
            prim_path="/World/envs/env_.*/test_block",
            size=(0.04, 0.04, 0.04),
            diffuse_color=(1.0, 0.2, 0.2),
            density=1000.0,
            init_pos=(0.5, 0.0, 0.03),
            return_cfg=True,
        )
        from isaaclab.assets.rigid_object import RigidObject
        self.block = RigidObject(cfg=block_cfg)
        self.scene.rigid_objects["test_block"] = self.block

    def _reset_idx(self, env_ids):
        super()._reset_idx(env_ids)
        b = len(env_ids)
        block_pos = torch.tensor([0.5, 0.0, 0.03], device=self.device).unsqueeze(0).expand(b, -1).clone()
        block_pos_w = self.scene.env_origins[env_ids] + block_pos
        block_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).expand(b, -1)
        block_pose = torch.cat([block_pos_w, block_quat], dim=-1)
        self.block.write_root_pose_to_sim(block_pose, env_ids=env_ids)
        self.block.write_root_velocity_to_sim(
            torch.zeros(b, 6, device=self.device), env_ids=env_ids
        )


def _run_trial(base_env: SlideTestEnv, velocity: float, num_steps: int) -> float:
    """Reset, apply velocity, step, return mean X displacement."""
    base_env.reset()
    env_ids = torch.arange(base_env.num_envs, device=base_env.device)
    block = base_env.scene.rigid_objects["test_block"]

    initial_pos = block.data.root_pos_w.clone()

    lateral_vel = torch.zeros(base_env.num_envs, 6, device=base_env.device)
    lateral_vel[:, 0] = velocity
    block.write_root_velocity_to_sim(lateral_vel, env_ids=env_ids)
    base_env.scene.write_data_to_sim()
    base_env.sim.forward()

    action = torch.zeros(base_env.action_space.shape, device=base_env.device)
    for _ in range(num_steps):
        base_env.step(action)

    final_pos = block.data.root_pos_w.clone()
    return (final_pos[:, 0] - initial_pos[:, 0]).mean().item()


@pytest.fixture
def slide_env(scene_id):
    """Create a SlideTestEnv, yield it, then fully tear down."""
    import omni.usd
    omni.usd.get_context().new_stage()

    cfg = SlideTestEnvCfg()
    cfg.scene.num_envs = 1
    cfg.sim.device = "cuda:0"
    cfg.scene_preset_id = scene_id
    cfg.randomize_friction = False
    cfg.auto_reset = False
    cfg.ignore_terminations = True
    cfg.robot_id = SCENE_ROBOT_MAP[scene_id]

    env = SlideTestEnv(cfg=cfg)
    try:
        env.reset()
        yield env
    finally:
        env.close()
        teardown_sim()


def test_low_friction_more_displacement(slide_env):
    """Low friction should produce more sliding than high friction."""
    env_ids = torch.arange(slide_env.num_envs, device=slide_env.device)
    table = slide_env.scene.rigid_objects["workspace_table"]
    block = slide_env.scene.rigid_objects["test_block"]

    set_rigid_body_friction(block, 1.0, 1.0, 0.0, env_ids)

    # Low friction trial
    set_rigid_body_friction(table, 0.01, 0.01, 0.0, env_ids)
    disp_low = _run_trial(slide_env, VELOCITY, NUM_STEPS)

    # High friction trial
    set_rigid_body_friction(table, 1.0, 1.0, 0.0, env_ids)
    disp_high = _run_trial(slide_env, VELOCITY, NUM_STEPS)

    assert disp_low > disp_high + 0.001, (
        f"Expected low friction displacement ({disp_low:.4f}) > "
        f"high friction ({disp_high:.4f})"
    )
