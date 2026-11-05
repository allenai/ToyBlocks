"""R-on-Red task — red R-shaped block on a red target cube.

Spawns the five objects used by the TAMP ``mar16_red_high`` data-collection
scene (``block_r``, ``distractor1``, ``distractor2``, ``distractor3``,
``target_obj``) with the same sizes, colors, and asset types as the old
``BlocksSceneCfg``, plus a table at ``[0.5, 0, 0]``.

This task does not currently provide rewards / success criteria — it's
meant for replaying recorded trajectories through ``evaluate.py``:

    python scripts/evaluate.py --task Blocksuite-ROnRed-v0 \
        --replay_dataset_path /path/to/<episode>/blocks \
        --num_envs 1 --num_eval_steps 500
"""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import RigidObject, RigidObjectCfg

from blocksuite import OBJECT_ASSET_DIR
from blocksuite.scenes import SCENE_PRESETS
from blocksuite.tasks.base_env import BaseEnv

from .r_on_red_env_cfg import ROnRedEnvCfg


BLOCK_USD_PATH = str(
    OBJECT_ASSET_DIR / "train_blocks" / "usd" / "R.usd"
)
BLOCK_USD_SCALE = (1.65, 1.65, 1.65)

OBJECT_NAMES = ["block_r", "distractor1", "distractor2", "distractor3", "target_obj"]

OBJECT_COLORS = {
    "block_r": (1.0, 0.0, 0.0),
    "distractor1": (0.0, 0.0, 1.0),
    "distractor2": (0.0, 0.0, 1.0),
    "distractor3": (0.0, 0.0, 1.0),
    "target_obj": (1.0, 0.0, 0.0),
}

OBJECT_TYPES = {
    "block_r": "usd",
    "distractor1": "cuboid",
    "distractor2": "cuboid",
    "distractor3": "cuboid",
    "target_obj": "cuboid",
}


class ROnRedEnv(BaseEnv):
    cfg: ROnRedEnvCfg

    def _setup_scene(self):
        preset = SCENE_PRESETS[self.cfg.scene_preset_id]
        if "table" in preset.asset_cfgs:
            preset.asset_cfgs["table"].init_state.pos = list(self.cfg.table_pos)
        super()._setup_scene()

    def _setup_scene_objects(self):
        cube_size = tuple(self.cfg.block_size)
        for name in OBJECT_NAMES:
            if OBJECT_TYPES[name] == "usd":
                spawn = sim_utils.UsdFileCfg(
                    usd_path=BLOCK_USD_PATH,
                    scale=BLOCK_USD_SCALE,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                )
            else:
                spawn = sim_utils.CuboidCfg(
                    size=cube_size,
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=OBJECT_COLORS[name]
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
                    collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
                )
            cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_.*/{name}",
                spawn=spawn,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.6, 0.0, cube_size[2] / 2),
                ),
            )
            self.scene.rigid_objects[name] = RigidObject(cfg=cfg)

    def _get_obs_extra(self) -> dict:
        obs = {}
        for name in OBJECT_NAMES:
            obj = self.scene.rigid_objects[name]
            pos = obj.data.root_pos_w - self.scene.env_origins
            quat = obj.data.root_quat_w
            obs[f"object_pose_{name}"] = torch.cat([pos, quat], dim=-1)
        return obs

    def _get_obs_goal(self) -> dict:
        return {
            "dummy": torch.zeros(self.num_envs, 1, device=self.device),
        }

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def evaluate(self) -> dict:
        result = super().evaluate()
        result["success"] = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        return result

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
