"""The simulator surface the stage-1 predicate samplers use, over an Isaac Lab scene.

The samplers in :mod:`blocksuite.vecgen.sampling.predicates` were written against
``tmp3``'s ``IsaacLabSimulatorVectorized``; this class keeps the same call names and
semantics (``get_pose`` / ``set_rigid_object_pose`` with ``local``, ``get_size`` per env
through :meth:`get_rigid_object`, ``obj_prim_scene_mapping`` as the set of block names)
so the samplers port line for line and sample the same distributions.

Frames: the local env frame is the robot base frame (robot root pose, as TAMP's
``env_base_poses``); with the base at the env origin and identity rotation this is world
minus ``scene.env_origins``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import Tensor

from blocksuite.vecgen.math import quat_apply, quat_inverse, quat_mul


class SceneObject:
    """Per-object size lookup (``IsaacObjectWrapper.get_size`` from tmp3).

    Sizes follow the spawner config of the env's asset slot. For USD-file assets the
    size is the env-0 prim's local bounding box times its world scale, as in tmp3; for
    multi-asset letter blocks that box is not the block's own extent (it measures about
    0.35 x 0.31 x 0.038 m for every letter), which the samplers have always used for block
    heights, so it is kept here to sample the same states.
    """

    def __init__(self, asset: Any, name: str, stage: Any) -> None:
        self.object = asset
        self.name = name
        prim_path = asset.cfg.prim_path.replace(".*", "0")
        self.prim = stage.GetPrimAtPath(prim_path)
        if not self.prim or not self.prim.IsValid():
            raise ValueError(f"Prim '{prim_path}' not found in the stage.")
        self._size_cache: dict[int, tuple[float, float, float]] = {}

    def _resolve_slot(self, env_idx: int | None) -> int:
        from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiAssetSpawnerCfg

        from blocksuite.utils.rigid_objects.spawn_indexed import IndexedMultiAssetSpawnerCfg

        spawn = self.object.cfg.spawn
        if not isinstance(spawn, MultiAssetSpawnerCfg):
            return -1
        if isinstance(spawn, IndexedMultiAssetSpawnerCfg) and spawn.asset_indices is not None:
            return spawn.asset_indices[env_idx] if env_idx is not None else spawn.asset_indices[0]
        n = len(spawn.assets_cfg) if spawn.assets_cfg else 1
        return (env_idx % n) if env_idx is not None else 0

    def _resolve_asset_cfg(self, env_idx: int | None) -> Any:
        from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiAssetSpawnerCfg

        spawn = self.object.cfg.spawn
        if not isinstance(spawn, MultiAssetSpawnerCfg):
            return spawn
        if not spawn.assets_cfg:
            raise ValueError("MultiAssetSpawnerCfg has empty assets_cfg")
        return spawn.assets_cfg[max(self._resolve_slot(env_idx), 0)]

    def get_size(self, env_idx: int | None = None) -> tuple[float, float, float]:
        import isaaclab.sim as sim_utils
        from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg, CylinderCfg, SphereCfg
        from pxr import Usd, UsdGeom

        slot = self._resolve_slot(env_idx)
        if slot in self._size_cache:
            return self._size_cache[slot]
        cfg = self._resolve_asset_cfg(env_idx)
        if isinstance(cfg, CuboidCfg):
            size = tuple(cfg.size)
        elif isinstance(cfg, SphereCfg):
            size = (cfg.radius, cfg.radius, cfg.radius)
        elif isinstance(cfg, CylinderCfg):
            size = (cfg.radius, cfg.radius, cfg.height)
        elif isinstance(cfg, sim_utils.UsdFileCfg):
            cache = UsdGeom.BBoxCache(
                Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_]
            )
            local_size = np.asarray(
                cache.ComputeLocalBound(self.prim).GetRange().GetSize(), dtype=np.float64
            )
            wt = UsdGeom.Xformable(self.prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            scale = np.array(
                [float(wt.GetColumn(i).GetLength()) for i in range(3)], dtype=np.float64
            )
            size = tuple(float(v) for v in local_size * scale)
        else:
            raise ValueError(f"Unsupported object type: {type(cfg)}")
        self._size_cache[slot] = size
        return size


class SceneSim:
    """Batched pose reads/writes and per-env sizes for the scene's rigid objects.

    ``block_names`` plays the role of tmp3's ``obj_prim_scene_mapping`` keys: the objects
    an on-table placement must keep clear of (every block, not the table).
    """

    def __init__(self, env: Any, block_names: list[str]) -> None:
        self.base_env = env.unwrapped
        self.scene = self.base_env.scene
        self.device = torch.device(self.base_env.device)
        self.num_envs = int(self.scene.num_envs)
        self.obj_prim_scene_mapping: dict[str, str] = {
            n: self.scene[n].cfg.prim_path for n in block_names
        }
        self._objects: dict[str, SceneObject] = {}
        self._base_pose: Tensor = self.scene["robot"].data.root_pose_w[:, :7].clone()

    def get_rigid_object(self, name: str) -> SceneObject:
        if name not in self._objects:
            self._objects[name] = SceneObject(self.scene[name], name, self.scene.stage)
        return self._objects[name]

    def _to_local(self, pose: Tensor, env_idx: int) -> Tensor:
        base = self._base_pose[env_idx]
        inv = quat_inverse(base[3:])
        return torch.cat([quat_apply(inv, pose[:3] - base[:3]), quat_mul(inv, pose[3:])])

    def _to_world(self, pose: Tensor, env_idx: int) -> Tensor:
        base = self._base_pose[env_idx]
        return torch.cat([quat_apply(base[3:], pose[:3]) + base[:3], quat_mul(base[3:], pose[3:])])

    def get_pose(self, name: str, env_idx: int | None = 0, local: bool = True) -> Tensor:
        """``[num_envs, 7]`` when ``env_idx`` is None (local = minus env origins), else ``[7]``."""
        root = self.scene[name].data.root_state_w
        if env_idx is None:
            poses = root[:, :7].clone()
            if local:
                poses[:, :3] -= self.scene.env_origins
            return poses
        pose = root[env_idx, :7]
        return self._to_local(pose, env_idx) if local else pose.clone()

    def set_rigid_object_pose(
        self, name: str, pose: Tensor, env_idx: int | None = None, local: bool = False
    ) -> None:
        """Write one env's root pose (and zero its velocity)."""
        env_idx = 0 if env_idx is None else env_idx
        pose = torch.as_tensor(pose, device=self.device, dtype=torch.float32)
        if local:
            pose = self._to_world(pose, env_idx)
        asset = self.scene[name]
        ids = torch.tensor([env_idx], device=asset.device)
        asset.write_root_pose_to_sim(pose.to(asset.device).unsqueeze(0), env_ids=ids)
        asset.write_root_velocity_to_sim(torch.zeros(1, 6, device=asset.device), env_ids=ids)
