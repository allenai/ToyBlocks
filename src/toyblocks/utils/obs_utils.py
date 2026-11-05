from isaaclab.sim.spawners.wrappers.wrappers_cfg import MultiAssetSpawnerCfg
from pxr import Usd, UsdGeom
import torch
from isaaclab.assets.asset_base import AssetBase
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.sim.spawners.shapes.shapes_cfg import CuboidCfg
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
import numpy as np
from typing import List, Tuple


ENDPOINT_SIGNS = torch.tensor([
    [-1, -1, +1],  # top_left_1
    [-1, +1, +1],  # top_left_2
    [+1, -1, +1],  # top_right_1
    [+1, +1, +1],  # top_right_2
    [+1, +1, -1],  # bottom_right_1
    [+1, -1, -1],  # bottom_right_2
    [-1, +1, -1],  # bottom_left_1
    [-1, -1, -1],  # bottom_left_2
], dtype=torch.float32)

def cube_corner_points(
    cube: RigidObject, cube_pos_w: torch.Tensor | None = None, cube_quat_w: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Args:
        cube: The cube asset to use for computing a bounding box for the corner points
        cube_pos_w: The position of the cube in the world frame. If none will attempt to use the asset's data.
        cube_quat_w: The quaternion of the cube in the world frame. If none will attempt to use the asset's data.

    Returns the xyz positions of eight endpoints of the cuboid in the world frame
    - Two top left endpoints (top face, left side)
    - Two bottom right endpoints (bottom face, right side)
    
    Supports cuboids with different length (x), width (y), and height (z) dimensions.
    
    Returns:
        torch.Tensor: Concatenated positions of eight endpoints, shape (num_envs, 8, 3)
            [top_left_1 (3D), top_left_2 (3D), top_right_1 (3D), top_right_2 (3D),
             bottom_right_1 (3D), bottom_right_2 (3D), bottom_left_1 (3D), bottom_left_2 (3D)]
    """
    
    num_envs = cube.data.root_pose_w.shape[0]
    bbox = get_bounding_box(cube)
    half_sizes = (bbox[:, 1] - bbox[:, 0]) / 2.0
    cube_pos_w = cube.data.root_pos_w if cube_pos_w is None else cube_pos_w
    cube_quat_w = cube.data.root_quat_w if cube_quat_w is None else cube_quat_w

    signs = ENDPOINT_SIGNS.to(cube.device)  # (8, 3)
    endpoints_local = signs.unsqueeze(0) * half_sizes.unsqueeze(1)  # (num_envs, 8, 3)
    
    # Transform endpoints from cube local frame to world frame (batched)
    # Reshape for quat_apply: (num_envs, 8, 3) -> (num_envs * 8, 3)
    # quat_apply will handle the batching automatically
    endpoints_local_flat = endpoints_local.reshape(-1, 3)  # (num_envs * 8, 3)
    cube_quat_w_expanded = cube_quat_w.unsqueeze(1).expand(-1, 8, -1).reshape(-1, 4)  # (num_envs * 8, 4)
    cube_pos_w_expanded = cube_pos_w.unsqueeze(1).expand(-1, 8, -1)  # (num_envs, 8, 3)
    
    # Apply rotation and translation
    endpoints_w_flat = cube_pos_w_expanded.reshape(-1, 3) + math_utils.quat_apply(cube_quat_w_expanded, endpoints_local_flat)
    endpoints_w = endpoints_w_flat.reshape(num_envs, 8, 3)  # (num_envs, 8, 3)
    return endpoints_w

_BBOX_CACHE: dict[str, torch.Tensor] = {}

def get_bounding_box(asset: AssetBase):
    """
    Get the bounding box of an asset.

    This script tries to center the bounding box at the origin but this part might be bugged. The extents are correct however.
    """
    if asset.cfg.prim_path in _BBOX_CACHE:
        return _BBOX_CACHE[asset.cfg.prim_path]
    obj_spawn = asset.cfg.spawn
    if isinstance(obj_spawn, CuboidCfg):
        b = asset.data.root_pose_w.shape[0]
        assert isinstance(asset, RigidObject)
        s = torch.tensor(obj_spawn.size, dtype=torch.float32)  # (3,)
        bbox = torch.stack([-s / 2, s / 2], dim=0).unsqueeze(0).expand(b, -1, -1)  # (b, 2, 3)
        _BBOX_CACHE[asset.cfg.prim_path] = bbox.to(asset.device)
        return bbox.to(asset.device)
    elif isinstance(obj_spawn, UsdFileCfg) or isinstance(obj_spawn, MultiAssetSpawnerCfg):
        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])

        num_envs = asset.data.root_pose_w.shape[0]
        bboxes = torch.zeros(num_envs, 2, 3, dtype=torch.float32, device=asset.device)
        for i in range(num_envs):
            prim_path = asset.cfg.prim_path 
            if ".*" in prim_path: 
                prim_path = prim_path.replace(".*", str(i))
            prim = asset.stage.GetPrimAtPath(prim_path)
            bbox = bbox_cache.ComputeLocalBound(prim)
            transform = np.asarray(bbox.GetMatrix())
            lower = np.asarray(bbox.GetBox().GetMin())
            upper = np.asarray(bbox.GetBox().GetMax())
            lower = torch.tensor(np.concatenate([lower, [1]]) @ transform)[:3]
            upper = torch.tensor(np.concatenate([upper, [1]]) @ transform)[:3]
            bboxes[i] = torch.stack([lower, upper], dim=0)
        _BBOX_CACHE[asset.cfg.prim_path] = bboxes
        return bboxes
    else:
        raise ValueError(f"Unsupported object type: {type(asset)}")


def distractors_pos_and_quat(distractors: List[RigidObject]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the pos and quat of a list of distractors.
    Args:
        distractors: List of distractors.
    Returns:
        Tuple of pos and quat tensors. (num_envs * num_distractors, 3) and (num_envs * num_distractors, 4)
    """

    poses = torch.stack([distractor.data.root_pos_w for distractor in distractors], dim=1).reshape(-1, 3)
    quats = torch.stack([distractor.data.root_quat_w for distractor in distractors], dim=1).reshape(-1, 4)
    return poses, quats