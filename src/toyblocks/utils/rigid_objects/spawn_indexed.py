"""Indexed multi-asset spawning for large vectorized env counts.

Isaac Lab's MultiAssetSpawner assigns proto i to env i % num_protos.
When per-env randomized block scales are used with thousands of envs, loading
one USD prototype per env is prohibitively slow. This module provides:

- ``bucket_isotropic_cube_sizes``: quantize continuous side lengths into a small
  number of discrete buckets.
- ``IndexedMultiAssetSpawnerCfg`` / ``spawn_multi_asset_indexed``: a variant of
  MultiAssetSpawner that uses an explicit per-env index into the prototype list.
"""

from __future__ import annotations

import random
import re

import carb
from pxr import Sdf, Usd

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass


def bucket_isotropic_cube_sizes(
    sides: list[float],
    num_buckets: int,
    size_range: tuple[float, float],
) -> tuple[list[int], list[tuple[float, float, float]]]:
    """Quantize isotropic side lengths into discrete spawn-prototype buckets.

    Args:
        sides: Per-env side lengths.
        num_buckets: Number of discrete prototypes to create.
        size_range: (min_side, max_side) range for bucketing.

    Returns:
        asset_indices: Per-env bucket index (length == len(sides)).
        bucket_sizes: Unique (s, s, s) tuples (length == num_buckets).
    """
    lo, hi = size_range
    if num_buckets <= 1:
        bucket_sides = [float((lo + hi) / 2)]
    else:
        bucket_sides = [
            lo + (hi - lo) * i / (num_buckets - 1) for i in range(num_buckets)
        ]
    span = hi - lo
    asset_indices: list[int] = []
    for side in sides:
        if span <= 0:
            idx = 0
        else:
            t = (float(side) - lo) / span
            idx = max(0, min(num_buckets - 1, int(t * num_buckets)))
        asset_indices.append(idx)
    bucket_sizes = [(s, s, s) for s in bucket_sides]
    return asset_indices, bucket_sizes


def spawn_multi_asset_indexed(
    prim_path: str,
    cfg: "IndexedMultiAssetSpawnerCfg",
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    clone_in_fabric: bool = False,
    replicate_physics: bool = False,
) -> Usd.Prim:
    """Spawn multi-asset with explicit per-env prototype assignment."""
    if cfg.asset_indices is None:
        raise ValueError("IndexedMultiAssetSpawnerCfg.asset_indices must be set.")

    stage = sim_utils.get_current_stage()
    root_path, asset_path = prim_path.rsplit("/", 1)
    is_regex_expression = re.match(r"^[a-zA-Z0-9/_]+$", root_path) is None

    if is_regex_expression and root_path != "":
        source_prim_paths = sim_utils.find_matching_prim_paths(root_path)
        if len(source_prim_paths) == 0:
            raise RuntimeError(
                f"Unable to find source prim path: '{root_path}'."
            )
    else:
        source_prim_paths = [root_path]

    if len(cfg.asset_indices) != len(source_prim_paths):
        raise ValueError(
            f"asset_indices length ({len(cfg.asset_indices)}) != num envs "
            f"({len(source_prim_paths)})."
        )

    template_prim_path = sim_utils.get_next_free_prim_path("/World/Template", stage=stage)
    sim_utils.create_prim(template_prim_path, "Scope", stage=stage)

    proto_prim_paths: list[str] = []
    for index, asset_cfg in enumerate(cfg.assets_cfg):
        if cfg.semantic_tags is not None:
            if asset_cfg.semantic_tags is None:
                asset_cfg.semantic_tags = cfg.semantic_tags
            else:
                asset_cfg.semantic_tags += cfg.semantic_tags
        for attr_name in ("mass_props", "rigid_props", "collision_props", "activate_contact_sensors", "deformable_props"):
            attr_value = getattr(cfg, attr_name, None)
            if hasattr(asset_cfg, attr_name) and attr_value is not None:
                setattr(asset_cfg, attr_name, attr_value)
        proto_prim_path = f"{template_prim_path}/Asset_{index:04d}"
        asset_cfg.func(
            proto_prim_path,
            asset_cfg,
            translation=translation,
            orientation=orientation,
            clone_in_fabric=clone_in_fabric,
            replicate_physics=replicate_physics,
        )
        proto_prim_paths.append(proto_prim_path)

    prim_paths = [f"{sp}/{asset_path}" for sp in source_prim_paths]

    with Sdf.ChangeBlock():
        for index, env_prim_path in enumerate(prim_paths):
            env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), env_prim_path)
            if cfg.random_choice:
                proto_path = random.choice(proto_prim_paths)
            else:
                proto_path = proto_prim_paths[cfg.asset_indices[index]]
            Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_path), env_spec.layer, Sdf.Path(env_prim_path))

    sim_utils.delete_prim(template_prim_path, stage=stage)

    carb_settings_iface = carb.settings.get_settings()
    carb_settings_iface.set_bool("/isaaclab/spawn/multi_assets", True)

    return stage.GetPrimAtPath(prim_paths[0])


@configclass
class IndexedMultiAssetSpawnerCfg(sim_utils.MultiAssetSpawnerCfg):
    """MultiAssetSpawnerCfg variant that maps each env to a prototype by explicit index."""

    func = spawn_multi_asset_indexed
    asset_indices: list[int] | None = None
    """Per-environment index into ``assets_cfg`` (length must equal num envs)."""
