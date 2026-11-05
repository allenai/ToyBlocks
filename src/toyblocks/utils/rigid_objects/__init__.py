from toyblocks import OBJECT_ASSET_DIR
import functools
import logging
from toyblocks.utils.rigid_objects.spawn_indexed import (
    IndexedMultiAssetSpawnerCfg,
    bucket_isotropic_cube_sizes,
)
from isaaclab.assets.rigid_object import RigidObjectCfg, RigidObject
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=None)
def _usd_base_side_lengths(usd_path: str) -> tuple[float, float, float]:
    """Read the actual (x, y, z) bounding-box extent of a toy-block USD's default prim.

    Toy-block USDs are not perfect unit cubes (their base extents differ slightly per axis and from
    any hardcoded normalizer), so scaling by a wrong reference makes the spawned block a different
    size than requested. Reading the true per-axis extent lets ``size / base_side_length`` produce a
    block whose extent matches ``size`` exactly, so the spawn height (``block_half + table_top``)
    seats it on the table instead of clipping in. Cached per path (cheap, read once).
    """
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path)
    prim = stage.GetDefaultPrim()
    if not prim or not prim.IsValid():
        prim = stage.GetPseudoRoot()
    # match obs_utils.get_bounding_box (default purpose) so block_sizes stays consistent with what
    # the observation/bounding-box code measures for the same asset.
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
    rng = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    return (float(mx[0] - mn[0]), float(mx[1] - mn[1]), float(mx[2] - mn[2]))


def _optional_mass_props(
    mass: float | None, density: float | None
) -> sim_utils.MassPropertiesCfg | None:
    if mass is None and density is None:
        return None
    return sim_utils.MassPropertiesCfg(mass=mass, density=density)


def _build_physics_material(
    static_friction: float | None,
    dynamic_friction: float | None,
    restitution: float | None,
) -> "sim_utils.RigidBodyMaterialCfg | None":
    """Build a physics material cfg from friction params, or ``None`` if none are provided.

    Only the provided fields are set; unspecified fields keep the IsaacLab defaults.
    """
    if static_friction is None and dynamic_friction is None and restitution is None:
        return None
    kwargs = {}
    if static_friction is not None:
        kwargs["static_friction"] = static_friction
    if dynamic_friction is not None:
        kwargs["dynamic_friction"] = dynamic_friction
    if restitution is not None:
        kwargs["restitution"] = restitution
    return sim_utils.RigidBodyMaterialCfg(**kwargs)


def _wrap_spawn_with_physics_material(spawn_cfg, material_cfg) -> None:
    """Wrap ``spawn_cfg.func`` so that, after the prims are spawned, a physics material is authored and
    bound to each spawned prim's collider.

    This is needed for USD-based spawners (``UsdFileCfg`` / ``MultiAssetSpawnerCfg``) which, unlike
    ``ShapeCfg``, have no ``physics_material`` field. The spawn func is called with the (possibly regex)
    prim path and clones across all envs, so after it returns we resolve every matching prim and bind the
    material via :func:`bind_physics_material` (which descends to the collider).
    """
    original_func = spawn_cfg.func

    def _wrapped(prim_path, cfg, translation=None, orientation=None, **kwargs):
        prim = original_func(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
        for env_prim_path in sim_utils.find_matching_prim_paths(prim_path):
            material_path = f"{env_prim_path}/physicsMaterial"
            material_cfg.func(material_path, material_cfg)
            sim_utils.bind_physics_material(env_prim_path, material_path)
        return prim

    spawn_cfg.func = _wrapped


def _wrap_spawn_with_per_env_physics_materials(spawn_cfg, material_cfgs: list) -> None:
    """Like :func:`_wrap_spawn_with_physics_material` but binds a different
    physics material per environment prim (cycling if there are more envs than
    materials)."""
    original_func = spawn_cfg.func

    def _wrapped(prim_path, cfg, translation=None, orientation=None, **kwargs):
        prim = original_func(prim_path, cfg, translation=translation, orientation=orientation, **kwargs)
        for i, env_prim_path in enumerate(sim_utils.find_matching_prim_paths(prim_path)):
            mat_cfg = material_cfgs[i % len(material_cfgs)]
            material_path = f"{env_prim_path}/physicsMaterial"
            mat_cfg.func(material_path, mat_cfg)
            sim_utils.bind_physics_material(env_prim_path, material_path)
        return prim

    spawn_cfg.func = _wrapped


def create_cuboid(
    prim_path: str,
    size: tuple[float, float, float] | list[tuple[float, float, float]],
    diffuse_color: tuple[float, float, float],
    mass: float | list[float] | None = None,
    density: float | list[float] | None = None,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    kinematic: bool = False,
    collision: bool = True,
    return_cfg: bool = False,
    metallic: float = 0.0,
    static_friction: float | list[float] | None = None,
    dynamic_friction: float | list[float] | None = None,
    restitution: float | list[float] | None = None,
    max_unique_assets: int | None = 64,
    size_range: tuple[float, float] | None = None,
):
    """
    Create a cuboid. If any of the following arguments (size, mass, density) are provided as lists, then this function will use the Isaac MultiAssetSpawner instead
    to spawn a different asset in each parallel environment. Generally you should pass lists of length equal to the number of parallel environments.

    When ``size`` is a per-env list longer than ``max_unique_assets``, the sizes
    are quantized into buckets so that only a bounded number of prototypes are
    spawned (mirrors ``create_toy_block``; avoids O(num_envs) asset creation).

    Args:
        prim_path: The prim path
        size: The size of the cuboid. These are the width, length, height values (not half-sizes)
        diffuse_color: The diffuse color of the cuboid
        mass: The mass of the cuboid. If None, the density will be used to compute the mass
        density: The density of the cuboid. If None, the mass will be used to compute the density
        init_pos: The initial position of the cuboid
        init_quat: The initial quaternion of the cuboid in format (w, x, y, z)
        kinematic: Whether the cuboid is kinematic (doesn't move due to external forces)
        collision: Whether to enable collisions. Useful to turn kinematic = True and collision = False for visual markers/objects
        metallic: Metallic property for the visual material (0.0 = dielectric, 1.0 = metal).
        static_friction: Static friction coefficient. If provided (along with the other friction params), a physics material is created and bound to the cuboid.
        dynamic_friction: Dynamic friction coefficient.
        restitution: Restitution (bounciness) coefficient.
        max_unique_assets: When per-env ``size`` list is longer than this, quantize
            into at most this many spawn prototypes. None disables bucketing.
        size_range: (min, max) isotropic side length for bucketing. Inferred from
            the size list if not provided.
    """
    if isinstance(size, list):
        first_size = size[0]
    else:
        first_size = size
    if isinstance(mass, list):
        first_mass = mass[0]
    else:
        first_mass = mass
    if isinstance(density, list):
        first_density = density[0]
    else:
        first_density = density
    first_sf = static_friction[0] if isinstance(static_friction, list) else static_friction
    first_df = dynamic_friction[0] if isinstance(dynamic_friction, list) else dynamic_friction
    first_rest = restitution[0] if isinstance(restitution, list) else restitution

    base_cfg = sim_utils.CuboidCfg(
        size=first_size,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color, metallic=metallic),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            solver_position_iteration_count=16, solver_velocity_iteration_count=0, kinematic_enabled=kinematic
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=first_mass, density=first_density),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
        physics_material=_build_physics_material(first_sf, first_df, first_rest),
    )

    multi_asset_spawn = (
        isinstance(size, list) or isinstance(mass, list) or isinstance(density, list)
        or isinstance(static_friction, list) or isinstance(dynamic_friction, list) or isinstance(restitution, list)
    )

    if multi_asset_spawn:
        assets_count = 0
        for arg in [size, mass, density, static_friction, dynamic_friction, restitution]:
            if isinstance(arg, list):
                assets_count = len(arg)
                break

        use_bucketing = (
            isinstance(size, list)
            and max_unique_assets is not None
            and assets_count > max_unique_assets
        )

        if use_bucketing:
            assert isinstance(size, list) and max_unique_assets is not None
            sides = [float(s[0]) for s in size]
            if size_range is None:
                size_range = (min(sides), max(sides))
            asset_indices, bucket_sizes = bucket_isotropic_cube_sizes(
                sides, max_unique_assets, size_range
            )
            logger.info(
                "Bucketing %d per-env cuboid sizes into %d prototypes (range %s).",
                assets_count, len(bucket_sizes), size_range,
            )
            assets_cfg = []
            for bsz in bucket_sizes:
                acfg: sim_utils.CuboidCfg = base_cfg.copy()  # type: ignore
                acfg.size = bsz
                assets_cfg.append(acfg)
            spawn = IndexedMultiAssetSpawnerCfg(
                assets_cfg=assets_cfg,
                asset_indices=asset_indices,
                random_choice=False,
            )
        else:
            assets_cfg = []
            for i in range(assets_count):
                asset_cfg: sim_utils.CuboidCfg = base_cfg.copy() # type: ignore
                if isinstance(size, list):
                    asset_cfg.size = size[i]
                if isinstance(mass, list):
                    asset_cfg.mass_props.mass = mass[i] # type: ignore
                if isinstance(density, list):
                    asset_cfg.mass_props.density = density[i] # type: ignore
                if isinstance(static_friction, list) or isinstance(dynamic_friction, list) or isinstance(restitution, list):
                    sf_i = static_friction[i] if isinstance(static_friction, list) else static_friction
                    df_i = dynamic_friction[i] if isinstance(dynamic_friction, list) else dynamic_friction
                    rest_i = restitution[i] if isinstance(restitution, list) else restitution
                    asset_cfg.physics_material = _build_physics_material(sf_i, df_i, rest_i)
                assets_cfg.append(asset_cfg)
            spawn = sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=assets_cfg,
                random_choice=False,
            )

        cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_quat)
        )
    else:
        cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=base_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_quat)
        )
    
    if return_cfg:
        return cfg
    else:
        return RigidObject(cfg=cfg)

def create_sphere(
    prim_path: str,
    radius: float,
    diffuse_color: tuple[float, float, float],
    mass: float | None = None,
    density: float | None = None,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    kinematic: bool = False,
    collision: bool = True,
    return_cfg: bool = False,
):
    """
    Create a sphere
    """
    cfg = RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.SphereCfg(
            radius=radius, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=diffuse_color),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, solver_velocity_iteration_count=0, kinematic_enabled=kinematic
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass, density=density),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_quat)
    )
    if return_cfg:
        return cfg
    else:
        return RigidObject(cfg=cfg)

def create_toy_block(
    prim_path: str,
    size: tuple[float, float, float] | list[tuple[float, float, float]],
    block_id: str | list[str] = "R",
    mass: float | list[float] | None = None,
    density: float | list[float] | None = None,
    init_pos: tuple[float, float, float] = (0.0, 0.0, 0.0),
    init_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    kinematic: bool = False,
    collision: bool = True,
    return_cfg: bool = False,
    usd_path: str | None = None,
    scale: tuple[float, float, float] | None = None,
    static_friction: float | list[float] | None = None,
    dynamic_friction: float | list[float] | None = None,
    restitution: float | list[float] | None = None,
    max_unique_assets: int | None = 64,
    size_range: tuple[float, float] | None = None,
):
    """
    Create a toy block

    If any of the following arguments (size, mass, density, block_id) are provided as lists, then this function will use the Isaac MultiAssetSpawner instead
    to spawn a different asset in each parallel environment. Generally you should pass lists of length equal to the number of parallel environments.

    When ``size`` is a per-env list longer than ``max_unique_assets``, the sizes
    are quantized into buckets so that only a bounded number of USD prototypes
    are loaded (avoids O(num_envs) USD loads that appear to hang on startup).

    Args:
        prim_path: The prim path
        size: The size of the block. These are the width, length, height values (not half-sizes).
        block_id: The id of the block. This is used to construct the prim path
        mass: The mass of the block. If None, the density will be used to compute the mass
        density: The density of the block. If None, the mass will be used to compute the density
        init_pos: The initial position of the block
        init_quat: The initial quaternion of the block in format (w, x, y, z)
        kinematic: Whether the block is kinematic (doesn't move due to external forces)
        collision: Whether to enable collisions. Useful to turn kinematic = True and collision = False for visual markers/objects
        usd_path: Explicit USD file path. Overrides the default ``block{block_id}`` lookup.
        scale: Explicit (sx, sy, sz) scale. Overrides the automatic ``size / base_side_length`` computation
            (where base_side_length is read from the asset USD's actual bounding box).
        static_friction: Static friction coefficient (or per-env list). If provided, a physics material
            is created and bound to the block. When a list is given, each environment gets its own material.
        dynamic_friction: Dynamic friction coefficient (or per-env list).
        restitution: Restitution (bounciness) coefficient (or per-env list).
        max_unique_assets: When per-env ``size`` list is longer than this, quantize
            into at most this many USD prototypes. None disables bucketing.
        size_range: (min, max) isotropic side length for bucketing. Inferred from
            the size list if not provided.
    """
    if isinstance(size, list):
        first_size = size[0]
    else:
        first_size = size
    if isinstance(mass, list):
        first_mass = mass[0]
    else:
        first_mass = mass
    if isinstance(density, list):
        first_density = density[0]
    else:
        first_density = density
    if isinstance(block_id, list):
        first_block_id = block_id[0]
    else:
        first_block_id = block_id
    resolved_usd_path = usd_path or str(OBJECT_ASSET_DIR / f"train_blocks/usd/{first_block_id}.usd")
    # actual per-axis base extent read from the asset USD 
    base_side_lengths = _usd_base_side_lengths(resolved_usd_path)
    resolved_scale = scale or (first_size[0] / base_side_lengths[0], first_size[1] / base_side_lengths[1], first_size[2] / base_side_lengths[2])
    base_cfg = sim_utils.UsdFileCfg(
        usd_path=resolved_usd_path,
        scale=resolved_scale,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=0, kinematic_enabled=kinematic),
        mass_props=_optional_mass_props(first_mass, first_density),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=collision),
    )
    multi_asset_spawn = isinstance(size, list) or isinstance(mass, list) or isinstance(density, list) or isinstance(block_id, list)

    if multi_asset_spawn:
        assets_count = 0
        if isinstance(size, list):
            assets_count = len(size)
        elif isinstance(mass, list):
            assets_count = len(mass)
        elif isinstance(density, list):
            assets_count = len(density)
        elif isinstance(block_id, list):
            assets_count = len(block_id)

        use_bucketing = (
            isinstance(size, list)
            and max_unique_assets is not None
            and assets_count > max_unique_assets
        )

        if use_bucketing:
            assert isinstance(size, list) and max_unique_assets is not None
            if size_range is None:
                sides = [float(s[0]) for s in size]
                size_range = (min(sides), max(sides))
            else:
                sides = [float(s[0]) for s in size]
            asset_indices, bucket_sizes = bucket_isotropic_cube_sizes(
                sides, max_unique_assets, size_range
            )
            logger.info(
                "Bucketing %d per-env sizes into %d prototypes (range %s).",
                assets_count, len(bucket_sizes), size_range,
            )
            assets_cfg = []
            for bsz in bucket_sizes:
                acfg: UsdFileCfg = base_cfg.copy()
                acfg.scale = (bsz[0] / base_side_lengths[0], bsz[1] / base_side_lengths[1], bsz[2] / base_side_lengths[2])
                assets_cfg.append(acfg)
            spawn = IndexedMultiAssetSpawnerCfg(
                assets_cfg=assets_cfg,
                asset_indices=asset_indices,
                random_choice=False,
            )
        else:
            assets_cfg = []
            for i in range(assets_count):
                asset_cfg: UsdFileCfg = base_cfg.copy()
                if isinstance(block_id, list):
                    asset_cfg.usd_path = str(OBJECT_ASSET_DIR / f"train_blocks/usd/{block_id[i]}.usd")
                asset_base = _usd_base_side_lengths(asset_cfg.usd_path)
                if isinstance(size, list):
                    asset_cfg.scale = (size[i][0] / asset_base[0], size[i][1] / asset_base[1], size[i][2] / asset_base[2])
                if isinstance(mass, list):
                    if asset_cfg.mass_props is None:
                        asset_cfg.mass_props = sim_utils.MassPropertiesCfg()
                    asset_cfg.mass_props.mass = mass[i]
                if isinstance(density, list):
                    if asset_cfg.mass_props is None:
                        asset_cfg.mass_props = sim_utils.MassPropertiesCfg()
                    asset_cfg.mass_props.density = density[i]
                assets_cfg.append(asset_cfg)
            spawn = sim_utils.MultiAssetSpawnerCfg(
                assets_cfg=assets_cfg,
                random_choice=False,
            )

        cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=spawn,
            init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_quat)
        )
    else:
        cfg = RigidObjectCfg(
            prim_path=prim_path,
            spawn=base_cfg,
            init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=init_quat)
        )
    # UsdFileCfg / MultiAssetSpawnerCfg do not support a `physics_material` field (unlike ShapeCfg),
    # so for USD blocks we bind the friction material to the spawned prims via a post-spawn wrapper.
    # This covers both the single-asset (UsdFileCfg) and multi-asset (MultiAssetSpawnerCfg) paths.
    has_per_env_friction = isinstance(static_friction, list) or isinstance(dynamic_friction, list) or isinstance(restitution, list)
    if has_per_env_friction:
        n = max(
            len(static_friction) if isinstance(static_friction, list) else 0,
            len(dynamic_friction) if isinstance(dynamic_friction, list) else 0,
            len(restitution) if isinstance(restitution, list) else 0,
        )
        per_env_mats = []
        for i in range(n):
            sf_i = static_friction[i] if isinstance(static_friction, list) else static_friction
            df_i = dynamic_friction[i] if isinstance(dynamic_friction, list) else dynamic_friction
            rest_i = restitution[i] if isinstance(restitution, list) else restitution
            per_env_mats.append(_build_physics_material(sf_i, df_i, rest_i))
        if per_env_mats and per_env_mats[0] is not None:
            _wrap_spawn_with_per_env_physics_materials(cfg.spawn, per_env_mats)
    else:
        phys_mat = _build_physics_material(
            static_friction,  # type: ignore[arg-type]
            dynamic_friction,  # type: ignore[arg-type]
            restitution,  # type: ignore[arg-type]
        )
        if phys_mat is not None:
            _wrap_spawn_with_physics_material(cfg.spawn, phys_mat)
    if return_cfg:
        return cfg
    else:
        return RigidObject(cfg=cfg)
