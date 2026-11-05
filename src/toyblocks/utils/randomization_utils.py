"""
Utilities for sampling/randomizing data
"""
import numpy as np
import torch
from isaaclab.utils import math
from isaaclab.assets.rigid_object.rigid_object import RigidObject

def random_quaterion(n: int, x_range: tuple[float, float], y_range: tuple[float, float], z_range: tuple[float, float], device: torch.device):
    """Uniformly samples a random quaternion. This is done by sampling a random euler angle within given X, Y, Z ranges (roll, pitch, yaw) and then converting to a quaternion.

    Args:
        n: Number of quaternions to sample
        x_range: Range for x component
        y_range: Range for y component
        z_range: Range for z component
        device: Device to sample on

    Returns:
        Tensor of shape (n, 4) containing the sampled quaternions
    """
    return math.quat_from_euler_xyz(torch.rand(n) * (x_range[1] - x_range[0]) + x_range[0], torch.rand(n) * (y_range[1] - y_range[0]) + y_range[0], torch.rand(n) * (z_range[1] - z_range[0]) + z_range[0])

ORI_QUATERNIONS = np.array([
    [1.0,0.0,0.0,0.0],
    [0.7071,0,0,-0.7071],
    [0.0,0.0,0.0,-1.0],
    [0.7071,0,0,0.7071],
    [0.7071,0,-0.7071,0],
    [0.5,-0.5,-0.5,-0.5],
    [0,0.7071,0,0.7071],
    [0.5,0.5,-0.5,0.5],
    [0.0,0.0,1.0,0.0],
    [0,0.7071,0.7071,0],
    [0.0,-1.0,0.0,0.0],
    [0.0,-0.7071,0.7071,0.0],
    [0.7071,0.0,0.7071,0.0],
    [0.5,0.5,0.5,-0.5],
    [0,-0.7071,0,0.7071],
    [0.5,-0.5,0.5,0.5],
    [0.5,0.5,-0.5,-0.5],
    [0,0,0.7071,0.7071],
    [0.5,0.5,0.5,0.5],
    [0.7071,0.7071,0,0],
    [0.5,-0.5,0.5,-0.5],
    [0,0,0.7071,-0.7071],
    [0.5,-0.5,-0.5,0.5],
    [0.7071,-0.7071,0,0],
])
def random_cube_orientation_symmetry(n: int, device: torch.device):
    global ORI_QUATERNIONS
    # NOTE: this is copied from tamp codebase, not the best place to put this code...
    if isinstance(ORI_QUATERNIONS, np.ndarray):
        ORI_QUATERNIONS = torch.tensor(ORI_QUATERNIONS, device=device)
    return ORI_QUATERNIONS[torch.randint(0, ORI_QUATERNIONS.shape[0], (n,))]


def randomize_rigid_body_friction(
    asset: RigidObject,
    static_friction_range: tuple[float, float],
    dynamic_friction_range: tuple[float, float],
    restitution_range: tuple[float, float],
    env_ids: torch.Tensor,
    make_consistent: bool = True,
) -> torch.Tensor:
    """Randomize the friction of a rigid body. Meant to roughly mimic Isaac Lab's randomize_rigid_body_material function.

    Returns:
        The sampled friction values of shape ``(n_envs, 3)`` (static, dynamic, restitution) on CPU, so callers
        can store them (e.g. for privileged/asymmetric-critic observations).
    """

    n_envs = len(env_ids)
    total_num_shapes = asset.root_physx_view.max_shapes
    range_list = [static_friction_range, dynamic_friction_range, restitution_range]
    ranges = torch.tensor(range_list, device="cpu")
    friction_samples = math.sample_uniform(ranges[:, 0], ranges[:, 1], (n_envs, 3), device="cpu")

    # ensure dynamic friction is always less than static friction
    if make_consistent:
        friction_samples[:, 1] = torch.min(friction_samples[:, 0], friction_samples[:, 1])

    env_ids_cpu = env_ids.cpu()
    materials = asset.root_physx_view.get_material_properties().clone()
    materials[env_ids_cpu] = friction_samples.unsqueeze(1).expand(-1, total_num_shapes, -1)
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)
    return friction_samples


def set_rigid_body_friction(
    asset: RigidObject,
    static_friction: float | torch.Tensor,
    dynamic_friction: float | torch.Tensor,
    restitution: float | torch.Tensor,
    env_ids: torch.Tensor,
):
    """Set the friction of a rigid body to fixed values for the given envs."""
    def _to_cpu_tensor(v):
        if isinstance(v, torch.Tensor):
            return v.cpu()
        return torch.tensor(v, device="cpu")

    s = _to_cpu_tensor(static_friction)
    d = _to_cpu_tensor(dynamic_friction)
    r = _to_cpu_tensor(restitution)

    n_envs = len(env_ids)
    total_num_shapes = asset.root_physx_view.max_shapes
    env_ids_cpu = env_ids.cpu()

    materials = asset.root_physx_view.get_material_properties().clone()
    materials[env_ids_cpu, :, 0] = s if s.dim() == 0 else s.unsqueeze(1)
    materials[env_ids_cpu, :, 1] = d if d.dim() == 0 else d.unsqueeze(1)
    materials[env_ids_cpu, :, 2] = r if r.dim() == 0 else r.unsqueeze(1)
    asset.root_physx_view.set_material_properties(materials, env_ids_cpu)


def closest_block_size_slot(
    side_length: float,
    sizes_list: list,
) -> int:
    """Return the index of the closest isotropic block size entry."""
    best_idx = 0
    best_err = float("inf")
    for idx, size in enumerate(sizes_list):
        side = float(np.asarray(size, dtype=np.float32).reshape(-1)[0])
        err = abs(side - side_length)
        if err < best_err:
            best_err = err
            best_idx = idx
    return best_idx


def apply_rigid_object_uniform_scale(
    asset: RigidObject,
    env_idx: int,
    factor: float,
) -> None:
    """Multiply a rigid object's USD ``xformOp:scale`` by *factor*."""
    if abs(factor - 1.0) < 1e-4:
        return

    import isaaclab.sim as sim_utils
    from pxr import Gf, UsdGeom

    prim_path = asset.root_physx_view.prim_paths[env_idx]
    stage = sim_utils.get_current_stage()
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return

    xformable = UsdGeom.Xformable(prim)
    scale_attr = prim.GetAttribute("xformOp:scale")
    if scale_attr.IsValid():
        cur = scale_attr.Get()
        new_scale = Gf.Vec3d(
            cur[0] * factor,
            cur[1] * factor,
            cur[2] * factor,
        )
        scale_attr.Set(new_scale)
    else:
        xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(factor, factor, factor)
        )
