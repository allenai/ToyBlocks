"""Adapt a ToyBlocks ``BaseEnv`` to the simulator interface the oracle uses.

``OracleHighLevelPolicy`` (ported from TAMP's ``evals/oracle_hl_policy.py``) reads a
tmp3-``IsaacLabSimulatorVectorized``-shaped simulator.  The ToyBlocks eval harness
instead drives a ``BaseEnv`` (an ``isaaclab`` ``DirectRLEnv`` subclass).  This module
exposes the **exact subset** of that interface the oracle touches, backed by ``BaseEnv``.

One adapter is built **per environment** and bound to a fixed ``env_index``.  The
oracle is constructed with ``n_envs=1``, so internally it always passes
``env_idx=0``; we ignore that and operate on the bound real env.  This keeps the
oracle code unmodified while supporting many parallel envs (one oracle each).

Interface exposed:
- ``get_pose(obj_name, env_idx, local)`` -> ``(7,)`` ``[x,y,z, qw,qx,qy,qz]`` tensor
- ``robots["robot"]`` with ``_gripper_open`` (shape ``(1,)``), ``_robot``, ``_ee_link_id``
- ``scene`` with ``sensors.get("front_camera")``, ``rigid_objects``, ``env_origins``, ``keys()``
- ``num_envs`` (1), ``device``, ``env_profile.get_cube_scale(env_idx)``
- ``get_rigid_object(obj_name).get_size(env_idx)`` -> the bound env's spawned ``(x, y, z)``
  extent (``on_table.evaluate`` reads table and block heights through it)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import torch

# Local frame = the robot base frame, as tmp3's simulator defined it (so projected
# pixels line up with the env-local camera extrinsics the oracle reads).
from toyblocks.vecgen.math import quat_apply, quat_inverse, quat_mul

# The canonical TCP (grasp point) is the ``ee`` frame transformer every task env
# uses: panda_link7 + a local +0.2418 m offset. Measured ground truth: during a
# real grasp the cube sits 0.03-0.09 m from this frame (well under the oracle's 0.10 m
# gate), whereas panda_link7 alone is ~0.28-0.43 m and the inner-finger link
# midpoint is ~0.15-0.18 m (the link origins sit at the knuckle, not the pads).
_EE_TCP_SENSOR = "frame_transformer:ee"


class _EEDataProxy:
    """Re-slices ``body_link_pose_w`` so index 0 maps to the bound real env.

    The EE row's xyz is rewritten to the canonical TCP (the ``ee`` frame
    transformer's ``target_pos_w``) so the oracle's grasp-distance gate measures
    cube-to-grasp-point instead of cube-to-wrist-link.
    """

    def __init__(
        self,
        articulation: Any,
        env_index: int,
        *,
        ee_index: int,
        tcp_getter: Any,
    ) -> None:
        self._art = articulation
        self._i = env_index
        self._ee_index = ee_index
        self._tcp_getter = tcp_getter

    @property
    def body_link_pose_w(self) -> torch.Tensor:
        # Clone: the raw slice is a view into the live sim buffer, so overwriting
        # the EE row in place would corrupt the physics state every step.
        pose = self._art.data.body_link_pose_w[self._i : self._i + 1].clone()
        pose[0, self._ee_index, :3] = self._tcp_getter()
        return pose


class _RobotProxy:
    """Minimal stand-in exposing ``.data.body_link_pose_w`` for env-0 indexing."""

    def __init__(
        self,
        articulation: Any,
        env_index: int,
        *,
        ee_index: int,
        tcp_getter: Any,
    ) -> None:
        self.data = _EEDataProxy(
            articulation, env_index, ee_index=ee_index, tcp_getter=tcp_getter
        )


class _RobotAdapter:
    """Expose the tmp3 ``IsaacRobotVectorized`` attributes the oracle reads."""

    def __init__(
        self,
        base_env: Any,
        env_index: int,
        *,
        gripper_joint_index: Optional[int] = None,
        gripper_open_below: float = 0.25,
    ) -> None:
        self._art = base_env.robot.articulation  # isaaclab Articulation
        self._i = env_index
        self._gripper_open_below = gripper_open_below
        self._gripper_idx = (
            gripper_joint_index
            if gripper_joint_index is not None
            else _resolve_gripper_joint_index(self._art)
        )
        # The row's xyz is overwritten with the TCP in the proxy, so this only needs
        # to name the right link: the same one tmp3's real wrapper resolves.
        self._ee_idx = base_env._resolve_ee_body_idx()
        # The canonical TCP source. Fail loudly if absent -- the grasp gate
        # cannot work without it (no silent fallback to a wrist body).
        self._ee_frame = base_env.scene.sensors.get(_EE_TCP_SENSOR)
        if self._ee_frame is None:
            raise RuntimeError(
                f"oracle grasp gate requires the {_EE_TCP_SENSOR!r} frame "
                "transformer (the canonical TCP), but it is not in the scene. "
                f"Available sensors: {sorted(base_env.scene.sensors.keys())}."
            )

    def _tcp_world(self) -> torch.Tensor:
        """Canonical TCP (grasp point) world xyz for the bound env, ``(3,)``."""
        return self._ee_frame.data.target_pos_w[self._i, 0]

    # -- attributes named to match what the oracle accesses --------------------
    @property
    def _robot(self) -> Any:
        """Proxy whose ``.data.body_link_pose_w[0]`` is the bound real env."""
        return _RobotProxy(
            self._art, self._i, ee_index=self._ee_idx, tcp_getter=self._tcp_world
        )

    @property
    def _ee_link_id(self) -> int:
        return self._ee_idx

    def ee_world_pos(self) -> torch.Tensor:
        """Canonical TCP world xyz for the bound env, ``(3,)`` (what the oracle reads)."""
        return self._tcp_world()

    @property
    def _gripper_open(self) -> torch.Tensor:
        """Bool tensor ``(1,)`` — True when the bound env's gripper is open.

        Derived from the live ``finger_joint`` position (Robotiq: open≈0.0,
        closed≈0.5). tmp3 tracks a *commanded* flag instead, but the oracle only
        uses this as one gate of grasp detection (alongside object lift + EE
        distance), so a physical readout is equivalent in practice.
        """
        jp = self._art.data.joint_pos[self._i : self._i + 1, self._gripper_idx]
        return jp < self._gripper_open_below


class _EnvProfileAdapter:
    """Stand-in for tmp3's ``env_profile`` (only ``get_cube_scale`` is used)."""

    def __init__(self, cube_scale: float = 1.0) -> None:
        self._cube_scale = float(cube_scale)

    def get_cube_scale(self, env_idx: int = 0) -> float:  # noqa: ARG002 - parity with tmp3
        return self._cube_scale

    def set_cube_scale(self, cube_scale: float) -> None:
        self._cube_scale = float(cube_scale)


class _SensorShim:
    """Maps the oracle's ``scene.sensors.get('front_camera')`` to a BaseEnv camera."""

    def __init__(self, base_env: Any) -> None:
        self._base_env = base_env

    def get(self, name: str, default: Any = None) -> Any:
        cam = self._base_env.cameras.get(name)
        if cam is not None:
            return cam
        # Fall back to the registered sensor key (``camera:<name>``); the oracle
        # tolerates ``None`` and uses default intrinsics if neither exists.
        return self._base_env.scene.sensors.get(f"camera:{name}", default)


class _SceneShim:
    """Expose the ``scene`` attributes the oracle reads, backed by BaseEnv."""

    def __init__(self, base_env: Any) -> None:
        self._base_env = base_env
        self.sensors = _SensorShim(base_env)

    @property
    def rigid_objects(self) -> Any:
        # dict keyed by PDDL-matching names; iterating yields the names.
        return self._base_env.scene.rigid_objects

    @property
    def env_origins(self) -> torch.Tensor:
        return self._base_env.scene.env_origins

    def keys(self):
        return self._base_env.scene.rigid_objects.keys()


class _SizedObject:
    """``get_rigid_object(name)`` result: tmp3's ``get_size(env_idx)`` for one scene object.

    Like ``get_pose``, the size is the bound env's: ``env_idx`` is ignored, because the
    single-env oracle passes ``None`` or ``0`` while its block may have spawned in any
    per-env scale slot.
    """

    def __init__(self, size: Callable[[], tuple[float, float, float]]) -> None:
        self._size = size

    def get_size(self, env_idx: Optional[int] = None) -> tuple[float, float, float]:  # noqa: ARG002
        return self._size()


def _spawn_size(spawn: Any, env_index: int, name: str) -> tuple[float, float, float]:
    """``(x, y, z)`` extent an object's spawn config gives env ``env_index``.

    Read by field, not by config class: a multi-asset spawner (``assets_cfg``, with the
    per-env slot in ``asset_indices`` or round-robin), a cuboid (``size``), or a USD file
    (``usd_path``: its base extent times ``scale``).
    """
    assets = getattr(spawn, "assets_cfg", None)
    if assets:
        indices = getattr(spawn, "asset_indices", None)
        slot = indices[env_index] if indices is not None else env_index % len(assets)
        return _spawn_size(assets[slot], env_index, name)
    size = getattr(spawn, "size", None)
    if size is not None:
        return tuple(float(v) for v in size)
    usd_path = getattr(spawn, "usd_path", None)
    if usd_path is not None:
        from toyblocks.utils.rigid_objects import _usd_base_side_lengths

        scale = getattr(spawn, "scale", None) or (1.0, 1.0, 1.0)
        return tuple(float(b) * float(k) for b, k in zip(_usd_base_side_lengths(usd_path), scale))
    raise ValueError(f"no size for {name!r}: unsupported spawn config {type(spawn).__name__}")


class BaseEnvOracleSimAdapter:
    """tmp3-simulator-compatible view over one env of a ToyBlocks ``BaseEnv``."""

    def __init__(
        self,
        base_env: Any,
        env_index: int = 0,
        *,
        gripper_joint_index: Optional[int] = None,
        cube_scale: float = 1.0,
    ) -> None:
        self._base_env = base_env
        self._i = int(env_index)
        self.num_envs = 1
        self.device = base_env.device
        self.scene = _SceneShim(base_env)
        self.robots = {
            "robot": _RobotAdapter(base_env, self._i, gripper_joint_index=gripper_joint_index)
        }
        self.env_profile = _EnvProfileAdapter(cube_scale)

    @property
    def env_origins(self) -> torch.Tensor:
        return self._base_env.scene.env_origins

    @property
    def base_env(self) -> Any:
        """The wrapped ``BaseEnv`` (all envs)."""
        return self._base_env

    @property
    def env_index(self) -> int:
        """The real env this adapter (and its single-env oracle) is bound to."""
        return self._i

    def get_rigid_object(self, obj_name: str) -> _SizedObject:
        """tmp3's ``get_rigid_object``, reduced to the ``get_size`` the predicates use."""
        self._get_asset(obj_name)  # unknown names fail here, as in get_pose
        return _SizedObject(lambda: self._object_size(obj_name))

    def _object_size(self, obj_name: str) -> tuple[float, float, float]:
        """Spawned extent of ``obj_name`` in the bound env: the env package's per-env block
        sizes (what datagen spawned) for blocks, the spawn config otherwise (the table)."""
        from toyblocks.eval.scene_verification import _read_block_sizes

        block_sizes = _read_block_sizes(self._base_env, self._i)
        if obj_name in block_sizes:
            return tuple(float(v) for v in block_sizes[obj_name])
        return _spawn_size(self._get_asset(obj_name).cfg.spawn, self._i, obj_name)

    def ee_world_pos(self) -> torch.Tensor:
        """Canonical TCP world xyz for the bound env, ``(3,)`` (what the oracle reads)."""
        return self.robots["robot"].ee_world_pos()

    def set_cube_scale(self, cube_scale: float) -> None:
        """Update the per-env cube scale the oracle reads via ``env_profile``.

        Adapters are built once (bound to ``env_index``) but the cube size is a
        per-episode property, so the controller pushes the reset episode's scale
        in on each ``on_reset``.
        """
        self.env_profile.set_cube_scale(cube_scale)

    def _get_asset(self, obj_name: str) -> Any:
        scene = self._base_env.scene
        ro = scene.rigid_objects
        if obj_name in ro:
            return ro[obj_name]
        arts = scene.articulations
        if obj_name in arts:
            return arts[obj_name]
        raise ValueError(
            f"get_pose: object '{obj_name}' not found in scene.rigid_objects "
            f"{list(ro.keys())} or articulations {list(arts.keys())}"
        )

    def get_velocity(self, obj_name: str, env_idx: Optional[int] = 0, local: bool = False) -> torch.Tensor:
        """Return ``(6,)`` ``[vx,vy,vz, wx,wy,wz]`` (world frame) for the bound env.

        ``env_idx`` and ``local`` are ignored (the oracle is single-env, bound to
        ``env_index``, and only reads velocity *magnitudes* for the settle check).
        """
        return self._get_asset(obj_name).data.root_state_w[self._i, 7:13]

    def get_pose(self, obj_name: str, env_idx: Optional[int] = 0, local: bool = True) -> torch.Tensor:
        """Return ``(7,)`` ``[x,y,z, qw,qx,qy,qz]`` for the bound env.

        ``env_idx`` is ignored (the oracle is single-env and bound to ``env_index``).
        World frame mirrors ``root_state_w[i, :7]``; local frame is the full SE(3)
        transform into the robot base frame.
        """
        i = self._i
        obj = self._get_asset(obj_name)
        pose_w = obj.data.root_state_w[i, :7]
        if not local:
            return pose_w
        base = self._base_env.robot.articulation.data.root_state_w[i, :7]
        base_quat_inv = quat_inverse(base[3:])
        local_pos = quat_apply(base_quat_inv, pose_w[:3] - base[:3])
        local_quat = quat_mul(base_quat_inv, pose_w[3:])
        return torch.cat([local_pos, local_quat])


def _resolve_gripper_joint_index(articulation: Any) -> int:
    """Index of the driven ``finger_joint`` in ``joint_pos``.

    This index drives grasp-open detection, so guessing it (the old default of 7)
    inverts the oracle's grasp gate on any robot whose joint order differs: the
    oracle then believes it is holding a block it dropped, or vice versa, and the
    plan advances against a false world state. Raises instead.
    """
    ids, _ = articulation.find_joints(["finger_joint"])
    if not ids:
        raise RuntimeError(
            "robot articulation has no 'finger_joint'; the oracle's grasp detection "
            f"reads that joint's position (joints: {list(articulation.joint_names)})."
        )
    return int(ids[0])
