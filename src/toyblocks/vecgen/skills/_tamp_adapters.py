"""Adapters that let the TAMP scripted executors' helpers run on VecSim / Geometry.

The flip and rotate executors compute their joint targets through a ``robot`` object
(IK via ``robot._motion_gen.solve_ik``, FK via ``compute_ee_targets_batched``, the
articulation for finger-tip measurement) and a ``simulator`` (``scene`` and
``get_pose``). These classes provide exactly that surface for one env slot.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from curobo.types import Pose
from torch import Tensor

T = TypeVar("T")


class SlotIK:
    """``motion_gen.solve_ik`` for one pool slot (``ik_seed.solve_ik_seeded`` seeds it)."""

    def __init__(self, geo: Any, slot: int) -> None:
        self.geo = geo
        self.slot = slot

    def solve_ik(
        self, pose: Any, retract_config: Any = None, seed_config: Any = None, **_: Any
    ) -> Any:
        seed = None if seed_config is None else seed_config.reshape(-1)[: self.geo.dof]
        return self.geo.ik_result(self.slot, pose, seed)


def run_per_env(geo: Any, env_ids: list[int], fn: Callable[[int], T]) -> list[T]:
    """Run ``fn(env)`` for every env in sequence (the scripted builders are IK-bound)."""
    return [fn(e) for e in env_ids]


class SlotRobot:
    """The subset of ``IsaacRobotVectorized`` the scripted executors touch."""

    ROBOTIQ = True

    def __init__(self, sim: Any, geo: Any, slot: int) -> None:
        self.sim = sim
        self.geo = geo
        self.slot = slot
        self.device = sim.device
        self.joint_names = list(geo.pool.joint_names)
        self._robot = sim.art
        self._ee_link_id = [sim.ee_body_id]
        self._gripper_joint_ids = [sim.gripper_id]
        self._motion_gen = SlotIK(geo, slot)

    def get_curr_pose(self, env_idx: int | None = None) -> Pose:
        e = self.slot if env_idx is None else env_idx
        p = self.sim.ee_pose()[e]
        return Pose(position=p[:3].reshape(1, 3), quaternion=p[3:7].reshape(1, 4))

    def get_joint_states_batched(self, env_ids: Tensor) -> Tensor:
        return self.sim.joint_pos()[env_ids]

    def compute_ee_targets_batched(self, q: Tensor) -> Tensor:
        return self.geo.fk(q)


class SlotSim:
    def __init__(self, sim: Any) -> None:
        self.sim = sim
        self.scene = sim.scene
        self.device = sim.device

    def get_pose(self, name: str, env_idx: int = 0, local: bool = True) -> Tensor:
        return self.sim.object_pose(name)[env_idx]

    def block_half_extents(self, name: str, env_idx: int) -> Tensor | None:
        """Collider half extents ``[3]`` (body frame) of block ``name`` in env ``env_idx``,
        or None before ``VecSim.measure_block_dims`` ran."""
        if self.sim.block_dims is None:
            return None
        return 0.5 * self.sim.block_dims[env_idx, self.sim.block_names.index(name)]
