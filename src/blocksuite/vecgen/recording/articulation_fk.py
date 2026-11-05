"""End-effector FK from the simulated robot's own kinematics, for ``action.ee_target``.

The recorder stores ``action.ee_target = FK(commanded arm joint targets)``. Datagen computes it
with cuRobo; a caller without a motion planner (eval) uses this instead. The fixed
parent-to-child transform of every arm joint is read once off the live articulation
(``T_child = T_parent * F * Rz(q)``), after which FK of any joint vector is a chain of 4x4
products. Poses come out in the robot root frame, ``[x, y, z, qw, qx, qy, qz]``, like the
recorder's other robot-frame columns.

The calibration assumes each arm joint turns about its child link's z axis (the Franka/FR3
URDF convention); :meth:`ArticulationEEFK.check` measures it against the observed EE pose and
the FK disables itself (zero-filled columns, as without FK) if the error is not negligible.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

log = logging.getLogger(__name__)

# FK vs observed EE pose beyond this (m) means the joint-axis assumption does not hold
MAX_FK_ERROR_M = 1e-3


def _pose_to_matrix(pose: torch.Tensor) -> torch.Tensor:
    """``[..., 7]`` ``[x, y, z, qw, qx, qy, qz]`` -> ``[..., 4, 4]`` (float64)."""
    from isaaclab.utils.math import make_pose, matrix_from_quat

    pose = pose.to(torch.float64)
    return make_pose(pose[..., :3], matrix_from_quat(pose[..., 3:7]))


def _rot_z(q: torch.Tensor) -> torch.Tensor:
    """``[N]`` angles -> ``[N, 4, 4]`` rotations about z (float64)."""
    from isaaclab.utils.math import make_pose, matrix_from_euler

    q = q.to(torch.float64)
    zero = torch.zeros_like(q)
    rot = matrix_from_euler(torch.stack([zero, zero, q], dim=-1), "XYZ")
    return make_pose(torch.stack([zero] * 3, dim=-1), rot)


class ArticulationEEFK:
    """``ee_target_fk`` callable for ``IsaacDataRecorderLeRobot`` built from a live articulation.

    Calibrates lazily on the first call (the scene must hold a physically valid state by then,
    i.e. after a reset), from env 0. The chain comes from the robot config
    (``BaseRobotCfg.arm_joint_names`` / ``arm_link_names``): ``arm_link_names[i]`` is the child
    link of ``arm_joint_names[i]``. The chain starts at the articulation's root link, so the base
    link's name (``panda_link0`` or ``base_link``, depending on the USD) is never needed.
    """

    def __init__(
        self,
        articulation: Any,
        ee_body_name: str,
        arm_joint_names: Sequence[str],
        arm_link_names: Sequence[str],
    ) -> None:
        if len(arm_link_names) != len(arm_joint_names):
            raise ValueError(
                f"arm_link_names needs one child link per arm joint, got "
                f"{len(arm_link_names)} links for {len(arm_joint_names)} joints"
            )
        self._art = articulation
        self._ee_body_name = ee_body_name
        self._joint_names = tuple(arm_joint_names)
        self._link_names = tuple(arm_link_names)
        self._fixed: list[torch.Tensor] | None = None
        self._root_to_base: torch.Tensor | None = None
        self._ee_offset: torch.Tensor | None = None
        self.disabled = False

    def _calibrate(self) -> None:
        art = self._art
        bodies = list(art.body_names)
        joints = list(art.joint_names)
        link_ids = [bodies.index(n) for n in self._link_names]
        ee_id = bodies.index(self._ee_body_name)
        self._joint_ids = [joints.index(n) for n in self._joint_names]
        poses = _pose_to_matrix(art.data.body_link_pose_w[0])  # [B, 4, 4]
        root = _pose_to_matrix(art.data.root_link_pose_w[0])
        q = art.data.joint_pos[0, self._joint_ids].to(torch.float64)
        # Fixed base: everything from the root link to the first arm joint is one rigid transform.
        frames = [root] + [poses[i] for i in link_ids]
        self._root_to_base = torch.eye(4, dtype=torch.float64, device=root.device)
        self._fixed = [
            torch.linalg.inv(frames[j])
            @ frames[j + 1]
            @ torch.linalg.inv(_rot_z(q[j : j + 1])[0])
            for j in range(len(self._joint_ids))
        ]
        self._ee_offset = torch.linalg.inv(poses[link_ids[-1]]) @ poses[ee_id]

    def _chain(self, q: torch.Tensor) -> torch.Tensor:
        """``[N, 7]`` joint angles -> ``[N, 4, 4]`` EE poses in the robot root frame."""
        assert self._fixed is not None and self._root_to_base is not None
        t = self._root_to_base.expand(q.shape[0], 4, 4)
        for j, fixed in enumerate(self._fixed):
            t = t @ fixed @ _rot_z(q[:, j])
        return t @ self._ee_offset

    def check(self) -> float:
        """Max position error (m) of FK(current joints) against the observed EE pose, over all
        envs; disables the FK when it exceeds :data:`MAX_FK_ERROR_M`."""
        if self._fixed is None:
            self._calibrate()
        art = self._art
        q = art.data.joint_pos[:, self._joint_ids].to(torch.float64)
        fk = self._chain(q)
        root = _pose_to_matrix(art.data.root_link_pose_w)
        ee = torch.linalg.inv(root) @ _pose_to_matrix(
            art.data.body_link_pose_w[:, list(art.body_names).index(self._ee_body_name)]
        )
        err = float((fk[:, :3, 3] - ee[:, :3, 3]).norm(dim=-1).max())
        if err > MAX_FK_ERROR_M:
            log.error(
                "articulation FK disagrees with the observed EE pose by %.4f m; "
                "action.ee_target / relative_ee_target will be zero-filled",
                err,
            )
            self.disabled = True
        return err

    def __call__(self, joint_targets: np.ndarray) -> np.ndarray:
        if self._fixed is None:
            self._calibrate()
            self.check()
        if self.disabled:
            raise RuntimeError("articulation FK disabled (failed its check against the scene)")
        from isaaclab.utils.math import quat_from_matrix

        q = torch.as_tensor(
            np.asarray(joint_targets), dtype=torch.float64, device=self._fixed[0].device
        )
        t = self._chain(q)
        quat = quat_from_matrix(t[:, :3, :3])
        quat = torch.where(quat[:, :1] < 0, -quat, quat)
        return torch.cat([t[:, :3, 3], quat], dim=-1).to(torch.float32).cpu().numpy()
