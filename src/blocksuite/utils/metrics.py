"""MetricsTracker: batched per-step metric collection for blocksuite environments.

Collects joint kinematics (position/velocity/acceleration norms), end-effector
kinematics, and collision statistics from Isaac Lab sensors.  All outputs are
``[num_envs]`` tensors suitable for the ``stats`` dict consumed by
:class:`~blocksuite.wrappers.record_metrics.RecordMetrics`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.sensors.contact_sensor import ContactSensor
    from isaaclab.sensors.frame_transformer import FrameTransformer


class MetricsTracker:
    """Batched metric recorder for blocksuite ``BaseEnv``.

    Instantiated once in ``BaseEnv.__init__`` and driven by ``step()`` /
    ``reset()`` calls from the environment.  Each ``step()`` returns a flat
    ``dict[str, Tensor[num_envs]]`` that is merged into the ``evaluate()``
    ``stats`` dict.
    """

    def __init__(
        self,
        robot: Articulation,
        ee_frame: FrameTransformer | None = None,
        contact_sensor: ContactSensor | None = None,
        contact_filter_sensors: dict[str, ContactSensor] | None = None,
        *,
        track_joint_kinematics: bool = True,
        track_ee_kinematics: bool = True,
        track_collisions: bool = False,
        dt: float = 0.01,
        num_envs: int = 1,
        device: torch.device | str = "cuda",
    ):
        self._robot = robot
        self._ee_frame = ee_frame
        self._contact_sensor = contact_sensor
        self._contact_filter_sensors: dict[str, ContactSensor] = contact_filter_sensors or {}

        self._track_joint = track_joint_kinematics
        self._track_ee = track_ee_kinematics and ee_frame is not None
        self._track_collisions = track_collisions and contact_sensor is not None

        self._dt = dt
        self._num_envs = num_envs
        self._device = torch.device(device)

        # -- joint acceleration: previous velocity buffer
        if self._track_joint:
            num_joints = robot.data.joint_pos.shape[-1]
            self._prev_joint_vel = torch.zeros(
                num_envs, num_joints, device=self._device
            )
            self._joint_names: list[str] = list(robot.joint_names) if hasattr(robot, "joint_names") else [f"joint_{i}" for i in range(num_joints)]

        # -- EE acceleration: previous position / velocity buffers
        if self._track_ee:
            self._prev_ee_pos = ee_frame.data.target_pos_w[:, 0].clone()
            self._prev_ee_vel = torch.zeros(num_envs, 3, device=self._device)

        # -- collision cumulative counters (per-episode, reset with env)
        if self._track_collisions:
            self._self_collision_count = torch.zeros(
                num_envs, device=self._device
            )
            self._env_collision_count = torch.zeros(
                num_envs, device=self._device
            )
            self._contact_body_names: list[str] = list(contact_sensor.body_names)
            first_filter_sensor = next(iter(self._contact_filter_sensors.values()), None)
            self._filter_names: list[str] = [
                p.rsplit("/", 1)[-1]
                for p in first_filter_sensor.cfg.filter_prim_paths_expr
            ] if first_filter_sensor is not None else []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def step(self) -> dict[str, torch.Tensor]:
        """Compute all enabled metrics for the current sim state.

        Returns a flat dict mapping metric names to ``[num_envs]`` tensors.
        """
        stats: dict[str, torch.Tensor] = {}
        if self._track_joint:
            stats.update(self._compute_joint_metrics())
        if self._track_ee:
            stats.update(self._compute_ee_metrics())
        if self._track_collisions:
            stats.update(self._compute_collision_metrics())
        return stats

    def reset(self, env_ids: torch.Tensor) -> None:
        """Zero out buffers for environments that just reset."""
        if self._track_joint:
            self._prev_joint_vel[env_ids] = 0.0
        if self._track_ee:
            self._prev_ee_pos[env_ids] = self._ee_frame.data.target_pos_w[env_ids, 0]
            self._prev_ee_vel[env_ids] = 0.0
        if self._track_collisions:
            self._self_collision_count[env_ids] = 0.0
            self._env_collision_count[env_ids] = 0.0

    # ------------------------------------------------------------------
    # Convenience helpers (for use in task evaluate())
    # ------------------------------------------------------------------

    def ee_to_object_distance(self, object_pos_w: torch.Tensor) -> torch.Tensor:
        """L2 distance from the end effector to an arbitrary world-frame position.

        Args:
            object_pos_w: ``[num_envs, 3]`` world-frame object positions.

        Returns:
            ``[num_envs]`` distance tensor.
        """
        ee_pos = self._ee_frame.data.target_pos_w[:, 0]
        return torch.linalg.norm(ee_pos - object_pos_w, dim=-1)

    def object_to_target_distance(
        self, object_pos_w: torch.Tensor, target_pos_w: torch.Tensor
    ) -> torch.Tensor:
        """L2 distance between two world-frame positions (e.g. object -> goal).

        Args:
            object_pos_w: ``[num_envs, 3]`` source positions.
            target_pos_w: ``[num_envs, 3]`` target positions.

        Returns:
            ``[num_envs]`` distance tensor.
        """
        return torch.linalg.norm(object_pos_w - target_pos_w, dim=-1)

    def ee_position(self) -> torch.Tensor:
        """Current end-effector world position.

        Returns:
            ``[num_envs, 3]`` position tensor.
        """
        return self._ee_frame.data.target_pos_w[:, 0]

    @property
    def robot(self):
        """Reference to the tracked :class:`Articulation`."""
        return self._robot

    # ------------------------------------------------------------------
    # Internal metric computation
    # ------------------------------------------------------------------

    def _compute_joint_metrics(self) -> dict[str, torch.Tensor]:
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        joint_accel = (joint_vel - self._prev_joint_vel) / self._dt
        self._prev_joint_vel = joint_vel.clone()

        stats: dict[str, torch.Tensor] = {
            "joint_pos_norm": torch.linalg.norm(joint_pos, dim=-1),
            "joint_vel_norm": torch.linalg.norm(joint_vel, dim=-1),
            "joint_accel_norm": torch.linalg.norm(joint_accel, dim=-1),
        }

        if self._robot.data.applied_torque is not None:
            stats["joint_effort_norm"] = torch.linalg.norm(
                self._robot.data.applied_torque, dim=-1
            )

        for j, name in enumerate(self._joint_names):
            stats[f"joint_pos/{name}"] = joint_pos[:, j]
            stats[f"joint_vel/{name}"] = joint_vel[:, j]
            stats[f"joint_accel/{name}"] = joint_accel[:, j]
            if self._robot.data.applied_torque is not None:
                stats[f"joint_effort/{name}"] = self._robot.data.applied_torque[:, j]

        return stats

    def _compute_ee_metrics(self) -> dict[str, torch.Tensor]:
        ee_pos = self._ee_frame.data.target_pos_w[:, 0]
        ee_quat = self._ee_frame.data.target_quat_w[:, 0]  # (w, x, y, z)

        ee_vel = (ee_pos - self._prev_ee_pos) / self._dt
        ee_accel = (ee_vel - self._prev_ee_vel) / self._dt
        self._prev_ee_pos = ee_pos.clone()
        self._prev_ee_vel = ee_vel.clone()

        return {
            "ee_pos_x": ee_pos[:, 0],
            "ee_pos_y": ee_pos[:, 1],
            "ee_pos_z": ee_pos[:, 2],
            "ee_quat_w": ee_quat[:, 0],
            "ee_quat_x": ee_quat[:, 1],
            "ee_quat_y": ee_quat[:, 2],
            "ee_quat_z": ee_quat[:, 3],
            "ee_vel_norm": torch.linalg.norm(ee_vel, dim=-1),
            "ee_accel_norm": torch.linalg.norm(ee_accel, dim=-1),
        }

    def _compute_collision_metrics(self) -> dict[str, torch.Tensor]:
        sensor = self._contact_sensor
        net_forces = sensor.data.net_forces_w  # [num_envs, num_bodies, 3]

        per_body_force_norm = torch.linalg.norm(net_forces, dim=-1)
        total_contact_force = per_body_force_norm.sum(dim=-1)

        force_threshold = sensor.cfg.force_threshold
        in_contact = per_body_force_norm > force_threshold
        bodies_in_contact = in_contact.sum(dim=-1).float()

        self._env_collision_count += bodies_in_contact

        stats: dict[str, torch.Tensor] = {
            "contact_force_norm": total_contact_force,
            "bodies_in_contact": bodies_in_contact,
            "env_collision_count": self._env_collision_count.clone(),
            "self_collision_count": self._self_collision_count.clone(),
        }

        # per-body 3D contact forces (from the multi-body sensor)
        for b, name in enumerate(self._contact_body_names):
            stats[f"contact_force/{name}/x"] = net_forces[:, b, 0]
            stats[f"contact_force/{name}/y"] = net_forces[:, b, 1]
            stats[f"contact_force/{name}/z"] = net_forces[:, b, 2]

        # per (body, filter) pair forces from per-body filter sensors
        for bname, filter_sensor in self._contact_filter_sensors.items():
            fm = filter_sensor.data.force_matrix_w
            if fm is None:
                continue
            # fm shape: [num_envs, 1, num_filters, 3] (single-body sensor)
            filter_force_norms = torch.linalg.norm(fm[:, 0], dim=-1)  # [num_envs, num_filters]
            self._self_collision_count += (filter_force_norms > force_threshold).sum(dim=-1).float()

            for f, fname in enumerate(self._filter_names):
                stats[f"contact_pair/{bname}/{fname}/x"] = fm[:, 0, f, 0]
                stats[f"contact_pair/{bname}/{fname}/y"] = fm[:, 0, f, 1]
                stats[f"contact_pair/{bname}/{fname}/z"] = fm[:, 0, f, 2]

        return stats
