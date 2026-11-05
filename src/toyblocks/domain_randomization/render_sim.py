"""The simulator view render drives: a ToyBlocks autogen env, without TAMP.

Render used TAMP's ``tmp3.Backends.IsaacLabVectorized.simulator.IsaacLabSimulatorVectorized``
(pinned ea570fd2) as a wrapper around the gym env, but only ever touched a small part of it.
:class:`RenderSim` is that part, with the same behaviour:

- ``env`` / ``base_env`` / ``sim`` (the ``SimulationContext``) / ``scene`` / ``device`` /
  ``num_envs`` (taken from the scene when it disagrees) / ``env_indices`` / ``cameras``;
- construction resets the sim and, unless ``skip_initial_step``, steps it 10 times without
  recording; the robot base pose of every env is then cached in ``env_base_poses``;
- ``step(num_steps, record_data, active_env_indices)``: one physics + render step each, the
  scene updated by ``physics_dt * PHYS_STEP_PER_RENDER_INTERVAL``, and while ``execution`` is
  set the ``data_recorder`` records the selected envs (``record_data`` True / False / a bool
  tensor);
- ``set_rigid_object_pose(name, pose, env_idx, local)``: rigid objects get pose and zero
  velocity, ``XFormPrim`` objects a world pose; ``local`` composes the env base pose (full SE3);
- ``prim_scene_obj_mapping`` / ``obj_prim_scene_mapping`` (collider prim path <-> scene name,
  see :func:`fix_object_name_mappings`);
- ``close()`` finalizes the recorder.

TAMP's wrapper also built its cuRobo robot model on construction. That set the arm's PD gains to
the ``medium`` profile (``TMP3_MOTION_PROFILE`` overrides it) and teleported the arm to a start
pose. Render writes the arm's joint state every frame, so the pose is irrelevant; the gains
matter only with ``--physics_step``, so :class:`RenderSim` applies the same profile.
"""

from __future__ import annotations

import logging
import os
import sys
from collections.abc import Mapping
from typing import Any

import torch

from toyblocks.vecgen.math import quat_apply, quat_mul

logger = logging.getLogger(__name__)

#: Physics steps per rendered frame (TAMP's constant; the scene update ``dt`` scales by it).
PHYS_STEP_PER_RENDER_INTERVAL = 4
#: Arm gain profile TAMP's robot model applied at construction (``TMP3_MOTION_PROFILE``).
_DEFAULT_MOTION_PROFILE = "medium"


def _collider_prim_paths(prim: Any) -> list[str]:
    """Paths of every collider prim beneath *prim*: the children of any ``Collisions`` scope."""
    paths: list[str] = []
    for child in prim.GetChildren():
        if child.GetName() == "Collisions":
            paths.extend(str(collider.GetPath()) for collider in child.GetChildren())
        else:
            paths.extend(_collider_prim_paths(child))
    return paths


def fix_object_name_mappings(scene: Any, rigid_objects: Mapping[str, Any]) -> dict[str, str]:
    """Collider prim path (env 0) -> scene name, for every rigid object.

    Cuboid spawns keep their collision mesh at ``<prim>/geometry/mesh``; USD-asset spawns (the
    letter blocks) nest colliders under ``<prim>/<AssetName>/Collisions/<collider>``.
    """
    stage = scene.stage
    mapping: dict[str, str] = {}
    for scene_name, rigid_object in rigid_objects.items():
        stage_prim_path = rigid_object.cfg.prim_path.replace(".*", "0")
        stage_prim = stage.GetPrimAtPath(stage_prim_path)
        if not (stage_prim and stage_prim.IsValid()):
            continue
        geometry_prim = stage_prim.GetChild("geometry")
        if geometry_prim and geometry_prim.IsValid():
            mapping[str(stage_prim_path + "/geometry/mesh")] = scene_name
            continue
        collider_paths = _collider_prim_paths(stage_prim)
        if collider_paths:
            for collider_path in collider_paths:
                mapping[collider_path] = scene_name
            continue
        logger.warning(
            "fix_object_name_mappings: '%s' at '%s' has no 'geometry' child and no Collisions "
            "scope",
            scene_name,
            stage_prim_path,
        )
    return mapping


class RenderSim:
    """Render's view of a ToyBlocks env (see the module docstring)."""

    def __init__(self, env: Any, num_envs: int = 1, skip_initial_step: bool = False) -> None:
        from isaaclab.sensors.camera.camera import Camera

        self.data_recorder: Any | None = None
        self.execution = False
        self.env = env
        self.base_env = env.unwrapped
        self.sim = self.base_env.sim
        self.device = self.sim.device
        self.scene = self.base_env.scene
        self.num_envs = int(num_envs)
        scene_num_envs = getattr(getattr(self.scene, "cfg", None), "num_envs", None)
        if scene_num_envs is not None and scene_num_envs != self.num_envs:
            print(
                f"Warning: Scene created with num_envs={scene_num_envs}, but simulator expects "
                f"{self.num_envs}; using the scene's"
            )
            self.num_envs = int(scene_num_envs)
        self.env_indices: list[int] = list(range(self.num_envs))

        self.prim_scene_obj_mapping = fix_object_name_mappings(self.scene, self.scene.rigid_objects)
        self.obj_prim_scene_mapping = {v: k for k, v in self.prim_scene_obj_mapping.items()}
        self.sim.reset()
        self.cameras: dict[str, Any] = {
            name: sensor
            for name, sensor in self.scene.sensors.items()
            if isinstance(sensor, Camera)
        }
        if not skip_initial_step:
            self.step(num_steps=10, record_data=False)
        self._apply_motion_profile()

        robot_pose = self.scene["robot"].data.root_pose_w
        self.env_base_poses: dict[int, torch.Tensor] = {
            i: robot_pose[i, :] for i in self.env_indices
        }

    def _apply_motion_profile(self) -> None:
        """The arm gains TAMP's robot model set at construction (targets snapped to the arm)."""
        from toyblocks.utils.stiffness_profiles import get_profile_tensors

        name = os.environ.get("TMP3_MOTION_PROFILE", _DEFAULT_MOTION_PROFILE).strip()
        name = name or _DEFAULT_MOTION_PROFILE
        robot = self.base_env.robot
        art = robot.articulation
        arm_ids, _ = art.find_joints(list(robot.cfg.arm_joint_names), preserve_order=True)
        env_ids = torch.arange(self.num_envs, device=self.device)
        q = art.data.joint_pos[:, arm_ids]
        art.set_joint_position_target(q, joint_ids=arm_ids, env_ids=env_ids)
        art.set_joint_velocity_target(torch.zeros_like(q), joint_ids=arm_ids, env_ids=env_ids)
        stiffness, damping, effort, velocity = get_profile_tensors(
            name, self.device, batch_size=self.num_envs
        )
        art.write_joint_velocity_limit_to_sim(velocity, joint_ids=arm_ids, env_ids=env_ids)
        art.write_joint_stiffness_to_sim(stiffness, joint_ids=arm_ids, env_ids=env_ids)
        art.write_joint_damping_to_sim(damping, joint_ids=arm_ids, env_ids=env_ids)
        art.write_joint_effort_limit_to_sim(effort, joint_ids=arm_ids, env_ids=env_ids)
        art.write_data_to_sim()

    # -- stepping ------------------------------------------------------------------------------
    def step(
        self,
        num_steps: int = 1,
        record_data: bool | torch.Tensor = True,
        active_env_indices: list[int] | None = None,
    ) -> None:
        """``num_steps`` physics + render steps; records while ``execution`` is set (see module doc)."""
        for _ in range(num_steps):
            self.sim.step()
            self.scene.update(self.sim.get_physics_dt() * PHYS_STEP_PER_RENDER_INTERVAL)
            if not (self.execution and self.data_recorder):
                continue
            env_indices = active_env_indices if active_env_indices is not None else self.env_indices
            if isinstance(record_data, torch.Tensor):
                flags = record_data.detach().cpu().numpy()
                record = [i for i in env_indices if i < len(flags) and bool(flags[i])]
            elif record_data:
                record = list(env_indices)
            else:
                record = []
            if record:
                self.data_recorder.record_step_batched(record)

    # -- frames --------------------------------------------------------------------------------
    def _get_env_base_pose(self, env_idx: int) -> torch.Tensor:
        """The robot base pose of ``env_idx`` (world frame, xyz + wxyz), cached at construction."""
        return self.env_base_poses[env_idx]

    def _transform_to_world_frame(self, local_pose: torch.Tensor, env_idx: int) -> torch.Tensor:
        """``world = base * local`` (full SE3) for one ``[7]`` pose."""
        base = self._get_env_base_pose(env_idx)
        world_pos = quat_apply(base[3:], local_pose[:3]) + base[:3]
        world_quat = quat_mul(base[3:], local_pose[3:])
        return torch.cat([world_pos, world_quat])

    def set_rigid_object_pose(
        self,
        obj_name: str,
        pose: Any,
        env_idx: int | None = None,
        local: bool = False,
    ) -> None:
        """Teleport ``obj_name`` in ``env_idx`` (``[7]``) or in every env (``[num_envs, 7]``)."""
        from isaaclab.assets.rigid_object.rigid_object import RigidObject
        from isaacsim.core.prims.impl.xform_prim import XFormPrim

        if not isinstance(pose, torch.Tensor):
            pose = torch.tensor(pose, device=self.device, dtype=torch.float32)
        pose = pose.to(device=self.device)
        env_idx = 0 if env_idx is None else env_idx
        if local:
            pose = self._transform_to_world_frame(pose, env_idx)
        if obj_name not in self.scene.keys():  # noqa: SIM118 - InteractiveScene has keys(), no __contains__
            raise ValueError(f"Rigid object '{obj_name}' not found in the simulator.")
        obj = self.scene[obj_name]
        if isinstance(obj, RigidObject):
            pose = pose.to(device=obj.device)
            if pose.ndim == 1:
                env_ids = torch.tensor([env_idx], device=obj.device)
                obj.write_root_pose_to_sim(pose.unsqueeze(0), env_ids=env_ids)
                obj.write_root_velocity_to_sim(
                    torch.zeros(1, 6, device=obj.device), env_ids=env_ids
                )
            elif pose.ndim == 2 and pose.shape[0] == self.num_envs:
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros(self.num_envs, 6, device=obj.device))
            else:
                raise ValueError(
                    f"Invalid pose shape: {tuple(pose.shape)}. Expected (7,) or ({self.num_envs}, 7)"
                )
        elif isinstance(obj, XFormPrim):
            row = pose if pose.ndim == 1 else pose[0]
            obj.set_world_poses(position=row[:3].unsqueeze(0), quaternion=row[3:].unsqueeze(0))

    # -- lifecycle -----------------------------------------------------------------------------
    def close(self) -> None:
        """Finalize the recorder, if one is attached (Isaac itself is left to the caller)."""
        if self.data_recorder:
            self.data_recorder.finalize()
        sys.stdout.flush()
