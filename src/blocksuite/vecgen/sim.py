"""Thin batched interface over an Isaac Lab autogen env for the vecgen engine.

Everything here is tensor-batched over envs. One *tick* is ``sim.step()`` = 4 physics
substeps at 100 Hz (0.04 s), the same cadence the TAMP orchestrator uses, so stiffness
profiles, gripper wait counts and the recorder's ``fps=25`` carry over unchanged.

Frames: the robot base sits at the env origin with identity rotation, so "base frame"
is world minus ``scene.env_origins``. Quaternions are wxyz.

``launch_app()`` must have been called before this module is imported (it imports
isaaclab lazily inside ``make_env`` to keep that order explicit).
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 0.7  # ROBOTIQ_DEFAULT_CLOSE_TARGET in tmp3.Backends.IsaacLab.robot
GRIPPER_WAIT_TICKS = 6  # 24 physics steps, as GripperActionMarker(wait_steps=24) // 4
PHYSICS_STEPS_PER_TICK = 4
TICK_DT = 0.04
ARM_JOINT_NAMES = [f"panda_joint{i}" for i in range(1, 8)]
GRIPPER_JOINT_NAME = "finger_joint"
EE_BODY_NAME = "robotiq_arg2f_base_link"


#: Cameras vecgen spawns (and so records) with ``cameras=True``; the scene's left camera is not
#: needed in the videos. Names are BaseEnvCfg.camera_cfgs keys, which are only filled in (scene
#: preset + robot cameras) when the env is built, hence a fixed list.
VIDEO_CAMERAS: tuple[str, ...] = ("front_camera", "gripper_camera")


def make_env(
    env_path: str | Path, num_envs: int, device: str, cameras: bool = False
) -> Any:
    """Build an autogen env package's env (``cameras``: spawn the package's :data:`VIDEO_CAMERAS`)."""
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    env_path = Path(env_path)
    spec = importlib.util.spec_from_file_location(env_path.name, env_path / "__init__.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load env package at {env_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[env_path.name] = module  # the entry point is "<pkg>.env:<pkg>_Env"
    spec.loader.exec_module(module)  # registers <pkg>-v0
    task_id = f"{os.path.basename(env_path)}-v0"
    env_cfg = parse_env_cfg(task_id, device=device, num_envs=num_envs, use_fabric=True)
    env_cfg.always_spawn_cameras = cameras
    if cameras:
        env_cfg.spawn_only_cameras = VIDEO_CAMERAS
    env_cfg.enable_visual_markers = False
    return gym.make(task_id, cfg=env_cfg)


@dataclass
class ProfileTensors:
    stiffness: Tensor
    damping: Tensor
    effort: Tensor
    velocity_limit: Tensor


#: tanh scale (rad) of the friction-compensation push; see VecSim.friction_comp_offset.
FRICTION_COMP_ERR_SCALE: float = 0.003
#: |joint velocity| (rad/s) above which dynamic rather than static friction applies.
FRICTION_COMP_MOVING_VEL: float = 0.01


class VecSim:
    """Batched reads, writes, stepping and snapshots on an autogen env."""

    def __init__(self, env: Any, device: torch.device) -> None:
        self.env = env
        self.base_env = env.unwrapped
        self.scene = self.base_env.scene
        self.sim = self.base_env.sim
        self.device = device
        self.art = self.scene["robot"]
        self.num_envs = int(self.scene.num_envs)
        self.arm_ids, _ = self.art.find_joints(ARM_JOINT_NAMES)
        self.arm_ids = list(self.arm_ids)
        gripper_ids, _ = self.art.find_joints([GRIPPER_JOINT_NAME])
        self.gripper_id = int(gripper_ids[0])
        ee_ids, _ = self.art.find_bodies([EE_BODY_NAME])
        self.ee_body_id = int(ee_ids[0])
        self.env_origins: Tensor = self.scene.env_origins
        self.object_names: list[str] = list(self.scene.rigid_objects.keys())
        self.block_names: list[str] = [n for n in self.object_names if n.startswith("block")]
        self._arm_idx = torch.as_tensor(self.arm_ids, dtype=torch.long, device=device)
        self._all = torch.arange(self.num_envs, device=device)
        self._gripper_target = torch.full((self.num_envs, 1), GRIPPER_OPEN, device=device)
        # collider edge lengths [N, K, 3] (block_names order), set by measure_block_dims
        self.block_dims: Tensor | None = None

    # ----------------------------------------------------------------- reads
    def joint_pos(self) -> Tensor:
        """Arm joint positions ``[N, 7]``."""
        return self.art.data.joint_pos[:, self.arm_ids]

    def joint_vel(self) -> Tensor:
        return self.art.data.joint_vel[:, self.arm_ids]

    def gripper_pos(self) -> Tensor:
        """``[N]`` finger joint position (0 open .. ~0.7 closed)."""
        return self.art.data.joint_pos[:, self.gripper_id]

    def ee_pose(self) -> Tensor:
        """End-effector link pose in the base frame ``[N, 7]`` (xyz, wxyz)."""
        pose = self.art.data.body_link_pose_w[:, self.ee_body_id].clone()
        pose[:, :3] -= self.env_origins
        return pose

    def object_pose(self, name: str) -> Tensor:
        """Rigid object root pose in the base frame ``[N, 7]``."""
        pose = self.scene.rigid_objects[name].data.root_pose_w.clone()
        pose[:, :3] -= self.env_origins
        return pose

    def object_poses(self, names: list[str] | None = None) -> Tensor:
        """Stack of object poses ``[N, K, 7]`` in the base frame."""
        names = self.block_names if names is None else names
        return torch.stack([self.object_pose(n) for n in names], dim=1)

    def object_velocity(self, name: str) -> Tensor:
        return self.scene.rigid_objects[name].data.root_vel_w

    def collider_bounds(self, name: str) -> tuple[Tensor, Tensor]:
        """Axis-aligned bounds ``(lo, hi)`` ``[N, 3]`` of rigid object ``name``'s collision
        geometry in its body frame, read from the spawned stage.

        Only prims with ``PhysicsCollisionAPI`` count (visual-only parts, e.g. a letter
        block's decals, do not); the body prim's own scale is applied, its pose is not.
        """
        import omni.usd
        from pxr import Gf, Usd, UsdGeom, UsdPhysics

        stage = omni.usd.get_context().get_stage()
        cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[
                UsdGeom.Tokens.default_,
                UsdGeom.Tokens.render,
                UsdGeom.Tokens.proxy,
                UsdGeom.Tokens.guide,
            ],
        )
        lo = torch.zeros(self.num_envs, 3)
        hi = torch.zeros(self.num_envs, 3)
        pattern = self.scene.rigid_objects[name].cfg.prim_path
        for e in range(self.num_envs):
            path = pattern.replace("env_.*", f"env_{e}").replace("[^/]+", str(e))
            root = stage.GetPrimAtPath(path)
            if not root.IsValid():
                raise RuntimeError(f"prim {path} not found")
            rng = Gf.Range3d()
            for prim in Usd.PrimRange(root):
                if prim.HasAPI(UsdPhysics.CollisionAPI) and prim.IsA(UsdGeom.Gprim):
                    rng.UnionWith(cache.ComputeRelativeBound(prim, root).ComputeAlignedRange())
            if rng.IsEmpty():
                raise RuntimeError(f"{path} has no collision geometry")
            scale = Gf.Transform(UsdGeom.Xformable(root).GetLocalTransformation()).GetScale()
            mn, mx = rng.GetMin(), rng.GetMax()
            for a in range(3):
                lo[e, a] = mn[a] * scale[a]
                hi[e, a] = mx[a] * scale[a]
        return lo.to(self.device), hi.to(self.device)

    def measure_block_dims(self) -> tuple[Tensor, Tensor]:
        """Collider edge lengths ``[N, K, 3]`` (body frame, ``block_names`` order) and the
        collider centre's offset from the body origin ``[N, K, 3]``. Planning uses these
        instead of the package's nominal sizes: the toy-block spawn scales the
        decal-inclusive bounding box to the nominal size, so a letter block's collider is
        smaller than it."""
        dims, centre = [], []
        for name in self.block_names:
            lo, hi = self.collider_bounds(name)
            dims.append(hi - lo)
            centre.append(0.5 * (hi + lo))
        self.block_dims = torch.stack(dims, dim=1)
        return self.block_dims, torch.stack(centre, dim=1)

    def measure_table_top(self, name: str = "workspace_table") -> Tensor:
        """Height ``[N]`` of the table collider's top face above the table body origin."""
        _lo, hi = self.collider_bounds(name)
        return hi[:, 2]

    # ---------------------------------------------------------------- writes
    def apply_profile(self, name: str, env_ids: Tensor | None = None) -> None:
        """Write a named stiffness profile's gains and snap targets to the current state."""
        from blocksuite.utils.stiffness_profiles import get_profile_tensors

        ids = self._all if env_ids is None else env_ids
        if ids.numel() == 0:
            return
        q = self.art.data.joint_pos[ids][:, self.arm_ids]
        self.art.set_joint_position_target(q, joint_ids=self.arm_ids, env_ids=ids)
        self.art.set_joint_velocity_target(torch.zeros_like(q), joint_ids=self.arm_ids, env_ids=ids)
        kp, kd, effort, vlim = get_profile_tensors(name, self.device, batch_size=int(ids.numel()))
        self.art.write_joint_velocity_limit_to_sim(vlim, joint_ids=self.arm_ids, env_ids=ids)
        self.art.write_joint_stiffness_to_sim(kp, joint_ids=self.arm_ids, env_ids=ids)
        self.art.write_joint_damping_to_sim(kd, joint_ids=self.arm_ids, env_ids=ids)
        self.art.write_joint_effort_limit_to_sim(effort, joint_ids=self.arm_ids, env_ids=ids)
        self.art.write_data_to_sim()

    def command_arm(
        self,
        q_des: Tensor,
        env_ids: Tensor,
        qd: Tensor | None = None,
        qdd: Tensor | None = None,
        target_ff: bool = True,
        friction_comp: bool = False,
    ) -> None:
        """Command arm joint targets for ``env_ids`` with optional target feed-forward.

        ``q_cmd = q_des + (M qdd + C qd + kd qd) / kp`` (gravity is disabled on the arm),
        the same law as ``IsaacRobotVectorized.set_arm_ff_target``. ``friction_comp`` adds
        :meth:`friction_comp_offset` so the arm reaches ``q_des`` despite joint friction.
        Does not write to sim; call :meth:`flush` once per tick after all commands.
        """
        q_cmd = q_des
        if friction_comp:
            q_cmd = q_cmd + self.friction_comp_offset(q_des, env_ids, qd)
        if target_ff and qd is not None and qdd is not None:
            mass = self.art.root_physx_view.get_generalized_mass_matrices()
            view = self.art.root_physx_view
            if hasattr(view, "get_coriolis_and_centrifugal_compensation_forces"):
                coriolis = view.get_coriolis_and_centrifugal_compensation_forces()
            else:
                coriolis = view.get_coriolis_and_centrifugal_forces()
            m_arm = mass.index_select(1, self._arm_idx).index_select(2, self._arm_idx)[env_ids]
            cor = coriolis.index_select(1, self._arm_idx)[env_ids]
            kp = self.art.data.joint_stiffness[env_ids][:, self.arm_ids]
            kd = self.art.data.joint_damping[env_ids][:, self.arm_ids]
            tau_ff = torch.bmm(m_arm, qdd.unsqueeze(-1)).squeeze(-1) + cor + kd * qd
            q_cmd = q_cmd + tau_ff / kp.clamp_min(1e-6)
        self.art.set_joint_position_target(q_cmd, joint_ids=self.arm_ids, env_ids=env_ids)

    def friction_comp_offset(self, q_des: Tensor, env_ids: Tensor, qd: Tensor | None = None) -> Tensor:
        """Target offset that cancels the arm's joint friction: ``tau_f / kp`` per joint.

        With the identified joint friction on the arm (Coulomb ~1.5-1.8 Nm, robot configs'
        ``friction``), a PD joint at rest stalls anywhere inside ``friction / kp`` of its
        target -- ~0.03 rad on the kp=50 wrist at the medium profile, so grasps and places
        landed 4-6 mm / ~1 deg off the commanded pose. This pushes with the friction torque
        toward the target: ``tau_f = f_c * tanh(err / FRICTION_COMP_ERR_SCALE)`` where
        ``err = q_des - q`` and ``f_c`` is the static coefficient at rest, the dynamic one
        while the joint moves; plus viscous ``b * qd`` along the planned velocity. ``tanh``
        fades the push out near the target, leaving a residual band of about
        ``f_c / (kp + f_c / FRICTION_COMP_ERR_SCALE)`` instead of ``f_c / kp``.
        """
        data = self.art.data
        arm = self.arm_ids
        q = data.joint_pos[env_ids][:, arm]
        kp = data.joint_stiffness[env_ids][:, arm].clamp_min(1e-6)
        f_static = data.joint_friction_coeff[env_ids][:, arm]
        f_dyn = getattr(data, "joint_dynamic_friction_coeff", None)
        f_dyn = f_static if f_dyn is None else f_dyn[env_ids][:, arm]
        moving = data.joint_vel[env_ids][:, arm].abs() > FRICTION_COMP_MOVING_VEL
        f_c = torch.where(moving, f_dyn, f_static)
        tau = f_c * torch.tanh((q_des - q) / FRICTION_COMP_ERR_SCALE)
        visc = getattr(data, "joint_viscous_friction_coeff", None)
        if qd is not None and visc is not None:
            tau = tau + visc[env_ids][:, arm] * qd
        return tau / kp

    def set_arm_joint_friction(self, scale: float) -> None:
        """Scale the arm's static / dynamic / viscous joint friction in sim (A/B: 0 = off)."""
        data = self.art.data
        arm = self.arm_ids
        for name, writer in (
            ("joint_friction_coeff", "write_joint_friction_coefficient_to_sim"),
            ("joint_dynamic_friction_coeff", "write_joint_dynamic_friction_coefficient_to_sim"),
            ("joint_viscous_friction_coeff", "write_joint_viscous_friction_coefficient_to_sim"),
        ):
            values = getattr(data, name, None)
            if values is not None and hasattr(self.art, writer):
                getattr(self.art, writer)(values[:, arm] * scale, joint_ids=arm)

    def set_arm_armature(self, scale: float) -> None:
        """Scale the arm's joint armature in sim (0 = none, as the pre-sysid robot of the Sep data)."""
        armature = getattr(self.art.data, "joint_armature", None)
        if armature is not None:
            self.art.write_joint_armature_to_sim(armature[:, self.arm_ids] * scale, joint_ids=self.arm_ids)

    def arm_dynamics_summary(self) -> str:
        """Env-0 arm stiffness / damping / friction / armature as written to sim (for the run log)."""
        data = self.art.data
        parts = []
        for name in ("joint_stiffness", "joint_damping", "joint_friction_coeff", "joint_dynamic_friction_coeff",
                     "joint_viscous_friction_coeff", "joint_armature"):
            values = getattr(data, name, None)
            if values is not None:
                parts.append(f"{name.removeprefix('joint_')}={[round(float(v), 3) for v in values[0, self.arm_ids]]}")
        return " ".join(parts)

    def command_gripper(self, closed: Tensor, env_ids: Tensor) -> None:
        """``closed`` is a bool ``[B]``; sets the finger joint target (no write yet)."""
        target = torch.where(closed, GRIPPER_CLOSE, GRIPPER_OPEN).to(self.device).unsqueeze(-1)
        self._gripper_target[env_ids] = target
        self.art.set_joint_position_target(target, joint_ids=[self.gripper_id], env_ids=env_ids)

    def command_gripper_value(self, target: Tensor, env_ids: Tensor) -> None:
        """Explicit finger joint targets ``[B]`` (e.g. 0.9 for the flip's full close)."""
        t = target.to(self.device).reshape(-1, 1)
        self._gripper_target[env_ids] = t
        self.art.set_joint_position_target(t, joint_ids=[self.gripper_id], env_ids=env_ids)

    def flush(self) -> None:
        self.art.write_data_to_sim()

    # -------------------------------------------------------------- stepping
    def step(self) -> None:
        """One tick: 4 physics substeps with the current targets."""
        self.sim.step()
        self.scene.update(PHYSICS_STEPS_PER_TICK * self.sim.get_physics_dt())

    def forward(self) -> None:
        """Refresh derived buffers (link poses) after a teleport without stepping."""
        self.sim.forward()
        self.scene.update(0.0)

    # ------------------------------------------------------------ resets etc.
    def reset_to_instances(self, env_ids: Tensor, instance_ids: Tensor) -> None:
        """Load reset states ``instance_ids`` into ``env_ids``; the env raises if a state was
        sampled under a different scene profile than the env was spawned with."""
        self.base_env._reset_idx(env_ids=env_ids, instance_ids=instance_ids)
        self._gripper_target[env_ids] = GRIPPER_OPEN

    def snapshot(self, env_ids: Tensor) -> dict[str, Any]:
        """Batched state of ``env_ids`` (articulation + rigid objects), base-relative."""
        return self.base_env.get_state_dict(env_ids=env_ids, is_relative=True)

    def restore(self, state: dict[str, Any], env_ids: Tensor) -> None:
        """Write a :meth:`snapshot` back. Targets snap to the restored joint positions."""
        self.base_env.set_state_dict(state, env_ids=env_ids, is_relative=True)
        self.art.set_joint_position_target(
            self._gripper_target[env_ids], joint_ids=[self.gripper_id], env_ids=env_ids
        )
        self.art.write_data_to_sim()

    def num_reset_states(self) -> int:
        return int(self.base_env.num_reset_states)

    def _require_scene_profiles(self) -> None:
        if not hasattr(self.base_env, "env_scene_profile"):
            raise RuntimeError(
                f"env package {type(self.base_env).__name__} was generated before codegen v4 "
                "and records no per-env scene profile; regenerate it (generate_task_from_pddl.py "
                "or make_superset_env.py) so envs and reset states can be matched by value"
            )

    def env_slots(self) -> list[int]:
        """Spawn slot (scene profile) of every env, as the env package recorded it."""
        self._require_scene_profiles()
        return [int(slot) for slot in self.base_env.env_slots]

    def env_scene_profile(self, e: int) -> dict[str, Any]:
        """Block sizes, friction and table height env ``e`` was spawned with."""
        self._require_scene_profiles()
        return self.base_env.env_scene_profile(e)

    def reset_states_by_slot(self) -> dict[int, list[int]]:
        """Reset states each spawned slot may run, matched by profile value: a state's
        block poses are only valid at the block sizes / friction it was sampled under.
        Packages whose reset states record no profile keep the legacy ``id % K`` map."""
        slots = sorted(set(self.env_slots()))
        ok = self.base_env._reset_state_slot_compat()
        if ok is None:
            n_prof = int(self.base_env.MAX_NON_SIM_STATE_RANDOMIZATIONS)
            return {
                s: [i for i in range(self.num_reset_states()) if i % n_prof == s] for s in slots
            }
        return {s: ok[:, s].nonzero().flatten().tolist() for s in slots}

    def block_sizes(self) -> Tensor:
        """Nominal block dims ``[N, K, 3]`` (``block_names`` order) each env was spawned with."""
        out = torch.zeros(self.num_envs, len(self.block_names), 3)
        for e in range(self.num_envs):
            sizes = self.env_scene_profile(e)["block_sizes"]
            for j, name in enumerate(self.block_names):
                out[e, j] = torch.tensor(sizes[name], dtype=torch.float32)
        return out
