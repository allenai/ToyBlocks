"""Replaying a recorded episode frame back into a live sim.

Moved verbatim out of scripts/rerender/rerender.py (the robot joint constants,
``_resolve_joint_ids`` and ``_set_table_height``) so that dagger -- and any other consumer
of this package -- can reproduce rerender's render-only replay without duplicating it.
rerender.py imports these names back under their original spellings, so
``rerender_from_params``'s ``T._resolve_joint_ids`` / ``T._set_table_height`` keep working.

``apply_recorded_frame`` is NEW and is not used by rerender: it composes exactly what
rerender's per-frame loop writes (arm joints, gripper drive + passive mimic linkage, object
poses in the LOCAL frame, and the per-episode table height) for callers that want one frame
rather than a whole render loop.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch


ARM_JOINTS = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
              "panda_joint5", "panda_joint6", "panda_joint7"]
GRIPPER_DRIVE_JOINT = "finger_joint"
# Robotiq passive linkage: the 5 unactuated gripper joints all follow finger_joint
# with multiplier +1.0 (verified by a physics sweep: finger=X -> every passive joint=X).
# Writing them directly (render-only) reproduces the closed-gripper geometry without
# physics. NOTE: the inner-finger joints are +1.0 (not -1.0); a wrong sign here splays
# the finger pads outward as the gripper closes ("inverted gripper / no contact").
GRIPPER_MIMIC = {
    "right_outer_knuckle_joint": 1.0,
    "left_inner_knuckle_joint": 1.0,
    "right_inner_knuckle_joint": 1.0,
    "left_inner_finger_joint": 1.0,
    "right_inner_finger_joint": 1.0,
}


def _resolve_joint_ids(sim):
    art = sim.base_env.robot.articulation
    arm_ids = art.find_joints(ARM_JOINTS)[0]
    drive_id = art.find_joints([GRIPPER_DRIVE_JOINT])[0]
    mimic_ids, mimic_signs = [], []
    for jname, sign in GRIPPER_MIMIC.items():
        try:
            mimic_ids.append(art.find_joints([jname])[0][0])
            mimic_signs.append(sign)
        except Exception:
            pass
    return art, arm_ids, drive_id, mimic_ids, mimic_signs


def _set_table_height(sim, env_ids: List[int], deltas: List[float]) -> None:
    """Override workspace_table height per env from recorded table_height_delta."""
    if "workspace_table" not in sim.scene.rigid_objects:
        return
    table = sim.scene.rigid_objects["workspace_table"]
    eids = torch.tensor(env_ids, device=sim.device)
    pose = table.data.root_pose_w[eids].clone()
    base_z = table.data.default_root_state[eids, 2] if hasattr(table.data, "default_root_state") else pose[:, 2]
    pose[:, 2] = base_z + torch.tensor(deltas, device=sim.device, dtype=pose.dtype)
    table.write_root_pose_to_sim(pose, env_ids=eids)


#: Where blocks outside an episode's problem are parked (env-local): far below the scene, past
#: every camera's view, one PARK_STEP apart so parked bodies never overlap (matters only with
#: physics). Written as a finite pose -- NaN poses in PhysX/Fabric are undefined behaviour.
PARK_LOCAL_Z = -100.0
PARK_STEP = 0.25


def park_inactive_objects(sim: Any, wave: Sequence[Dict[str, Any]],
                          active: Optional[Sequence[int]] = None) -> List[str]:
    """Move each env's ``parked_objects`` (see episodes.split_parked_objects) out of view.

    Returns the names parked in at least one env. Objects not in the scene are skipped.
    """
    envs = list(range(len(wave))) if active is None else list(active)
    names = sorted({n for i in envs for n in (wave[i].get("parked_objects") or ())})
    scene_keys = set(sim.scene.keys())
    origins = sim.scene.env_origins
    for k, name in enumerate(names):
        if name not in scene_keys:
            continue
        obj = sim.scene[name]
        ids = [i for i in envs if name in (wave[i].get("parked_objects") or ())]
        if not ids or not hasattr(obj, "write_root_pose_to_sim"):
            continue
        idx = torch.tensor(ids, device=sim.device)
        pose = torch.zeros(len(ids), 7, device=sim.device)
        pose[:, :3] = origins[idx]
        pose[:, 0] += PARK_STEP * k
        pose[:, 2] += PARK_LOCAL_Z
        pose[:, 3] = 1.0
        obj.write_root_pose_to_sim(pose, env_ids=idx)
        if hasattr(obj, "write_root_velocity_to_sim"):
            obj.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=sim.device), env_ids=idx)
    return names


def _robot_articulation(sim: Any) -> Any:
    """The robot articulation, whether given rerender's sim wrapper or a bare env.

    rerender always holds a wrapper exposing ``.base_env``; dagger drives the env object
    directly. Accept both rather than making the caller know which it has.
    """
    base = getattr(sim, "base_env", sim)
    robot = getattr(base, "robot", None)
    if robot is None:
        raise AttributeError(f"{type(sim).__name__} exposes no .robot to replay onto")
    return robot.articulation


def resolve_joint_ids_for(sim: Any):
    """``_resolve_joint_ids`` but tolerant of a bare env (see ``_robot_articulation``)."""
    art = _robot_articulation(sim)
    arm_ids = art.find_joints(ARM_JOINTS)[0]
    drive_id = art.find_joints([GRIPPER_DRIVE_JOINT])[0]
    mimic_ids, mimic_signs = [], []
    for jname, sign in GRIPPER_MIMIC.items():
        try:
            mimic_ids.append(art.find_joints([jname])[0][0])
            mimic_signs.append(sign)
        except Exception:  # noqa: BLE001 -- a gripper without this passive joint
            pass
    return art, arm_ids, drive_id, mimic_ids, mimic_signs


def _write_object_pose(sim: Any, name: str, pose7: torch.Tensor, env_idx: int,
                       local: bool = True) -> None:
    """Set one rigid object's root pose, with or without the template_env helper.

    ``set_rigid_object_pose`` is defined on the blocks template env, so dagger's own envs
    have it -- but an AUTOGEN env generated against upstream's BaseEnv does NOT. Fall back
    to the identical write the helper performs (env-origin offset for the local frame,
    then root pose + zeroed root velocity).
    """
    helper = getattr(sim, "set_rigid_object_pose", None)
    if helper is not None:
        helper(name, pose7, env_idx=env_idx, local=local)
        return
    device = sim.device
    env_ids = torch.tensor([env_idx], device=device, dtype=torch.long)
    pose = pose7.to(device).reshape(-1, 7).clone()
    if local:
        pose[:, :3] += sim.scene.env_origins[env_ids]
    asset = sim.scene[name]
    asset.write_root_pose_to_sim(pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=device),
                                     env_ids=env_ids)


def apply_recorded_frame(sim: Any, wave: Sequence[Dict[str, Any]], t: int = 0, *,
                         active: Optional[Sequence[int]] = None,
                         set_table: bool = True,
                         joint_ids: Optional[tuple] = None) -> List[int]:
    """Write frame ``t`` of each episode in ``wave`` into the sim. Returns the slots written.

    ``wave[i]`` is an episode record from :mod:`toyblocks.domain_randomization.episodes`
    and lands in env slot ``i``. This mirrors rerender's render-only replay body:

      * arm joints          <- wave[i]["joint"][t]
      * gripper drive       <- wave[i]["gripper"][t], plus the passive mimic joints
      * object poses        <- wave[i]["objects"][name][t], LOCAL frame (wxyz)
      * table height        <- wave[i]["table_height_delta"] (once, all slots)

    Velocities are written as zeros, exactly as rerender does -- this is a pose replay, not
    a physics rollout. Objects absent from the scene are skipped, which is what rerender does
    (``if name in sim.scene.keys()``).
    """
    if active is None:
        active = [i for i in range(len(wave)) if t < wave[i]["length"]]
    active = list(active)
    if not active:
        return []

    art, arm_ids, drive_id, mimic_ids, mimic_signs = (
        joint_ids if joint_ids is not None else resolve_joint_ids_for(sim))
    device = sim.device
    eids = torch.tensor(active, device=device)

    if t == 0:
        park_inactive_objects(sim, wave, active)
    if set_table:
        thd = [float(wave[i].get("table_height_delta", 0.0) or 0.0) for i in active]
        if any(thd):
            _set_table_height(sim, active, thd)

    arm = torch.tensor(np.stack([wave[i]["joint"][t] for i in active]),
                       device=device, dtype=torch.float32)
    art.write_joint_state_to_sim(arm, torch.zeros_like(arm), joint_ids=arm_ids, env_ids=eids)

    grip = torch.tensor([wave[i]["gripper"][t] for i in active],
                        device=device, dtype=torch.float32).reshape(-1, 1)
    art.write_joint_state_to_sim(grip, torch.zeros_like(grip), joint_ids=drive_id, env_ids=eids)
    for jid, sign in zip(mimic_ids, mimic_signs):
        mj = grip * sign
        art.write_joint_state_to_sim(mj, torch.zeros_like(mj), joint_ids=[jid], env_ids=eids)

    scene_keys = set(sim.scene.keys())
    for i in active:
        for name, poses in wave[i]["objects"].items():
            if name in scene_keys:
                _write_object_pose(
                    sim, name, torch.tensor(poses[t], device=device, dtype=torch.float32),
                    env_idx=i, local=True)
    return active


def _moving_cameras(episodes: List[Dict[str, Any]], cam_keys) -> set:
    """Cameras whose recorded pose changes over time (i.e. link-mounted, e.g. the
    gripper camera). These MUST be driven from the recorded pose each frame — a
    link-mounted camera otherwise lags the gripper (Isaac syncs it via scene.update,
    one render behind) and starts at its default pose on frame 0."""
    moving = set()
    for key in cam_keys:
        for e in episodes:
            cp = e["cam_pose"].get(key)
            if cp is not None and cp.shape[0] > 1 and float(np.abs(cp - cp[0:1]).max()) > 1e-4:
                moving.add(key)
                break
    return moving


# --- Per-episode randomization defaults (overridable in flags.yaml `randomize`) ---
# Angles are camera-local degrees; vertical_fov_deg perturbs the VERTICAL field of view.
# HDRI/skybox pool + table-material defaults, the pure HDRI resolver, and the SkyBuckets picker now
# live in the reusable toyblocks.domain_randomization package (shared with eval; no module globals).
# rerender keeps the old private names as thin adapters so every call site is unchanged -- a render
# render_params diff confirms byte-identical DR sampling.
from toyblocks.domain_randomization import resolve as _dr_resolve
from toyblocks.domain_randomization.assets import (
    default_hdri_dir as _dr_default_hdri_dir,
    default_table_materials as _dr_default_table_materials,
)
from toyblocks.domain_randomization.buckets import (
    SkyBuckets as _PkgSkyBuckets,
    bucket_weight as _bucket_weight,
    bucket_seq as _bucket_seq,
)

DEFAULT_HDRI_DIR = _dr_default_hdri_dir()
DEFAULT_TABLE_MATERIALS = _dr_default_table_materials()

SKYBOX_MANIFEST = _dr_resolve.SKYBOX_MANIFEST
_expand_path = _dr_resolve.expand_path
_resolve_hdri_names = _dr_resolve.resolve_hdri_names
_discover_datasets = _dr_resolve.discover_datasets
_files_under = _dr_resolve.files_under
_resolve_set_route = _dr_resolve.resolve_set_route
_shuffled_texture_order = _dr_resolve.shuffled_texture_order


def select_override_cameras(sim: Any, episodes: Sequence[Dict[str, Any]],
                            flag_camera_names: Sequence[str] = (),
                            match_all: bool = False) -> Dict[str, Any]:
    """The cameras to drive from the RECORDED pose, exactly as rerender chooses them.

    Mirrors rerender.replay_job: a camera qualifies when the scene has it as a sensor with
    ``set_world_poses``, the flags yaml does NOT override it, and at least one episode records
    a pose for it. Of those, MOVING (link-mounted, e.g. the gripper) cameras are always driven
    -- otherwise they lag the gripper and start at a default pose on frame 0 -- and the static
    ones only under ``match_all`` (rerender's --match_recorded_cameras).
    """
    flag_keys = {f"camera:{n}" for n in (flag_camera_names or ())}
    scene_cam_keys = [k for k, s in sim.scene.sensors.items()
                      if k.startswith("camera:") and hasattr(s, "set_world_poses")
                      and k not in flag_keys
                      and any(k in e.get("cam_pose", {}) for e in episodes)]
    moving = _moving_cameras(episodes, scene_cam_keys)
    return {k: sim.scene.sensors[k] for k in scene_cam_keys if k in moving or match_all}
