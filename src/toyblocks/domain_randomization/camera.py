"""Camera randomization -- per-episode position/orientation/FOV offsets + quaternion math + base
capture/override. Moved verbatim from render.py (pure samplers + USD/torch appliers for one
concern). No module globals. pxr/torch imports are function-local so this imports without a launched
Isaac app (samplers pure; _capture_*/_apply_*/_override_* require a live stage)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

RAND_KEYS = ("position_offset_cm", "pitch_deg", "yaw_deg", "roll_deg", "vertical_fov_deg")


RAND_DEFAULTS = {"position_offset_cm": [3.0, 3.0, 3.0], "pitch_deg": 5.0,
                 "yaw_deg": 5.0, "roll_deg": 2.0, "vertical_fov_deg": 3.0}


def _quat_mul(a, b):  # (...,4) wxyz
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _qmul(a, b):  # python-list quaternion product (wxyz), for the JSON side
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _delta_quat(pitch_rad: float, yaw_rad: float, roll_rad: float = 0.0):
    """Camera-local delta rotation from pitch (about local X), yaw (local Y) and roll
    (local Z), composed yaw*pitch*roll (wxyz). For small angles order is ~irrelevant."""
    def ax(axis, ang):
        h = ang / 2.0
        s = math.sin(h)
        return [math.cos(h), axis[0] * s, axis[1] * s, axis[2] * s]
    q = ax((0.0, 1.0, 0.0), yaw_rad)          # yaw about local Y
    q = _qmul(q, ax((1.0, 0.0, 0.0), pitch_rad))   # then pitch about local X
    q = _qmul(q, ax((0.0, 0.0, 1.0), roll_rad))    # then roll about local Z
    return q


def _norm_rand_dict(d: dict) -> dict:
    """Normalize a randomize spec: accept `fov_deg` as an alias for `vertical_fov_deg`."""
    d = dict(d or {})
    if "vertical_fov_deg" not in d and "fov_deg" in d:
        d["vertical_fov_deg"] = d["fov_deg"]
    return d


def _resolve_cam_ranges(flags: dict, cam_keys) -> Tuple[Dict[str, dict], int]:
    """Per-camera randomization ranges from flags['randomize'] (global + per-camera
    overrides). Returns ({cam_key: {position_offset_cm,pitch_deg,yaw_deg,roll_deg,
    vertical_fov_deg}}, seed)."""
    rz = (flags or {}).get("randomize")
    if not rz:
        return {}, 0
    rz = _norm_rand_dict(rz)
    glob = {k: rz.get(k, RAND_DEFAULTS[k]) for k in RAND_KEYS}
    per = rz.get("cameras") or {}
    out: Dict[str, dict] = {}
    for cam in cam_keys:
        r = dict(glob)
        # sim keys are "camera:<name>"; yaml overrides may use either form
        cam_name = cam[len("camera:"):] if cam.startswith("camera:") else cam
        override = _norm_rand_dict(per.get(cam) or per.get(cam_name) or {})
        r.update({k: v for k, v in override.items() if k in RAND_KEYS})
        pc = r["position_offset_cm"]
        pc = [pc, pc, pc] if isinstance(pc, (int, float)) else [float(x) for x in pc]
        r["position_offset_cm"] = pc
        for k in ("pitch_deg", "yaw_deg", "roll_deg", "vertical_fov_deg"):
            r[k] = float(r.get(k, 0.0))
        if any(abs(x) > 0 for x in pc) or any(abs(r[k]) > 0
                                              for k in ("pitch_deg", "yaw_deg", "roll_deg", "vertical_fov_deg")):
            out[cam] = r
    return out, int(rz.get("seed", 0))


def _sample_episode_offsets(ranges: Dict[str, dict], episode_indices, seed: int,
                            indexed: "bool | str" = False) -> Dict[int, dict]:
    """Sample per-episode, per-camera offsets uniformly from the ranges (reproducible by seed).

    indexed=True derives each episode's generator from (seed, episode_index) instead of advancing
    one shared stream, so a shard renders an episode identically to a full run (see
    seeds.episode_rng). Default False keeps previously rendered datasets reproducible.
    """
    rng = np.random.default_rng(seed)
    out: Dict[int, dict] = {}
    for e in sorted(episode_indices):
        if indexed:
            from toyblocks.domain_randomization.seeds import rng_for_episode
            rng = rng_for_episode(indexed, seed, e, "camera")
        d = {}
        for cam in sorted(ranges):
            r = ranges[cam]
            pc = r["position_offset_cm"]
            def u(key):
                return float(rng.uniform(-r[key], r[key])) if r[key] else 0.0
            d[cam] = {
                "position_offset_cm": [float(rng.uniform(-pc[0], pc[0])),
                                       float(rng.uniform(-pc[1], pc[1])),
                                       float(rng.uniform(-pc[2], pc[2]))],
                "pitch_deg": u("pitch_deg"),
                "yaw_deg": u("yaw_deg"),
                "roll_deg": u("roll_deg"),
                "vertical_fov_deg": u("vertical_fov_deg"),
            }
        out[e] = d
    return out


def _is_link_mounted(cam) -> bool:
    """True for cameras parented to a moving robot link (e.g. the gripper cam), which are
    re-posed by scene.update() every frame. Static (env-mounted) cameras don't move."""
    p = getattr(getattr(cam, "cfg", None), "prim_path", "") or ""
    return any(tok in p for tok in ("Robot", "robot", "panda", "robotiq", "hand", "link"))


def _capture_camera_bases(sim, randomize_cams: Dict[str, Any], rand_static: Dict[str, Any],
                          dt: float, *, pose_cams: Optional[dict] = None) -> Tuple[dict, dict]:
    """Snapshot pristine pose + intrinsics for randomized cameras.

    Must run BEFORE any per-episode offset is applied, and the returned tensors must be reused
    for every problem that shares this sim. Re-capturing from live cam.data after a prior job
    would bake that job's ending FOV/pose into the "base" and compound across problems
    (front-cam zoom drift over a multi-problem run).
    """
    cam_base_pose, cam_base_K = {}, {}
    if not randomize_cams and not pose_cams:
        return cam_base_pose, cam_base_K
    sim.sim.render()
    sim.scene.update(dt)
    for k, c in randomize_cams.items():
        cam_base_K[k] = c.data.intrinsic_matrices.clone()
    # `pose_cams` widens the POSE snapshot beyond the randomized set: --camera_base data re-bases
    # static cameras onto the dataset's own extrinsics, and a yaml with no `randomize:` block has
    # no randomize_cams at all -- so without this the correction would have no pristine oracle on
    # exactly the nocamrand runs.
    for k, c in (pose_cams or rand_static).items():
        cam_base_pose[k] = (c.data.pos_w.clone(), c.data.quat_w_world.clone())
    return cam_base_pose, cam_base_K


def recorded_static_bases(rand_static: Dict[str, Any], wave, active,
                          base_pose: Dict[str, Any], robot_root, device) -> Dict[str, Any]:
    """Re-base each STATIC camera's local mount onto the pose the DATASET records.

    Why only static cameras. `observation.camera_pose.*` is NOT a measurement of where the camera
    was: IsaacLab's `update_latest_camera_pose` defaults to False and no ToyBlocks camera sets it
    (only droid_franka does), so the recorder writes the sensor's INIT-TIME pose every frame -- see
    utils/dataset_provenance.py, which already says so. For an env-static camera that is exactly
    right: its init pose IS its pose for the whole episode. For the LINK-MOUNTED gripper camera it
    describes an arm configuration that never occurs in the episode -- measured 827 mm from
    cfg.offset on aug07, with an implied mount that drifts 380 mm within one episode and sits
    outside the trajectory entirely. Re-basing that camera onto its recorded pose injects ~0.5-0.8 m
    of error, so the link cameras keep cfg.offset, which rides FK correctly.

    The algebra (the unknown parent cancels, so it is never reconstructed): with A the robot-base ->
    camera-parent transform, T_live = A . M0 and T_rec = A . M_rec, both read in the RECORDED frame,
    so M_rec = M0 . (T_live^-1 . T_rec) and A drops out. Returns {cam_key: (pos, quat_cfg)} in the
    cfg convention, ready to stand in for cfg.offset.
    """
    from isaaclab.utils.math import (convert_camera_frame_orientation_convention, quat_apply,
                                     quat_inv, quat_mul)

    out: Dict[str, Any] = {}
    if robot_root is None:
        return out
    p_b, q_b = robot_root[:, :3], robot_root[:, 3:7]
    q_b_inv = quat_inv(q_b)
    for cam_key, cam in rand_static.items():
        off_cfg = getattr(getattr(cam, "cfg", None), "offset", None)
        live = base_pose.get(cam_key)
        if off_cfg is None or live is None:
            continue
        conv = getattr(off_cfg, "convention", "ros") or "ros"
        p_cfg = torch.tensor(list(off_cfg.pos), device=device, dtype=torch.float32)
        q_cfg = torch.tensor(list(off_cfg.rot), device=device, dtype=torch.float32)
        p_live_w, q_live_w = live
        P, Q, seen = {}, {}, False
        for i in active:
            rec = (wave[i].get("cam_pose") or {}).get(cam_key)
            if rec is None:
                continue
            arr = np.asarray(rec, dtype=np.float32)
            arr = arr[0] if arr.ndim == 2 else arr
            if arr.size != 7 or not np.isfinite(arr).all() or float(np.abs(arr[3:]).max()) == 0.0:
                continue                                          # recorder writes zeros when absent
            p_rec = torch.tensor(arr[:3], device=device, dtype=torch.float32).unsqueeze(0)
            q_rec = torch.tensor(arr[3:], device=device, dtype=torch.float32).unsqueeze(0)
            # the pristine pose, expressed in the same robot-base frame the recorder used
            qi, bi = q_b_inv[i:i + 1], p_b[i:i + 1]
            q_live_rb = quat_mul(qi, q_live_w[i:i + 1])
            p_live_rb = quat_apply(qi, p_live_w[i:i + 1] - bi)
            qt_live = convert_camera_frame_orientation_convention(q_live_rb, origin="world",
                                                                  target=conv)
            qt_rec = convert_camera_frame_orientation_convention(q_rec, origin="world", target=conv)
            dq = quat_mul(quat_inv(qt_live), qt_rec)
            q_a = quat_mul(qt_live, quat_inv(q_cfg.unsqueeze(0)))
            dp = quat_apply(quat_inv(q_a), p_rec - p_live_rb)
            P[i] = (p_cfg.unsqueeze(0) + dp)[0]
            Q[i] = quat_mul(q_cfg.unsqueeze(0), dq)[0]
            seen = True
        if seen:
            out[cam_key] = (P, Q)
    return out


def _apply_camera_local_offsets(cams: Dict[str, Any], wave, active,
                                ep_off: Dict[int, dict], device,
                                data_bases: Optional[dict] = None) -> None:
    """Apply pose randomization via LOCAL USD mount offset (cfg.offset ⊗ sampled delta).

    Tiled/Omniverse renderers consume USD-authored camera transforms. Fabric
    ``set_world_poses`` (even with sync_usd_on_fabric_write) does not reliably change the
    rendered view for either link-mounted OR env-static cameras — front/left stayed at the
    default framing while render_params still logged large pose/FOV deltas. Writing local
    translate/orient on the camera prim is what the renderer actually picks up.

    For link-mounted cams this also rides with FK (parent ⊗ local). For static cams the
    parent is the env Xform, so cfg.offset is the calibrated local mount — same formula.
    Always compose from cfg.offset (not live pose) so offsets never compound across episodes.
    """
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    for cam_key, cam in cams.items():
        off_cfg = getattr(getattr(cam, "cfg", None), "offset", None)
        view = getattr(cam, "_view", None)
        if off_cfg is None or view is None:
            print(f"[render] WARNING: cannot set local mount offset for {cam_key} "
                  f"(missing cfg.offset or _view); skipping pose DR")
            continue
        conv = getattr(off_cfg, "convention", "ros") or "ros"
        base_pos = torch.tensor(list(off_cfg.pos), device=device, dtype=torch.float32)
        base_quat_cfg = torch.tensor(list(off_cfg.rot), device=device, dtype=torch.float32)
        # camera_base="data": this static camera's base is the pose the DATASET records, not the
        # SceneConfig mount, so editing the config cannot silently redefine what rendering this
        # data means. Per-env, because each env carries its own recorded row.
        rebased = (data_bases or {}).get(cam_key)
        P, Q, ce = [], [], []
        for i in active:
            off = ep_off.get(wave[i]["episode_index"], {}).get(cam_key)
            if off is None:
                continue
            b_pos, b_quat = base_pos, base_quat_cfg
            if rebased is not None and i in rebased[0]:
                b_pos, b_quat = rebased[0][i], rebased[1][i]
            pos = b_pos + torch.tensor([x / 100.0 for x in off["position_offset_cm"]],
                                       device=device, dtype=torch.float32)
            # `quat_delta` is the REPLAY path: render_from_params recovers a rotation delta
            # from recorded absolute params, which cannot be decomposed back into pitch/yaw/roll
            # without ambiguity, so it hands the composed quaternion over directly.
            if off.get("quat_delta") is not None:
                dq = torch.tensor(list(off["quat_delta"]), device=device, dtype=torch.float32)
            else:
                dq = torch.tensor(_delta_quat(math.radians(off["pitch_deg"]),
                                              math.radians(off["yaw_deg"]),
                                              math.radians(off.get("roll_deg", 0.0))),
                                  device=device, dtype=torch.float32)
            # Compose in the cfg convention (usually ROS), then convert to OpenGL for USD local.
            quat_cfg = _quat_mul(b_quat.unsqueeze(0), dq.unsqueeze(0))[0]
            quat_gl = convert_camera_frame_orientation_convention(
                quat_cfg.unsqueeze(0), origin=conv, target="opengl")[0]
            P.append(pos); Q.append(quat_gl); ce.append(i)
        if ce:
            view.set_local_poses(torch.stack(P), torch.stack(Q), indices=ce)


def recorded_K(wave, i, cam_key, scene_K, device):
    """The episode's OWN recorded intrinsics, or None if unusable.

    The dataset records observation.camera_intrinsics.<cam> as a flattened 3x3. Preferring it over
    the built scene's matrix is the point: the scene's value is a SceneConfig constant, so editing
    the config silently changes what a "render of this data" means, while the data itself is the
    calibration the frames were actually captured with.

    Two ways it is unusable, both real: the recorder writes ZEROS when a camera had no intrinsics
    that step, and a source captured at a different resolution has a principal point that does not
    describe the image being rendered now. Reject both and fall back to the scene, loudly.
    """
    rec = (wave[i].get("cam_K") or {}).get(cam_key)
    if rec is None:
        return None
    arr = np.asarray(rec, dtype=np.float32)
    arr = arr[0] if arr.ndim == 2 else arr                       # per-frame rows -> first frame
    if arr.size != 9:
        return None
    k = arr.reshape(3, 3)
    if not np.isfinite(k).all() or float(k[0, 0]) <= 0 or float(k[1, 1]) <= 0:
        return None                                              # zeros/NaN => never recorded
    if scene_K is not None:
        sk = scene_K.detach().cpu().numpy()
        for r, c in ((0, 2), (1, 2)):                            # cx, cy must describe THIS image
            if float(sk[r, c]) > 0 and abs(float(k[r, c]) - float(sk[r, c])) > 0.2 * float(sk[r, c]):
                return "mismatch"
    return torch.tensor(k, device=device, dtype=torch.float32)


# Tiled rendering builds ONE render product from the whole camera list (TiledCamera.__init__ ->
# rep.create.render_product_tiled), and the projection it rasterises with does not follow the
# per-prim focal length. Measured on shipped data: every episode in a wave rendered with env
# slot 0's focal length while the parquet faithfully recorded its own planned FOV. Predicting
# each frame's error as (fx[episode] - fx[wave leader]) / fx[episode] -- no free parameters --
# explained 98.3% of the variance of the measured radial error across nine frames (r = +0.996).
#
# Writing distinct per-env matrices therefore does not change the image. It only makes the
# recorded intrinsics disagree with the pixels, which is worse than having no jitter: every
# downstream projection (annotate_target_xy's goal.position above all) lands off-target by
# (fx_recorded/fx_rendered - 1) x the radius from the principal point, ~10 px at r=200.
#
# So apply ONE matrix to the whole wave -- the one that actually renders. FOV jitter becomes per
# WAVE rather than per episode, and recorded intrinsics match the pixels by construction.
# Flip this to True only alongside evidence that tiled rendering honours per-prim intrinsics.
ALLOW_PER_ENV_INTRINSICS = False


def _set_wave_intrinsics(cam, cam_key, KK, ce, warned):
    """Set intrinsics for a wave's envs, collapsing to one matrix unless per-env is known to work.

    KK[j] is the matrix intended for env ce[j]; KK[0] is the wave's first active env, which is
    the value the tiled renderer actually uses for every tile.
    """
    if ALLOW_PER_ENV_INTRINSICS or len(KK) <= 1:
        cam.set_intrinsic_matrices(torch.stack(KK), env_ids=ce)
        return
    k0 = KK[0]
    spread = max(float(torch.abs(k - k0).max()) for k in KK[1:])
    if spread > 1e-3 and cam_key not in warned:
        warned.add(cam_key)
        print(f"[render] {cam_key}: tiled rendering cannot vary intrinsics per env; applying "
              f"the wave's first value to all {len(ce)} env(s) (discarding up to {spread:.2f} px "
              f"of fx jitter). FOV randomization is per WAVE, not per episode -- recorded "
              f"intrinsics now match the rendered pixels.")
    cam.set_intrinsic_matrices(torch.stack([k0] * len(KK)), env_ids=ce)


def applied_wave_offsets(off, wave, active, ep_off):
    """One episode's camera offsets with the FOV replaced by the one actually rendered.

    The wave renders with a single intrinsic matrix (see ``_set_wave_intrinsics``), taken from
    its first active env, so every episode in the wave was framed with that FOV whatever it
    sampled for itself. Recording the sampled value makes render_params.json describe a camera
    that never existed, and everything that rebuilds intrinsics from it -- recover_camera_columns
    for camera-less sources, render_from_params for replays -- then reproduces the wrong one.

    Position and rotation are left alone: those DO apply per env.
    """
    if ALLOW_PER_ENV_INTRINSICS or not off:
        return off
    out = {}
    for ck, o in off.items():
        o2 = dict(o)
        if "vertical_fov_deg" in o2:
            for i in active:
                lead = (ep_off.get(wave[i]["episode_index"]) or {}).get(ck)
                if lead is not None:
                    o2["vertical_fov_deg"] = float(lead.get("vertical_fov_deg", 0.0))
                    break
        out[ck] = o2
    return out


def _apply_episode_camera_fov(cams: Dict[str, Any], wave, active,
                              ep_off: Dict[int, dict], device,
                              base_K: Optional[dict] = None,
                              camera_base: str = "data") -> None:
    """Apply per-episode FOV offsets from a pristine base_K (never compound).

    Pose is handled separately by ``_apply_camera_local_offsets``. Pass ``base_K`` captured
    once at sim build so intrinsics don't drift across waves/problems.

    camera_base="data" (the default) takes that pristine base from the EPISODE'S OWN recorded
    intrinsics instead, so the FOV jitter rides on the calibration in the dataset rather than on a
    SceneConfig constant that merely happens to match it today. "scene" restores the old base.
    """
    warned = getattr(_apply_episode_camera_fov, "_warned", None)
    if warned is None:
        warned = _apply_episode_camera_fov._warned = set()
    for cam_key, cam in cams.items():
        K = cam.data.intrinsic_matrices
        bK_all = base_K.get(cam_key) if base_K else None
        KK, ce = [], []
        for i in active:
            off = ep_off.get(wave[i]["episode_index"], {}).get(cam_key)
            if off is None:
                continue
            bk = (bK_all[i].clone() if bK_all is not None else K[i].clone())
            if camera_base == "data":
                rk = recorded_K(wave, i, cam_key, bk, device)
                if rk == "mismatch":
                    if cam_key not in warned:
                        warned.add(cam_key)
                        print(f"[render] WARNING {cam_key}: recorded intrinsics disagree with the "
                              f"rendered image (principal point off by >20%); using the scene's. "
                              f"A source captured at a different resolution cannot supply K.")
                elif rk is not None:
                    bk = rk
            fy, cy = float(bk[1, 1]), float(bk[1, 2])
            dfov = float(off.get("vertical_fov_deg") or 0.0)
            if fy > 0 and dfov != 0.0:
                new_vfov = max(math.radians(1.0),
                               min(math.radians(179.0),
                                   2 * math.atan(cy / fy) + math.radians(dfov)))
                new_fy = cy / math.tan(new_vfov / 2)
                bk[0, 0] = bk[0, 0] * (new_fy / fy)
                bk[1, 1] = new_fy
            KK.append(bk); ce.append(i)
        if ce:
            # Prefer plain indices — matches set_local_poses and avoids device-tensor quirks.
            _set_wave_intrinsics(cam, cam_key, KK, ce, warned)


def _apply_episode_camera_offsets(sim, cams: Dict[str, Any], wave, active,
                                  ep_off: Dict[int, dict], device,
                                  set_pose: bool = True, set_fov: bool = True,
                                  base_pose: Optional[dict] = None,
                                  base_K: Optional[dict] = None) -> None:
    """Legacy helper: pose via world writes (unreliable for tiled cams) and/or FOV.

    Prefer ``_apply_camera_local_offsets`` for pose and ``_apply_episode_camera_fov`` for FOV.
    Kept for ``_override_cameras``-adjacent call patterns and any set_pose fallback.
    """
    if set_fov:
        _apply_episode_camera_fov(cams, wave, active, ep_off, device, base_K=base_K)
    if not set_pose:
        return
    for cam_key, cam in cams.items():
        pos_w, quat_w = cam.data.pos_w, cam.data.quat_w_world
        bpose = base_pose.get(cam_key) if base_pose else None
        P, Q, ce = [], [], []
        for i in active:
            off = ep_off.get(wave[i]["episode_index"], {}).get(cam_key)
            if off is None:
                continue
            if bpose is not None:
                bp, bq = bpose[0][i].clone(), bpose[1][i].clone()
            else:
                bp, bq = pos_w[i].clone(), quat_w[i].clone()
            bp = bp + torch.tensor([x / 100.0 for x in off["position_offset_cm"]],
                                   device=device, dtype=bp.dtype)
            dq = torch.tensor(_delta_quat(math.radians(off["pitch_deg"]),
                                          math.radians(off["yaw_deg"]),
                                          math.radians(off.get("roll_deg", 0.0))),
                              device=device, dtype=bq.dtype)
            bq = _quat_mul(bq.unsqueeze(0), dq.unsqueeze(0))[0]
            P.append(bp); Q.append(bq); ce.append(i)
        if ce:
            ce_t = torch.tensor(ce, device=device)
            cam.set_world_poses(torch.stack(P), torch.stack(Q), env_ids=ce_t, convention="world")


def _initial_camera_params(episodes: List[Dict[str, Any]]) -> dict:
    """Initial per-camera params read from the recorded data: world position,
    rotation quaternion (wxyz), and VERTICAL fov in degrees."""
    cams = {}
    if episodes:
        e0 = episodes[0]
        for cam_key, cp in e0["cam_pose"].items():
            entry = {"position": [float(x) for x in cp[0, :3]],
                     "rotation": [float(x) for x in cp[0, 3:7]]}
            # recorded cam_K is a flat per-frame 9-vector [fx,0,cx, 0,fy,cy, 0,0,1]
            ck = e0["cam_K"].get(cam_key)
            if ck is not None and float(ck[0, 4]) > 0:
                fy, cy = float(ck[0, 4]), float(ck[0, 5])
                entry["vertical_fov_deg"] = math.degrees(2 * math.atan(cy / fy))
            cams[cam_key] = entry
    return cams


def _scene_camera_params(sim, cam_keys,
                         base_K: Optional[Dict[str, torch.Tensor]] = None) -> dict:
    """Initial per-camera params read from the SCENE CONFIG -- the base `--camera_base scene`
    actually renders from.

    Counterpart to `_initial_camera_params`, which reads the DATASET's recorded pose. Same output
    shape, so `_absolute_cam_params` composes on either. Without this, render_params.json reported
    the recorded base no matter which base was rendered, and a `scene` run's file described a
    camera that was never used.

    cfg.offset is stored in the cfg's own convention (ros for ToyBlocks cameras); the recorded
    pose is world-convention, so convert to keep the two directly comparable.

    The FOV comes from `base_K` -- the pristine intrinsics `_capture_camera_bases` snapshots once at
    sim build, the same tensors the renderer composes the FOV offset onto -- wherever it covers the
    camera. The live intrinsics are pristine only for the first problem a sim renders: render
    builds this record per problem, so every later one read the previous wave's jittered FOV as
    its base and recorded base + offset off by that jitter, while the parquet's intrinsics were
    right. A camera outside `base_K` is not FOV-randomized, so its live K is still its base.
    """
    import math
    import torch
    from isaaclab.utils.math import convert_camera_frame_orientation_convention

    cams = {}
    for key in cam_keys:
        c = sim.scene.sensors.get(key)
        off = getattr(getattr(c, "cfg", None), "offset", None)
        if off is None:
            continue
        q = torch.tensor([list(off.rot)], dtype=torch.float32)
        q_world = convert_camera_frame_orientation_convention(
            q, origin=getattr(off, "convention", "ros"), target="world")[0]
        entry = {"position": [float(v) for v in off.pos],
                 "rotation": [float(v) for v in q_world]}
        try:
            bK = (base_K or {}).get(key)
            K = bK[0] if bK is not None else c.data.intrinsic_matrices[0]
            fy, cy = float(K[1, 1]), float(K[1, 2])
            if fy > 0:
                entry["vertical_fov_deg"] = math.degrees(2 * math.atan(cy / fy))
        except Exception:
            pass
        cams[key] = entry
    return cams


def _absolute_cam_params(base: dict, off: dict) -> dict:
    """Per-episode ABSOLUTE camera params (same space as `initial`): base pose + sampled
    offset. position is world-frame; rotation = base ⊗ local delta (wxyz); vertical fov
    += the sampled delta."""
    pos = [float(base["position"][k]) + off["position_offset_cm"][k] / 100.0 for k in range(3)]
    dq = _delta_quat(math.radians(off.get("pitch_deg", 0.0)),
                     math.radians(off.get("yaw_deg", 0.0)),
                     math.radians(off.get("roll_deg", 0.0)))
    out = {"position": pos, "rotation": _qmul([float(x) for x in base["rotation"]], dq)}
    if "vertical_fov_deg" in base:
        out["vertical_fov_deg"] = float(base["vertical_fov_deg"]) + float(off.get("vertical_fov_deg", 0.0))
    return out


def offsets_from_absolute(initial_cams: dict, episodes: List[Dict[str, Any]],
                          index_key: str = "episode_index") -> Dict[int, dict]:
    """Recover per-episode camera OFFSETS from recorded ABSOLUTE params -- the inverse of
    ``_absolute_cam_params``.

    render_params stores, per episode, the absolute pose/FOV a camera was rendered at, plus the
    pristine base under ``initial.cameras``. A replay needs the OFFSET, because the offset is what
    the renderer actually consumes (``_apply_camera_local_offsets`` writes cfg.offset (x) delta as a
    USD local transform). Reconstructing it here means a replay drives cameras through exactly the
    same code path as the original randomization instead of a parallel implementation.

    abs = base (x) dq  =>  dq = conj(base) (x) abs, handed over as `quat_delta` rather than
    decomposed back into pitch/yaw/roll (that decomposition is not unique).
    """
    out: Dict[int, dict] = {}
    for e in episodes:
        idx = int(e.get(index_key, e.get("episode_index")))
        # Preferred: the run recorded the RAW offsets, so no reconstruction is needed and the
        # replay is exact. Datasets written before render.py started recording these fall back
        # to the derivation below, which is correct for position on env-mounted cameras but CANNOT
        # recover rotation (the recorded absolute is composed onto a different-convention base
        # than the renderer uses).
        raw = e.get("camera_offsets")
        if raw:
            out[idx] = {ck: dict(v) for ck, v in raw.items()}
            continue
        per = {}
        for cam_key, absv in (e.get("cameras") or {}).items():
            base = (initial_cams or {}).get(cam_key)
            if not base or "position" not in absv:
                continue
            dpos_cm = [(float(absv["position"][k]) - float(base["position"][k])) * 100.0
                       for k in range(3)]
            bq = [float(x) for x in base["rotation"]]
            conj = [bq[0], -bq[1], -bq[2], -bq[3]]          # unit quat inverse
            dq = _qmul(conj, [float(x) for x in absv["rotation"]])
            per[cam_key] = {"position_offset_cm": dpos_cm, "quat_delta": dq,
                            "pitch_deg": 0.0, "yaw_deg": 0.0, "roll_deg": 0.0,
                            "vertical_fov_deg": (float(absv.get("vertical_fov_deg", 0.0))
                                                 - float(base.get("vertical_fov_deg", 0.0)))}
        if per:
            out[idx] = per
    return out


_OVERRIDE_WARNED: set = set()


def _override_cameras(sim, cams: Dict[str, Any], wave, active, t, device) -> None:
    """Set each camera's world pose + intrinsics from recorded (local) values."""
    for cam_key, cam_obj in cams.items():
        poss, quats, Ks, ce = [], [], [], []
        for i in active:
            cp = wave[i]["cam_pose"].get(cam_key)
            if cp is None:
                continue
            local = torch.tensor(cp[min(t, cp.shape[0] - 1)], device=device, dtype=torch.float32)
            world = sim._transform_to_world_frame(local, env_idx=i)
            poss.append(world[:3]); quats.append(world[3:7]); ce.append(i)
            ck = wave[i]["cam_K"].get(cam_key)
            if ck is not None:
                Ks.append(torch.tensor(ck[min(t, ck.shape[0] - 1)],
                                       device=device, dtype=torch.float32).reshape(3, 3))
        if not ce:
            continue
        ce_t = torch.tensor(ce, device=device)
        cam_obj.set_world_poses(torch.stack(poss), torch.stack(quats),
                                env_ids=ce_t, convention="world")
        if len(Ks) == len(ce):
            _set_wave_intrinsics(cam_obj, cam_key, Ks, ce_t, _OVERRIDE_WARNED)
