"""Build-time scene configuration for domain randomization -- the "start a scene + apply the loaded
parameters" seam, reusable by rerender and eval. `apply_build_flags` mutates an env_cfg BEFORE the sim
is built (env_spacing, GI, RTX render-mode/carb, anti-aliasing, per-preset camera cfg); the moved
STAGE helpers (apply_lighting_flags / hide_*) run AFTER the sim exists. pxr/omni/blocksuite imports
are function-local so this module imports without a launched Isaac app. No module globals.
"""
from __future__ import annotations

import fnmatch
import math
import re
from typing import Any, Optional

import numpy as np
import torch

def drop_cameras(names) -> None:
    """Remove cameras from the sim2real preset BEFORE scene creation.

    Deleting the cfg (rather than filtering at record time) means the camera is never spawned, so we
    skip its RENDER cost as well as its rgb+depth streams and video files -- on a 3-camera scene,
    dropping one is roughly a third of the per-frame tiled-render work.
    """
    if not names:
        return
    from blocksuite.scenes import SCENE_PRESETS
    preset = SCENE_PRESETS["sim2real"]
    for n in names:
        key = str(n)[len("camera:"):] if str(n).startswith("camera:") else str(n)
        if key in preset.camera_cfgs:
            del preset.camera_cfgs[key]
            print(f"[dr.build] --skip_cameras: '{key}' NOT spawned (no rgb, no depth, no render cost)")
        else:
            print(f"[dr.build] WARNING --skip_cameras: '{key}' not in the sim2real preset "
                  f"(have: {sorted(preset.camera_cfgs)})")


def scale_cameras(factor) -> None:
    """Scale every sim2real camera's render resolution BEFORE scene creation.

    DLSS derives its internal render resolution from the output size. At the stock 640x360 it path
    traces at 371x209, which Kit reports as "below minimal input resolution of 300" -- i.e. the
    neural reconstructor runs outside its supported input range while also upscaling ~1.7x from
    1 spp. Rendering larger lifts it back into range; the caller can downscale afterwards.
    """
    if not factor or float(factor) == 1.0:
        return
    from blocksuite.scenes import SCENE_PRESETS
    f = float(factor)
    for name, cam in SCENE_PRESETS["sim2real"].camera_cfgs.items():
        w, h = getattr(cam, "width", None), getattr(cam, "height", None)
        if not w or not h:
            continue
        nw, nh = int(round(w * f)), int(round(h * f))
        if nw % 2 or nh % 2:
            raise ValueError(f"--render_scale {f}: '{name}' would be {nw}x{nh}; yuv420p video "
                             f"needs even width and height")
        cam.width, cam.height = nw, nh
        print(f"[dr.build] --render_scale {f}: '{name}' {w}x{h} -> {cam.width}x{cam.height}")


DEPTH_DATA_TYPES = ("distance_to_image_plane", "depth", "distance_to_camera")


def drop_depth() -> None:
    """Stop the sim2real preset cameras from rendering depth (--skip_depth), BEFORE scene creation.
    Robot-mounted cameras are not in the preset; the recorder skips their depth separately."""
    from blocksuite.scenes import SCENE_PRESETS
    for name, cam in SCENE_PRESETS["sim2real"].camera_cfgs.items():
        types = list(getattr(cam, "data_types", None) or [])
        kept = [t for t in types if t not in DEPTH_DATA_TYPES]
        if kept != types:
            cam.data_types = kept
            print(f"[dr.build] --skip_depth: '{name}' renders {kept} (no depth)")


def apply_camera_flag_overrides(flags: dict, env_cfg=None) -> None:
    """Override camera extrinsics / FOV from ``flags["cameras"]``.

    flags["cameras"] maps camera name -> {position:[x,y,z], quaternion:[w,x,y,z],
    fov_deg: float}.

    Two kinds of camera, two application points. Scene cameras (front/left) live in the scene
    preset and are mutated here, before gym.make. ROBOT cameras -- the wrist/gripper camera --
    are not in the preset at all: ``BaseEnv._setup_scene`` merges ``robot_cfg.cameras`` into
    ``cfg.camera_cfgs`` AFTER this runs, so there is nothing here to override and the old code
    printed "not in sim2real preset; skipping" and dropped the request on the floor. Measured
    2026-11-03: rerender_flags_camtest.yaml asks for gripper position z=0.062 / fov 105 and
    every dataset it ever produced rendered at the robot default z=0.022 / ~88 while
    render_params still recorded the request.

    So those are DEFERRED into ``env_cfg.camera_overrides_json``, which BaseEnv applies
    immediately after that merge (base_env.py, right after ``camera_cfgs.update``). fov_deg is
    passed through rather than converted here, because the conversion needs the camera's own
    horizontal_aperture (the gripper's is 6.916, not the 20.955 default) and that cfg does not
    exist yet at this point.

    Note this is only the DETERMINISTIC ``cameras:`` block. The ``randomize:`` camera block
    already reaches the gripper correctly: it resolves against the live scene sensors and
    writes the local USD mount offset, which rides with FK.
    """
    cams = (flags or {}).get("cameras")
    if not cams:
        return
    from blocksuite.scenes import SCENE_PRESETS
    preset = (SCENE_PRESETS.get(getattr(env_cfg, "scene_preset_id", None))
              or SCENE_PRESETS["sim2real"])
    deferred: dict = {}
    for name, ov in cams.items():
        if name not in preset.camera_cfgs:
            body = {}
            if "position" in ov:
                body["pos"] = [float(x) for x in ov["position"]]
            if "quaternion" in ov:
                body["rot"] = [float(x) for x in ov["quaternion"]]
            if "fov_deg" in ov:
                body["fov_deg"] = float(ov["fov_deg"])
            if body:
                deferred[name] = body
            continue
        cfg = preset.camera_cfgs[name]
        if "position" in ov:
            cfg.offset.pos = tuple(float(x) for x in ov["position"])
        if "quaternion" in ov:
            cfg.offset.rot = tuple(float(x) for x in ov["quaternion"])
        if "fov_deg" in ov:
            ha = float(getattr(cfg.spawn, "horizontal_aperture", 20.955))
            cfg.spawn.focal_length = ha / (2.0 * math.tan(math.radians(float(ov["fov_deg"])) / 2.0))
        print(f"[rerender] flags: overrode camera '{name}' -> {ov}")
    if deferred:
        _defer_camera_overrides(deferred, env_cfg)


def _defer_camera_overrides(deferred: dict, env_cfg) -> None:
    """Hand robot-owned camera overrides to ``camera_overrides_json`` for BaseEnv to apply.

    Merged key-by-key and never clobbering: an explicit ``camera_overrides_json`` passed by the
    caller is more specific than the flags YAML, so it wins per field.
    """
    if env_cfg is None:
        print(f"[dr.build] WARNING: cannot apply camera override(s) {sorted(deferred)} -- they "
              f"are not scene-preset cameras and no env_cfg was given to defer them to")
        return
    import json as _json
    try:
        existing = _json.loads(getattr(env_cfg, "camera_overrides_json", None) or "{}")
    except ValueError:
        print("[dr.build] WARNING: camera_overrides_json is not valid JSON; not merging flags "
              "camera overrides into it")
        return
    for name, body in deferred.items():
        cur = existing.setdefault(name, {})
        for k, v in body.items():
            cur.setdefault(k, v)
    env_cfg.camera_overrides_json = _json.dumps(existing)
    print(f"[dr.build] deferred camera override(s) to camera_overrides_json: "
          f"{ {k: deferred[k] for k in sorted(deferred)} }")


def hide_debug_visuals() -> None:
    """Hide all debug visualization markers under /Visuals (RGB frame-axis gizmos, corner
    spheres, block/goal arrows, ee-frame). These are per-env VisualizationMarkers created for
    datagen; some leak into camera views. MakeInvisible on the /Visuals scope propagates to
    all children by inheritance."""
    try:
        from pxr import UsdGeom
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        hidden = []
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            # top-level /Visuals scope(s) — hide the whole subtree
            if p == "/Visuals" or p.startswith("/Visuals/") and p.count("/") == 2:
                if prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeInvisible()
                    hidden.append(p)
        if hidden:
            print(f"[rerender] hid debug visuals: {sorted(set(hidden))}")
        else:
            print("[rerender] no /Visuals debug markers found to hide")
    except Exception as e:
        print(f"[rerender] hide_debug_visuals failed: {e}")


def hide_scene_objects(flags: dict) -> None:
    """Hide (MakeInvisible) scene geometry named in flags['hide_objects']. Each entry is either:
      * a per-env object NAME / fnmatch (e.g. "modular_wall_*") — matched against
        /World/envs/env_N/<name> (scene-preset objects, see sim2real.py); hiding the top prim
        propagates to its subtree. Render-only replay -> no physics effect.
      * "floor" / "ground" — hides the ground-plane GEOMETRY under /World/ground but KEEPS its
        SphereLight (so the floor disappears / the dome shows through, lighting unchanged).
      * "backdrop" — convenience keyword expanding to the full sim2real environment backdrop
        (modular_wall_*, wall, window_background_*, window_frame_*, FlatGrid, floor), i.e. strip
        the scene to just the table + robot + workstation + blocks. Combine with extra patterns.
      * an absolute prim-path fnmatch (starts with "/") — hides any matching Imageable prim.
    Scene-preset object names (src/blocksuite/assets/scene.json): workspace_table, robot_workstation
    (kept), modular_wall_{left,right}_{0,1}, wall (back wall), window_background_{top,bottom},
    window_frame_{top,bottom}, FlatGrid. Pair backdrop/wall/floor removal with a large
    --env_spacing so the cameras' far-clip drops neighbor envs."""
    patterns = (flags or {}).get("hide_objects")
    if not patterns:
        return
    if isinstance(patterns, str):
        patterns = [patterns]
    # Convenience keyword "backdrop": expands to the whole sim2real environment backdrop
    # (side wall panels, back wall, window + frames, flat grid, floor) so a run can strip the
    # scene down to just the table + robot + workstation (+ the manipulated blocks, which are
    # never matched here). Fully customizable: mix "backdrop" with extra explicit patterns, or
    # skip it and list your own object-name fnmatch patterns instead.
    BACKDROP = ["modular_wall_*", "wall", "window_background_*", "window_frame_*", "FlatGrid", "floor"]
    if any(isinstance(p, str) and p.lower() == "backdrop" for p in patterns):
        patterns = [p for p in patterns if not (isinstance(p, str) and p.lower() == "backdrop")]
        patterns = patterns + [b for b in BACKDROP if b not in patterns]
    try:
        import fnmatch
        from pxr import Usd, UsdGeom, UsdLux
        import omni.usd
        stage = omni.usd.get_context().get_stage()
        hide_floor = any(str(p).lower() in ("floor", "ground") for p in patterns)
        abs_pats = [p for p in patterns if isinstance(p, str) and p.startswith("/")]
        name_pats = [p for p in patterns if isinstance(p, str) and not p.startswith("/")
                     and p.lower() not in ("floor", "ground")]
        names_hit, n_prims = set(), 0
        for prim in stage.Traverse():
            p = prim.GetPath().pathString
            if abs_pats and prim.IsA(UsdGeom.Imageable) and any(fnmatch.fnmatch(p, a) for a in abs_pats):
                UsdGeom.Imageable(prim).MakeInvisible(); n_prims += 1; continue
            m = re.match(r"^/World/envs/env_\d+/([^/]+)$", p)
            if m and any(fnmatch.fnmatch(m.group(1), pat) for pat in name_pats) and prim.IsA(UsdGeom.Imageable):
                UsdGeom.Imageable(prim).MakeInvisible()
                names_hit.add(m.group(1)); n_prims += 1
        if hide_floor:
            ground = stage.GetPrimAtPath("/World/ground")
            nf = 0
            if ground and ground.IsValid():
                for pr in Usd.PrimRange(ground):
                    # geometry only (Mesh/Plane/etc.) -> keep the SphereLight (not a Gprim)
                    if pr.IsA(UsdGeom.Gprim) and not pr.HasAPI(UsdLux.LightAPI):
                        UsdGeom.Imageable(pr).MakeInvisible(); nf += 1
            print(f"[rerender] hide_objects: hid floor geometry under /World/ground "
                  f"({nf} gprim(s); kept SphereLight)")
            n_prims += nf
        if names_hit or abs_pats:
            print(f"[rerender] hide_objects {name_pats + abs_pats}: hid across envs -> {sorted(names_hit)}")
        if not n_prims:
            print(f"[rerender] hide_objects: nothing matched {patterns}")
    except Exception as e:
        print(f"[rerender] hide_scene_objects failed: {e}")


def apply_lighting_flags(flags: dict, sim) -> None:
    """Apply lighting overrides after scene creation (dome intensity + optional
    per-env colored lights), mirroring the vectorized_recorder rerender."""
    if not flags:
        return
    from pxr import UsdLux, Gf, UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()

    if "lighting_intensity" in flags:
        val = flags["lighting_intensity"]
        # A [min,max] list is a per-batch sampling range handled in replay_job
        # (_dome_intensity per wave) -> don't fix a value here.
        if isinstance(val, (list, tuple)):
            print(f"[rerender] flags: lighting_intensity range {list(val)} "
                  "(sampled per batch at replay)")
        else:
            intensity = float(np.random.uniform(200, 5000)) if val == "random" else float(val)
            prim = stage.GetPrimAtPath("/World/Light")
            if prim and prim.IsValid():
                UsdLux.DomeLight(prim).GetIntensityAttr().Set(intensity)
                print(f"[rerender] flags: lighting_intensity = {intensity:.1f}")
            else:
                print("[rerender] flags: /World/Light not found; skipping lighting_intensity")

    if "randomized_colored_lighting" in flags:
        # Best-effort: prims created after the first sim.reset may need a render
        # refresh to register with RTX. Kept for yaml-format compatibility.
        val = flags["randomized_colored_lighting"]
        env_origins = sim.scene.env_origins.cpu()
        offset = torch.tensor([0.3, -0.1, 1.0])
        for env_idx in range(sim.num_envs):
            r, g, b = (np.random.uniform(0, 1, 3).tolist() if val == "random"
                       else [float(val[0]), float(val[1]), float(val[2])])
            pos = (env_origins[env_idx] + offset).tolist()
            light = UsdLux.DiskLight.Define(stage, f"/World/ColoredLight_{env_idx}")
            light.GetRadiusAttr().Set(0.5)
            light.GetColorAttr().Set(Gf.Vec3f(r, g, b))
            light.GetIntensityAttr().Set(10000.0)
            xf = UsdGeom.Xformable(light.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTranslateOp().Set(Gf.Vec3d(*pos))
        print(f"[rerender] flags: randomized_colored_lighting applied ({sim.num_envs} envs)")


def apply_build_flags(env_cfg, flags, *, cli_env_spacing=None, antialiasing=None,
                      disable_dlss: bool = False, enable_gi: bool = False,
                      skip_cameras=None, render_scale=None,
                      camera_update_period: float | None = None,
                      skip_depth: bool = False) -> None:
    """Apply build-time DR scene config to `env_cfg` BEFORE the sim is built (mutates env_cfg + the
    scene preset's camera cfgs). env_spacing precedence: yaml flags['env_spacing'] > cli_env_spacing >
    InteractiveScene default. GI auto-on when flags has a `skybox` block (or enable_gi). Order-
    independent env_cfg mutations, so this reproduces rerender's build config exactly.

    camera_update_period: None (default) leaves each preset camera's own update_period from the
    scene config untouched; a float overrides every one of them. Rerender passes 0.0."""
    flags = flags or {}
    ys = flags.get("env_spacing")
    if ys is not None:
        env_cfg.scene.env_spacing = float(ys)
        _ov = (f" (yaml overrides --env_spacing {cli_env_spacing})"
               if cli_env_spacing is not None else " (from yaml)")
        print(f"[dr.build] env_spacing = {float(ys)}{_ov}")
    elif cli_env_spacing is not None:
        env_cfg.scene.env_spacing = float(cli_env_spacing)
        print(f"[dr.build] env_spacing = {cli_env_spacing} (from --env_spacing)")
    # None keeps whatever the scene config declares (0.04 = the 25fps recording cadence).
    # Rerender passes 0.0: that default is time-gated and occasionally skips a refresh under
    # rerender's manual render loop, leaving a duplicate frame; 0 forces a fresh capture on
    # every scene.update().
    if camera_update_period is not None:
        period = float(camera_update_period)
        try:
            from blocksuite.scenes import SCENE_PRESETS
            preset = SCENE_PRESETS.get(getattr(env_cfg, "scene_preset_id", None))
            if preset is not None:
                for cam_cfg in preset.camera_cfgs.values():
                    if hasattr(cam_cfg, "update_period"):
                        cam_cfg.update_period = period
                print(f"[dr.build] camera update_period = {period}")
        except Exception as e:
            print(f"[dr.build] could not set camera update_period={period}: {e}")
    aa_mode = antialiasing or ("FXAA" if disable_dlss else None)
    if aa_mode:
        try:
            env_cfg.sim.render.antialiasing_mode = aa_mode
            print(f"[dr.build] antialiasing_mode = {aa_mode}")
        except Exception as e:
            print(f"[dr.build] could not set antialiasing_mode: {e}")
    rmode = flags.get("render_mode")
    if rmode:
        try:
            cs = dict(getattr(env_cfg.sim.render, "carb_settings", None) or {})
            cs["/rtx/rendermode"] = str(rmode)
            env_cfg.sim.render.carb_settings = cs
            print(f"[dr.build] render.carb_settings['/rtx/rendermode'] = {rmode} (applied at launch)")
        except Exception as e:
            print(f"[dr.build] could not set render_mode via carb_settings: {e}")
    if enable_gi or flags.get("skybox"):
        r = env_cfg.sim.render
        for attr, val in (("enable_global_illumination", True), ("enable_reflections", True),
                          ("dome_light_upper_lower_strategy", 0)):
            try:
                setattr(r, attr, val)
            except Exception as e:
                print(f"[dr.build] could not set render.{attr}: {e}")
        print("[dr.build] enabled RTX global illumination + reflections (dome IBL sampling)")
    apply_camera_flag_overrides(flags, env_cfg)
    drop_cameras(skip_cameras)
    scale_cameras(render_scale)
    if skip_depth:
        drop_depth()


# Carb keys worth comparing between two renderers that are supposed to produce the same image.
# kit_args match by construction when both sides call DRSession.kit_args(flags); what can still
# differ is the RUNTIME state after each app applies its own defaults on top.
RENDER_CONTEXT_KEYS = (
    "/rtx/rendermode",
    "/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp", "/rtx/pathtracing/maxBounces",
    "/rtx/pathtracing/clampSpp", "/rtx/pathtracing/fireflyFilter/enabled",
    "/rtx/directLighting/sampledLighting/enabled",
    "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "/rtx/directLighting/sampledLighting/maxRayIntensity",
    "/rtx/indirectDiffuse/enabled", "/rtx/reflections/enabled",
    "/rtx/ambientOcclusion/enabled",
    "/rtx/post/tonemap/op", "/rtx/post/tonemap/filmIso", "/rtx/post/tonemap/fNumber",
    "/rtx/post/tonemap/cameraShutter", "/rtx/post/tonemap/cameraExposure",
    "/rtx/post/histogram/enabled",
    "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/post/dlss/enabled",
    "/rtx/newDenoiser/enabled", "/rtx/raytracing/subsurface/enabled",
    "/rtx/sceneDb/ambientLightIntensity",
    "/rtx/hydra/subdivision/refinementLevel",
    "/rtx/materialDb/syncLoads", "/omni/replicator/captureOnPlay",
    "/app/renderer/skipWhileMinimized", "/app/hydraEngine/waitIdle",
    "/rtx-transient/dlssg/enabled", "/rtx/ecoMode/enabled",
)


def dump_render_context(extra_keys=()) -> dict:
    """Read back the live carb render settings, for diffing two renderers.

    Two processes can be launched with identical kit_args and still end up in different RTX
    states -- a different Kit experience file, a different AppLauncher default, or a setting one
    side writes after launch. This dumps the values that actually govern the image.
    """
    out: dict = {}
    try:
        import carb

        s = carb.settings.get_settings()
    except Exception as e:  # noqa: BLE001 -- callable from a non-Isaac process
        return {"_error": f"{type(e).__name__}: {e}"}
    for key in tuple(RENDER_CONTEXT_KEYS) + tuple(extra_keys):
        try:
            out[key] = s.get(key)
        except Exception:  # noqa: BLE001 -- unset keys are informative as None
            out[key] = None
    return out


def dump_render_products(sim) -> dict:
    """Per-camera render resolution actually in the scene -- the RTX work per frame.

    A renderer with fewer or smaller render products converges differently at the same sample
    count, which shows up as a shading difference without any DR value moving.
    """
    out: dict = {}
    total = 0
    try:
        sensors = dict(sim.scene.sensors)
    except Exception as e:  # noqa: BLE001
        return {"_error": f"{type(e).__name__}: {e}"}
    for name, sensor in sensors.items():
        cfg = getattr(sensor, "cfg", None)
        w, h = getattr(cfg, "width", None), getattr(cfg, "height", None)
        if w and h:
            n = int(getattr(sim.scene, "num_envs", 1) or 1)
            out[name] = {"width": int(w), "height": int(h), "num_envs": n,
                         "pixels_per_frame": int(w) * int(h) * n}
            total += out[name]["pixels_per_frame"]
    out["_total_pixels_per_frame"] = total
    return out
