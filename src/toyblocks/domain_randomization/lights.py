"""Scene light randomization -- per-env local lights (sphere/disk) + a shared distant sun, plus
color/temperature sampling. Moved verbatim from render.py (pure samplers + USD/pxr appliers for one
concern). No module globals. pxr imports are function-local so this imports without a launched Isaac
app (pure samplers usable anywhere; the _setup_*/_apply_* functions require a live stage)."""
from __future__ import annotations

import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np



# ---------------------------------------------------------------------------
# Per-env LOCAL lights (in-batch light randomization)
#
# The DomeLight is a single global prim -> its intensity/HDRI is shared by the whole batch
# (per-wave), can't differ per env (proven by the light-linking test). FINITE lights
# (sphere/disk) are different: they're spatially local, so a per-env light positioned at each
# env's origin lights mainly THAT env (with a large --env_spacing the neighbors are past the
# inverse-square falloff / the cameras' far-clip). So local lights CAN be randomized PER ENV
# within one vectorized render -> true in-batch lighting variation.
#
# flags: scene_lights.local_lights = {count, type[sphere|disk], radius, intensity, color
# (white|random|[r,g,b]), height, xy_range, seed}. count/intensity/height accept scalar or
# [min,max]. Sampled per EPISODE (per env within a wave); recorded to episodes[].local_lights.
# ---------------------------------------------------------------------------
LOCAL_LIGHT_ROOT = "/World/LocalLights"


def _kelvin_to_rgb(kelvin: float) -> list:
    """Blackbody color temperature (K) -> linear-ish RGB (Tanner Helland approx), normalized so
    the max channel is 1.0 (color = HUE only; brightness stays with `intensity`). ~3000K warm
    orange, ~6500K near white, higher = cool blue."""
    t = max(1000.0, min(40000.0, float(kelvin))) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * ((t - 60) ** -0.1332047592)
        g = 288.1221695283 * ((t - 60) ** -0.0755148492)
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    rgb = [max(0.0, min(255.0, c)) for c in (r, g, b)]
    m = max(rgb) or 1.0
    return [c / m for c in rgb]


def _sample_color(color, rng) -> Tuple[list, dict]:
    """Sample a light color. Returns (rgb[0..1], meta) where meta records the mode+draw for
    render_params. Accepts: 'white' | 'random' | [r,g,b] | {temperature:K|[min,max]} |
    {hsv:[[h..],[s..],[v..]] or {h,s,v}} | {rgb:[r,g,b]}."""
    def rng_(v):  # scalar or [min,max]
        return float(rng.uniform(v[0], v[1])) if isinstance(v, (list, tuple)) else float(v)
    if color is None:
        return [1.0, 1.0, 1.0], {"mode": "white"}
    if isinstance(color, (list, tuple)) and len(color) == 3 and all(isinstance(c, (int, float)) for c in color):
        return [float(c) for c in color], {"mode": "rgb", "rgb": [float(c) for c in color]}
    if isinstance(color, str):
        if color.lower() == "random":
            rgb = [float(rng.uniform(0.35, 1.0)) for _ in range(3)]
            return rgb, {"mode": "random", "rgb": rgb}
        return [1.0, 1.0, 1.0], {"mode": "white"}
    if isinstance(color, dict):
        if "temperature" in color:
            k = rng_(color["temperature"])
            return _kelvin_to_rgb(k), {"mode": "temperature", "kelvin": round(k, 1)}
        if "hsv" in color:
            import colorsys
            hsv = color["hsv"]
            if isinstance(hsv, dict):
                hr, sr, vr = hsv.get("h", [0, 360]), hsv.get("s", [0, 1]), hsv.get("v", [1, 1])
            else:  # [[h..],[s..],[v..]]
                hr, sr, vr = hsv[0], hsv[1], hsv[2]
            h, s, v = rng_(hr), rng_(sr), rng_(vr)
            rgb = list(colorsys.hsv_to_rgb((h % 360) / 360.0, s, v))
            return rgb, {"mode": "hsv", "hsv": [round(h, 1), round(s, 3), round(v, 3)]}
        if "rgb" in color:
            return [float(c) for c in color["rgb"]], {"mode": "rgb", "rgb": list(color["rgb"])}
    return [1.0, 1.0, 1.0], {"mode": "white"}


def _sample_light(lspec: dict, rng) -> dict:
    """Sample one light's intensity / color / radius / xyz-offset from its own spec (each light
    is parametrized independently). xy from `xy:[[xmin,xmax],[ymin,ymax]]` box or `xy_range:r`."""
    def s(v, default):
        v = lspec.get(v, default) if isinstance(v, str) else v
        return float(rng.uniform(v[0], v[1])) if isinstance(v, (list, tuple)) else float(v)
    intensity = s("intensity", 30000.0)
    # intensity_auto_ratio: [min,max] (or scalar) -> this light's intensity is resolved at wave time
    # as ratio * effective_dome (randomized per episode). Low ratio = soft/no shadow (dome fills it),
    # high ratio = strong dark cast shadow -> a wide range yields both strong- and weak-shadow scenes.
    _auto = lspec.get("intensity_auto_ratio")
    auto_ratio = (float(rng.uniform(_auto[0], _auto[1])) if isinstance(_auto, (list, tuple))
                  else (float(_auto) if _auto is not None else None))
    radius = s("radius", 0.15)
    z = s("height", [0.8, 1.5])
    if "xy" in lspec:  # box [[xmin,xmax],[ymin,ymax]]
        (xmn, xmx), (ymn, ymx) = lspec["xy"]
        x, y = float(rng.uniform(xmn, xmx)), float(rng.uniform(ymn, ymx))
    else:
        r = float(lspec.get("xy_range", 0.5))
        x, y = float(rng.uniform(-r, r)), float(rng.uniform(-r, r))
    rgb, cmeta = _sample_color(lspec.get("color", "white"), rng)
    out = {"intensity": intensity, "auto_ratio": auto_ratio, "radius": radius, "color": rgb,
           "color_meta": cmeta, "offset": [x, y, z]}
    # DistantLight ("sun") params -- sampled ONLY for type: distant so the sphere/disk RNG stream
    # (and thus the existing open_hybrid path) is unchanged. `angle`=angular size deg (small=sharp),
    # `elevation`=deg above horizon (low=long raking shadows), `azimuth`=deg compass direction.
    if (lspec.get("type") or "").lower() == "distant":
        out["ltype"] = "distant"
        out["angle"] = s("angle", [0.3, 1.5])
        out["elevation"] = s("elevation", [22, 60])
        out["azimuth"] = s("azimuth", [0, 360])
        # distance > 0 -> FALLOFF SUN: a SphereLight placed this far in the sun direction instead of a
        # true (infinite) DistantLight. Near-parallel rays + a directional shadow, but inverse-square
        # falloff confines it to its own env (needs env_spacing >> distance). `radius` sets softness
        # (angular size = radius/distance); intensity must be MUCH higher than a DistantLight (~d^2).
        out["distance"] = s("distance", 0.0)
    return out


def _local_light_slot_types(spec: dict) -> list:
    """Per-slot light type ('sphere'|'disk'): from each entry of `lights`, else the shared `type`
    repeated `max(count)` times."""
    lights = spec.get("lights")
    if lights:
        return [(l.get("type") or spec.get("type") or "sphere").lower() for l in lights]
    cnt = spec.get("count", 1)
    n = int(cnt[1] if isinstance(cnt, (list, tuple)) else cnt)
    return [(spec.get("type") or "sphere").lower()] * n


def _setup_local_lights(sim, spec: dict):
    """Pre-create the per-env local lights (one per slot, off until applied). Returns [env][k] prims."""
    from pxr import UsdLux, UsdGeom, Sdf
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    slot_types = _local_light_slot_types(spec)
    # link_per_env: restrict each env's lights to ITS OWN env via USD light-link + shadow-link
    # collections. REQUIRED for a DistantLight ("sun") in a num_envs>1 batch: a distant light has no
    # falloff, so without linking every env is lit by ALL envs' suns (washed shadows, wrong lighting
    # -> col1 != col5). Default OFF to preserve the committed sphere-light paths (those have falloff).
    link_per_env = bool(spec.get("link_per_env", False))
    # per-slot spec (to detect a FALLOFF sun: a `type: distant` light with `distance` > 0 -> built as
    # a positioned SphereLight instead of an infinite DistantLight, so it has inverse-square falloff
    # and stays in its own env WITHOUT light-linking).
    _lspecs = spec.get("lights")
    def _falloff(k, ltype):
        s = (_lspecs[k] if _lspecs else spec).get("distance", 0)
        hi = s[1] if isinstance(s, (list, tuple)) else s
        return ltype == "distant" and float(hi or 0) > 0
    origins = sim.scene.env_origins.cpu().numpy()
    prims = []
    for i in range(sim.num_envs):
        row = []
        for k, ltype in enumerate(slot_types):
            path = f"{LOCAL_LIGHT_ROOT}/env_{i}/L{k}"
            existing = stage.GetPrimAtPath(path)
            if ltype == "distant" and not _falloff(k, ltype):
                # DistantLight = a "sun": parallel rays -> razor-sharp shadows; oriented by rotation
                # (elevation/azimuth), not position. `angle` = angular size (deg); small = sharpest.
                lt = UsdLux.DistantLight(existing) if (existing and existing.IsValid()) \
                    else UsdLux.DistantLight.Define(stage, path)
                if not (existing and existing.IsValid()):
                    lt.CreateAngleAttr(1.0)
                    xf = UsdGeom.Xformable(lt.GetPrim()); xf.ClearXformOpOrder(); xf.AddRotateXYZOp()
            else:
                # DiskLight/SphereLight (positioned) — incl. the falloff sun (a far SphereLight)
                Light = UsdLux.DiskLight if ltype == "disk" else UsdLux.SphereLight
                lt = Light(existing) if (existing and existing.IsValid()) else Light.Define(stage, path)
                if not (existing and existing.IsValid()):
                    lt.CreateRadiusAttr(0.15)
                    xf = UsdGeom.Xformable(lt.GetPrim()); xf.ClearXformOpOrder(); xf.AddTranslateOp()
            if link_per_env:                       # confine this light to /World/envs/env_{i}
                api = UsdLux.LightAPI(lt.GetPrim())
                for coll in (api.GetLightLinkCollectionAPI(), api.GetShadowLinkCollectionAPI()):
                    coll.CreateIncludeRootAttr().Set(False)
                    coll.CreateIncludesRel().SetTargets([Sdf.Path(f"/World/envs/env_{i}")])
            lt.GetIntensityAttr().Set(0.0)
            row.append(lt.GetPrim())
        prims.append(row)
    mode = f"{len(spec['lights'])} independent light(s)" if spec.get("lights") else f"count {spec.get('count', 1)}"
    print(f"[render] local_lights: {len(slot_types)} slot(s) ({mode}) x {sim.num_envs} env(s)"
          f"{' [link_per_env]' if link_per_env else ''}")
    return prims


def _sample_episode_local_lights(spec: dict, episode_indices, seed: int,
                                 indexed: "bool | str" = False) -> Dict[int, list]:
    """Per episode, sample the active lights (reproducible by seed). Each sample carries `_slot`
    (its fixed prim slot, so subsetting preserves per-slot light type). Modes:
      * `lights` list + `count`: activate a RANDOM SUBSET of `count` (int|[min,max]) of the
        defined lights this episode (count omitted -> all of them).
      * `count` alone (no `lights`): `count` lights all drawn from the shared spec."""
    rng = np.random.default_rng(seed)
    light_specs = spec.get("lights")
    cnt = spec.get("count")

    def n_from(c, hi):
        if c is None:
            return hi
        return int(rng.integers(c[0], c[1] + 1)) if isinstance(c, (list, tuple)) else int(c)

    out: Dict[int, list] = {}
    for e in sorted(episode_indices):
        if indexed:
            from toyblocks.domain_randomization.seeds import rng_for_episode
            rng = rng_for_episode(indexed, seed, e, "lights")
        if light_specs:
            m = len(light_specs)
            n = min(max(0, n_from(cnt, m)), m)
            idxs = list(range(m)) if (cnt is None or n == m) else sorted(
                int(j) for j in rng.choice(m, size=n, replace=False))
            out[e] = [dict(_sample_light(light_specs[j], rng), _slot=j) for j in idxs]
        else:
            n = n_from(cnt, 1)
            out[e] = [dict(_sample_light(spec, rng), _slot=k) for k in range(n)]
    return out


def _apply_local_lights(prims, wave, active, ep_lights, origins, eff_dome=None,
                        enabled=True) -> Dict[int, list]:
    """Set each active env's own local lights (intensity/color/radius/position) from its episode
    sample; unused slots -> intensity 0. Each env's lights sit at its own origin -> genuine
    per-env in-batch variation. `enabled=False` (used by a scenario bucket that turns the sun off)
    forces every slot to intensity 0 and records no local_lights for the wave."""
    from pxr import UsdLux, UsdGeom, Gf
    max_count = len(prims[0]) if prims else 0
    applied: Dict[int, list] = {}
    for i in active:
        eid = wave[i]["episode_index"]
        lights = ep_lights.get(eid, [])
        if not enabled:                                   # scenario turned local lights off
            for k in range(max_count):
                UsdLux.LightAPI(prims[i][k]).GetIntensityAttr().Set(0.0)
            applied[eid] = []
            continue
        by_slot = {L.get("_slot", k): L for k, L in enumerate(lights)}  # place each in its fixed slot
        ox, oy, oz = float(origins[i][0]), float(origins[i][1]), float(origins[i][2])
        for k in range(max_count):
            prim = prims[i][k]
            api = UsdLux.LightAPI(prim)
            L = by_slot.get(k)
            if L is not None:
                inten = float(L["intensity"])
                if L.get("auto_ratio") is not None and eff_dome is not None:
                    inten = float(L["auto_ratio"]) * float(eff_dome)  # key<->dome coupling
                    L["intensity"] = inten  # record the resolved value in render_params
                api.GetIntensityAttr().Set(inten)
                api.GetColorAttr().Set(Gf.Vec3f(*L["color"]))
                if L.get("ltype") == "distant" and float(L.get("distance", 0) or 0) > 0:
                    # FALLOFF SUN: a SphereLight positioned `distance` away in the sun direction
                    # (elevation/azimuth), pointing back at the scene by omni-directional emission.
                    # Near-parallel rays over the ~1m scene -> a directional shadow, but inverse-square
                    # falloff means neighbor envs (env_spacing >> distance) get a negligible share.
                    d = float(L["distance"])
                    el = math.radians(float(L["elevation"])); az = math.radians(float(L["azimuth"]))
                    dxu = math.cos(el) * math.cos(az); dyu = math.cos(el) * math.sin(az); dzu = math.sin(el)
                    rad = prim.GetAttribute("inputs:radius") or prim.GetAttribute("radius")
                    if rad:
                        rad.Set(float(L.get("radius", 0.3)))
                    op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
                    op.Set(Gf.Vec3d(ox + d * dxu, oy + d * dyu, oz + 0.5 + d * dzu))
                elif L.get("ltype") == "distant":
                    # DistantLight: set angular size + orient it. Default emits along -Z (straight
                    # down when unrotated); RotateX(90-elev) tilts it to `elevation` above the
                    # horizon, RotateZ(azimuth) spins the compass direction. Position is irrelevant.
                    ang = prim.GetAttribute("inputs:angle") or prim.GetAttribute("angle")
                    if ang:
                        ang.Set(float(L.get("angle", 1.0)))
                    op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
                    op.Set(Gf.Vec3f(90.0 - float(L["elevation"]), 0.0, float(L["azimuth"])))
                else:
                    rad = prim.GetAttribute("inputs:radius") or prim.GetAttribute("radius")
                    if rad:
                        rad.Set(float(L["radius"]))
                    op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
                    op.Set(Gf.Vec3d(ox + L["offset"][0], oy + L["offset"][1], oz + L["offset"][2]))
            else:
                api.GetIntensityAttr().Set(0.0)  # inactive slot this episode
        applied[eid] = lights
    # DIAGNOSTIC (opt-in via DR_DUMP_LIVE=1): read back what is ACTUALLY on the USD stage after
    # applying, so a num_envs-dependent divergence shows up even when render_params records
    # identical values. Prints once per wave.
    if os.environ.get("DR_DUMP_LIVE"):
        try:
            import omni.usd as _ou
            from pxr import UsdLux as _UL, UsdGeom as _UG
            st = _ou.get_context().get_stage()
            tot = 0.0
            for i in active:
                for k in range(max_count):
                    v = _UL.LightAPI(prims[i][k]).GetIntensityAttr().Get()
                    tot += float(v or 0.0)
            dome = st.GetPrimAtPath("/World/DomeLight") or st.GetPrimAtPath("/Environment/DomeLight")
            dv = float(_UL.LightAPI(dome).GetIntensityAttr().Get() or 0) if dome and dome.IsValid() else -1
            sun = st.GetPrimAtPath("/World/GlobalSun")
            sv = float(_UL.LightAPI(sun).GetIntensityAttr().Get() or 0) if sun and sun.IsValid() else -1
            print(f"[DR_DUMP_LIVE] envs={len(active)} local_light_sum={tot:.1f} "
                  f"mean_per_env={tot/max(1,len(active)):.1f} dome={dv:.1f} global_sun={sv:.1f}", flush=True)
        except Exception as _e:
            print(f"[DR_DUMP_LIVE] failed: {_e}", flush=True)
    return applied


def _setup_global_sun(spec: dict):
    """One SHARED DistantLight for the WHOLE scene (all envs). A directional sun with NO per-env
    cross-contamination: a per-env DistantLight has no distance falloff, so in a num_envs>1 batch it
    lights EVERY env (washed shadows), and USD light-linking is not honored by the default RTX mode.
    This single sun is sampled per WAVE (like the dome) -> one consistent shadow direction per batch.
    Use `scene_lights.sun` (same keys as a `type: distant` local light) instead of a per-env sun for
    correct multi-env renders."""
    from pxr import UsdLux, UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    lt = UsdLux.DistantLight.Define(stage, "/World/GlobalSun")
    lt.CreateAngleAttr(1.0)
    xf = UsdGeom.Xformable(lt.GetPrim()); xf.ClearXformOpOrder(); xf.AddRotateXYZOp()
    lt.GetIntensityAttr().Set(0.0)
    print("[render] global sun: 1 shared DistantLight (per-wave; no per-env cross-contamination)")
    return lt.GetPrim()


def _apply_global_sun(prim, L: dict) -> dict:
    """Point + set the shared sun from a sampled light dict (elevation/azimuth/angle/intensity/color).
    Returns the recorded params for render_params.json."""
    from pxr import UsdLux, UsdGeom, Gf
    api = UsdLux.LightAPI(prim)
    api.GetIntensityAttr().Set(float(L["intensity"]))
    api.GetColorAttr().Set(Gf.Vec3f(*L["color"]))
    ang = prim.GetAttribute("inputs:angle") or prim.GetAttribute("angle")
    if ang:
        ang.Set(float(L.get("angle", 1.0)))
    op = UsdGeom.Xformable(prim).GetOrderedXformOps()[0]
    op.Set(Gf.Vec3f(90.0 - float(L["elevation"]), 0.0, float(L["azimuth"])))
    return {"intensity": round(float(L["intensity"]), 1), "elevation": round(float(L["elevation"]), 1),
            "azimuth": round(float(L["azimuth"]), 1), "angle": round(float(L.get("angle", 1.0)), 3),
            "color_meta": L.get("color_meta")}


# ============================================================================
# EXPERIMENTAL / MODULAR: per-env "environment sphere" (flags['env_sphere'])
# A large textured sphere per env used as a PER-ENV background/skybox (each env its own HDRI,
# randomized within a batch — unlike the single global dome). `emissive: false` = HDRI as a
# diffuse surface (lit by the scene); `emissive: true` = HDRI self-illuminating (visible skybox
# that also feeds GI). Reuses `_resolve_skybox_textures` (set/names/textures_dir) for the pool
# and `_apply_material_to_shader` for the texture.
#   TO REMOVE this whole idea: delete this section + the 3 `env_sphere` call-sites in replay_job.
#   CAVEAT: the left_camera far-clips at 3m, so `radius` must be < 3m to show in that view.
# ============================================================================
