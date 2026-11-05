"""Per-env "environment sphere" backdrop -- a large emissive UV-sphere per env showing an HDRI as a
camera-only skybox. Moved verbatim from rerender.py. The in-process luminance memo (_LUMINANCE) is
keyed by source path and holds successes only (see _hdri_luminance), and SkyBuckets takes the
default HDRI dir explicitly. pxr/omni/cv2 imports are function-local so this module imports without a
launched Isaac app (the sampler is pure; _setup_*/_apply_* require a live stage)."""
from __future__ import annotations

import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import texture_cache as _texcache

from .buckets import SkyBuckets
from .materials import _apply_material_to_shader


ENV_SPHERE_ROOT = "/World/EnvSpheres"
# Threads for the HDRI fetch + luminance pass; cv2.imread and the file copy release the GIL.
_LUM_WORKERS = int(os.environ.get("TOYBLOCKS_HDRI_WORKERS", "16"))


def _make_uv_sphere_mesh(stage, path: str, radius: float, rings: int = 64, sectors: int = 128):
    """Author a UsdGeom.Mesh UV sphere with EQUIRECTANGULAR `st` UVs (u=longitude, v=latitude)
    so a latlong HDRI projects correctly — the analytic UsdGeom.Sphere / isaaclab mesh sphere
    author NO st, which is why the diffuse HDRI came out as a flat average color."""
    from pxr import UsdGeom, Vt, Gf, Sdf
    pts, sts = [], []
    for i in range(rings + 1):
        v = i / rings
        theta = v * math.pi                     # 0..pi   (pole to pole)
        st, ct = math.sin(theta), math.cos(theta)
        for j in range(sectors + 1):
            u = j / sectors
            phi = u * 2.0 * math.pi             # 0..2pi  (around)
            pts.append((radius * st * math.cos(phi), radius * st * math.sin(phi), radius * ct))
            # FLIP u (1-u): viewing the sphere from INSIDE mirrors the equirect horizontally, so a
            # plain `u` shows the HDRI mirrored vs an Omni DomeLight. 1-u aligns the sphere backdrop
            # to the dome exactly (verified in test_skybox_rot.py: flipped u, 0 offset -> best match).
            sts.append((1.0 - u, 1.0 - v))      # equirectangular, un-mirrored for inside view
    faces, counts = [], []
    W = sectors + 1
    for i in range(rings):
        for j in range(sectors):
            a = i * W + j
            faces += [a, a + 1, a + W + 1, a + W]
            counts.append(4)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in pts]))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(faces))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray(counts))
    mesh.CreateDoubleSidedAttr(True)
    pv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                 UsdGeom.Tokens.vertex)
    pv.Set(Vt.Vec2fArray([Gf.Vec2f(*s) for s in sts]))
    return mesh


def _set_camera_far_clip(far: float) -> None:
    """Raise the far clip on every scene camera so a large env_sphere (infinite-looking backdrop) is
    NOT clipped away. Some env cams (e.g. left_camera) ship with a tiny far clip (~3m), which is why
    the sphere previously had to stay < 3m. Set env_sphere.far_clip to use a big radius instead."""
    import omni.usd
    from pxr import UsdGeom, Gf
    stage = omni.usd.get_context().get_stage()
    n = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Camera):
            cam = UsdGeom.Camera(prim)
            cr = cam.GetClippingRangeAttr().Get()
            near = float(cr[0]) if cr is not None else 0.01
            cam.GetClippingRangeAttr().Set(Gf.Vec2f(near, float(far)))
            n += 1
    print(f"[rerender] env_sphere: raised camera far-clip -> {far} on {n} camera prim(s)")


def _setup_env_spheres(sim, spec: dict):
    """Create one large double-sided UV sphere + OmniPBR material per env. Returns [env] ->
    (shader_prim, rotateZ_op). Positioned at the env origin; textured per-episode in _apply.
    Uses a proper equirectangular-UV mesh so the HDRI actually projects (see _make_uv_sphere_mesh)."""
    from pxr import UsdGeom, UsdShade, Gf, Sdf
    from omni.usd.commands import CreateMdlMaterialPrimCommand
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    radius = float(spec.get("radius", 2.5))
    # Skybox behaviour flags (make the sphere behave like a game-engine skybox, not an occluder):
    #  * cast_shadows (default False) -> set primvars:doNotCastShadows so the sphere does NOT block
    #    other lights: a DistantLight "sun" (or any light outside the sphere) passes THROUGH it and
    #    still lights + shadows the interior scene ("lets light pass"). RTX honours this primvar.
    #  * backdrop_only (default False) -> also set primvars:invisibleToSecondaryRays: the sphere is
    #    seen by the CAMERA only (primary rays), invisible to GI/reflection/shadow rays, so it is a
    #    PURE visual backdrop that contributes NO lighting (scene lit solely by explicit lights).
    no_shadow = not bool(spec.get("cast_shadows", False))
    backdrop_only = bool(spec.get("backdrop_only", False))
    # material: "usdpreview" (portable UNLIT UsdPreviewSurface skybox — emissive HDRI via an explicit
    # `st` primvar reader; robust equirectangular projection, verified in test_skybox.py) or "omnipbr"
    # (legacy MDL emissive/diffuse). Default omnipbr for back-compat.
    material = str(spec.get("material", "omnipbr")).lower()
    # far_clip: raise all camera far clips so a BIG radius (infinite-looking backdrop, matching the
    # dome's extent) is not clipped by the env cams' small default far clip.
    far_clip = spec.get("far_clip")
    if far_clip is not None:
        _set_camera_far_clip(float(far_clip))
    origins = sim.scene.env_origins.cpu().numpy()
    out = []
    for i in range(sim.num_envs):
        base = f"{ENV_SPHERE_ROOT}/env_{i}"
        mesh = _make_uv_sphere_mesh(stage, f"{base}/Sphere", radius)
        prim = mesh.GetPrim()
        pv = UsdGeom.PrimvarsAPI(prim)
        if no_shadow:                              # light passes through -> no occlusion/shadow
            pv.CreatePrimvar("doNotCastShadows", Sdf.ValueTypeNames.Bool,
                             UsdGeom.Tokens.constant).Set(True)
        if backdrop_only:                          # camera-only: no GI/reflection/shadow interaction
            pv.CreatePrimvar("invisibleToSecondaryRays", Sdf.ValueTypeNames.Bool,
                             UsdGeom.Tokens.constant).Set(True)
        xf = UsdGeom.Xformable(prim); xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(float(origins[i][0]), float(origins[i][1]),
                                         float(origins[i][2]) + 0.5))
        rot_op = xf.AddRotateZOp(UsdGeom.XformOp.PrecisionDouble)
        mat_path = f"{base}/Looks/EnvMat"
        if material == "usdpreview":
            # UNLIT skybox: UsdPreviewSurface, emissiveColor <- UsdUVTexture(HDRI) wired through an
            # explicit `st` primvar reader (equirectangular); diffuse black. Per-episode file + scale
            # are set in _apply_env_spheres. Handle returned = the UsdUVTexture shader prim.
            mat = UsdShade.Material.Define(stage, mat_path)
            sh = UsdShade.Shader.Define(stage, mat_path + "/Shader"); sh.CreateIdAttr("UsdPreviewSurface")
            sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0, 0, 0))
            sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(1.0)
            sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
            rdr = UsdShade.Shader.Define(stage, mat_path + "/stReader"); rdr.CreateIdAttr("UsdPrimvarReader_float2")
            rdr.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            st_out = rdr.CreateOutput("result", Sdf.ValueTypeNames.Float2)
            tex = UsdShade.Shader.Define(stage, mat_path + "/Tex"); tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("raw")
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_out)
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset)      # set per-episode
            tex.CreateInput("scale", Sdf.ValueTypeNames.Float4)    # set per-episode (HDR tame)
            rgb = tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            sh.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(rgb)
            mat.CreateSurfaceOutput().ConnectToSource(sh.CreateOutput("surface", Sdf.ValueTypeNames.Token))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat)
            out.append((tex.GetPrim(), rot_op))
        else:
            CreateMdlMaterialPrimCommand(mtl_url="OmniPBR.mdl", mtl_name="OmniPBR",
                                         mtl_path=mat_path, stage=stage, select_new_prim=False).do()
            UsdShade.MaterialBindingAPI(prim).Apply(prim)
            UsdShade.MaterialBindingAPI(prim).Bind(UsdShade.Material(stage.GetPrimAtPath(mat_path)))
            out.append((stage.GetPrimAtPath(f"{mat_path}/Shader"), rot_op))
    print(f"[rerender] env_sphere: {sim.num_envs} UV-sphere(s) radius {radius} material={material} "
          f"emissive={bool(spec.get('emissive'))} no_shadow={no_shadow} backdrop_only={backdrop_only}")
    return out


_LUMINANCE: Dict[str, float] = {}   # source path -> median luminance (successes only)
_LUMINANCE_LOCK = threading.Lock()


def _read_luminance(path: str) -> Optional[float]:
    import cv2
    a = cv2.imread(path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_COLOR)  # float32 BGR HDR
    if a is None:
        return None
    L = 0.2126 * a[..., 2] + 0.7152 * a[..., 1] + 0.0722 * a[..., 0]
    return float(np.median(L[::4, ::4]))  # 16x subsample for speed


def _hdri_luminance(src: str) -> Optional[float]:
    """Median linear luminance of an HDRI (robust to bright sun/window spots), cached in-process.
    Used by env_sphere `emissive_normalize` to scale the emissive PER SKYBOX so every backdrop is
    visible regardless of the HDRI's absolute brightness -- e.g. the indoor-2018 (Laval) HDRIs are
    stored ~100-1000x dimmer than Poly Haven, so a fixed emissive scale makes them near-black.

    Keyed by the SOURCE path and read from the texture cache's copy when there is one, else from
    the source: a size-bounded cache can evict the copy between resolve() and the read. Only
    successes are memoized, so one failed read does not leave that backdrop un-normalized for the
    rest of the process."""
    with _LUMINANCE_LOCK:
        if src in _LUMINANCE:
            return _LUMINANCE[src]
    val = None
    for path in dict.fromkeys((_texcache.resolve(src), src)):
        try:
            val = _read_luminance(path)
        except Exception as e:
            print(f"[rerender] env_sphere: HDRI luminance read failed for {os.path.basename(path)}: {e}")
        if val is not None:
            break
    if val is not None:
        with _LUMINANCE_LOCK:
            _LUMINANCE[src] = val
    return val


def _sample_episode_env_spheres(spec: dict, episode_indices, seed: int, default_hdri_dir: str,
                                indexed: "bool | str" = False) -> Dict[int, dict]:
    """Per episode (=per env in a wave): pick one HDRI + Z rotation + emissive_intensity
    (scalar, or [min,max] to sweep/randomize per env). Reproducible by seed.
    env_sphere.sets: [ {set|names|textures..., prob}, ... ] -> MULTI-BUCKET: each episode first
    picks a bucket by weight, then advances THAT bucket's OWN without-replacement queue. Omitted ->
    single pool (unchanged; shuffle:false keeps the resolved list in order for a 1:1 episode map).
    emissive_normalize: true -> scale the emissive PER SKYBOX so every backdrop shows at a common
    level (emissive_intensity is then the TARGET median luminance; actual scale = target/median_lum,
    clamped to emissive_clamp [lo,hi] default [0.05,100]). Off (default) -> emissive_intensity is a
    fixed scale as before."""
    rng = np.random.default_rng(seed)
    buckets = SkyBuckets(spec, rng, default_hdri_dir)
    if not buckets:
        print("[rerender] env_sphere: WARNING no textures resolved")
    elif buckets.multi:
        print(f"[rerender] env_sphere: {len(buckets.buckets)} buckets {buckets.summary()} "
              f"(per-episode bucket -> per-set queue)")
    rot = spec.get("rotation_deg", [0, 360])
    ei = spec.get("emissive_intensity", 1.0)
    normalize = bool(spec.get("emissive_normalize", False))
    clamp = spec.get("emissive_clamp", [0.05, 100.0])
    if normalize:
        print(f"[rerender] env_sphere: emissive_normalize ON -> per-skybox scale = target/median_lum "
              f"(target={ei}, clamp={list(clamp)}) so dim HDRIs stay visible")
    # Draw every episode first (the rng stream is consumed in exactly the original order), then
    # fetch + measure the unique HDRIs in parallel: serially, one copy off weka plus one full HDR
    # decode per episode was ~16 min of a 310-episode item's setup on Beaker.
    draws: List[Tuple[int, Optional[str], Optional[str], float, float]] = []
    for e in sorted(episode_indices):
        if indexed:
            from toyblocks.domain_randomization.seeds import rng_for_episode
            rng = rng_for_episode(indexed, seed, e, "env_sphere")
        tex, bname = buckets.next(rng)           # single bucket -> no rng draw (back-compat)
        tex = str(tex) if tex else None
        r = float(rng.uniform(rot[0], rot[1])) if isinstance(rot, (list, tuple)) else float(rot)
        target = float(rng.uniform(ei[0], ei[1])) if isinstance(ei, (list, tuple)) else float(ei)
        draws.append((e, tex, bname, r, target))
    lums: Dict[str, Optional[float]] = {}
    if normalize:
        # read the CACHED copy: else emissive_normalize pulls the full file off weka
        # once per unique HDRI even with the renderer already on tmpfs.
        uniq = sorted({d[1] for d in draws if d[1]})
        with ThreadPoolExecutor(max_workers=max(1, min(_LUM_WORKERS, len(uniq)))) as ex:
            lums = dict(zip(uniq, ex.map(_hdri_luminance, uniq)))
    out: Dict[int, dict] = {}
    for e, tex, bname, r, target in draws:
        em, lum = target, None
        if normalize and tex:
            lum = lums.get(tex)
            if lum is not None:
                em = float(np.clip(target / max(lum, 1e-3), clamp[0], clamp[1]))
        rec = {"texture": tex, "rotation": r, "emissive_intensity": em}
        if buckets.multi:
            rec["bucket"] = bname
        if normalize:
            rec["target"], rec["hdri_lum"] = round(target, 3), (round(lum, 5) if lum is not None else None)
        out[e] = rec
    return out


def _apply_env_spheres(env_prims, spec: dict, wave, active, ep_data) -> Dict[int, dict]:
    """Set each active env's own sphere: HDRI as diffuse (emissive off) or self-illuminating
    (emissive on) + Z rotation. Genuine per-env background within one batch."""
    emissive = bool(spec.get("emissive", False))
    material = str(spec.get("material", "omnipbr")).lower()
    applied: Dict[int, dict] = {}
    for i in active:
        eid = wave[i]["episode_index"]
        data = ep_data.get(eid, {})
        tex = data.get("texture")
        em_int = float(data.get("emissive_intensity", 1.0))
        shader, rot_op = env_prims[i]
        if material == "usdpreview":
            # unlit UsdPreviewSurface skybox: set the UsdUVTexture `file` + HDR-tame `scale`.
            # `emissive_intensity` (sampled per-episode) is reused as the scale here (~0.15-0.4).
            from pxr import Sdf as _Sdf, Gf as _Gf
            if tex:
                shader.GetAttribute("inputs:file").Set(_Sdf.AssetPath(_texcache.resolve(tex)))
            sc = em_int
            shader.GetAttribute("inputs:scale").Set(_Gf.Vec4f(sc, sc, sc, 1.0))
            rot_op.Set(float(data.get("rotation", 0.0)))
            applied[eid] = {"texture": os.path.basename(tex) if tex else None,
                            "rotation_deg": data.get("rotation"), "material": "usdpreview",
                            "scale": round(sc, 3)}
            continue
        solid = spec.get("emissive_color")  # diagnostic: solid emissive color instead of HDRI
        if emissive and solid is not None:
            props = {"enable_emission": True, "emissive_color": [float(c) for c in solid],
                     "emissive_intensity": em_int, "diffuse_color_constant": [0.0, 0.0, 0.0]}
            _apply_material_to_shader(shader, props, "")
        elif tex:
            tex_abs = _texcache.resolve(tex)
            if emissive:
                props = {"enable_emission": True, "emissive_color_texture": tex_abs,
                         "emissive_intensity": em_int, "diffuse_color_constant": [0.0, 0.0, 0.0]}
            else:
                props = {"enable_emission": False, "diffuse_texture": tex_abs,
                         "albedo_brightness": 1.0, "diffuse_color_constant": [1.0, 1.0, 1.0]}
            _apply_material_to_shader(shader, props, "")
        rot_op.Set(float(data.get("rotation", 0.0)))
        applied[eid] = {"texture": os.path.basename(tex) if tex else None,
                        "rotation_deg": data.get("rotation"), "emissive": emissive,
                        **({"emissive_intensity": round(em_int, 1)} if emissive else {})}
    return applied
