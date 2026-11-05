"""Table (object) material randomization -- moved verbatim from rerender.py so this DR concern
(pure samplers + USD/pxr appliers) is importable outside the rerender pipeline. No module globals:
the former DEFAULT_TABLE_MATERIALS fallback is now the explicit `default_table_materials` argument of
_resolve_material_specs. pxr imports are function-local, so this module imports without a launched
Isaac app (pure samplers usable anywhere; the _apply_* functions require a live stage)."""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

from .resolve import expand_path as _expand_path



# ---------------------------------------------------------------------------
# Object material randomization (flags.yaml `materials`)
#
# flags["materials"] maps an arbitrary label -> {object, library, seed, randomize, overrides}:
#   object:     scene entity name (e.g. "workspace_table"), spawned per-env at
#               /World/envs/env_<i>/<object>/... (see toyblocks/scenes/sim2real.py
#               `_load_asset_cfgs_from_json`, which gives every scene.json object its own
#               "{ENV_REGEX_NS}/<name>" prim path). Because each env is a FULL, independent
#               USD prim clone (required for RigidObject physics state to diverge per env),
#               its material Shader prim is ALSO independent -> unlike the skybox's single
#               global DomeLight, an object's material CAN be sampled per EPISODE WITHIN a
#               batch (one draw per env), not just once per batch. That is what this does.
#   library:    path to a materials JSON (list of {name, <OmniPBR input>: value, ...}).
#               Property names match OmniPBR.mdl shader inputs 1:1 (texture path values are
#               resolved relative to the library file's directory). Omitted inputs keep
#               whatever the base table.usda authored.
#   randomize:  optional +/- jitter ranges layered on the CHOSEN material's own values,
#               sampled per episode (same style as the camera `randomize` block). Only
#               numeric/2-vector inputs are jitterable (MATERIAL_JITTER_KEYS below).
#   overrides:  optional per-material-NAME jitter range overrides (mirrors
#               `randomize.cameras.<cam>`).
# The final per-episode material (name + resolved properties) is written into
# render_params.json episodes[].materials, in the same "applied" style as cameras/skybox.
MATERIAL_VEC2_KEYS = ("texture_translate", "texture_scale", "detail_texture_translate",
                     "detail_texture_scale")
MATERIAL_VEC3_KEYS = ("diffuse_color_constant", "diffuse_tint", "emissive_color")
MATERIAL_JITTER_KEYS = ("texture_translate", "texture_scale", "texture_rotate",
                        "albedo_brightness", "albedo_add", "reflection_roughness_constant",
                        "metallic_constant", "specular_level")
MATERIAL_ASSET_KEYS = ("diffuse_texture", "reflectionroughness_texture", "metallic_texture",
                       "normalmap_texture", "ao_texture", "opacity_texture",
                       "emissive_color_texture", "emissive_mask_texture", "ORM_texture")
# OmniPBR.mdl-declared defaults for jitterable keys a material omits (so jitter is added to
# the MDL default, not silently to 0.0, e.g. an un-authored albedo_brightness is 1.0 not 0.0).
MATERIAL_MDL_DEFAULTS = {
    "texture_translate": [0.0, 0.0], "texture_scale": [1.0, 1.0], "texture_rotate": 0.0,
    "albedo_brightness": 1.0, "albedo_add": 0.0, "reflection_roughness_constant": 0.5,
    "metallic_constant": 0.0, "specular_level": 0.5,
}
# Friendly aliases -> canonical OmniPBR input names (accept the human-readable form used by the
# authoring/export JSON, e.g. table_textures_2/materials.json, alongside the snake_case names).
MATERIAL_NAME_ALIASES = {
    "id": "name",
    "albedo map": "diffuse_texture", "albedo_map": "diffuse_texture",
    "albedo brightness": "albedo_brightness",
    "albedo add": "albedo_add",
    "color tint": "diffuse_tint", "color_tint": "diffuse_tint",
    "texture translate": "texture_translate",
    "texture rotate": "texture_rotate",
    "texture tiling": "texture_scale", "texture_tiling": "texture_scale", "texture scale": "texture_scale",
    "roughness": "reflection_roughness_constant", "metallic": "metallic_constant",
    "specular": "specular_level",
}
# HSV tint components (any of these keys) collapse into diffuse_tint = {hue,saturation,value};
# `vibrance` is treated as HSV Value (per the authoring convention).
_TINT_HSV_KEYS = {
    "color tint hue": "hue", "color_tint_hue": "hue", "tint hue": "hue", "hue": "hue",
    "color tint saturation": "saturation", "color_tint_saturation": "saturation",
    "tint saturation": "saturation", "saturation": "saturation",
    "color tint vibrance": "value", "color tint value": "value", "color_tint_value": "value",
    "tint value": "value", "vibrance": "value", "value": "value",
}


def _srange(v, rng):
    """Sample a scalar spec: a number -> itself; a [min,max] pair -> uniform in the range."""
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return float(rng.uniform(v[0], v[1]))
    return float(v)


def _normalize_material(raw: dict) -> dict:
    """Canonicalize one material dict: map friendly keys -> OmniPBR names, collect HSV tint into
    `diffuse_tint` = {hue,saturation,value}, set name (from id) + enabled (default True).
    Values keep their scalar | [min,max] | {x,y} | [r,g,b] shape (sampled later)."""
    m, tint = {}, {}
    for k, v in raw.items():
        kl = str(k).strip().lower()
        if kl in _TINT_HSV_KEYS:
            tint[_TINT_HSV_KEYS[kl]] = v
            continue
        m[MATERIAL_NAME_ALIASES.get(kl, k)] = v
    if tint:
        m["diffuse_tint"] = tint            # HSV dict -> converted to RGB at sample time
    m["name"] = m.get("name") or raw.get("id") or raw.get("name")
    m.setdefault("enabled", True)
    return m


def _sample_material_props(canon: dict, rng) -> dict:
    """Produce a concrete OmniPBR props dict from a (normalized) material, sampling any embedded
    ranges: scalars pass through, [min,max] -> uniform, {x,y}/[x,y] vec2 sampled per-axis,
    diffuse_tint given as HSV dict -> sampled + converted to RGB. Textures pass through as-is."""
    import colorsys
    out = {}
    for k, v in canon.items():
        if k in ("name", "enabled"):
            continue
        if k in MATERIAL_ASSET_KEYS:
            out[k] = v
        elif k == "diffuse_tint" and isinstance(v, dict):   # HSV -> RGB
            h = _srange(v.get("hue", 0.0), rng) % 1.0
            s = _srange(v.get("saturation", 0.0), rng)
            val = _srange(v.get("value", 1.0), rng)
            out["diffuse_tint"] = list(colorsys.hsv_to_rgb(h, s, val))
        elif k in MATERIAL_VEC3_KEYS:
            out[k] = [_srange(c, rng) for c in v]
        elif k in MATERIAL_VEC2_KEYS:
            xy = (v.get("x", 0.0), v.get("y", 0.0)) if isinstance(v, dict) else (v[0], v[1])
            out[k] = [_srange(xy[0], rng), _srange(xy[1], rng)]
        else:
            out[k] = _srange(v, rng)
    return out


def _load_material_library(path: str) -> Tuple[List[dict], str]:
    """Load a materials JSON (a list, or {"materials": [...]}) -> (normalized enabled materials,
    base_dir for resolving relative texture paths). Accepts both the snake_case format and the
    friendly authoring format (id / 'albedo map' / 'color tint hue' / {x,y} / embedded ranges)."""
    path = os.path.abspath(_expand_path(path))
    with open(path) as f:
        data = json.load(f)
    raw = data["materials"] if isinstance(data, dict) else data
    mats = []
    for i, m in enumerate(raw):
        cm = _normalize_material(m)
        if not cm.get("enabled", True):
            continue
        cm["name"] = cm.get("name") or f"material_{i}"
        mats.append(cm)
    return mats, os.path.dirname(path)


def _resolve_material_specs(flags: dict, default_table_materials: str) -> Dict[str, dict]:
    """Parse flags['materials'] into {label: {object, materials, lib_dir, randomize,
    overrides, seed}}. Skips (with a warning) any entry missing/failing to load `library`."""
    mats_flags = (flags or {}).get("materials")
    if not mats_flags:
        return {}
    out: Dict[str, dict] = {}
    for key, spec in mats_flags.items():
        # `library` omitted -> the default table material library (toyblocks.TABLE_MATERIALS_FILEPATH,
        # env-overridable). Keeps flag yamls free of hardcoded asset paths.
        lib_path = spec.get("library") or default_table_materials
        try:
            materials, lib_dir = _load_material_library(lib_path)
        except Exception as e:
            print(f"[rerender] materials: '{key}' failed to load library {lib_path}: {e}")
            continue
        if not materials:
            print(f"[rerender] materials: '{key}' library {lib_path} is empty; skipping")
            continue
        rz = spec.get("randomize") or {}
        out[key] = {
            "object": spec.get("object", key),
            "materials": materials,
            "lib_dir": lib_dir,
            "randomize": {k: rz[k] for k in MATERIAL_JITTER_KEYS if rz.get(k) is not None},
            "overrides": spec.get("overrides") or {},
            "seed": int(spec.get("seed", 0)),
        }
        print(f"[rerender] materials: '{key}' -> object={out[key]['object']}, "
              f"{len(materials)} material(s) from {lib_path}, "
              f"randomize={out[key]['randomize']}")
    return out


def _find_shader_prim(root_path: str):
    """First Shader-typed prim under `root_path` (depth-first), or None."""
    from pxr import Usd
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(root_path)
    if not (root and root.IsValid()):
        return None
    for p in Usd.PrimRange(root):
        if p.GetTypeName() == "Shader":
            return p
    return None


def _resolve_object_shader_prims(object_name: str, num_envs: int) -> List[Any]:
    """Per-env Shader prim for a scene object spawned at /World/envs/env_<i>/<object_name>/...
    (one independent prim per env; None where not found)."""
    prims = [_find_shader_prim(f"/World/envs/env_{i}/{object_name}") for i in range(num_envs)]
    missing = sum(1 for p in prims if p is None)
    if missing:
        print(f"[rerender] materials: object '{object_name}' shader not found in "
              f"{missing}/{num_envs} env(s)")
    return prims


def _read_shader_props(shader_prim, keys) -> dict:
    """Current authored values of `keys` (+ diffuse_texture) off a Shader prim, for the
    render_params.json `initial` (pre-randomization) baseline."""
    if shader_prim is None:
        return {}
    out = {}
    tex_attr = shader_prim.GetAttribute("inputs:diffuse_texture")
    if tex_attr and tex_attr.Get():
        v = tex_attr.Get()
        out["diffuse_texture"] = str(getattr(v, "path", v))
    for key in keys:
        attr = shader_prim.GetAttribute(f"inputs:{key}")
        v = attr.Get() if attr else None
        if v is None:
            continue
        out[key] = [float(v[0]), float(v[1])] if key in MATERIAL_VEC2_KEYS else float(v)
    return out


def _initial_material_params(specs: Dict[str, dict], shader_prims: Dict[str, list]) -> dict:
    """Baseline (from table.usda, pre-randomization) material params, one entry per label."""
    return {key: _read_shader_props((shader_prims.get(key) or [None])[0], MATERIAL_JITTER_KEYS)
            for key in specs}


def _baseline_material_props(specs: Dict[str, dict], shader_prims: Dict[str, list]) -> Dict[str, dict]:
    """Per label: the asset's OWN value for every key the library can write, ``None`` where the
    asset does not author it.

    Distinct from :func:`_initial_material_params`, which is for render_params provenance and
    DROPS unauthored keys -- here the ``None`` is the point. Applying a draw on top of this
    baseline means a key the chosen material omits reverts to the asset (or, if the asset does
    not author it either, to the MDL default) instead of keeping whatever the PREVIOUS episode's
    material left on the prim. One entry per label, not per env: the asset is the same in every
    env, so the first prim is representative.
    """
    out: Dict[str, dict] = {}
    for key, spec in specs.items():
        prim = (shader_prims.get(key) or [None])[0]
        if prim is None:
            out[key] = {}
            continue
        writable = set(MATERIAL_JITTER_KEYS)
        for mat in spec.get("materials", []) or []:
            writable |= {k for k in mat if k != "name"}
        base: dict = {}
        for k in sorted(writable):
            attr = prim.GetAttribute(f"inputs:{k}")
            v = attr.Get() if attr else None
            if v is None:
                base[k] = None
            elif k in MATERIAL_ASSET_KEYS:
                base[k] = str(getattr(v, "path", v))
            elif k in MATERIAL_VEC3_KEYS or k in MATERIAL_VEC2_KEYS:
                base[k] = [float(x) for x in v]
            elif isinstance(v, bool):
                base[k] = v
            else:
                base[k] = float(v)
        out[key] = base
    return out


def _combined_material_props(base: dict, jitter: dict) -> dict:
    """Chosen material's authored properties with this episode's sampled jitter added on top
    (vec2 keys added componentwise; a key the base material didn't author falls back to its
    OmniPBR.mdl default, e.g. un-authored albedo_brightness is 1.0, not 0.0)."""
    props = dict(base)
    for key, delta in jitter.items():
        if key in MATERIAL_VEC2_KEYS:
            cur = props.get(key, MATERIAL_MDL_DEFAULTS[key])
            props[key] = [float(cur[0]) + delta[0], float(cur[1]) + delta[1]]
        else:
            props[key] = float(props.get(key, MATERIAL_MDL_DEFAULTS.get(key, 0.0))) + delta
    return props


def _sample_episode_materials(specs: Dict[str, dict], episode_indices,
                              indexed: "bool | str" = False) -> Dict[int, Dict[str, dict]]:
    """Per episode, per configured label: pick one material (uniform over its library) and
    sample +/- jitter for any `randomize` (or per-material-name `overrides`) properties.
    Each label draws from its OWN rng (seeded by that label's `seed`, independent of every
    other label/object). Returns {episode_index: {label: {material_index, material_name,
    jitter}}}."""
    rngs = {key: np.random.default_rng(spec["seed"]) for key, spec in specs.items()}
    out: Dict[int, Dict[str, dict]] = {}
    for e in sorted(episode_indices):
        if indexed:
            from toyblocks.domain_randomization.seeds import rng_for_episode
            rngs = {k: rng_for_episode(indexed, sp['seed'], e, f"materials:{k}")
                    for k, sp in specs.items()}
        picks = {}
        for key, spec in specs.items():
            rng = rngs[key]
            mats = spec["materials"]
            idx = int(rng.integers(0, len(mats)))
            base = mats[idx]
            # 1) sample the material's own embedded ranges (incl. HSV tint) -> concrete props
            sampled = _sample_material_props(base, rng)
            # 2) optional extra +/- jitter from flags.randomize / per-material overrides (usually
            #    empty for range-based libraries; kept for back-compat with scalar libraries)
            rz = dict(spec["randomize"])
            rz.update(spec["overrides"].get(base["name"], {}))
            jitter = {}
            for prop, rng_val in rz.items():
                if not rng_val:
                    continue
                if prop in MATERIAL_VEC2_KEYS:
                    r = [rng_val, rng_val] if isinstance(rng_val, (int, float)) else [float(x) for x in rng_val]
                    jitter[prop] = [float(rng.uniform(-r[0], r[0])), float(rng.uniform(-r[1], r[1]))]
                else:
                    jitter[prop] = float(rng.uniform(-float(rng_val), float(rng_val)))
            picks[key] = {"material_name": base["name"],
                          "props": _combined_material_props(sampled, jitter)}
        out[e] = picks
    return out


def _resolve_material_texture(value: str, lib_dir: str) -> str:
    """Resolve a material texture path -> absolute. Tries the path as given (rel to lib_dir),
    then falls back to <lib_dir>/<basename> (so an authoring path like './Textures/x.png' still
    finds x.png sitting directly in the library dir)."""
    cand = value if os.path.isabs(value) else os.path.join(lib_dir, value)
    if os.path.exists(cand):
        return os.path.abspath(cand)
    alt = os.path.join(lib_dir, os.path.basename(value))
    if os.path.exists(alt):
        return os.path.abspath(alt)
    print(f"[rerender] materials: WARNING texture not found: '{value}' (tried {cand}, {alt})")
    return os.path.abspath(cand)


def _apply_material_to_shader(shader_prim, props: dict, lib_dir: str) -> None:
    """Set each property in `props` (OmniPBR input names) on `shader_prim`. Texture-asset
    inputs resolve relative to `lib_dir`. Values are wrapped in the exact Gf/Sdf type the USD
    attribute expects (Usd.Attribute.Set() does NOT auto-convert a plain python list/tuple to
    e.g. Gf.Vec3f -- passing one raises 'Type mismatch: expected GfVec3f, got vector<VtValue>')."""
    from pxr import Sdf, Gf

    def _set(attr_name, value, sdf_type):
        attr = shader_prim.GetAttribute(attr_name)
        if not attr:
            attr = shader_prim.CreateAttribute(attr_name, sdf_type)
        attr.Set(value)

    for key, value in props.items():
        if value is None:
            # authored by neither the draw nor the asset -> clear so the prim falls
            # back to the MDL default rather than keeping a stale value.
            _a = shader_prim.GetAttribute(f"inputs:{key}")
            if _a:
                _a.Clear()
            continue
        if key == "name":
            continue
        attr_name = f"inputs:{key}"
        if key in MATERIAL_ASSET_KEYS:
            _set(attr_name, Sdf.AssetPath(_resolve_material_texture(value, lib_dir)),
                 Sdf.ValueTypeNames.Asset)
        elif key in MATERIAL_VEC3_KEYS:
            _set(attr_name, Gf.Vec3f(*[float(v) for v in value]), Sdf.ValueTypeNames.Color3f)
        elif key in MATERIAL_VEC2_KEYS:
            _set(attr_name, Gf.Vec2f(*[float(v) for v in value]), Sdf.ValueTypeNames.Float2)
        elif isinstance(value, bool):
            _set(attr_name, value, Sdf.ValueTypeNames.Bool)
        elif isinstance(value, int):
            _set(attr_name, value, Sdf.ValueTypeNames.Int)
        else:
            _set(attr_name, float(value), Sdf.ValueTypeNames.Float)


def _apply_episode_materials(specs: Dict[str, dict], shader_prims: Dict[str, list],
                             wave, active, ep_mats: Dict[int, dict],
                             baselines: Dict[str, dict] = None) -> Dict[int, dict]:
    """Apply this wave's per-episode material pick to each active env's OWN shader prim (one
    independent USD prim per env -> every env in the batch can get a DIFFERENT material at
    the same time; no cross-env toggling trick needed, unlike the global skybox dome).
    Returns {episode_index: {label: {name, <applied properties...>}}} for render_params.json."""
    applied: Dict[int, dict] = {}
    for i in active:
        eidx = wave[i]["episode_index"]
        picks = ep_mats.get(eidx, {})
        rec = {}
        for key, pick in picks.items():
            spec = specs[key]
            prims = shader_prims.get(key) or []
            prim = prims[i] if i < len(prims) else None
            # baselines is None => today's behaviour: write only what this draw sampled, so a
            # key it omits keeps the previous episode's value (materials bleed). Passing
            # baselines reverts those keys to the asset first. Opt-in, see setup(
            # material_baseline_revert=).
            final = ({**(baselines or {}).get(key, {}), **pick["props"]}
                     if baselines else pick["props"])
            if prim is not None:
                _apply_material_to_shader(prim, final, spec["lib_dir"])
            rec[key] = {"name": pick["material_name"],
                        **{k: v for k, v in final.items() if k != "name"}}
        if rec:
            applied[eidx] = rec
    return applied


def dump_applied_materials(specs: Dict[str, dict], shader_prims: Dict[str, list],
                           max_envs: int = 8) -> dict:
    """Read back what is ACTUALLY authored on the stage, per material label and per env.

    The DR *values* the samplers draw are already known to match between rerender and any other
    consumer. This answers the different question: did the same values actually land on the same
    shader, with the same textures bound? A wrong texture path, a texture that failed to resolve,
    or a value written to a prim that is not the one being rendered would all look like "the table
    texture is different" while every sampled value matched.

    Returns {label: {"object": <scene object>, "envs": [ {attr: value}, ... ]}}. Values are plain
    JSON types so two runs can be diffed directly. Reads both the jitterable numeric keys and the
    texture ASSET keys, which _read_shader_props deliberately does not cover.
    """
    out: dict = {}
    for label, spec in (specs or {}).items():
        prims = (shader_prims or {}).get(label) or []
        envs = []
        for prim in prims[:max_envs]:
            if prim is None:
                envs.append(None)
                continue
            rec = {"prim": str(prim.GetPath())}
            for key in MATERIAL_ASSET_KEYS:
                attr = prim.GetAttribute(f"inputs:{key}")
                v = attr.Get() if attr else None
                if v is not None:
                    rec[key] = str(getattr(v, "path", v))
            for key in MATERIAL_JITTER_KEYS:
                attr = prim.GetAttribute(f"inputs:{key}")
                v = attr.Get() if attr else None
                if v is None:
                    continue
                rec[key] = ([float(v[0]), float(v[1])] if key in MATERIAL_VEC2_KEYS
                            else float(v))
            envs.append(rec)
        out[label] = {"object": spec.get("object"), "envs": envs}
    return out
