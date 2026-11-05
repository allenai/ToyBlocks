"""Load + schema-validate a rerender DR flags yaml, so rerender and eval parse the same file the
same way with the same checks. Pure (no Isaac). Validation is LENIENT by design -- every existing
rerender_flags_*.yaml must load without errors; unknown keys and suspicious shapes are reported as
warnings, and only structural type errors (e.g. `scenarios.mix` not a list) are hard errors (raised
only when strict=True).
"""
from __future__ import annotations

from typing import Any, List, Tuple

import yaml

# Known top-level keys (anything else -> a warning, not an error, so forward-compat yamls still load).
TOP_LEVEL = {
    "carb_settings", "hide_objects", "env_spacing", "render_mode", "lighting_intensity",
    "skybox", "exposure", "scene_lights", "env_sphere", "scenarios", "materials",
    "randomize", "cameras",
    # Per-env block glyph swap for task augmentation: each block draws one of the 36 ASSIGNABLE
    # atlas glyphs (A-Z and 1-10 -- there is no 0 tile; "10" is the one two-character glyph) and `pddl_goal` + the LeRobot `task` string are rewritten to
    # match, so the instruction still describes what the camera sees. Either a bool or a mapping
    # {glyphs: ...} constraining what each character may become. See glyphs.py.
    "randomize_task_letters",
    # Per-env CUBOID colour swap. Procedural colour blocks (spec type "cuboid") each draw a
    # colour from a named HSV palette, and `pddl_goal` + the LeRobot `task` string are rewritten
    # to match -- for a cuboid the colour word IS the referring expression, so recolouring without
    # rewriting says "the red block" over a cyan one. Either a bool or {colors: [{name, hsv}]}.
    # Disjoint from randomize_task_letters by spec type, not by name. See block_colors.py.
    "randomize_block_colors",
    # RAM-backed HDRI path cache: {enabled, dir, preload, max_gb, verbose}. Rewrites the asset
    # path handed to USD to a tmpfs copy so the per-wave re-read hits memory, not weka.
    "texture_cache",
}
_SKYBOX_KEYS = {"set", "sets", "names", "textures", "textures_dir", "texture_format", "rotation_deg",
                "intensity", "visible_background", "seed", "shuffle", "dedup", "exhaust_before_repeat"}
_ENV_SPHERE_KEYS = {"set", "sets", "names", "textures", "textures_dir", "radius", "far_clip",
                    "material", "emissive", "emissive_intensity", "emissive_normalize",
                    "emissive_clamp", "cast_shadows", "backdrop_only", "rotation_deg", "shuffle",
                    "dedup", "seed", "exhaust_before_repeat"}
_EXPOSURE_KEYS = {"auto", "iso", "fstop", "shutter", "iso_dist", "seed"}
_SCENE_LIGHT_KEYS = {"fill_intensity", "fill_auto_ratio", "local_lights", "sun"}


def _is_range(v) -> bool:
    return isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v)


def _is_scalar_or_range(v) -> bool:
    return isinstance(v, (int, float)) or _is_range(v)


def validate_flags(flags: Any) -> Tuple[List[str], List[str]]:
    """Return (errors, warnings). errors = structural problems that would crash; warnings = unknown
    keys / suspicious shapes. Empty errors => the yaml is structurally usable."""
    errs: List[str] = []
    warns: List[str] = []
    if flags is None:
        return errs, warns
    if not isinstance(flags, dict):
        return ["top-level flags must be a mapping"], warns

    for k in flags:
        if k not in TOP_LEVEL:
            warns.append(f"unknown top-level key '{k}'")

    if "env_spacing" in flags and not isinstance(flags["env_spacing"], (int, float)):
        errs.append("env_spacing must be a number")
    if "lighting_intensity" in flags and not _is_range(flags["lighting_intensity"]):
        warns.append("lighting_intensity should be a [min,max] range")

    sky = flags.get("skybox")
    if sky is not None:
        if not isinstance(sky, dict):
            errs.append("skybox must be a mapping")
        else:
            for k in sky:
                if k not in _SKYBOX_KEYS:
                    warns.append(f"skybox: unknown key '{k}'")
            if not any(k in sky for k in ("set", "sets", "names", "textures", "textures_dir")):
                warns.append("skybox: no texture selection (set/sets/names/textures/textures_dir)")
            if "intensity" in sky and not _is_scalar_or_range(sky["intensity"]):
                warns.append("skybox.intensity should be a number or [min,max]")

    es = flags.get("env_sphere")
    if es is not None:
        if not isinstance(es, dict):
            errs.append("env_sphere must be a mapping")
        else:
            for k in es:
                if k not in _ENV_SPHERE_KEYS:
                    warns.append(f"env_sphere: unknown key '{k}'")
            if "radius" in es and not isinstance(es["radius"], (int, float)):
                errs.append("env_sphere.radius must be a number")
            if "emissive_intensity" in es and not _is_scalar_or_range(es["emissive_intensity"]):
                warns.append("env_sphere.emissive_intensity should be a number or [min,max]")
            if "emissive_clamp" in es and not _is_range(es["emissive_clamp"]):
                warns.append("env_sphere.emissive_clamp should be a [lo,hi] range")

    exp = flags.get("exposure")
    if exp is not None:
        if not isinstance(exp, dict):
            errs.append("exposure must be a mapping")
        else:
            for k in exp:
                if k not in _EXPOSURE_KEYS:
                    warns.append(f"exposure: unknown key '{k}'")
            for k in ("iso", "fstop", "shutter"):
                if k in exp and not _is_scalar_or_range(exp[k]):
                    warns.append(f"exposure.{k} should be a number or [min,max]")

    sl = flags.get("scene_lights")
    if sl is not None:
        if not isinstance(sl, dict):
            errs.append("scene_lights must be a mapping")
        else:
            for k in sl:
                if k not in _SCENE_LIGHT_KEYS:
                    warns.append(f"scene_lights: unknown key '{k}'")
            fi = sl.get("fill_intensity")
            if fi is not None and not (_is_scalar_or_range(fi)
                                       or (isinstance(fi, str) and fi.lower() == "auto")):
                warns.append("scene_lights.fill_intensity should be number | [min,max] | 'auto'")

    scen = flags.get("scenarios")
    if scen is not None:
        mix = scen.get("mix") if isinstance(scen, dict) else scen
        if not isinstance(mix, list) or not mix:
            errs.append("scenarios.mix must be a non-empty list")
        else:
            for i, s in enumerate(mix):
                if not isinstance(s, dict):
                    errs.append(f"scenarios.mix[{i}] must be a mapping")
                    continue
                if "name" not in s:
                    warns.append(f"scenarios.mix[{i}] has no 'name'")
                for wk in ("prob", "probability", "weight"):
                    if wk in s and not isinstance(s[wk], (int, float)):
                        errs.append(f"scenarios.mix[{i}].{wk} must be a number")

    mats = flags.get("materials")
    if mats is not None:
        if not isinstance(mats, dict):
            errs.append("materials must be a mapping of label -> spec")
        else:
            for label, spec in mats.items():
                if not isinstance(spec, dict):
                    errs.append(f"materials.{label} must be a mapping")
                elif "object" not in spec:
                    warns.append(f"materials.{label} has no 'object' (scene entity name)")

    rnd = flags.get("randomize")
    if rnd is not None and not isinstance(rnd, dict):
        errs.append("randomize must be a mapping")

    rbc = flags.get("randomize_block_colors")
    if rbc is not None and rbc is not False:
        if not isinstance(rbc, (bool, dict, list, tuple)):
            errs.append("randomize_block_colors must be a bool or {colors: [{name, hsv}]}, "
                        f"got {type(rbc).__name__}")
        else:
            try:
                from .block_colors import parse_palette
                pal = parse_palette(rbc)
                if len(pal) == 1:
                    warns.append(f"randomize_block_colors: only one colour ({pal[0][0]!r}), so "
                                 f"every cuboid becomes it")
            except ValueError as e:
                errs.append(str(e))

    rtl = flags.get("randomize_task_letters")
    # A LIST is valid too -- parse_glyph_set and interpreter.setup both accept one and the
    # train/test charset splits use it. The validator rejecting it printed a spurious
    # "[dr.config] ERROR" on every wave of every glyph run while the run worked fine, which is
    # exactly the noise that hides a real error.
    if rtl is not None and not isinstance(rtl, (bool, dict, list, tuple)):
        errs.append("randomize_task_letters must be a bool, a list, or a mapping, e.g. "
                    "true, [A-F, 1-9], {glyphs: [A-Z]} or {glyphs: {\"*\": [A-Z], 7: [1-9]}}")
    elif isinstance(rtl, (list, tuple)):
        try:
            from .glyphs import parse_glyph_set
            parse_glyph_set(rtl, "randomize_task_letters")
        except ValueError as e:
            errs.append(f"randomize_task_letters: {e}")
    elif isinstance(rtl, dict):
        unknown = sorted(set(rtl) - {"glyphs"})
        if unknown:
            errs.append(f"randomize_task_letters: unknown key(s) {unknown}; only 'glyphs'")
        if "glyphs" not in rtl:
            warns.append("randomize_task_letters has no 'glyphs' key, so every character keeps "
                         "its own glyph and nothing will be swapped")
        else:
            # Parse here so a typo is one loud line at startup, not a mis-labelled dataset.
            from . import glyphs as _g
            try:
                _sets = _g.resolve_glyph_sets(rtl["glyphs"])
            except ValueError as e:
                errs.append(f"randomize_task_letters: {e}")
            else:
                _moving = sorted(g for g, a in _sets.items() if a != frozenset((g,)))
                if not _moving:
                    warns.append("randomize_task_letters: every character resolves to itself, "
                                 "so nothing will be swapped")

    return errs, warns


def canonicalize_flags(flags: Any) -> Any:
    """Rewrite dr_schema 3 spellings to the internal (v2) names the interpreter reads:
    scene_lights.global_distant -> sun, scenarios.mix[].local_lights -> sun, and
    type: sphere_light -> distant + distance. Idempotent; a no-op for v2 yamls."""
    if not isinstance(flags, dict):
        return flags
    sl = flags.get("scene_lights")
    if isinstance(sl, dict):
        if "global_distant" in sl and "sun" not in sl:
            sl["sun"] = sl.pop("global_distant")
        ll = sl.get("local_lights")
        if isinstance(ll, dict):
            ll.pop("random_subset", None)
            for L in (ll.get("lights") or []):
                if isinstance(L, dict) and (L.get("type") or "").lower() == "sphere_light":
                    L["type"] = "distant"
                    L.setdefault("distance", [40, 70])
    scen = flags.get("scenarios")
    mix = scen.get("mix") if isinstance(scen, dict) else None
    for m in (mix or []):
        if isinstance(m, dict) and "local_lights" in m and "sun" not in m:
            m["sun"] = m.pop("local_lights")
    return flags


def load_flags(path: str, *, strict: bool = False, verbose: bool = True) -> dict:
    """Load a DR flags yaml and validate its schema. Returns the flags dict (never None). Prints
    warnings; with strict=True, raises ValueError on any structural error, else prints them."""
    with open(path) as f:
        flags = yaml.safe_load(f) or {}
    errs, warns = validate_flags(flags)
    flags = canonicalize_flags(flags)  # validate what the author wrote, run the internal names
    if verbose:
        for w in warns:
            print(f"[dr.config] WARN {path}: {w}")
    if errs:
        msg = "; ".join(errs)
        if strict:
            raise ValueError(f"invalid DR flags {path}: {msg}")
        if verbose:
            for e in errs:
                print(f"[dr.config] ERROR {path}: {e}")
    return flags
