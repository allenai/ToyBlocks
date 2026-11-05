"""The yaml INTERPRETER: given a launched sim + a DR flags dict, it builds the per-problem seed hub,
creates all DR prims (dome buckets, local/sun lights, env-spheres), resolves material specs + shader
prims, samples every per-episode value, and captures the pre-randomization `initial` baseline --
exactly as rerender's replay_job did (moved verbatim). Returns a namespace of all of it so any
consumer (rerender's replay loop, eval's wave runner) shares ONE interpreter with no code duplication.

pxr/omni imports live inside the underlying stage functions, so this module imports without a launched
Isaac app; `setup()` itself requires a live sim.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np

from .assets import AssetPaths
from . import texture_cache as _texcache
from . import camera as _cam, env_sphere as _es, exposure as _exp, lights as _li, \
    materials as _mat, seeds as _sd, skybox as _sk
from .buckets import SkyBuckets

# bind the exact private names the moved body calls -> package functions (verbatim body, no edits)
_problem_seed_salt = _sd._problem_seed_salt
_resolve_cam_ranges = _cam._resolve_cam_ranges
_is_link_mounted = _cam._is_link_mounted
_sample_episode_offsets = _cam._sample_episode_offsets
_initial_camera_params = _cam._initial_camera_params
_scene_camera_params = _cam._scene_camera_params
_setup_local_lights = _li._setup_local_lights
_sample_episode_local_lights = _li._sample_episode_local_lights
_setup_global_sun = _li._setup_global_sun
_sphere_fill_intensity = _sk._sphere_fill_intensity
_dome_intensity = _sk._dome_intensity
_set_render_mode = _exp._set_render_mode
_apply_carb_settings = _exp._apply_carb_settings
_apply_exposure = _exp._apply_exposure
_dump_exposure_settings = _exp._dump_exposure_settings
_setup_env_spheres = _es._setup_env_spheres
_sample_episode_materials = _mat._sample_episode_materials
_initial_material_params = _mat._initial_material_params
_baseline_material_props = _mat._baseline_material_props
_resolve_object_shader_prims = _mat._resolve_object_shader_prims
# per-wave camera appliers (used by apply_wave_cameras)
_apply_camera_local_offsets = _cam._apply_camera_local_offsets
_apply_episode_camera_fov = _cam._apply_episode_camera_fov
_recorded_static_bases = _cam.recorded_static_bases
# per-wave appliers (used by apply_wave)
_apply_skybox = _sk._apply_skybox
_apply_global_sun = _li._apply_global_sun
_sample_light = _li._sample_light
_apply_episode_materials = _mat._apply_episode_materials
_apply_local_lights = _li._apply_local_lights
_apply_env_spheres = _es._apply_env_spheres


def setup(sim, flags, job=None, units=None, episodes=(), override_cams=None, *,
          camera_base="data", cam_base_K: dict | None = None,
          salt=None, episode_keys=None, assets: AssetPaths = None,
          indexed_seeding: "bool | str" = False,
          material_baseline_revert: bool = False):
    # `indexed_seeding` carries the MODE, not just a flag: False = the legacy stream, True/"indexed"
    # = per-episode from (seed, episode_index), "hash" = per-episode from a hash of
    # (seed, concern, episode_index). It is threaded verbatim into every sampler, so one value
    # decides the whole run's draws and render_params records which was used.
    assets = assets or AssetPaths.default()
    flags = flags or {}
    _texcache.configure(flags.get("texture_cache"))
    # dir-injecting adapters (were rerender globals DEFAULT_HDRI_DIR / DEFAULT_TABLE_MATERIALS)
    def _SkyBuckets(flag, rng):
        return SkyBuckets(flag, rng, assets.hdri_dirs)
    def _resolve_material_specs(_flags):
        return _mat._resolve_material_specs(_flags, assets.table_materials)
    def _sample_episode_env_spheres(spec, idx, seed):
        return _es._sample_episode_env_spheres(spec, idx, seed, assets.hdri_dirs,
                                              indexed=indexed_seeding)

    # A caller with no dataset (a gym wrapper) has no `job` dict and no `units` records, but
    # both reduce to one scalar and one list of episode keys -- so take those directly rather
    # than forking the function. rerender.py keeps passing job/units positionally and lands on
    # exactly the same two values.
    if salt is None and job is None:
        raise TypeError("setup() needs either job= (dataset) or salt= (a caller-chosen int)")
    prob_salt = int(salt) if salt is not None else _problem_seed_salt(job)
    if episode_keys is None and units is None:
        raise TypeError("setup() needs either units= (dataset) or episode_keys=")
    ep_keys = (list(episode_keys) if episode_keys is not None
               else [e["episode_index"] for e in units])
    all_cam_keys = [k for k, s in sim.scene.sensors.items()
                    if k.startswith("camera:") and hasattr(s, "set_world_poses")]
    cam_ranges, rseed_base = _resolve_cam_ranges(flags, all_cam_keys)
    rseed = int(rseed_base) + int(prob_salt)
    randomize_cams = {k: sim.scene.sensors[k] for k in cam_ranges}
    rand_link = {k: c for k, c in randomize_cams.items() if _is_link_mounted(c)}
    rand_static = {k: c for k, c in randomize_cams.items() if k not in rand_link}
    ep_offsets = (_sample_episode_offsets(cam_ranges, ep_keys, rseed,
                                      indexed=indexed_seeding)
                  if randomize_cams else {})
    if randomize_cams:
        print(f"[rerender] per-episode camera randomization: {sorted(randomize_cams)} "
              f"(all via local USD mount; static={sorted(rand_static)}, link={sorted(rand_link)}; "
              f"seed={rseed_base}+problem_salt={prob_salt} -> {rseed})")
        override_cams = {}  # randomization supersedes recorded-pose matching
    li = (flags or {}).get("lighting_intensity")
    light_range = li if (isinstance(li, (list, tuple)) and len(li) == 2) else None
    light_rng = np.random.default_rng(rseed + 12345)
    # --- Skybox / HDRI environment (flags['skybox']): per-BATCH (dome is global) ---
    sky = (flags or {}).get("skybox") or {}
    sky_format = sky.get("texture_format") or "latlong"
    sky_rot = sky.get("rotation_deg")           # [min,max] | scalar | None
    sky_int = sky.get("intensity")              # [min,max] | scalar | None
    sky_int_dist = str(sky.get("intensity_dist", "uniform")).lower()  # uniform | log
    # visible_background: True/False (fixed), or a probability p in (0,1) -> the dome is drawn
    # (bright HDRI backdrop) with prob p per batch, else hidden (lighting-only) -> that batch goes
    # genuinely dark/moody. Randomizing it is the extra lever for TRUE darkness alongside low ISO.
    _sky_vb_raw = sky.get("visible_background", False)
    sky_vb_prob = (float(_sky_vb_raw) if isinstance(_sky_vb_raw, (int, float))
                   and not isinstance(_sky_vb_raw, bool) else None)
    sky_visible_bg = bool(_sky_vb_raw) if sky_vb_prob is None else True
    # per-problem diversity (b15373f0): salt the (possibly fixed) skybox seed with prob_salt so
    # each problem draws a different HDRI/rotation sequence. rseed_base+prob_salt (not rseed) so an
    # explicit skybox.seed in the flags still diverges across problems.
    sky_rng = np.random.default_rng(int(sky.get("seed", rseed_base)) + 999 + prob_salt)
    # skybox.shuffle: false -> cycle the resolved textures IN ORDER (match a sphere run on the same
    # skyboxes: episode/wave i -> textures[i]). Default: shuffle.
    # skybox.sets: [ {set|names|textures..., prob}, ... ] -> MULTI-BUCKET selection. Each wave first
    # picks a bucket by weight (prob/probability/weight; equal by default), then advances THAT
    # bucket's OWN without-replacement shuffled queue. Omitted -> single pool (unchanged behavior).
    sky_buckets = _SkyBuckets(sky, sky_rng)
    sky_textures = sky_buckets.pool
    # sky_order: single-bucket queue (for the render_params `order` record); combined pool for multi.
    sky_order = sky_buckets.buckets[0]["order"] if len(sky_buckets.buckets) == 1 else sky_textures
    # Same trap as _sseed/_w: assigned only under a conditional but handed to the returned
    # namespace unconditionally. `missing` needs a skybox and `desc` needs fill or local
    # lights, so any yaml without those died here. Nothing reads either value back --
    # rerender rebinds desc to a local it never uses -- so the empty defaults are inert.
    missing = []
    desc = ""
    if sky_textures:
        missing = [t for t in sky_textures if not os.path.exists(t)]
        if missing:
            print(f"[rerender] WARNING skybox textures not found: {missing}")
        if sky_buckets.multi:
            print(f"[rerender] skybox: {len(sky_buckets.buckets)} buckets {sky_buckets.summary()} "
                  f"total {len(sky_textures)} texture(s), format={sky_format}, rotation_deg={sky_rot}, "
                  f"intensity={sky_int}, selection=per-wave bucket -> per-set queue")
        else:
            print(f"[rerender] skybox: {len(sky_textures)} texture(s), format={sky_format}, "
                  f"rotation_deg={sky_rot}, intensity={sky_int}, selection=cycle (without replacement)")
    # --- Scene fill light (flags['scene_lights']['fill_intensity']): controls the neutral
    # SphereLight bundled in the ground-plane asset (a single GLOBAL prim), sampled/set per
    # BATCH like the dome. Modes: number (fixed; 0=off) | [min,max] (sampled/batch) |
    # "auto" (fill = fill_auto_ratio * this batch's dome/HDRI intensity, so it tracks the
    # skybox and stays subordinate) | omitted (leave the scene default untouched). ---
    scene_lights = (flags or {}).get("scene_lights") or {}
    fill_spec = scene_lights.get("fill_intensity")  # None|number|[min,max]|"auto"
    fill_auto = isinstance(fill_spec, str) and fill_spec.lower() == "auto"
    fill_range = fill_spec if (isinstance(fill_spec, (list, tuple)) and len(fill_spec) == 2) else None
    fill_fixed = fill_spec if isinstance(fill_spec, (int, float)) else None
    fill_auto_ratio = float(scene_lights.get("fill_auto_ratio", 1.0))
    fill_rng = np.random.default_rng(rseed + 4242)
    if fill_spec is not None:
        desc = ("auto (ratio %.2f x dome)" % fill_auto_ratio if fill_auto
                else (f"range {list(fill_range)}" if fill_range is not None else f"fixed {fill_fixed}"))
        print(f"[rerender] scene_lights.fill_intensity = {desc} "
              f"(default fill was {_sphere_fill_intensity()})")
    # --- Per-env local lights (scene_lights.local_lights): the ONE way to randomize lighting
    # PER ENV within a batch (the dome can't). Set up once here; sampled per episode + applied
    # per wave below. ---
    local_spec = scene_lights.get("local_lights")
    local_prims = _setup_local_lights(sim, local_spec) if local_spec else []
    local_rng_seed = (int(local_spec.get("seed", rseed_base)) + prob_salt) if local_spec else 0
    # --- Global SHARED sun (scene_lights.sun): one DistantLight for all envs, sampled per WAVE.
    # Use this (not a per-env `local_lights` distant) for correct multi-env renders -> no washed
    # shadows from N per-env suns. Same keys as a distant light (elevation/azimuth/angle/intensity). ---
    sun_spec = scene_lights.get("sun")
    sun_prim = _setup_global_sun(sun_spec) if sun_spec else None
    sun_rng = (np.random.default_rng(int(sun_spec.get("seed", rseed_base)) + prob_salt)
               if sun_spec else None)
    # optional RTX render-mode override (e.g. RealTimePathTracing = RT 2.0 for USD light-linking)
    _set_render_mode((flags or {}).get("render_mode"))
    _apply_carb_settings((flags or {}).get("carb_settings"))  # e.g. raise the firefly clamp
    # --- RTX exposure (flags['exposure']): {auto: bool, iso|fstop|shutter: [min,max]|scalar,
    # iso_dist: log, seed}. Disables auto-exposure once here; per-wave values sampled + applied
    # below (global, like the dome). Dumps the settings so we can VERIFY they exist + take effect. ---
    exp_spec = (flags or {}).get("exposure")
    # rseed_base+prob_salt so a fixed exposure.seed still diverges per problem (b15373f0 policy).
    exp_rng = (np.random.default_rng(int(exp_spec.get("seed", rseed_base)) + 777 + prob_salt)
               if exp_spec else None)
    if exp_spec is not None:
        _dump_exposure_settings("(before override)")
        if exp_spec.get("auto") is not None:
            _apply_exposure(auto=bool(exp_spec["auto"]))
            print(f"[rerender] exposure: auto-exposure -> {bool(exp_spec['auto'])}")
    # --- EXPERIMENTAL per-env environment sphere (flags['env_sphere']): a per-env background
    # sphere (see the _setup_env_spheres section). Set up once here; sampled + applied below. ---
    env_sphere_spec = (flags or {}).get("env_sphere")
    env_sphere_prims = _setup_env_spheres(sim, env_sphere_spec) if env_sphere_spec else []
    env_sphere_seed = (int(env_sphere_spec.get("seed", rseed_base)) + prob_salt) if env_sphere_spec else 0
    # --- Scenario buckets (flags['scenarios']): a per-WAVE MIXTURE of coordinated "looks". Each
    # wave samples one scenario by weight; the scenario OVERRIDES that wave's global levers
    # (skybox_intensity[+_dist], iso[+_dist]/fstop/shutter, fill_intensity) and toggles the local
    # `sun` on/off. This is the ONE way to get BOTH a hard-shadow distribution AND a soft
    # open-hybrid distribution in a single run — those looks need DIFFERENT global dome+ISO+fill,
    # which are per-wave, so a single set of ranges can't produce both. Omitted -> single look
    # (unchanged). Keys per scenario: name, prob (== probability == weight; relative, normalized),
    # sun:bool, and any of the per-wave override keys above.
    scen_spec = (flags or {}).get("scenarios")
    _scen_mix = (scen_spec.get("mix") if isinstance(scen_spec, dict) else scen_spec) or None
    # _sseed/_w are only assigned inside `if _scen_mix:` below, yet both are passed to the
    # returned namespace unconditionally -- so every yaml without a `scenarios:` block
    # (67 of the 106 checked-in flag yamls) died here with UnboundLocalError. Bind them
    # with the other scenario locals; when _scen_mix is truthy both are overwritten, so
    # a yaml that worked before is bit-identical.
    scen_rng = scen_p = _sseed = _w = None
    if _scen_mix:
        # rseed_base+prob_salt so a fixed scenarios.seed still diverges per problem (b15373f0 policy).
        _sseed = ((int(scen_spec.get("seed", rseed_base)) + prob_salt)
                  if isinstance(scen_spec, dict) else rseed)
        scen_rng = np.random.default_rng(_sseed + 31337)
        # per-bucket selection probability: `prob` (== `probability` == `weight`, first one wins;
        # default 1.0). RELATIVE and normalized -> 0.7/0.3 gives 70%/30%, 2/1 gives 66%/33%, so
        # they need not sum to 1. Tune these to draw more/less from a bucket.
        def _scen_weight(s):
            for k in ("prob", "probability", "weight"):
                if k in s:
                    return float(s[k])
            return 1.0
        _w = np.array([_scen_weight(s) for s in _scen_mix], dtype=float)
        scen_p = _w / _w.sum()
        # scenarios drive per-wave ISO -> the exposure/auto-exposure path must be live even if the
        # base flags omit an `exposure:` block. Synthesize a minimal one (auto-exposure off) here.
        if exp_spec is None and any(k in s for s in _scen_mix for k in ("iso", "fstop", "shutter")):
            exp_spec = {"auto": False}
            exp_rng = np.random.default_rng(rseed + 777)
            _apply_exposure(auto=False)
            print("[rerender] scenarios: enabled RTX exposure path (auto=off) for per-wave ISO")
        print(f"[rerender] scenarios: {[s.get('name', '?') for s in _scen_mix]} "
              f"weights={scen_p.round(3).tolist()} (per-wave mixture)")
    # --- Object material randomization (flags['materials']): per-EPISODE, since each scene
    # object is its own per-env USD prim clone (see _resolve_object_shader_prims docstring).
    object_specs = _resolve_material_specs(flags)
    if object_specs and prob_salt:
        # Salt each label's seed so material picks diverge across problems for the same index.
        object_specs = {
            k: {**spec, "seed": int(spec["seed"]) + prob_salt}
            for k, spec in object_specs.items()
        }
    episode_records: list = []
    # Pristine dome intensity BEFORE any per-batch randomization, for render_params.json's
    # "initial" (from-data) block. Read at finalize it would be the last wave's sampled value.
    initial_dome = _dome_intensity()
    initial_fill = _sphere_fill_intensity()
    # Initial (from-data) camera params; per-episode records below are stored in THIS space
    # (absolute = initial + sampled offset), so they are directly comparable to `initial`.
    # Record the base the RENDERER will use, not always the dataset's. `--camera_base scene`
    # leaves the static cameras on cfg.offset, so reporting the recorded pose there described a
    # camera that was never rendered (and made a fixed run indistinguishable from a broken one).
    if camera_base == "scene":
        # Keys come from the SCENE, not from the dataset. Taking them from episodes[0]["cam_pose"]
        # asked the SOURCE which cameras to record -- but a camera-less source records none, so the
        # list was empty on exactly the runs --camera_base scene exists to serve, and
        # `initial.cameras` came out {}. That emptied every episodes[].cameras too, because the
        # per-episode record composes offsets only for keys present here, leaving render_params
        # with the DR deltas but no absolute camera to compose them onto.
        # FOV from the sim-build snapshot the renderer composes onto (cam_base_K), not the live
        # camera: on a sim reused across problems that still holds the last wave's jittered FOV.
        _init_cams = _scene_camera_params(sim, all_cam_keys, base_K=cam_base_K)
    else:
        _init_cams = _initial_camera_params(episodes)
    initial = {"cameras": _init_cams,
               "camera_base": camera_base,
               "lighting": {"dome_intensity": initial_dome, "fill_intensity": initial_fill}}
    # Record the RESOLVED skybox input pool (the full candidate list this run drew from), not
    # just the selection criteria in flags -> render_params stays reproducible even if the sets
    # manifest / hdris dir changes later. episodes[].skybox.texture are the per-wave samples.
    if sky_textures:
        initial["skybox"] = {
            "set": sky.get("set"),
            "names": sky.get("names"),
            "textures_dir": sky.get("textures_dir"),
            "count": len(sky_textures),
            "selection": "cycle",
            "textures": [os.path.basename(t) for t in sky_textures],
            "order": [os.path.basename(t) for t in sky_order],
        }
        if sky_buckets.multi:
            # multi-bucket: per-wave pick a bucket by weight, then advance that bucket's own queue
            initial["skybox"]["selection"] = "bucket->queue"
            initial["skybox"]["buckets"] = sky_buckets.metadata()

    # cube_scale is an edge length in metres (1.0 = not recorded). rerender matches it itself, by
    # respawning blocks per problem (rerender.plan_block_sizes); this flag is informational only.
    n_obj_warn = any(abs(e["cube_scale"] - 1.0) > 1e-6 for e in episodes)

    device = sim.device
    NE = sim.num_envs

    # Resolve each configured object's per-env shader prims once (cheap stage lookup), sample
    # every episode's material pick + jitter up front (reproducible by seed, indexed by
    # episode_index like ep_offsets), and record the pre-randomization baseline into `initial`.
    object_shader_prims = {key: _resolve_object_shader_prims(spec["object"], NE)
                           for key, spec in object_specs.items()}
    # Opt-in: snapshot the asset's own value for every key the library can write, so a draw can
    # be applied ON TOP and a key the chosen material omits reverts to the asset instead of
    # keeping the previous episode's. Captured before any randomization has touched a prim --
    # later is too late.
    material_baselines = (_baseline_material_props(object_specs, object_shader_prims)
                          if (material_baseline_revert and object_specs) else None)
    if object_specs:
        initial["materials"] = _initial_material_params(object_specs, object_shader_prims)
        ep_mats = _sample_episode_materials(object_specs, ep_keys,
                                           indexed=indexed_seeding)
        print(f"[rerender] per-episode material randomization: {sorted(object_specs)} "
              f"(per-env, since each is an independent USD prim clone)")
    else:
        ep_mats = {}

    # per-env environment-sphere samples (experimental per-env background)
    ep_spheres = (_sample_episode_env_spheres(env_sphere_spec, ep_keys,
                                              env_sphere_seed) if env_sphere_spec else {})
    # Warm the texture cache from its plan (preload: auto -> this worker's own keys).
    _texcache.warm(flags, assets.hdri_dirs, episode_keys=ep_keys,
                   indexed=(indexed_seeding or "hash"), prob_salt=prob_salt)
    if env_sphere_spec:
        print(f"[rerender] per-episode env_sphere: emissive={bool(env_sphere_spec.get('emissive'))} "
              f"radius={env_sphere_spec.get('radius', 2.5)} (per-env background within a batch)")

    # --- GI <-> env-sphere coupling (flags['couple_gi_to_env_sphere']) ---
    # When ONLY the env_sphere defines a skybox pool (the `skybox` block has none) and this flag is
    # on, the per-batch dome GI texture is taken from the batch's FIRST env-sphere pick: one shared
    # queue, so GI == env0's backdrop, envs 1..N-1 keep that GI but show their own env-spheres, and
    # env-sphere picks (per episode) are what advance the queue that drives exhaust_before_repeat.
    # If BOTH skybox and env_sphere define pools they stay INDEPENDENT (each its own queue) and the
    # flag is IGNORED -- you've explicitly chosen separate GI vs backdrop sets.
    _env_has_pool = bool(ep_spheres) and any(v.get("texture") for v in ep_spheres.values())
    coupled_gi = (bool((flags or {}).get("couple_gi_to_env_sphere"))
                  and _env_has_pool and not bool(sky_buckets))
    if (flags or {}).get("couple_gi_to_env_sphere"):
        if coupled_gi:
            print("[rerender] couple_gi_to_env_sphere: dome GI texture = each batch's FIRST "
                  "env-sphere pick (GI == env0 backdrop; env-sphere queue drives selection)")
        elif bool(sky_buckets) and _env_has_pool:
            print("[rerender] couple_gi_to_env_sphere: IGNORED -> both skybox and env_sphere define "
                  "pools (independent queues)")

    # per-env local-light samples (in-batch lighting randomization)
    ep_local = (_sample_episode_local_lights(local_spec, ep_keys,
                                             local_rng_seed, indexed=indexed_seeding)
                if local_spec else {})
    if local_spec:
        def _cdesc(c):
            if isinstance(c, dict):
                return next((k for k in ("temperature", "hsv", "rgb") if k in c), "?")
            return str(c)
        if local_spec.get("lights"):
            desc = (f"{len(local_spec['lights'])} independent light(s): "
                    + ", ".join(_cdesc(l.get("color", "white")) for l in local_spec["lights"]))
        else:
            desc = (f"count={local_spec.get('count')} intensity={local_spec.get('intensity')} "
                    f"color={_cdesc(local_spec.get('color', 'white'))}")
        print(f"[rerender] per-episode local-light randomization: {desc} (per-env within a batch)")

    # Output video fps must match the SOURCE dataset (e.g. 25 or 100), not a constant.

    # Per-env block glyph swap. Flat bool rather than the whole flags dict, so ir keeps
    # holding resolved values rather than raw yaml (every other field here is resolved).
    _rtl = flags.get("randomize_task_letters")
    task_letters = bool(_rtl)
    # Resolved here, not at sample time: ir holds resolved values, and a bad spec must fail once
    # at startup rather than once per wave. None on the unconstrained path keeps assign_glyphs on
    # its historical without-replacement draw, so existing datasets still reproduce.
    glyph_sets = None
    if task_letters and not (_rtl is True):
        from . import glyphs as _g
        glyph_sets = _g.resolve_glyph_sets(_rtl.get("glyphs") if isinstance(_rtl, dict) else _rtl)

    # Per-env cuboid colour swap. Resolved here for the same reason as glyph_sets: a bad palette
    # must fail once at startup, not once per wave.
    _rbc = flags.get("randomize_block_colors")
    block_colors = bool(_rbc)
    block_palette = None
    if block_colors:
        from . import block_colors as _bc
        block_palette = _bc.parse_palette(_rbc)

    return SimpleNamespace(
        task_letters=task_letters,
        glyph_sets=glyph_sets,
        block_colors=block_colors,
        block_palette=block_palette,
        block_colors_applied={},
        block_colors_sampled={},
        glyphs_applied={},
        glyphs_sampled={},
        job=job,                      # the glyph library check reads its env_file
        episode_records=episode_records,
        prob_salt=prob_salt,
        all_cam_keys=all_cam_keys,
        cam_ranges=cam_ranges,
        rseed_base=rseed_base,
        rseed=rseed,
        randomize_cams=randomize_cams,
        rand_link=rand_link,
        rand_static=rand_static,
        assets=assets,
        indexed_seeding=indexed_seeding,
        ep_offsets=ep_offsets,
        override_cams=override_cams,
        li=li,
        light_range=light_range,
        light_rng=light_rng,
        sky=sky,
        sky_format=sky_format,
        sky_rot=sky_rot,
        sky_int=sky_int,
        sky_int_dist=sky_int_dist,
        _sky_vb_raw=_sky_vb_raw,
        sky_vb_prob=sky_vb_prob,
        sky_visible_bg=sky_visible_bg,
        sky_rng=sky_rng,
        sky_buckets=sky_buckets,
        sky_textures=sky_textures,
        sky_order=sky_order,
        missing=missing,
        scene_lights=scene_lights,
        fill_spec=fill_spec,
        fill_auto=fill_auto,
        fill_range=fill_range,
        fill_fixed=fill_fixed,
        fill_auto_ratio=fill_auto_ratio,
        fill_rng=fill_rng,
        desc=desc,
        local_spec=local_spec,
        local_prims=local_prims,
        local_rng_seed=local_rng_seed,
        sun_spec=sun_spec,
        sun_prim=sun_prim,
        sun_rng=sun_rng,
        exp_spec=exp_spec,
        exp_rng=exp_rng,
        env_sphere_spec=env_sphere_spec,
        env_sphere_prims=env_sphere_prims,
        env_sphere_seed=env_sphere_seed,
        scen_spec=scen_spec,
        _scen_mix=_scen_mix,
        scen_rng=scen_rng,
        _sseed=_sseed,
        _w=_w,
        scen_p=scen_p,
        object_specs=object_specs,
        material_baselines=material_baselines,
        initial_dome=initial_dome,
        initial_fill=initial_fill,
        initial=initial,
        n_obj_warn=n_obj_warn,
        device=device,
        NE=NE,
        object_shader_prims=object_shader_prims,
        ep_mats=ep_mats,
        ep_spheres=ep_spheres,
        _env_has_pool=_env_has_pool,
        coupled_gi=coupled_gi,
        ep_local=ep_local,
    )



def sample_episodes(ir, episode_keys):
    """Sample the four per-episode DR dicts for `episode_keys` and merge them into `ir`.

    setup() samples the whole episode list up front because it is driven by a dataset it
    already holds. A gym wrapper does not know its episodes in advance, so it calls this per
    reset with the new key(s). Both paths call the same four samplers with the same inputs --
    every one of which setup() already recorded on `ir` -- so a key sampled here gets the same
    value it would have got in a bulk call.

    Legal only under prefix-free seeding. Under `stream` an episode's draw depends on how many
    episodes preceded it, so sampling one key alone would silently return a different value
    than the bulk call; that is a wrong number, not a slower path, so it raises.
    """
    mode = getattr(ir, "indexed_seeding", False)
    if not mode:
        raise ValueError(
            "sample_episodes() needs prefix-free seeding (indexed_seeding='hash' or "
            "'indexed'); under 'stream' a per-episode draw depends on its prefix, so an "
            "incremental call would not match the bulk one."
        )
    keys = list(episode_keys)
    if not keys:
        return ir
    if ir.randomize_cams:
        ir.ep_offsets.update(_sample_episode_offsets(ir.cam_ranges, keys, ir.rseed, indexed=mode))
    if ir.object_specs:
        ir.ep_mats.update(_sample_episode_materials(ir.object_specs, keys, indexed=mode))
    if ir.env_sphere_spec:
        # NB: setup() calls a LOCAL closure of this name that injects assets.hdri_dirs; there is
        # no module-level binding, so call the real function and inject from ir. (This raised
        # NameError the first time a gym wrapper drove sample_episodes with an env_sphere block --
        # rerender never hits it because setup() samples in bulk through the closure.)
        ir.ep_spheres.update(_es._sample_episode_env_spheres(
            ir.env_sphere_spec, keys, ir.env_sphere_seed, ir.assets.hdri_dirs, indexed=mode))
    if ir.local_spec:
        ir.ep_local.update(_sample_episode_local_lights(ir.local_spec, keys, ir.local_rng_seed,
                                                        indexed=mode))
    return ir



def _block_specs(job):
    """{block_name: spec dict} parsed from the job's generated env.py, or {} when unavailable.

    The generated `_setup_scene_objects` declares `block_specs` as a literal, so this is a pure
    source read -- no import, no Isaac. Used to keep the glyph swap off asset libraries whose
    decal atlas this module does not describe.
    """
    import ast as _ast
    import os as _os

    env_file = (job or {}).get("env_file")
    if not env_file or not _os.path.exists(env_file):
        return {}
    try:
        src = open(env_file).read()
        for node in _ast.walk(_ast.parse(src)):
            if not (isinstance(node, _ast.Assign) and len(node.targets) == 1):
                continue
            tgt = node.targets[0]
            if getattr(tgt, "id", None) != "block_specs":
                continue
            specs = _ast.literal_eval(node.value)
            return {str(k): (v or {}) for k, v in specs.items()}
    except (OSError, SyntaxError, ValueError):
        return {}
    return {}


def _glyph_asset_paths(job):
    """{block_name: usd_path}. Thin wrapper so the glyph path is textually unchanged."""
    return {k: (v or {}).get("usd_path") for k, v in _block_specs(job).items()}


def sample_wave_task_letters(ir, wave, active=None):
    """PURE: {env_slot: {block: glyph}} for this wave. No USD, safe before the sim is touched.

    Sampling is split from application because the episode RECORD is written before the camera
    window where the shaders are set. Sampling twice would risk the recorded map and the rendered
    glyphs disagreeing -- the one inconsistency nothing downstream could detect. The result is
    cached on ``ir`` and reused by the applier.

    Block names come from each unit's own recorded ``objects``, not the scene: scenes differ in
    block count, and a scene-wide list would assign glyphs to blocks this episode does not have.
    """
    ir.glyphs_sampled = {}
    if not getattr(ir, "task_letters", False):
        return {}
    from . import glyphs as _glyphs

    active = list(range(len(wave))) if active is None else list(active)
    # Only blocks whose AUTHORED glyph is recoverable are assignable. `startswith("block_")`
    # also matched the colour cuboids that share mixed scenes (block_red1, block_cyan2,
    # block_green, and the legacy block_o1/o2 aliases): those have no glyph decal, so a name
    # picked here would be renamed in pddl_goal and could never be rendered -- a guaranteed
    # text/pixel mismatch on every mixed-scene dataset (aug14_2i_*, sep11_ai2_kp200_10blocks).
    assets = getattr(ir, "glyph_assets", None)
    if assets is None:
        assets = ir.glyph_assets = _glyph_asset_paths(getattr(ir, "job", None))
    # PER EPISODE, not unioned across the wave. The union made the draw SIZE depend on which
    # other episodes shared the wave, and rng.choice(35, size=n, replace=False) does not agree
    # with itself across n -- so episode 300 beside a 4-block episode drew different letters than
    # episode 300 alone, breaking the shard-invariance the episode-keyed rng exists to give. It
    # also handed glyphs to blocks the episode does not have, which this docstring already
    # forbade. Reproduced 2026-10-06: alone {block_a:4} vs in-wave {block_a:R}.
    names_by_slot = {}
    cand_all, names_all = set(), set()
    for i in active:
        cand_i = sorted({str(k) for k in (wave[i].get("objects") or {})
                         if _glyphs.authored_glyph(str(k)) is not None})
        keep = [n for n in cand_i if _glyphs.library_supported(assets.get(n))]
        names_by_slot[i] = keep
        cand_all.update(cand_i)
        names_all.update(keep)
    cand = sorted(cand_all)
    names = sorted(names_all)          # union, for the logging/guard lines below only
    dropped = [n for n in cand if n not in names_all]
    if dropped and not getattr(ir, "_logged_glyph_lib", False):
        ir._logged_glyph_lib = True
        print(f"[rerender] randomize_task_letters: SKIPPING {sorted(dropped)} -- their asset is "
              f"not from {_glyphs.SUPPORTED_LIBRARY} (e.g. {assets.get(dropped[0])!r}). This "
              f"module's atlas maths only describe that library; on any other the same UV shift "
              f"renders a DIFFERENT, real-looking glyph. Those blocks keep their letters and are "
              f"left out of the goal/task rewrite.")
    if not names:
        return {}
    if len(names) > len(_glyphs.GLYPHS):
        print(f"[rerender] randomize_task_letters: {len(names)} blocks > {len(_glyphs.GLYPHS)} "
              f"glyphs; sampling WITH replacement, goals may name a glyph twice")
    # ir.rseed, NOT ir.rseed_base: rseed = rseed_base + prob_salt, and _problem_seed_salt exists
    # precisely so "the same episode_index draws different samples across problems". Seeding on
    # the base made episode 0 of EVERY problem draw the same letters, so a 167-problem job
    # carried ~30 distinct assignments instead of ~5000 -- the augmentation was mostly duplicate
    # and nothing downstream would look wrong.
    seed = int(getattr(ir, "rseed", ir.rseed_base))
    ir.glyphs_sampled = _glyphs.sample_wave_glyphs(wave, active, names_by_slot, seed,
                                                   getattr(ir, "glyph_sets", None))
    return ir.glyphs_sampled


def apply_wave_task_letters(ir, wave, active=None):
    """PUBLIC: write this wave's sampled glyphs onto the scene; cache the applied map on `ir`.

    Callers must invoke this for EVERY render branch, not only the camera-DR one. It used to be
    reachable only from apply_wave_cameras, which rerender.py calls from a single `elif
    randomize_cams:` arm -- so a flags file carrying randomize_task_letters and no camera DR
    rewrote pddl_goal/task/glyph_map and rendered the ORIGINAL letters. Idempotent: the value
    written is a pure function of the glyph, so applying twice in one wave is a no-op rewrite.
    """
    active = list(range(len(wave))) if active is None else list(active)
    ir.glyphs_applied = _apply_wave_task_letters(ir, wave, active)
    return ir.glyphs_applied


def _apply_wave_task_letters(ir, wave, active):
    """Write the sampled glyphs onto the per-env decals shaders; returns what was APPLIED."""
    if not getattr(ir, "task_letters", False):
        return {}
    from . import glyphs as _glyphs

    asg = getattr(ir, "glyphs_sampled", None)
    if not asg:                                   # caller skipped sampling -- do it now
        asg = sample_wave_task_letters(ir, wave, active)
    if not asg:
        return {}
    applied = _glyphs.apply_wave_glyphs(asg)
    # The episode record and the rewritten pddl_goal/task come from the SAMPLED map, so any block
    # that failed to apply leaves the dataset claiming a glyph it never rendered. Surface it here
    # too, keyed by env, so the wave log names the slots and not just a count.
    gap = _glyphs.reconcile(asg, applied)
    if gap:
        print(f"[rerender] randomize_task_letters: text/pixel MISMATCH in {len(gap)} env(s): {gap}")
    return applied

def sample_wave_block_colors(ir, wave, active=None):
    """PUBLIC: draw this wave's cuboid colours; cache on `ir`. Pure -- touches no prims.

    Block names come from each unit's own recorded ``objects`` intersected with the generated
    env's ``block_specs``, so ownership is the spec ``type`` and never a name test: ``block_b`` is
    a letter block and ``block_blue`` is a cuboid, and any prefix rule gets that wrong in the
    direction that still renders.
    """
    ir.block_colors_sampled = {}
    if not getattr(ir, "block_colors", False):
        return {}
    from . import block_colors as _bc

    active = list(range(len(wave))) if active is None else list(active)
    specs = getattr(ir, "block_specs_cache", None)
    if specs is None:
        specs = ir.block_specs_cache = _block_specs(getattr(ir, "job", None))
    cuboids = set(_bc.cuboid_names(specs))
    # PER EPISODE, never a wave-wide union -- same rule as the glyph draw above, and for the same
    # reason: a union assigns colours to blocks the episode does not have.
    names_by_slot = {i: sorted({str(k) for k in (wave[i].get("objects") or {})} & cuboids)
                     for i in active}
    if not any(names_by_slot.values()):
        return {}
    # Hard-fail on drift: a block may not be owned by both randomizers.
    from . import glyphs as _g
    both = {n for v in names_by_slot.values() for n in v if _g.authored_glyph(n) is not None}
    if both:
        raise AssertionError(f"block(s) {sorted(both)} claimed by BOTH randomize_task_letters and "
                             f"randomize_block_colors; the spec-type partition has drifted")
    # ir.rseed (includes prob_salt), not rseed_base -- see sample_wave_task_letters.
    seed = int(getattr(ir, "rseed", ir.rseed_base))
    ir.block_colors_sampled = _bc.sample_wave_block_colors(
        wave, active, names_by_slot, seed, getattr(ir, "block_palette", None))
    return ir.block_colors_sampled


def apply_wave_block_colors(ir, wave, active=None):
    """PUBLIC: write this wave's sampled colours onto the scene; cache the APPLIED map on `ir`.

    Called from apply_wave, which runs once per wave BEFORE the frame loop branches -- so this is
    reached by all three render arms and by DRSession. The glyph applier's original bug was living
    inside apply_wave_cameras, reachable only from the `elif randomize_cams:` arm; do not
    reintroduce that shape by adding a second, branch-local call site.
    """
    if not getattr(ir, "block_colors", False):
        return {}
    from . import block_colors as _bc

    sampled = getattr(ir, "block_colors_sampled", None) or sample_wave_block_colors(ir, wave, active)
    if not sampled:
        ir.block_colors_applied = {}
        return {}
    ir.block_colors_applied = _bc.apply_wave_block_colors(
        sampled, getattr(ir, "block_palette", None))
    return ir.block_colors_applied


def apply_wave_cameras(ir, sim, wave, active, *, dt, camera_base="scene",
                       cam_base_pose=None, cam_base_K=None, rtx_warmup_frames=1):
    """Seat this wave's cameras: pre-render, optional re-base onto the recorded pose, per-episode
    offsets + FOV, then RTX warm-up. Moved verbatim out of rerender's frame loop at t==0.

    apply_wave() deliberately does NOT touch cameras, so this is the other half of "randomize this
    batch" and a caller must run it too. It owns two ordering constraints that are easy to get
    wrong and silent when you do:
      * one render()+update(dt) BEFORE the offsets, so cam.data is populated and the link-mounted
        cameras are seated on their real poses rather than the default (the "zoomed-out" first
        frame);
      * the RTX warm-up AFTER them, so temporal history is built against the final framing.
    Both live in here precisely so a gym wrapper cannot omit one.

    `camera_base="data"` re-bases the static cameras onto the pose recorded in the dataset instead
    of the scene cfg.offset; it needs cam_base_pose, and a robot articulation to express it
    against. Without a dataset a caller passes camera_base="scene" and omits both bases.
    """
    if not ir.randomize_cams:
        return None
    sim.sim.render()
    sim.scene.update(dt)

    data_bases = None
    if camera_base == "data" and ir.rand_static:
        arts = getattr(sim.scene, "articulations", None) or {}
        rob = next(iter(arts.values()), None)
        try:
            root = rob.data.root_state_w[:, :7] if rob is not None else None
        except Exception:                                     # noqa: BLE001
            root = None
        if root is None:
            # once per run, not once per wave: ir carries the latch so this module keeps no
            # mutable state of its own (see the package docstring).
            if not getattr(ir, "_warned_no_robot", False):
                ir._warned_no_robot = True
                print("[rerender] WARNING --camera_base data: no robot articulation "
                      "to express the recorded pose against; keeping cfg.offset")
        else:
            data_bases = _recorded_static_bases(ir.rand_static, wave, active, cam_base_pose,
                                                root, ir.device)
            if data_bases and not getattr(ir, "_logged_cam_base", False):
                import torch
                ir._logged_cam_base = True
                for ck, (P, _Q) in data_bases.items():
                    i = next(iter(P))
                    cfg = ir.randomize_cams[ck].cfg.offset
                    d = float(torch.linalg.norm(
                        P[i] - torch.tensor(list(cfg.pos), device=ir.device)))
                    print(f"[rerender] camera_base=data: {ck} re-based onto the "
                          f"recorded pose ({d * 1000:.2f} mm from cfg.offset)")

    _apply_camera_local_offsets(ir.randomize_cams, wave, active, ir.ep_offsets, ir.device,
                                data_bases=data_bases)
    _apply_episode_camera_fov(ir.randomize_cams, wave, active, ir.ep_offsets, ir.device,
                              base_K=cam_base_K, camera_base=camera_base)
    # Same pre-render window as the camera DR: the RTX warm-up below must see the final
    # framing AND the final glyphs, or the first frames carry the previous wave's letters.
    apply_wave_task_letters(ir, wave, active)
    for _ in range(max(1, rtx_warmup_frames)):
        sim.sim.render()
        sim.scene.update(dt)
    return data_bases

def apply_wave(ir, sim, wave, active=None):
    """Apply ONE wave of the yaml-interpreted DR to the live scene (per-batch dome/skybox/fill/
    exposure/sun + per-env materials/lights/spheres) and return this wave's applied values. Moved
    verbatim from replay_job's wave loop; `ir` is the namespace returned by setup(). Mutates ir's
    per-set queue cursors (sky_buckets) as intended."""
    # `active` is the list of ENV SLOTS this call should touch. Default is every slot,
    # which is the literal the three per-env appliers used to be handed inline -- so a
    # caller that omits it (rerender.py) is unchanged. A gym wrapper resetting a subset
    # passes just those slots; all three appliers already iterate `for i in active`.
    active = list(range(len(wave))) if active is None else list(active)
    # Cuboid colours: here and nowhere else. apply_wave runs once per wave before the frame loop
    # branches, so this reaches every render arm and the DRSession path too.
    if getattr(ir, "block_colors", False):
        sample_wave_block_colors(ir, wave, active)
        apply_wave_block_colors(ir, wave, active)
    scen_rng = ir.scen_rng
    scen_p = ir.scen_p
    _scen_mix = ir._scen_mix
    light_range = ir.light_range
    li = ir.li
    light_rng = ir.light_rng
    sky_buckets = ir.sky_buckets
    coupled_gi = ir.coupled_gi
    sky_rng = ir.sky_rng
    sky_int = ir.sky_int
    sky_int_dist = ir.sky_int_dist
    sky_rot = ir.sky_rot
    sky_format = ir.sky_format
    sky_vb_prob = ir.sky_vb_prob
    sky_visible_bg = ir.sky_visible_bg
    ep_spheres = ir.ep_spheres
    initial_dome = ir.initial_dome
    fill_spec = ir.fill_spec
    fill_auto_ratio = ir.fill_auto_ratio
    fill_rng = ir.fill_rng
    sun_prim = ir.sun_prim
    sun_spec = ir.sun_spec
    sun_rng = ir.sun_rng
    exp_spec = ir.exp_spec
    exp_rng = ir.exp_rng
    object_specs = ir.object_specs
    object_shader_prims = ir.object_shader_prims
    ep_mats = ir.ep_mats
    local_prims = ir.local_prims
    local_spec = ir.local_spec
    ep_local = ir.ep_local
    env_sphere_prims = ir.env_sphere_prims
    env_sphere_spec = ir.env_sphere_spec
    wave_scen = _scen_mix[int(scen_rng.choice(len(_scen_mix), p=scen_p))] if _scen_mix else None
    # per-batch lighting: scalar = fixed; [min,max] = sampled once per wave (shared
    # by all episodes in the batch, since the dome light is global).
    if light_range is not None:
        wave_light = float(light_rng.uniform(light_range[0], light_range[1]))
        _dome_intensity(wave_light)
    elif isinstance(li, (int, float)):
        wave_light = float(li)
    else:
        wave_light = None

    # per-batch skybox: cycle through a shuffled pool (each HDRI used once before any
    # repeats). Pick texture + Z rotation (+ intensity) once per wave (dome is global).
    # Applied AFTER lighting so its intensity wins when both are set.
    wave_sky = None
    if sky_buckets or coupled_gi:
        def _samp(v):
            return (float(sky_rng.uniform(v[0], v[1])) if isinstance(v, (list, tuple))
                    else (float(v) if v is not None else None))
        def _samp_int(v, dist=sky_int_dist):
            # Dome intensity supports intensity_dist: log -> sample a WIDE [dim,bright] range
            # with even PERCEPTUAL spacing (log). Uniform sampling of e.g. [40,1100] spends
            # most draws in the bright half; log gives far more dark/high-contrast batches.
            if not isinstance(v, (list, tuple)):
                return float(v) if v is not None else None
            lo, hi = float(v[0]), float(v[1])
            if dist == "log" and lo > 0 and hi > 0:
                return float(np.exp(sky_rng.uniform(np.log(lo), np.log(hi))))
            return float(sky_rng.uniform(lo, hi))
        # a scenario bucket may override this wave's dome/HDRI intensity (+ its own dist)
        _eff_sky_int, _eff_sky_dist = sky_int, sky_int_dist
        if wave_scen is not None and "skybox_intensity" in wave_scen:
            _eff_sky_int = wave_scen["skybox_intensity"]
            _eff_sky_dist = str(wave_scen.get("skybox_intensity_dist", "uniform")).lower()
        if coupled_gi:
            # dome GI texture = this batch's FIRST env-sphere pick (shared queue); env0's
            # backdrop == the GI, envs 1..N-1 keep this GI but show their own env-spheres.
            _first = wave[0]["episode_index"]
            _fsph = ep_spheres.get(_first) or {}
            tex, _sky_bucket = _fsph.get("texture"), _fsph.get("bucket", "env_sphere")
        else:
            # pick the bucket (weighted; NO rng draw for a single bucket) then advance ITS queue
            tex, _sky_bucket = sky_buckets.next(sky_rng)
        tex = str(tex) if tex else None
        rot = _samp(sky_rot) if sky_rot is not None else 0.0
        inten = _samp_int(_eff_sky_int, _eff_sky_dist)
        wave_vb = (bool(sky_rng.random() < sky_vb_prob) if sky_vb_prob is not None
                   else sky_visible_bg)
        if tex:
            _apply_skybox(tex, sky_format, rot, inten, visible_background=wave_vb)
            wave_sky = {"texture": os.path.basename(tex), "texture_format": sky_format,
                        "rotation_deg": rot, "visible_background": wave_vb}
            if coupled_gi:
                wave_sky["gi_coupled"] = True
                wave_sky["bucket"] = _sky_bucket
            elif sky_buckets.multi:
                wave_sky["bucket"] = _sky_bucket
            if inten is not None:
                wave_sky["intensity"] = inten

    # per-batch scene fill light. Applied AFTER dome+skybox so `auto` can read the dome
    # intensity actually in effect this wave (HDRI intensity if a skybox was set, else the
    # sampled/fixed dome light, else the pristine default). Leaves the fill untouched when
    # scene_lights.fill_intensity is omitted.
    # effective dome intensity in effect this wave (skybox HDRI > sampled dome light > default).
    # Used by fill_auto AND by any local light with `intensity_auto_ratio` (key<->dome coupling
    # so the key casts shadows proportional to the dome -> shadows survive a bright skybox).
    eff_dome = (wave_sky.get("intensity") if (wave_sky and wave_sky.get("intensity") is not None)
                else (wave_light if wave_light is not None else initial_dome))
    # effective fill spec: a scenario bucket may override it (number | [min,max] | "auto");
    # otherwise use the base scene_lights.fill_intensity modes parsed above.
    _fspec = wave_scen["fill_intensity"] if (wave_scen is not None
                                             and "fill_intensity" in wave_scen) else fill_spec
    wave_fill = None
    if isinstance(_fspec, str) and _fspec.lower() == "auto":
        if eff_dome is not None:
            wave_fill = fill_auto_ratio * float(eff_dome)
    elif isinstance(_fspec, (list, tuple)) and len(_fspec) == 2:
        wave_fill = float(fill_rng.uniform(_fspec[0], _fspec[1]))
    elif isinstance(_fspec, (int, float)):
        wave_fill = float(_fspec)
    if wave_fill is not None:
        _sphere_fill_intensity(wave_fill)

    # per-batch SHARED sun (global DistantLight): one sampled direction for the whole wave, so
    # every env in a num_envs>1 batch gets the SAME clean directional shadow (no N-sun wash).
    wave_sun = _apply_global_sun(sun_prim, _sample_light(sun_spec, sun_rng)) if sun_prim else None

    # per-batch RTX exposure (global): sample iso/fstop/shutter this wave -> real low-key<->
    # high-key brightness dial on top of the lighting. iso_dist: log for even perceptual span.
    wave_exp = None
    if exp_spec is not None:
        def _samp_exp(v, logd):
            if not isinstance(v, (list, tuple)):
                return float(v) if v is not None else None
            lo, hi = float(v[0]), float(v[1])
            if logd and lo > 0 and hi > 0:
                return float(np.exp(exp_rng.uniform(np.log(lo), np.log(hi))))
            return float(exp_rng.uniform(lo, hi))
        wave_exp = {}
        for key in ("iso", "fstop", "shutter"):
            # a scenario bucket may override this wave's exposure (+ its own _dist)
            _eval, _elog = exp_spec.get(key), exp_spec.get(f"{key}_dist", "") == "log"
            if wave_scen is not None and key in wave_scen:
                _eval = wave_scen[key]
                _elog = str(wave_scen.get(f"{key}_dist", "")).lower() == "log"
            val = _samp_exp(_eval, _elog)
            if val is not None:
                wave_exp[key] = val
        _apply_exposure(auto=exp_spec.get("auto"), **wave_exp)

    # per-episode materials: each active env gets ITS OWN sampled material applied to
    # ITS OWN shader prim (independent per-env USD clones -> true per-episode variation
    # within the batch, not per-batch like the skybox above).
    wave_materials = (_apply_episode_materials(object_specs, object_shader_prims, wave,
                                                active, ep_mats,
                                                getattr(ir, "material_baselines", None))
                      if object_specs else {})

    # per-env local lights: each active env's own lights set from its episode sample ->
    # genuine per-env (in-batch) lighting variation despite the shared dome.
    # a scenario bucket may disable the local sun for this wave (sun: false) -> soft/open look
    _sun_on = bool(wave_scen.get("sun", True)) if wave_scen is not None else True
    wave_local = (_apply_local_lights(local_prims, wave, active, ep_local,
                                      sim.scene.env_origins.cpu().numpy(), eff_dome=eff_dome,
                                      enabled=_sun_on)
                  if local_spec else {})

    # per-env environment sphere (experimental per-env background)
    wave_spheres = (_apply_env_spheres(env_sphere_prims, env_sphere_spec, wave,
                                       active, ep_spheres)
                    if env_sphere_spec else {})
    return SimpleNamespace(wave_light=wave_light, wave_sky=wave_sky, wave_fill=wave_fill, eff_dome=eff_dome, wave_sun=wave_sun, wave_exp=wave_exp, wave_scen=wave_scen, wave_materials=wave_materials, wave_local=wave_local, wave_spheres=wave_spheres)


# ---------------------------------------------------------------------------
# render_params RE-RENDER path: build the scene (prims) from flags, then apply a specific episode's
# RECORDED values (from render_params.json) via the same _apply_* tools -- no yaml sampling, no rng.
# ---------------------------------------------------------------------------
from . import resolve as _res


def setup_prims(sim, flags, num_envs, *, assets: AssetPaths = None):
    """ENVIRONMENT CREATION, split from sampling: create the DR prims (env-spheres, per-env local
    lights, shared sun, per-object shader prims) from `flags` only -- no rng, no per-episode values.
    Also applies build-time carb/render-mode + exposure-auto. Returns a namespace of prim handles +
    specs that apply_record() drives with explicit per-episode values."""
    assets = assets or AssetPaths.default()
    flags = flags or {}
    sky = flags.get("skybox") or {}
    scene_lights = flags.get("scene_lights") or {}
    local_spec = scene_lights.get("local_lights")
    sun_spec = scene_lights.get("sun")
    env_sphere_spec = flags.get("env_sphere")
    exp_spec = flags.get("exposure")
    _set_render_mode(flags.get("render_mode"))
    _apply_carb_settings(flags.get("carb_settings"))
    if exp_spec is not None and exp_spec.get("auto") is not None:
        _apply_exposure(auto=bool(exp_spec["auto"]))
    local_prims = _setup_local_lights(sim, local_spec) if local_spec else []
    sun_prim = _setup_global_sun(sun_spec) if sun_spec else None
    env_sphere_prims = _setup_env_spheres(sim, env_sphere_spec) if env_sphere_spec else []
    object_specs = _mat._resolve_material_specs(flags, assets.table_materials)
    object_shader_prims = {k: _resolve_object_shader_prims(s["object"], num_envs)
                           for k, s in object_specs.items()}
    return SimpleNamespace(assets=assets, sky=sky, local_spec=local_spec, sun_spec=sun_spec,
                           env_sphere_spec=env_sphere_spec, exp_spec=exp_spec, local_prims=local_prims,
                           sun_prim=sun_prim, env_sphere_prims=env_sphere_prims,
                           object_specs=object_specs, object_shader_prims=object_shader_prims)


def apply_record(prims, sim, record, *, active=(0,)):
    """Apply ONE episode's RECORDED render_params values (record = a render_params episodes[] dict) to
    the live scene using the same tools the sampler uses -- the DR just PICKS the stored values and
    passes them through. `prims` = setup_prims() namespace. Recorded->apply conversions: HDRI
    basenames -> full paths; materials {name,...props} -> {material_name, props}; env_sphere scale ->
    emissive_intensity. Camera (recorded absolute) is handled by the caller. num_envs=1 per episode
    guarantees each episode reproduces its exact per-batch globals."""
    hdri = prims.assets.hdri_dirs
    key = record.get("episode_index", 0)
    act = list(active)
    # `active` holds ENV indices, and the appliers below index wave[i] for i in act, so the wave must
    # be long enough to be addressed by them. One record is applied to one env, so every slot carries
    # this record's key. active=(0,) reproduces the original single-element wave exactly; higher
    # indices are what a multi-env (batched) caller passes.
    wave = [{"episode_index": key} for _ in range(max(act, default=0) + 1)]
    eff_dome = record.get("lighting_intensity")
    rec_sky = record.get("skybox")
    if rec_sky:
        tex = _res.resolve_basename(rec_sky.get("texture"), hdri)
        inten = rec_sky.get("intensity")
        if inten is not None:
            _dome_intensity(float(inten)); eff_dome = float(inten)
        elif eff_dome is not None:
            _dome_intensity(float(eff_dome))
        if tex:
            _apply_skybox(tex, rec_sky.get("texture_format", "latlong"),
                          float(rec_sky.get("rotation_deg", 0.0)), inten,
                          visible_background=bool(rec_sky.get("visible_background", False)))
    elif eff_dome is not None:
        _dome_intensity(float(eff_dome))
    if record.get("fill_intensity") is not None:
        _sphere_fill_intensity(float(record["fill_intensity"]))
    if record.get("sun") is not None and prims.sun_prim is not None:
        _apply_global_sun(prims.sun_prim, record["sun"])
    rec_exp = record.get("exposure")
    if rec_exp:
        _apply_exposure(auto=(prims.exp_spec or {}).get("auto"),
                        **{k: rec_exp[k] for k in ("iso", "fstop", "shutter") if k in rec_exp})
    if record.get("materials") and prims.object_specs:
        ep_mats = {key: {lab: {"material_name": v.get("name"),
                               "props": {k: vv for k, vv in v.items() if k != "name"}}
                         for lab, v in record["materials"].items()}}
        _apply_episode_materials(prims.object_specs, prims.object_shader_prims, wave, act, ep_mats)
    if record.get("env_sphere") and prims.env_sphere_spec:
        es = record["env_sphere"]
        ep_data = {key: {"texture": _res.resolve_basename(es.get("texture"), hdri),
                         "emissive_intensity": es.get("scale", 1.0),
                         "rotation": es.get("rotation_deg", 0.0)}}
        _apply_env_spheres(prims.env_sphere_prims, prims.env_sphere_spec, wave, act, ep_data)
    if prims.local_spec:
        # Apply UNCONDITIONALLY. An episode with no local lights must ZERO the slots, not leave the
        # previous episode's burning: apply_record replays against one persistent stage. Both the
        # empty-list case (scenario sun:false -> lights.py records []) and the absent-key case are
        # falsy, and the old `record.get(...) and ...` guard skipped both, so those episodes
        # replayed with a stale per-env light the record says is off. The empty list already
        # zeroes every slot in _apply_local_lights.
        _apply_local_lights(prims.local_prims, wave, act,
                            {key: record.get("local_lights") or []},
                            sim.scene.env_origins.cpu().numpy(), eff_dome=eff_dome, enabled=True)
    return record
