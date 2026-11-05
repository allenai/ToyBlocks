"""Render specific episodes from a recorded dataset's render_params.json -- NO yaml.

Reads the per-episode DR values recorded in render_params.json and re-applies them EXACTLY (the
domain randomizer just PICKS the stored values and passes them to the tools -- no sampling, no rng),
replays each episode's recorded trajectory + recorded camera pose, and writes videos in the LeRobot
dataset structure. Faithful for EVERY run (incl. camera-randomized ones): the scene DR comes from the
recorded render_params values (via domain_randomization.interpreter.apply_record) and the camera
framing from the recorded per-frame pose/intrinsics in the source parquet.

    python render_from_params.py <recorded_dataset_dir> --output <out> \
        --envs_dir <envs> [--episodes 3,7,12] --headless --enable_cameras

<recorded_dataset_dir> is a LeRobot dataset with BOTH meta/render_params.json (flags + recorded
per-episode DR values) and data/ (the trajectories to replay). Reuses:
  * domain_randomization.interpreter.setup_prims  -- environment creation from flags (no rng)
  * domain_randomization.interpreter.apply_record -- pick recorded {x,..} values -> apply tools
  * render.{build_simulator, load_episodes, _resolve_joint_ids, _set_table_height,
    load_env_and_task, _resolve_env_file} + the same purender replay loop.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

# ---- args parsed BEFORE the Isaac app launch (json read is pure Python) -------------------------
parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("source", type=str, help="recorded dataset dir (meta/render_params.json + data/)")
parser.add_argument("--output", type=str, required=True, help="output dataset dir")
parser.add_argument("--episodes", type=str, default=None,
                    help="comma-separated src_episode_index list to render (default: all recorded)")
parser.add_argument("--envs_dir", type=str, default=None, help="per-problem env packages dir")
parser.add_argument("--env_file", type=str, default=None, help="explicit env.py (else discovered)")
parser.add_argument("--rtx_warmup_frames", type=int, default=16,
                    help="extra RTX passes at each episode's frame 0 (denoiser/DLSS convergence)")
parser.add_argument("--depth_codec", type=str, default="hevc", choices=["av1", "hevc"])
parser.add_argument("--depth_format", type=str, default="12bit", choices=["rg8", "12bit"],
                    help="MAIN depth storage: 'rg8' = RG8-packed uint12 in a lossless RGB frame, "
                         "'12bit' = native gray12le via --depth_codec. Same meaning as render.py.")
parser.add_argument("--depth_gop", type=int, default=1,
                    help="GOP for the MAIN depth video (see render.py). Higher = smaller and "
                         "faster to encode, worse to seek. The --extra_depth stream ignores this and uses its VARIANT's GOP: rg8_224 stays at 1 (every frame intra), g12_224 uses the reference's short per-camera GOP (8 static / 4 gripper).")
parser.add_argument("--extra_depth", type=str, default=None, choices=["rg8_224", "g12_224"],
                    help="also write a second depth video per camera: rg8_224 (lossless rgb byte-split, GOP 1) or g12_224 (lossless gray12le HEVC, short per-camera GOP; ~0.2x the bytes and far cheaper cold reads). See render.py")
parser.add_argument("--extra_rgb", type=str, default=None, choices=["224"],
                    help="also write videos/observation.images.<cam>_224/ (see render.py)")
parser.add_argument("--assets_paths", type=str, nargs="*", default=None,
                    help="Ordered extra HDRI/skybox asset roots (first match wins, repo default last).")
parser.add_argument("--verify_cam_pose", action="store_true",
                    help="After applying the recorded camera offsets, read the LIVE camera pose and "
                         "intrinsics back out of the sim and print them against the recorded target. "
                         "Comparing against the output render_params is circular (both derive from "
                         "the same offsets); this is what actually proves the camera ended up where "
                         "the record says it was.")
parser.add_argument("--reproduce_placement", action="store_true",
                    help="Rebuild the ORIGINAL batch placement from the record: allocate the "
                         "recorded num_envs and render each episode in its recorded env_slot. "
                         "Scene CONTENT (skybox/exposure/materials/lights) is already replayed "
                         "exactly; placement is the remaining variable, because env_origins[slot] "
                         "and the presence of neighbouring envs change GI / dome IBL sampling. "
                         "Without this, a num_envs=1 replay reproduces the content but NOT the "
                         "batch context, so it can render brighter/cleaner than the original.")
parser.add_argument("--num_envs", type=int, default=1,
                    help="Episodes rendered per wave. NOTE: skybox/GI/exposure are GLOBAL to the "
                         "stage, so a wave >1 shares whatever the first slot set -- which is "
                         "exactly what the batch render does. Use >1 to reproduce batch-path "
                         "behaviour; 1 gives exact per-episode globals. The scene always has at "
                         "least one env per block-size slot, so each episode renders at its "
                         "recorded block size.")
parser.add_argument("--antialiasing", type=str, default=None,
                    choices=["Off", "FXAA", "TAA", "DLSS", "DLAA"],
                    help="sim.render.antialiasing_mode. DLAA = DLSS at NATIVE resolution (no "
                         "upscale), which keeps ray reconstruction but avoids its below-minimum "
                         "input regime at small output sizes.")
parser.add_argument("--render_scale", type=float, default=None,
                    help="Multiply every camera's width/height before scene creation, to lift "
                         "DLSS's internal render resolution above its minimum input.")
parser.add_argument("--skip_cameras", type=str, nargs="*", default=None,
                    help="Camera names NOT to spawn (same semantics as render.py): dropped from "
                         "the scene preset before creation, so their render cost is never paid.")

Placed = List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]   # (env, src episode, episode, record)


def plan_waves(pending: List[Tuple[int, Dict[str, Any], Dict[str, Any]]], env_file: str,
               num_envs: int, reproduce_placement: bool, T: Any) -> Tuple[List[Placed], int]:
    """Waves to render and the scene size, with every episode in an env of its recorded size.

    render spawns env ``i`` at size slot ``i % K`` (K = the env package's size table) and renders
    an episode only in an env of the slot matching its recorded size: render_params ``block_size``,
    else the source's ``observation.state.cube_scale``. Replaying it in any other env keeps the
    recorded centres at the wrong block size (floating / sunken stacks).

    * ``reproduce_placement``: the recorded waves (``wave``) at the recorded ``env_slot``s, at the
      recorded ``num_envs``. Records written before ``wave`` existed cannot be placed exactly and
      are refused rather than silently re-packed.
    * otherwise: ``num_envs`` <= 1 renders one episode per wave (exact per-episode globals) in the
      first env of its slot; larger ``num_envs`` packs episodes by slot as render does. The scene
      gets at least one env per slot.

    Returns ``(waves, scene_num_envs)``; each wave lists ``(env, src_episode, episode, record)``.
    Raises SystemExit when an episode cannot be placed at its size.
    """
    sw = T._scene_swap
    table = sw.env_size_table(env_file)
    sized = bool(table) and any(
        r.get("block_size") is not None or abs(float(ep.get("cube_scale", 1.0)) - 1.0) > 1e-9
        for _s, ep, r in pending)

    def slot_of(sidx: int, ep: Dict[str, Any], rec: Dict[str, Any]) -> int:
        size = rec.get("block_size")
        size = float(size) if size is not None else float(ep.get("cube_scale", 1.0))
        slot = sw.match_size_slots(table, [size])[0]
        if slot is None:
            raise SystemExit(f"src episode {sidx}: recorded block size {size:.5f} m is not in "
                             f"{env_file}'s size table {[round(x, 5) for x in table]}")
        return int(slot)

    if not sized:
        print("[from_params] WARNING no block-size table / recorded sizes: blocks use the scene's "
              "default sizes, which need not match the recording")

    if reproduce_placement:
        missing = [s for s, _e, r in pending if "wave" not in r or "env_slot" not in r]
        if missing:
            raise SystemExit(f"--reproduce_placement: {len(missing)} record(s) (e.g. src episode "
                             f"{missing[0]}) carry no 'wave'/'env_slot': written by a render "
                             f"that predates per-episode wave recording, so the original batches "
                             f"cannot be rebuilt exactly")
        ne = max(int(r.get("num_envs", 0) or 0) for _s, _e, r in pending) or num_envs
        env_map = sw.round_robin_slots(ne, len(table)) if sized else None
        by_wave: Dict[int, Placed] = {}
        for sidx, ep, rec in pending:
            env = int(rec["env_slot"])
            if env >= ne:
                raise SystemExit(f"src episode {sidx}: recorded env_slot {env} >= num_envs {ne}")
            if env_map is not None and env_map[env] != slot_of(sidx, ep, rec):
                raise SystemExit(f"src episode {sidx}: env {env} spawns size slot {env_map[env]}, "
                                 f"its recorded size is slot {slot_of(sidx, ep, rec)} (a render "
                                 f"that did not use the round-robin slot map)")
            by_wave.setdefault(int(rec["wave"]), []).append((env, sidx, ep, rec))
        for w, placed in by_wave.items():
            envs = [env for env, *_ in placed]
            if len(set(envs)) != len(envs):
                raise SystemExit(f"recorded wave {w} puts two episodes in one env: {sorted(envs)}")
        return [sorted(by_wave[w], key=lambda x: x[0]) for w in sorted(by_wave)], ne

    if not sized:
        ne = max(1, int(num_envs))
        return ([[(k, *pending[i + k]) for k in range(min(ne, len(pending) - i))]
                 for i in range(0, len(pending), ne)], ne)
    ne = max(int(num_envs), len(table))
    env_map = sw.round_robin_slots(ne, len(table))
    slots = [slot_of(sidx, ep, rec) for sidx, ep, rec in pending]
    if num_envs <= 1:
        return [[(env_map.index(s), *pending[k])] for k, s in enumerate(slots)], ne
    units = [{"episode_index": k, "length": int(ep["length"]), "_spawn_slot": s}
             for k, ((_sidx, ep, _rec), s) in enumerate(zip(pending, slots))]
    waves = T.pack_units_by_spawn_slot(units, env_map, 0)
    return [[(env, *pending[int(u["episode_index"])]) for env, u in enumerate(w)
             if not u.get("_placeholder")] for w in waves], ne


from isaaclab.app import AppLauncher  # noqa: E402

if __name__ == "__main__":
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True
    # Defaults for the trim build/render helpers we reuse (they read these off the CLI namespace).
    # env_spacing/enable_gi left unset so the RECORDED flags decide (faithful render).
    # Every flag the reused render/build helpers read off the CLI namespace must be present here,
    # even when this script does not expose it: they do a bare args_cli.<name> and a missing entry
    # is an AttributeError at build time, not a graceful default. skip_cameras was added to
    # render.py without being mirrored here, which broke this script entirely.
    # Every flag the reused render/build helpers read off the CLI namespace must be present here,
    # even when this script does not expose it: they do a bare args_cli.<name> and a missing entry
    # is an AttributeError at build time, not a graceful default. Anything that IS a real parser
    # argument above (e.g. skip_cameras) must stay OUT of this dict or it would clobber the parsed
    # value with the placeholder.
    for _n, _v in dict(env_spacing=None, disable_dlss=False, enable_gi=False,
                       skip_initial_step=False, show_gizmos=False, physics_step=False,
                       copy_source_data=False).items():
        setattr(args, _n, _v)

    # --- locate render_params.json (accept a dataset dir or a render output root) ---
    src_dir = Path(args.source)
    rp_path = src_dir / "meta" / "render_params.json"
    if not rp_path.exists():
        cand = sorted(src_dir.glob("**/meta/render_params.json"))
        if not cand:
            raise SystemExit(f"no render_params.json under {src_dir}")
        rp_path = cand[0]
        src_dir = rp_path.parent.parent
    render_params = json.load(open(rp_path))
    flags = render_params.get("flags") or {}

    # kit_args from the recorded carb_settings (launch-time RTX settings), same as render.
    _cs = flags.get("carb_settings")
    if isinstance(_cs, dict) and _cs:
        def _kv(k, v):
            v = ("true" if v else "false") if isinstance(v, bool) else v
            return f"--/{str(k).lstrip('/')}={v}"
        args.kit_args = ((getattr(args, "kit_args", "") or "") + " "
                         + " ".join(_kv(k, v) for k, v in _cs.items())).strip()
        print(f"[from_params] kit_args: {args.kit_args}")

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    # Isaac-dependent imports AFTER launch, then reuse trim's machinery by injecting them into it
    # (trim reads these off its module scope, populated only inside its own __main__).
    import torch
    import numpy as np
    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg
    from toyblocks.domain_randomization.render_sim import PHYS_STEP_PER_RENDER_INTERVAL, RenderSim
    from toyblocks.vecgen.recording.lerobot_recorder import IsaacDataRecorderLeRobot

    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "rr_main", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "render.py"))
    T = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(T)
    T.args_cli = args
    # torch too: render imports it inside its own __main__, so a helper we reuse here
    # (_set_table_height, which builds a torch tensor) hits NameError without it injected.
    # numpy is already a module-level import in render, so it needs no injection.
    T.torch = torch
    T.gym = gym
    T.parse_env_cfg = parse_env_cfg
    T.RenderSim = RenderSim
    T.IsaacDataRecorderLeRobot = IsaacDataRecorderLeRobot
    T.simulation_app = simulation_app

    from toyblocks.domain_randomization import interpreter as DR
    from toyblocks.domain_randomization import render_params as RP
    from toyblocks.domain_randomization import camera as DRCAM
    from toyblocks.domain_randomization import resolve as DRRES
    from toyblocks.domain_randomization.assets import AssetPaths
    _assets = AssetPaths.with_search_paths(args.assets_paths)
    print(f"[from_params] asset map: {DRRES.asset_source_summary(_assets.hdri_dirs)}")

    # --- resolve env + task, build the sim with the RECORDED flags (env_spacing/GI/hide/etc.) ---
    cands = sorted(Path(args.envs_dir).glob("**/env.py")) if args.envs_dir else []
    env_file = args.env_file or T._resolve_env_file(cands, render_params.get("env_file"))
    if not env_file:
        raise SystemExit("could not resolve an env.py (pass --env_file or --envs_dir)")
    task_id = T.load_env_and_task(str(env_file))

    # --- which episodes to render (by src_episode_index) ---
    # Selected BEFORE the sim is built: --reproduce_placement reads the recorded num_envs off these
    # records to size the scene.
    want = None
    if args.episodes:
        want = {int(x) for x in args.episodes.split(",") if x.strip() != ""}
    recs = [e for e in render_params.get("episodes", [])
            if want is None or int(e.get("src_episode_index", e["episode_index"])) in want]
    if not recs:
        raise SystemExit(f"no matching episodes in {rp_path} (--episodes={args.episodes})")
    print(f"[from_params] rendering {len(recs)} episode(s) from {rp_path}")

    # --- load the trajectories to replay from the SOURCE dataset (pure Python, before the sim) ---
    src_eps = T.load_episodes(src_dir, default_success=None, max_episodes=None)
    by_src = {int(e["episode_index"]): e for e in src_eps}
    pending = []
    for rec in recs:
        sidx = int(rec.get("src_episode_index", rec["episode_index"]))
        ep = by_src.get(sidx)
        if ep is None:
            print(f"[from_params] WARNING src episode {sidx} not in source dataset; skipping")
            continue
        pending.append((sidx, ep, rec))
    waves_plan, args.num_envs = plan_waves(pending, str(env_file), args.num_envs,
                                           args.reproduce_placement, T)
    sim = T.build_simulator(task_id, num_envs=args.num_envs, device="cuda:0", flags=flags)
    device = sim.device
    print(f"[from_params] sim ready: num_envs={sim.num_envs}")

    # --- environment creation from flags (prims only, no rng) ---
    prims = DR.setup_prims(sim, flags, sim.num_envs, assets=_assets)
    art, arm_ids, drive_id, mimic_ids, mimic_signs = T._resolve_joint_ids(sim)
    dt = sim.sim.get_physics_dt() * PHYS_STEP_PER_RENDER_INTERVAL
    src_fps = int(render_params.get("fps", T.FPS))

    # Camera pose comes from TWO different places and mixing them up silently ruins the replay.
    #
    #   * render_params episodes[i]["cameras"] -- the ABSOLUTE pose/FOV the camera was actually
    #     RENDERED at, i.e. base (x) the randomization that run sampled. This is the only correct
    #     source for a camera that was randomized.
    #   * the SOURCE parquet's per-frame cam_pose -- the pose the ORIGINAL execution recorded,
    #     before any render-time randomization.
    #
    # This used to drive every camera from the source parquet while claiming it reproduced "any
    # camera randomization the source run applied". It did not: replaying an allrand dataset put
    # the cameras back at their un-randomized poses. Measured on aug17 rotate_cw, that was a 121 mm
    # mean depth error on the objects/arm, 1.48 m on the table, and a backdrop silhouette
    # overlapping only 72%. It also wrote pose via Fabric set_world_poses, which per
    # _apply_camera_local_offsets does not reliably reach the renderer at all.
    #
    # So: randomized cameras are driven through the SAME path the original randomization used
    # (recovered offset -> USD local mount transform + FOV from a pristine base_K), and only
    # cameras with no recorded entry keep the source-pose path -- for those the source pose IS the
    # rendered pose, so it is correct.
    flag_cam_keys = {f"camera:{n}" for n in (flags.get("cameras") or {})}
    _ini_cams = ((render_params.get("initial") or {}).get("cameras") or {})
    ep_cam_off = DRCAM.offsets_from_absolute(_ini_cams, recs, index_key="src_episode_index")
    # The recorded ABSOLUTE target per (episode, camera), kept for --verify_cam_pose.
    ep_cam_abs = {int(e.get("src_episode_index", e.get("episode_index"))): (e.get("cameras") or {})
                  for e in recs}
    # A camera counts as RANDOMIZED only if it actually moved. render.py records
    # episodes[].cameras for a deterministic `cameras:` block too (so the dataset says what was
    # applied, not what was asked for), and offsets_from_absolute then recovers an all-ZERO
    # offset for those. Classifying them as randomized would replay them as cfg.offset (x)
    # identity -- correct only when the render itself was based on cfg.offset, i.e.
    # camera_base=scene. Under the default camera_base=data the static cams were re-based onto
    # the DATASET's recorded extrinsics, so cfg.offset is not where they were rendered and the
    # source-pose path (override_cams) is the right one.
    _cam_base = ((render_params.get("initial") or {}).get("camera_base") or "data")

    def _moved(off: dict) -> bool:
        if _cam_base == "scene":
            return True  # cfg.offset IS the render base, so a zero delta replays exactly
        q = off.get("quat_delta") or (1.0, 0.0, 0.0, 0.0)
        return (any(abs(float(x)) > 1e-9 for x in (off.get("position_offset_cm") or ()))
                or abs(1.0 - abs(float(q[0]))) > 1e-9
                or abs(float(off.get("vertical_fov_deg") or 0.0)) > 1e-9)

    # Union over episodes: a camera that happens to sample zero on ONE episode is still randomized.
    dr_cam_keys = {k for off in ep_cam_off.values()
                   for k, v in off.items() if _moved(v)} - flag_cam_keys
    dr_cams = {k: sim.scene.sensors[k] for k in sorted(dr_cam_keys) if k in sim.scene.keys()}
    override_cams = {k: sim.scene.sensors[k] for k, s in sim.scene.sensors.items()
                    if k.startswith("camera:") and hasattr(s, "set_world_poses")
                    and k not in flag_cam_keys and k not in dr_cams
                    and any(k in e["cam_pose"] for e in src_eps)}
    cam_base_K = {}
    if dr_cams:
        _, cam_base_K = DRCAM._capture_camera_bases(sim, dr_cams, {}, dt)
        print(f"[from_params] replaying RANDOMIZED camera pose/FOV from render_params: "
              f"{sorted(dr_cams)}")
    if override_cams:
        print(f"[from_params] driving un-randomized cameras from source pose: {sorted(override_cams)}")

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    recorder = IsaacDataRecorderLeRobot(sim.scene, data_dir=out_dir, dataset_name="blocks",
                                        num_envs=args.num_envs, record_every_n_steps=1, fps=src_fps, robotiq=True,
                                        depth_codec=("rg8" if args.depth_format == "rg8"
                                                     else args.depth_codec),
                                        depth_gop=args.depth_gop,
                                        extra_depth=args.extra_depth,
                                        extra_rgb=args.extra_rgb,
                                        # One dataset per output dir, same as render.py: this
                                        # script mkdirs out_dir on the line above, and sharding
                                        # would put the data in a sibling __sNNN that nothing
                                        # reading --output ever looks at.
                                        enable_sharding=False)
    sim.data_recorder = recorder
    sim.execution = True

    split_out_count = {True: 0, False: 0}   # per-split output-episode counter (aligns with videos)
    out_records = []
    # Wave loop mirroring render.py: the episodes of a wave share one stage, so anything GLOBAL
    # (skybox, GI, exposure/tonemap) is whatever the wave collectively set, while per-env prims
    # (env_sphere, local lights, table material) stay per-slot. plan_waves puts every episode in an
    # env spawned at its recorded block size.
    from toyblocks.domain_randomization.replay import park_inactive_objects
    for wi_plan, placed in enumerate(waves_plan):
        chunk = [(sidx, ep, rec) for _env, sidx, ep, rec in placed]
        slots = [env for env, _sidx, _ep, _rec in placed]
        wave = [None] * (max(slots, default=0) + 1)
        for k, (_s, ep, _r) in enumerate(chunk):
            wave[slots[k]] = ep
        for k, (sidx, ep, rec) in enumerate(chunk):
            slot = slots[k]
            recorder.reset_env_state(slot)
            thd = ep.get("table_height_delta", 0.0)
            if abs(thd) > 1e-9:
                T._set_table_height(sim, [slot], [thd])
            DR.apply_record(prims, sim, rec, active=(slot,))   # recorded values -> tools, per env
        park_inactive_objects(sim, [w if w is not None else {} for w in wave])
        print(f"[from_params] wave {wi_plan + 1}/{len(waves_plan)}: src episodes "
              f"{[s for s, _e, _r in chunk]} across {len(chunk)} env(s)")
        max_t = max(e["length"] for e in wave if e is not None)
        for t in range(max_t):
            active = [i for i in range(len(wave))
                      if wave[i] is not None and t < wave[i]["length"]]
            if not active:
                break
            eids = torch.tensor(active, device=device)
            arm = torch.tensor(np.stack([wave[i]["joint"][t] for i in active]),
                               device=device, dtype=torch.float32)
            art.write_joint_state_to_sim(arm, torch.zeros_like(arm), joint_ids=arm_ids, env_ids=eids)
            grip = torch.tensor([wave[i]["gripper"][t] for i in active],
                                device=device, dtype=torch.float32).reshape(-1, 1)
            art.write_joint_state_to_sim(grip, torch.zeros_like(grip), joint_ids=drive_id, env_ids=eids)
            for jid, sign in zip(mimic_ids, mimic_signs):
                mj = grip * sign
                art.write_joint_state_to_sim(mj, torch.zeros_like(mj), joint_ids=[jid], env_ids=eids)
            for i in active:
                for name, poses in wave[i]["objects"].items():
                    if name in sim.scene.keys():
                        sim.set_rigid_object_pose(
                            name, torch.tensor(poses[t], device=device, dtype=torch.float32),
                            env_idx=i, local=True)
                    elif name not in _WARNED_MISSING_OBJECTS:
                        # Silently skipping leaves the object at its SPAWN pose for the whole
                        # replay while every other surface matches -- a large local depth error
                        # that looks like a rendering problem rather than a missing write.
                        _WARNED_MISSING_OBJECTS.add(name)
                        print(f"[from_params] WARNING recorded object {name!r} is not in the "
                              f"scene ({sorted(sim.scene.keys())}); its pose is NOT replayed and "
                              f"it stays where the env spawned it")
            if override_cams:
                DRCAM._override_cameras(sim, override_cams, wave, active, t, device)
            if t == 0:
                # Camera offsets are applied HERE, not in the wave setup, to mirror render.py
                # exactly: render + scene.update FIRST, then write the local mount offsets, then
                # the RTX warmup passes. scene.update() re-authors camera transforms, so an offset
                # written before it gets partly clobbered -- that ordering alone left a ~1 px
                # residual against the original render even once the offsets themselves were right.
                sim.sim.render()
                sim.scene.update(dt)
                if dr_cams:
                    DRCAM._apply_camera_local_offsets(dr_cams, wave, active, ep_cam_off, device)
                    DRCAM._apply_episode_camera_fov(dr_cams, wave, active, ep_cam_off, device,
                                                    base_K=cam_base_K)
                    if args.verify_cam_pose:
                        sim.sim.render(); sim.scene.update(dt)
                        for _ck in sorted(dr_cams):
                            _cam = dr_cams[_ck]
                            _oc = getattr(getattr(_cam, "cfg", None), "offset", None)
                            _vw = getattr(_cam, "_view", None)
                            # cfg.offset is the base _apply_camera_local_offsets composes ONTO.
                            # initial.cameras is the base the recorded offset was derived FROM.
                            # If these disagree, every replayed pose inherits the difference.
                            _ini = (_ini_cams or {}).get(_ck, {})
                            if _oc is not None:
                                print(f"[verify] {_ck} cfg.offset pos={[round(float(x),6) for x in _oc.pos]} "
                                      f"rot={[round(float(x),6) for x in _oc.rot]} conv={getattr(_oc,'convention',None)}")
                            if _ini:
                                print(f"[verify] {_ck} initial   pos={[round(float(x),6) for x in _ini.get('position',[])]} "
                                      f"rot={[round(float(x),6) for x in _ini.get('rotation',[])]}")
                                if _oc is not None:
                                    _d=[(float(_oc.pos[j])-float(_ini['position'][j]))*1000 for j in range(3)]
                                    print(f"[verify] {_ck} cfg.offset-initial pos_mm="
                                          f"[{_d[0]:+.3f},{_d[1]:+.3f},{_d[2]:+.3f}]")
                            # THE oracle: the USD local transform the renderer actually consumes.
                            if _vw is not None:
                                try:
                                    _lp, _lq = _vw.get_local_poses()
                                    for _i in active:
                                        print(f"[verify] {_ck} slot{_i} USD local pos="
                                              f"{[round(float(x),6) for x in _lp[_i].tolist()]} "
                                              f"quat={[round(float(x),6) for x in _lq[_i].tolist()]}")
                                except Exception as _e:
                                    print(f"[verify] {_ck} get_local_poses failed: {type(_e).__name__}: {_e}")
                            for _i in active:
                                _want = (ep_cam_abs.get(int(wave[_i]["episode_index"])) or {}).get(_ck)
                                if _want is None:
                                    continue
                                _off = (ep_cam_off.get(int(wave[_i]["episode_index"])) or {}).get(_ck, {})
                                _K = _cam.data.intrinsic_matrices[_i]
                                _fy, _cy = float(_K[1, 1]), float(_K[1, 2])
                                _vf = math.degrees(2 * math.atan(_cy / _fy)) if _fy > 0 else float("nan")
                                print(f"[verify] {_ck} slot{_i} applied off_cm="
                                      f"{[round(x,4) for x in _off.get('position_offset_cm',[])]} "
                                      f"dq={[round(x,6) for x in _off.get('quat_delta',[])]}")
                                print(f"[verify] {_ck} slot{_i} vfov live={_vf:.5f} want="
                                      f"{float(_want.get('vertical_fov_deg', 0)):.5f}")
                for _ in range(max(1, args.rtx_warmup_frames)):
                    sim.sim.render()
                    sim.scene.update(dt)
            sim.sim.render()
            sim.scene.update(dt)
            recorder.record_step_batched(active)
        for k, (sidx, ep, rec) in enumerate(chunk):
            slot = slots[k]
            if rec.get("split") in ("blocks_success", "blocks_failure"):
                succ = (rec["split"] == "blocks_success")
            else:
                succ = bool(ep["success"])
            recorder.write_success(succ, env_idx=slot)
            recorder.save_episode(slot)
            oidx = split_out_count[succ]; split_out_count[succ] += 1
            out_rec = dict(rec)
            out_rec["episode_index"] = oidx
            out_rec["split"] = "blocks_success" if succ else "blocks_failure"
            out_rec["src_episode_index"] = sidx
            out_records.append(out_rec)
            print(f"[from_params] rendered src episode {sidx} -> "
                  f"{out_rec['split']}/file-{oidx} ({ep['length']} frames)")

    print(f"[from_params] finalizing recorder for {out_dir} ...")
    recorder.finalize()
    sim.data_recorder = None
    job = {"sources": [(str(src_dir), s) for s in sorted(split_out_count) if split_out_count[s]],
           "env_file": str(env_file)}
    RP.write_render_params(out_dir, job, flags, src_fps, False, render_params.get("initial", {}),
                           out_records, {"from_render_params": str(rp_path), "output": out_dir,
                                         "episodes": args.episodes},
                           completed=True, total_episodes=len(out_records),
                           script="scripts/render/render_from_params.py")
    print(f"[from_params] done -> {out_dir}")
    simulation_app.close()
