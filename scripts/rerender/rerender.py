# Re-render recorded trajectories with different settings (vectorized).
#
# Loads recorded LeRobot datasets, replays each episode's recorded states by
# teleporting the robot + objects frame-by-frame (no physics stepping), renders,
# and re-records via the current vectorized data recorder — producing videos that
# reproduce the originals (modulo driver/scene-version pixel differences).
#
# Supports:
#   * "unmerged" layout:  datasets_dir/problem_*/blocks_{success,failure}/{data,meta,videos}
#   * "flat" merged layout: datasets_dir/{data,meta,videos}
# Output mirrors the input layout under --output.
#
# A YAML --flags file can override camera extrinsics + FOV (per camera) and
# lighting (see apply_flags / apply_camera_flag_overrides).
#
# Usage:
#   python rerender.py datasets/jun9_flip_2_3b \
#       --env_file datasets/jun9_flip_2_3b/env/env.py \
#       --output  datasets/rerendered/jun9_flip_2_3b \
#       --max_episodes 20 --num_envs 8 [--flags flags.yaml]

import argparse
import importlib
import importlib.util
import glob
import json
import math
import fcntl
import os
import shutil
import re
import sys
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __name__ == "__main__":  # forkserver/spawn workers re-run this module -> must NOT start forkserver / launch Isaac
    # --- Start the video-encoder forkserver BEFORE Isaac loads --------------------------------
    # The LeRobot video encoder runs in a subprocess. Under the old plain-fork default it forked
    # from THIS process AFTER Isaac was up, copy-on-write inheriting Isaac's (num_envs-scaled) ~26GB
    # footprint -> OOM at high num_envs. spawn is unusable (Isaac launches at module import below, so
    # a spawned child would re-launch it). So we start a `forkserver` now, while this process is still
    # light, preloading the isaaclab-free encoder module; the encoder later forks from that clean
    # ~2-3GB server (torch+lerobot+av, no Isaac). MUST run before importing isaaclab.
    import multiprocessing as _mp  # noqa: E402
    _ENC_CTX = _mp.get_context("forkserver")
    _ENC_CTX.set_forkserver_preload(["toyblocks.vecgen.recording._video_encoder"])
    _warm = _ENC_CTX.Process(target=int)  # trivial -> forces the (clean, pre-Isaac) server to start now
    _warm.start(); _warm.join()

    from isaaclab.app import AppLauncher  # noqa: E402

parser = argparse.ArgumentParser(
    description="Re-render recorded LeRobot trajectories with the vectorized recorder.",
)
parser.add_argument("datasets_dir", type=str, help="Root containing the recorded dataset(s).")
parser.add_argument("--env_file", type=str, default=None,
                    help="Path to the generated env package's env.py (gym.register BaseEnv). "
                         "For UNMERGED datasets this is ignored and the env is auto-matched per "
                         "problem from <datasets_dir>/envs/<problem_name>/env.py. Used as a "
                         "fallback (and for merged/flat datasets, which also fall back to "
                         "<datasets_dir>/env/env.py if not given).")
parser.add_argument("--output", type=str, required=True,
                    help="Output directory; mirrors the input dataset structure.")
parser.add_argument("--envs_dir", type=str, default=None,
                    help="Optional root dir holding per-problem env packages as "
                         "<envs_dir>/<problem_name>/env.py. Use when envs live separately from "
                         "the data (e.g. S3 basics/<exp>/envs/ vs unmerged/<exp>/). Takes "
                         "precedence over <datasets_dir>/envs/<problem_name>/ and --env_file.")
parser.add_argument("--flags", type=str, default=None,
                    help="YAML file with camera/lighting overrides (see docstring).")
parser.add_argument("--skip_cameras", type=str, nargs="*", default=None,
                    help="Camera names NOT to spawn, e.g. --skip_cameras left_camera. Dropped from "
                         "the scene preset BEFORE creation, so both their rgb AND depth streams are "
                         "absent and their render cost is never paid (~1/3 of the tiled render for "
                         "one of three cameras). Names may be given with or without the 'camera:' "
                         "prefix.")
parser.add_argument("--skip_failure_episodes", action="store_true",
                    help="Render only blocks_success, skipping blocks_failure entirely. The merge "
                         "keeps only blocks_success anyway (merge_success.get_dataset_paths), so "
                         "for a merge-bound run the failure split is pure cost -- 45%% of episodes "
                         "on the reorient source. Off by default: failure episodes are still real "
                         "data. NOTE this renumbers the global episode index, so a run started "
                         "with it cannot resume an output built without it (the plan fingerprint "
                         "will differ, which is exactly the check that catches it).")
parser.add_argument("--assets_paths", type=str, nargs="*", default=None,
                    help="Ordered EXTRA asset search roots for HDRIs/skyboxes. Each may be an "
                         "assets-root (has an environments/ subdir) or an environments-root directly "
                         "(auto-detected). Searched IN ORDER (first match wins per file), then the "
                         "repo default (assets/environments) LAST. Lets a run pull skyboxes from a "
                         "fast/local or shared store (e.g. /tmp/assets/environments) so the Docker "
                         "image can ship without the 46GB skybox pool.")
parser.add_argument("--max_episodes", type=int, default=None,
                    help="Max episodes re-rendered per source dataset (default: all).")
parser.add_argument("--scene_swap_warmup", type=int, default=0,
                    help="Extra RTX passes after a scene swap, for DLSS/denoiser convergence. "
                         "0 relies on the passes the wave-0 t==0 path already does.")
parser.add_argument("--scene_swap", default=True, action=argparse.BooleanOptionalAction,
                    help="At each problem boundary, make the live Isaac scene match that problem's "
                         "block set (stop, delete/spawn block prims, play). Costs NOTHING when "
                         "consecutive problems share a set -- the common case, and the whole of a "
                         "one-problem-per-worker run. --no-scene_swap restores the old behaviour, "
                         "where a mismatched problem is skipped instead.")
parser.add_argument("--fail_on_video_defects", action="store_true",
                    help="Mark a problem INCOMPLETE (completed=false, full re-render) if any video "
                         "could not be published. Default off: the problem stays complete with "
                         "status=incomplete_videos and the exact missing list in render_params, "
                         "which the orchestrator patches at the end of the run.")
parser.add_argument("--episodes_file", type=str, default=None,
                    help="Render ONLY the listed episodes (for targeted re-render / patching a subset). "
                         "A TSV/CSV with a header containing columns 'problem','split','episode' (extra "
                         "columns ignored), OR the full video-key/path list (one per line; problem = the "
                         "problem_* path component, split = blocks_success/blocks_failure, episode = the "
                         "file-NNN index). Only episodes present for a problem+split are rendered; other "
                         "problems produce no output. Output episodes are renumbered per split but each "
                         "records its src_episode_index for remapping back to the original dataset.")
parser.add_argument("--episode_offset", type=int, default=None,
                    help="PARALLEL FRAGMENT selection (flat episode-level). Render the GLOBAL episode "
                         "slice [episode_offset, episode_offset+max_episodes) across all problems/splits "
                         "in canonical order (sorted problems x [success,failure] x episode 0..n-1). "
                         "Each parallel worker renders a disjoint contiguous slice (same --num_envs, "
                         "--N). Slices are problem-aligned, so workers can share one --output. Here "
                         "--max_episodes is the fragment size M (M source episodes -> M*N render-jobs). "
                         "Overrides --episodes_file.")
parser.add_argument("--split_groups", action="store_true",
                    help="Render each member problem of a vecgen problem_grp_* dir as its own unit: the "
                         "group's source and env, only that problem's episodes (by the data's "
                         "source_problem), written to <output>/<member problem>/. A group of 10 "
                         "problems becomes 10 one-wave items instead of one ten-wave item. Members "
                         "take per-fragment DR (no prefix: the prefix would be the whole group). Off by "
                         "default.")
parser.add_argument("--fragment_invariant", default=True, action=argparse.BooleanOptionalAction,
                    help="LEVEL-2 parallel invariance (default ON). In a fragment (--episode_offset "
                         "or --episodes_file), keep the FULL per-problem episode list so render-uids, "
                         "wave grouping, per-episode DR sampling AND the per-wave dome/skybox rng "
                         "draws are IDENTICAL to a single-machine run; only this fragment's episodes "
                         "are rendered/recorded. => de-fragmented+merged output is byte-identical to "
                         "one big run. --no-fragment_invariant reverts to per-fragment DR (Level 1: "
                         "smaller working set, but a fragment's DR differs from a single run).")
parser.add_argument("-N", "--renders_per_episode", type=int, default=1,
                    help="Renders generated per source episode (default 1). N>1 replays each "
                         "episode N times, each with freshly sampled IN-BATCH randomization "
                         "(per-env materials/local-lights/cameras + per-wave dome), producing Nx "
                         "the output episodes for visual data augmentation. The recorded "
                         "state/action trajectory is identical across the N copies (only the "
                         "rendered pixels differ); each output episode records src_episode_index "
                         "+ iter in render_params. NOTE: multiplies dataset SIZE by ~N.")
parser.add_argument("--num_envs", type=int, default=8,
                    help="Parallel envs / replay batch size (multiple of 8, or <8).")
parser.add_argument("--num_writers", type=int, default=1,
                    help="Encoder processes per render process (envs are split across them, each a "
                         "full encode pool). Raise it when the encoder, not the GPU, is the limit.")
parser.add_argument("--wave_sort_block", type=int, default=10,
                    help="Group episodes of similar length into the same wave, within blocks of this many "
                         "waves of the render order (0 = keep source order). A wave renders every env "
                         "until its LONGEST episode ends, so source order wasted ~36%% of rendered "
                         "env-frames on oct21/oct23 (10-wave blocks: 64%% -> 88%% useful). Fixed "
                         "blocks keep a fragment's waves identical to the full run's: a fragment "
                         "loads its prefix through the end of its last block. Changes which "
                         "episodes share a wave (so per-wave draws and env slots), not the "
                         "per-episode DR, which is keyed by render uid.")
parser.add_argument("--skip_initial_step", action="store_true",
                    help="Skip the simulator's construction-time physics settle "
                         "(RenderSim step(10)). Replay overrides all state per "
                         "frame, so this is safe and removes the only physics stepping that "
                         "remains in render-only mode. (PhysX is still used for forward kinematics "
                         "and init; full removal needs an external renderer.)")
parser.add_argument("--physics_step", action="store_true",
                    help="Use one physics step per frame (pins joints via PD targets) instead of "
                         "render-only. Stepping advances the sim timestamp so the time-gated tiled "
                         "cameras always refresh (eliminates occasional duplicate frames), at the cost "
                         "of ~0.01s/frame physics drift. Default OFF = pure teleport + render, no physics.")
parser.add_argument("--disable_dlss", action="store_true",
                    help="Shortcut for --antialiasing FXAA (disable DLSS). DLSS is ON by default. "
                         "DLSS/TAA/DLAA accumulate a temporal history that is empty on frame 0, "
                         "which can leave warm-up artifacts (e.g. edge lines) in the first frame(s); "
                         "FXAA/Off have no temporal history. Ignored if --antialiasing is given.")
parser.add_argument("--antialiasing", type=str, default=None,
                    choices=["Off", "FXAA", "TAA", "DLSS", "DLAA"],
                    help="Anti-aliasing mode (sim.render.antialiasing_mode). Default: keep the "
                         "preset (DLSS). Off/FXAA are single-frame (no temporal warm-up artifact); "
                         "TAA/DLSS/DLAA are temporal. Takes precedence over --disable_dlss.")
parser.add_argument("--rtx_warmup_frames", type=int, default=1,
                    help="Number of extra render()+update() passes at t==0 of each batch before "
                         "the first captured frame (default 1). Lets temporal renderers (DLSS + the "
                         "RTX denoiser) converge so frame 0 has no warm-up artifacts; also seats "
                         "link-mounted cameras on their initial pose. Try 8-16 if frame-0 artifacts "
                         "persist. Cost: N extra renders per batch, once (not per frame).")
parser.add_argument("--depth_format", type=str, default="12bit", choices=["rg8", "12bit"],
                    help="How the MAIN depth video is stored (default 12bit). 'rg8' = the SAME "
                         "uint12 value packed into a lossless 8-bit RGB frame (R=low byte, G=high "
                         "byte, B=0; depth = R + (G<<8)), every frame a keyframe -- no precision "
                         "is lost and it decodes far faster via torchcodec random access, but the "
                         "files are ~3x larger. '12bit' = native gray12le using --depth_codec, "
                         "encoded at GOP 1000 for size. info.json records which "
                         "(depth_encoding rg8_mm vs gray12le_mm). Was named '8bit', which was "
                         "misleading: RG8 reduces nothing.")
parser.add_argument("--depth_gop", type=int, default=1,
                    help="GOP (keyframe interval) for the MAIN depth video. 1 = every frame an "
                         "intra keyframe (random-access friendly, largest). Higher is both "
                         "SMALLER and FASTER to encode -- measured on 245 real 640x360 gray12 "
                         "frames, GOP 1000 gave 1210 KiB in 1.57s vs GOP 2's 4974 KiB in 4.90s -- "
                         "at the cost of seeking, since a reader decodes from the last keyframe. "
                         "The --extra_depth stream ignores this and uses its VARIANT's GOP: rg8_224 stays at 1 (every frame intra), g12_224 uses the reference's short per-camera GOP (8 / 4 gripper).")
parser.add_argument("--extra_rgb", type=str, default=None, choices=["224"],
                    help="Additionally write a SECOND RGB video per camera, into "
                         "videos/observation.images.<cam>_224/ -- the layout the HVLA training "
                         "sidecars use (no codec prefix: rg8 is a DEPTH packing). Each frame is "
                         "the SAME uint8 the main stream gets, resized with Pillow BILINEAR and "
                         "encoded libsvtav1 crf 26 preset 8 yuv420p, matching "
                         "maniflow-tb/scripts/rgb_conversion/reencode_to_224_batch.py. "
                         "ffmpeg 'scale' is NOT equivalent -- it disagrees with Pillow on ~70%% "
                         "of the 224x224 pixels (max abs 35), so a sidecar built that way is not "
                         "the uint8 the model reads.")
parser.add_argument("--rgb_GOP", type=int, default=2,
                    help="keyframe interval for the MAIN RGB streams (default 2, which is what "
                         "every dataset so far carries -- it was a hardcoded constant until now). "
                         "0 means 'let SVT-AV1 choose'. Depth has had --depth_gop all along; GOP "
                         "is the same size/seek lever for RGB.")
parser.add_argument("--extra_rgb_GOP", type=int, default=0,
                    help="keyframe interval for the --extra_rgb sidecars. DEFAULT 0 = pass no "
                         "-g and let SVT-AV1 choose, which is what the reference sidecars carry "
                         "(measured 161 on PARTIAL5k) -- so the default reproduces them. Set a "
                         "value to force one; smaller costs size (g=2 measured 1,781,024 B "
                         "against 306,410 B with no -g on a 622-frame clip).")
parser.add_argument("--extra_depth", type=str, default=None,
                    choices=["rg8_224", "g12_224", "rg8_55_224"],
                    help="Additionally write a SECOND depth video per camera, into "
                         "videos/observation.depth.<cam>_<variant>/. 'rg8_224' = the RG8 packing "
                         "downscaled to 224x224 (nearest-neighbour, aspect NOT preserved), "
                         "lossless qp=0 at GOP 1 -- small and random-access-friendly for training. "
                         "Matches the layout/encode of the reference dataset "
                         "aug07_color_pnp_rerendering_textrand_only_updated_N1. Independent of "
                         "--depth_format: the main depth is unaffected.")
parser.add_argument("--render_scale", type=float, default=None,
                    help="Scale the sim2real preset cameras' render resolution, keeping their "
                         "field of view (e.g. 2.0: front 640x360 -> 1280x720). Default off. The "
                         "recorder writes one resolution for all cameras and robot-mounted cameras "
                         "are not scaled, so this requires --skip_cameras gripper_camera "
                         "left_camera.")
parser.add_argument("--skip_depth", action="store_true",
                    help="Do not render or record the main depth stream for any camera (no "
                         "observation.depth.* features or videos). Default off. Cannot be combined "
                         "with --extra_depth, which is derived from the main depth.")
parser.add_argument("--depth_codec", type=str, default="hevc", choices=["av1", "hevc"],
                    help="Depth video codec (only matters if the scene records depth). Both encode "
                         "depth as lossless 12-bit millimetres in gray12le (0-4095mm = 0-4.095m, "
                         "decode: uint16 mm / 1000 = metres). 'hevc' (libx265) is ~15x faster to "
                         "encode; 'av1' (libaom) produces smaller files. Default hevc.")
parser.add_argument("--env_spacing", type=float, default=None,
                    help="Override the InteractiveScene env_spacing (default 5.0). Larger spacing "
                         "moves neighbor envs out of a camera's peripheral view (mitigates a "
                         "neighbor-env's backdrop leaking through the left camera's gap when a "
                         "skybox is used).")
parser.add_argument("--dump_scene", action="store_true",
                    help="Debug: after building the scene, print the stage layout (env origins, "
                         "and key prims' scope + world position: backdrop walls, window_background, "
                         "dome light, cameras) to diagnose per-env vs global geometry, then exit.")
parser.add_argument("--show_gizmos", action="store_true",
                    help="Keep debug visualization markers in the render (the RGB frame-axis "
                         "gizmos, corner spheres, block/goal arrows, ee-frame — all under /Visuals). "
                         "Default: hidden (clean videos). These are per-env markers created for "
                         "datagen and otherwise leak into some camera views.")
parser.add_argument("--enable_gi", action="store_true",
                    help="Enable RTX global illumination + reflections (RenderCfg."
                         "enable_global_illumination / enable_reflections, dome IBL sampling). "
                         "Auto-enabled when flags.yaml has a `skybox` block. Off by default "
                         "(the 'balanced' preset has indirect diffuse off).")
parser.add_argument("--resume", action="store_true",
                    help="Skip problems whose output already has a COMPLETED "
                         "render_params.json (meta.completed=true). Use to continue a run that "
                         "was interrupted; finished problems are not re-rendered.")
parser.add_argument("--patch_actions_from_source", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Copy the SOURCE episode's action.* columns into the re-rendered parquet "
                         "(default ON). A render-only replay never writes joint targets or a "
                         "gripper command, so the recorder would otherwise emit a constant "
                         "action.joint_target and an all-zero action.gripper_position.")
parser.add_argument("--material_baseline_revert", action="store_true",
                    help="Before each episode's material draw, revert every key the library "
                         "can write to the ASSET's own value. Without it a key the chosen "
                         "material omits keeps the PREVIOUS episode's value, so materials "
                         "bleed across episodes. Off by default: it changes rendered output.")
parser.add_argument("--dr_seeding", default="hash", choices=["stream", "indexed", "hash"],
                    help="How per-episode DR draws are seeded. NEVER mix two modes within one "
                         "dataset -- each produces different values -- so to reproduce or PATCH an "
                         "existing dataset pass the mode recorded in its meta/render_params.json "
                         "(args.dr_seeding). "
                         "'stream' is the legacy loop-order path: ONE rng advanced over the episode "
                         "list, so an episode's values depend on how many episodes precede it, "
                         "which is why a fragment has to replay the whole prefix. "
                         "'indexed' derives each episode's rng from (seed, episode_index), so a "
                         "shard reproduces an episode with no prefix -- what lets a patch jump "
                         "straight to a missing episode. "
                         "'hash' (DEFAULT) is indexed with the identity HASHED rather than summed "
                         "and truncated: the per-problem salt is added into a 32-bit seed and each "
                         "concern offsets it by a small constant (+777, +999, +4242, ...), so "
                         "neighbouring problems and concerns enter the mixer at neighbouring "
                         "points. Hashing (seed, concern, episode_index) gives each its own 128 "
                         "bits with full avalanche, and keeps the prefix-free property.")
parser.add_argument("--passthrough_source_columns", default=True,
                    action=argparse.BooleanOptionalAction,
                    help="Copy EVERY column (values + arrow dtype) from the matching SOURCE episode "
                         "into the re-rendered parquet, so the two datasets differ only where the "
                         "re-render is authoritative: the randomized camera pose/intrinsics. "
                         "Supersedes --patch_actions_from_source, which only copied action.*. "
                         "Default ON. Without it the re-render emits zeros for joint "
                         "velocity/acceleration/effort and the contact wrench, constants for the PD "
                         "gains and scene scalars, drifted FK link poses, re-derived label strings, "
                         "and drops 4 source columns entirely.")
parser.add_argument("--copy_source_data", action="store_true",
                    help="After each problem, replace the recorder's re-derived data/ with a "
                         "COPY of the source dataset's original data/ (full per-episode files + "
                         "provenance columns instance_id/robot_id/scene_id/source_problem). "
                         "Default OFF: the re-render keeps the recorder's compact data (all "
                         "episodes/frames, packed into fewer files) and you can instead attach "
                         "data/ from the source (copy or symlink it). The re-rendered videos match "
                         "the source trajectories "
                         "frame-for-frame, so the source data is the canonical state/action data.")
parser.add_argument("--camera_base", default="data", choices=["data", "scene"],
                    help="Where the camera's PRISTINE base comes from, before per-episode DR is "
                         "composed on top. 'data' (DEFAULT) uses the episode's own recorded "
                         "observation.camera_intrinsics (and, once the pose rebase lands, "
                         "observation.camera_pose), so a re-render reproduces the calibration the "
                         "source frames were captured with. 'scene' uses the built SceneConfig "
                         "(cfg.offset / cam.data.intrinsic_matrices), which is what every run "
                         "before this flag did -- pass it to reproduce those byte-for-byte. "
                         "Unusable recorded values (the recorder writes zeros when a camera had "
                         "none, or a source captured at another resolution) fall back to the "
                         "scene with a warning rather than silently.")
parser.add_argument("--dump_probe", type=str, default=None,
                    help="DIAGNOSTIC, opt-in: write the APPLIED material values, the live carb "
                         "render settings and the per-camera render products to this JSON path "
                         "after the first wave is applied, then continue rendering normally. "
                         "For diffing this renderer's actual state against another consumer of "
                         "the DR package. Off by default; does not change any rendered output.")
parser.add_argument("--match_recorded_cameras", action="store_true",
                    help="(experimental) Override each camera's extrinsics+intrinsics from the "
                         "dataset's recorded camera_pose/camera_intrinsics. Off by default: the "
                         "current scene's cameras already match the recorded ones for these "
                         "datasets, so this is unnecessary (and the orientation handling still "
                         "needs work). flags.yaml camera overrides always take precedence.")
_GLYPH_EPISODES_FILE_MSG = (
    "[rerender] randomize_task_letters cannot be combined with --episodes_file: a patch run "
    "re-renders an arbitrary subset of episodes and renumbers them, so it redraws glyphs that the "
    "dataset's already-written meta/tasks.parquet does not describe -- the frames and the "
    "instruction text would disagree, silently. Re-render the whole problem instead, or drop the "
    "flag. (Problem-aligned --episode_offset slicing IS supported.)")

if __name__ == "__main__":  # (import-safety) app launch + isaaclab imports only in the real entry, not in re-imported workers
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()
    args_cli.headless = True
    args_cli.enable_cameras = True

    # Pre-read flags['carb_settings'] BEFORE launch and pass them as kit_args (`--/path=value`), which
    # take effect at LAUNCH. The offscreen tiled-camera render ignores runtime carb.set for RTX
    # lighting settings (render mode, firefly clamp maxRayUnexposedIntensity, ...) -> kit_args is the
    # only reliable path. yaml is a pure-Python import (safe before the Isaac app launch).
    _early = {}          # bound before the try: the refusal below reads it even if the load fails
    try:
        import yaml as _yaml
        _early = (_yaml.safe_load(open(args_cli.flags)) or {}) if getattr(args_cli, "flags", None) else {}
        _cs = _early.get("carb_settings")
        if isinstance(_cs, dict) and _cs:
            def _kv(k, v):
                v = ("true" if v else "false") if isinstance(v, bool) else v  # kit wants lowercase bools
                return f"--/{str(k).lstrip('/')}={v}"
            _kit = " ".join(_kv(k, v) for k, v in _cs.items())
            args_cli.kit_args = ((getattr(args_cli, "kit_args", "") or "") + " " + _kit).strip()
            print(f"[rerender] kit_args (launch-time carb overrides): {args_cli.kit_args}")
    except Exception as e:
        print(f"[rerender] could not build kit_args from carb_settings: {e}")

    # The glyph feature spans BOTH repos: this file calls into toyblocks's
    # domain_randomization, and TAMP is pinned by toyblocks's submodule pointer, so the two
    # halves can land at different times. On 2026-10-06 ToyBlocks main pinned a TAMP that calls
    # interpreter.sample_wave_task_letters while main's own interpreter had no such function and
    # no glyphs.py at all: every fragment of every run died with AttributeError and the worker
    # then WEDGED at SM 2% holding its GPUs. Check the pairing up front, where the exit code is
    # still honest and no Kit startup has been paid for.
    if (_early.get("randomize_task_letters") if isinstance(_early, dict) else False):
        try:
            from toyblocks.domain_randomization import interpreter as _probe
        except Exception as _e:                       # noqa: BLE001
            raise SystemExit(f"[rerender] randomize_task_letters is set but toyblocks's "
                             f"domain_randomization could not be imported: {_e}")
        _need = [f for f in ("sample_wave_task_letters", "apply_wave_task_letters")
                 if not hasattr(_probe, f)]
        if _need:
            raise SystemExit(
                f"[rerender] randomize_task_letters is set, but this ToyBlocks checkout is "
                f"missing {_need} (and almost certainly src/toyblocks/domain_randomization/"
                f"glyphs.py). Update ToyBlocks to a commit carrying glyphs.py, or remove "
                f"randomize_task_letters from the flags file.")

    # Refuse an impossible flag/argv combination BEFORE the app launches. Raised from main() the
    # same SystemExit is flattened to rc=0 by Isaac's shutdown path, so the worker saw a clean exit
    # with no output instead of an error -- and paid ~245s of Kit startup per attempt to get it.
    if (_early.get("randomize_task_letters") if isinstance(_early, dict) else False) \
            and getattr(args_cli, "episodes_file", None):
        raise SystemExit(_GLYPH_EPISODES_FILE_MSG)
    if getattr(args_cli, "skip_depth", False) and args_cli.extra_depth:
        raise SystemExit("[rerender] --skip_depth cannot be combined with --extra_depth: the "
                         "extra depth stream is derived from the main depth.")
    if getattr(args_cli, "render_scale", None) is not None:
        _skipped = {str(c).split(":")[-1] for c in (args_cli.skip_cameras or [])}
        _missing = {"gripper_camera", "left_camera"} - _skipped
        if _missing:
            raise SystemExit(f"[rerender] --render_scale needs --skip_cameras {' '.join(sorted(_missing))}: "
                             f"robot-mounted cameras are not scaled and the recorder writes one "
                             f"resolution for every camera, so mixed sizes would be resampled.")
        if args_cli.match_recorded_cameras:
            raise SystemExit("[rerender] --render_scale cannot be combined with "
                             "--match_recorded_cameras: recorded intrinsics are at the old size.")

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import torch  # noqa: E402
    import yaml  # noqa: E402
    import gymnasium as gym  # noqa: E402
    from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
    from isaaclab.assets.rigid_object.rigid_object import RigidObject  # noqa: E402
    from toyblocks.domain_randomization.render_sim import (  # noqa: E402
        PHYS_STEP_PER_RENDER_INTERVAL, RenderSim,
    )
    # toyblocks's recorder (the one vecgen datagen and eval use): GPU-side frame snapshot + copy in
    # the frame-build thread, shared-memory frame slots instead of pickling, parallel encoder
    # processes (--num_writers), lossless depth at x265 ultrafast (same decoded values).
    from toyblocks.vecgen.recording.lerobot_recorder import (  # noqa: E402
        IsaacDataRecorderLeRobot, EncoderProcessError,
    )

# Distinct exit code for "the video-encoder subprocess died": lets the launcher
# tell a resumable encoder crash apart from a clean finish or other failure, and
# retry with --resume (which skips already-completed problems).
EXIT_ENCODER_DIED = 3

FPS = 25
# Robot joint constants now live in the shared DR package (dagger replays frames with
# them too); re-exported here under their original names.
from toyblocks.domain_randomization.replay import (  # noqa: E402
    ARM_JOINTS, GRIPPER_DRIVE_JOINT, GRIPPER_MIMIC,
)

# --- Compat shim: stale generated env.py calls the removed XFormPrim table API
# (get_local_poses/set_local_poses) inside _setup_scene_objects.  rerender sets
# table height from recorded data, so these are harmless no-ops (CPU tensors;
# physx view isn't ready at scene-setup time).
_COMPAT_NUM_ENVS = 1
if __name__ == "__main__" and not hasattr(RigidObject, "get_local_poses"):
    def _compat_get_local_poses(self):
        return torch.zeros(_COMPAT_NUM_ENVS, 3), torch.zeros(_COMPAT_NUM_ENVS, 4)

    def _compat_set_local_poses(self, positions=None, orientations=None, *a, **k):
        return None

    RigidObject.get_local_poses = _compat_get_local_poses
    RigidObject.set_local_poses = _compat_set_local_poses


# ---------------------------------------------------------------------------
# Env construction
# ---------------------------------------------------------------------------

def _patch_env_source(src: str) -> str:
    """Neutralize the stale in-setup table-height block (uses removed XFormPrim API
    and assumes scene objects the current sim2real preset names differently)."""
    return re.sub(
        r"table_prim\s*=\s*self\.scene\[[\"']workspace_table[\"']\].*?"
        r"set_local_poses\(\s*pos\s*,\s*quat\s*\)",
        "pass  # [rerender] stale table-height block neutralized",
        src, flags=re.DOTALL,
    )


def load_env_and_task(env_file: str) -> str:
    """Import the generated env package (registers gym id), inject a compat-patched
    ``<pkg>.env`` module, and return the registered task id."""
    env_file = os.path.abspath(env_file)
    pkg_dir = os.path.dirname(env_file)
    parent = os.path.dirname(pkg_dir)
    pkg_name = os.path.basename(pkg_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    before = set(gym.registry.keys())
    importlib.import_module(pkg_name)
    new_ids = sorted(set(gym.registry.keys()) - before)
    if not new_ids:
        raise RuntimeError(f"No gym id registered by env package at {env_file}")

    env_mod_name = f"{pkg_name}.env"
    if env_mod_name not in sys.modules:
        with open(env_file) as f:
            src = _patch_env_source(f.read())
        spec = importlib.util.spec_from_file_location(env_mod_name, env_file)
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[env_mod_name] = mod
        exec(compile(src, env_file, "exec"), mod.__dict__)
    return new_ids[0]


def dump_scene(sim) -> None:
    """Print the stage layout to diagnose per-env vs global geometry."""
    from pxr import Usd, UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    try:
        print("[dump] env_origins:", [[round(x, 3) for x in o]
                                       for o in sim.scene.env_origins.cpu().numpy().tolist()])
    except Exception as e:
        print("[dump] env_origins n/a:", e)

    def wpos(prim):
        try:
            t = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0).ExtractTranslation()
            return (round(t[0], 3), round(t[1], 3), round(t[2], 3))
        except Exception:
            return "?"

    print("[dump] === top-level prims under / ===")
    for c in stage.GetPseudoRoot().GetChildren():
        print("   ", c.GetPath().pathString, f"({c.GetTypeName()})")
    w = stage.GetPrimAtPath("/World")
    if w and w.IsValid():
        print("[dump] === /World children ===")
        for c in w.GetChildren():
            print("   ", c.GetPath().pathString, f"({c.GetTypeName()})")

    # GLOBAL geometry = Imageable prims with a real gprim (Mesh/Plane/etc.) NOT under an env clone.
    # This is where a cross-env 'white plane' would live. Report the top-most such prims + world pos.
    print("[dump] === GLOBAL geometry (Mesh/Gprim NOT under /World/envs/env_*) ===")
    seen = 0
    for prim in stage.Traverse():
        p = prim.GetPath().pathString
        if "/envs/env_" in p or "/Visuals" in p:
            continue
        tn = prim.GetTypeName()
        if tn in ("Mesh", "Plane", "Cube", "Sphere", "Cylinder", "Cone", "Capsule",
                  "GeomSubset") or (prim.IsA(UsdGeom.Gprim)):
            print(f"   [GLOBAL] {p}  ({tn})  pos={wpos(prim)}")
            seen += 1
            if seen > 60:
                print("   ... (truncated)"); break
    if seen == 0:
        print("   (none found — geometry is all per-env)")


def _remap_robot_id(rid: str, registered) -> str:
    """Pick the registered robot id sharing the longest name prefix with `rid`
    (e.g. 'fr3_robotiq_gripper_softer' -> 'fr3_robotiq_gripper_soft')."""
    keys = list(registered)
    if not keys:
        return rid
    return max(keys, key=lambda k: len(os.path.commonprefix([rid, k])))


class _robot_camera_options:
    """Apply --skip_cameras / --render_scale / --skip_depth to ROBOT-mounted cameras too.

    build.py edits the sim2real preset, but BaseEnv adds the robot's cameras (gripper_camera)
    afterwards, so those were never dropped, scaled or stripped of depth. BaseEnv calls
    common_utils.apply_camera_overrides_json right after adding them; this wraps that call for the
    duration of gym.make. A no-op unless one of the options names or affects a robot camera."""

    _DEPTH = ("distance_to_image_plane", "depth", "distance_to_camera")

    def __init__(self, skip_cameras, render_scale, skip_depth: bool):
        self.skip = {str(c).split(":")[-1] for c in (skip_cameras or [])}
        self.scale = float(render_scale) if render_scale else None
        self.skip_depth = skip_depth

    def __enter__(self):
        from toyblocks.scenes import SCENE_PRESETS
        from toyblocks.utils import common_utils as _cu
        self._cu, self._orig = _cu, _cu.apply_camera_overrides_json
        preset = set(SCENE_PRESETS["sim2real"].camera_cfgs)
        opts = self

        def _wrapped(cfg):
            opts._orig(cfg)
            cams = getattr(cfg, "camera_cfgs", None) or {}
            for name in [n for n in cams if n not in preset]:
                if name in opts.skip:
                    del cams[name]
                    print(f"[rerender] --skip_cameras: robot camera '{name}' NOT spawned")
                    continue
                cam = cams[name]
                if opts.scale and getattr(cam, "width", None) and getattr(cam, "height", None):
                    w, h = int(round(cam.width * opts.scale)), int(round(cam.height * opts.scale))
                    if w % 2 or h % 2:
                        raise ValueError(f"--render_scale {opts.scale}: '{name}' would be {w}x{h}")
                    print(f"[rerender] --render_scale {opts.scale}: robot camera '{name}' "
                          f"{cam.width}x{cam.height} -> {w}x{h}")
                    cam.width, cam.height = w, h
                if opts.skip_depth and getattr(cam, "data_types", None):
                    cam.data_types = [t for t in cam.data_types if t not in opts._DEPTH]
                    print(f"[rerender] --skip_depth: robot camera '{name}' renders {cam.data_types}")

        if self.skip or self.scale or self.skip_depth:
            _cu.apply_camera_overrides_json = _wrapped
        return self

    def __exit__(self, *exc):
        self._cu.apply_camera_overrides_json = self._orig
        return False


def build_simulator(task_id: str, num_envs: int, device: str, flags: dict):
    env_cfg = parse_env_cfg(task_id)
    env_cfg.scene.num_envs = num_envs
    env_cfg.sim.device = device
    # build-time DR scene config (env_spacing/GI/render-mode/carb/AA/camera cfg) -> shared package.
    _dr_build.apply_build_flags(env_cfg, flags, cli_env_spacing=args_cli.env_spacing,
                                antialiasing=args_cli.antialiasing,
                                disable_dlss=args_cli.disable_dlss, enable_gi=args_cli.enable_gi,
                                skip_cameras=args_cli.skip_cameras,
                                render_scale=getattr(args_cli, "render_scale", None),
                                skip_depth=bool(getattr(args_cli, "skip_depth", False)),
                                # 0 = fresh capture per scene.update(); the preset default
                                # (0.04s) is time-gated and can skip a refresh under the manual
                                # render loop below, duplicating a frame.
                                camera_update_period=0.0)
    # --- rerender-specific env_cfg setup (render-only replay) ---
    rid = getattr(env_cfg, "robot_id", None)
    if rid is not None:
        try:
            from toyblocks.robots import REGISTERED_ROBOTS
            if rid not in REGISTERED_ROBOTS:
                fb = _remap_robot_id(rid, REGISTERED_ROBOTS)
                print(f"[rerender] robot_id '{rid}' not registered; remapping to '{fb}' "
                      "(render-only: physics/softness variant irrelevant)")
                env_cfg.robot_id = fb
        except Exception as e:
            print(f"[rerender] robot_id remap skipped: {e}")
    if hasattr(env_cfg, "always_spawn_cameras"):
        env_cfg.always_spawn_cameras = True
    else:
        env_cfg.obs_mode = "rgb+state"
    if hasattr(env_cfg, "reset_states_path"):
        env_cfg.reset_states_path = "/nonexistent_rerender_bypass"
    # Compat: stub any missing goal-conversion helper (replay ignores live goal eval).
    try:
        import torch as _torch
        import toyblocks.tamp.blocks.pddl_goal_tensor_conversions as _gtc
        def _rr_goal_stub(env, *a, **k):
            return _torch.ones(env.num_envs, dtype=_torch.bool, device=env.device)
        if not getattr(_gtc, "_rr_stub_installed", False):
            def _rr_mod_getattr(name):
                if name.startswith("__"):
                    raise AttributeError(name)
                print(f"[rerender] compat: stubbing missing goal helper {name!r} (replay ignores goal eval)")
                return _rr_goal_stub
            _gtc.__getattr__ = _rr_mod_getattr
            _gtc._rr_stub_installed = True
    except Exception as _e:
        print(f"[rerender] compat goal-stub skipped: {_e}")
    with _robot_camera_options(args_cli.skip_cameras, getattr(args_cli, "render_scale", None),
                               bool(getattr(args_cli, "skip_depth", False))):
        env = gym.make(id=task_id, cfg=env_cfg)
    sim = RenderSim(env, num_envs=num_envs, skip_initial_step=args_cli.skip_initial_step)
    _dr_build.apply_lighting_flags(flags, sim)
    if not args_cli.show_gizmos:
        _dr_build.hide_debug_visuals()
    _dr_build.hide_scene_objects(flags)
    return sim


def _is_lerobot_dataset(p: Path) -> bool:
    # is_dir(), not exists(): this predicate MUST match render_sources._is_ds exactly. The
    # planner enumerates splits with that one and --episode_offset indexes the resulting ordered
    # list, so a split one side accepts and the other rejects shifts every later offset silently.
    # A `data` that is a file rather than a directory did exactly that (worker +2 episodes).
    return (p / "meta" / "info.json").exists() and (p / "data").is_dir()


# Datagen emits seed variants of a problem as <problem>__s001, __s002, ... but writes ONE env per
# BASE problem, so a variant never has an exactly-named env.py. The variants are the same PDDL
# problem re-run with different seeds -- same scene, same block set -- so the base's env is the
# correct one for them.
_SEED_SUFFIX = re.compile(r"__s\d+$")


def _base_problem(name: str) -> str:
    """'problem_x__s001' -> 'problem_x'; unchanged when there is no seed suffix."""
    return _SEED_SUFFIX.sub("", name)


def _resolve_env_file(candidates, fallback: Optional[str]) -> Optional[str]:
    """Return the first existing candidate env.py path, else the fallback --env_file.
    `candidates` is an ordered list of Path|None (highest precedence first)."""
    for c in candidates:
        if c is not None and c.exists():
            return str(c)
    return fallback


# Problems skipped because their env's block set differs from the built scene (see the job loop).
_SCENE_SKIPPED: List[str] = []


def _env_block_names(env_file: Optional[str]) -> Optional[frozenset]:
    """Cheap scene-signature: the env's block_names list, parsed from source (no import)."""
    if not env_file or not os.path.exists(env_file):
        return None
    with open(env_file) as f:
        src = f.read()
    m = re.search(r"block_names\s*=\s*\[([^\]]*)\]", src)
    if not m:
        return None
    return frozenset(re.findall(r"['\"]([^'\"]+)['\"]", m.group(1)))


def discover_jobs(datasets_root: str, output_root: str,
                  explicit_env_file: Optional[str],
                  envs_dir: Optional[str] = None,
                  skip_failures: bool = False,
                  split_groups: bool = False) -> List[Dict[str, Any]]:
    """Return a list of jobs: {out_dir, sources:[(dataset_path, success_bool|None)], env_file}.

    Auto-detects layout:
    - Unmerged: <root>/problem_*/blocks_{success,failure}/  -> one job per problem
      (success + failure routed through one recorder).
    - Flat/merged: <root>/{data,meta,videos}  -> one job.

    Per-problem env is resolved in precedence order: <envs_dir>/<problem_name>/env.py
    (when --envs_dir is given; matches the S3 layout where envs live under a separate
    root such as basics/<exp>/envs/), then <root>/envs/<problem_name>/env.py, then the
    explicit --env_file. (Flat falls back to <root>/env/env.py.)
    """
    root = Path(datasets_root)
    out = Path(output_root)
    ed = Path(envs_dir) if envs_dir else None
    if not root.exists():
        raise FileNotFoundError(f"datasets_dir not found: {datasets_root}")

    problem_dirs = sorted(d for d in root.iterdir()
                          if d.is_dir() and d.name.startswith("problem_"))
    if problem_dirs:  # unmerged layout
        jobs = []
        members: Dict[str, Any] = {}
        if split_groups:
            from render_sources import group_members
            members = group_members(str(root), (("blocks_success",) if skip_failures
                                                else ("blocks_success", "blocks_failure")))
        for pd_ in problem_dirs:
            sources = []
            _splits = ((("blocks_success", True),) if skip_failures
                       else (("blocks_success", True), ("blocks_failure", False)))
            for split, succ in _splits:
                sub = pd_ / split
                if _is_lerobot_dataset(sub):
                    sources.append((sub, succ))
            if sources:
                # Exact name first, then the BASE problem for a __sNNN seed variant. Without the
                # base fallback an item made up only of variants resolves no env at all and the
                # whole item dies at "No env found" -- 4 items of aug27_pnp_2r2g_soft failed 8x
                # into the DLQ exactly this way.
                _bp = _base_problem(pd_.name)
                env_file = _resolve_env_file(
                    [ed / pd_.name / "env.py" if ed else None,
                     root / "envs" / pd_.name / "env.py",
                     (ed / _bp / "env.py") if (ed and _bp != pd_.name) else None,
                     (root / "envs" / _bp / "env.py") if _bp != pd_.name else None],
                    explicit_env_file)
                if pd_.name in members:
                    # One job per member problem: the group's source and env, its own output dir
                    # (which also salts its DR), and only its episodes (see main's fragment filter).
                    _by_split = members[pd_.name]
                    for _m in sorted({m for v in _by_split.values() for m in v}):
                        jobs.append({"out_dir": str(out / _m), "sources": sources,
                                     "env_file": env_file, "group": pd_.name, "member": _m,
                                     "member_episodes": {sp: list(v.get(_m, ()))
                                                         for sp, v in _by_split.items()}})
                    continue
                jobs.append({"out_dir": str(out / pd_.name), "sources": sources,
                             "env_file": env_file})
        return jobs

    if _is_lerobot_dataset(root):  # flat merged layout
        env_file = _resolve_env_file(
            [ed / root.name / "env.py" if ed else None,
             ed / "env.py" if ed else None,
             root / "env" / "env.py"],
            explicit_env_file)
        return [{"out_dir": str(out / root.name), "sources": [(root, None)], "env_file": env_file}]

    # Fallback: any nested LeRobot datasets.
    jobs = []
    for meta in sorted(root.rglob("meta/info.json")):
        ds = meta.parent.parent
        rel = ds.relative_to(root)
        env_file = _resolve_env_file(
            [ed / ds.name / "env.py" if ed else None, ds.parent / "env" / "env.py"],
            explicit_env_file)
        jobs.append({"out_dir": str(out / rel), "sources": [(ds, None)], "env_file": env_file})
    return jobs


# load_episodes now lives in the shared DR package so dagger (and any other
# consumer) can build the same episode records; imported at module scope below.
from toyblocks.domain_randomization import episodes as _dr_episodes  # noqa: E402
from toyblocks.domain_randomization.episodes import load_episodes  # noqa: E402

# _resolve_joint_ids / _set_table_height moved to the shared DR package. Imported under
# their original names so rerender_from_params' T._resolve_joint_ids still resolves.
from toyblocks.domain_randomization.replay import (  # noqa: E402
    _resolve_joint_ids, _set_table_height, park_inactive_objects,
)

# _moving_cameras moved to the shared DR package (dagger needs the same selection).
from toyblocks.domain_randomization.replay import _moving_cameras  # noqa: E402


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


def _resolve_skybox_textures(sky):
    """Adapter: inject the package-resolved default HDRI dir (was the DEFAULT_HDRI_DIR global)."""
    return _dr_resolve.resolve_skybox_textures(sky, DEFAULT_HDRI_DIR)


def _SkyBuckets(flag, rng):
    """Adapter: construct the package SkyBuckets with rerender's default HDRI dir."""
    return _PkgSkyBuckets(flag, rng, DEFAULT_HDRI_DIR)


# Camera / seed-salt / dome-stage randomization moved to toyblocks.domain_randomization.{camera,seeds,skybox}
# (per-concern modules). Names re-bound so rerender call sites are unchanged.
from toyblocks.domain_randomization import camera as _dr_camera, seeds as _dr_seeds, skybox as _dr_skybox
_stable_seed_salt = _dr_seeds._stable_seed_salt
_problem_seed_salt = _dr_seeds._problem_seed_salt
RAND_KEYS = _dr_camera.RAND_KEYS
RAND_DEFAULTS = _dr_camera.RAND_DEFAULTS
_quat_mul = _dr_camera._quat_mul
_qmul = _dr_camera._qmul
_delta_quat = _dr_camera._delta_quat
_norm_rand_dict = _dr_camera._norm_rand_dict
_resolve_cam_ranges = _dr_camera._resolve_cam_ranges
_sample_episode_offsets = _dr_camera._sample_episode_offsets
_is_link_mounted = _dr_camera._is_link_mounted
_capture_camera_bases = _dr_camera._capture_camera_bases
# A camera that has a recorded base but no DR draw still needs its mount written; this carries it
# through the same composition with nothing added.
ZERO_CAM_OFFSET = {"position_offset_cm": [0.0, 0.0, 0.0], "pitch_deg": 0.0, "yaw_deg": 0.0,
                   "roll_deg": 0.0, "vertical_fov_deg": 0.0}
_apply_camera_local_offsets = _dr_camera._apply_camera_local_offsets
_apply_episode_camera_fov = _dr_camera._apply_episode_camera_fov
_apply_episode_camera_offsets = _dr_camera._apply_episode_camera_offsets
_initial_camera_params = _dr_camera._initial_camera_params
_absolute_cam_params = _dr_camera._absolute_cam_params
_override_cameras = _dr_camera._override_cameras
SPHERE_FILL_PATH = _dr_skybox.SPHERE_FILL_PATH
_dome_intensity = _dr_skybox._dome_intensity
_sphere_fill_intensity = _dr_skybox._sphere_fill_intensity
_apply_skybox = _dr_skybox._apply_skybox

# The neutral "fill" SphereLight bundled inside Isaac Lab's default ground-plane asset
# (spawn_ground_plane -> GroundPlaneCfg, base_env.py). It's a single GLOBAL prim (not per-env)
# at a fixed high intensity (100000), and GroundPlaneCfg exposes no knob for it, so we set it
# on the prim directly (like the dome). Empirically its effect on the enclosed tabletop is
# modest (small radius, below the ground), but this lets the skybox flow control it for a
# cleaner HDRI look / lighting DR. See scene_lights in the flags docstring.
# Exposure/carb and scene-light randomization moved to toyblocks.domain_randomization.{exposure,lights}
# (per-concern modules). Names re-bound so rerender call sites are unchanged.
from toyblocks.domain_randomization import exposure as _dr_exposure, lights as _dr_lights
_EXPOSURE_KEYS = _dr_exposure._EXPOSURE_KEYS
_dump_exposure_settings = _dr_exposure._dump_exposure_settings
_apply_exposure = _dr_exposure._apply_exposure
_apply_carb_settings = _dr_exposure._apply_carb_settings
_set_render_mode = _dr_exposure._set_render_mode
LOCAL_LIGHT_ROOT = _dr_lights.LOCAL_LIGHT_ROOT
_kelvin_to_rgb = _dr_lights._kelvin_to_rgb
_sample_color = _dr_lights._sample_color
_sample_light = _dr_lights._sample_light
_local_light_slot_types = _dr_lights._local_light_slot_types
_setup_local_lights = _dr_lights._setup_local_lights
_sample_episode_local_lights = _dr_lights._sample_episode_local_lights
_apply_local_lights = _dr_lights._apply_local_lights
_setup_global_sun = _dr_lights._setup_global_sun
_apply_global_sun = _dr_lights._apply_global_sun
# env_sphere backdrop randomization moved to toyblocks.domain_randomization.env_sphere
# (per-concern module). Names re-bound so rerender call sites are unchanged.
from toyblocks.domain_randomization import env_sphere as _dr_env_sphere
ENV_SPHERE_ROOT = _dr_env_sphere.ENV_SPHERE_ROOT
_make_uv_sphere_mesh = _dr_env_sphere._make_uv_sphere_mesh
_set_camera_far_clip = _dr_env_sphere._set_camera_far_clip
_setup_env_spheres = _dr_env_sphere._setup_env_spheres
_hdri_luminance = _dr_env_sphere._hdri_luminance
_apply_env_spheres = _dr_env_sphere._apply_env_spheres


def _sample_episode_env_spheres(spec, episode_indices, seed):
    """Adapter: inject rerender default HDRI dir (was DEFAULT_HDRI_DIR global) for the bucket picker."""
    return _dr_env_sphere._sample_episode_env_spheres(spec, episode_indices, seed, DEFAULT_HDRI_DIR)
# Backward-compatible alias (gripper-only call sites / older notes).
_apply_link_camera_mount_offsets = _apply_camera_local_offsets


# Table material randomization moved to toyblocks.domain_randomization.materials (per-concern
# module: pure samplers + pxr appliers). Names re-bound so rerender call sites are unchanged.
from toyblocks.domain_randomization import materials as _dr_materials
MATERIAL_VEC2_KEYS = _dr_materials.MATERIAL_VEC2_KEYS
MATERIAL_VEC3_KEYS = _dr_materials.MATERIAL_VEC3_KEYS
MATERIAL_JITTER_KEYS = _dr_materials.MATERIAL_JITTER_KEYS
MATERIAL_ASSET_KEYS = _dr_materials.MATERIAL_ASSET_KEYS
MATERIAL_MDL_DEFAULTS = _dr_materials.MATERIAL_MDL_DEFAULTS
MATERIAL_NAME_ALIASES = _dr_materials.MATERIAL_NAME_ALIASES
_TINT_HSV_KEYS = _dr_materials._TINT_HSV_KEYS
_srange = _dr_materials._srange
_normalize_material = _dr_materials._normalize_material
_sample_material_props = _dr_materials._sample_material_props
_load_material_library = _dr_materials._load_material_library
_find_shader_prim = _dr_materials._find_shader_prim
_resolve_object_shader_prims = _dr_materials._resolve_object_shader_prims
_read_shader_props = _dr_materials._read_shader_props
_initial_material_params = _dr_materials._initial_material_params
_combined_material_props = _dr_materials._combined_material_props
_sample_episode_materials = _dr_materials._sample_episode_materials
_resolve_material_texture = _dr_materials._resolve_material_texture
_apply_material_to_shader = _dr_materials._apply_material_to_shader
_apply_episode_materials = _dr_materials._apply_episode_materials


def _resolve_material_specs(flags):
    """Adapter: inject rerender default table-material library (was the DEFAULT_TABLE_MATERIALS global)."""
    return _dr_materials._resolve_material_specs(flags, DEFAULT_TABLE_MATERIALS)
# render_params.json writer + DR flags loader moved to toyblocks.domain_randomization.
# {render_params, config} (reusable by eval). Names re-bound so rerender call sites are unchanged;
# the run-args block (was inline from args_cli) is supplied here via _rp_args().
from toyblocks.domain_randomization import render_params as _dr_rp, config as _dr_config
from toyblocks.domain_randomization import build as _dr_build
import scene_swap as _scene_swap
from toyblocks.domain_randomization import interpreter as _dr_interp
from toyblocks.domain_randomization.assets import AssetPaths as _AssetPaths
_ASSETS = None                              # ordered HDRI/skybox roots; built in main() from --assets_paths


def _rp_args() -> dict:
    a = args_cli
    return {
        "datasets_dir": a.datasets_dir, "output": a.output, "env_file": a.env_file,
        "envs_dir": a.envs_dir, "flags_file": a.flags, "max_episodes": a.max_episodes,
        "dr_seeding": a.dr_seeding,
        "renders_per_episode": int(a.renders_per_episode), "num_envs": a.num_envs,
        "skip_initial_step": bool(a.skip_initial_step), "physics_step": bool(a.physics_step),
        "match_recorded_cameras": bool(a.match_recorded_cameras),
        "camera_base": a.camera_base,
        "copy_source_data": bool(a.copy_source_data), "disable_dlss": bool(a.disable_dlss),
        "antialiasing": a.antialiasing, "rtx_warmup_frames": int(a.rtx_warmup_frames),
        "depth_codec": a.depth_codec, "depth_format": a.depth_format,
        "extra_depth": a.extra_depth, "extra_rgb": a.extra_rgb,
        "depth_gop": int(a.depth_gop),
        "extra_rgb_GOP": int(a.extra_rgb_GOP), "rgb_GOP": int(a.rgb_GOP),
        "show_gizmos": bool(a.show_gizmos),
        "enable_gi": bool(a.enable_gi),
        # Recorded only when set, so runs without them keep their exact render_params.
        # env_spacing is geometry: too small and neighbor envs appear in frame, so a dataset must
        # carry it. Conditional because fast_patch reads it as rargs.get("env_spacing", 60) -- a
        # recorded None would drop --env_spacing from its rebuilt argv instead of defaulting to 60.
        # The effective value is yaml flags['env_spacing'] > this > preset (see dr.build).
        **({"env_spacing": float(a.env_spacing)} if a.env_spacing is not None else {}),
        **({"render_scale": float(a.render_scale)} if getattr(a, "render_scale", None) else {}),
        **({"skip_depth": True} if getattr(a, "skip_depth", False) else {}),
        # fast_patch rebuilds the render argv from these args; the new options need the cameras.
        **({"skip_cameras": list(a.skip_cameras)}
           if a.skip_cameras and (getattr(a, "render_scale", None) or getattr(a, "skip_depth", False))
           else {}),
    }


_split_meta_dirs = _dr_rp.split_meta_dirs
_write_json_locked = _dr_rp.write_json_locked
_job_completed = _dr_rp.job_completed


def _render_params_payload(job, flags, fps, use_physics, initial, episode_records, completed,
                           total_episodes=None):
    return _dr_rp.render_params_payload(job, flags, fps, use_physics, initial, episode_records,
                                        completed, _rp_args(), total_episodes)



def _patch_actions_from_source(out_dir, job, episode_records, verbose=True):
    """Overwrite the recorder's re-derived action.* columns with the SOURCE episode's values.

    The replay writes joint STATES (write_joint_state_to_sim) and never calls
    set_joint_position_target or issues a gripper command, so the recorder samples
    articulation.data.joint_pos_target -- a buffer nothing in a render-only replay writes. The
    result: action.joint_target is one constant vector for the whole episode (an unset buffer, not
    a trajectory), action.gripper_position is uniformly 0, and action.relative_joint_target derives
    from both. observation.state.* are unaffected because they come from the states we DO write,
    which is why a re-rendered dataset looks correct until the actions are used for training.

    Each output episode maps to exactly one source episode via src_episode_index, so this copy is
    exact and -N>1 safe (each of the N copies takes the same source episode's actions).
    """
    import pandas as pd
    ACTION_PREFIX = "action."
    src_by_split = {}
    for ds_path, succ in job["sources"]:
        split = "blocks_success" if succ else ("blocks_failure" if succ is False else "blocks")
        files = sorted((Path(ds_path) / "data").glob("chunk-*/file-*.parquet"))
        if not files:
            continue
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        src_by_split[split] = {int(e): g.reset_index(drop=True)
                               for e, g in df.groupby("episode_index")}
    if not src_by_split:
        print("[rerender] patch-actions: no source data/ found; leaving actions as recorded")
        return
    # output episode_index -> src_episode_index, per split
    out2src = {}
    for r in episode_records:
        out2src.setdefault(r.get("split", "blocks_success"), {})[int(r["episode_index"])] = \
            int(r["src_episode_index"])
    patched = skipped = 0
    for split, mapping in out2src.items():
        src_eps = src_by_split.get(split) or {}
        for pf in sorted((Path(out_dir) / split / "data").glob("chunk-*/file-*.parquet")):
            df = pd.read_parquet(pf)
            cols = [c for c in df.columns if c.startswith(ACTION_PREFIX)]
            if not cols or "episode_index" not in df.columns:
                continue
            changed = False
            for oidx, grp in df.groupby("episode_index"):
                sidx = mapping.get(int(oidx))
                sdf = src_eps.get(sidx) if sidx is not None else None
                if sdf is None or len(sdf) != len(grp):
                    skipped += 1
                    continue
                for c in cols:
                    if c in sdf.columns:
                        df.loc[grp.index, c] = sdf[c].values
                changed = True; patched += 1
            if changed:
                df.to_parquet(pf, index=False)
    msg = f"[rerender] patch-actions: copied source action.* into {patched} episode(s)"
    if skipped:
        msg += f"; SKIPPED {skipped} (no source match or frame-count mismatch -- actions left as recorded)"
    if verbose:
        print(msg)


def _cam_pose_deltas(randomize_cams, ep_offsets):
    """Per-episode local mount delta for EVERY randomized camera.

    The recorder stores each camera pose BASE-RELATIVE, as base^-1 . parent_world . M where M is the
    USD mount transform. DR replaces M_orig=(pos,rot) with M_new=(pos+dpos, rot*dq), so

        T_rel_new = T_rel_orig . (M_orig^-1 . M_new)

    i.e. the stored pose composes with a FIXED per-episode delta. This holds for BOTH mounts: for a
    link-mounted cam the parent is the moving gripper link, for a static cam it is the env Xform.

    Needed for both, because the recorder's pose is wrong in both cases -- DR is applied by writing
    the USD LOCAL transform (the only thing the tiled renderer honours), and cam_data.pos_w does not
    reflect that write. Measured on the allrand yaml: the front cam's recorded pose moved 22um and
    0.007deg against a configured +-3cm/+-5deg, while its intrinsics (set through a different API)
    did change -- so the stored pose described a camera that never rendered the image. The gripper
    cam has the extra problem that pos_w never refreshes at all in a render-only replay.

    Returns, per camera key, the cfg convention plus the base quaternion and each episode's
    (dpos, dq) -- everything the patcher needs to redo the composition without a live sim.
    """
    out = {}
    for cam_key, cam in (randomize_cams or {}).items():
        off_cfg = getattr(getattr(cam, "cfg", None), "offset", None)
        if off_cfg is None:
            continue
        per_ep = {}
        for ep, cams in (ep_offsets or {}).items():
            o = cams.get(cam_key)
            if o is None:
                continue
            per_ep[int(ep)] = {
                "dpos": [x / 100.0 for x in o["position_offset_cm"]],
                "dq": list(_dr_camera._delta_quat(math.radians(o["pitch_deg"]),
                                                  math.radians(o["yaw_deg"]),
                                                  math.radians(o.get("roll_deg", 0.0)))),
            }
        out[cam_key] = {"conv": getattr(off_cfg, "convention", "ros") or "ros",
                        "q_base": list(off_cfg.rot), "per_ep": per_ep}
    return out


def _quat_rotate_wxyz(q, v):
    """Rotate v by quaternion q (torch, wxyz, broadcastable).

    v + 2w(qv x v) + 2 qv x (qv x v)
    """
    w, qv = q[:, :1], q[:, 1:]
    t = torch.cross(qv, v, dim=-1)
    return v + 2.0 * (w * t + torch.cross(qv, t, dim=-1))


def _compose_link_cam_pose(pose7, dpos, dq, q_base, conv):
    """Apply a local mount delta to stored BASE-RELATIVE camera poses. pose7: (T,7) xyz+wxyz.

    Conventions are done with isaaclab's own converter rather than a hardcoded axis swap: the
    stored quaternion is in the `world` camera convention while cfg.offset.rot and the sampled
    delta are in the cfg convention (usually ros), so the delta has to be applied in cfg space.
    """
    from isaaclab.utils.math import convert_camera_frame_orientation_convention
    p = torch.as_tensor(pose7[:, :3], dtype=torch.float64)
    qw = torch.as_tensor(pose7[:, 3:], dtype=torch.float64)
    qb = torch.as_tensor([q_base], dtype=torch.float64)
    dqt = torch.as_tensor([dq], dtype=torch.float64)
    dp = torch.as_tensor(dpos, dtype=torch.float64)

    q_cfg = convert_camera_frame_orientation_convention(qw.float(), origin="world",
                                                        target=conv).double()
    # parent link rotation = cam_cfg (x) base^-1 ; a link is not a camera, so it is convention-free
    qb_inv = torch.cat([qb[:, :1], -qb[:, 1:]], dim=1)
    q_par = _quat_mul(q_cfg, qb_inv.expand_as(q_cfg))
    # rotate dpos (parent frame) into the stored frame
    p_new = p + _quat_rotate_wxyz(q_par, dp.unsqueeze(0).expand(q_par.shape[0], 3))
    q_new_cfg = _quat_mul(q_cfg, dqt.expand_as(q_cfg))
    q_new = convert_camera_frame_orientation_convention(q_new_cfg.float(), origin=conv,
                                                        target="world").double()
    return torch.cat([p_new, q_new], dim=1).numpy().astype(np.float32)


def _src_episode_index(ds_path):
    """Locate every source episode WITHOUT reading the data.

    Returns [(ParquetFile, {episode: (row_start, n_rows, n_rendered)}, row_group_starts)] per
    file, ``n_rendered`` = rows not flagged in ``FAILED_ATTEMPT_COL`` (what load_episodes keeps).
    Only episode_index (and that flag) is read (a few hundred KB for a 78k-row split), so the
    caller can then fetch one episode at a time instead of holding the whole split resident.
    """
    import pyarrow.parquet as pq
    from toyblocks.domain_randomization.episodes import FAILED_ATTEMPT_COL, flag_values
    out = []
    for f in sorted((Path(ds_path) / "data").glob("chunk-*/file-*.parquet")):
        # A corrupt SOURCE file must not take the render process down. Observed 2026-09-11:
        # "ArrowInvalid: Parquet magic bytes not found in footer" propagated out of replay_job to
        # main(), and Isaac's teardown then hung -- so the process stopped emitting and the
        # worker's watchdog SIGKILLed it 3600s later. One bad source file therefore burned a full
        # hour of a render slot AND discarded the fragment's in-flight problem. Skip the file,
        # say so loudly, and let the rest of the split render.
        try:
            pf = pq.ParquetFile(f)
            has_flag = FAILED_ATTEMPT_COL in pf.schema_arrow.names
            tbl = pf.read(columns=["episode_index"] + ([FAILED_ATTEMPT_COL] if has_flag else []))
            eps = np.asarray(tbl.column("episode_index").to_numpy())
            flag = (flag_values(tbl.column(FAILED_ATTEMPT_COL).to_pylist()) if has_flag
                    else np.zeros(len(eps), dtype=bool))
        except Exception as e:                       # noqa: BLE001 -- arrow raises several types
            print(f"[rerender] SKIPPING CORRUPT SOURCE PARQUET {f}: {type(e).__name__}: {e}",
                  file=sys.stderr, flush=True)
            continue
        spans = {}
        if len(eps):
            # episodes are contiguous runs; record each run's [start, len)
            edges = np.nonzero(np.diff(eps))[0] + 1
            starts = np.concatenate([[0], edges])
            ends = np.concatenate([edges, [len(eps)]])
            for s, e in zip(starts, ends):
                spans[int(eps[s])] = (int(s), int(e - s), int((~flag[s:e]).sum()))
        md = pf.metadata
        rg = np.cumsum([0] + [md.row_group(i).num_rows for i in range(md.num_row_groups)])
        out.append((pf, spans, rg))
    return out


def _read_src_rows(pf, rg_starts, start, n):
    """Read rows [start, start+n) reading ONLY the row groups that overlap them.

    These sources are written one row group per episode (~0.4 MiB), so an episode costs about that
    much instead of the 1.2 GiB a whole-split read costs.
    """
    lo = int(np.searchsorted(rg_starts, start, "right") - 1)
    hi = int(np.searchsorted(rg_starts, start + n - 1, "right") - 1)
    tbl = pf.read_row_groups(list(range(lo, hi + 1)))
    return tbl.slice(start - int(rg_starts[lo]), n)


#: Columns the RE-RENDER owns; passthrough copies every other source column verbatim.
PASSTHROUGH_OWNED_BASE = frozenset({"episode_index", "index"})
#: EMPTY. Nothing extra is owned when randomize_task_letters is on -- every text column is
#: REBUILT from the source instead:
#:   * pddl_goal  -- owning it discarded the source's real goal (the recorder has no PDDL problem
#:                   file in a re-render, so its own is empty: observed as an all-null column).
#:                   Rewritten from the source value in the rebuild loop below.
#:   * task_index -- owning it took the RECORDER's index, which points into the recorder's own
#:                   one-row table holding the generic string "manipulation". That silently threw
#:                   away every real task string ("pnp_table_block ['robot', 'block_e', ...]") and
#:                   left the dataset with no instruction text at all. It is now re-pointed at a
#:                   task table rebuilt per episode through that episode's glyph mapping.
#:   * glyph_map  -- ADDED in the rebuild loop; the rebuild iterates src_schema.names, so a column
#:                   the source lacks can never survive by being owned.
PASSTHROUGH_OWNED_TASK_LETTERS = frozenset()


def _bc_palette_for_run(flags):
    """The resolved colour palette for this run, or None when colour DR is off."""
    spec = (flags or {}).get("randomize_block_colors")
    if not spec:
        return None
    from toyblocks.domain_randomization.block_colors import parse_palette
    return parse_palette(spec)


def passthrough_owned_columns(task_letters: bool) -> frozenset:
    """Source columns passthrough must NOT copy over the render's own values."""
    return (PASSTHROUGH_OWNED_BASE | PASSTHROUGH_OWNED_TASK_LETTERS) if task_letters \
        else PASSTHROUGH_OWNED_BASE


def _patch_episode_task_meta(out_dir, split, strings, ep_idx, verbose=True):
    """Make meta/episodes agree with the rebuilt task table.

    Two columns there describe the task and both go stale once task_index is re-pointed:
      * `tasks` -- the per-episode language instruction LeRobot consumers actually read. The
        recorder writes the placeholder ["manipulation"] in every re-render (the unswapped control
        shows it too), so leaving it would publish the placeholder next to a correctly rewritten
        tasks.parquet.
      * `stats/task_index/*` -- min/max/mean/std/count/quantiles over a column whose values have
        just changed. They used to be trivially right because task_index was all zeros.
    """
    import numpy as _np
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = os.path.join(out_dir, split, "meta", "episodes")
    files = sorted(glob.glob(os.path.join(d, "chunk-*", "file-*.parquet")))
    if not files or not ep_idx:
        return
    for f in files:
        t = pq.read_table(f)
        names = t.schema.names
        if "episode_index" not in names:
            continue
        eps = t.column("episode_index").to_pylist()
        cols, out_names = [], []
        for c in names:
            col = t.column(c)
            if c == "tasks":
                vals = []
                for e in eps:
                    ix = ep_idx.get(int(e))
                    vals.append(sorted({strings[i] for i in ix if i < len(strings)})
                                if ix else col.to_pylist()[eps.index(e)])
                col = pa.array(vals, type=t.schema.field(c).type)
            elif c.startswith("stats/task_index/"):
                stat = c.rsplit("/", 1)[1]
                vals = []
                for e in eps:
                    ix = ep_idx.get(int(e))
                    if not ix:
                        vals.append(None)
                        continue
                    a = _np.asarray(ix, dtype=float)
                    v = {"min": a.min(), "max": a.max(), "mean": a.mean(), "std": a.std(),
                         "count": float(len(a))}.get(stat)
                    if v is None and stat.startswith("q"):
                        v = float(_np.quantile(a, int(stat[1:]) / 100.0))
                    vals.append([v] if v is not None else None)
                try:
                    col = pa.array(vals, type=t.schema.field(c).type)
                except (pa.ArrowInvalid, pa.ArrowTypeError):
                    col = t.column(c)                  # leave it rather than corrupt the schema
            cols.append(col if isinstance(col, pa.ChunkedArray) else pa.chunked_array([col]))
            out_names.append(c)
        tmp = f + ".tmp"
        pq.write_table(pa.Table.from_arrays(cols, names=out_names), tmp)
        os.replace(tmp, f)
    if verbose:
        print(f"[rerender] passthrough: patched meta/episodes tasks + task_index stats for {split}")


def _src_task_strings(src_ds):
    """{task_index: task string} from a source dataset's meta/tasks.parquet, or {} if unreadable."""
    import pyarrow.parquet as pq          # function-local, as everywhere else in this module

    p = os.path.join(src_ds, "meta", "tasks.parquet")
    if not os.path.exists(p):
        return {}
    try:
        t = pq.read_table(p).to_pydict()
    except Exception:                                  # noqa: BLE001 - never fail a render on this
        return {}
    idx = t.get("task_index") or []
    txt = next((v for k, v in t.items() if k != "task_index"), [])
    return {int(i): str(x) for i, x in zip(idx, txt)}


def _src_task_schema(src_ds):
    """The source tasks.parquet schema, or None. Carries the pandas metadata."""
    import pyarrow.parquet as pq

    p = os.path.join(src_ds, "meta", "tasks.parquet")
    if not os.path.exists(p):
        return None
    try:
        return pq.ParquetFile(p).schema_arrow
    except Exception:                                  # noqa: BLE001 - never fail a render on this
        return None


def _write_task_table(out_dir, split, strings, schema=None, verbose=True, src_strings=()):
    """Write meta/tasks.parquet for `split` from an ordered list of task strings."""
    import json as _json

    import pyarrow as pa
    import pyarrow.parquet as pq

    meta = os.path.join(out_dir, split, "meta")
    if not os.path.isdir(meta) or not strings:
        return
    # Reuse the SOURCE schema: it carries the pandas metadata (index_columns =
    # ["__index_level_0__"]) that makes to_pandas() return the task STRING as the index, which is
    # how every consumer reads it. Writing a bare schema instead handed them a RangeIndex and the
    # string as an ordinary column -- a silent shape change, not an error.
    if schema is not None and len(schema) == 2:
        arrays = [pa.array(range(len(strings)), type=f.type) if f.name == "task_index"
                  else pa.array(strings, type=f.type) for f in schema]
        tbl = pa.Table.from_arrays(arrays, schema=schema)
    else:
        tbl = pa.table({"task_index": pa.array(range(len(strings)), type=pa.int64()),
                        "__index_level_0__": pa.array(strings, type=pa.large_string())})
    pq.write_table(tbl, os.path.join(meta, "tasks.parquet"))
    info_p = os.path.join(meta, "info.json")
    if os.path.exists(info_p):
        with open(info_p) as fh:
            info = _json.load(fh)
        info["total_tasks"] = len(strings)
        tmp = info_p + ".tmp"
        with open(tmp, "w") as fh:
            _json.dump(info, fh, indent=4)
        os.replace(tmp, info_p)
    if verbose:
        # Say whether anything actually CHANGED. When every assignable block was skipped (e.g. an
        # asset library this module's atlas maths do not describe) the rebuilt table is identical
        # to the source, and "rewrote ... after the glyph swap" reads as a swap that never
        # happened -- the kind of confident log line that hid earlier bugs here.
        changed = len([t for t in strings if t not in set(src_strings or ())])
        how = (f"{changed} changed by the glyph swap" if changed
               else "unchanged -- no block was swapped")
        print(f"[rerender] passthrough: task table rebuilt for {split} "
              f"({len(strings)} task(s), {how})")


def _uid_at_rank(rank_to_uid, rank):
    """Post-reorder per-split rank -> the render UID cam_delta's per_ep is keyed by.

    finalize() sorts each split back into uid order and renumbers 0..k-1, so the output
    episode_index is a RANK. An empty map means the caller did not supply one: fall back to
    identity, which is what the code did before this inversion existed.
    """
    return rank_to_uid[rank] if 0 <= rank < len(rank_to_uid) else rank


def _patch_data_from_source(out_dir, job, episode_records, cam_delta=None, verbose=True,
                            scene_bases=None, task_letters=False, block_colors=False,
                            _bc_palette=None, uid_of_out=None):
    """Make the output parquet identical to the SOURCE except the camera columns we re-render.

    A render-only replay cannot reproduce most recorded columns: nothing writes joint
    velocity/acceleration/effort, the contact wrench, the PD gains or the scene scalars, so the
    recorder emits zeros or unset-buffer constants; FK-derived link poses drift a few hundred
    microns; label/provenance strings and instance_id are re-derived; the aggregate `action` column
    gets filled here but is all-zero at source; and four source columns (action.is_valid,
    observation.state.profile_id, left_camera pose+intrinsics) are missing entirely because that
    camera is not spawned. Teaching the recorder to fake each one would drift out of date the next
    time the source schema changes.

    So copy every column straight from the matching source episode -- VALUES and arrow dtype, which
    also restores fixed_size_list/string in place of list/large_string. A later fix to the source
    then flows into a re-render for free instead of the two silently disagreeing.

    Kept from the re-render, because they are the point of re-rendering:
      * camera intrinsics -- FOV DR goes through set_intrinsic_matrices, which the recorder captures.
      * camera POSE -- RECOMPUTED as source_pose . delta for every randomized camera. The recorder's
        pose is wrong for both mounts: DR is applied by writing the USD LOCAL transform (the only
        thing the tiled renderer honours) and cam_data.pos_w does not reflect that write, so a
        static cam's stored pose moved 22um against a configured +-3cm, while a link-mounted cam's
        pos_w never refreshes at all and stays constant for the episode.
    Owned by the output and never copied: episode_index and index, renumbered per shard and per
    -N copy.

    STREAMING: one episode is resident at a time and the result is written incrementally, so peak
    memory is flat in problem size. Reading whole splits (which the previous action-only patch also
    did, via pd.concat) cost 2.6 GiB on a 98 MB aug13 problem and would have multiplied by
    --procs_per_gpu.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from toyblocks.domain_randomization.episodes import drop_failed_attempt_rows

    OWNED = passthrough_owned_columns(task_letters)
    src_by_split = {}
    for ds_path, succ in job["sources"]:
        split = "blocks_success" if succ else ("blocks_failure" if succ is False else "blocks")
        parts = _src_episode_index(ds_path)
        if parts:
            src_by_split[split] = parts
    if not src_by_split:
        print("[rerender] passthrough: no source data/ found; leaving recorded columns as-is")
        return

    ep_block_colors = {}        # (split, episode_index) -> {block: colour name}
    ep_block_recs = {}          # (split, episode_index) -> {block: {color,hsv,rgb,token}}
    src_task_txt = {}           # split -> {source task_index: task string}
    src_task_sch = {}           # split -> the source table's arrow schema (pandas metadata)
    for ds_path, succ in job["sources"]:
        split = "blocks_success" if succ else ("blocks_failure" if succ is False else "blocks")
        if split in src_by_split:
            src_task_txt[split] = _src_task_strings(ds_path)
            src_task_sch[split] = _src_task_schema(ds_path)

    out2src = {}
    ep_glyphs = {}
    for r in episode_records:
        _sp = r.get("split", "blocks_success")
        out2src.setdefault(_sp, {})[int(r["episode_index"])] = int(r["src_episode_index"])
        if r.get("glyphs"):
            # Keyed by (split, episode) like out2src one line above, NOT by episode alone.
            # episode_index is the PER-SPLIT output counter, so a problem that renders both
            # blocks_success and blocks_failure numbers both from 0. Keyed on the episode alone
            # the second split's mapping silently overwrote the first's, and the rebuild below
            # then stamped the OTHER split's letters into task_index/pddl_goal/glyph_map for
            # every episode index the two splits share -- at least min(n_success, n_failure)
            # episodes carrying instruction text that contradicts their own pixels.
            ep_glyphs[(_sp, int(r["episode_index"]))] = dict(r["glyphs"])
        if r.get("block_randomization"):
            # {block: {color, hsv, rgb, token}} -> {block: color}, the shape rewrite_tokens wants.
            # The record is keyed by GOAL TOKEN; rewrite_tokens wants {original: colour}, so
            # go through "from". Deriving it here keeps the inverse convention in exactly one
            # place (block_randomization_record) instead of two.
            ep_block_colors[(_sp, int(r["episode_index"]))] = {
                v["from"]: v["color"] for v in r["block_randomization"].values()
                if v.get("from") and v.get("color")}
            # The full record too: the column carries rgb, because the HSV->RGB palette lives in
            # a flags yaml that does not travel with the dataset.
            ep_block_recs[(_sp, int(r["episode_index"]))] = dict(r["block_randomization"])

    patched = skipped = 0
    added: set = set()          # camera columns synthesized because the source carried none
    # Rewritten task strings, interned per split. The glyph mapping is PER EPISODE, so one source
    # task yields as many output tasks as there are distinct mappings that used it -- the table
    # has to be rebuilt, not copied.
    new_tasks: dict = {}        # split -> {rewritten string: new index}
    ep_task_idx: dict = {}      # split -> {episode: [task_index per frame]} for the meta patch
    for split, mapping in out2src.items():
        parts = src_by_split.get(split)
        if not parts:
            continue
        ep_loc = {}                                    # episode -> (part_idx, row_start, n_rows)
        for pi, (_pf, spans, _rg) in enumerate(parts):
            for e, (s, n, n_rendered) in spans.items():
                ep_loc.setdefault(e, (pi, s, n, n_rendered))

        # cam_delta's per_ep is keyed by RENDER UID (the unit's global index), but the output
        # episode_index read below is the POST-REORDER RANK within this split: finalize() sorts
        # each split back into uid order and renumbers 0..k-1. Those agree only when the split's
        # uids are contiguous from zero -- i.e. a single split. With blocks_success AND
        # blocks_failure present the success uids are a sparse subset ({0,2,3,5} -> ranks
        # {0,1,2,3}), so looking up by rank returns ANOTHER episode's camera delta rather than
        # None, and observation.camera_pose silently describes a camera that never rendered that
        # frame. sorted(values)[rank] inverts the renumbering. Empty => identity, so any caller
        # that does not pass uid_of_out keeps today's behaviour.
        _rank_to_uid = sorted((uid_of_out or {}).get(split, {}).values())

        for pfile in sorted((Path(out_dir) / split / "data").glob("chunk-*/file-*.parquet")):
            opf = pq.ParquetFile(pfile)
            if "episode_index" not in opf.schema_arrow.names:
                continue
            oeps = np.asarray(opf.read(columns=["episode_index"])
                              .column("episode_index").to_numpy())
            # output episode -> (row_start, n_rows), in file order
            order, ok, at = [], True, 0
            for e in dict.fromkeys(oeps.tolist()):
                n_out = int((oeps == e).sum())
                loc = ep_loc.get(mapping.get(int(e), -1))
                if loc is None or loc[3] != n_out:
                    skipped += 1
                    ok = False
                    break
                order.append((int(e), at, n_out, loc))
                at += n_out
            if not ok:
                continue

            # The only output columns we keep are the cameras (+ the two we own); reading just
            # those is ~6 MiB against ~800 MiB for the whole table, so no need to stream them.
            src_schema = parts[0][0].schema_arrow
            out_names = set(opf.schema_arrow.names)
            keep = [c for c in src_schema.names
                    if c in out_names
                    and (c in OWNED or c.startswith("observation.camera_intrinsics."))]
            # Camera columns the SOURCE does not carry but THIS render can supply. An execution
            # that recorded no cameras has none at all in its parquet -- and since the output
            # is rebuilt from src_schema.names below, the re-render's own were dropped too. The
            # merged dataset then holds no record of the camera that made its pixels, and
            # annotate_target_xy silently projects goal.position through a nominal camera
            # (fx 527.36 against a real ~394, plus one fixed pose) instead of the real one.
            #   intrinsics <- the re-render's own recorder: DR sets FOV through the camera API, so
            #                 cam.data.intrinsic_matrices does track it.
            #   pose       <- the SceneConfig base composed with this episode's DR delta. NOT the
            #                 recorder's pose: that reads cam.data.pos_w, a Fabric view that never
            #                 observes the USD local-mount write DR actually applies.
            # Link-mounted cams are absent from scene_bases by construction, so only the static
            # cameras gain a pose here -- which is the one goal.position depends on.
            extra_intr = sorted(c for c in out_names
                                if c.startswith("observation.camera_intrinsics.")
                                and c not in src_schema.names)
            extra_pose = sorted((f"observation.camera_pose.camera:{cam}", cam)
                                for cam in (scene_bases or {})
                                if f"observation.camera_pose.camera:{cam}" not in src_schema.names)
            added.update(extra_intr)
            added.update(c for c, _ in extra_pose)
            okeep = (pq.read_table(pfile, columns=keep + extra_intr)
                     if (keep or extra_intr) else None)

            tmp = str(pfile) + ".tmp"
            writer = None
            try:
                for oidx, o_at, n, (pi, s_start, s_n, _s_rendered) in order:
                    spf, _spans, rg = parts[pi]
                    # the rows load_episodes rendered: failed-attempt rows dropped, renumbered
                    src_rows = drop_failed_attempt_rows(_read_src_rows(spf, rg, s_start, s_n))
                    cols, names = [], []
                    for c in src_schema.names:
                        cam = c.rsplit(".", 1)[-1].split(":")[-1]
                        has_delta = bool(cam_delta) and any(k.split(":")[-1] == cam
                                                            for k in cam_delta)
                        if okeep is not None and c in keep:
                            col = okeep.column(c).slice(o_at, n).cast(src_schema.field(c).type)
                        elif c.startswith("observation.camera_pose.") and (
                                has_delta or (scene_bases or {}).get(cam) is not None):
                            col = _patched_cam_pose(src_rows, c, _uid_at_rank(_rank_to_uid, int(oidx)), cam_delta,
                                                    src_schema.field(c).type,
                                                    scene_base=(scene_bases or {}).get(cam))
                        else:
                            col = src_rows.column(c)
                        if (task_letters or block_colors) and c == "task_index" \
                                and src_task_txt.get(split):
                            # Re-point every frame at a task string rewritten through THIS
                            # episode's mapping. Owning this column instead took the recorder's
                            # index into its own one-row "manipulation" table, discarding the
                            # real instruction text entirely.
                            from toyblocks.domain_randomization.glyphs import rewrite_tokens
                            # Every episode in the split goes through here, including any with
                            # no glyphs (empty mapping -> text unchanged). Rebuilding only SOME of
                            # them would leave the rest indexing the old table.
                            gm = ep_glyphs.get((split, int(oidx))) or {}
                            cm = ep_block_colors.get((split, int(oidx))) or {}
                            from toyblocks.domain_randomization.block_colors import (
                                rewrite_tokens as rewrite_colors)
                            tmap = src_task_txt[split]
                            interned = new_tasks.setdefault(split, {})
                            out_idx = []
                            for v in (col.to_pylist() if hasattr(col, "to_pylist") else list(col)):
                                txt = tmap.get(int(v)) if v is not None else None
                                txt = rewrite_tokens(txt, gm) if txt else (txt or "")
                                # Colours too, and in the SAME pass over the table. `task` is the
                                # column the dataloaders actually read (lerobot_wrapper's
                                # question_key defaults to it); pddl_goal is only parsed from
                                # .pddl FILES by the eval scripts. So rewriting the goal and not
                                # this would leave the one field training sees naming the colour
                                # a cuboid no longer is -- 1095 of oct06_stack3's 5478 task
                                # strings name a colour cuboid.
                                if cm and txt:
                                    txt = rewrite_colors(txt, cm)
                                if txt not in interned:
                                    interned[txt] = len(interned)
                                out_idx.append(interned[txt])
                            ep_task_idx.setdefault(split, {})[int(oidx)] = out_idx
                            col = pa.array(out_idx, type=src_schema.field(c).type)
                        if task_letters and c == "pddl_goal" and ep_glyphs.get((split, int(oidx))):
                            # Rewrite the SOURCE goal -- the recorder's is empty in a re-render,
                            # which is why owning this column produced an all-null one -- so the
                            # symbolic task names the glyphs actually rendered.
                            from toyblocks.domain_randomization.glyphs import rewrite_tokens
                            gm = ep_glyphs[(split, int(oidx))]
                            vals = col.to_pylist() if hasattr(col, "to_pylist") else list(col)
                            col = pa.array([rewrite_tokens(v, gm) if v else v for v in vals],
                                           type=src_schema.field(c).type)
                        if c == "pddl_goal" and ep_block_colors.get((split, int(oidx))):
                            # Same rule as glyphs, different vocabulary: for a cuboid the colour
                            # word IS the referring expression, so a recoloured block whose goal
                            # still says "red" is a real contradiction, not a cosmetic one.
                            from toyblocks.domain_randomization.block_colors import (
                                rewrite_tokens as rewrite_colors)
                            cm = ep_block_colors[(split, int(oidx))]
                            vals = col.to_pylist() if hasattr(col, "to_pylist") else list(col)
                            col = pa.array([rewrite_colors(v, cm) if v else v for v in vals],
                                           type=src_schema.field(c).type)
                        cols.append(col if isinstance(col, pa.ChunkedArray)
                                    else pa.chunked_array([col]))
                        names.append(c)
                    if task_letters:
                        # glyph_map cannot survive by being OWNED: this loop iterates
                        # src_schema.names, so when the source lacks the column it must be added
                        # the way camera columns a camera-less source lacks are added.
                        from toyblocks.domain_randomization.glyphs import glyph_map_json
                        _gm = ep_glyphs.get((split, int(oidx))) or {}
                        _gcol = pa.chunked_array(
                            [pa.array([glyph_map_json(_gm) if _gm else ""] * n, type=pa.string())])
                        if "glyph_map" in names:
                            # ...but the recorder DECLARES glyph_map unconditionally, so a source
                            # recorded by a current recorder already has one and the loop above
                            # copied it. Appending a second produced a parquet with a duplicate
                            # field name that pyarrow refuses to open at all:
                            #   ArrowInvalid: Multiple matches for FieldRef.Name(glyph_map)
                            # Reproduced 2026-10-13 re-rendering oct06_stack3 (source HAS the
                            # column) where sep13_play_rr (source lacks it) was unaffected -- so
                            # this only bites on sources newer than that recorder change.
                            # This episode's map is the live one; the source's is stale.
                            cols[names.index("glyph_map")] = _gcol
                        else:
                            cols.append(_gcol)
                            names.append("glyph_map")
                        added.add("glyph_map")
                    if task_letters or block_colors:
                        # ONE map for both families, keyed by the rewritten pddl name. Add-or-
                        # REPLACE from the outset: a source recorded by a current recorder already
                        # declares this column, and appending a second field makes the parquet
                        # unreadable (ArrowInvalid: Multiple matches) -- the bug the glyph branch
                        # above had to be fixed for. Do not reintroduce it here.
                        from toyblocks.domain_randomization.block_variables import (
                            block_variable_map_json)
                        _vm = block_variable_map_json(
                            ep_glyphs.get((split, int(oidx))) or None,
                            ep_block_colors.get((split, int(oidx))) or None,
                            _bc_palette)
                        _vcol = pa.chunked_array([pa.array([_vm if _vm != "{}" else ""] * n,
                                                           type=pa.string())])
                        if "block_variable_map" in names:
                            cols[names.index("block_variable_map")] = _vcol
                        else:
                            cols.append(_vcol)
                            names.append("block_variable_map")
                        added.add("block_variable_map")
                    for c in extra_intr:
                        col = okeep.column(c).slice(o_at, n)
                        cols.append(col if isinstance(col, pa.ChunkedArray)
                                    else pa.chunked_array([col]))
                        names.append(c)
                    for c, cam in extra_pose:
                        col = _patched_cam_pose(None, c, _uid_at_rank(_rank_to_uid, int(oidx)), cam_delta,
                                                pa.list_(pa.float32()),
                                                scene_base=(scene_bases or {}).get(cam),
                                                n_rows=n)
                        cols.append(pa.chunked_array([col]))
                        names.append(c)
                    batch = pa.Table.from_arrays(cols, names=names)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp, batch.schema, compression="snappy")
                    writer.write_table(batch)
                    patched += 1
            finally:
                if writer is not None:
                    writer.close()
            if writer is not None:
                os.replace(tmp, pfile)
            elif os.path.exists(tmp):
                os.remove(tmp)

    msg = f"[rerender] passthrough: copied source columns into {patched} episode(s)"
    if added:
        msg += (f"; ADDED {len(added)} camera column(s) the source lacked "
                f"({', '.join(sorted(added))}) -- without them the merged dataset has no record "
                f"of the camera that rendered it and annotation falls back to a nominal one")
    if skipped:
        msg += (f"; SKIPPED {skipped} (no source match or frame-count mismatch -- those episodes "
                f"keep the recorder's columns)")
    if verbose:
        print(msg)

    if patched:
        if task_letters or block_colors:
            # The source table names the ORIGINAL glyphs, so it cannot be copied; the RECORDER's
            # holds one generic "manipulation" row, so keeping it (what this branch used to do)
            # threw the instruction text away. Write the table rebuilt above instead.
            for split, interned in new_tasks.items():
                strings = [t for t, _ in sorted(interned.items(), key=lambda kv: kv[1])]
                _write_task_table(job["out_dir"], split, strings,
                                  schema=src_task_sch.get(split), verbose=verbose,
                                  src_strings=(src_task_txt.get(split) or {}).values())
                _patch_episode_task_meta(job["out_dir"], split, strings,
                                         ep_task_idx.get(split) or {}, verbose)
            if not new_tasks:
                _passthrough_task_table(job, src_by_split, verbose)
        else:
            _passthrough_task_table(job, src_by_split, verbose)


def _passthrough_task_table(job, src_by_split, verbose=True):
    """Carry the SOURCE's task table across with the task_index column.

    OWNED is {episode_index, index}: passthrough copies every other source column verbatim,
    task_index included. But the recorder writes its own meta/tasks.parquet holding a single
    generic 'manipulation' task, so the copied indices point at rows that do not exist -- a source
    with 3 tasks yields data referencing 0,1,2 against a 1-row table. Nothing in the render notices;
    it surfaces later as a merge failing to map task_index, or silently as frames resolving to the
    wrong task string. Since the whole point of passthrough is that the two datasets "differ only
    where the pixels differ", the task table has to come from the source as well.
    """
    import json
    import shutil

    for split, _parts in src_by_split.items():
        src_ds = next((p for p, succ in job["sources"]
                       if ("blocks_success" if succ else
                           ("blocks_failure" if succ is False else "blocks")) == split), None)
        if not src_ds:
            continue
        src_tasks = os.path.join(src_ds, "meta", "tasks.parquet")
        dst_meta = os.path.join(job["out_dir"], split, "meta")
        dst_tasks = os.path.join(dst_meta, "tasks.parquet")
        if not (os.path.exists(src_tasks) and os.path.isdir(dst_meta)):
            continue
        try:
            shutil.copyfile(src_tasks, dst_tasks)
            n_tasks = None
            src_info = os.path.join(src_ds, "meta", "info.json")
            if os.path.exists(src_info):
                with open(src_info) as fh:
                    n_tasks = json.load(fh).get("total_tasks")
            dst_info = os.path.join(dst_meta, "info.json")
            if n_tasks is not None and os.path.exists(dst_info):
                with open(dst_info) as fh:
                    info = json.load(fh)
                info["total_tasks"] = n_tasks
                tmp = dst_info + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(info, fh, indent=4)
                os.replace(tmp, dst_info)
            if verbose:
                print(f"[rerender] passthrough: carried source task table into {split} "
                      f"(total_tasks={n_tasks})")
        except OSError as e:
            print(f"[rerender] WARNING could not carry the source task table into {split}: {e}")


def _scene_static_bases(static_cams, cam_base_pose, sim):
    """Pristine SceneConfig pose of each STATIC camera, in the robot-base frame the column uses.

    Deliberately reuses the SAME tensors the renderer was placed from -- cam_base_pose, snapshotted
    pre-DR by _capture_camera_bases -- and the recorder's own world_pose_to_local, so the written
    column and the rendered image describe one camera rather than two. The only transform between
    them is world -> robot base, because the column is stored base-relative.

    Link-mounted cameras are excluded by construction: static_cams already filters them out, and a
    gripper camera's pose is not scene-determined, so composing it from the SceneConfig would be
    wrong. Those keep the existing source-based composition.

    Returns {} when anything needed is missing, which leaves every column on its previous path.

    The camera side is taken from cfg.offset (the static camera's mount in its env Xform, i.e.
    what the renderer actually places -- the same source render_params' `initial` uses), NOT
    from cam_base_pose's Fabric pos_w. Found 2026-10-09 on oct06_stack3_1000x32: in some render
    processes pos_w for env 0 still read the un-cloned prim (no env origin, default rotation), so
    every problem that process rendered wrote pose = -env_origin = (-1500, 1000, 0) at
    env_spacing 500 -- correct pixels, and goal.position NaN on every frame of 517 episodes. The
    robot base is put in the same env frame (root - env origin), so the env origin cancels.
    cam_base_pose is kept only to flag such a process.
    """
    try:
        import torch
        from toyblocks.utils.tf import world_pose_to_local
        from isaaclab.utils.math import convert_camera_frame_orientation_convention
    except Exception:                                    # noqa: BLE001
        return {}
    scene = getattr(sim, "scene", None)
    arts = getattr(scene, "articulations", None) or {}
    rob = arts.get("robot") or next(iter(arts.values()), None)
    root = getattr(getattr(rob, "data", None), "root_state_w", None)
    origins = getattr(scene, "env_origins", None)
    if root is None or origins is None:
        return {}
    base = root[:1, :7].detach().cpu().numpy().astype(np.float32).copy()
    base_w = base.copy()
    base[:, :3] -= origins[:1].detach().cpu().numpy().astype(np.float32)   # robot base in env frame
    out = {}
    for k, c in (static_cams or {}).items():
        off = getattr(getattr(c, "cfg", None), "offset", None)
        if off is None:
            continue
        q = torch.tensor([list(off.rot)], dtype=torch.float32)
        q_world = convert_camera_frame_orientation_convention(
            q, origin=getattr(off, "convention", "ros") or "ros", target="world")
        pe = np.concatenate([np.asarray([off.pos], dtype=np.float32),
                             q_world.detach().cpu().numpy().astype(np.float32)], axis=-1)
        name = k.split(":")[-1]
        out[name] = world_pose_to_local(pe, base[:, :3], base[:, 3:7]).reshape(7)
        bp = (cam_base_pose or {}).get(k)
        if bp is not None:
            pw = np.concatenate([bp[0][:1].detach().cpu().numpy().astype(np.float32),
                                 bp[1][:1].detach().cpu().numpy().astype(np.float32)], axis=-1)
            live = world_pose_to_local(pw, base_w[:, :3], base_w[:, 3:7]).reshape(7)
            if float(np.abs(live[:3] - out[name][:3]).max()) > 0.01:
                print(f"[rerender] WARNING: {k} live pos_w gives base-frame position "
                      f"{np.round(live[:3], 3).tolist()} but cfg.offset gives "
                      f"{np.round(out[name][:3], 3).tolist()}; writing the cfg.offset pose (the one "
                      f"the renderer places).", flush=True)
    return out


def _patched_cam_pose(src_rows, col, oidx, cam_delta, arrow_type, scene_base=None,
                      n_rows=None):
    """One episode's camera pose composed with its mount delta (_compose_link_cam_pose).

    The base is the SOURCE's recorded pose by default -- correct for --camera_base data, where the
    renderer is deliberately placed onto the dataset's own extrinsics.

    With --camera_base scene the renderer is placed from the SceneConfig instead, so composing on
    the source base writes a pose describing a camera that never rendered. `scene_base` is that
    pristine SceneConfig pose (the very tensor _capture_camera_bases handed the renderer, put into
    the base frame with the recorder's own world_pose_to_local), so the column becomes
    scene_base (+) the same per-episode delta the renderer applied. Found 2026-09-11: 29 source
    problems carry a bad recorded front-camera pose, and every rerender of them reproduced it
    faithfully -- correct pixels, wrong numbers -- because this composition started from the source.
    """
    import pyarrow as pa
    cam_key = col.rsplit(".", 1)[-1].split(":")[-1]
    spec = cam_delta.get(cam_key) or next(
        (v for k, v in cam_delta.items() if k.split(":")[-1] == cam_key), None)
    d = spec["per_ep"].get(oidx) if spec else None
    # n_rows is passed explicitly when the SOURCE has no such column to measure -- the
    # camera-less-source case, where this function synthesizes the column outright rather than
    # patching one.
    if n_rows is None:
        n_rows = len(src_rows.column(col))
    if scene_base is not None:
        # Constant for the episode: a STATIC camera does not move, and DR is a fixed per-episode
        # mount delta -- exactly the invariant the source-based path already relies on.
        pose = np.repeat(np.asarray(scene_base, dtype=np.float32).reshape(1, 7), n_rows, axis=0)
        if d is None:
            return pa.array(list(pose), type=arrow_type)
    elif d is None:
        return src_rows.column(col)
    else:
        pose = np.asarray(src_rows.column(col).to_pylist(), dtype=np.float32).reshape(-1, 7)
    out = _compose_link_cam_pose(pose, d["dpos"], d["dq"], spec["q_base"], spec["conv"])
    return pa.array(list(out), type=arrow_type)


def _verify_problem_outputs(out_dir, job, episode_records):
    """Check that the files this problem was supposed to produce actually exist.

    render_params records what the render INTENDED; it is written from episode_records, which are
    appended before the frames are encoded. So a problem can report completed=true with 3 episodes
    and have no data/ and no videos at all -- exactly what a scene mismatch produced on 2026-08-24,
    and what the orchestrator's own episode count caught while render_params insisted all was well.
    Trust the filesystem here.

    Returns (fatal, holes). The split matters, because the two call for opposite remedies:

      fatal -> completed=false: the problem produced NOTHING (no data/*.parquet, or no videos/
               dir at all). There is nothing to patch onto, so it must be re-rendered.
      holes -> completed=true + status=incomplete_videos: output landed but some (episode, feature)
               videos are absent. ANY shortfall is a hole, however large: the completion phase
               re-renders exactly the listed episodes, so there is never a reason to spend 45
               minutes re-rendering 256 good episodes for the sake of the bad ones.

    Video file index == the split's episode index (episodes_for_split renumbers records 0..k-1 in
    render order, and the recorder writes file-NNN at that index), so a missing index names the
    episode to re-render directly.
    """
    fatal: List[str] = []
    holes: List[Dict[str, Any]] = []
    per_split: Dict[str, int] = {}
    for rec in episode_records:
        split = rec.get("split") or "blocks_success"
        per_split[split] = per_split.get(split, 0) + 1
    for split, n_eps in per_split.items():
        root = Path(out_dir) / split
        if n_eps <= 0:
            continue
        data = root / "data"
        if not data.exists() or not any(data.rglob("*.parquet")):
            fatal.append(f"{split}: no data/*.parquet for {n_eps} recorded episode(s)")
        vdir = root / "videos"
        if not vdir.exists():
            fatal.append(f"{split}: no videos/ dir for {n_eps} recorded episode(s)")
            continue
        for feat_dir in sorted(p for p in vdir.iterdir() if p.is_dir()):
            have = set()
            for mp4 in feat_dir.rglob("*.mp4"):
                m = re.match(r"file-(\d+)\.mp4$", mp4.name)
                if m:
                    have.add(int(m.group(1)))
            missing = sorted(set(range(n_eps)) - have)
            if not missing:
                continue
            holes += [{"episode": e, "split": split, "feature": feat_dir.name,
                       "reason": "missing_on_disk"} for e in missing]
    return fatal, holes


def _read_video_defects(out_dir):
    """Defect rows the encoder appended for this problem, if any.

    Written by the encoder process (which is the only thing that knows a publish failed) rather
    than inferred later: reconstructing the list from weka afterwards means listing ~115k objects.
    """
    path = os.path.join(out_dir, "_video_defects.jsonl")
    rows = []
    try:
        with open(path) as fh:
            for ln in fh:
                ln = ln.strip()
                if ln:
                    rows.append(json.loads(ln))
    except FileNotFoundError:
        return []
    except Exception as e:
        print(f"[rerender] WARNING could not read {path}: {type(e).__name__}: {e}")
    return rows


def _write_render_params(out_dir: str, job: Dict[str, Any], flags: dict, fps: int,
                         use_physics: bool, initial: dict, episode_records: list,
                         completed: bool = True, total_episodes: Optional[int] = None,
                         verbose: bool = True, video_defects: Optional[list] = None,
                         empty_reason: Optional[str] = None) -> None:
    _dr_rp.write_render_params(out_dir, job, flags, fps, use_physics, initial, episode_records,
                              _rp_args(), completed, total_episodes, verbose,
                              video_defects=video_defects, empty_reason=empty_reason)

def _has_partial_output(out_dir: str) -> bool:
    """True if this problem's output dir exists with some content but is NOT completed.

    Distinguishes an interrupted problem (started, videos/parquet on disk, but
    render_params.completed != true) from one that was never started. Used by
    --resume to know it must clear stale partial data before re-rendering.
    """
    if not os.path.isdir(out_dir):
        return False
    if _job_completed(out_dir):
        return False
    # Any split dir, staging dir, or render_params present => it was started.
    if glob.glob(os.path.join(out_dir, "*", "meta")) or \
       glob.glob(os.path.join(out_dir, "_video_staging")):
        return True
    return False


def _count_output_episodes(out_dir: str) -> int:
    """Number of episodes already written under this problem, summed over splits.

    Counts .mp4 files under a SINGLE camera stream per split (there are multiple
    streams per episode — front/left/gripper — so counting all of them would
    inflate the total). Videos are the reliable on-disk record of episodes the
    encoder actually finished (render_params can over-count — it is written by the
    main loop per wave, ahead of the async encoder)."""
    n = 0
    for videos_dir in glob.glob(os.path.join(out_dir, "*", "videos")):
        streams = sorted(glob.glob(os.path.join(videos_dir, "*")))
        if not streams:
            continue
        # one canonical stream per split (prefer the front camera), count its files
        front = [s for s in streams if s.endswith("camera:front_camera")]
        stream = (front or streams)[0]
        n += len(glob.glob(os.path.join(stream, "chunk-*", "file-*.mp4")))
    return n


def _clean_incomplete_output(out_dir: str) -> None:
    """Remove a partial (never-finalized) problem output so it re-renders cleanly.

    Safe because the caller only invokes this when the problem is NOT completed;
    every byte here is going to be regenerated. Also clears the recorder's
    ``_video_staging`` scratch dir left behind by a crash mid-encode."""
    try:
        shutil.rmtree(out_dir, ignore_errors=True)
    except Exception as e:
        print(f"[rerender] WARNING could not clean {out_dir}: {e}")


def _count_source_episodes(dataset_path, max_episodes: Optional[int]) -> int:
    """EPISODE count for a source, capped by max_episodes.

    This used to count data parquet FILES, on the assumption that file-NNN == episode NNN. That
    held for every source written one-episode-per-file, but aug13-style sources pack a whole split
    (e.g. 250 episodes) into a single file-000.parquet. Because --episode_offset builds its global
    episode list from this function, the file count made a fragment select only episode_index 0 of
    each split -- 103 of 12800 episodes fleet-wide, silently, with every job reporting success.

    meta/info.json is authoritative and free; the parquet union is the fallback. Must stay in step
    with render_sources._split_episode_count, which the planner's item offsets index into.
    """
    ds = Path(dataset_path)
    n = None
    try:
        _ti = json.loads((ds / "meta" / "info.json").read_text()).get("total_episodes")
        if isinstance(_ti, int) and _ti > 0:
            n = _ti
    except (OSError, ValueError, TypeError):
        pass
    if n is None:
        ids = set()
        for f in sorted((ds / "data").glob("chunk-*/file-*.parquet")):
            try:
                ids.update(pd.read_parquet(f, columns=["episode_index"])["episode_index"].tolist())
            except Exception:
                pass
        n = len(ids)
    return min(n, max_episodes) if max_episodes else n


ENUM_CACHE_NAME = ".rerender_enum_cache.json"


def _enum_counts_from_cache(src: str):
    """{(problem, split): n_episodes} from the planner's cache, or None if absent/stale/unusable.

    Reads the SAME file render_sources._enum_cache_store writes, with the same schema. The
    planner already pays for this scan once per run and the counts it stores are the ones the item
    offsets index into -- so the cache is not merely faster here, it is the authoritative table.
    Recomputing risks the two drifting, which mis-slices every fragment silently.
    """
    # render_sources documents RERENDER_ENUM_CACHE=0 as bypassing the cache ENTIRELY. Honour it
    # on this side too: it is the escape hatch for a cache believed wrong, and a bypass that only
    # the planner obeys leaves the worker reading the very file the operator is trying to ignore.
    if os.environ.get("RERENDER_ENUM_CACHE", "1") in ("0", "false", "no"):
        return None
    try:
        c = json.loads((Path(src) / ENUM_CACHE_NAME).read_text())
    except (OSError, ValueError):
        return None
    if c.get("schema") != 1:
        return None
    try:
        return {(p, s): int(n) for p, s, n in c["splits"]}
    except (KeyError, TypeError, ValueError):
        return None


# Set from --episodes_file in main(): dict[(problem_name, split_str)] -> set(episode_index). When
# not None, replay_job renders ONLY the listed episodes (targeted re-render for patching a subset).
_EPISODE_FILTER = None

# True only on the --episode_offset path, where jobs are PRUNED to the problems the slice touches.
# There, a job that renders nothing means the filter disagrees with the source (the parquet-file vs
# episode_index bug did exactly that) -- fail loudly instead of finalizing an empty dataset as a
# success. Left False for --episodes_file, where untouched problems legitimately render nothing.
_FRAGMENT_PRUNED = False


def _load_episode_filter(path: Optional[str]):
    """Parse --episodes_file into {(problem, split): {episode indices}}. Accepts a TSV/CSV whose header
    has problem/split/episode columns, OR a plain list of video keys/paths (problem_* + blocks_{success,
    failure} + file-NNN.mp4 parsed out of each line)."""
    if not path:
        return None
    import re
    lines = [l.rstrip("\n") for l in open(path) if l.strip()]
    if not lines:
        return {}
    filt: Dict[Any, set] = {}
    hdr = lines[0].lower()
    if "problem" in hdr and "episode" in hdr:                 # structured TSV/CSV
        delim = "\t" if "\t" in lines[0] else ","
        cols = [c.strip().lower() for c in lines[0].split(delim)]
        pi, ei = cols.index("problem"), cols.index("episode")
        si = cols.index("split") if "split" in cols else None
        for ln in lines[1:]:
            p = ln.split(delim)
            if len(p) <= max(pi, ei):
                continue
            try:
                ep = int(p[ei])
            except ValueError:
                continue
            split = p[si].strip() if (si is not None and len(p) > si) else "blocks_success"
            filt.setdefault((p[pi].strip(), split), set()).add(ep)
    else:                                                     # list of video keys/paths
        for ln in lines:
            mp = re.search(r"(problem_[^/]+)", ln)
            me = re.search(r"file-(\d+)\.mp4", ln)
            if not (mp and me):
                continue
            ms = re.search(r"(blocks_success|blocks_failure)", ln)
            split = ms.group(1) if ms else "blocks_success"
            # An episode lives at chunk-{ep//1000}/file-{ep%1000}.mp4, so the chunk is part of the
            # index -- beaker_orchestrator decodes it the same way (base = chunk*1000). Reading only
            # file-NNN made chunk-001/file-003 mean episode 3 instead of 1003, so on any split with
            # 1000+ episodes a video-key list patched the WRONG episodes. Silently, too: the
            # empty-fragment guard is deliberately off for --episodes_file, where a problem
            # legitimately renders nothing.
            mc = re.search(r"chunk-(\d+)[/\\]", ln)
            ep = (int(mc.group(1)) * 1000 if mc else 0) + int(me.group(1))
            filt.setdefault((mp.group(1), split), set()).add(ep)
    return filt


# How often replay_job speaks from INSIDE a wave, and how finely. The wave rollout is the whole
# silent window of a render: a 32-env wave of 2500-frame episodes sits in it 60+ min, and the next
# print is the wave-boundary [progress] line. Every render watchdog keys on OUTPUT, so that one
# silence trips all three -- beaker_worker --stall_timeout (3600s, SIGKILL), beaker_orchestrator
# --worker_idle_timeout (4200s, cancel) and --no_output_timeout (14400s, exit 9). Measured
# 2026-10-09 on oct09_stack2_rg855: a HEALTHY 18-episode batch took 60.5 min -- 30s past
# --stall_timeout -- and 12 jobs were killed for rendering slowly rather than for being stuck.
#
# BOTH gates are required; they guard opposite mistakes. The seconds gate keeps the line rare
# (log volume, and worker_pulse only reads WORKER_PULSE_TAIL=40 lines). The frames gate keeps
# --worker_idle_timeout able to do its job: a render parked on encoder backpressure blocks per
# FRAME, so it advances none, stays silent, and is still correctly reaped as a corpse. A
# wall-clock-only heartbeat would instead make any render that merely creeps forward immortal.
def sort_units_into_waves(units: List[Dict[str, Any]], num_envs: int,
                          block_waves: int) -> List[List[Dict[str, Any]]]:
    """Waves of `num_envs` units, longest episodes first within each block of `block_waves` waves.

    Blocks are cut on the unit order (render uid order), so they do not depend on which subset of a
    problem a fragment renders: as long as the fragment loads its prefix through the end of its last
    block (see sort_block_cap), every block it touches is the full run's block, sorted the same way.
    Ties keep uid order. block_waves <= 0 keeps the source order.
    """
    if block_waves <= 0:
        return [units[i:i + num_envs] for i in range(0, len(units), num_envs)]
    block = num_envs * block_waves
    waves: List[List[Dict[str, Any]]] = []
    for b in range(0, len(units), block):
        blk = sorted(units[b:b + block], key=lambda u: (-int(u["length"]), int(u["episode_index"])))
        # Inside a wave, env slots stay in uid order: the tiled renderer applies the wave's first
        # active env's FOV to every env, so the leader stays the wave's lowest uid.
        waves += [sorted(blk[i:i + num_envs], key=lambda u: int(u["episode_index"]))
                  for i in range(0, len(blk), num_envs)]
    return waves


def pack_units_by_spawn_slot(units: List[Dict[str, Any]], env_slots: List[int],
                             block_waves: int) -> List[List[Dict[str, Any]]]:
    """Waves in which env ``e`` only ever renders units of spawn slot ``env_slots[e]``.

    Blocks spawn at a size fixed per env, and an episode is only drawn at its recorded size in an
    env spawned at its slot (``unit["_spawn_slot"]``). Same block cut as sort_units_into_waves
    (``len(env_slots) * block_waves`` units of uid order; the whole list when block_waves <= 0),
    so a Level-2 fragment still renders its blocks exactly as the full run does. Within a block each
    slot's units queue longest first (uid order when unsorted) and wave ``w`` takes the next unit of
    its slot for every env. An env whose slot has run out gets a placeholder -- a copy of the
    wave's first unit with ``length`` 0 and ``_render`` False -- which the render loop already
    treats as present but never stepped, recorded or counted.
    """
    num_envs = len(env_slots)
    block = num_envs * block_waves if block_waves > 0 else max(1, len(units))
    waves: List[List[Dict[str, Any]]] = []
    for b in range(0, len(units), block):
        blk = units[b:b + block]
        if block_waves > 0:
            blk = sorted(blk, key=lambda u: (-int(u["length"]), int(u["episode_index"])))
        queues: Dict[int, List[Dict[str, Any]]] = {}
        for u in blk:
            queues.setdefault(int(u["_spawn_slot"]), []).append(u)
        missing = sorted(set(queues) - set(env_slots))
        if missing:
            raise ValueError(f"spawn slot(s) {missing} have units but no env (env_slots={env_slots})")
        heads = {s: 0 for s in queues}
        while any(heads[s] < len(q) for s, q in queues.items()):
            wave: List[Optional[Dict[str, Any]]] = []
            for s in env_slots:
                q = queues.get(s, [])
                wave.append(q[heads[s]] if heads.get(s, 0) < len(q) else None)
                if s in heads and heads[s] < len(q):
                    heads[s] += 1
            lead = next(u for u in wave if u is not None)
            waves.append([u if u is not None
                          else dict(lead, length=0, _render=False, _placeholder=True)
                          for u in wave])
    return waves


def sort_block_cap(n_loaded_before: int, last_wanted: int, num_envs: int, block_waves: int) -> int:
    """Episodes to load from the LAST source of a Level-2 fragment: through its last wanted
    episode, extended to the end of that episode's sort block (units are uid-contiguous)."""
    cap = last_wanted + 1
    if block_waves <= 0:
        return cap
    block = num_envs * block_waves
    block_end = ((n_loaded_before + last_wanted) // block + 1) * block
    return max(cap, block_end - n_loaded_before)


def reorder_split_episodes(split_dir: Path, new_of_old: Dict[int, int]) -> None:
    """Renumber one finalized split in place so episode `old` becomes episode `new_of_old[old]`.

    Sorted waves save episodes in render order, but everything downstream (apply_video_patch,
    fast_patch, hole scans, resume) treats output episode i as source episode i. This puts the split
    back into source order: video files are renamed, data rows are regrouped (whole episodes per
    file, global `index` renumbered) and meta/episodes is rebuilt with the new frame ranges, data
    file pointers and video indices. Requires one episode per video file, which the recorder writes.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    n = len(new_of_old)
    if sorted(new_of_old) != list(range(n)) or sorted(new_of_old.values()) != list(range(n)):
        raise ValueError(f"reorder of {split_dir}: not a permutation of 0..{n - 1}")
    lut = np.empty(n, dtype=np.int64)
    for o, nw in new_of_old.items():
        lut[o] = nw

    mfiles = sorted((split_dir / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    dfiles = sorted((split_dir / "data").glob("chunk-*/file-*.parquet"))
    meta = pa.concat_tables([pq.read_table(f) for f in mfiles])
    data = pa.concat_tables([pq.read_table(f) for f in dfiles])
    if meta.num_rows != n:
        raise ValueError(f"reorder of {split_dir}: {meta.num_rows} episode rows, expected {n}")
    vid_keys = sorted({c[len("videos/"):-len("/chunk_index")] for c in meta.column_names
                       if c.startswith("videos/") and c.endswith("/chunk_index")})
    old_ep = meta.column("episode_index").to_numpy().astype(np.int64)
    for vk in vid_keys:                    # one episode per video file, else renaming is wrong
        fi = meta.column(f"videos/{vk}/file_index").to_numpy()
        ci = meta.column(f"videos/{vk}/chunk_index").to_numpy()
        if not np.array_equal(ci * 1000 + fi, old_ep):
            raise ValueError(f"reorder of {split_dir}: {vk} does not hold one episode per file")

    # episode metadata, in new order
    meta = meta.take(pa.array(np.argsort(lut[old_ep], kind="stable")))
    new_ep = lut[meta.column("episode_index").to_numpy().astype(np.int64)]
    lengths = meta.column("length").to_numpy().astype(np.int64)

    # data rows, regrouped by new episode (frames keep their order inside an episode)
    d_ep = data.column("episode_index").to_numpy().astype(np.int64)
    data = data.take(pa.array(np.argsort(lut[d_ep], kind="stable")))
    base = int(pq.read_table(dfiles[0], columns=["index"]).column(0)[0].as_py()) if data.num_rows else 0

    def _col(t, name, values):
        i = t.schema.get_field_index(name)
        return t.set_column(i, t.schema.field(i), pa.array(values, type=t.schema.field(i).type))

    data = _col(data, "episode_index", lut[data.column("episode_index").to_numpy().astype(np.int64)])
    data = _col(data, "index", np.arange(base, base + data.num_rows, dtype=np.int64))

    # pack whole episodes into files no larger than the recorder's largest
    target = max(pq.ParquetFile(f).metadata.num_rows for f in dfiles)
    starts = np.concatenate([[0], np.cumsum(lengths)[:-1]])
    file_of, files, cur, cur_rows = [], [], [], 0
    for k, L in enumerate(lengths):
        if cur and cur_rows + L > target:
            files.append(cur)
            cur, cur_rows = [], 0
        cur.append(k)
        cur_rows += int(L)
        file_of.append(len(files))
    files.append(cur)
    meta = _col(meta, "episode_index", new_ep)
    meta = _col(meta, "dataset_from_index", base + starts)
    meta = _col(meta, "dataset_to_index", base + starts + lengths)
    meta = _col(meta, "data/chunk_index", np.asarray(file_of) // 1000)
    meta = _col(meta, "data/file_index", np.asarray(file_of) % 1000)
    for vk in vid_keys:
        meta = _col(meta, f"videos/{vk}/chunk_index", new_ep // 1000)
        meta = _col(meta, f"videos/{vk}/file_index", new_ep % 1000)
    for c in ("meta/episodes/chunk_index", "meta/episodes/file_index"):
        if c in meta.column_names:
            meta = _col(meta, c, np.zeros(n, dtype=np.int64))

    # write the new data/ and meta/episodes/ next to the old, then swap
    for sub, tables in (("data", [data.slice(int(starts[f[0]]), int(sum(lengths[k] for k in f)))
                                  for f in files]),
                        ("meta/episodes", [meta])):
        root = split_dir / sub
        tmp = Path(f"{root}.reorder_tmp")
        shutil.rmtree(tmp, ignore_errors=True)
        for j, t in enumerate(tables):
            dst = tmp / f"chunk-{j // 1000:03d}" / f"file-{j % 1000:03d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(t, dst)
        old = Path(f"{root}.reorder_old")
        root.rename(old)
        tmp.rename(root)
        shutil.rmtree(old, ignore_errors=True)

    # videos: two-phase rename so no target overwrites a file not yet moved
    for vdir in sorted(p for p in (split_dir / "videos").iterdir() if p.is_dir()):
        moves = []
        for o, nw in new_of_old.items():
            src = vdir / f"chunk-{o // 1000:03d}" / f"file-{o % 1000:03d}.mp4"
            if src.exists():
                tmp = src.with_name(src.name + ".reorder")
                src.rename(tmp)
                moves.append((tmp, vdir / f"chunk-{nw // 1000:03d}" / f"file-{nw % 1000:03d}.mp4"))
        for tmp, dst in moves:
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp.rename(dst)


class WavePoses:
    """One wave's joint and object trajectories on the GPU, uploaded once.

    Replaces per-frame, per-env, per-object writes (each a CPU->GPU copy plus two single-env PhysX
    writes: 41-107 ms per frame at 32 envs, as much as the path-traced render itself) with one
    write per object per frame for all active envs (~2 ms). World poses use the same SE(3) as
    IsaacLabSimulator._transform_to_world_frame (env base pose composed with the local pose).
    """

    def __init__(self, sim, wave: List[Dict[str, Any]], max_t: int, device) -> None:
        from isaaclab.assets import RigidObject
        from isaaclab.utils.math import quat_apply, quat_mul

        self.n = len(wave)
        self.device = device

        def _padded(rows, width):
            out = np.zeros((max_t, len(rows), width), np.float32)
            for k, r in enumerate(rows):
                a = np.asarray(r, np.float32).reshape(len(r), width)[:max_t]
                out[:len(a), k] = a
                out[len(a):, k] = a[-1]
            return torch.as_tensor(out, device=device)

        self.arm = _padded([w["joint"] for w in wave], np.asarray(wave[0]["joint"]).shape[-1])
        self.grip = _padded([w["gripper"] for w in wave], 1)[..., 0]
        bases = []
        for i in range(self.n):
            bp = sim._get_env_base_pose(i)
            bases.append(torch.as_tensor(bp, device=device, dtype=torch.float32) if bp is not None
                         else torch.tensor([0, 0, 0, 1, 0, 0, 0], device=device, dtype=torch.float32))
        base = torch.stack(bases)
        # (rigid object, [T, H, 7] world poses, {env slot -> column}); other prim types keep the
        # per-env path (set_rigid_object_pose handles XFormPrims).
        self.objects: List[Tuple[Any, torch.Tensor, Dict[int, int]]] = []
        self.fallback: List[str] = []
        for name in sorted({n for w in wave for n in w["objects"]}):
            if name not in sim.scene.keys():
                continue
            obj = sim.scene[name]
            if not isinstance(obj, RigidObject):
                self.fallback.append(name)
                continue
            has = [i for i in range(self.n) if name in wave[i]["objects"]]
            local = _padded([wave[i]["objects"][name] for i in has], 7)
            b = base[has].unsqueeze(0).expand(max_t, -1, -1)
            pos = quat_apply(b[..., 3:].reshape(-1, 4), local[..., :3].reshape(-1, 3))
            quat = quat_mul(b[..., 3:].reshape(-1, 4), local[..., 3:].reshape(-1, 4))
            world = torch.cat([pos.reshape(max_t, len(has), 3) + b[..., :3],
                               quat.reshape(max_t, len(has), 4)], dim=-1)
            self.objects.append((obj, world, {i: k for k, i in enumerate(has)}))

    def zero_velocities(self) -> None:
        for obj, _world, slot in self.objects:
            envs = list(slot)
            obj.write_root_velocity_to_sim(torch.zeros(len(envs), 6, device=self.device),
                                           env_ids=torch.tensor(envs, device=self.device))

    def write_objects(self, t: int, active: List[int], zero_velocity: bool) -> None:
        for obj, world, slot in self.objects:
            envs = [i for i in active if i in slot]
            if not envs:
                continue
            ids = torch.tensor(envs, device=self.device)
            obj.write_root_pose_to_sim(world[t, [slot[i] for i in envs]], env_ids=ids)
            if zero_velocity:
                obj.write_root_velocity_to_sim(torch.zeros(len(envs), 6, device=self.device), env_ids=ids)


WAVE_HEARTBEAT_EVERY_S = 60.0
WAVE_HEARTBEAT_STEPS = 200    # ~this many per wave, so the frame stride scales with episode length


def plan_block_sizes(sim, job: Dict[str, Any], units: List[Dict[str, Any]]
                     ) -> Tuple[str, Optional[List[int]], Optional[List[float]]]:
    """Spawn this problem's blocks at the sizes its episodes were recorded with.

    Returns ``(status, env_slots, size_table)``. Datagen spawns each env at one slot of the PDDL
    package's size table (8 random edge lengths, 3-5 cm), records that edge as
    ``observation.state.cube_scale``, and every recorded pose is consistent with it. Rendering an
    episode in an env of another size keeps the recorded centres, so too-small blocks float (half
    the size difference above the table, the full difference per stacked level) and too-large ones
    sink into each other. Each unit gets ``_spawn_slot`` (its recorded size, matched by value
    against this problem's table) and pack_units_by_spawn_slot keeps it in an env of that slot.

    The env -> slot map is always the template's round-robin (env ``i`` -> slot ``i % K``), so the
    waves depend only on the units of each sort block: a Level-2 fragment renders the full run's
    waves and the map never depends on which problem the process rendered before. The scene is
    respawned only when it does not already hold this problem's table at that map (a process that
    rendered another problem first). A size with few slots' worth of envs costs extra waves.

    ``status``: ``"ok"``; ``"legacy"`` (env_slots None: no size table, no recorded cube_scale or
    --no-scene_swap -- the scene's sizes are used, with a warning); ``"bad_size"`` (a recorded
    size is not in the table -- permanent, the caller finalizes the problem empty); ``"error"``
    (the respawn failed, or num_envs cannot cover a needed slot -- left incomplete, redelivered).
    """
    name = Path(job["out_dir"]).name
    env_file = job.get("env_file")
    table = _scene_swap.env_size_table(env_file)
    if not table:
        print(f"[rerender] {name}: no block-size table in {env_file}; blocks keep the scene's "
              f"sizes", flush=True)
        return "legacy", None, None
    recorded = [float(u.get("cube_scale", 1.0)) for u in units]
    if all(abs(v - 1.0) < 1e-9 for v in recorded):
        print(f"[rerender] WARNING {name}: source records no observation.state.cube_scale; block "
              f"sizes CANNOT be matched and use the scene's (round-robin) sizes", flush=True)
        return "legacy", None, None
    if not args_cli.scene_swap:
        print(f"[rerender] WARNING {name}: --no-scene_swap, so blocks cannot be respawned at the "
              f"recorded sizes; they keep the scene's sizes", flush=True)
        return "legacy", None, None
    slots = _scene_swap.match_size_slots(table, recorded)
    bad = sorted({round(v, 5) for v, s in zip(recorded, slots) if s is None})
    if bad:
        print(f"[rerender] ERROR {name}: recorded block size(s) {bad} m are not in this problem's "
              f"size table {[round(x, 5) for x in table]} ({env_file}); it cannot be rendered at "
              f"the right sizes, so it is finalized EMPTY (not redelivered).", flush=True)
        _SCENE_SKIPPED.append(name)
        return "bad_size", None, None
    counts: Dict[int, int] = {}
    for u, s in zip(units, slots):
        u["_spawn_slot"] = int(s)
        counts[int(s)] = counts.get(int(s), 0) + 1
    env_slots = _scene_swap.round_robin_slots(int(sim.num_envs), len(table))
    missing = sorted(set(counts) - set(env_slots))
    if missing:
        print(f"[rerender] ERROR {name}: num_envs={sim.num_envs} leaves block-size slot(s) "
              f"{missing} without an env (the table has {len(table)}); rerun with --num_envs a "
              f"multiple of {len(table)}. SKIPPING this problem", flush=True)
        _SCENE_SKIPPED.append(name)
        return "error", None, None
    try:
        if _scene_swap.reconcile(sim, env_file, env_slots=env_slots,
                                 warmup=args_cli.scene_swap_warmup):
            print(f"[rerender] {name}: respawned blocks at this problem's sizes", flush=True)
    except Exception as e:                               # noqa: BLE001 - never render wrong sizes
        print(f"[rerender] ERROR {name}: could not spawn blocks at the recorded sizes "
              f"({type(e).__name__}: {e}); SKIPPING this problem", flush=True)
        _SCENE_SKIPPED.append(name)
        return "error", None, None
    per_slot = {s: env_slots.count(s) for s in sorted(counts)}
    waves = max(-(-counts[s] // per_slot[s]) for s in counts)
    print(f"[rerender] {name}: block sizes matched ({waves} wave(s) min) -- "
          + ", ".join(f"slot {s} {1000 * table[s]:.1f} mm: {counts[s]} unit(s) / {per_slot[s]} env(s)"
                      for s in sorted(counts)), flush=True)
    return "ok", env_slots, table


def replay_job(sim, job: Dict[str, Any], max_episodes: Optional[int],
               flags: dict, match_cameras: bool, progress: Optional[dict] = None,
               cam_base_pose: Optional[dict] = None,
               cam_base_K: Optional[dict] = None) -> None:
    # Unconditional, at the top: there is no module-level `time`, and main() imports it inside a
    # branch -- see tests/test_rerender_tools_no_shadowed_imports.py for why that shape matters.
    import time
    out_dir = job["out_dir"]
    # A member problem's "prefix" would be every other member's episodes of the whole group, so a
    # one-wave member would step through the group's ten waves; it takes per-fragment DR instead
    # (still salted by its own output dir, so no two members share draws).
    _invariant = bool(getattr(args_cli, "fragment_invariant", True)) and not job.get("member")
    # A member problem always renders only its own episodes: the fragment filter when there is one,
    # else (a whole-dataset run) its member list.
    _filter = _EPISODE_FILTER
    if _filter is None and job.get("member"):
        _filter = {(Path(out_dir).name, sp): set(eps)
                   for sp, eps in job["member_episodes"].items()}
    # LEVEL-2 minimal-prefix cap: a fragment needs the FULL episode prefix up to its LAST rendered
    # episode (so uids/waves/per-wave rng match a single run) but nothing beyond it (later episodes
    # don't affect earlier waves; per-episode DR is a sequential prefix so it's stable). Cap the
    # concatenation (sources in order) to that minimal prefix. Only for N==1 (N>1 interleaves splits).
    _N = max(1, int(getattr(args_cli, "renders_per_episode", 1)))
    _caps: List[Optional[int]] = [None] * len(job["sources"])
    if _invariant and _filter is not None and _N == 1:
        _wants = [_filter.get(
            (Path(out_dir).name,
             "blocks_success" if sc else ("blocks_failure" if sc is False else "blocks")), set())
            for _ds, sc in job["sources"]]
        _last = max((i for i, w in enumerate(_wants) if w), default=-1)
        for i, w in enumerate(_wants):
            _caps[i] = None if i < _last else ((max(w) + 1) if i == _last else 0)
    episodes: List[Dict[str, Any]] = []
    # Source parquet skipped as unreadable across every split of this problem. Decides, below,
    # whether "no episodes" means a dead source (skip) or a broken enumeration (fail loudly).
    _skipped_parquet: List[Path] = []
    _sort_block = max(0, int(getattr(args_cli, "wave_sort_block", 0) or 0))
    for _si, (ds_path, succ) in enumerate(job["sources"]):
        _cap = _caps[_si]
        if _cap and _sort_block:
            # Load through the end of the last wanted episode's sort block, so this fragment's
            # waves are the full run's waves (see sort_units_into_waves).
            _cap = sort_block_cap(len(episodes), _cap - 1, int(args_cli.num_envs), _sort_block)
        # With an episode filter, load the minimal prefix (Level 2) or ALL then keep requested ones,
        # so a high target index isn't dropped by a low max_episodes cap.
        # LEVEL 1 (and every member problem) keeps only the requested episodes below, so convert
        # only those: a group's parquet holds every member's episodes (~9x more than one member).
        _keep = None
        if _filter is not None and not _invariant:
            _keep = _filter.get((Path(out_dir).name,
                                 "blocks_success" if succ else
                                 ("blocks_failure" if succ is False else "blocks")), set())
        if _cap == 0:
            eps = []
        else:
            eps = load_episodes(Path(ds_path), succ,
                                _cap if _cap is not None
                                else (None if _filter is not None else max_episodes),
                                keep=_keep)
            _skipped_parquet.extend(_dr_episodes.LAST_SKIPPED_PARQUET)
        for e in eps:
            e["_render"] = True
        if _filter is not None:
            split = "blocks_success" if succ else ("blocks_failure" if succ is False else "blocks")
            want = _filter.get((Path(out_dir).name, split), set())
            n0 = len(eps)
            if _invariant:
                # LEVEL 2 (default): keep the prefix episodes so render-uids, wave grouping, per-episode
                # DR sampling AND the per-wave dome/skybox rng draws are IDENTICAL to a single run; only
                # the requested episodes are RENDERED/recorded. Non-rendered units still occupy their
                # uid slot + advance the per-wave rng -> this fragment == a slice of one big run.
                for e in eps:
                    e["_render"] = int(e["episode_index"]) in want
                print(f"[rerender]   episode-filter (invariant) {Path(out_dir).name}/{split}: "
                      f"render {sum(e['_render'] for e in eps)}/{n0} (prefix kept for DR alignment)")
            else:
                # LEVEL 1: drop non-requested episodes (per-fragment DR; smaller, NOT identical).
                eps = [e for e in eps if int(e["episode_index"]) in want]
                print(f"[rerender]   episode-filter {Path(out_dir).name}/{split}: {n0} -> {len(eps)} kept")
        print(f"[rerender]   {ds_path} -> {len(eps)} episode(s)")
        episodes.extend(eps)
    if not episodes or not any(e.get("_render") for e in episodes):
        if _skipped_parquet:
            # A problem whose every parquet is corrupt legitimately has nothing to render. Raising
            # here instead killed the whole fragment, so its item was never acked: the lease
            # expired, SQS redelivered it, and the next worker died the same way -- 64 items of the
            # reorient run wedged exactly like this, each burning an hour of a render slot before
            # the watchdog SIGKILLed it. Skip the problem and let the rest of the fragment finish.
            _reason = (f"every source parquet unreadable "
                       f"({len(_skipped_parquet)} file(s), e.g. {_skipped_parquet[0].name})")
            print(f"[rerender] {Path(out_dir).name}: nothing renderable -- {_reason}. "
                  f"Skipping this problem; NOT a fragment error.", flush=True)
            # Mark it FINISHED. beaker_worker acks an item only when bp.item_complete() is true,
            # which needs every problem to carry render_params.completed=true. Returning without
            # one left the item un-acked forever: the lease expired, SQS redelivered it, and the
            # next worker skipped it just as fast and still could not ack -- so the item count
            # could never reach the total, COMPLETION never ran, and the merge never happened.
            # There is nothing to render here ever, so "completed" is the truthful terminal state;
            # empty_reason records why it is empty.
            _write_render_params(out_dir, job, flags, 0, False, {}, [], completed=True,
                                 total_episodes=0, empty_reason=_reason)
            return
        if _FRAGMENT_PRUNED:
            raise SystemExit(
                f"[rerender] FATAL: fragment selected {Path(out_dir).name} but none of its "
                f"{len(episodes)} loaded episode(s) matched the requested indices. The episode "
                f"enumeration disagrees with the source (see _count_source_episodes). Refusing to "
                f"finalize an empty dataset as a success.")
        print(f"[rerender] no episodes to render for {out_dir}; skipping")
        return

    # -N: multiply renders per episode. Each source episode is replayed N times as a distinct
    # "render unit" -> N x the output episodes (same recorded trajectory/state+action, but freshly
    # sampled IN-BATCH randomization: per-env materials/local-lights/cameras, and — because units
    # spread across waves — different per-batch domes too). Iteration-major order so each pass over
    # the episodes gets a fresh dome/skybox per wave. render_uid is a unique per-unit key: it drives
    # the per-env randomization sampling AND equals the sequential output episode index (the recorder
    # appends in save order == unit order). src_episode_index + iter are recorded for traceability.
    N = max(1, int(getattr(args_cli, "renders_per_episode", 1)))
    units: List[Dict[str, Any]] = []
    for j in range(N):
        for e in episodes:
            u = dict(e)                          # shallow copy (trajectory arrays shared, read-only)
            u["src_episode_index"] = int(e["episode_index"])
            u["iter"] = j
            u["episode_index"] = len(units)      # render_uid == output index == sampling key
            units.append(u)
    if N > 1:
        print(f"[rerender] -N {N}: {len(episodes)} episode(s) x {N} = {len(units)} render unit(s)")
    # Spawn blocks at the recorded sizes BEFORE the DR interpreter binds to the block prims and the
    # recorder to the scene (a respawn replaces both); see plan_block_sizes.
    _size_status, _env_slots, _size_table = plan_block_sizes(sim, job, units)
    if _size_status == "bad_size":
        # Permanent: finalize it empty, as an all-unreadable source is, so the item is acked
        # instead of being redelivered into the dead-letter queue.
        _write_render_params(out_dir, job, flags, 0, False, {}, [], completed=True,
                             total_episodes=0,
                             empty_reason="recorded block size not in the problem's size table")
        return
    if _size_status == "error":
        return

    # Cameras to drive from recorded pose (skip those overridden via flags.yaml).
    # MOVING (link-mounted, e.g. gripper) cameras are ALWAYS driven from the recorded
    # pose — they otherwise lag the gripper and start at a default pose (frame-0
    # "zoomed out" bug). Static cameras (front/left) are left to the scene cfg unless
    # --match_recorded_cameras forces all of them.
    flag_cam_keys = {f"camera:{n}" for n in (flags.get("cameras") or {})}
    scene_cam_keys = [k for k, s in sim.scene.sensors.items()
                      if k.startswith("camera:") and hasattr(s, "set_world_poses")
                      and k not in flag_cam_keys and any(k in e["cam_pose"] for e in episodes)]
    moving = _moving_cameras(episodes, scene_cam_keys)
    override_cams: Dict[str, Any] = {
        k: sim.scene.sensors[k] for k in scene_cam_keys
        if k in moving or match_cameras
    }
    if override_cams:
        print(f"[rerender] driving cameras from recorded pose: {sorted(override_cams)} "
              f"(moving={sorted(moving)})")

    # --- Per-episode camera randomization + per-batch lighting (flags['randomize']) ---
    # Mix a stable per-problem salt into every RNG so the same episode_index (and wave
    # index) does not reuse identical camera/light/skybox/material draws across problems.
    # --- DR yaml interpreter (seed hub + prim setup + per-episode sampling) -> shared package.
    #     replay_job re-binds the returned locals so the render loop below is byte-identical.
    # --camera_base data re-bases the cameras onto the SOURCE's recorded extrinsics. An execution
    # that recorded no cameras has no observation.camera_pose.* in its parquet
    # and there is nothing to re-base onto -- the DR delta would be composed onto nothing and the
    # written column would describe a camera that never rendered. Fall back to the SceneConfig base,
    # which is what the renderer is actually placed from, and say so instead of failing silently.
    # Re-derived per job from the originally requested value, so one camera-less source cannot
    # pin every later problem in a multi-problem run onto 'scene'.
    _cb_requested = getattr(args_cli, "_camera_base_requested", args_cli.camera_base)
    args_cli._camera_base_requested = _cb_requested
    _src_has_cam_pose = any(e.get("cam_pose") for e in episodes)
    if _cb_requested == "data" and not _src_has_cam_pose:
        print("[rerender] WARNING: --camera_base data IGNORED for this problem -- the source "
              "dataset carries no observation.camera_pose.* columns (its execution recorded no "
              "cameras). Falling back to --camera_base scene: camera columns "
              "are written from the SceneConfig base the renderer uses, composed with this run's "
              "per-episode DR.", flush=True)
        args_cli.camera_base = "scene"
    else:
        args_cli.camera_base = _cb_requested

    _ir = _dr_interp.setup(sim, flags, job, units, episodes, override_cams, assets=_ASSETS,
                           camera_base=args_cli.camera_base, cam_base_K=cam_base_K,
                           material_baseline_revert=args_cli.material_baseline_revert,
                           indexed_seeding=(args_cli.dr_seeding
                                            if args_cli.dr_seeding in ("indexed", "hash")
                                            else False))
    episode_records = _ir.episode_records
    prob_salt = _ir.prob_salt
    all_cam_keys = _ir.all_cam_keys
    cam_ranges = _ir.cam_ranges
    rseed_base = _ir.rseed_base
    rseed = _ir.rseed
    randomize_cams = _ir.randomize_cams
    # Static (env-mounted) cameras, independent of whether DR touches them: --camera_base data
    # re-bases these onto the dataset's extrinsics even on a yaml with no `randomize:` block.
    static_cams = {k: c for k, c in sim.scene.sensors.items()
                   if k.startswith("camera:") and hasattr(c, "set_world_poses")
                   and not _is_link_mounted(c)}
    rand_link = _ir.rand_link
    rand_static = _ir.rand_static
    ep_offsets = _ir.ep_offsets
    override_cams = _ir.override_cams
    li = _ir.li
    light_range = _ir.light_range
    light_rng = _ir.light_rng
    sky = _ir.sky
    sky_format = _ir.sky_format
    sky_rot = _ir.sky_rot
    sky_int = _ir.sky_int
    sky_int_dist = _ir.sky_int_dist
    _sky_vb_raw = _ir._sky_vb_raw
    sky_vb_prob = _ir.sky_vb_prob
    sky_visible_bg = _ir.sky_visible_bg
    sky_rng = _ir.sky_rng
    sky_buckets = _ir.sky_buckets
    sky_textures = _ir.sky_textures
    sky_order = _ir.sky_order
    missing = _ir.missing
    scene_lights = _ir.scene_lights
    fill_spec = _ir.fill_spec
    fill_auto = _ir.fill_auto
    fill_range = _ir.fill_range
    fill_fixed = _ir.fill_fixed
    fill_auto_ratio = _ir.fill_auto_ratio
    fill_rng = _ir.fill_rng
    desc = _ir.desc
    local_spec = _ir.local_spec
    local_prims = _ir.local_prims
    local_rng_seed = _ir.local_rng_seed
    sun_spec = _ir.sun_spec
    sun_prim = _ir.sun_prim
    sun_rng = _ir.sun_rng
    exp_spec = _ir.exp_spec
    exp_rng = _ir.exp_rng
    env_sphere_spec = _ir.env_sphere_spec
    env_sphere_prims = _ir.env_sphere_prims
    env_sphere_seed = _ir.env_sphere_seed
    scen_spec = _ir.scen_spec
    _scen_mix = _ir._scen_mix
    scen_rng = _ir.scen_rng
    _sseed = _ir._sseed
    _w = _ir._w
    scen_p = _ir.scen_p
    object_specs = _ir.object_specs
    initial_dome = _ir.initial_dome
    initial_fill = _ir.initial_fill
    initial = _ir.initial
    n_obj_warn = _ir.n_obj_warn
    device = _ir.device
    NE = _ir.NE
    object_shader_prims = _ir.object_shader_prims
    ep_mats = _ir.ep_mats
    ep_spheres = _ir.ep_spheres
    _env_has_pool = _ir._env_has_pool
    coupled_gi = _ir.coupled_gi
    ep_local = _ir.ep_local
    art, arm_ids, drive_id, mimic_ids, mimic_signs = _resolve_joint_ids(sim)
    src_fps = FPS
    try:
        _info = json.load(open(Path(job["sources"][0][0]) / "meta" / "info.json"))
        src_fps = int(_info.get("fps", FPS))
    except Exception as e:
        print(f"[rerender] could not read source fps (using {FPS}): {e}")
    print(f"[rerender] output fps = {src_fps}")

    # Stale PARTIAL output is garbage by definition (never finalized, every byte about to be
    # regenerated), and leaving it would let the recorder append to a half-written dataset. --resume
    # already clears it during job selection; do it here too so a run WITHOUT --resume behaves
    # identically. A FINALIZED dir is left alone -- re-rendering over it is the caller's business,
    # and deleting a complete dataset is not this function's decision to make.
    if _has_partial_output(out_dir):
        print(f"[rerender] clearing partial (never finalized) output in "
              f"{Path(out_dir).name} before re-rendering", flush=True)
        _clean_incomplete_output(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    recorder = IsaacDataRecorderLeRobot(
        sim.scene, data_dir=out_dir, dataset_name="blocks",
        num_envs=NE, record_every_n_steps=1, fps=src_fps, robotiq=True,
        # --depth_format rg8 selects the RG8 packing; 12bit keeps the native gray12le codec.
        depth_codec=("rg8" if args_cli.depth_format == "rg8" else args_cli.depth_codec),
        depth_gop=args_cli.depth_gop,
        extra_depth=args_cli.extra_depth,
        extra_rgb=args_cli.extra_rgb,
        extra_rgb_gop=args_cli.extra_rgb_GOP,
        rgb_gop=args_cli.rgb_GOP,
        record_depth=not getattr(args_cli, "skip_depth", False),
        num_writers=max(1, int(getattr(args_cli, "num_writers", 1) or 1)),
        # rerender writes ONE dataset per problem dir. Sharding exists for datagen's
        # preemption-resume, where a relaunch must not overwrite a shard a previous attempt
        # finalized; rerender has no such story -- its verifier, --resume, the orchestrator's
        # completion oracle, merge and annotate all read the problem dir itself, so a sibling
        # __sNNN renders fully and then fails verification forever. Say so, rather than depending
        # on the dir being empty for next_free_shard to return shard 0.
        enable_sharding=False,
    )
    # Backstop. If the recorder still chose a shard other than out_dir, every later check in this
    # process and in the orchestrator would look somewhere the data is not. Fail here, loudly,
    # instead of rendering an item that can never verify.
    if Path(recorder.data_dir).resolve() != Path(out_dir).resolve():
        raise RuntimeError(
            f"[rerender] recorder chose {recorder.data_dir!r} instead of {out_dir!r}. rerender "
            f"requires shard 0: the output verifier, --resume, the completion oracle and the "
            f"merge all read the problem dir itself. Refusing to render into a sibling shard.")

    sim.data_recorder = recorder
    sim.execution = True

    use_physics = args_cli.physics_step
    # Render-only path advances scene time by one render interval per frame so the
    # cameras (update_period ~ 1/fps) refresh. (Physics-step path advances the sim
    # timestamp itself, which is more reliable but introduces tiny physics drift.)
    dt = sim.sim.get_physics_dt() * PHYS_STEP_PER_RENDER_INTERVAL
    print(f"[rerender] mode = {'physics-step' if use_physics else 'render-only (no physics)'}")
    # Use sim-wide pristine camera FOV bases (captured once at sim build, before any problem).
    # Re-capturing per job from live cam.data would bake the previous problem's ending FOV
    # into the base and compound across the multi-problem run. Pose DR uses cfg.offset via
    # local USD writes (not these bases).
    if randomize_cams:
        if cam_base_K:
            print(f"[rerender] using sim-wide pristine camera base FOV "
                  f"(cams={sorted(randomize_cams)}; pose via local USD mount)")
        else:
            cam_base_pose, cam_base_K = _capture_camera_bases(sim, randomize_cams, rand_static, dt)
            print(f"[rerender] WARNING: no sim-wide camera bases passed; captured per-job "
                  f"(cams={sorted(randomize_cams)}) — fine for a single problem, but FOV will "
                  f"drift if this sim is reused across problems")
    else:
        cam_base_pose, cam_base_K = {}, {}
    waves = (pack_units_by_spawn_slot(units, _env_slots, _sort_block) if _env_slots
             else sort_units_into_waves(units, NE, _sort_block))
    if _sort_block:
        _ren = sum(max(u["length"] for u in w) * len(w) for w in waves)
        _src = sum(max(u["length"] for u in units[i:i + NE]) * len(units[i:i + NE])
                   for i in range(0, len(units), NE))
        _rec = sum(u["length"] for u in units if u.get("_render", True))
        print(f"[rerender] waves sorted by length in blocks of {_sort_block}: env-frames rendered "
              f"{_ren} vs {_src} in source order ({100.0 * _rec / max(_ren, 1):.0f}% useful)")
    split_out_count = {True: 0, False: 0}  # per-split (success/failure) output-episode counter
    _uid_of_out: Dict[str, Dict[int, int]] = {}  # split -> {output index: render uid}
    for wi, wave in enumerate(waves):
        print(f"[rerender] {Path(out_dir).name}: wave {wi+1}/{len(waves)} ({len(wave)} episodes)")
        # per-wave scenario pick: the chosen bucket overrides this wave's global levers below.
        for env_idx in range(len(wave)):
            if wave[env_idx].get("_render", True):
                recorder.reset_env_state(env_idx)
        # per-env table height (recorded)
        thd = [w["table_height_delta"] for w in wave]
        if any(abs(x) > 1e-9 for x in thd):
            _set_table_height(sim, list(range(len(wave))), thd)

        _w = _dr_interp.apply_wave(_ir, sim, wave)
        if args_cli.dump_probe and not globals().get("_PROBE_WRITTEN"):
            # Diagnostic only, opt-in, once per process: what actually landed on the stage.
            try:
                import json as _json

                from toyblocks.domain_randomization.build import (dump_render_context,
                                                                   dump_render_products)
                from toyblocks.domain_randomization.materials import dump_applied_materials
                _probe = {"who": "rerender",
                          "materials": dump_applied_materials(_ir.object_specs,
                                                              _ir.object_shader_prims),
                          "carb": dump_render_context(),
                          "render_products": dump_render_products(sim)}
                with open(args_cli.dump_probe, "w") as _f:
                    _json.dump(_probe, _f, indent=1, sort_keys=True, default=str)
                print(f"[rerender] dump_probe -> {args_cli.dump_probe}")
                globals()["_PROBE_WRITTEN"] = True
            except Exception as _e:            # noqa: BLE001 -- never fail a render for a probe
                print(f"[rerender] dump_probe FAILED: {type(_e).__name__}: {_e}")
        wave_light = _w.wave_light
        wave_sky = _w.wave_sky
        wave_fill = _w.wave_fill
        eff_dome = _w.eff_dome
        wave_sun = _w.wave_sun
        wave_exp = _w.wave_exp
        wave_scen = _w.wave_scen
        wave_materials = _w.wave_materials
        wave_local = _w.wave_local
        wave_spheres = _w.wave_spheres
        # record per-episode params for render_params.json (empty when no flags applied).
        # Per-env block glyph swap: sample BEFORE the record loop so the recorded map and the
        # shader written later in the camera window are the same draw (interpreter caches it on
        # _ir and the applier reuses it). Empty dict when the flag is off.
        # Gated on the FLAG, not called unconditionally: a ToyBlocks that predates glyphs.py
        # has no such attribute, and reaching for it broke every run on main (see the pairing
        # check in __main__). Both helpers already no-op when the flag is off, so this only
        # changes which repo has to be new enough.
        _wave_glyphs = (_dr_interp.sample_wave_task_letters(_ir, wave)
                        if getattr(_ir, "task_letters", False) else {})

        # Envs this wave will actually render, in the order apply_wave_cameras sees them at
        # t == 0 -- the first of them is the one whose FOV the tiled renderer uses for all.
        _wave_active0 = [j for j in range(len(wave))
                         if wave[j]["length"] > 0 and wave[j].get("_render", True)]
        # `key` = unit's global sampling id (unique per render unit, keys all the per-env samples).
        # rec["episode_index"] = the PER-SPLIT output index (save order within blocks_success /
        # blocks_failure) so records align 1:1 with the video files; src_episode_index+iter+split
        # trace an -N augmented render back to its source episode.
        for i in range(len(wave)):
            if not wave[i].get("_render", True):
                continue                          # LEVEL 2: uid kept for DR alignment, not recorded
            key = wave[i]["episode_index"]
            succ = bool(wave[i]["success"])
            oidx = split_out_count[succ]; split_out_count[succ] += 1
            # Always record split + source index so every record is self-describing:
            # it maps 1:1 to a per-split output video (file-{episode_index} in `split`) and
            # back to its source episode (`src_episode_index`), regardless of -N. Needed for
            # split-correct provenance AND for resume reconstruction (which keys off split +
            # src). `iter` only carries meaning under -N>1 augmentation.
            rec = {"episode_index": oidx,
                   "split": "blocks_success" if succ else "blocks_failure",
                   "src_episode_index": int(wave[i]["src_episode_index"])}
            _uid_of_out.setdefault(rec["split"], {})[oidx] = int(key)
            # BATCH PLACEMENT. Everything above is scene CONTENT; this is scene POSITION, and
            # without it a record cannot be reproduced: an episode rendered in env slot i sits at
            # env_origins[i], and GI / dome IBL / neighbouring envs' lights are all sampled against
            # that placement. Two runs can therefore record identical lighting yet render
            # differently purely because the episode landed in a different slot or a differently
            # sized wave. Recording it lets rerender_from_params reproduce the exact placement
            # (and makes num_envs=1 reproduction explicit about what it is NOT reproducing).
            rec["env_slot"] = int(i)
            rec["wave"] = int(wi)   # with env_slot: the exact batch placement (rerender_from_params)
            if _env_slots:
                # The edge length this episode's blocks were spawned at (plan_block_sizes).
                rec["block_size"] = round(float(_size_table[_env_slots[i]]), 6)
            rec["wave_size"] = int(len(wave))
            rec["num_envs"] = int(NE)
            try:
                rec["env_origin"] = [round(float(v), 6)
                                     for v in sim.scene.env_origins[i].cpu().numpy().tolist()]
            except Exception:
                pass
            if N > 1:                            # -N augmentation: which of the N copies
                rec["iter"] = int(wave[i]["iter"])
            # The table height actually applied for this episode (_set_table_height, from the
            # source's observation.state.table_height_delta). It was applied but never recorded,
            # so rerender_from_params.py:292 -- which does ep.get("table_height_delta", 0.0) --
            # silently replayed a FLAT table and produced different depth for every episode whose
            # source had a non-zero delta. Additive: an existing reader ignores the new key.
            _thd = wave[i].get("table_height_delta")
            if _thd is not None:
                rec["table_height_delta"] = float(_thd)
            _gm = _wave_glyphs.get(i) or {}
            if _gm:
                # Recorded like cameras/materials: what was APPLIED for this episode. The
                # recorder rewrites pddl_goal + task through the same map at frame-build time.
                rec["glyphs"] = dict(_gm)
                recorder.set_glyph_map(_gm, env_idx=i)
            # Cuboid colours. Recorded from APPLIED, not sampled: apply_wave ran before the frame
            # loop and dropped any block whose shader it could not find, so this says what the
            # camera saw rather than what we intended.
            _bcm = (getattr(_ir, "block_colors_applied", None) or {}).get(i) or {}
            if _bcm:
                from toyblocks.domain_randomization.block_colors import block_randomization_record
                rec["block_randomization"] = block_randomization_record(
                    _bcm, getattr(_ir, "block_palette", None))
                recorder.set_cuboid_color_map(_bcm, env_idx=i,
                                              palette=getattr(_ir, "block_palette", None))
            if randomize_cams:
                offs = ep_offsets.get(key, {})
                # The wave rendered with ONE intrinsic matrix, so this episode was framed with
                # the wave leader's FOV rather than its own sample. Record what was applied.
                offs = _dr_camera.applied_wave_offsets(offs, wave, _wave_active0, ep_offsets)
                rec["cameras"] = {ck: _absolute_cam_params(initial["cameras"][ck], off)
                                  for ck, off in offs.items() if ck in initial["cameras"]}
                # Record the RAW offsets too, because "cameras" above cannot be inverted back into
                # them: _absolute_cam_params composes the delta onto initial["cameras"] (env-frame
                # position, world-convention rotation) while the RENDERER composes the same delta
                # onto cfg.offset (link-relative position, ROS-convention rotation). Those bases
                # differ -- 546 mm apart for the gripper cam, and a different quaternion convention
                # for every cam -- so the recorded absolute does not describe the rendered
                # orientation and a replay cannot reconstruct it. These are what
                # _apply_camera_local_offsets actually consumes, so rerender_from_params can replay
                # exactly instead of approximately.
                rec["camera_offsets"] = {ck: dict(off) for ck, off in offs.items()}
            elif flags.get("cameras"):
                # What was APPLIED, not what was asked for -- and in the same schema the
                # randomize branch above writes, which is what every consumer expects
                # (rerender_from_params.py: "the ABSOLUTE pose/FOV the camera was actually"
                # rendered with; also repair_scene_vfov.py, render_replay.py).
                #
                # This used to copy flags["cameras"] verbatim. That is the REQUEST: it claimed a
                # gripper_camera pose the renderer had silently dropped (fixed 2026-11-03, see
                # dr/build.apply_camera_flag_overrides), and it listed left_camera, which
                # --skip_cameras meant was never rendered at all. initial["cameras"] is read back
                # from the built scene, so it carries only cameras that exist and only values
                # that took effect. The request is still recoverable: render_params["flags"]
                # records the whole resolved yaml, cameras section included.
                rec["cameras"] = {ck: dict(v) for ck, v in (initial.get("cameras") or {}).items()}
            if wave_light is not None:
                rec["lighting_intensity"] = wave_light
            if wave_fill is not None:
                rec["fill_intensity"] = wave_fill
            if wave_sky is not None:
                rec["skybox"] = wave_sky
            if wave_exp:
                rec["exposure"] = wave_exp
            if wave_sun is not None:
                rec["sun"] = wave_sun
            if wave_scen is not None:
                rec["scenario"] = wave_scen.get("name", "?")
            ep_mat = wave_materials.get(key)
            if ep_mat:
                rec["materials"] = ep_mat
            ep_ll = wave_local.get(key)
            if ep_ll:
                rec["local_lights"] = ep_ll
            ep_sph = wave_spheres.get(key)
            if ep_sph:
                rec["env_sphere"] = ep_sph
            # Always append: every record now carries split + src_episode_index provenance
            # (plus any randomization applied this wave), giving a complete source->output map.
            episode_records.append(rec)

        max_t = max(w["length"] for w in wave)
        _poses = WavePoses(sim, wave, max_t, device)
        # blocks outside each episode's problem: a finite pose out of view, never NaN
        park_inactive_objects(sim, wave)
        if not use_physics:
            _poses.zero_velocities()      # render-only: nothing steps, so once per wave suffices
        _t_wave0 = time.time()
        _t_pose = _t_render = _t_record = 0.0
        _hb_stride = max(1, max_t // WAVE_HEARTBEAT_STEPS)
        _hb_t, _hb_next = -1, time.time() + WAVE_HEARTBEAT_EVERY_S
        for t in range(max_t):
            active = [i for i in range(len(wave))
                      if t < wave[i]["length"] and wave[i].get("_render", True)]
            if not active:
                break
            # Speak from inside the wave (see WAVE_HEARTBEAT_EVERY_S). Deliberately NOT the
            # "[progress] ... episodes N/M" shape of the boundary line below: that exact string is
            # what feeds the orchestrator's rendered_hwm (beaker_worker gates on `"[progress]" in
            # line`, beaker_orchestrator reads HEARTBEAT_EPISODES), and rendered_hwm is a
            # never-reset high-water mark gating BOTH the exit-9 give-up path and the
            # launch-cooldown reset. Frames pushed into an encoder queue are not episodes on disk
            # -- under backpressure a render sits in this loop for hours having saved none -- so
            # crediting them there would reopen the hole --no_output_timeout exists to close.
            if t - _hb_t >= _hb_stride and time.time() >= _hb_next:
                _hb_t, _hb_next = t, time.time() + WAVE_HEARTBEAT_EVERY_S
                print(f"[rerender] {Path(out_dir).name}: wave {wi+1}/{len(waves)} "
                      f"frame {t+1}/{max_t} ({100.0 * (t + 1) / max_t:.1f}%)", flush=True)
            _tp = time.time()
            eids = torch.tensor(active, device=device)

            # arm joints (from the wave's trajectories, uploaded once)
            arm = _poses.arm[t, eids]
            art.write_joint_state_to_sim(arm, torch.zeros_like(arm),
                                         joint_ids=arm_ids, env_ids=eids)
            # gripper drive + mimic
            grip = _poses.grip[t, eids].reshape(-1, 1)
            art.write_joint_state_to_sim(grip, torch.zeros_like(grip),
                                         joint_ids=drive_id, env_ids=eids)
            for jid, sign in zip(mimic_ids, mimic_signs):
                mj = grip * sign
                art.write_joint_state_to_sim(mj, torch.zeros_like(mj),
                                             joint_ids=[jid], env_ids=eids)
            if use_physics:
                # Pin actuated joints (arm + gripper drive) at the recorded pose via PD
                # targets so they hold during the physics step (no jerk toward stale targets).
                tgt = torch.cat([arm, grip], dim=1)
                art.set_joint_position_target(tgt, joint_ids=list(arm_ids) + list(drive_id),
                                              env_ids=eids)
            # objects: one batched write per rigid object for all active envs
            _poses.write_objects(t, active, zero_velocity=use_physics)
            for i in active:
                for name in _poses.fallback:
                    poses = wave[i]["objects"].get(name)
                    if poses is not None:
                        sim.set_rigid_object_pose(
                            name, torch.tensor(poses[t], device=device, dtype=torch.float32),
                            env_idx=i, local=True)
            _t_pose += time.time() - _tp

            # override cameras from recorded extrinsics/intrinsics (faithful repro)
            if override_cams:
                _override_cameras(sim, override_cams, wave, active, t, device)

            # --camera_base data with NO `randomize:` block: there is no jitter to compose, but
            # the static cameras must still be moved onto the extrinsics the dataset records --
            # otherwise the nocamrand path silently keeps using the SceneConfig mount. Written once,
            # and only when the dataset actually disagrees with the config by more than 0.1 mm, so
            # a dataset whose calibration matches (the common case) is left byte-for-byte alone.
            if t == 0 and not randomize_cams and args_cli.camera_base == "data" and static_cams:
                _arts0 = getattr(sim.scene, "articulations", None) or {}
                _rob0 = next(iter(_arts0.values()), None)
                _root0 = getattr(getattr(_rob0, "data", None), "root_state_w", None)
                if _root0 is not None:
                    sim.sim.render()
                    sim.scene.update(dt)
                    _db = _dr_camera.recorded_static_bases(static_cams, wave, active,
                                                           cam_base_pose, _root0[:, :7], device)
                    _moved = {}
                    for _ck, (_P, _Q) in (_db or {}).items():
                        _cfgp = torch.tensor(list(static_cams[_ck].cfg.offset.pos), device=device)
                        if max(float(torch.linalg.norm(_P[i] - _cfgp)) for i in _P) > 1e-4:
                            _moved[_ck] = (_P, _Q)
                    if _moved:
                        if not globals().get("_LOGGED_NODR_BASE"):
                            globals()["_LOGGED_NODR_BASE"] = True
                            print(f"[rerender] camera_base=data (no DR): re-basing "
                                  f"{sorted(_moved)} onto the recorded extrinsics")
                        _zero = {wave[i]["episode_index"]: {k: dict(ZERO_CAM_OFFSET)
                                                            for k in _moved} for i in active}
                        _apply_camera_local_offsets(static_cams, wave, active, _zero, device,
                                                    data_bases=_moved)
            # The glyph swap must not be coupled to camera DR. It used to run ONLY inside
            # apply_wave_cameras, which is reached only by the `elif randomize_cams:` arm below,
            # so a flags file with randomize_task_letters and no camera DR produced a dataset
            # whose pddl_goal/task/glyph_map all named the new letters while the pixels kept the
            # authored ones -- and logged nothing at all, because the applier (which is the only
            # thing that warns) never ran. Apply here so every branch gets it, before each
            # branch's RTX warm-up. Idempotent, so the camera arm's own call is a no-op rewrite.
            if t == 0 and getattr(_ir, "task_letters", False):
                _dr_interp.apply_wave_task_letters(_ir, wave, active)
            if use_physics:
                # One physics step via the sim's own step() path: renders all cameras
                # and records. Advancing the sim timestamp makes the time-gated tiled
                # cameras refresh reliably (no duplicate frames). Joints pinned via
                # targets; objects re-teleported each frame -> drift negligible.
                sim.step(num_steps=1, record_data=True, active_env_indices=active)
            elif randomize_cams:
                # Per-episode randomization, single render per frame:
                #  * ALL cameras (front/left static AND gripper link): LOCAL USD mount
                #    (cfg.offset ⊗ delta) once at t==0. World-pose writes do not affect the
                #    tiled renderer; local USD writes do.
                #  * FOV via set_intrinsic_matrices from pristine base_K once at t==0.
                # Warm up AFTER offsets so temporal history matches the offset framing.
                if t == 0:
                    # camera DR + its two ordering constraints (pre-render so cam.data is
                    # populated; RTX warm-up after so temporal history matches the final
                    # framing) now live together in the interpreter, so a gym wrapper gets
                    # them as one call instead of having to rediscover the ordering.
                    _dr_interp.apply_wave_cameras(
                        _ir, sim, wave, active, dt=dt,
                        camera_base=args_cli.camera_base,
                        cam_base_pose=cam_base_pose, cam_base_K=cam_base_K,
                        rtx_warmup_frames=args_cli.rtx_warmup_frames)
                _tr = time.time()
                sim.sim.render()
                sim.scene.update(dt)
                _t_render += time.time() - _tr
                _tr = time.time()
                recorder.record_step_batched(active)
                _t_record += time.time() - _tr
            else:
                # Pure render: no physics dynamics. Link-mounted cameras (e.g. the
                # gripper cam) follow their parent link in scene.update(), which is one
                # render behind the capture. That 1-frame lag is negligible mid-trajectory,
                # but on the FIRST frame of a batch the camera is still at its default pose
                # (the "zoomed-out" bug). So warm up once at t==0 only: an extra
                # render()+update() seats the cameras on the initial link poses before the
                # first capture. Every other frame uses a single render (no extra cost).
                # --rtx_warmup_frames does extra t0 passes so DLSS/denoiser converge (removes
                # frame-0 temporal warm-up artifacts, e.g. the front-cam edge lines).
                if t == 0:
                    for _ in range(max(1, args_cli.rtx_warmup_frames)):
                        sim.sim.render()
                        sim.scene.update(dt)
                _tr = time.time()
                sim.sim.render()
                sim.scene.update(dt)
                _t_render += time.time() - _tr
                _tr = time.time()
                recorder.record_step_batched(active)
                _t_record += time.time() - _tr

        _dt_wave = time.time() - _t_wave0
        print(f"[rerender] timing {Path(out_dir).name}: wave {wi+1}/{len(waves)} {max_t} steps x "
              f"{len(wave)} envs in {_dt_wave:.1f}s ({max_t / max(_dt_wave, 1e-9):.2f} steps/s) | per step: "
              f"poses {1000 * _t_pose / max_t:.1f} ms, render {1000 * _t_render / max_t:.1f} ms, "
              f"record {1000 * _t_record / max_t:.1f} ms", flush=True)

        # Save episodes in INPUT order (not finish order) so output episode_index
        # matches the source. Buffers persist until save_episode / reset_env_state.
        for i in range(len(wave)):
            if not wave[i].get("_render", True):
                continue
            recorder.write_success(bool(wave[i]["success"]), env_idx=i)
            recorder.save_episode(i)

        # Per-batch incremental checkpoint of render_params.json (locked+atomic) so the
        # sampled params survive a crash mid-problem, plus a global progress line.
        _write_render_params(out_dir, job, flags, src_fps, use_physics, initial,
                             episode_records, completed=False, total_episodes=len(episodes),
                             verbose=False)
        if progress is not None:
            progress["done"] = progress.get("done", 0) + sum(
                1 for u in wave if not u.get("_placeholder"))
            pct = 100.0 * progress["done"] / max(progress.get("total", 1), 1)
            print(f"[progress] problem {progress.get('prob_idx','?')}/{progress.get('n_probs','?')} "
                  f"'{Path(out_dir).name}' | wave {wi+1}/{len(waves)} | "
                  f"episodes {progress['done']}/{progress.get('total','?')} ({pct:.1f}%)", flush=True)

    print(f"[rerender] finalizing recorder for {out_dir} ...")
    recorder.finalize()
    # Sorted waves saved episodes in render order; put each split back into source (uid) order so
    # output episode i is source episode i again, as patching, hole scans and resume all assume.
    for _split, _m in _uid_of_out.items():
        _order = sorted(_m, key=lambda o: _m[o])
        _new_of_old = {o: r for r, o in enumerate(_order)}
        if all(o == r for o, r in _new_of_old.items()):
            continue
        _sd = Path(out_dir) / _split
        if not (_sd / "meta" / "episodes").is_dir():
            continue
        reorder_split_episodes(_sd, _new_of_old)
        for rec in episode_records:
            if rec.get("split") == _split:
                rec["episode_index"] = _new_of_old[int(rec["episode_index"])]
        # render_params renumbers each split's records by list position (episodes_for_split), so
        # the list order must follow the new indices too, or entry i describes another episode.
        episode_records.sort(key=lambda r: (r.get("split") != _split, int(r["episode_index"])))
        _dpath = os.path.join(out_dir, "_video_defects.jsonl")
        if os.path.exists(_dpath):
            _rows = _read_video_defects(out_dir)
            for _r in _rows:
                if _r.get("split") == _split and _r.get("episode") is not None:
                    _r["episode"] = _new_of_old.get(int(_r["episode"]), _r["episode"])
            with open(_dpath, "w") as _fh:
                _fh.write("".join(json.dumps(_r) + "\n" for _r in _rows))
        print(f"[rerender] {_split}: restored source episode order after length-sorted waves "
              f"({sum(o != r for o, r in _new_of_old.items())}/{len(_new_of_old)} episodes moved)", flush=True)
    # If the encoder died / had to be force-terminated (hang), its tail videos are truncated -> do NOT
    # mark this problem completed, so a --resume relaunch re-renders it instead of shipping bad videos.
    encoder_clean = not getattr(recorder, "_writer_force_terminated", False)
    if not encoder_clean:
        print(f"[rerender] WARNING encoder did not finish cleanly for {out_dir}; "
              f"marking problem INCOMPLETE (completed=false) for --resume re-render.")
    # A video the encoder could not publish leaves a HOLE: the parquet row and the other cameras
    # exist, so nothing downstream notices. That is how 35 videos shipped silently on 2026-08-24.
    # Failing the whole problem over one missing file is too blunt -- at the observed rate that
    # would have re-rendered 22 of 50 problems (~16 GPU-hours) to recover 35 videos. So by default
    # the problem stays completed but records exactly which videos are missing, and the orchestrator
    # patches just those episodes at the end. --fail_on_video_defects restores the strict behaviour.
    _output_gaps, _output_holes = _verify_problem_outputs(out_dir, job, episode_records)
    if _output_gaps:
        encoder_clean = False
        print(f"[rerender] WARNING output verification FAILED for {out_dir}; marking INCOMPLETE "
              f"(completed=false) so it is re-rendered:")
        for g in _output_gaps[:8]:
            print(f"[rerender]     {g}")
    # Union of what the encoder knew it dropped and what is actually absent on disk. The second is
    # the larger set: every hole found in allrand and textrand predates the defect bookkeeping, so
    # an encoder-only list would have called both datasets whole.
    _defect_rows = _read_video_defects(out_dir)
    _seen = {(d.get("split"), d.get("episode"), d.get("feature")) for d in _defect_rows}
    for h in _output_holes:
        if (h["split"], h["episode"], h["feature"]) not in _seen:
            _defect_rows.append(h)
    if _output_holes:
        print(f"[rerender] {len(_output_holes)} video(s) are missing on disk for {out_dir}; "
              f"recording them in render_params for a targeted patch pass")
    if _defect_rows:
        print(f"[rerender] WARNING {len(_defect_rows)} video(s) could not be published for "
              f"{out_dir}: " + ", ".join(f"ep{d['episode']}/{d['feature'].split(':')[-1]}"
                                         for d in _defect_rows[:8]))
        if args_cli.fail_on_video_defects:
            encoder_clean = False
            print("[rerender] --fail_on_video_defects: marking problem INCOMPLETE "
                  "(completed=false) for a full re-render.")
        else:
            print("[rerender] recording them in render_params (status=incomplete_videos) for a "
                  "targeted patch pass; the problem itself stays complete.")
    sim.data_recorder = None

    # Optionally replace the recorder's compact data/ with the source's original data/
    # (full per-episode files + provenance columns). Default OFF (prefer symlinks).
    # Full source passthrough subsumes the action-only patch, so don't run both (the second pass
    # would just rewrite the same rows). Falls back to the narrow patch if passthrough is disabled.
    if args_cli.passthrough_source_columns:
        # --camera_base scene: the renderer was placed from the SceneConfig, so the pose column
        # must be composed from that same base instead of the source's recorded one. Otherwise a
        # source with a bad extrinsic (29 such problems found 2026-09-11) yields correct pixels and
        # wrong numbers. --camera_base data keeps the source base, which is what it asks for.
        _scene_b = (_scene_static_bases(static_cams, cam_base_pose, sim)
                    if args_cli.camera_base == "scene" else None)
        if _scene_b:
            print(f"[rerender] camera_base=scene: writing {sorted(_scene_b)} pose column(s) from "
                  f"the SceneConfig base the renderer used, not the source's", flush=True)
        _patch_data_from_source(out_dir, job, episode_records, uid_of_out=_uid_of_out,
                                cam_delta=_cam_pose_deltas(randomize_cams, ep_offsets),
                                scene_bases=_scene_b,
                                task_letters=bool(flags.get("randomize_task_letters")),
                                block_colors=bool(flags.get("randomize_block_colors")),
                                _bc_palette=_bc_palette_for_run(flags))
    elif args_cli.patch_actions_from_source:
        _patch_actions_from_source(out_dir, job, episode_records)

    if args_cli.copy_source_data:
        for ds_path, succ in job["sources"]:
            split = "blocks_success" if succ else ("blocks_failure" if succ is False else "blocks")
            src_data = Path(ds_path) / "data"
            dst_data = Path(out_dir) / split / "data"
            if src_data.exists():
                if dst_data.exists() or dst_data.is_symlink():
                    shutil.rmtree(dst_data, ignore_errors=True)
                    if dst_data.is_symlink():
                        dst_data.unlink(missing_ok=True)
                shutil.copytree(src_data, dst_data)
                print(f"[rerender] copied source data -> {dst_data} ({len(list(src_data.rglob('*.parquet')))} files)")

    _write_render_params(out_dir, job, flags, src_fps, use_physics, initial,
                         episode_records, completed=encoder_clean, total_episodes=len(episodes),
                         video_defects=_defect_rows)


def main():
    global _COMPAT_NUM_ENVS
    _COMPAT_NUM_ENVS = args_cli.num_envs

    flags = _dr_config.load_flags(args_cli.flags) if args_cli.flags else {}
    if args_cli.flags:
        print(f"[rerender] loaded flags: {flags}")

    # randomize_task_letters REBUILDS meta/tasks.parquet: the glyph mapping is per episode, so one
    # source task becomes several output tasks and the indices are interned as episodes are seen.
    # Two processes each holding PART of one problem would intern independently and both write that
    # problem's tasks.parquet into the shared --output; the last writer wins and the other's frames
    # point at the wrong strings. Nothing fails; the dataset just carries wrong instruction text for
    # part of its episodes.
    # (Copying the source table, the non-glyph path, is safe under sharding because every worker
    # writes byte-identical content.)
    #
    # --episode_offset is NOT refused. It was, and that made the flag unusable on the fleet: the
    # Beaker worker ALWAYS passes a slice, so on 2026-10-02 all 123 workers of a 5000-episode run
    # exited here having rendered nothing. The planner's slices land on problem boundaries
    # (beaker_plan.build_items / build_items_k: episode_offset = starts[group[0]],
    # max_episodes = sum of the group's sizes), so a fragment owns every episode of every problem
    # it touches and its interning is self-consistent. That is CHECKED below once the slice has
    # been compiled against THIS process's own enumeration -- search _GLYPH_SHARD -- rather than
    # assumed here, because the argv alone cannot tell an aligned slice from a mid-problem one
    # (--rerender_extra can append a narrower --max_episodes after the planner's).
    #
    # --episodes_file stays refused, and is refused BEFORE the app launches (see __main__): a
    # patch run re-renders an arbitrary subset, renumbers episode_index to the unit's position in
    # the pruned list (see fragment_invariant), and so redraws glyphs that the already-written
    # tasks.parquet does not describe.
    if flags.get("randomize_task_letters") and args_cli.episodes_file:
        raise SystemExit(_GLYPH_EPISODES_FILE_MSG)

    # Ordered HDRI/skybox asset roots: --assets_paths (in order) then the repo default (last). The
    # resolver builds a unified basename->path map (first root that has a file wins) so a run can pull
    # skyboxes from a fast/local/shared store first and fall back to the repo default.
    global _ASSETS
    _ASSETS = _AssetPaths.with_search_paths(args_cli.assets_paths)
    print(f"[rerender] asset roots (priority order): {list(_ASSETS.hdri_dirs)}")
    print(f"[rerender] asset map: {_dr_resolve.asset_source_summary(_ASSETS.hdri_dirs)}")

    global _EPISODE_FILTER, _FRAGMENT_PRUNED
    _EPISODE_FILTER = _load_episode_filter(args_cli.episodes_file)
    if _EPISODE_FILTER is not None:
        _nf = sum(len(v) for v in _EPISODE_FILTER.values())
        print(f"[rerender] --episodes_file: rendering only {_nf} episode(s) across "
              f"{len(_EPISODE_FILTER)} problem/split(s) from {args_cli.episodes_file}")

    jobs = discover_jobs(args_cli.datasets_dir, args_cli.output, args_cli.env_file,
                         envs_dir=args_cli.envs_dir,
                         skip_failures=args_cli.skip_failure_episodes,
                         split_groups=args_cli.split_groups)
    layout = "unmerged" if any(len(j["sources"]) > 1 or j["sources"][0][1] is not None
                               for j in jobs) else "merged/flat"
    print(f"[rerender] discovered {len(jobs)} job(s) under {args_cli.datasets_dir} (layout: {layout})")
    if not jobs:
        print("[rerender] nothing to do")
        simulation_app.close()
        return

    # --- Parallel fragment selection (flat episode-level): render the GLOBAL episode slice
    #     [episode_offset, episode_offset+max_episodes) across all problems/splits, in the same
    #     canonical order every worker sees. Compiles to the {(problem,split): {ep}} filter that
    #     --episodes_file uses, then prunes jobs to the touched problems. Disjoint slices across
    #     workers tile the dataset and can all write into one --output. ---
    if args_cli.episode_offset is not None:
        _off = int(args_cli.episode_offset)
        _M = args_cli.max_episodes
        # canonical order: sorted problems x [success,failure] x ep 0..n-1
        _pairs: List[Any] = []
        _split_glob: Optional[List[Any]] = None
        if args_cli.split_groups:
            # Member problems index episodes inside their GROUP's split, so the list is not
            # range(count) per problem: take it from the planner's own enumeration (and its caches)
            # rather than rebuilding it here, which is what keeps every offset aligned.
            from render_sources import enumerate_source_episodes as _enum_src
            _split_glob = _enum_src(args_cli.datasets_dir,
                                    skip_failures=args_cli.skip_failure_episodes, split_groups=True)
        for job in jobs:
            _pname = Path(job["out_dir"]).name
            for _src, _succ in job["sources"]:
                _split = ("blocks_success" if _succ else
                          ("blocks_failure" if _succ is False else "blocks"))
                _pairs.append((_pname, _split, _src))

        # The offset is ABSOLUTE into the global list, so every problem's count is needed even
        # though this fragment renders ~10 of them. Counting them one at a time is ~1 weka read
        # each with NO output in between: on this 2382-problem source that measured ~1541 s in a
        # single process, and with 134 render processes starting at once on 2026-09-10 it
        # congested weka past the worker's --stall_timeout 3600, so every fragment was SIGKILLed
        # (rc=-9) before build_simulator ever ran. The fleet produced zero episodes for hours
        # while the orchestrator's own counters stayed clean. Prefer the planner's cache; fall
        # back to a PARALLEL scan, and print on both paths so the watchdog sees a live process.
        _cache = _enum_counts_from_cache(args_cli.datasets_dir)
        if _split_glob is not None:
            _cnt: Dict[Any, int] = {}
            for _p, _s, _ in _split_glob:
                _cnt[(_p, _s)] = _cnt.get((_p, _s), 0) + 1
            _pairs = [(_p, _s, None) for (_p, _s) in _cnt]
            _counts = list(_cnt.values())
            print(f"[rerender] --split_groups: {len(_split_glob)} episode(s) across "
                  f"{len({_p for _p, _s in _cnt})} problem(s) (members included)", flush=True)
        elif _cache is not None and all((_p, _s) in _cache for _p, _s, _ in _pairs):
            _counts = [_cache[(_p, _s)] for _p, _s, _ in _pairs]
            print(f"[rerender] episode counts: cache hit for all {len(_pairs)} problem/split(s) "
                  f"({ENUM_CACHE_NAME})", flush=True)
        else:
            import time
            from concurrent.futures import ThreadPoolExecutor
            _nw = min(32, max(1, len(_pairs)))
            print(f"[rerender] episode counts: no usable cache; scanning {len(_pairs)} "
                  f"problem/split(s) with {_nw} threads", flush=True)
            _t0 = time.time()
            with ThreadPoolExecutor(max_workers=_nw) as _ex:
                _counts = list(_ex.map(lambda t: _count_source_episodes(t[2], None), _pairs))
            print(f"[rerender] episode counts: scanned {len(_pairs)} in {time.time() - _t0:.0f}s",
                  flush=True)

        _glob: List[Any] = (_split_glob if _split_glob is not None else
                            [(_p, _s, _e)
                             for (_p, _s, _), _k in zip(_pairs, _counts)
                             for _e in range(_k)])
        _T = len(_glob)
        _end = min(_off + _M, _T) if _M is not None else _T
        _chosen = _glob[_off:_end]
        _filt: Dict[Any, set] = {}
        for _pname, _split, _e in _chosen:
            _filt.setdefault((_pname, _split), set()).add(_e)
        # _GLYPH_SHARD: with randomize_task_letters the fragment must own every episode of every
        # problem it touches. The unit is the PROBLEM DIRECTORY, not the (problem, split) pair:
        # replay_job rmtree's the whole problem dir when it finds un-finalized output, and
        # write_render_params writes into every split's meta dir of a touched problem -- so a
        # fragment holding one split of a problem still writes over the other split's metadata.
        # Requiring every split of every touched problem to be complete gives exclusive ownership.
        if flags.get("randomize_task_letters"):
            _tot = {(_p, _s): _k for (_p, _s, _), _k in zip(_pairs, _counts)}
            _tp = {_p for (_p, _s) in _filt}
            _partial = [f"{_p}/{_s}: {len(_filt.get((_p, _s), ()))} of {_k} episode(s)"
                        for (_p, _s), _k in sorted(_tot.items())
                        if _p in _tp and len(_filt.get((_p, _s), ())) != _k]
            if _partial:
                raise SystemExit(
                    "[rerender] randomize_task_letters needs PROBLEM-ALIGNED sharding, but "
                    f"--episode_offset {_off} --max_episodes {_M} selects only part of "
                    f"{len(_partial)} problem/split(s):\n  " + "\n  ".join(_partial[:10])
                    + ("\n  ..." if len(_partial) > 10 else "")
                    + "\nAnother worker holds the rest, and both would rebuild that problem's "
                      "meta/tasks.parquet from their own episodes only -- last writer wins and the "
                      "other's frames point at the wrong instruction text. Shard by problem "
                      "or drop the flag.")
        _EPISODE_FILTER = _filt
        _FRAGMENT_PRUNED = True
        _touched = {p for (p, _s) in _filt}
        jobs = [j for j in jobs if Path(j["out_dir"]).name in _touched]
        print(f"[rerender] --episode_offset {_off} --max_episodes {_M}: global episodes "
              f"[{_off},{_end}) of {_T} -> {len(_chosen)} episode(s) across {len(_filt)} "
              f"problem/split(s), {len(jobs)} problem(s)")
        if not jobs:
            print("[rerender] fragment selects no episodes; nothing to do")
            simulation_app.close()
            return

    # Resolve the env to BUILD the scene from (first job with a resolvable env_file).
    build_env_file = next((j["env_file"] for j in jobs if j.get("env_file")), None)
    if build_env_file is None:
        raise SystemExit("[rerender] No env found. For unmerged datasets put envs at "
                         "<datasets_dir>/envs/<problem>/env.py, or pass --env_file.")
    print(f"[rerender] building scene from env: {build_env_file}")

    # One Isaac scene is built and reused for all jobs. All problems in an unmerged
    # dataset normally share the same scene (same blocks), so this is correct; we warn
    # if a job's env has a different block set (its objects would be skipped).
    built_blocks = _env_block_names(build_env_file)
    task_id = load_env_and_task(build_env_file)
    print(f"[rerender] env task id: {task_id} | blocks={sorted(built_blocks) if built_blocks else '?'}")
    sim = build_simulator(task_id, args_cli.num_envs, args_cli.device, flags)
    print(f"[rerender] sim ready: num_envs={sim.num_envs} cameras={list(sim.cameras.keys())} "
          f"scene_sensors={list(sim.scene.sensors.keys())}")
    # The sizes the scene was just spawned at: this env's table, at the env's own slot map (the
    # template's round-robin). plan_block_sizes reuses them when they cover a problem, so the common
    # one-problem-per-process run never respawns (a mid-run stop/play hung ~1/3 of production
    # workers on 2026-11-02).
    _built_table = _scene_swap.env_size_table(build_env_file)
    _built_slots = getattr(sim.base_env, "env_slots", None)
    if _built_table and _built_slots is not None and len(_built_slots) == sim.num_envs:
        sim._rr_size_key = (tuple(_built_table), tuple(int(s) for s in _built_slots))

    if args_cli.dump_scene:
        dump_scene(sim)
        sim.close()
        simulation_app.close()
        return

    # Capture pristine camera pose/FOV ONCE for the whole multi-problem run. The sim is
    # reused across jobs; per-job re-capture from live state previously compounded FOV
    # (especially front_camera) across problems while render_params still logged
    # recorded_base+offset (so JSON looked in-range but videos zoomed in).
    cam_base_pose, cam_base_K = {}, {}
    all_cam_keys = [k for k, s in sim.scene.sensors.items()
                    if k.startswith("camera:") and hasattr(s, "set_world_poses")]
    _cam_ranges, _ = _resolve_cam_ranges(flags, all_cam_keys)
    # --camera_base data re-bases the STATIC cameras onto the dataset's own extrinsics, and it must
    # work on a yaml with no `randomize:` block too -- where _cam_ranges is empty and there would
    # otherwise be no pristine pose to difference against. Snapshot every static camera in that
    # case; it is two tensor clones each.
    _static_all = {k: sim.scene.sensors[k] for k in all_cam_keys
                   if not _is_link_mounted(sim.scene.sensors[k])}
    _pose_cams = _static_all if args_cli.camera_base == "data" else None
    if _cam_ranges or _pose_cams:
        _rand_cams = {k: sim.scene.sensors[k] for k in _cam_ranges}
        _rand_static = {k: c for k, c in _rand_cams.items() if not _is_link_mounted(c)}
        _dt = sim.sim.get_physics_dt() * PHYS_STEP_PER_RENDER_INTERVAL
        cam_base_pose, cam_base_K = _capture_camera_bases(sim, _rand_cams, _rand_static, _dt,
                                                          pose_cams=_pose_cams)
        print(f"[rerender] captured sim-wide pristine camera base pose/FOV "
              f"(static={sorted(_rand_static)}, link={sorted(set(_rand_cams) - set(_rand_static))}"
              f"{', pose-only=' + str(sorted(set(_pose_cams) - set(_rand_static))) if _pose_cams else ''}) "
              f"— reused for every problem to avoid cross-problem FOV/pose drift")

    match_cameras = args_cli.match_recorded_cameras

    # --resume: skip problems already finished (render_params.completed=true), and
    # for any problem with only PARTIAL output (started but never finalized — e.g. a
    # prior encoder crash), wipe that partial output so it is re-rendered cleanly from
    # scratch. Resume granularity is per-PROBLEM: completed problems are skipped whole,
    # an interrupted problem is redone whole. (Episode-level append is intentionally not
    # done — LeRobot v3 packs episodes into shared/aggregated parquet files, so appending
    # into a crashed, non-finalized dataset risks silently corrupting it.)
    jobs_to_run = []
    for job in jobs:
        out_dir = job["out_dir"]
        if args_cli.resume and _job_completed(out_dir):
            print(f"[rerender] resume: SKIP completed {Path(out_dir).name}")
            continue
        if args_cli.resume and _has_partial_output(out_dir):
            n = _count_output_episodes(out_dir)
            print(f"[rerender] resume: RE-RENDER incomplete {Path(out_dir).name} "
                  f"(discarding {n} partial episode(s) — never finalized); starting clean")
            _clean_incomplete_output(out_dir)
        jobs_to_run.append(job)

    # Progress totals across the problems we will run (episodes ~ #data parquet files x -N).
    def _job_source_eps(j, src):
        if j.get("member"):
            sp = "blocks_success" if src[1] else ("blocks_failure" if src[1] is False else "blocks")
            return len(j["member_episodes"].get(sp, ()))
        return _count_source_episodes(src[0], args_cli.max_episodes)
    total_eps = sum(_job_source_eps(j, s)
                    for j in jobs_to_run for s in j["sources"]) * max(1, args_cli.renders_per_episode)
    progress = {"done": 0, "total": total_eps, "n_probs": len(jobs_to_run), "prob_idx": 0}
    print(f"[rerender] to run: {len(jobs_to_run)}/{len(jobs)} problems, ~{total_eps} episodes total"
          + (f" ({len(jobs) - len(jobs_to_run)} skipped via --resume)" if args_cli.resume else ""))

    for pi, job in enumerate(jobs_to_run, 1):
        progress["prob_idx"] = pi
        # Reconcile the scene to THIS problem's blocks. A no-op (no timeline, no stage work) when
        # the set already matches, so a one-problem-per-worker run pays nothing. Must run before
        # replay_job: the recorder, the DR interpreter and the joint ids are all rebuilt inside it,
        # which is what keeps them consistent with the post-swap scene for free.
        if args_cli.scene_swap:
            try:
                _scene_swap.reconcile(sim, job.get("env_file"),
                                      warmup=args_cli.scene_swap_warmup)
            except Exception as e:                       # noqa: BLE001 - never render a wrong scene
                print(f"[rerender] ERROR scene swap failed for {Path(job['out_dir']).name} "
                      f"({type(e).__name__}: {e}); SKIPPING this problem", flush=True)
                _SCENE_SKIPPED.append(Path(job["out_dir"]).name)
                continue

        jb = _env_block_names(job.get("env_file"))
        built_now = _scene_swap.scene_signature(sim) if args_cli.scene_swap else built_blocks
        # Subset, not equality: after a swap the scene holds exactly this problem's blocks, and a
        # scene that is a strict SUPERSET would still render this problem correctly only if the
        # extra objects were absent -- which is why the swap deletes them rather than hiding them.
        if jb is not None and built_now is not None and not jb.issubset(built_now):
            # SKIP, do not render. One Isaac scene is built from the FIRST job's env and reused
            # for every job in this process, so a problem whose blocks differ would render into a
            # scene that lacks its objects. It used to only warn: the render then produced meta/
            # and render_params with completed=true, defects=0 and NO data/ and NO videos -- an
            # empty problem that every downstream check called complete. Writing nothing at all
            # leaves problem_complete() false, so the orchestrator redelivers it and it renders
            # correctly in an item where it is first (or alone).
            print(f"[rerender] ERROR {Path(job['out_dir']).name}: env blocks {sorted(jb)} are not "
                  f"in the built scene {sorted(built_now)} -- SKIPPING this problem rather than "
                  f"rendering it into a scene without its objects. It stays incomplete and will be "
                  f"redelivered; run it with --problems_per_item 1 (or alone) for an exact scene.",
                  flush=True)
            _SCENE_SKIPPED.append(Path(job["out_dir"]).name)
            continue
        print(f"[rerender] === problem {pi}/{len(jobs_to_run)} {Path(job['out_dir']).name}: "
              f"env={job.get('env_file')} ===", flush=True)
        try:
            replay_job(sim, job, args_cli.max_episodes, flags, match_cameras, progress,
                       cam_base_pose=cam_base_pose, cam_base_K=cam_base_K)
        except EncoderProcessError as e:
            # The video-encoder subprocess died. The current problem is left
            # non-finalized (completed=false) so a --resume relaunch skips the
            # problems already done and re-renders this one cleanly. Exit HARD
            # with a distinct code: os._exit skips Isaac's teardown (which itself
            # tends to hang), so the launcher detects death promptly and retries.
            print(f"[rerender] FATAL: video encoder died during problem {pi}/"
                  f"{len(jobs_to_run)} '{Path(job['out_dir']).name}': {e}", flush=True)
            print(f"[rerender] aborting (exit {EXIT_ENCODER_DIED}); relaunch with --resume "
                  f"to skip completed problems and re-render this one.", flush=True)
            sys.stdout.flush(); sys.stderr.flush()
            os._exit(EXIT_ENCODER_DIED)

    print(f"[rerender] all jobs done ({progress['done']}/{progress['total']} episodes); closing")
    sim.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
