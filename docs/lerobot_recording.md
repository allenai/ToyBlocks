# LeRobot dataset recording

blocksuite has one LeRobot recorder,
[`IsaacDataRecorderLeRobot`](../src/blocksuite/vecgen/recording/lerobot_recorder.py) in
`blocksuite.vecgen.recording`. It is a port of TAMP's `tmp3.DataRecorder` recorder (the one
TAMP stage 3 uses) and needs nothing from TAMP. Eval (`scripts/evaluate.py --save_lerobot`)
records with it, so an eval rollout and a datagen episode have the same schema, the same row
layout and the same video formats.

## Row layout

Each row holds an observed state and the action taken *in* it:

    row t:  observation.* = s_t,  action.* = a_{t+1}  (drives s_t -> s_{t+1})
    row N:  observation.* = s_N,  action.* = 0, action.is_valid = False

Recording samples the sim after a step, so a sample carries s_k beside the command that produced
it; the recorder holds each frame one record call and moves the next call's action block onto it.
The PD-gain columns (`observation.state.current_*`) and `action.pd_mode` move with the action.
The recorder supports per-step gain switching. Current datagen uses one uniform gain profile,
so within a dataset these columns stay constant. Every dataset's `meta/info.json` carries `blocksuite_row_layout`.

Eval envs auto-reset inside `env.step`, so the sample taken on an episode's last step is already
the next episode's s_0. `record_step_batched(reset_env_ids=...)` holds that sample as the next
episode's row 0 and the finished episode's last held frame becomes its terminal row;
`capture_reset_frames` records the first episode's s_0 after `env.reset()`.

## Output layout

    <data_dir>/<name>_success/     LeRobot v3 dataset of the successful episodes
    <data_dir>/<name>_failure/     ... and of the failed ones
        data/chunk-000/file-000.parquet
        meta/{info.json, stats.json, tasks.parquet, episodes/, recording_config.json, provenance.json}
        videos/observation.images.<cam>/chunk-{ep // 1000:03d}/file-{ep % 1000:03d}.mp4
        videos/observation.depth.<cam>/chunk-.../file-....mp4

One video file per episode and camera. RGB is encoded with the `rgb_codec` constructor argument.
Its default, `libsvtav1`, is what datagen uses. Eval passes `libx264` (`--video_codec`), because
an eval run keeps hundreds of streams open at once and x264 needs far less native memory per
stream: measured 5–14 MiB, against 27–57 MiB for SVT-AV1.
Depth is lossless 12-bit millimetres (`gray12le` HEVC, GOP 2, so a training reader decodes at
most one extra frame per random access). The rerender pipeline's `rg8_55_224` depth is a
separate, lossy stream: 224×224, clipped at 1,023 mm, 5+5-bit RG packing. The recorder can add
it alongside the main depth with the `extra_depth` argument, but eval does not set that. Eval
writes under `<output_dir>/lerobot_dataset/` and records,
per episode, `output_lerobot_dataset_dir` / `output_lerobot_episode_index` in
`eval_episodes.json`.

## Schema

The datagen schema: robot state (`observation.state.*`: joints, gripper, EE pose, wrench, gains,
torques), actions (`action.joint_target`, `action.relative_joint_target`, `action.ee_target` =
FK of the joint target, `action.relative_ee_target`, `action.gripper_position` binarised,
`action.gripper_raw`, `action.pd_mode`, `action`), poses of every object
(`observation.state.object_pose.*`), robot link (`robot.poses.*`) and camera
(`observation.camera_pose.*`, `observation.camera_intrinsics.*`), scene profile
(`cube_scale`, `table_height_delta`, `table_friction`, `friction_properties`, `profile_id`),
symbolic actions/state, `pddl_goal`, `task`, `glyph_map`, `block_variable_map`,
`source_problem`, `instance_id`, `is_episode_successful`, `error`.

Optional columns (constructor flags, off for datagen, on for eval):

| Flag | Columns |
|---|---|
| `action_dim` | `action.policy`, the raw policy action (staged with `set_pending_actions_batched`) |
| `record_hl_model` | `action.hl_model_output`, `action.hl_prev_action`, `action.hl_input_prompt`, `action.hl_done` (`set_hl_model_outputs_batched`) |
| `query_physx_properties` | `observation.state.object_mass`, `table_mass`; friction columns from the live PhysX materials |

`set_pending_pd_modes_batched` overrides the stiffness-derived `action.pd_mode` for a step.
Eval computes `action.ee_target` with
[`ArticulationEEFK`](../src/blocksuite/vecgen/recording/articulation_fk.py) (the sim robot's own
kinematics, checked against the observed EE pose); datagen passes its cuRobo FK.

## Throughput and memory

What recording costs the simulation thread is one batched snapshot per record call: robot state
copied to the host in one transfer, camera images converted to their recorded dtypes and copied
*on the GPU*. Everything else runs elsewhere:

- **Frame builder thread.** Copies the camera snapshot to the host on a side CUDA stream (it
  waits only for the snapshot, never for later simulation work), builds the rows and hands them
  to the writers.
- **Shared-memory frames.** Camera images go to their writer through a shared-memory slot
  arena (`_shm_slots.py`); only a descriptor is queued, so the images are never pickled in the
  simulation process. Falls back to the queue when `/dev/shm` is too small.
- **Writers.** `num_writers` encoder subprocesses (eval: `--num_shards`, default 8); env `i`
  goes to writer `i // ceil(N / K)`. Each stages a private dataset pair that `finalize` stitches
  into the published pair, so the layout does not depend on `K`. A writer encodes on a pool of
  `cores / K` threads, one codec thread per stream.
- **Small encoder processes.** They fork from a forkserver that preloads only the encoder
  module and never re-import the caller's script (which launches Isaac): ~0.35-0.65 GB each.
- **Bounded RAM.** Frames in flight may take `TMP3_RECORDER_INFLIGHT_GB` (default
  min(4 GiB, RAM / 4)), split across the writers; with an arena, half of a writer's share is the
  arena. Past the budget, frames are spilled to node-local disk (`BLOCKSUITE_RECORDER_SPILL_DIR`,
  else the video staging root, else `/tmp`; never network storage) and read back in order. The caller only
  waits when that disk is short of space (`BLOCKSUITE_RECORDER_SPILL_MIN_FREE_GB`, default 20).
  The spill is a burst buffer: an encoder that is slower than the simulation for good fills it.
- **Encode cost.** Depth (lossless 12-bit, GOP 2) is the expensive stream; x265 runs it at
  `ultrafast`, which changes speed and memory, not the decoded values (640x360: 273 fps per
  core and <1 MiB per stream vs 52 fps / 85 MiB at x265's default). x264 RGB takes ~14 MiB per
  open stream at 640x480 against SVT-AV1's ~57 MiB.

Measured on one RTX 5090 box (32 cores, 60 GB), stub policy, 2 cameras at 640x360, 10 s
episodes, 8 writers:

| 64 envs | wall | lowest available RAM | caller thread per record call |
|---|---|---|---|
| no recording | 100 s | 36.7 GB | - |
| previous eval recorder | 137 s | 28.3 GB | - |
| this recorder | 115 s | 30.6 GB | 4 ms |

At finalize the recorder prints the caller-thread cost (`[recorder] caller thread: ...`) and any
waits on the encoder budget.

## Eval flags

| Flag | Default | |
|---|---|---|
| `--save_lerobot` | off | record |
| `--lerobot_dataset_name` | run name | `<name>_success` / `<name>_failure` |
| `--num_shards` | 8 | encoder subprocesses |
| `--video_codec` | `libx264` | RGB codec (`libx264` / `libsvtav1`) |

## Tests

`tests/vecgen/test_recorder_rows.py` (row layout, auto-reset rows, captured s_0, staged extras,
discards, writer routing and the episode index map, spilling) and
`tests/vecgen/test_recorder_stitch.py` (stitching writer datasets) run without Isaac.
