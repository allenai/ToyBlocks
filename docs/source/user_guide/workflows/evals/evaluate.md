# `evaluate.py` Reference

[`scripts/evaluate.py`](https://github.com/allenai/blocksuite/blob/main/scripts/evaluate.py)
is the eval worker: `launch_eval.py` runs one per eval GPU, each pulling episodes from the
episode queue. You can also run it by hand, for example to debug one worker against a
policy server you started yourself. This page covers its action sources, how to run it
against a server, the observation transforms, its flags and its outputs. For the full
pipeline, see [Policy Evaluation](../evaluations.md).

`evaluate.py` parses the `Args` dataclass in
[`src/blocksuite/eval/args.py`](https://github.com/allenai/blocksuite/blob/main/src/blocksuite/eval/args.py)
with `tyro`, and hands it to `blocksuite.eval.run_eval.evaluate`. `Args` also inherits the
Isaac launcher options of
[`BaseArgs`](https://github.com/allenai/blocksuite/blob/main/src/blocksuite/cli/base_args.py)
(`--device` (default `cuda:0`), `--headless` (on for eval), `--enable_cameras`,
`--kit_args`, `--rendering_mode`) and the nested env-config overrides `--env-cfg.<field>`
(for example `--env-cfg.robot-id droid_franka`). Flags accept `-` or `_` between words,
and boolean flags are turned off with `--no-<flag>`. Isaac starts before the arguments
are parsed, so even `--help` launches it.

Before the run starts, `evaluate.py` also:

- adds `--enable_cameras` when `--obs_mode` contains `rgb`;
- snaps `--num_envs` to a value the task accepts (`reconcile_num_envs_with_task`), since
  autogen block tasks bake a fixed number of cube-size slots;
- with `--dataset_path`, defaults the eval `robot_id` and `scene_preset_id` to the
  dataset's recorded provenance, warning on a mismatching override and failing when the
  provenance is missing and no override was given (see
  [Dataset scene provenance](launch_eval.md#dataset-scene-provenance)).

## Action sources

`evaluate.py` picks where actions come from based on its flags:

| Source | Selected by | Use for |
|---|---|---|
| Remote policy server | `--policy_host` (+ `--policy_port`, default `8000`) | Evaluating the served ManiFlow / Molmo2 policies, or any model behind the `BasePolicy` interface. |
| Random actions | `--random_actions` | Sanity checks: uniform in `[-1, 1]`. |
| Zero actions | none of the above | Sanity checks of task spawning and metrics. |
| Dataset replay | `--replay` with `--queue_url`, `--dataset_path` and `--num_envs 1` | Playing back each queued episode's recorded actions and reporting how far the simulation diverges from the recording. |

`--obs_mode state` raises when combined with `--policy_host` (see the flag table below).
`--random_actions` raises when combined with `--policy_host`, and `--replay` raises when
combined with either.

### Random and zero actions

```bash
python scripts/evaluate.py --task Blocksuite-Lift-v0 \
    --random_actions --num_envs 4 --num_eval_steps 200
```

`--random_actions` samples every action component uniformly from `[-1, 1]`; with no
action source at all, every step gets a zero action. Both are handy for confirming that a
task spawns correctly and that recording and metrics work before plugging in a policy.

### Dataset replay

Replay is easiest through the launcher, which also starts the queue:

```bash
python scripts/launch_eval.py --replay \
    --dataset-path data/datasets/my_dataset --episode-indices "0"
```

See [Dataset Replay](replay.md) for how replay works, its metrics and its requirements.

## Running against a policy server by hand

Running the server and the eval worker in two terminals is the easiest way to debug
either side with breakpoints.

**1. Start a policy server** (terminal 1),
[`serve_maniflow_policy`](https://github.com/allenai/blocksuite/blob/main/src/blocksuite/serving/serve_maniflow_policy.py).
`--hl oracle` serves the ManiFlow LL alone (the client supplies the high-level commands);
`--hl model` serves Molmo2 + ManiFlow:

```bash
export H_VLA_PATH=$PWD/maniflow-tb PYTHONPATH=$PYTHONPATH:$PWD/maniflow-tb

# ManiFlow, with the high-level oracle on the client
python -m blocksuite.serving.serve_maniflow_policy --port 8000 --hl oracle \
    --checkpoint data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
    --config data/checkpoints/my_maniflow_run/config.yaml \
    --train_config maniflow-tb/examples/Franka/train_maniflow_multimodal.yaml

# Molmo2 high level + ManiFlow low level
python -m blocksuite.serving.serve_maniflow_policy --port 8000 --hl model \
    --checkpoint data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
    --hl_checkpoint data/checkpoints/my_molmo2_run \
    --config data/checkpoints/my_maniflow_run/config.yaml \
    --train_config maniflow-tb/examples/Franka/train_maniflow_multimodal.yaml
```

**2. Run the eval worker** (terminal 2). Prompts and per-episode setups come from the
episode queue (`--queue_url`, started with
`python scripts/serve_episode_queue.py --dataset-path <root> --port 8001`) and the
dataset (`--dataset_path`), and the task must already be generated (see
[Policy Evaluation](../evaluations.md)):

```bash
python scripts/evaluate.py \
    --task <task-slug>-v0 \
    --policy_host localhost --policy_port 8000 \
    --queue_url http://localhost:8001/next --dataset_path data/datasets/my_dataset \
    --control_mode arm_joint_gripper_binary \
    --obs_mode rgb+state --obs_transform maniflow_blocks_table \
    --action_horizon 25 --sync_action_chunks --num_envs 2 \
    --use_oracle --save_lerobot
```

Drop `--use_oracle` against the `--hl model` server; `launch_eval.py` passes
`--hl_from_model` there instead.

**Split-terminal scripts.** `scripts/eval/serve_local_server.sh` (terminal 1) and
`scripts/eval/serve_local_client.sh` (terminal 2) wrap these steps and match
`launch_eval.py`'s ManiFlow and Molmo2 inputs. Both run in the foreground with plain
`python`, so a `breakpoint()` in the server, the eval client or an obs transform is hit
interactively (`launch_eval.py` backgrounds everything with redirected output). Activate
the venv first:

```bash
# Terminal 1: policy server (POLICY_NAME=maniflow serves --hl oracle)
MODEL_PATH=data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
HL_MODEL_PATH=data/checkpoints/my_molmo2_run \
    bash scripts/eval/serve_local_server.sh

# Terminal 2: task generation + episode queue + evaluate.py
DATASET_PATH=data/datasets/my_dataset EPISODE_INDICES=0,3 \
    bash scripts/eval/serve_local_client.sh
```

Their configuration (policy, checkpoints, dataset, ports, `num_envs` cap) lives once in
`scripts/eval/_common.sh`, sourced by both; override any variable inline. `POLICY_NAME`
defaults to `molmo2_maniflow` (`--hl model`). `MODEL_PATH` (and `HL_MODEL_PATH` for
`molmo2_maniflow`) and `DATASET_PATH` (or `PDDL_PROBLEM_FILE`, to generate the task from
a PDDL problem instead) have no defaults, and the scripts exit with a message when one is
missing. The client waits for the server's port, then either runs one batched pass sized
to the queue (`PER_EPISODE=0`, the default with `--hl oracle`) or one task generation,
queue and single-environment `evaluate.py` per episode (`PER_EPISODE=1`, the default with
`--hl model`). Results go under `runs/` (`OUT_PATH`). Note that both scripts first kill
your own stale servers, queues and `evaluate.py` processes.

Key flags:

- `--obs_transform`: how observations are converted for the policy (see
  [Observation transforms](#observation-transforms)).
- `--unbatched`: call the server once per environment instead of once per step with a
  batch of all environments, for servers that add their own batch dimension.
  `launch_eval.py` always runs batched.
- `--action_horizon N`: the server returns chunks of actions and one is applied per
  step; `--sync_action_chunks` puts all environments on the same chunk boundary so the
  server is queried once per chunk (the pipeline always sets it).
- `--save_lerobot`: record the rollouts as a LeRobot dataset, including the raw policy
  actions.

A policy that returns 9-dim actions (7 arm joints, gripper, `pd_mode`) switches the
controller gains through its trailing `pd_mode` channel; no flag is needed. The channel
drives the same PD-mode controller as replay, and is stripped before `env.step`. Depth
images are added to the observations automatically when the served checkpoint expects
them.

Any model can be served by implementing the small `BasePolicy` interface; see
[Setting up a new policy server](../imitation_learning/new_policy_server.md).

## Observation transforms

Registered in
[`src/blocksuite/serving/obs_transforms/__init__.py`](https://github.com/allenai/blocksuite/blob/main/src/blocksuite/serving/obs_transforms/__init__.py)
and selected with `--obs_transform` (default `none`, which sends the raw observations):

| Name | Use for |
|---|---|
| `maniflow` | ManiFlow Franka policies. |
| `maniflow_blocks_table` | Same as `maniflow`, mapping the blocks-table cameras (`front_camera` / `gripper_camera`) onto the policy's `front_rgb` / `gripper_rgb`. The default of `launch_eval.py`. |

The ManiFlow `agent_pos` width is the served checkpoint's, not a flag: the server
advertises `state_dim` in its metadata, and the client builds the state from
`STATE_DIM_LAYOUTS` (`obs_transforms/maniflow.py`): 8 (7 arm + 1 gripper), 9 (+ `pd_mode`)
or 19 (+ 7 joint velocities and 3 EE xyz). Without metadata it defaults to 19; any other
width raises.

These flags apply to the ManiFlow transforms only, and raise with any other transform:

| Flag | Default | Description |
|---|---|---|
| `--center_crop_ratio` | none | After the resize to 224, centre-crop RGB, depth and `target_heatmap` to `int(224 * ratio)` and resize back (inference counterpart of `random_crop_ratio`, e.g. `0.95`). |
| `--depth_aug` | none | Live depth augmentation before the policy sees depth; `ejbl` matches HVLA's `augment_depth`. |
| `--enable_crop` | off | Build `target_front_crop_rgb` on the client from the native front camera and target pixels. Needs a checkpoint with `use_target_front_crop >= 0`. |

To add a transform, subclass `BaseObsTransform`, override `transform` (and optionally
`action_postprocess`), and register it in `_lazy_registry`.

## Flags

The tables below list the user-facing flags with their defaults. Every flag's docstring in
`args.py` has the full details.

### Run and environment

| Flag | Default | Description |
|---|---|---|
| `--task` | `Blocksuite-Lift-v0` | Gymnasium task id. Queue evals use the autogen `<slug>-v0` task generated from the dataset. |
| `--num_envs` | `4` | Parallel environments (snapped to the task's slot count). |
| `--num_eval_steps` | `200` | Maximum environment steps. Ignored under `--queue_url`, where the run ends when the queue is exhausted. |
| `--num_episodes` | none | Stop after this many finished episodes; with `--num_eval_steps`, whichever comes first. |
| `--obs_mode` | `state` | `state`, `rgb` or `rgb+state`. A `depth` component is added automatically when the served checkpoint takes depth. **Not combinable with `--policy_host`:** `state` flattens the observation into one tensor, while the server protocol needs a dict of named keys, so the pair is rejected up front. Use `rgb+state` (add `--obs_transform none` to send the raw keys). |
| `--control_mode` | env default | Control mode override, e.g. `arm_joint_gripper_binary` (needed for absolute joint targets: replay, `--dataset_path` runs, `--sync_action_chunks`, `--wave_resets`, `--ll_oracle`, play-mode checkpoints). |
| `--seed` | none | Random seed; no fixed seed when unset. |
| `--deterministic_cuda` | off | Deterministic cuDNN plus PhysX enhanced determinism and a single solver partition; slower, and still not bit-for-bit across hardware. |
| `--exp_name` | auto | Name of the run. |
| `--env_spacing` | env default | Grid spacing (m) between parallel envs. |

### Policy server

| Flag | Default | Description |
|---|---|---|
| `--policy_host`, `--policy_port` | none, `8000` | WebSocket policy server. Without a host, no server is contacted (zero actions, unless `--random_actions` or `--replay`). |
| `--action_key` | `actions` | Key of the server response that holds the actions. |
| `--unbatched` | off | One server call per environment instead of one batched call per step. |
| `--obs_transform` | `none` | See [Observation transforms](#observation-transforms). |
| `--action_horizon` | `0` | Actions per server chunk, one applied per step; `0` disables chunking. |
| `--sync_action_chunks` | off | Batched chunked evals: one shared chunk boundary for all environments, so the server is queried once per horizon. An environment whose episode ends mid-chunk holds its reset pose, hidden from the episode clock and the recording, until the boundary. |

### Queue and dataset

| Flag | Default | Description |
|---|---|---|
| `--queue_url` | none | Episode-queue URL (e.g. `http://localhost:8001/next`) the per-episode setups come from. |
| `--dataset_path` | none | LeRobot dataset root: reconciles robot and scene with its provenance, overlays its targets on plots, and enables the dataset PD-switch settle. Pass the dataset the queue serves. |
| `--expert_timeout_scale` | `0` | End each episode after this multiple of its expert demonstration's duration (needs `--queue_url` and `--dataset_path`; `0` disables). Such episodes end as truncations, so they keep their full metrics and TAMPER tags them `timeout`. |
| `--use_calibrated_cameras` | on | Use the fixed sim2real + FR3 gripper camera intrinsics instead of the dataset's. Disables the two options below. |
| `--ingest_camera_intrinsics` | on | Apply each episode's recorded camera intrinsics (needs `--no-use_calibrated_cameras`). |
| `--use_render_params` | off | On a rerendered dataset, reproduce each episode's recorded camera pose, HDRI, materials and lights (needs `--no-use_calibrated_cameras`). See also `--render_replay_assets_paths` (extra HDRI roots), `--render_replay_warmup_frames` (default `16`) and `--wave_resets`. |
| `--replay_block_identity` | on | On glyph/colour-rerendered datasets, paint each episode's recorded block letters and colours and prompt with the rendered goal. |
| `--verify_scene` / `--verify_scene_strict` | off | After the first queue reset, compare the scene with the recording and write `scene_verification.json`; strict aborts on hard mismatches. |

### High-level conditioning

| Flag | Default | Description |
|---|---|---|
| `--use_oracle` | off | Drive `task_name` and `target_heatmap` from the client-side symbolic oracle instead of a learned HL. Needs `--policy_host`, a ManiFlow transform and a PDDL problem per episode: `--pddl_problem_file`, or the queue (each setup's `pddl_problem_file`, else a problem rebuilt from its `pddl_goal` and recorded high-level state). The oracle re-plans from the live state (up to 3 times) when the next step's preconditions no longer hold. |
| `--pddl_problem_file` | none | One PDDL problem for every environment; a queue setup's own problem overrides it. |
| `--async_hl`, `--async_hl_call_scale` | off, `0` | Molmo2 HL with a play-mode LL: query the HL only at episode start and when the LL ends a skill; optionally cap HL calls at a multiple of the expert's. |
| `--ll_oracle`, `--ll_oracle_hl_call_scale` | off, `3.0` | Run Molmo2's commands with the vecgen motion-planned skills instead of the served LL (needs `--action_horizon 0`). |
| `--go_home` | off | With `--async_hl` or `--ll_oracle`: drive the arm home after a place, flip or rotate, before the next HL query. |
| `--post_place_lift` | `0` | Play-mode LL: lift the end effector this many metres after a release before the next LL query. |
| `--oracle_comparison` | off | Also run a client-side oracle (not fed to the LL) and show it next to the model in the HL overlay video. |
| `--hl_update_interval_ms`, `--hl_update_interval_jitter_ms`, `--estimated_hl_inference_time_ms` | `0` | Simulated HL latency: update interval K (with jitter) and inference time M. Must match training for checkpoints that use the previous front frame; see [HL latency](launch_eval.md#hl-latency). |

`--hl_from_model` is set by `launch_eval.py` for `--policy molmo2_maniflow`; it is not
meant to be passed by hand.

### Episode end and scoring

| Flag | Default | Description |
|---|---|---|
| `--episode_end` | `goal` | `goal` ends on the goal predicate (oracle runs: plan completion), `horizon` runs to `episode_length_s`, `ll_done` ends on a play-mode LL's -1. A stuck oracle plan and a learned HL's `done` end the episode in every mode. |
| `--collision` | `end` | A non-target block moving more than 2 cm or 15 deg fails the episode; `end` also ends it, `penalize` lets it run, `off` ignores collisions. |
| `--goal_ori_yaw_tol_deg` | `40.0` | Yaw tolerance of the goal's orientation atoms when judging goal success. |

A place-on-table sub-goal counts only within 25 front-camera pixels of the HL's table
point (a fixed tolerance, not a flag).

### Recording and video

| Flag | Default | Description |
|---|---|---|
| `--output_dir` | `runs/<run_name>` | Where everything is written. |
| `--save_lerobot` | off | Record the rollouts as a LeRobot dataset (scene data and raw policy actions), with the datagen recorder; see [LeRobot recording](https://github.com/allenai/blocksuite/blob/main/docs/lerobot_recording.md). |
| `--lerobot_dataset_name` | run name | Name of that dataset. |
| `--num_shards` | `8` | Parallel LeRobot writer processes, stitched into one `<name>_success` / `<name>_failure` pair at the end. |
| `--video_codec` | `libx264` | LeRobot RGB codec: `libx264` or `libsvtav1`. |
| `--video_fps` | control rate | Frame rate of saved videos and LeRobot datasets. |
| `--save_sensor_obs_video` | off | One MP4 per finished episode from the RGB observations (needs `rgb` in `--obs_mode`). |
| `--video_env_ids`, `--save_video_all_envs` | `(0,)`, off | Which envs the videos record. |
| `--video_camera` | all, tiled | Camera used for videos. |
| `--drop_cameras` | `left_camera` | Scene cameras not spawned for a policy eval; pass it with no names to keep every camera. |
| `--save_hl_overlay_video`, `--hl_overlay_camera` | off, `front_camera` | With `--save_lerobot` and an HL, also record the frame with the HL heatmap and text drawn on it. |
| `--save_hl_latency_video` | off | With `--save_sensor_obs_video` and HL conditioning, a side-by-side of the fresh HL view and the aged one the LL acted on. |
| `--save_expert_video` | off | Extract each episode's expert demonstration video for the W&B episodes table (set by `launch_eval.py --wandb`). |

### Metrics

`--track_metrics`, `--track_joint_metrics` and `--track_ee_metrics` are on by default;
`--save_step_metrics` (per-step CSVs) is off. Contact-force tracking is an env-config
option (`--env-cfg.track-collisions`). See [Metrics](metrics.md) for every metric and
output format.

### Replay

`--replay` and `--replay_clip_stats`; see [Dataset Replay](replay.md).

### TAMPER

`--tamper_output_dir` captures the terminal sim state of every unsuccessful episode as a
vecgen env package, so new expert demonstrations can be planned from those states.
`--tamper_capture_successes`, `--tamper_max_states` and `--tamper_source_env_path` refine
it. See [TAMPER](https://github.com/allenai/blocksuite/blob/main/docs/tamper.md).

### W&B

`--wandb` mirrors throughput, per-episode metrics and videos to a W&B run (needs
`WANDB_API_KEY`). `--wandb_project` (default `blocksuite-eval`), `--wandb_entity`,
`--wandb_group` (none by default), `--wandb_name`, `--wandb_job_type`, `--wandb_tags` and
`--wandb_log_interval_seconds` configure it. Pipeline runs leave it off and merge all
workers into one run instead (see [`launch_eval.py` outputs](launch_eval.md#outputs)).

### Debugging

| Flag | Default | Description |
|---|---|---|
| `--debug` | off | t=0 sim-vs-training comparison, per-episode seed logs, and per-step action prints. |
| `--save_diagnostic_plots` | off | Live-rollout diagnostics for the env `video_env_ids` names: policy-input panels in `policy_input_debug/`, HL staleness plot, and server-side LL / HL input traces. The per-step comparison plots are offline: run `scripts/plot_diagnostics_from_eval_run.py --eval-dir <output_dir>` on a `--save_lerobot` run. |
| `--save_first_batch_data` | off | On step 0, have the (ManiFlow) server dump the model-ready batch to `first_batch_data/` for comparison with training. |
| `--enable_viewer` | off | Launch the viser 3D viewer. |

**Policy-input verification** (`--save_diagnostic_plots`) answers "is the policy being
fed what it was trained on?" rather than "did it succeed". For every environment at the
first reset, then for the followed video environment at each episode start, it renders
two PNGs into `policy_input_debug/` from the live obs transform's own output:

- per depth camera: the raw sim depth, the depth the model receives (after
  `process_depth`, optional depth augmentation, point-map Z masking and the centre crop),
  the episode's recorded frame-0 depth, their signed difference and both histograms. A
  constant offset is a calibration or units mismatch; structured residue is a scene
  mismatch.
- the state panel (`*_state.png`): each `agent_pos` dimension against the `[q01, q99]`
  band the server normalizes it into. Dimensions drawn in red are outside the training
  band.

The panels are capability-gated: a run with no policy server, no ManiFlow transform, or
a server that advertises no normalization bands prints why it skipped them and carries
on. On its first inference the server also logs, per depth camera, how much of the
back-projected point map its clamp saturates (`ptmap saturation ...`); high saturation
means the scene sits outside the checkpoint's point-map bounds even when both depth
columns look healthy. The server-side traces (`ll_policy_input_trace/`, and
`hl_policy_input_trace/` for Molmo2) are rendered inside the policy server just before
the model call, one PNG per handover, so they show the tensors the model actually
receives.

## Outputs

Everything is written under `--output_dir`:

| Path | When written |
|---|---|
| `eval_results.json` | Every run: arguments, git checkout, per-episode values and aggregates. |
| `eval_episodes.json` | Every run: one record per episode (prompt, setup, metrics, goal progress, video paths). |
| `step_metrics/env{idx}_episode_NNNN.csv` | `--save_step_metrics` |
| `sensor_obs_videos/` | `--save_sensor_obs_video` |
| `hl_latency_videos/` | `--save_hl_latency_video` |
| `source_episode_videos/` | `--save_expert_video` |
| LeRobot dataset | `--save_lerobot` |
| `policy_input_debug/`, `first_batch_data/`, `scene_verification.json` | The debugging and verification flags above |

See [Metrics](metrics.md) for the fields of the two JSON files. Aggregated results are also
printed at the end of every run:

```text
============================================================
  Evaluation complete  |  episodes: 50  |  time: 312.4s
============================================================
metric                     mean       std       min       max    count
------------------------------------------------------------
return                   0.7854    0.4108    0.0000    1.0000       50
success_once             0.7800    0.4143    0.0000    1.0000       50
...
```
