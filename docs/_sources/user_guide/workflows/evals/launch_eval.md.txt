# `launch_eval.py` Reference

[`scripts/launch_eval.py`](https://github.com/allenai/ToyBlocks/blob/main/scripts/launch_eval.py)
is the recommended entry point for evaluation. It does not run anything itself: it turns
its flags into the environment variables of
[`scripts/eval/run_eval_pipeline.sh`](https://github.com/allenai/ToyBlocks/blob/main/scripts/eval/run_eval_pipeline.sh),
the one pipeline that starts the policy server, generates the gym task from the dataset,
serves the episode queue, runs the eval workers and merges the results (see
[Policy Evaluation](../evaluations.md)). This page lists every flag it accepts.

Everything runs as local subprocesses, from your working tree as it is:

- **GPU layout.** GPU 0 runs the policy server and GPUs 1..N run the N eval workers
  (GPUs 0..N-1 under `--replay`, which has no server). With fewer GPUs than that, the
  workers share the visible GPUs round-robin; on a one-GPU workstation the single
  worker shares GPU 0 with the policy server.
- **`--dry-run`** prints the `env VAR=... bash scripts/eval/run_eval_pipeline.sh`
  invocation instead of running it (`--output` writes it to a file), so you can inspect,
  edit and rerun it by hand.
- **Submodules.** Policy evals need the `maniflow-tb` (package `hvla`) and `molmo2`
  (package `olmo`) submodules, installed by the `maniflow` extra (see
  [Installation and setup](../installation_and_setup_instructions.md)). The pipeline
  refuses to start if either is missing, or if `maniflow-tb` is not checked out at the
  commit this checkout pins; fix it with `git submodule update --init maniflow-tb molmo2`.
  `--replay` needs neither.

The gym task is generated from `--dataset-path`: its slug is the dataset directory's
basename with `-` replaced by `_` (gym id `<slug>-v0`), unless `--task-name` overrides it.

Flags are spelled with hyphens (a few also accept an underscore alias), and on/off flags
marked "on" below are turned off with `--no-<flag>`. Only the flags on this page reach
`evaluate.py`; there is no pass-through for other [`evaluate.py` flags](evaluate.md).

## Run profiles

By default a run uses the default profile (the full queue on one eval GPU);
`--smoke-test` switches to a small validation profile. Each value can still be
overridden by its own flag:

| Setting | Default | `--smoke-test` | Flag |
|---|---|---|---|
| Queue size | unlimited | 10 episodes | `--queue-limit` |
| Queue log level | `INFO` | `DEBUG` | `--queue-log-level` |
| Parallel envs per eval GPU | 50 | 2 | `--num-envs` |
| Eval GPUs | 1 | 1 | `--num-eval-gpus` |
| Episode timeout (sim seconds) | 30.0 | 3.0 | `--episode-length-s` |
| Output subdirectory | `evals` | `queue_test` | `--out-subdir` |

Pass `--num-eval-gpus N` to spread the queue over N eval workers (`--policy maniflow`
and `--replay` only; see the limits below).

## Per-policy defaults

`--policy` (or `--replay`) sets these defaults; the matching flags override them:

| Setting | `molmo2_maniflow` (default) | `maniflow` | `--replay` |
|---|---|---|---|
| High level / low level | Molmo2 / ManiFlow (`--ll-oracle`: Molmo2 / motion-planned skills) | symbolic oracle / ManiFlow | recorded actions |
| Policy server | `serve_maniflow_policy --hl model` | `serve_maniflow_policy --hl oracle` | none |
| `--obs-transform` | `maniflow_blocks_table` | `maniflow_blocks_table` | none |
| `--action-horizon` | 25 (0 with `--ll-oracle`) | 25 | 0 |
| `--action-mode` | auto-detected from `config.yaml` | auto-detected | none |
| `--robot-id`, `--scene-preset` | the dataset's recorded provenance | same | same |
| `--center-crop-ratio` | 0.95 | 0.95 | none |
| `--use-oracle` | not applied (Molmo2 is the HL) | on | not applied |

`molmo2_maniflow` needs both `--model-path` (the low-level ManiFlow, also required with
`--ll-oracle`) and `--hl-model-path` (the Molmo2 high level). `maniflow` needs
`--model-path`. `--replay` takes no checkpoint.

**Limits of `molmo2_maniflow`.** Its server accepts a single eval-client connection and
pre-registers 128 HL-latency slots, so the launcher rejects `--num-eval-gpus` above 1
and `--num-envs` above 128. The launcher currently applies this check under `--replay`
too while `--policy` is left at its default, so pass `--policy maniflow` together with
`--replay --num-eval-gpus N`.

## The episode queue

Every eval episode starts from a recorded episode of `--dataset-path`. The pipeline starts
[`scripts/serve_episode_queue.py`](https://github.com/allenai/ToyBlocks/blob/main/scripts/serve_episode_queue.py),
a small HTTP server (port 8001) that loads one setup per dataset episode at startup and
hands them out one at a time. All eval workers pull from this one queue, so each episode
runs exactly once across all GPUs:

```text
LeRobot dataset ──> serve_episode_queue.py ──GET /next──> evaluate.py (worker 1)
                         │                  ──GET /next──> evaluate.py (worker N)
                         └──GET /size──> run_eval_pipeline.sh (sizing)
```

**What a setup holds.** `load_setups` in `utils/episode_dataset.py` builds one setup per
episode from the LeRobot parquet columns. Among other things:

- the episode's `pddl_goal`, and its `hl_prompt` when one is recorded;
- the robot's starting joint and gripper positions, and every block's pose;
- the table height or pose, the camera intrinsics, the block size (`cube_scale`) and friction;
- the expert episode's length and videos, and the frame it starts from (`init_frame_index`).

A trimmed example:

```json
{
  "episode_id": "ep_000001",
  "pddl_goal": "(and (on block_b block_a))",
  "robot_joint_positions": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
  "robot_gripper_position": 0.04,
  "objects": {
    "block_a": {"position": [0.5, 0.0, 0.05], "quaternion": [1.0, 0.0, 0.0, 0.0]}
  },
  "table_height_delta": 0.0,
  "expert_episode_length": 310
}
```

Whenever environments reset, the worker fetches the next setup for each of them
(`EpisodeQueueClient.fetch_reset_states`) and writes it into the simulation
(`apply_setup`). Joints and objects a setup does not list keep their reset values, but
`apply_setup` raises instead of guessing on a setup/scene mismatch: an object the scene
does not spawn, a missing gripper joint, or a table with neither a recorded pose nor a
height delta. One step later, the worker checks that the simulation state matches the
setup; with `--queue-log-level DEBUG` the client also logs every fetched setup and every
joint and object write.

**Prompt resolution.** The language instruction sent to the policy (and logged as the
episode's `prompt`) is the setup's `hl_prompt` (parquet `goal.hl_prompt`) when present;
otherwise the setup's `pddl_goal` is turned into English by `pddl_goal_to_prompt` (for
example `(and (on block_b block_a))` becomes "put the B block on the A block"). Each environment
gets its own episode's prompt. For `molmo2_maniflow` the same string is the Molmo2 task;
`--hl-prompt "stack 2 blocks"` forces one fixed goal for the whole run.

**Which episodes are queued.**

- `--episode-indices` (with `--allow-duplicate-episodes` to repeat an episode) selects the
  episodes, as `0,3,5,10-19` or `@file.txt` (a file of the same entries). The pipeline
  passes the same selection to task generation and to the queue.
- `--init-after-hl-action` moves each episode's start to just after a given high-level
  action. The robot and block poses, the expert first frames and the
  `--expert-timeout-scale` budget then come from the cut (the remaining expert length).
- `--queue-limit` caps how many are queued.

The queue hands out the longest expert demonstrations first, so the episodes likely to run
longest start early and the last stragglers are short ones.

**Sizing from the queue.** Before starting the workers, the pipeline reads `GET /size`:

- Each worker gets `min(ceil(queue size / --num-eval-gpus), --num-envs)` parallel
  environments, so a small queue does not spawn idle environments.
- With `--expert-timeout-scale`, `episode_length_s` is raised to cover that multiple of
  the queue's longest expert demonstration (`expert_seconds_max`), so each episode's own
  budget is the timeout that binds.

**Block-size slots.** Block size cannot change after spawn, so when the dataset records a
`cube_scale` per episode, the generated task spawns one slot per distinct size and the queue
serves each environment only episodes of its slot's size. A worker spreads its environments
over the slots in proportion to each slot's queued workload, which it reads from `/size`
(`slot_counts`, `slot_expert_frames`). Every fetch also sends the task's slot ordering,
and the queue fails if it differs from its own. That catches a task generated from
different `--episode-indices` than the queue was started with. The queue also refuses to
start if only some of the episodes record a `cube_scale`.

**When the queue drains.** An environment whose fetch comes back empty keeps stepping on
the task's default randomization. These "phantom" episodes are never scored or recorded,
but they still cost policy inference and physics. A worker stops once all of its
environments have drained, and the pipeline prints the queue's final status at the end.

**Endpoints:**

- `GET /next`: `{"setup": ...}`, or `null` once drained. With block-size slots, add
  `?slot=<i>&slot_scales=<s0,s1,...>`.
- `GET /size`: `remaining`, `total`, `expert_seconds_max`, `slot_counts` and
  `slot_expert_frames`. The queue fails at startup if the dataset's `meta/info.json`
  has no `fps`, since `expert_seconds_max` needs it.
- `GET /healthz`: `{"ok": true}`.

## Dataset requirements

A run consumes one artifact, the LeRobot dataset at `--dataset-path`. It is the source of
truth for:

1. **The per-episode setups** above. Each needs at least
   `observation.state.joint_position` and `pddl_goal`; the other columns
   (gripper position, block poses, table pose, friction, intrinsics, ...) are used when
   present.
2. **The gym task.** `generate_task_from_dataset.py` reads the block list from the
   `observation.state.object_pose.<name>` columns and every distinct `pddl_goal`
   (quantified, typed goals such as `(exists (?b1 ?b2 - block_red) ...)` are grounded
   against the scene's blocks). Block names resolve to assets by name: a glyph name
   (`block_a` .. `block_z`, `block_1` .. `block_9`) spawns that glyph block from
   `--usd-library`, a name in `USD_OVERRIDES` (in `tamp/blocks/template/codegen.py`) its
   USD, and anything else a cuboid coloured by the colour word in its name.
3. **The scene.** See [Dataset scene provenance](#dataset-scene-provenance).

If the dataset has `observation.images.*` videos, the queue also resolves each episode's
expert clips, which the W&B episodes table shows next to the rollout.

## Dataset scene provenance

The goal is to rebuild a recorded episode's scene from the dataset path and episode index
alone. The rebuild has two halves: the **scene config** (robot, scene preset, table,
cameras, lighting, physics) comes from code and flags, and the **per-episode state**
(joints, gripper, block poses, table height) comes from the data through `apply_setup`. If
the scene config does not match the one datagen used, the state lands on the wrong scene,
so eval pins the scene to the dataset's recorded provenance:

- Datagen records `scene_id` and `robot_id` (per-step columns, or `meta/info.json`).
  `generate_task_from_dataset.py` bakes them into the generated `env_cfg.py`, and
  `evaluate.py --dataset_path` defaults the eval robot and scene preset to them
  (`reconcile_env_cfg_with_dataset` in `utils/dataset_provenance.py`).
- A dataset that records neither, with no `--robot-id` / `--scene-preset` override, is
  rejected rather than run on a possibly wrong scene. An explicit override that disagrees
  is honoured with a warning; regenerating the task from the dataset (or dropping the
  override) is the usual fix.
- The recorded block heights and `table_height_delta` are relative to the datagen
  preset's table, so `apply_setup` restores the table from `table_pose` (absolute) or
  `table_height_delta` and fails rather than inferring it from block positions.

What each part of the scene is rebuilt from:

| Input | Recorded in the dataset | Used by eval |
|---|---|---|
| Robot joints and gripper | `joint_position`, `gripper_position` | written at reset |
| Block poses | `object_pose.<block>` | written at reset (the block must be spawned) |
| PDDL goal | `pddl_goal` | selects each environment's goal at reset |
| Table height | `table_pose` or `table_height_delta` | written at reset |
| Block size | per-episode `cube_scale` | one spawn slot per distinct size |
| Block and table friction | `block_friction`, `table_friction` | written at reset when recorded |
| `scene_id`, `robot_id` | per-step columns | default the eval scene (see above) |
| Camera intrinsics | `camera_intrinsics.<camera>` | off by default: `--use-calibrated-cameras` uses one fixed sim2real set; `--no-use-calibrated-cameras` applies each episode's recorded intrinsics |
| Camera poses and look | `camera_pose.<camera>`, `meta/render_params.json` on rendered datasets | checked by scene verification; reproduced only with `--use-render-params` |
| Actuator gains | `current_stiffness` / `current_damping` / `current_effort_limit` | read by replay and the dataset PD-switch settle (see [Dataset Replay](replay.md)) |
| Block glyphs and colours | glyph/colour renders: `block_variable_map` / `glyph_map` | painted with `--replay-block-identity` (on) |

## Validating dataset loading

To confirm that the eval scene matches the dataset it was generated from:

- [`scripts/eval/verify_dataset_loading.py`](https://github.com/allenai/ToyBlocks/blob/main/scripts/eval/verify_dataset_loading.py)
  (standalone, needs a GPU) applies each requested episode's setup to a freshly reset
  environment, all in one Isaac session, and writes a scene diff (table height, blocks
  present, block and table poses, block sizes, camera intrinsics and pose), first-frame
  MAE / MSE / SSIM with side-by-side PNGs, and a check of the policy input built from
  the reset observation. It exits non-zero on any hard mismatch; each camera's SSIM has a
  hard floor (`--min-first-frame-ssim`, default 0.90). Generate the task package first:

  ```bash
  python scripts/generate_task_from_dataset.py --dataset-path data/datasets/my_dataset --headless
  python scripts/eval/verify_dataset_loading.py \
      --dataset-path data/datasets/my_dataset --task my_dataset-v0 --episode-indices 0 4 \
      --obs-mode rgb+state --output-dir runs/verify --headless --enable_cameras
  ```

- `evaluate.py --verify_scene` runs the same scene checks on the first batch of queue
  setups during a live eval and writes `scene_verification.json` with comparison PNGs.
  Hard mismatches are only logged; `--verify_scene_strict` aborts the run after the
  report is written. Use strict whenever the numbers will be reported. These two flags
  are not exposed by `launch_eval.py`.

## Evaluating on rendered datasets

The [render stage](../datagen/render.md) gives every episode its own look:

- its own camera offsets, HDRI sky, table material, lights, exposure and backdrop, recorded
  in `meta/render_params.json` (for a merged dataset, one file per source, reached through
  `meta/provenance.json`);
- on glyph and colour renders, its own block letters and colours (`block_variable_map` /
  `glyph_map`).

By default, eval reproduces only the block identities. Everything else is the single
calibrated sim2real scene. Four flags control how much of the recorded look is
reproduced:

| Flag | Default | Effect |
|---|---|---|
| `--use-calibrated-cameras` | on | Fixed sim2real camera intrinsics for every environment; also turns the render replay off. `--no-use-calibrated-cameras` gives each environment its episode's recorded intrinsics instead. |
| `--use-render-params` | off | Reproduce each episode's camera poses (and the wrist camera's mount offset), HDRI sky, table material, lights, exposure and backdrop. Does nothing on datasets without `render_params.json`. |
| `--wave-resets` | off | Start episodes in waves, so all environments in a wave share one episode's lighting (see the caveats below). |
| `--replay-block-identity` | on | Paint each episode's recorded block letters and colours, and prompt the policy with the goal as rendered. Success is always graded on the source block names. `--no-replay-block-identity` keeps the original look and the source-name prompts. |

To evaluate with the full recorded look:

```bash
python scripts/launch_eval.py --policy maniflow \
    --model-path data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
    --dataset-path data/datasets/my_rendered_dataset \
    --no-use-calibrated-cameras --use-render-params --wave-resets
```

With `--replay`, drop `--wave-resets`, since replay runs a single environment per GPU.

**What the launcher checks** before it starts anything:

- `--use-render-params` needs `--no-use-calibrated-cameras`.
- `--wave-resets` needs `--use-render-params` and more than one environment, so it cannot
  be combined with `--replay` or with calibrated cameras.

**Caveats:**

- **Lighting is shared by the whole scene.** The sky, lights and exposure are one setting
  for all environments. Without `--wave-resets` they are set once at the first reset, so
  with several environments most episodes run under another episode's lighting.
- **Tiled rendering uses one set of intrinsics.** Every environment is drawn with
  environment 0's intrinsics, so an environment whose episode recorded different ones
  reports intrinsics that don't match its pixels.
- **A mismatched record is skipped, not fatal.** An episode whose render record fails the
  lookup checks (the episode count per source problem, the front camera position, the
  table height) runs with nothing applied, and the run carries on.

To check what was applied, look at each episode's `render_replay_status` and
`block_identity_status` (`applied` or `none`) in `eval_episodes.json`; see
[Metrics](metrics.md).

## HL latency

The four HL latency flags decouple the high-level conditioning (the oracle for
`--policy maniflow`, the Molmo2 forward for `--policy molmo2_maniflow`) from per-step
low-level inference, in milliseconds of sim time
(`sim_time_ms = step * step_dt * 1000`; see `src/toyblocks/serving/hl_latency.py`). All
default to 0, which runs the HL on every LL inference with no delay:

- **K** (`--hl-update-interval-ms`): invoke the HL at most once per K ms and buffer its
  output in between. `--hl-update-interval-jitter-ms j` adds `U(-j, +j)` ms to each
  interval to mimic HL timing variance.
- **M** (`--estimated-hl-inference-time-ms`): feed the LL the freshest HL output at
  least M ms old, faking the arm and physics advancing during HL inference.

The Molmo2 server only sees observations at action-chunk boundaries, so a K below
`action_horizon * step_dt` is a no-op there; the client-side oracle buffer ticks every
control step.

**Checkpoints that read the previous front frame.** A ManiFlow checkpoint trained with
`use_previous_front_frame` also takes the aged HL frame as `prev_front_rgb`. There is no
flag for it: the server advertises the input and the client sends it. M and K then have
to match training: the dataloader pairs each frame with one
`n_action_steps * U{1,2,3,4}` steps back (`n_action_steps` is in the checkpoint's
`config.yaml`), while the conditioning age cycles within `[M, M + K]`. So set
`M ≈ n_action_steps * step_dt * 1000` and size K so the age reaches about 4x that. Two
settings print a loud startup warning naming the resulting age (neither is fatal, since
each is also how a synchronous HL is configured on purpose):

- `M = 0`: the conditioning frame is the current one at every chunk boundary.
- K an integer multiple of the action-chunk period (`action_horizon * step_dt * 1000`):
  the HL commits phase-lock with the inferences, so the age is a constant M instead of
  sweeping `[M, M + K)`. Offset K or add jitter to break the lock.

`--save-diagnostic-plots` writes `hl_conditioning_staleness.png` (client-side oracle HL
only), the quickest check of what actually happened: K is the sawtooth period of the
conditioning age and M its floor, and LL calls and HL commits are drawn as event lines.
For the Molmo2 HL, the server-side HL input trace is the equivalent record.

## Outputs

Results go under `RUNS_ROOT`, which defaults to `runs/` in the repository root (set
`RUNS_ROOT=/path` in the environment of `launch_eval.py` to change it). Each eval worker
`N` (seed `N`, from 1) writes to

```text
runs/<task>-v0/<policy>/<out_subdir>/<problem_id>/seed_<N>/<YYYYmmdd_HHMMSS>_local/
```

with `<policy>` from `--policy-name` (`maniflow`, `molmo2_maniflow` or `replay` by
default), `<out_subdir>` from the profile or `--out-subdir`, and `<problem_id>` from
`--problem-id`. Besides the [`evaluate.py` outputs](evaluate.md#outputs)
(`eval_results.json`, `eval_episodes.json`, the LeRobot dataset, videos), each worker
directory holds `launch_params.json` (the resolved launch parameters, including the
commit), `launch_command.sh` (the command line), `std_out_err.log` (the pipeline log
followed by that worker's output) and `skill_success.json` (per-skill success rates from
[`scripts/skill_success_from_eval_run.py`](https://github.com/allenai/ToyBlocks/blob/main/scripts/skill_success_from_eval_run.py)).

At the end, [`scripts/merge_eval_runs.py`](https://github.com/allenai/ToyBlocks/blob/main/scripts/merge_eval_runs.py)
merges the workers into one run directory, `runs/<task>-v0/<policy>/<out_subdir>/<problem_id>/<run_id>/`,
with every worker's episodes in one `eval_episodes.json`, the merged success LeRobot
dataset (moved out of the workers; the failure datasets stay in them) and its own
`skill_success.json`.

With `--wandb`, [`scripts/wandb_consolidate.py`](https://github.com/allenai/ToyBlocks/blob/main/scripts/wandb_consolidate.py)
merges every worker into one W&B run named `<policy>__<task>-v0__<out_subdir>` (needs
`WANDB_API_KEY`). It carries `summary/run_stats_table` (one row of headline numbers:
success rates, episode lengths against the expert's, timeout rate, throughput),
`eval/aggregates_table` (every metric's mean, std, min and max),
`videos/episodes_with_expert_table` (one row per episode, with the expert video next to
the rollout), and, under `--save-diagnostic-plots`, the `debug/policy_inputs_table`,
`debug/ll_policy_inputs_table` and `debug/hl_policy_inputs_table` debugging tables.

## Examples

```bash
# Smoke test of Molmo2 high level + ManiFlow low level (the default policy)
python scripts/launch_eval.py --smoke-test \
    --model-path data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
    --hl-model-path data/checkpoints/my_molmo2_run \
    --dataset-path data/datasets/my_dataset

# ManiFlow with the high-level oracle, on 4 eval GPUs, logged to W&B
python scripts/launch_eval.py --policy maniflow --num-eval-gpus 4 \
    --model-path data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
    --dataset-path data/datasets/my_dataset --wandb

# Molmo2 high level + motion-planned skills (low-level oracle), episodes 0-19 only
python scripts/launch_eval.py --ll-oracle \
    --model-path data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
    --hl-model-path data/checkpoints/my_molmo2_run \
    --dataset-path data/datasets/my_dataset --episode-indices "0-19"

# Start after every flip or rotate, with a per-episode timeout of 1.5x the expert's
python scripts/launch_eval.py --policy maniflow \
    --model-path data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
    --dataset-path data/datasets/my_dataset \
    --init-after-hl-action flip,rotate --expert-timeout-scale 1.5

# Replay the dataset's recorded actions (no policy server; one cube scale)
python scripts/launch_eval.py --replay \
    --dataset-path data/datasets/my_dataset --episode-indices "0"

# Print the pipeline invocation instead of running it
python scripts/launch_eval.py --dry-run --policy maniflow \
    --model-path data/checkpoints/my_maniflow_run/checkpoints/steps_400000_model.safetensors \
    --dataset-path data/datasets/my_dataset --output /tmp/eval_run.sh
```

## Flags

### Policy and run mode

| Flag | Default | Description |
|---|---|---|
| `--policy` | `molmo2_maniflow` | Policy to evaluate: `molmo2_maniflow` or `maniflow`. |
| `--replay` | off | Replay the dataset's recorded actions instead of evaluating a policy: no policy server, each worker runs `evaluate.py --replay`. Overrides `--policy`; `--num-envs` defaults to and must stay 1. Rejects `--model-path`, `--hl-model-path`, `--oracle-comparison`, `--replicate-noise` and `--init-after-hl-action`. See [Dataset Replay](replay.md). |
| `--replay-clip-stats` | none | With `--replay`: a `stats.json` whose `action.joint_target` q01/q99 bands clamp every replayed arm joint target. Adds `replay_clipped_frames` / `replay_clip_max_rad`. |
| `--smoke-test` | off | Use the smoke-test profile above. |
| `--dry-run` | off | Print the `env VAR=... bash run_eval_pipeline.sh` invocation without running it. |
| `--output` | none | With `--dry-run`: write the invocation to this path instead of stdout. |

### Dataset, task and queue

See [The episode queue](#the-episode-queue) for how these flags shape the queue.

| Flag | Default | Description |
|---|---|---|
| `--dataset-path` | required | LeRobot dataset the queue reads its episode setups from and the gym task is generated from. |
| `--task-name` | dataset basename | Override the gym task slug (a valid Python identifier). |
| `--episode-indices` | all | Episodes to queue, e.g. `0,3,5,10-19` or `@file.txt`. |
| `--allow-duplicate-episodes` | off | Allow repeated indices, e.g. `0,0,0,0,0` with `--num-envs 5` to run episode 0 five times in parallel. |
| `--init-after-hl-action` | none | Start each episode at the first frame after every instance of these high-level actions (comma or `\|` separated, e.g. `flip,rotate`; `flip` also matches `reorient`). Episodes without a match start at frame 0; episodes whose last match is their last frame are skipped. |
| `--queue-limit` | profile | Cap the queue at N episodes. |
| `--queue-log-level` | profile | Log level of the episode queue server. |
| `--usd-library` | `train_blocks_large` | Glyph asset library for bare block names in task generation; `train_blocks` for legacy small-block datasets. |

### Scale and episode budget

| Flag | Default | Description |
|---|---|---|
| `--num-envs` | profile | Maximum parallel environments per eval GPU (the pipeline right-sizes it to the queue). At most 128 for `molmo2_maniflow`. |
| `--num-eval-gpus` | profile (1) | Eval workers, one per GPU after the policy server's (GPU 0). Must be 1 for `molmo2_maniflow`. |
| `--episode-length-s` | profile | Episode timeout in sim seconds. |
| `--expert-timeout-scale` | `0` | Give each episode its own timeout, this multiple of its expert demonstration's length (e.g. `1.5`). The pipeline raises `--episode-length-s` to cover the longest expert in the queue. `0` keeps one shared timeout. |

### Episode end and scoring

| Flag | Default | Description |
|---|---|---|
| `--episode-end` | `goal` | What ends an episode before the timeout: `goal` (the goal predicate; oracle runs: plan completion), `horizon` (nothing), or `ll_done` (a play-mode LL's -1; `--policy maniflow` only). Replay always uses `horizon`. |
| `--collision` | `end` | A non-target block moving more than 2 cm or 15 deg: `end` fails and ends the episode, `penalize` fails it but lets it run on, `off` does not score collisions. |
| `--goal-ori-yaw-tol-deg` | `40.0` | Yaw tolerance of the goal's orientation atoms when judging success. |
| `--post-place-lift` | `0` | After a play-mode LL releases a block, lift the arm this many metres before the next LL query (`0.035` matches datagen). |

### ManiFlow model

| Flag | Default | Description |
|---|---|---|
| `--model-path` | none | ManiFlow `.safetensors` checkpoint (required for `maniflow` and `molmo2_maniflow`). |
| `--config-path` | `<model-path>/../config.yaml` | Model config (for a checkpoint inside `checkpoints/`, the run directory's `config.yaml`). |
| `--stats-path` | checkpoint's own | Normalization `stats.json`; by default the server uses the checkpoint's recorded training stats, else the `stats.json` beside its `config.yaml`. |
| `--train-config-path` | `maniflow-tb/examples/Franka/train_maniflow_multimodal.yaml` | Training config merged under the checkpoint's config; keep it at the pinned `maniflow-tb`. |
| `--action-mode` | auto-detected | `relative` or `absolute`. |
| `--policy-seed` | `0` | Seed of the policy server's action sampling. |
| `--replicate-noise` | off | Start every sample of a batched inference call from the same flow-matching noise, so identical observations give identical actions (`maniflow` only). |

### Molmo2 high-level model

| Flag | Default | Description |
|---|---|---|
| `--hl-model-path` | none | Molmo2 checkpoint (required for `molmo2_maniflow`). |
| `--hl-prompt` | per episode | Force a fixed language goal (e.g. `stack 2 blocks`) instead of each episode's prompt. |
| `--no-hl-prev-action` | off | Never put the previous action in the Molmo2 question (for HL checkpoints trained without it). |
| `--hl-max-new-tokens` | server default | Molmo2 decode budget per answer. |

### High-level conditioning

| Flag | Default | Description |
|---|---|---|
| `--use-oracle` | on | `maniflow` only: drive the task string and target heatmap from the client-side symbolic oracle (`--no-use-oracle` to disable). |
| `--oracle-comparison` | off | `molmo2_maniflow` only: also run a client-side oracle and show it next to the model in the HL overlay video. Needs `--save-hl-overlay-video`. |
| `--async-hl` | off | `molmo2_maniflow` with a play-mode LL: query Molmo2 only at episode start and when the LL ends a skill, instead of at every chunk boundary. |
| `--async-hl-call-scale` | no cap | With `--async-hl`: end an episode once Molmo2 has been called more than this multiple of its expert's HL calls. |
| `--ll-oracle` | off | `molmo2_maniflow`: run Molmo2's commands with the datagen motion-planned skills (cuRobo v2) instead of the served LL. Each command acts on the block nearest its pixel (or the table within 25 px), and Molmo2 is queried again per environment once it ends (implies asynchronous HL queries). |
| `--ll-oracle-hl-call-scale` | `3.0` | With `--ll-oracle`: the same HL call cap. |
| `--go-home` | off | With `--async-hl` or `--ll-oracle`: drive the arm home after a place, flip or rotate, before the next Molmo2 query. |

### HL latency

See [HL latency](#hl-latency) for how to choose K and M.

| Flag | Default | Description |
|---|---|---|
| `--hl-update-interval-ms` | `0` | HL invocation interval K in ms (`0` = every step). |
| `--hl-update-interval-jitter-ms` | `0` | Uniform jitter added to each interval. |
| `--estimated-hl-inference-time-ms` | `0` | Assumed HL inference time M in ms. |
| `--save-hl-latency-video` | off | Per-episode side-by-side of the fresh and the aged HL view (needs `--save-sensor-obs-video`). |

### Observations, transform and scene

See [Evaluating on rendered datasets](#evaluating-on-rendered-datasets) for how the camera,
render and block-identity flags work together.

| Flag | Default | Description |
|---|---|---|
| `--obs-transform` | per policy | Observation transform passed to `evaluate.py`. |
| `--action-horizon` | per policy | Actions applied per policy chunk. |
| `--robot-id` | dataset provenance | Robot config (`--env-cfg.robot-id`). |
| `--scene-preset` | dataset provenance | Scene preset (`--env-cfg.scene-preset-id`). |
| `--center-crop-ratio` | 0.95 for ManiFlow transforms | Centre-crop each resized 224x224 input to `int(224 * ratio)` and resize it back, matching a checkpoint trained with `random_crop_ratio`; `1.0` feeds the full frame. |
| `--enable-crop` | off | ManiFlow transforms only: build `target_front_crop_rgb` on the client. |
| `--use-calibrated-cameras` | on | Use the fixed sim2real camera intrinsics; `--no-use-calibrated-cameras` uses each episode's recorded intrinsics. |
| `--use-render-params` | off | On a rendered dataset, reproduce each episode's camera pose, HDRI, table material, lights, exposure and backdrop. Needs `--no-use-calibrated-cameras`; with `--num-envs > 1` add `--wave-resets`. |
| `--replay-block-identity` | on | On a glyph/colour-rendered dataset, paint each episode's recorded block letters and colours and prompt with the recorded goal. Success is graded on the source block names either way. |
| `--wave-resets` | off | Start episodes in waves, so each wave shares one episode's recorded scene lighting. Needs a rendered dataset and `--num-envs > 1`; not with `--replay` or calibrated cameras. |
| `--eval-table-z-offset` | none | Extra world-frame Z offset (m) for the table and blocks after the queue setup, e.g. `-0.02`. |
| `--replay-gripper-close-target` | none | Override the binary gripper close target (radians). |

### Recording

| Flag | Default | Description |
|---|---|---|
| `--save-lerobot` | on | Record every rollout as a LeRobot dataset, encoded by sharded writer processes as frames arrive. |
| `--lerobot-dataset-name` | run name | Base name of the merged dataset. |
| `--num-shards` | `8` | Parallel LeRobot writer processes, stitched into one success / failure pair at the end. |
| `--video-codec` | `libx264` | `libx264` or `libsvtav1` (about 4x the memory per open stream). |
| `--save-sensor-obs-video` | off | Per-episode MP4s from the RGB observations (buffered in RAM, unlike the LeRobot recording). |
| `--save-hl-overlay-video` | off | Record the HL overlay as a video stream in the LeRobot dataset (needs `--save-lerobot` and an HL). |
| `--save-step-metrics` | off | Per-step metric CSVs in each worker's output directory; for debugging. |

### Diagnostics

| Flag | Default | Description |
|---|---|---|
| `--save-diagnostic-plots` | off | Live-rollout diagnostics: policy-input verification panels, the HL staleness plot, and the server-side LL and HL input traces. Also renders every episode's offline control-diagnostic plots at the end of the pipeline. Implies `--save-first-batch-data` and `--save-lerobot`. |
| `--save-first-batch-data` | off | Dump the model-ready batch the server saw on step 0 to `first_batch_data/` (`maniflow` only). |
| `--deterministic-cuda` | off | Deterministic cuDNN and PhysX enhanced determinism on the server and the workers; slower, and still not bit-for-bit across hardware (see [Policy Evaluation](../evaluations.md)). |

### W&B and output naming

| Flag | Default | Description |
|---|---|---|
| `--wandb` | off | Log the run to W&B, merging every worker into one run. Also extracts each episode's expert video for the episodes table. |
| `--wandb-project` | `toyblocks-eval` | W&B project. |
| `--out-subdir` | profile | Output subdirectory under `runs/<task>-v0/<policy>/`. |
| `--policy-name` | from `--policy` | Policy label used in W&B names and output paths. Currently only the default works: the pipeline also reads this label to choose the policy server, so any other value stops it with `unknown POLICY_NAME`. |
| `--problem-id` | `problem_000000` | `<problem_id>` segment of the output path. |

### Deprecated

| Flag | Description |
|---|---|
| `--legacy-ee-pos` | Rejected with an explanation: `ee_pos` now always comes from the robot config's `ee_body_name`. |
