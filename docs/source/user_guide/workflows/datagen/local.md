# Running the pipeline on this machine

`scripts/datagen/run_datagen_pipeline.py` runs the whole data generation pipeline on the machine
you launch it on: PDDL problems, env packages, executed episodes, rendered videos and a merged
LeRobot dataset, all on local disk and local GPUs. The same command works on a single-GPU
workstation and on a multi-GPU node; the pipeline spreads problems over every GPU it can see.

This page covers what is specific to running it: choosing GPUs and concurrency, where outputs and
logs go, rendering, resuming, and checking the result. For the stage-by-stage reference, see
[Automatic Data Generation](../data_generation.md).

| Stage | How it runs |
|---|---|
| task-gen | one subprocess, on the CPU |
| env | one process per problem, across the GPU slots (skipped with `--vecgen-problems-per-env` > 1, the default) |
| execution | one process per problem or problem group, across the GPU slots |
| render | one render per problem (`--render-randomized`), scheduled into the shared GPU-slot pool or inside each execution job (`--render-schedule`) |
| merge | one subprocess, on the CPU |
| annotate | one subprocess, on the CPU |

The pipeline needs no network access once the assets are installed. Run it from the root of the
checkout: the stage scripts are called by relative path (`scripts/...`), and the pipeline runs
`git rev-parse HEAD` to pin the commit into the run config for reproducibility.

## Before you start

Follow [the installation instructions](../installation_and_setup_instructions.md): with the
virtual environment active, install the datagen (vecgen) extra and fetch the assets:

```bash
uv pip install -e ".[dev,isaac,torch-cu128,vecgen]"
```

```bash
get-toyblocks-assets --asset-manifest asset_manifest.json
```

Randomized rendering draws HDRI backdrops from polyhaven (installed with the base
assets) and the separately licensed Laval indoor set; see
[HDRI backdrops](https://github.com/allenai/ToyBlocks/blob/main/docs/source/user_guide/workflows/datagen/local.md#hdri-backdrops).

Rendering needs real disk. A run of a few problems at `--render-n 2` writes several GB of
video; check free space before a long run rather than discovering it mid-encode.

## A complete run

This generates 4 stacking problems, executes 16 episodes each, renders them, and merges the
result into one LeRobot dataset under `./outputs`:

```bash
python scripts/datagen/run_datagen_pipeline.py --render \
    --render-flags src/toyblocks/domain_randomization/flags/render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml \
    --experiment-id my_run --output-root outputs \
    --task-gen-script scripts/task_gen/stack_three.py --task-gen-count 4 \
    --env-per-problem 16 --num-envs 16 --exec-num-envs 16 \
    --vecgen-problems-per-env 1
```

The merged dataset is annotated at the end, since the render supplies the camera data that
annotation projects goals into. Drop `--render-randomized` to stop after execution and merge the raw
parquet (state and actions, no video); the pipeline then logs a warning and skips annotation,
which needs video (`--render-randomized` or `--vecgen-record-videos`).

Without any flags beyond `--experiment-id`, the pipeline uses the settings the RI stack-3 training
set was generated with: 1000 `stack_three.py --train` problems, randomized scenes, 32 reset states
per problem, 8 problems per vecgen env package at 128 envs.

Each stage also runs on its own; see [task generation](task_generation.md),
[env generation](env_generation.md), [execution](execution.md), [rendering](render.md),
[merge](merge.md) and [annotate](annotate.md).

## Choosing GPUs and how many problems run at once

| Flag | Default | Meaning |
|---|---|---|
| `--local-gpus ID [ID ...]` | every GPU visible to the process | Which GPU ids the stages may use. |
| `--local-gpu-workers N` | `1` | Concurrent problems **per GPU**. `1` is a strict no-op. |

The pipeline builds one *slot* per GPU per worker, so `len(gpus) * workers` problems run at once.
Each slot owns one `CUDA_VISIBLE_DEVICES` value, so every job sees a single GPU; a stage scales
out by spreading problems over more slots, never by giving one problem several GPUs. Listing a
GPU id twice in `--local-gpus` gives it two slots, like `--local-gpu-workers 2` does.

GPU discovery honours an existing `CUDA_VISIBLE_DEVICES`; failing that it asks `nvidia-smi`, and
if that is unavailable it runs one problem at a time on the default GPU.

A failing problem does not abort the stage. The stage keeps going and reports the failure count at
the end, and the next stage works with the problems that succeeded.

### How much fits

Measured on one RTX 5090 (32,607 MiB) at `--render-local-num-envs 8`:

| | peak VRAM | peak GPU util | mean util while busy |
|---|---|---|---|
| render | 7,673 MiB | 76% | — |
| vecgen execution | 5,198 MiB | 88% | 61% |

Two concurrent processes fit comfortably on a 32 GiB card and one does not saturate it, so
`--local-gpu-workers 2` is usually free throughput. Read "Host RAM and disk, not VRAM, are the
limits" under [Things that will bite you](#things-that-will-bite-you) first.

## Logs

With a single slot, every job's output is streamed to your console. With more than one slot, each
problem's output goes to its own log file instead, since concurrent Isaac Sim logs are unreadable
interleaved:

```text
<base>/local_logs/env/<slug>.log
<base>/local_logs/execution/<slug>.log
<base>/local_logs/render/<slug>.log
```

The pipeline logs the directory when it does this, and a `FAILED <stage>: <slug> (see ...)` line
names the log of each problem that failed. `[sched] <stage> start|end <slug> slot=... t=...` lines
record when each job ran, which is how per-stage timings are measured. The pipeline's own log goes
to the console only; pipe it through `tee` if you want to keep it.

## Outputs

| Flag | Default | Meaning |
|---|---|---|
| `--output-root` | `data` | Root for every dataset the pipeline writes. A relative path is taken from the working directory and resolved to an absolute one. |
| `--dataset-folder` | `unmerged` | Subfolder of `<output-root>/datasets/` for the per-problem outputs. |
| `--merge-output-dir` | `<output-root>/datasets/merged/train/<experiment-id>` | Where the merged dataset goes. |

Everything else lands under `<base>` = `<output-root>/datasets/<dataset-folder>/<experiment-id>/`:

```text
<experiment-id>/
├── tasks/              # generated PDDL problem files
├── envs/               # env packages + reset states, one per problem or problem_grp_<slug>
├── execution/          # per-problem LeRobot datasets (blocks_success + blocks_failure)
├── render/
│   ├── src/                      # per-problem staging links the renders read from
│   ├── out/job_local/            # rendered problems
│   ├── out/job_local_nonrand/    # the --render-non-randomized pass, if enabled
│   └── non_randomized_flags.yaml # derived flags for that pass
├── local_logs/         # per-problem logs, only when >1 slot runs at a time
├── .pipeline_state/    # <stage>.done markers that --resume reads
└── run_datagen_pipeline_config_<timestamp>.json
```

The merged dataset gets a copy of the run config and a plot of its first episode,
`episode_0.png`.

## HDRI backdrops

The `allrand` flags (including the default) light each episode with an HDRI backdrop sampled from
two pools: `env_sphere.sets: [polyhaven/all, indoor-2018/all]`. The base asset download includes
polyhaven; indoor-2018 is installed separately.
Each pool is a folder `<root>/<pool>/` holding the `.hdr` files plus a `dataset.json` that lists
the files in each named set (`all`, `rooms`, ...). `<root>` is `assets/environments/`, or the
directory in `RENDER_HDRI_DIR` when that is set.

| pool | what | size | license | how to get it |
|---|---|---|---|---|
| `polyhaven` | 299 Poly Haven HDRIs at 2k (`all` = 134 of them) | 1.8 GB | CC0 (credits in its `ATTRIBUTION.md`) | included in `get-toyblocks-assets --asset-manifest asset_manifest.json` |
| `indoor-2018` | Laval Indoor HDR Dataset (Gardner et al. 2017), 2,113 panoramas as 4096x2048 `.hdr` | 44 GB | research / non-commercial | request it from Université Laval ([indoor.hdrdb.com](http://indoor.hdrdb.com)); not redistributed |

`get-toyblocks-assets` installs the polyhaven pool, `dataset.json` and `ATTRIBUTION.md` included,
under `assets/environments/polyhaven`. For indoor-2018, convert the panoramas to 4096x2048 Radiance
`.hdr` and put them in `<root>/indoor-2018/` together with a `dataset.json` whose `sets.all` lists
the file names:

```json
{"source": "Laval Indoor HDR Dataset 2018", "format": "hdr", "sets": {"all": ["9C4A0003-e05009bcad.hdr", "..."]}}
```

With only polyhaven installed the render still runs. The indoor-2018 bucket resolves to nothing
(`[dr] WARNING skybox bucket resolved 0 textures`), so every backdrop comes from polyhaven, and the
images differ from the RI training data, which drew from both pools. The log line
`[render] asset map:` shows which pools were found.

## Rendering locally

Execution records state and actions only — vecgen spawns no cameras by default. The videos come
from the render stage, which replays the recorded states with no physics and renders them
under the domain randomization in a flags YAML. See [Rendering](render.md) for what that
stage does and every flag it takes, and the
[YAML reference](https://github.com/allenai/ToyBlocks/blob/main/src/toyblocks/domain_randomization/YAML_REFERENCE.md)
for how to write the randomization.

Two flags turn the stage on, and they pick how it is scheduled against the executions. Both
produce **the same pixels** — the `render.py` argv, the staged source layout and the
per-problem output directory name (which is the domain-randomization seed salt) are identical, so
a problem draws the same randomization either way. The choice is about throughput and
observability, not about how the dataset looks.

| | `--render-schedule coupled` (default) | `--render-schedule pipelined` |
|---|---|---|
| where the render runs | its own item in the shared GPU-slot pool | inside the execution job's own shell chain, `vecgen && render` |
| when it becomes eligible | the moment that problem's execution finalizes | immediately, holding the slot its execution just used |
| can overlap the next execution | yes, whenever a slot is free | no — one job, one slot |
| exec vs render timing | separate items, separate logs, separate `[sched]` lines | not separable (one process span) |
| a render failure | its own item with its own log | indistinguishable from an execution failure in the job's exit code |

Neither schedule can be combined with `--vecgen-record-videos`: recording in-physics video and
then overwriting it pays the slow physics-render cost for frames nothing reads.

There is no default YAML: `--render-flags` always names the look. The RI training set uses
`src/toyblocks/domain_randomization/flags/render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml`
(`--render-flags`), one copy per episode (`--render-n 1`), the 224 RGB and `rg8_55_224` depth
sidecars, `left_camera` skipped and failure episodes skipped. The flags YAML must be readable on
this machine; the pipeline checks it before starting.

### Which schedule to use

Measured on one RTX 5090 over 4 problems, 2 repetitions, timing the execution + render
window only, with pipelining and oversubscription varied separately so neither is credited with
the other's win:

| | 1 slot | 2 slots (`--local-gpu-workers 2`) |
|---|---|---|
| coupled | 921 s | **667 s** |
| pipelined | 906 s | 733 s |

**For throughput, add `--local-gpu-workers 2`** — 28% faster than a single coupled slot, and
it needs none of the scheduler. Pipelining costs nothing at one slot but *loses* about 10% at
two, because its render-first ordering pairs render against render (two GPU-saturating
processes), whereas the coupled chains naturally pair one problem's execution against another's
render.

Use the pipelined mode when you want separable exec/render timings and logs, failure isolation
between the two, a first finished problem early rather than at the end of a chain, or bounded
un-rendered disk.

### Options that matter on one machine

| Flag | Default | Meaning |
|---|---|---|
| `--render-local-num-envs` | `8` | Envs per render wave, under either schedule. A **floor**, not a preference: the render spawns env *i* at block-size slot *i* % *profiles*, so a value below `--num-scene-profiles` leaves the high slots without an env and **every problem is skipped**. The pipeline refuses that pairing up front. 8 keeps the encoder's RSS survivable beside Isaac. |
| `--render-local-stall-timeout` | `3600` | SIGKILL the render if it prints nothing for this many seconds. |
| `--render-local-fail-on-video-defects` | off | Fail a problem whose encoder could not publish every video. Off by default because a failed problem renders whole; `scripts/render/fast_patch.py` fills video holes afterwards. |

`--render-n`, `--render-non-randomized`, `--render-skip-cameras` and `--render-extra` are
described in [Rendering](render.md), along with what `--render-n` and
`--render-non-randomized` actually multiply and how to check that they did.

## What counts as "rendered"

Once the render stage is marked complete, **merge reads only `render/out/job_*`**. A problem
that executed but was never rendered would simply vanish from the merged dataset. To stop that
being silent, a post-stage sweep classifies every problem on disk into three states:

| State | Meaning | Retried? |
|---|---|---|
| `RENDER_SHIPPED` | finalized under `blocks_success` **and** merge can read it (`info.json` present) | no — it is done |
| `RENDER_UNRENDERABLE` | finalized, but it can never ship: it declared itself empty, recorded zero episodes, or finalized with only a `blocks_failure` split | no — permanent, and failing on it would wedge every later run |
| `RENDER_ABSENT` | not finalized, or finalized without a readable dataset | yes — the run exits non-zero and a later run or `--resume` finishes it |

Two of these are worth understanding. `completed: true` in `render_params.json` means *terminal*,
not *has data*. And a problem whose episodes all failed finalizes with `blocks_failure` alone, so
no `blocks_success` is ever created; re-running cannot change that, because the source has no
successful episodes.

Both modes report from this sweep, never from the job list, so it also covers problems the
execution loop skipped, runs with `--skip-execution`, resumed runs where the execution stage was
disabled, and preemption shards (`problem_x__s001`). The merge gate asks merge's own discovery
what it would ship, rather than globbing `problem_*` — the execution stage creates each problem's
output directory *before* running it, so a crashed execution leaves an empty directory behind for
good. Pass `--merge-allow-missing-problems` to merge what rendered and drop the rest.

## Flags that do not do what their name suggests

| Flag | What actually happens |
|---|---|
| `--num-reset-states` | **Removed.** It never had an effect — the env stage passes `--env-per-problem` through as the generator's `--num_reset_states` — so passing it is now an error rather than a silent no-op. Use **`--env-per-problem`** (default `32`). |
| `--env-per-problem` | Despite the name, this is the number of **reset states** per problem, not a number of environments. The environment count is `--num-envs` (sampling) or `--exec-num-envs` (execution). |
| `--num-scene-profiles` | The count is **baked into the env package at codegen time**, so changing it does nothing for a problem whose package already exists — see below. |
| `--seed` | With `--vecgen-problems-per-env 1`, it does not reach env generation: the env stage is hard-wired to `ENV_GEN_SEED = 0`, so re-running with a different `--seed` regenerates **identical reset states** and `--seed` affects execution only. With multi-problem packages (the default), the package for the group starting at problem *g* is sampled with `--seed + g`, so `--seed` changes the reset states too. |
| `--vecgen-episodes` | Only used with `--vecgen-problems-per-env 1`. A multi-problem job runs `--env-per-problem` episodes per problem in its group. |

The env-count constraint is "a multiple of the profile count, **or** strictly smaller than it" —
`--num-envs 4` against 8 profiles is legal and spawns only the low slots. It is not "must be a
multiple".

With `--vecgen-problems-per-env > 1` the env stage is **switched off entirely** — the execution
jobs build the multi-problem packages themselves with `scripts/vecgen/make_superset_env.py`.

### Render arguments the pipeline pins for you

The pipeline builds its own `render.py` command line, and three of its arguments differ from
`render.py`'s own defaults. They are worth knowing if you are comparing a pipeline render
against one you ran by hand:

| Argument | Pipeline passes | `render.py` default |
|---|---|---|
| `--env_spacing` | `60` | unset (scene default `5.0`) |
| `--camera_base` | `scene` | `data` |
| `--rtx_warmup_frames` | `16` | `1` |

`--num_envs` comes from `--render-local-num-envs`. Anything you pass through
`--render-extra` is appended last and therefore wins over all of these.

Env generation also **skips any problem whose `reset_states.pkl` already exists**, and a
multi-problem job skips building a package whose `problems.json` exists. That is what makes
re-running cheap — and what makes a stale env package silently survive a parameter change.
Delete the problem's env directory, or use a fresh `--experiment-id`, when you change env-stage
parameters. This applies to `--num-scene-profiles` in particular: the profile count is a property
of the package, not of the run, so re-running an existing experiment with a new count keeps the
old bake.

## Resuming

`--resume` continues a previous run of the same `--experiment-id`: stages already marked complete
in `<base>/.pipeline_state/` are skipped, and the first unfinished stage is rerun **from scratch,
deleting its outputs** first. It is a no-op if the experiment directory does not exist.

What that deletes matters for long runs. If the execution stage was interrupted, `--resume`
deletes `execution/` and `render/`, including the problems that had already finished. The
render stage is the exception: it deletes nothing, and `render.py --resume` renders only the
problems not yet completed. A run interrupted after its executions finalized therefore resumes
straight into `render` instead of silently merging raw parquet.

A stage is marked complete when it finishes, even if some of its problems failed, so `--resume`
does not retry failed problems. To keep finished problems and run only the missing or failed
ones, rerun the same command **without** `--resume` and with `--skip-task-gen` (otherwise task
generation writes a fresh batch of problems next to the old ones): env generation skips problems
whose `reset_states.pkl` exists, and execution skips problems whose `.execution_complete` marker
exists.

## Verify the output

**Metadata cannot see a broken render.** A dataset can carry `completed: true` and still contain
videos no decoder will open — a host-RAM OOM mid-encode produces `moov atom not found` mp4s behind
a perfectly healthy-looking marker. Validate the pixels after any run above one worker:

```bash
python scripts/render/validate_dataset_videos.py <base>/render/out/job_local
```

To confirm that `--render-n` genuinely multiplied, count episodes, frames and distinct video
files — never bytes. Merge hardlinks videos into the merged tree, so `du` of the merged directory
is meaningless, and different random draws compress differently. See
[Rendering](render.md).

## Things that will bite you

### Rendering segfaults at startup on some GPU/driver combinations

On an RTX 5090 with driver 595.99.02, every local render died about 7 s into Isaac startup with
a SIGSEGV in `librtx.scenedb.plugin.so`, while env generation and execution ran fine on the same
GPU. This is an Isaac Sim 5.1 RTX-renderer problem, not ToyBlocks. Check your machine with:

```bash
python -c "from isaaclab.app import AppLauncher; AppLauncher(headless=True, enable_cameras=True)"
```

If that crashes, the machine cannot render: run the pipeline without a render flag and
render the executed episodes on a machine that can (see [Rendering](render.md)). The same
crash affects `--vecgen-record-videos` and eval video capture. Kit logs are under
`.venv/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/`.

### Harmless PhysX errors

Some otherwise clean runs print
`PxDirectGPUAPI::getRigidDynamicData(): not allowed while simulation is running`. The episodes
still succeed; ignore it.

### Isaac hangs after it finishes

`render.py` prints `all jobs done` and then hangs in Isaac teardown — observed holding a GPU for
53 minutes after finalizing all 16 episodes. Every pipeline render therefore runs through
`scripts/render/run_local_render.py`, which watches for the marker and kills the process group
after a grace period. **Never key completion on an exit code** here; key it on the log.

### Host RAM and disk, not VRAM, are the limits

VRAM is not the binding constraint when raising `--local-gpu-workers` — host RAM is, and a
host-RAM OOM mid-encode corrupts videos behind a `completed: true` marker. Raise the worker count
only if you validate the output afterwards.

### Concurrent slots share Isaac's caches

The pipeline sets `KIT_CACHE_ROOT`, `KIT_LOG_DIR` and `OMNI_USER_CONFIG_DIR` per slot, but the
installed Isaac Sim **does not read any of them**. Concurrent slots still share `~/.cache/ov`,
`~/.local/share/ov` and `~/.nvidia-omniverse`. The one per-slot variable that does take effect is
`TMP3_ENCODER_THREADS`, which stops N processes each sizing an encode pool from the whole CPU set.

### `--camera_base data` does nothing here

It tells the renderer to base each static camera on the pose the *dataset* records. The execution
stage records no camera pose at all — vecgen is parquet-only, so there is never a recorded pose to
base on. `render.py` detects this, warns loudly, and falls back to `--camera_base scene`; the
pipeline passes `scene` explicitly and drops a `data` arriving through `--render-extra`. Same
outcome, said at argv-build time instead of minutes into Kit startup.

### `--render-extra` needs the `=` form

Argparse reads a space-separated value starting with `-` as an option of its own:

```bash
--render-extra="--extra_rgb 224 --extra_depth g12_224 --depth_gop 1000"   # works
--render-extra "--skip_failure_episodes"                                  # rejected
```

It is appended last, so it beats the pipeline's own `--depth_gop`, `--dr_seeding`,
`--camera_base` and `--env_spacing`, and the default sidecars. `--output`, `--envs_dir`,
`--episodes_file`, `--episode_offset` and `--max_episodes` are dropped with a warning — they break
the pipeline's accounting rather than the render.

## See also

- [Automatic Data Generation](../data_generation.md) — the pipeline overview and every pipeline flag
- [Rendering](render.md) — the render stage and its flags
- [YAML reference](https://github.com/allenai/ToyBlocks/blob/main/src/toyblocks/domain_randomization/YAML_REFERENCE.md) — writing a randomization YAML
- [docs/source/user_guide/workflows/datagen/local.md](https://github.com/allenai/ToyBlocks/blob/main/docs/source/user_guide/workflows/datagen/local.md) — implementation notes, HDRI backdrops and invariants for the pipeline render
