# Running Pipeline Locally

`scripts/datagen/run_datagen_pipeline.py` runs the whole data generation pipeline on the machine
you launch it on: PDDL problems, env packages, executed episodes, rendered videos and a merged
LeRobot dataset, all on local disk and local GPUs. The same command works on a single-GPU
workstation and on a multi-GPU node. The pipeline spreads problems over every GPU it can see.

This page covers what is specific to running it: asking for renders, choosing GPUs and
concurrency, where outputs and logs go, resuming, and checking the result. For the
stage-by-stage reference, see [Automatic Data Generation](../data_generation.md).

| Stage | How it runs |
|---|---|
| task-gen | one subprocess, on the CPU |
| env | one process per problem, across the GPU slots (skipped with `--vecgen-problems-per-env` > 1, the default) |
| execution | one process per problem or problem group, across the GPU slots |
| render | one render per problem per pass (`--render-pass`), scheduled into the shared GPU-slot pool or inside each execution job |
| merge | one subprocess, on the CPU |
| annotate | one subprocess, on the CPU |

The pipeline needs no network access once the assets are installed. Run it from the root of the
checkout: the stage scripts are called by relative path (`scripts/...`), and the pipeline runs
`git rev-parse HEAD` to pin the commit into the run config for reproducibility.

## Before you start

Follow [the installation instructions](../installation_and_setup_instructions.md). With the
virtual environment active, install the datagen (vecgen) extra and fetch the assets:

```bash
uv pip install -e ".[dev,isaac,torch-cu128,vecgen]"
```

```bash
get-toyblocks-assets --asset-manifest asset_manifest.json
```

Randomized rendering draws HDRI backdrops from the polyhaven pool, installed with the base
assets, and from the separately licensed Laval indoor set. See
[HDRI backdrops](#hdri-backdrops).

Rendering needs real disk. A few problems rendered three times over write several GB of video,
so check free space before a long run rather than discovering it mid-encode.

## A complete run

This generates 4 stacking problems, executes 16 episodes each, renders them with domain
randomization, and merges the result into one LeRobot dataset under `./outputs`:

```bash
python scripts/datagen/run_datagen_pipeline.py \
    --render-pass src/toyblocks/domain_randomization/flags/render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml \
    --experiment-id my_run --output-root outputs \
    --task-gen-script scripts/task_gen/stack_three.py --task-gen-count 4 \
    --env-per-problem 16 --num-envs 16 --exec-num-envs 16 \
    --vecgen-problems-per-env 1
```

The merged dataset is annotated at the end, because the render supplies the camera data that
annotation projects goals into. Drop the `--render-pass` to stop after execution and merge the
raw parquet, which holds state and actions but no video. The pipeline then logs a warning and
skips annotation, which needs video.

With no flags beyond `--experiment-id` the pipeline runs its full default configuration: 1000
`stack_three.py --train` problems, randomized scenes, 32 reset states per problem and 8 problems
per env package at 128 envs. That is a large run, so pass the task-gen and env flags above for
something smaller.

Each stage also runs on its own. See [task generation](task_generation.md),
[env generation](env_generation.md), [execution](execution.md), [rendering](render.md),
[merge](merge.md) and [annotate](annotate.md).

## Rendering locally

Execution records state and actions only, because vecgen spawns no cameras. The videos come from
the render stage, which replays the recorded states with no physics and draws them under the
settings in a flags YAML. See [Rendering](render.md) for what that stage does and every flag it
takes, and the
[YAML reference](https://github.com/allenai/ToyBlocks/blob/main/src/toyblocks/domain_randomization/YAML_REFERENCE.md)
for how to write one.

**`--render-pass FLAGS.yaml[:N]` is how you ask for a render.** Repeat the flag for more looks:

```bash
python scripts/datagen/run_datagen_pipeline.py \
    --render-pass src/toyblocks/domain_randomization/flags/render_flags_plain.yaml \
    --render-pass src/toyblocks/domain_randomization/flags/render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml:3 \
    --experiment-id my_run --output-root outputs
```

Each pass renders **every** source episode under one YAML, into its own
`render/out/job_<name>/` directory named after that YAML. The optional `:N` renders N copies of
each episode within the pass, resampling the randomization for each copy. The merge reads every
`job_*` directory, so the run above ships 1 + 3 = 4 merged episodes per executed episode.

Each pass is a separate renderer invocation rather than extra copies inside one, because the
per-wave settings (dome, skybox, exposure) are global to the scene, so two differently-flagged
looks cannot share a wave. [Render passes](render.md#render-passes) covers this in more detail.

### Which flag does what

| Flag | Default | What it does |
|---|---|---|
| `--render-pass FLAGS.yaml[:N]` | none | Adds one pass. Repeatable. |
| `--render-flags FLAGS.yaml` | none | Names the YAML that the two shorthand flags below use. It adds no pass on its own. |
| `--render-randomized` | off | Shorthand for one pass of `--render-flags`, repeated `--render-n` times. |
| `--render-n N` | `1` | Copies per source episode for `--render-randomized`. |
| `--render-non-randomized` | off | Shorthand for one pass of `render_flags_plain.yaml`, which needs no YAML of your own. |
| `--render-non-randomized-flags` | `render_flags_plain.yaml` | A different YAML for that pass. |

The two shorthands name the common cases, and they compose with each other and with
`--render-pass` exactly as further `--render-pass` entries would. These two commands are
equivalent:

```bash
--render-randomized --render-flags path/to/dr.yaml --render-n 3 --render-non-randomized
```

```bash
--render-pass path/to/dr.yaml:3 --render-pass src/toyblocks/domain_randomization/flags/render_flags_plain.yaml
```

**`--render-flags` on its own renders nothing.** It carries a value for the shorthands rather
than requesting a pass, so a command with `--render-flags` and no `--render-randomized` runs the
pipeline with the render stage off. Older commands pair the two, because `--render-randomized`
was once the only way to render. `--render`, `--render-local` and the `--rerender*` spellings are
superseded, and the pipeline says so when it sees them.

**With no render flag at all the stage is off.** The merge then takes the execution data
directly, the dataset holds state and actions with no video, and annotation is skipped with a
warning, because projecting goals into the front camera needs a camera.

**Asking for a render that resolves to no pass is an error.** `--render-randomized` without
`--render-flags` stops the run before any work and names the shipped YAMLs, rather than
quietly skipping the stage.

A render cannot be combined with `--vecgen-record-videos`. Recording in-physics video and then
overwriting it pays the slow physics-render cost for frames nothing reads.

### Parity across passes

Passes are symmetric over episodes. Every pass renders the same source episodes, so the merged
dataset holds (sum of the copies across passes) x (source episodes), and every output episode
records the `src_episode_index` it came from. A problem counts as rendered only when **every**
pass has landed on disk, checked by the gate in
[What counts as "rendered"](#what-counts-as-rendered). Without that check a half-finished
problem would contribute 3 copies where 4 were promised and nothing would report it.

Two passes whose YAML file names reduce to the same `job_<name>` would write to the same
directory, so the pipeline keeps the first and warns.

### How renders are scheduled

`--render-schedule` chooses how renders are scheduled against the executions. Both modes run
locally and produce **the same pixels**, because the renderer argv, the staged source layout and
the per-problem output directory name (which seeds the randomization) are identical.
The choice is about throughput and observability.

- `coupled` (the default) appends the render to that problem's execution job, holding one slot
  across both phases. Exec and render are one process span, so they share a log and an exit code.
- `pipelined` queues each render as its own item in the shared GPU-slot pool once that problem's
  execution finalizes, so it can overlap the next execution and gets its own log, timing and
  exit status.

Measured on one RTX 5090 over 4 problems, timing the execution and render window only:

| | 1 slot | 2 slots (`--local-gpu-workers 2`) |
|---|---|---|
| coupled | 921 s | **667 s** |
| pipelined | 906 s | 733 s |

**For throughput, add `--local-gpu-workers 2`**, which is 28% faster than a single coupled slot
and needs none of the scheduler. Pipelining costs nothing at one slot but loses about 10% at
two, because its render-first ordering pairs render against render, two GPU-saturating
processes, whereas the coupled chains naturally pair one problem's execution against another's
render.

Use `pipelined` when you want separable exec and render timings and logs, failure isolation
between the two, or a first finished problem early rather than at the end of a chain.

### Options that matter on one machine

| Flag | Default | Meaning |
|---|---|---|
| `--render-local-num-envs` | `8` | Envs per render wave, under either schedule. A **floor**, not a preference: the render spawns env *i* at block-size slot *i* % *profiles*, so a value below `--num-scene-profiles` leaves the high slots without an env and **every problem is skipped**. The pipeline refuses that pairing up front. 8 keeps the encoder's memory use survivable beside Isaac. |
| `--render-local-stall-timeout` | `3600` | SIGKILL the render if it prints nothing for this many seconds. |
| `--render-local-fail-on-video-defects` | off | Fail a problem whose encoder could not publish every video. Off by default because a failed problem renders whole, and `scripts/render/fast_patch.py` fills video holes afterwards. |

Despite their names, the `--render-local-*` flags apply to both schedules.
`--render-skip-cameras` and `--render-extra` are described in [Rendering](render.md).

### Render arguments the pipeline pins for you

The pipeline builds its own `render.py` command line, and three of its arguments differ from
`render.py`'s own defaults. They are worth knowing if you are comparing a pipeline render
against one you ran by hand:

| Argument | Pipeline passes | `render.py` default |
|---|---|---|
| `--env_spacing` | `60` | unset (scene default `5.0`) |
| `--camera_base` | `scene` | `data` |
| `--rtx_warmup_frames` | `16` | `1` |

`--num_envs` comes from `--render-local-num-envs`. `--render-extra` is appended after these and
wins over them, apart from the exceptions noted under
[`--render-extra` needs the `=` form](#troubleshooting).

## HDRI backdrops

The `allrand` flags light each episode with an HDRI backdrop sampled from two pools:
`env_sphere.sets: [polyhaven/all, indoor-2018/all]`. The base asset download includes polyhaven,
and indoor-2018 is installed separately. Each pool is a folder `<root>/<pool>/` holding the
`.hdr` files plus a `dataset.json` that lists the files in each named set (`all`, `rooms`, and
so on). `<root>` is `assets/environments/`, or the directory in `RENDER_HDRI_DIR` when that is
set.

| pool | what | size | license | how to get it |
|---|---|---|---|---|
| `polyhaven` | 299 Poly Haven HDRIs at 2k (`all` = 134 of them) | 1.8 GB | CC0 (credits in its `ATTRIBUTION.md`) | included in `get-toyblocks-assets --asset-manifest asset_manifest.json` |
| `indoor-2018` | Laval Indoor HDR Dataset (Gardner et al. 2017), 2,113 panoramas as 4096x2048 `.hdr` | 44 GB | research / non-commercial | request it from Université Laval ([indoor.hdrdb.com](http://indoor.hdrdb.com)), it is not redistributed |

`get-toyblocks-assets` installs the polyhaven pool under `assets/environments/polyhaven`. For
indoor-2018, convert the panoramas to 4096x2048 Radiance `.hdr` and put them in
`<root>/indoor-2018/` together with a `dataset.json` whose `sets.all` lists the file names:

```json
{"source": "Laval Indoor HDR Dataset 2018", "format": "hdr", "sets": {"all": ["9C4A0003-e05009bcad.hdr", "..."]}}
```

With only polyhaven installed the render still runs. The indoor-2018 bucket resolves to nothing
(`[dr] WARNING skybox bucket resolved 0 textures`), so every backdrop comes from polyhaven and
the images differ from data rendered with both pools. The log line `[render] asset map:` shows
which pools were found.

## Choosing GPUs and how many problems run at once

| Flag | Default | Meaning |
|---|---|---|
| `--local-gpus ID [ID ...]` | every GPU visible to the process | Which GPU ids the stages may use. |
| `--local-gpu-workers N` | `1` | Concurrent problems **per GPU**. `1` is a strict no-op. |

The pipeline builds one *slot* per GPU per worker, so `len(gpus) * workers` problems run at once.
Each slot owns one `CUDA_VISIBLE_DEVICES` value, so every job sees a single GPU. A stage scales
out by spreading problems over more slots, never by giving one problem several GPUs. Listing a
GPU id twice in `--local-gpus` gives it two slots, like `--local-gpu-workers 2` does.

GPU discovery honours an existing `CUDA_VISIBLE_DEVICES`. Failing that it asks `nvidia-smi`, and
if that is unavailable it runs one problem at a time on the default GPU.

A failing problem does not abort the stage. The stage keeps going and reports the failure count
at the end, and the next stage works with the problems that succeeded.

### How much fits

Measured on one RTX 5090 (32,607 MiB) at `--render-local-num-envs 8`:

| | peak VRAM | peak GPU util | mean util while busy |
|---|---|---|---|
| render | 7,673 MiB | 76% | — |
| vecgen execution | 5,198 MiB | 88% | 61% |

Two concurrent processes fit comfortably on a 32 GiB card and one does not saturate it, so
`--local-gpu-workers 2` is usually free throughput. Host RAM and disk, not VRAM, are what bind
first, so validate the videos afterwards (see [Troubleshooting](#troubleshooting)).

## Logs

With a single slot, every job's output is streamed to your console. With more than one slot, each
problem's output goes to its own log file instead, because concurrent Isaac Sim logs are
unreadable interleaved:

```text
<base>/local_logs/env/<slug>.log
<base>/local_logs/execution/<slug>.log
<base>/local_logs/render/<slug>.log
```

The pipeline logs the directory when it does this, and a `FAILED <stage>: <slug> (see ...)` line
names the log of each problem that failed. `[sched] <stage> start|end <slug> slot=... t=...` lines
record when each job ran, which is how per-stage timings are measured. The pipeline's own log goes
to the console only, so pipe it through `tee` if you want to keep it.

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
│   ├── src/            # per-problem staging links the renders read from
│   └── out/            # one job_<name> directory per render pass, named after its flags YAML
├── local_logs/         # per-problem logs, only when >1 slot runs at a time
├── .pipeline_state/    # <stage>.done markers that --resume reads
└── run_datagen_pipeline_config_<timestamp>.json
```

A pass run with `render_flags_plain.yaml` lands in `render/out/job_plain`, and one run with
`render_flags_allrand_texcache.yaml` lands in `render/out/job_allrand_texcache`. The merged
dataset gets a copy of the run config and a plot of its first episode, `episode_0.png`.

## What counts as "rendered"

Once the render stage is marked complete, **merge reads only `render/out/job_*`**. A problem
that executed but was never rendered would simply vanish from the merged dataset. To stop that
being silent, a post-stage sweep classifies every problem on disk into three states:

| State | Meaning | Retried? |
|---|---|---|
| `RENDER_SHIPPED` | finalized under `blocks_success` **and** merge can read it (`info.json` present) | no, it is done |
| `RENDER_UNRENDERABLE` | finalized, but it can never ship: it declared itself empty, recorded zero episodes, or finalized with only a `blocks_failure` split | no, it is permanent, and failing on it would wedge every later run |
| `RENDER_ABSENT` | not finalized, or finalized without a readable dataset | yes, the run exits non-zero and a later run or `--resume` finishes it |

Two of these are worth understanding. `completed: true` in `render_params.json` means *terminal*,
not *has data*. And a problem whose episodes all failed finalizes with `blocks_failure` alone, so
no `blocks_success` is ever created. Re-running cannot change that, because the source has no
successful episodes.

With more than one pass a problem takes the worst state across its passes, so every pass has to
land before that problem ships.

Both schedules report from this sweep rather than from the job list, so it also covers problems
the execution loop skipped and runs with `--skip-execution`. Pass
`--merge-allow-missing-problems` to merge what rendered and drop the rest.

## Flags that do not do what their name suggests

| Flag | What actually happens |
|---|---|
| `--num-reset-states` | **Removed.** It never had an effect, because the env stage passes `--env-per-problem` through as the generator's `--num_reset_states`, so passing it is now an error rather than a silent no-op. Use **`--env-per-problem`** (default `32`). |
| `--env-per-problem` | Despite the name, this is the number of **reset states** per problem, not a number of environments. The environment count is `--num-envs` (sampling) or `--exec-num-envs` (execution). |
| `--num-scene-profiles` | The count is **baked into the env package at codegen time**, so changing it does nothing for a problem whose package already exists. See below. |
| `--seed` | With `--vecgen-problems-per-env 1`, it does not reach env generation: the env stage is hard-wired to `ENV_GEN_SEED = 0`, so re-running with a different `--seed` regenerates **identical reset states** and `--seed` affects execution only. With multi-problem packages (the default), the package for the group starting at problem *g* is sampled with `--seed + g`, so `--seed` changes the reset states too. |
| `--vecgen-episodes` | Only used with `--vecgen-problems-per-env 1`. A multi-problem job runs `--env-per-problem` episodes per problem in its group. |

The env-count constraint is "a multiple of the profile count, **or** strictly smaller than it".
`--num-envs 4` against 8 profiles is legal and spawns only the low slots. It is not "must be a
multiple".

With `--vecgen-problems-per-env > 1` the env stage is **switched off entirely**, because the
execution jobs build the multi-problem packages themselves with
`scripts/vecgen/make_superset_env.py`.

Env generation also **skips any problem whose `reset_states.pkl` already exists**, and a
multi-problem job skips building a package whose `problems.json` exists. That is what makes
re-running cheap, and what makes a stale env package silently survive a parameter change.
Delete the problem's env directory, or use a fresh `--experiment-id`, when you change env-stage
parameters. This applies to `--num-scene-profiles` in particular: the profile count is a property
of the package, not of the run, so re-running an existing experiment with a new count keeps the
old bake.

## Resuming

`--resume` continues a previous run of the same `--experiment-id`. Stages already marked complete
in `<base>/.pipeline_state/` are skipped, and the first unfinished stage is rerun **from scratch,
deleting its outputs** first. It is a no-op if the experiment directory does not exist.

What that deletes matters for long runs. If the execution stage was interrupted, `--resume`
deletes `execution/` and `render/`, including the problems that had already finished. The
render stage is the exception: it deletes nothing, and `render.py --resume` renders only the
problems not yet completed, pass by pass. A run interrupted after its executions finalized
therefore resumes straight into `render` instead of silently merging raw parquet.

A stage is marked complete when it finishes, even if some of its problems failed, so `--resume`
does not retry failed problems. To keep the finished problems and run only the missing or failed
ones, rerun the same command **without** `--resume` and with `--skip-task-gen`, otherwise task
generation writes a fresh batch of problems next to the old ones. Each stage then skips the
problems it already has.

## Verify the output

**Metadata cannot see a broken render.** A dataset can carry `completed: true` and still contain
videos no decoder will open. A host-RAM OOM mid-encode produces `moov atom not found` mp4s behind
a perfectly healthy-looking marker. Validate the pixels after any run above one worker, once per
pass:

```bash
python scripts/render/validate_dataset_videos.py <base>/render/out/job_<name> --frames
```

`--frames` is what decodes the videos and compares their frame counts to the episode lengths.
Without it only the structural check runs, which catches an unopenable file but passes a short
one.

To confirm that a `:N` suffix genuinely multiplied, count episodes, frames and distinct video
files, never bytes. Merge hardlinks videos into the merged tree, so `du` of the merged directory
is meaningless, and different random draws compress differently. See [Rendering](render.md).

## Troubleshooting

### Rendering segfaults at startup on some GPU and driver combinations

On an RTX 5090 with driver 595.99.02, every local render died about 7 s into Isaac startup with
a SIGSEGV in `librtx.scenedb.plugin.so`, while env generation and execution ran fine on the same
GPU. This is an Isaac Sim 5.1 RTX-renderer problem, not a ToyBlocks one. Check your machine with:

```bash
python -c "from isaaclab.app import AppLauncher; AppLauncher(headless=True, enable_cameras=True)"
```

If that crashes, the machine cannot render. Run the pipeline without a render pass and render the
executed episodes on a machine that can (see [Rendering](render.md)). The same crash affects
`--vecgen-record-videos` and eval video capture. Kit logs are under
`.venv/lib/python3.11/site-packages/isaacsim/kit/logs/Kit/Isaac-Sim/5.1/`.

### Harmless PhysX errors

Some otherwise clean runs print
`PxDirectGPUAPI::getRigidDynamicData(): not allowed while simulation is running`. The episodes
still succeed, so ignore it.

### Isaac hangs after it finishes

`render.py` prints `all jobs done` and then hangs in Isaac teardown, observed holding a GPU for
53 minutes after finalizing all 16 episodes. Every pipeline render therefore runs through
`scripts/render/run_local_render.py`, which watches for the marker and kills the process group
after a grace period. **Never key completion on an exit code** here. Key it on the log.

### Host RAM and disk, not VRAM, are the limits

VRAM is not the binding constraint when raising `--local-gpu-workers`. Host RAM is, and a
host-RAM OOM mid-encode corrupts videos behind a `completed: true` marker. Raise the worker count
only if you validate the output afterwards.

### A job directory from an earlier run is merged again

The merge globs every `job_*` directory under `render/out/`, not only the passes this command
asked for. Re-running an experiment with fewer passes therefore still ships the older pass, and
the merged dataset stays larger than the command says. Delete the job directory you no longer
want, or use a fresh `--experiment-id`.

### Concurrent slots share Isaac's caches

The pipeline sets `KIT_CACHE_ROOT`, `KIT_LOG_DIR` and `OMNI_USER_CONFIG_DIR` per slot, but the
installed Isaac Sim **does not read any of them**. Concurrent slots still share `~/.cache/ov`,
`~/.local/share/ov` and `~/.nvidia-omniverse`. The one per-slot variable that does take effect is
`TMP3_ENCODER_THREADS`, which stops N processes each sizing an encode pool from the whole CPU set.

### `--camera_base data` does nothing here

It tells the renderer to base each static camera on the pose the *dataset* records. The execution
stage records no camera pose at all, because vecgen is parquet-only, so there is never a recorded
pose to base on. `render.py` detects this, warns loudly, and falls back to `--camera_base scene`.
The pipeline passes `scene` explicitly and drops a `data` arriving through `--render-extra`. Same
outcome, said at argv-build time instead of minutes into Kit startup.

### `--render-extra` needs the `=` form

Argparse reads a space-separated value starting with `-` as an option of its own:

```bash
--render-extra="--extra_rgb 224 --extra_depth g12_224 --depth_gop 1000"   # works
--render-extra "--skip_failure_episodes"                                  # rejected
```

It is appended after the pipeline's own arguments, so it beats `--depth_gop`, `--dr_seeding`,
`--env_spacing` and the default sidecars. Three things it cannot change. `--output`,
`--envs_dir`, `--episodes_file`, `--episode_offset` and `--max_episodes` are dropped with a
warning, because they break the pipeline's accounting rather than the render.
`--camera_base data` is dropped for the reason above. And each pass re-applies its own `--flags`
and `-N` afterwards, so an extra carrying either cannot rewrite one pass into another.

## See also

- [Automatic Data Generation](../data_generation.md) for the pipeline overview and every pipeline flag
- [Rendering](render.md) for the render stage, render passes and every render flag
- [Merge](merge.md) for how the rendered passes become one dataset
- [YAML reference](https://github.com/allenai/ToyBlocks/blob/main/src/toyblocks/domain_randomization/YAML_REFERENCE.md) for writing a randomization YAML
