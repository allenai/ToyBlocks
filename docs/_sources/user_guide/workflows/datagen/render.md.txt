# Stage 4: Rendering

Execution records what the robot and the blocks did, but it records no camera images. This stage
produces them, by replaying each recorded episode frame by frame: the robot and objects are
placed at their recorded poses, with no physics stepping and no replanning, and each frame is
rendered.

A render is described by a **flags YAML**: how the scene is set up and how it is drawn. The
file is recorded in every problem's `render_params.json`, so a dataset always carries the look
it was made with. There is no default — the look is always named.

The simplest one is
[`render_flags_plain.yaml`](https://github.com/allenai/ToyBlocks/blob/main/src/toyblocks/domain_randomization/flags/render_flags_plain.yaml),
which ships with the repo. Every key in it is fixed, so each episode renders identically:

```yaml
carb_settings: {...}     # RTX path tracing
env_spacing: 500         # how far apart the envs sit
texture_cache: {...}     # where decoded textures are kept
hide_objects: [backdrop] # the backdrop geometry is hidden...
skybox:                  # ...because this sky replaces it
  names: [abandoned_bakery_2k]
  rotation_deg: 0
  intensity: 800
  visible_background: true
```

```bash
--render-non-randomized
```

Add sampled sections to that file and the same renderer produces a different look per episode:
see [Domain randomization](#domain-randomization). Ask for more than one look in a run with
[render passes](#render-passes).

The renderer is also a plain script you can run on a single GPU of this machine, either on a
dataset that has no images yet or on one that does, replacing them. To run it as part of a
pipeline run, see [Through the pipeline](#through-the-pipeline) and
[Running locally](local.md).

## Why the pixels are a separate stage

Keeping rendering apart from execution has several benefits:

- One expensive planning-and-physics run can be rendered many times with different visuals, for
  example N copies per episode or separate train and test letter sets, **without re-planning**.
- Rendering can run later, or on a different GPU, separately from data generation.
- The render settings sampled for each episode are recorded, so evaluation can reproduce an
  episode's look.

Because execution does not spawn cameras by default, rendering is not an optional extra for a
dataset with video: it is **where the video comes from**.

### What rendering does not change

The state and action trajectory is identical across renders; only the pixels and the camera
columns differ. Rendering cannot fix a bad episode, change what the robot did, or alter the
success labels.

Rendered pixels match an original in-physics rendering closely but not exactly, because
driver and scene-version differences remain. See
[DETERMINISM.md](https://github.com/allenai/ToyBlocks/blob/main/docs/render/DETERMINISM.md)
for exactly what is bit-reproducible and what is not.

## Where it fits

- **Input:** the recorded datasets from [stage 3, execution](execution.md), the env packages
  they were recorded with, and a flags YAML describing the randomization.
- **Output:** a rendered copy of each dataset, with camera videos, read by [merge](merge.md).

The stage is optional in the pipeline, enabled with `--render-randomized`, but it is
the only way to get camera videos without recording them during execution.

## Running it

The quickest way to try it is to render a handful of episodes from an existing dataset on your
own GPU. This is also the best way to preview a flags YAML before committing to a large run:

```bash
python scripts/render/render.py <dataset_dir> \
    --env_file <env_package>/env.py \
    --output <out_dir> \
    --max_episodes 20 --num_envs 8 \
    --flags src/toyblocks/domain_randomization/flags/render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml
```

For an *unmerged* dataset, with one directory per problem, drop `--env_file`. The env is then
matched automatically per problem from `<datasets_dir>/envs/<problem_name>/env.py`. Use
`--envs_dir` when the env packages live somewhere other than next to the data. For a pipeline run
that is the experiment's own `envs/` directory:

```bash
python scripts/render/render.py data/datasets/unmerged/<experiment-id>/execution \
    --envs_dir data/datasets/unmerged/<experiment-id>/envs \
    --output <out_dir> --max_episodes 20 \
    --flags src/toyblocks/domain_randomization/flags/render_flags_allrand_texcache.yaml
```

`render.py` often hangs during Isaac teardown after it finishes; see
[Common pitfalls](#common-pitfalls) for how to run it safely.

## Through the pipeline

Add `--render-randomized` to a pipeline run. Each problem is rendered on this machine as soon as its
execution finishes, and the merge then uses the rendered episodes instead of the execution
data.

Pass `--render-non-randomized` for a plain render, `--render-randomized` for a randomized one,
or both to get both passes (see [Multiplexing](#render-passes)). Either
way `--render-flags` is required.

`--render-schedule` chooses how those renders are scheduled against the executions:

- `coupled` (the default) runs the render as an extra step in the same job as that problem's
  execution, holding one slot across both phases. Faster with more than one worker.
- `pipelined` queues each problem's render as its own item in the shared GPU-slot pool, so it
  can overlap the next execution when a slot is free, and each render gets its own log, timing
  and exit status — at some throughput cost.

Both schedules run the same renderer with the same arguments and produce identical pixels, so
this is a throughput and observability choice. [Running locally](local.md#rendering-locally)
has the measured difference and the three-state gate that decides what the merge ships.

```bash
python scripts/datagen/run_datagen_pipeline.py --render \
    --render-flags src/toyblocks/domain_randomization/flags/render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml \
    --experiment-id my_run --output-root data \
    --task-gen-script scripts/task_gen/flip.py --task-gen-count 4 \
    --vecgen-problems-per-env 1 --exec-num-envs 8 --vecgen-episodes 16 \
    --local-gpu-workers 2
```

With no other render flags this renders the way the RI training set was rendered: the
`render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml` flags, one copy per episode,
`left_camera` skipped, failure episodes skipped, and the 224-px RGB and `rg8_55_224` depth
sidecars (`--extra_rgb 224 --extra_depth rg8_55_224 --extra_rgb_GOP 2 --rgb_GOP 2`). Override any
`render.py` option with `--render-extra`, in the `=` form:

```bash
--render-extra="--extra_depth g12_224"
```

The flags YAML draws HDRI backdrops from the `polyhaven` and `indoor-2018` pools. Where to get
each, and what happens when one is missing, is in
[HDRI backdrops](local.md#hdri-backdrops).

The pipeline flags for this stage are:

| Flag | Default | Description |
|------|---------|-------------|
| `--render-randomized` | off | Run the render stage and merge the rendered episodes. Cannot be combined with `--vecgen-record-videos`. |
| `--render-schedule` | `coupled` | `coupled` runs each render inside the problem's execution job; `pipelined` queues it in the shared GPU-slot pool. Identical pixels either way. |
| `--render-local` | off | Deprecated spelling of `--render --render-schedule coupled`. |
| `--render-flags` | none — **required** | The render flags YAML, as a path readable on this machine. It describes how the scene is set up and drawn, and is recorded in `render_params.json`. Shipped in `src/toyblocks/domain_randomization/flags/`: `render_flags_plain.yaml` for a plain render, `render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml` for the RI training set's randomization. |
| `--render-n` | 1 | The number of rendered copies per source episode. |
| `--render-skip-cameras` | `left_camera` | Cameras not to render. Pass the flag with no names to render every camera. |
| `--render-skip-failure-episodes` | on | Render only successful episodes. The merge keeps only those anyway. `--no-render-skip-failure-episodes` renders both splits. |
| `--render-extra` | none | Extra `render.py` options, appended last so they override the defaults above, written as one quoted `=` string. `--output`, `--envs_dir`, `--episodes_file`, `--episode_offset` and `--max_episodes` are dropped with a warning, because the pipeline owns them. |
| `--render-non-randomized` | off | Render a plain pass: the YAML's scene with nothing sampled. On its own it is the only pass; alongside `--render-randomized` it adds one, making the merged dataset (N+1)x the source. See [Multiplexing](#render-passes). |
| `--render-non-randomized-flags` | derived | The flags YAML for that pass, instead of the derived one. |
| `--render-local-num-envs` | 8 | Envs per render wave. Must be at least `--num-scene-profiles`. |
| `--render-local-stall-timeout` | 3600 | Kill a render that prints nothing for this many seconds. |
| `--render-local-fail-on-video-defects` | off | Fail a problem whose encoder could not publish every video, instead of finalizing it as `incomplete_videos`. |

Despite their names, the `--render-local-*` flags apply to both schedules.

## Output

Every rendered split gets a `meta/render_params.json` describing how it was rendered. Its
top-level keys are:

| Key | Meaning |
|---|---|
| `rendered_at`, `render_script` | When the render ran, and which script ran it. |
| `completed` | Whether the render reached a final state, **not** whether it succeeded; see [Common pitfalls](#common-pitfalls). |
| `status` | `complete`, `incomplete_videos` or `failed`. |
| `video_defects` | The exact `(episode, feature)` videos that are missing, for a patch pass. |
| `empty_reason` | Why a finished render recorded nothing, when it recorded nothing. |
| `episodes_recorded` | The number of episodes in **this split**. |
| `episodes_total` | The expected total for the whole problem, across all splits. May be `null`. |
| `mode`, `fps` | `render-only` or `physics-step`, and the recording rate. |
| `env_file`, `source_datasets` | What was rendered, and from where. |
| `flags`, `flags_sha256`, `flags_resolved_sha256` | The randomization, the digest of the literal file, and the digest of the resolved dict. Group shards by the **resolved** digest. |
| `dr_seeding` | Which seeding mode produced these values. A consumer must branch on it. |
| `initial` | The scene as actually built: cameras, materials and lights. |
| `episodes` | Per-episode records, including `src_episode_index` and `iter`. |
| `split_group` | Present only for a `--split_groups` member. |

`completed` and `video_defects` together give three states, so a gap in the output is neither
silently accepted nor treated as a total failure:

| `completed` | `video_defects` | Meaning |
|---|---|---|
| `true` | `[]` | Whole and usable. |
| `true` | non-empty | Usable but incomplete. The listed videos are missing and need a patch pass. |
| `false` | — | The problem must be rendered. |

### Validating the output

Metadata alone cannot prove a render is good, so check the videos themselves:

```bash
python scripts/render/validate_dataset_videos.py <out_dir>
```

## Domain randomization

The plain YAML above fixes every value. Replace any of them with a **set** or a **range** and
the renderer samples instead, giving each episode its own look from the same file:

```yaml
skybox:
  set: polyhaven/rooms        # a pool of skies, not one name
  rotation_deg: [0, 360]      # a range, not 0
  intensity: [500, 1000]
materials: {...}              # table and block surfaces
scene_lights: {...}           # fill, local lights, the sun
env_sphere: {...}             # the visible backdrop sphere
randomize: {...}              # camera pose and field of view
randomize_task_letters: {...} # the glyph on each block
randomize_block_colors: {...} # cuboid colours
```

```bash
--render-randomized --render-flags <your.yaml>
```

Nothing else changes: same renderer, same stage, same output layout. The whole file is recorded
in every problem's `render_params.json` as the dataset's visual provenance.

Four presets live in `src/toyblocks/domain_randomization/flags/`:

| Preset | Use |
|---|---|
| `render_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml` | The RI training set's render and the pipeline default: `allrand_texcache` plus the train glyph split (symmetry-preserving) and per-env block colours. |
| `render_flags_allrand_texcache.yaml` | All-random lighting, materials and cameras, without glyph or colour swaps. |
| `render_flags_camtest.yaml` | A small camera preset for quick tests. It randomizes nothing; see [Common pitfalls](#common-pitfalls). |
| `render_flags_open_hybrid_camera_rand_only.yaml` | Camera randomization only. |

The two `allrand` presets sample HDRI backdrops from the `polyhaven` and `indoor-2018` pools; see
[HDRI backdrops](local.md#hdri-backdrops).

To write your own, see the
[YAML reference](https://github.com/allenai/ToyBlocks/blob/main/src/toyblocks/domain_randomization/YAML_REFERENCE.md).
It covers every top-level key, the value forms the resolver accepts, and, importantly, *when*
each value is drawn: per wave, per episode or per env.

## Render passes

One run can produce several looks of the same episodes. Each **pass** is one YAML, rendered over
every source episode into its own `job_<name>/` directory, and the merge takes them all:

```bash
--render-pass render_flags_plain.yaml
--render-pass render_flags_allrand_texcache.yaml:3
```

That renders the plain look once and the randomized look three times, so the merged dataset is
4x the source episodes. The `:N` suffix is per pass; without it a pass renders once.

`--render-non-randomized` and `--render-randomized` are shorthand for the two most common
passes, and compose with each other exactly as two `--render-pass` entries would.

Each pass is a separate renderer invocation rather than extra `-N` iterations within one,
because a process applies one flags dict before the sim is built, and the per-wave levers (dome,
skybox, exposure) are scene-global: a differently-flagged episode sharing a wave would still be
lit by that wave's draw. Separate job directories also mean each pass resumes independently, and
a problem counts as rendered only when **every** pass has landed.

## Multiplexing: N copies within one pass

Often you want several visual variants of the same trajectory, for example to make a policy
robust to lighting. `-N` replays each source episode N times, each time with freshly sampled
randomization. The state and action trajectory is identical across the copies and only the pixels
and camera columns differ, so the result has **N×** the source's episodes and frames. Each output
episode records its `src_episode_index` and `iter`.

In the pipeline, `--render-non-randomized` adds one further pass with the randomized sections of
the YAML removed, making the merged dataset **(N+1)×** the source:

```bash
--render-n 3 --render-non-randomized
```

That extra pass is a separate invocation into a sibling job directory, not an extra `-N`
iteration. One process has one flags dict, applied before the sim is built, and the per-wave
settings (dome, skybox, exposure, sun) are global to the scene. So an "un-randomized" copy sharing
a wave with randomized ones would still be lit by that wave's draw.

**"Without randomization" keeps the scene and drops the draws.** The point of a plain pass beside
randomized ones is that they are comparable: same backdrop, same renderer, only the sampling
gone. That is why it takes a YAML rather than rendering with no flags at all — dropping the YAML
would bring the backdrop geometry back and change the render mode, giving a different scene
rather than the same one unrandomized. The derived YAML therefore keeps
`cameras`, `hide_objects`, `env_spacing`, `render_mode`, `carb_settings` and `texture_cache`,
drops everything else, and passes `--enable_gi` when the source had a skybox. It is written to
`<experiment>/render/non_randomized_flags.yaml`, with a header naming what was dropped. You can
replace it with `--render-non-randomized-flags`.

A problem counts as rendered only when **every** pass has landed. Otherwise a failed original
pass would contribute N copies where N+1 were promised, and nothing would report it.

## Reproducing a recorded look

Because each episode's sampled settings are recorded, a render can be reproduced later:

```bash
python scripts/render/render_from_params.py --help
```

Pass the `dr_seeding` mode recorded in the dataset's own `render_params.json`.

## Flags

These are the options of `render.py`. `datasets_dir` is positional and `--output` is required;
everything else has a default.

### Source and output

| Flag | Default | Meaning |
|---|---|---|
| `datasets_dir` | (positional, required) | The root containing the recorded dataset or datasets. |
| `--output` | (required) | The output directory. It mirrors the structure of the input. |
| `--env_file` | `None` | The env package's `env.py`. Ignored for unmerged datasets, which are matched per problem. |
| `--envs_dir` | `None` | A root holding per-problem env packages as `<envs_dir>/<problem>/env.py`. Takes precedence over both of the above. |
| `--assets_paths` | `None` | Extra asset search roots for HDRIs and skyboxes, searched in order before the repo default (`assets/environments`). Use it when the HDRI pools live somewhere else, for example on another disk. The `RENDER_HDRI_DIR` environment variable does the same for the default root. |
| `--copy_source_data` | off | Replace the recorder's re-derived `data/` with a copy of the source's. Off by default, because the source data is canonical and can be copied or symlinked instead. |

### Episode selection

| Flag | Default | Meaning |
|---|---|---|
| `--max_episodes` | all | The maximum number of episodes rendered per source dataset. |
| `--skip_failure_episodes` | off | Render only `blocks_success`. The merge keeps only that split anyway, so for a merge-bound run the failure split is pure cost. This **renumbers the global episode index**, so such a run cannot resume an output built without it. |
| `--episodes_file` | `None` | Render only the listed episodes, for a targeted patch. Output episodes are renumbered per split, but each records its `src_episode_index`. |
| `--episode_offset` | `None` | Select a fragment for parallel rendering: the global slice `[offset, offset+max_episodes)`. Overrides `--episodes_file`. |
| `--fragment_invariant` | **on** | Keep the full per-problem episode list inside a fragment, so render-uids, wave grouping and randomization draws match a single-machine run. Reassembling the fragments then gives a merge that is byte-identical to one big run. |
| `--split_groups` | off | Render each member of a vecgen `problem_grp_*` directory as its own unit. Must match how execution was run. |

### Cameras and video streams

| Flag | Default | Meaning |
|---|---|---|
| `--skip_cameras` | `None` | Cameras **not to spawn**. They are dropped before the scene is created, so both their RGB and depth are absent and their render cost is never paid. Names work with or without the `camera:` prefix. |
| `--skip_depth` | off | Do not render or record the main depth stream. Cannot be combined with `--extra_depth`. |
| `--depth_format` | `12bit` | How the main depth is stored. `12bit` is native `gray12le`; `rg8` is the same uint12 packed into lossless 8-bit RGB, about 3x larger but much faster to read at random positions. Neither loses precision. |
| `--depth_codec` | `hevc` | `hevc` encodes about 15x faster; `av1` produces smaller files. Both are lossless 12-bit millimetres. |
| `--depth_gop` | `1` | The keyframe interval for the main depth. `1` is the friendliest for random access and the largest; higher values are smaller and faster to encode, at the cost of seeking. |
| `--rgb_GOP` | `2` | The keyframe interval for the main RGB. `0` lets SVT-AV1 choose. |
| `--extra_rgb` | `None` | Also write a 224-px RGB video per camera, in the HVLA training sidecar layout. It is resized with Pillow BILINEAR; ffmpeg `scale` is **not** equivalent. |
| `--extra_depth` | `None` | Also write a 224-px depth video per camera: `rg8_224`, `g12_224` or `rg8_55_224`. `g12_224` and `rg8_224` are lossless and decode to identical depth, `g12_224` at a fraction of the size. `rg8_55_224` packs 10 bits (R = mm >> 5, G = mm & 31) and **clips at 1023 mm**; it is what the RI training set and the pipeline default use. |
| `--extra_rgb_GOP` | `0` | The keyframe interval for the `--extra_rgb` sidecars. `0` lets SVT-AV1 choose. The pipeline passes `2`. |
| `--render_scale` | `None` | Scale the preset cameras' resolution while keeping the field of view. Requires `--skip_cameras gripper_camera left_camera`, because robot-mounted cameras are not scaled. |
| `--camera_base` | `data` | Where each camera's starting pose comes from before randomization. `data` reproduces the calibration the source was captured with; `scene` uses the built SceneConfig. Unusable recorded values fall back to `scene` **with a warning**, never silently. |
| `--match_recorded_cameras` | off | (Experimental.) Override extrinsics and intrinsics with the recorded values. Camera overrides in the flags YAML always take precedence. |

### Randomization and seeding

| Flag | Default | Meaning |
|---|---|---|
| `--flags` | `None` | The randomization YAML. |
| `-N`, `--renders_per_episode` | `1` | The number of renders per source episode. See [Multiplexing](#render-passes). |
| `--dr_seeding` | `hash` | How per-episode draws are seeded: `hash` (the default), `indexed`, or the legacy `stream`. |
| `--material_baseline_revert` | off | Reset every writable material key to the asset's own value before each draw. Without it, a key the chosen material leaves out keeps the previous episode's value, so materials bleed across episodes. Off by default because it changes the rendered output. |
| `--enable_gi` | off | Turn on RTX global illumination and reflections. Enabled automatically when the YAML has a `skybox` block. |

The seeding modes differ in how an episode's values are derived. `stream` seeds one RNG over the
whole episode list, so an episode's values depend on how many episodes come before it. `indexed`
and `hash` derive each episode independently, which is what lets a patch jump straight to a
missing episode.

### Performance and rendering

| Flag | Default | Meaning |
|---|---|---|
| `--num_envs` | `8` | The number of parallel envs, which is also the replay batch size. Use a multiple of 8, or fewer than 8. |
| `--num_writers` | `1` | Encoder processes per render process. Raise it when the encoder, not the GPU, is the bottleneck. |
| `--wave_sort_block` | `10` | Group episodes of similar length into a *wave* (a batch of episodes rendered together), within blocks of this many waves. A wave renders every env until its longest episode ends, so rendering in source order wasted about 36% of rendered env-frames on measured runs. |
| `--scene_swap` | **on** | At each problem boundary, make the live scene match that problem's block set. Costs nothing when consecutive problems share a set. `--no-scene_swap` skips mismatched problems instead. |
| `--scene_swap_warmup` | `0` | Extra RTX passes after a scene swap, so the denoiser can converge. |
| `--rtx_warmup_frames` | `1` | Extra render passes at `t==0` of each batch, so temporal renderers converge and frame 0 has no warm-up artifact. Try 8–16 if artifacts persist. |
| `--antialiasing` | preset (DLSS) | `Off`, `FXAA`, `TAA`, `DLSS` or `DLAA`. `Off` and `FXAA` work on single frames; the others carry a temporal history that is empty on frame 0. |
| `--disable_dlss` | off | A shortcut for `--antialiasing FXAA`. Ignored if `--antialiasing` is given. |
| `--skip_initial_step` | off | Skip the physics settle at construction time. Safe, because the replay overrides all state on every frame. |
| `--physics_step` | off | Run one physics step per frame instead of rendering only. This removes occasional duplicate frames at the cost of slight drift. The default is to place everything and render, with no physics. |
| `--env_spacing` | `None` (5.0) | Override the scene's env spacing. Larger spacing stops a neighbouring env's backdrop from leaking into a camera's peripheral view when a skybox is used. |

### Resume, integrity and debugging

| Flag | Default | Meaning |
|---|---|---|
| `--resume` | off | Skip problems whose output already has `render_params.completed == true`. See [Common pitfalls](#common-pitfalls) before using it. |
| `--fail_on_video_defects` | off | Mark a problem incomplete, forcing a full render, if any video could not be published. Off by default: the problem stays complete with `status=incomplete_videos` and an exact list of missing videos for a patch pass (`scripts/render/fast_patch.py`). |
| `--passthrough_source_columns` | **on** | Copy every column from the matching source episode, so the two datasets differ only where the render is authoritative. |
| `--patch_actions_from_source` | **on** | Copy the source's `action.*` columns. Superseded by the flag above. |
| `--dump_scene` | off | Print the stage layout and exit. |
| `--dump_probe` | `None` | Write the applied material values, live carb settings and render products to JSON after the first wave, then continue. |
| `--show_gizmos` | off | Keep debug visualization markers in the render. Hidden by default. |

## Common pitfalls

### `render.py` hangs after it finishes

`render.py` prints `all jobs done` and then often **hangs in Isaac teardown**. Run it through
`scripts/render/run_local_render.py`, which watches for that marker and kills the process group
30 s later; the pipeline always does. Never judge success from the exit status; judge it from the
marker and the result on disk.

```bash
python scripts/render/run_local_render.py -- python scripts/render/render.py <dataset_dir> ...
```

### `completed: true` does not mean the render produced data

`completed: true` means the render reached a final state, not that it *has data*. It is written
so that a re-run does not retry the problem forever, so it is also written when the recorded block
size is not in the problem's size table, or when every source parquet is unreadable. Those cases write a
`render_params.json` but **no `info.json`**, and `info.json` is what merge actually needs. Note
also that there is no `total_episodes` key; the problem-wide count is `episodes_total`.

### A broken render can still look complete

**Metadata cannot see a broken render.** A host-RAM out-of-memory error during encoding produces
videos that fail with `moov atom not found`, inside a dataset whose `render_params.json` still
says `completed: true`. Always run `validate_dataset_videos.py`, shown under [Output](#output),
before using a rendered dataset.

### `render_flags_camtest.yaml` produces identical copies

`render_flags_camtest.yaml` has **no random section at all**. An `-N 4` run with it produces
four pixel-identical copies, while every size-based check still reports 4x. Use it only to test
the camera rig, never to check that randomization works.

### Mixing seeding modes in one dataset

**Never mix two `--dr_seeding` modes within one dataset**, because each produces different
values. To reproduce or patch an existing dataset, pass the mode recorded in its
`meta/render_params.json`.

### Two renders into the same output delete each other's work

`--resume` **`rmtree`s a non-finalized `<output>/<problem>`**. So two renders pointed at the same
problem directory delete each other's work rather than failing. Never run two renders against the
same output problem directory.

### Checking `-N` by file size gives the wrong answer

To confirm that `-N` really multiplied the dataset, count episodes, frames and distinct video
files, **never bytes**. Merge hardlinks videos into the merged tree, so `du` of the merged
directory is meaningless, and different draws compress differently (measured at 2.055× for N=2,
not 2×).

The failure a size check cannot catch is N copies that are pixel-identical because nothing was
randomized. Check that the copies actually *differ*: there should be distinct
`observation.camera_pose.*` values per episode under camera randomization, and differing
**depth** frames. Depth is deterministic and RGB is not, so RGB can prove a change but never its
absence.

## Next

[Merge](merge.md) the rendered problems into one training dataset. Once this stage is marked
complete, the merge reads `render/out/job_*` and nothing else, so a problem that was never
rendered simply will not appear. See
[what counts as "rendered"](local.md#what-counts-as-rendered) for how the local flow guards
against that.

## See also

- [Running locally](local.md): the local pipeline, and the gate that decides what gets merged
- [Execution](execution.md): the stage that produces the trajectories this stage replays
- [Merge](merge.md): how rendered problems become one training dataset
- [YAML reference](https://github.com/allenai/ToyBlocks/blob/main/src/toyblocks/domain_randomization/YAML_REFERENCE.md): every randomization key
- [DETERMINISM.md](https://github.com/allenai/ToyBlocks/blob/main/docs/render/DETERMINISM.md): what is reproducible, and what is not
- [Running locally](local.md): the two pipeline render modes, HDRI backdrops, and the invariants behind the gate
