# Re-rendering locally, as part of the datagen pipeline

Two ways to re-render on the machine running the pipeline. Both produce the same pixels — the
rerender.py argv, the staged source layout, and the per-problem output directory name (which is the
domain-randomization seed salt) are identical, so a problem draws the same randomization either way.

| | `--rerender-local` (coupled) | `--rerender` (pipelined) |
|---|---|---|
| where the render runs | inside the execution job's own shell chain, `vecgen && rerender` | its own item in the shared GPU-slot pool |
| when it becomes eligible | immediately, holding the slot its execution just used | the moment that problem's execution finalizes |
| can overlap the next execution | no — one job, one slot | yes, whenever a slot is free |
| exec vs render timing | not separable (one process span) | separate items, separate logs, separate `[sched]` lines |
| a render failure | indistinguishable from an execution failure in the job's exit code | its own item with its own log |

Both modes run the same post-stage sweep and the same gate, so they fail identically: a problem that
executed but did not render exits the run non-zero with the marker unset, and a later run (or
`--resume`) finishes it. Neither mode merges a short set unless `--merge-allow-missing-problems`.

## Quick start

```bash
python scripts/datagen/run_datagen_pipeline.py --rerender-local \
  --experiment-id my_run --output-root data \
  --task-gen-script scripts/task_gen/stack_three.py --task-gen-count 1 \
  --env-per-problem 16 --num-envs 16 --exec-num-envs 16 \
  --vecgen-problems-per-env 1 --rerender-local-num-envs 16
```

The render defaults are the RI training set's: `--rerender-flags`
`rerender_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml`, one copy per episode
(`--rerender-n 1`), the 224 RGB and `rg8_55_224` depth sidecars, `left_camera` skipped and
failure episodes skipped. `--rerender-extra` overrides any rerender.py option (use the `=` form).
The flags draw HDRI backdrops from two pools; see [HDRI backdrops](#hdri-backdrops).

Swap `--rerender` for `--rerender-local` to get the coupled mode. They are alternatives; passing
both is refused.

## HDRI backdrops

The `allrand` flags (including the default) light each episode with an HDRI backdrop sampled from
two pools: `env_sphere.sets: [polyhaven/all, indoor-2018/all]`. The base asset download includes
polyhaven; indoor-2018 is installed separately.
Each pool is a folder `<root>/<pool>/` holding the `.hdr` files plus a `dataset.json` that lists
the files in each named set (`all`, `rooms`, ...). `<root>` is `assets/environments/`, or the
directory in `RERENDER_HDRI_DIR` when that is set.

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
`[rerender] asset map:` shows which pools were found.

## Which mode to use

Measured on this box (4 problems, 2 reps, execution+re-render window only — see
[BENCHMARK_local.md](BENCHMARK_local.md)):

| | 1 slot | 2 slots (`--local-gpu-workers 2`) |
|---|---|---|
| coupled (`--rerender-local`) | 921s | **667s** |
| pipelined (`--rerender`) | 906s | 733s |

**For throughput, use `--rerender-local --local-gpu-workers 2`** — 28% faster than the default and
it needs none of the scheduler. Pipelining costs nothing at one slot but *loses* ~10% at two,
because its render-first ordering pairs render against render (two GPU-saturating processes)
whereas the coupled chains naturally pair one problem's execution against another's render.

Use the pipelined mode when you want separable exec/render timings and logs, failure isolation
between the two, a first finished problem at ~250s rather than at the end of a chain, or bounded
un-rendered disk.

## Running more than one problem at a time

`--local-gpu-workers N` gives each GPU N slots, so N problems run concurrently. Default is 1, which
is a strict no-op.

```bash
--local-gpus 0 --local-gpu-workers 2     # two concurrent jobs on GPU 0
```

Measured on one RTX 5090 (32,607 MiB) at `--rerender-local-num-envs 8`:

| | peak VRAM | peak GPU util | mean util while busy |
|---|---|---|---|
| re-render | 7,673 MiB | 76% | — |
| vecgen execution | 5,198 MiB | 88% | 61% |

So two concurrent processes fit comfortably and a single one does not saturate the card. **But VRAM
is not the binding constraint — host RAM is**, and a host-RAM OOM mid-encode produces
`moov atom not found` videos inside a dataset whose `render_params.json` still says
`completed: true`. Validate the output of any run above 1 worker:

```bash
python scripts/rerender/validate_dataset_videos.py <base>/rerender/out/job_local
```

### What is *not* isolated between slots

`run_local_stage` sets `KIT_CACHE_ROOT`, `KIT_LOG_DIR` and `OMNI_USER_CONFIG_DIR` per slot, but
**the installed Isaac Sim does not read any of them** (grepping the whole `isaacsim` tree finds zero
references; a real variable such as `CARB_APP_PATH` finds 14). Concurrent slots therefore share
`~/.cache/ov`, `~/.local/share/ov` and `~/.nvidia-omniverse`. The one per-slot variable that does
take effect is `TMP3_ENCODER_THREADS`, which stops N processes each sizing an encode pool from the
whole CPU set.

## How the pipelined mode is wired

`LocalRerenderScheduler` has two drive sites:

1. **During the execution stage** — `arm()` loads the execution items into a `JobFeed`, and
   `on_done` queues a problem's render as soon as its execution finalizes.
2. **A post-stage sweep** — renders anything the execution stage did not hand over. This exists
   because `apply_resume()` disables every stage before the first incomplete one, so a resumed run
   that lands on `rerender` has `enabled["exec"] == False` and the execution block never runs at all.

`JobFeed` is a priority queue the stage keeps pulling from while it runs; re-render items are served
ahead of queued executions, so at one slot the interleaving matches the coupled chain and at most one
un-rendered execution dataset sits on disk at a time.

**Back-pressure is load-bearing.** Each slot runs a worker loop that pulls the next job only when it
is free. The obvious alternative — iterating the feed and `pool.submit`-ing each item — drains the
whole feed into the thread pool's own FIFO queue before anything runs, which destroys the priority
ordering entirely: every execution runs before any render, and N un-rendered datasets pile up on
disk instead of 1.

## Multiplexing: N randomized copies, optionally plus the original

`-N` (`--rerender-n`) replays each source episode N times with freshly sampled randomization. The
state/action trajectory is identical across the copies; only the pixels and the camera columns
differ, so the merged dataset is **N x** the source in episodes and frames.

`--render-non-randomized` (off by default) adds one more pass with the randomized sections of
the flags YAML removed, making the merged dataset **(N+1) x**.

```bash
--rerender-flags .../rerender_flags_allrand_texcache.yaml --rerender-n 3 --render-non-randomized
```

**It is a separate `rerender.py` invocation into a sibling job dir, not an extra `-N` iteration.**
Three things force that: one process has one flags dict, applied to `env_cfg` before the sim is
built; the per-wave levers (dome, skybox, exposure, sun) are scene-global and the wave packers
interleave `iter` across waves, so an "un-randomized" unit sharing a wave would still be lit by
that wave's draw; and `--resume` SKIPS a problem whose `render_params` says completed, so a second
pass into the same `--output` renders nothing and exits 0. The sibling is named `job_local_nonrand`,
and since merge globs `rerender/out/job_*` it is picked up with no merge change.

**"Without domain randomization" keeps the scene and drops the draws.** Dropping the YAML entirely
is a different scene, not the same scene unrandomized: the backdrop geometry comes back, GI and
reflections go off (they are on only because a `skybox:` block is PRESENT), and `render_mode` and
`env_spacing` revert. So the derived flags keep `cameras`, `hide_objects`, `env_spacing`,
`render_mode`, `carb_settings` and `texture_cache`, drop everything else, and pass `--enable_gi`
when the source had a skybox. The derived YAML is written to `<base>/rerender/non_randomized_flags.yaml`
with a header naming exactly what was dropped. `--render-non-randomized-flags` overrides it.

A problem counts as rendered only when EVERY pass has landed -- otherwise a failed original pass
would contribute N copies where N+1 were promised and nothing would report it.

### Checking that N actually multiplied

Count episodes, frames and distinct video files -- **not bytes**. Merge hardlinks videos into the
merged tree, so `du` of the merged directory is meaningless, and different draws compress
differently (measured 2.055x for N=2, not 2x).

The failure that counts cannot see: N copies that are pixel-identical because nothing was
randomized. `rerender_flags_camtest.yaml` is the one shipped YAML of 117 with no random section at
all, so an N=4 run with it produces four identical copies and every size check still reads 4x.
Verify the copies differ -- distinct `observation.camera_pose.*` per episode under camera DR, and
differing DEPTH frames (depth is deterministic; RGB is not, so RGB can prove a change but never
its absence).

## Invariants worth knowing before changing this

- **Dispatch must be exactly-once.** `rerender.py --resume` `shutil.rmtree`s a non-finalized
  `<out>/<problem>`, so two renders on one problem dir delete each other's work rather than failing.
  The scheduler's `_seen` set is what makes that structurally impossible. Note `problem_x__s001` is a
  *preemption shard of* `problem_x`, not a problem of its own: `stage_rerender_source` already folds
  shards into the base slug's staging set, so the sweep collapses them with `base_problem_slugs()`
  before dispatching (a disk glob for `problem_*` matches shards too).
- **Never key completion on an exit code.** `rerender.py` prints `all jobs done` and then hangs in
  Isaac teardown. Every render runs through `scripts/rerender/run_local_render.py`, which watches for
  the marker and SIGKILLs the process group after a 30 s grace. Observed without it: 16/16 episodes
  finalized, then 53 minutes of nothing, holding the GPU.
- **Once the rerender stage marker is set, merge reads ONLY `rerender/out/job_*`.** A problem that
  executed but was never re-rendered then vanishes from the merged dataset with no error. That is why
  a problem skipped on purpose (its execution failed, or it has no finalized dataset) is still
  reported, and why a short set exits non-zero and leaves the marker unset unless
  `--merge-allow-missing-problems` is passed.
- **Both modes report from the post-stage sweep, never from the job list.** The sweep's universe is
  the disk, so it covers problems the execution loops skipped on `.execution_complete` or on a
  missing env dir, a run with `--skip-execution`, a resumed run where `apply_resume` disabled exec,
  and `__sNNN` shards. The coupled mode's in-chain render is a scheduling optimisation; correctness
  comes from the sweep. Computing `done`/`missing` over this invocation's `exec_jobs` instead is
  what used to drop already-executed problems from the dataset silently.
- **"Rendered" means what merge ships, not what the renderer claims.** `rerender.py` finalizes a
  problem with `completed: true, total_episodes: 0` when the recorded block size is not in the
  problem's size table or when every source parquet is unreadable — `completed` means *terminal*,
  not *has data*. Those write
  `blocks_success/meta/render_params.json` and no `info.json`, and `merge.get_dataset_paths` needs
  the `info.json`. `local_render_state()` therefore returns three states, and `unrenderable` is
  never fatal: it is permanent, so failing on it would wedge every later run.
- **Never quantify a gate over `glob("problem_*")`.** The execution stage `mkdir`s each problem's
  output dir *before* running its job, so every crashed execution leaves an empty `problem_*` dir
  behind for good. A gate over the glob would refuse to merge the other forty-nine problems forever
  over a directory merge would never have read. Use `merge_shipped_problems()`, which asks merge's
  own discovery. Dispatch still uses the glob — nothing should stop being rendered.

## `--camera_base` and why `data` is refused locally

`--camera_base data` tells the renderer to base each static camera on the pose **the dataset
records**, rather than on the SceneConfig mount. It is `rerender.py`'s default and it is the right
choice for a source whose parquet carries `observation.camera_pose.*` -- editing the scene config
then cannot silently redefine what "a re-render of this data" means.

**The local execution stage records no camera pose at all.** vecgen runs parquet-only -- it spawns
no cameras, so its output has `video features = 0` and no `observation.camera_pose.*` columns, in
BOTH local modes. This is not a timing question: coupled runs `vecgen && rerender` in one chain and
pipelined queues the render at finalize, so in both cases that problem's execution is complete
before its render starts. There is simply never a recorded pose to base on.

Asked for anyway, `rerender.py` already catches it: it tests `any(e.get("cam_pose") for e in
episodes)` and, finding none, prints

    [rerender] WARNING: --camera_base data IGNORED for this problem -- the source dataset
    carries no observation.camera_pose.* columns ... Falling back to --camera_base scene

and sets `scene` for that problem (re-derived per problem, so one camera-less source cannot pin
later ones). So the fallback is correct and LOUD, not silent. The pipeline passes
`--camera_base scene` explicitly and drops a `--camera_base data` arriving through
`--rerender-extra`; that is belt-and-braces -- same outcome, said at argv-build time instead of
~245 s into Kit startup.

Comments here used to call such a source "an execution that ran `--no-video`". That flag is real
but **not in this checkout**: it lives on the older plan-execution path (`execute_plans.py`).
Here the recording
switches are *opt-in and inverted*: vecgen takes `--record-videos` (via `--vecgen-record-videos`,
which the pipeline refuses alongside a local re-render) and the play pipeline takes
`--play-record-videos`. So every local execution is camera-less by default, and the comments now
say "an execution that recorded no cameras" rather than naming a flag that does not parse here.

Where `data` genuinely matters, and where the trap lives: on a source that DOES carry poses, the
re-base at `t == 0` moves the static cameras onto the recorded extrinsics **even for cameras the
flags yaml pinned with a `cameras:` block** -- `static_cams` is not filtered by `flag_cam_keys` the
way the randomization path is. The position/quaternion half of the override is reverted while
`fov_deg` survives (it goes through the camera cfg, not the pose), so the camera renders at a
pose/FOV pair nobody asked for. That is pre-existing and does not affect the local pipeline, which
never has recorded poses and always passes `scene`.

## Known gaps

- `--rerender-extra` is forwarded locally as of this change, appended last so it beats the
  pipeline's own `--depth_gop` / `--dr_seeding` / `--camera_base` / `--env_spacing` (rerender.py
  declares no append/extend/count actions, so a repeat is last-wins). `--output`, `--envs_dir`,
  `--episodes_file`, `--episode_offset` and `--max_episodes` are dropped with a warning: they
  break the pipeline's own accounting rather than the render, and dropping them restores the
  correct value.
- `--env_spacing` reaches `render_params.json` only when a run passes it explicitly; `--skip_cameras`
  is recorded only when `--render_scale` or `--skip_depth` is also set.
- **Episode-count parity is unchecked.** A problem that rendered 3 of 64 episodes has a finalized
  `info.json` and passes every gate here. `merge.py` already detects it — `video_coverage_report`
  logs *"INCOMPLETE VIDEO COVERAGE … which a path-resolving dataloader (hvla) reads as ZEROS"* —
  and then continues. Turning that warning into an exit code is the cheapest remaining win.
  `--rerender-local-fail-on-video-defects` covers only the encoder-defect half, and is off by
  default because a failed problem re-renders whole; `scripts/rerender/fast_patch.py` fills video
  holes afterwards.
