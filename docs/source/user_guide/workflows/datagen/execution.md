# Stage 3: Execution (vecgen)

Execution is where the demonstrations are actually produced. Given an env package and its reset
states, the robot has to work out how to reach the goal from each starting scene, carry out that
plan in simulation, and have every step recorded as training data.

`scripts/vecgen/run.py` does all of this in **one Isaac process**. It loads the env package,
plans every environment with the FF symbolic planner, runs the planned skills as batched cuRobo
motion plans, checks each skill's effect against the live simulation, and records a LeRobot v3
dataset. Planning happens inside the same process, per environment, so there are no
pre-computed plan pickles and no TAMP checkout to manage.

## Where it fits

- **Input:** an env package with `reset_states.pkl`, from [stage 2](env_generation.md), plus
  the PDDL problem and domain.
- **Output:** a LeRobot dataset split into successful and failed episodes, read by
  [rendering](render.md) or directly by [merge](merge.md).

This stage always runs. By default it records state and actions only; the camera videos come
from rendering, as explained in [What a recorded episode contains](#what-a-recorded-episode-contains).

## Correct by construction

The point of this stage is that every episode it labels successful really is a success. That
guarantee comes from a plan, act and verify loop:

1. FF plans each environment from its own symbolic state.
2. After every skill, that skill's `verify()` runs **against the live simulation**.
3. If the check fails, the environment is restored to its pre-skill snapshot (sim state, held
   block and cuRobo attachment), the retried frames are **dropped from the dataset**, and the
   skill is retried with freshly sampled parameters.
4. After `--max-attempts` failed attempts, FF re-plans from the live state, at most twice.
5. If that also fails, the episode ends as a failure.
6. At the end of the program the PDDL goal facts are checked again, so an episode is labelled
   successful **only if the goal actually holds**.

Failed episodes go to a separate `blocks_failure` split rather than being mixed in or discarded,
so you can study them later.

Each skill and its `verify()` live in `src/toyblocks/vecgen/skills/`. To teach vecgen a new PDDL
action, see [Adding a skill to vecgen](adding_a_skill.md).

## Running it

To run the flip env package from stage 2 and record 32 episodes with 16 environments in
parallel:

```bash
python scripts/vecgen/run.py \
    --env-path outputs/flip \
    --pddl src/toyblocks/tamp/blocks/problems/flip.pddl \
    --domain src/toyblocks/tamp/blocks/domain.pddl \
    --num-envs 16 --episodes 32 \
    --record datasets/flip --mark-complete --headless
```

The two counts mean different things:

- `--num-envs` is **concurrency**: how many environments run side by side on the GPU.
- `--episodes` is the **total number of episodes to finish**, successes and failures together.

When an environment finishes an episode, it is immediately reset onto the next reset state of
its own scene-profile slot and starts planning again. Environments keep recycling like this until
the episode target is met.

`--record` names the output directory, and creates `blocks_success/` and `blocks_failure/`
underneath it. `--mark-complete` writes a summary and a completion marker when the run finishes
cleanly; see [Output](#output).

## Through the pipeline

When `run_datagen_pipeline.py` runs this stage, it sets `run.py`'s flags from its own:

| Pipeline flag | Default | Becomes | Notes |
|---|---|---|---|
| `--exec-num-envs` | `128` | `--num-envs` | Must be a multiple of `--num-scene-profiles`. |
| `--vecgen-episodes` | `--env-per-problem` | `--episodes` | Episodes to run. Left unset, a multi-problem job runs one episode per reset state across the group (`members x --env-per-problem`) and a single-problem job runs `--env-per-problem`. Set it explicitly and that count is used on either path — it used to be silently replaced on the multi-problem path, which is the default. |
| `--seed` | `42` | `--seed` | |
| `--vecgen-exclude-actions` | `push_to_right,push_next_to` | `--exclude-actions` | Domain actions FF may never plan, also when re-planning. |
| `--vecgen-pool-slots` | `32` | `--pool-slots` | cuRobo planner pool slots per process. |
| `--vecgen-record-videos` | off | `--record-videos --enable_cameras` | Normally videos come from [rendering](render.md); this cannot be combined with a `--render-pass`. |
| `--vecgen-problems-per-env` | `8` | — | The number of problems sharing one multi-problem env package and process. Above 1 the env stage is skipped: each execution job builds its group's package with `scripts/vecgen/make_superset_env.py` (unless it already exists) and then streams every episode of the group through one `run.py` process, which reads the problems from `problems.json` instead of `--pddl`. |

The pipeline always passes `--record <base>/execution/<slug> --mark-complete --headless`, where
`<slug>` is the problem's slug, or `problem_grp_<first slug>` for a group. It skips any problem
whose `.execution_complete` marker already exists. Each problem (or group) is one job in the
pipeline's GPU-slot pool; see [Running the pipeline locally](local.md).

The pipeline does **not** forward `--profile`, `--max-attempts`, `--friction-comp`,
`--arm-joint-friction-scale` or `--arm-armature-scale`, so a pipeline run always uses their
defaults (see [Flags](#flags)). To change them, run `run.py` by hand.

## Environment variables

A few recording settings have no flag at all and are read from environment variables. The
pipeline's stages inherit your shell's environment, so export them before launching, or set them
on the `run.py` command line:

```bash
TMP3_RECORDER_INFLIGHT_GB=8 python scripts/datagen/run_datagen_pipeline.py --experiment-id ...
```

| Variable | Default | Effect |
|---|---|---|
| `TMP3_RECORDER_INFLIGHT_GB` | 4 GiB or a quarter of RAM, whichever is smaller | How much un-encoded video may be queued before recording pauses the simulation. Only matters with `--record-videos`. |
| `TMP3_ENCODER_THREADS` | all available cores | The size of the video encoder's thread pool. When more than one GPU slot runs at a time, the pipeline sets it to the cores divided by the number of slots, unless you set it yourself. |

The arm dynamics and the motion profile are flags, not environment variables; see [Flags](#flags).
The flip and rotate skill tuning and the cuRobo collision settings are fixed in code.

## Output

Everything is written under the `--record` directory:

| Path | Contents |
|---|---|
| `<--record>/blocks_success/` | Episodes whose PDDL goal held at the end. |
| `<--record>/blocks_failure/` | Every episode the engine ended as a failure, with the reason. |
| `<--record>/results.json` | Only with `--mark-complete`: per-episode records, failure reasons and wall time. |
| `<--record>/.execution_complete` | Only with `--mark-complete`. The pipeline uses it to decide whether to resume or skip this problem. |
| `<--record>__s001/`, `__s002/` | Shards written when a preempted run resumes; see [Common pitfalls](#common-pitfalls). |

To check a run, look for `RUN_OK` and the `RESULT ...` line in the log, and read the failure
reasons in `results.json`.

## What a recorded episode contains

**By default this stage records no cameras at all.** The env is built with `cameras=False`, so
the camera prims are never spawned, and the recorder leaves out every video feature. There is no
`observation.images.*` and no `observation.depth.*`, and also **no `observation.camera_pose.*` or
`observation.camera_intrinsics.*` columns**. The output is Parquet only.

This keeps execution fast; the pixels normally come from [the render stage](render.md)
instead. It is also why the render's `--camera_base data` cannot work on a dataset this stage
recorded: there is no recorded camera pose to start from, so the pipeline renders with
`--camera_base scene`.

Each row is one engine tick, written at 25 fps. A row holds `is_first` and `is_last`;
`observation.state` and the per-joint position, velocity, acceleration and effort vectors; the
gripper, `ee_position` and wrench; the 13-DoF stiffness, damping, effort-limit, velocity-limit
and torque vectors; `observation.state.object_pose.<name>` for every rigid object;
`observation.state.contact_force.<sensor>` per contact sensor; `robot.poses.<link>` for every
robot body; the scene parameters; the provenance columns (`scene_id`, `robot_id`,
`source_problem`, `instance_id`); `pddl_goal`; the symbolic state; the action block, including
`action.joint_position` and `action.joint_target`; and the high- and low-level action labels.

Every column is described in [Dataset fields](dataset_fields.md).

If you do want video from this stage, `--record-videos` is not enough on its own. Isaac also has
to be told to render, so pass `--enable_cameras` as well:

```bash
python scripts/vecgen/run.py \
    --env-path outputs/flip \
    --pddl src/toyblocks/tamp/blocks/problems/flip.pddl \
    --domain src/toyblocks/tamp/blocks/domain.pddl \
    --num-envs 16 --episodes 32 \
    --record datasets/flip_videos --record-videos --enable_cameras --mark-complete --headless
```

This adds `observation.images.camera:<camera>` and `observation.depth.camera:<camera>` videos
for the front and gripper cameras (640x360) to the dataset.

## Flags

These are the flags you are most likely to need:

| Flag | Default | Meaning |
|---|---|---|
| `--env-path` | (required) | The autogen env package directory. |
| `--pddl` | `None` | The problem file. Omit it only when `<env-path>/problems.json` exists, which is the case for multi-problem streaming. |
| `--domain` | `src/toyblocks/tamp/blocks/domain.pddl` | The domain that FF plans over. |
| `--num-envs` | `16` | The number of environments stepped in lockstep. |
| `--episodes` | `16` | The total number of episodes to finish. |
| `--seed` | `0` | Seeds the engine RNG and the geometry sampler. |
| `--record` | `None` | The output directory for the LeRobot dataset. |
| `--record-videos` | off | **Opt-in.** Also spawn and record the scene cameras. |
| `--mark-complete` | off | Write `results.json` and `.execution_complete` when the run finishes cleanly. |
| `--max-attempts` | `3` | Attempts per skill before the engine gives up on it. |
| `--exclude-actions` | `""` | A comma-separated list of domain actions FF may never plan, for example `reorient,rotate` for pick-and-place-only data. |
| `--init-from-scene` | off | Plan each episode from its live scene rather than from the PDDL `:init`. |
| `--no-derived` | off | Plan from the PDDL `:init` only, without facts derived from the live scene. |
| `--reset-states-once` | off | Start every reset state at most once; the run ends when all of them have run. |
| `--reset-state-ids` | `None` | Run only these reset-state ids, each once. |
| `--cut-tail` | off | Keep every env busy until the episode target is reached, then close the episodes still running as failures. |
| `--profile` | `medium` | The motion and stiffness profile: the arm gains, and the press depth of the scripted flip. |
| `--friction-comp` / `--no-friction-comp` | off | Joint-friction compensation of motion-planned arm targets. |
| `--arm-joint-friction-scale` | `0.0` | Scales the arm's static, dynamic and viscous joint friction in sim. `0` gives a frictionless arm, `1` the robot model's friction. |
| `--arm-armature-scale` | `0.0` | Scales the arm's joint armature in sim. `0` removes it, `1` keeps the robot model's armature. |
| `--no-ff` | off | Disable the target feed-forward. |
| `--max-ticks` | `None` | Stop after this many engine ticks. One tick is 4 physics steps, or 0.04 s. |
| `--play` | off | Collect play data instead: each reset state plays a chain of policy-chosen goals, with no reset between plans. |

The arm-dynamics defaults are the frictionless, armature-free arm without friction compensation
that the Nov 2026 training sets (e.g. RI) were generated with.

Performance knobs (`--async-workers`, `--pool-slots`, `--build-slots`, `--plan-gather-ticks`,
`--plan-gather-min`, `--sync-plans`, `--sync-programs`, `--pool-ik-seeds`,
`--pool-trajopt-seeds`, `--build-ik-seeds`), the play-data options, and debugging aids
(`--log-level`, `--profile-out`) are listed by `python scripts/vecgen/run.py --headless --help`.

## Common pitfalls

### A mistyped flag is ignored, or quietly does something else

`run.py` uses `parse_known_args`, which it needs for Isaac's AppLauncher, so a flag it does not
define is dropped without a word.

There is a subtlety. The parser leaves `allow_abbrev` at its default, so any **unique prefix** of
a real flag is accepted as that flag. `--record-video` is therefore not ignored: it turns video
recording *on*. `--num-env 4` and `--pool-s 8` work too, while an ambiguous prefix such as `--pool` is an error. A typo is either silently ignored or
silently honoured, depending on whether it happens to be a prefix, so double-check flag names
against the table above.

### Some limits cannot be changed from the command line

Three limits live in `EngineConfig` and have no CLI flag: `max_replans = 2`,
`max_episode_ticks = 6000` (the episode is ended with reason `"episode tick limit"`), and
`done_hold_ticks = 25` (the hold recorded after the last skill).

### A run aborts because a scene profile has no reset state

Reset states are bound to scene profiles, not to environments. Environment *i* spawns in slot
`i % MAX_NON_SIM_STATE_RANDOMIZATIONS` and may only run reset states whose block sizes and
friction match that slot. If a spawned slot has **no** compatible reset state, the run aborts
with a `RuntimeError`.

Without `--reset-states-once`, a profile's reset states **cycle**, so asking for more episodes
than there are reset states replays some of them.

### `--mark-complete` does not notice lost videos

It writes `.execution_complete` without consulting the video encoder's defect counter, so a
`--record-videos` run that lost videos can still be marked complete. Only the render tooling
checks `video_defects()`.

### Episodes go missing at merge time because of the directory name

Merge only picks up `base_dir/problem_*/blocks_success`, and the `source_problem` column is taken
from the **deepest** path component starting with `problem_`, falling back to the directory's own
name. For a nested path like `.../problem_grp_flip/problem_0007` that is `problem_0007`, not the
group. Name the `--record` directory accordingly.

An *empty* directory still counts as shard 0, which is why the pipeline creates the problem
directory before launching. `run.py` never scans for shards itself, so a retried attempt
leaves the previous partial shard on disk.

### The process seems to hang after finishing

After the run, `run.py` arms a 15 s timer and calls `os._exit(rc)`, because
`simulation_app.close()` can hang when headless. So the process may take a while to go away.
**Poll the log for `RUN_OK` / `RESULT ...` rather than waiting for the process to exit.**

### `results.json` counts more episodes than the dataset

If FF fails, or the plan uses an action vecgen has no skill for, the episode is counted in
`results.json` as a failure with reason `"no program"`, but it produces **no dataset episode**,
because `start_episode` was never called. Episodes with zero frames are also skipped at save
time. So the two counts can legitimately differ.

## Next

[Render](render.md) the episodes to produce camera videos, then [merge](merge.md) them into
one training dataset.
