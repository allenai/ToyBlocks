# Automatic Data Generation

The data generation pipeline turns a symbolic PDDL description of a task into a large,
automatically-labelled dataset of robot demonstrations in simulation. You describe what should be
true at the end of the task, for example "the red block is on the green block", and the
pipeline does the rest: it samples scenes in which the task can start, plans every
episode with a symbolic planner, carries the plan out with motion-planned skills on
hundreds of simulated environments at once, checks every step against the simulator,
and records the demonstrations in LeRobot format, including sensor observations, ready for training.

Because tasks are written as goals rather than as simulation code, the same pipeline
produces data for stacking, flipping, rotating, pick-and-place and other task
families, and it runs the same way on a single GPU or across every GPU of one machine.

## Design goals for the datagen system

- **Declarative tasks.** A task is a Planning Domain Definition Language (PDDL)
  problem: a set of initial facts and a goal over predicates such as `on`, `on_table`
  and `ori_of`. Task generators in `scripts/task_gen/` produce whole task families, and
  the simulation environment and its reset states are generated from the PDDL
  automatically. Adding a task means writing a goal, not simulation code.
- **Correct by construction.** The FF symbolic planner plans each environment from its
  own symbolic state. After every skill, the skill's effects are checked against the
  live simulation. If a check fails, the environment returns to its pre-skill snapshot
  and retries; once the retries run out, the planner re-plans from the current state.
  An episode is labelled successful only if the PDDL goal holds in the final state, and
  failed episodes are stored in a separate failure split instead of being mixed in.
- **Scalable, with CPU and GPU parallelization.** One Isaac process per GPU runs hundreds of
  environments at once (eg. 256 per GPU in measured runs). Skills run as batched tensor
  operations on a pool of cuRobo inverse-kinematics and trajectory-optimization solvers,
  with a separate collision world per environment, while each environment's FF plan
  takes milliseconds on the CPU. Finished environments stream straight into their next
  episode, and several problems share one process (8 by default) so the Isaac start-up
  cost is paid once per group.
- **Diverse tasks on simple assets.** Scene randomization covers block size, table height, friction,
  distractor blocks and block placement, and the robot's starting pose can be randomized
  (always collision-free).
- **Rerendering, with visuals separate from physics.** Execution records the
  trajectory states; the camera videos come from a separate rerendering stage. It replays
  each recorded episode by placing the robot and objects frame by frame, with no physics
  stepping, and renders it under domain randomization: lighting and sky images, table
  materials, camera poses and field of view, and block letters and colours. This means:
  - one expensive planning-and-physics run can be rendered many times with different
    visuals, for example several copies per episode, or separate train and test letter
    sets, without re-planning;
  - rendering is scheduled separately from data generation: as its own job in the
    pipeline's GPU-slot pool, appended to each execution job, or run again later on the
    stored states;
  - the render settings sampled for each episode are recorded, so evaluation can
    reproduce an episode's look.

  Re-rendered pixels match the original rendering closely but not exactly: driver and
  scene-version differences remain.
- **Rich, training-ready output.** Episodes are written in LeRobot v3 format. Alongside
  the robot state, camera streams and actions, every frame records the symbolic state,
  the current high-level and low-level action labels, and the goal. A merge and an
  annotation step produce one training dataset.
- **Reproducible and traceable.** Each run pins the git commit, saves its full
  configuration, and writes provenance into the dataset (the commit, checksums of the
  source files, copies of the PDDL files). An interrupted run can be resumed, skipping
  the stages that already finished.
- **One command, one machine.** The orchestrator runs every stage on the machine you
  launch it on, spreading problems over the visible GPUs. The same command works on a
  single-GPU workstation and on a multi-GPU node (see [Running on this machine](datagen/local.md)).

## What can you do with the datagen pipeline?

- **Generate a demonstration dataset for a task family**, such as 3-block stacking,
  flipping, rotating or pick-and-place: pick a generator in `scripts/task_gen/` and run
  `run_datagen_pipeline.py` (see [Full-pipeline data generation](#full-pipeline-data-generation)).
- **Generate a single task end to end on one GPU** for debugging, with
  `generate_task_from_pddl.py` followed by `scripts/vecgen/run.py` (see
  [Env generation](datagen/env_generation.md) and
  [Execution](datagen/execution.md)).
- **Randomize the physical scene** (block size, table height, friction, distractors and
  robot start pose); scene randomization is on by default, and `--randomize-robot-init`
  adds the robot start pose.
- **Re-render a dataset** with new lighting, materials, cameras, and block letters or
  colours, as many times as you like, without re-planning (see
  [Re-rendering](datagen/rerender.md)).
- **Generate play data**, where each scene plays a chain of goals with no reset in
  between, with `scripts/datagen/run_playdata_pipeline.py` or
  `scripts/vecgen/run.py --play`.
- **Relabel a policy's failures**: turn the states where a policy failed during
  evaluation into new expert demonstrations, with
  `scripts/datagen/run_tamper_pipeline.py`
  ([TAMPER](https://github.com/allenai/ToyBlocks/blob/main/docs/tamper.md)).
- **Produce training-ready output**: one merged, annotated LeRobot dataset, ready for
  imitation learning and for [evaluation](evaluations.md).

## How to get started

`scripts/datagen/run_datagen_pipeline.py` runs the stages below in order, all on your machine.
Each stage can also be run on its own, as its page (linked under "Datagen Pipeline Steps" below)
shows.

```text
run_datagen_pipeline.py
  ├─ task-gen    scripts/task_gen/*.py           PDDL problems
  ├─ env         generate_task_from_pddl.py      env package + sampled reset states
  ├─ execution   scripts/vecgen/run.py           FF per env -> batched skills -> verify -> record
  ├─ rerender    (optional, --rerender or        new visuals from the recorded states
  │               --rerender-local)
  ├─ merge       merge_datasets.py               one training dataset
  └─ annotate    annotate_target_xy.py           target annotations
```

If you simply want to generate some example datasets with already tested scripts, start from these entry points in the [scripts/datagen folder](https://github.com/allenai/ToyBlocks/tree/main/scripts/datagen):

- [run_datagen_pipeline.py](https://github.com/allenai/ToyBlocks/blob/main/scripts/datagen/run_datagen_pipeline.py): the end-to-end pipeline described on this page; its docstring lists ready-to-run example commands.
- [run_playdata_pipeline.py](https://github.com/allenai/ToyBlocks/blob/main/scripts/datagen/run_playdata_pipeline.py): the same pipeline for play data, where each scene plays a chain of goals without resetting.
- [run_tamper_pipeline.py](https://github.com/allenai/ToyBlocks/blob/main/scripts/datagen/run_tamper_pipeline.py): turns states where a policy failed during evaluation into new expert demonstrations.
- [merge_datasets.py](https://github.com/allenai/ToyBlocks/blob/main/scripts/datagen/merge_datasets.py): merges per-problem LeRobot datasets into one.
- [plot_episode.py](https://github.com/allenai/ToyBlocks/blob/main/scripts/datagen/plot_episode.py): plots every signal of one episode, a quick way to check a generated dataset.



## Full Data Generation Pipeline

The `scripts/datagen/run_datagen_pipeline.py` script orchestrates the entire data generation process end-to-end. It runs every stage on this machine, one problem per GPU slot (all visible GPUs by default), pins the git commit for reproducibility, and saves a full config JSON alongside the output. How the GPU slots, logs and resume work is described in [Running on this machine](datagen/local.md).

### Datagen Pipeline Steps

1. **[Task generation](datagen/task_generation.md)**: runs a task gen script (default `scripts/task_gen/stack_three.py --train`) to produce randomised PDDL problem files.
2. **[Env generation](datagen/env_generation.md)**: builds one env package per problem and samples its reset states (`generate_task_from_pddl.py`). Skipped when `--vecgen-problems-per-env` > 1 (the default is 8): the execution step then builds one multi-problem package per group itself.
3. **[Execution](datagen/execution.md)**: runs vecgen on each problem or group (`scripts/vecgen/run.py`): FF planning per environment, batched skills, verification, and LeRobot recording.
4. **[Rerender](datagen/rerender.md)** (optional): re-renders each problem's episodes as soon as its execution finishes, either as its own job in the GPU-slot pool (`--rerender`) or inside the execution job (`--rerender-local`).
5. **[Merge](datagen/merge.md)**: merges all successful per-problem LeRobot datasets (the re-rendered ones when the rerender stage ran) into a single LeRobot training dataset (`merge_datasets.py`).
6. **[Annotate](datagen/annotate.md)**: adds per-frame goal columns to the merged LeRobot dataset (`annotate_target_xy.py`): `goal.action` (the current action as text, e.g. `pick`, `push left`) and `goal.position` (the target object's pixel coordinates in the front camera). It needs camera data, so it is skipped with a warning unless the episodes were re-rendered or recorded with `--vecgen-record-videos`.


### Basic Usage

```bash
# Full pipeline with the defaults (the RI stack-3 training-set settings), on every visible GPU
python scripts/datagen/run_datagen_pipeline.py --experiment-id nov05_stack3

# The same, re-rendered: the camera videos come from the rerender stage
python scripts/datagen/run_datagen_pipeline.py --experiment-id nov05_stack3 --rerender-local

# A small run: 2 problems, 16 episodes each, re-rendered, outputs under ./outputs
python scripts/datagen/run_datagen_pipeline.py --experiment-id nov05_stack3_small \
    --output-root outputs --rerender-local \
    --task-gen-count 2 --env-per-problem 16 --num-envs 16 --exec-num-envs 16 \
    --vecgen-problems-per-env 1

# Another task generator
python scripts/datagen/run_datagen_pipeline.py --experiment-id nov05_rotate \
    --task-gen-script scripts/task_gen/rotate.py --task-gen-count 100

# Interactive mode: choose which steps to run and edit parameters
python scripts/datagen/run_datagen_pipeline.py --experiment-id nov05_stack3 -i

# Quick smoke test (1 problem, 16 envs, no randomization)
python scripts/datagen/run_datagen_pipeline.py --experiment-id nov05_smoketest --smoke-test

# Continue an interrupted run: finished stages are skipped
python scripts/datagen/run_datagen_pipeline.py --experiment-id nov05_stack3 --resume
```

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--experiment-id` | (required) | Name for this experiment run. Used for directory naming. |
| `--interactive` / `-i` | off | Interactively select steps and edit parameters before running. |
| `--resume` | off | Skip stages already marked complete and restart the first unfinished one. |
| `--smoke-test` | off | Override to minimal params: 1 problem, 16 envs, no distractors, no scene randomization. |
| `--task-gen-script` | `scripts/task_gen/stack_three.py --train` | Which task generator to use (`flip.py`, `rotate.py`, `pnp.py`, `stack_three.py`, ...), optionally with its own flags. |
| `--task-gen-count` | 1000 | Number of PDDL problems to generate. |
| `--task-gen-seed` | 6303 | Task generation seed. |
| `--randomize-scene` / `--no-randomize-scene` | on | Block size, table height, and friction randomization. |
| `--randomize-robot-init` | off | Give each reset state a random collision-free arm configuration and gripper state. |
| `--num-distractors` | 0 | Distractor objects per problem (passed to the generator as `--num_distractors`; see [Task generation](datagen/task_generation.md#common-pitfalls)). |
| `--domain-path` | `src/toyblocks/tamp/blocks` | Path to the PDDL domain directory. |
| `--usd-library` | `train_blocks_large` | Glyph asset library that block names resolve against at code generation. |
| `--skip-task-gen`, `--skip-env`, `--skip-execution`, `--skip-merge`, `--skip-annotate` | off | Skip individual steps for re-runs or debugging. |

### Scene Randomization

The pipeline can randomize the following parts of the physical scene:

| Randomization | Flags | Default | Notes |
|---------------|-------|---------|-------|
| Block size | `--block-size-range MIN MAX` | `0.03 0.05` | Ignored with `--no-randomize-scene`. |
| Table height | `--table-height-range MIN MAX` | `-0.05 0.05` | Height offset; ignored with `--no-randomize-scene`. |
| Friction | `--friction-range MIN MAX` | `0.5 0.5` | Ignored with `--no-randomize-scene`. |
| Distractor objects | `--num-distractors N` | `0` | Extra blocks that are not part of the goal. |
| Block placement | `--block-x-range SPEC`, `--block-y-range SPEC` | PDDL value, else `[-0.1, 0.25]` / `[-0.30, 0.30]` | Table-frame window for on-table block spawns; accepts one interval or several disjoint intervals. |
| Robot start pose | `--randomize-robot-init` | off | Collision-free random arm configuration per reset state. |
| Robot start pose ranges | `--robot-init-ee-x-range`, `--robot-init-ee-y-range`, `--robot-init-ee-z-min` | `0.25 0.65`, `-0.35 0.35`, `0.20` | End-effector window in the robot base frame. |
| Gripper start state | `--robot-init-gripper-closed-prob` | `0.5` | Probability a reset state starts with the gripper closed. |

Block size, table height and friction are not sampled per episode: each env package bakes in `--num-scene-profiles` (default 8) profiles drawn from the ranges, so `--num-envs` and `--exec-num-envs` must be multiples of it. How the ranges travel through the PDDL file to the simulator is in [How scene randomization reaches the simulator](datagen/task_generation.md#how-scene-randomization-reaches-the-simulator). Visual randomization (lighting, materials, cameras, block letters and colours) happens at rerender time; see [Re-rendering](datagen/rerender.md).

Scene randomization is on by default, over the ranges above. With `--no-randomize-scene` (and with `--smoke-test`), block size, table height and friction are fixed: block size = 0.04, table height delta = 0, friction = 0.4. To use other ranges:

```bash
python scripts/datagen/run_datagen_pipeline.py \
    --experiment-id nov05_stack3_rand \
    --block-size-range 0.03 0.05 \
    --table-height-range 0.0 0.04 \
    --friction-range 0.1 0.4
```

### Output and GPU Options

| Flag | Default | Description |
|------|---------|-------------|
| `--output-root` | `data` | Root for every dataset the pipeline writes, relative to the working directory (resolved to an absolute path). |
| `--dataset-folder` | `unmerged` | Subfolder of `<output-root>/datasets/` for the per-problem outputs. |
| `--merge-output-dir` | `<output-root>/datasets/merged/train/<experiment-id>` | Where the merged dataset goes. |
| `--local-gpus` | every visible GPU | GPU ids the stages may use, e.g. `--local-gpus 0 1`. |
| `--local-gpu-workers` | 1 | Concurrent problems per GPU. Host RAM is usually the limit; validate videos when raising it. |

See [Running on this machine](datagen/local.md) for how the GPU slots are filled, where logs go,
and what `--resume` deletes.

### Re-rendering

Execution records state and actions only; the videos come from the optional rerender stage. Turn
it on with one of two flags, which produce the same pixels and differ only in scheduling:

| Flag | What it does |
|------|--------------|
| `--rerender` | Pipelined: each problem's render is queued as its own item in the GPU-slot pool the moment its execution finishes, so it can overlap the next execution. |
| `--rerender-local` | Coupled: the render is appended to each execution job and holds the slot its execution just used. With `--local-gpu-workers 2` this is the faster choice. |

They are alternatives; passing both is refused. The render defaults are the RI training set's:
the flags YAML `rerender_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml`, one copy per
episode, the 224 RGB and depth sidecars, `left_camera` skipped and failure episodes skipped. The
default flags draw HDRI backdrops from polyhaven (part of the base asset download) and the
separately licensed Laval indoor set; see
[HDRI backdrops](https://github.com/allenai/ToyBlocks/blob/main/docs/rerender/README_local.md#hdri-backdrops).
Every rerender flag is described in [Re-rendering](datagen/rerender.md).

### Per-Stage Parameters

Flags that set one stage's options are listed on that stage's page, with the stage flag each
one becomes: [env generation](datagen/env_generation.md#through-the-pipeline),
[execution](datagen/execution.md#through-the-pipeline), [re-rendering](datagen/rerender.md),
[merge](datagen/merge.md#flags) and [annotation](datagen/annotate.md#running-it).

### Execution Settings Without a Pipeline Flag

Some execution settings are `scripts/vecgen/run.py` flags that the pipeline does not forward: the
motion profile (`--profile`, default `medium`) and the arm dynamics (`--friction-comp`,
`--arm-joint-friction-scale`, `--arm-armature-scale`). Their defaults are the frictionless,
armature-free arm without friction compensation that the RI training set was generated with, so
a pipeline run always uses them. To change them, run the execution stage by hand; see
[Execution](datagen/execution.md#flags).

A few recording settings are read from environment variables. The pipeline's stages inherit your
shell's environment, so export them before launching. They are listed in
[Execution: Environment variables](datagen/execution.md#environment-variables).


### Datagen Output Structure

If successful, all outputs of the script `scripts/datagen/run_datagen_pipeline.py` will be written
under `<output-root>/datasets/<dataset-folder>/<experiment-id>/` (by default
`data/datasets/unmerged/<experiment-id>/`), containing the following:

```
<experiment-id>/
├── tasks/                    # Generated PDDL problem files
├── envs/                     # Env packages + reset states, one per problem or problem group
├── execution/                # Per-problem LeRobot datasets (success + failure splits)
├── rerender/                 # Re-rendered datasets (only with --rerender / --rerender-local)
├── local_logs/               # Per-problem logs, only when more than one GPU slot runs at a time
├── .pipeline_state/          # <stage>.done markers read by --resume
└── run_datagen_pipeline_config_<timestamp>.json  # Full reproducibility config
```

The final merged dataset is written to `<output-root>/datasets/merged/train/<experiment-id>/` (or `--merge-output-dir`). Every column it contains is described in [Dataset fields](datagen/dataset_fields.md).

```{toctree}
:hidden:

datagen/task_generation
datagen/env_generation
datagen/execution
datagen/rerender
datagen/merge
datagen/annotate
datagen/dataset_fields
datagen/local
```
