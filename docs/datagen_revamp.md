# Datagen revamp (`naman/datagen-revamp`)

Goal: one vectorized simulation process per GPU that samples scenes, plans symbolically
per env, executes skills as batched tensor operations over hundreds of envs, verifies
every skill against the simulator, and records episodes. The planning stage (stage 2)
disappears; the simulator is the planner's model.

## Why

Measured on the Oct 2026 Beaker runs (stack-3, cuRobo v2 branch):

| | today | target |
|---|---|---|
| stage 2 | 50-140 s of TAMP Python per instance (PRRefinement) | none |
| stage 3 Python per step | 5-48 ms at 32 envs, grows with live envs | < 10 ms, flat in envs |
| envs per GPU | 32 | 256-1024 (PhysX ~25 ms/step, flat to 4k envs) |
| env-steps / s / GPU | ~400 | 20k-30k |

TAMP's refinement loop (predict effects, refine, backtrack on error modes) exists to cope
with not having a model of the world. In sim the model is exact and cheap, and envs are
the search budget: every "retry" becomes "run K variants in K envs from the same snapshot".

## Pipeline

1. **Task gen** (CPU, unchanged): PDDL problems with symbolic init + goal, block set, scene
   randomization metadata.
2. **Vectorized datagen job** (one process per GPU, this branch):
   - *Scene*: build the Isaac env with N envs for the problem's block set; sample initial
     poses on GPU that realize the PDDL init (stacks, orientations, table windows), settle a
     few ticks, verify the init predicates hold (batched).
   - *Symbolic state*: PDDL init plus **derived geometric predicates** computed from the sim
     with batched tensor ops: reachable(pose), flip_clear(b, side), approach_clear(b).
   - *Plan*: FF per env (CPU thread pool, ms each) -> skill program per env.
   - *Execute*: each env runs its own program; per tick the stepper gathers one joint-target
     row per env from precomputed segment tables and writes all targets at once. Per skill,
     keyframes are sampled in batch for all envs currently at that skill, converted to joint
     targets with batched IK / batched trajopt (cuRobo pool, per-env worlds), stored as
     segment tables.
   - *Verify*: after each skill, batched predicate checks on the real state. Failing envs
     restore the pre-skill snapshot and resample parameters; after K tries the episode is
     dropped. If the world changed (e.g. a flip landed elsewhere), recompute the env's
     symbolic state and call FF again from where it is.
   - *Record*: per-env GPU ring buffers for the dataset columns, flushed per episode by an
     async writer into LeRobot shards (same schema as today, videos via rerender).
   - *Stream*: a finished env takes the next (problem, reset state) from a queue; no
     per-problem waves.
3. **Rerender** (unchanged): videos from recorded states.
4. **Merge / annotate** (unchanged).

## Package layout (`src/blocksuite/vecgen/`)

- `program.py`     skill programs and per-env program state (pure data, torch-free)
- `segments.py`    batched joint-target segment tables, gather per tick
- `symbolic/`      batched predicates from sim state, PDDL init construction, FF per env
- `skills/`        batched keyframe generators (pick, place on/table/side, flip, rotate) and
                   `rl.py`: the same skills driven by their RL checkpoints (`policy` phase kind,
                   checkpoint gains kp 40 applied per env for the RL window, run profile restored)
- `geometry.py`    batched IK / transport plans on the cuRobo pool, per-env worlds
- `engine.py`      the lockstep loop: scheduler, snapshot/restore, verification, retries
- `recorder.py`    columnar per-episode recording -> LeRobot writer
- `scripts/vecgen/run.py` entry point (`--pddl` for one problem, or a superset package with
  `problems.json` from `scripts/vecgen/make_superset_env.py` to stream a problem family);
  `scripts/datagen/run_datagen_pipeline.py` runs one package per GPU slot

## What is reused from the old system

vecgen runs without the TAMP submodule (`tmp3` not installed: `uv pip install -e
.[dev,isaac,torch-cu128,vecgen]`). What it reuses from TAMP was copied into blocksuite, verbatim except imports, and
checked against the original (same IK/plan results, same dataset files, same task-gen output):

| reused | from | now in |
|---|---|---|
| `CuroboPlannerPool` (cuRobo v2 batched IK/trajopt, per-env tensor worlds, attachments) + robot config / URDF / spheres | `tmp3.Backends.Curobo.coordinator`, `motion_gen`, `types`, `robot_config` | `vecgen/planner/` |
| `IsaacDataRecorderLeRobot` (schema, shards, success/failure split, writer process) | `tmp3.DataRecorder` | `vecgen/recording/` (goal annotation not ported) |
| stiffness profiles | `tmp3.Backends._shared.stiffness_profiles` | `blocksuite/utils/stiffness_profiles.py` |
| scripted flip (`ReOrientTrajectory` constants + helpers: waypoints, tip offset, anchor cost, press depth) | `blocksuite.tamp.blocks.executors` | `skills/reorient_trajectory.py` |
| scripted rotate (`RotateTrajectory` constants + helpers: sub-targets, anchors, timing) | `blocksuite.tamp.blocks.executors` | `skills/rotate_trajectory.py` |
| RL checkpoint loading + pseudo env (`BaseRLTrajectory`) | `blocksuite.tamp.blocks.executors` | `skills/rl_trajectory.py` |
| stage-1 predicate samplers (superset package) | `blocksuite.tamp.blocks.predicates` | `vecgen/sampling/` |
| FF-v2.3modified binary + sources | `TAMP/planners` | `symbolic/ff_planner/` |
| PDDL task generators' library | `tmp3.data_gen.task_gen` | `blocksuite/task_gen/` |

Used in place: autogen env packages, reset states, scene profiles, `_reset_idx`,
`get/set_state_dict`, stage-1 codegen (`blocksuite.tamp.blocks.template`), `camera_view`,
`domain.pddl`, RL checkpoints. Not reused: per-env executors (pick/place), scalar predicates
(now batched tensors), PRRefinement, orchestrator, `robot.py` / simulator backends, policy
pickles. The legacy 3-stage pipeline has since been removed; play data (`run.py --play`) and the
eval oracle are TAMP-free too.

## Phases

1. Replay-only engine on existing autogen envs: pick/place from PDDL goals, batched IK and
   trajopt, batched verification, async writer. Milestone: 256 envs, Python < 10 ms/step.
2. Flip / rotate skills (scripted flip ported to tensors), stack-3 family end to end with no
   stage 2. Milestone: 200-problem run, success >= today's 78-83%.
3. Snapshot/branch retries, FF per env with derived predicates, streaming across problems,
   universal scene (superset of blocks), stage 1 folded in. Milestone: 1024 envs/GPU.
4. PhysX tuning on H100, multi-process per node, Beaker job type, rerender hookup.

## Status (Oct 20 2026)

Phases 1-3 are implemented on this branch (see `docs/datagen_revamp.html` for the full system
document and measured results): plan gathering across skills, derived geometric facts before the
first FF call, RL skills as a `policy` phase with per-env stiffness, superset packages with
problem streaming, 256 envs/GPU (red_tower_3: 504/512 in 81 s = 380 ep/min on a 5090), and the
Beaker job generator (rendered, not yet submitted).

Oct 20 optimisation pass: flip search in three pool rounds per pull scale (3-7 IK rounds per flip
instead of ~48), pool slot regions per build thread, batched RL observations (one forward pass per
policy per tick), `--cut-tail` streaming. 4 streamed stack-3 problems with recording: 970 s -> 790 s
(115/128). Pool cost model: every call solves all slots, so keep the pool no wider than the env count.
