# Stage 2: Environment generation

A PDDL problem says what the task is, but the simulator needs much more: which blocks to spawn,
how big they are, where they may sit, and what a valid starting scene looks like. Environment
generation fills that gap. It turns one PDDL problem into **a self-contained Isaac Lab env
package plus a set of sampled reset states**, and it is the only stage that writes simulation
code.

`scripts/generate_task_from_pddl.py` does two jobs:

1. **Code generation (always).** It renders the domain's `template_env.py` and
   `template_env_cfg.py` into `<output_dir>/<slug>/` and registers the gym id `<slug>-v0`. The
   *slug* is the task's name, taken from the PDDL file name. The scene randomization tables are
   baked into the generated source at this point.
2. **Reset-state sampling (when `--num_reset_states > 0`).** A *reset state* is one concrete
   starting scene that satisfies the problem's `:init` facts: exact block positions,
   orientations and, optionally, a starting arm pose. Execution starts each episode from one of
   them, so sampling many gives the dataset many different starting scenes. The script launches
   Isaac, creates the generated env with `--num_envs` parallel copies, and runs the `:init`
   predicates through vectorized samplers. Any env whose blocks land outside the front camera's
   view is resampled. The result is pickled to `reset_states.pkl`.

With `--num_reset_states 0` no simulation runs at all: you get code generation plus a
`compileall` pass, which is a quick way to check that a problem produces valid code.

## Where it fits

- **Input:** one PDDL problem file, from [stage 1](task_generation.md) or from the checked-in
  problems.
- **Output:** an env package directory, read by [stage 3, execution](execution.md).

The pipeline runs this stage once per problem when `--vecgen-problems-per-env` is `1`. With its
default of `8`, the pipeline skips this stage and builds one multi-problem package per group of
problems inside the execution jobs instead; see [Through the pipeline](#through-the-pipeline).
Run it by hand to debug a single problem, or to look at the scenes it samples.

## Running it

A typical run generates 32 reset states for the checked-in flip problem, using 16 environments
in parallel:

```bash
python scripts/generate_task_from_pddl.py \
    --domain_path src/toyblocks/tamp/blocks \
    --pddl src/toyblocks/tamp/blocks/problems/flip.pddl \
    --output_dir outputs \
    --num_reset_states 32 --num_envs 16 \
    --seed 0 --headless
```

`--pddl` can be a checked-in problem from `src/toyblocks/tamp/blocks/problems/` or one written
by [stage 1](task_generation.md), for example
`outputs/rotate/tasks/problem_declarative_8_20260602_171400_1234.pddl`. Of the checked-in ones,
`flip.pddl` has cube and table randomizations and `rotate.pddl` has none. The package lands in
`<output_dir>/<slug>/`, where the slug is the file's basename with hyphens replaced by
underscores. In a shell script you can compute it like this:

```bash
task=$(basename "${problem}" .pddl | tr '-' '_')
```

The script parses its arguments with `tyro`, so **every flag accepts both spellings**:
`--num_reset_states` and `--num-reset-states` are the same flag. The Isaac flags inherited from
`BaseArgs` are the one exception; see [Common pitfalls](#common-pitfalls).

To also give each reset state a random, collision-free starting arm pose, add
`--randomize_robot_init`. The three range flags limit where the end effector may start:

```bash
python scripts/generate_task_from_pddl.py \
    --domain_path src/toyblocks/tamp/blocks \
    --pddl src/toyblocks/tamp/blocks/problems/flip.pddl \
    --output_dir outputs --num_reset_states 32 --num_envs 16 \
    --randomize_robot_init \
    --robot_init_ee_x_range 0.25 0.65 --robot_init_ee_y_range -0.35 0.35 \
    --robot_init_ee_z_min 0.20 --seed 0 --headless
```

## Through the pipeline

When `run_datagen_pipeline.py` runs this stage, it sets the flags above from its own:

| Pipeline flag | Default | Becomes | Notes |
|---|---|---|---|
| `--env-per-problem` | `32` | `--num_reset_states` | Despite the name, this is the number of **reset states** per problem, not environments. It must be divisible by `--num-envs`. |
| `--num-envs` | `16` | `--num_envs` | Parallel envs during reset-state sampling. |
| `--num-scene-profiles` | `8` | `--num_scene_profiles` | Baked into the package. |
| `--usd-library` | `train_blocks_large` | `--usd_library` | Glyph asset library. |
| `--block-x-range`, `--block-y-range` | `""` | `--block_x_range`, `--block_y_range` | Only forwarded when set. |
| `--randomize-robot-init` and its range flags | off | `--randomize_robot_init` and the matching `--robot_init_*` flags | Only forwarded with `--randomize-robot-init`. |
| `--num-reset-states` | — | — | **Removed from the pipeline** (the standalone script below still has it). The pipeline always passed `--env-per-problem`, so this is now rejected with an error instead of ignored. |
| `--seed` | `42` | — | Does not reach this stage. The env-generation seed is always `0`, so a different `--seed` regenerates identical reset states. |

The pipeline runs one env-generation process per problem, spread over the GPU slots (see
[Running the pipeline locally](local.md)), and skips any problem whose `reset_states.pkl` already
exists.

With `--vecgen-problems-per-env > 1` (the default is `8`) this stage is skipped. Each execution
job first builds one multi-problem package for its group of problems with
`scripts/vecgen/make_superset_env.py`: the union of the group's blocks, the blocks a problem does
not use parked out of the way, `--env-per-problem` reset states per problem, and a
`problems.json` that lists each problem's PDDL and reset-state range. The package lands in
`envs/problem_grp_<first slug>/`. `--block-x-range` / `--block-y-range` are forwarded and resolved
per problem exactly as on the per-problem path (the flag, else that problem's `;block_x_range` /
`;block_y_range` comment, else the default window). `--seed` behaves differently: it does reach
this path (the package for the group starting at problem *g* is sampled with seed `--seed + g`).
See
[Flags that do not do what their name suggests](local.md#flags-that-do-not-do-what-their-name-suggests).

## Output

The package directory `<output_dir>/<slug>/` contains six files:

| File | Contents |
|---|---|
| `env.py` | The generated env class, with the block names and specs, the randomization dict and `MAX_NON_SIM_STATE_RANDOMIZATIONS` written inline. It starts with `# GENERATED CODE`. |
| `env_cfg.py` | The generated config class, with `scene_preset_id` and `robot_id` overridden if you passed them. |
| `__init__.py` | A `gym.register` call for the id `<slug>-v0`. |
| `non_sim_state_randomizations.json` | The raw randomization tables: per-block `block_sizes`, `table_height` and `friction`. |
| `profiles.json` | The same values seen per scene-profile slot, plus a `codegen_version` provenance stamp. |
| `reset_states.pkl` | The sampled snapshots, with a `profile_ids` array recording which scene profile each was sampled under. Only written when `--num_reset_states > 0`. |

An env package is tied to the code that generated it. `profiles.json` carries a
`codegen_version`, and evaluation **refuses** a package whose version does not match the current
checkout, so regenerate packages after updating the code.

Watch the log for the sampling warnings described under [Common pitfalls](#common-pitfalls).

## Flags

| Flag | Default | Meaning |
|---|---|---|
| `--pddl` | `toyblocks/tamp/blocks/problems/pnp_on_table.pddl` | The problem file. Its basename becomes the task slug. The default no longer works: that file does not exist any more, and the path is missing the leading `src/`. Omitting `--pddl` therefore raises `FileNotFoundError`, so always pass it. |
| `--domain_path` | `src/toyblocks/tamp/blocks` | The folder holding `template/template_env.py` and `template_env_cfg.py`. |
| `--output_dir` | `.` | The directory that will **contain** `<slug>/`. An existing `<slug>/` is deleted first; see [Common pitfalls](#common-pitfalls). |
| `--num_reset_states` | `1` | The number of snapshots to sample. It must be an exact multiple of `--num_envs`. `0` means code generation only. |
| `--num_envs` | `1` | The number of parallel envs used for sampling. |
| `--num_scene_profiles` | `8` | The number of distinct (block size, table height, friction) profiles baked into the package. |
| `--seed`, `-s` | `0` | Seeds both code generation and sampling. |
| `--block_x_range` | `""` | The table-frame x window for blocks spawned on the table, overriding `;block_x_range` in the PDDL. Give one interval, `"[-0.2, 0.58]"`, or several disjoint ones, `"[[-0.2,-0.1],[0.3,0.5]]"`. |
| `--block_y_range` | `""` | The same for y. The sampler defaults are `[-0.1, 0.25]` for x and `[-0.30, 0.30]` for y. |
| `--randomize_robot_init` | off | Give each reset state a random collision-free arm configuration and gripper state. |
| `--robot_init_ee_x_range` | `(0.25, 0.65)` | The end-effector x window, in the robot base frame. |
| `--robot_init_ee_y_range` | `(-0.35, 0.35)` | The end-effector y window. |
| `--robot_init_ee_z_min` | `0.20` | The lowest end-effector height. There is no flag for the upper bound: it is the end-effector height of the home configuration. |
| `--robot_init_candidates` | `32` | The number of poses tried per env before falling back to the home pose. |
| `--robot_init_gripper_closed_prob` | `0.5` | The probability that a reset state starts with the gripper closed. |
| `--scene_preset_id` | `""` (template: `sim2real`) | The scene preset baked into the generated `env_cfg.py`. |
| `--robot_id` | `""` (template: `fr3_robotiq_gripper_sysid`) | The robot baked into the generated `env_cfg.py`. |
| `--usd_library` | `""` (→ `train_blocks_large`) | The glyph asset library that bare block names resolve against. The only other value is `train_blocks`. |
| `--my_verbose` | off | Print per-predicate sampling details. This is a different flag from `--verbose`. |
| `--headless` | off | Inherited from `BaseArgs`; turns the display off. |

Only `--block_x_range` and `--block_y_range` can be overridden here. There is **no**
`--block_size_range`, `--table_height_range` or `--friction_range` flag, because those values
come from the PDDL comment lines written by
[stage 1](task_generation.md#how-scene-randomization-reaches-the-simulator).

## Common pitfalls

### Re-running env generation deletes the previous output

If `<output_dir>/<slug>` already exists, it is removed with `shutil.rmtree` before anything is
written. There is no `--force` flag and no confirmation, so re-running the same problem throws
away the previous `reset_states.pkl`. Copy it elsewhere first if you want to keep it.

### Reset-state and env counts must line up

Two size constraints apply, and they are easy to mix up:

- `--num_reset_states` must be an **exact multiple** of `--num_envs`, enforced by a bare
  `assert`. For example, 8 reset states with 16 envs fails rather than running one partial batch.
- `--num_envs` must be **a multiple of the profile count, or strictly smaller than it**. With the
  default 8 profiles, `1`–`7`, `8`, `16` and `24` are all valid, but `12` and `20` are not. A value
  below the profile count simply spawns only the lower profile slots.

### Changing the profile count does not affect existing packages

`--num_scene_profiles` is a property of the **package**, not of a run. Isaac cannot reconfigure
assets at runtime, so a new value only affects packages generated from then on. Every later stage
also inherits the `--num-envs` constraint from the value baked into the package.

### `--enable-cameras` is silently ignored

`tyro` accepts both spellings, but the Isaac flags inherited from `BaseArgs` are parsed *twice*.
Isaac's own `AppLauncher` runs first, with `parse_known_args`, and declares them with
underscores. So `--enable-cameras` is **silently discarded** and cameras stay off, while
`--enable_cameras` works. The same applies to `--kit_args` and `--rendering_mode`. Always use the
underscore spelling for Isaac flags.

This stage does not need cameras anyway: the front-view visibility check is pure projection
maths, not rendering.

### `--help` takes tens of seconds

`launch_app()` runs at import time, before `tyro` parses anything, so asking for help starts the
simulation app first. Expect a few seconds on a warm cache (much longer on a cold one), a GPU
allocation, and the need for `--headless` on a machine without a display, just to see the usage
listing. The [Flags](#flags) table above is the faster reference.

### Sampling can give up without an error

Each predicate gets at most 200 sampling rounds, and envs that never satisfy it are dropped from
the pending list **with no error**. Separately, envs whose blocks land outside the front camera
view are resampled at most 20 times. After that the script prints
`Warning: N env(s) still have a block outside the front camera view` and keeps those states
anyway. Watch the output for that warning, and for
`[on_table] WARNING: only N% of the spawn window ... is usable`, which fires when less than 20% of
the window is usable.

Not every `:init` predicate is sampled. `at` and `clear` are skipped outright, and a predicate
with no registered sampler is skipped with only a `--my_verbose` message. The samplers that exist
are `on_table`, `on`, `ori_of`, `next_to`, `on_right`, `on_left`, `on_front` and `on_back`.

## Next

Feed the env package to [stage 3, execution](execution.md), which plans and records episodes
starting from these reset states.
