# Stage 1: Task generation

Before the simulator can produce any demonstrations, the pipeline needs to know what task to
solve. Task generation answers that question by writing **PDDL problem files**. PDDL (Planning
Domain Definition Language) describes a task as two lists: the facts that are true at the start,
and the goal that must be true at the end. Both are written with predicates such as `on`,
`on_table` and `ori_of`, so a goal like "the red block is on the green block" becomes one line
of PDDL.

Each generator in `scripts/task_gen/` describes a whole *family* of tasks, such as "flip one
block" or "stack three blocks". It builds a declarative `ProblemSpec` (the blocks, distractor
pools, init rules and goal), and `ProblemSpec.generate()` then samples `--count` concrete problems
from it, writing one `.pddl` file per problem into `--out-dir`. This stage is pure CPU: it needs no
Isaac and no USD assets, and finishes in a few seconds of Python.

Usually the pipeline runs this stage for you. Run it by hand when you want to inspect the
problems a generator produces, or to build a problem set once and reuse it.

## Where it fits

- **Input:** nothing but the generator's own flags.
- **Output:** a directory of `.pddl` problem files, read by [stage 2, env generation](env_generation.md).

The stage is optional. You can skip it and feed stage 2 the checked-in problems in
`src/toyblocks/tamp/blocks/problems/`, or PDDL you wrote by hand.

## Running it

The smallest useful run picks a generator and tells it how many problems to write and where:

```bash
python scripts/task_gen/flip.py --count 50 --out-dir outputs/flip/tasks --seed 42
```

`--count` is the number of problems, `--out-dir` is where the `.pddl` files go, and `--seed` makes
the sampling reproducible. Always pass `--out-dir`, for the reason explained under
[Common pitfalls](#common-pitfalls).

To look at what a generator produces without writing anything, add `--preview`. It prints one
sampled problem to the terminal:

```bash
python scripts/task_gen/stack_three.py --train --seed 42 --preview
```

### Examples

One command per task family, ready to copy:

```bash
# Flip block_r once (any direction, goal from the flip table)
python scripts/task_gen/flip.py \
    --count 50 --out-dir outputs/flip/tasks --seed 42

# Rotate block_r by one yaw step to o2
python scripts/task_gen/rotate.py --goal-ori o2 \
    --count 50 --out-dir outputs/rotate/tasks --seed 42

# Pick-and-place block_r onto block_green, or off it onto the table
python scripts/task_gen/pnp.py \
    --count 50 --out-dir outputs/pnp/tasks --seed 42

# 3-block towers in letter + digit + colour scenes (--train or --val glyph pools)
python scripts/task_gen/stack_three.py --train \
    --count 50 --out-dir outputs/stack3/tasks --seed 42

# Play-data scenes (used by scripts/datagen/run_playdata_pipeline.py)
python scripts/task_gen/play_scene.py \
    --count 50 --out-dir outputs/play/tasks --seed 42

# Search scenes for the DFA goal policies (src/toyblocks/playdata/dfa/AUTHORING.md)
python scripts/task_gen/find_as.py --count 5 --out-dir outputs/find_as/tasks --seed 42
python scripts/task_gen/search.py --letter R --distractors 2 4 \
    --count 5 --out-dir outputs/search/tasks --seed 42
```

## Through the pipeline

In `run_datagen_pipeline.py` you choose the generator with `--task-gen-script` (default
`scripts/task_gen/stack_three.py --train`). The string can carry the generator's own flags as well:

```bash
python scripts/datagen/run_datagen_pipeline.py --experiment-id nov04_stack3 \
    --task-gen-script "scripts/task_gen/stack_three.py --num-blocks 4 6 --glyph-ori o2" \
    --task-gen-count 50 --task-gen-seed 42
```

The pipeline splits that string with `shlex.split` and then **appends** its own values for
`--count`, `--out-dir`, `--seed`, `--block-size-range`, `--table-height-range`, `--friction-range`
and `--num_distractors`. If you also set one of those inside the string, it is passed twice, and
argparse keeps the last one, which is the pipeline's. So set them with the pipeline's flags
(`--task-gen-count`, `--task-gen-seed`, `--num-distractors` and so on) rather than inside the
string.

## Output

Each problem is one file, named `problem_<internal problem name>.pddl`. For the declarative
generators that is `problem_declarative_<n_blocks>_<timestamp>_<random>.pddl`. The body is a
standard PDDL problem:

```text
(define (problem ...) (:domain blocks-tamp) (:objects ...) (:init ...) (:goal ...))
```

The scene-randomization comment lines described below are appended after the goal.

To check a run, open one of the files, or rerun the same generator with `--preview` and read the
`:init` and `:goal` sections.

## The generators

There are 7 generators. They fall into three families.

**Single-block skills.** One target block performs one skill.

| Script | Task family |
|---|---|
| `flip.py` | One contact flip of a target block (default `block_r`). `--direction` constrains the flip, and `--goal-ori` pins the target face. |
| `rotate.py` | One yaw step (clockwise or counter-clockwise) of a target block. `--direction cw|ccw` and `--goal-ori` work as on `flip.py`. |
| `pnp.py` | One pick-and-place of `block_r`, keeping its orientation: onto a support block (default `block_green`), or off it onto the table. `--mode stack|unstack|all`. |

**Stacking.**

| Script | Task family |
|---|---|
| `stack_three.py` | 3-block towers in a mixed letter, digit and colour scene of 4–10 blocks, with `--train` and `--val` glyph pools. Every glyph block in the tower must end readable; symmetric letters also accept the orientations that show the same glyph. `--init-readable`, `--init-flip-unreadable` and `--init-ori` control the starting orientations. This is the default generator, and the one the RI training set was made with. |

**Play and search scenes.** The PDDL goal exists only to satisfy code generation; the play
pipeline picks its own goals per episode.

| Script | Task family |
|---|---|
| `play_scene.py` | A few glyph blocks (`--blocks`, default `A I 2 E`) on the table, clear, at random orientations. |
| `search.py` | One target letter (`--letter`, default `R`) that starts non-readable, among `--distractors MIN MAX` (default 2–4) distractors. |
| `find_as.py` | Four `A` blocks plus six distractor letters, flat on the table; five fixed distractor sets. |

## How scene randomization reaches the simulator

The pipeline can vary block size, table height and friction from one environment to the next.
These are physical properties rather than facts about the task, so they are **not** PDDL
predicates. Instead, the generator writes them as comment lines at the end of the problem file:

```text
    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [-0.05, 0.05]
    ;friction_range = [0.5, 0.5]
```

[Stage 2](env_generation.md) reads these lines back and bakes a set of scene profiles from them.
Two details of the format matter if you ever edit these lines by hand; both are covered under
[Common pitfalls](#common-pitfalls).

No generator writes `;block_x_range` or `;block_y_range`. The window where blocks may spawn on
the table is set by a stage 2 flag instead.

Through the pipeline, scene randomization is on by default (`--randomize-scene`), and the ranges
above are the pipeline's defaults. With `--no-randomize-scene` or `--smoke-test`, the pipeline
**overwrites** whatever ranges you passed with the fixed values `[0.04, 0.04]`, `[0.0, 0.0]` and
`[0.4, 0.4]` after parsing. So `--no-randomize-scene --block-size-range 0.03 0.05` discards the
range rather than honouring it.

## Flags

Most generators share these flags. Each generator also has its own, listed by `--help`.

| Flag | Usual default | Meaning |
|---|---|---|
| `--count` | `5` | The number of problems to write. `play_scene.py` defaults to `1`. |
| `--out-dir` | `None` | The directory the `.pddl` files go into. Always pass it; see [Common pitfalls](#common-pitfalls). |
| `--seed` | `None` | The seed for `random.seed()`. `find_as.py` defaults to `42`, and `play_scene.py` to `0`. |
| `--preview` | off | Print one sampled PDDL problem and write nothing. Available on every generator except `play_scene.py`. |
| `--block-size-range MIN MAX` | `[0.03, 0.05]` | The cube edge length range, written as a `;block_size_range` comment. |
| `--table-height-range MIN MAX` | `[0, 0.04]` | The table height offset range, written as `;table_height_range`. |
| `--friction-range MIN MAX` | `[0.1, 0.4]` | The friction range, written as `;friction_range`. |

`stack_three.py` and `play_scene.py` have no default for the three range flags: run by hand
without them, they write no randomization comment at all, and stage 2 uses its fixed fallbacks.
The pipeline always passes all three.

## Common pitfalls

### Problems land in `src/toyblocks/Tasks/` when `--out-dir` is omitted

You might expect a missing `--out-dir` to mean `./outputs`. The generators instead fall back to
`src/toyblocks/Tasks/`, inside the source tree. The exception is `play_scene.py`, which makes
the flag required. Always pass `--out-dir` so the files end up where you expect.

### A hand-edited randomization line is silently ignored

The parser in stage 2 looks for the exact text `;block_size_range = [`, so **the ` = [` spacing
matters**. A line written as `;block_size_range=[0.03,0.05]` does not match and is ignored without
a warning. You then get the fixed fallbacks instead: block size `0.04`, table delta `0` and
friction `0.3`. Keep the spacing exactly as the generators write it.

### Problem files from elsewhere are executed as code

The randomization values are parsed with `eval()`, so a hand-edited PDDL comment is run as
Python. Treat problem files from someone else as code, not data, and read them before using them.

### `--num_distractors` needs an underscore, and its meaning varies

Every generator spells the flag with an underscore and rejects the hyphenated
`--num-distractors`. The pipeline passes the underscore form, which is why it works there. Use
`--num_distractors` when you run a generator by hand.

What the number means also depends on the generator:

- **The maximum of a 0..N range** (default `5`): `flip.py`, `rotate.py`, `pnp.py`. Set
  `--distractor-count-min` to the same value for an exact count, and `--distractor-blocks` to
  replace the built-in pool.
- **Ignored**: `stack_three.py`, `play_scene.py`, `search.py`, `find_as.py`. `search.py` takes
  its distractors from `--distractors MIN MAX` instead; the others have fixed scene recipes.

Through the pipeline, `--num-distractors` defaults to `0`, and it is always passed. So a
pipeline run with `flip.py`, `rotate.py` or `pnp.py` produces scenes **without distractors**
unless you pass `--num-distractors N`, even though the generator on its own defaults to up to 5.

### `flip.py` and `rotate.py` do not target `o2` by default

Without `--goal-ori`, both generators run in *dynamic-goal* mode. The initial orientation is drawn
uniformly from all 24, and the goal is computed forward through the flip or yaw table, so **every
problem gets a different goal face**. `flip.py` also flips in any of the four directions. If you
want every problem to end on the same face, pass `--goal-ori o2`; the generator then fixes the
goal and works backwards to the initial orientation.

### Renaming a problem file renames its task

Downstream stages name a task after its file, not after the `(problem ...)` name inside it.
`task_slug_from_pddl_path` takes the file's basename stem and replaces hyphens with underscores.
So renaming a generated `.pddl` file silently renames the task, and two problems with the same
basename collide in one output directory.

### `--block` is misread on `rotate.py`

argparse accepts any unique prefix of a flag name, and `--block` is a prefix of
`--block-size-range`. `flip.py` and `pnp.py` turn this off with `allow_abbrev=False` (and
`flip.py` accepts `--block` as an alias of `--target-block`), but `rotate.py` does not, so on
`rotate.py` `--block foo` is taken as `--block-size-range` and fails with "expected 2
arguments". Use `--target-block` there.

## Next

Feed the problems to [stage 2, env generation](env_generation.md), or let
[the pipeline](../data_generation.md) run every stage for you on this machine
([Running the pipeline locally](local.md)).
