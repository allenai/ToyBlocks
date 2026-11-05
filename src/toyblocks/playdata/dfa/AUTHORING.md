# Authoring DFA Strategies

How to write a new DFA YAML for a new task and strategy, and every input needed to run it.

A **DFA is a per-environment goal policy**: it watches the symbolic state and proposes the next PDDL
goal. The planner and executor are unchanged — you author only the goal-selection logic.

---

## 1. Mental model

Per environment the loop is: read the symbolic state → **DFA picks the next goal** → write a PDDL
problem → TAMP solves it → execute one episode → repeat, never resetting until the DFA accepts or
rejects.

A DFA is **states** + ordered **transitions**. Each transition has:

- a **guard** — when it fires,
- optional **effects** — writes to per-env memory,
- an optional **move** — the emitted PDDL goal (omit for a pure state change),
- a **next** state — a state name, or the reserved `ACCEPT` / `REJECT`.

Reaching `ACCEPT` or `REJECT` **terminates** the search; the env then resets to a fresh reset-state and
the DFA restarts. Every environment runs the same YAML with its **own seeded RNG**, so `bind_random_block`
and the random moves diverge — 8 envs = 8 independent searches.

---

## 2. Anatomy of a spec

```yaml
name:    my_strategy          # identifier for logs (optional)
initial: START                # start state (required; a key of `states`)
memory:  { n: 0, found: [] }  # initial per-env memory (optional)
params:  { letter: A, k: 3 }  # values for $token substitution (optional)
states:                       # required
  START:                      # a state = an ORDERED list of transitions
    - guard:   <GuardNode>    # default: always
      effects: [ <EffectNode> ]  # optional; run when this transition fires
      emit:    <MoveNode>     # optional; the goal to pursue (null = pure transition)
      next:    NEXT_STATE     # required; a state, or ACCEPT / REJECT
```

**Named nodes** — a guard/effect/move is a bare string (no params) or a single-key mapping
`{name: {param: value}}`:

```yaml
always                                    # guard, no params
{ counter_lt: { name: n, n: $k } }        # guard with params ($k from `params`)
{ flip: { var: c, direction: random } }   # move
{ incr: { name: n } }                     # effect
{ not: { found_letter: { letter: A } } }  # combinator: not / all / any
```

`$token` anywhere outside `params` is replaced at load time by `params[token]`, overridable per run
with `--dfa-param token=value`.

---

## 3. The vocabulary

Get this live at any time:

```bash
python -c "from toyblocks.playdata.dfa import catalog; import json; print(json.dumps(catalog(), indent=2, default=str))"
```

### Guards — decide if a transition fires (→ bool)

| guard | params | fires when |
|---|---|---|
| `always` | — | always (default / fall-through edge) |
| `found_letter` | `letter`, `oris=`(8 camera-visible) | the block whose asset is `letter` is at a camera-visible orientation |
| `upright_letter` | `letter`, `oris=(o2,)` | that block is upright / cleanly readable |
| `ori_is` | `var`, `o` | the block bound to `var` is at orientation `o` |
| `counter_ge` / `counter_lt` | `name`, `n` | memory counter `name` ≥ / < `n` |
| `collected_ge` | `name`, `n` | memory *list* `name` has ≥ `n` items |
| `holds` | `pred`, `args` | raw predicate; args may be literals or `@memkey` |

Combinators `not`, `all`, `any` nest guards.

### Effects — mutate per-env memory when a transition fires

| effect | params | does |
|---|---|---|
| `bind_random_block` | `var`, `clear_only`, `glyph`, `exclude_glyph` | bind `var` to a random block (blind pick; optional filters) |
| `bind_letter_block` | `var`, `letter` | bind `var` to the block whose asset is `letter` |
| `incr` | `name`, `by=1` | increment a counter |
| `reset` | `name`, `value=0` | reset a counter/value |
| `set` | `name`, `value` | set a memory value |
| `record_found_letter` | `var`, `letter`, `into` | if `var` is a revealed `letter` block, append it to list `into` (deduped) |

### Moves — the emitted PDDL goal for one episode (→ ProposedGoal)

| move | params | emits |
|---|---|---|
| `flip` | `var`, `direction`(forward/back/left/right/**random**), `turns` | `(ori_of B oN)` — one quarter-turn from current |
| `rotate` | `var`, `turns` | `(ori_of B oN)` — spin about Z |
| `flip_or_rotate` | `var`, `rotate_prob=0.5` | random flip OR rotate (the honest random-walk move) |
| `set_ori` | `var`, `o` | `(ori_of B o)` — absolute target (⚠ see §5) |
| `on_table` | `var` | `(on_table B table)` |
| `stack` | `a`, `b` | `(on A B)` |
| `next_to` | `a`, `b` | `(next_to A B)` |

`var` names a **memory key** whose value is a block name (bound by an earlier effect).
`stack`/`next_to` args may be a memory key or a literal block name.

---

## 4. Execution semantics

One `next_goal` call walks transitions until it emits a goal or a decision:

1. Current state is `ACCEPT`/`REJECT` → **TERMINATE** (env resets, DFA restarts).
2. Evaluate the state's transitions top-to-bottom; the **first** guard that passes fires. None pass →
   **RETRY** (re-snapshot soon).
3. Run its `effects`, move to `next`.
4. If it has a `move`: the move returns a goal → **GOAL** (plan + execute one episode); returns nothing
   → RETRY. If `emit` is null (a *pure transition*) → loop to step 1 immediately, so setup chains like
   *pick → reset counter → move* resolve in a single call.

Memory is a per-env dict, reset to `memory` at the start of each search. The reserved key `_reached`
holds the last plan's goal-reached flag if a guard wants it.

---

## 5. The ground-truth rule (keep it distillation-safe)

Read ground truth to build a *single-action* goal or to decide *"done"* — never to steer the action.

- **Moves** read the block's current orientation only to express one random flip/rotate as a concrete
  `(ori_of B oN)` goal — a legitimate single-action planning problem.
- **Found guards** (`found_letter`, `record_found_letter`) use ground-truth identity + orientation as a
  stand-in for "the camera sees the letter" — a *termination* signal, not an action.
- **Avoid** `set_ori(c, o2)`-style moves for training data: pointing the block straight at the readable
  face bakes privileged knowledge into the action sequence, so a distilled policy learns the shortcut,
  not the real search. Use `flip_or_rotate`.
- **Blind picks:** keep `bind_random_block` asset-blind (no `glyph:` filter) so the search doesn't
  magically select the target block.

---

## 6. Worked example — a new strategy from scratch

"Find K blocks of a target letter by random-walk flipping." This is `library/find_as.yaml`:

```yaml
name: find_as
initial: PICK
memory: { n: 0, picks: 0, found: [] }   # found = distinct revealed target blocks
params: { letter: A, target_count: 3, moves_per_block: 2, max_picks: 60 }
states:
  PICK:                                 # pick a random clear block, reset the move counter
    - guard: always
      effects:
        - bind_random_block: { var: c, clear_only: true }
        - reset: { name: n }
        - incr:  { name: picks }
      next: MOVE
  MOVE:                                 # random-walk it a few times, then look
    - guard: { counter_lt: { name: n, n: $moves_per_block } }
      emit:  { flip_or_rotate: { var: c } }
      effects: [ { incr: { name: n } } ]
      next: MOVE
    - guard: always
      next: CHECK
  CHECK:                                # if it is a revealed target letter, record it
    - guard: always
      effects: [ { record_found_letter: { var: c, letter: $letter, into: found } } ]
      next: DECIDE
  DECIDE:
    - guard: { collected_ge: { name: found, n: $target_count } }
      next: ACCEPT
    - guard: { counter_ge: { name: picks, n: $max_picks } }
      next: REJECT
    - guard: always
      next: PICK
```

Reusable recipe: a **PICK** state that binds a fresh subject and resets a counter, a **MOVE** self-loop
gated by a counter, a **CHECK** that records progress, and a **DECIDE** state that accepts on success,
rejects on a budget, else loops. Change the move, the guard, and the params to make a different task.

---

## 7. Running a new DFA — every input

### 1 · Scene → PDDL (a task generator)

Copy `scripts/task_gen/find_as.py` or `search.py`. You specify: which glyph/color blocks, how many,
their init (on-table, clear, random or fixed orientations), and a groundable PDDL goal (the play loop
ignores it, but stage-1 codegen needs one).

```bash
python scripts/task_gen/find_as.py --count 5 --out-dir outputs/mytask/tasks --seed 42
```

### 2 · Env package + reset states (stage 1)

```bash
python scripts/generate_task_from_pddl.py \
  --domain_path src/toyblocks/tamp/blocks \
  --pddl outputs/mytask/tasks/<problem>.pddl \
  --num_reset_states 16 --num_envs 16 --num_scene_profiles 8 \
  --output_dir outputs/mytask --headless
# -> outputs/mytask/<slug>/  (env package + reset_states.pkl)
```

### 3 · Run the DFA (vecgen play: FF plans each goal, vecgen executes it)

```bash
python scripts/vecgen/run.py --play \
  --env-path outputs/mytask/<slug> \
  --pddl outputs/mytask/tasks/<problem>.pddl \
  --domain src/toyblocks/tamp/blocks/domain.pddl \
  --dfa-spec src/toyblocks/playdata/dfa/library/find_as.yaml \
  --dfa-param letter=A target_count=3 \
  --num-envs 8 --max-plans-total 800 \
  --disallowed-actions push_to_right push_next_to --missed-goals-to-failure \
  --record datasets/mytask --summary datasets/mytask/summary.json \
  --record-videos --enable_cameras --headless
```

The play pipeline takes the same strategy: `scripts/datagen/run_playdata_pipeline.py ... --dfa-spec <yaml>
--dfa-param k=v ...` (add `--play-record-videos` for camera videos while playing).

| input | what it controls |
|---|---|
| `--dfa-spec` | path to your strategy YAML (omit → random-goal proposer, the old behavior) |
| `--dfa-param k=v` | override a `params` value; repeatable (values parsed as YAML scalars) |
| `--pddl` | supplies the PDDL object declarations (the block universe) |
| `--env-path` | the stage-1 env package (its reset states seed each env) |
| `--num-envs` | parallel independent searches (envs are matched to reset states by scene profile) |
| `--max-plans-total` | stop after this many executed plans (the run's budget) |
| `--one-chain-per-reset-state` | after ACCEPT / REJECT move to the next unplayed reset state (else a random one) |
| `--record-videos --enable_cameras` | record the front / gripper cameras; without them Parquet only (images can come from the render stage) |

---

## 8. When YAML isn't enough — add a registry function

A new guard/effect/move is one decorated function in `registries.py`. It receives a `DFACtx` (live
symbolic `state`, per-env `memory` and `rng`, the `holds` predicate oracle, `glyph_of` asset identity,
the `table` name) and is referenced from YAML by its registered name.

```python
@guard("digit_readable")
def _digit_readable(ctx, *, at_least="1"):
    """A digit block >= `at_least` is camera-visible."""
    for b in ctx.state.blocks:
        g = ctx.glyph_of(b)
        if g and g.isdigit() and g >= str(at_least):
            if any(ctx.holds("ori_of", [b, o]) for o in FOUND_ORIS):
                return True
    return False
```

Then `{ digit_readable: { at_least: 3 } }` works in any YAML. Moves return
`ProposedGoal(predicates=["(pred args)"])`; effects mutate `ctx.memory` and return `None`.

---

## 9. Checklist & gotchas

- **Validate on load.** The loader rejects an unknown guard/move/effect, a dangling `next`, or a
  missing `initial`. Test offline:
  `python -c "from toyblocks.playdata.dfa import load_dfa_spec; load_dfa_spec('.../mine.yaml')"`.
- **Always give a budget out.** Include a `REJECT` edge (e.g. `counter_ge(picks, max_picks)`) so a
  search that never succeeds still terminates and resets, instead of looping forever.
- **Cluttered scenes (> 5 blocks):** keep the `isolate` action commented out in `domain.pddl` — its RL
  policy is sized for 5 distractors and crashes the batch otherwise.
- **"Found" orientations** are the 8 camera-verified poses in `registries.FOUND_ORIS` (rendered via
  `scripts/render_orientations.py`). Override per guard with `oris: [...]` if your asset differs.
- **Reproducible, divergent:** per-env RNG is seeded from the run seed + env index; re-running the same
  command reproduces the run, and each env still searches differently.

---

*Strategy = one YAML of states + guarded transitions. Task = a scene generator + stage-1 env. Run =
`--dfa-spec` + `--dfa-param`. The planner and executor never change. See the full grammar in
`engine.py`'s module docstring and the live vocabulary via `registries.catalog()`.*
