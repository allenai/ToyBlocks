# Adding a skill to vecgen

[Execution](execution.md) can only demonstrate what it has a **skill** for. The FF planner plans
over every action in `src/toyblocks/tamp/blocks/domain.pddl`, but each planned action is carried
out by a vecgen skill: a small Python class that turns "rotate `block_r` clockwise" into joint
targets, runs them on every environment at once, and then checks the result against the
simulation. When FF plans an action that has no skill, the environment gets no program and the
episode is recorded as a failure with reason `"no program"`.

The domain already contains actions that have no skill: `push_to_right` and `push_next_to`.
That is why the pipeline excludes them by default (`--vecgen-exclude-actions`). This page walks
through every place a new skill has to be wired in, from the PDDL action to the annotated
dataset.

The worked reference throughout is the **`rotate`** skill (`src/toyblocks/vecgen/skills/rotate.py`).
It maps one PDDL action to one skill, uses derived facts to keep FF away from rotates it cannot
do, and checks its effect with its own tolerance. Its grasp-and-twist geometry comes from scripted
helpers in `rotate_trajectory.py`, which a new skill does not need. `Pick`
(`src/toyblocks/vecgen/skills/pnp.py`, about 90 lines) is the smallest complete skill, and a good
template for a new one.

## Where it fits

A skill sits in the middle of a chain. Each link is a separate piece of code:

```text
domain.pddl action        FF plans it per environment
  -> symbolic/programs.py  skills_for_action(): PDDL action -> SkillCall(s)
  -> run.py skills dict    SkillCall name -> Skill instance
  -> Skill phases          begin / request / after_phase, resolved by the engine (IK, cuRobo, rows)
  -> Skill.verify()        effect check on the live sim: advance, retry, or re-plan
  -> recorder              action.high_level_* and action.low_level_* columns
  -> annotation            goal.action text and goal.position pixel
```

These are the files you may have to touch, in the order this page covers them:

| Step | File | When |
|---|---|---|
| Skill class | `src/toyblocks/vecgen/skills/<name>.py` | Always. |
| PDDL action | `src/toyblocks/tamp/blocks/domain.pddl` | Unless the action already exists. |
| Predicates | `src/toyblocks/vecgen/symbolic/derived.py`, `predicates.py`, `state.py`, and others | Only for predicates whose truth value must come from the scene. |
| Action to skill | `src/toyblocks/vecgen/symbolic/programs.py` | Always. |
| Registration | `scripts/vecgen/run.py` | Always. |
| Annotation | `src/toyblocks/data_annotation/annotation_utils.py` | When the action should get `goal.action` text and a target pixel. |
| Task generator | `scripts/task_gen/<name>.py` | When you want a family of problems, not one hand-written file. |

There is no plugin registry. Every link above is a hard-coded `if` branch, `dict` or `frozenset`,
so the work is editing these files directly.

## The skill interface

A skill subclasses `Skill` from `src/toyblocks/vecgen/skills/base.py` and lives next to the
existing ones in `src/toyblocks/vecgen/skills/`. The whole base class is this:

```python
class Skill:
    """Base class; subclasses set ``name`` and ``phases`` and implement the hooks."""

    name: str = ""
    phases: list[str] = []
    #: retry from the live state (no snapshot restore), e.g. a flip that moved the block
    retry_from_live: bool = False
    #: phases whose request() is expensive per env (scripted searches): built on a worker
    #: thread while the sim keeps stepping the other envs
    async_phases: frozenset[str] = frozenset()

    def begin(self, ctx: EngineContext, env_ids: Tensor, args: list[tuple[str, ...]]) -> Tensor:
        """Sample per-env parameters for a fresh attempt; return ``ok [B]``."""
        return torch.ones(int(env_ids.numel()), dtype=torch.bool, device=ctx.device)

    def request(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> PhaseRequest:
        raise NotImplementedError

    def skips(self, ctx: EngineContext, phase: str, env: int) -> bool:
        """True when ``phase`` is a no-op for env ``env`` this attempt (e.g. an optional
        sub-step it does not need): the engine moves past it without a segment, so no tick
        is spent or recorded and neither ``request`` nor ``after_phase`` runs for it."""
        return False

    def after_phase(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> None:
        """Bookkeeping once the phase's segment finished (attach/detach, cached poses)."""

    def verify(self, ctx: EngineContext, env_ids: Tensor) -> Tensor:
        """Effect check on the live state after the last phase; ``ok [B]``."""
        return torch.ones(int(env_ids.numel()), dtype=torch.bool, device=ctx.device)

    def hl_name(self) -> str:
        return self.name
```

The engine (`src/toyblocks/vecgen/engine.py`) calls the hooks in this order for each attempt of a
skill:

1. **`begin(ctx, env_ids, args)`** runs once, when an environment starts the skill's first phase.
   `args` holds each environment's `SkillCall` arguments, for example `("block_r", "cw", "o1", "o2")`
   for a rotate. Sample the attempt's parameters here (grasp pose, start pose, place spot), check
   them with IK, and store them for the later phases. Return a `[B]` bool mask. An environment
   whose entry is `False` fails this attempt with reason `"<name> begin: no feasible parameters"`.
2. **`request(ctx, phase, env_ids)`** runs once per phase. It returns one `PhaseRequest` that says
   what that group of environments should do next. The engine resolves it into one
   **segment** per environment, a table of joint targets that is played back one row per tick.
3. **`skips(ctx, phase, env)`** is asked before each phase after the first. Returning `True`
   skips the phase entirely: no segment, no recorded frame, and no `request` or `after_phase`
   call.
4. **`after_phase(ctx, phase, env_ids)`** runs when a phase's segments have finished playing. Use
   it for bookkeeping such as attaching a grasped block to the cuRobo robot model.
5. **`verify(ctx, env_ids)`** runs after the last phase, on the live simulation state, and returns a
   `[B]` bool mask. This is the skill's effect check; see
   [Effect checks, retries and re-planning](#effect-checks-retries-and-re-planning).

`hl_name()` is only a fallback for the recorded high-level action name. Skills planned from a
PDDL action always record that action's name instead (see
[Labels recorded for the dataset](#labels-recorded-for-the-dataset)).

### The batched contract

Skills are written for a batch of environments, not for one:

- `env_ids` is a `[B]` long tensor of environment indices: the group of environments that
  reached the same phase of the same skill on this tick. Every tensor a hook returns is `[B]` (or
  `[B, ...]`) in the same order as `env_ids`. `B` changes from call to call.
- `ctx` is the `EngineContext`. Its scene reads cover **all** `N` environments, so index them with
  `env_ids`: `ctx.block_poses()` is `[N, K, 7]`, `ctx.sim.ee_pose()` is `[N, 7]`,
  `ctx.sim.joint_pos()` is `[N, 7]`, `ctx.sizes` is `[N, K, 3]` (the measured collider edge
  lengths), and `ctx.held` is `[N]` (the index of the block in hand, `-1` for none). Poses are in
  the robot base frame as `[x, y, z, qw, qx, qy, qz]`. `ctx.block_index` maps a PDDL block
  name to its index along the `K` axis.
- `ctx.geo` is the batched cuRobo `Geometry`. `ctx.geo.ik(env_ids, goals, seeds=...)` returns
  `(ok, q, pos_err, rot_err)`, `ctx.geo.update_worlds(...)` refreshes the collision worlds, and
  `ctx.geo.attach(...)` / `ctx.geo.detach(...)` add or remove a held block from the robot model.
- Per-environment state lives in `ctx.params[e]`, a plain dict. One skill instance serves every
  environment and episode, so keep nothing per-environment on `self`. The engine clears
  `ctx.params[e]` when an episode starts and when it re-plans, but **not** between the skills of
  one program, so `begin` should set every key that the later phases read. The keys `_special`,
  `_outcome` and `_attempt` belong to the engine; `_attempt` holds the current attempt number,
  which `flip` uses to escalate its retries.
- Skills never step the simulation or write joint targets themselves. They only read the scene,
  run IK and planning through `ctx.geo`, and return requests.

### Phases and `PhaseRequest`

`phases` is the skill's state machine: a fixed list of phase names that every attempt walks
through in order. Each phase is one `PhaseRequest` (`src/toyblocks/vecgen/skills/base.py`), and
its `kind` says how the engine turns it into a segment:

| `kind` | What the engine does |
|---|---|
| `"plan"` | Plans a collision-free cuRobo trajectory to `goals` (`[B, 7]` end-effector poses). Plan requests from every group are batched into one pool call, which runs on a planner thread while the simulation keeps stepping. |
| `"ik_interp"` | Solves IK for `goals`, then interpolates `ticks` joint rows from the current joints. A goal is accepted when IK succeeds, or when it lands within 5 mm and 0.05 rad. |
| `"rows"` | Plays `rows[i]`, a `[T_i, 7]` joint-target tensor per environment, as given. Use it for scripted motions. |
| `"gripper"` | Holds the current arm target for `ticks` ticks while commanding the gripper `closed` (or to `gripper_target`). |
| `"hold"` | Holds the current arm target for `ticks` ticks. |
| `"policy"` | Hands the environments to a closed-loop runner in `extra["runner"]` until it reports done. No shipped skill uses it. |

The other fields apply to several kinds:

- `label` is the low-level action name recorded on every frame of the segment.
- `hold` adds ticks at the last row. On a `plan` or `rows` request, `settle_tol` ends the hold
  early once every joint is within `settle_tol` radians of the last row, so `hold` becomes a cap.
- `closed` or `gripper_target` on a `plan`, `ik_interp` or `rows` request also commands the
  gripper when the segment starts.
- `exclude` (`[B, K]` bool) removes blocks from the collision world for this phase, for example
  the block being grasped. `pad_fraction` and `no_pad` widen the remaining blocks in XY.
- `ok` (`[B]` bool) marks environments whose request could not be built. They fail the attempt
  with reason `"<name>/<phase>: request not formed"`, without touching the simulation.

A segment holds at most `EngineConfig.segment_capacity` rows (1500 ticks, 60 s). Installing a
longer one raises a `ValueError`, so split long scripted motions over several phases.

Here is how `rotate` declares its phases. A rotate may need two sub-steps (an alignment correction
plus the 90° turn), so it lists the sub-step phases twice:

```python
class Rotate(Skill):
    """``rotate(block, direction, ori, ori2)``."""

    name = "rotate"
    phases = ["open", "start", *_substep_phases(0), *_substep_phases(1)]
    async_phases = frozenset({"descend_0", "descend_1"})
```

Its `begin` looks for a start pose above the block that IK can reach, trying nine small XY
offsets for every environment in the batch at once:

```python
        ctx.geo.update_worlds(ctx.block_poses(), ctx.table_pose(), env_ids=env_ids)
        ok = torch.zeros(b, dtype=torch.bool, device=dev)
        chosen = torch.zeros(b, 7, device=dev)
        for dx, dy in START_OFFSETS:
            todo = ~ok
            if not bool(todo.any()):
                break
            sub = ar[todo]
            goal = torch.zeros(int(sub.numel()), 7, device=dev)
            goal[:, 0] = blk[sub, 0] + dx
            goal[:, 1] = blk[sub, 1] + dy
            goal[:, 2] = blk[sub, 2] + START_HEIGHT
            goal[:, 3:7] = start_q[sub]
            ok_c, _q, perr, rerr = ctx.geo.ik(env_ids[sub], goal, seeds=q_now[sub])
            acc = ok_c | ((perr <= 0.005) & (rerr <= 0.05))
            chosen[sub[acc]] = goal[acc]
            ok[sub[acc]] = True
```

`request` then reads the stored pose back from `ctx.params` and returns one request per phase.
The first two phases open the gripper and plan to the start pose:

```python
        if phase == "open":
            return PhaseRequest("gripper", "init_gripper_open_traj", closed=False, ticks=6)
        if phase == "start":
            # The descend anchors take the gripper's yaw from the CURRENT pose: arriving with the
            # wrist still lagging (~15-40 deg) grasps the block skewed and the fingers twist it.
            goals = torch.stack([ctx.params[e]["start_pose"] for e in envs])
            return PhaseRequest(
                "plan",
                "rotate_start_traj",
                goals=goals,
                hold=WRIST_SETTLE_MAX_TICKS,
                settle_tol=WRIST_SETTLE_TOL,
            )
```

When a block is grasped or released, `after_phase` keeps the cuRobo model and `ctx.held` in step,
so the next plan treats the block as part of the robot:

```python
    def after_phase(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> None:
        kind = phase.rsplit("_", 1)[0]
        if kind == "close":
            envs = [e for e in env_ids.tolist() if ctx.params[e].get("_twist_rows") is not None]
            if not envs:
                return
            ids = torch.tensor(envs, dtype=torch.long, device=ctx.device)
            j = torch.tensor(
                [ctx.params[e]["block"] for e in envs], dtype=torch.long, device=ctx.device
            )
            poses = ctx.block_poses()[ids]
            blk = poses[torch.arange(len(envs), device=ctx.device), j]
            ctx.geo.attach(ids, j, ctx.sim.ee_pose()[ids], blk)
            ctx.held[ids] = j
        elif kind == "release":
            ctx.geo.detach(env_ids)
            ctx.held[env_ids] = -1
```

A rotate that needs only one sub-step skips the second sub-step's phases, so they record no idle
frames:

```python
    def skips(self, ctx: EngineContext, phase: str, env: int) -> bool:
        """Sub-step 1's phases are no-ops for a rotate that was not split (one sub-target):
        skipped, they record no idle frames after the retreat."""
        if phase in ("open", "start"):
            return False
        return len(ctx.params[env]["subs"]) <= int(phase.rsplit("_", 1)[1])
```

### Expensive phases: `async_phases`

A phase whose `request` does slow per-environment work, such as a scripted IK search, can be listed
in `async_phases`. The engine then calls `request` on a worker thread (`--async-workers`, default
4) and keeps stepping the other environments meanwhile. IK called from a worker thread goes to a
separate cuRobo build pool (`--build-slots`). Because several worker threads may run the same
skill at once, `request` must not mutate shared state on `self`. `rotate` builds its descend rows
this way. An exception raised on the worker fails those environments' attempts with reason
`"<name>/<phase>: build error"`.

### How a skill finishes

A skill succeeds for an environment when its last phase finishes, or is skipped, and `verify`
returns `True`. The environment then moves to the next `SkillCall` of its program. An attempt
fails in any of these ways, each with its own recorded reason:

| Failure | Reason string |
|---|---|
| `begin` returned `False` | `<name> begin: no feasible parameters` |
| `PhaseRequest.ok` was `False` | `<name>/<phase>: request not formed` |
| IK failed on an `ik_interp` request | `<name>/<phase>: IK failed (<err> m)` |
| cuRobo found no plan | `<name>/<phase>: plan <status>` |
| A `rows` request had an empty row tensor | `<name>/<phase>: no rows` |
| An async build raised | `<name>/<phase>: build error` |
| `verify` returned `False` | `<name> effect check failed` |

A skill cannot abort halfway through a phase: once a segment is installed, it plays to the end.
To fail early, return `False` from `begin` or mark the environment in a request's `ok` mask.

## Effect checks, retries and re-planning

`verify` is what makes the data [correct by construction](execution.md#correct-by-construction).
It reads the live simulation and checks that the PDDL effect of the action actually holds.
`rotate` checks the block's orientation against the target, with a tighter yaw tolerance (15°)
than the 40° of the shared `ori_of` predicate:

```python
    def verify(self, ctx: EngineContext, env_ids: Tensor) -> Tensor:
        envs = env_ids.tolist()
        j = torch.tensor(
            [ctx.params[e]["block"] for e in envs], dtype=torch.long, device=ctx.device
        )
        idx = torch.tensor([ori_index(ctx.params[e]["ori2"]) for e in envs], device=ctx.device)
        poses = ctx.block_poses()[env_ids]
        q = poses[torch.arange(len(envs), device=ctx.device), j, 3:7]
        target = ORI_QUATERNIONS.to(ctx.device)[idx]
        return check_quat(q, target, VERIFY_YAW_TOL_DEG, AXIS_TOL_DEG)
```

The batched predicate functions in `src/toyblocks/vecgen/symbolic/predicates.py` (`on`,
`on_table`, `clear`, `ori_of`, `next_to`, `on_right` and the others) are the usual building blocks
for a check. Use them, or something stricter, so that a verified skill also satisfies the goal
check at the end of the episode.

Every failure from the table above goes through the engine's `_retry_or_fail`:

1. **Retry.** Up to `--max-attempts` attempts (default 3) per skill. Before each attempt the engine
   snapshots the environment: the simulation state, the held block, its cuRobo attachment, and the
   recorder's frame position. A failed attempt restores that snapshot and **drops the attempt's
   frames from the dataset**, then `begin` runs again with fresh samples.
2. **Retry from the live state.** A skill that sets `retry_from_live = True` (`flip` does, because
   the block has already moved) is retried without a restore, and its failed attempt's frames are
   **kept**.
3. **Re-plan.** Once the attempts run out, the engine drops any held block, reads the symbolic state
   back from the scene, and runs FF again toward the original goal. This happens at most twice per
   episode (`EngineConfig.max_replans`, no CLI flag).
4. **Fail.** If re-planning is exhausted or FF finds no plan, the episode ends as a failure with the
   last reason, and goes to the `blocks_failure` split.

After the last skill of a program, the PDDL goal is checked once more against the scene. An episode
is recorded as successful only if that check passes too; otherwise the reason is
`goal not satisfied: [...]`.

Re-planning reads back only a fixed set of facts (`derive_init_facts` in
`src/toyblocks/vecgen/symbolic/state.py`): `on`, `on_table`, `clear`, `ori_of`, `handempty`,
`gripper_at_init`, plus the derived facts (see [The PDDL side](#the-pddl-side)) and any
`next_to` / `on_right` / `on_left` / `on_front` / `on_back` / `pick_obstructs` fact the problem
declares and that still holds. If your action's effect is some other predicate, a re-plan does not
see it, and there is no hook for that: you have to extend `derive_init_facts` yourself.

## The PDDL side

### Adding the action

If the action is new, add an `(:action ...)` block to `src/toyblocks/tamp/blocks/domain.pddl`. The
`rotate` action takes the gripper, the block, a direction and the two orientations:

```text
  (:action rotate
    :parameters (?g - gripper ?b - block ?d - rot_dir ?o - ori ?o2 - ori ?table - table)
    :precondition (and 
      (initialized)
      (at ?g ?b)
      (handempty ?g)
      (on_table ?b ?table)
      (clear ?b)
      (ori_of ?b ?o)
      (yaw_next ?d ?o ?o2)
      (not (cant_rotate ?b))
      (not (on_table_side ?b ?table))
      (forall (?b2 - block) (not (rotate_obstructs ?b2 ?b ?d)))
```

The rest of its precondition rules out blocks that were pushed against a neighbour, and its effect
swaps the orientation:

```text
    :effect (and 
      (not (ori_of ?b ?o)) 
      (ori_of ?b ?o2)
    )
```

Three conventions are worth keeping:

- **Put the gripper first and the manipulated block second.** FF's argument list is recorded
  verbatim (lowercased) as `action.high_level_args`, and the annotator reads `args[1]` as the target
  object and, for direction actions, `args[2]` as the direction.
- **Require `(initialized)` and `(at ?g ?b)`.** Every manipulation action does. FF's `init` and
  `move_to` actions establish these facts, and vecgen ignores both, so they cost nothing at run
  time.
- **Reuse existing constants and predicates** (`ori_of`, `yaw_next`, `flip_next`, the `dir`
  types) where they fit, since every predicate you add may need Python code, as described next.

To check the action, run FF on a problem whose goal needs it. The vendored FF binary is in the
repository, so this needs no GPU:

```python
from pathlib import Path

from toyblocks.vecgen.symbolic.ff import run_ff
from toyblocks.vecgen.symbolic.pddl import parse_problem, render_problem

problem = parse_problem("src/toyblocks/tamp/blocks/problems/rotate.pddl")
result = run_ff(Path("src/toyblocks/tamp/blocks/domain.pddl"), render_problem(problem))
print(result.ok, [str(a) for a in result.actions])
# True ['(init)', '(move_to robot block_r block_r)', '(rotate robot block_r cw o1 o2 workspace_table)']
```

### Predicates

A predicate that only appears in the domain and is set by action effects or by the `init` action
(like `yaw_next`) needs nothing else. A predicate whose truth value comes from the scene needs
Python code, and where that code goes depends on who reads it:

| The predicate is used in... | Add it to |
|---|---|
| A precondition, true or false depending on the scene before planning | `derived_facts` in `src/toyblocks/vecgen/symbolic/derived.py` |
| A goal | `GEOMETRIC_PREDICATES` and `goal_facts_satisfied` in `src/toyblocks/vecgen/symbolic/predicates.py` (checked by vecgen), **and** `SUPPORTED_GOAL_PREDICATES` in `src/toyblocks/tamp/blocks/goal_predicates.py` with its tensor check in `TENSOR_GOAL_DISPATCH` in `pddl_goal_tensor_conversions.py` (compiled into the env package; the two are asserted equal at import) |
| A problem's `:init`, for which env generation samples poses | `SAMPLERS` in `src/toyblocks/vecgen/sampling/predicates.py` |
| A state that must survive a re-plan | `derive_init_facts` in `src/toyblocks/vecgen/symbolic/state.py` |
| A goal written by a task generator | A `Pred` subclass in `src/toyblocks/task_gen/predicates.py` |

`run.py` calls `assert_goal_supported` on every problem before it starts, so a goal predicate
missing from `GEOMETRIC_PREDICATES` fails immediately rather than at the first episode end.

### Facts derived from the scene

The most useful predicates for a new skill are usually **derived facts**: geometric conditions under
which the skill would fail, computed from the live scene before FF runs, so that FF plans around
them (for example, by moving an obstructing block first) instead of planning an action the skill
cannot carry out. `rotate` has two, `cant_rotate` (the block is out of the rotate workspace or out
of view) and `rotate_obstructs` (a neighbour is too close):

```python
    cant_rot = (
        (xy[..., 0] < ROTATE_X_RANGE[0])
        | (xy[..., 0] > ROTATE_X_RANGE[1])
        | (xy[..., 1] < ROTATE_Y_RANGE[0])
        | (xy[..., 1] > ROTATE_Y_RANGE[1])
        | ~in_view
    )
    for e in range(b):
        for j in range(k):
            if bool(cant_rot[e, j]):
                out[e].append(Fact("cant_rotate", (blocks[j],)))
        for x, j in rot_obs[e].nonzero().tolist():
            for d in ROTATE_DIRS:
                out[e].append(Fact("rotate_obstructs", (blocks[x], blocks[j], d)))
```

`derived_facts` runs for every environment when its episode is assigned, and again on every
re-plan, unless `--no-derived` is passed. Keep its limits and the skill's `begin` in agreement:
every fact that rules out an action here is one less attempt that fails at run time.

## Mapping the action to the skill

`skills_for_action` in `src/toyblocks/vecgen/symbolic/programs.py` turns each action of an FF plan
into one or more `SkillCall`s. It unpacks the action's positional arguments and keeps the ones the
skill needs. `rotate` maps one-to-one:

```python
    if name == "rotate":
        _, b, d, o, o2, _t = a
        return [SkillCall("rotate", (b, d, o, o2))]
```

One action may also expand into several skills. `pnp_table_block`, for example, becomes
`pick(b1)` followed by `place_on(b1 b2 o2)`. The module docstring lists every mapping. Two sets at
the top of the file matter too:

```python
IGNORED = {"init", "move_to", "done", "reach-goal", "stop"}
UNSUPPORTED = {"push_to_right", "push_next_to", "isolate"}
```

If you write a skill for an action in `UNSUPPORTED`, add its branch and remove it from the set.
Play mode (`--play`) uses the set to strip actions from the domain before FF ever sees them.
Datagen mode does not filter the domain with it: when FF plans an action that is in the set, or
that has no branch, `skills_for_action` raises `UnsupportedAction` and the environment gets no
program.

`program_from_plan` copies the PDDL action's name, its full argument list and FF's symbolic state
before the action onto every `SkillCall` the action produces (`hl_name`, `hl_args`, `hl_state`).
These are what the recorder writes, so you do not set them yourself.

## Registering the skill

The engine looks a skill up by the `SkillCall` name. The table is a plain dict in `main()` of
`scripts/vecgen/run.py`, and the same dict is used for play data:

```python
    skills: dict[str, Skill] = {
        "pick": Pick(),
        "place_on": Place("on"),
        "place_table": Place("table"),
        "place_side": Place("side"),
        "flip": Flip(profile=args.profile),
        "rotate": Rotate(),
    }
```

Import your class at the top of `run.py` and add it under the name `skills_for_action` uses. Keep
the dict key and the class's `name` the same: some failure reasons are built from the key and
others from `name`. A `SkillCall` whose name is not in the dict ends its episode with reason
`unsupported skill <name>`.

Evaluation's low-level oracle, `src/toyblocks/eval/ll_oracle.py`, builds its own skill dict for the
commands it parses. Add the skill there only if that oracle should be able to run it.

## Labels recorded for the dataset

Whenever a segment is installed, the engine tells the recorder which action is running:

```python
                self.rec.symbolic(
                    e, cur.hl_name or skill.hl_name(), cur.hl_args or cur.args, req.label, cur.hl_state
                )
```

From then on, every frame of that segment carries these labels:

| Column | Value |
|---|---|
| `action.high_level_action` | The PDDL action name, for example `rotate`. |
| `action.high_level_args` | FF's argument list as JSON, for example `["robot", "block_r", "cw", "o1", "o2", "workspace_table"]`. |
| `action.low_level_action` | The `label` of the phase's `PhaseRequest`, for example `rotate_start_traj`. |
| `action.low_level_args` | `{"ll_action_name": <label>}` as JSON. |
| `observation.state.high_level_state` | FF's symbolic state before the action, one fact per line. |

`pddl_goal`, the episode's goal text, is the problem's `:goal` block as written in the PDDL file;
it does not change from skill to skill. Every recorded episode ends with a 25-tick hold, which is
labelled `done` when the episode succeeded. All columns are described in
[Dataset fields](dataset_fields.md).

The skill only chooses the low-level labels. Reuse the existing ones where a phase means the same
thing, for example `init_gripper_open_traj` for an opening gripper or `gripper_close_traj` for a
grasp, because the annotator keys on them.

## Annotation: goal text and target pixel

[Annotation](annotate.md) turns the labels into the `goal.action` text and the `goal.position`
pixel. Both come from `build_action_goal_str` in
`src/toyblocks/data_annotation/annotation_utils.py`, which looks at a frame's high-level action,
low-level action and high-level arguments, and returns the text and the object to project. For
`rotate` and `reorient`, the verb comes from the action and the direction from `args[2]`:

```python
    elif hl_action in _DIRECTION_VOCABULARY and len(hl_args) > 2:
        parsed = _parse_reorient_direction(hl_action, str(hl_args[2]))
        if parsed is not None:
            verb, direction = parsed
            primary_str = f"{verb} {direction}"
            secondary_str = (
                "gripper_open" if ll_action in _GRIPPER_OPEN_LL_ACTIONS else "move_robot"
            )
        else:
            target = None
```

so every frame of a clockwise rotate of `block_r` reads `rotate cw`, with `goal.position` on
`block_r`.

The annotator knows only the actions it has a branch for. **A frame of any other action gets an
empty `goal.action` and a NaN `goal.position`, and annotation still exits 0.** For a new action,
add:

- an entry in `_HL_ARG_COUNTS` with the action's argument count, so that a frame whose arguments
  fell out of sync with its action name is left unlabelled instead of mislabelled;
- a branch in `build_action_goal_str` that names the action and picks the target object. By default
  the target is `args[1]`. Return a second object as `target_obj2` if the action has one, as the
  push actions do for `goal.position_target`;
- if the target pixel should be taken at the end of the action rather than on each frame (as for a
  place on the table), add the text to `SEGMENT_END_TARGET_LABELS`.

The branches match low-level labels by name, so check every label your skill emits against them.

## A problem to exercise it

The quickest test case is one hand-written problem whose goal can only be reached through the new
action. `src/toyblocks/tamp/blocks/problems/rotate.pddl` is the one for `rotate`: one block at `o1`,
goal `(ori_of block_r o2)`. Build its env package with [env generation](env_generation.md):

```bash
python scripts/generate_task_from_pddl.py \
    --domain_path src/toyblocks/tamp/blocks \
    --pddl src/toyblocks/tamp/blocks/problems/rotate.pddl \
    --output_dir outputs \
    --num_reset_states 32 --num_envs 16 \
    --seed 0 --headless
```

For a whole family of problems, write a generator in `scripts/task_gen/` (see
[Task generation](task_generation.md)). `scripts/task_gen/rotate.py` builds a `ProblemSpec` whose
goal is computed from each sampled initial orientation:

```python
        def rotate_goal(init):
            init_ori = init.arg("ori_of", target_block)
            goal_o = yaw(init_ori, random.choice(directions))
            assert goal_o != init_ori, (
                f"yaw produced a no-op: {init_ori} -> {goal_o}"
            )
            return And(Clear(target_block), OriOf(target_block, goal_o))

        spec.goal = rotate_goal
```

Run it with `--preview` to print one problem without writing anything. Through the pipeline, pass
the script with `--task-gen-script`. If your skill implements `push_to_right` or `push_next_to`,
also override `--vecgen-exclude-actions`, whose default excludes both.

Play data does not pick up a new skill automatically: the balanced goal policy
(`src/toyblocks/playdata/skill_balance.py`, the `--balanced-skills` default of the play pipeline)
only proposes pick-and-place, flip and rotate goals.

## Testing it

### Unit tests, on the CPU

The symbolic layer runs without a GPU. `tests/vecgen/test_symbolic.py` plans a checked-in problem
with FF and checks the resulting program, and the same pattern covers a new mapping:

```python
from pathlib import Path

from toyblocks.vecgen.symbolic.ff import run_ff
from toyblocks.vecgen.symbolic.pddl import parse_problem, render_problem
from toyblocks.vecgen.symbolic.programs import program_from_plan

ROOT = Path(__file__).resolve().parents[2]


def test_rotate_problem_maps_to_one_rotate_skill():
    p = parse_problem(ROOT / "src/toyblocks/tamp/blocks/problems/rotate.pddl")
    res = run_ff(ROOT / "src/toyblocks/tamp/blocks/domain.pddl", render_problem(p))
    assert res.ok, res.error
    prog = program_from_plan(res.actions, p.name, 0, res.states)
    assert [str(s) for s in prog.skills] == ["rotate(block_r, cw, o1, o2)"]
```

`tests/vecgen/test_predicates.py` shows how to test a geometric predicate on hand-built poses. Run
the CPU tests with:

```bash
pytest tests/vecgen/test_symbolic.py tests/vecgen/test_predicates.py
```

A skill class itself cannot be imported without cuRobo: every skill module imports
`toyblocks.vecgen.engine`, which imports the cuRobo-backed `Geometry`. Test the skill in a short
run instead.

### A short run

Run the new env package with a handful of environments, then look at the log and the
`results.json`:

```bash
python scripts/vecgen/run.py \
    --env-path outputs/rotate \
    --pddl src/toyblocks/tamp/blocks/problems/rotate.pddl \
    --num-envs 16 --episodes 32 \
    --record outputs/rotate_rec --mark-complete --headless
```

In the log:

- `FF plan (... ms): rotate(block_r, cw, o1, o2)` is printed once at start-up, from the problem's
  `:init`. If FF cannot plan the problem, the run stops right there with `FF failed`.
- `env N: retry K of <skill call> (<reason>)` and `env N: re-planned from live state after <reason>`
  show the retries and re-plans as they happen.
- `env N episode M: SUCCESS` or `FAILURE <reason>` closes each episode, and the
  `RESULT episodes=... success=...` line and `failure reasons: {...}` sum them up.

`results.json` holds the same failure reasons per episode. A high share of `<name> effect check
failed` points at the skill's motion or at a too-strict `verify`. `begin: no feasible parameters`
points at the start-pose search, or at derived facts that let FF plan actions the skill cannot
reach. `no program` means FF planned an action with no skill, or found no plan at all.

To watch the skill, add `--record-videos --enable_cameras`, which records the scene cameras as well
(see [What a recorded episode contains](execution.md#what-a-recorded-episode-contains)).

### Checking the recorded data

Check that every frame of the new action carries the labels you expect:

```python
import glob

import pandas as pd

cols = ["action.high_level_action", "action.low_level_action"]
files = sorted(glob.glob("outputs/rotate_rec/blocks_success/data/chunk-*/file-*.parquet"))
df = pd.concat(pd.read_parquet(f, columns=cols) for f in files)
print(df.groupby(["action.high_level_action", "action.low_level_action"]).size())
```

Each of your phases' labels should appear under your action. Expect a few `init` frames at the
start of each episode, the `done` hold at the end, and an empty label on each episode's terminal
row. For a whole episode at a glance, `scripts/datagen/plot_episode.py` plots every signal of one
episode. Finally, annotate a
copy of a dataset that has camera data (one recorded with `--record-videos`, or rendered) and check
that `goal.action` is non-empty and `goal.position` is not NaN on the new action's frames, as
described under [Annotation's output](annotate.md#output).
