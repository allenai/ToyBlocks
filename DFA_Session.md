# DFA Play-Data — Session Handoff

Everything a next session needs to understand and continue this work. Written 2026-08-16.

Branch: **`naman/dev-playdata`** (HEAD = merge commit `f6642601`).

---

## TL;DR

We built a **DFA-driven play-data system**: a declarative finite-state machine decides, per
environment, what goal to pursue next (e.g. "flip blocks until letter R is visible"), replacing random
goal sampling. The planner/executor/recorder are unchanged. Along the way we found and fixed a **major
RL-executor bug** (batched observations built from the wrong block) that also affects the classic
pipeline — that fix is **PR #206**.

The DFA layer lives in `src/blocksuite/playdata/dfa/`. Author strategies in YAML; run them via
`scripts/generate_play_data.py --dfa-spec <yaml>`.

Full authoring guide: **`src/blocksuite/playdata/dfa/AUTHORING.md`**. Grammar reference: the module
docstring in `dfa/engine.py`. Live vocabulary: `python -c "from blocksuite.playdata.dfa import catalog"`.

---

## 1. The workflow

Classic 3-stage pipeline: stage-1 (env + reset states) → stage-2 (TAMP plan, offline) → stage-3
(execute, reset per plan). **DFA play data** keeps stage-1 but fuses stage-2+3 into one running loop:

```
per env, repeat (never reset until DFA accept/reject):
  read symbolic state  (playdata/symbolic_state.py::read_symbolic_state)
  -> DFA.next_goal(state)      (dfa/engine.py::DFAGoalPolicy)   # GOAL | RETRY | TERMINATE
  -> write PDDL problem        (playdata/problem_writer.py)     # objects fixed, init=live snapshot, goal=DFA move
  -> submit to planner pool    (playdata/planner_pool.py)       # stage-2 TAMP solve, async CuRobo
  -> install & execute plan    (playdata/orchestrator.py)       # stage-3, records one episode
  on DFA TERMINATE (ACCEPT/REJECT): reset env to a new reset-state, reset DFA, restart
```

One DFA goal → one PDDL problem → one TAMP solve → one recorded episode. Each of the N parallel envs has
its **own seeded RNG**, so they run independent searches of the same strategy.

Two workflow docs were published as artifacts this session (also reproducible):
- "DFA Play-Data Workflow" (the loop, the intermediate PDDL, when TAMP runs).
- "Authoring DFA Strategies" == `dfa/AUTHORING.md`.

---

## 2. File map (all new/changed this session)

**New DFA subpackage** `src/blocksuite/playdata/dfa/`:
- `engine.py` — typed `DFASpec`/`Transition`/`GuardSpec`/`MoveSpec`/`EffectSpec` + `DFAGoalPolicy`. Its
  module docstring is the **spec grammar**.
- `registries.py` — the vocabulary: `GUARDS`/`EFFECTS`/`MOVES` dicts + `catalog()`. All guard/effect/move
  functions live here.
- `context.py` — `DFACtx` (what a guard sees: state, memory, rng, holds, glyph_of, pose_of, table) +
  `glyph_of` (block name → USD glyph letter).
- `ori_flip.py` — `ori_after_flip` / `ori_after_rotate` (quaternion compose + snap to o1..o24).
- `spec_loader.py` — YAML/JSON → validated `DFASpec` (`$param` substitution, loud validation).
- `library/*.yaml` — the strategies (see §4).
- `AUTHORING.md` — the how-to guide.

**New in playdata** (mostly pre-existing from earlier commit `20809807`, feature committed `2427d022`):
- `goal_policy.py` — `GoalPolicy` protocol, `RandomGoalPolicy` adapter.
- `orchestrator.py` — `PlayDataOrchestrator` (edited to use `GoalPolicy` + reset-on-accept).
- `symbolic_state.py`, `problem_writer.py`, `planner_pool.py`, `scene_config.py`.

**Scripts**:
- `scripts/generate_play_data.py` — the run entry point (edited: `--dfa-spec`, `--dfa-param`).
- `scripts/task_gen/search.py` — search scenes (1 target letter + distractors).
- `scripts/task_gen/find_as.py` — 10-block scenes with 4 A blocks (5 variants).
- `scripts/render_orientations.py` — renders a block in all 24 orientations from the front camera
  (used to define `FOUND_ORIS`).

**RL-executor fixes** (the big bug — see §5): `src/blocksuite/tamp/blocks/executors/`
`ReOrientTrajectory.py`, `RotateTrajectory.py`, `PushTrajectory.py`, `NudgeTrajectory.py`,
`IsolateTrajectory.py`.

**Domain**: `src/blocksuite/tamp/blocks/domain.pddl` — `isolate` action commented out (see §5).

---

## 3. How to run (three steps)

```bash
# 1. Scenes -> PDDL
python scripts/task_gen/find_as.py --count 5 --out-dir outputs/find_as/tasks --seed 42

# 2. Env package + 16 reset states (Isaac; needs OMNI_KIT_ACCEPT_EULA=yes)
python scripts/generate_task_from_pddl.py --domain_path src/blocksuite/tamp/blocks \
  --pddl outputs/find_as/tasks/<problem>.pddl \
  --num_reset_states 16 --num_envs 16 --num_scene_profiles 8 \
  --output_dir outputs/find_as --headless

# 3. Run a DFA strategy (stage 2+3 interleaved)
python scripts/generate_play_data.py --domain_path src/blocksuite/tamp/blocks \
  --pddl_problem_file outputs/find_as/tasks/<problem>.pddl \
  --env_path outputs/find_as/<slug> \
  --dfa-spec src/blocksuite/playdata/dfa/library/find_as_structured.yaml \
  --dfa-param letter=A --dfa-param target_count=3 --dfa-param max_picks=40 \
  --num-envs 8 --num-planners 4 --max-plan-time 50 --max-plans-total 600 \
  --data_dir datasets/<out> --status-log-seconds 60 \
  --record-data --headless --enable_cameras --generate-data --robotiq
```

Camera videos are recorded into the LeRobot dataset exactly as in the legacy pipeline (one video
per camera per episode, in the `<data_dir>__sNNN/` shard). `--no-video` records Parquet only and
spawns no cameras (much faster); images can be produced afterwards with the rerender stage.
`--cameras front_camera` spawns and records only the named cameras.
**Let runs exit naturally** — killing mid-finalize truncates the mp4 `moov` atom (unplayable). The
bundled ffmpeg for repair/probe: `.venv/lib/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-*`.

---

## 4. The strategies (`dfa/library/`)

| spec | what it does | notes |
|---|---|---|
| `flip_search.yaml` | pick random block → flip forward × N → check R visible → repeat | the simple example |
| `reveal_search.yaml` | pick block → `set_ori(o2)` → check | ⚠ **leaks the answer** (targets o2 directly) — demo only, NOT training data |
| `find_as.yaml` | random-walk `flip_or_rotate`, accumulate 3 distinct A blocks | honest but slow |
| `find_as_structured.yaml` | pick unresolved block → look → flip **fwd,fwd,left,left,left** checking after **each** flip → record on find, else `move_to_side`; never re-pick found/sided; accept at 3 | the current best; efficient |

**Ground-truth rule (distillation-safe):** moves read the block's *current* orientation only to express
one flip as a single-action PDDL goal; found-guards use ground truth only as a *camera stand-in for
termination*, never to steer the action. Do **not** use `set_ori(c, o2)`-style targeted moves for
training data. Keep block picks blind (no `glyph:` filter).

**"Found" orientations** = `registries.FOUND_ORIS = (o2, o6, o10, o14, o17, o18, o19, o20)` — the 8
poses where a glyph face is camera-visible, verified via `scripts/render_orientations.py`.
`upright_letter` guard = strictly o2.

---

## 5. Critical bug fixes

### 5a. Wrong-block observations in batched RL executors (PR #206)
**Symptom:** flips landed on the wrong face (tilt-90°); ~40–60% reorient effect-failures in play data.
**Cause:** the batched RL coordinator shares **one** pseudo `rl_env`; each executor set
`rl_env.block = scene[env_data[env_ids[0]]["obj"]]` (only the FIRST env's block), and
`_compute_intermediate_values` built *every* block observation (`object_pos/quat`, corners,
`axis_angle_error`, `reach_target`) from it. In play data each env flips a *different* block, so all but
the first env got the wrong block's pose. Invisible in the classic pipeline's single-problem case
(all envs flip the same block), but it **does** hit classic multi-reorient/multi-problem batches.
**Fix:** in each executor's `_compute_goals` (runs per-env in `step()`, before obs), repoint:
```python
rl_env.block = base_env.scene[env_data["obj"]]
bbox = obs_utils.get_bounding_box(rl_env.block); rl_env.block_sizes[env_idx] = bbox[env_idx,1]-bbox[env_idx,0]
```
Applied to ReOrient/Rotate/Push/Nudge/Isolate. **Validated:** `flip_search` effect-failures 60 → 0,
flip success ~65% → 100%. **PR #206** (`fix/rl-executor-per-env-block` → `naman/dev`) contains ONLY
these 5 files, created via a git worktree so the main tree was undisturbed.

### 5b. `isolate` action crashes cluttered scenes
`isolate` binds every other block as a distractor, but its RL policy is sized for `num_distractors=5`,
so in a 10-block scene the planner picks `isolate` and it crashes (`RuntimeError: expanded size (9) must
match existing size (5) ... [8,9,24] vs [8,5,1]`). **Fix:** commented out the `isolate` action in
`domain.pddl` (the planner reads it live — no stage-1 regen needed; it falls back to analytic
move_to_side/CuRobo). This reverts a merge choice (we had taken main's isolate-enabled `domain.pddl`).

### 5c. (context, from main via merge) train/inference observation mismatch
Pre-merge `base_rl_trajectory` bypassed the actor obs-key allowlist and zero-padded the asymmetric-critic
privileged tail → the actor was fed the wrong observation. **Main already fixed this** (allowlist +
tail slicing + `_validate_obs_widths`) and ships a retrained `flip` checkpoint. We got these by merging
`origin/main`. Not our fix, but important context; if `naman/dev` lacks `main`, merge it.

---

## 6. Investigations / findings

- **Observation frames** (train vs inference) are consistent by construction — same functions over the
  shared scene; `object_pos`/`tcp_pos` env-local, relative vectors invariant, `block_corners_in_goal_frame`
  goal-frame, `flip_push_dir_world` canonical world-frame. `block_sizes` is body-frame cached (not a
  world AABB). The only remaining frame nuance is the goal-yaw composition in `ReOrientPoseGenerator`
  (flagged "still testing" in-code) — a minor contributor to yaw-off misses, secondary to §5a.
- **`block_r` is the flip policy's training asset** (`create_toy_block` defaults to `R.usd`), so the
  earlier "USD vs cuboid OOD" hypothesis was wrong — the failures were the §5a wrong-block bug.
- **24-orientation render** (`scripts/render_orientations.py`, output `outputs/ori_views/`) shows a
  clear R at o2/o17/o19 etc. → defined `FOUND_ORIS` (8 poses).

---

## 7. Git & PR state

- Branch `naman/dev-playdata`, HEAD `f6642601` (merge of `origin/main`).
- Committed: DFA feature (`2427d022`) + the main merge (`f6642601`).
- **Uncommitted in the working tree** (as of this handoff): all the post-merge work —
  `dfa/registries.py` (many guard/effect/move additions), `dfa/library/{flip_search,reveal_search,
  find_as,find_as_structured}.yaml`, `dfa/AUTHORING.md`, `scripts/task_gen/find_as.py`,
  `scripts/render_orientations.py`, the **5 trajectory executors**, and `domain.pddl` (isolate off).
  These are NOT committed on the branch yet — commit them before relying on them long-term.
- **PR #206** (`fix/rl-executor-per-env-block` → `naman/dev`) has the 5 trajectory fixes only.
- `backup/pre-main-merge` branch exists as a pre-merge safety net.
- `naman/dev` is 65 commits ahead of / 35 behind our HEAD (diverged); its trajectory files matched ours
  (clean patch base for #206).

---

## 8. Running experiments (tmux)

Long runs are launched under tmux so they survive disconnection: `tmux attach -t <name>`.
- `findas_struct` — **the current run**: `find_as_structured.yaml`, 8 envs, 600-plan budget, videos on,
  env `outputs/find_as/problem_declarative_10_20260816_132707_061728_1824`, output
  `datasets/find_as_structured_deep/<slug>/` (log `deep.log`). Was healthy at ~29 plans, `move_to_side`
  executing, 0 crashes.
- Earlier finished runs (videos on disk): `datasets/find_as_deep/<slug>/videos` (the random-walk
  `find_as`, 801 plans, 2 accepts), `datasets/play_accepts_reveal/videos` (reveal_search, 23 accepts).

**gotcha:** the file-transfer tool (`SendUserFile`) is disabled this session — deliver videos by
embedding compressed clips (bundled ffmpeg → ~2 MB each) as data URIs in an HTML Artifact, or point to
the disk paths. Artifact publish had a transient 502 outage mid-session (recovered).

---

## 9. Gotchas / rules

- **Isaac needs** `export OMNI_KIT_ACCEPT_EULA=yes`. `uv run` uses `.venv` (has isaac + tamp extras).
- `num-envs` must divide the env's `num_scene_profiles` (8) or be < it.
- **Cluttered scenes (>5 blocks): keep `isolate` off** in `domain.pddl` (§5b).
- **`move_to_side`** emits `(on_table_side B table)` — plannable (analytic action, no RL distractor
  issue), but `on_table_side` has no goal-checker, so the episode's goal-reached flag reads False
  (episode still kept; the DFA marks "sided" in memory, not via the goal check).
- Every strategy needs a `REJECT`/budget exit or it loops forever.
- Validate a spec offline: `python -c "from blocksuite.playdata.dfa import load_dfa_spec; load_dfa_spec('<yaml>')"`.
- pkill-then-launch in one shell line can abort the launch (nonzero pkill) — separate the commands.

---

## 10. Open items / suggested next steps

- **Commit the uncommitted work** on `naman/dev-playdata` (or split into PRs: the DFA registry/spec
  additions, `find_as` scenes, `domain.pddl` isolate-off, `AUTHORING.md`).
- **Finish/collect the `findas_struct` run** — grab accept-env videos, report the found/sided behavior.
- Optionally run the other 4 `find_as` packages deep with the structured strategy.
- If you want higher per-plan flip success in dense scenes: fewer blocks, bias `flip_or_rotate` toward
  flips, or looser tolerance — but the current behavior is the honest search signal.
- The `ReOrientPoseGenerator` goal-yaw composition (§6) is worth a proper fix if yaw-off misses matter.
