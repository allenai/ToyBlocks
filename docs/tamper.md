# TAMPER — TAMP Expert Relabeling

A DAgger-style loop with a TAMP expert. An evaluation run surfaces the states a
policy actually gets wrong; the expert then plans demonstrations *from those
states* instead of from freshly sampled ones, so the next round of training data
covers the distribution the policy visits rather than the one datagen happens to
sample.

The expert is the vecgen engine (`scripts/vecgen/run.py`, the same one datagen
runs): FF over the blocks domain from the captured scene, cuRobo v2 motion plans,
scripted skills with verified effects. No step needs the TAMP submodule.

Related docs:

- Evaluation: [evaluations.md](source/user_guide/workflows/evaluations.md) and [`evaluate.py` reference](source/user_guide/workflows/evals/evaluate.md)
- Dataset recording: [docs/lerobot_recording.md](lerobot_recording.md)
- Datagen pipeline: the **Datagen Pipeline** section in `CLAUDE.md`

## How it works

Capture happens inside `BaseEnv._before_auto_reset`, the only point where a
finished episode's terminal state is still in the sim. By the time
`infos["final_info"]` reaches the eval loop, the env has auto-reset and the sim
holds the *next* episode. `_success_once` and the queue setup are both still
live there too, so the outcome and the episode's goal are recoverable.

Each captured directory is a multi-problem **vecgen env package**:

- a clone of the eval's env package, with its gym id retargeted to the directory;
- the captured states as its `reset_states.pkl`;
- a `problems.json` index with one entry per state: its goal problem, its episode
  blocks, and reset-state range `[i, i + 1]`. The env's other spawned blocks stay
  parked.

`scripts/vecgen/run.py` then runs on it directly. `--init-from-scene` plans each
episode with FF from the scene read off the captured state (`on` / `on_table` /
`clear` / `ori_of`, as a vecgen re-plan does) toward the recorded goal, instead of
from the problem's `:init`. `--reset-state-ids` runs the selected states once each.

```
eval (policy fails)  ──►  TAMPER dir  ──►  vecgen (plan + execute + record)  ──►  dataset
        ▲                                                                            │
        └─────────────────────────  retrain, evaluate again  ◄───────────────────────┘
```

Where a state's goal comes from:

| Eval setup | Problem written |
|---|---|
| Queue setup with `pddl_problem_file` (file exists) | that file |
| Queue setup with `pddl_goal` (dataset queues, `launch_eval.py`) | rebuilt: `:objects` = the episode's blocks under their domain subtypes, `:goal` = `pddl_goal`, `:init` = the recorded `high_level_state` (vecgen ignores it under `--init-from-scene`) |
| No queue | none; pass `run_tamper_pipeline.py --pddl-problem-file` |

## Prerequisites

You need an eval env package and an episode queue: what `launch_eval.py` builds
from a LeRobot dataset. To build them by hand:

```bash
DS=/path/to/lerobot_dataset            # any datagen output (vecgen or merged)

# eval env package from the dataset -> src/blocksuite/tasks/autogen/tamper_src (gym id tamper_src-v0)
python scripts/generate_task_from_dataset.py --dataset-path $DS --task-name tamper_src --headless

# episode queue (one setup per dataset episode)
python scripts/serve_episode_queue.py --dataset-path $DS --port 8011 &
```

## Step 1 — Evaluate with capture enabled

```bash
python scripts/evaluate.py \
    --task tamper_src-v0 \
    --random_actions \
    --queue_url http://localhost:8011/next \
    --obs_mode rgb --enable_cameras \
    --num_envs 4 --num_eval_steps 600 \
    --env-cfg.episode-length-s 4.0 \
    --tamper_output_dir outputs/tamper/round1 \
    --output_dir runs/tamper_eval_round1 \
    --headless
```

`--random_actions` is the cheapest way to exercise the loop. For a real policy,
point it at a policy server with `--policy_host` / `--policy_port`, or pass `--tamper_output_dir` through
your usual eval. Capture is off unless `--tamper_output_dir` is set, so existing
eval runs are unaffected.

> **Episodes must end.** Autogen env packages ship `episode_length_s = 99999`. An
> episode that never succeeds never terminates, never auto-resets, and is
> therefore never captured. Pass `--env-cfg.episode-length-s`, or use the queue's
> expert-timeout scaling, to give the policy a fair attempt with a finite end.

### Capture flags

| Flag | Default | Description |
|---|---|---|
| `--tamper_output_dir` | `None` | Root for captured states. Unset disables capture entirely. |
| `--tamper_capture_successes` | `False` | Also snapshot successful episodes (tagged `success`), so you can filter at planning time instead. |
| `--tamper_max_states` | `None` | Cap the number of snapshots. Bounds the output of a long run against a policy that fails often. |
| `--tamper_source_env_path` | auto | Env package to clone. Auto-detected from `cfg.reset_states_path`; set it only when detection fails. |

On completion the run prints:

```
[TAMPER]: captured 16 state(s) across 1 problem(s) -> outputs/tamper/round1
```

## Step 2 — Inspect what was captured

```
outputs/tamper/round1/
    tamper_summary.json
    <problem_slug>/                       # PDDL stem, or unknown_problem (dataset queues)
        reset_states.pkl                  # captured states + scene profile ids
        problems.json, problems/*.pddl    # vecgen index: one entry per state
        tamper_failures.json              # instance_id -> episode, outcome, goal
        __init__.py, env.py, env_cfg.py,  # cloned env package (gym id retargeted)
        profiles.json, non_sim_state_randomizations.json
```

States are grouped by source PDDL problem file; a dataset queue records none, so
everything lands in `unknown_problem`, and each state's own goal sits in
`problems.json`. `tamper_failures.json` is the join key back to the eval run:

```json
{
  "pddl_problem_file": null,
  "num_instances": 16,
  "num_scene_profiles": 1,
  "reason_counts": {"timeout": 16},
  "failures": [
    {"instance_id": 0, "env_idx": 0, "reason": "timeout", "episode_len": 100,
     "episode_id": "ep_000015", "pddl_goal": "(and (on_table block_pink workspace_table) ...)",
     "scene_profile_id": 0, "source_problem": "vg_notamp_out",
     "episode_blocks": ["block_yellow", "block_o", "..."],
     "vecgen_problem": "tamper_00000"}
  ]
}
```

Outcomes are tagged `failure` (the env terminated) or `timeout` (it ran out of
episode length, or the end was requested while unsuccessful). `vecgen_problem` is
the state's `problems.json` entry. It is `null` when no problem could be recovered,
and the pipeline skips those states.

### What gets normalized, and why

A terminal state is captured mid-motion. Before writing, the collector zeroes
joint and root velocities and aligns `joint_position_target` with
`joint_position`. Without this, the PD controller would yank the arm toward a
stale target the moment vecgen wrote the state, so the scene executed would not
be the scene planned against.

Object **poses** are left exactly as captured; those are the TAMPER signal.

## Steps 3–5 in one command

`scripts/datagen/run_tamper_pipeline.py` runs vecgen, annotation and merge over a
TAMPER directory. Eval is deliberately *not* part of it: it needs a policy and a
live sim, so it stays step 1.

```bash
python scripts/datagen/run_tamper_pipeline.py \
    --tamper-dir outputs/tamper/round1 --exec-num-envs 32

# Only hard failures, record cameras so annotation runs, no merge
python scripts/datagen/run_tamper_pipeline.py \
    --tamper-dir outputs/tamper/round1 \
    --reasons failure --record-videos --skip-merge
```

The pipeline reads `tamper_summary.json` and each `tamper_failures.json`, selects
instances, and runs `scripts/vecgen/run.py` once per problem dir. Episodes go to
`<data-dir>/problem_<slug>/` (`blocks_success` / `blocks_failure`), in the layout
`merge_datasets.py` collects.

| Flag | Default | Description |
|---|---|---|
| `--tamper-dir` | required | Directory from `--tamper_output_dir` |
| `--reasons` | all | Filter instances by outcome (`failure` / `timeout` / `success`) |
| `--max-instances` | `None` | Cap instances per problem dir |
| `--pddl-problem-file` | recorded | Plan every state toward this one problem; required when eval ran without `--queue_url` |
| `--skip-execution` / `--skip-annotate` / `--skip-merge` | off | Step toggles |
| `--exec-num-envs` | `32` | vecgen envs. Capped at the selected states and rounded up to a multiple of the scene-profile count. |
| `--record-videos` | off | Record the scene cameras. Annotation needs them; without, re-render the episodes, then annotate. |
| `--vecgen-args` | `""` | Extra `run.py` flags, as one quoted string, e.g. `'--max-attempts 2 --pool-slots 16'` |
| `--data-dir` / `--merge-output-dir` | under `--tamper-dir` | Output overrides |

The pipeline continues past a failed problem dir and exits non-zero with a summary
of which steps failed.

## Running the steps individually

### Step 3 — vecgen: plan, execute, record

The goal is unchanged; the failure state is the starting configuration. That is
the recovery demonstration.

```bash
python scripts/vecgen/run.py \
    --env-path outputs/tamper/round1/unknown_problem \
    --domain src/blocksuite/tamp/blocks/domain.pddl \
    --init-from-scene --reset-state-ids 0 1 2 3 --episodes 4 \
    --num-envs 4 --seed 0 \
    --record outputs/tamper/round1/datasets/problem_unknown_problem --mark-complete \
    --headless            # + --record-videos --enable_cameras for camera streams
```

`--reset-state-ids` index `reset_states.pkl`: instance *i* is the *i*-th entry in
that dir's `failures` list. Expect some states to fail. A failure state can be
genuinely unrecoverable for the scripted expert: a block shoved out of reach or
tilted on another, or the arm left pressed into the scene, which cuRobo reports as
`INVALID_START_STATE_WORLD_COLLISION`. Treat the success rate as signal about how
far off-distribution the policy has wandered.

### Step 4 — Annotate

Pixel goals project into the front camera, so the annotator needs recorded
cameras: `--record-videos`, or re-rendered episodes.

```bash
python src/blocksuite/data_annotation/annotate_target_xy.py \
    outputs/tamper/round1/datasets/problem_unknown_problem -j 8
```

### Step 5 — Merge

```bash
python scripts/datagen/merge_datasets.py \
    --input_dirs outputs/tamper/round1/datasets \
    --output_dir datasets/merged/train/tamper_round1 \
    --cameras all --num_workers 8
```

The recovery demos merge with your baseline data. The point of the loop is to
train on the union, not on the recoveries alone.

## Step 6 — Retrain and repeat

Retrain on baseline plus round-1 recoveries, evaluate again with
`--tamper_output_dir outputs/tamper/round2`, and iterate. Each round should show
the failure count dropping and the remaining failures moving to harder states.

## Worked example

Run end to end on one RTX 5090 (Nov 2026, `naman/dev`). The source was a
16-episode vecgen stack-3 dataset with 10 blocks per scene.

| Step | Command | Result |
|---|---|---|
| Eval package | `generate_task_from_dataset.py --task-name tamper_src` | 1 scene profile |
| Eval | `evaluate.py --random_actions --queue_url … --tamper_output_dir …` (4 envs, 4 s episodes) | 16 episodes, **16 states captured** (all `timeout`), 16 goals rebuilt from `pddl_goal` |
| Pipeline | `run_tamper_pipeline.py --exec-num-envs 8` | vecgen **12–13/16 SUCCESS** (two runs) in about 72 s. Every failure is `INVALID_START_STATE_WORLD_COLLISION`: the random policy left the arm in contact. |
| Merge | (part of the pipeline) | 12 episodes, 12,554 frames |
| With cameras | `… --reasons timeout --max-instances 4 --record-videos --exec-num-envs 4` | 4/4 SUCCESS, annotated, merged (5,092 frames) |

The recordings are recoveries: vecgen planned from what the random policy left
behind, re-planning from the live scene when a pick's effect check failed.

## Troubleshooting

**Nothing was captured.** Almost always `episode_length_s = 99999`: failing
episodes never end. Pass `--env-cfg.episode-length-s`. Also confirm the policy is
actually failing: successes are skipped unless `--tamper_capture_successes` is set.

**`no problems.json … and no --pddl-problem-file given`.** Eval ran without a
queue (no goal recorded), or the directory was captured before the vecgen port.
Pass the problem with `--pddl-problem-file`.

**`IndexError` in `flatten_state_dict` at env creation.** Autogen envs return an
empty `goal` observation group, which `--obs_mode state` cannot flatten. Use
`--obs_mode rgb --enable_cameras`.

**`the deferred PD-switch settle (enabled by --dataset_path) … needs control_mode
'arm_joint_gripper_binary'`.** `--dataset_path` with `--random_actions` (relative
control). Drop `--dataset_path` for random-action smoke tests.

**vecgen cannot load the directory.** The source env package was not found, so
nothing was cloned. The collector warns about this at capture time; pass
`--tamper_source_env_path`. `task_id` in `tamper_summary.json` is `null` when the
source package had no `__init__.py` to clone.

**`profiles.json … codegen version None`.** The env package predates the current
codegen. Regenerate it; it cannot be evaluated or replayed as-is.

**Merged dataset is empty or holds failures.** `merge_datasets.py` collects only
`problem_*/blocks_success`. Record under `<data-dir>/problem_<slug>` (the pipeline
does).
