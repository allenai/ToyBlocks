# Replaying Actions from a Dataset

Replay plays a LeRobot dataset's recorded actions back through the **same
queue-driven eval loop a policy uses** (`blocksuite.eval.run_eval._run_eval`).
There is no separate replay runner any more: `--replay` is just another action
source next to a policy server, random or zero actions.

What replay measures is the *open-loop ceiling* of the sim: replaying the
expert's own actions from the recorded initial state reports both the usual
eval metrics (`success_once`, `episode_len`, ...) and per-episode divergence
from the recording (`replay_*` metrics).

## Quickstart

```bash
# One eval GPU, episodes 0-39 of a merged LeRobot root (all of one cube scale)
python scripts/launch_eval.py --replay \
    --dataset-path data/datasets/my_flip_dataset \
    --episode-indices "0-39" \
    --save-diagnostic-plots --deterministic-cuda
```

`--num-envs` stays at 1 and the LeRobot recording is on by default. To replay on several
GPUs, add `--num-eval-gpus N`; the launcher currently also needs `--policy maniflow`
alongside it, because its one-GPU check for the default `molmo2_maniflow` policy runs
under `--replay` too.

`--replay` selects `POLICY_NAME=replay` in
[`scripts/eval/run_eval_pipeline.sh`](https://github.com/allenai/blocksuite/blob/main/scripts/eval/run_eval_pipeline.sh):
the policy server is skipped, the autogen task is generated from the dataset,
the episode queue is started, and each eval worker runs

```bash
python scripts/evaluate.py --task <slug>-v0 --replay \
    --queue_url http://localhost:8001/next --dataset_path <root> \
    --control_mode arm_joint_gripper_binary --obs_mode rgb+state --num_envs 1 --headless
```

Output lands under `runs/<task>-v0/replay/<out_subdir>/...` like any other eval (see
[`launch_eval.py` outputs](launch_eval.md#outputs))
(`eval_episodes.json`, `eval_results.json`, videos, the optional LeRobot
recording), so the offline diagnostics
([`scripts/plot_diagnostics_from_eval_run.py`](https://github.com/allenai/blocksuite/blob/main/scripts/plot_diagnostics_from_eval_run.py))
work unchanged.

Add `--dry-run` to print the pipeline invocation instead of running it.

## How it works

| Piece | Where | Role |
|---|---|---|
| Initial state | `apply_setup` (`utils/episode_queue.py`) via the queue setup | Robot joints, gripper, objects, table height, friction, `pddl_goal`, cube-scale slot. Under `--replay` the recorded frame-0 arm joint **velocity** is restored too (`setup["robot_joint_velocities"]`). |
| Actions | `ReplayPolicy` (`eval/replay.py`) | Emits a 9-dim action per step: 7 recorded joint targets, binary gripper command, and the recorded `action.pd_mode` as the 9th channel. |
| Gains | `PDModeController` (`eval/pd_mode_controller.py`) | The pd_mode channel drives the same controller a policy's prediction would. Whenever a ground-truth `--dataset_path` is given, a mode switch settles for the TAMP tick counts `DATASET_SWITCH_SETTLE_TICKS = (5, 10)` (compliant/stiff) with the arm frozen and the finger target active. The settle runs as per-env hold steps (`DeferredSettle`) that the policy, the recording and the episode clock never see. Initial writes at reset never settle. |
| Skipped ticks | `ReplayPolicy.pop_gap_holds` (`eval/replay.py`) | Where the recording skipped control ticks (detected by `_detect_unrecorded_ticks` from the recorded joint velocities and timestamps), the previous row's action is held for that many ticks before the next row runs, so the sim stays in step with the recording. Uses the same hold steps as the settle. |
| Termination | `ReplayPolicy.done_indices()` → `base_env.request_episode_end` | The episode ends on the step that applies the last recorded action; `episode_end` is forced to `horizon` and `episode_length_s` is raised to cover the longest recording. |
| Divergence | `ReplayPolicy.record_divergence()` / `episode_metrics()` | Joint error vs the recorded row, per-object position / quaternion error, pre/post PD-switch phases. Scalars only, folded into the episode metrics under `replay_*`. |

## `replay_*` metrics

Per episode (in `eval_episodes.json`; aggregated in `eval_results.json` and the
wandb episodes table):

- `replay_num_frames`, `replay_profile_switch_frame` (NaN when no switch),
  `replay_unrecorded_ticks` (control ticks the recording skipped; see "Skipped ticks"
  above)
- `replay_joint_pos_error_{mean,max,final}`,
  `replay_{pre,post}_switch_joint_pos_error_{mean,max,final}`
- `replay_object_<name>_{pos,quat}_error_{mean,max,final}`,
  `replay_{pre,post}_switch_object_<name>_{pos,quat}_error_{mean,max,final}`
- With `--replay_clip_stats` only: `replay_clipped_frames`, `replay_clip_max_rad`
  (see [Clipping replayed targets](#clipping-replayed-targets)).

Any metric without data is `NaN`; the aggregation is NaN-safe.

## Clipping replayed targets

`--replay_clip_stats <stats.json>` (`--replay-clip-stats` on `launch_eval.py`)
clamps every replayed arm joint target to the `action.joint_target` q01/q99
bands of that `stats.json`, for example a ManiFlow run's. It answers whether a
policy whose labels were clipped to those bands could still succeed, and adds
the per-episode metrics `replay_clipped_frames` and `replay_clip_max_rad`. It
only applies to `--replay` and raises otherwise.

## Requirements and failure modes

- **`--num_envs 1`.** `ReplayPolicy` advances every env's recording on each
  step, so it cannot hold one env through a PD-switch settle while the others
  move on. Replay therefore runs one env per worker (use `--num-eval-gpus` for
  parallelism). If the autogen task bakes several cube-scale slots,
  `reconcile_num_envs_with_task` raises `num_envs` and `evaluate()` errors
  out — restrict `--episode-indices` to episodes of a single cube scale, or
  use a task whose package bakes one slot.
- **`--control_mode arm_joint_gripper_binary`** — datagen records absolute
  joint targets, and the deferred settle holds the arm with absolute joint
  targets, so any run with `--dataset_path` (replay or policy eval) raises
  under another control mode. The pipeline sets this; pass it yourself when
  calling `evaluate.py` directly.
- **The dataset must record `action.pd_mode`** and the gain columns
  (`observation.state.current_stiffness` / `current_damping` /
  `current_effort_limit`). Missing gains fail at startup
  (`load_pd_gains_from_dataset`); a missing or all-NaN `action.pd_mode` fails
  when the episode is loaded (`ReplayPolicy.on_reset`).
- `replay_gripper_close_target` defaults to `0.72` (TAMP datagen's Robotiq close
  target); override with `--env-cfg.replay-gripper-close-target`.
- `apply_setup` is strict: a setup without a table pose / height delta, or
  referencing an object the scene does not spawn, raises instead of warning.

## Behavior notes

- Gain *values* are the dataset-wide stiff/compliant vectors from
  `load_pd_gains_from_dataset` (first parquet chunk, one representative vector
  per mode), not the exact per-frame recorded ones; joint velocity limits are
  not written. Identical for datasets recorded under two fixed profiles. When
  the dataset has no frames of one mode, that mode falls back to the blocksuite
  defaults.
- The 5/10-tick settle is a fixed convention (settle steps are not recorded in
  the source dataset). It applies to policy evals with a `--dataset_path` as
  well, so a pd_mode switch in a policy eval also settles.
- The LeRobot recorder (the same one datagen uses) pairs each row's observation
  with the action applied next, so a replay recording has the same row layout
  as the dataset it replays.

## Historical note

The former standalone path (`--replay_lerobot_dataset`, `--replay_dataset_path`,
`--max_replay_episodes`, `replay_results.json`, `replay_plots/`, comparison
videos) has been removed.
[`scripts/extract_lerobot_episode.py`](https://github.com/allenai/blocksuite/blob/main/scripts/extract_lerobot_episode.py)
and the side-by-side video helpers in `utils/tamp_dataset.py` remain for offline
inspection but are no longer used by eval.
