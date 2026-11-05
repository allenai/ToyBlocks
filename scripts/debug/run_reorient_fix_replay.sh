#!/usr/bin/env bash
# GT-action replay of the reorient_fix datasets through scripts/evaluate.py
# (open-loop: recorded action.joint_target + gripper, recorded gains/friction/
# scale/initial state). Two repeats per dataset because fixed-seed replay
# swings ~+-8% from PhysX nondeterminism; baseline datasets for comparison.
set -uo pipefail
cd /home/naman/blocksuite
status=.logs/reorient_fix_replay.status
log=.logs/reorient_fix_replay.log
note() { echo "[$(date '+%F %T')] $*" | tee -a "$status"; }
replay() {  # $1 task slug  $2 dataset root  $3 run name  $4 n_episodes
  idx=$(seq 0 $(( $4 - 1 )) | tr '\n' ' ')
  note "replay $3 ($4 episodes)"
  timeout -s KILL 2400 uv run --no-sync python scripts/evaluate.py --task ${1}-v0 \
      --replay_lerobot_dataset "$2" \
      --replay_lerobot_episode_indices $idx \
      --control_mode arm_joint_gripper_binary --num_envs 1 \
      --output_dir runs/reorient_fix_replay/$3 --headless >>"$log" 2>&1
  rc=$?
  gs=$(python3 -c "import json,sys; d=json.load(open('runs/reorient_fix_replay/$3/replay_results.json')); f=[m.get('goal_success') for m in (d if isinstance(d,list) else d.get('episodes', d.get('per_episode', []))) if isinstance(m,dict) and 'goal_success' in m]; print(f'goal success {sum(f)}/{len(f)}')" 2>/dev/null || echo "(no replay_results.json)")
  note "replay $3 done (rc=$rc) $gs"
}
for k in 1 2; do
  replay reorient_fix_right datasets/reorient_fix/reorient_fix_right_fixed_v5/blocks_success right_v5_run$k 27
  replay reorient_fix_left  datasets/reorient_fix/reorient_fix_left_fixed_v5/blocks_success  left_v5_run$k 25
done
replay reorient_fix_right datasets/reorient_fix/reorient_fix_right_baseline/blocks_success right_baseline_run1 17
replay reorient_fix_left  datasets/reorient_fix/reorient_fix_left_baseline/blocks_success  left_baseline_run1 8
note "DONE-replay"
