#!/usr/bin/env bash
# Foreground queue + evaluate.py for the split-terminal local eval workflow.
#
# Runs task codegen, the episode queue, and evaluate.py in the FOREGROUND so a
# `breakpoint()`/`pdb` in the eval client / obs transform is hit interactively.
# Pair with scripts/eval/serve_local_server.sh (policy server) in another
# terminal; this script does NOT start or tear down the server.
#
# PER_EPISODE selects the eval strategy:
#   0 (default for --hl oracle): one codegen + queue + batched evaluate.py pass
#       sized to the queue (num_envs up to MAX_ENVS_PER_GPU).
#   1 (default for --hl model):  one codegen + queue + num_envs=1 evaluate.py
#       pass PER episode. The model HL server only supports single-agent
#       inference (B==1), and each episode's recorded cube_scale needs its own
#       spawn slot, so multi-scale episode sets can't share one process.
#
# Usage: activate the venv, then `bash scripts/eval/serve_local_client.sh`.
# Override config inline, e.g. `PER_EPISODE=1 EPISODE_INDICES=0,3 INIT_AFTER_HL_ACTION=flip,rotate bash ...`.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/eval/_common.sh
source "${SCRIPT_DIR}/_common.sh"

# DATASET_PATH has no repo default (per-user weka path), and without it the queue
# is silently skipped and codegen runs against an empty task slug.
if [ -z "$PDDL_PROBLEM_FILE" ] && [ -z "$DATASET_PATH" ]; then
  echo "DATASET_PATH (or PDDL_PROBLEM_FILE) is required; set it in scripts/eval/_common.sh or export it" >&2
  exit 1
fi

# Per-episode loop defaults on for the single-agent model HL server.
if [ -z "${PER_EPISODE:-}" ]; then
  if [ "$HL" = "model" ]; then PER_EPISODE=1; else PER_EPISODE=0; fi
fi

SCRIPT_START=$(date +%s)

# Kill any stale queue / eval clients (the server lives in its own terminal).
# Scoped to this user: `pkill -f` matches whole command lines, so on a shared node
# an unscoped "evaluate.py" pattern kills other people's runs.
_ME="$(id -u)"
pkill -9 -u "$_ME" -f "serve_episode_queue.py" || true
pkill -9 -u "$_ME" -f "evaluate.py" || true
for _ in $(seq 1 10); do
  if ! pgrep -u "$_ME" -f "serve_episode_queue.py" >/dev/null \
     && ! pgrep -u "$_ME" -f "evaluate.py" >/dev/null; then
    break
  fi
  sleep 1
done

# The policy server runs in the other terminal, so make the dependency explicit:
# without this the first evaluate.py connect races the server's model load and
# fails with a connection error that looks like a config problem.
echo "===> waiting for the policy server on :${POLICY_PORT} (start serve_local_server.sh if it isn't running)"
for _ in $(seq 1 360); do
  if port_is_listening "$POLICY_PORT"; then break; fi
  sleep 5
done
if ! port_is_listening "$POLICY_PORT"; then
  echo "no policy server listening on :${POLICY_PORT} after 30m; start scripts/eval/serve_local_server.sh first" >&2
  exit 1
fi

# Generate the autogen task package for one episode subset.
generate_task() {
  local episode_indices="$1"
  if [ -n "$PDDL_PROBLEM_FILE" ]; then
    local pddl_env_path="${ENV_PATH:-src/toyblocks/tasks/autogen}"
    echo "===> generating task from PDDL: ${PDDL_PROBLEM_FILE}"
    python scripts/generate_task_from_pddl.py \
      --pddl "$PDDL_PROBLEM_FILE" \
      --domain_path src/toyblocks/tamp/blocks \
      --output_dir "$pddl_env_path" \
      --num_reset_states 8 \
      --headless || exit 1
    local pddl_stem
    pddl_stem=$(basename "$PDDL_PROBLEM_FILE" .pddl)
    # TASK_SLUG has no dataset to derive from in PDDL-direct mode; it names the
    # per-episode output dir below.
    TASK_SLUG="$(echo "$pddl_stem" | tr '-' '_')"
    TASK="${TASK_SLUG}-v0"
  else
    echo "===> generating autogen task from ${DATASET_PATH} (episodes ${episode_indices:-all})"
    local episode_arg=()
    if [ -n "$episode_indices" ]; then episode_arg=(--episode-indices "$episode_indices"); fi
    local usd_library_arg=()
    if [ -n "$USD_LIBRARY" ]; then usd_library_arg=(--usd-library "$USD_LIBRARY"); fi
    python scripts/generate_task_from_dataset.py \
      --dataset-path "$DATASET_PATH" \
      --domain-path src/toyblocks/tamp/blocks \
      --output-dir src/toyblocks/tasks/autogen \
      --task-name "$TASK_SLUG" \
      "${episode_arg[@]}" \
      "${usd_library_arg[@]}" \
      --headless || exit 1
  fi
}

# Start the episode queue for one episode subset; sets QUEUE_PID + QUEUE_URL_ARG.
start_queue() {
  local episode_indices="$1"
  QUEUE_PID=""
  QUEUE_URL_ARG=""
  if [ -z "$PDDL_PROBLEM_FILE" ] && [ -n "$DATASET_PATH" ]; then
    echo "===> starting queue server on :${QUEUE_PORT} (episodes ${episode_indices:-all})"
    local episode_arg=()
    if [ -n "$episode_indices" ]; then episode_arg=(--episode-indices "$episode_indices"); fi
    local cut_arg=()
    if [ -n "${INIT_AFTER_HL_ACTION:-}" ]; then
      cut_arg=(--init-after-hl-action "$INIT_AFTER_HL_ACTION")
    fi
    python scripts/serve_episode_queue.py \
      --dataset-path "$DATASET_PATH" \
      "${episode_arg[@]}" \
      "${cut_arg[@]}" \
      $QUEUE_LIMIT_ARG \
      --log-level "$QUEUE_LOG_LEVEL" \
      --port "$QUEUE_PORT" &
    QUEUE_PID=$!
    wait_policy_port "$QUEUE_PORT" "$QUEUE_PID" "queue"
    QUEUE_URL_ARG="--queue_url http://localhost:${QUEUE_PORT}/next"
  fi
}

stop_queue() {
  if [ -n "${QUEUE_PID:-}" ]; then
    kill "$QUEUE_PID" 2>/dev/null || true
    wait "$QUEUE_PID" 2>/dev/null || true
  fi
}

# Run evaluate.py against the running policy server.
run_eval() {
  local num_envs="$1" out="$2"
  mkdir -p "$out"
  export KIT_CACHE_ROOT="/tmp/kit-cache-0"
  export KIT_LOG_DIR="/tmp/kit-logs-0"
  export OMNI_USER_CONFIG_DIR="/tmp/omni-config-0"

  local pddl_arg=()
  if [ -n "$PDDL_PROBLEM_FILE" ]; then pddl_arg=(--pddl_problem_file "$PDDL_PROBLEM_FILE"); fi

  # Oracle HL only when HL=oracle (maniflow); model HL (molmo2_maniflow) omits it,
  # matching run_eval_pipeline.sh's HL_FLAG gating.
  local oracle_arg=()
  if [ "$HL" = "oracle" ]; then oracle_arg=(--use_oracle); fi

  echo "===> launching evaluate.py (num_envs=${num_envs}, out=${out})"
  # --dataset_path enables reconcile_env_cfg_with_dataset + camera-intrinsics
  # ingestion so scene/robot/calibration follow the dataset provenance. Robot/
  # scene are only passed when explicitly set, so empty defers to the dataset
  # instead of registering as an override mismatch.
  CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --task "$TASK" \
    --seed 0 \
    ${QUEUE_URL_ARG} \
    "${pddl_arg[@]}" \
    ${DATASET_PATH:+--dataset_path "$DATASET_PATH"} \
    --policy_host localhost \
    --policy_port "$POLICY_PORT" \
    --output_dir "$out" \
    ${ROBOT_ID:+--env-cfg.robot-id "$ROBOT_ID"} \
    ${SCENE_PRESET:+--env-cfg.scene-preset-id "$SCENE_PRESET"} \
    --env-cfg.episode-length-s "$EPISODE_LENGTH_S" \
    --env-cfg.num-renders-on-reset 4 \
    --control_mode arm_joint_gripper_binary \
    --obs_mode rgb+state \
    --obs_transform "$OBS_TRANSFORM" \
    ${CENTER_CROP_RATIO:+--center_crop_ratio "$CENTER_CROP_RATIO"} \
    --action_horizon "$ACTION_HORIZON" \
    ${SYNC_ACTION_CHUNKS:+--sync_action_chunks} \
    --num_envs "$num_envs" \
    --track_metrics \
    --save_sensor_obs_video \
    --save_video_all_envs \
    $FIRST_BATCH_ARG \
    --device cuda:0 \
    "${oracle_arg[@]}"
}

ec=0

if [ "$PER_EPISODE" = "1" ]; then
  # One codegen + queue + num_envs=1 eval pass per episode.
  if [ -n "$PDDL_PROBLEM_FILE" ]; then
    EPISODE_LIST=("")
  else
    IFS=',' read -ra EPISODE_LIST <<< "$EPISODE_INDICES"
  fi
  for EP in "${EPISODE_LIST[@]}"; do
    EP_LABEL="${EP:-pddl}"
    echo "============================================================"
    echo "===> EPISODE ${EP_LABEL}"
    echo "============================================================"
    generate_task "$EP"
    start_queue "$EP"
    run_eval 1 "${OUT_PATH}/verify/${TASK_SLUG}/ep${EP_LABEL}" || ec=1
    stop_queue
  done
  echo "===> all episodes done (ec=${ec})"
else
  # Single batched pass: codegen once, size num_envs to the queue.
  generate_task "$EPISODE_INDICES"
  start_queue "$EPISODE_INDICES"

  QUEUE_SIZE=0
  if [ -n "${QUEUE_URL_ARG}" ]; then
    QUEUE_SIZE=$(curl -sS "http://localhost:${QUEUE_PORT}/size" \
      | python3 -c "import sys,json; print(json.load(sys.stdin)['total'])")
  fi
  if [ "$QUEUE_SIZE" -gt 0 ] 2>/dev/null; then
    NUM_ENVS=$(python3 -c \
      "import math; print(min(math.ceil(${QUEUE_SIZE} / ${NUM_EVAL_GPUS}), ${MAX_ENVS_PER_GPU}))")
  else
    NUM_ENVS=${MAX_ENVS_PER_GPU}
  fi
  echo "===> num_envs per GPU: ${NUM_ENVS} (queue=${QUEUE_SIZE}, gpus=${NUM_EVAL_GPUS}, max=${MAX_ENVS_PER_GPU})"

  OUT="${OUT_PATH}/${TASK}/${POLICY_NAME}/${OUT_SUBDIR}/${PROBLEM_ID}/seed_0/local"
  run_eval "$NUM_ENVS" "$OUT" || ec=1

  echo "===> eval done (ec=${ec})"
  if [ -n "${QUEUE_URL_ARG}" ]; then
    echo "queue final status:"
    curl -sS "http://localhost:${QUEUE_PORT}/size" && echo
  fi
  stop_queue
fi

SCRIPT_END=$(date +%s)
ELAPSED=$(( SCRIPT_END - SCRIPT_START ))
printf "===> total elapsed: %02dh %02dm %02ds\n" \
  $(( ELAPSED / 3600 )) $(( (ELAPSED % 3600) / 60 )) $(( ELAPSED % 60 ))

exit $ec
