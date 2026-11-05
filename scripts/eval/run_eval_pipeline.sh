#!/usr/bin/env bash
# Single source of truth for the queue-driven eval pipeline.
#
# scripts/launch_eval.py runs this with an environment it builds (or prints the
# `env VAR=... bash <this>` invocation under --dry-run). Everything is driven by
# environment variables.
#
# Pipeline: policy server on GPU 0 -> autogen task codegen -> episode queue ->
# right-size num_envs from the queue -> optional wandb coordinator ->
# NUM_EVAL_GPUS eval workers on GPUs 1..N -> teardown.
# POLICY_NAME=replay skips the server and runs the workers with --replay
# (recorded actions from the queue's dataset) on GPUs 0..N-1 instead.

# -e matters here: without it a failed `git checkout` / `make` / `curl` only logged
# an error and the pipeline carried on, so a run could serve the wrong hvla build
# or an unbuilt FF planner and still report results. Sites that legitimately
# tolerate failure (teardown kills, the final queue-status curl) opt out with
# `|| true`; worker exit codes are collected into `ec` explicitly.
set -euo pipefail
set -x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/eval/_wait.sh
source "${SCRIPT_DIR}/_wait.sh"

# The model policies need the maniflow-tb and molmo2 submodules (the maniflow extra installs both).
if [ "${POLICY_NAME:-molmo2_maniflow}" != "replay" ] \
   && { [ ! -f "${REPO_ROOT}/maniflow-tb/pyproject.toml" ] || [ ! -f "${REPO_ROOT}/molmo2/pyproject.toml" ]; }; then
  echo "maniflow-tb / molmo2 submodules missing: git submodule update --init maniflow-tb molmo2" >&2
  exit 1
fi

RUNS_ROOT="${RUNS_ROOT:-$REPO_ROOT/runs}"
RUN_ID="$(date +%Y%m%d_%H%M%S)_${RUN_ID:-local}"
ENABLE_WANDB="${ENABLE_WANDB:-}"               # 1 from launch_eval.py --wandb

# -- Policy --
POLICY_NAME="${POLICY_NAME:-molmo2_maniflow}"
MODEL_PATH="${MODEL_PATH:-}"
HL_MODEL_PATH="${HL_MODEL_PATH:-}"
HL_PROMPT="${HL_PROMPT:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
STATS_PATH="${STATS_PATH:-}"
ACTION_MODE="${ACTION_MODE:-}"
POLICY_SEED="${POLICY_SEED:-}"
DETERMINISTIC_CUDA="${DETERMINISTIC_CUDA:-}"
REPLICATE_NOISE="${REPLICATE_NOISE:-}"   # 1 -> shared initial flow-matching noise across the batch (LL-only server)
H_VLA_PATH="${H_VLA_PATH:-$REPO_ROOT/maniflow-tb}"
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-$H_VLA_PATH/examples/Franka/train_maniflow_multimodal.yaml}"
# The policy server imports hvla from H_VLA_PATH as checked out (in the image, or by
# `git submodule update` locally); a job's `git checkout` does not move submodules. Serving a
# maniflow-tb other than the one this ToyBlocks commit pins would be silent, so refuse it.
if [ "$POLICY_NAME" != "replay" ]; then
  _MTB_PIN="$(git -C "$REPO_ROOT" rev-parse "HEAD:maniflow-tb" 2>/dev/null || true)"
  _MTB_HEAD="$(git -C "$H_VLA_PATH" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$_MTB_PIN" ] && [ "$_MTB_HEAD" != "$_MTB_PIN" ]; then
    echo "maniflow-tb at ${H_VLA_PATH} is ${_MTB_HEAD:-missing}, but this ToyBlocks commit pins ${_MTB_PIN}." >&2
    echo "Run: git submodule update --init maniflow-tb (in a docker image: rebuild it, docker/build_eval_image.sh)." >&2
    exit 1
  fi
fi

# -- Scene / task --
ROBOT_ID="${ROBOT_ID:-}"
SCENE_PRESET="${SCENE_PRESET:-}"
TASK_SLUG="${TASK_SLUG:?TASK_SLUG is required}"
TASK="${TASK_SLUG}-v0"
PROBLEM_ID="${PROBLEM_ID:-problem_000000}"
OBS_TRANSFORM="${OBS_TRANSFORM:-}"
LEGACY_EE_POS="${LEGACY_EE_POS:-}"   # 1 -> ee_pos from the last body in the robot base frame (pre-ee_body_name convention)
DATASET_PATH="${DATASET_PATH:-}"
OUT_SUBDIR="${OUT_SUBDIR:-evals}"

# -- Queue --
QUEUE_LIMIT_ARG="${QUEUE_LIMIT_ARG:-}"
QUEUE_LOG_LEVEL="${QUEUE_LOG_LEVEL:-INFO}"
EPISODE_INDICES="${EPISODE_INDICES:-}"
# Start each queued episode after these HL actions (flip,rotate). Empty = frame 0.
# Passed to serve_episode_queue.py only — codegen stays on frame 0.
INIT_AFTER_HL_ACTION="${INIT_AFTER_HL_ACTION:-}"
USD_LIBRARY="${USD_LIBRARY:-train_blocks_large}"
ALLOW_DUPLICATE_EPISODES="${ALLOW_DUPLICATE_EPISODES:-}"

# -- Eval sizing --
ACTION_HORIZON="${ACTION_HORIZON:-10}"
SYNC_ACTION_CHUNKS="${SYNC_ACTION_CHUNKS-1}"    # 1 -> one shared chunk boundary for all envs (--sync_action_chunks); empty -> per-env phases
HL_FLAG="${HL_FLAG:-}"                          # client-side HL: --use_oracle or empty
UNBATCHED="${UNBATCHED:-}"                      # 1 -> per-env inference (--unbatched)
HL_FROM_MODEL="${HL_FROM_MODEL:-}"             # 1 -> learned server HL (molmo2); enables the model HL overlay video
ASYNC_HL="${ASYNC_HL:-}"                       # 1 -> molmo2_maniflow: HL only at episode start + LL skill end (--async_hl)
ASYNC_HL_CALL_SCALE="${ASYNC_HL_CALL_SCALE:-}" # with ASYNC_HL: end an episode past this x its expert's HL calls (--async_hl_call_scale)
GO_HOME="${GO_HOME:-}"                         # 1 -> with ASYNC_HL / LL_ORACLE: arm goes home after place/flip/rotate, before the next HL query (--go_home)
LL_ORACLE="${LL_ORACLE:-}"                     # HL-call scale -> molmo2_maniflow: vecgen / cuRobo low-level oracle runs Molmo2's commands (--ll_oracle)
HL_NO_PREV_ACTION="${HL_NO_PREV_ACTION:-}"     # 1 -> molmo2_maniflow: no previous action in the Molmo2 question (--no-hl-prev-action)
HL_MAX_NEW_TOKENS="${HL_MAX_NEW_TOKENS:-}"     # molmo2_maniflow: Molmo2 decode budget (--hl_max_new_tokens); empty -> server default
ORACLE_COMPARISON="${ORACLE_COMPARISON:-}"     # 1 -> also tick a client-side comparison oracle for a model | oracle overlay panel
# HL latency (update interval K + estimated HL inference time M, milliseconds)
# + viz. Defaults reproduce today's fresh-every-step behavior. See
# toyblocks.serving.hl_latency.
HL_UPDATE_INTERVAL_MS="${HL_UPDATE_INTERVAL_MS:-0.0}"
HL_UPDATE_INTERVAL_JITTER_MS="${HL_UPDATE_INTERVAL_JITTER_MS:-0.0}"
ESTIMATED_HL_INFERENCE_TIME_MS="${ESTIMATED_HL_INFERENCE_TIME_MS:-0.0}"
SAVE_HL_LATENCY_VIDEO="${SAVE_HL_LATENCY_VIDEO:-}"
NUM_EVAL_GPUS="${NUM_EVAL_GPUS:-1}"
MAX_ENVS_PER_GPU="${MAX_ENVS_PER_GPU:-50}"
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-30}"
# Per-episode timeout as a multiple of that episode's own expert demonstration
# (1.5 = 150%). 0 keeps EPISODE_LENGTH_S as the only timeout. When set,
# section 3 raises EPISODE_LENGTH_S to cover the queue's longest expert so the
# per-episode budget is the binding one.
EXPERT_TIMEOUT_SCALE="${EXPERT_TIMEOUT_SCALE:-0}"
# POLICY_NAME=replay only: stats.json whose action.joint_target q01/q99 bands the
# replayed joint targets are clamped to. Empty replays the recording unclipped.
REPLAY_CLIP_STATS="${REPLAY_CLIP_STATS:-}"
# Policy evals with a maniflow obs transform: centre-crop ratio for the RGB inputs
# (evaluate.py --center_crop_ratio). Empty feeds the full frame.
CENTER_CROP_RATIO="${CENTER_CROP_RATIO:-}"
# Maniflow obs transform live depth aug (evaluate.py --depth_aug), e.g. ejbl.
DEPTH_AUG="${DEPTH_AUG:-}"
# Maniflow crop-trained checkpoints: client target_front_crop_rgb (evaluate.py --enable_crop).
ENABLE_CROP="${ENABLE_CROP:-}"
USE_CALIBRATED_CAMERAS="${USE_CALIBRATED_CAMERAS-1}"   # "" -> dataset camera K (--no-use-calibrated-cameras)
# Replay a rendered dataset's recorded camera pose and look (evaluate.py --use_render_params).
USE_RENDER_PARAMS="${USE_RENDER_PARAMS:-}"
# Paint a glyph/colour render's recorded block identities and prompt with its goal (evaluate.py --replay_block_identity).
REPLAY_BLOCK_IDENTITY="${REPLAY_BLOCK_IDENTITY-1}"   # "" -> authored look, source-name prompts (--no-replay_block_identity)
# Start episodes in waves sharing the dataset's recorded lighting (evaluate.py --wave_resets).
WAVE_RESETS="${WAVE_RESETS:-}"
REPLAY_GRIPPER_CLOSE_TARGET="${REPLAY_GRIPPER_CLOSE_TARGET:-}"
# Extra world Z offset (m) for workspace_table + blocks after queue setup (e.g. -0.02).
EVAL_TABLE_Z_OFFSET="${EVAL_TABLE_Z_OFFSET:-}"

# -- Recording / diagnostics --
FIRST_BATCH_ARG="${FIRST_BATCH_ARG:-}"
DIAGNOSTIC_PLOTS_ARG="${DIAGNOSTIC_PLOTS_ARG:-}"
DIAGNOSTIC_PLOT_WORKERS="${DIAGNOSTIC_PLOT_WORKERS:-8}"   # processes for the end-of-run offline diagnostics render
SENSOR_OBS_VIDEO_ARG="${SENSOR_OBS_VIDEO_ARG:-}"
HL_OVERLAY_VIDEO_ARG="${HL_OVERLAY_VIDEO_ARG:-}"   # --save_hl_overlay_video: HL overlay stream in the LeRobot dataset
STEP_METRICS_ARG="${STEP_METRICS_ARG:-}"    # --save_step_metrics: per-step stats CSVs (debugging)
TERMINATE_ON_SUCCESS_ARG="${TERMINATE_ON_SUCCESS_ARG:-}"
SAVE_LEROBOT="${SAVE_LEROBOT-1}"            # LeRobot recording on unless set empty
LEROBOT_DATASET_NAME="${LEROBOT_DATASET_NAME:-}"
LEROBOT_NUM_SHARDS="${LEROBOT_NUM_SHARDS:-8}"
LEROBOT_VIDEO_CODEC="${LEROBOT_VIDEO_CODEC:-libx264}"

# -- Wandb --
WANDB_PROJECT="${WANDB_PROJECT:-toyblocks-eval}"
# Extract source-dataset expert MP4s for the coordinator's episodes table.
SAVE_EXPERT_VIDEO="${SAVE_EXPERT_VIDEO:-${ENABLE_WANDB:-}}"
EVAL_PARAMS="${EVAL_PARAMS:-}"
LAUNCH_COMMAND="${LAUNCH_COMMAND:-}"

# -- Ports --
POLICY_PORT="${POLICY_PORT:-8000}"
QUEUE_PORT="${QUEUE_PORT:-8001}"

ec=0
SERVE0=""
QUEUE_PID=""
COORD_PID=""
EVAL_PIDS=()

_TEE_PID=""
cleanup() {
  kill "${SERVE0:-}" "${QUEUE_PID:-}" "${COORD_PID:-}" 2>/dev/null || true
  for pid in "${EVAL_PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  kill "${_TEE_PID:-}" 2>/dev/null || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# -- Save a copy of all stdout/stderr next to the run artifacts --
# Each seed gets its own std_out_err.log.  The shared pre-worker output
# (policy server, codegen, queue) is captured into a temporary file first,
# then prepended into every per-seed log inside the eval-worker loop.
_SHARED_LOG="$(mktemp /tmp/eval_shared_log.XXXXXX)"
exec > >(tee -a "${_SHARED_LOG}") 2>&1
_TEE_PID=$!
echo "===> shared pipeline log (temporary): ${_SHARED_LOG}"

# =============================================================================
# 1. Policy server on GPU 0
# =============================================================================
if [ "$POLICY_NAME" = "replay" ]; then
  # Dataset replay: the eval workers play back each queue episode's recorded
  # actions (evaluate.py --replay), so there is no policy server to start or
  # wait for. cleanup/teardown tolerate the empty pid.
  echo "===> POLICY_NAME=replay: no policy server (workers replay recorded actions)"
  SERVE0=""
elif [ "$POLICY_NAME" = "maniflow" ]; then
  export H_VLA_PATH
  export PYTHONPATH="${PYTHONPATH:-}:${H_VLA_PATH}"
  CUDA_VISIBLE_DEVICES=0 python -m toyblocks.serving.serve_maniflow_policy \
    --port "$POLICY_PORT" \
    --hl oracle \
    --checkpoint "$MODEL_PATH" \
    --config "$CONFIG_PATH" \
    --train_config "$TRAIN_CONFIG_PATH" \
    --stats_path "$STATS_PATH" \
    ${POLICY_SEED:+--seed "$POLICY_SEED"} \
    ${DETERMINISTIC_CUDA:+--deterministic-cuda} \
    ${REPLICATE_NOISE:+--replicate-noise} \
    ${ACTION_MODE:+--action-mode "$ACTION_MODE"} &
  SERVE0=$!
elif [ "$POLICY_NAME" = "molmo2_maniflow" ]; then
  export H_VLA_PATH
  export PYTHONPATH="${PYTHONPATH:-}:${H_VLA_PATH}"
  CUDA_VISIBLE_DEVICES=0 python -m toyblocks.serving.serve_maniflow_policy \
    --port "$POLICY_PORT" \
    --hl model \
    --checkpoint "$MODEL_PATH" \
    --hl_checkpoint "$HL_MODEL_PATH" \
    ${LL_ORACLE:+--skip-ll} \
    ${HL_PROMPT:+--hl_prompt "$HL_PROMPT"} \
    ${HL_NO_PREV_ACTION:+--no-hl-prev-action} \
    ${HL_MAX_NEW_TOKENS:+--hl_max_new_tokens "$HL_MAX_NEW_TOKENS"} \
    --config "$CONFIG_PATH" \
    --train_config "$TRAIN_CONFIG_PATH" \
    --stats_path "$STATS_PATH" \
    ${POLICY_SEED:+--seed "$POLICY_SEED"} \
    ${DETERMINISTIC_CUDA:+--deterministic-cuda} \
    --hl_update_interval_ms "$HL_UPDATE_INTERVAL_MS" \
    --hl_update_interval_jitter_ms "$HL_UPDATE_INTERVAL_JITTER_MS" \
    --estimated_hl_inference_time_ms "$ESTIMATED_HL_INFERENCE_TIME_MS" \
    ${ACTION_MODE:+--action-mode "$ACTION_MODE"} &
  SERVE0=$!
else
  echo "unknown POLICY_NAME=${POLICY_NAME}" >&2
  exit 1
fi
if [ -n "$SERVE0" ]; then
  wait_policy_port "$POLICY_PORT" "$SERVE0" "policy"
  sleep 5
fi

# =============================================================================
# 2. Autogen task codegen from the dataset
# =============================================================================
echo "===> generating autogen task from ${DATASET_PATH}"
CODEGEN_EPISODE_INDICES_ARG=""
if [ -n "$EPISODE_INDICES" ]; then
  CODEGEN_EPISODE_INDICES_ARG="--episode-indices $EPISODE_INDICES"
fi
CODEGEN_USD_LIBRARY_ARG=""
if [ -n "$USD_LIBRARY" ]; then
  CODEGEN_USD_LIBRARY_ARG="--usd-library $USD_LIBRARY"
fi
python scripts/generate_task_from_dataset.py \
  --dataset-path "$DATASET_PATH" \
  --domain-path src/toyblocks/tamp/blocks \
  --output-dir src/toyblocks/tasks/autogen \
  --task-name "$TASK_SLUG" \
  $CODEGEN_EPISODE_INDICES_ARG \
  $CODEGEN_USD_LIBRARY_ARG \
  --headless || exit 1

# =============================================================================
# 3. Episode queue + right-size num_envs
# =============================================================================
if [ -n "$DATASET_PATH" ]; then
  echo "===> starting queue server on :${QUEUE_PORT} (log_level=${QUEUE_LOG_LEVEL}, limit_arg='${QUEUE_LIMIT_ARG}')"
  EPISODE_INDICES_ARG=""
  if [ -n "$EPISODE_INDICES" ]; then
    EPISODE_INDICES_ARG="--episode-indices $EPISODE_INDICES"
  fi
  ALLOW_DUP_ARG=""
  if [ -n "$ALLOW_DUPLICATE_EPISODES" ]; then
    ALLOW_DUP_ARG="--allow-duplicate-episodes"
  fi
  INIT_AFTER_HL_ACTION_ARG=""
  if [ -n "$INIT_AFTER_HL_ACTION" ]; then
    INIT_AFTER_HL_ACTION_ARG="--init-after-hl-action $INIT_AFTER_HL_ACTION"
  fi
  python scripts/serve_episode_queue.py \
    --dataset-path "$DATASET_PATH" \
    $EPISODE_INDICES_ARG \
    $ALLOW_DUP_ARG \
    $INIT_AFTER_HL_ACTION_ARG \
    $QUEUE_LIMIT_ARG \
    --log-level "$QUEUE_LOG_LEVEL" \
    --port "$QUEUE_PORT" &
  QUEUE_PID=$!
  wait_policy_port "$QUEUE_PORT" "$QUEUE_PID" "queue"
  QUEUE_STATUS=$(curl -sS "http://localhost:${QUEUE_PORT}/size")
  QUEUE_SIZE=$(printf '%s' "$QUEUE_STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['total'])")
  echo "===> queue size: ${QUEUE_SIZE}"
  QUEUE_URL_ARG="--queue_url http://localhost:${QUEUE_PORT}/next"
  # Isaac Lab's max_episode_length is a single scalar, so the per-episode expert
  # budget can only ever end an episode EARLY. Raise the global timeout to cover
  # the queue's longest expert (expert_seconds_max from /size) so it does, rather
  # than having every long episode cut off at the shared horizon instead.
  # Numeric test, not a string one: the launcher renders a float, so "0.0" and
  # "0" both mean off.
  if python3 -c "import sys; sys.exit(0 if float('${EXPERT_TIMEOUT_SCALE}') > 0 else 1)"; then
    _RAISED=$(printf '%s' "$QUEUE_STATUS" | python3 -c "import json,math,sys; print(max(${EPISODE_LENGTH_S}, math.ceil(${EXPERT_TIMEOUT_SCALE} * json.load(sys.stdin)['expert_seconds_max'])))")
    echo "===> expert_timeout_scale=${EXPERT_TIMEOUT_SCALE}: episode_length_s ${EPISODE_LENGTH_S} -> ${_RAISED}"
    EPISODE_LENGTH_S="${_RAISED}"
  fi
else
  QUEUE_URL_ARG=""
  QUEUE_SIZE=0
fi

# Upper bound only; evaluate.py spreads num_envs over the task's baked cube-scale
# slots by queued workload (see reconcile_num_envs_with_task).
if [ "$QUEUE_SIZE" -gt 0 ] 2>/dev/null; then
  NUM_ENVS=$(python3 -c "import math; print(min(math.ceil(${QUEUE_SIZE} / ${NUM_EVAL_GPUS}), ${MAX_ENVS_PER_GPU}))")
else
  NUM_ENVS=${MAX_ENVS_PER_GPU}
fi
echo "===> num_envs per GPU: ${NUM_ENVS} (queue=${QUEUE_SIZE}, gpus=${NUM_EVAL_GPUS}, max=${MAX_ENVS_PER_GPU})"

# =============================================================================
# 4. Optional wandb coordinator
# =============================================================================
EVAL_SENTINEL="/tmp/eval_done_${RUN_ID}"
rm -f "${EVAL_SENTINEL}"
if [ -n "$ENABLE_WANDB" ]; then
  # --ll_oracle runs are told apart in wandb; POLICY_NAME still picks the server and run paths.
  WANDB_POLICY="$POLICY_NAME"
  [ -n "$LL_ORACLE" ] && WANDB_POLICY="molmo2_ll_oracle"
  python scripts/wandb_consolidate.py \
    --wandb-project "$WANDB_PROJECT" \
    --wandb-name "${WANDB_POLICY}__${TASK}__${OUT_SUBDIR}" \
    --wandb-tags "$WANDB_POLICY" "$TASK" "$OUT_SUBDIR" \
    --output-dirs "${RUNS_ROOT}/${TASK}/${POLICY_NAME}/${OUT_SUBDIR}/${PROBLEM_ID}/seed_*/${RUN_ID}" \
    --eval-params "$EVAL_PARAMS" \
    --sentinel "${EVAL_SENTINEL}" &
  COORD_PID=$!
else
  echo "===> wandb disabled (set ENABLE_WANDB=1 to enable)"
fi

# =============================================================================
# 4b. OpenCV contrib for live depth augmentation (client-side --depth_aug)
# =============================================================================
if [ -n "${DEPTH_AUG:-}" ]; then
  # shellcheck source=scripts/eval/install_hvla_opencv_contrib.sh
  source "${SCRIPT_DIR}/install_hvla_opencv_contrib.sh"
  install_hvla_opencv_contrib
fi

# =============================================================================
# 5. Eval workers on GPUs 1..N (0..N-1 under replay: no server holds GPU 0)
# =============================================================================
GPU_OFFSET=0
if [ "$POLICY_NAME" = "replay" ]; then
  GPU_OFFSET=1
fi
# Fewer GPUs than the layout asks for (a one-GPU workstation): workers share GPUs round-robin,
# worker 1 with the policy server on GPU 0. A no-op when there are enough GPUs.
NUM_VISIBLE_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
[ "$NUM_VISIBLE_GPUS" -ge 1 ] || NUM_VISIBLE_GPUS=1
# Action-source args differ per branch: a policy server connection with its
# obs transform / chunking / HL flags, or --replay (recorded actions from the
# queue's dataset; no server, no transform, no HL).
if [ "$POLICY_NAME" = "replay" ]; then
  POLICY_ARGS=(--replay ${REPLAY_CLIP_STATS:+--replay_clip_stats "$REPLAY_CLIP_STATS"})
else
  POLICY_ARGS=(
    --policy_host localhost --policy_port "$POLICY_PORT"
    --obs_transform "${OBS_TRANSFORM}"
    ${CENTER_CROP_RATIO:+--center_crop_ratio "$CENTER_CROP_RATIO"}
    ${DEPTH_AUG:+--depth_aug "$DEPTH_AUG"}
    ${ENABLE_CROP:+--enable_crop}
    --action_horizon "${ACTION_HORIZON}"
    ${HL_FLAG}
    ${UNBATCHED:+--unbatched}
    --hl_update_interval_ms "$HL_UPDATE_INTERVAL_MS"
    --hl_update_interval_jitter_ms "$HL_UPDATE_INTERVAL_JITTER_MS"
    --estimated_hl_inference_time_ms "$ESTIMATED_HL_INFERENCE_TIME_MS"
    ${SAVE_HL_LATENCY_VIDEO:+--save_hl_latency_video}
    ${HL_FROM_MODEL:+--hl_from_model}
    ${ORACLE_COMPARISON:+--oracle_comparison}
  )
fi
# Path of a seed's run directory relative to RUNS_ROOT; the merge step below reuses it.
run_rel() { echo "${TASK}/${POLICY_NAME}/${OUT_SUBDIR}/${PROBLEM_ID}/seed_$1/${RUN_ID}"; }
for EVAL_GPU in $(seq 1 "$NUM_EVAL_GPUS"); do
  SEED=${EVAL_GPU}
  OUT="${RUNS_ROOT}/$(run_rel "${SEED}")"
  mkdir -p "${OUT}"
  # Provenance next to the run artifacts. wandb_consolidate.py already puts the same
  # payload in the run config, but that is unavailable when ENABLE_WANDB is unset.
  if [ -n "$EVAL_PARAMS" ]; then
    printf '%s\n' "$EVAL_PARAMS" > "${OUT}/launch_params.json"
  fi
  if [ -n "$LAUNCH_COMMAND" ]; then
    printf '#!/usr/bin/env bash\n%s\n' "$LAUNCH_COMMAND" > "${OUT}/launch_command.sh"
    chmod +x "${OUT}/launch_command.sh"
  fi
  # Per-seed log: shared preamble (server startup, codegen, queue) + worker output.
  _SEED_LOG="${OUT}/std_out_err.log"
  cp "${_SHARED_LOG}" "${_SEED_LOG}"
  export KIT_CACHE_ROOT="/tmp/kit-cache-${EVAL_GPU}"
  export KIT_LOG_DIR="/tmp/kit-logs-${EVAL_GPU}"
  export OMNI_USER_CONFIG_DIR="/tmp/omni-config-${EVAL_GPU}"
  CUDA_DEVICE=$(( (EVAL_GPU - GPU_OFFSET) % NUM_VISIBLE_GPUS ))
  echo "===> launching evaluate.py on GPU ${CUDA_DEVICE} (seed=${SEED})"
  CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python scripts/evaluate.py \
    --task "${TASK}" --seed "${SEED}" \
    ${QUEUE_URL_ARG} \
    "${POLICY_ARGS[@]}" \
    --output_dir "${OUT}" \
    ${ROBOT_ID:+--env-cfg.robot-id "${ROBOT_ID}"} \
    ${SCENE_PRESET:+--env-cfg.scene-preset-id "${SCENE_PRESET}"} \
    ${REPLAY_GRIPPER_CLOSE_TARGET:+--env-cfg.replay-gripper-close-target "${REPLAY_GRIPPER_CLOSE_TARGET}"} \
    ${EVAL_TABLE_Z_OFFSET:+--env-cfg.eval-table-z-offset "${EVAL_TABLE_Z_OFFSET}"} \
    --env-cfg.episode-length-s "${EPISODE_LENGTH_S}" \
    --expert_timeout_scale "${EXPERT_TIMEOUT_SCALE}" \
    --env-cfg.num-renders-on-reset 4 \
    ${LEGACY_EE_POS:+--env-cfg.legacy-ee-pos} \
    --control_mode arm_joint_gripper_binary \
    --obs_mode rgb+state \
    --num_envs "${NUM_ENVS}" \
    --track_metrics \
    ${SYNC_ACTION_CHUNKS:+--sync_action_chunks} \
    ${ASYNC_HL:+--async_hl} \
    ${ASYNC_HL_CALL_SCALE:+--async_hl_call_scale "$ASYNC_HL_CALL_SCALE"} \
    ${GO_HOME:+--go_home} \
    ${LL_ORACLE:+--ll_oracle --ll_oracle_hl_call_scale "$LL_ORACLE"} \
    $SENSOR_OBS_VIDEO_ARG \
    ${SENSOR_OBS_VIDEO_ARG:+--save_video_all_envs} \
    $HL_OVERLAY_VIDEO_ARG \
    $STEP_METRICS_ARG \
    ${SAVE_EXPERT_VIDEO:+--save_expert_video} \
    $FIRST_BATCH_ARG \
    $DIAGNOSTIC_PLOTS_ARG \
    $TERMINATE_ON_SUCCESS_ARG \
    ${DETERMINISTIC_CUDA:+--deterministic_cuda} \
    ${DATASET_PATH:+--dataset_path "${DATASET_PATH}"} \
    $([ -n "$USE_CALIBRATED_CAMERAS" ] || echo --no-use-calibrated-cameras) \
    ${USE_RENDER_PARAMS:+--use_render_params} \
    $([ -n "$REPLAY_BLOCK_IDENTITY" ] || echo --no-replay_block_identity) \
    ${WAVE_RESETS:+--wave_resets} \
    ${SAVE_LEROBOT:+--save_lerobot --num_shards "${LEROBOT_NUM_SHARDS}" --video_codec "${LEROBOT_VIDEO_CODEC}"} \
    ${LEROBOT_DATASET_NAME:+--lerobot_dataset_name "${LEROBOT_DATASET_NAME}"} \
    --device cuda:0 > >(tee -a "${_SEED_LOG}") 2>&1 &
  EVAL_PIDS+=($!)
done

# =============================================================================
# 6. Wait for eval workers, then the coordinator
# =============================================================================
echo "===> waiting on eval client(s)"
for pid in "${EVAL_PIDS[@]}"; do
  wait "$pid" || ec=1
done
echo "===> eval done (ec=${ec})"
if [ -n "$QUEUE_URL_ARG" ]; then
  echo "queue final status:"
  curl -sS "http://localhost:${QUEUE_PORT}/size" && echo || true
fi

touch "${EVAL_SENTINEL}"
if [ -n "$COORD_PID" ]; then
  echo "===> waiting on wandb coordinator"
  wait "$COORD_PID" || ec=1
fi

# =============================================================================
# 7. Teardown
# =============================================================================
kill "${QUEUE_PID:-}" 2>/dev/null || true
kill "${SERVE0:-}" 2>/dev/null || true
wait "${SERVE0:-}" 2>/dev/null || true

# Every episode's control diagnostics, rendered offline from the recorded
# LeRobot data (the wandb coordinator only renders one representative episode).
# A plotting failure is reported but must not fail the eval.
if [ -n "$DIAGNOSTIC_PLOTS_ARG" ]; then
  for EVAL_GPU in $(seq 1 "$NUM_EVAL_GPUS"); do
    OUT="${RUNS_ROOT}/$(run_rel "${EVAL_GPU}")"
    [ -f "${OUT}/eval_episodes.json" ] || continue
    echo "===> rendering offline diagnostics for every episode in ${OUT}"
    python scripts/plot_diagnostics_from_eval_run.py \
      --eval-dir "${OUT}" --num-workers "${DIAGNOSTIC_PLOT_WORKERS}" \
      || echo "WARNING: offline diagnostics failed for ${OUT}" >&2
  done
fi

# Per-skill success rates (skill_success.json) from each run's episode table.
# A failure is reported but must not fail the eval.
for EVAL_GPU in $(seq 1 "$NUM_EVAL_GPUS"); do
  OUT="${RUNS_ROOT}/$(run_rel "${EVAL_GPU}")"
  [ -f "${OUT}/eval_episodes.json" ] || continue
  echo "===> per-skill success for ${OUT}"
  python scripts/skill_success_from_eval_run.py --eval-dir "${OUT}" \
    || echo "WARNING: per-skill success failed for ${OUT}" >&2
done

# The seed runs merged into one run directory (success LeRobot dataset + episode
# table) with its own skill_success.json; after the per-seed steps, which read the
# seeds' success datasets the merge moves out.
MERGED_REL="${TASK}/${POLICY_NAME}/${OUT_SUBDIR}/${PROBLEM_ID}/${RUN_ID}"
python scripts/merge_eval_runs.py --out-dir "${RUNS_ROOT}/${MERGED_REL}" \
  --seed-dirs $(for g in $(seq 1 "$NUM_EVAL_GPUS"); do echo "${RUNS_ROOT}/$(run_rel "$g")"; done) \
  && python scripts/skill_success_from_eval_run.py --eval-dir "${RUNS_ROOT}/${MERGED_REL}" \
  || echo "WARNING: merging the seed runs into ${RUNS_ROOT}/${MERGED_REL} failed" >&2

rm -f "${_SHARED_LOG}"
exit $ec
