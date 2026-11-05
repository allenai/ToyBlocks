#!/usr/bin/env bash
# Foreground policy server for the split-terminal local eval workflow.
#
# Runs the merged serve_maniflow_policy in the FOREGROUND so you can drop a
# `breakpoint()`/`pdb` into the server and hit it interactively. --hl selects
# the HL conditioning source (see _common.sh): oracle = LL-only batched server
# (client injects oracle HL), model = single-agent Molmo2 HL + LL server.
#
# Usage: activate the venv, then `bash scripts/eval/serve_local_server.sh`.
# Override any config var inline, e.g. `POLICY_NAME=molmo2_maniflow bash ...`.
# Pair with scripts/eval/serve_local_client.sh in a second terminal.

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/eval/_common.sh
source "${SCRIPT_DIR}/_common.sh"

# Free any process squatting the policy port / GPU from a previous run. Scoped to
# this user's processes: `pkill -f` matches on the whole command line, so on a
# shared node an unscoped pattern kills other people's servers too.
pkill -9 -u "$(id -u)" -f "blocksuite.serving.serve_maniflow_policy" || true

# =============================================================================
# POLICY SERVER LAUNCH (foreground)
# =============================================================================

if [ "$POLICY_NAME" = "maniflow" ] || [ "$POLICY_NAME" = "molmo2_maniflow" ]; then
  export H_VLA_PATH
  export PYTHONPATH="${PYTHONPATH:-}:${H_VLA_PATH}"

  # No repo default for the checkpoint (a per-user weka path), so check it here rather
  # than letting the model load fail on an empty path.
  if [ -z "$MODEL_PATH" ]; then
    echo "MODEL_PATH is required for ${POLICY_NAME}; set it in scripts/eval/_common.sh or export it" >&2
    exit 1
  fi
  # config.yaml sits next to the checkpoint dir: <run>/config.yaml for
  # <run>/checkpoints/steps_N_model.safetensors, as launch_eval.py derives it.
  if [ -z "$CONFIG_PATH" ]; then
    case "$MODEL_PATH" in
      *.safetensors) CONFIG_PATH="$(dirname "$(dirname "$MODEL_PATH")")/config.yaml" ;;
      *) CONFIG_PATH="$(dirname "$MODEL_PATH")/config.yaml" ;;
    esac
  fi

  if [ "$HL" = "model" ] && [ -z "$HL_MODEL_PATH" ]; then
    echo "HL_MODEL_PATH is required for molmo2_maniflow (--hl model)" >&2
    exit 1
  fi

  HL_CHECKPOINT_ARG=()
  if [ "$HL" = "model" ]; then
    HL_CHECKPOINT_ARG=(--hl_checkpoint "$HL_MODEL_PATH")
  fi

  CUDA_VISIBLE_DEVICES=0 python -m blocksuite.serving.serve_maniflow_policy \
      --port "$POLICY_PORT" \
      --hl "$HL" \
      --checkpoint "$MODEL_PATH" \
      "${HL_CHECKPOINT_ARG[@]}" \
      ${HL_NO_PREV_ACTION:+--no-hl-prev-action} \
      --config "$CONFIG_PATH" \
      --train_config "$TRAIN_CONFIG_PATH" \
      --stats_path "$STATS_PATH" \
      ${ACTION_MODE:+--action-mode "$ACTION_MODE"}

elif [ "$POLICY_NAME" = "none" ]; then
  echo "Not launching policy"
else
  echo "unknown POLICY_NAME=${POLICY_NAME}" >&2
  exit 1
fi
