# Shared config + helpers for the split-terminal local eval workflow.
#
# Sourced by serve_local_server.sh (policy server, one terminal) and
# serve_local_client.sh (queue + evaluate.py, another terminal). Running the
# server and client as foreground processes in separate terminals is what lets
# you drop a plain `breakpoint()`/`pdb` into either side -- unlike
# `launch_eval.py`, which backgrounds both as subprocesses with
# redirected stdio. Edit paths/ports here once; both scripts pick them up.
#
# Activate the venv before sourcing (e.g. `source .venv/bin/activate`); both
# scripts invoke bare `python` so breakpoints attach directly.

# -- Policy selection --
# maniflow        -> LL-only batched server, client-side oracle HL (--hl oracle)
# molmo2_maniflow -> single-agent HL+LL server, Molmo2 HL model  (--hl model)
POLICY_NAME="${POLICY_NAME:-molmo2_maniflow}"

# --hl passed to serve_maniflow_policy, derived from POLICY_NAME.
if [ "$POLICY_NAME" = "molmo2_maniflow" ]; then
  HL="model"
else
  HL="oracle"
fi

# -- Scene / robot --
# Empty by default: defer to the dataset's recorded scene_id/robot_id via
# reconcile_env_cfg_with_dataset (matching launch_eval.py's _MANIFLOW_DEFAULTS).
# Export them to force an explicit override.
ROBOT_ID="${ROBOT_ID:-}"
SCENE_PRESET="${SCENE_PRESET:-}"
ACTION_MODE="${ACTION_MODE:-}"

# -- maniflow-tb (hvla package) source --
# The venv's editable `hvla` is installed from this repo's maniflow-tb submodule, and that
# editable finder wins over PYTHONPATH, so H_VLA_PATH must be that same checkout (as in
# run_eval_pipeline.sh). It serves whatever the submodule has checked out: keep it at the pin
# with `git submodule update --init maniflow-tb`.
_BS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
H_VLA_PATH="${H_VLA_PATH:-$_BS_ROOT/maniflow-tb}"

# -- Model args (LL ManiFlow checkpoint + HL Molmo2 checkpoint) --
# Empty by default: these are per-user paths, so a repo default would
# silently evaluate someone else's checkpoint (or fail deep inside the model
# load) on any box where it happens to resolve. Export them, or edit them here
# for a local workflow; serve_local_server.sh requires MODEL_PATH (and HL_MODEL_PATH for
# molmo2_maniflow). CONFIG_PATH defaults to the config.yaml next to the checkpoint, as in
# launch_eval.py. STATS_PATH is an explicit override only: empty lets the server use the
# checkpoint's own normalizer.
MODEL_PATH="${MODEL_PATH:-}"
CONFIG_PATH="${CONFIG_PATH:-}"
STATS_PATH="${STATS_PATH:-}"
# Must be the in-repo hvla train config (at the maniflow-tb pin): it is merged
# under the checkpoint's own config in
# maniflow_serving_utils.load_maniflow_policy_common, so it supplies the
# framework fields the checkpoint omits. A train yaml from a different revision
# supplies a different architecture. Matches launch_eval.py's
# train_config_path default.
TRAIN_CONFIG_PATH="${TRAIN_CONFIG_PATH:-$H_VLA_PATH/examples/Franka/train_maniflow_multimodal.yaml}"
# Molmo2 HL checkpoint (required when POLICY_NAME=molmo2_maniflow / --hl model).
HL_MODEL_PATH="${HL_MODEL_PATH:-}"

# -- Task / eval args --
PROBLEM_ID="${PROBLEM_ID:-problem_000000}"
OBS_TRANSFORM="${OBS_TRANSFORM:-maniflow_blocks_table}"
# Empty by default (a per-user path, same reasoning as the checkpoints above).
# serve_local_client.sh requires either this or PDDL_PROBLEM_FILE.
DATASET_PATH="${DATASET_PATH:-}"
# Derive the autogen task slug from the dataset basename when unset, mirroring
# task_slug_from_dataset_path (hyphens -> underscores) so codegen and the gym id
# follow DATASET_PATH instead of a stale hardcoded default.
if [ -z "${TASK_SLUG:-}" ] && [ -n "$DATASET_PATH" ]; then
  TASK_SLUG="$(basename "$DATASET_PATH" | tr '-' '_')"
fi
TASK_SLUG="${TASK_SLUG:-}"
TASK="${TASK:-${TASK_SLUG}-v0}"
OUT_SUBDIR="${OUT_SUBDIR:-evals}"
EPISODE_INDICES="${EPISODE_INDICES:-0}"
# Start each queued episode after these HL actions (flip,rotate). Empty = frame 0.
# Passed to serve_episode_queue.py only — codegen stays on frame 0.
INIT_AFTER_HL_ACTION="${INIT_AFTER_HL_ACTION:-}"
# Glyph USD library for bare block names (AlphabetBlockLarge). Override with
# train_blocks for legacy small-block datasets.
USD_LIBRARY="${USD_LIBRARY:-train_blocks_large}"
# Per-episode sim time cap; launcher parity (run_eval_pipeline.sh default 30).
EPISODE_LENGTH_S="${EPISODE_LENGTH_S:-30}"

# -- PDDL-direct mode (set to bypass the dataset queue; uses generate_task_from_pddl.py) --
PDDL_PROBLEM_FILE="${PDDL_PROBLEM_FILE:-}"
ENV_PATH="${ENV_PATH:-}"   # Output dir for generate_task_from_pddl.py (default: src/blocksuite/tasks/autogen)

# -- Server / queue ports --
POLICY_PORT="${POLICY_PORT:-8000}"
QUEUE_PORT="${QUEUE_PORT:-8003}"
QUEUE_LIMIT_ARG="${QUEUE_LIMIT_ARG:-}"
QUEUE_LOG_LEVEL="${QUEUE_LOG_LEVEL:-INFO}"

# -- Eval capacity --
MAX_ENVS_PER_GPU="${MAX_ENVS_PER_GPU:-50}"
NUM_EVAL_GPUS="${NUM_EVAL_GPUS:-1}"
# Action chunk length the client requests per inference: ManiFlow checkpoints are trained at
# 25, launch_eval.py's default.
ACTION_HORIZON="${ACTION_HORIZON:-25}"
# Centre-crop of each resized 224x224 RGB input, launch_eval.py's default for the maniflow obs
# transforms ("" or 1.0 = no crop).
CENTER_CROP_RATIO="${CENTER_CROP_RATIO:-0.95}"
# 1 -> serve_maniflow_policy --no-hl-prev-action, for HL checkpoints trained without the
# "previous action" prompt (launch_eval.py --no-hl-prev-action).
HL_NO_PREV_ACTION="${HL_NO_PREV_ACTION:-}"
# 1 -> one shared action-chunk boundary for all envs (evaluate.py --sync_action_chunks), as in
# run_eval_pipeline.sh; the Molmo2 HL + LL server requires it. Empty -> per-env phases.
SYNC_ACTION_CHUNKS="${SYNC_ACTION_CHUNKS-1}"

# -- Output root for run artifacts (videos, eval_results.json, ...) --
OUT_PATH="${OUT_PATH:-runs}"

# -- Misc --
FIRST_BATCH_ARG="${FIRST_BATCH_ARG:-}"

# wait_policy_port (used for both the policy server and the episode queue) lives
# in _wait.sh, shared with scripts/eval/run_eval_pipeline.sh.
_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/eval/_wait.sh
source "${_COMMON_DIR}/_wait.sh"
