#!/usr/bin/env bash

# Eval-side counterpart to tests/tamp/datagen.sh: check that eval can rebuild the
# scene a datagen dataset was recorded in.
#
# Per dataset it generates the eval task package from the dataset (which bakes
# the recorded scene_preset_id / robot_id), picks one episode per distinct
# randomization profile, and runs verify_dataset_loading.py ONCE with all of
# them. Almost every check is against the dataset's own recorded ground truth, so
# there is nothing to commit and nothing to re-baseline.
#
# The exception is the policy-input leg, which also feeds the reset observation
# through the ManiFlow obs transform and checks what the model would actually
# receive. Its dataset-derived checks need no golden either; the one committed
# file is the observation *contract* (keys, dtypes, ranks, native resolution,
# emitted wire layout), keyed per scene preset because the preset is what decides
# camera names and native resolution. The two presets here currently agree on all
# of it -- the committed goldens differ only in scene_preset_id -- so the second
# dataset pins that agreement rather than covering a second layout. That is also
# the depth in --obs-mode below, since the transform is driven at its widest
# layout. Re-record with --write-policy-input-golden and review
# `git diff tests/eval/goldens/`.
#
# One Isaac process per dataset, not per profile: the env is built with one
# spawn slot per recorded cube_scale, so a single session already contains every
# profile's scene and a second cold start would rebuild the same thing. Datasets
# still get a process each (the script calls os._exit, and the scene differs),
# so their exit codes are collected and reported together at the end instead of
# aborting on the first failure.
#
# Requires tests/tamp/datagen.sh to have run first (both datasets carry different
# recorded scene_ids, which is what makes the scene reconciliation observable).
# Chain them with && -- a failed datagen otherwise leaves this script silently
# verifying the previous run's datasets, which the -d check below cannot catch.
#
#   ./tests/tamp/datagen.sh --mode all --record-data --no-ui && ./tests/eval/scene_load.sh

# No `set -e`: one failing case must not hide the others (see above). No `set -u`
# either, so the empty FAILURES/SUMMARY arrays expand cleanly on older bash.
set -o pipefail

# Resolved from the script's own location, not `git rev-parse`: this has to work
# in a plain working copy on a shared filesystem, which is not a git repo.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

SHOW_HELP=false

DATASET=all
PROFILES=2
OUTPUT_DIR=runs/eval_scene_load
WRITE_POLICY_INPUT_GOLDEN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --profiles) PROFILES="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --write-policy-input-golden) WRITE_POLICY_INPUT_GOLDEN=true; shift ;;
        *) echo "Unknown option: $1. Run with --help for options"; exit 1 ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    cat <<EOF
Verify eval rebuilds each datagen dataset's scene.
Usage: ./tests/eval/scene_load.sh [OPTIONS]

Options:
  -h, --help       Display this help message
  --dataset        One of (pnp | rotate | all). Default: all
                     pnp    -> datasets/pnp_test    (sim2real scene)
                     rotate -> datasets/rotate_test (droid_with_box_table scene)
  --profiles       Distinct (cube_scale, table_height_delta) profiles to cover
                   per dataset, all within that dataset's single Isaac launch.
                   Default: 2
  --output-dir     Where reports + first-frame PNGs land. Default: ${OUTPUT_DIR}
  --write-policy-input-golden
                   Re-record each dataset's observation-contract golden under
                   tests/eval/goldens/ instead of comparing against it. Only a
                   run whose dataset-derived policy-input checks pass may write.
                   Review \`git diff tests/eval/goldens/\` before committing.

Example: ./tests/eval/scene_load.sh --dataset rotate --profiles 1
EOF
    exit 0
fi

mkdir -p "${OUTPUT_DIR}"

# Per-dataset configuration. DATA_DIR mirrors the --data_dir that
# tests/tamp/datagen.sh passes to execute_plans.py; blocks_success is the
# LeRobot root the recorder writes successful episodes into.
select_dataset() {
    case "$1" in
        pnp)
            DATA_DIR=datasets/pnp_test/blocks_success
            EVAL_TASK_NAME=pnp_eval_test
            ;;
        rotate)
            DATA_DIR=datasets/rotate_test/blocks_success
            EVAL_TASK_NAME=rotate_eval_test
            ;;
        *)
            echo "Unknown --dataset '$1'. Use [pnp|rotate|all]"; exit 1 ;;
    esac
}

FAILURES=()
SUMMARY=()

run_dataset() {
    local name="$1"
    select_dataset "${name}"

    if [[ ! -d "${DATA_DIR}" ]]; then
        echo "[ERROR]: ${DATA_DIR} does not exist. Run ./tests/tamp/datagen.sh first."
        FAILURES+=("${name} (no dataset)")
        return
    fi

    echo "=== dataset=${name} data_dir=${DATA_DIR} task=${EVAL_TASK_NAME} ==="

    # Rebuilds the eval task package from the dataset every run: the package
    # bakes the recorded scene/robot and one spawn slot per recorded cube_scale,
    # so a stale package from a previous dataset is exactly the mismatch this
    # test exists to catch.
    if ! python scripts/generate_task_from_dataset.py \
        --dataset-path "${DATA_DIR}" \
        --task-name "${EVAL_TASK_NAME}" \
        --output-dir src/toyblocks/tasks/autogen \
        --headless; then
        echo "[ERROR]: generate_task_from_dataset failed for ${name}"
        FAILURES+=("${name} (task generation)")
        return
    fi

    # Raises when the dataset has fewer than --profiles distinct profiles, which
    # means the PDDL randomization annotations did not take effect or too few
    # policies succeeded. That is a real failure, not a reason to verify less.
    local episodes
    if ! episodes=$(python tests/eval/pick_profile_episodes.py \
        --dataset-path "${DATA_DIR}" \
        --num-profiles "${PROFILES}" \
        --verbose); then
        echo "[ERROR]: could not pick ${PROFILES} distinct profile(s) for ${name}"
        FAILURES+=("${name} (profile selection)")
        return
    fi

    # Episodes are also reported by profile RANK: which episode indices exist
    # depends on which policies happened to succeed, so the rank is what stays
    # comparable between runs. Rank is just the position in
    # pick_profile_episodes.py's profile-sorted output (0 = smallest cube_scale),
    # and verify_dataset_loading.py matches the two lists positionally.
    # A here-string, not a pipe, so the arrays survive the loop body.
    local episodes_arr=() ranks_arr=() rank=0
    while read -r episode; do
        [[ -z "${episode}" ]] && continue
        episodes_arr+=("${episode}")
        ranks_arr+=("${rank}")
        rank=$((rank + 1))
    done <<< "${episodes}"

    local case_out="${OUTPUT_DIR}/${name}"
    echo "[INFO]: verifying ${name} episodes ${episodes_arr[*]} (profile ranks ${ranks_arr[*]}, task ${EVAL_TASK_NAME}-v0) ..."
    # depth is in obs_mode for the policy-input leg: the ManiFlow transform is
    # driven at its widest layout, which reads depth + camera intrinsics.
    local golden_flag=()
    if [ "${WRITE_POLICY_INPUT_GOLDEN}" = true ]; then
        golden_flag=(--write-policy-input-golden)
    fi
    python scripts/eval/verify_dataset_loading.py \
        --dataset-path "${DATA_DIR}" \
        --task "${EVAL_TASK_NAME}-v0" \
        --episode-indices "${episodes_arr[@]}" \
        --profile-ranks "${ranks_arr[@]}" \
        --obs-mode rgb+state+depth \
        --output-dir "${case_out}" \
        "${golden_flag[@]}" \
        --headless \
        --enable_cameras
    local code=$?

    SUMMARY+=("${name} profiles ${ranks_arr[*]} (ep ${episodes_arr[*]}): exit=${code} (${case_out})")
    if [[ ${code} -ne 0 ]]; then
        FAILURES+=("${name}")
    fi
}

if [[ "${DATASET}" == "all" ]]; then
    DATASETS=(pnp rotate)
else
    DATASETS=("${DATASET}")
fi

for d in "${DATASETS[@]}"; do
    run_dataset "${d}"
done

echo ""
echo "=== scene_load summary ==="
for line in "${SUMMARY[@]}"; do
    echo "  ${line}"
done

if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo ""
    echo "FAILED (${#FAILURES[@]}): ${FAILURES[*]}"
    echo "Inspect ${OUTPUT_DIR}/<name>/ep<N>/scene_verification_ep<N>.json for every field"
    echo "diff with its tolerance and margin, and first_frame_*.png for the overlay."
    exit 1
fi

echo ""
echo "All ${#SUMMARY[@]} dataset(s) passed."
