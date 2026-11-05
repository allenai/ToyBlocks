#!/usr/bin/env bash

# End-to-end datagen smoke test: PDDL -> task package -> vecgen (FF + cuRobo
# planning and Isaac execution in one loop) -> LeRobot dataset. Parameterized by --task so CI covers more than
# one scene: `flip_pnp` runs the template's sim2real preset, `rotate` runs
# droid_with_box_table (baked via generate_task_from_pddl.py --scene_preset_id).
# Both PDDLs carry the production randomization annotations, so codegen samples 8
# distinct (block_size, table_height, friction) profiles -- which is what
# tests/eval/scene_load.sh then needs in order to pick episodes across profiles.

set -e

# Resolved from the script's own location, not `git rev-parse`: this has to work
# in a plain working copy on a shared filesystem (weka), which is not a git repo.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${REPO_ROOT}"

SHOW_HELP=false

MODE=generate-task
TASK=all
RECORD_DATA=false
USE_DEBUGGER=false
NO_UI=false
USE_WANDB=false

# Overridden per --task by select_task(); both values are calibrated from a
# measured runner execution, not assumed.
SUCCESS_THRESHOLD=0.9

# Floor on the gate's denominator. A rate alone cannot tell a healthy run from a
# collapsed one (2 of 2 finished episodes is still 1.0). 8 is one full
# --num-envs 8 batch, the minimum that touches all 8 randomization slots, and a
# healthy run reports total=16 (--episodes 16), so this cannot false-fail while
# datagen works.
MIN_POLICIES=8

command -v fzf >/dev/null 2>&1 && HAS_FZF=true || HAS_FZF=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        --mode) MODE="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --record-data) RECORD_DATA=true; shift ;;
        --debug) USE_DEBUGGER=true; shift ;;
        --no-ui) NO_UI=true; shift ;;
        --use-wandb) USE_WANDB=true; shift ;;
        *) echo "Unkown option: $1. Run with --help for options"; exit 1 ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    cat <<EOF
Not sure what to put here, but this script does some cool stuff :D
Usage: ./tests/tamp/datagen.sh [OPTIONS]

Options:
  -h, --help    Display this help message
  --mode        One of (generate-task | vecgen | save-to-wandb | all)
  --task        One of (flip_pnp | rotate | all). Default: all
                  flip_pnp -> sim2real scene,             datasets/pnp_test
                  rotate   -> droid_with_box_table scene, datasets/rotate_test
  --record-data Also record camera videos (needed by tests/eval/scene_load.sh)
  --debug       Run with post-mortem debugging mode enabled to stop and break when we get a crash
  --no-ui       Runs the script without fzf prompting with its TUI for user input

Example: ./tests/tamp/datagen.sh --mode all --task rotate --record-data --no-ui
EOF
    exit 0
fi

# jq reads the success rate and the policy count that the only quality gate in
# this script compares against SUCCESS_THRESHOLD and MIN_POLICIES. Without it the
# run would report a pass nobody checked.
if ! command -v jq >/dev/null 2>&1; then
    echo "[ERROR]: jq is required to evaluate the success-rate gate. Install it and re-run."
    exit 1
fi

[ "$USE_DEBUGGER" == true ] && POST_MORTEM_CMD="-mipdb" || POST_MORTEM_CMD=""
[ "$RECORD_DATA" == true ] && RECORD_VIDEO_ARGS=(--record-videos --enable_cameras) || RECORD_VIDEO_ARGS=()

TMP_DIR="/tmp/$(whoami)"
mkdir -p "${TMP_DIR}"

# ToyBlocks has NO PDDL-problem generator (its scripts only CONSUME PDDLs).
# Start block_r at o17; goal o2 + on block_red1 -> REORIENT then PNP_TABLE_BLOCK.
# The three ;*_range annotations are read by
# codegen.parse_pddl_randomization_ranges_from_text and match the checked-in
# production problems (src/toyblocks/tamp/blocks/problems/flip.pddl); using
# narrower ranges here would test something large-scale datagen never runs.
cat << 'EOF' > "${TMP_DIR}/flip_pnp_r_on_red.pddl"
(define (problem flip-pnp-r-on-red)
    (:domain blocks-tamp)
    (:objects
        block_r - block_r_e
        block_red1 - block
        workspace_table - table
        robot - gripper
    )
    (:init
        (on_table block_r workspace_table)
        (on_table block_red1 workspace_table)
        (clear block_r)
        (clear block_red1)
        (handempty robot)
        (gripper_at_init)
        (ori_of block_r o17)
        (ori_of block_red1 o1)
    )
    (:goal (and
        (ori_of block_r o2)
        (on block_r block_red1)
    ))
    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)
EOF

# Second scene coverage: same randomization triple, ROTATE action only, run in
# droid_with_box_table so the two recorded datasets carry different scene_ids.
# block_r is declared under block_r_e, not the abstract `block` supertype: the
# rotate action takes `?b - block` so either grounds it, but every task_gen
# script, every checked-in problem and the oracle's _LETTER_BLOCK_TYPES map
# declare the letter cube under its subtype, and problem reconstruction at eval
# time emits block_r_e regardless of what datagen wrote.
cat << 'EOF' > "${TMP_DIR}/rotate_test.pddl"
(define (problem rotate-test)
    (:domain blocks-tamp)
    (:objects
        block_r - block_r_e
        workspace_table - table
        robot - gripper
    )
    (:init
        (on_table block_r workspace_table)
        (clear block_r)
        (handempty robot)
        (gripper_at_init)
        (ori_of block_r o1)
    )
    (:goal (and
        (ori_of block_r o2)
    ))
    ;block_size_range = [0.03, 0.05]
    ;table_height_range = [0, 0.04]
    ;friction_range = [0.1, 0.4]
)
EOF

# Per-task configuration. SCENE_ARGS is passed only to
# generate_task_from_pddl.py: vecgen loads the generated env package, so the
# scene follows automatically from there.
select_task() {
    case "$1" in
        flip_pnp)
            PDDL_PATH="${TMP_DIR}/flip_pnp_r_on_red.pddl"
            TASK_SLUG=flip_pnp_r_on_red
            DATA_DIR=datasets/pnp_test
            # The robot the committed obs-contract goldens (tests/eval/goldens) pin;
            # tests/eval/scene_load.sh checks the recorded robot_id against them.
            SCENE_ARGS=(--robot_id fr3_robotiq_gripper_softer)
            # vecgen measured 13/16, 14/16, 14/16 (seeds 0-2, RTX 5090; the
            # misses are flip effect checks). With --episodes 16 below, 0.75
            # tolerates four of 16 failing and fails at the fifth.
            SUCCESS_THRESHOLD=0.75
            MIN_POLICIES=8
            ;;
        rotate)
            PDDL_PATH="${TMP_DIR}/rotate_test.pddl"
            TASK_SLUG=rotate_test
            DATA_DIR=datasets/rotate_test
            # robot_id: see flip_pnp (the obs-contract goldens).
            SCENE_ARGS=(--scene_preset_id droid_with_box_table --robot_id fr3_robotiq_gripper_softer)
            # vecgen measured 16/16 (seed 0, RTX 5090); 0.9 fails at the second
            # failure of 16.
            # The droid_with_box_table table baseline differs from sim2real, so
            # sampled reset states can sit further from the arm's comfortable
            # workspace -- planning success is the first thing to check if this
            # case degrades below 0.9 or trips MIN_POLICIES.
            SUCCESS_THRESHOLD=0.9
            MIN_POLICIES=8
            ;;
        *)
            echo "Unknown --task '$1'. Use [flip_pnp|rotate|all]"; exit 1 ;;
    esac
    ENV_PATH="src/toyblocks/tasks/autogen/${TASK_SLUG}"
}

codegen_task_from_pddl() {
    echo "[INFO]: Generating task '${TASK_SLUG}' from pddl file ..."

    python ${POST_MORTEM_CMD} scripts/generate_task_from_pddl.py \
        --pddl "${PDDL_PATH}" \
        --domain_path src/toyblocks/tamp/blocks \
        --output_dir src/toyblocks/tasks/autogen \
        --num_reset_states 32 \
        --num_envs 8 \
        "${SCENE_ARGS[@]}" \
        --headless

    echo "[INFO]: DONE generating task :D"
}

vecgen_record() {
    echo "[INFO]: Recording '${TASK_SLUG}' with vecgen ..."

    # --episodes must stay above the 8 randomization slots, but only enough to
    # survive partial success: a single --num-envs 8 batch already touches every
    # slot, and only successful episodes land in blocks_success, so 16 (two
    # batches) is what keeps at least 2 distinct recorded (cube_scale,
    # table_height_delta) pairs -- which is what tests/eval/pick_profile_episodes.py
    # needs. vecgen records into ${DATA_DIR}/blocks_success and blocks_failure.
    rm -rf "${DATA_DIR}"
    python ${POST_MORTEM_CMD} scripts/vecgen/run.py \
        --env-path "${ENV_PATH}" \
        --pddl "${PDDL_PATH}" \
        --num-envs 8 \
        --episodes 16 \
        --seed 0 \
        --record "${DATA_DIR}" \
        --mark-complete \
        "${RECORD_VIDEO_ARGS[@]}" \
        --headless

    # results.json carries episodes / success; add the summary keys the gate
    # below and push_to_wandb.py read.
    jq '. + {total: .episodes, failure: (.episodes - .success),
             success_rate: (if .episodes > 0 then .success / .episodes else 0 end)}' \
        "${DATA_DIR}/results.json" > "${DATA_DIR}/results.json.tmp"
    mv "${DATA_DIR}/results.json.tmp" "${DATA_DIR}/results.json"

    echo "[INFO]: DONE recording :D"
}

save_to_wandb() {
    echo "Saving results to wandb ..."

    python ${POST_MORTEM_CMD} tests/tamp/push_to_wandb.py \
        --output-dir "${DATA_DIR}"

    echo "Done saving results! :D"
}

modes=(
    "generate-task"
    "vecgen"
    "save-to-wandb"
)

if [[ "$HAS_FZF" == true && "$NO_UI" != true ]]; then
    MODE=$(printf "%s\n" "${modes[@]}" | fzf --header="Select stage:" --layout=reverse)
fi

cat << 'EOF'
 _______ _________     ______   _______ _________ _______  _______  _______  _
(  ____ \\__   __/    (  __  \ (  ___  )\__   __/(  ___  )(  ____ \(  ____ \( (    /|
| (    \/   ) (       | (  \  )| (   ) |   ) (   | (   ) || (    \/| (    \/|  \  ( |
| |         | | _____ | |   ) || (___) |   | |   | (___) || |      | (__    |   \ | |
| |         | |(_____)| |   | ||  ___  |   | |   |  ___  || | ____ |  __)   | (\ \) |
| |         | |       | |   ) || (   ) |   | |   | (   ) || | \_  )| (      | | \   |
| (____/\___) (___    | (__/  )| )   ( |   | |   | )   ( || (___) || (____/\| )  \  |
(_______/\_______/    (______/ |/     \|   )_(   |/     \|(_______)(_______/|/    )_)

⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣤⡀⠀⠀⠀⠀⠀⠀⠀⣀⣀⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⣀⣀⡀⠀⠀⠀⢀⣾⠋⠉⢿⣆⠀⠀⠀⠀⢀⣾⠟⠙⢿⣄⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣶⢿⣿⢻⡛⠛⢛⡟⠛⠛⠿⠿⠃⠀⠀⠀⠻⣿⢿⣿⢿⣿⠁⠀⠀⠀⢻⡆⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⡾⠋⠫⣀⡠⠃⠀⠑⠶⠚⣥⣀⣀⡀⠀⠀⠀⠀⠀⠀⠋⠀⠋⠈⠋⠀⠀⠀⠀⠈⣿⡀⣀⣀⣀⡀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⡾⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⣉⣉⣛⠃⠀⠀⣴⣦⠀⠀⠀⣦⠀⠀⢠⣶⡄⠀⠀⢸⣟⠛⠉⠉⠁
⢰⣿⣿⣦⠀⠀⠀⠀⣰⡟⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠛⠛⠛⠃⠀⠀⠛⠛⠀⠘⠿⠻⠟⠀⠘⠛⠃⠀⠀⠈⣿⠛⠛⠛⠃
⢸⣯⠟⠛⣷⡀⠀⢠⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀
⠀⠹⣧⣴⣿⡿⢶⣾⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣇⠀⠀⠀
⠀⠀⠘⢿⣮⡀⡞⢹⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀
⠀⠀⠀⠀⠈⠛⢿⣾⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣿⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠈⣿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠹⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⡿⠁⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠻⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣰⡟⠁⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣤⡾⠋⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣤⡿⠛⠛⠛⠛⢿⣤⣿⠛⠛⠛⠛⠛⠛⠛⢿⣤⣾⠛⠛⠛⠛⠻⣧⣼⡏⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠀⠀⠀⠀⠀⠀⠉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠉⠁⠀⠀⠀⠀⠀⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀
EOF

show_happy_pusheen() {

cat << 'EOF'
 _______           _______  _______  _______  _______  _______  _
(  ____ \|\     /|(  ____ \(  ____ \(  ____ \(  ____ \(  ____ \( )
| (    \/| )   ( || (    \/| (    \/| (    \/| (    \/| (    \/| |
| (_____ | |   | || |      | |      | (__    | (_____ | (_____ | |
(_____  )| |   | || |      | |      |  __)   (_____  )(_____  )| |
      ) || |   | || |      | |      | (            ) |      ) |(_)
/\____) || (___) || (____/\| (____/\| (____/\/\____) |/\____) | _
\_______)(_______)(_______/(_______/(_______/\_______)\_______)(_)

⣤⣤⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣾⡿⣿⣇⠀⠀⠀⠀
⣿⢏⣹⣳⣯⣗⣤⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣾⣿⡿⠃⠒⣜⣮⢧⡀⠀⠀
⣿⡞⠁⡉⠙⠻⣷⣿⢦⣤⣤⣶⣶⣶⣶⣶⣶⣾⣿⡿⠋⠀⠌⡐⠈⢿⣿⣣⠀⠀
⣿⠀⢂⠐⡁⢂⣬⣿⣿⢫⠉⠀⠀⠀⠀⠀⠀⠜⡹⢿⣿⣿⣶⣶⣤⣈⣿⣷⣗⠀
⡇⢀⣦⣼⣾⣿⣿⣿⡭⡃⠌⠀⠀⠀⠀⠀⠀⠀⠑⡹⣚⢿⣿⣿⣿⣿⣿⣿⣼⠀
⣿⣿⣿⣿⣿⣿⣟⢧⢃⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⠉⢎⠳⢯⡟⣿⣻⢿⣯⡷
⣿⣿⡿⣟⡿⡓⢎⠂⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⣰⣬⣧⡝⢊⠙⣷
⠟⢧⠛⠥⠃⢉⠀⣴⣾⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣿⣿⣿⠀⠀⢸
⠈⠄⡈⠤⣁⠢⡀⢿⣿⣿⣿⠃⠀⠀⠀⠀⢠⡄⠀⣴⠀⠀⡀⢙⢛⡛⠭⢠⠃⢆
⠐⡠⢑⡒⡄⠓⡌⣌⢩⣩⠷⠶⣤⠀⠀⠀⠀⠳⠾⠃⢀⢸⡼⠋⠋⠛⢦⡃⠞⡠
⢀⠱⡈⢖⡈⢣⠜⣠⠟⠀⠀⠀⠀⢳⡄⠀⠀⠀⠀⠀⠐⣾⠁⠀⠀⠀⠈⢧⢣⢸
⣆⠠⢑⠢⣉⠆⢼⡟⠀⠀⠀⠀⠀⠈⣷⠀⠀⠀⠀⠀⢸⡇⠀⠀⠀⠀⠀⠈⣷⢯
⡏⠀⠀⢁⠂⢌⡟⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠈⣗⠀⠀⠀⠀⠀⠀⠈⢯
⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⣿⠀⠀⠀⠀⠀⠀⢿⡀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⡇⠀⠀⠀⠀⠀⠀⠘⣷⠀⠀⠀⠀⠀⠀⠀
EOF

}

show_sad_pusheen() {

cat << 'EOF'

 _______  _______ _________ _                 _______  _______
(  ____ \(  ___  )\__   __/( \      |\     /|(  ____ )(  ____ \
| (    \/| (   ) |   ) (   | (      | )   ( || (    )|| (    \/
| (__    | (___) |   | |   | |      | |   | || (____)|| (__
|  __)   |  ___  |   | |   | |      | |   | ||     __)|  __)
| (      | (   ) |   | |   | |      | |   | || (\ (   | (
| )      | )   ( |___) (___| (____/\| (___) || ) \ \__| (____/\
|/       |/     \|\_______/(_______/(_______)|/   \__/(_______/



⠀⠀⠀⠀⢀⡴⣄⠀⠀⠀⠀⢠⣄⠀⠀⠀⠀⠀⠀⠀⣼⣿⡟⠃⠀⠀⠀
⠀⠀⠀⣰⠋⠀⠈⠓⠒⠒⠒⠚⠈⢳⡄⠀⠀⠀⠀⠀⣿⣿⠀⠀⠀⠀
⠀⠀⣼⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣤⣤⣤⣤⣤⣿⣿⣄⠀⠀⠀⠀
⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣷⠀⠀⠀⠀⠀⠀⠙⣷⡴⠶⣦
⠀⠀⢷⡀⠀⠉⠉⠀⠀⠀⠉⠉⠀⠀⣠⡿⠀⠀⠀⢀⣀⣠⣤⠿⠞⠛⠋
⣠⠾⠋⠙⣶⠤⠤⠤⠤⣤⡤⠤⠤⠞⣠⡴⠶⠚⠋⠉⠁⠀⠀⠀⠀⠀⠀
⠛⠒⠛⠉⠉⠀⠀⠀⣴⠟⢃⡴⠛⠋⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠛⠛⠋⠀⠀

EOF

}

run_all_stages() {
    codegen_task_from_pddl
    vecgen_record
    if [[ "$USE_WANDB" == "true" ]]; then
        save_to_wandb
    fi

    success_rate=$(jq ".success_rate" "${DATA_DIR}/results.json")
    total_policies=$(jq ".total" "${DATA_DIR}/results.json")
    echo "[INFO]: ${DATA_DIR} total=${total_policies} success_rate=${success_rate}"

    # Both halves of the gate exit non-zero, not just sad: this is the only
    # quality gate -- CI chains ./tests/eval/scene_load.sh after this script, and
    # verifying the scene of a 10%-success datagen run is verifying almost
    # nothing. Note this aborts any remaining --task, by design.
    if (( total_policies < MIN_POLICIES )); then
        show_sad_pusheen
        echo "[ERROR]: only ${total_policies} episodes finished (< ${MIN_POLICIES}) for ${DATA_DIR}; the success rate is meaningless at this sample size"
        exit 1
    fi

    if (( $(echo "$success_rate < ${SUCCESS_THRESHOLD}" | bc -l) )); then
        show_sad_pusheen
        echo "[ERROR]: success_rate ${success_rate} < ${SUCCESS_THRESHOLD} for ${DATA_DIR}"
        exit 1
    else
        show_happy_pusheen
    fi
}

if [[ "${TASK}" == "all" ]]; then
    TASKS=(flip_pnp rotate)
else
    TASKS=("${TASK}")
fi

for t in "${TASKS[@]}"; do
    select_task "${t}"
    echo "=== task=${t} slug=${TASK_SLUG} data_dir=${DATA_DIR} mode=${MODE} ==="
    case "${MODE}" in
        generate-task) codegen_task_from_pddl ;;
        vecgen) vecgen_record ;;
        save-to-wandb) save_to_wandb ;;
        all) run_all_stages ;;
        *) echo "Unkown 'mode' option ${MODE}. Use [generate-task|vecgen|save-to-wandb|all]"; exit 1 ;;
    esac
done
