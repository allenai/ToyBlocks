#!/usr/bin/env bash

set -e

SHOW_HELP=false
NUM_ENVS=8192
NUM_MINIBATCHES=32
ROBOT_ID=fr3_robotiq_gripper_sysid
SHARE_URL=false
USE_DEBUGGER=false
SEED=42

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        --num-envs) NUM_ENVS="$2"; shift 2;;
        --num-minibatches) NUM_MINIBATCHES="$2"; shift 2;;
        --robot-id) ROBOT_ID="$2"; shift 2;;
        --share-url) SHARE_URL=true; shift ;;
        --debug) USE_DEBUGGER=true; shift ;;
        *) echo "Unkown option: $1"; exit 1 ;;
    esac
done

[ "$SHARE_URL" == true ] && SHARE_URL_CMD="--share-url" || SHARE_URL_CMD=""
[ "$USE_DEBUGGER" == true ] && POST_MORTEM_CMD="-mipdb" || POST_MORTEM_CMD=""

DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

RUN_NAME="${DATETIME}-isolate-${ROBOT_ID}-seed${SEED}"

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

python ${POST_MORTEM_CMD} scripts/ppo_fast.py \
    --headless \
    --exp-name ppo-Blocksuite-Flip-v0-0 \
    --task Blocksuite-Isolate-v0 \
    --num-envs ${NUM_ENVS} \
    --num_minibatches ${NUM_MINIBATCHES} \
    --update-epochs 8 \
    --eval-freq 64 \
    --gamma 0.95 \
    --num-steps 16 \
    --total_timesteps 2000000000 \
    --reward-scale 0.1 \
    --env-cfg.robot-id ${ROBOT_ID} \
    --env-cfg.table-height-range 0.0 \
    --env-cfg.randomize-friction false \
    --env-cfg.randomize-block-size false \
    --track \
    --seed ${SEED} \
    ${SHARE_URL_CMD} \
    --exp-name ${RUN_NAME} \
    --disable-viewer \
    --wandb-project-name Blocksuite-RL \
    --wandb-entity ai2-robotics

