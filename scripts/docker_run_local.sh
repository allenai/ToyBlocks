#!/usr/bin/env bash

set -e

DOCKER_IMAGE=toyblocks-dev-maniflow-tamp:latest
SRC_FOLDER=""

command -v fzf >/dev/null 2>&1 && HAS_FZF=true || HAS_FZF=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) SHOW_HELP=true; shift ;;
        --docker-image) DOCKER_IMAGE="$2"; shift 2 ;;
        --source-folder) SRC_FOLDER="$2"; shift 2 ;;
        *) echo "Unkown option: $1"; exit 1 ;;
    esac
done

# TODO: get the ones that are built locally using the docker cli
image_choices=(
    "toyblocks-main-base:latest"
    "toyblocks-main-maniflow:latest"
    "toyblocks-dev-base:latest"
    "toyblocks-dev-maniflow:latest"
)

if [ "${HAS_FZF}" == "true" ]; then
    DOCKER_IMAGE=$(printf "%s\n" "${image_choices[@]}" | fzf --header="Select docker image:" --layout=reverse)
fi

REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
cd "${REPO_ROOT}"

SRC_FOLDER_CMD=""
if [ -n "${SRC_FOLDER}" ]; then
    SRC_FOLDER_CMD="-v ${SRC_FOLDER}:/root/toyblocks:rw"
    echo "[INFO]: mounting source folder @ ${SRC_FOLDER_CMD}"
fi

docker run -it --rm \
    --gpus all --runtime=nvidia \
    -v ~/.gitconfig:/root/.gitconfig:ro \
    -v ~/.ssh/id_ed25519:/root/.ssh/id_ed25519 \
    -v $(pwd)/datasets:/root/toyblocks/datasets:rw \
    ${SRC_FOLDER_CMD} \
    ${DOCKER_IMAGE}
