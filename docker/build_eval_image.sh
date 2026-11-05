#!/bin/bash
# Build the eval image (docker/eval.dockerfile) from a fresh blobless clone of blocksuite at REF
# with the curobo, maniflow-tb and molmo2 submodules.
#
#   docker/build_eval_image.sh <ref> <work-dir>
#   e.g. docker/build_eval_image.sh release /tmp/eval_img
set -euo pipefail
REF=${1:?ref (branch or commit)}
WORK=${2:?work dir for the build context}
HERE=$(cd "$(dirname "$0")" && pwd)

rm -rf "$WORK/blocksuite"
mkdir -p "$WORK"
git clone -q --filter=blob:none --no-checkout git@github.com:allenai/blocksuite.git "$WORK/blocksuite"
cd "$WORK/blocksuite"
git checkout -q "$REF"
git submodule update --init --filter=blob:none curobo maniflow-tb molmo2
SHA=$(git rev-parse --short HEAD)
TAG="blocksuite-eval:$SHA"

DOCKER_BUILDKIT=1 docker build -f "$HERE/eval.dockerfile" -t "$TAG" .
docker image ls "$TAG"
