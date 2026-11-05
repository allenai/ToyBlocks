#!/bin/bash
# Build the eval image (docker/eval.dockerfile) from a fresh blobless clone of ToyBlocks at REF
# with the curobo, maniflow-tb and molmo2 submodules.
#
#   docker/build_eval_image.sh <ref> <work-dir>
#   e.g. docker/build_eval_image.sh main /tmp/eval_img
set -euo pipefail
REF=${1:?ref (branch or commit)}
WORK=${2:?work dir for the build context}
HERE=$(cd "$(dirname "$0")" && pwd)

rm -rf "$WORK/toyblocks"
mkdir -p "$WORK"
git clone -q --filter=blob:none --no-checkout https://github.com/allenai/ToyBlocks.git "$WORK/toyblocks"
cd "$WORK/toyblocks"
git checkout -q "$REF"
git submodule update --init --filter=blob:none curobo maniflow-tb molmo2
SHA=$(git rev-parse --short HEAD)
TAG="toyblocks-eval:$SHA"

DOCKER_BUILDKIT=1 docker build -f "$HERE/eval.dockerfile" -t "$TAG" .
docker image ls "$TAG"
