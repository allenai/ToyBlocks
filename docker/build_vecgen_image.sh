#!/bin/bash
# Build the vecgen image (docker/vecgen.dockerfile) from a fresh blobless clone of ToyBlocks at
# REF with only the curobo submodule (no TAMP).
#
#   docker/build_vecgen_image.sh <ref> <work-dir>
#   e.g. docker/build_vecgen_image.sh main /tmp/vecgen_img
set -euo pipefail
REF=${1:?ref (branch or commit)}
WORK=${2:?work dir for the build context}
HERE=$(cd "$(dirname "$0")" && pwd)

rm -rf "$WORK/toyblocks"
mkdir -p "$WORK"
git clone -q --filter=blob:none --no-checkout https://github.com/allenai/ToyBlocks.git "$WORK/toyblocks"
cd "$WORK/toyblocks"
git checkout -q "$REF"
git submodule update --init --filter=blob:none curobo
SHA=$(git rev-parse --short HEAD)
TAG="toyblocks-vecgen:$SHA"

DOCKER_BUILDKIT=1 docker build -f "$HERE/vecgen.dockerfile" -t "$TAG" .
docker image ls "$TAG"
