#!/bin/bash
# Build the vecgen image (docker/vecgen.dockerfile) from a fresh blobless clone of blocksuite at
# REF with only the curobo submodule (no TAMP).
#
#   docker/build_vecgen_image.sh <ref> <work-dir>
#   e.g. docker/build_vecgen_image.sh release /tmp/vecgen_img
set -euo pipefail
REF=${1:?ref (branch or commit)}
WORK=${2:?work dir for the build context}
HERE=$(cd "$(dirname "$0")" && pwd)

rm -rf "$WORK/blocksuite"
mkdir -p "$WORK"
git clone -q --filter=blob:none --no-checkout git@github.com:allenai/blocksuite.git "$WORK/blocksuite"
cd "$WORK/blocksuite"
git checkout -q "$REF"
git submodule update --init --filter=blob:none curobo
SHA=$(git rev-parse --short HEAD)
TAG="blocksuite-vecgen:$SHA"

DOCKER_BUILDKIT=1 docker build -f "$HERE/vecgen.dockerfile" -t "$TAG" .
docker image ls "$TAG"
