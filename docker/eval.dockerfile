# Eval image: the vecgen image (docker/vecgen.dockerfile: slim ubuntu, cuRobo v2 with no CUDA
# toolkit) plus what the queue-driven eval needs (scripts/launch_eval.py ->
# scripts/eval/run_eval_pipeline.sh):
#   - the maniflow extra: molmo2 (Molmo2 HL) and maniflow-tb (ManiFlow LL), editable;
#   - an ffmpeg CLI (expert clip extraction).
#
# Build with docker/build_eval_image.sh: the context is a fresh blobless clone of ToyBlocks at
# the requested ref with the curobo, maniflow-tb and molmo2 submodules. One
# `uv pip install` of every extra resolves the same torch / CUDA / Isaac versions as the vecgen
# image. The repo is at /root/toyblocks and the venv at /root/.venv.

FROM ubuntu:24.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -q \
    && apt-get install -y --no-install-recommends ca-certificates curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN UV_INSTALL_DIR=/usr/local/bin sh /uv-installer.sh && rm /uv-installer.sh

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    UV_NO_CACHE=1 \
    OMNI_KIT_ACCEPT_EULA=YES
RUN uv venv /root/.venv --python 3.11
ENV VIRTUAL_ENV=/root/.venv \
    PATH="/root/.venv/bin:$PATH"

COPY . /root/toyblocks
WORKDIR /root/toyblocks

RUN uv pip install -e ".[dev,isaac,torch-cu128,vecgen,maniflow]" --config-settings editable_mode=strict \
    && get-toyblocks-assets --asset-manifest asset_manifest.json \
    && python -c "import curobo.scene, warp, lerobot, hvla, olmo; print('imports ok; warp', warp.__version__)"

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# As docker/vecgen.dockerfile (git/ssh for the job's checkout; the X11 / GL / EGL / Vulkan client
# libraries Isaac Sim loads; FFmpeg 6 shared libraries for torchcodec), plus the ffmpeg CLI.
RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git openssh-client jq ffmpeg \
        libgl1 libegl1 libglu1-mesa libvulkan1 libglib2.0-0t64 libgomp1 \
        libx11-6 libxext6 libxrender1 libxt6t64 libxrandr2 libxinerama1 libxcursor1 libxi6 \
        libsm6 libice6 libxkbcommon0 \
        libavcodec60 libavformat60 libavutil58 libavdevice60 libavfilter9 libswscale7 \
        libswresample4 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/
COPY --from=build /opt/uv-python /opt/uv-python
COPY --from=build /root/.venv /root/.venv
COPY --from=build /root/toyblocks /root/toyblocks

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_LINK_MODE=copy \
    VIRTUAL_ENV=/root/.venv \
    PATH="/root/.venv/bin:$PATH" \
    OMNI_KIT_ACCEPT_EULA=YES \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=all
WORKDIR /root/toyblocks
