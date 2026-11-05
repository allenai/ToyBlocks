# Slim image for the vecgen datagen flow: task gen -> env packages (stage 1) -> vecgen
# (scripts/vecgen/run.py, optional camera videos) -> merge -> annotate, plus the pipeline driver.
#
# Build with docker/build_vecgen_image.sh: the context is a fresh blobless clone of ToyBlocks at
# the requested ref with only the curobo (v2) submodule. No TAMP (the vecgen flow is TAMP-free), no openpi / hvla /
# molmo2, no CUDA toolkit: cuRobo v2 JIT-compiles its kernels with NVRTC from the pip wheels,
# so it needs no rebuild at run time. Repo at /root/toyblocks, venv at /root/.venv.

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
# Pin every package to the tested vecgen venv (docker/vecgen.constraints.txt).
ENV UV_CONSTRAINT=/root/toyblocks/docker/vecgen.constraints.txt

# cuRobo v2 builds against the installed torch (no build isolation), once at build time.
RUN uv pip install -e ".[isaac,torch-cu128]" --config-settings editable_mode=strict \
    && uv pip install "setuptools<81" setuptools_scm ninja wheel "warp-lang==1.12.1" \
        av "lerobot==0.4.3" flask \
    && uv pip install --no-build-isolation -e "./curobo[cu12]" \
    && get-toyblocks-assets --asset-manifest asset_manifest.json \
    && python -c "import curobo.scene, warp, lerobot; print('imports ok; warp', warp.__version__)"

FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
# git/ssh for the job scripts' checkout; the X11 / GL / EGL / Vulkan client libraries Isaac Sim
# loads (the driver libraries come from the NVIDIA container runtime); the FFmpeg 6 shared
# libraries torchcodec loads to decode videos (LeRobot dataset validation).
RUN apt-get update -q \
    && apt-get install -y --no-install-recommends \
        ca-certificates curl git openssh-client jq \
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
