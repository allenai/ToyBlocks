# ToyBlocks Installation and Setup Instructions

These instructions use `uv` and set up everything needed for evaluation, data generation and training.

## Prerequisites

- Linux with an NVIDIA GPU and a driver that supports CUDA 12.8 (driver 570 or newer).
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) on your `PATH` (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- Git access to GitHub over HTTPS. ToyBlocks and all of its submodules (`curobo`, `maniflow-tb`, `molmo2` and `agentic_robotics`) are public.
- `tmux` (`sudo apt install tmux`): long-running jobs (pipeline runs, merges, renders) are run in detached `tmux` sessions so they survive a dropped connection.

## Setup

```bash
git clone https://github.com/allenai/ToyBlocks.git
cd ToyBlocks

# Create a virtual environment for Python 3.11 (required for Isaac Sim 5.1) and activate it
uv venv --python 3.11
source .venv/bin/activate

# Fetch the submodules (curobo, plus maniflow-tb and molmo2 for the policies and agentic_robotics
# for LLM agents)
git submodule update --init --recursive
# Without ManiFlow, only curobo is needed:
# git submodule update --init curobo

# Install like any Python package (uv sync is not recommended)
uv pip install -e ".[dev,isaac,torch-cu128]"

# Or, to also install the policies (the maniflow extra: maniflow-tb and molmo2, editable)
uv pip install -e ".[dev,isaac,torch-cu128,maniflow]"
```

Isaac Sim asks you to accept the NVIDIA Omniverse EULA the first time it starts. Headless and scripted runs have no prompt to answer and abort with `EOF when reading a line`, so accept it up front:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
```

## Datagen (cuRobo) setup

Data generation (vecgen) needs cuRobo v2, which comes from the `curobo` submodule. Install the `vecgen` optional group, or `tamp`, which adds the rerun and flask debug viewers on top. The submodule must be initialized first (`git submodule update --init curobo`); the `--recursive` command above already does this. There is no TAMP submodule any more: the FF symbolic planner is vendored in `src/toyblocks/vecgen/symbolic/ff_planner/`.

```bash
uv pip install -e ".[dev,isaac,torch-cu128,vecgen]"

# Or datagen + ManiFlow
uv pip install -e ".[dev,isaac,torch-cu128,vecgen,maniflow]"
```

`uv` builds `curobo` from the submodule itself (it is registered in `pyproject.toml` as an editable source), so no separate `uv pip install -e curobo` step is needed.

No local CUDA toolkit is needed: cuRobo v2 compiles its kernels at runtime with NVRTC, using the `nvidia-*-cu12` pip packages installed with torch. The `cuda-12.8` toolkit install and the `setup_cuda` shell function that cuRobo v1 needed are no longer required.

:::{note}
**Docker:** `docker/vecgen.dockerfile` still installs `curobo` as a separate step after the main install instead of passing the `vecgen` group, because a one-step install failed inside `docker build`. `docker/eval.dockerfile` installs every extra, `vecgen` included, in one step.
:::

## Agentic robotics setup

The `agentic_robotics` submodule runs LLM agents that drive the robot through the skills as MCP
tools. It needs the `vecgen` extra for its scripted skills, plus the `agentic` extra:

```bash
git submodule update --init agentic_robotics
uv pip install -e ".[dev,isaac,torch-cu128,vecgen,agentic]"
```

Episodes call the model through OpenRouter, so you need your own OpenRouter API key. See
[Agentic Robotics](agentic_robotics.md) for the key, the web UI and adding tasks.

## Download the assets

The assets are downloaded from a public R2 bucket (about 280 MB, no credentials needed) into a cache at `~/.toyblocks`, and symlinked into the `assets` folder at the root of the repository. The cache is shared by every checkout on the machine, so the disk usage lands in your home directory and deleting `assets/` frees nothing. Run it with the virtual environment active, since the command is installed into `.venv/bin`:

```bash
get-toyblocks-assets --asset-manifest asset_manifest.json
```

This also installs the CC0 Poly Haven HDRI pool (1.8 GB, `assets/environments/polyhaven`) that rendering uses for backdrops. The second pool the default render flags use, the Laval Indoor HDR Dataset (`indoor-2018`), is licensed for research use and has to be requested from Université Laval. See [HDRI backdrops](https://github.com/allenai/ToyBlocks/blob/main/docs/source/user_guide/workflows/datagen/local.md#hdri-backdrops) in `docs/source/user_guide/workflows/datagen/local.md` for how to install it.

## Verify the installation

Run the installation checks from the repository root, with the virtual environment active:

```bash
pytest tests/install                    # seconds: packages, versions, submodules, assets
export OMNI_KIT_ACCEPT_EULA=YES
pytest tests/install --install-gpu      # several minutes: also Isaac Sim, cuRobo and vecgen
```

The first command checks each extra you installed and skips the others. For every installed extra, it checks that the packages are installed in the versions `pyproject.toml` allows and that what its workflows import imports. It also checks that the submodules the extra needs are checked out at the pinned commits and are the ones Python imports. Finally, it checks that the assets are downloaded. A failure names what is wrong and the command that fixes it.

`--install-gpu` also checks that PyTorch sees the GPU and that Isaac Sim starts headless with the right Warp. With the `vecgen` extra, it runs the vecgen check below, with and without videos. With the `agentic` extra, it imports the agentic skill servers.

To run the vecgen check by hand: with the `vecgen` group installed, generate one environment package from a checked-in problem and record a few episodes. This exercises Isaac Sim, cuRobo and LeRobot recording, and should end with a `RESULT episodes=4 success=...` line:

```bash
export OMNI_KIT_ACCEPT_EULA=YES
problem=src/toyblocks/tamp/blocks/problems/flip.pddl
python scripts/generate_task_from_pddl.py --domain_path src/toyblocks/tamp/blocks \
    --pddl ${problem} --num_reset_states 8 --num_envs 4 --seed 0 --output_dir outputs --headless
python scripts/vecgen/run.py --env-path outputs/flip --pddl ${problem} \
    --domain src/toyblocks/tamp/blocks/domain.pddl --num-envs 4 --episodes 4 \
    --record datasets/flip/flip --mark-complete --headless
```

That run records state and actions only. To also record camera videos (RGB and depth from the front and gripper cameras, 640x360), add `--record-videos --enable_cameras`: `--record-videos` adds the cameras to the scene, and `--enable_cameras` lets Isaac Sim render them. Each simulation step takes about 1.5 times as long:

```bash
python scripts/vecgen/run.py --env-path outputs/flip --pddl ${problem} \
    --domain src/toyblocks/tamp/blocks/domain.pddl --num-envs 4 --episodes 4 \
    --record datasets/flip/flip_videos --record-videos --enable_cameras --mark-complete --headless
```

The videos land in `datasets/flip/flip_videos/blocks_success/videos/` (and `blocks_failure/videos/` for failed episodes), one folder per camera stream.

A full pipeline run ([Running the pipeline locally](datagen/local.md)) also leaves two things worth checking: a plot of a sample episode, `episode_0.png`, next to the merged dataset, and the full run configuration, `run_datagen_pipeline_config_<timestamp>.json`, in the run's base directory.

## Build the docs

These docs are built with Sphinx and MyST from `docs/source`:

```bash
uv pip install -e ".[docs]"
sphinx-build docs/source docs/build/html
```

Then open `docs/build/html/index.html`.

## Docker images

Both images build from a fresh clone of a git ref (so push your branch first) and are tagged with its short SHA:

```bash
# eval image: Isaac + cuRobo + maniflow-tb + molmo2 (policy servers and evaluate.py)
docker/build_eval_image.sh main /tmp/eval_img        # -> toyblocks-eval:<sha>

# slim datagen image: vecgen + cuRobo v2, no policies, no CUDA toolkit
docker/build_vecgen_image.sh main /tmp/vecgen_img    # -> toyblocks-vecgen:<sha>
```

:::{note}
**Hardware:** the images need an x86_64 Linux host with an NVIDIA GPU (see [Prerequisites](#prerequisites)); they do not build on arm64 such as Apple Silicon, since the pinned CUDA torch has no arm64 wheels. Running them needs the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Tested on an NVIDIA RTX 5090 (32 GB).
:::

## Code Organization

All source code is in `src/toyblocks/`:

- `src/toyblocks/tasks`: all hand-created tasks, as well as templates for PDDL-based task generation
- `src/toyblocks/vecgen`: the datagen engine (FF planning, batched skills, cuRobo planner pool, LeRobot recording)
- `src/toyblocks/robots`: all robot assets and configurations, including controllers
- `src/toyblocks/serving`: policy serving infrastructure (`BasePolicy`, `PolicyServer`, WebSocket client, obs transforms)
- `src/toyblocks/eval`: evaluation of learned models (with or without oracles) and replay
- `src/toyblocks/wrappers`: gym `Wrapper` subclasses (recording, stats, obs flattening)
- `src/toyblocks/cli`: command line utilities
- `src/toyblocks/data_annotation`: dataset annotation utilities
- `src/toyblocks/utils`: utilities for ToyBlocks
- `src/toyblocks/utils/rigid_objects`: utilities for working with rigid objects (e.g. creating them more easily)

The ManiFlow and Molmo2 policy code is in the `maniflow-tb/` (package `hvla`) and `molmo2/` (package `olmo`) submodules.

## Next steps

- [Automatic Data Generation](data_generation.md): generate demonstration datasets.
- [Pre-training ManiFlow Policies](imitation_learning.md): train a policy on them.
- [Policy Evaluation](evaluations.md): evaluate it.
- [Agentic Robotics](agentic_robotics.md): let an LLM agent drive the robot through the skills.
