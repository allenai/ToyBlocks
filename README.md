<p align="center">
  <img src="ToyBlocksLogo_letest.svg" alt="ToyBlocks" width="720">
</p>

# blocksuite

This repo contains all code for creating toy block domain tasks and (maybe?) RL training

All docs / example commands / workflows can be found at https://allenai.github.io/blocksuite

If you need to add details about your work (e.g. new rl script, new datagen script, new details on evals etc.) please edit the markdown files in docs/source

## Setup

This guide uses `uv` to set everything up. The setup instructions here are for setting up the evaluation environments and data generation systems.

Prerequisites: an NVIDIA driver that supports CUDA 12.8 (570+) and [`uv`](https://docs.astral.sh/uv/getting-started/installation/).
ToyBlocks and its `curobo` and `molmo2` submodules are public; the `maniflow-tb` submodule is currently
internal to Ai2.

```bash
git clone https://github.com/allenai/ToyBlocks.git
cd ToyBlocks

# Create a virtual environment for py311 (required for isaacsim 5.1) and activate it
uv venv --python 3.11
source .venv/bin/activate

# Grab the submodules: curobo, plus maniflow-tb and molmo2 (without maniflow, `--init curobo` is enough)
git submodule update --init --recursive

# Just install like any python package (don't do uv sync, it's not recommended)
uv pip install -e ".[dev,isaac,torch-cu128]"

# If you want to install with maniflow (maniflow-tb + molmo2)
uv pip install -e ".[dev,isaac,torch-cu128,maniflow]"

# Headless / scripted Isaac Sim runs can't answer the EULA prompt
export OMNI_KIT_ACCEPT_EULA=YES
```

### Datagen (curobo) setup

Datagen (vecgen) needs cuRobo v2 (the `curobo` submodule): install the `vecgen` optional group (or
`tamp`, which adds the rerun / flask debug viewers). The submodule must be initialized first
(`git submodule update --init curobo`; the `--recursive` command above already does it).
There is no TAMP submodule any more; the FF planner is vendored in `src/blocksuite/vecgen/symbolic/ff_planner/`.

```bash
uv pip install -e ".[dev,isaac,torch-cu128,vecgen]"

# or datagen + maniflow
uv pip install -e ".[dev,isaac,torch-cu128,vecgen,maniflow]"
```

`uv` builds `curobo` from the submodule itself (it is registered in `pyproject.toml` as an editable
source), so no separate `uv pip install -e curobo` step is needed.

No local CUDA toolkit is needed: cuRobo v2 compiles its kernels at runtime with NVRTC from the
`nvidia-*-cu12` pip packages installed with torch. The `cuda-12.8` toolkit install and the
`setup_cuda` shell function that cuRobo v1 needed can be dropped.

:warning: **Docker:** `docker/vecgen.dockerfile` still installs `curobo` as a separate step after
the main install instead of passing the `vecgen` group (a one-step install failed inside `docker build`).

### Download the assets

The assets can be downloaded from `R2` using the following command, with the venv active (it caches
them in `~/.toyblocks` and symlinks them into the `assets` folder at the root of the repo)

```bash
get-toyblocks-assets --asset-manifest asset_manifest.json
```

This also installs the CC0 Poly Haven HDRI pool (1.8 GB, `assets/environments/polyhaven`) that
re-rendering uses for backdrops. The second pool the default render flags use, the Laval Indoor
HDR Dataset (`indoor-2018`), is licensed for research use and has to be requested from Université
Laval. See
[docs/rerender/README_local.md](docs/rerender/README_local.md#hdri-backdrops) for how to install it.

## Code Organization

All source code in `src/blocksuite/`

- `src/blocksuite/tasks`: all hand created tasks as well as templates for PDDL based task generation
- `src/blocksuite/vecgen`: the datagen engine (FF planning, batched skills, cuRobo planner pool, LeRobot recording)
- `src/blocksuite/robots`: all robot assets and configurations, including controllers
- `src/blocksuite/serving`: policy serving infrastructure (`BasePolicy`, `PolicyServer`, WebSocket client, obs transforms)
- `src/blocksuite/wrappers`: gym `Wrapper` subclasses (recording, stats, obs flattening)
- `src/blocksuite/cli`: command line utilities
- `src/blocksuite/data_annotation`: dataset annotation utilities
- `src/blocksuite/utils`: utilities for blocksuite
- `src/blocksuite/utils/rigid_objects`: utilities for working with rigid objects (e.g. creating them more easily)

The ManiFlow and Molmo2 policy code is in the `maniflow-tb/` and `molmo2/` submodules

## Docker images

Both images build from a fresh SSH clone of a ref (push your branch first) and are tagged with its short SHA:

```bash
# eval image: Isaac + cuRobo + maniflow-tb + molmo2 (policy servers and evaluate.py)
docker/build_eval_image.sh release /tmp/eval_img        # -> blocksuite-eval:<sha>

# slim datagen image: vecgen + cuRobo v2, no policies, no CUDA toolkit
docker/build_vecgen_image.sh release /tmp/vecgen_img    # -> blocksuite-vecgen:<sha>
```
