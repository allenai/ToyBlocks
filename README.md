<p align="center">
  <img src="ToyBlocksLogo_latest.svg" alt="ToyBlocks" width="720">
</p>

# ToyBlocks

<h3 align="center">
  <a href="https://allenai.org/blog/toyblocks">Blog post</a> &nbsp;|&nbsp;
  <a href="https://allenai.github.io/ToyBlocks/docs/user_guide/index.html">Docs</a> &nbsp;|&nbsp;
  <a href="https://console.cloud.google.com/storage/browser/toyblocks-data">Data</a> &nbsp;|&nbsp;
  Tech report (coming soon)
</h3>

This repo contains ToyBlocks: an open simulation framework based on IsaacSim/IsaacLab, built around 
a robot arm manipulating blocks on a tabletop. The setting is deliberately simple, but the robot 
must still reason about how the blocks interact and how to arrange them while pushing, flipping, 
and stacking.

ToyBlocks automatically and scalably generates demonstration data for policy training, without requiring 
time-consuming and expensive teleoperation. Researchers can easily vary these training experiences. 
For example, a team could generate more pushing or flipping demonstrations, then evaluate performance 
on tasks kept out of training.

The generated datasets are recorded in the LeRobot format and can be used directly to train VLAs 
or other policies. ToyBlocks ships with training infrastructure for [ManiFlow](https://maniflow-policy.github.io/), a low-level visuomotor 
policy conditioned on language and on a target heatmap, and serves it either on its own or under a 
high-level policy (such as [Molmo2](https://allenai.org/blog/molmo2), Astra, Claude) that decides what to do next. 

Policies are evaluated in closed loop in simulation, either on freshly generated unseen scenes 
and tasks or on the exact initial states of episodes recorded in a dataset. Evaluation is GPU-parallel. 
Because every task's goal is expressed in PDDL, ToyBlocks automatically measures success and partial progress, 
with no task-specific metric code. Either level of a hierarchical policy can be replaced by an oracle 
to isolate where failures come from, and the states where a policy failed can be sent back to data generation 
to plan new expert demonstrations and recoveries from them.

The overall goal of ToyBlocks is to provide a controlled world for robot learning and evaluation. 
By providing full control over the data generation process, ToyBlocks provides tools for researchers 
to trace the outcomes of trained policies back to the training data.


## Setup

This guide uses `uv` to set everything up. The setup instructions here are for setting up the evaluation environments and data generation systems.

Prerequisites: an NVIDIA driver that supports CUDA 12.8 (570+) and 
[`uv`](https://docs.astral.sh/uv/getting-started/installation/). ToyBlocks and 
its `curobo` and `molmo2` submodules are public; the `maniflow-tb` submodule is currently internal to Ai2.

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
There is no TAMP submodule any more; the FF planner is vendored in `src/toyblocks/vecgen/symbolic/ff_planner/`.

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

All source code in `src/toyblocks/`

- `src/toyblocks/tasks`: all hand created tasks as well as templates for PDDL based task generation
- `src/toyblocks/vecgen`: the datagen engine (FF planning, batched skills, cuRobo planner pool, LeRobot recording)
- `src/toyblocks/robots`: all robot assets and configurations, including controllers
- `src/toyblocks/serving`: policy serving infrastructure (`BasePolicy`, `PolicyServer`, WebSocket client, obs transforms)
- `src/toyblocks/wrappers`: gym `Wrapper` subclasses (recording, stats, obs flattening)
- `src/toyblocks/cli`: command line utilities
- `src/toyblocks/data_annotation`: dataset annotation utilities
- `src/toyblocks/utils`: utilities for ToyBlocks
- `src/toyblocks/utils/rigid_objects`: utilities for working with rigid objects (e.g. creating them more easily)

The ManiFlow and Molmo2 policy code is in the `maniflow-tb/` and `molmo2/` submodules

## Docker images

Both images build from a fresh SSH clone of a ref (push your branch first) and are tagged with its short SHA:

```bash
# eval image: Isaac + cuRobo + maniflow-tb + molmo2 (policy servers and evaluate.py)
docker/build_eval_image.sh main /tmp/eval_img        # -> toyblocks-eval:<sha>

# slim datagen image: vecgen + cuRobo v2, no policies, no CUDA toolkit
docker/build_vecgen_image.sh main /tmp/vecgen_img    # -> toyblocks-vecgen:<sha>
```
