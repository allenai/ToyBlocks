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

## How to get started

The [docs](https://allenai.github.io/ToyBlocks/docs/user_guide/index.html) walk through each workflow:

- [Installation and Setup](https://allenai.github.io/ToyBlocks/docs/user_guide/workflows/installation_and_setup_instructions.html)
- [Automatic Data Generation](https://allenai.github.io/ToyBlocks/docs/user_guide/workflows/data_generation.html)
- [Training ManiFlow Policies on Automatically Generated Data](https://allenai.github.io/ToyBlocks/docs/user_guide/workflows/imitation_learning.html)
- [Policy Evaluation](https://allenai.github.io/ToyBlocks/docs/user_guide/workflows/evaluations.html)


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

## License

ToyBlocks is licensed under the [Apache License 2.0](LICENSE).
The `curobo`, `maniflow-tb` and `molmo2` submodules and the downloaded assets (e.g. the HDRI pools) are
distributed under their own licenses.
