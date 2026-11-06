# ToyBlocks Docs

Documentation on the ToyBlocks project and its toy-blocks domain.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.2em;sd-text-primary` Installation and Setup
:link: user_guide/workflows/installation_and_setup_instructions
:link-type: doc

Set up ToyBlocks, its submodules and the assets with `uv`, then verify the installation.
:::

:::{grid-item-card} {octicon}`database;1.2em;sd-text-primary` Automatic Data Generation
:link: user_guide/workflows/data_generation
:link-type: doc

Turn a symbolic PDDL task description into a large, automatically-labelled dataset of robot
demonstrations in simulation.
:::

:::{grid-item-card} {octicon}`mortar-board;1.2em;sd-text-primary` Training ManiFlow Policies
:link: user_guide/workflows/imitation_learning
:link-type: doc

Train ManiFlow policies on a merged LeRobot dataset with `scripts/train_maniflow.py`.
:::

:::{grid-item-card} {octicon}`graph;1.2em;sd-text-primary` Policy Evaluation
:link: user_guide/workflows/evaluations
:link-type: doc

Measure how well a trained policy performs in closed loop in simulation.
:::

:::{grid-item-card} {octicon}`hubot;1.2em;sd-text-primary` Agentic Robotics
:link: user_guide/workflows/agentic_robotics
:link-type: doc

Put a frontier vision-language model in charge of the robot, acting through MCP skill tools.
:::

::::

```{toctree}
:maxdepth: 4
:caption: Contents:
:hidden:

user_guide/index
```
