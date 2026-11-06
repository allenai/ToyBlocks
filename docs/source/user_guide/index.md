![ToyBlocks](../_static/ToyBlocksLogo_latest.svg)

# Workflows

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.2em;sd-text-primary` 1 · [Installation and Setup](workflows/installation_and_setup_instructions.md)

- [Prerequisites](workflows/installation_and_setup_instructions.md#prerequisites)
- [Setup](workflows/installation_and_setup_instructions.md#setup)
- [Download the assets](workflows/installation_and_setup_instructions.md#download-the-assets)
- [Verify the installation](workflows/installation_and_setup_instructions.md#verify-the-installation)
- [Docker images](workflows/installation_and_setup_instructions.md#docker-images)
:::

:::{grid-item-card} {octicon}`database;1.2em;sd-text-primary` 2 · [Automatic Data Generation](workflows/data_generation.md)

- [Running Pipeline Locally](workflows/datagen/local.md)
- Stages: [1 Tasks](workflows/datagen/task_generation.md) · [2 Environments](workflows/datagen/env_generation.md) ·
  [3 Execution](workflows/datagen/execution.md) · [4 Rendering](workflows/datagen/render.md) ·
  [5 Merge](workflows/datagen/merge.md) · [6 Annotation](workflows/datagen/annotate.md)
- [Recorded Dataset Fields](workflows/datagen/dataset_fields.md)
:::

:::{grid-item-card} {octicon}`mortar-board;1.2em;sd-text-primary` 3 · [Training ManiFlow Policies](workflows/imitation_learning.md)

- [Training on a generated dataset](workflows/imitation_learning.md#training-on-a-generated-dataset)
- [Choosing GPUs and nodes](workflows/imitation_learning.md#choosing-gpus-and-nodes)
- [Resuming and fine-tuning](workflows/imitation_learning.md#resuming-and-fine-tuning)
- [`train_maniflow.py` Reference](workflows/imitation_learning/train_maniflow.md)
- [Setting up a new policy server](workflows/imitation_learning/new_policy_server.md)
:::

:::{grid-item-card} {octicon}`graph;1.2em;sd-text-primary` 4 · [Policy Evaluation](workflows/evaluations.md)

- [Quick start with `launch_eval.py`](workflows/evaluations.md#quick-start-with-launch_evalpy)
- [Replaying Actions from a Dataset](workflows/evals/replay.md)
- [Eval Metrics](workflows/evals/metrics.md)
- [`launch_eval.py` Reference](workflows/evals/launch_eval.md) ·
  [`evaluate.py` Reference](workflows/evals/evaluate.md)
:::

:::{grid-item-card} {octicon}`hubot;1.2em;sd-text-primary` [Agentic Robotics](workflows/agentic_robotics.md)
:columns: 12

[Setup](workflows/agentic_robotics.md#setup) ·
[Run an episode](workflows/agentic_robotics.md#run-an-episode) ·
[Web UI](workflows/agentic_robotics.md#web-ui-the-episode-pages-and-the-portal) ·
[Create a new task](workflows/agentic_robotics.md#create-a-new-task) ·
[ManiFlow skills](workflows/agentic_robotics.md#maniflow-skills)
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

workflows/installation_and_setup_instructions
workflows/data_generation
workflows/imitation_learning
workflows/evaluations
workflows/agentic_robotics
```
