# Workflows

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`download;1.2em;sd-text-primary` 1 · [Installation and Setup](installation_and_setup_instructions.md)

- [Prerequisites](installation_and_setup_instructions.md#prerequisites)
- [Setup](installation_and_setup_instructions.md#setup)
- [Download the assets](installation_and_setup_instructions.md#download-the-assets)
- [Verify the installation](installation_and_setup_instructions.md#verify-the-installation)
- [Docker images](installation_and_setup_instructions.md#docker-images)
:::

:::{grid-item-card} {octicon}`database;1.2em;sd-text-primary` 2 · [Automatic Data Generation](data_generation.md)

- [Running Pipeline Locally](datagen/local.md)
- Stages: [1 Tasks](datagen/task_generation.md) · [2 Environments](datagen/env_generation.md) ·
  [3 Execution](datagen/execution.md) · [4 Rendering](datagen/render.md) ·
  [5 Merge](datagen/merge.md) · [6 Annotation](datagen/annotate.md)
- [Recorded Dataset Fields](datagen/dataset_fields.md)
:::

:::{grid-item-card} {octicon}`mortar-board;1.2em;sd-text-primary` 3 · [Training ManiFlow Policies](imitation_learning.md)

- [Training on a generated dataset](imitation_learning.md#training-on-a-generated-dataset)
- [Choosing GPUs and nodes](imitation_learning.md#choosing-gpus-and-nodes)
- [Resuming and fine-tuning](imitation_learning.md#resuming-and-fine-tuning)
- [`train_maniflow.py` Reference](imitation_learning/train_maniflow.md)
- [Setting up a new policy server](imitation_learning/new_policy_server.md)
:::

:::{grid-item-card} {octicon}`graph;1.2em;sd-text-primary` 4 · [Policy Evaluation](evaluations.md)

- [Quick start with `launch_eval.py`](evaluations.md#quick-start-with-launch_evalpy)
- [Replaying Actions from a Dataset](evals/replay.md)
- [Eval Metrics](evals/metrics.md)
- [`launch_eval.py` Reference](evals/launch_eval.md) ·
  [`evaluate.py` Reference](evals/evaluate.md)
:::

:::{grid-item-card} {octicon}`hubot;1.2em;sd-text-primary` [Agentic Robotics](agentic_robotics.md)
:columns: 12

[Setup](agentic_robotics.md#setup) ·
[Run an episode](agentic_robotics.md#run-an-episode) ·
[Web UI](agentic_robotics.md#web-ui-the-episode-pages-and-the-portal) ·
[Create a new task](agentic_robotics.md#create-a-new-task) ·
[ManiFlow skills](agentic_robotics.md#maniflow-skills)
:::

::::

```{toctree}
:maxdepth: 2
:hidden:

installation_and_setup_instructions
data_generation
imitation_learning
evaluations
agentic_robotics
```
