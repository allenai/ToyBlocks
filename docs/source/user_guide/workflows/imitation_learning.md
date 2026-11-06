# Training ManiFlow Policies on Automatically Generated Data

ManiFlow policies are trained with the `maniflow-tb` submodule
([allenai/maniflow-tb](https://github.com/allenai/maniflow-tb), Python package `hvla`) and
its local launcher, `scripts/train_maniflow.py`. The script turns its flags into a single
training command,

```text
accelerate launch --config_file hvla/config/deepseeds/deepspeed_zero2.yaml ... hvla/training/train_hvla.py \
    --config_yaml ./examples/Franka/train_maniflow_multimodal.yaml --framework.name ManiFlowMultiModal ...
```

and runs it locally: `accelerate launch` on all visible GPUs by default, or
`torchrun` across several nodes. It trains on a merged LeRobot dataset, the output of the
[data generation pipeline](data_generation.md). Every flag is listed in the
`train_maniflow.py` [Reference](imitation_learning/train_maniflow.md).

## Setup

These docs describe maniflow-tb commit `c04eab7`, the commit the
`maniflow-tb` submodule is pinned to. The `maniflow` extra installs it in editable mode, together
with `molmo2` (see [Installation and Setup](installation_and_setup_instructions.md)):

```bash
git submodule update --init maniflow-tb molmo2
uv pip install -e ".[dev,isaac,torch-cu128,maniflow]"
```

The dataloader decodes the RGB videos with TorchCodec, which needs the FFmpeg shared libraries
(FFmpeg 4 to 7) on the machine. `python -c "import torchcodec"` should succeed before you train.
maniflow-tb's own `maniflow-tb/INSTALL.md` describes a standalone conda environment instead,
including FFmpeg and the optional `muon-optimizer` package for `--optimizer muon`.

Training downloads pretrained weights from the Hugging Face Hub on first use (the timm RGB
backbone and `t5-small`); set `HF_HOME` to choose the cache location. Metrics go to W&B when
credentials are available (`WANDB_API_KEY` or a `wandb login`); otherwise W&B runs offline and
its logs stay in the run directory.

## Training on a generated dataset

```bash
python maniflow-tb/scripts/train_maniflow.py  --data_root_dir data/datasets/merged/train/my_dataset 
```

Relative `--data_root_dir` and `--run_root_dir` paths are resolved against the directory you run
the script from; the training command itself runs inside `maniflow-tb/`.

- `--data_root_dir`: the merged LeRobot dataset (by default the pipeline writes it to  
`<output-root>/datasets/merged/train/<experiment-id>`). Pass several directories to co-train  
on them; `--dataset_ratio` sets their sampling ratio.  

Everything else keeps its default, which gives:

- absolute 9-D joint actions (7 joints, gripper, Play mode), predicting chunks of 50 actions  
and executing 25;
- a 9-D state (joint positions, gripper, Play mode); The play mode is used my Maniflow to indicate that it has completed the task and request the next instruction.
- front and gripper RGB through a fine-tuned CLIP ViT-B/16. Depth is supported but off by default.
- the target heatmap as cross-attention tokens through the shared RGB encoder (dropped
with probability 0.1, sigma 0.8 patches);
- T5-small language conditioning on the action string only (`action_only`);
- one node with all visible GPUs, AdamW with DeepSpeed ZeRO-2, a batch of 128 per GPU,
200k steps, a checkpoint every 10k steps and a 5% validation split.

The script prints the run ID, the run directory and the full command before it starts;
`--dry_run` prints them and exits. The run ID is built from the dataset name and the flags,
plus a timestamp.
`--user_prefix` puts a prefix in front of it, and `--run_id` replaces it.

## Choosing GPUs and nodes

`--num_gpus_per_node` defaults to the number of visible GPUs, so `CUDA_VISIBLE_DEVICES` picks
them. A short debug run on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python maniflow-tb/scripts/train_maniflow.py \
    --run_root_dir /tmp/maniflow_debug --duration 100 --save_interval 100 \
    --logging_frequency 10 --dbs 4 --num_workers 0 --max_episodes 500 \
    --data_root_dir data/datasets/merged/train/my_dataset
```

For several nodes, run the script on every node with the same flags, plus
`--num_nodes N --node_rank R --master_addr <address of node 0> --run_id <name>`. It then
launches `torchrun` instead of `accelerate launch`; every node must use the same `--run_id`
(or `--resume`) so that all of them write to the same run directory, which therefore has to be
on a filesystem the nodes share.

## The last batch of an epoch

The train dataloader keeps the last, partial batch of every epoch, and ManiFlow splits every batch
into a flow part (`int(0.75 * B)` samples) and a consistency part (`int(0.25 * B)` samples). A last
batch of 1 to 3 samples leaves the consistency part empty, and training stops at that step with

```text
IndexError: list index out of range
```

raised from the T5 tokenizer (`encode_text_input_T5`), because it is handed an empty list of
instructions. The `train: <episodes> episodes, <N> samples` line that `[FrankaDataset]` prints
before training gives `N`; the last batch holds `N mod --dbs` samples (per GPU). Pick a `--dbs` for
which that is 0 or at least 4. For example, 242 samples crash at `--dbs 16` (last batch: 2) and
train at `--dbs 32` (last batch: 18). The debug run above, at `--dbs 4`, crashes whenever `N` is
not a multiple of 4.

`N` is a property of the dataset, not of the training flags, and it is not fixed for a given
datagen command. The pipeline seeds task generation (`--task-gen-seed`, default `6303`), reset
state sampling (a fixed `0` with `--vecgen-problems-per-env 1`, `--seed` plus the group offset
otherwise) and execution (`--seed`, default `42`), so the problems and start states repeat.
The executed trajectories are not guaranteed to: vecgen steps GPU physics and builds motion plans
on worker threads, and a skill that is retried from the live state (a flip that misses keeps its
failed attempt in the episode) changes the episode length. Rerunning the same command can
therefore change `N`, and a `--dbs` that worked before can stop working. Check the log line after
each regeneration.

## Resuming and fine-tuning

- `--resume <run_id>` continues a run in `--run_root_dir` from its latest complete checkpoint,
and fails if there is none. The architecture (actions, state, cameras, conditioning, depth)
comes from the run's saved `config.yaml`, so the flags that define it are not emitted; only
the training-loop flags still apply. `--data_root_dir` is not needed. The run must resume on
the same number of GPUs it was saved with, since the optimizer state is sharded per GPU.
- `--pretrained_checkpoint <file>` starts a new run from a checkpoint's weights. The
optimizer, scheduler, EMA and W&B state start fresh.

## Outputs

Each run writes to `<run_root_dir>/<run_id>/` (`--run_root_dir` defaults to `./runs`):

| Path | Contents |
|---|---|
| `config.yaml` | The full training config, which the policy server reads. |
| `stats.json` | Normalization statistics of the (merged) training data. |
| `checkpoints/steps_<N>_model.safetensors` | Weights every `--save_interval` steps. |
| `checkpoints/steps_<N>_training_state.pt`, `checkpoints/state_<N>/`, `checkpoints/steps_<N>.done` | Scheduler and EMA state, the per-GPU optimizer state (newest only) and the completion marker `--resume` looks for. |
| `final_model/` | `model.safetensors`, `config.yaml` and `config.json` at the end of training. |
| `summary.jsonl` | One line per saved checkpoint. |
| `wandb/`, `wandb_run_id.txt` | W&B logs (also the offline ones) and the W&B run ID. |

Metrics go to the W&B project `maniflow-tb` (`--wandb_project`, or `$WANDB_PROJECT`), in your
default entity unless `--wandb_entity` or `$WANDB_ENTITY` names another.

## Evaluating the checkpoint

Pass a checkpoint to [`launch_eval.py`](evals/launch_eval.md). It finds `config.yaml` in
the run directory, and the policy server reads the training data's normalization stats
from it:

```bash
python scripts/launch_eval.py --policy maniflow \
    --model-path runs/<run_id>/checkpoints/steps_200000_model.safetensors \
    --dataset-path /path/to/lerobot_dataset
```

The policy server imports `hvla` from the `maniflow-tb` submodule, and the eval pipeline refuses
to start if that checkout is not the commit ToyBlocks pins (`git submodule update --init maniflow-tb` fixes it). A checkpoint trained with this code is on Hugging Face,
[allenai/maniflow-tb](https://huggingface.co/allenai/maniflow-tb), with the same layout
(`hf download allenai/maniflow-tb --local-dir <dir>`). See [Policy Evaluation](evaluations.md)
for the rest of the eval workflow, and
[Setting up a new policy server](imitation_learning/new_policy_server.md) to evaluate a
policy that is not ManiFlow.

```{toctree}
:hidden:

imitation_learning/train_maniflow
imitation_learning/new_policy_server
```
