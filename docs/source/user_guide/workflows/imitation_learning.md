# Pre-training ManiFlow Policies on Automatically Generated Data 

ManiFlow policies are trained with the `maniflow-tb` submodule
([allenai/maniflow-tb](https://github.com/allenai/maniflow-tb), Python package `hvla`) and
its local launcher, `scripts/train_maniflow.py`. The script turns its flags into a single
training command,

```text
accelerate launch --config_file hvla/config/deepseeds/deepspeed_zero2.yaml ... hvla/training/train_hvla.py \
    --config_yaml ./examples/Franka/train_maniflow_multimodal.yaml --framework.name ManiFlowMultiModal ...
```

and runs it on this machine: `accelerate launch` on all visible GPUs by default, or
`torchrun` across several nodes. It trains on a merged LeRobot dataset, the output of the
[data generation pipeline](data_generation.md). Every flag is listed in the
[`train_maniflow.py` Reference](imitation_learning/train_maniflow.md).

## Setup

These docs describe maniflow-tb commit `7b10edf` ("toyblocks release"), the commit the
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
python maniflow-tb/scripts/train_maniflow.py \
    --data_root_dir data/datasets/merged/train/oct06_stack3_1000x32 \
    --goal_action_filter 'pick,place on table,place,flip backward,flip forward,flip left,flip right,rotate ccw,rotate cw,done' \
    --rgb_224
```

Relative `--data_root_dir` and `--run_root_dir` paths are resolved against the directory you run
the script from; the training command itself runs inside `maniflow-tb/`.

- `--data_root_dir`: the merged LeRobot dataset (by default the pipeline writes it to
  `<output-root>/datasets/merged/train/<experiment-id>`). Pass several directories to co-train
  on them; `--dataset_ratio` sets their sampling ratio.
- `--goal_action_filter`: training samples only start on frames whose `goal.action` is in
  this list. With several datasets, give one `;`-separated group per dataset.
- `--rgb_224`: read the pre-resized `<camera>_224` videos instead of decoding the native
  640x360 ones. The run fails if a dataset has no 224 videos; the
  [re-render step](datagen/rerender.md) writes them.

Everything else keeps its default, which gives:

- absolute 9-D joint actions (7 joints, gripper, PD mode), predicting chunks of 50 actions
  and executing 25;
- a 9-D state (joint positions, gripper, PD mode);
- front and gripper RGB through a fine-tuned CLIP ViT-B/16, no depth;
- the target heatmap as cross-attention tokens through the shared RGB encoder (dropped
  with probability 0.1, sigma 0.8 patches);
- T5-small language conditioning on the action string only (`action_only`);
- one node with all visible GPUs, AdamW with DeepSpeed ZeRO-2, a batch of 128 per GPU,
  200k steps, a checkpoint every 10k steps and a 5% validation split.

The script prints the run ID, the run directory and the full command before it starts;
`--dry_run` prints them and exits. The run ID is built from the dataset name and the flags,
plus a timestamp, for example
`rgb224_oct06_stack3_1000x32_50_25_abs_token_shared_rgb_mask0.1_sigma0.8_lang_t5-small_action_only-11-05-09-30-00`.
`--user_prefix` puts a prefix in front of it, and `--run_id` replaces it.

## Choosing GPUs and nodes

`--num_gpus_per_node` defaults to the number of visible GPUs, so `CUDA_VISIBLE_DEVICES` picks
them. A short debug run on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python maniflow-tb/scripts/train_maniflow.py \
    --run_root_dir /tmp/maniflow_debug --duration 100 --save_interval 100 \
    --logging_frequency 10 --dbs 4 --num_workers 0 --max_episodes 500 \
    --data_root_dir data/datasets/merged/train/my_dataset --rgb_224
```

For several nodes, run the script on every node with the same flags, plus
`--num_nodes N --node_rank R --master_addr <address of node 0> --run_id <name>`. It then
launches `torchrun` instead of `accelerate launch`; every node must use the same `--run_id`
(or `--resume`) so that all of them write to the same run directory, which therefore has to be
on a filesystem the nodes share.

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
to start if that checkout is not the commit ToyBlocks pins (`git submodule update --init
maniflow-tb` fixes it). A checkpoint trained with this code is on Hugging Face,
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
