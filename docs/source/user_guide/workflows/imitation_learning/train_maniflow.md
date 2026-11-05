# `train_maniflow.py` Reference

Every flag of maniflow-tb's `scripts/train_maniflow.py` at commit `7b10edf`, the commit the
`maniflow-tb` submodule is pinned to (`python maniflow-tb/scripts/train_maniflow.py --help`
prints the same list). For the workflow (setup, an example run, outputs, evaluation) see
[Pre-training Policies on Automatically Generated Data](../imitation_learning.md).

The flags are spelled with underscores. Each one either sets a field of
`examples/Franka/train_maniflow_multimodal.yaml` on the `train_hvla.py` command line or
configures the launcher. With `--resume`, the architecture flags (marked "fresh runs
only") are not emitted: the run's saved `config.yaml` defines them.

## Launcher and logging

| Flag | Default | Description |
|---|---|---|
| `--dry_run` | off | Print the run ID, run directory and training command, then exit without training. |
| `--user_prefix` | none | Prefix of the run ID (`<prefix>-<run name>`). |
| `--run_id` | derived | Run name; replaces the ID built from the flags (see [Run name](#run-name)). Must be shorter than 123 characters. |
| `--num_gpus_per_node` | all visible GPUs | GPUs per node (`CUDA_VISIBLE_DEVICES` aware). |
| `--num_nodes` | `1` | Nodes. More than one launches with `torch.distributed.run` (`torchrun`) instead of `accelerate launch`; run the script once per node with the same flags and its own `--node_rank`. |
| `--node_rank` | `$NODE_RANK` or `0` | Rank of this node (multi-node only). |
| `--master_addr` | `$MASTER_ADDR` or `127.0.0.1` | Address of node 0 (multi-node only). |
| `--master_port` | `$MASTER_PORT` or `29500` | Rendezvous port on node 0 (multi-node only). |
| `--wandb_project` | `$WANDB_PROJECT` or `maniflow-tb` | W&B project. |
| `--wandb_entity` | `$WANDB_ENTITY` or your default entity | W&B entity. |
| `--wandb_mode` | see description | `online`, `offline` or `disabled`. Default: `$WANDB_MODE`, else `online` when `WANDB_API_KEY` is set or `~/.netrc` holds W&B credentials, else `offline`. |

The training command runs inside `maniflow-tb/`. `--data_root_dir` and `--run_root_dir` are
resolved against the directory you launch from, so relative paths work from anywhere.

## Data

All fresh runs only, except `--num_workers`.

| Flag | Default | Description |
|---|---|---|
| `--data_root_dir` | none | One or more LeRobot datasets; several are co-trained, merged at load time. Required for a new run; a `--resume` run reads it from its saved config. |
| `--data_mix` | `apr13_test` | Dataset mixture name (`datasets.vla_data.data_mix`). |
| `--val_ratio` | `0.05` | Fraction of episodes held out for validation. |
| `--max_episodes` | all | Use only N episodes, then split them into validation and training. A comma-separated list (e.g. `10000,5000,2000`) caps each `--data_root_dir` separately. |
| `--dataset_ratio` | none | Sampling ratio across datasets, e.g. `1,1` or `10,5,2`; larger datasets are subsampled to match. |
| `--num_workers` | `12` | Dataloader workers per process. |
| `--filter_config` | none | Episode filter preset name or YAML path; presets are in `maniflow-tb/hvla/config/episode_filtering/` (`spatial_filter`). A relative YAML path is resolved from `maniflow-tb/`. |
| `--filter_override` | none | Dotlist overrides of the filter config, e.g. `spatial_region=inside bounds='[0.4,0.7,-0.1,0.1]'`. |
| `--filter_joint_jumps` | off | Drop episodes in which any joint moves more than RAD radians between consecutive frames (teleport resets). The bare flag uses 0.5. Applies to training and validation. |
| `--filter_small_rotates` | off | Drop episodes with a rotate cycle that turns the block less than DEG degrees about z (failed rotate attempts). The bare flag uses 10. Applies to training and validation. |
| `--goal_action_filter` | none | Allow-list of `goal.action` values a training sample may start on, e.g. `'pick,place'`. With several datasets, `'pick,place;stack,done;lift'` gives one `;`-separated group per dataset, and an empty group leaves that dataset unfiltered. Not to be combined with `--phase_sample_weights`. |
| `--phase_sample_weights` | none | Sampling weights per `action.low_level_action` phase: a preset string or a dictionary. |
| `--skip_actions` | none | Comma-separated low-level actions to skip. |
| `--drop_redundant_table_moves` | off | Drop redundant back-to-back table-to-table retries, keeping the last attempt. |
| `--max_frames_per_episode` | all | Truncate each episode to its first N frames. |
| `--skip_first_n_steps_per_episode` | none | Skip the first N frames of every episode. |

## Training loop and optimizer

| Flag | Default | Description |
|---|---|---|
| `--duration` | `200000` | Training steps (`trainer.max_train_steps`). |
| `--dbs` | `128` | Batch size per GPU. |
| `--grad_accum` | `1` | Gradient accumulation steps. |
| `--optimizer` | `adamw` | `adamw` (DeepSpeed ZeRO-2) or `muon` (distributed Muon + AdamW on plain DDP, bf16). `muon` needs the `muon-optimizer` package (see `maniflow-tb/INSTALL.md`). |
| `--muon_lr` | `0.02` | Muon learning rate; lower it (e.g. 0.01 or 0.005) if training diverges. |
| `--save_interval` | `10000` | Save a checkpoint every N steps. |
| `--logging_frequency` | `100` | Log metrics every N steps. |
| `--run_root_dir` | `runs` | Parent directory of the run directory (`<run_root_dir>/<run_id>`). |
| `--resume` | none | Run ID to resume from `<run_root_dir>/<run_id>/`, using its `config.yaml` and latest complete checkpoint. Fails if the run has no loadable checkpoint. |
| `--pretrained_checkpoint` | none | Checkpoint whose weights initialize a new run at step 0; optimizer, scheduler, EMA and W&B state are not restored. |

## Action and state

All fresh runs only.

| Flag | Default | Description |
|---|---|---|
| `--action_mode` | `abs` | Action space: `abs` or `relative` (9-D joints), `ee` (8-D end effector), `ee_target_7d` or `rel_ee_target_7d` (9-D). |
| `--action_keys` | none | Custom action columns, e.g. `action.cube_position.block_r action.cube_distance.block_r`. The action size is read from the first dataset's `meta/info.json`. |
| `--action_norm_band` | `q01_q99` | Arm-action normalization band: `q01_q99` (the `stats.json` quantiles, clipping the outer 1% of labels) or `valid_min_max` (min/max over the `action.is_valid` rows, no clipping). |
| `--state_estimation` | off | Predict `cube_position.block_red1` and `cube_distance.block_red1` (4-D) with a horizon of 1, saving every 2000 steps. |
| `--state_dim` | `9` | Proprioceptive state size: joint positions (7), gripper and PD mode. |
| `--include_jvel_ee_pos` | off | Add joint velocities (7) and the end-effector XYZ (3), giving a 19-D state. |
| `--zero_state` | off | Zero the state vector, keeping its size. |
| `--ft` | off | Add the 6-D wrench to the state. |
| `--horizon` | `50` | Predicted action chunk length. |
| `--n_action_steps` | `25` | Actions executed per chunk. |
| `--num_inference_steps` | `12` | ODE solver steps at inference. |

## Cameras and RGB

All fresh runs only.

| Flag | Default | Description |
|---|---|---|
| `--camera_config` | front and gripper RGB | Camera preset (table below). |
| `--no_gripper` | off | Legacy: same as `--camera_config front_rgb` (`front_rgb_depth` with `--depth`). |
| `--rgb_224` | off | Read pre-resized RGB videos from `<camera>_224` instead of the native 640x360 ones. Missing 224 videos fail the run. |
| `--rgb_backbone` | `vit_base_patch16_clip_224.openai` | timm RGB backbone, e.g. `vit_base_patch16_siglip_224.webli`, `vit_base_patch16_siglip_224.v2_webli` (SigLIP2) or `vit_so400m_patch14_siglip_224.webli`. The normalization mean and std are read from the checkpoint. |
| `--siglip` | off | Shorthand for `--rgb_backbone vit_base_patch16_siglip_224.v2_webli`. |
| `--freeze_rgb_backbone` | off | Freeze the RGB backbone instead of fine-tuning it. |
| `--use_previous_front_frame` | off | Add the previous front frame as a third RGB view. |
| `--previous_front_frame_lookback_weights` | none | Weighted lookback ranges of that frame, e.g. `0-1_1`. |
| `--target_crop` | `-1` | Target-centred square front crop: `-1` off, `0` = all of it above the target, `50` centred, `33` = 67/33 top/bottom, `25` = 75/25. Not with `--use_previous_front_frame`. |

`--camera_config` presets:

| Preset | RGB | Depth |
|---|---|---|
| `front_rgb` | front | none |
| `front_gripper_rgb` | front, gripper | none |
| `front_rgb_depth` | front | front |
| `front_gripper_rgb_front_depth` | front, gripper | front |
| `front_gripper_rgb_depth` | front, gripper | front, gripper |
| `front_gripper_rgb_gripper_depth` | front, gripper | gripper |
| `front_gripper_depth_only` | none | front, gripper |
| `gripper_rgb_depth` | gripper | gripper |
| `gripper_rgb` | gripper | none |

A preset without RGB turns target conditioning off.

## Depth

All fresh runs only. Any depth input also turns on `--per_cam_norm`.

| Flag | Default | Description |
|---|---|---|
| `--depth` | off | Use depth (raw depth, back-projected to a pointmap on the GPU). Without `--camera_config`, both cameras' depth. |
| `--depth_encoder` | `shared_rgb` | Depth encoder: `dense`, `sparse`, `shared_rgb` (the RGB backbone) or `defm`. |
| `--sparse_depth` | off | Sparse PointNet encoder. Only read when `--depth_encoder` is empty, which its choices do not allow, so it has no effect; use `--depth_encoder sparse`. |
| `--ptmap_backbone` | `convnextv2_tiny.fcmae_ft_in22k_in1k` | Pointmap encoder backbone (e.g. `vit_small_patch14_dinov2`). Ignored with `--depth_encoder shared_rgb`, which uses `--rgb_backbone`. |
| `--depth_rg8` | none | Decode depth from RG8-packed videos in `<camera>_<variant>`: `rg8` / `rg8_224` (depth = R + 256 G) or `rg8_55` / `rg8_55_224` (10-bit 5+5 pack, clipped to 1023 mm). Only for datasets re-encoded with `scripts/depth_conversion/reencode_to_rg8_batch.py`. |
| `--depth_g12` | none | Read depth from gray HEVC videos `<depth_key>_<variant>`: `g12_224` (lossless gray12le), `g10_224` (lossless gray10le, clipped to 1023 mm) or `g10_224_crf035` (x265 CRF 0.35). Not with `--depth_rg8`. |
| `--add_depth_noise` | off | RGB-guided depth noise augmentation. |
| `--depth_noise_ideal_mix` | `0.6` | Blend of clean and corrupted depth (1.0 = clean). |
| `--per_cam_norm` | off | Per-camera pointmap normalization bounds instead of global ones. |

## Target conditioning

All fresh runs only.

| Flag | Default | Description |
|---|---|---|
| `--token` | on | Target heatmap as cross-attention context tokens. |
| `--timestep_concat` | off | Heatmap fused into the AdaLN time conditioning. |
| `--target_embedding` | off | Gaussian bias on the RGB tokens, without a heatmap encoder. |
| `--target_pe` | off | Learned per-patch encoding on the RGB tokens, without a heatmap encoder. |
| `--no_target` | off | No target conditioning. |
| `--heatmap_encoder_type` | `shared_rgb` | Heatmap encoder for `--token`: `shared_rgb` (the CLIP ViT), `convnextv2` (a separate pretrained one), `lightweight`, or `cnn` (from scratch on the same 14x14 grid). |
| `--target_mask_prob` | `0.1` | Probability of dropping the target conditioning during training. |
| `--target_sigma` | `0.8` | Gaussian sigma of the heatmap, in patches. |

The five mode flags are mutually exclusive, but at this commit `--token` is on by default and
checked first, so `--timestep_concat`, `--target_embedding` and `--target_pe` have no
effect: the mode is `token` unless `--no_target` is given.

## Language conditioning

All fresh runs only.

| Flag | Default | Description |
|---|---|---|
| `--language_conditioned` | on | Language conditioning (already on; turn it off with `--no_language`). |
| `--no_language` | off | Disable language conditioning. |
| `--language_model` | `t5-small` | T5 encoder: `t5-small`, `t5-base` or `t5-large`. |
| `--language_mode` | `action_only` | `action_only` drops the coordinates from the language string; `action_goal` keeps them. |
| `--lang_injection_mode` | `token` | `token` (cross-attention) or `timestep_concat` (fused into AdaLN). |

## Action-chunk labelling

All fresh runs only.

| Flag | Default | Description |
|---|---|---|
| `--pause_on_attempt` | off | When a pick, place, flip or rotate switch falls inside a chunk, repeat that frame's action for the rest of the chunk and set done = 1 from the switch. Not with `--use_previous_front_frame`. |
| `--freeze_on_goal_change` | off | When the lookback crosses a goal change, relabel the chunk as "hold position". Needs `--use_previous_front_frame`. |

## Diagnostics

| Flag | Default | Description |
|---|---|---|
| `--save_first_batch` | off | Save the preprocessed first batch to `<run_dir>/first_batch_data/`. |
| `--save_arch_diagram` | off | Save an ASCII architecture diagram to `<run_dir>/architecture.txt`. |
| `--profile_steps` | `0` | cProfile the first N steps on rank 0, dumped to `<run_dir>/profiles/`. |
| `--faulthandler` | off | Set `PYTHONFAULTHANDLER=1`, so a native crash dumps a Python traceback per thread. |

## Checks the script enforces

The script stops with an error for:

- a new run without `--data_root_dir`;
- `--num_nodes` above 1 without `--run_id` or `--resume`;
- `--depth_rg8` together with `--depth_g12`;
- `--freeze_on_goal_change` without `--use_previous_front_frame`;
- `--pause_on_attempt` together with `--use_previous_front_frame`;
- no visible GPU and no `--num_gpus_per_node` (not checked with `--dry_run`);
- `--optimizer muon` without the `muon` package installed (not checked with `--dry_run`);
- `--resume` of a run without `<run_root_dir>/<run_id>/config.yaml` (not checked with `--dry_run`).

## Run name

A fresh run's ID is `<datasets>_<horizon>_<n_action_steps>_<action_mode>` plus a suffix for
each non-default choice, then a `-<MM-DD-HH-MM-SS>` timestamp:

- the target mode, heatmap encoder, mask and sigma (`_token_shared_rgb_mask0.1_sigma0.8`);
- the language model and mode (`_lang_t5-small_action_only`);
- the camera preset (for example `_FGd`) and depth encoder;
- the backbone, state, filtering, ratio, optimizer and labelling flags.

`--rgb_224`, `--max_episodes` and `--target_crop` add the prefixes `rgb224_`, `Ep<N>_` and
`tcrop<N>_`, and `--user_prefix` goes in front of everything. IDs longer than 122 characters
are truncated before the timestamp, so that `--resume` accepts them. `--run_id` replaces the
whole ID, timestamp included.
