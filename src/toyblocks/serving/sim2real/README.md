# Sim-to-real with RCS: ManiFlow + Molmo2 on the FR3, sysid, camera calibration

This directory runs ManiFlow (low-level) and Molmo2 (high-level) checkpoints on a
single FR3/DROID-style hardware setup through RCS, plus the system identification
and camera calibration for that setup. Run the policy servers on the machine with
the model and GPU, then run the client on the robot control machine.

```
sim2real/
  README.md
  vlagent_rcs_client.py   main entry point: drives the policies on the robot
  rollout/                everything the client runs: inference loop, config, GUI,
                          eval log, ManiFlow / Molmo2 / FFStereo servers, env and
                          hardware setup, TAM, episode replay, RCS sim eval
  sysid/                  FR3 system identification (see sysid/SYSID.md)
  calibration/            ZED wrist + side camera calibration
```

Reports and notebooks for replays and eval logs are in `scripts/sim2real/`.

> **Safety:** The client controls hardware. Check the robot IP, gripper serial,
> camera serials, home pose, workspace, and relative-motion limits in
> [`rollout/hardware_config.py`](rollout/hardware_config.py) and
> [`rollout/env_setup.py`](rollout/env_setup.py) before executing an episode.
> Begin with the robot clear of obstacles and use conservative limits.

## Prerequisites

1. Create and activate a Python environment with ToyBlocks installed. The
   server must be able to import `toyblocks`, `hvla`, and `vlagents`: install
   with `uv sync --extra maniflow --extra deployment` (`hvla` comes with the
   `maniflow` extra; the ManiFlow server raises an `ImportError` without it).

2. Clone RCS and follow its installation instructions, including the extension
   READMEs for the hardware in use:

   ```shell
   git clone https://github.com/RobotControlStack/robot-control-stack.git
   cd robot-control-stack
   # Follow these READMEs for the exact dependencies and installation commands:
   # rcs-core README.md
   uv pip install -ve .
   # rcs_fr3 extensions/rcs_fr3/README.md
   uv pip install scikit_build_core
   uv pip install --no-build-isolation  -ve extensions/rcs_fr3
   # rcs_zed extensions/rcs_zed/README.md make sure to install zed SDK
   uv pip install -ve extensions/rcs_zed

   uv pip install -ve extensions/rcs_robotiq2f85
   ```

   The client imports `rcs`, `rcs_fr3`, and, when ZED cameras are configured,
   `rcs_zed`. Install the local RCS packages into the same environment used to
   launch the client.

3. Copy the ManiFlow checkpoint directory to the server machine. At minimum it
   needs the model checkpoint (`*.safetensors`), its `stats.json`, and the
   checkpoint `config.yaml`. Keep the files together where possible so the
   loader can find `config.yaml` automatically. If the training configuration
   is stored separately, copy that as well and pass it with `--train-config`.

### Checkpoint configuration compatibility

Older checkpoints may omit fields that the serving loader needs. Add the
following entries to the checkpoint's `config.yaml` (under the existing
`framework` mapping) if they are missing:

```yaml
framework:
  encoder_output_dim: 768
  shape_meta:
    action:
      shape: [9]
```

The 9 policy outputs are seven FR3 joint targets, one gripper value, and one
mode value (`pd_mode`, or `play_mode` for newer checkpoints). The RCS client
forwards only the seven joint targets and the gripper command to the
environment; the policy nevertheless has to be constructed with the same 9-D
action shape it was trained with.

Also ensure that `shape_meta.obs` has the camera/state keys used by the
checkpoint. The keys must agree with the client camera names and the server
state mode. In particular, use `--joint-only-state` only for a checkpoint that
was trained with joint-angle-only state.

## Policy server

Start the ManiFlow vlagents server on the model/GPU machine (run in its own
terminal):

```bash
export CK=/path/to/checkpoint_dir

python -m toyblocks.serving.sim2real.rollout.vlagents_server \
  --checkpoint $CK/steps_160000_model.safetensors \
  --stats-path $CK/stats.json \
  --train-config $CK/config.yaml \
  --device cuda:0 \
  --host 0.0.0.0 \
  --port 8080 \
  --joint-only-state
```

`--checkpoint` and `--stats-path` are required. `--checkpoint` selects the
`.safetensors` model, while `--stats-path` provides the normalization statistics
used at inference. `--train-config` is optional; use it when a separate
training YAML supplies settings absent from the checkpoint `config.yaml`.
Without it, the loader searches alongside the checkpoint for `config.yaml` and
for a `train_*.yaml` file.

Other server arguments:

- `--device`: PyTorch device, such as `cuda:0` (default) or `cpu`.
- `--use-ema`: load EMA weights when the checkpoint contains them.
- `--host` and `--port`: network address exposed to the RCS client. Use
  `0.0.0.0` to accept connections from another machine; use the server's IP in
  the client configuration.
- `--joint-only-state`: send only the seven joint angles plus gripper state to
  the policy. Omit it for a checkpoint trained with the full state.
- `--no-play-mode`: omit play mode from the policy state. Leave it unset for the
  default state format that includes PD mode.

The model identifier served by this wrapper is always `maniflow`; it must match
the client's `vlagents_model` setting.

The Molmo2 and FFStereo servers do not need to be started by hand: the client
launches them on the same machine when they are enabled (see below).

## RCS client

Review the hardware constants in
[`rollout/hardware_config.py`](rollout/hardware_config.py), especially
`ROBOT2IP`, `ROBOTIQ_SERIAL`, `ZED_CAMERA_DICT`, `HOME_POSE`, and
`ROBOT_INSTANCE`, and the PD gains and limits in
[`rollout/env_setup.py`](rollout/env_setup.py). Then edit
[`rollout/vlagent_rcs_client.json`](rollout/vlagent_rcs_client.json). It is
also created with defaults on the first run if absent.

For a server at `192.168.1.50:8080`, a typical single-arm configuration is:

```json
{
  "vlagents_host": "192.168.1.50",
  "vlagents_port": 8080,
  "vlagents_model": "maniflow",
  "instruction": "pick up the green cube",
  "robot_keys": ["right"],
  "jpeg_encoding": true,
  "on_same_machine": false,
  "fps": 25,
  "record_path": "inference_recordings/green_cube",
  "show_camera_windows": false,
  "n_action_steps": 25,
  "max_rel_mov_joints": 0.08726646259971647,
  "max_rel_mov_cart": [0.5, 1.5707963267948966]
}
```

Configuration keys:

- `vlagents_host`, `vlagents_port`, `vlagents_model`: address and model name of
  the server. For this server, `vlagents_model` is `maniflow`.
- `instruction`: language instruction included in every policy observation.
- `robot_keys`: RCS robot names to include in observations and actions. This
  client is configured for the single `right` robot by default.
- `jpeg_encoding`: compress camera images before network transfer. Usually
  enable this for a remote server.
- `on_same_machine`: set to `true` only when client and server run on the same
  machine **as the same UNIX user** and the `vlagents` shared-memory transport
  is appropriate. If you see `PermissionError` on `/psm_*`, set
  `on_same_machine=false` and `jpeg_encoding=true`.
- `point_goal_format`: `action_only` (default) or `action_goal`. Overridden by
  the checkpoint's `framework.language_mode` when that can be read.
- `model_config_path`: checkpoint whose `framework.language_mode` decides
  `point_goal_format` — its `config.yaml`, its directory, or the
  `*.safetensors` in it (CLI `--model-config`). `null` uses the checkpoint path
  the server reports.
- `fps`: desired hardware control-loop rate.
- `record_path`: directory used by `r` recordings.
- `show_camera_windows`: display OpenCV camera windows for debugging.
- `n_action_steps`: number of actions retained from each returned policy chunk.
  Set to `null` to request and execute one action each control step.
- `max_rel_mov_joints`: maximum relative joint motion per step, in radians,
  for joint control.
- `max_rel_mov_cart`: `[translation_m, rotation_rad]` limits for Cartesian
  control. The supplied client uses joint control by default.
- `tam`: enable the Torque Adaptation Module (hardware only; CLI `--tam` /
  `--no-tam`). A learned residual torque is added on top of the Franka PD
  controller (the PD gains selected in `rollout/env_setup.py` are left
  unchanged) and the controller uses the ideal-model gravity compensation. The
  JAX history encoder runs in the client process (GPU recommended) and feeds
  latents at ~5 Hz. Requires the `torque-adaptation-module` (`simadaptor`)
  package; the checkpoint is downloaded once into `~/.cache/simadaptor`. Ported
  from `examples/inference/franka_tam.py`.
- `tam_warmup`, `tam_warmup_target_ms`, `tam_warmup_streak`,
  `tam_warmup_timeout_s`: blocking JAX warmup before the robot moves (CLI
  `--no-tam-warmup` skips it); a push is "fast" below `target_ms`, warmup ends
  after `streak` consecutive fast pushes or warns after `timeout_s`.
- `rate_limit`: apply `franka::limitRate` to the final torque when TAM is on.
  Keep `true`: PD + residual can trip FCI's torque-rate reflex otherwise.
- `tam_history_pre_ratelimit`: record the pre-rate-limit intended torque in
  TAM's history so the limiter does not distort what TAM conditions on.
- `tam_residual_clip`: per-joint `|clip|` in Nm on the TAM residual; `null`
  keeps the controller default `[10,10,10,10,2,2,2]`. The wrist joints go
  bang-bang at high clips, hence the low default on J5-J7.
- `use_molmo`, `molmo_*`, `async_molmo`: Molmo2 high-level policy, see
  [Molmo2 high-level policy](#molmo2-high-level-policy).
- `use_ffstereo`, `ffstereo_*`: Fast-FoundationStereo depth, see
  [FFStereo depth](#ffstereo-depth).

All `tam_*` keys are applied when the environment is created; restart the
client after changing them.

Start the client from the same environment in which RCS and its FR3/ZED
extensions are installed. The module docstring of
[`vlagent_rcs_client.py`](vlagent_rcs_client.py) lists every command-line
flag with examples.

### Full run with robot (terminal control)

```bash
sudo chmod 777 /dev/ttyUSB0   # if using Robotiq gripper
python -m toyblocks.serving.sim2real.vlagent_rcs_client
```

The client keeps retrying until the server is reachable. Once connected, it
resets the environment and waits for a command on standard input.

### Interactive GUI (pick action + click point)

```bash
python -m toyblocks.serving.sim2real.vlagent_rcs_client --interactive-gui
```

The GUI shows `front_rgb` and `gripper_rgb`. Click-to-pick point conditioning
is enabled on **`front_rgb` only**; `gripper_rgb` is view-only. With
`--point-goal-format action_goal`, submitted prompts look like `pick 340,329`.
Use `action_only` (the default) to send the action string without coordinates.

The prompt format is not a free choice: it has to match the one the checkpoint
was trained with, recorded as `framework.language_mode` in the checkpoint's
`config.yaml`. The client reads that field and corrects `point_goal_format` on
its own, from `model_config_path` if set, otherwise from the checkpoint path the
server reports (which resolves whenever the server runs on the same machine). An
explicit `--point-goal-format` is never overridden, only warned about, and a
checkpoint that cannot be read locally leaves the configured value alone.

Submit Goal stays highlighted, like the selected action button, while the last
submitted command is still the one staged; picking a different action, clicking
a different point, or switching the prompt format clears the highlight.

### Molmo2 high-level policy

With `--use-molmo` (or `"use_molmo": true`), Molmo2 replaces the manual click +
action goal: the front frame and a goal prompt go to the Molmo2 server, and the
action and target point it returns become ManiFlow's `instruction` and
`target_heatmap`. The client starts the Molmo2 server on port 8082
(`molmo_port`) itself when `molmo_host` is local.

```bash
python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo --molmo-checkpoint /path/to/molmo2_checkpoint
python -m toyblocks.serving.sim2real.vlagent_rcs_client --use-molmo --async-molmo
```

- `--use-molmo` implies `--interactive-gui`: the goal is typed into the browser
  text box (seeded by `molmo_prompt`) and point clicking on `front_rgb` is off.
- `--async-molmo` queries Molmo2 only at episode start, when the low-level
  skill ends, and when the goal changes, reusing the last instruction and
  heatmap in between. Molmo2 answering `done` ends the episode.
- **Type only the goal**, in the form the checkpoint was trained on, e.g.
  `stack G V L`. The server builds the rest of the question (the previous
  action and, for html-v2 checkpoints, the `… point to target. predict
  action.` instruction) and rejects a prompt that already contains the
  instruction.
- **The checkpoint's training config is the source of truth.** The server runs
  [`serving/molmo2_hl.py`](../molmo2_hl.py), the same Molmo2 code as the sim
  eval, which reads the crop, front-only, frame size, preprocessor style and
  question format from the checkpoint's `config.yaml`. The client always sends
  the full front frame resized to `molmo_input_size`, and the server crops it.
- `molmo_input_size`, `molmo_front_crop` and `molmo_front_view_only` (CLI
  `--molmo-input-size`, `--molmo-front-crop`, `--molmo-front-view-only`) must
  match the checkpoint. They are cross-checked, and a mismatch raises an
  error naming the setting.
- `molmo_use_prev_action` (CLI `--no-molmo-prev-action`) drops the previous
  action from every question.

### FFStereo depth

With `--ffstereo` (or `"use_ffstereo": true`), depth comes from
Fast-FoundationStereo on the ZED stereo pair instead of ZED depth. Set
`INCLUDE_RIGHT = True` and `INCLUDE_DEPTH = False` in
[`rollout/hardware_config.py`](rollout/hardware_config.py). The client starts
the depth server on port 8081 (`ffstereo_port`). The model folder defaults to
`/home/yejin/code/ffstereo`; point `FFSTEREO_ROOT` at your checkout otherwise.

```bash
FFSTEREO_ROOT=/path/to/ffstereo python -m toyblocks.serving.sim2real.vlagent_rcs_client --ffstereo --ffstereo-clip-m 1.5
```

### Dry-run (no robot)

Uses hardware cameras when connected, otherwise mock placeholder frames. Useful
for testing prompt formatting and server connectivity with the robot off:

```bash
python -m toyblocks.serving.sim2real.vlagent_rcs_client --dry-run --point-goal-format action_goal
```

### Browser GUI

`--interactive-gui` and `--dry-run` always start a web GUI on port 8765 (override
with `--gui-port`). Open http://localhost:8765 locally, or forward over SSH:

```bash
# On robot workstation
python -m toyblocks.serving.sim2real.vlagent_rcs_client --dry-run --point-goal-format action_goal

# On your laptop
ssh -L 8765:localhost:8765 user@robots-workstation1
# Open http://localhost:8765
```

Custom port:

```bash
python -m toyblocks.serving.sim2real.vlagent_rcs_client --dry-run --gui-port 8765
```

### GUI episode controls

In the GUI, **Submit Goal** is the only way to start an episode. It always reads
the goal text box (and clicked point, if set), updates the instruction, and
starts the policy. Optionally check **Record episode** before submitting to
save trajectories to `record_path`.

Episode control buttons (or keys when the goal text box is not focused):

| Button / key | Action |
|--------------|--------|
| **Submit Goal** | Read text box, update goal, start episode (optionally record) |
| **Stop (q)** | Stop the running episode and reset the environment |
| **Success (s)** | Mark the episode successful and reset |
| **Reload (o)** | Reload `vlagent_rcs_client.json` and reconnect to the server |
| **Quit (x)** | Exit the client |

Use **Stop** mid-trajectory to abort without quitting the client.

In dry-run mode, **Submit Goal** sends a one-shot server ping; episode control
buttons are accepted but do not move the robot.

### Eval log

Each command graded from the GUI is appended to
`<checkpoint dir>/<record_path>/eval_log_<YYYY-MM-DD>.json` (JSON Lines). The
first line records what the run was made of (commits, checkpoint, stereo
source, PD gains). Summarise success rate and time per skill with:

```bash
python scripts/sim2real/eval_summary.py LOG_OR_DIR
```

## Terminal keyboard commands (non-GUI mode)

| Key | Action |
|-----|--------|
| `e` | Start episode |
| `r` | Start episode + record |
| `s` | Mark success + reset |
| `q` | Stop episode + reset |
| `o` | Reload `vlagent_rcs_client.json` |
| `x` | Exit |

`o` immediately applies `instruction`, `n_action_steps`, `robot_keys`, server
address/model, `record_path`, JPEG transport, and camera-window settings.
`fps`, motion limits, and other environment settings are applied when the
environment is created, so restart the client after changing those values.

## Config and utilities

Edit client config (server address, instruction, transport, etc.) in
[`rollout/vlagent_rcs_client.json`](rollout/vlagent_rcs_client.json). Key
transport settings:

```json
"jpeg_encoding": true,
"on_same_machine": false
```

This works across users and over the network. Use shared memory only when both
processes run as the same UNIX user.

List connected ZED cameras:

```bash
python -m rcs_zed serials
```

View a ZED feed directly:

```bash
python -m rcs_zed rgb-view 35115330   # front_rgb
python -m rcs_zed rgb-view 14943057   # gripper_rgb
```

## Typical workflow

```bash
# Terminal 1: start server
export CK=/path/to/checkpoint_dir
python -m toyblocks.serving.sim2real.rollout.vlagents_server \
  --checkpoint $CK/steps_160000_model.safetensors \
  --stats-path $CK/stats.json \
  --train-config $CK/config.yaml \
  --joint-only-state

# Terminal 2: dry-run GUI to test clicks + prompts
python -m toyblocks.serving.sim2real.vlagent_rcs_client --dry-run --point-goal-format action_goal

# Terminal 2: full run with robot
python -m toyblocks.serving.sim2real.vlagent_rcs_client --interactive-gui --point-goal-format action_goal
```

## Hardware smoke test

Before running, make sure the play_mode starts at 1, and pd gains are set to:

```
# rollout/env_setup.py
        rcfg.kp = np.array([600, 600, 600, 600, 250, 150, 50])
        rcfg.kd = np.array([50, 50, 50, 50, 30, 25, 15])

# rollout/inference.py
        self._prev_play_mode = 1.0
        ...
        play_mode = 1.0
```

```shell
export CK=/path/to/joint_only_checkpoint_dir
python -m toyblocks.serving.sim2real.rollout.vlagents_server \
  --checkpoint $CK/steps_150000_model.safetensors \
  --stats-path $CK/stats.json \
  --train-config $CK/config.json \
  --joint-only-state

# second shell (make sure the initial pd mode is 1!)
python -m toyblocks.serving.sim2real.vlagent_rcs_client
```

Export recorded episodes as one MP4 per camera:

```
Usage: python -m rcs camera-episode-videos [OPTIONS] {dataset} {output}

 Export a simple raw-frame MP4 for every camera in every episode.

╭─ Arguments ──────────────────────────────────────────────────────────────────────────╮
│ *    dataset      <path>  Parquet dataset file or directory with parquet parts. [required] │
│ *    output       <path>  Output directory for camera episode mp4 files. [required]        │
╰──────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ────────────────────────────────────────────────────────────────────────────╮
│ --fps            <int>  Video frames per second. [default: 30]                       │
│ --camera         <str>  Only export this camera. By default, exports every camera.   │
│ --episode        <int>  Only export this zero-based, recording-order episode.        │
│ --help                  Show this message and exit.                                  │
╰──────────────────────────────────────────────────────────────────────────────────────╯
```

## Sim eval in RCS (sim2sim)

[`rollout/vlagents_rcs_eval.py`](rollout/vlagents_rcs_eval.py) runs the same
client stack against RCS's MuJoCo sim over many episodes and writes the success
rate to `results.json`, with a video per episode:

```bash
python -m toyblocks.serving.sim2real.rollout.vlagents_rcs_eval \
    --num-episodes 20 --max-steps 300 --output-dir vlagent_eval_runs --seed 0
```

## Replay

[`rollout/episode_replay.py`](rollout/episode_replay.py) replays a recorded
episode's joint targets on the robot with no policy in the loop, to tell
policy errors apart from execution errors. It builds the env exactly like the
client (`rollout/env_setup.py` + `vlagent_rcs_client.json`), and writes a
per-frame log to `outputs/replay_logs/<stamp>/`.

```bash
python -m toyblocks.serving.sim2real.rollout.episode_replay                 # episode 0
EPISODE=3 REPLAY_DATASET=/path/to/blocks_success python -m toyblocks.serving.sim2real.rollout.episode_replay
```

`tam` comes from `vlagent_rcs_client.json`; `ReplayConfig.tam` in the script
overrides it. Compare runs with the reports in `scripts/sim2real/`:

```bash
# two real replays, e.g. without vs with sysid (defaults to the 2 newest logs)
python scripts/sim2real/replay_report.py [A_dir B_dir] [-o out.pdf]
# one real replay vs the no-sysid and sysid sim rollouts
python scripts/sim2real/replay_sim_cmp_report.py LOG_DIR ...
```

`scripts/sim2real/visualize_trajectory.ipynb` plots joint tracking errors for a
single run.

To use TAM's panda-specific checkpoint:

```bash
pip install --upgrade --force-reinstall --no-deps "torque-adaptation-module @ git+https://github.com/Dongwon-Son/TAM"
SIMADAPTOR_DEFAULT_CHECKPOINT=panda_specific python examples/inference/franka_tam.py
```

## System identification

Fits FR3 joint dynamics for MuJoCo and Isaac from a recording on the real robot,
using the same env and gains as the policy runs. See
[`sysid/SYSID.md`](sysid/SYSID.md).

## Camera calibration

ChArUco calibration of the wrist camera (eye-in-hand) and the side camera
(eye-to-hand), with a Tk GUI. The board stays on the ground while the arm moves:

```bash
python -m toyblocks.serving.sim2real.calibration.camera_calibration
```
