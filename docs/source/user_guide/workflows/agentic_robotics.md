# Agentic Robotics

The `agentic_robotics/` submodule ([allenai/agentic_robotics](https://github.com/allenai/agentic_robotics))
puts a frontier vision-language model in charge of the robot. The model sees the front camera
image and acts only through MCP tools, one per skill: `pick`, `stack`, `place_on_table`,
`rotate`, `flip`, `return_home`, the gripper, and a zoomed view. Nothing tells it whether it
succeeded; it decides from the images when the task is done or impossible. Every run is recorded
(video, transcript, final-state score) so it can be browsed afterwards.

Two skill backends execute the tools:

| Backend | What runs the skill | Needs |
|---|---|---|
| `oracle` (default) | ToyBlocks's scripted vecgen skills (the ones that record the training demonstrations), planned with cuRobo v2 on privileged simulator state | the `vecgen` extra |
| `maniflow` | a trained ManiFlow low-level policy, through its policy server | the `maniflow` extra and a checkpoint |

## Setup

This is the [Installation](installation_and_setup_instructions.md) sequence with the `vecgen`
and `agentic` extras and the `agentic_robotics` submodule; it needs an NVIDIA GPU (driver 570 or
newer) and [`uv`](https://docs.astral.sh/uv/getting-started/installation/):

```bash
git clone https://github.com/allenai/ToyBlocks.git && cd ToyBlocks
uv venv --python 3.11 && source .venv/bin/activate
git submodule update --init curobo agentic_robotics
uv pip install -e ".[dev,isaac,torch-cu128,vecgen,agentic]"
export OMNI_KIT_ACCEPT_EULA=YES
get-toyblocks-assets --asset-manifest asset_manifest.json   # the robot, block and scene assets
```

The assets are downloaded once per machine into `~/.toyblocks` and symlinked into `assets/`; without
them the first task build fails with a missing robot USD. `pytest tests/install` checks the install,
the submodule pins and the assets (see [Verify the installation](installation_and_setup_instructions.md#verify-the-installation)).

The `agentic` extra installs the `agentic-robotics` package (editable, from the submodule) and
its `agentic-robotics` command. Its MCP SDK needs newer `starlette`, `fastapi`, `uvicorn` and
`idna` than Isaac Sim pins; `pyproject.toml` overrides those four (they serve Isaac's livestream
and web services, which headless runs do not use).

### Your own OpenRouter key

Episodes call the model through [OpenRouter](https://openrouter.ai), and **you need your own
OpenRouter API key**: none ships with the repository, and every episode is billed to the key it
runs with. Create one at <https://openrouter.ai/keys>, then either export it or store it in a file
only you can read:

```bash
export OPENROUTER_API_KEY=sk-or-...
# or
echo "sk-or-..." > ~/.openrouter_key && chmod 600 ~/.openrouter_key
```

`agentic_robotics/scripts/openrouter_spend.sh` prints the key's remaining budget. The model is the
experiment file's `model` (default `fable-5.1`); `--model` picks another key from the adapter's
model table (`opus-5.5`, `sonnet-5.5`, `haiku-4.5`, `gpt-6.1-sol-pro`, `qwen-3.8`). An unknown
key fails with the list of known ones.

Runs are written under `$AGENTIC_ROBOTICS_RUNS_DIR`, or `agentic_robotics/runs/`.

## Build a task's environment packages

A task's scenes are ordinary ToyBlocks env packages, generated with ToyBlocks's codegen the first
time they are needed and kept in `agentic_robotics/data/tasks/` (not tracked in git). List what is
defined and what is built on this machine, then build a task:

```bash
agentic-robotics tasks
agentic-robotics generate-env --task r_on_red        # one package per PDDL variant (r_on_red has one)
```

Passing `--generate-env` to `episode`, `page` or `human` builds a missing package on the spot.

## Run an episode

```bash
# One headless, model-driven episode with the scripted skills
agentic-robotics episode --task r_on_red --backend oracle
```

The run directory holds the video (`video.mp4`), the model's transcript, the resolved
`config.json`, and `evaluation_summary.json`: whether the final scene satisfies the task's PDDL
goal (scored with the package's goal classifier), the tool calls made and the model cost.

## Web UI: the episode pages and the portal

The commands below start local web pages; open the printed URL in a browser.

| Command | Page | What it does |
|---|---|---|
| `agentic-robotics menu --backend oracle` | <http://127.0.0.1:8083> | Isaac starts with no scene; pick tasks one after another from the page, watch the model act, and interject |
| `agentic-robotics page --task r_on_red --backend oracle` | <http://127.0.0.1:8083> | one model-driven episode with the interjection page around it |
| `agentic-robotics human --task r_on_red --backend oracle` | <http://127.0.0.1:8083> | the same episode with you in the model's chair: call the skills yourself |
| `agentic-robotics portal` | <http://127.0.0.1:8082> | browse finished runs (videos, transcripts, scores); no Isaac, no model |
| `agentic-robotics portal --static` | files | write the portal as static pages under `<runs dir>/portal/` |
| `agentic-robotics mock` | <http://127.0.0.1:8083> | the page alone, without Isaac or a GPU, for working on the UI |

An episode page embeds the simulator's 3D viewer, which listens on its own port (8080, or the next
free one) and must be reachable from the browser too. The pages bind to localhost; on a remote
machine either forward the ports or pass `--lan` to bind every interface:

```bash
ssh -L 8083:localhost:8083 -L 8080:localhost:8080 -L 8082:localhost:8082 <gpu-machine>
```

`--port` changes the page's port.

## Create a new task

A task is a PDDL problem (or a directory of variant problems) plus an entry giving the model its
prompt and tool-call budget.

1. **Write the problem.** Problems use ToyBlocks's blocks domain
   (`src/toyblocks/tamp/blocks/domain.pddl`). Put one problem at
   `agentic_robotics/data/problems/<slug>.pddl`, or several variants in
   `agentic_robotics/data/problems/<slug>/*.pddl` (a run draws one at random). Each variant's file
   stem becomes its env-package name, so it must be a valid Python identifier once hyphens become
   underscores. ToyBlocks's task generators write suitable problems, for example:

   ```bash
   python scripts/task_gen/stack_three.py --train --count 3 --seed 0 \
       --out-dir agentic_robotics/data/problems/my_stack
   ```

   See [Task generation](datagen/task_generation.md) for the generators and the PDDL comments
   that randomize the scene.

2. **Register it.** Add a `TaskConfig` to `TOYBLOCKS_TASKS` in
   `agentic_robotics/src/agentic_robotics/adapters/toyblocks/config.py`:

   ```python
   "my_stack": TaskConfig(
       name="Custom New Task",
       task_slug="my_stack",          # data/problems/my_stack(.pddl | /)
       prompt=PROMPT_PREFIX
       + "Stack any three letter blocks so their letters read top to bottom. "
       + PROMPT_SUFFIX,
       max_tool_calls=30,             # the model's budget of MCP tool calls
   ),
   ```

   `PROMPT_PREFIX` and `PROMPT_SUFFIX` frame every task (the robot's role, how pixel coordinates
   work, how to end the episode); the middle is what the model is asked to do. An optional
   `glyph_library` field names the asset library the letter and digit blocks are drawn from,
   `train_blocks` or `train_blocks_large` (ToyBlocks's `GLYPH_LIBRARIES`); leave it out for the
   default, `train_blocks_large`.

3. **Build and run it.**

   ```bash
   agentic-robotics generate-env --task my_stack
   agentic-robotics human --task my_stack --backend oracle     # try it yourself first
   agentic-robotics page --task my_stack --backend oracle      # then watch the model try, and interject
   ```

   `generate-env` builds one package per variant, so a task with many variants takes a while;
   `page` serves the episode with the chat page around it (`episode` runs it headless).

   `generate-env` reports reset states with unreachable or overlapping blocks; adjust the
   problem's ranges if many are flagged.

## ManiFlow skills

The `maniflow` backend runs a trained ManiFlow policy instead of the scripted skills. It needs the
`maniflow` extra, which builds the `maniflow-tb` and `molmo2` submodules, and a checkpoint
directory in the layout a training run writes (`checkpoints/steps_N_model.safetensors`,
`config.yaml`, `stats.json`; see [Imitation learning](imitation_learning.md)). The released
checkpoint, [allenai/maniflow-tb](https://huggingface.co/allenai/maniflow-tb), has that layout:

```bash
git submodule update --init maniflow-tb molmo2
uv pip install -e ".[dev,isaac,torch-cu128,vecgen,maniflow,agentic]"
hf download allenai/maniflow-tb --local-dir checkpoints/maniflow-tb
```

The adapter's checkpoint table (`maniflow_config.py`, `MANIFLOW_CONFIGS`) names each checkpoint
directory relative to `$MANIFLOW_CHECKPOINT_ROOT` (default: `checkpoints/` under the working
directory); `maniflow-tb` is the entry for the released checkpoint. Start the policy server in its
own shell, then run the episode against it:

```bash
agentic-robotics serve --maniflow maniflow-tb                                     # policy server, port 8000
agentic-robotics episode --task ai2_stacked --backend maniflow --maniflow maniflow-tb   # in another shell
```

Without `--maniflow`, both use the experiment file's `maniflow` key; without `--backend`, the
episode uses its `skill_backend`. A missing checkpoint file is reported by name before Isaac starts.

## Experiments and further reading

What a run holds fixed -- backend, model, checkpoint, camera, seed, the scripted skills'
parameters -- comes from an experiment file,
`agentic_robotics/src/agentic_robotics/adapters/toyblocks/experiments/<name>.toml`, chosen with
`--experiment <name>` (`default.toml` otherwise). `agentic-robotics --help` lists every mode and
flag. The submodule's README covers the evaluation suites and the Search & Spell experiments.
