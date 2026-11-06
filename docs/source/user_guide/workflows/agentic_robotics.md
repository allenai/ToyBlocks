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
time they are needed and kept in `agentic_robotics/data/tasks/` (not tracked in git). A task key
names a family of PDDL variants, and a run draws one; a variant's slug names that scene alone.

This walkthrough runs `spell_hen_patterned_0_feasible_0`, one variant of the Search & Spell family
`spell_hen_patterned` from the released experiments: ten blocks on the table, some stacked, some
letters hidden, and the goal of spelling HEN in a tower. It is a feasible variant, its letters are
ones the released ManiFlow checkpoint trained on, and its blocks come from that checkpoint's
library, so the same scene runs under both skill backends. List what is defined and what is built
locally, then build the scene (a family key would build every variant, twelve here, one
Isaac boot each):

```bash
agentic-robotics tasks
agentic-robotics generate-env --task spell_hen_patterned_0_feasible_0
```

Passing `--generate-env` to `episode`, `page` or `human` builds a missing package on the spot.

## Run an episode

```bash
# One headless, model-driven episode with the scripted skills
agentic-robotics episode --task spell_hen_patterned_0_feasible_0 --backend oracle
```

The run directory holds the video (`video.mp4`), the model's transcript, the resolved
`config.json`, and `evaluation_summary.json`: whether the final scene satisfies the task's PDDL
goal (scored with the package's goal classifier), the tool calls made and the model cost.

## Web UI: the episode pages and the portal

The commands below start local web pages. Open <http://127.0.0.1:8083> for the interactive pages
(the task menu, a model-driven episode, the human-controller interface, the mock page) and
<http://127.0.0.1:8082> for the portal, which browses the logs of current and previous runs; the
two ports differ so the portal can stay open while an episode runs.

```bash
agentic-robotics menu --backend oracle                       # Isaac starts with no scene; pick tasks one after another, watch the model act, interject
agentic-robotics page --task spell_hen_patterned_0_feasible_0 --backend oracle    # one model-driven episode with the interjection page around it
agentic-robotics human --task spell_hen_patterned_0_feasible_0 --backend oracle   # the same episode with you in the model's chair: call the skills yourself
agentic-robotics portal                                      # browse finished runs (videos, transcripts, scores); no Isaac, no model
agentic-robotics portal --static                             # or write the portal as static pages under <runs dir>/portal/
agentic-robotics mock                                        # the page alone, without Isaac or a GPU, for working on the UI
```

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
   underscores. ToyBlocks's task generators write suitable problems. This one writes three
   variants of the scene the ToyBlocks quick start generates training data for, the letter-M
   block beside a red block, each with the goal of one flip of the M block that brings the face
   on its top around to the front:

   ```bash
   python scripts/task_gen/flip.py --direction backward --target-block block_m \
       --distractor-blocks block_red --distractor-count-min 1 \
       --count 3 --seed 0 --out-dir agentic_robotics/data/problems/flip_m
   ```

   See [Task generation](datagen/task_generation.md) for the generators and the PDDL comments
   that randomize the scene.

2. **Register it.** Add a `TaskConfig` to `TOYBLOCKS_TASKS` in
   `agentic_robotics/src/agentic_robotics/adapters/toyblocks/config.py`:

   ```python
   "flip_m": TaskConfig(
       name="Flip M",
       task_slug="flip_m",            # data/problems/flip_m(.pddl | /)
       prompt=PROMPT_PREFIX
       + (
           "The letter block has the letter 'M' on one of its faces. Your task is to flip "
           "that block once so that the face now on its top ends up facing the camera.\n\n"
       )
       + PROMPT_SUFFIX,
       max_tool_calls=10,             # the model's budget of MCP tool calls
   ),
   ```

   `PROMPT_PREFIX` and `PROMPT_SUFFIX` frame every task (the robot's role, how pixel coordinates
   work, how to end the episode); the middle is what the model is asked to do, and it should say
   in words what the problem's PDDL goal checks: a run is scored by the package's goal classifier,
   so a prompt that asks for more or less than the goal scores wrongly. An optional
   `glyph_library` field names the asset library the letter and digit blocks are drawn from,
   `train_blocks` or `train_blocks_large` (ToyBlocks's `GLYPH_LIBRARIES`); leave it out for the
   default, `train_blocks_large`.

3. **Build and run it.**

   ```bash
   agentic-robotics generate-env --task flip_m
   agentic-robotics human --task flip_m --backend oracle       # try it yourself first
   agentic-robotics page --task flip_m --backend oracle        # then watch the model try, and interject
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
agentic-robotics serve --maniflow maniflow-tb        # the policy server, on port 8000; leave it running

# In a second terminal. A new shell starts without this one's environment, so restore it:
cd ToyBlocks && source .venv/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES                      # and OPENROUTER_API_KEY, unless it is in ~/.openrouter_key
agentic-robotics page --task spell_hen_patterned_0_feasible_0 --backend maniflow --maniflow maniflow-tb
```

Without `--maniflow`, both use the experiment file's `maniflow` key; without `--backend`, the
episode uses its `skill_backend`. A missing checkpoint file is reported by name before Isaac starts.

## Experiments and further reading

What a run holds fixed -- backend, model, checkpoint, camera, seed, the scripted skills'
parameters -- comes from an experiment file,
`agentic_robotics/src/agentic_robotics/adapters/toyblocks/experiments/<name>.toml`, chosen with
`--experiment <name>` (`default.toml` otherwise). `agentic-robotics --help` lists every mode and
flag. The submodule's README covers the evaluation suites and the Search & Spell experiments.
