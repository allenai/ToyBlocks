# CLAUDE Instructions for ToyBlocks

## Critical Rules for AI Code Changes

1. **Minimize file modifications.** Touch only the files strictly necessary to implement the requested change. Do not refactor, reorganize, or "improve" unrelated code. If a change can be accomplished by editing one file, do not edit two.
2. **Summarize every file change.** After making changes, provide a detailed summary listing each modified file with:
   - The file path
   - What was changed and why
   - Any side effects or dependencies introduced
3. **No speculative edits.** Do not modify files "just in case" or to preemptively fix issues that were not requested. Each file touched must be directly justified by the task.
4. **Preserve existing patterns.** When editing a file, match the surrounding code style, import conventions, and structure already present in that file. Do not introduce new patterns unless explicitly asked.
5. **Scope awareness.** Before editing, consider whether the change belongs in the file you're about to touch or whether it should go in a more appropriate location. Prefer surgical edits over sweeping refactors.
6. **No TAMP.** The TAMP submodule (`tmp3`) is gone: do not import `tmp3` / `evals` / `test_domains` (`tests/tamp/test_no_tamp_submodule.py` enforces it).
7. For all newley generated code, make sure the objects are properly typed.
---

## Running Commands

**Always run long-running commands inside a detached `tmux` session.** Never launch
them as a plain background shell job. A background job is a child of the calling
agent process, so it dies when that session disconnects or is torn down; a tmux
session reparents to pid 1 and survives independently.

```bash
LOG=/path/to/job.log
tmux new-session -d -s <session-name> "<command> > $LOG 2>&1; echo EXIT=\$? >> $LOG; sleep infinity"

# Then poll:
tail -f $LOG
tmux ls
```

- Keep `sleep infinity` at the end so the session stays alive after the command
  exits and the exit code remains readable.
- Verify detachment: the parent chain of the process should end at `tmux: server`,
  not at the agent process (`ps -o pid=,ppid=,comm= -p <pid>`).
- This applies to dataset merges, renders, large scans over shared storage, training runs —
  anything expected to outlive a few minutes.

---

## Project Overview
- **ToyBlocks** is a research codebase for robotic manipulation task generation, data collection, policy training, and evaluation. It integrates Isaac Lab simulation, symbolic task planning (FF) with CuRobo motion planning (the vecgen datagen engine), and policy serving infrastructure.
- The architecture is modular: simulation environments, datagen (vecgen), policy serving, and evaluation are separated into distinct packages.
- Key directories:
  - `src/toyblocks/`: Core Python package — tasks, blocks domain, datagen (vecgen), serving, evaluation, utilities.
  - `src/toyblocks/tamp/blocks/`: Blocks domain — PDDL domain + problems, action config, goal checks, env-package templates.
  - `src/toyblocks/vecgen/`: Datagen engine — FF planning, batched skills, cuRobo v2 planner pool, recording.
  - `src/toyblocks/tasks/`: Isaac Lab gymnasium environments (lift, flip, rotate, push, pile, etc.).
  - `src/toyblocks/serving/`: Policy server/client infrastructure (WebSocket + msgpack).
  - `src/toyblocks/eval/`: Evaluation utilities (replay, metrics, resets).
  - `scripts/`: Entry points for datagen, evaluation, task generation, and policy serving.
  - `curobo/`: NVIDIA CuRobo motion planning (editable dependency).
  - `maniflow-tb/`: ManiFlow low-level policy (installs as the `hvla` package; submodule of allenai/maniflow-tb).
  - `molmo2/`: Molmo2 high-level policy model (installs as `ai2-molmo2`, package `olmo`; submodule of allenai/molmo2).
  - `agentic_robotics/`: LLM agents driving the robot through MCP skill tools (installs as `agentic-robotics`, `agentic-robotics` CLI; submodule of allenai/agentic_robotics). Its scripted skills run vecgen's skills via `toyblocks.eval.ll_oracle`'s engine wiring, so changes there can affect it.
  - `data/`: Default pipeline output root (`--output-root`): `data/datasets/<folder>/<exp>`, merged sets under `data/datasets/merged/train/<exp>`.

## Critical Files & Data Flow

### Datagen Pipeline (vecgen)
Orchestrated by `scripts/datagen/run_datagen_pipeline.py` on this machine (one problem per GPU slot: `--local-gpus` x `--local-gpu-workers`; output under `--output-root`, default `data`; `--resume` skips stages marked done in `<base>/.pipeline_state/`):
1. **Task Generation** (`scripts/task_gen/*.py`): PDDL problems.
2. **Env** (`scripts/generate_task_from_pddl.py`): PDDL problem → autogen Isaac Lab env package + sampled reset states (`--randomize_robot_init`: collision-free cuRobo IK starts). With `--vecgen-problems-per-env G > 1` the vecgen jobs build multi-problem packages instead (`scripts/vecgen/make_superset_env.py`).
3. **Execution** (`scripts/vecgen/run.py`, `src/toyblocks/vecgen/`): FF plans per env (`symbolic/`), batched skills on a cuRobo v2 planner pool (`skills/`, `planner/`), verified effects and goals (`symbolic/predicates.py`), LeRobot recording → `execution/<slug>/`.
4. **Merge / annotate / plot** (`scripts/datagen/merge_datasets.py`, `src/toyblocks/data_annotation/annotate_target_xy.py`, `scripts/datagen/plot_episode.py`).

Play data and TAMPER (`scripts/datagen/run_tamper_pipeline.py`, `docs/tamper.md`) run on vecgen too. Rendering (`scripts/render/`) runs on `toyblocks.domain_randomization.render_sim.RenderSim`. It runs inside the datagen and play pipelines via `--render-pass FLAGS.yaml[:N]`, the only flag that turns the stage on (repeatable, one pass per look, no default YAML); datagen's `--render-schedule` picks whether each render is its own job or rides the execution job. HDRI backdrops: polyhaven comes with `asset_manifest.json`, the Laval indoor-2018 pool is installed separately; see `docs/source/user_guide/workflows/datagen/render.md`.

### Evaluation
- `scripts/evaluate.py`: Eval worker: remote policy server, LL oracle, dataset replay, or random actions.
- `src/toyblocks/serving/serve_maniflow_policy.py`: ManiFlow LL policy server (`--hl oracle`), or Molmo2 HL + ManiFlow LL (`--hl model`).
- `scripts/launch_eval.py`: Queue-driven eval launcher: policy server, episode queue and eval workers on this machine.

### TAMP Domain (Blocks)
- `src/toyblocks/tamp/blocks/domain.pddl`: Symbolic domain definition.
- `src/toyblocks/tamp/blocks/action_config.yaml`: Per-action argument layouts (`HL_ARGS`).
- `src/toyblocks/tamp/blocks/pddl_goal_tensor_conversions.py`: Batched goal-predicate checks on the live sim (eval success gate, oracle step checks).
- `src/toyblocks/tamp/blocks/template/`: Codegen for autogen env packages from PDDL.
- `src/toyblocks/tamp/blocks/problems/`: Checked-in PDDL problem files.

### Serving
- `src/toyblocks/serving/base_policy.py`: `BasePolicy` interface all policy servers must implement.
- `src/toyblocks/serving/policy_server.py`: WebSocket+msgpack server.
- `src/toyblocks/serving/policy_client.py`: Client used by `evaluate.py`.
- `src/toyblocks/serving/obs_transforms/`: Observation transforms (maniflow, maniflow_blocks_table).

## Developer Workflows

### Environment & Install
- **Package manager**: `uv` (see `pyproject.toml`).
- **Python**: Requires >= 3.11.
- **Setup**: `uv venv --python 3.11` and `git submodule update --init --recursive`; install with `uv pip install -e .[...]`, not `uv sync` (see README).
- **Install with Isaac Lab**: `uv pip install -e .[dev,isaac,torch-cu128]`
- **Install for datagen (vecgen, builds cuRobo v2 from `./curobo`, no CUDA toolkit needed)**: `uv pip install -e .[dev,isaac,torch-cu128,vecgen]`
- **Install with ManiFlow**: `uv pip install -e .[dev,isaac,torch-cu128,maniflow]` (extras combine in one command)
- **Install agentic robotics**: `uv pip install -e .[dev,isaac,torch-cu128,vecgen,agentic]` (needs the `agentic_robotics` submodule and the user's own OpenRouter key; see `docs/source/user_guide/workflows/agentic_robotics.md`)
- **Linting**: `ruff check` / `ruff format` (config in `pyproject.toml`).

### Running the Datagen Pipeline Locally

```bash
# End to end (task gen -> env -> vecgen -> merge -> annotate), one problem per GPU
python scripts/datagen/run_datagen_pipeline.py --experiment-id oct31_stack3 \
    --output-root data --task-gen-script scripts/task_gen/stack_three.py --task-gen-count 2 \
    --env-per-problem 16 --num-envs 16 --exec-num-envs 16

# Or stage by stage for one problem
problem=src/toyblocks/tamp/blocks/problems/flip.pddl
task=$(basename "${problem}" .pddl | tr '-' '_')
python scripts/generate_task_from_pddl.py --domain_path src/toyblocks/tamp/blocks \
    --pddl ${problem} --num_reset_states 32 --num_envs 16 --seed 0 --output_dir outputs --headless
python scripts/vecgen/run.py --env-path outputs/${task} --pddl ${problem} \
    --domain src/toyblocks/tamp/blocks/domain.pddl --num-envs 16 --episodes 32 \
    --record datasets/flip/${task} --mark-complete --headless
```

### Evaluation (Molmo2 HL + ManiFlow LL)

```bash
# Starts the policy server, episode queue and eval worker on this machine
uv run python scripts/launch_eval.py --num-eval-gpus 1 \
    --dataset-path <merged LeRobot val set> --episode-indices 0-3 --queue-limit 4 \
    --model-path <run>/checkpoints/steps_N_model.safetensors --hl-model-path <molmo2 step dir> \
    --no-hl-prev-action
```

Add `--ll-oracle` (Molmo2 HL + oracle LL), `--policy maniflow` (oracle HL + ManiFlow LL), `--replay`
(recorded actions) or `--oracle-comparison --save-hl-overlay-video`. For pdb debugging, run
`scripts/eval/serve_local_server.sh` and `scripts/eval/serve_local_client.sh` in two terminals.

### PDDL Task Generation

```bash
# Atomic skills on block_r (with distractors): one flip, one rotate, one pick-and-place
python scripts/task_gen/flip.py --count 50 --out-dir outputs/flip/tasks --seed 42
python scripts/task_gen/rotate.py --count 50 --out-dir outputs/rotate/tasks --seed 42
python scripts/task_gen/pnp.py --count 50 --out-dir outputs/pnp/tasks --seed 42

# 3-block stacking in letter + digit + color scenes (--train / --val glyph splits)
python scripts/task_gen/stack_three.py --train --count 50 --out-dir outputs/stack3/tasks --seed 42

# Play-data scenes (used by scripts/datagen/run_playdata_pipeline.py)
python scripts/task_gen/play_scene.py --count 50 --out-dir outputs/play/tasks --seed 42
# ...and the search scenes the DFA goal policies run on (src/toyblocks/playdata/dfa/AUTHORING.md)
python scripts/task_gen/find_as.py --count 5 --out-dir outputs/find_as/tasks --seed 42
python scripts/task_gen/search.py --count 5 --out-dir outputs/search/tasks --seed 42
```

## Project-Specific Conventions
- **Isaac App Launch**: Always call `launch_app()` from `toyblocks.cli.isaac_app` before importing any Isaac Lab modules. This is a hard requirement.
- **Autogen Tasks**: Generated env packages go under `src/toyblocks/tasks/autogen/*/` (gitignored). They register as `<slug>-v0` gymnasium environments.
- **PDDL Naming**: PDDL problem basenames become task slugs via `task_slug_from_pddl_path` (hyphens → underscores).
- **Predicates / skills**: vecgen checks PDDL facts on the sim in `vecgen/symbolic/predicates.py` and executes each domain action with a batched skill in `vecgen/skills/`.
- **LeRobot Format**: All dataset recording (datagen and eval) uses the one `IsaacDataRecorderLeRobot` in `toyblocks.vecgen.recording` (LeRobot v3, parallel writers stitched at finalize); see `docs/lerobot_recording.md`.
- **Observation Transforms**: Policy servers and `evaluate.py` use pluggable obs transforms registered in `src/toyblocks/serving/obs_transforms/`.
- **`tyro`**: Most CLI scripts use `tyro` for argument parsing (dataclass-based configs).
- **`uv`**: Use `uv run` to execute scripts within the managed virtualenv.

## Integration Points
- **Isaac Lab**: All simulation via Isaac Lab's Python API (requires `--extra isaac`).
- **FF**: Symbolic planner, vendored (with its binary) in `src/toyblocks/vecgen/symbolic/ff_planner/`.
- **CuRobo**: Motion planning in `./curobo/` (editable install as `nvidia-curobo`).
- **ManiFlow**: Molmo2 HL in `./molmo2/`, ManiFlow LL in `./maniflow-tb/` (`hvla` package).
- **Agentic robotics**: `./agentic_robotics/` (`agentic-robotics` package): episodes, web UI (`agentic-robotics menu` / `portal`), task packages in `agentic_robotics/data/tasks/`.

## Key Entry Points

| Script | Purpose |
|--------|---------|
| `scripts/generate_task_from_pddl.py` | PDDL → autogen env package + reset states |
| `scripts/vecgen/run.py` | Datagen execution: FF + batched skills → LeRobot dataset |
| `scripts/evaluate.py` | Eval worker (policy server, LL oracle, replay, random) |
| `src/toyblocks/serving/serve_maniflow_policy.py` | ManiFlow / Molmo2 WebSocket policy server |
| `scripts/launch_eval.py` | Queue-driven eval launcher (local, single or multi GPU) |
| `scripts/task_gen/*.py` | PDDL problem generators (flip, rotate, push, etc.) |
| `scripts/datagen/run_datagen_pipeline.py` | End-to-end pipeline orchestrator |

---

For more details, see docstrings in the scripts and the documentation in `docs/`.
