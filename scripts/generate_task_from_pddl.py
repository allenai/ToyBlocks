"""Build an autogen Isaac Lab task package from a PDDL problem file.

This is the legacy entry point: pass a checked-in or planner-emitted PDDL via
``--pddl`` and the script writes ``<output_dir>/<slug>/{env.py, env_cfg.py,
__init__.py}`` registering ``<slug>-v0`` with gymnasium. With ``--num_reset_states 0``
no Isaac stepping happens (only codegen); with ``--num_reset_states > 0`` the
script also samples reset states by stepping the env through the TAMP-free predicate
samplers in :mod:`toyblocks.vecgen.sampling` (the same ones ``make_superset_env.py`` uses).

For the queue-driven eval flow, the dataset-only sibling
``scripts/generate_task_from_dataset.py`` skips the GCS PDDL bundle by reading
``pddl_goal`` and ``observation.state.object_pose.*`` directly from the LeRobot
dataset. Both scripts share the codegen helpers in
:mod:`toyblocks.tamp.blocks.template.codegen`.
"""

import tyro

from toyblocks.cli.isaac_app import launch_app

simulation_app = launch_app()
import compileall
import importlib
import os
import pickle
import shutil
import sys
import zlib
from dataclasses import dataclass
from typing import Annotated

import gymnasium as gym
import numpy as np
import torch
import tqdm

from toyblocks.cli.base_args import BaseArgs
from toyblocks.tamp.blocks.camera_view import blocks_in_front_view
from toyblocks.tamp.blocks.template.codegen import (
    SKIP_PREDICATES,
    assert_goal_text_supported,
    extract_clear_blocks,
    extract_goal_block,
    generate_environment_code,
    parse_pddl_init,
    parse_pddl_randomization_ranges_from_text,
    reorder_predicates_by_dependencies,
    task_slug_from_pddl_path,
    write_env_package,
)
from toyblocks.tasks.base_env import BaseEnv
from toyblocks.tasks.base_env_cfg import BaseEnvCfg
from toyblocks.utils import common_utils, import_utils, tree_utils
from toyblocks.utils.reset_states_io import pack_reset_states
from toyblocks.utils.robot_init_sampling import (
    CollisionFreeRobotInitSampler,
    RobotInitSamplingConfig,
)
from toyblocks.vecgen.sampling import SceneSim, make_sampler
from toyblocks.vecgen.sampling.predicates import set_spawn_ranges
from toyblocks.vecgen.sim import VecSim

#: Rounds of re-sampling an env whose blocks are not all inside the front camera view.
MAX_OUT_OF_VIEW_RESAMPLES = 20


@dataclass
class Args(BaseArgs):
    """
    Script to create a task/data generator from a PDDL/problem file.

    Example command:

    python scripts/generate_task_from_pddl.py --pddl src/toyblocks/tamp/blocks/problems/pnp_on_table.pddl \
        --domain_path src/toyblocks/tamp/blocks --output_dir outputs --headless --my_verbose

    For local tasks that load without editing ``toyblocks.tasks`` (see ``tasks/autogen/__init__.py``),
    write under ``src/toyblocks/tasks/autogen`` (ignored by git for ``problem_*`` packages).

    The output package folder name, Gym registration id prefix, and generated Python class names use the
    PDDL file stem with hyphens (``-``) replaced by underscores (``_``).
    """

    pddl: str = "src/toyblocks/tamp/blocks/problems/flip.pddl"
    """The PDDL file to generate task from."""
    seed: Annotated[int, tyro.conf.arg(aliases=["-s"])] = 0
    """Seed for the random number generator."""
    domain_path: str = "src/toyblocks/tamp/blocks"
    """Path to the folder containing all domain specific files, including predicate definitions, environment template, and problem definitions"""
    output_dir: str = "."
    """Directory that will contain the generated task folder (slug). Use ``src/toyblocks/tasks/autogen`` for runtime autoload."""
    num_reset_states: int = 1
    """How many initial physics snapshots to sample and save as reset_states.pkl. If 0, only env code is generated (no sim, no pickle)."""
    my_verbose: bool = False
    """Verbose output."""
    num_envs: int = 1
    """Parallel env count for reset-state sampling. Ignored when num_reset_states is 0. Must be >= 1 and divide num_reset_states when num_reset_states > 0."""
    num_scene_profiles: int = 8
    """How many distinct block-size / table-height / friction profiles to bake into the generated env package. Baked at codegen time, so every later stage that loads this package inherits it. num_envs must be a multiple of this (or smaller) in both task generation and plan execution."""
    block_x_range: str = ""
    """Table-frame x window for on-table block spawns, overriding any ;block_x_range in the PDDL. Either one interval ("[-0.2, 0.58]") or disjoint intervals ("[[-0.2,-0.1],[0.3,0.5]]"), which spawn nothing in the gap. Empty keeps the PDDL value, else the sampler default [-0.1, 0.25]. Offsets are from the table origin, whose robot-frame position depends on the scene preset, so the usable window is bounded by MAX_REACH_RADIUS (0.65 m from the robot base) rather than by the table edge; stage 1 prints the actual anchor and the usable fraction."""
    block_y_range: str = ""
    """Table-frame y window for on-table block spawns, overriding any ;block_y_range in the PDDL. Same format as block_x_range. Empty keeps the PDDL value, else the sampler default [-0.30, 0.30]."""
    randomize_robot_init: bool = False
    """If True, give each reset state a random collision-free arm configuration and a random open/closed gripper. The end-effector position is randomized uniformly in a workspace box capped at the home end-effector height, with the orientation held at the home end-effector orientation, solved with CuRobo IK (self-collision and scene collision checked)."""
    robot_init_ee_x_range: tuple[float, float] = (0.25, 0.65)
    """End-effector x range in meters, robot base frame (randomize_robot_init only)."""
    robot_init_ee_y_range: tuple[float, float] = (-0.35, 0.35)
    """End-effector y range in meters, robot base frame (randomize_robot_init only)."""
    robot_init_ee_z_min: float = 0.20
    """Lowest end-effector height in meters, robot base frame; the upper bound is the home pose height (randomize_robot_init only)."""
    robot_init_candidates: int = 32
    """End-effector poses tried per env before keeping the home pose (randomize_robot_init only)."""
    robot_init_gripper_closed_prob: float = 0.5
    """Probability that a reset state starts with the gripper closed (randomize_robot_init only)."""
    scene_preset_id: str = ""
    """Scene preset baked into the generated env_cfg.py. Empty keeps the template's default."""
    robot_id: str = ""
    """Robot id baked into the generated env_cfg.py. Empty keeps the template's default."""
    usd_library: str = ""
    """Glyph asset library that bare glyph block names resolve against (e.g. "train_blocks_large"). Empty uses the default (train_blocks_large)."""


def generate_env_from_pddl(args: Args, pddl_path: str, output_dir: str):
    """Read ``pddl_path`` and codegen + (optionally) sample reset states for it."""
    predicates, block_names = parse_pddl_init(pddl_path)

    domain_package = import_utils.resolve_module_from_path(args.domain_path)
    if domain_package is None:
        raise ValueError(
            f"Could not find python package for {args.domain_path}. "
            "None of the folders correspond with any imported modules."
        )

    task_slug = task_slug_from_pddl_path(pddl_path)
    env_id = f"{task_slug}-v0"
    env_folder = os.path.join(output_dir, task_slug)
    if os.path.exists(env_folder):
        shutil.rmtree(env_folder)
    os.makedirs(env_folder, exist_ok=True)
    print(f"Created environment folder: {env_folder}")

    with open(pddl_path, "r") as f:
        pddl_text = f.read()
    goal_text = extract_goal_block(pddl_text)
    assert_goal_text_supported(goal_text, block_names)
    randomization_ranges = parse_pddl_randomization_ranges_from_text(pddl_text)

    # Where on-table blocks may spawn. The PDDL comments carry the per-problem
    # value; an explicit CLI range wins over it. Module-level in the sampler because
    # reset-state sampling instantiates predicates generically (name + args only).
    set_spawn_ranges(
        block_x_range=args.block_x_range or randomization_ranges.get("block_x_range"),
        block_y_range=args.block_y_range or randomization_ranges.get("block_y_range"),
    )

    # codegen samples the randomization profiles from the global RNG, so the
    # generated (block_size, table_height, friction) slots -- and therefore every
    # recorded cube_scale / table_height_delta -- are only reproducible if it is
    # seeded here, as generate_plans.py and execute_plans.py already do.
    np.random.seed(args.seed)

    # TODO - shared boilerplate with generate_task_from_dataset.py; extract helper
    env_info = generate_environment_code(
        domain_path=args.domain_path,
        task_name=task_slug,
        block_names=block_names,
        goal_texts=[goal_text],
        goal_display_name=os.path.basename(pddl_path),
        randomization_ranges=randomization_ranges,
        max_non_sim_state_randomizations=args.num_scene_profiles,
        scene_preset_id=args.scene_preset_id or None,
        robot_id=args.robot_id or None,
        usd_library=args.usd_library or None,
    )
    write_env_package(env_folder, env_info=env_info, env_id=env_id)
    if args.verbose:
        print(f"Generated env package at {env_folder} with env_id={env_id}")
    env_cfg_class_name = env_info["env_cfg_class_name"]

    if args.num_reset_states == 0:
        print("num_reset_states is 0: skipping sim setup and reset_states.pkl (env code only).")
        compile_ok = compileall.compile_dir(env_folder, quiet=not args.verbose, legacy=False)
        if not compile_ok:
            print("Warning: compileall failed for one or more files under the env folder.")
        return

    if args.num_envs < 1:
        raise ValueError("num_envs must be >= 1 when num_reset_states > 0.")

    if os.path.abspath(output_dir) not in sys.path:
        sys.path.insert(0, os.path.abspath(output_dir))
    # ``import toyblocks.tasks`` (top of this file) already imported
    # ``toyblocks.tasks.autogen``, so a FileFinder for this very directory is
    # sitting in sys.path_importer_cache with a directory listing taken before
    # write_env_package created env_folder. That finder only refills its cache
    # when the directory mtime changes, so on a filesystem with coarse mtime
    # granularity (weka/NFS: 1s) a package written within the same tick is
    # invisible and this import raises ModuleNotFoundError.
    importlib.invalidate_caches()
    importlib.import_module(f"{task_slug}")
    generated_env_cfg_module = importlib.import_module(f"{task_slug}.env_cfg")
    env_cfg_class: BaseEnvCfg = getattr(generated_env_cfg_module, env_cfg_class_name)
    env_cfg = env_cfg_class()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(id=env_id, cfg=env_cfg)
    base_env: BaseEnv = env.unwrapped
    env.reset(seed=args.seed)
    sim = SceneSim(env, block_names)
    # Re-apply the template / robot home pose before snapshotting original_reset_state
    # (make_superset_env.py resets twice the same way), otherwise reset_states.pkl
    # stores a mid-drift transient (wrong joint targets).
    env.reset(seed=args.seed)

    clear_blocks = extract_clear_blocks(predicates)
    if clear_blocks:
        print(f"Clear blocks (no blocks will be placed on top): {clear_blocks}")

    predicates = reorder_predicates_by_dependencies(predicates, clear_blocks)

    original_reset_state = base_env.get_state_dict()
    for k in original_reset_state["rigid_object"]:
        original_reset_state["rigid_object"][k]["root_velocity"] *= 0
    reset_states: list[dict] = []
    batch_size = args.num_envs
    num_reset_batches = args.num_reset_states // batch_size
    assert args.num_reset_states % args.num_envs == 0, (
        "currently behavior when number of reset states is not divisible by number of environments is not handled"
    )
    # Mix the task slug into the seed: the pipeline runs every problem with the same seed, and
    # the sampler's candidates barely depend on the scene, so a bare seed would give every
    # problem the same robot start configurations.
    rng = np.random.default_rng([args.seed, zlib.crc32(os.path.basename(env_folder).encode())])
    robot_init_sampler: CollisionFreeRobotInitSampler | None = None
    if args.randomize_robot_init:
        # Collision-free IK with the robot model and collision world vecgen plans with.
        robot_init_sampler = CollisionFreeRobotInitSampler(
            base_env,
            VecSim(env, torch.device(base_env.device)),
            RobotInitSamplingConfig(
                ee_x_range=tuple(args.robot_init_ee_x_range),
                ee_y_range=tuple(args.robot_init_ee_y_range),
                ee_z_min=args.robot_init_ee_z_min,
                num_candidates=args.robot_init_candidates,
                gripper_closed_prob=args.robot_init_gripper_closed_prob,
            ),
            rng,
        )

    def sample_predicates(env_list: list[int]) -> None:
        for pred in predicates:
            if pred["name"] in SKIP_PREDICATES:
                continue
            sampler = make_sampler(pred["name"], pred["args"])
            if sampler is None:
                if args.my_verbose:
                    print(f"Predicate {pred['name']} has no sampler, skipping")
                continue
            max_rounds = 200

            pending = list(env_list)
            for _ in range(max_rounds):
                if not pending:
                    break

                sampler.sample(sim, pending)
                for _ in range(5):
                    base_env.step(None)

                eval_results = sampler.evaluate(sim, pending)
                still_pending = []
                for env_idx, ok in zip(pending, eval_results):
                    if ok:
                        if args.my_verbose:
                            print(f"  [env {env_idx}] {pred['name']}({pred['args']}) OK")
                    else:
                        still_pending.append(env_idx)
                pending = still_pending

    # Blocks placed by on_table / on; relative predicates (next_to, on_right, ...) can
    # still push one out of the front camera's view, so those envs are resampled.
    block_names = sorted(
        {pred["args"][0] for pred in predicates if pred["name"] in ("on_table", "on")}
        | {pred["args"][1] for pred in predicates if pred["name"] == "on"}
    )
    block_edges = {
        name: torch.tensor(
            [sim.get_rigid_object(name).get_size(ei)[2] for ei in range(batch_size)],
            device=base_env.device,
            dtype=torch.float32,
        )
        for name in block_names
    }

    def envs_with_blocks_out_of_view() -> list[int]:
        visible = torch.ones(batch_size, dtype=torch.bool, device=base_env.device)
        for name in block_names:
            centers = sim.get_pose(name, env_idx=None, local=True)[:batch_size, :3]
            visible &= blocks_in_front_view(centers.to(base_env.device), block_edges[name])
        return (~visible).nonzero(as_tuple=True)[0].tolist()

    for _ in tqdm.tqdm(range(num_reset_batches)):
        base_env.set_state_dict(original_reset_state)

        sample_predicates(list(range(batch_size)))
        out_of_view = envs_with_blocks_out_of_view()
        for _ in range(MAX_OUT_OF_VIEW_RESAMPLES):
            if not out_of_view:
                break
            sample_predicates(out_of_view)
            out_of_view = envs_with_blocks_out_of_view()
        if out_of_view:
            print(
                f"Warning: {len(out_of_view)} env(s) still have a block outside the front "
                f"camera view after {MAX_OUT_OF_VIEW_RESAMPLES} resamples: {out_of_view}"
            )

        # After the blocks are placed, so the collision check sees the final scene.
        if robot_init_sampler is not None:
            robot_init_sampler.sample_and_apply()

        reset_state = base_env.get_state_dict(is_relative=True)
        reset_states.append(reset_state)
    reset_states = tree_utils.cat(reset_states)
    reset_states = tree_utils.reshape(reset_states, (args.num_reset_states, -1))

    reset_states_np = common_utils.to_numpy(reset_states)
    # Reset states are sampled one batch of ``num_envs`` at a time and concatenated,
    # so snapshot i was produced by env ``i % num_envs``. Record the profile that
    # env was actually spawned with (and that profile's values), so later stages
    # match each reset state to an env by value instead of recomputing an index.
    profile_ids = np.array(
        [base_env.env_slots[i % args.num_envs] for i in range(args.num_reset_states)],
        dtype=np.int64,
    )
    profiles = [base_env.scene_profile(int(pid)) for pid in sorted(set(profile_ids.tolist()))]
    with open(os.path.join(env_folder, "reset_states.pkl"), "wb") as f:
        pickle.dump(
            pack_reset_states(
                reset_states_np, profile_ids, args.num_scene_profiles, profiles=profiles
            ),
            f,
        )


def main(args: Args):
    # codegen samples block sizes and non-sim randomizations through the global
    # np.random state, which the local rng in generate_env_from_pddl does not cover.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        generate_env_from_pddl(args, args.pddl, args.output_dir)
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(130)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        # simulation_app.close() can hang, so we skip it and exit directly.
        # The OS reclaims resources when the process terminates.
        print("Exiting...")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
    simulation_app.close()
