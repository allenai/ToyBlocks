"""Build one autogen env package that serves many PDDL problems (problem streaming).

The package spawns the union of the problems' blocks. For each problem, reset states are
sampled with the stage-1 predicate samplers (``toyblocks.vecgen.sampling``, no TAMP)
for that problem's blocks while the other blocks are parked on the ground plane behind
the robot (z = -1.05 m). The package is an
ordinary gym-registered env (same codegen as ``generate_task_from_pddl.py``) plus
``problems.json`` listing, per problem, its PDDL (copied into ``problems/``), its active
blocks and its reset-state id range; ``scripts/vecgen/run.py --env-path <pkg>`` streams
episodes over all of them in one process.

Example:
    python scripts/vecgen/make_superset_env.py --pddls tasks/*.pddl --out-dir outputs \\
        --name stack3_superset --num-reset-states 32 --num-envs 16 --headless
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import shutil
import sys
import zlib
from pathlib import Path

from toyblocks.cli.isaac_app import launch_app

simulation_app = launch_app()

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from toyblocks.tamp.blocks.camera_view import blocks_in_front_view  # noqa: E402
from toyblocks.tamp.blocks.template.codegen import (  # noqa: E402
    SKIP_PREDICATES,
    extract_clear_blocks,
    extract_goal_block,
    generate_environment_code,
    parse_pddl_init,
    parse_pddl_randomization_ranges_from_text,
    reorder_predicates_by_dependencies,
    write_env_package,
)
from toyblocks.utils import common_utils, tree_utils  # noqa: E402
from toyblocks.utils.reset_states_io import pack_reset_states  # noqa: E402
from toyblocks.utils.robot_init_sampling import (  # noqa: E402
    CollisionFreeRobotInitSampler,
    RobotInitSamplingConfig,
)
from toyblocks.vecgen.sampling import SceneSim, make_sampler  # noqa: E402
from toyblocks.vecgen.sampling import predicates as spawn_window  # noqa: E402
from toyblocks.vecgen.sim import VecSim  # noqa: E402
from toyblocks.vecgen.symbolic.pddl import parse_problem  # noqa: E402

GROUND_Z = -1.05
PARK_ROWS_X = (-0.7, -0.85, -1.0, -1.15, -1.3, -1.45)
PARK_COLS_Y = (-0.9, -0.75, -0.6, -0.45, -0.3, -0.15, 0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9)
MAX_OUT_OF_VIEW_RESAMPLES = 20


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pddls", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", required=True, help="package / task slug")
    ap.add_argument("--domain-path", default="src/toyblocks/tamp/blocks")
    ap.add_argument(
        "--num-reset-states", type=int, default=32, help="per problem (multiple of --num-envs)"
    )
    ap.add_argument("--num-envs", type=int, default=16)
    ap.add_argument("--num-scene-profiles", type=int, default=8)
    ap.add_argument("--usd-library", default="")
    # Same as generate_task_from_pddl.py's --block_x_range / --block_y_range: an explicit window
    # beats every member's ;block_x_range / ;block_y_range PDDL comment.
    ap.add_argument("--block-x-range", default="",
                    help="Table-frame x window for on-table spawns, e.g. '[-0.2, 0.58]' or "
                    "'[[-0.2, -0.1], [0.3, 0.5]]'. Empty: each problem's PDDL value, else the default")
    ap.add_argument("--block-y-range", default="",
                    help="Table-frame y window for on-table spawns (same format as --block-x-range)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    # Same as generate_task_from_pddl.py's --randomize_robot_init options.
    ap.add_argument("--randomize-robot-init", action="store_true",
                    help="Give each reset state a random collision-free arm configuration and gripper state")
    ap.add_argument("--robot-init-ee-x-range", type=float, nargs=2, default=[0.25, 0.65])
    ap.add_argument("--robot-init-ee-y-range", type=float, nargs=2, default=[-0.35, 0.35])
    ap.add_argument("--robot-init-ee-z-min", type=float, default=0.20)
    ap.add_argument("--robot-init-candidates", type=int, default=32)
    ap.add_argument("--robot-init-gripper-closed-prob", type=float, default=0.5)
    args, _ = ap.parse_known_args()
    return args


def park_pose(j: int) -> list[float]:
    row, col = divmod(j, len(PARK_COLS_Y))
    x = PARK_ROWS_X[min(row, len(PARK_ROWS_X) - 1)]
    return [x, PARK_COLS_Y[col], GROUND_Z + 0.03, 1.0, 0.0, 0.0, 0.0]


def main() -> int:
    args = parse_args()
    pddls = [Path(p) for p in args.pddls]
    problems = [parse_problem(p) for p in pddls]
    union = sorted({b for p in problems for b in p.blocks})
    print(f"{len(problems)} problems, {len(union)} blocks in the union: {union}", flush=True)
    out_dir = Path(args.out_dir)
    env_folder = out_dir / args.name
    if env_folder.exists():
        shutil.rmtree(env_folder)
    env_folder.mkdir(parents=True)
    (env_folder / "problems").mkdir()
    ranges = parse_pddl_randomization_ranges_from_text(pddls[0].read_text())
    env_id = f"{args.name}-v0"
    np.random.seed(args.seed)
    goal_text = extract_goal_block(pddls[0].read_text())  # the package needs one goal to compile
    env_info = generate_environment_code(
        domain_path=args.domain_path,
        task_name=args.name,
        block_names=union,
        goal_texts=[goal_text],
        goal_display_name=args.name,
        randomization_ranges=ranges,
        max_non_sim_state_randomizations=args.num_scene_profiles,
        usd_library=args.usd_library or None,
    )
    write_env_package(str(env_folder), env_info=env_info, env_id=env_id)
    # ---- sim
    if str(out_dir.resolve()) not in sys.path:
        sys.path.insert(0, str(out_dir.resolve()))
    importlib.invalidate_caches()
    importlib.import_module(args.name)
    cfg_mod = importlib.import_module(f"{args.name}.env_cfg")
    env_cfg = getattr(cfg_mod, env_info["env_cfg_class_name"])()
    env_cfg.scene.num_envs = args.num_envs
    env = gym.make(id=env_id, cfg=env_cfg)
    base_env = env.unwrapped
    env.reset(seed=args.seed)
    sim = SceneSim(env, union)
    env.reset(seed=args.seed)
    original = base_env.get_state_dict()
    for k in original["rigid_object"]:
        original["rigid_object"][k]["root_velocity"] *= 0
    batch = args.num_envs
    assert args.num_reset_states % batch == 0
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
            # Mix in the package name: every job builds its first package with the same --seed,
            # so a bare seed would give every package the same robot start configurations.
            np.random.default_rng([args.seed, zlib.crc32(args.name.encode())]),
        )
    all_states: list[dict] = []
    index: list[dict] = []
    # set_spawn_ranges only ever narrows the module window, so remember the default to restore
    # it for members whose PDDL names no window (else they inherit the previous member's).
    default_x = list(spawn_window.SPAWN_X_INTERVALS)
    default_y = list(spawn_window.SPAWN_Y_INTERVALS)
    for pi, (pddl, prob) in enumerate(zip(pddls, problems, strict=True)):
        shutil.copy(pddl, env_folder / "problems" / pddl.name)
        # Per member, like generate_task_from_pddl.py: CLI window, else this PDDL's, else default.
        member_ranges = parse_pddl_randomization_ranges_from_text(pddl.read_text())
        spawn_window.set_spawn_ranges(
            block_x_range=args.block_x_range or member_ranges.get("block_x_range") or default_x,
            block_y_range=args.block_y_range or member_ranges.get("block_y_range") or default_y,
        )
        predicates, _ = parse_pddl_init(str(pddl))
        clear_blocks = extract_clear_blocks(predicates)
        predicates = reorder_predicates_by_dependencies(predicates, clear_blocks)
        active = set(prob.blocks)
        parked = {
            "articulation": original["articulation"],
            "rigid_object": {
                k: {kk: vv.clone() for kk, vv in v.items()}
                for k, v in original["rigid_object"].items()
            },
        }
        for j, name in enumerate(union):
            if name not in active:
                pose = torch.tensor(park_pose(j), device=base_env.device)
                parked["rigid_object"][name]["root_pose"][:, :7] = pose
                parked["rigid_object"][name]["root_pose"][:, :3] += base_env.scene.env_origins
        block_edges = {
            name: torch.tensor(
                [sim.get_rigid_object(name).get_size(ei)[2] for ei in range(batch)],
                device=base_env.device,
                dtype=torch.float32,
            )
            for name in prob.blocks
        }

        def sample_predicates(env_list: list[int], predicates=predicates) -> None:
            for pred in predicates:
                if pred["name"] in SKIP_PREDICATES:
                    continue
                sampler = make_sampler(pred["name"], pred["args"])
                if sampler is None:
                    continue
                pending = list(env_list)
                for _ in range(200):
                    if not pending:
                        break
                    sampler.sample(sim, pending)
                    for _ in range(5):
                        base_env.step(None)
                    res = sampler.evaluate(sim, pending)
                    pending = [e for e, ok in zip(pending, res, strict=True) if not ok]

        def out_of_view(prob=prob, block_edges=block_edges) -> list[int]:
            vis = torch.ones(batch, dtype=torch.bool, device=base_env.device)
            for name in prob.blocks:
                centers = sim.get_pose(name, env_idx=None, local=True)[:batch, :3]
                vis &= blocks_in_front_view(centers.to(base_env.device), block_edges[name])
            return (~vis).nonzero(as_tuple=True)[0].tolist()

        start = len(all_states) * batch
        for _ in range(args.num_reset_states // batch):
            base_env.set_state_dict(parked)
            sample_predicates(list(range(batch)))
            bad = out_of_view()
            for _ in range(MAX_OUT_OF_VIEW_RESAMPLES):
                if not bad:
                    break
                sample_predicates(bad)
                bad = out_of_view()
            # After the blocks are placed, so the collision check sees the final scene.
            if robot_init_sampler is not None:
                robot_init_sampler.sample_and_apply()
            all_states.append(base_env.get_state_dict(is_relative=True))
        index.append(
            {
                "slug": pddl.stem.replace("-", "_"),
                "pddl": f"problems/{pddl.name}",
                "blocks": prob.blocks,
                "reset_states": [start, len(all_states) * batch],
            }
        )
        print(
            f"[{pi + 1}/{len(problems)}] {pddl.stem}: reset states {start}-{len(all_states) * batch}",
            flush=True,
        )
    states = tree_utils.reshape(tree_utils.cat(all_states), (len(all_states) * batch, -1))
    states_np = common_utils.to_numpy(states)
    n = len(all_states) * batch
    # Each state's profile is the spawn slot of the env that sampled it (env i % batch),
    # recorded with that slot's values so resets can match it to an env by value.
    profile_ids = np.array([base_env.env_slots[i % batch] for i in range(n)], dtype=np.int64)
    profiles = [base_env.scene_profile(int(pid)) for pid in sorted(set(profile_ids.tolist()))]
    with open(env_folder / "reset_states.pkl", "wb") as f:
        pickle.dump(
            pack_reset_states(states_np, profile_ids, args.num_scene_profiles, profiles=profiles),
            f,
        )
    (env_folder / "problems.json").write_text(
        json.dumps({"blocks": union, "problems": index}, indent=2)
    )
    print(f"wrote {env_folder} with {n} reset states for {len(problems)} problems", flush=True)
    print("SUPERSET_OK", flush=True)
    return 0


if __name__ == "__main__":
    rc = main()
    import threading

    threading.Timer(15.0, lambda: os._exit(rc)).start()
    try:
        simulation_app.close()
    finally:
        os._exit(rc)
