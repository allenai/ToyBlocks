"""Build an autogen Isaac Lab task package directly from a LeRobot dataset.

Sibling to ``scripts/generate_task_from_pddl.py``: same env package output, no
PDDL bundle dependency. Reads goal expressions from the dataset's
``pddl_goal`` column and the block list from the
``observation.state.object_pose.<name>`` column schema, both of which datagen
already writes into every LeRobot recording. Datasets with multiple distinct
goals are supported: the generated env evaluates all goals and selects per
sub-env based on the queue setup's ``pddl_goal`` field at runtime (see
:meth:`BaseEnv.set_episode_goal`). Block-spawn shape and color are derived from
each block name by :meth:`toyblocks.tamp.blocks.template.codegen.Block.from_names`
(USD assets via ``USD_OVERRIDES``; every other name is a cuboid colored from a
color word in the name, or a rotating palette when it has none).

The dataset's recorded ``scene_id`` / ``robot_id`` provenance is baked into the
generated ``env_cfg.py`` (see
:func:`~toyblocks.utils.dataset_provenance.read_dataset_provenance`), so the
package targets the scene the data was collected in rather than the template's
default; a dataset with no recorded provenance is rejected.

Output package name + gym id are derived from the dataset directory basename
(``apr15_pnp`` -> ``apr15_pnp-v0``); pass ``--task-name`` to override. Always
runs in codegen-only mode (``num_reset_states`` is implicitly 0); reset diversity
comes from the queue's per-episode setups, not from re-sampling here.
"""

from __future__ import annotations

import tyro

from toyblocks.cli.isaac_app import launch_app

simulation_app = launch_app()
import compileall
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Tuple

from toyblocks.cli.base_args import BaseArgs
from toyblocks.domain_randomization.block_variables import (
    identity_assignment,
    parse_block_variable_map,
    source_goal,
)
from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import (
    assert_goal_atoms_supported,
    filter_goal_atoms,
)
from toyblocks.tamp.blocks.template.codegen import (
    assert_goal_text_supported,
    generate_environment_code,
    parse_pddl_goal_from_text,
    task_slug_from_dataset_path,
    write_env_package,
)
from toyblocks.utils import episode_dataset
from toyblocks.utils.dataset_provenance import read_dataset_provenance
from toyblocks.utils.episode_dataset import (
    SCENE_FURNITURE_PREFIXES,
    distinct_cube_scales,
    normalize_cube_scale,
)

_COL_OBJ_POSE_PREFIX = "observation.state.object_pose."
_COL_PDDL_GOAL = "pddl_goal"
_COL_EPISODE_INDEX = "episode_index"
_COL_CUBE_SCALE = "observation.state.cube_scale"
_COL_BLOCK_VARIABLE_MAP = "block_variable_map"


@dataclass
class Args(BaseArgs):
    """
    Build an autogen task package from a LeRobot dataset (no PDDL needed).

    Example command:

    python scripts/generate_task_from_dataset.py \
        --dataset-path data/datasets/merged/train/apr15_pnp \
        --domain-path src/toyblocks/tamp/blocks \
        --output-dir src/toyblocks/tasks/autogen --headless

    The output package folder, gym id, and class names use the dataset directory
    basename with hyphens replaced by underscores (override with ``--task-name``).
    """

    dataset_path: str = ""
    """LeRobot dataset directory (root with data.parquet, data/chunk-*/, or blocks/data/chunk-*/)."""
    domain_path: str = "src/toyblocks/tamp/blocks"
    """Path to the folder containing the env template + predicate definitions."""
    output_dir: str = "src/toyblocks/tasks/autogen"
    """Directory that will contain the generated task folder (slug)."""
    task_name: str = ""
    """Override the slug derived from the dataset basename (must be a valid Python identifier)."""
    episode_indices: str = ""
    """Optional episode filter (e.g. '0,3,5,10-19'). When set, only goals from these episodes are codegen'd."""
    usd_library: str = ""
    """Glyph asset library that bare glyph block names resolve against (e.g. "train_blocks_large"). Empty uses the default (train_blocks_large)."""


def _to_source_goal(bvm_col: Any, row_idx: int, goal: str) -> str:
    """``goal`` in source block names, through the row's ``block_variable_map`` (if any)."""
    if bvm_col is None:
        return goal
    identity = identity_assignment(parse_block_variable_map(bvm_col[row_idx].as_py()))
    return source_goal(goal, identity)


def read_task_spec_from_dataset(
    dataset_path: Path,
    *,
    episode_indices: list[int] | None = None,
) -> Tuple[List[str], List[str], dict, List[float]]:
    """Return ``(block_names, goal_texts, goal_episode_counts, cube_scales)`` from the dataset.

    ``block_names`` is the sorted list of suffixes of ``observation.state.object_pose.<name>``
    columns (the same set the queue's ``apply_setup`` writes to sim per episode).
    Derived from parquet schema metadata (no row reads).

    ``goal_texts`` is a sorted list of unique non-empty ``pddl_goal`` values found
    across all parquet files. Multi-goal datasets are supported: the generated env
    evaluates all goals and selects per sub-env based on the queue setup's
    ``pddl_goal`` field (see :meth:`BaseEnv.set_episode_goal`). On glyph/colour
    rerenders each goal is mapped back to source block names through the row's
    ``block_variable_map``, matching the grading goal the queue sends.

    ``goal_episode_counts`` maps each goal text to the number of episodes
    (``is_first==True`` rows) that carry it.

    ``cube_scales`` is the sorted list of per-episode recorded cube side lengths
    (``observation.state.cube_scale``), one per episode; empty when the dataset
    has no such column. Passed to codegen so each distinct scale gets its own
    spawn slot for exact per-episode block-size matching.

    When ``episode_indices`` is provided, only rows matching those episodes are
    scanned for goals (via pyarrow predicate pushdown). Block names always come
    from the schema regardless of episode filtering.

    Multi-root collections (a directory of LeRobot datasets) are rejected here:
    they may carry heterogeneous tasks and there's no safe way to pick one. Pass
    a single root if you need to retarget per child.
    """
    import pyarrow.parquet as pq

    roots = episode_dataset._discover_lerobot_roots(dataset_path)
    if not roots:
        raise FileNotFoundError(
            f"no LeRobot dataset (data.parquet, data/chunk-*/file-*.parquet, "
            f"or blocks/data/chunk-*/file-*.parquet) found at or under {dataset_path}"
        )
    if len(roots) > 1:
        raise ValueError(
            f"{dataset_path} resolves to {len(roots)} LeRobot roots: "
            f"{[str(r) for r in roots]}. Pass one root explicitly so the task spec "
            "is unambiguous."
        )
    parquet_files = episode_dataset._resolve_parquet_files(roots[0])
    if not parquet_files:
        raise FileNotFoundError(f"no parquet files found under {roots[0]}")

    # When episodes are filtered, project down to just the shards that hold
    # them (via meta/episodes data/chunk_index + data/file_index) so we don't
    # open every file on a high-latency filesystem.
    wanted = set(episode_indices) if episode_indices is not None else None
    scan_files = episode_dataset._data_shards_for_episodes(parquet_files, roots[0], wanted)

    # -- Block names from schema (no row reads) --
    block_names: list[str] | None = None
    for pq_path in scan_files:
        schema = pq.read_schema(str(pq_path))
        cols = schema.names

        object_cols = sorted(c for c in cols if c.startswith(_COL_OBJ_POSE_PREFIX))
        all_names = [c[len(_COL_OBJ_POSE_PREFIX) :] for c in object_cols]
        furniture = [n for n in all_names if n.startswith(SCENE_FURNITURE_PREFIXES)]
        names = [n for n in all_names if n not in furniture]
        if furniture:
            print(f"  [dataset] skipping scene-furniture object_pose columns: {furniture}")
        if not names:
            raise ValueError(
                f"{pq_path}: no {_COL_OBJ_POSE_PREFIX}* columns. "
                "This dataset doesn't carry the per-object poses required to "
                "spawn the env's blocks."
            )
        if block_names is None:
            block_names = names
        elif block_names != names:
            raise ValueError(
                f"{pq_path}: object_pose column set differs across parquet files: "
                f"{block_names} vs {names}"
            )

        if _COL_PDDL_GOAL not in cols:
            raise ValueError(
                f"{pq_path}: missing required column {_COL_PDDL_GOAL!r}. "
                "If your datagen pipeline doesn't write this column yet, fall back "
                "to scripts/generate_task_from_pddl.py until it does."
            )

    # -- Goal + cube-scale scan: column projection + optional episode filter --
    first_schema = pq.read_schema(str(scan_files[0])).names
    has_cube_scale = _COL_CUBE_SCALE in first_schema
    # Glyph/colour rerenders record goals in rendered names; the queue grades on the
    # source names (episode_dataset._row_to_setup), so compile those.
    has_block_variable_map = _COL_BLOCK_VARIABLE_MAP in first_schema
    read_columns = [_COL_PDDL_GOAL, _COL_EPISODE_INDEX]
    if has_cube_scale:
        read_columns.append(_COL_CUBE_SCALE)
    if has_block_variable_map:
        read_columns.append(_COL_BLOCK_VARIABLE_MAP)
    filters = None
    if episode_indices is not None:
        filters = [(_COL_EPISODE_INDEX, "in", episode_indices)]

    goal_counts: dict[str, int] = {}
    # episode_index -> recorded cube side length (one per episode; constant within).
    cube_scale_by_episode: dict[int, float] = {}
    # Episodes whose rows were actually read, collected from the projected
    # episode_index column rather than the goal branch below, so a requested
    # episode that was never scanned reports as missing instead of surfacing as
    # an "empty pddl_goal" error.
    scanned_episodes: set[int] = set()
    for pq_path in scan_files:
        table = pq.read_table(str(pq_path), columns=read_columns, filters=filters)
        if table.num_rows == 0:
            continue

        goal_col = table.column(_COL_PDDL_GOAL)
        cube_col = table.column(_COL_CUBE_SCALE) if has_cube_scale else None
        bvm_col = table.column(_COL_BLOCK_VARIABLE_MAP) if has_block_variable_map else None

        ep_col = (
            table.column(_COL_EPISODE_INDEX)
            if _COL_EPISODE_INDEX in table.column_names
            else None
        )
        if ep_col is not None:
            for ep_val in ep_col.to_pylist():
                if isinstance(ep_val, (list, tuple)):
                    ep_val = ep_val[0] if ep_val else None
                if ep_val is not None:
                    scanned_episodes.add(int(ep_val))

        seen_episodes: set[Tuple[int, str]] = set()
        for row_idx in range(table.num_rows):
            v = goal_col[row_idx].as_py()
            if v is None:
                continue
            if isinstance(v, (list, tuple)):
                v = v[0] if v else None
            if isinstance(v, bytes):
                v = v.decode("utf-8")
            if not isinstance(v, str) or not v.strip():
                continue
            v = v.strip()
            if ep_col is not None:
                ep = ep_col[row_idx].as_py()
                if isinstance(ep, (list, tuple)):
                    # Defaulting an empty cell to 0 would file this row's goal and
                    # cube_scale under episode 0, corrupting both the goal set and
                    # the per-episode scale slots.
                    if not ep:
                        raise ValueError(
                            f"{dataset_path}: row {row_idx} has an empty "
                            f"{_COL_EPISODE_INDEX!r}, so its {_COL_PDDL_GOAL!r} cannot be "
                            "attributed to an episode."
                        )
                    ep = ep[0]
                ep = int(ep)
                if cube_col is not None and ep not in cube_scale_by_episode:
                    cs = cube_col[row_idx].as_py()
                    if isinstance(cs, (list, tuple)):
                        cs = cs[0] if cs else None
                    if cs is not None:
                        cube_scale_by_episode[ep] = normalize_cube_scale(cs)
                key = (ep, v)
                if key in seen_episodes:
                    continue
                seen_episodes.add(key)
                v = _to_source_goal(bvm_col, row_idx, v)
            else:
                v = _to_source_goal(bvm_col, row_idx, v)
                if v in goal_counts:
                    continue
            goal_counts[v] = goal_counts.get(v, 0) + 1

    if episode_indices is not None:
        missing = sorted(set(episode_indices) - scanned_episodes)
        if missing:
            raise ValueError(
                f"{dataset_path}: requested episode_indices not found in dataset: "
                f"{missing} (scanned {len(scanned_episodes)} episode(s) across "
                f"{len(scan_files)} data shard(s)). Generating from the remaining "
                "episodes would bake the wrong goal set and the wrong cube-scale "
                "spawn slots into the env."
            )

    if not goal_counts:
        raise ValueError(
            f"{dataset_path}: {_COL_PDDL_GOAL!r} is empty in every row across all parquet files."
        )

    assert block_names is not None
    goal_texts = sorted(goal_counts.keys())
    cube_scales = sorted(cube_scale_by_episode.values())
    return block_names, goal_texts, goal_counts, cube_scales


def _is_valid_identifier(name: str) -> bool:
    return name.isidentifier()


def generate_env_from_dataset(args: Args) -> None:
    if not args.dataset_path:
        raise ValueError("--dataset-path is required.")
    dataset_path = Path(args.dataset_path).resolve()

    ep_indices: list[int] | None = None
    if args.episode_indices.strip():
        ep_indices = episode_dataset.parse_episode_indices_spec(args.episode_indices)
        print(
            f"[codegen] filtering to {len(ep_indices)} episode(s): {ep_indices[:10]}{'...' if len(ep_indices) > 10 else ''}"
        )

    block_names, goal_texts, goal_counts, cube_scales = read_task_spec_from_dataset(
        dataset_path, episode_indices=ep_indices,
    )
    total_episodes = sum(goal_counts.values())
    print(
        f"[dataset spec] {len(goal_texts)} goal(s), {total_episodes} episodes, {len(block_names)} blocks: {block_names}"
    )
    for i, gt in enumerate(goal_texts):
        print(f"  goal[{i}] ({goal_counts[gt]} episodes): {gt!r}")
    if cube_scales:
        slots = distinct_cube_scales(cube_scales)
        print(
            f"[cube scale] {len(slots)} distinct recorded scale(s) -> "
            f"{len(slots)} spawn slot(s): {[round(s, 4) for s in slots]}. "
            f"Eval num_envs must be a positive multiple of {len(slots)}."
        )
    else:
        print("[cube scale] no observation.state.cube_scale column; using codegen default block size.")

    # Validate (parse + ground quantifiers + check supported predicates) before
    # touching the output folder, so a bad goal fails loudly without side effects.
    for gt in goal_texts:
        assert_goal_text_supported(gt, block_names)

    task_slug = args.task_name.strip() or task_slug_from_dataset_path(str(dataset_path))
    if not _is_valid_identifier(task_slug):
        raise ValueError(
            f"task slug {task_slug!r} is not a valid Python identifier "
            "(use --task-name to override; valid chars: letters, digits, underscore; "
            "cannot start with a digit)."
        )
    env_id = f"{task_slug}-v0"
    env_folder = os.path.join(args.output_dir, task_slug)
    if os.path.exists(env_folder):
        shutil.rmtree(env_folder)
    os.makedirs(env_folder, exist_ok=True)
    print(f"Created environment folder: {env_folder}")

    # The scene preset and robot ARE baked in (the template config is copied
    # verbatim otherwise), so a package generated for a droid_with_box_table
    # dataset would default to the template's sim2real and only the runtime
    # reconcile in evaluate.py would rescue it. Raise when the dataset recorded
    # nothing: the same contract as reconcile_env_cfg_with_dataset, since the
    # table baseline every recorded table_height_delta is relative to is exactly
    # what the preset decides.
    provenance = read_dataset_provenance(dataset_path)
    missing = [k for k in ("scene_preset_id", "robot_id") if not provenance.get(k)]
    if missing:
        raise ValueError(
            f"{dataset_path} records no {', '.join(missing)} (checked meta/info.json and "
            "the per-step provenance columns), so the generated env package cannot be "
            "pinned to the scene the data was collected in. Emitting the template's "
            "default would put the table at a different baseline than every recorded "
            "table_height_delta assumes. Record the provenance at datagen time; see "
            "docs/source/user_guide/workflows/evals/launch_eval.md 'Dataset scene provenance'."
        )
    print(
        f"[provenance] baking scene_preset_id={provenance['scene_preset_id']!r} "
        f"robot_id={provenance['robot_id']!r} (from {provenance.get('source')})"
    )

    # Table height is sim state, applied at runtime (write_root_pose_to_sim),
    # not a spawn-time property — so it is NOT baked here. The queue eval owns it
    # per-episode via episode_queue._apply_table_height (recorded
    # table_height_delta); codegen only bakes properties that require a fresh
    # spawn (block sizes/colors, cube-scale USD prototypes).
    env_info = generate_environment_code(
        domain_path=args.domain_path,
        task_name=task_slug,
        block_names=block_names,
        goal_texts=goal_texts,
        goal_display_name=str(dataset_path),
        randomization_ranges={},
        cube_scales=cube_scales,
        scene_preset_id=provenance["scene_preset_id"],
        robot_id=provenance["robot_id"],
        usd_library=args.usd_library or None,
    )
    write_env_package(env_folder, env_info=env_info, env_id=env_id)
    if args.verbose:
        print(f"Generated env package at {env_folder} with env_id={env_id}")

    compile_ok = compileall.compile_dir(env_folder, quiet=not args.verbose, legacy=False)
    if not compile_ok:
        print("Warning: compileall failed for one or more files under the env folder.")


def main(args: Args) -> None:
    try:
        generate_env_from_dataset(args)
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
        # simulation_app.close() can hang; rely on OS to reclaim on exit.
        print("Exiting...")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main(tyro.cli(Args))
