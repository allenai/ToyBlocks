"""Load queue-eval episode setups from a LeRobot dataset on disk.

Streams one setup per episode (first frame, ``is_first==True``, or a later
row when ``init_after_hl_action`` is set) out of a LeRobot dataset directory.
Each setup dict carries (all optional): the robot
pose (``robot_joint_positions`` = ``joint_position[:7]``, ``robot_gripper_position``),
``objects`` (per-block local ``position``/``quaternion`` from
``object_pose.<name>``), ``pddl_goal``, ``hl_prompt`` (parquet
``goal.hl_prompt``; when present this is the policy prompt, otherwise
English is derived from ``pddl_goal`` — see
:func:`~toyblocks.utils.prompt_resolver.parse_prompt_for_setup`),
``high_level_action`` (recorded PDDL action name at the setup row, e.g.
``"pnp_table_block"``; metadata for replay filtering and the oracle
reconstruction gate), expert-video metadata (``expert_videos``,
``expert_episode_length``), ``init_frame_index`` (0 unless a mid-episode
cut moved the start), and the episode/source ids. Plus these newer
scene-reconstruction keys:

- ``table_pose`` / ``has_table_column``: recorded ``object_pose.table*`` furniture
  pose, written to ``workspace_table`` by :func:`~toyblocks.utils.episode_queue.apply_setup`.
- ``table_height_delta``: table offset vs the preset baseline; reconstructs table
  height when there's no absolute table pose.
- ``scene_id`` / ``robot_id``: top-level per-step scene-identity columns (constant
  per dataset; see :func:`~toyblocks.utils.dataset_provenance.read_dataset_provenance`).
- ``block_identity`` / ``policy_pddl_goal``: on glyph/colour renders, the
  recorded ``block_variable_map`` / ``glyph_map``; ``pddl_goal`` is then mapped
  back to source block names and the recorded (rendered) goal kept as
  ``policy_pddl_goal``.
- ``cube_scale``: recorded isotropic block side length (m); codegen + queue spawn
  each env at the matching size (see :func:`distinct_cube_scales`).
- ``camera_intrinsics``: per-camera first-frame 3x3 K (row-major 9-vector),
  keyed by short camera name. Randomized-camera datasets vary K per episode;
  :func:`~toyblocks.utils.episode_queue.apply_setup` writes it per env.
- ``initial_pd_mode``: recorded first-frame ``action.pd_mode`` (0=compliant,
  1=stiff). Eval pre-applies the matching actuator gains at reset so the very
  first policy step runs under the impedance the checkpoint was trained on.

Accepted layouts (priority order): ``data.parquet`` at root,
``data/chunk-*/file-*.parquet``, or a directory of LeRobot datasets
(flattened in sorted child order).

Datasets without ``observation.state.object_pose.*`` (e.g. toyblocks's
in-house recorder) override only the robot pose; ``_reset_idx`` fills in
object positions. ``pyarrow`` is imported lazily inside :func:`load_setups`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from toyblocks.domain_randomization.block_variables import (
    BlockIdentity,
    identity_assignment,
    parse_block_variable_map,
    parse_glyph_map,
    source_goal,
)

logger = logging.getLogger(__name__)


def _noisy_warning(message: str, *args: Any) -> None:
    """Warn about missing expert-video data."""
    logger.warning(message, *args)


_COL_JOINT_POS = "observation.state.joint_position"
_COL_GRIPPER_POS = "observation.state.gripper_position"
_COL_PDDL_GOAL = "pddl_goal"
# Recorded per-episode language instruction (first frame). When present this is
# the policy / wandb prompt as-is; otherwise parse_prompt_for_setup derives
# English from pddl_goal. Read into setup["hl_prompt"] by _row_to_setup.
_COL_HL_PROMPT = "goal.hl_prompt"
# Glyph/colour renders rename pddl_goal into the rendered identities while the
# pose columns keep the source block names. These per-episode maps (see
# toyblocks.domain_randomization.block_variables) let _row_to_setup grade on the
# source names (setup["pddl_goal"]) and keep the rendered goal for the policy
# (setup["policy_pddl_goal"]).
_COL_GLYPH_MAP = "glyph_map"
_COL_BLOCK_VARIABLE_MAP = "block_variable_map"
# Recorded high-level action for the episode: the raw PDDL action name (e.g.
# "pnp_table_block", "reorient", "rotate"), written by TAMP's LeRobot recorder
# (IsaacDataRecorderLeRobot) from LLActionSpecV2.name -- NOT the oracle/Molmo
# language form ("flip back <360, 260>"), whose direction words and pixel target
# live in action.high_level_args. Datagen writes one per frame; eval reads the
# value at the setup row into setup["high_level_action"] (first frame, or the
# first remaining skill after --init-after-hl-action). Used by replay's
# high-level-action filtering and the oracle reconstruction gate.
_COL_HIGH_LEVEL_ACTION = "action.high_level_action"
# Its arguments; tells consecutive same-named actions (two pnps in a row) apart.
_COL_HIGH_LEVEL_ARGS = "action.high_level_args"
# Recorded symbolic HL state: the datagen planner's true-prop set (newline-joined
# PDDL literals like "(on_table block_r workspace_table)") for each frame. The
# first-frame value is the exact PDDL :init the datagen planned from, so the eval
# oracle rebuilds its problem from it verbatim instead of re-deriving :init
# geometrically. Read into setup["high_level_state"] by _row_to_setup.
_COL_HIGH_LEVEL_STATE = "observation.state.high_level_state"
# Provenance-only id of the source problem the episode was generated from
# (datagen derives it from the data dir name). Read into setup["source_problem"]
# for logging; not used to reconstruct the problem.
_COL_SOURCE_PROBLEM = "source_problem"
# Per-frame sub-skill label ("pick", "place on table", "flip forward", ...) and its
# front-camera target pixel (annotate_target_xy projects where the gripper ends the
# sub-skill). Read into setup["demo_goal_targets"] so the eval oracle conditions on
# the demonstration's own target where the plan has none (a table placement).
_COL_GOAL_ACTION = "goal.action"
_COL_GOAL_POSITION = "goal.position"
_COL_EPISODE_INDEX = "episode_index"
_COL_IS_FIRST = "is_first"
_COL_OBJ_POSE_PREFIX = "observation.state.object_pose."
_COL_IMAGE_PREFIX = "observation.images."
# Per-camera first-frame 3x3 K (row-major 9-vector). Randomized-camera renders
# vary K across episodes; eval applies this per env at reset instead of one
# dataset-wide matrix. Short name is the segment after the last ':' (same
# convention as dataset_provenance.camera_short_name).
_COL_CAMERA_INTRINSICS_PREFIX = "observation.camera_intrinsics."
# Per-camera first-frame pose [x, y, z, qw, qx, qy, qz] in the robot-base frame
# (world convention). Render replay seats env-mounted cameras on it; see
# toyblocks.eval.render_replay. Same short-name keying as the intrinsics.
_COL_CAMERA_POSE_PREFIX = "observation.camera_pose."
# Recorded depth video features. Kept strictly separate from the RGB prefix
# above (and surfaced on a separate ``expert_depth_videos`` setup key) so the
# "an expert first frame is never a depth map" invariant in wandb_logging holds
# by construction rather than by every consumer remembering to filter.
_COL_DEPTH_PREFIX = "observation.depth."
# Recorded table-height offset from the scene-preset baseline (datagen randomizes
# the table up/down by this delta). Lets eval reconstruct the table height even
# when there is no object_pose.table column (see apply_setup).
_COL_TABLE_HEIGHT_DELTA = "observation.state.table_height_delta"
# Scene-identity columns. Datagen records these as TOP-LEVEL per-step columns
# (not under observation.state.*, and NOT in meta/info.json, which only stores a
# generic robot_type). They are constant per dataset and anchor the recorded
# deltas/poses to the datagen scene; read into each setup here and resolved by
# toyblocks.utils.dataset_provenance.read_dataset_provenance.
_COL_SCENE_ID = "scene_id"
_COL_ROBOT_ID = "robot_id"
# Recorded isotropic cube side length (meters) for this episode's blocks. Block
# scale is non-sim-state (can't change at runtime), so eval must spawn each env
# at the recorded size; the codegen + queue use distinct_cube_scales() to map
# each distinct value to one spawn slot and route episodes to the matching env.
_COL_CUBE_SCALE = "observation.state.cube_scale"
# Per-episode contact material. Datagen randomizes block and table friction per
# episode and records them
_COL_BLOCK_FRICTION = "observation.state.friction_properties"
_COL_TABLE_FRICTION = "observation.state.table_friction"
# First-frame PD-mode command recorded by datagen (0=compliant, 1=stiff). Read
# into ``setup["initial_pd_mode"]`` so eval can pre-write the same actuator
# gains at reset that the policy was trained under at step 0 rather than
# hardcoding a value. Missing column / value leaves the key unset and the
# caller falls back to its own default.
_COL_ACTION_PD_MODE = "action.pd_mode"
# Recorded arm joint velocity (rich-schema datasets). The first-frame value is
# read into ``setup["robot_joint_velocities"]`` so ``--replay`` can restore the
# exact recorded frame-0 motion; policy evals ignore it (zero initial velocity).
_COL_JOINT_VEL = "observation.state.joint_velocity"

# Round cube scales to this many decimals (micrometers) before treating two as
# the same spawn slot — collapses float-repr noise without merging real sizes.
CUBE_SCALE_DECIMALS = 6

# Nominal block side length (metres) used as the multiplier base for old
# datasets (see normalize_cube_scale).
BASE_BLOCK_SIDE_M = 0.049
# A recorded cube_scale above this many metres can't be a real block side length
# (blocks are ~2-10 cm), so it's interpreted as a scale multiplier instead.
MAX_PLAUSIBLE_CUBE_SIDE_M = 0.2


def normalize_cube_scale(raw: float | None) -> float:
    """Return an absolute isotropic block side length (metres) from a recorded value.

    Current datagen records ``observation.state.cube_scale`` directly as a side
    length in metres (~0.02-0.10), which is what eval spawns at. Some older
    datasets instead recorded a *scale multiplier* relative to
    :data:`BASE_BLOCK_SIDE_M`, and the two units are indistinguishable for values
    in the plausible metric range (``0.05`` is both a 5 cm cube and a
    2.5 mm one). Guessing therefore risks spawning every block at the wrong size
    while the run still reports a success rate, so an out-of-range value raises
    with instructions instead of being reinterpreted. Also raises on
    ``None``/non-finite.
    """
    if raw is None:
        raise ValueError("cube_scale value is None; dataset recorded no scale to spawn at")
    value = float(raw)
    if not (value > 0.0 and value == value and value != float("inf")):
        raise ValueError(f"cube_scale value {raw!r} is not a positive finite number")
    if value > MAX_PLAUSIBLE_CUBE_SIDE_M:
        raise ValueError(
            f"recorded cube_scale {value!r} exceeds {MAX_PLAUSIBLE_CUBE_SIDE_M} m, so it "
            "cannot be a block side length. It is most likely a scale multiplier from "
            f"older datagen (side = value * {BASE_BLOCK_SIDE_M}), but that unit is "
            "indistinguishable from metres inside the plausible range, so eval will not "
            "guess: silently picking the wrong unit spawns every block at the wrong size "
            "and still yields a success rate. Regenerate the dataset recording "
            "observation.state.cube_scale in metres."
        )
    return value


def distinct_cube_scales(scales: Iterable[float]) -> list[float]:
    """Sorted, de-duplicated cube side lengths (rounded to CUBE_SCALE_DECIMALS).

    Shared source of truth for the per-slot spawn sizes: codegen bakes one env
    slot per entry (in this order) and the queue server routes each episode to
    the slot whose value matches its recorded ``cube_scale``. Both derive this
    from the SAME episode set, so the orderings agree. Raises on a ``None`` /
    non-finite value (a dataset that recorded no usable scale fails loudly).
    """
    out: set[float] = set()
    for s in scales:
        if s is None:
            raise ValueError("cube_scale value is None; dataset recorded no scale to spawn at")
        out.add(round(float(s), CUBE_SCALE_DECIMALS))
    return sorted(out)

# Object-pose columns whose suffix starts with any of these are scene
# furniture (spawned by the scene preset, e.g. ``droid_with_box_table`` spawns
# ``workspace_table`` / ``robot_table``), not manipulable blocks. Blocks are
# always named ``block_*``; everything matched here is excluded from the block
# set so codegen doesn't try to spawn the table as a cuboid (the preset already
# provides it). Used here and re-imported by generate_task_from_dataset.py.
SCENE_FURNITURE_PREFIXES: tuple[str, ...] = ("table", "workspace_table", "robot_table")


# CLI / PDDL aliases for ``--init-after-hl-action``. ``flip`` is the task
# family name; the recorded ``action.high_level_action`` value is ``reorient``.
# Matching always includes the token itself, so a column that literally says
# ``flip`` still matches ``--init-after-hl-action flip``.
_HL_ACTION_ALIASES: dict[str, str] = {
    "flip": "reorient",
}


def parse_init_after_hl_actions(spec: str) -> frozenset[str]:
    """Parse ``flip,rotate`` / ``flip|rotate`` into casefolded names to match.

    Each token matches itself and its alias (``flip`` → ``flip`` and
    ``reorient``). Raises on an empty spec.
    """
    if not spec.strip():
        raise ValueError(
            "init_after_hl_action is empty; omit the flag to start each episode at frame 0"
        )
    tokens = [part.strip() for part in spec.replace("|", ",").split(",")]
    tokens = [t for t in tokens if t]
    if not tokens:
        raise ValueError(
            f"init_after_hl_action {spec!r} contains no action names "
            "(use e.g. 'flip', 'rotate', or 'flip,rotate')"
        )
    names: set[str] = set()
    for token in tokens:
        key = token.lower()
        names.add(key)
        alias = _HL_ACTION_ALIASES.get(key)
        if alias is not None:
            names.add(alias.lower())
    return frozenset(names)


def _canonical_init_after_label(match_names: frozenset[str]) -> str:
    """Stable setup/log label: drop alias sources when the PDDL name is present."""
    shown = set(match_names)
    for source, target in _HL_ACTION_ALIASES.items():
        if source in shown and target in shown:
            shown.discard(source)
    return ",".join(sorted(shown))


def _hl_action_cell_str(raw: Any) -> str:
    """Normalize a parquet ``action.high_level_action`` cell to a plain string."""
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw).strip() if raw is not None else ""


def _hl_state_cell_str(table: Any, row: int) -> str:
    """Normalize ``observation.state.high_level_state`` at ``row`` to a plain string."""
    if _COL_HIGH_LEVEL_STATE not in table.column_names:
        return ""
    raw = table.column(_COL_HIGH_LEVEL_STATE)[row].as_py()
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return str(raw).strip() if raw is not None else ""


def _advance_init_row_for_hl_state_lag(
    table: Any,
    *,
    next_idx: int,
    last_match: int,
    start_row: int,
) -> int | None:
    """Skip one row when a new HL action label appears before state updates.

    LeRobot rows pair observation ``high_level_state`` with the *next* command in
    ``high_level_action`` (see ``test_recorder_state_action_alignment``). After
    cutting on action runs, the first row past the match can therefore show the
    next skill's label while symbolic state still matches the prior frame (e.g.
    flip→pnp on ``compliant_test_flip_pnp``).
    """
    if start_row >= next_idx or start_row != last_match + 1:
        return start_row
    col = table.column(_COL_HIGH_LEVEL_ACTION)
    action_at_run_end = _hl_action_cell_str(col[last_match].as_py())
    action_at_cut = _hl_action_cell_str(col[start_row].as_py())
    if not action_at_cut or action_at_cut == action_at_run_end:
        return start_row
    state_at_run_end = _hl_state_cell_str(table, last_match)
    state_at_cut = _hl_state_cell_str(table, start_row)
    if not state_at_run_end or state_at_cut != state_at_run_end:
        return start_row
    lag_row = start_row + 1
    if lag_row >= next_idx:
        return None
    return lag_row


def _block_identity_at(table: Any, row: int, source: str) -> BlockIdentity:
    """The episode's recorded glyph/colour identities; empty for datasets without the maps."""
    cols = table.column_names
    try:
        bvm = (parse_block_variable_map(table.column(_COL_BLOCK_VARIABLE_MAP)[row].as_py())
               if _COL_BLOCK_VARIABLE_MAP in cols else {})
        glyph_map = (parse_glyph_map(table.column(_COL_GLYPH_MAP)[row].as_py())
                     if _COL_GLYPH_MAP in cols else {})
        return identity_assignment(bvm, glyph_map)
    except ValueError as exc:
        raise ValueError(f"{source} row {row}: {exc}") from exc


def _first_skill_hl_action(table: Any, row: int) -> str:
    """Recorded ``action.high_level_action`` at ``row``, skipping vecgen's ``init`` frames.

    The vecgen recorder labels an episode's first frames ``init`` until the env's
    first skill starts (1-3 frames on ~1/3 of vecgen episodes). ``init`` is not a
    PDDL action, so return the first later label of the same episode instead.
    """
    col = table.column(_COL_HIGH_LEVEL_ACTION)
    episode = _episode_index_at(table, row)
    name = _hl_action_cell_str(col[row].as_py())
    i = row
    while name == "init" and i + 1 < table.num_rows and _episode_index_at(table, i + 1) == episode:
        i += 1
        name = _hl_action_cell_str(col[i].as_py())
    return "" if name == "init" else name


def _episode_index_at(table: Any, row: int) -> int | None:
    """``episode_index`` at ``row``, or ``None`` when the column/cell is absent."""
    if _COL_EPISODE_INDEX not in table.column_names:
        return None
    raw = table.column(_COL_EPISODE_INDEX)[row].as_py()
    if isinstance(raw, (list, tuple)):
        raw = raw[0] if raw else None
    if raw is None:
        return None
    return int(raw)


def _scan_hl_action_runs(
    table: Any, first_idx: int, next_idx: int
) -> list[tuple[str, int, int]]:
    """Run-length encode ``action.high_level_action`` over ``[first_idx, next_idx)``.

    Each entry is ``(name, abs_start, abs_end_inclusive)``. Empty cells are
    kept as ``""`` so gaps show up in the log.
    """
    col = table.column(_COL_HIGH_LEVEL_ACTION)
    runs: list[tuple[str, int, int]] = []
    for i in range(first_idx, next_idx):
        name = _hl_action_cell_str(col[i].as_py())
        if runs and runs[-1][0] == name and runs[-1][2] == i - 1:
            start = runs[-1][1]
            runs[-1] = (name, start, i)
        else:
            runs.append((name, i, i))
    return runs


def _expert_hl_calls(table: Any, first_idx: int, next_idx: int) -> int:
    """HL commands the expert issued over ``[first_idx, next_idx)``, counted as Molmo2 calls.

    One per recorded (action, args) run, a ``pnp`` counting twice (Molmo2 says pick, then
    place), plus the final ``done``. vecgen's ``init`` frames and empty cells are not actions.
    A dataset without recorded HL actions counts only that ``done``.
    """
    if _COL_HIGH_LEVEL_ACTION not in table.column_names:
        return 1
    names = table.column(_COL_HIGH_LEVEL_ACTION)
    args = table.column(_COL_HIGH_LEVEL_ARGS) if _COL_HIGH_LEVEL_ARGS in table.column_names else None
    calls, prev = 1, None
    for i in range(first_idx, next_idx):
        name = _hl_action_cell_str(names[i].as_py())
        key = (name, None if args is None else str(args[i].as_py()))
        if name not in ("", "init") and key != prev:
            calls += 2 if name.startswith("pnp") else 1
        prev = key
    return calls


def _init_row_after_hl_action(
    table: Any,
    *,
    first_idx: int,
    next_idx: int,
    actions: frozenset[str],
    label: str,
    source: str,
) -> int | None:
    """Return the first row after every instance of any name in ``actions``.

    ``actions`` are casefolded recorded names (already aliased). No match
    returns ``first_idx`` (start the episode at frame 0). A last match on the
    episode's final frame returns ``None`` (nothing left; caller skips).
    Raises only when the column is missing. Logs the recorded action runs and
    the cut decision at INFO so a queue load can be audited episode by episode.
    """
    ep_idx = _episode_index_at(table, first_idx)
    ep_label = f"episode_index={ep_idx}" if ep_idx is not None else source
    n_frames = next_idx - first_idx
    if _COL_HIGH_LEVEL_ACTION not in table.column_names:
        raise ValueError(
            f"{source} {ep_label}: --init-after-hl-action requires column "
            f"{_COL_HIGH_LEVEL_ACTION!r}, which is missing"
        )
    runs = _scan_hl_action_runs(table, first_idx, next_idx)
    unique = [name for name, _, _ in runs if name]
    # Preserve first-seen order, drop empties from the unique list.
    seen: set[str] = set()
    unique_ordered: list[str] = []
    for name in unique:
        if name not in seen:
            seen.add(name)
            unique_ordered.append(name)

    last_match: int | None = None
    last_match_name = ""
    run_lines: list[str] = []
    for name, abs_start, abs_end in runs:
        rel0 = abs_start - first_idx
        rel1 = abs_end - first_idx
        count = abs_end - abs_start + 1
        folded = name.lower()
        matched = bool(name) and folded in actions
        if matched:
            last_match = abs_end
            last_match_name = name
        display = name if name else "(empty)"
        mark = "  MATCH" if matched else ""
        run_lines.append(f"    [{rel0}:{rel1}] {display} ({count} frame(s){mark})")

    if last_match is None:
        decision = (
            f"START_AT_0 — none of {sorted(actions)} appear in this episode; "
            f"using the full {n_frames} frame(s)"
        )
        start_row: int | None = first_idx
    else:
        action_cut_row = last_match + 1
        start_row = action_cut_row
        if start_row < next_idx and _COL_HIGH_LEVEL_STATE in table.column_names:
            aligned = _advance_init_row_for_hl_state_lag(
                table,
                next_idx=next_idx,
                last_match=last_match,
                start_row=start_row,
            )
            state_lag_frames = (
                (aligned - action_cut_row) if aligned is not None else None
            )
            start_row = aligned
        else:
            state_lag_frames = None
        last_rel = last_match - first_idx
        if start_row is None or start_row >= next_idx:
            decision = (
                f"SKIP — last match is {last_match_name!r} at episode frame "
                f"{last_rel}/{n_frames - 1} (the last frame); nothing left after the cut"
            )
            start_row = None
        else:
            init_frame = start_row - first_idx
            remaining = next_idx - start_row
            lag_note = ""
            if state_lag_frames:
                lag_note = (
                    f", advanced +{state_lag_frames} frame(s) for high_level_state lag"
                )
            decision = (
                f"CUT — last match is {last_match_name!r} at episode frame "
                f"{last_rel}; start at frame {init_frame} "
                f"(action cut last_match+1={action_cut_row - first_idx}{lag_note}, "
                f"remaining {remaining}/{n_frames} frames)"
            )

    logger.info(
        "init-after-hl-action %s (%s, %d frame(s), label=%r, looking for any of %s):\n"
        "  recorded high_level_action values (first-seen order): %s\n"
        "  runs (episode-frame [lo:hi], MATCH = in the search set):\n%s\n"
        "  decision: %s",
        ep_label,
        source,
        n_frames,
        label,
        sorted(actions),
        unique_ordered if unique_ordered else "(none — every cell empty)",
        "\n".join(run_lines) if run_lines else "    (no rows)",
        decision,
    )
    return start_row


def _seek_expert_clips_to_init(
    setup: dict,
    *,
    init_frame_index: int,
    dataset_fps: float,
) -> None:
    """Seek expert RGB/depth clips to ``init_frame_index`` in place.

    Packed mp4s that already have ``from_timestamp`` and ``to_timestamp`` use
    the clip's own duration (``(to-from) * init/original_len``), so a video
    clock that disagrees with ``meta/info.json`` fps still lands on the cut.
    Per-episode mp4s have no span; those use ``init_frame_index / dataset_fps``.
    ``to_timestamp`` stays at the episode end. Still-image parquet paths are
    left alone — a timestamp is not a seek into a PNG.
    """
    if init_frame_index <= 0:
        return
    remaining = int(setup["expert_episode_length"])
    original_len = init_frame_index + remaining
    if original_len <= 0:
        raise ValueError(
            f"episode {setup.get('episode_id')!r} has original_len={original_len} "
            f"(init_frame_index={init_frame_index}, expert_episode_length={remaining})"
        )
    ep = setup.get("episode_index", setup.get("episode_id"))
    for key in ("expert_videos", "expert_depth_videos"):
        clips = setup.get(key)
        if not clips:
            continue
        for clip_name, entry in clips.items():
            path = str(entry.get("path") or "")
            is_video = "from_timestamp" in entry or path.lower().endswith(".mp4")
            if not is_video:
                logger.info(
                    "init-after-hl-action episode_index=%s: %s %s is not a video "
                    "clip; leaving timestamps unset (path=%s)",
                    ep,
                    key,
                    clip_name,
                    path or "(none)",
                )
                continue
            has_span = "from_timestamp" in entry and "to_timestamp" in entry
            if has_span:
                start = float(entry["from_timestamp"])
                end = float(entry["to_timestamp"])
                duration = end - start
                if duration <= 0.0:
                    raise ValueError(
                        f"expert clip {path!r} has non-positive duration "
                        f"(from_timestamp={start}, to_timestamp={end})"
                    )
                new_from = start + duration * (init_frame_index / original_len)
                logger.info(
                    "init-after-hl-action episode_index=%s: seeking %s %s by clip "
                    "span (frame %d/%d of [%.4fs, %.4fs] → from_timestamp=%.4fs, "
                    "to_timestamp=%.4fs) path=%s",
                    ep,
                    key,
                    clip_name,
                    init_frame_index,
                    original_len,
                    start,
                    end,
                    new_from,
                    end,
                    path,
                )
                entry["from_timestamp"] = new_from
            else:
                new_from = init_frame_index / dataset_fps
                logger.info(
                    "init-after-hl-action episode_index=%s: seeking %s %s by "
                    "dataset fps (frame %d / %.4g Hz → from_timestamp=%.4fs) path=%s",
                    ep,
                    key,
                    clip_name,
                    init_frame_index,
                    dataset_fps,
                    new_from,
                    path,
                )
                entry["from_timestamp"] = new_from


def load_setups(
    path: str | Path,
    *,
    limit: int | None = None,
    episode_indices: Iterable[int] | None = None,
    allow_duplicate_episodes: bool = False,
    init_after_hl_action: str | None = None,
) -> list[dict]:
    """Load one setup per episode from the LeRobot dataset at ``path``.

    ``path`` is a LeRobot root or a directory of LeRobot roots.
    ``episode_indices`` filters by ``episode_index`` (raises on missing); for
    multi-root layouts this matches across all children — see
    ``docs/source/user_guide/workflows/evals/launch_eval.md``. ``limit`` (applied after filtering) caps output.

    When *allow_duplicate_episodes* is True and ``episode_indices`` contains
    repeated values, the corresponding setup is duplicated in the output list
    (each copy is a shallow dict copy with an appended ``_repeat_N`` suffix on
    ``episode_id`` to keep provenance distinct).

    ``init_after_hl_action`` (e.g. ``flip``, ``flip,rotate``, ``flip|rotate``)
    starts each episode at the first frame after every instance of any named
    high-level action (``flip`` matches both ``flip`` and ``reorient``). Reset
    poses, remaining ``expert_episode_length``, and expert-clip
    ``from_timestamp`` all come from that cut. ``pddl_goal`` / ``hl_prompt``
    stay the episode's original task strings. An episode with no matching
    frames starts at frame 0. An episode whose last match is its last frame
    is skipped (nothing left after the cut). Raises if the column is missing.
    Empty / ``None`` keeps today's frame-0 start. Each setup stores
    ``init_frame_index`` (0 when the flag is off or no match) and, when the
    flag is set, the resolved ``init_after_hl_action`` label.
    """
    p = Path(path)
    if not p.is_dir():
        raise NotADirectoryError(
            f"dataset path must be a directory containing a LeRobot dataset: {p}"
        )

    roots = _discover_lerobot_roots(p)
    if not roots:
        raise FileNotFoundError(
            f"no LeRobot dataset (data.parquet, data/chunk-*/file-*.parquet, "
            f"or blocks/data/chunk-*/file-*.parquet) found at or under {p}"
        )

    indices_list: list[int] | None = None
    if episode_indices is not None:
        indices_list = list(episode_indices)
    wanted: set[int] | None = None
    if indices_list is not None:
        wanted = set(indices_list)

    cut_names: frozenset[str] | None = None
    cut_label: str | None = None
    if init_after_hl_action is not None and init_after_hl_action.strip():
        cut_names = parse_init_after_hl_actions(init_after_hl_action)
        cut_label = _canonical_init_after_label(cut_names)
        logger.info(
            "init-after-hl-action: spec=%r → match any of %s (stored label=%r). "
            "Per episode: no match → start at frame 0; last match on the last "
            "frame → skip; otherwise start at last_match+1.",
            init_after_hl_action.strip(),
            sorted(cut_names),
            cut_label,
        )

    setups: list[dict] = []
    skipped_empty: set[int] = set()
    for root in roots:
        parquet_files = _resolve_parquet_files(root)
        root_setups, root_skipped = _load_setups_from_lerobot(
            parquet_files,
            root=root,
            wanted_episode_indices=wanted,
            init_after_actions=cut_names,
            init_after_label=cut_label,
        )
        skipped_empty.update(root_skipped)
        root_str = str(root.resolve())
        # v3 metadata enriches with from_timestamp / to_timestamp (needed for
        # shared video files where multiple episodes live in one mp4).
        video_meta, depth_meta = _load_episode_video_metadata(root, episode_indices=wanted)
        if video_meta:
            logger.info(
                "Resolved expert video metadata for %d episode(s) under %s",
                len(video_meta),
                root,
            )
        else:
            _noisy_warning(
                "%s produced no expert video metadata from meta/info.json, "
                "meta/episodes, or observation.images.* parquet rows; W&B "
                "expert_first_frame will be absent unless row image paths resolve.",
                root,
            )
        fps: float | None = None
        for s in root_setups:
            s["source_dataset_path"] = root_str
            ep_idx = s.get("episode_index")
            if ep_idx is not None and ep_idx in video_meta:
                s["expert_videos"] = video_meta[ep_idx]
            if ep_idx is not None and ep_idx in depth_meta:
                s["expert_depth_videos"] = depth_meta[ep_idx]
            init_frame = int(s.get("init_frame_index") or 0)
            if init_frame > 0:
                if fps is None:
                    fps = dataset_fps(root)
                _seek_expert_clips_to_init(
                    s, init_frame_index=init_frame, dataset_fps=fps
                )
        missing_expert_videos = sum(1 for s in root_setups if not s.get("expert_videos"))
        if missing_expert_videos:
            _noisy_warning(
                "%s: %d/%d loaded setup(s) have no expert_videos; those rows "
                "will not get expert_first_frame.",
                root,
                missing_expert_videos,
                len(root_setups),
            )
        if cut_label is not None and root_setups:
            frames = [int(s["init_frame_index"]) for s in root_setups]
            logger.info(
                "%s: %d setup(s) start after %s (init_frame_index min=%d max=%d)",
                root,
                len(root_setups),
                cut_label,
                min(frames),
                max(frames),
            )
        if len(roots) > 1:
            # Each child has its own episode_index numbering — namespace by root.
            for s in root_setups:
                s["episode_id"] = f"{root.name}/{s['episode_id']}"
        setups.extend(root_setups)

    if skipped_empty:
        logger.warning(
            "skipped %d episode(s) with nothing left after %s: %s",
            len(skipped_empty),
            cut_label,
            sorted(skipped_empty),
        )

    if wanted is not None:
        present = {s["episode_index"] for s in setups if "episode_index" in s}
        missing = wanted - present - skipped_empty
        if missing:
            raise ValueError(
                f"requested episode_indices not found in dataset: "
                f"{sorted(missing)} (dataset had {len(setups)} setups, "
                f"episode_index range "
                f"{min(present) if present else 'n/a'}..{max(present) if present else 'n/a'})"
            )
    if not setups:
        extra = ""
        if skipped_empty:
            extra = (
                f" ({len(skipped_empty)} episode(s) skipped because nothing remained "
                f"after {cut_label})"
            )
        if wanted is not None:
            raise ValueError(
                f"episode-indices filter {sorted(wanted)} matched no setups{extra}"
            )
        raise ValueError(f"loaded 0 setups{extra}")

    # Expand duplicates: if the spec requests the same index N times, produce
    # N copies of its setup (each with a distinct episode_id suffix).
    if allow_duplicate_episodes and indices_list is not None:
        from collections import Counter

        counts = Counter(indices_list)
        has_duplicates = any(c > 1 for c in counts.values())
        if has_duplicates:
            by_idx: dict[int, dict] = {
                s["episode_index"]: s for s in setups if "episode_index" in s
            }
            expanded: list[dict] = []
            for idx in indices_list:
                base = by_idx.get(idx)
                if base is None:
                    continue
                copy = dict(base)
                rep_num = sum(1 for e in expanded if e.get("episode_index") == idx)
                if rep_num > 0:
                    copy["episode_id"] = f"{base.get('episode_id', str(idx))}_repeat{rep_num}"
                expanded.append(copy)
            setups = expanded

    if limit is not None:
        if limit < 0:
            raise ValueError(f"limit must be non-negative, got {limit}")
        setups = setups[:limit]
    return setups


def parse_episode_indices_spec(
    spec: str,
    *,
    allow_duplicates: bool = False,
) -> list[int]:
    """Parse '0,3,5,10-19' into a sorted, deduplicated list of ints.

    Supports comma-separated entries and inclusive ``lo-hi`` ranges. A
    ``@PATH`` prefix reads entries from ``PATH``. Raises on malformed input.

    When *allow_duplicates* is True, repeated entries are preserved in order
    (e.g. ``'0,0,0'`` → ``[0, 0, 0]``). Ranges are still expanded but not
    repeated.
    """
    spec = spec.strip()
    if not spec:
        raise ValueError("empty episode-indices spec")
    if spec.startswith("@"):
        text = Path(spec[1:]).read_text()
        items = [s for s in text.replace("\n", ",").split(",")]
    else:
        items = spec.split(",")

    if allow_duplicates:
        out_list: list[int] = []
        for raw in items:
            item = raw.strip()
            if not item:
                continue
            if "-" in item:
                lo_s, hi_s = item.split("-", 1)
                try:
                    lo, hi = int(lo_s), int(hi_s)
                except ValueError as e:
                    raise ValueError(f"bad range {item!r} in episode-indices: {e}") from None
                if hi < lo:
                    raise ValueError(f"bad range {item!r}: hi < lo")
                out_list.extend(range(lo, hi + 1))
            else:
                try:
                    out_list.append(int(item))
                except ValueError as e:
                    raise ValueError(f"bad token {item!r} in episode-indices: {e}") from None
        if not out_list:
            raise ValueError(f"no episode indices parsed from spec {spec!r}")
        return out_list

    out: set[int] = set()
    for raw in items:
        item = raw.strip()
        if not item:
            continue
        if "-" in item:
            lo_s, hi_s = item.split("-", 1)
            try:
                lo, hi = int(lo_s), int(hi_s)
            except ValueError as e:
                raise ValueError(f"bad range {item!r} in episode-indices: {e}") from None
            if hi < lo:
                raise ValueError(f"bad range {item!r}: hi < lo")
            out.update(range(lo, hi + 1))
        else:
            try:
                out.add(int(item))
            except ValueError as e:
                raise ValueError(f"bad token {item!r} in episode-indices: {e}") from None
    if not out:
        raise ValueError(f"no episode indices parsed from spec {spec!r}")
    return sorted(out)


def _discover_lerobot_roots(path: Path) -> list[Path]:
    """Return one or more LeRobot dataset roots reachable from ``path``."""
    if _is_lerobot_root(path):
        return [path]
    return [c for c in sorted(path.iterdir()) if c.is_dir() and _is_lerobot_root(c)]


def _is_lerobot_root(path: Path) -> bool:
    if not path.is_dir():
        return False
    if (path / "data.parquet").is_file():
        return True
    if (path / "data").is_dir() and any((path / "data").glob("chunk-*/file-*.parquet")):
        return True
    blocks_data = path / "blocks" / "data"
    return blocks_data.is_dir() and any(blocks_data.glob("chunk-*/file-*.parquet"))


_VIDEO_KEY_PREFIX = "videos/"
_VIDEO_KEY_FIELDS = ("/chunk_index", "/file_index", "/from_timestamp", "/to_timestamp")


def _load_episode_video_metadata(
    root: Path,
    *,
    episode_indices: set[int] | None = None,
) -> tuple[dict[int, dict[str, dict]], dict[int, dict[str, dict]]]:
    """Read LeRobot v3 ``meta/episodes/`` parquet for per-episode mp4 metadata.

    Returns ``(rgb, depth)``, each ``{episode_index: {video_key: {path,
    from_timestamp, to_timestamp}}}`` and each empty when ``meta/episodes/`` is
    absent or lacks that modality's video columns. Depth entries additionally
    carry ``channels`` and ``depth_scale`` from the feature spec, which identify
    metric clips and convert their encoded units.

    When ``episode_indices`` is given, a pyarrow predicate restricts the scan to
    those episodes, so the per-row ``is_file()`` resolution below runs only for
    requested episodes instead of all rows.
    """
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.is_dir():
        logger.info("%s has no meta/episodes directory for expert video lookup", root)
        return {}, {}
    parquet_files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
    if not parquet_files:
        _noisy_warning("%s has meta/episodes but no chunk-*/file-*.parquet shards", root)
        return {}, {}

    import pyarrow.parquet as pq

    info = _load_lerobot_info(root)
    info_video_keys = _video_keys_from_info(info) + _video_keys_from_info(
        info, prefix=_COL_DEPTH_PREFIX
    )
    depth_channels = _depth_feature_channels(info)
    depth_scales = _depth_feature_scales(info)
    video_path_template = info.get("video_path") or (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    if not info.get("video_path"):
        logger.info(
            "%s has no meta/info.json video_path template; trying LeRobot "
            "default videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
            root,
        )
    if info_video_keys:
        logger.debug("%s meta/info.json video feature keys: %s", root, info_video_keys)
    else:
        logger.info(
            "%s meta/info.json has no observation.images.* feature keys; "
            "deriving expert video keys from meta/episodes columns",
            root,
        )

    meta_filters = (
        [(_COL_EPISODE_INDEX, "in", list(episode_indices))]
        if episode_indices is not None
        else None
    )

    out: dict[int, dict[str, dict]] = {}
    out_depth: dict[int, dict[str, dict]] = {}
    # chunks_size maps an episode to the video file that holds it, so a wrong value
    # silently resolves to another episode's expert video (or none at all). It is a
    # required LeRobot v3 info.json field; guessing 1000 would fabricate the mapping.
    if not info.get("chunks_size"):
        raise ValueError(
            f"{root}/meta/info.json has no 'chunks_size', so episodes cannot be mapped "
            "to their video files. Re-export the dataset with a complete info.json."
        )
    chunks_size = int(info["chunks_size"])
    for pq_path in parquet_files:
        table = pq.read_table(str(pq_path), filters=meta_filters)
        cols = set(table.column_names)
        if _COL_EPISODE_INDEX not in cols:
            raise ValueError(
                f"{pq_path}: missing required column {_COL_EPISODE_INDEX!r} "
                "in LeRobot meta/episodes parquet"
            )
        meta_video_keys = _video_keys_from_episode_meta_cols(
            cols
        ) + _video_keys_from_episode_meta_cols(cols, prefix=_COL_DEPTH_PREFIX)
        video_keys = sorted(set(info_video_keys) | set(meta_video_keys))
        logger.debug(
            "%s expert-video candidate keys: info=%s meta=%s union=%s",
            pq_path,
            info_video_keys,
            meta_video_keys,
            video_keys,
        )
        if not video_keys:
            _noisy_warning("%s has no videos/* metadata columns", pq_path)
            continue
        ep_col = table.column(_COL_EPISODE_INDEX).to_pylist()
        per_key_data = {
            key: {
                field: table.column(f"{_VIDEO_KEY_PREFIX}{key}{field}").to_pylist()
                for field in _VIDEO_KEY_FIELDS
                if f"{_VIDEO_KEY_PREFIX}{key}{field}" in cols
            }
            for key in video_keys
        }
        # Resolve each camera's video file (+ optional in-file seek). Priority:
        #  (1) per-video-key ``videos/<key>/chunk_index`` + ``file_index`` columns
        #      (authoritative LeRobot v3 coordinates; ``from_timestamp`` seeks into
        #      a shared/packed mp4); else
        #  (2) per-episode video files, where each episode has its OWN mp4 named by
        #      episode index (e.g. videos/<key>/chunk-000/file-<ep>.mp4) and starts
        #      at its own t=0 (no offset). The DATA parquet may be packed into a
        #      single shard (data/file_index=0 for every episode), so the data
        #      shard index MUST NOT be used to locate the per-episode video.
        for row, ep in enumerate(ep_col):
            ep_idx = int(ep) if ep is not None else None
            if ep_idx is None:
                continue
            videos: dict[str, dict] = {}
            depth_videos: dict[str, dict] = {}
            for key, fields in per_key_data.items():
                chunk = fields.get("/chunk_index", [None] * len(ep_col))[row]
                file_ = fields.get("/file_index", [None] * len(ep_col))[row]
                per_video_coords = chunk is not None and file_ is not None
                if not per_video_coords:
                    # Per-episode layout: video file index == episode index within
                    # a chunk of ``chunks_size`` episodes (LeRobot's chunk/file
                    # numbering), decoded from the episode's own start.
                    chunk = ep_idx // chunks_size
                    file_ = ep_idx % chunks_size
                chunk, file_ = int(chunk), int(file_)
                rel = _format_video_path_template(
                    video_path_template,
                    video_key=key,
                    chunk_index=chunk,
                    file_index=file_,
                )
                video_path = Path(rel)
                if not video_path.is_absolute():
                    video_path = root / video_path
                if not video_path.is_file():
                    _noisy_warning(
                        "%s episode_index=%s camera=%s resolved expert video to missing file %s",
                        pq_path,
                        ep_idx,
                        key,
                        video_path,
                    )
                    continue
                entry: dict[str, Any] = {"path": str(video_path.resolve())}
                # Only a per-video ``from_timestamp`` (packed mp4) implies a seek;
                # per-episode files start at t=0, so leave it unset.
                if per_video_coords and "/from_timestamp" in fields:
                    entry["from_timestamp"] = float(fields["/from_timestamp"][row])
                if per_video_coords and "/to_timestamp" in fields:
                    entry["to_timestamp"] = float(fields["/to_timestamp"][row])
                if key.startswith(_COL_DEPTH_PREFIX):
                    if key in depth_channels:
                        entry["channels"] = depth_channels[key]
                    if key in depth_scales:
                        entry["depth_scale"] = depth_scales[key]
                    depth_videos[key] = entry
                else:
                    videos[key] = entry
            if videos:
                out[ep_idx] = videos
            else:
                logger.debug(
                    "%s episode_index=%s resolved zero existing expert videos",
                    pq_path,
                    ep_idx,
                )
            if depth_videos:
                out_depth[ep_idx] = depth_videos
    if out:
        logger.info(
            "Resolved existing expert video paths for %d episode(s) from %s",
            len(out),
            root,
        )
    if out_depth:
        logger.info(
            "Resolved existing expert depth video paths for %d episode(s) from %s",
            len(out_depth),
            root,
        )
    return out, out_depth


def _depth_feature_channels(info: dict) -> dict[str, int]:
    """``{depth_video_key: trailing channel count}`` from ``meta/info.json``.

    The channel count is the only thing in the metadata that distinguishes the
    two depth encodings in circulation: TAMP datagen writes 1-channel uint16
    millimetres (metric), while toyblocks's own recorder writes a 3-channel
    per-frame min-max uint8 image (not metric). Consumers that need meters check
    this rather than decoding and guessing from the value range.
    """
    features = info.get("features", {}) or {}
    out: dict[str, int] = {}
    for key, spec in features.items():
        if not key.startswith(_COL_DEPTH_PREFIX):
            continue
        shape = (spec or {}).get("shape")
        if shape:
            out[key] = int(shape[-1])
    return out


def _depth_feature_scales(info: dict) -> dict[str, float]:
    """``{depth_video_key: encoded units per meter}`` from ``meta/info.json``."""
    features = info.get("features", {}) or {}
    out: dict[str, float] = {}
    for key, spec in features.items():
        if not key.startswith(_COL_DEPTH_PREFIX):
            continue
        scale = ((spec or {}).get("info") or {}).get("video.depth_scale")
        if scale is not None:
            out[key] = float(scale)
    return out


def decode_expert_depth_meters(entry: dict[str, Any]) -> np.ndarray:
    """Decode one ``expert_depth_videos`` entry's first frame to HxW meters.

    ``entry`` is a value from a setup's ``expert_depth_videos`` (``path`` plus an
    optional ``from_timestamp`` seek into a packed mp4 and the feature's
    ``channels`` and ``depth_scale`` metadata). Raises on the 3-channel encoding that
    ``lerobot_recorder._prepare_depth`` writes: that is a per-frame min-max uint8
    image, so its pixels carry no absolute scale and comparing them against
    metric eval depth would produce a plausible-looking but meaningless diff.
    """
    channels = entry.get("channels")
    if channels != 1:
        raise ValueError(
            f"depth clip {entry.get('path')!r} has channels={channels}; only the "
            "1-channel uint16-millimetre encoding (TAMP datagen) is metric. A "
            "3-channel clip is lerobot_recorder's per-frame min-max uint8 "
            "visualization, which has no absolute scale and cannot be compared "
            "against eval depth in meters."
        )
    if "depth_scale" not in entry:
        raise ValueError(
            f"depth clip {entry.get('path')!r} has no 'video.depth_scale' in its "
            "meta/info.json feature metadata; encoded depth units cannot be "
            "converted to meters without it."
        )
    depth_scale = float(entry["depth_scale"])
    if not np.isfinite(depth_scale) or depth_scale <= 0:
        raise ValueError(
            f"depth clip {entry.get('path')!r} has invalid "
            f"'video.depth_scale'={depth_scale!r}; expected a positive finite value."
        )

    import av

    path = entry["path"]
    if not Path(path).is_file():
        raise FileNotFoundError(f"expert depth video not found: {path}")

    from_timestamp = entry.get("from_timestamp")
    container = av.open(path)
    try:
        stream = container.streams.video[0]
        frame = None
        if from_timestamp:
            # Same packed-mp4 seek discipline as _extract_expert_first_frame:
            # seek lands on the keyframe at or before the target, so decode
            # forward to avoid returning a neighbouring episode's frame.
            target_pts = int(from_timestamp / stream.time_base)
            container.seek(target_pts, stream=stream)
            for f in container.decode(video=0):
                if f.pts is not None and f.pts >= target_pts:
                    frame = f
                    break
            if frame is None:
                raise ValueError(
                    f"depth video {path} has no frame at/after "
                    f"from_timestamp={from_timestamp:.4f}s"
                )
        else:
            for f in container.decode(video=0):
                frame = f
                break
            if frame is None:
                raise ValueError(f"depth video has no decodable frames: {path}")
        # ``to_ndarray(format=...)`` goes through swscale, which rescales a
        # narrower source (gray12le) into the full 16-bit range rather than
        # zero-extending it, i.e. it multiplies the millimetres by 2^(16-bits).
        # Undo that from the *decoded* frame's own bit depth so the samples are
        # the millimetres the recorder wrote, whatever depth it chose.
        bits = frame.format.components[0].bits
        raw = frame.to_ndarray(format="gray16le")
    finally:
        container.close()
    swscale_gain = float(1 << (16 - bits)) if bits < 16 else 1.0
    return np.squeeze(raw).astype(np.float32) / (swscale_gain * depth_scale)


def _video_keys_from_episode_meta_cols(
    cols: set[str],
    prefix: str = _COL_IMAGE_PREFIX,
) -> list[str]:
    """Derive video keys under ``prefix`` (e.g. ``observation.images.front_camera``).

    ``prefix`` matches :func:`_video_keys_from_info` and defaults to RGB.
    Datasets also record ``observation.depth.*`` (and may record segmentation)
    video features, which must never be selected as the expert RGB clip —
    otherwise :func:`_pick_primary_video_key` can pick e.g.
    ``observation.depth.front_camera`` (it sorts before ``observation.images.*``
    yet contains ``front``) and the W&B ``expert_first_frame`` renders a depth
    map instead of RGB. Depth is requested explicitly instead, and lands on a
    separate setup key.
    """
    keys: set[str] = set()
    for col in cols:
        if not col.startswith(_VIDEO_KEY_PREFIX):
            continue
        rest = col[len(_VIDEO_KEY_PREFIX) :]
        key, _, field = rest.rpartition("/")
        if f"/{field}" in _VIDEO_KEY_FIELDS and key and key.startswith(prefix):
            keys.add(key)
    return sorted(keys)


def _format_video_path_template(
    template: str,
    *,
    video_key: str,
    chunk_index: int,
    file_index: int,
) -> str:
    """Format common LeRobot video templates, including older chunk/file aliases."""
    return template.format(
        video_key=video_key,
        chunk_index=chunk_index,
        file_index=file_index,
        chunk=chunk_index,
        file=file_index,
        episode_chunk=chunk_index,
        episode_index=file_index,
    )


def _load_lerobot_info(root: Path) -> dict:
    """Read ``meta/info.json`` when present; video metadata depends on it."""
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        return {}
    with open(info_path) as f:
        return json.load(f)


def dataset_fps(path: str | Path) -> float:
    """Control rate (Hz) the LeRobot dataset(s) at ``path`` were recorded at.

    ``path`` is a LeRobot root or a directory of roots, matching
    :func:`load_setups`. Callers use this to turn a recorded frame count (e.g. a
    setup's ``expert_episode_length``) into a duration, so every root has to
    agree: one queue mixes episodes from all of them, and a per-root rate would
    make the same frame count mean different things depending on which child an
    episode came from. Disagreement therefore raises rather than picking one.

    Also raises when ``fps`` is absent. It is a required LeRobot v3
    ``meta/info.json`` field and there is no correct guess -- a wrong rate would
    silently scale every derived duration.
    """
    roots = _discover_lerobot_roots(Path(path))
    if not roots:
        raise FileNotFoundError(f"no LeRobot dataset found at or under {path}")
    by_root: dict[Path, float] = {}
    for root in roots:
        raw = _load_lerobot_info(root).get("fps")
        if raw is None:
            raise ValueError(
                f"{root}/meta/info.json has no 'fps', so recorded frame counts cannot "
                "be converted to seconds. Re-export the dataset with a complete "
                "info.json."
            )
        value = float(raw)
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"{root}/meta/info.json has 'fps'={raw!r}, expected a positive number")
        by_root[root] = value
    distinct = sorted(set(by_root.values()))
    if len(distinct) > 1:
        per_root = ", ".join(f"{root.name}={fps:g}" for root, fps in by_root.items())
        raise ValueError(
            f"the LeRobot roots under {path} disagree on 'fps' ({per_root}), so the same "
            "recorded frame count means a different duration depending on which root an "
            "episode came from. Serve them as separate datasets."
        )
    return distinct[0]


def max_episode_frames(path: str | Path) -> int:
    """Longest recorded episode (in frames) across the LeRobot root(s) at ``path``.

    Reads the ``length`` column of ``meta/episodes/chunk-*/file-*.parquet``
    for every discovered root; when a root has no such metadata, falls back to
    the ``expert_episode_length`` of the setups :func:`load_setups` produces
    (one full pass over the data parquet). Used by ``--replay`` to bound the
    global episode timeout so no recording is cut short.
    """
    roots = _discover_lerobot_roots(Path(path))
    if not roots:
        raise FileNotFoundError(f"no LeRobot dataset found at or under {path}")

    import pyarrow.parquet as pq

    longest = 0
    fallback_roots: list[Path] = []
    for root in roots:
        meta_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
        if not meta_files or "length" not in pq.read_schema(str(meta_files[0])).names:
            fallback_roots.append(root)
            continue
        for pq_path in meta_files:
            lengths = pq.read_table(str(pq_path), columns=["length"]).column("length").to_pylist()
            longest = max([longest, *(int(v) for v in lengths if v is not None)])
    for root in fallback_roots:
        longest = max(
            [longest, *(int(s["expert_episode_length"]) for s in load_setups(root))]
        )
    if longest <= 0:
        raise ValueError(f"no episode lengths could be read from the LeRobot dataset(s) at {path}")
    return longest


def _video_keys_from_info(info: dict, prefix: str = _COL_IMAGE_PREFIX) -> list[str]:
    """LeRobot video feature keys under ``prefix`` (default: RGB ``observation.images.*``)."""
    features = info.get("features", {}) or {}
    return sorted(k for k in features if k.startswith(prefix))


def _resolve_parquet_files(root: Path) -> list[Path]:
    """List the parquet files that make up ``root``, in episode order."""
    flat = root / "data.parquet"
    if flat.is_file():
        return [flat]
    for sub in (root / "data", root / "blocks" / "data"):
        if sub.is_dir():
            files = sorted(sub.glob("chunk-*/file-*.parquet"))
            if files:
                return files
    raise FileNotFoundError(f"no parquet files found under {root}")


_COL_DATA_CHUNK_INDEX = "data/chunk_index"
_COL_DATA_FILE_INDEX = "data/file_index"


def _episode_data_coords(
    root: Path,
    episode_indices: set[int],
) -> dict[int, tuple[int, int]] | None:
    """``{episode_index: (data_chunk_index, data_file_index)}`` from ``meta/episodes``.

    These are the authoritative per-episode data-file coordinates LeRobot v3
    records (the columns a merge rewrites when it renumbers episodes). Returns ``None`` when ``meta/episodes`` is
    absent or does not carry both mapping columns, so the caller can fall back
    to scanning every shard instead of inventing a mapping.
    """
    meta_dir = root / "meta" / "episodes"
    if not meta_dir.is_dir():
        return None
    meta_files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
    if not meta_files:
        return None

    import pyarrow.parquet as pq

    columns = [_COL_EPISODE_INDEX, _COL_DATA_CHUNK_INDEX, _COL_DATA_FILE_INDEX]
    if not set(columns) <= set(pq.read_schema(str(meta_files[0])).names):
        return None

    filters = [(_COL_EPISODE_INDEX, "in", sorted(episode_indices))]
    coords: dict[int, tuple[int, int]] = {}
    for pq_path in meta_files:
        table = pq.read_table(str(pq_path), columns=columns, filters=filters)
        eps = table.column(_COL_EPISODE_INDEX).to_pylist()
        chunks = table.column(_COL_DATA_CHUNK_INDEX).to_pylist()
        files = table.column(_COL_DATA_FILE_INDEX).to_pylist()
        for ep, chunk, file_ in zip(eps, chunks, files):
            if ep is None or chunk is None or file_ is None:
                continue
            coords[int(ep)] = (int(chunk), int(file_))
    return coords


def _data_shards_for_episodes(
    parquet_files: list[Path],
    root: Path,
    episode_indices: set[int] | None,
) -> list[Path]:
    """Return only the data shards that contain ``episode_indices``.

    LeRobot v3 splits data files by size, not by episode count, and records
    each episode's ``data/chunk_index`` + ``data/file_index`` in
    ``meta/episodes``; those coordinates are the only way to locate an
    episode's rows. (``meta/info.json``'s ``chunks_size`` counts files per
    chunk directory, so arithmetic on it points at the wrong shard and silently
    reads none of the requested episodes.)

    Returns all files when no filter is requested, when the dataset is a single
    flat ``data.parquet``, or when ``meta/episodes`` cannot supply the mapping —
    predicate pushdown still restricts the rows, so that is correct, only
    slower. Raises when a requested episode has no ``meta/episodes`` row, or
    maps to a shard that is not on disk.
    """
    if episode_indices is None or len(parquet_files) <= 1:
        return parquet_files
    coords = _episode_data_coords(root, episode_indices)
    if coords is None:
        logger.info(
            "%s: meta/episodes carries no %r + %r mapping, so all %d data shard(s) "
            "will be scanned for the requested episode(s).",
            root,
            _COL_DATA_CHUNK_INDEX,
            _COL_DATA_FILE_INDEX,
            len(parquet_files),
        )
        return parquet_files

    missing = sorted(episode_indices - set(coords))
    if missing:
        total = _load_lerobot_info(root).get("total_episodes")
        raise ValueError(
            f"{root}: no meta/episodes row for requested episode_indices {missing}, "
            f"so their data rows cannot be located (dataset reports "
            f"total_episodes={total})."
        )

    # ``_resolve_parquet_files`` may have resolved either ``data/`` or
    # ``blocks/data/``; derive the same directory from the listed shards so the
    # coordinates rebuild paths in whichever layout this dataset uses.
    data_dir = parquet_files[0].parent.parent
    known = set(parquet_files)
    wanted: set[Path] = set()
    for ep in sorted(episode_indices):
        chunk, file_ = coords[ep]
        shard = data_dir / f"chunk-{chunk:03d}" / f"file-{file_:03d}.parquet"
        if shard not in known:
            raise ValueError(
                f"{root}: meta/episodes maps episode {ep} to data shard {shard}, "
                f"which is not among the {len(parquet_files)} shard(s) found on disk."
            )
        wanted.add(shard)
    return [p for p in parquet_files if p in wanted]


def _episode_lengths_from_meta(
    root: Path, episode_indices: set[int] | None = None
) -> dict[int, int] | None:
    """``{episode_index: length}`` (frames) from LeRobot v3 ``meta/episodes``.

    Data files can be cut at a fixed row count, which splits an episode across
    two files; the frames counted in the file an episode starts in then miss its
    tail, while ``meta/episodes`` records the whole length. Returns ``None`` when
    the metadata (or its ``length`` column) is absent.
    """
    meta_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not meta_files:
        return None

    import pyarrow.parquet as pq

    columns = [_COL_EPISODE_INDEX, "length"]
    if not set(columns) <= set(pq.read_schema(str(meta_files[0])).names):
        return None
    filters = (
        None
        if episode_indices is None
        else [(_COL_EPISODE_INDEX, "in", sorted(episode_indices))]
    )
    lengths: dict[int, int] = {}
    for pq_path in meta_files:
        table = pq.read_table(str(pq_path), columns=columns, filters=filters)
        for ep, length in zip(
            table.column(_COL_EPISODE_INDEX).to_pylist(), table.column("length").to_pylist()
        ):
            if ep is not None and length is not None:
                lengths[int(ep)] = int(length)
    return lengths


def _load_setups_from_lerobot(
    parquet_files: list[Path],
    *,
    root: Path,
    wanted_episode_indices: set[int] | None = None,
    init_after_actions: frozenset[str] | None = None,
    init_after_label: str | None = None,
) -> tuple[list[dict], set[int]]:
    """Stream one setup per episode out of the given parquet files in order.

    When ``wanted_episode_indices`` is provided, pyarrow predicate pushdown
    skips row groups that don't contain the requested episodes, and column
    projection avoids reading large columns (action arrays, raw images) that
    aren't needed for setup construction.

    ``init_after_actions`` is the casefolded match set (or ``None``).
    Returns ``(setups, skipped_empty_remainder)``.
    """
    import pyarrow.parquet as pq

    setups: list[dict] = []
    skipped_empty: set[int] = set()
    object_columns_warned = False
    info = _load_lerobot_info(root)
    depth_channels = _depth_feature_channels(info)
    depth_scales = _depth_feature_scales(info)

    filters = None
    if wanted_episode_indices is not None:
        filters = [(_COL_EPISODE_INDEX, "in", list(wanted_episode_indices))]

    scan_files = _data_shards_for_episodes(parquet_files, root, wanted_episode_indices)
    # The authoritative episode lengths: an episode cut across two data files
    # would otherwise count only the rows of the file it starts in.
    meta_lengths = _episode_lengths_from_meta(root, wanted_episode_indices) or {}
    for pq_path in scan_files:
        # Read schema first to determine needed columns (avoids reading
        # action arrays, large observation tensors, etc.).
        schema = pq.read_schema(str(pq_path))
        all_cols = schema.names

        if _COL_JOINT_POS not in all_cols:
            raise ValueError(
                f"{pq_path}: missing required column {_COL_JOINT_POS!r}; "
                f"is this a toyblocks/TAMP LeRobot dataset?"
            )
        if _COL_IS_FIRST not in all_cols:
            raise ValueError(
                f"{pq_path}: missing required column {_COL_IS_FIRST!r}; "
                f"is this a toyblocks/TAMP LeRobot dataset?"
            )

        object_cols = [
            c
            for c in all_cols
            if c.startswith(_COL_OBJ_POSE_PREFIX)
            and not c[len(_COL_OBJ_POSE_PREFIX) :].startswith(SCENE_FURNITURE_PREFIXES)
        ]
        # Scene-furniture pose columns (e.g. observation.state.object_pose.table).
        # These are NOT manipulable blocks (excluded from ``objects`` above and
        # from codegen's block list), but the recorded table pose is needed to
        # reconstruct the datagen scene during eval — block Z is recorded
        # relative to this table, so a mismatched eval table makes blocks
        # float/sink. Captured into ``table_pose`` by ``_row_to_setup``.
        table_cols = [
            c for c in all_cols
            if c.startswith(_COL_OBJ_POSE_PREFIX)
            and c[len(_COL_OBJ_POSE_PREFIX):].startswith(SCENE_FURNITURE_PREFIXES)
        ]
        if not object_cols and not object_columns_warned:
            logger.info(
                "%s has no %s* columns; produced setups will only "
                "override the robot pose, not object poses.",
                pq_path,
                _COL_OBJ_POSE_PREFIX,
            )
            object_columns_warned = True

        image_cols = [c for c in all_cols if c.startswith(_COL_IMAGE_PREFIX)]
        depth_cols = [c for c in all_cols if c.startswith(_COL_DEPTH_PREFIX)]
        camera_intrinsics_cols = [
            c for c in all_cols if c.startswith(_COL_CAMERA_INTRINSICS_PREFIX)
        ]
        camera_pose_cols = [c for c in all_cols if c.startswith(_COL_CAMERA_POSE_PREFIX)]
        if image_cols:
            logger.debug("%s has parquet image columns: %s", pq_path, image_cols)
        else:
            logger.debug(
                "%s has no observation.images.* parquet columns; relying on "
                "meta/episodes video metadata for expert_first_frame",
                pq_path,
            )

        needed_cols = {
            _COL_IS_FIRST, _COL_EPISODE_INDEX, _COL_JOINT_POS,
            _COL_GRIPPER_POS, _COL_PDDL_GOAL, _COL_HL_PROMPT, _COL_TABLE_HEIGHT_DELTA,
            _COL_GLYPH_MAP, _COL_BLOCK_VARIABLE_MAP,
            _COL_SCENE_ID, _COL_ROBOT_ID, _COL_CUBE_SCALE,
            _COL_HIGH_LEVEL_ACTION, _COL_HIGH_LEVEL_ARGS, _COL_HIGH_LEVEL_STATE, _COL_SOURCE_PROBLEM,
            _COL_BLOCK_FRICTION, _COL_TABLE_FRICTION, _COL_ACTION_PD_MODE,
            _COL_JOINT_VEL, _COL_GOAL_ACTION, _COL_GOAL_POSITION,
        }
        needed_cols.update(object_cols)
        needed_cols.update(table_cols)
        needed_cols.update(image_cols)
        needed_cols.update(depth_cols)
        needed_cols.update(camera_intrinsics_cols)
        needed_cols.update(camera_pose_cols)
        columns = sorted(c for c in needed_cols if c in all_cols)

        table = pq.read_table(str(pq_path), columns=columns, filters=filters)

        if table.num_rows == 0:
            continue

        is_first_col = table.column(_COL_IS_FIRST)
        first_indices = [
            i for i in range(len(is_first_col)) if _scalar_bool(is_first_col[i].as_py())
        ]
        if not first_indices:
            first_indices = _infer_episode_starts(table, pq_path)

        for i, idx in enumerate(first_indices):
            next_idx = first_indices[i + 1] if i + 1 < len(first_indices) else table.num_rows
            ep_index = _episode_index_at(table, idx)
            episode_frames = next_idx - idx
            if ep_index is not None:
                episode_frames = meta_lengths.get(ep_index, episode_frames)
            start_row = idx
            if init_after_actions is not None:
                start_row = _init_row_after_hl_action(
                    table,
                    first_idx=idx,
                    next_idx=next_idx,
                    actions=init_after_actions,
                    label=init_after_label or "",
                    source=f"{pq_path}#{idx}",
                )
                if start_row is None:
                    ep_idx = _episode_index_at(table, idx)
                    if ep_idx is not None:
                        skipped_empty.add(ep_idx)
                    logger.warning(
                        "%s episode_index=%s: last %s frame is the episode end; "
                        "skipping (nothing left after the cut)",
                        pq_path,
                        ep_idx if ep_idx is not None else "?",
                        init_after_label,
                    )
                    continue
            setup = _row_to_setup(
                table=table, row=start_row, object_cols=object_cols,
                table_cols=table_cols, image_cols=image_cols,
                depth_cols=depth_cols, depth_channels=depth_channels,
                depth_scales=depth_scales, root=root,
                source=f"{pq_path}#{start_row}",
                expert_episode_length=episode_frames - (start_row - idx),
                task_row=idx if start_row != idx else None,
            )
            setup["init_frame_index"] = start_row - idx
            setup["expert_hl_calls"] = _expert_hl_calls(table, start_row, next_idx)
            demo_targets = _demo_goal_targets(table, start_row, next_idx)
            if demo_targets:
                setup["demo_goal_targets"] = demo_targets
            if init_after_label is not None:
                setup["init_after_hl_action"] = init_after_label
            setups.append(setup)

    if not setups and not skipped_empty:
        raise ValueError(
            f"loaded 0 setups from {len(parquet_files)} parquet file(s); nothing to serve"
        )
    return setups, skipped_empty


def _infer_episode_starts(table: Any, pq_path: Path) -> list[int]:
    """Detect episode boundaries via ``episode_index`` transitions (when ``is_first`` is all-False)."""
    if _COL_EPISODE_INDEX not in table.column_names:
        raise ValueError(
            f"{pq_path}: {_COL_IS_FIRST} is all-False and "
            f"{_COL_EPISODE_INDEX!r} column is missing; cannot infer "
            f"episode starts"
        )
    ep_col = table.column(_COL_EPISODE_INDEX)
    starts = [0]
    prev = ep_col[0].as_py()
    for i in range(1, len(ep_col)):
        cur = ep_col[i].as_py()
        if cur != prev:
            starts.append(i)
            prev = cur
    logger.info(
        "%s has %s all-False; inferred %d episode start(s) from %s transitions",
        pq_path,
        _COL_IS_FIRST,
        len(starts),
        _COL_EPISODE_INDEX,
    )
    return starts


def _demo_goal_targets(table: Any, start: int, stop: int) -> list[list]:
    """``[[goal.action, [u, v]], ...]``: each sub-skill run in rows ``[start, stop)``.

    A run is a stretch of the same non-empty ``goal.action``; its target is the
    run's first finite ``goal.position``. Empty when the columns are missing.
    """
    cols = table.column_names
    if _COL_GOAL_ACTION not in cols or _COL_GOAL_POSITION not in cols:
        return []
    actions = table.column(_COL_GOAL_ACTION).slice(start, stop - start).to_pylist()
    positions = table.column(_COL_GOAL_POSITION).slice(start, stop - start).to_pylist()
    runs: list[list] = []
    prev = None
    for action, pos in zip(actions, positions):
        action = (action or "").strip()
        if action != prev:
            prev = action
            if action:
                runs.append([action, None])
        if action and runs[-1][1] is None and pos is not None:
            uv = [float(x) for x in pos[:2]]
            if all(np.isfinite(uv)):
                runs[-1][1] = uv
    return [r for r in runs if r[1] is not None]


def demo_goal_target(setup: dict | None, action: str) -> list[float] | None:
    """The ``[u, v]`` target of the setup's first ``action`` sub-skill (``demo_goal_targets``), or None."""
    return next((uv for name, uv in (setup or {}).get("demo_goal_targets") or [] if name == action), None)


def _row_to_setup(
    *,
    table: Any,
    row: int,
    object_cols: list[str],
    table_cols: list[str],
    image_cols: list[str],
    depth_cols: list[str],
    depth_channels: dict[str, int],
    depth_scales: dict[str, float],
    root: Path,
    source: str,
    expert_episode_length: int,
    task_row: int | None = None,
) -> dict:
    cols = table.column_names
    # Episode-level task strings stay on the first frame when the setup row is
    # a mid-episode cut: later rows can be empty and ``goal.hl_prompt`` raises
    # on an empty cell. Poses / HL state / PD mode come from ``row``.
    prompt_row = row if task_row is None else task_row

    # TODO - hardcoded 7-DOF; derive from env config
    joints = _to_list(table.column(_COL_JOINT_POS)[row].as_py())
    if len(joints) < 7:
        raise ValueError(f"{source}: {_COL_JOINT_POS} has {len(joints)} entries, expected >= 7")

    setup: dict[str, Any] = {
        "robot_joint_positions": [float(x) for x in joints[:7]],
        "expert_episode_length": int(expert_episode_length),
    }

    if _COL_GRIPPER_POS in cols:
        gripper = _to_list(table.column(_COL_GRIPPER_POS)[row].as_py())
        if gripper:
            setup["robot_gripper_position"] = float(gripper[0])

    # Recorded frame-0 arm joint velocity (rich-schema datasets only). Applied
    # by apply_setup only when asked (``--replay``); see _COL_JOINT_VEL.
    if _COL_JOINT_VEL in cols:
        joint_vel = _to_list(table.column(_COL_JOINT_VEL)[row].as_py())
        if len(joint_vel) >= 7:
            setup["robot_joint_velocities"] = [float(x) for x in joint_vel[:7]]

    # Recorded PD-mode command at ``row``. Eval uses this to pre-apply the
    # actuator gains the training data used at that step (see
    # ``PDModeController.initialize_after_reset`` and its callers in
    # run_eval). At frame 0 this is the t=0 command; at a mid-episode cut it
    # is the mode the remaining skill starts under.
    if _COL_ACTION_PD_MODE in cols:
        pd = _to_list(table.column(_COL_ACTION_PD_MODE)[row].as_py())
        if pd:
            mode = int(round(float(pd[0])))
            if mode in (0, 1):
                setup["initial_pd_mode"] = mode
            else:
                logger.warning(
                    "%s: %s at row %d is %r (rounded to %d); expected 0 or 1, "
                    "leaving initial_pd_mode unset",
                    source, _COL_ACTION_PD_MODE, row, pd[0], mode,
                )

    # Per-episode contact material (see the column constants above). Recorded as
    # (static, dynamic[, restitution]); restitution defaults to 0.0 when the
    # dataset only carries the two friction coefficients.
    if _COL_BLOCK_FRICTION in cols:
        props = _to_list(table.column(_COL_BLOCK_FRICTION)[row].as_py())
        if len(props) >= 2:
            setup["block_friction"] = [
                float(props[0]),
                float(props[1]),
                float(props[2]) if len(props) > 2 else 0.0,
            ]

    if _COL_TABLE_FRICTION in cols:
        table_friction = _to_list(table.column(_COL_TABLE_FRICTION)[row].as_py())
        if table_friction:
            setup["table_friction"] = float(table_friction[0])

    if object_cols:
        objects: dict[str, dict[str, list[float]]] = {}
        # Variable-block datasets store an empty pose for blocks absent from a
        # given episode -- merged datasets an all-NaN one; those are skipped (not
        # every episode uses every block in the union schema). A partial (1..6
        # entry) pose is still malformed.
        absent_blocks: list[str] = []
        for col in object_cols:
            name = col[len(_COL_OBJ_POSE_PREFIX) :]
            pose = _to_list(table.column(col)[row].as_py())
            if len(pose) == 0 or all(x is None or np.isnan(x) for x in pose):
                absent_blocks.append(name)
                continue
            if len(pose) < 7:
                raise ValueError(
                    f"{source}: {col} has {len(pose)} entries, expected 7 (xyz + wxyz)"
                )
            objects[name] = {
                "position": [float(x) for x in pose[:3]],
                "quaternion": [float(x) for x in pose[3:7]],
            }
        if objects:
            setup["objects"] = objects
        if absent_blocks:
            logger.debug(
                "%s: %d block(s) absent from this episode (empty pose), skipped: %s",
                source,
                len(absent_blocks),
                absent_blocks,
            )

    # Scene-furniture (table) pose. Recorded block Z is relative to this table,
    # so eval must restore it (apply_setup writes it to ``workspace_table``).
    # ``has_table_column`` lets the verification diagnostic distinguish "table
    # matched" from "dataset never recorded a table" (the latter can't be
    # recovered from data — see scene_verification.compare_scene).
    setup["has_table_column"] = bool(table_cols)
    if table_cols:
        furniture_poses: dict[str, dict[str, list[float]]] = {}
        for col in table_cols:
            name = col[len(_COL_OBJ_POSE_PREFIX):]
            pose = _to_list(table.column(col)[row].as_py())
            if len(pose) < 7:
                raise ValueError(
                    f"{source}: {col} has {len(pose)} entries, expected 7 (xyz + wxyz)"
                )
            furniture_poses[name] = {
                "position": [float(x) for x in pose[:3]],
                "quaternion": [float(x) for x in pose[3:7]],
            }
        setup["furniture_poses"] = furniture_poses
        # Primary table pose written to the scene's ``workspace_table`` object:
        # prefer the exact ``table`` suffix, else the first furniture column.
        primary = furniture_poses.get("table") or next(iter(furniture_poses.values()))
        setup["table_pose"] = primary

    # Recorded table-height delta (offset from the preset baseline). Used by
    # apply_setup to reconstruct the table height when there is no absolute
    # table pose column (the common case).
    if _COL_TABLE_HEIGHT_DELTA in cols:
        delta = _to_list(table.column(_COL_TABLE_HEIGHT_DELTA)[row].as_py())
        if delta:
            setup["table_height_delta"] = float(delta[0])

    # Scene-identity columns (constant per dataset; see dataset_provenance).
    for col, key in ((_COL_SCENE_ID, "scene_id"), (_COL_ROBOT_ID, "robot_id")):
        if col in cols:
            val = table.column(col)[row].as_py()
            if isinstance(val, (list, tuple)):
                val = val[0] if val else None
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            if val is not None and str(val).strip():
                setup[key] = str(val).strip()

    # Recorded isotropic cube side length (meters) for this episode. Used to
    # spawn each env at the matching size (block scale is non-sim-state).
    if _COL_CUBE_SCALE in cols:
        scale = table.column(_COL_CUBE_SCALE)[row].as_py()
        if isinstance(scale, (list, tuple)):
            scale = scale[0] if scale else None
        if scale is not None:
            setup["cube_scale"] = normalize_cube_scale(scale)

    camera_intrinsics: dict[str, list[float]] = {}
    for col in cols:
        if not col.startswith(_COL_CAMERA_INTRINSICS_PREFIX):
            continue
        name = col[len(_COL_CAMERA_INTRINSICS_PREFIX) :].rpartition(":")[-1]
        raw = _to_list(table.column(col)[row].as_py())
        if len(raw) != 9:
            raise ValueError(
                f"{source}: {col!r} has {len(raw)} entries, expected 9 "
                "(row-major 3x3 K)"
            )
        camera_intrinsics[name] = [float(x) for x in raw]
    if camera_intrinsics:
        setup["camera_intrinsics"] = camera_intrinsics

    # The recorder writes zeros when a camera had no pose; leave those out.
    camera_poses: dict[str, list[float]] = {}
    for col in cols:
        if not col.startswith(_COL_CAMERA_POSE_PREFIX):
            continue
        raw = [float(x) for x in _to_list(table.column(col)[row].as_py())]
        if len(raw) == 7 and np.isfinite(raw).all() and any(raw[3:]):
            camera_poses[col[len(_COL_CAMERA_POSE_PREFIX) :].rpartition(":")[-1]] = raw
    if camera_poses:
        setup["camera_poses"] = camera_poses

    # observation.images.* columns store relative video paths; resolve them
    # so downstream consumers (expert first-frame extraction, eval table)
    # have an absolute path ready to open.
    if image_cols:
        expert_videos: dict[str, dict[str, Any]] = {}
        for col in image_cols:
            video_key = col[len(_COL_IMAGE_PREFIX) :]
            val = table.column(col)[row].as_py()
            resolved = _resolve_parquet_image_value(root, val)
            if resolved is not None:
                expert_videos[video_key] = {"path": str(resolved)}
                logger.debug(
                    "%s image column %s row=%d resolved parquet image/video path: %s",
                    source,
                    col,
                    row,
                    resolved,
                )
            else:
                logger.debug(
                    "%s image column %s row=%d value type=%s did not resolve "
                    "to an existing expert video/image path (value=%r)",
                    source,
                    col,
                    row,
                    type(val).__name__,
                    val,
                )
        if expert_videos:
            setup["expert_videos"] = expert_videos

    # Same resolution for the depth feed, kept under its own key so nothing that
    # looks for an RGB expert frame can ever land on a depth clip.
    if depth_cols:
        expert_depth_videos: dict[str, dict[str, Any]] = {}
        for col in depth_cols:
            resolved = _resolve_parquet_image_value(root, table.column(col)[row].as_py())
            if resolved is None:
                continue
            entry: dict[str, Any] = {"path": str(resolved)}
            if col in depth_channels:
                entry["channels"] = depth_channels[col]
            if col in depth_scales:
                entry["depth_scale"] = depth_scales[col]
            # Full feature key, matching what _load_episode_video_metadata writes.
            expert_depth_videos[col] = entry
        if expert_depth_videos:
            setup["expert_depth_videos"] = expert_depth_videos

    if _COL_PDDL_GOAL in cols:
        goal = table.column(_COL_PDDL_GOAL)[prompt_row].as_py()
        if isinstance(goal, (list, tuple)) and goal:
            goal = goal[0]
        if isinstance(goal, bytes):
            goal = goal.decode("utf-8")
        if isinstance(goal, str) and goal.strip():
            setup["pddl_goal"] = goal.strip()

    identity = _block_identity_at(table, prompt_row, source)
    if identity:
        setup["block_identity"] = identity.to_dict()
        if identity.to_source and "pddl_goal" in setup:
            setup["policy_pddl_goal"] = setup["pddl_goal"]
            setup["pddl_goal"] = source_goal(setup["pddl_goal"], identity)

    if _COL_HL_PROMPT in cols:
        hl_prompt = table.column(_COL_HL_PROMPT)[prompt_row].as_py()
        if isinstance(hl_prompt, (list, tuple)) and hl_prompt:
            hl_prompt = hl_prompt[0]
        if isinstance(hl_prompt, bytes):
            hl_prompt = hl_prompt.decode("utf-8")
        if not isinstance(hl_prompt, str) or not hl_prompt.strip():
            raise ValueError(
                f"{source}: {_COL_HL_PROMPT!r} must be a non-empty string, got {hl_prompt!r}"
            )
        setup["hl_prompt"] = hl_prompt.strip()

    # Recorded high-level instruction at ``row``. At a mid-episode cut this is
    # the first remaining skill (the oracle reconstruction gate compares FF's
    # first planned action against it). Optional metadata; older datasets that
    # predate the column produce no key. Decode bytes / unwrap a shape-[1]
    # list like pddl_goal above.
    if _COL_HIGH_LEVEL_ACTION in cols:
        hla = _first_skill_hl_action(table, row)
        if hla:
            setup["high_level_action"] = hla

    # Recorded symbolic HL state at ``row`` = the datagen planner's PDDL :init
    # at that step. At a mid-episode cut this is already post-skill, so the
    # oracle rebuilds from the remaining world. Unwrap the shape-[1]/bytes
    # wrappers like pddl_goal above. Older datasets that predate the column
    # produce no key (the oracle then fails loudly at reset).
    if _COL_HIGH_LEVEL_STATE in cols:
        hls = table.column(_COL_HIGH_LEVEL_STATE)[row].as_py()
        if isinstance(hls, (list, tuple)) and hls:
            hls = hls[0]
        if isinstance(hls, bytes):
            hls = hls.decode("utf-8")
        if isinstance(hls, str) and hls.strip():
            setup["high_level_state"] = hls.strip()

    # Provenance-only source-problem id (first frame); logged by the oracle.
    if _COL_SOURCE_PROBLEM in cols:
        sp = table.column(_COL_SOURCE_PROBLEM)[row].as_py()
        if isinstance(sp, (list, tuple)) and sp:
            sp = sp[0]
        if isinstance(sp, bytes):
            sp = sp.decode("utf-8")
        if isinstance(sp, str) and sp.strip():
            setup["source_problem"] = sp.strip()

    if _COL_EPISODE_INDEX in cols:
        ep = table.column(_COL_EPISODE_INDEX)[row].as_py()
        if isinstance(ep, (list, tuple)):
            # An empty cell used to reach int([]) and raise TypeError from deep in
            # here; the episode index is this setup's identity (queue routing, scale
            # slot), so say so instead.
            if not ep:
                raise ValueError(
                    f"{_COL_EPISODE_INDEX!r} is empty for row {row}, so this episode's "
                    "recorded setup cannot be identified."
                )
            ep = ep[0]
        if ep is not None:
            ep_int = int(ep)
            setup["episode_index"] = ep_int
            setup["episode_id"] = f"ep_{ep_int:06d}"
    setup.setdefault("episode_id", source)

    return setup


def _resolve_parquet_image_value(root: Path, value: Any) -> Path | None:
    """Resolve row-level LeRobot image/video references when present in data parquet."""
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, dict):
        for key in ("path", "video_path", "file", "uri"):
            candidate = value.get(key)
            if candidate:
                return _existing_path(root, str(candidate))
        return None
    if isinstance(value, (list, tuple)) and value:
        return _resolve_parquet_image_value(root, value[0])
    if isinstance(value, str) and value.strip():
        return _existing_path(root, value.strip())
    return None


def _existing_path(root: Path, raw_path: str) -> Path | None:
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    return path if path.is_file() else None


def _scalar_bool(v: Any) -> bool:
    if isinstance(v, (list, tuple)):
        return bool(v[0]) if v else False
    return bool(v)


def _to_list(v: Any) -> list:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, (int, float)):
        return [v]
    raise TypeError(
        f"_to_list: expected list, tuple, ndarray, scalar, or None; got {type(v).__name__}: {v!r}"
    )
