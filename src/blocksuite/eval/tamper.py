"""TAMPER — TAMP Expert Relabeling.

Capture the terminal states of unsuccessful eval episodes and feed them back
into datagen as seeds.  An evaluation run surfaces the states a policy actually
gets wrong; TAMP then plans expert demonstrations *from those states* rather
than from freshly sampled ones, so the next round of training data covers the
distribution the policy visits instead of the one datagen happens to sample.
The DAgger idea with TAMP as the expert doing the relabeling; the expert is the
vecgen engine (``scripts/vecgen/run.py``: FF from the captured scene + scripted
skills), so nothing here needs the TAMP submodule.

The snapshot has to be taken inside ``BaseEnv._before_auto_reset``.  That hook
fires while the finished episode's terminal state is still in the sim; by the
time ``infos["final_info"]`` reaches the eval loop the env has auto-reset and
the sim holds the next episode.  ``_success_once`` and the queue setup are both
still live at that point too — ``fetch_reset_states`` runs later, inside
``_reset_idx`` — so the outcome and the source PDDL problem are recoverable.

States are written in the layout of an env package's reset states
(``reset_states.pkl`` packed by :func:`~blocksuite.utils.reset_states_io.pack_reset_states`)
next to a clone of the eval's env package and a vecgen problem index
(``problems.json`` + ``problems/*.pddl``: one entry per captured state, its goal and
episode blocks), so the directory is a multi-problem vecgen env package::

    python scripts/datagen/run_tamper_pipeline.py --tamper-dir <tamper_dir>
    # = per problem dir:
    python scripts/vecgen/run.py --env-path <tamper_dir>/<problem_slug> \\
        --init-from-scene --reset-state-ids 0 1 2 ... --record <data_dir> --headless

Usage from the eval side::

    collector = TamperCollector(base_env, output_dir, source_env_path=...)
    collector.attach()
    ...                       # run the eval loop
    summary = collector.save()
"""

from __future__ import annotations

import json
import logging
import pickle
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import blocksuite
from blocksuite.eval.oracle.problem_reconstruction import (
    ProblemReconstructionError,
    build_problem_from_recorded_state,
    filter_block_names,
    parse_block_subtypes,
)
from blocksuite.utils import common_utils, tree_utils
from blocksuite.utils.reset_states_io import pack_reset_states
from blocksuite.vecgen.symbolic.pddl import parse_problem

logger = logging.getLogger(__name__)

# Env package configs a TAMPER dir needs (the whole source package is cloned, see
# TamperCollector._copy_env_package_files).
_ENV_PACKAGE_FILES = (
    "non_sim_state_randomizations.json",
    "block_config.json",
)

# generate_plans.load_non_sim_state_randomizations_config opens this one with no
# existence check, so a TAMPER dir without it is unusable.
_REQUIRED_ENV_PACKAGE_FILE = "non_sim_state_randomizations.json"

FAILURES_FILENAME = "tamper_failures.json"
RESET_STATES_FILENAME = "reset_states.pkl"
# vecgen's multi-problem index: written per TAMPER dir (the source package's own index, if
# any, maps the source's reset states, not the captured ones, so it is never copied)
PROBLEMS_INDEX_FILENAME = "problems.json"
PROBLEMS_SUBDIR = "problems"

#: Domain whose ``block`` subtypes declare a reconstructed problem's objects.
_DOMAIN_FILE = Path(blocksuite.__file__).parent / "tamp" / "blocks" / "domain.pddl"
#: Stand-in ``:init`` when the queue setup recorded no ``high_level_state``; vecgen plans
#: TAMPER problems from the captured scene (``run.py --init-from-scene``), not from ``:init``.
_PLACEHOLDER_INIT = "(handempty robot)"

# Episode outcomes worth replanning from.
REASON_FAILURE = "failure"
REASON_TIMEOUT = "timeout"
REASON_SUCCESS = "success"

_UNKNOWN_PROBLEM = "unknown_problem"


def _quiesce_state(state: dict[str, Any]) -> dict[str, Any]:
    """Turn a live mid-episode snapshot into a valid static reset state.

    A terminal state is captured mid-motion: joints carry velocity and
    ``joint_position_target`` still holds whatever the policy last commanded.
    Stage-1 reset states have neither — it re-applies the home pose before
    snapshotting precisely so the pickle does not store "a mid-drift transient
    (wrong joint targets)" (see ``generate_task_from_pddl``).  Replaying a state
    that kept them would have the PD controller yank the arm toward a stale
    target the moment vecgen wrote it, so the scene vecgen executes would not
    be the scene it planned against.

    Object *poses* are left exactly as captured — those are the TAMPER signal.
    """
    for asset_state in state.get("articulation", {}).values():
        if "joint_velocity" in asset_state:
            asset_state["joint_velocity"] = torch.zeros_like(asset_state["joint_velocity"])
        if "root_velocity" in asset_state:
            asset_state["root_velocity"] = torch.zeros_like(asset_state["root_velocity"])
        if "joint_position_target" in asset_state and "joint_position" in asset_state:
            asset_state["joint_position_target"] = asset_state["joint_position"].clone()
    for obj_state in state.get("rigid_object", {}).values():
        if "root_velocity" in obj_state:
            obj_state["root_velocity"] = torch.zeros_like(obj_state["root_velocity"])
    return state


def _infer_env_package_dir(base_env) -> Path | None:
    """Locate the env package the eval task was generated from.

    ``TemplateEnvCfg.reset_states_path`` points at that package's own
    ``reset_states.pkl``, so its parent is the directory holding the configs
    stage 2 needs. Saves the caller from passing a path it already implied.
    """
    cfg = getattr(base_env, "cfg", None)
    reset_states_path = getattr(cfg, "reset_states_path", None) if cfg else None
    if not reset_states_path:
        return None
    parent = Path(str(reset_states_path)).parent
    return parent if parent.exists() else None


def _problem_slug(pddl_problem_file: str | None) -> str:
    """Directory name for one source problem.

    Mirrors ``task_slug_from_pddl_path``: basename without extension, hyphens
    to underscores, so a TAMPER dir sits alongside its stage-1 counterpart.
    """
    if not pddl_problem_file:
        return _UNKNOWN_PROBLEM
    return Path(str(pddl_problem_file)).stem.replace("-", "_")


class TamperCollector:
    """Snapshots terminal states of unsuccessful episodes during evaluation.

    Args:
        base_env: The unwrapped environment exposing ``get_state_dict`` and
            ``_before_auto_reset``.
        output_dir: Root to write per-problem TAMPER dirs under.
        source_env_path: Env package the eval task was generated from; its
            config files are copied next to each ``reset_states.pkl`` so vecgen
            can run against the TAMPER dir directly.
        capture_successes: Also snapshot successful episodes (tagged
            ``success``), for callers that would rather filter at planning time.
        max_failures: Stop collecting after this many snapshots (None = no cap).
        quiesce: Zero velocities and align joint targets with joint positions so
            the snapshot is a valid static reset state (see :func:`_quiesce_state`).
            Disable only to inspect the raw terminal state; the result is not
            safe to feed back into vecgen.
    """

    def __init__(
        self,
        base_env,
        output_dir: str | Path,
        *,
        source_env_path: str | Path | None = None,
        capture_successes: bool = False,
        max_failures: int | None = None,
        quiesce: bool = True,
    ) -> None:
        self._quiesce = bool(quiesce)
        self._base_env = base_env
        self._output_dir = Path(output_dir)
        self._source_env_path = (
            Path(source_env_path) if source_env_path else _infer_env_package_dir(base_env)
        )
        self._capture_successes = bool(capture_successes)
        self._max_failures = max_failures

        # Parallel lists: one entry per captured episode, in capture order.
        self._states: list[dict[str, Any]] = []
        self._records: list[dict[str, Any]] = []

        self._attached = False
        self._original_hook = None
        self._skipped_for_cap = 0

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------
    def attach(self) -> None:
        """Wrap ``base_env._before_auto_reset`` so snapshots run pre-reset.

        Wrapping the bound attribute rather than subclassing keeps this usable
        with whatever env class the eval task registered; the viewer installs
        its own pre-reset hook the same way.
        """
        if self._attached:
            return

        base_env = self._base_env
        original = base_env.__class__._before_auto_reset

        def _hook(reset_env_ids: Sequence[int], *args: Any, **kwargs: Any):
            try:
                self._capture(reset_env_ids)
            except Exception:
                # A collector bug must not take down the eval run.
                logger.exception("TAMPER capture failed; continuing evaluation")
            return original(base_env, reset_env_ids, *args, **kwargs)

        self._original_hook = original
        base_env._before_auto_reset = _hook
        self._attached = True
        logger.info(
            "TAMPER collector attached (output_dir=%s, capture_successes=%s, max=%s)",
            self._output_dir, self._capture_successes, self._max_failures,
        )

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------
    def _outcome(self, env_idx: int) -> str:
        """Classify how the episode that just ended in *env_idx* finished."""
        base_env = self._base_env
        if bool(base_env._success_once[env_idx].item()):
            return REASON_SUCCESS
        # reset_terminated / reset_time_outs are set earlier in the same step.
        # request_episode_end also lands in reset_time_outs, but a requested end
        # on an unsuccessful episode is still a timeout from TAMPER's point of view.
        terminated = getattr(base_env, "reset_terminated", None)
        if terminated is not None and bool(terminated[env_idx].item()):
            return REASON_FAILURE
        return REASON_TIMEOUT

    def _setup_for(self, env_idx: int) -> dict | None:
        """The setup of the episode that is ending in *env_idx*, if any.

        ``current_setups`` still points at the finishing episode here: the queue
        only advances inside ``_reset_idx``, which runs after this hook.
        """
        queue_client = getattr(self._base_env, "queue_client", None)
        if queue_client is None:
            return None
        setups = getattr(queue_client, "current_setups", None)
        if not setups or env_idx >= len(setups):
            return None
        return setups[env_idx]

    def _capture(self, reset_env_ids: Sequence[int]) -> None:
        base_env = self._base_env
        if isinstance(reset_env_ids, torch.Tensor):
            env_indices = reset_env_ids.detach().cpu().tolist()
        else:
            env_indices = list(reset_env_ids)

        env_slots = getattr(base_env, "env_slots", None)

        for env_idx in env_indices:
            env_idx = int(env_idx)
            reason = self._outcome(env_idx)
            if reason == REASON_SUCCESS and not self._capture_successes:
                continue

            if self._max_failures is not None and len(self._states) >= self._max_failures:
                self._skipped_for_cap += 1
                continue

            # is_relative=True matches stage-1 capture, so poses are env-local
            # and reusable in whichever slot vecgen later runs them in.
            state = base_env.get_state_dict(env_ids=[env_idx], is_relative=True)
            if self._quiesce:
                state = _quiesce_state(state)
            setup = self._setup_for(env_idx) or {}

            episode_len = None
            length_buf = getattr(base_env, "episode_length_buf", None)
            if length_buf is not None:
                episode_len = int(length_buf[env_idx].item())

            record = {
                "reason": reason,
                "env_idx": env_idx,
                "episode_len": episode_len,
                "episode_id": setup.get("episode_id"),
                "episode_index": setup.get("episode_index"),
                "pddl_problem_file": setup.get("pddl_problem_file"),
                "pddl_goal": setup.get("pddl_goal"),
                # The env slot's baked scene profile; vecgen may only run this
                # state in a slot whose profile matches.
                "scene_profile_id": int(env_slots[env_idx]) if env_slots is not None else 0,
                "cube_scale": setup.get("cube_scale"),
                "table_height": setup.get("table_height"),
                "source_problem": setup.get("source_problem"),
                # the blocks this episode placed (the env parks the rest of its spawned
                # union) and its recorded first-frame state: the reconstructed problem's
                # :objects / :init when no PDDL problem file is recorded
                "episode_blocks": filter_block_names((setup.get("objects") or {}).keys()),
                "high_level_state": setup.get("high_level_state"),
            }

            self._states.append(state)
            self._records.append(record)
            logger.info(
                "TAMPER captured %s: env=%d episode_id=%s problem=%s (total=%d)",
                reason, env_idx, record["episode_id"],
                _problem_slug(record["pddl_problem_file"]), len(self._states),
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @property
    def num_captured(self) -> int:
        return len(self._states)

    def _copy_env_package_files(self, dest: Path) -> list[str]:
        """Clone the source env package next to the captured reset states.

        vecgen imports the directory as a package and instantiates
        ``<dirname>-v0``, so a bare pickle is unusable there.  Copying the whole
        package makes the TAMPER directory an env package: ``reset_states_path``
        is derived from ``__file__``, so the clone automatically reads the states
        written here, and the registered gym id is retargeted to this
        directory's name.

        A missing source package only fails once vecgen loads the directory. Say
        so now, while the run that produced the states is on screen.
        """
        copied: list[str] = []
        if self._source_env_path is not None and self._source_env_path.exists():
            for src in sorted(self._source_env_path.iterdir()):
                # reset_states.pkl / problems.json are ours; __pycache__ would shadow the rewrite.
                skip = ("__pycache__", RESET_STATES_FILENAME, PROBLEMS_INDEX_FILENAME)
                if src.name in skip or src.is_dir():
                    continue
                shutil.copy(src, dest / src.name)
                copied.append(src.name)
            self._retarget_gym_id(dest)

        if _REQUIRED_ENV_PACKAGE_FILE not in copied:
            logger.warning(
                "%s has no %s (source_env_path=%s), so it is no env package vecgen can "
                "load; copy the env package this task was generated from, or pass "
                "--tamper_source_env_path.",
                dest, _REQUIRED_ENV_PACKAGE_FILE, self._source_env_path,
            )
        return copied

    def _problem_for(
        self,
        record: dict[str, Any],
        scene_blocks: list[str],
        subtypes: frozenset[str],
        names: dict[tuple, str],
    ) -> tuple[str, list[str]] | None:
        """PDDL problem text and episode blocks of one captured episode, or None.

        The recorded ``pddl_problem_file`` when it exists; otherwise a problem rebuilt
        from the queue setup's ``pddl_goal`` over the episode's blocks (see
        :func:`build_problem_from_recorded_state`), named ``tamper_<k>`` per distinct
        (blocks, goal) so vecgen, which keys problems by name, keeps them apart.
        """
        pddl_file = record.get("pddl_problem_file")
        if pddl_file and Path(pddl_file).exists():
            blocks = record.get("episode_blocks") or list(parse_problem(pddl_file).blocks)
            return Path(pddl_file).read_text(), blocks
        goal = (record.get("pddl_goal") or "").strip()
        if not goal:
            return None
        blocks = record.get("episode_blocks") or scene_blocks
        hl_state = (record.get("high_level_state") or "").strip() or _PLACEHOLDER_INIT
        key = (tuple(sorted(blocks)), goal, hl_state)
        name = names.setdefault(key, f"tamper_{len(names):05d}")
        try:
            text = build_problem_from_recorded_state(
                blocks=blocks,
                high_level_state=hl_state,
                pddl_goal=goal,
                valid_block_subtypes=subtypes,
                problem_name=name,
            )
        except ProblemReconstructionError as exc:
            logger.warning("TAMPER: no problem for episode %s: %s", record.get("episode_id"), exc)
            return None
        return text, list(blocks)

    def _write_problems_index(self, dest: Path, indices: list[int]) -> list[str | None]:
        """Write vecgen's problem index for one TAMPER dir: ``problems/*.pddl`` (one file
        per distinct problem) and ``problems.json`` with one entry per captured state
        (``reset_states`` ``[i, i + 1]``, the episode's blocks; the env's other spawned
        blocks are parked). Returns each instance's entry slug, None when no problem could
        be recovered (no recorded problem file or ``pddl_goal``); vecgen cannot run those.
        """
        scene_blocks = filter_block_names(self._base_env.scene.rigid_objects.keys())
        subtypes = parse_block_subtypes(str(_DOMAIN_FILE))
        files: dict[str, str] = {}  # problem text -> its file (relative to dest)
        names: dict[tuple, str] = {}
        entries: list[dict[str, Any]] = []
        slugs: list[str | None] = []
        for instance_id, source_index in enumerate(indices):
            found = self._problem_for(self._records[source_index], scene_blocks, subtypes, names)
            if found is None:
                slugs.append(None)
                continue
            text, blocks = found
            rel = files.setdefault(text, f"{PROBLEMS_SUBDIR}/tamper_{len(files):05d}.pddl")
            slug = f"tamper_{instance_id:05d}"
            entries.append({
                "slug": slug,
                "pddl": rel,
                "blocks": blocks,
                "reset_states": [instance_id, instance_id + 1],
            })
            slugs.append(slug)
        if not entries:
            logger.warning(
                "%s: no captured episode recorded a PDDL problem or goal (eval ran without "
                "--queue_url?); pass the problem to run_tamper_pipeline.py --pddl-problem-file",
                dest,
            )
            return slugs
        (dest / PROBLEMS_SUBDIR).mkdir(exist_ok=True)
        for text, rel in files.items():
            (dest / rel).write_text(text)
        index = {"blocks": sorted(set(scene_blocks) | {b for e in entries for b in e["blocks"]}),
                 "problems": entries}
        with open(dest / PROBLEMS_INDEX_FILENAME, "w") as f:
            json.dump(index, f, indent=2)
            f.write("\n")
        return slugs

    def _retarget_gym_id(self, dest: Path) -> str | None:
        """Point the copied package's ``gym.register`` id at this directory.

        ``execute_plans`` derives the task id it instantiates from the env
        directory name, so a clone that kept the source package's id would
        register one task and be asked for another.

        Returns the new task id, or None when there is nothing to rewrite.
        """
        init_py = dest / "__init__.py"
        if not init_py.exists():
            return None
        source_name = self._source_env_path.name if self._source_env_path else None
        if not source_name:
            return None
        text = init_py.read_text()
        old_id = f'id="{source_name}-v0"'
        new_id = f'id="{dest.name}-v0"'
        if old_id not in text:
            logger.warning(
                "%s: could not find %s to retarget; stage-3 execution against this "
                "directory will not find its task id", init_py, old_id,
            )
            return None
        init_py.write_text(text.replace(old_id, new_id))
        return f"{dest.name}-v0"

    def save(self) -> dict[str, Any]:
        """Write one stage-1-shaped dir per source problem.

        Returns:
            Summary dict: captured count, per-problem dirs and their instance ids.
        """
        if not self._states:
            logger.info("TAMPER collector captured nothing; writing no output")
            return {"num_captured": 0, "problems": {}}

        # Group by source problem (one TAMPER dir = one vecgen env package each).
        by_problem: dict[str, list[int]] = {}
        for index, record in enumerate(self._records):
            by_problem.setdefault(_problem_slug(record["pddl_problem_file"]), []).append(index)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        problems: dict[str, Any] = {}

        for slug, indices in sorted(by_problem.items()):
            dest = self._output_dir / slug
            dest.mkdir(parents=True, exist_ok=True)

            # Instance i of this dir's reset_states.pkl is indices[i], so the
            # metadata below is what maps a plan back to the episode it came from.
            states = tree_utils.cat([self._states[i] for i in indices])
            states = tree_utils.reshape(states, (len(indices), -1))
            states_np = common_utils.to_numpy(states)

            profile_ids = np.array(
                [self._records[i]["scene_profile_id"] for i in indices], dtype=np.int64
            )
            num_profiles = (
                getattr(self._base_env, "MAX_NON_SIM_STATE_RANDOMIZATIONS", None)
                or getattr(self._base_env, "num_scene_profiles", None)
                or 1
            )
            scene_profile = getattr(self._base_env, "scene_profile", None)
            profiles = (
                [scene_profile(int(pid)) for pid in sorted(set(profile_ids.tolist()))]
                if callable(scene_profile)
                else None
            )

            with open(dest / RESET_STATES_FILENAME, "wb") as f:
                pickle.dump(
                    pack_reset_states(states_np, profile_ids, int(num_profiles), profiles=profiles),
                    f,
                )

            vecgen_slugs = self._write_problems_index(dest, indices)
            failures = []
            for instance_id, source_index in enumerate(indices):
                entry = dict(self._records[source_index])
                entry["instance_id"] = instance_id
                # its problems.json entry (None: no problem recovered, vecgen skips it)
                entry["vecgen_problem"] = vecgen_slugs[instance_id]
                failures.append(entry)

            pddl_problem_file = next(
                (self._records[i]["pddl_problem_file"] for i in indices
                 if self._records[i]["pddl_problem_file"]),
                None,
            )
            payload = {
                "pddl_problem_file": pddl_problem_file,
                "source_env_path": str(self._source_env_path) if self._source_env_path else None,
                "num_instances": len(indices),
                "num_scene_profiles": int(num_profiles),
                "reason_counts": _count_reasons(failures),
                "failures": failures,
            }
            with open(dest / FAILURES_FILENAME, "w") as f:
                json.dump(payload, f, indent=2)
                f.write("\n")

            copied = self._copy_env_package_files(dest)
            problems[slug] = {
                "dir": str(dest),
                "num_instances": len(indices),
                "instance_ids": list(range(len(indices))),
                "pddl_problem_file": pddl_problem_file,
                "copied_env_files": copied,
                # What vecgen will instantiate for this directory.
                "task_id": f"{dest.name}-v0" if "__init__.py" in copied else None,
                "num_vecgen_problems": sum(s is not None for s in vecgen_slugs),
            }
            logger.info(
                "TAMPER wrote %d state(s) to %s (problem=%s)", len(indices), dest, slug
            )

        if self._skipped_for_cap:
            logger.warning(
                "TAMPER skipped %d episode(s) after hitting max_failures=%s",
                self._skipped_for_cap, self._max_failures,
            )

        summary = {
            "num_captured": len(self._states),
            "num_skipped_for_cap": self._skipped_for_cap,
            "output_dir": str(self._output_dir),
            "problems": problems,
        }
        with open(self._output_dir / "tamper_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        return summary


def _count_reasons(failures: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in failures:
        counts[entry["reason"]] = counts.get(entry["reason"], 0) + 1
    return counts
