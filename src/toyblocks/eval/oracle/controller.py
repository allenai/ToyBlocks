"""Drive a per-env symbolic oracle and turn its output into LL-policy conditioning.

The eval harness owns the live sim (``base_env``).  For each env we keep an
independent :class:`OracleHighLevelPolicy` (built ``n_envs=1`` and bound to that
env via :class:`BaseEnvOracleSimAdapter`), because the queue can run a different
PDDL problem in every env.  Each control step :meth:`compute` ticks every oracle,
parses its action string into a language phrase + target pixel, and produces the
``task_name`` list and ``target_heatmap`` batch the maniflow LL server consumes.

``task_name`` is rebuilt in the served checkpoint's ``language_mode`` (advertised
by the policy server): ``action_only`` gets the bare phrase, ``action_goal`` gets
the phrase plus the target pixels, as HVLA's dataloader formatted it in training.
An ``action_goal`` checkpoint conditioned on a bare verb has no spatial goal at
all — its ``target_heatmap`` is dropped when ``target_injection_mode`` is unset.

The serving code is untouched: the controller's output is injected into the obs
dict in ``_run_eval`` and forwarded by ``ManiflowObsTransform``.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from toyblocks.serving.maniflow_serving_utils import (
    DEFAULT_TARGET_SIGMA,
    MANIFLOW_IMAGE_SIZE,
    _build_heatmap,
    format_task_name,
)

from toyblocks.eval.oracle.recovering import OraclePlanStuck, make_oracle
from toyblocks.eval.oracle.parse import parse_oracle_action
from toyblocks.eval.oracle.problem_reconstruction import (
    ProblemReconstructionError,
    assert_oracle_consistent_with_recording,
    build_problem_from_recorded_state,
    filter_block_names,
    parse_block_subtypes,
)
from toyblocks.eval.oracle.sim_adapter import BaseEnvOracleSimAdapter
from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import _CUBE
from toyblocks.utils.episode_dataset import demo_goal_target

logger = logging.getLogger(__name__)

# Action phrase emitted for a queue-exhausted (idle) env. Such envs run default
# randomization episodes whose results are masked out at finalization
# (``previous_setups[i] is None``), so this string never reaches a *scored*
# episode — it only keeps the batched LL server's input length == num_envs.
# It is intentionally a bare in-distribution verb, never the task-level goal.
# Under ``language_mode='action_goal'`` it is completed with the frame centre,
# since a verb with no coordinates is out of distribution for such a checkpoint.
_IDLE_PLACEHOLDER_ACTION = "pick"


def _demo_target_px(setup: Optional[dict], action: str) -> Optional[tuple[int, int]]:
    """The demonstration's front-camera target pixel for its first ``action`` sub-skill."""
    uv = demo_goal_target(setup, action)
    return None if uv is None else (int(round(uv[0])), int(round(uv[1])))


class OracleHLController:
    """Owns one oracle per env and emits per-step ``task_name`` + ``target_heatmap``."""

    def __init__(
        self,
        base_env: Any,
        *,
        domain_file: str,
        default_problem_file: str | None,
        language_mode: str,
        action_config_path: str | None = None,
        target_sigma: float = DEFAULT_TARGET_SIGMA,
        orig_cam_w: float = 640.0,
        orig_cam_h: float = 360.0,
        front_cam_name: str = "front_camera",
        reconstruct_pddl: bool = True,
        problem_dump_dir: str | None = None,
        replan: bool = True,
        max_replans: int = 3,
    ) -> None:
        self._base_env = base_env
        self._num_envs = int(base_env.num_envs)
        self._domain_file = domain_file
        # Domain ``block`` subtypes, parsed once, so reconstructed :objects can
        # declare each cube under its true subtype and typed goal quantifiers
        # (e.g. ``(exists (?b - block_red) ...)``) ground instead of collapsing to
        # FALSE over an empty type.
        self._block_subtypes = parse_block_subtypes(domain_file)
        self._default_problem_file = default_problem_file
        # The served LL checkpoint's text contract (advertised by the policy
        # server). It decides whether ``task_name`` is the bare verb phrase or
        # carries the target pixels; there is no default, because a mismatch
        # silently leaves the policy with no spatial goal.
        self._language_mode = language_mode
        self._action_config_path = action_config_path
        self._target_sigma = float(target_sigma)
        self._reconstruct_pddl = bool(reconstruct_pddl)
        # Re-plan an env's oracle from the live state when its next step's preconditions
        # no longer hold (see RecoveringOracleHLPolicy).
        self._replan = bool(replan)
        self._max_replans = int(max_replans)
        self._problem_dump_dir = Path(problem_dump_dir) if problem_dump_dir else None
        if self._problem_dump_dir is not None:
            self._problem_dump_dir.mkdir(parents=True, exist_ok=True)
        # Reconstructed-problem text cache keyed by (block set, goal, recorded
        # high_level_state) so identical episodes across envs/resets skip rebuild.
        self._recon_cache: dict[tuple, str] = {}
        # Per-env temp .pddl paths written for reconstructed problems; unlinked
        # and replaced on the next reset of that env.
        self._temp_problem_paths: list[Optional[Path]] = [None] * self._num_envs

        # Pixel coords come out of the oracle in the front-camera resolution; the
        # heatmap maps them onto the 224x224 grid via orig_cam_w/h. Prefer the
        # live camera resolution so it matches whatever the oracle projected into.
        self._orig_cam_w, self._orig_cam_h = self._resolve_cam_resolution(
            front_cam_name, orig_cam_w, orig_cam_h
        )
        # Idle-env placeholder target: frame centre, so the placeholder string is
        # well-formed under ``action_goal``. Its heatmap stays zero and its
        # episode is masked out, so the exact pixel is immaterial.
        self._idle_placeholder_task = format_task_name(
            _IDLE_PLACEHOLDER_ACTION,
            np.array([self._orig_cam_w / 2.0, self._orig_cam_h / 2.0], dtype=np.float32),
            self._language_mode,
        )

        # One adapter + oracle per env. Adapters are stable (bound to env_index);
        # oracles are rebuilt on reset when the env's problem file is known.
        self._adapters = [BaseEnvOracleSimAdapter(base_env, i) for i in range(self._num_envs)]
        self._oracles: list[Optional[Any]] = [None] * self._num_envs
        self._zero_heatmap = np.zeros(
            (3, MANIFLOW_IMAGE_SIZE, MANIFLOW_IMAGE_SIZE), dtype=np.float32
        )
        # Per-env state for plan completion: the LL model never saw "done" as an
        # instruction during training, so when the oracle finishes we repeat the
        # last real instruction for the final step and flag the env so the
        # harness can end the episode (after final evaluation).
        self._last_task_names: list[Optional[str]] = [None] * self._num_envs
        self._last_heatmaps: list[Optional[np.ndarray]] = [None] * self._num_envs
        self._last_target_xy: list[Optional[np.ndarray]] = [None] * self._num_envs
        self._target_xy_batch = np.full((self._num_envs, 2), np.nan, dtype=np.float32)
        self._done_mask: list[bool] = [False] * self._num_envs
        # Place->pick recovery cap: per-env, not process-fatal. See OraclePlanStuck.
        self._stuck_mask: list[bool] = [False] * self._num_envs
        # Full HL action string per env (language + ``<u, v>`` coords), for the
        # video burn-in. Distinct from ``task_name`` (coords stripped) sent to the
        # LL policy. Updated every ``compute``; read via ``display_texts``.
        self._display_texts: list[str] = [""] * self._num_envs
        self._last_display_texts: list[Optional[str]] = [None] * self._num_envs
        # Last HL task_name logged per env, so ``compute`` only emits a line when
        # the instruction actually changes (pick -> place -> done) rather than
        # every control step.
        self._logged_hl: list[Optional[str]] = [None] * self._num_envs
        # Per-env episode identity captured on each reset. Logs are labelled by
        # episode index (not env index) because every oracle is built n_envs=1 and
        # always self-reports env0, while envs recycle distinct episodes from the
        # queue -- the episode index is the informative identity.
        self._episode_indices: list[Optional[int]] = [None] * self._num_envs
        self._episode_ids: list[Optional[str]] = [None] * self._num_envs
        # Whether each env currently holds an active queue setup. A queue with
        # fewer remaining episodes than env slots leaves some envs idle (setup
        # ``None``); those run masked-out default episodes and legitimately have
        # no oracle, so ``compute`` must not raise for them (only for an *active*
        # setup whose oracle is missing).
        self._active_setup: list[bool] = [False] * self._num_envs
        # Per-env oracle progress mirrored from public accessors, so phase
        # transitions (pick->place) and plan advancement are logged
        # episode-indexed on the ToyBlocks side (the oracle's own prints always
        # self-report ``env0`` because each oracle is built ``n_envs=1``).
        self._last_phase: list[Optional[str]] = [None] * self._num_envs
        self._last_plan_idx: list[Optional[int]] = [None] * self._num_envs
        self._pick_log_step: list[int] = [0] * self._num_envs

    # ------------------------------------------------------------------
    def _resolve_cam_resolution(
        self, front_cam_name: str, default_w: float, default_h: float
    ) -> tuple[float, float]:
        """Live front-camera pixel dimensions; raises if they can't be read.

        ``_build_heatmap`` rescales the oracle's pixel target by
        ``224 / orig_cam_w``, so a wrong resolution shifts every heatmap by a
        fixed factor without any error — the LL is then aimed slightly off every
        target and the run just looks bad. The ``default_w``/``default_h``
        arguments are the caller's declared expectation, not a fallback: they are
        only used to report the discrepancy.
        """
        cameras = getattr(self._base_env, "cameras", None) or {}
        cam = cameras.get(front_cam_name)
        if cam is None:
            raise RuntimeError(
                f"[OracleHL] front camera {front_cam_name!r} is not in the live scene "
                f"(cameras: {sorted(cameras)}). Its pixel dimensions scale every "
                "oracle heatmap coordinate, so they cannot be assumed."
            )
        # isaaclab Camera exposes image dims via cfg width/height.
        w = float(cam.cfg.width)
        h = float(cam.cfg.height)
        if not (w > 0 and h > 0):
            raise RuntimeError(
                f"[OracleHL] front camera {front_cam_name!r} reports a non-positive "
                f"resolution ({w}x{h}); oracle heatmap coordinates scale by it."
            )
        if (w, h) != (float(default_w), float(default_h)):
            logger.info(
                "[OracleHL] front camera %r renders at %gx%g (caller expected %gx%g); "
                "using the live resolution to scale oracle pixel targets.",
                front_cam_name, w, h, float(default_w), float(default_h),
            )
        return w, h

    # ------------------------------------------------------------------
    def on_reset(self, env_indices: Sequence[int], setups: Sequence[Optional[dict]]) -> None:
        """(Re)build the oracle for each reset env from its PDDL problem file.

        ``setups`` aligns with ``env_indices``.  An env with no problem file and
        no reconstructable ``pddl_goal`` gets ``oracle=None``; ``compute`` then
        raises for that env (there is no goal-prompt fallback — the LL must only
        ever receive oracle verbs).
        """
        for env_idx, setup in zip(env_indices, setups):
            self._last_task_names[env_idx] = None
            self._last_heatmaps[env_idx] = None
            self._last_target_xy[env_idx] = None
            self._done_mask[env_idx] = False
            self._stuck_mask[env_idx] = False
            self._logged_hl[env_idx] = None
            self._last_phase[env_idx] = None
            self._last_plan_idx[env_idx] = None
            self._pick_log_step[env_idx] = 0
            self._active_setup[env_idx] = setup is not None
            self._episode_indices[env_idx] = (
                setup.get("episode_index") if setup is not None else None
            )

            # Push this episode's cube scale into the per-env adapter so the
            # oracle's projection/offset math sees the right block size (the
            # adapter is built once but cube size is per-episode).
            cube_scale = self._cube_scale_multiplier(setup)
            self._adapters[env_idx].set_cube_scale(cube_scale)

            problem_file = None
            if setup is not None:
                problem_file = setup.get("pddl_problem_file")
            problem_file = problem_file or self._default_problem_file

            # Precedence: explicit/default problem file first, then
            # reconstruction from pddl_goal. Build failures propagate (a
            # missing/broken oracle is a bad run — surface it, don't degrade
            # into feeding the goal string to the LL).
            if problem_file is not None:
                self._validate_problem_objects(problem_file)
                oracle = make_oracle(
                    domain_file=self._domain_file,
                    problem_file=problem_file,
                    simulator=self._adapters[env_idx],
                    scene=self._adapters[env_idx].scene,
                    action_config_path=self._action_config_path,
                    include_coords=True,
                    replan=self._replan,
                    max_replans=self._max_replans,
                )
                oracle.demo_place_on_table_px = _demo_target_px(setup, "place on table")
                self._oracles[env_idx] = oracle
                continue

            if self._reconstruct_pddl and setup is not None and setup.get("pddl_goal"):
                # Reconstruction path fails LOUDLY (no silent zero-heatmap): a
                # wrong :init makes FF plan a different task, so a bad episode
                # must surface rather than quietly degrade.
                self._build_reconstructed_oracle(env_idx, setup)
                continue

            self._oracles[env_idx] = None

    def _cube_scale_multiplier(self, setup: Optional[dict]) -> float:
        """Multiplier on the nominal 0.045 m cube width for this episode.

        The queue records ``cube_scale`` as an absolute side length in metres
        (see ``normalize_cube_scale``); the oracle + reconstructor both expect a
        multiplier relative to the nominal cube (1.0 == nominal), matching the
        adapter's default.
        """
        if setup is None:
            return 1.0
        side_m = setup.get("cube_scale")
        if side_m is None:
            return 1.0
        return float(side_m) / _CUBE

    def _block_names(self) -> list[str]:
        """Filtered manipulable block names present in the live scene."""
        return filter_block_names(self._base_env.scene.rigid_objects.keys())

    def _recon_cache_key(
        self, blocks: Sequence[str], pddl_goal: str, high_level_state: str
    ) -> tuple:
        """Key a reconstructed problem by block set, goal, and recorded HL state.

        The problem text is a pure function of these three (no live poses), so
        identical episodes across envs/resets reuse the cached text -- and the key
        is both exact and cheaper than the old rounded-pose hashing.
        """
        return (frozenset(blocks), pddl_goal.strip(), high_level_state.strip())

    def _build_reconstructed_oracle(self, env_idx: int, setup: dict) -> None:
        """Build a PDDL problem from the recorded HL state, build the oracle, gate it.

        Fails LOUDLY when the episode has no recorded
        ``observation.state.high_level_state``: a wrong :init makes FF plan a
        different task, so rather than geometrically guessing we require the exact
        recorded state and instruct the user to regenerate stale datasets.
        """
        high_level_state = setup.get("high_level_state")
        if not high_level_state:
            raise ProblemReconstructionError(
                f"env={env_idx} episode {setup.get('episode_id')!r} has no recorded "
                "observation.state.high_level_state; cannot build the oracle :init. "
                "Regenerate the dataset with a datagen that records "
                "observation.state.high_level_state (no geometric fallback)."
            )

        pddl_goal = setup["pddl_goal"]

        # Build :objects from the blocks actually placed in this episode. The
        # autogen env spawns every block in the dataset's *union* schema (e.g. 33
        # blocks) and parks the ones absent from this episode far out at x=i+1
        # (see codegen init_pos); those phantom blocks aren't part of the task.
        scene_blocks = self._block_names()
        blocks = filter_block_names(sorted((setup.get("objects") or {}).keys()))
        unused = len(scene_blocks) - len(blocks)
        logger.info(
            "[OracleHL] env=%d building PDDL problem from recorded high_level_state "
            "for %d episode block(s)%s; source_problem=%r; goal=%r",
            env_idx,
            len(blocks),
            "" if unused <= 0 else f" ({unused} unused spawn-slot block(s) in scene excluded)",
            setup.get("source_problem"),
            pddl_goal,
        )

        cache_key = self._recon_cache_key(blocks, pddl_goal, high_level_state)
        problem_text = self._recon_cache.get(cache_key)
        if problem_text is None:
            problem_text = build_problem_from_recorded_state(
                blocks=blocks,
                high_level_state=high_level_state,
                pddl_goal=pddl_goal,
                valid_block_subtypes=self._block_subtypes,
            )
            self._recon_cache[cache_key] = problem_text

        problem_path = self._write_problem_file(env_idx, setup, problem_text)

        oracle = make_oracle(
            domain_file=self._domain_file,
            problem_file=str(problem_path),
            simulator=self._adapters[env_idx],
            scene=self._adapters[env_idx].scene,
            action_config_path=self._action_config_path,
            include_coords=True,
            replan=self._replan,
            max_replans=self._max_replans,
        )
        assert_oracle_consistent_with_recording(
            oracle,
            setup.get("high_level_action"),
            problem_path=str(problem_path),
        )
        oracle.demo_place_on_table_px = _demo_target_px(setup, "place on table")
        self._oracles[env_idx] = oracle

    def _write_problem_file(self, env_idx: int, setup: dict, text: str) -> Path:
        """Write reconstructed PDDL to a per-env temp file (+ optional dump)."""
        prev = self._temp_problem_paths[env_idx]
        if prev is not None:
            prev.unlink(missing_ok=True)
        fd = tempfile.NamedTemporaryFile(
            mode="w", suffix=".pddl", prefix=f"recon_env{env_idx}_", delete=False
        )
        fd.write(text)
        fd.close()
        path = Path(fd.name)
        self._temp_problem_paths[env_idx] = path

        if self._problem_dump_dir is not None:
            ep = setup.get("episode_id", f"env{env_idx}")
            dump = self._problem_dump_dir / f"env{env_idx}_{ep}.pddl"
            dump.write_text(text)
        return path

    def _validate_problem_objects(self, problem_file: str) -> None:
        """Warn if PDDL objects are missing from the scene (would project to 0,0)."""
        try:
            text = Path(problem_file).read_text()
        except OSError as e:
            raise FileNotFoundError(f"PDDL problem file unreadable: {problem_file}: {e}")
        scene_objs = set(self._base_env.scene.rigid_objects.keys())
        # Parse object tokens from the (:objects ...) block (best-effort).
        import re
        m = re.search(r"\(:objects(.*?)\)", text, re.DOTALL | re.IGNORECASE)
        if not m:
            return
        tokens = [t for t in re.split(r"\s+", m.group(1)) if t and t != "-"]
        # Drop PDDL type names that follow a '-'; keep names that look like blocks.
        missing = [t for t in tokens if t.startswith("block_") and t not in scene_objs]
        if missing:
            logger.warning(
                "[OracleHL] problem %s references objects not in scene: %s "
                "(present: %s)", problem_file, missing, sorted(scene_objs),
            )

    # ------------------------------------------------------------------
    @property
    def target_xy_batch(self) -> np.ndarray:
        """Native camera pixel targets from the last ``compute`` (B, 2) float32."""
        return self._target_xy_batch

    def compute(self) -> tuple[list[str], np.ndarray]:
        """Tick every oracle; return ``(task_names, heatmaps (B,1,3,224,224))``."""
        task_names: list[str] = []
        heatmaps: list[np.ndarray] = []
        target_xys: list[np.ndarray] = []
        display_texts: list[str] = []
        for env_idx in range(self._num_envs):
            oracle = self._oracles[env_idx]
            if oracle is None:
                if self._active_setup[env_idx]:
                    # Active episode with no oracle = a bad run (setup carried
                    # neither a problem file nor a reconstructable goal). Raise
                    # rather than silently feeding the goal string to the LL.
                    raise RuntimeError(
                        f"{self._log_prefix(env_idx)} has an active setup but no "
                        "oracle; every queue setup must build one (goal-prompt "
                        "fallback removed)."
                    )
                # Queue-exhausted idle env: no scored episode here, so emit a
                # benign placeholder verb (result is masked out) instead of the
                # goal string. This is not a real-episode fallback.
                task_names.append(self._idle_placeholder_task)
                heatmaps.append(self._zero_heatmap)
                target_xys.append(np.array([np.nan, np.nan], dtype=np.float32))
                display_texts.append(task_names[-1])
                continue
            if self._stuck_mask[env_idx]:
                self._append_held_instruction(
                    env_idx, task_names, heatmaps, target_xys, display_texts, reason="stuck"
                )
                continue
            # A broken oracle (missing problem, empty verb) still raises. A
            # stuck place->pick cap is per-env: mark it and hold the last
            # overlay so the other envs keep evaluating.
            try:
                action_str = oracle.get_current_high_level_action(0)
            except OraclePlanStuck as exc:
                logger.error("%s %s", self._log_prefix(env_idx), exc)
                self._stuck_mask[env_idx] = True
                self._append_held_instruction(
                    env_idx, task_names, heatmaps, target_xys, display_texts, reason="stuck"
                )
                continue
            lang, xy = parse_oracle_action(action_str)
            self._log_grasp_progress(env_idx, oracle)

            if lang == "done":
                # Plan complete: flag the env for episode end and keep the LL
                # model in-distribution for the final step by repeating the
                # last real instruction ("done" never appears in training data).
                if not self._done_mask[env_idx]:
                    logger.info("%s plan complete -> requesting episode end",
                                self._log_prefix(env_idx))
                self._done_mask[env_idx] = True
                self._append_held_instruction(
                    env_idx, task_names, heatmaps, target_xys, display_texts, reason="done"
                )
                continue

            if not lang:
                raise RuntimeError(
                    f"{self._log_prefix(env_idx)} oracle produced an empty action "
                    f"from {action_str!r}; expected a verb."
                )
            if xy is None:
                # Every oracle verb except "done" (handled above) carries a pixel
                # target. A zero heatmap is a valid tensor meaning "no spatial
                # goal", so substituting one would evaluate the LL unconditioned
                # and still produce a success rate.
                raise RuntimeError(
                    f"{self._log_prefix(env_idx)} oracle action {action_str!r} carries "
                    "no pixel target, but the verb is not 'done'. Conditioning the LL "
                    "on a zero heatmap would silently drop the spatial goal."
                )
            task_names.append(format_task_name(lang, xy, self._language_mode))
            heatmaps.append(
                _build_heatmap(xy, self._orig_cam_w, self._orig_cam_h, self._target_sigma)
            )
            # The oracle's raw string drives the burn-in; task_name is rebuilt in
            # the checkpoint's language_mode and drives the LL policy. The two
            # differ in word order as well as content -- the oracle writes
            # "flip 310,195 left", training wrote "flip left 310,195".
            display_texts.append(action_str.strip() or task_names[-1])
            xy_native = np.asarray(xy, dtype=np.float32).reshape(2)
            target_xys.append(xy_native)
            self._last_task_names[env_idx] = task_names[-1]
            self._last_heatmaps[env_idx] = heatmaps[-1]
            self._last_target_xy[env_idx] = xy_native
            self._last_display_texts[env_idx] = display_texts[-1]

        self._display_texts = display_texts
        self._target_xy_batch = np.stack(target_xys, axis=0).astype(np.float32)
        # Log the HL instruction for each env only when it changes, so the run
        # log reads as a clean per-env action trace instead of per-step spam.
        # Idle (queue-exhausted) envs emit a masked placeholder — don't log them.
        for env_idx in range(self._num_envs):
            if self._active_setup[env_idx]:
                self._maybe_log_hl(env_idx, task_names[env_idx], display_texts[env_idx])
        batch = np.stack(heatmaps, axis=0)[:, None, ...].astype(np.float32)  # (B,1,3,224,224)
        return task_names, batch

    def _log_prefix(self, env_idx: int) -> str:
        """``[OracleHL] ep=<episode_index> env=<i>`` — episode-indexed log prefix."""
        ep = self._episode_indices[env_idx]
        return f"[OracleHL] ep={ep if ep is not None else '?'} env={env_idx}"

    def _maybe_log_hl(self, env_idx: int, task_name: str, display_text: str) -> None:
        """Emit an INFO line when env ``env_idx``'s HL instruction changes."""
        if task_name and task_name != self._logged_hl[env_idx]:
            logger.info("%s HL action: %s", self._log_prefix(env_idx), display_text or task_name)
            self._logged_hl[env_idx] = task_name

    @property
    def display_texts(self) -> list[str]:
        """Full HL action strings (with coords) from the last ``compute`` — burn-in text."""
        return list(self._display_texts)

    def done_indices(self) -> list[int]:
        """Envs whose oracle has completed its plan (as of the last ``compute()``).

        The harness should pass these to ``base_env.request_episode_end`` so the
        episode finalizes (with full evaluation) on the next ``env.step``.
        """
        return [i for i, d in enumerate(self._done_mask) if d]

    def final_command_indices(self) -> list[int]:
        """Envs whose oracle is on its plan's final command, or done (as of the last ``compute()``).

        The final command is the last plan step's place for a pnp step (its pick
        is not) or the step itself for a flip / rotate.
        """
        out: list[int] = []
        for i, oracle in enumerate(self._oracles):
            if oracle is None:
                continue
            if self._done_mask[i] or oracle.is_done_of(0):
                out.append(i)
            elif (
                oracle.current_plan_index_of(0) == oracle.plan_length - 1
                and oracle.phase_of(0) != "pick"
            ):
                out.append(i)
        return out

    def current_commands(
        self,
    ) -> list[Optional[tuple[str, Optional[str], Optional[str], Optional[str], int]]]:
        """Per env, the current command ``(verb, obj, place target, direction, plan index)``.

        ``verb`` is pick / place (by pnp phase), flip / rotate, or the plan action
        for other steps; ``obj`` is the block the step moves; the place target is
        ``None`` for a table placement. ``None`` for envs with no oracle or a
        finished plan (keep the last command).
        """
        out: list[Optional[tuple[str, Optional[str], Optional[str], Optional[str], int]]] = []
        for oracle in self._oracles:
            if oracle is None or oracle.is_done_of(0):
                out.append(None)
                continue
            plan_idx = oracle.current_plan_index_of(0)
            step = oracle.get_plan_summary()[plan_idx]
            action = step.get("action", "")
            sem = step.get("semantic") or {}
            args = step.get("args") or []
            obj = sem.get("obj") or (args[1] if len(args) > 1 else None)
            if action in ("reorient", "rotate"):
                raw_dir = sem.get("direction", "")
                verb = oracle._parse_direction(raw_dir)[0]
                out.append((verb, obj, None, raw_dir.lower(), plan_idx))
            elif oracle.phase_of(0) == "pick":
                out.append(("pick", obj, None, None, plan_idx))
            elif action.startswith("pnp_") or action in ("move_to_side", "move_to_center"):
                on = sem.get("target_obj") if action in ("pnp_table_block", "pnp_block_block") else None
                out.append(("place", obj, on, None, plan_idx))
            else:
                out.append((action, obj, None, None, plan_idx))
        return out

    def stuck_indices(self) -> list[int]:
        """Envs whose recovering oracle hit the place->pick cap this episode.

        Driving-oracle runs should end these episodes (``oracle_stuck`` metric).
        Comparison overlay should keep the last panel and let the HL model
        finish the episode.
        """
        return [i for i, s in enumerate(self._stuck_mask) if s]

    def _append_held_instruction(
        self,
        env_idx: int,
        task_names: list[str],
        heatmaps: list[np.ndarray],
        target_xys: list[np.ndarray],
        display_texts: list[str],
        *,
        reason: str,
    ) -> None:
        """Reuse the last real oracle verb/heatmap for ``done`` or ``stuck``.

        Both cases need a real verb: a zero heatmap would uncondition the LL,
        and an empty overlay would look like the comparison panel died.
        """
        last_task = self._last_task_names[env_idx]
        if last_task is None:
            raise RuntimeError(
                f"{self._log_prefix(env_idx)} oracle {reason} with no prior "
                "action; cannot hold a real verb for the LL / overlay."
            )
        last_hm = self._last_heatmaps[env_idx]
        if last_hm is None:
            raise RuntimeError(
                f"{self._log_prefix(env_idx)} has a last task name "
                f"({last_task!r}) but no last heatmap; the two are always "
                "recorded together."
            )
        last_xy = self._last_target_xy[env_idx]
        if last_xy is None:
            raise RuntimeError(
                f"{self._log_prefix(env_idx)} has a last heatmap but no last "
                "target_xy; the two are always recorded together."
            )
        task_names.append(last_task)
        heatmaps.append(last_hm)
        target_xys.append(np.asarray(last_xy, dtype=np.float32).reshape(2))
        display_texts.append(self._last_display_texts[env_idx] or last_task)

    # ------------------------------------------------------------------
    def _log_grasp_progress(self, env_idx: int, oracle: Any) -> None:
        """Log phase transitions, plan advancement, and grasp distance per env.

        Reads the oracle's public per-env accessors (``phase_of`` /
        ``current_plan_index_of``) after ``get_current_high_level_action`` has
        ticked them, so the pick->place transition and step advancement are
        reported *episode-indexed* here — the oracle's own prints always
        self-report ``env0`` (each oracle is built ``n_envs=1``).
        """
        phase = oracle.phase_of(0)
        plan_idx = oracle.current_plan_index_of(0)
        prev_phase = self._last_phase[env_idx]
        prev_idx = self._last_plan_idx[env_idx]

        if prev_idx is not None and plan_idx != prev_idx:
            logger.info(
                "%s advanced to plan step %d/%d (phase=%s)",
                self._log_prefix(env_idx), plan_idx, oracle.plan_length, phase,
            )
            self._pick_log_step[env_idx] = 0
        elif prev_phase is not None and phase != prev_phase:
            dist = self._grasp_distance(env_idx, oracle, plan_idx)
            dist_str = f" (cube-to-TCP={dist:.4f} m)" if dist is not None else ""
            logger.info(
                "%s phase %s -> %s at step %d%s",
                self._log_prefix(env_idx), prev_phase, phase, plan_idx, dist_str,
            )

        # While picking, periodically report cube-to-TCP so a re-run shows the
        # distance drop below the grasp threshold (confirms the TCP correction).
        if phase == "pick":
            step = self._pick_log_step[env_idx]
            if step % 10 == 0:
                dist = self._grasp_distance(env_idx, oracle, plan_idx)
                if dist is not None:
                    logger.info(
                        "%s pick step %d: cube-to-TCP=%.4f m",
                        self._log_prefix(env_idx), step, dist,
                    )
            self._pick_log_step[env_idx] = step + 1

        self._last_phase[env_idx] = phase
        self._last_plan_idx[env_idx] = plan_idx

    def _grasp_distance(
        self, env_idx: int, oracle: Any, plan_idx: int
    ) -> Optional[float]:
        """Cube-to-TCP distance (m) for the current pnp pick target, or ``None``."""
        target = self._pnp_pick_target(oracle, plan_idx)
        if not target:
            return None
        adapter = self._adapters[env_idx]
        try:
            obj_xyz = adapter.get_pose(target, local=False)[:3]
            ee_xyz = adapter.ee_world_pos()
            return float(torch.linalg.norm(obj_xyz - ee_xyz).item())
        except Exception:
            return None

    @staticmethod
    def _pnp_pick_target(oracle: Any, plan_idx: int) -> Optional[str]:
        """Pick-target object name for the current step if it is a pnp action."""
        plan = oracle.get_plan_summary()
        if plan_idx < 0 or plan_idx >= len(plan):
            return None
        step = plan[plan_idx]
        action = step.get("action", "")
        if not (action.startswith("pnp_") or action in ("move_to_side", "move_to_center")):
            return None
        sem = step.get("semantic") or {}
        if sem.get("obj"):
            return sem["obj"]
        args = step.get("args") or []
        return args[1] if len(args) > 1 else None
