"""The eval oracle: :class:`~toyblocks.eval.oracle.hl_policy.OracleHighLevelPolicy` with
place->pick recovery, settle-gated step completion and re-planning when the world drifts."""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from toyblocks.eval.oracle import live_state as ls
from toyblocks.eval.oracle.hl_policy import _EVALUABLE_PREDICATES, OracleHighLevelPolicy

# Shared with the runtime success gate so the oracle's "settled" notion and
# the PDDL goal success gate agree (see pddl_goal_tensor_conversions.py).
from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import _SETTLED_ANG_VEL, _SETTLED_LIN_VEL

logger = logging.getLogger(__name__)


class OraclePlanStuck(RuntimeError):
    """One env's recovering oracle hit the place->pick cap for a plan step.

    This is a per-env outcome (the policy keeps missing the placement), not a
    broken oracle. :class:`OracleHLController` marks that env stuck and the
    eval harness ends the episode (driving oracle) or keeps the last overlay
    (comparison). It must not abort ``evaluate.py``.
    """


class RecoveringOracleHLPolicy(OracleHighLevelPolicy):
    """The oracle the eval harness runs: :class:`OracleHighLevelPolicy` plus recovery.

    The base oracle never leaves the 'place' phase until on/on_table holds, so a
    dropped/failed placement stalls forever. We re-pick the same object.

    We also defer plan advancement (and therefore 'done') for pnp steps until
    the placed block is actually released *and* settled: the base commits the
    moment the pose-only ``on``/``on_table`` effect holds, which is true while
    the block is still grasped/hovering, so a fall-off at the final step would
    otherwise latch 'done' instead of being retried.

    The plan's last action also gets checkable effects: FF prints the state before
    each action but none after the last, so upstream gave it none and it could never
    complete. For a plain-conjunction goal they are the goal literals still false
    before it (other goals end in FF's ``REACH-GOAL``, which already has a state).

    With ``replan``, the current step's preconditions are re-checked on the live sim
    (pick block and place target clear; a rotated / flipped block on the table): at
    every step boundary, and every ``_DRIFT_CHECK_EVERY`` ticks while picking or
    rotating / flipping (a skill that never completes has no boundary; two
    consecutive failing checks are needed there, and the place phase is skipped
    because the held block hovers over its target). When the world has drifted from
    the plan, FF re-plans from the plan's expected state with
    ``on``/``on_table``/``clear``/``ori_of`` read off the sim, at most
    ``max_replans`` times per episode.
    """

    _PLACE_SETTLE_STEPS = 20
    # Cap on place->pick reverts for a single plan step. Recovery is meant to
    # absorb an occasional dropped placement, not to loop: without a cap a
    # chronically-misplacing policy re-picks forever, the episode runs to the
    # episode_length_s timeout, and the result reads as a plain failure with no
    # indication the oracle was stuck.
    _MAX_PLACE_RECOVERIES = 5
    _DRIFT_CHECK_EVERY = 10  # ticks between mid-step drift checks

    def __init__(
        self, *args: Any, replan: bool = False, max_replans: int = 3, **kwargs: Any
    ) -> None:
        # Set before super().__init__, which plans (and so calls the overrides below).
        self._domain_file = kwargs["domain_file"]
        self._problem_text = Path(kwargs["problem_file"]).read_text()
        self._goal_literals = ls.flat_goal_literals(self._problem_text)
        self._ff_states: list[set[str]] = []
        self._step_ff_idx: list[int] = []  # plan step -> its FF step index (1-based)
        self._replan_enabled = bool(replan)
        self._max_replans = int(max_replans)
        self.replans = 0
        self._eval_errors_logged: set[tuple[str, str]] = set()
        self._ticks: dict[int, int] = {}
        self._drift_pending: dict[int, str] = {}  # env -> drift seen once mid-step
        self._budget_logged: set[tuple[int, int]] = set()
        self._ready = False
        self._replanning = False
        super().__init__(*args, **kwargs)
        # _EnvState uses __slots__, so these counters can't live there.
        self._place_settle_steps: dict[int, int] = {}
        self._place_recoveries: dict[tuple[int, int], int] = {}
        # The demonstration's "place on table" target pixel (set by the
        # controller from the episode's recorded goal.position). A table
        # placement's plan step names no spot, so without it upstream samples
        # a random free one the LL never saw in training.
        self.demo_place_on_table_px: tuple[int, int] | None = None
        self._ready = True

    def _sample_table_placement(self, env_idx: int) -> tuple[int, int]:
        if self.demo_place_on_table_px is not None:
            return self.demo_place_on_table_px
        return super()._sample_table_placement(env_idx)

    # -- plan bookkeeping (called from upstream _compute_plan) -------------------
    def _parse_ff_states(self, raw_output: str) -> list[set[str]]:
        self._ff_states = OracleHighLevelPolicy._parse_ff_states(raw_output)
        return self._ff_states

    def _compute_effect_predicates(
        self, ff_states: list, step_ff_idx: int, **kwargs: Any
    ) -> list[dict]:
        self._step_ff_idx.append(step_ff_idx)
        # kwargs: TAMP's goal_props, the goal the last step is checked against.
        effects = super()._compute_effect_predicates(ff_states, step_ff_idx, **kwargs)
        if effects or not ff_states or step_ff_idx != len(ff_states):
            return effects
        # The plan's last action: FF printed no state after it.
        before = {ls.norm_fact(p) for p in ff_states[-1]}
        effects = [
            {"name": n, "args": list(a), "prop_str": ls.fact(n, a)}
            for n, a in self._goal_literals
            if n in _EVALUABLE_PREDICATES and ls.fact(n, a) not in before
        ]
        if not effects:
            logger.warning(
                "[OracleHL] the plan's last action has no checkable effects (goal is not a "
                "plain conjunction of %s literals); it cannot complete on its own",
                sorted(_EVALUABLE_PREDICATES),
            )
        return effects

    def _expected_state(self, plan_idx: int) -> set[str]:
        """The plan's symbolic state before step ``plan_idx`` (FF's printed state)."""
        return {ls.norm_fact(p) for p in self._ff_states[self._step_ff_idx[plan_idx] - 1]}

    # -- re-planning when the world drifts from the plan -------------------------
    def get_current_high_level_action(self, env_idx: int = 0) -> str:
        if self._ready and self._replan_enabled:
            n = self._ticks.get(env_idx, 0) + 1
            self._ticks[env_idx] = n
            phase = self._env_states[env_idx].phase
            if phase in ("pick", "active") and n % self._DRIFT_CHECK_EVERY == 0:
                self._replan_if_drifted(env_idx, confirm=True)
        return super().get_current_high_level_action(env_idx)

    def _set_phase_for_current_action(self, env_idx: int) -> None:
        super()._set_phase_for_current_action(env_idx)
        if self._ready and self._replan_enabled and not self._replanning:
            self._drift_pending.pop(env_idx, None)
            self._replan_if_drifted(env_idx)

    def _replan_if_drifted(self, env_idx: int, confirm: bool = False) -> None:
        """Re-plan when the current step's preconditions fail on the live sim.

        ``confirm``: only on the second consecutive failing check (mid-step checks).
        """
        st = self._env_states[env_idx]
        if st.plan_idx >= len(self._plan) or not self._ff_states:
            return
        step = self._plan[st.plan_idx]
        expected = self._expected_state(st.plan_idx)
        blocks, table = ls.blocks_and_table(expected)
        sim = self._simulator
        live = ls.read_live_state(sim.base_env, sim.env_index, blocks)
        action = step["hl_action"]
        target = self._place_target(step) if self._is_pnp_action(action) else None
        why = ls.step_violations(action, self._pick_object(step), target, live)
        if not why:
            self._drift_pending.pop(env_idx, None)
            return
        desc = f"step {st.plan_idx} {action} {step['hl_args']}: {'; '.join(why)}"
        if confirm and self._drift_pending.get(env_idx) != desc:
            self._drift_pending[env_idx] = desc
            return
        self._drift_pending.pop(env_idx, None)
        if self.replans >= self._max_replans:
            if (env_idx, st.plan_idx) not in self._budget_logged:
                self._budget_logged.add((env_idx, st.plan_idx))
                logger.warning(
                    "[OracleHL] world drifted from the plan at %s, but the re-plan budget "
                    "(%d) is spent; keeping the plan", desc, self._max_replans,
                )
            return
        self.replans += 1
        gripper = next(
            (ls.parse_fact(f)[1][0] for f in expected if f.startswith("(handempty ")), "robot"
        )
        init = ls.corrected_init(expected, live, table, gripper)
        old = (self._plan, self._ff_states, self._step_ff_idx)
        self._plan, self._ff_states, self._step_ff_idx = [], [], []
        # FF prints no plan for a goal that already holds ("simplified to TRUE"), which
        # upstream reads as a failure: the drifted world may simply be done.
        if not (self._goal_literals and {ls.fact(n, a) for n, a in self._goal_literals} <= set(init)):
            fd, path = tempfile.mkstemp(suffix=".pddl", prefix="oracle_replan_")
            with os.fdopen(fd, "w") as fh:
                fh.write(ls.replace_init(self._problem_text, init))
            try:
                self._compute_plan(self._domain_file, path)
            except Exception as exc:
                if "simplified to TRUE" not in str(exc):
                    self._plan, self._ff_states, self._step_ff_idx = old
                    logger.error(
                        "[OracleHL] re-plan after %s failed (%r); keeping the plan", desc, exc
                    )
                    return
                self._plan, self._ff_states, self._step_ff_idx = [], [], []
            finally:
                os.unlink(path)
        logger.warning(
            "[OracleHL] world drifted from the plan at %s -> re-planned from the live "
            "state (%d/%d): %s", desc, self.replans, self._max_replans,
            [f"{s['hl_action']} {' '.join(s['hl_args'])}" for s in self._plan],
        )
        st.plan_idx = 0
        self._place_recoveries.clear()
        self._place_settle_steps[env_idx] = 0
        self._replanning = True
        try:
            self._set_phase_for_current_action(env_idx)
        finally:
            self._replanning = False

    def _placement_stable(self, env_idx: int) -> bool:
        """True when the current pnp step's block is released and at rest."""
        st = self._env_states[env_idx]
        if st.plan_idx >= len(self._plan):
            return True
        step = self._plan[st.plan_idx]
        obj = self._pick_object(step)
        if not obj:
            return True
        if self._is_object_grasped(env_idx):
            return False
        vel = self._simulator.get_velocity(obj, env_idx=env_idx)
        lin = float(torch.linalg.norm(vel[:3]).item())
        ang = float(torch.linalg.norm(vel[3:]).item())
        return lin <= _SETTLED_LIN_VEL and ang <= _SETTLED_ANG_VEL

    def _advance(self, env_idx: int) -> None:
        # Defer pnp completion (and reaching 'done') until the just-placed
        # block is released + settled. Upstream calls _advance only after the
        # pose-on effect already holds, so this delays *only* the commit.
        st = self._env_states[env_idx]
        if st.plan_idx < len(self._plan):
            step = self._plan[st.plan_idx]
            if self._is_pnp_action(step["hl_action"]) and not self._placement_stable(env_idx):
                return
        super()._advance(env_idx)

    def _update_phase(self, env_idx: int) -> None:
        prev_idx = self._env_states[env_idx].plan_idx
        super()._update_phase(env_idx)
        st = self._env_states[env_idx]
        if st.plan_idx != prev_idx or st.phase != "place":
            self._place_settle_steps[env_idx] = 0
            return
        # Still placing the same step: count steps the object is released/lost.
        if self._is_object_grasped(env_idx):
            self._place_settle_steps[env_idx] = 0
            return
        # Released but held in 'place' because _advance was deferred: if the
        # pose-on effect still holds the block is on target and merely
        # settling, so wait (don't count toward a false re-pick).
        if self._evaluate_effect_predicates(env_idx):
            self._place_settle_steps[env_idx] = 0
            return
        n = self._place_settle_steps.get(env_idx, 0) + 1
        self._place_settle_steps[env_idx] = n
        if n > self._PLACE_SETTLE_STEPS:
            key = (env_idx, st.plan_idx)
            attempts = self._place_recoveries.get(key, 0) + 1
            if attempts > self._MAX_PLACE_RECOVERIES:
                raise OraclePlanStuck(
                    f"[OracleHL][env{env_idx}] placement for plan step "
                    f"{st.plan_idx} failed {attempts - 1} times in a row"
                )
            self._place_recoveries[key] = attempts
            st.phase = "pick"
            st.pick_phase_steps = 0
            st.pick_init_z = self._get_pick_object_z(env_idx)  # wherever it landed
            st.cached_table_pixel = None
            self._place_settle_steps[env_idx] = 0
            print(
                f"[OracleHL][env{env_idx}] placement failed (object released, "
                f"effects unmet) -> reverting to 'pick' "
                f"(attempt {attempts}/{self._MAX_PLACE_RECOVERIES})"
            )


def make_oracle(
    *,
    domain_file: str,
    problem_file: str,
    simulator: Any,
    scene: Any,
    action_config_path: str | None = None,
    include_coords: bool = True,
    replan: bool = False,
    max_replans: int = 3,
) -> RecoveringOracleHLPolicy:
    """A single-env :class:`RecoveringOracleHLPolicy`."""
    return RecoveringOracleHLPolicy(
        domain_file=domain_file,
        problem_file=problem_file,
        simulator=simulator,
        scene=scene,
        n_envs=1,
        action_config_path=action_config_path,
        include_coords=include_coords,
        replan=replan,
        max_replans=max_replans,
    )
