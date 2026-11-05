r"""The DFA engine: a Mealy machine that emits play-data goals.

This module defines the typed representation of a DFA strategy, the per-env
runtime that steps it, and :class:`DFAGoalPolicy`, which adapts a DFA to the
:class:`~toyblocks.playdata.goal_policy.GoalPolicy` contract.

===========================================================================
DFA SPEC GRAMMAR (this is what a YAML/JSON strategy file must contain)
===========================================================================

A spec is a mapping with these top-level keys::

    name:    str            # identifier for logs (default "dfa")
    initial: str            # the start state (required; must be a key of `states`)
    memory:  {str: any}     # initial per-env memory bindings (default {})
    params:  {str: any}     # substitution values for $tokens (default {})
    states:  {str: [Transition, ...]}   # required

`states` maps a state name to an ORDERED list of transitions. Two state names
are reserved and must NOT appear in `states`: ``ACCEPT`` and ``REJECT``. Both are
terminal — reaching either makes the policy return ``TERMINATE`` (the
orchestrator then resets the env and restarts the DFA).

A **Transition** is a mapping::

    guard:   GuardNode      # default "always"
    emit:    MoveNode|null  # a goal to pursue, or null for a pure transition
    effects: [EffectNode]   # memory mutations applied when this transition fires
    next:    str            # required; a state name, "ACCEPT", or "REJECT"

**Named nodes.** Guards, moves, and effects are written as either a bare string
(no params) or a single-key mapping ``{name: {param: value, ...}}``::

    always                                  # guard, no params
    {counter_lt: {name: n, n: 2}}           # guard with params
    {flip: {var: c, direction: forward}}    # move
    {incr: {name: n}}                        # effect

**Guard combinators** (handled by the engine, not the registry)::

    {not: <GuardNode>}
    {all: [<GuardNode>, ...]}     # conjunction
    {any: [<GuardNode>, ...]}     # disjunction

**$param substitution.** Any string value of the exact form ``$key`` anywhere
outside the `params` block is replaced at load time with ``params[key]`` (or a
CLI/programmatic override). This lets one spec parameterize many runs
(``letter: $letter``, ``n: $flips_per_block``).

Available guard / effect / move names and their parameters are the keys of
:data:`~toyblocks.playdata.dfa.registries.GUARDS` / ``EFFECTS`` / ``MOVES``;
call :func:`~toyblocks.playdata.dfa.registries.catalog` for a machine-readable
listing (names, params, defaults, docs) suitable for auto-generating specs.

===========================================================================
EXECUTION SEMANTICS (one `next_goal` call)
===========================================================================

Starting from the env's current state, the engine loops:

1. If the state is ``ACCEPT``/``REJECT`` -> return ``TERMINATE``.
2. Evaluate the state's transitions top-to-bottom; the FIRST whose guard passes
   fires. If none passes -> return ``RETRY`` (nothing to do yet).
3. Apply the fired transition's ``effects`` (in order), then move to ``next``.
4. If it has an ``emit``:
     * the move yields a goal  -> return ``GOAL`` (the caller plans/executes it);
     * the move yields nothing -> return ``RETRY``.
   If ``emit`` is null (a *pure transition*), loop back to step 1 immediately —
   so chains like ``PICK -> (bind var, reset counter) -> FLIP`` resolve within a
   single call without wasting a planning round. A ``max_chain`` bound guards
   against a misconfigured cycle of pure transitions.

Memory is a plain dict, one per env, mutated by effects and read by guards/moves.
The reserved key ``_reached`` is set by :meth:`DFAGoalPolicy.on_episode_result`
to the last executed plan's goal-reached flag, so a guard can branch on it.
"""

from __future__ import annotations

import copy
import logging
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

from toyblocks.playdata.dfa.context import DFACtx
from toyblocks.playdata.dfa.context import glyph_of as default_glyph_of
from toyblocks.playdata.dfa.registries import COMBINATORS, EFFECTS, GUARDS, MOVES
from toyblocks.playdata.goal_policy import Decision, HoldsFn, PolicyDecision
from toyblocks.playdata.symbolic_state import SymbolicState

ACCEPT = "ACCEPT"
REJECT = "REJECT"
TERMINAL: frozenset[str] = frozenset({ACCEPT, REJECT})


@dataclass
class GuardSpec:
    """A parsed guard: a registry name + params, or a combinator + children."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    children: list[GuardSpec] = field(default_factory=list)


@dataclass
class MoveSpec:
    """A parsed move: a registry name + params."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class EffectSpec:
    """A parsed effect: a registry name + params."""

    name: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Transition:
    """One edge out of a state."""

    guard: GuardSpec
    next: str
    emit: MoveSpec | None = None
    effects: list[EffectSpec] = field(default_factory=list)


@dataclass
class DFASpec:
    """A fully parsed, validated DFA strategy."""

    name: str
    initial: str
    states: dict[str, list[Transition]]
    memory0: dict[str, Any] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DFARuntime:
    """Per-env execution state: current node, memory, and RNG."""

    state: str
    memory: dict[str, Any]
    rng: random.Random


class DFAGoalPolicy:
    """Drive a :class:`DFASpec` per environment as a
    :class:`~toyblocks.playdata.goal_policy.GoalPolicy`.

    Each env gets its own :class:`DFARuntime` and a persistent RNG seeded from
    ``seed + env_idx``. The RNG persists across resets so restarts after an
    ``ACCEPT``/``REJECT`` diverge from earlier runs while the whole run stays
    reproducible.
    """

    def __init__(
        self,
        spec: DFASpec,
        *,
        seed: int = 0,
        table: str = "workspace_table",
        glyph_of: Callable[[str], str | None] = default_glyph_of,
        pose_of: Callable[[int, str], list[float] | None] | None = None,
        max_chain: int = 64,
    ) -> None:
        self.spec = spec
        self.base_seed = int(seed)
        self.table = table
        self._glyph_of = glyph_of
        self._pose_of = pose_of
        self.max_chain = max_chain
        self._runtimes: dict[int, DFARuntime] = {}
        self._rngs: dict[int, random.Random] = {}

    # -- optional scene binding (called by the orchestrator once the sim exists) --

    def set_scene_accessors(
        self,
        *,
        pose_of: Callable[[int, str], list[float] | None],
        table: str | None = None,
    ) -> None:
        """Bind the live pose accessor (and table) after the simulator is built."""
        self._pose_of = pose_of
        if table:
            self.table = table

    # -- GoalPolicy protocol --------------------------------------------------

    def reset(self, env_idx: int) -> None:
        if env_idx not in self._rngs:
            self._rngs[env_idx] = random.Random(self.base_seed * 100003 + env_idx * 7919 + 1)
        self._runtimes[env_idx] = DFARuntime(
            state=self.spec.initial,
            memory=copy.deepcopy(self.spec.memory0),
            rng=self._rngs[env_idx],
        )

    def on_episode_result(self, env_idx: int, reached: bool) -> None:
        runtime = self._runtimes.get(env_idx)
        if runtime is not None:
            runtime.memory["_reached"] = bool(reached)

    def next_goal(self, env_idx: int, state: SymbolicState, holds: HoldsFn) -> PolicyDecision:
        runtime = self._runtimes.get(env_idx)
        if runtime is None:
            self.reset(env_idx)
            runtime = self._runtimes[env_idx]

        pose_of = self._pose_of
        ctx = DFACtx(
            env_idx=env_idx,
            state=state,
            memory=runtime.memory,
            rng=runtime.rng,
            holds=holds,
            table=self.table,
            glyph_of=self._glyph_of,
            pose_of=(lambda b: pose_of(env_idx, b)) if pose_of is not None else (lambda b: None),
        )

        for _ in range(self.max_chain):
            if runtime.state in TERMINAL:
                return PolicyDecision(Decision.TERMINATE, reason=runtime.state)

            transitions = self.spec.states.get(runtime.state, [])
            fired = next((t for t in transitions if self._eval_guard(t.guard, ctx)), None)
            if fired is None:
                return PolicyDecision(
                    Decision.RETRY, reason=f"no transition fired in state {runtime.state}"
                )

            for eff in fired.effects:
                self._apply_effect(eff, ctx)
            prev = runtime.state
            runtime.state = fired.next

            if fired.emit is not None:
                goal = self._eval_move(fired.emit, ctx)
                if goal is None:
                    return PolicyDecision(
                        Decision.RETRY, reason=f"move {fired.emit.name} produced no goal"
                    )
                return PolicyDecision(
                    Decision.GOAL, goal=goal, reason=f"{prev}->{runtime.state}:{fired.emit.name}"
                )
            # Pure transition: keep chaining within this call.

        logger.warning(
            f"[DFA] {self.spec.name}: chain limit ({self.max_chain}) hit from state "
            f"{runtime.state}; check for a cycle of pure (emit-less) transitions"
        )
        return PolicyDecision(Decision.RETRY, reason="chain limit reached")

    # -- evaluation -----------------------------------------------------------

    def _eval_guard(self, g: GuardSpec, ctx: DFACtx) -> bool:
        if g.name == "not":
            return not self._eval_guard(g.children[0], ctx)
        if g.name == "all":
            return all(self._eval_guard(c, ctx) for c in g.children)
        if g.name == "any":
            return any(self._eval_guard(c, ctx) for c in g.children)
        fn = GUARDS.get(g.name)
        if fn is None:
            raise KeyError(f"unknown guard {g.name!r} (available: {sorted(GUARDS)})")
        return bool(fn(ctx, **g.params))

    def _apply_effect(self, e: EffectSpec, ctx: DFACtx) -> None:
        fn = EFFECTS.get(e.name)
        if fn is None:
            raise KeyError(f"unknown effect {e.name!r} (available: {sorted(EFFECTS)})")
        fn(ctx, **e.params)

    def _eval_move(self, m: MoveSpec, ctx: DFACtx) -> Any:
        fn = MOVES.get(m.name)
        if fn is None:
            raise KeyError(f"unknown move {m.name!r} (available: {sorted(MOVES)})")
        return fn(ctx, **m.params)


__all__ = [
    "ACCEPT",
    "REJECT",
    "TERMINAL",
    "COMBINATORS",
    "GuardSpec",
    "MoveSpec",
    "EffectSpec",
    "Transition",
    "DFASpec",
    "DFARuntime",
    "DFAGoalPolicy",
]
