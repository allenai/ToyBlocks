"""The goal-policy contract the play-data orchestrator talks to.

The orchestrator no longer depends on a concrete goal *proposer*; it depends on
a :class:`GoalPolicy` — anything that, given an environment's current symbolic
state, decides what that environment should do next. This is the single seam
that lets a stateful search strategy (see :mod:`toyblocks.playdata.dfa`) stand
in for the stateless :class:`~toyblocks.playdata.goal_proposer.RandomGoalProposer`
without touching the planner, executor, recorder, or video paths.

A policy answers each step with a :class:`PolicyDecision`:

* ``GOAL`` — pursue ``goal`` (a :class:`~toyblocks.playdata.goal_proposer.ProposedGoal`),
  which flows on to the PDDL writer and the planner pool exactly as a random
  goal would.
* ``RETRY`` — nothing to do from this state yet; the orchestrator re-snapshots
  and asks again shortly.
* ``TERMINATE`` — the strategy has finished for this env (accepted or rejected).
  The orchestrator resets the env to a fresh reset-state, resets the policy, and
  starts a new run — mirroring the stage-3 pipeline's per-plan reset.

Recovery after a planning failure is handled by the orchestrator itself, not the
policy, so a strategy never has to reason about an awkward physical state it did
not intend to create.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable

from toyblocks.playdata.goal_proposer import ProposedGoal, RandomGoalProposer
from toyblocks.playdata.symbolic_state import SymbolicState

#: Predicate oracle: ``holds("ori_of", ["block_r", "o2"]) -> bool``.
HoldsFn = Callable[[str, list[str]], bool]


class Decision(Enum):
    """What a policy wants the orchestrator to do this step."""

    GOAL = auto()
    """Pursue :attr:`PolicyDecision.goal`."""
    RETRY = auto()
    """No goal available from the current state; re-snapshot and ask again."""
    TERMINATE = auto()
    """The strategy is finished for this env; reset it and start over."""


@dataclass
class PolicyDecision:
    """A policy's answer for one environment at one step."""

    kind: Decision
    goal: ProposedGoal | None = None
    reason: str = ""
    """Human-readable label for logs / the monitor (e.g. the fired transition)."""


@runtime_checkable
class GoalPolicy(Protocol):
    """What the orchestrator requires of a goal-selection strategy."""

    def reset(self, env_idx: int) -> None:
        """Put ``env_idx``'s per-env strategy state back at its start."""
        ...

    def next_goal(self, env_idx: int, state: SymbolicState, holds: HoldsFn) -> PolicyDecision:
        """Decide what ``env_idx`` should do next, given its current state."""
        ...

    def on_episode_result(self, env_idx: int, reached: bool) -> None:
        """Feedback after an executed plan: whether its goal was reached."""
        ...


class RandomGoalPolicy:
    """Adapter that makes the stateless random proposer a :class:`GoalPolicy`.

    This is the default policy: with no ``--dfa-spec`` the run behaves exactly as
    it did before the policy seam existed. Recovery is intentionally *not* done
    here — the orchestrator applies it uniformly for every policy.
    """

    def __init__(self, proposer: RandomGoalProposer) -> None:
        self.proposer = proposer

    def reset(self, env_idx: int) -> None:  # noqa: ARG002 - stateless
        return None

    def next_goal(self, env_idx: int, state: SymbolicState, holds: HoldsFn) -> PolicyDecision:  # noqa: ARG002
        goal = self.proposer.propose(state, holds=holds)
        if goal is None:
            return PolicyDecision(Decision.RETRY, reason="no unsatisfied goal available")
        return PolicyDecision(Decision.GOAL, goal=goal)

    def on_episode_result(self, env_idx: int, reached: bool) -> None:  # noqa: ARG002 - stateless
        return None
