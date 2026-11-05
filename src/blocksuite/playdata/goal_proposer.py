"""Random goal proposal for play-data collection.

Goals are sampled from the *current* symbolic state of an environment so that
they are (a) not already satisfied and (b) expressible in the blocks domain.
Nothing here talks to the simulator: it works off a
:class:`~blocksuite.playdata.symbolic_state.SymbolicState` plus an optional
callback that reports whether a candidate predicate already holds.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass, field

from blocksuite.playdata.symbolic_state import ORI_NAMES, SymbolicState

# A goal family is a name plus a sampler that returns one PDDL predicate
# string, or None when the current state offers no candidate.
GOAL_FAMILIES: tuple[str, ...] = ("stack", "unstack", "ori", "next_to", "on_right")

# Canonical upright orientation, used by recovery goals.
_UPRIGHT_ORI = "o1"

HoldsFn = Callable[[str, list[str]], bool]


@dataclass
class GoalProposerCfg:
    """Which random goals to draw, and how many per problem."""

    families: tuple[str, ...] = GOAL_FAMILIES
    """Enabled goal families (subset of :data:`GOAL_FAMILIES`)."""
    weights: dict[str, float] = field(default_factory=dict)
    """Optional per-family sampling weight; families default to weight 1.0."""
    num_predicates: int = 1
    """How many goal predicates to conjoin. Each one targets a distinct block."""
    ori_blocks: tuple[str, ...] | None = None
    """Blocks eligible for ``ori_of`` goals. ``None`` means every block."""
    max_attempts: int = 64
    """Rejection-sampling budget per goal predicate."""


@dataclass
class ProposedGoal:
    """A sampled goal: the PDDL predicate lines plus a readable label."""

    predicates: list[str]
    families: list[str]

    @property
    def text(self) -> str:
        return " ".join(self.predicates)

    def render(self, indent: int = 8) -> str:
        pad = " " * indent
        return "\n".join(f"{pad}{p}" for p in self.predicates)


class RandomGoalProposer:
    """Sample random, currently-unsatisfied goals for the blocks domain."""

    def __init__(self, cfg: GoalProposerCfg, rng: random.Random | None = None) -> None:
        unknown = [f for f in cfg.families if f not in GOAL_FAMILIES]
        if unknown:
            raise ValueError(f"Unknown goal families: {unknown}")
        if not cfg.families:
            raise ValueError("At least one goal family must be enabled")
        self.cfg = cfg
        self.rng = rng or random.Random()

    # -- family samplers ------------------------------------------------------

    def _sample_stack(self, state: SymbolicState, avoid: set[str]) -> tuple[str, list[str]] | None:
        """``(on b1 b2)`` — put one block on top of another."""
        blocks = [b for b in state.blocks if b not in avoid]
        self.rng.shuffle(blocks)
        for b1 in blocks:
            targets = [b for b in state.blocks if b != b1 and b not in avoid]
            self.rng.shuffle(targets)
            for b2 in targets:
                if state.supporter(b1) == b2:
                    continue
                return "on", [b1, b2]
        return None

    def _sample_unstack(
        self, state: SymbolicState, avoid: set[str]
    ) -> tuple[str, list[str]] | None:
        """``(on_table b t)`` — bring a stacked block back down."""
        stacked = [b for b in state.blocks if state.supporter(b) is not None and b not in avoid]
        if not stacked:
            return None
        return "on_table", [self.rng.choice(stacked), state.table]

    def _sample_ori(self, state: SymbolicState, avoid: set[str]) -> tuple[str, list[str]] | None:
        """``(ori_of b oN)`` — reorient a block (rotate / flip)."""
        allowed = self.cfg.ori_blocks
        candidates = [
            b for b in state.blocks if b not in avoid and (allowed is None or b in allowed)
        ]
        if not candidates:
            return None
        block = self.rng.choice(candidates)
        current = state.ori.get(block)
        options = [o for o in ORI_NAMES if o != current]
        if not options:
            return None
        return "ori_of", [block, self.rng.choice(options)]

    def _sample_pair_on_table(
        self, state: SymbolicState, avoid: set[str], predicate: str
    ) -> tuple[str, list[str]] | None:
        """``(next_to b1 b2)`` / ``(on_right b1 b2)`` — a pushing goal."""
        blocks = [b for b in state.blocks if b not in avoid]
        if len(blocks) < 2:
            return None
        b1, b2 = self.rng.sample(blocks, 2)
        return predicate, [b1, b2]

    def _sample_family(
        self, family: str, state: SymbolicState, avoid: set[str]
    ) -> tuple[str, list[str]] | None:
        if family == "stack":
            return self._sample_stack(state, avoid)
        if family == "unstack":
            return self._sample_unstack(state, avoid)
        if family == "ori":
            return self._sample_ori(state, avoid)
        if family == "next_to":
            return self._sample_pair_on_table(state, avoid, "next_to")
        if family == "on_right":
            return self._sample_pair_on_table(state, avoid, "on_right")
        raise ValueError(f"Unknown goal family: {family}")

    # -- public API -----------------------------------------------------------

    def propose_recovery(
        self, state: SymbolicState, holds: HoldsFn | None = None
    ) -> ProposedGoal | None:
        """Propose an easy goal that tends to undo an awkward state.

        Play states drift: a block ends up flipped onto a corner, or buried in a
        stack, and then nothing the sampler picks can be planned — each attempt
        burns a full planning budget and starves the other envs of planner
        slots. Getting blocks upright and back on the table is the cheapest way
        out, so after a planning failure that is tried before random sampling.
        """
        candidates: list[tuple[str, list[str]]] = []
        for block in state.blocks:
            if state.ori.get(block) != _UPRIGHT_ORI:
                candidates.append(("ori_of", [block, _UPRIGHT_ORI]))
        for block in state.blocks:
            if state.supporter(block) is not None:
                candidates.append(("on_table", [block, state.table]))
        self.rng.shuffle(candidates)

        for name, args in candidates:
            if holds is not None and holds(name, args):
                continue
            return ProposedGoal(predicates=[f"({name} {' '.join(args)})"], families=["recovery"])
        return None

    def propose(self, state: SymbolicState, holds: HoldsFn | None = None) -> ProposedGoal | None:
        """Sample a goal for ``state``.

        Args:
            state: Current symbolic state of the environment.
            holds: Optional predicate oracle. Candidates that already hold are
                rejected so that every episode has something to do.

        Returns:
            The proposed goal, or None when no family could produce an
            unsatisfied candidate within the attempt budget.
        """
        families = list(self.cfg.families)
        weights = [self.cfg.weights.get(f, 1.0) for f in families]

        predicates: list[str] = []
        chosen_families: list[str] = []
        used_blocks: set[str] = set()

        for _ in range(max(1, self.cfg.num_predicates)):
            for _attempt in range(self.cfg.max_attempts):
                family = self.rng.choices(families, weights=weights, k=1)[0]
                candidate = self._sample_family(family, state, used_blocks)
                if candidate is None:
                    continue
                name, args = candidate
                if holds is not None and holds(name, args):
                    continue
                predicates.append(f"({name} {' '.join(args)})")
                chosen_families.append(family)
                # Keep conjunctions non-contradictory: one goal per subject block.
                used_blocks.add(args[0])
                break

        if not predicates:
            return None
        return ProposedGoal(predicates=predicates, families=chosen_families)
