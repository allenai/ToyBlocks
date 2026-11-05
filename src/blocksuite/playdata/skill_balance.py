"""Balanced skill mix for play data: roughly equal pick-and-place, flips and rotates.

Random goals give a lopsided dataset: a random ``(ori_of b oN)`` is usually several
flips and rotates away, and flip directions come out wherever the planner lands.
:class:`BalancedSkillGoalPolicy` instead picks, for every goal, the skill class
that is furthest behind its target share and asks for a goal that exactly one
action of that class achieves:

* ``flip:<dir>``   -> ``(ori_of b flip(ori, dir))`` for a clear glyph block on the table
* ``rotate:<dir>`` -> ``(ori_of b yaw(ori, dir))``  for a clear glyph block on the table
* ``pnp``          -> ``(on b1 b2)`` for two clear blocks, or ``(on_table b t)`` for a
  clear stacked block

Shares are counted from the skills the executed plans actually contain (see
:func:`plan_skills`), so extra actions the planner inserts (a ``move_to_side`` to
make room, an unstack before a flip) are counted too and the next goals correct
for them.

Target mix: pick-and-place 1/3, flip 1/3 (1/12 per direction), rotate 1/3 (1/6
per direction). Classes whose action is not allowed drop out and the rest are
re-normalised.
"""
from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from blocksuite.playdata.goal_policy import Decision, PolicyDecision
from blocksuite.playdata.goal_proposer import ProposedGoal
from blocksuite.playdata.symbolic_state import SymbolicState
from blocksuite.tamp.blocks.glyph_blocks import glyph_usd_subpath
from blocksuite.task_gen.orientations import flip as flip_ori
from blocksuite.task_gen.orientations import yaw as yaw_ori

HoldsFn = Callable[[str, list[str]], bool]

FLIP_DIRECTIONS: tuple[str, ...] = ("forward", "backward", "left_flip", "right_flip")
ROTATE_DIRECTIONS: tuple[str, ...] = ("cw", "ccw")
PNP_ACTIONS: frozenset[str] = frozenset(
    {"pnp_table_table", "pnp_table_block", "pnp_block_table", "pnp_block_block",
     "move_to_side", "move_to_center"}
)
#: The planner's goal marker, appended to every plan (gripper open + lift).
DONE_ACTIONS: frozenset[str] = frozenset({"done", "reach-goal", "reach_goal"})
#: Actions that carry no skill (plan bookkeeping and the planner's goal markers).
_BOOKKEEPING_ACTIONS: frozenset[str] = frozenset({"init", "move_to"}) | DONE_ACTIONS

PNP_CLASS = "pnp"
FLIP_CLASSES: tuple[str, ...] = tuple(f"flip:{d}" for d in FLIP_DIRECTIONS)
ROTATE_CLASSES: tuple[str, ...] = tuple(f"rotate:{d}" for d in ROTATE_DIRECTIONS)

#: Family shares before per-direction splitting.
_FAMILY_SHARE: dict[str, float] = {"pnp": 1.0 / 3.0, "flip": 1.0 / 3.0, "rotate": 1.0 / 3.0}


def skill_class(action_name: str, args: list[str]) -> str | None:
    """Skill class of one high-level action, or ``None`` for bookkeeping."""
    name = action_name.lower()
    if name in _BOOKKEEPING_ACTIONS:
        return None
    if name == "reorient" and len(args) > 2:
        return f"flip:{args[2].lower()}"
    if name == "rotate" and len(args) > 2:
        return f"rotate:{args[2].lower()}"
    if name in PNP_ACTIONS:
        return PNP_CLASS
    return name


def plan_skills(policy: Iterable[dict[str, Any]]) -> list[str]:
    """Skill classes of the high-level actions in a flat (parallel-format) policy.

    Consecutive low-level actions of the same high-level action share its
    ``_hl_action_name`` / ``_hl_action_args``, so each high-level action counts once.
    """
    skills: list[str] = []
    previous: tuple[str, tuple[str, ...]] | None = None
    for action in policy:
        values = action.get("generated_values", {}) or {}
        name = str(values.get("_hl_action_name", "") or "")
        if not name:
            continue
        args = tuple(str(a) for a in (values.get("_hl_action_args") or []))
        if (name, args) == previous:
            continue
        previous = (name, args)
        cls = skill_class(name, list(args))
        if cls is not None:
            skills.append(cls)
    return skills


def drop_done_after_non_pnp(policy: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop the plan's trailing ``done`` step unless the last real action is a pick-and-place.

    The planner ends every plan with ``done`` (gripper open + lift). After a place it
    lifts the gripper off the placed block; after a flip or rotate it is a redundant
    extra lift, so play data leaves it out. A plan with no real action (its goal
    already held) keeps its ``done`` so it is not left empty.
    """
    end = len(policy)
    while end > 0 and _hl_name(policy[end - 1]) in DONE_ACTIONS:
        end -= 1
    if end == len(policy):
        return policy
    last_real = next(
        (name for name in (_hl_name(a) for a in reversed(policy[:end])) if name not in _BOOKKEEPING_ACTIONS),
        None,
    )
    if last_real is None or last_real in PNP_ACTIONS:
        return policy
    return policy[:end]


def _hl_name(action: dict[str, Any]) -> str:
    return str((action.get("generated_values", {}) or {}).get("_hl_action_name", "") or "").lower()


def target_shares(allowed_actions: Iterable[str] | None) -> dict[str, float]:
    """Target share per skill class, restricted to the classes the allowed actions enable."""
    allowed = None if allowed_actions is None else {a.lower() for a in allowed_actions}
    enabled_families: list[str] = []
    if allowed is None or allowed & {"pnp_table_block", "pnp_block_table", "pnp_block_block"}:
        enabled_families.append("pnp")
    if allowed is None or "reorient" in allowed:
        enabled_families.append("flip")
    if allowed is None or "rotate" in allowed:
        enabled_families.append("rotate")
    if not enabled_families:
        raise ValueError("no pick-and-place, reorient or rotate action is allowed")
    total = sum(_FAMILY_SHARE[f] for f in enabled_families)
    shares: dict[str, float] = {}
    for family in enabled_families:
        share = _FAMILY_SHARE[family] / total
        if family == "pnp":
            shares[PNP_CLASS] = share
        elif family == "flip":
            shares.update({c: share / len(FLIP_CLASSES) for c in FLIP_CLASSES})
        else:
            shares.update({c: share / len(ROTATE_CLASSES) for c in ROTATE_CLASSES})
    return shares


@dataclass
class BalancedSkillCfg:
    """Knobs for :class:`BalancedSkillGoalPolicy`."""

    allowed_actions: tuple[str, ...] | None = None
    """High-level actions the planner may use (``None``: every action)."""
    max_stack_height: int = 3
    """Tallest tower a stacking goal may build."""
    count_successes: bool = False
    """Count a plan's skills only once its episode is kept (goal reached, or abandoned
    and kept), instead of when the plan is installed. With missed goals dropped from
    the dataset, a skill that often fails (e.g. forward flips) is then requested until
    it reaches its share of the kept data rather than of the attempts."""
    flip_min_x: dict[str, float] = field(default_factory=dict)
    """Per flip direction (``forward``, ``backward``, ``left_flip``, ``right_flip``), the
    minimum block x (m, from the robot base) for a flip goal. Flips near the robot
    mostly fail; blocks closer than this are not offered for that direction."""


class BalancedSkillGoalPolicy:
    """Goal policy that keeps executed skills close to :func:`target_shares`.

    Counts are shared across all envs of the run, so the mix is balanced for the
    run as a whole, not per env.
    """

    def __init__(self, cfg: BalancedSkillCfg, rng: random.Random | None = None) -> None:
        self.cfg = cfg
        self.rng = rng or random.Random()
        self.shares: dict[str, float] = target_shares(cfg.allowed_actions)
        allowed = None if cfg.allowed_actions is None else {a.lower() for a in cfg.allowed_actions}
        self._can_stack = allowed is None or bool(allowed & {"pnp_table_block", "pnp_block_block"})
        self._can_unstack = allowed is None or "pnp_block_table" in allowed
        self.counts: Counter[str] = Counter()
        self._pending: dict[int, list[str]] = {}
        # Per env: the (class, block) of the last flip/rotate goal handed out, and the
        # pairs that missed their goal. A missed pair is not offered again until the env
        # resets or the block is the subject of a reached goal: the same scripted flip
        # on an unchanged block fails the same way, and success counting would
        # otherwise re-request it in a loop.
        self._issued: dict[int, tuple[str, str]] = {}
        self._failed: dict[int, set[tuple[str, str]]] = {}
        self._pose_of: Callable[[int, str], list[float] | None] | None = None

    def set_scene_accessors(self, pose_of: Callable[[int, str], list[float] | None], table: str) -> None:  # noqa: ARG002
        """Live block poses (env-local frame, robot base at the origin) for position-aware goals."""
        self._pose_of = pose_of

    # -- GoalPolicy protocol ----------------------------------------------------

    def reset(self, env_idx: int) -> None:
        # Counts are run-wide on purpose; only the per-env missed goals are cleared.
        self._issued.pop(env_idx, None)
        self._failed.pop(env_idx, None)

    def on_episode_result(self, env_idx: int, reached: bool) -> None:
        skills = self._pending.pop(env_idx, [])
        if reached:
            self.counts.update(skills)
        issued = self._issued.pop(env_idx, None)
        if issued is None:
            return
        failed = self._failed.setdefault(env_idx, set())
        if reached:
            block = issued[1]
            failed.difference_update({pair for pair in failed if pair[1] == block})
        else:
            failed.add(issued)

    def on_plan_skills(self, env_idx: int, skills: list[str]) -> None:
        """Feedback from the orchestrator: the skills an installed plan contains."""
        if self.cfg.count_successes:
            self._pending[env_idx] = list(skills)
        else:
            self.counts.update(skills)

    def next_goal(self, env_idx: int, state: SymbolicState, holds: HoldsFn) -> PolicyDecision:  # noqa: ARG002
        self._issued.pop(env_idx, None)
        for cls in self._classes_by_deficit():
            goal = self._goal_for(cls, state, env_idx)
            if goal is not None:
                return PolicyDecision(Decision.GOAL, goal=goal, reason=f"balance:{cls}")
        return PolicyDecision(Decision.RETRY, reason="no one-step goal for any skill class")

    # -- internals ----------------------------------------------------------------

    def _classes_by_deficit(self) -> list[str]:
        """Enabled classes, furthest behind their target share first (ties shuffled)."""
        total = sum(self.counts[c] for c in self.shares)
        classes = list(self.shares)
        self.rng.shuffle(classes)
        return sorted(classes, key=lambda c: self.counts[c] - self.shares[c] * (total + 1))

    def _flippable(self, state: SymbolicState) -> list[str]:
        """Clear glyph blocks on the table with a known, clean orientation.

        Only glyph (letter / digit) blocks get flip and rotate goals: a plain colored
        cube looks the same in every orientation, so turning it records no visible
        change. Colored blocks still take part in pick-and-place goals.
        """
        clear = set(state.clear_blocks())
        skip = set(state.tilted) | set(state.unsupported)
        return [
            b for b in state.blocks
            if b in clear and state.supporter(b) is None and b not in skip and b in state.ori
            and glyph_usd_subpath(b) is not None
        ]

    def _height(self, state: SymbolicState, block: str) -> int:
        """Number of blocks in the tower from the table up to and including ``block``."""
        height = 1
        seen = {block}
        below = state.supporter(block)
        while below is not None and below not in seen:
            seen.add(below)
            height += 1
            below = state.supporter(below)
        return height

    def _far_enough(self, env_idx: int, block: str, direction: str) -> bool:
        """Whether ``block`` is at least ``flip_min_x`` from the robot for this flip direction."""
        min_x = self.cfg.flip_min_x.get(direction)
        if min_x is None or self._pose_of is None:
            return True
        pose = self._pose_of(env_idx, block)
        return pose is None or float(pose[0]) >= min_x

    def _goal_for(self, cls: str, state: SymbolicState, env_idx: int = 0) -> ProposedGoal | None:
        if cls == PNP_CLASS:
            return self._pnp_goal(state)
        family, direction = cls.split(":", 1)
        failed = self._failed.get(env_idx, set())
        candidates = [b for b in self._flippable(state) if (cls, b) not in failed]
        if family == "flip":
            candidates = [b for b in candidates if self._far_enough(env_idx, b, direction)]
        self.rng.shuffle(candidates)
        for block in candidates:
            current = state.ori[block]
            try:
                target = flip_ori(current, direction) if family == "flip" else yaw_ori(current, direction)
            except (KeyError, ValueError):
                continue
            if target != current:
                self._issued[env_idx] = (cls, block)
                return ProposedGoal(predicates=[f"(ori_of {block} {target})"], families=[cls])
        return None

    def _pnp_goal(self, state: SymbolicState) -> ProposedGoal | None:
        clear = [b for b in state.clear_blocks() if b not in set(state.unsupported)]
        options: list[tuple[str, list[str]]] = []
        if self._can_unstack:
            options += [("on_table", [b, state.table]) for b in clear if state.supporter(b) is not None]
        if self._can_stack:
            for b1 in clear:
                for b2 in clear:
                    if b1 == b2 or state.supporter(b1) == b2:
                        continue
                    if self._height(state, b2) + 1 > self.cfg.max_stack_height:
                        continue
                    options.append(("on", [b1, b2]))
        if not options:
            return None
        # Pick stack vs unstack evenly when both exist, so towers do not only grow.
        kinds = sorted({pred for pred, _ in options})
        kind = self.rng.choice(kinds)
        pred, args = self.rng.choice([o for o in options if o[0] == kind])
        return ProposedGoal(predicates=[f"({pred} {' '.join(args)})"], families=[PNP_CLASS])
