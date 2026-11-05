"""The live symbolic state the oracle checks its plan against, and the replan problem.

The oracle plans once with FF and then only checks each step's *new* facts. When the
world drifts from the plan (a missed or out-of-order skill, a dropped block), the
remaining steps can name a block that is stacked or covered. These helpers read the
facts that matter for that -- ``on`` / ``on_table`` / ``clear`` / ``ori_of`` -- off the
live sim with the success gate's own checks (:mod:`pddl_goal_tensor_conversions`), test
a plan step's preconditions against them, and build a corrected ``:init`` to re-plan
from. Everything but :func:`read_live_state` is pure.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

# Facts the sim can tell us, replaced wholesale on a re-plan. Everything else in the
# plan's expected state (static tables, ``initialized``, obstruction / side facts) is
# kept: those cannot be read reliably off the poses.
OBSERVED_PREDICATES = frozenset({"on", "on_table", "clear", "ori_of"})
# Gripper-side facts reset on a re-plan: it happens at a skill boundary, hand empty.
GRIPPER_PREDICATES = frozenset({"handempty", "holding", "at"})
PNP_ACTIONS = ("move_to_side", "move_to_center")
_COMPLEX_GOAL = re.compile(r"\(\s*(exists|forall|or|imply)\b", re.IGNORECASE)


def norm_fact(fact: str) -> str:
    """``"(on_table  a t )"`` -> ``"(on_table a t)"``; the oracle's FF-state parser leaves a
    space in zero-arity facts (``"(initialized )"``), which PDDL would not round-trip."""
    return "(" + " ".join(fact.strip().strip("()").split()).lower() + ")"


def fact(pred: str, args: Sequence[str]) -> str:
    return norm_fact(f"({pred} {' '.join(args)})")


def parse_fact(text: str) -> tuple[str, tuple[str, ...]]:
    parts = norm_fact(text)[1:-1].split()
    return parts[0], tuple(parts[1:])


def flat_goal_literals(problem_text: str) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """The goal's grounded positive literals, when the goal is a plain conjunction.

    Empty for ``exists`` / ``forall`` / ``or`` goals: their literals are not each
    required, and FF compiles such goals into a final ``REACH-GOAL`` action anyway, so
    the plan's last real action has a printed after-state.
    """
    from toyblocks.tamp.blocks.template.codegen import (
        extract_goal_block,
        parse_pddl_goal_from_text,
    )

    if _COMPLEX_GOAL.search(extract_goal_block(problem_text)):
        return ()
    out = []
    for p in parse_pddl_goal_from_text(problem_text):
        args = tuple(a.lower() for a in p.get("args", []))
        if p.get("negated") or any(a.startswith("?") for a in args):
            continue
        out.append((str(p["name"]).lower(), args))
    return tuple(out)


def replace_init(problem_text: str, facts: Iterable[str]) -> str:
    """``problem_text`` with its ``(:init ...)`` body replaced by ``facts``."""
    m = re.search(r"\(:init\b", problem_text, re.IGNORECASE)
    if not m:
        raise ValueError("problem has no (:init ...) section")
    depth, i = 0, m.start()
    while i < len(problem_text):
        if problem_text[i] == "(":
            depth += 1
        elif problem_text[i] == ")":
            depth -= 1
            if depth == 0:
                break
        i += 1
    if depth:
        raise ValueError("unbalanced (:init ...) section")
    body = "\n".join(f"        {f}" for f in sorted(facts))
    return problem_text[: m.start()] + f"(:init\n{body}\n    )" + problem_text[i + 1 :]


@dataclass
class LiveState:
    """What the sim shows for the problem's blocks."""

    support: dict[str, str | None] = field(default_factory=dict)  # block -> block it is on
    on_table: set[str] = field(default_factory=set)
    ori: dict[str, str] = field(default_factory=dict)  # only blocks within an orientation's tolerance

    def above(self, block: str) -> list[str]:
        return sorted(b for b, s in self.support.items() if s == block)

    def clear(self, block: str) -> bool:
        return not self.above(block)


def read_live_state(base_env: Any, env_index: int, blocks: Sequence[str]) -> LiveState:
    """``on`` / ``on_table`` / ``ori_of`` for ``blocks`` in env ``env_index``, with the success
    gate's checks. A block is on the highest block it is ``on``, and on the table when it is
    on no block (blocks never leave the table in these tasks; vecgen's live re-plan does
    the same), so a tilted block still gets a support FF can pick it from."""
    from toyblocks.tamp.blocks.pddl_goal_tensor_conversions import (
        ORI_QUATERNIONS,
        batched_goal_conjunct,
        resolve_root_pose_w,
    )

    def holds(pred: str, *args: str) -> bool:
        return bool(batched_goal_conjunct(base_env, pred, args)[env_index])

    z = {b: float(resolve_root_pose_w(base_env, b)[env_index, 2]) for b in blocks}
    live = LiveState()
    for a in blocks:
        below = [b for b in blocks if b != a and holds("on", a, b)]
        live.support[a] = max(below, key=z.__getitem__) if below else None
        if live.support[a] is None:
            live.on_table.add(a)
        for label in ORI_QUATERNIONS:
            if holds("ori_of", a, label):
                live.ori[a] = label
                break
    return live


def step_violations(action: str, obj: str, target: str | None, live: LiveState) -> list[str]:
    """Preconditions of a plan step the live world breaks (the domain requires a clear pick
    block and a clear place target; rotate / reorient also need the block on the table)."""
    why = []
    if obj and obj in live.support:
        if not live.clear(obj):
            why.append(f"{obj} is not clear ({', '.join(live.above(obj))} on it)")
        if action in ("rotate", "reorient") and live.support[obj] is not None:
            why.append(f"{obj} is stacked on {live.support[obj]}")
    if target and target in live.support and not live.clear(target):
        why.append(f"place target {target} is not clear ({', '.join(live.above(target))} on it)")
    return why


def corrected_init(expected: Iterable[str], live: LiveState, table: str, gripper: str) -> list[str]:
    """The plan's expected state with the observed facts replaced by the live ones.

    Observed facts are dropped for every block the sim reports on; an orientation the
    sim cannot classify (block tilted past tolerance) keeps the expected one.
    """
    blocks = set(live.support)
    out = set()
    for f in map(norm_fact, expected):
        pred, args = parse_fact(f)
        if pred in GRIPPER_PREDICATES:
            continue
        if pred in OBSERVED_PREDICATES and args and args[0] in blocks:
            if pred != "ori_of" or args[0] in live.ori:
                continue
        out.add(f)
    for b in blocks:
        if live.support[b] is not None:
            out.add(fact("on", (b, live.support[b])))
        elif b in live.on_table:
            out.add(fact("on_table", (b, table)))
        if live.clear(b):
            out.add(fact("clear", (b,)))
        if b in live.ori:
            out.add(fact("ori_of", (b, live.ori[b])))
    out.add(fact("handempty", (gripper,)))
    return sorted(out)


def blocks_and_table(expected: Iterable[str]) -> tuple[list[str], str]:
    """Problem blocks (every block has an ``ori_of``) and the table name, from a state."""
    blocks, table = set(), "workspace_table"
    for f in expected:
        pred, args = parse_fact(f)
        if pred == "ori_of" and args:
            blocks.add(args[0])
        elif pred == "on_table" and len(args) == 2:
            table = args[1]
    return sorted(blocks), table
