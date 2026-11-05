"""Symbolic PDDL problem representation; text is emitted only at render time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from blocksuite.task_gen.predicates import And, Expr


@dataclass
class SymbolicProblem:
    """Fully instantiated problem — init and goal are symbolic expression trees."""

    name: str
    domain: str
    objects_by_type: Dict[str, List[str]] = field(default_factory=dict)
    init: Expr = field(default_factory=lambda: And())
    goal: Expr = field(default_factory=lambda: And())


def render_objects(objects_by_type: Dict[str, List[str]], indent: int = 4) -> str:
    pad = " " * indent
    lines: List[str] = []
    for btype, names in objects_by_type.items():
        lines.append(f"{pad}{' '.join(names)} - {btype}")
    lines.append(f"{pad}workspace_table - table")
    lines.append(f"{pad}robot - gripper")
    return "\n".join(lines)


def render_problem(problem: SymbolicProblem) -> str:
    """Render a symbolic problem to PDDL text (single emission point)."""
    objects = render_objects(problem.objects_by_type)
    init_body = problem.init.render(indent=8)
    goal_body = problem.goal.render(indent=8)
    return (
        f"(define (problem {problem.name})\n"
        f"    (:domain {problem.domain})\n"
        f"    (:objects\n"
        f"{objects}    )\n"
        f"    (:init\n"
        f"{init_body}\n"
        f"    )\n"
        f"    (:goal (and\n"
        f"{goal_body}\n"
        f"    ))\n"
        f")"
    )
