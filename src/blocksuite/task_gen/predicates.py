"""Predicate DSL for declarative PDDL problem generation.

Supports composable And/Or logic and typed predicate constructors.
"""

# TODO: (wyc) integrate with tmp3.tamp.DataStructures.Predicate??

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from blocksuite.task_gen.orientations import ORIENTATIONS

_ORI_SET = frozenset(ORIENTATIONS)
_NON_BLOCK_ARGS = frozenset({"table", "workspace_table", "robot"})


class Expr:
    """Base class for all PDDL expressions."""

    def render(self, indent: int = 8) -> str:
        raise NotImplementedError

    def referenced_blocks(self) -> set:
        """Return block names referenced by this expression."""
        return set()


@dataclass
class Pred(Expr):
    """Generic predicate: Pred("on_table", "block_r", "table")."""
    name: str
    args: tuple

    def __init__(self, name: str, *args: str):
        self.name = name
        self.args = args

    def render(self, indent: int = 8) -> str:
        args_str = ' '.join(self.args)
        return f"{' ' * indent}({self.name} {args_str})"

    def referenced_blocks(self) -> set:
        if self.name == "ori_of":
            return {self.args[0]} if self.args else set()
        if self.name in ("handempty", "gripper_at_init"):
            return set()
        return {
            a
            for a in self.args
            if a not in _NON_BLOCK_ARGS and a not in _ORI_SET
        }

    def __repr__(self):
        return f"Pred({self.name!r}, {', '.join(repr(a) for a in self.args)})"


@dataclass
class And(Expr):
    """Conjunction: all children appear in the output."""
    children: List[Expr]

    def __init__(self, *children: Expr):
        self.children = list(children)

    def render(self, indent: int = 8) -> str:
        lines = []
        for child in self.children:
            lines.append(child.render(indent))
        return '\n'.join(lines)

    def referenced_blocks(self) -> set:
        result = set()
        for child in self.children:
            result |= child.referenced_blocks()
        return result

    def __repr__(self):
        return f"And({', '.join(repr(c) for c in self.children)})"


@dataclass
class Or(Expr):
    """Disjunction: renders as PDDL (or ...) for goals, or picks one randomly for init."""
    children: List[Expr]
    mode: str = "pddl_or"  # "pddl_or" or "random_choice"

    def __init__(self, *children: Expr, mode: str = "pddl_or"):
        self.children = list(children)
        self.mode = mode

    def render(self, indent: int = 8) -> str:
        if self.mode == "random_choice":
            raise RuntimeError(
                "Or(random_choice) must be materialized before rendering; "
                "use materialize() from blocksuite.task_gen.materialize"
            )
        # PDDL (or ...) syntax
        inner_parts = []
        for child in self.children:
            rendered = child.render(indent=0).strip()
            inner_parts.append(rendered)
        return f"{' ' * indent}(or {' '.join(inner_parts)})"

    def referenced_blocks(self) -> set:
        result = set()
        for child in self.children:
            result |= child.referenced_blocks()
        return result

    def __repr__(self):
        return f"Or({', '.join(repr(c) for c in self.children)})"


@dataclass
class Exists(Expr):
    """Existential quantification: ``(exists (?v1 - type1 ...) (and ...))``.

    *variables* is a list of ``(var_name, pddl_type)`` pairs, e.g.
    ``[("?b1", "block_red"), ("?b2", "block_red")]``.

    When *auto_inequality* is ``True`` (the default), pairwise
    ``(not (= ?vi ?vj))`` constraints are emitted for every pair of
    variables that share the same type.
    """
    variables: List[Tuple[str, str]]
    body: Expr
    auto_inequality: bool = True

    def __init__(
        self,
        variables: List[Tuple[str, str]],
        body: Expr,
        *,
        auto_inequality: bool = True,
    ):
        self.variables = list(variables)
        self.body = body
        self.auto_inequality = auto_inequality

    def render(self, indent: int = 8) -> str:
        pad = ' ' * indent
        var_decls = ' '.join(f"{name} - {vtype}" for name, vtype in self.variables)

        inner_parts: List[str] = []

        if self.auto_inequality:
            for i in range(len(self.variables)):
                for j in range(i + 1, len(self.variables)):
                    if self.variables[i][1] == self.variables[j][1]:
                        inner_parts.append(
                            f"{pad}        (not (= {self.variables[i][0]} {self.variables[j][0]}))"
                        )

        inner_parts.append(self.body.render(indent=indent + 8))

        inner_text = '\n'.join(inner_parts)
        return (
            f"{pad}(exists ({var_decls})\n"
            f"{pad}    (and\n"
            f"{inner_text}\n"
            f"{pad}    )\n"
            f"{pad})"
        )

    def referenced_blocks(self) -> set:
        return set()

    def __repr__(self):
        return f"Exists({self.variables!r}, {self.body!r})"


# ─── Typed predicate constructors ───────────────────────────────────────────


class OnTable(Pred):
    """(on_table ?block table)"""
    def __init__(self, block: str, table: str = "workspace_table"):
        super().__init__("on_table", block, table)


class On(Pred):
    """(on ?block1 ?block2)"""
    def __init__(self, top: str, bottom: str):
        super().__init__("on", top, bottom)


class OnRight(Pred):
    """(on_right ?block1 ?block2)"""
    def __init__(self, block1: str, block2: str):
        super().__init__("on_right", block1, block2)


class Clear(Pred):
    """(clear ?block)"""
    def __init__(self, block: str):
        super().__init__("clear", block)


class OriOf(Pred):
    """(ori_of ?block ?orientation)"""
    def __init__(self, block: str, ori: str):
        super().__init__("ori_of", block, ori)


class NextTo(Pred):
    """(next_to ?block1 ?block2) — block1 adjacent to block2 on the table."""
    def __init__(self, block1: str, block2: str):
        super().__init__("next_to", block1, block2)


class HandEmpty(Pred):
    """(handempty robot)"""
    def __init__(self, gripper: str = "robot"):
        super().__init__("handempty", gripper)

    def referenced_blocks(self) -> set:
        return set()


class GripperAtInit(Pred):
    """(gripper_at_init)"""
    def __init__(self):
        super().__init__("gripper_at_init")
        self.args = ()

    def render(self, indent: int = 8) -> str:
        return f"{' ' * indent}(gripper_at_init)"

    def referenced_blocks(self) -> set:
        return set()


class Raw(Expr):
    """Raw PDDL text, passed through verbatim."""
    def __init__(self, text: str):
        self.text = text.strip()

    def render(self, indent: int = 8) -> str:
        return f"{' ' * indent}{self.text}"

    def referenced_blocks(self) -> set:
        return set()

    def __repr__(self):
        return f"Raw({self.text!r})"
