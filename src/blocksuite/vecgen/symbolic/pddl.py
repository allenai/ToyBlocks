"""Minimal PDDL problem handling for the vecgen engine.

Parses the problem files task generation writes (objects by type, ``:init`` facts, a
``:goal`` over ``and`` / ``or`` / ``not`` / ``exists`` / ``forall``, trailing
``;key = value`` metadata) and writes variants with a replaced ``:init`` so the
symbolic planner can be run per env.

The goal is kept as a list of top-level conjuncts, each a :data:`GoalExpr` tree of
:class:`Fact`, :class:`And`, :class:`Or` and :class:`Not`. Quantifiers are grounded
against the problem's objects at parse time (``exists`` -> :class:`Or`, ``forall`` ->
:class:`And`), and ``(= a b)`` folds to a constant, so every consumer (FF rendering,
:func:`~blocksuite.vecgen.symbolic.predicates.goal_facts_satisfied`) only sees those
four node types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Fact:
    pred: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"({self.pred} {' '.join(self.args)})" if self.args else f"({self.pred})"


@dataclass(frozen=True)
class And:
    """Conjunction of goal expressions."""

    children: tuple[GoalExpr, ...]

    def __str__(self) -> str:
        return f"(and {' '.join(str(c) for c in self.children)})"


@dataclass(frozen=True)
class Or:
    """Disjunction of goal expressions."""

    children: tuple[GoalExpr, ...]

    def __str__(self) -> str:
        return f"(or {' '.join(str(c) for c in self.children)})"


@dataclass(frozen=True)
class Not:
    """Negated goal expression."""

    child: GoalExpr

    def __str__(self) -> str:
        return f"(not {self.child})"


GoalExpr = Fact | And | Or | Not


@dataclass
class Problem:
    name: str
    domain: str
    objects: dict[str, str]  # object name -> type
    init: list[Fact]
    goal: list[GoalExpr]  # top-level conjuncts
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def blocks(self) -> list[str]:
        return [o for o, t in self.objects.items() if t.startswith("block")]

    def objects_of_type(self, prefix: str) -> list[str]:
        return [o for o, t in self.objects.items() if t.startswith(prefix)]

    def init_has(self, pred: str, *args: str) -> bool:
        return Fact(pred, tuple(args)) in self.init

    def init_matching(self, pred: str) -> list[Fact]:
        return [f for f in self.init if f.pred == pred]


_FACT_RE = re.compile(r"\(\s*([^\s()]+)((?:\s+[^\s()]+)*)\s*\)")
_TOKEN_RE = re.compile(r"\(|\)|[^\s()]+")


def _strip_comments(text: str) -> str:
    return "\n".join(line.split(";", 1)[0] for line in text.splitlines())


def _section(text: str, name: str) -> str:
    """Return the balanced s-expression body following ``(:name``."""
    start = text.find(f"(:{name}")
    if start < 0:
        raise ValueError(f"no (:{name} section")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return text[start + len(name) + 2 : i]
    raise ValueError(f"unbalanced (:{name} section")


def parse_facts(body: str) -> list[Fact]:
    """Flat facts of an ``:init`` body (or a conjunctive goal rendered by the play goal policy)."""
    facts: list[Fact] = []
    for m in _FACT_RE.finditer(body):
        pred = m.group(1).lower()
        if pred in ("and", "not"):
            continue
        args = tuple(a.lower() for a in m.group(2).split())
        facts.append(Fact(pred, args))
    return facts


# --- goal parsing ---------------------------------------------------------------------

SExpr = str | list


def _parse_sexprs(text: str) -> list[SExpr]:
    """Nested lists / lowercased symbols of every s-expression in ``text``."""
    stack: list[list[SExpr]] = [[]]
    for tok in _TOKEN_RE.findall(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced ')' in goal")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok.lower())
    if len(stack) != 1:
        raise ValueError("unbalanced '(' in goal")
    return stack[0]


def _typed_vars(decl: SExpr) -> list[tuple[str, str]]:
    """``[?a ?b - block_red ?c - block]`` -> ``[(?a, block_red), (?b, block_red), (?c, block)]``."""
    if not isinstance(decl, list):
        raise ValueError(f"quantifier variables must be a list, got {decl!r}")
    out: list[tuple[str, str]] = []
    pending: list[str] = []
    i = 0
    while i < len(decl):
        tok = decl[i]
        if not isinstance(tok, str):
            raise ValueError(f"malformed quantifier variables {decl!r}")
        if tok == "-":
            typ = decl[i + 1]
            if not isinstance(typ, str):
                raise ValueError(f"malformed quantifier variables {decl!r}")
            out += [(v, typ) for v in pending]
            pending = []
            i += 2
            continue
        pending.append(tok)
        i += 1
    out += [(v, "object") for v in pending]
    return out


def _objects_of(objects: dict[str, str], typ: str) -> list[str]:
    """Objects a quantifier variable of PDDL type ``typ`` ranges over.

    The domain's type tree is not parsed here: a variable matches objects declared with
    exactly its type, ``block`` additionally matches every ``block*`` subtype and
    ``object`` matches everything.
    """
    if typ == "object":
        return list(objects)
    return [o for o, t in objects.items() if t == typ or (typ == "block" and t.startswith("block"))]


def _and(children: list[GoalExpr | bool]) -> GoalExpr | bool:
    kept: list[GoalExpr] = []
    for c in children:
        if c is False:
            return False
        if c is not True:
            kept.append(c)
    if not kept:
        return True
    return kept[0] if len(kept) == 1 else And(tuple(kept))


def _or(children: list[GoalExpr | bool]) -> GoalExpr | bool:
    kept: list[GoalExpr] = []
    for c in children:
        if c is True:
            return True
        if c is not False:
            kept.append(c)
    if not kept:
        return False
    return kept[0] if len(kept) == 1 else Or(tuple(kept))


def _goal_expr(sexpr: SExpr, objects: dict[str, str], subst: dict[str, str]) -> GoalExpr | bool:
    """Ground one goal s-expression; ``bool`` results come from folded ``=`` tests."""
    if isinstance(sexpr, str):
        return Fact(sexpr, ())
    if not sexpr:
        raise ValueError("empty s-expression in goal")
    head = sexpr[0]
    if not isinstance(head, str):
        raise ValueError(f"goal sub-expression must start with a symbol: {sexpr!r}")
    if head == "and":
        return _and([_goal_expr(c, objects, subst) for c in sexpr[1:]])
    if head == "or":
        return _or([_goal_expr(c, objects, subst) for c in sexpr[1:]])
    if head == "not":
        if len(sexpr) != 2:
            raise ValueError(f"(not ...) takes exactly one argument: {sexpr!r}")
        child = _goal_expr(sexpr[1], objects, subst)
        return (not child) if isinstance(child, bool) else Not(child)
    if head == "=":
        if len(sexpr) != 3 or not all(isinstance(a, str) for a in sexpr[1:]):
            raise ValueError(f"(= a b) malformed: {sexpr!r}")
        return subst.get(sexpr[1], sexpr[1]) == subst.get(sexpr[2], sexpr[2])
    if head in ("exists", "forall"):
        if len(sexpr) != 3:
            raise ValueError(f"({head} (vars) body) malformed: {sexpr!r}")
        typed = _typed_vars(sexpr[1])
        domains = [_objects_of(objects, typ) for _, typ in typed]
        for (var, typ), objs in zip(typed, domains, strict=True):
            if not objs:
                raise ValueError(f"quantifier variable {var} of type {typ!r} matches no object")
        grounded: list[GoalExpr | bool] = []

        def expand(i: int, sub: dict[str, str]) -> None:
            if i == len(typed):
                grounded.append(_goal_expr(sexpr[2], objects, sub))
                return
            for obj in domains[i]:
                expand(i + 1, {**sub, typed[i][0]: obj})

        expand(0, subst)
        return _or(grounded) if head == "exists" else _and(grounded)
    args: list[str] = []
    for a in sexpr[1:]:
        if not isinstance(a, str):
            raise ValueError(f"predicate {head!r} has a nested argument {a!r}")
        args.append(subst.get(a, a))
    return Fact(head, tuple(args))


def parse_goal(body: str, objects: dict[str, str]) -> list[GoalExpr]:
    """Top-level conjuncts of a ``:goal`` body, quantifiers grounded over ``objects``."""
    forms = _parse_sexprs(body)
    conjuncts: list[GoalExpr] = []
    for form in forms:
        expr = _goal_expr(form, objects, {})
        if expr is True:
            continue
        if expr is False:
            raise ValueError(f"goal conjunct {form!r} grounds to false")
        if isinstance(expr, And):
            conjuncts.extend(expr.children)
        else:
            conjuncts.append(expr)
    return conjuncts


def parse_problem(path: str | Path) -> Problem:
    raw = Path(path).read_text()
    text = _strip_comments(raw)
    name_m = re.search(r"\(define\s*\(problem\s+([^\s()]+)\s*\)", text)
    domain_m = re.search(r"\(:domain\s+([^\s()]+)\s*\)", text)
    objects: dict[str, str] = {}
    names: list[str] = []
    tokens = _section(text, "objects").replace("\n", " ").split()
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok == "-":
            typ = tokens[i + 1].lower()
            for n in names:
                objects[n] = typ
            names = []
            i += 2
            continue
        names.append(tok.lower())
        i += 1
    for n in names:
        objects[n] = "object"
    init = parse_facts(_section(text, "init"))
    goal = parse_goal(_section(text, "goal"), objects)
    metadata: dict[str, object] = {}
    for line in raw.splitlines():
        m = re.match(r"\s*;\s*([a-z_]+)\s*=\s*(.+)$", line)
        if m:
            try:
                metadata[m.group(1)] = eval(m.group(2), {"__builtins__": {}})  # noqa: S307 - trusted task-gen output
            except Exception:
                metadata[m.group(1)] = m.group(2).strip()
    return Problem(
        name=name_m.group(1) if name_m else Path(path).stem,
        domain=domain_m.group(1) if domain_m else "blocks-tamp",
        objects=objects,
        init=init,
        goal=goal,
        metadata=metadata,
    )


def render_problem(problem: Problem, init: list[Fact] | None = None) -> str:
    """PDDL text for ``problem`` with ``init`` (default: its own init)."""
    by_type: dict[str, list[str]] = {}
    for obj, typ in problem.objects.items():
        by_type.setdefault(typ, []).append(obj)
    obj_lines = "\n".join(f"    {' '.join(objs)} - {typ}" for typ, objs in by_type.items())
    init_lines = "\n".join(f"        {f}" for f in (problem.init if init is None else init))
    goal_lines = "\n".join(f"        {f}" for f in problem.goal)
    return (
        f"(define (problem {problem.name})\n"
        f"    (:domain {problem.domain})\n"
        f"    (:objects\n{obj_lines}\n    )\n"
        f"    (:init\n{init_lines}\n    )\n"
        f"    (:goal (and\n{goal_lines}\n    ))\n"
        ")\n"
    )
