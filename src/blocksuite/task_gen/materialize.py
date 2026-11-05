"""Resolve stochastic init specs into concrete symbolic expressions."""

from __future__ import annotations

import random
from typing import List

from blocksuite.task_gen.predicates import And, Expr, Or, Pred


def materialize(expr: Expr) -> Expr:
    """Expand ``Or(..., mode='random_choice')`` and flatten ``And`` trees."""
    if isinstance(expr, Or) and expr.mode == "random_choice":
        return materialize(random.choice(expr.children))
    if isinstance(expr, And):
        flat: List[Expr] = []
        for child in expr.children:
            resolved = materialize(child)
            if isinstance(resolved, And):
                flat.extend(resolved.children)
            else:
                flat.append(resolved)
        if not flat:
            raise ValueError("empty And after materialize")
        if len(flat) == 1:
            return flat[0]
        return And(*flat)
    return expr


def iter_predicates(expr: Expr):
    """Yield every ``Pred`` leaf in an expression tree."""
    if isinstance(expr, Pred):
        yield expr
    elif isinstance(expr, And):
        for child in expr.children:
            yield from iter_predicates(child)
    elif isinstance(expr, Or):
        for child in expr.children:
            yield from iter_predicates(child)


def collect_block_refs(expr: Expr) -> set[str]:
    """Return block object names referenced by an expression tree."""
    return expr.referenced_blocks()
