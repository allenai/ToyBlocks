"""Materialized init state passed to init-conditioned goal builders."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from toyblocks.task_gen.materialize import iter_predicates
from toyblocks.task_gen.predicates import Expr, Pred


@dataclass
class ResolvedInit:
    """Symbolic init facts for one sampled problem instance."""

    predicates: List[Pred] = field(default_factory=list)

    def absorb(self, expr: Expr) -> None:
        for pred in iter_predicates(expr):
            self.predicates.append(pred)

    def record(self, pred: Pred) -> None:
        self.predicates.append(pred)

    def matching(self, name: str, *prefix: str) -> List[tuple]:
        return [
            pred.args
            for pred in self.predicates
            if pred.name == name and pred.args[: len(prefix)] == prefix
        ]

    def arg(self, name: str, *prefix: str) -> str:
        matches = self.matching(name, *prefix)
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one ({name} {' '.join(prefix)} ...) in init, "
                f"found {len(matches)}"
            )
        rest = matches[0][len(prefix) :]
        if len(rest) != 1:
            raise ValueError(
                f"expected one trailing argument for ({name} {' '.join(prefix)} ...), "
                f"got {rest!r}"
            )
        return rest[0]

    def try_arg(self, name: str, *prefix: str) -> Optional[str]:
        matches = self.matching(name, *prefix)
        if len(matches) != 1:
            return None
        rest = matches[0][len(prefix) :]
        return rest[0] if len(rest) == 1 else None

    def has(self, name: str, *args: str) -> bool:
        return any(p.name == name and p.args == args for p in self.predicates)
