"""``ProblemSpec`` with the random init stacking capped at a tower height (stack task generators)."""

from __future__ import annotations

import random

from toyblocks.task_gen.predicates import Clear, On, OnTable, Pred
from toyblocks.task_gen.problem_spec import ProblemSpec


class HeightCappedSpec(ProblemSpec):
    """ProblemSpec whose random init stacking is limited to towers of n blocks.

    ``ProblemSpec._random_stacking`` lets a column grow without bound, so a
    scene of 14 distractors can contain a 5-block tower. This override keeps the
    same placement rule (0.4 chance of going on top of an eligible block) and
    only adds a depth check, so an uncapped spec samples exactly as before.

    Only blocks left unplaced by the explicit init are affected. Without --bury
    the goal cubes are pinned on-table and clear, so the cap applies to the
    distractors alone; with --bury the goal cubes join the same pass and are
    capped too.
    """

    def __init__(self, domain: str = "blocks-tamp", max_stack_height: int | None = None):
        super().__init__(domain=domain)
        if max_stack_height is not None and max_stack_height < 1:
            raise ValueError(f"max_stack_height must be >= 1, got {max_stack_height}")
        self.max_stack_height = max_stack_height

    def _random_stacking(self, blocks: list[str], explicit_clear: set[str]) -> list[Pred]:
        if self.max_stack_height is None:
            return super()._random_stacking(blocks, explicit_clear)

        table = "workspace_table"
        on_rel: dict[str, str] = {}
        above: dict[str, list[str]] = {b: [] for b in blocks}
        # Blocks in the column up to and including this block; a block on the
        # table is depth 1, so `depth[below] + 1` is the height `b` would create.
        depth: dict[str, int] = {}
        preds: list[Pred] = []

        shuffled = blocks.copy()
        random.shuffle(shuffled)

        for b in shuffled:
            if random.random() < 0.4:
                possible_below = [
                    x
                    for x in blocks
                    if x != b
                    and x in on_rel
                    and len(above[x]) == 0
                    and x not in explicit_clear
                    and depth[x] + 1 <= self.max_stack_height
                ]
                if possible_below:
                    below = random.choice(possible_below)
                    on_rel[b] = below
                    above[below].append(b)
                    depth[b] = depth[below] + 1
                    continue
            on_rel[b] = table
            depth[b] = 1

        for b in blocks:
            if on_rel.get(b) == table:
                preds.append(OnTable(b, table))
            elif b in on_rel:
                preds.append(On(b, on_rel[b]))

        for b in blocks:
            if not above[b] and b not in explicit_clear:
                preds.append(Clear(b))

        return preds
