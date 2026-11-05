"""Derive a PDDL initial state from the live simulator state (batched predicates).

Used to re-plan an env from where it actually is after a skill kept failing: the facts
are read off the sim (``on``, ``on_table``, ``clear``, ``ori_of`` for every block, plus the
declared spatial relations below) and handed to FF with the problem's original goal.
"""

from __future__ import annotations

import torch
from torch import Tensor

from blocksuite.vecgen.symbolic.pddl import Fact, Problem
from blocksuite.vecgen.symbolic.predicates import (
    ORI_NAMES,
    classify_orientation,
    goal_atoms,
    on,
    pair_predicate,
)

#: Two-block relations the task generators *declare* (in ``:init``, e.g. stack_three's
#: ``next_to`` / ``pick_obstructs`` obstructor, or the goal, e.g. spelling's ``on_right`` chain)
#: rather than assert for every pair that happens to satisfy them; ``next_to`` is also
#: directional in the domain (the pushed block cannot be picked, its neighbour can) and
#: ``pick_obstructs`` (10 cm proximity) would hold for most neighbours in a cluttered scene.
#: A re-plan keeps such a fact when it is declared in the problem and still holds in the sim.
DECLARED_RELATIONS: frozenset[str] = frozenset(
    {"next_to", "on_right", "on_left", "on_front", "on_back", "pick_obstructs"}
)


def declared_relations(problem: Problem) -> list[Fact]:
    """:data:`DECLARED_RELATIONS` facts named in the problem's ``:init`` or goal (deduplicated)."""
    atoms = list(problem.init) + [a for g in problem.goal for a in goal_atoms(g)]
    return list(dict.fromkeys(f for f in atoms if f.pred in DECLARED_RELATIONS))


def derive_init_facts(
    problem: Problem,
    block_index: dict[str, int],
    poses: Tensor,
    sizes: Tensor,
    table_pose: Tensor,
    table_name: str = "workspace_table",
    robot_name: str = "robot",
) -> list[Fact]:
    """Facts for one env; ``poses`` ``[K, 7]``, ``sizes`` ``[K, 3]``, ``table_pose`` ``[7]``.

    A block that is ``on`` no other block counts as ``on_table`` (blocks never leave the
    table in these tasks); ``clear`` holds when nothing is on the block. In a tower the
    ``on`` window (dz <= 9 cm) also spans the block two levels down, so the supporter is
    the highest block ``a`` is on. :func:`declared_relations` facts are kept when they hold.
    """
    blocks = [b for b in problem.blocks if b in block_index]
    k = len(blocks)
    idx = torch.tensor([block_index[b] for b in blocks], device=poses.device)
    p = poses[idx]  # [k, 7]
    facts: list[Fact] = []
    on_mat = on(p[:, None, :].expand(-1, k, -1), p[None, :, :].expand(k, -1, -1))  # [a, b]
    on_mat[torch.arange(k), torch.arange(k)] = False
    cls = classify_orientation(p[:, 3:7])
    for i, a in enumerate(blocks):
        below = on_mat[i].nonzero().flatten().tolist()
        if below:
            top = max(below, key=lambda j: float(p[j, 2]))
            facts.append(Fact("on", (a, blocks[top])))
        else:
            facts.append(Fact("on_table", (a, table_name)))
        if not bool(on_mat[:, i].any()):
            facts.append(Fact("clear", (a,)))
        facts.append(Fact("ori_of", (a, ORI_NAMES[int(cls[i])])))
    for f in declared_relations(problem):
        if not all(arg in block_index for arg in f.args):
            continue
        j, m = block_index[f.args[0]], block_index[f.args[1]]
        if bool(pair_predicate(f.pred, poses[j], poses[m], sizes[j], sizes[m])):
            facts.append(f)
    facts.append(Fact("handempty", (robot_name,)))
    facts.append(Fact("gripper_at_init", ()))
    return facts
