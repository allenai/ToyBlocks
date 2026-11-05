"""Import-light registry of PDDL goal predicates with a vectorized tensor check.

This is the single source of truth for *which* goal predicates can be compiled
into batched world-state checks. It is deliberately free of any heavy import
(``torch`` / ``isaaclab``) so that the codegen goal compiler
(:mod:`toyblocks.tamp.blocks.template.codegen`) and its unit tests can gate on
the supported set without pulling the simulator.

The actual tensor implementations live in
:mod:`toyblocks.tamp.blocks.pddl_goal_tensor_conversions`; that module asserts
its ``TENSOR_GOAL_DISPATCH`` keys equal :data:`SUPPORTED_GOAL_PREDICATES` at
import time, so the two cannot drift.
"""

from __future__ import annotations

# Predicates whose relational geometry is expressible as a pure function of
# world state (poses). Predicates that consult IK / motion planning / contact
# (IsValid*, at-with-pose, Grasped) are intentionally excluded.
SUPPORTED_GOAL_PREDICATES: frozenset[str] = frozenset(
    {
        "on",
        "on_table",
        "on_left",
        "on_right",
        "on_front",
        "on_back",
        "next_to",
        "ori_of",
        "clear",
        "pick_obstructs",
    }
)
