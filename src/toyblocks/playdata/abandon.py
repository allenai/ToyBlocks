"""Abandon a pick-and-place mid-transport and follow it with a new pick.

With ``--abandon-plan-percent N`` the play-data orchestrator arms N% of the plans
that contain a pick-and-place. Partway through the armed plan's transport segment
(block grasped, carried toward its place pose) the plan is cut off: the episode so
far is saved as a success, the arm holds still with the block still in the
gripper, and the next goal picks up a *different* block. That plan's first step is
the pick's ``init_gripper_open_traj``, so the next episode begins by dropping the
held block under a "pick <new block>" label, then picks the new block.

Only the pure helpers live here (policy inspection, follow-up goal sampling), so
they stay importable and testable without Isaac Sim.
"""
from __future__ import annotations

import random
from collections.abc import Iterable
from typing import Any

from toyblocks.playdata.goal_proposer import ProposedGoal
from toyblocks.playdata.skill_balance import PNP_ACTIONS, PNP_CLASS
from toyblocks.playdata.symbolic_state import SymbolicState

#: Low-level segments of a pick-and-place that carry the grasped block to its place pose.
TRANSPORT_LL_ACTIONS: frozenset[str] = frozenset(
    {"place_traj", "move_to_side_traj", "move_to_center_traj"}
)

#: XY distance (m) under which a block counts as below the held block, where the drop may land.
DROP_CLEARANCE_XY: float = 0.08

#: Actions a follow-up ``(on b1 b2)`` / ``(on_table b t)`` goal can be planned with.
_STACK_ACTIONS: frozenset[str] = frozenset({"pnp_table_block", "pnp_block_block"})
_UNSTACK_ACTIONS: frozenset[str] = frozenset({"pnp_block_table"})


def _generated_values(action: dict[str, Any]) -> dict[str, Any]:
    return action.get("generated_values", {}) or {}


def is_transport_step(action: dict[str, Any]) -> bool:
    """True for the transport segment of a pick-and-place high-level action."""
    values = _generated_values(action)
    hl_name = str(values.get("_hl_action_name", "") or "").lower()
    ll_name = str(action.get("ll_action_name") or values.get("ll_action_name") or "")
    return hl_name in PNP_ACTIONS and ll_name in TRANSPORT_LL_ACTIONS


def has_transport_step(policy: Iterable[dict[str, Any]]) -> bool:
    """True when a flat (parallel-format) policy contains a pick-and-place transport."""
    return any(is_transport_step(action) for action in policy)


def transport_num_waypoints(value: Any) -> int | None:
    """Waypoints in a transport step's motion plan, or ``None`` before it is planned.

    ``value`` is the step's ``PendingMotionPlan``; the executor stores the fresh
    CuRobo result on it (``motion_plan``) when the segment starts, and yields one
    waypoint per row of its interpolated plan. A pre-computed ``MotionGenResult``
    value is used as is.
    """
    motion_plan = getattr(value, "motion_plan", None)
    if motion_plan is None and hasattr(value, "get_interpolated_plan"):
        motion_plan = value
    if motion_plan is None:
        return None
    try:
        position = motion_plan.get_interpolated_plan().position
    except Exception:
        return None
    if position is None or position.dim() < 2:
        return None
    return int(position.shape[-2])


def held_block(action: dict[str, Any]) -> str | None:
    """The block a pick-and-place step carries (its ``obj`` argument), if known."""
    args = _generated_values(action).get("_hl_action_args") or []
    return str(args[1]) if len(args) > 1 else None


def blocks_below(
    held_xy: tuple[float, float],
    block_xy: dict[str, tuple[float, float]],
    clearance: float = DROP_CLEARANCE_XY,
) -> list[str]:
    """Blocks whose XY position is within ``clearance`` of the held block's."""
    hx, hy = held_xy
    return [
        name for name, (x, y) in block_xy.items()
        if ((x - hx) ** 2 + (y - hy) ** 2) ** 0.5 < clearance
    ]


def followup_actions_enabled(allowed_actions: Iterable[str] | None) -> tuple[bool, bool]:
    """``(can_stack, can_unstack)`` for follow-up goals given the planner's allowed actions."""
    if allowed_actions is None:
        return True, True
    allowed = {a.lower() for a in allowed_actions}
    return bool(allowed & _STACK_ACTIONS), bool(allowed & _UNSTACK_ACTIONS)


def _tower_height(state: SymbolicState, block: str) -> int:
    """Number of blocks from the table up to and including ``block``."""
    height = 1
    seen = {block}
    below = state.supporter(block)
    while below is not None and below not in seen:
        seen.add(below)
        height += 1
        below = state.supporter(below)
    return height


def followup_pnp_goal(
    state: SymbolicState,
    held: str,
    rng: random.Random,
    *,
    can_stack: bool = True,
    can_unstack: bool = True,
    max_stack_height: int = 3,
    avoid: Iterable[str] = (),
) -> ProposedGoal | None:
    """A one-step pick-and-place goal that moves a block other than ``held``.

    ``held`` is still in the gripper when the state is read, so it is neither the
    subject nor the destination: where it lands after the drop is unknown. Blocks in
    ``avoid`` (those below the held block, see :func:`blocks_below`) are skipped too. Stack
    and unstack goals are picked evenly when both exist, as in the balanced policy.
    Returns ``None`` when no such goal exists (e.g. ``held`` is the only clear block).
    """
    skip = {held} | set(state.unsupported) | set(avoid)
    clear = [b for b in state.clear_blocks() if b not in skip]
    options: list[tuple[str, list[str]]] = []
    if can_unstack:
        options += [("on_table", [b, state.table]) for b in clear if state.supporter(b) is not None]
    if can_stack:
        for b1 in clear:
            for b2 in clear:
                if b1 == b2 or state.supporter(b1) == b2:
                    continue
                if _tower_height(state, b2) + 1 > max_stack_height:
                    continue
                options.append(("on", [b1, b2]))
    if not options:
        return None
    kind = rng.choice(sorted({pred for pred, _ in options}))
    pred, args = rng.choice([o for o in options if o[0] == kind])
    return ProposedGoal(predicates=[f"({pred} {' '.join(args)})"], families=[PNP_CLASS])
