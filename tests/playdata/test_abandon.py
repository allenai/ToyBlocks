"""blocksuite.playdata.abandon: transport detection and follow-up pick goals."""
import random
from typing import Any

import pytest

from blocksuite.playdata.abandon import (
    blocks_below,
    followup_actions_enabled,
    followup_pnp_goal,
    has_transport_step,
    held_block,
    is_transport_step,
    transport_num_waypoints,
)
from blocksuite.playdata.symbolic_state import SymbolicState

TABLE = "workspace_table"


def _step(hl: str, args: list[str], ll: str) -> dict[str, Any]:
    return {"type": "x", "ll_action_name": ll,
            "generated_values": {"_hl_action_name": hl, "_hl_action_args": args}}


def _state(on: dict[str, str] | None = None, blocks: tuple[str, ...] = ("a", "b", "c")) -> SymbolicState:
    on = on or {}
    return SymbolicState(blocks=list(blocks), table=TABLE, gripper="robot", on=dict(on),
                         on_table=[b for b in blocks if b not in on], ori={b: "o1" for b in blocks})


def test_transport_detection() -> None:
    pnp = [_step("pnp_table_block", ["robot", "a", "b", TABLE], ll)
           for ll in ("init_gripper_open_traj", "grasp_traj", "gripper_close_traj", "place_traj")]
    assert has_transport_step(pnp)
    assert is_transport_step(pnp[3]) and not is_transport_step(pnp[1])
    assert held_block(pnp[3]) == "a"
    assert is_transport_step(_step("move_to_side", ["robot", "a", TABLE], "move_to_side_traj"))
    rotate = [_step("rotate", ["robot", "a", "cw", "o1", "o2", TABLE], "rotate_traj")]
    assert not has_transport_step(rotate)


def test_transport_num_waypoints() -> None:
    torch = pytest.importorskip("torch")

    class Plan:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self.position = torch.zeros(shape)

    class Result:
        def __init__(self, shape: tuple[int, ...]) -> None:
            self._plan = Plan(shape)

        def get_interpolated_plan(self) -> Plan:
            return self._plan

    class Pending:
        def __init__(self, motion_plan: Result | None) -> None:
            self.motion_plan = motion_plan

    assert transport_num_waypoints(Pending(None)) is None
    assert transport_num_waypoints(Pending(Result((1, 40, 7)))) == 40
    assert transport_num_waypoints(Result((25, 7))) == 25


def test_followup_goal_never_uses_held_block() -> None:
    rng = random.Random(0)
    state = _state(on={"b": "c"}, blocks=("a", "b", "c", "d"))
    seen = set()
    for _ in range(200):
        goal = followup_pnp_goal(state, "a", rng)
        assert goal is not None
        assert "a" not in goal.text.strip("()").split()[1:]
        seen.add(goal.text)
    # c is covered by b, so neither the subject nor the target can be c
    assert seen == {f"(on_table b {TABLE})", "(on b d)", "(on d b)"}


def test_followup_goal_respects_allowed_actions_and_height() -> None:
    rng = random.Random(1)
    state = _state(on={"b": "c"}, blocks=("a", "b", "c", "d"))
    goal = followup_pnp_goal(state, "a", rng, can_stack=False)
    assert goal is not None and goal.text == f"(on_table b {TABLE})"
    # (on d b) would build a 3-tower, so with max height 2 only (on b d) is left
    for _ in range(20):
        goal = followup_pnp_goal(state, "a", rng, can_unstack=False, max_stack_height=2)
        assert goal is not None and goal.text == "(on b d)"
    # the held block is the only other clear one: no follow-up
    assert followup_pnp_goal(_state(blocks=("a", "b")), "b", rng, can_stack=False) is None
    assert followup_actions_enabled(None) == (True, True)
    assert followup_actions_enabled(["pnp_block_table", "rotate"]) == (False, True)


def test_followup_goal_avoids_blocks_below_held() -> None:
    rng = random.Random(2)
    state = _state(on={"b": "c"}, blocks=("a", "b", "c", "d"))
    below = blocks_below((0.0, 0.0), {"b": (0.2, 0.0), "d": (0.03, 0.02)})
    assert below == ["d"]
    for _ in range(20):
        goal = followup_pnp_goal(state, "a", rng, avoid=below)
        assert goal is not None and goal.text == f"(on_table b {TABLE})"
