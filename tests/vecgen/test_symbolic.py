from pathlib import Path

import pytest

from toyblocks.vecgen.symbolic.ff import (
    default_ff_binary,
    parse_ff_output,
    parse_ff_states,
    run_ff,
)
from toyblocks.vecgen.symbolic.pddl import Fact, parse_problem, render_problem
from toyblocks.vecgen.symbolic.programs import program_from_plan

ROOT = Path(__file__).resolve().parents[2]
DOMAIN = ROOT / "src/toyblocks/tamp/blocks/domain.pddl"
RED_TOWER = ROOT / "src/toyblocks/tamp/blocks/problems/red_tower_3.pddl"


def test_parse_red_tower():
    p = parse_problem(RED_TOWER)
    assert p.name == "red_tower_3" and p.domain == "blocks-tamp"
    assert p.blocks == ["block_red1", "block_red2", "block_red3"]
    assert p.objects["workspace_table"] == "table" and p.objects["robot"] == "gripper"
    assert p.init_has("ori_of", "block_red1", "o8") and p.init_has("gripper_at_init")
    assert Fact("on", ("block_red2", "block_red1")) in p.goal
    text = render_problem(p)
    assert "(:goal (and" in text and "(on_table block_red1 workspace_table)" in text


def test_parse_ff_output_sample():
    sample = """ff: found legal plan as follows

step    0: INIT
        1: MOVE_TO ROBOT BLOCK_RED1 BLOCK_RED2
        2: PNP_TABLE_BLOCK ROBOT BLOCK_RED2 BLOCK_RED1 WORKSPACE_TABLE O8 O8


time spent:    0.00 seconds instantiating
"""
    acts = parse_ff_output(sample)
    assert [a.pred for a in acts] == ["init", "move_to", "pnp_table_block"]
    assert acts[2].args == ("robot", "block_red2", "block_red1", "workspace_table", "o8", "o8")


def test_parse_ff_states_sample():
    sample = """*** Plan 1 States ***
State

NOT-INITIALIZED()
GRIPPER_AT_INIT()
ON_TABLE(BLOCK_RED1 WORKSPACE_TABLE)
CLEAR(BLOCK_RED1)

State

INITIALIZED()
NOT-CLEAR(BLOCK_RED1)
ON(BLOCK_RED2 BLOCK_RED1)
"""
    assert parse_ff_states(sample) == [
        ("(clear block_red1)", "(gripper_at_init)", "(on_table block_red1 workspace_table)"),
        ("(initialized)", "(on block_red2 block_red1)"),
    ]
    assert parse_ff_states("ff: found legal plan as follows") == []


@pytest.mark.skipif(not DOMAIN.exists(), reason="domain not present")
def test_ff_red_tower_program():
    ff = default_ff_binary()
    p = parse_problem(RED_TOWER)
    res = run_ff(DOMAIN, render_problem(p), ff)
    assert res.ok, res.error
    prog = program_from_plan(res.actions, p.name, 0, res.states)
    assert [str(s) for s in prog.skills] == [
        "pick(block_red2)",
        "place_on(block_red2, block_red1, o8)",
        "pick(block_red3)",
        "place_on(block_red3, block_red2, o8)",
    ]
    assert res.seconds < 5.0
    # each skill carries the state its PDDL action starts from (FF's state before it)
    assert len(res.states) == len(res.actions)
    first, second = prog.skills[0].hl_state, prog.skills[2].hl_state
    assert prog.skills[1].hl_state == first and prog.skills[3].hl_state == second
    assert {"(initialized)", "(clear block_red1)", "(clear block_red2)"} <= set(first)
    assert "(on_table block_red2 workspace_table)" in first
    assert "(on block_red2 block_red1)" in second and "(clear block_red1)" not in second
    assert any(p.startswith("(yaw_next ") for p in second)  # statics from init are kept



GOAL_TREE_PROBLEM = """
(define (problem goal_tree)
    (:domain blocks-tamp)
    (:objects
        block_red1 block_red2 - block_red
        block_s - block_letter
        workspace_table - table
        robot - gripper
    )
    (:init
        (on_table block_red1 workspace_table) (on_table block_red2 workspace_table)
        (on_table block_s workspace_table) (clear block_red1) (clear block_red2) (clear block_s)
        (ori_of block_red1 o1) (ori_of block_red2 o1) (ori_of block_s o5)
        (handempty robot) (gripper_at_init)
    )
    (:goal (and
        (on block_s block_red1)
        (or (ori_of block_s o2) (ori_of block_s o10))
        (exists (?a - block_red ?b - block_red) (and (not (= ?a ?b)) (on ?a ?b)))
        (forall (?b - block_letter) (not (on_table ?b workspace_table)))
        (and (clear block_s))
    ))
)
"""


def test_parse_goal_tree(tmp_path):
    from toyblocks.vecgen.symbolic.pddl import And, Not, Or

    path = tmp_path / "goal_tree.pddl"
    path.write_text(GOAL_TREE_PROBLEM)
    p = parse_problem(path)
    assert p.goal[0] == Fact("on", ("block_s", "block_red1"))
    assert p.goal[1] == Or((Fact("ori_of", ("block_s", "o2")), Fact("ori_of", ("block_s", "o10"))))
    # exists over two block_red vars with the (= ?a ?b) pairs folded away
    assert p.goal[2] == Or((Fact("on", ("block_red1", "block_red2")), Fact("on", ("block_red2", "block_red1"))))
    # forall over the single letter block collapses to its one instance
    assert p.goal[3] == Not(Fact("on_table", ("block_s", "workspace_table")))
    # a nested (and ...) is flattened into the top-level conjuncts
    assert p.goal[4] == Fact("clear", ("block_s",))
    assert not any(isinstance(g, And) for g in p.goal)
    text = render_problem(p)
    assert "(or (ori_of block_s o2) (ori_of block_s o10))" in text
    assert "(not (on_table block_s workspace_table))" in text


def test_goal_tree_satisfied():
    import torch

    from toyblocks.vecgen.symbolic.pddl import And, Not, Or
    from toyblocks.vecgen.symbolic.predicates import ORI_QUATERNIONS, goal_facts_satisfied

    # block_a on the table at o10, block_b on top of it at o1; table top at z=0, 4 cm cubes
    size = 0.04
    poses = torch.zeros(1, 2, 7)
    poses[0, 0, 2] = size / 2
    poses[0, 0, 3:7] = ORI_QUATERNIONS[9]
    poses[0, 1, 2] = 1.5 * size
    poses[0, 1, 3:7] = ORI_QUATERNIONS[0]
    sizes = torch.full((1, 2, 3), size)
    table = torch.tensor([[0.0, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0]])
    index = {"block_a": 0, "block_b": 1}
    ori = lambda b, o: Fact("ori_of", (b, o))  # noqa: E731
    exprs = [
        Or((ori("block_a", "o2"), ori("block_a", "o10"))),
        Or((ori("block_a", "o2"), ori("block_a", "o4"))),
        And((Fact("on", ("block_b", "block_a")), Not(Fact("clear", ("block_a",))))),
        Not(Or((ori("block_a", "o2"), ori("block_a", "o10")))),
    ]
    ok = goal_facts_satisfied(exprs, index, poses, sizes, table)
    assert [bool(t[0]) for t in ok] == [True, False, True, False]


@pytest.mark.skipif(not DOMAIN.exists(), reason="domain not present")
def test_ff_or_goal_picks_cheapest(tmp_path):
    """An (or ...) orientation goal lets FF stop at whichever alternative is nearer."""
    path = tmp_path / "goal_tree.pddl"
    path.write_text(GOAL_TREE_PROBLEM)
    p = parse_problem(path)
    res = run_ff(DOMAIN, render_problem(p), default_ff_binary())
    assert res.ok, res.error
    final_ori = [a.args[-1] for a in res.actions if a.args and a.args[1] == "block_s"][-1]
    assert final_ori in ("o2", "o10")
