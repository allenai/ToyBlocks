import math

import torch

from toyblocks.vecgen.math import quat_yaw
from toyblocks.vecgen.symbolic.predicates import (
    ORI_QUATERNIONS,
    classify_orientation,
    clear,
    on,
    on_table,
    ori_index,
    ori_of,
)


def test_ori_of_and_classify_all_canonical():
    idx = torch.arange(24)
    assert bool(ori_of(ORI_QUATERNIONS, idx).all())
    assert torch.equal(classify_orientation(ORI_QUATERNIONS), idx)
    # 30 deg yaw perturbation keeps the class; 60 deg does not satisfy ori_of (40 deg yaw tol)
    yaw30 = quat_yaw(torch.full((24,), math.radians(30.0)))
    from toyblocks.vecgen.math import quat_mul

    q30 = quat_mul(yaw30, ORI_QUATERNIONS)
    assert bool(ori_of(q30, idx).all())
    assert torch.equal(classify_orientation(q30), idx)
    yaw60 = quat_yaw(torch.full((24,), math.radians(60.0)))
    assert not bool(ori_of(quat_mul(yaw60, ORI_QUATERNIONS), idx).any())
    assert ori_index("o2") == 1


def test_on_on_table_clear():
    a = torch.tensor([[0.4, 0.0, 0.07, 1, 0, 0, 0]])
    b = torch.tensor([[0.41, 0.01, 0.03, 1, 0, 0, 0]])
    assert bool(on(a, b)[0]) and not bool(on(b, a)[0])
    size = torch.tensor([[0.04, 0.04, 0.04]])
    table = torch.tensor([[0.49, 0.0, 0.0, 1, 0, 0, 0]])
    assert bool(on_table(b, size, table)[0])
    assert not bool(on_table(a, size, table)[0])
    poses = torch.stack([a, b], dim=1)  # [1, 2, 7]
    assert bool(clear(poses, torch.tensor([0]))[0])
    assert not bool(clear(poses, torch.tensor([1]))[0])


def _pose(x: float, y: float, z: float) -> torch.Tensor:
    return torch.tensor([x, y, z, 1.0, 0.0, 0.0, 0.0])


def test_side_relations():
    from toyblocks.vecgen.symbolic.predicates import (
        on_back,
        on_front,
        on_left,
        on_right,
        pick_obstructs,
    )

    size = torch.full((3,), 0.04)
    b = _pose(0.5, 0.0, 0.02)
    # on_right: |dy| = 0.02 + 0.02 + 0.005 (+-0.03), x within 0.04, z within 0.02
    assert bool(on_right(_pose(0.5, 0.045, 0.02), b, size, size))
    assert bool(on_right(_pose(0.5, -0.045, 0.02), b, size, size))  # side not checked
    assert bool(on_right(_pose(0.53, 0.07, 0.035), b, size, size))
    assert not bool(on_right(_pose(0.5, 0.08, 0.02), b, size, size))  # 3.5 cm too far
    assert not bool(on_right(_pose(0.55, 0.045, 0.02), b, size, size))  # x off by 5 cm
    assert not bool(on_right(_pose(0.5, 0.045, 0.05), b, size, size))  # z off by 3 cm
    # on_left / on_back: 6.5 cm (+-1 cm) along y / x; on_front: 5 cm along x
    assert bool(on_left(_pose(0.5, -0.065, 0.02), b)) and bool(on_left(_pose(0.5, 0.072, 0.02), b))
    assert not bool(on_left(_pose(0.5, -0.045, 0.02), b))
    assert not bool(on_left(_pose(0.5, -0.065, 0.035), b))  # z tolerance is 1 cm here
    assert bool(on_back(_pose(0.435, 0.0, 0.02), b)) and not bool(on_back(_pose(0.45, 0.0, 0.02), b))
    assert bool(on_front(_pose(0.55, 0.03, 0.02), b)) and not bool(on_front(_pose(0.57, 0.0, 0.02), b))
    # pick_obstructs: xy within 10 cm at the same height
    assert bool(pick_obstructs(_pose(0.57, 0.07, 0.02), b))
    assert not bool(pick_obstructs(_pose(0.58, 0.07, 0.02), b))
    assert not bool(pick_obstructs(_pose(0.5, 0.05, 0.05), b))


def test_goal_facts_side_relations_and_unknown_predicate_raises():
    import pytest

    from toyblocks.vecgen.symbolic.pddl import Fact, Not
    from toyblocks.vecgen.symbolic.predicates import assert_goal_supported, goal_facts_satisfied

    # word "CA" in a line: block_c to the right of block_a, plus a far-away block_t
    poses = torch.stack([_pose(0.5, 0.0, 0.02), _pose(0.5, 0.045, 0.02), _pose(0.3, -0.2, 0.02)])
    poses = poses.unsqueeze(0)
    sizes = torch.full((1, 3, 3), 0.04)
    table = torch.tensor([[0.5, 0.0, -0.01, 1.0, 0.0, 0.0, 0.0]])
    index = {"block_a": 0, "block_c": 1, "block_t": 2}
    goals = [
        Fact("on_right", ("block_c", "block_a")),
        Fact("next_to", ("block_c", "block_a")),
        Fact("pick_obstructs", ("block_a", "block_c")),
        Not(Fact("next_to", ("block_t", "block_a"))),
        Fact("on_left", ("block_t", "block_a")),
    ]
    assert_goal_supported(goals)
    ok = goal_facts_satisfied(goals, index, poses, sizes, table)
    assert [bool(t[0]) for t in ok] == [True, True, True, True, False]
    bad = [Fact("holding", ("robot", "block_a"))]
    with pytest.raises(ValueError, match="holding"):
        assert_goal_supported(bad)
    with pytest.raises(ValueError, match="holding"):
        goal_facts_satisfied(bad, index, poses, sizes, table)


def test_derive_init_facts_keeps_declared_relations(tmp_path):
    from toyblocks.vecgen.symbolic.pddl import Fact, parse_problem
    from toyblocks.vecgen.symbolic.state import derive_init_facts

    path = tmp_path / "isolate.pddl"
    path.write_text(
        """(define (problem iso) (:domain blocks-tamp)
  (:objects block_r block_n block_far - block workspace_table - table robot - gripper)
  (:init (on_table block_r workspace_table) (on_table block_n workspace_table)
    (on_table block_far workspace_table) (clear block_r) (clear block_n) (clear block_far)
    (next_to block_r block_n) (pick_obstructs block_n block_r)
    (ori_of block_r o1) (ori_of block_n o1) (ori_of block_far o1)
    (handempty robot) (gripper_at_init))
  (:goal (and (on_right block_far block_n))))"""
    )
    prob = parse_problem(path)
    index = {"block_r": 0, "block_n": 1, "block_far": 2}
    sizes = torch.full((3, 3), 0.04)
    table = torch.tensor([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    # block_r flush against block_n along x; block_far 8.6 cm from block_r (pick_obstructs-close,
    # but not declared for that pair)
    poses = torch.stack([_pose(0.5, 0.0, 0.03), _pose(0.54, 0.0, 0.03), _pose(0.45, 0.07, 0.03)])
    facts = set(derive_init_facts(prob, index, poses, sizes, table))
    assert facts == set(prob.init)
    # the goal's on_right is asserted once it holds; next_to is dropped once the blocks part
    poses[2] = _pose(0.54, 0.045, 0.03)
    poses[0] = _pose(0.3, -0.2, 0.03)
    facts = set(derive_init_facts(prob, index, poses, sizes, table))
    assert Fact("on_right", ("block_far", "block_n")) in facts
    assert Fact("next_to", ("block_r", "block_n")) not in facts
    assert Fact("pick_obstructs", ("block_n", "block_r")) not in facts
