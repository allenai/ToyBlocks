"""toyblocks.playdata.skill_balance: skill counting and the balanced goal policy."""
import random

import pytest

from toyblocks.playdata.goal_policy import Decision
from toyblocks.playdata.skill_balance import (
    BalancedSkillCfg,
    BalancedSkillGoalPolicy,
    drop_done_after_non_pnp,
    plan_skills,
    target_shares,
)
from toyblocks.playdata.symbolic_state import SymbolicState
from toyblocks.task_gen.orientations import flip, yaw

BLOCKS = ["block_a", "block_i", "block_2", "block_e"]


def _state(rng: random.Random) -> SymbolicState:
    return SymbolicState(blocks=list(BLOCKS), table="workspace_table", gripper="robot",
                         on={}, on_table=list(BLOCKS), ori={b: f"o{rng.randint(1, 24)}" for b in BLOCKS})


def _apply(state: SymbolicState, predicate: str) -> None:
    name, *args = predicate.strip("()").split()
    if name == "ori_of":
        state.ori[args[0]] = args[1]
    elif name == "on":
        state.on[args[0]] = args[1]
    elif name == "on_table":
        state.on.pop(args[0], None)


def test_plan_skills_counts_each_high_level_action_once() -> None:
    def a(name: str, args: list[str]) -> dict:
        return {"generated_values": {"_hl_action_name": name, "_hl_action_args": args}}
    flip_args = ["robot", "block_a", "left_flip", "o3", "o22", "workspace_table"]
    policy = [a("init", []), a("move_to", ["robot", "block_a", "block_a"]),
              a("reorient", flip_args), a("reorient", flip_args),
              a("rotate", ["robot", "block_i", "ccw", "o1", "o4", "workspace_table"]),
              a("pnp_table_block", ["robot", "block_2", "block_e", "workspace_table", "o6", "o6"]),
              a("done", [])]
    assert plan_skills(policy) == ["flip:left_flip", "rotate:ccw", "pnp"]


def test_target_shares_follow_the_allowed_actions() -> None:
    full = target_shares(None)
    assert full["pnp"] == pytest.approx(1 / 3)
    assert full["flip:forward"] == pytest.approx(1 / 12)
    assert full["rotate:cw"] == pytest.approx(1 / 6)
    no_rotate = target_shares(["init", "move_to", "pnp_table_block", "pnp_block_table", "reorient"])
    assert no_rotate["pnp"] == pytest.approx(0.5) and "rotate:cw" not in no_rotate


def test_flip_and_rotate_goals_are_one_action_away() -> None:
    rng = random.Random(0)
    state = _state(rng)
    policy = BalancedSkillGoalPolicy(BalancedSkillCfg(), rng=random.Random(1))
    for _ in range(50):
        decision = policy.next_goal(0, state, lambda *_: False)
        assert decision.kind is Decision.GOAL
        cls = decision.goal.families[0]
        name, *args = decision.goal.predicates[0].strip("()").split()
        if cls.startswith(("flip:", "rotate:")):
            family, direction = cls.split(":")
            expected = (flip if family == "flip" else yaw)(state.ori[args[0]], direction)
            assert (name, args[1]) == ("ori_of", expected)
        policy.on_plan_skills(0, [cls])
        _apply(state, decision.goal.predicates[0])


def test_mix_converges_even_with_extra_pick_and_place() -> None:
    rng = random.Random(0)
    state = _state(rng)
    policy = BalancedSkillGoalPolicy(BalancedSkillCfg(), rng=random.Random(1))
    for _ in range(600):
        decision = policy.next_goal(0, state, lambda *_: False)
        cls = decision.goal.families[0]
        # The planner sometimes has to add a pick-and-place (e.g. to make room).
        policy.on_plan_skills(0, [cls] + (["pnp"] if rng.random() < 0.2 else []))
        _apply(state, decision.goal.predicates[0])
    total = sum(policy.counts.values())
    for cls, share in policy.shares.items():
        assert policy.counts[cls] / total == pytest.approx(share, abs=0.02), cls


def test_stacking_respects_the_tower_cap() -> None:
    state = SymbolicState(blocks=list(BLOCKS), table="workspace_table", gripper="robot",
                          on={"block_i": "block_a", "block_2": "block_i"}, on_table=["block_a", "block_e"],
                          ori={b: "o1" for b in BLOCKS})
    policy = BalancedSkillGoalPolicy(BalancedSkillCfg(allowed_actions=("init", "move_to", "pnp_table_block")),
                                     rng=random.Random(0))
    for _ in range(20):
        goal = policy.next_goal(0, state, lambda *_: False).goal
        assert goal.predicates[0] != "(on block_e block_2)"  # would make a 4-block tower


def _plan(*hl_names: str) -> list[dict]:
    """A flat policy with two low-level entries per high-level action."""
    return [{"generated_values": {"_hl_action_name": n, "_hl_action_args": []}, "ll_action_name": f"{n}_{i}"}
            for n in hl_names for i in range(2)]


@pytest.mark.parametrize("last", ["reorient", "rotate"])
def test_done_dropped_after_flip_or_rotate(last: str) -> None:
    trimmed = drop_done_after_non_pnp(_plan("init", "pnp_block_table", last, "done"))
    assert [a["generated_values"]["_hl_action_name"] for a in trimmed][-1] == last
    assert len(trimmed) == 6


@pytest.mark.parametrize("last", ["pnp_table_block", "pnp_block_table", "move_to_side", "move_to_center"])
def test_done_kept_after_pick_and_place(last: str) -> None:
    plan = _plan("reorient", last, "done")
    assert drop_done_after_non_pnp(plan) == plan


def test_done_kept_when_plan_has_no_real_action() -> None:
    for plan in (_plan("done"), _plan("init", "reach-goal")):
        assert drop_done_after_non_pnp(plan) == plan


def test_plan_without_done_is_unchanged() -> None:
    plan = _plan("rotate")
    assert drop_done_after_non_pnp(plan) is plan


def test_flip_and_rotate_goals_only_target_glyph_blocks() -> None:
    rng = random.Random(3)
    mixed = ["block_a", "block_2", "block_red1", "block_green"]
    policy = BalancedSkillGoalPolicy(BalancedSkillCfg(), rng=random.Random(0))
    for _ in range(200):
        state = SymbolicState(blocks=list(mixed), table="workspace_table", gripper="robot",
                              on={}, on_table=list(mixed), ori={b: f"o{rng.randint(1, 24)}" for b in mixed})
        decision = policy.next_goal(0, state, lambda *_: False)
        assert decision.kind is Decision.GOAL and decision.goal is not None
        if decision.goal.text.startswith("(ori_of"):
            assert decision.goal.text.split()[1] in {"block_a", "block_2"}
        policy.on_plan_skills(0, list(decision.goal.families))


def test_color_only_scene_falls_back_to_pick_and_place() -> None:
    colors = ["block_red1", "block_green", "block_blue"]
    policy = BalancedSkillGoalPolicy(BalancedSkillCfg(), rng=random.Random(1))
    state = SymbolicState(blocks=list(colors), table="workspace_table", gripper="robot",
                          on={}, on_table=list(colors), ori={b: "o1" for b in colors})
    for _ in range(20):
        decision = policy.next_goal(0, state, lambda *_: False)
        assert decision.kind is Decision.GOAL and decision.goal is not None
        assert decision.goal.families == ["pnp"]


def test_count_successes_only_counts_kept_episodes() -> None:
    policy = BalancedSkillGoalPolicy(BalancedSkillCfg(count_successes=True), rng=random.Random(0))
    policy.on_plan_skills(0, ["flip:forward"])
    policy.on_episode_result(0, False)          # missed goal: dropped from the data
    policy.on_plan_skills(1, ["pnp"])
    policy.on_episode_result(1, True)           # kept
    assert policy.counts == {"pnp": 1}
    # default: counted when installed, regardless of the outcome
    legacy = BalancedSkillGoalPolicy(BalancedSkillCfg(), rng=random.Random(0))
    legacy.on_plan_skills(0, ["flip:forward"])
    legacy.on_episode_result(0, False)
    assert legacy.counts == {"flip:forward": 1}


def test_flip_min_x_skips_blocks_near_the_robot() -> None:
    blocks = ["block_a", "block_2"]
    xs = {"block_a": 0.30, "block_2": 0.55}
    policy = BalancedSkillGoalPolicy(BalancedSkillCfg(flip_min_x={"forward": 0.45}), rng=random.Random(2))
    policy.set_scene_accessors(pose_of=lambda env, b: [xs[b], 0.0, 0.02, 1, 0, 0, 0], table="workspace_table")
    state = SymbolicState(blocks=list(blocks), table="workspace_table", gripper="robot",
                          on={}, on_table=list(blocks), ori={b: "o1" for b in blocks})
    for _ in range(50):
        goal = policy._goal_for("flip:forward", state, env_idx=0)
        assert goal is not None and goal.text.split()[1] == "block_2"
    # other directions are unrestricted
    seen = {policy._goal_for("flip:backward", state, env_idx=0).text.split()[1] for _ in range(50)}
    assert seen == {"block_a", "block_2"}


def test_missed_flip_goal_is_not_requested_again_for_that_block() -> None:
    blocks = ["block_a", "block_2"]
    policy = BalancedSkillGoalPolicy(BalancedSkillCfg(count_successes=True), rng=random.Random(3))
    state = SymbolicState(blocks=list(blocks), table="workspace_table", gripper="robot",
                          on={}, on_table=list(blocks), ori={b: "o1" for b in blocks})
    missed = policy._goal_for("flip:forward", state, env_idx=0).text.split()[1]
    policy.on_episode_result(0, False)
    other = next(b for b in blocks if b != missed)
    for _ in range(20):
        assert policy._goal_for("flip:forward", state, env_idx=0).text.split()[1] == other
        policy._issued.clear()
    # other classes and other envs still offer the missed block
    assert {policy._goal_for("flip:backward", state, env_idx=0).text.split()[1] for _ in range(30)} == set(blocks)
    assert {policy._goal_for("flip:forward", state, env_idx=1).text.split()[1] for _ in range(30)} == set(blocks)
    # a reached goal on the block, or an env reset, lifts the exclusion
    policy._issued[0] = ("rotate:cw", missed)
    policy.on_episode_result(0, True)
    assert {policy._goal_for("flip:forward", state, env_idx=0).text.split()[1] for _ in range(30)} == set(blocks)
    policy._goal_for("flip:forward", state, env_idx=0)
    policy.on_episode_result(0, False)
    policy.reset(0)
    assert {policy._goal_for("flip:forward", state, env_idx=0).text.split()[1] for _ in range(30)} == set(blocks)
