"""Tests for PDDL goal → English prompt conversion."""

from __future__ import annotations

import pytest

from blocksuite.utils.prompt_resolver import (
    IDLE_PROMPT_PLACEHOLDER,
    PromptResolver,
    parse_prompt_for_setup,
    pddl_goal_to_prompt,
    policy_goal,
)

# Episode 0 of oct30_stack3_train_rg_readable_1000x32rs_glyphcolor_symm_N1, as a queue setup:
# graded in source block names, recorded (rendered) goal kept for the policy.
_RERENDER_SETUP = {
    "pddl_goal": "(and\n        (on_table block_blue workspace_table)\n        (on block_5 block_blue)\n"
                 "        (on block_1 block_5)\n        (clear block_1)\n    )",
    "policy_pddl_goal": "(and\n        (on_table block_green workspace_table)\n"
                        "        (on block_l block_green)\n        (on block_f block_l)\n"
                        "        (clear block_f)\n    )",
}


@pytest.mark.parametrize("flag,key", [(False, "pddl_goal"), (True, "policy_pddl_goal")])
def test_policy_goal_follows_replay_block_identity(flag: bool, key: str) -> None:
    assert policy_goal(_RERENDER_SETUP, flag) == _RERENDER_SETUP[key]
    assert parse_prompt_for_setup(_RERENDER_SETUP, replay_block_identity=flag) == (
        pddl_goal_to_prompt(_RERENDER_SETUP[key]))


def test_policy_goal_without_rendered_goal_is_the_grading_goal() -> None:
    setup = {"pddl_goal": _RERENDER_SETUP["pddl_goal"]}
    assert policy_goal(setup, True) == setup["pddl_goal"]


@pytest.mark.parametrize("flag", [False, True])
def test_resolver_prompts_with_its_flag(flag: bool) -> None:
    resolver = PromptResolver(1, replay_block_identity=flag)
    resolver.update_setups([_RERENDER_SETUP])
    assert resolver.prompt_for_env(0) == parse_prompt_for_setup(
        _RERENDER_SETUP, replay_block_identity=flag)


@pytest.mark.parametrize("flag,label", [(False, "Stack 1 5 BLUE"), (True, "Stack F L GREEN")])
def test_hl_stack_prompt_follows_replay_block_identity(flag: bool, label: str) -> None:
    from blocksuite.serving.molmo2_hl import stack_label

    assert stack_label(policy_goal(_RERENDER_SETUP, flag)) == label


def test_isolate_basic_skills_goal() -> None:
    goal = """(and
        (clear block_c)
        (not (pick_obstructs block_red4 block_c))
    )"""
    # Only one red cube in the goal → plain "the red block" (not RED4 / ordinals).
    assert pddl_goal_to_prompt(goal) == "isolate the C block from the red block"


def test_clear_without_pick_obstructs() -> None:
    assert pddl_goal_to_prompt("(and (clear block_r))") == "ensure the R block is clear"


def test_two_red_cubes_in_goal_get_red_and_dark_red() -> None:
    goal = "(and (on block_red1 block_red2) (clear block_red1))"
    assert "the red block" in pddl_goal_to_prompt(goal)
    assert "the dark red block" in pddl_goal_to_prompt(goal)


def test_single_red2_in_goal_is_plain_red() -> None:
    assert pddl_goal_to_prompt("(and (clear block_red2))") == "ensure the red block is clear"


def test_prompt_for_env_never_setup_returns_idle_placeholder() -> None:
    resolver = PromptResolver(2)
    assert resolver.prompt_for_env(0) == IDLE_PROMPT_PLACEHOLDER
    assert resolver.prompt_for_env(1) == IDLE_PROMPT_PLACEHOLDER


def test_prompt_for_env_drained_queue_keeps_last_prompt() -> None:
    resolver = PromptResolver(2)
    setup = {"pddl_goal": "(and (clear block_r))"}
    resolver.update_setups([setup, setup])
    prompt_env1 = resolver.prompt_for_env(1)
    assert prompt_env1 == "ensure the R block is clear"
    resolver.update_setups([setup, None])
    assert resolver.prompt_for_env(1) == prompt_env1
