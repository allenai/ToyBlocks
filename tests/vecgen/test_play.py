"""vecgen play data: TAMP-free goal policies, symbolic state from the live scene, chains."""

from __future__ import annotations

import math
import subprocess
import sys
from collections import deque

import torch

from blocksuite.vecgen.play import PlayConfig, PlayEngine, program_skills, read_symbolic_state
from blocksuite.vecgen.program import Program, SkillCall
from blocksuite.vecgen.symbolic.pddl import Fact, Problem
from blocksuite.vecgen.symbolic.programs import program_from_plan

EDGE = 0.04
TABLE = torch.tensor([0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
TOP = 0.01  # table collider top above its origin


def _pose(x: float, y: float, level: int, tilt_deg: float = 0.0) -> list[float]:
    half = math.radians(tilt_deg) / 2
    return [x, y, TOP + EDGE / 2 + level * EDGE, math.cos(half), math.sin(half), 0.0, 0.0]


def _problem(blocks: list[str]) -> Problem:
    objects = {b: "block_letter" for b in blocks}
    objects.update({"workspace_table": "table", "robot": "gripper"})
    return Problem("p", "blocks-tamp", objects, [], [])


def test_play_imports_without_tamp() -> None:
    code = (
        "import sys; sys.modules['tmp3'] = None\n"
        "import blocksuite.vecgen.play, blocksuite.playdata.skill_balance, "
        "blocksuite.playdata.abandon, blocksuite.playdata.goal_proposer\n"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_symbolic_state_reads_towers_cover_and_loose_blocks() -> None:
    # c on the table, b on c, a on b (a is also within the `on` window of c); an undeclared
    # block x sits on e; f floats; g is tilted 20 degrees
    names = ["block_a", "block_b", "block_c", "block_e", "block_x", "block_f", "block_g"]
    poses = torch.tensor(
        [
            _pose(0.5, 0.0, 2),
            _pose(0.5, 0.0, 1),
            _pose(0.5, 0.0, 0),
            _pose(0.4, 0.2, 0),
            _pose(0.4, 0.2, 1),
            _pose(0.6, -0.2, 6),
            _pose(0.6, 0.2, 0, tilt_deg=20.0),
        ]
    )
    index = {n: i for i, n in enumerate(names)}
    sizes = torch.full((len(names), 3), EDGE)
    prob = _problem([n for n in names if n != "block_x"])
    state, facts = read_symbolic_state(prob, index, poses, sizes, TABLE)
    assert state.on == {"block_a": "block_b", "block_b": "block_c"}
    assert Fact("on", ("block_a", "block_b")) in facts
    assert Fact("on", ("block_a", "block_c")) not in facts
    assert state.covered_by_untracked == ["block_e"]
    assert "block_e" not in state.clear_blocks()
    assert Fact("clear", ("block_e",)) not in facts
    assert state.unsupported == ["block_f"]
    assert state.tilted == ["block_g"]
    assert set(state.clear_blocks()) == {"block_a", "block_f", "block_g"}


def test_program_skills_count_each_high_level_action_once() -> None:
    plan = [
        Fact("pnp_table_block", ("robot", "block_a", "block_b", "workspace_table", "o1", "o1")),
        Fact("reorient", ("robot", "block_c", "left_flip", "o1", "o5", "workspace_table")),
        Fact("rotate", ("robot", "block_c", "cw", "o5", "o6", "workspace_table")),
        Fact("done", ()),
    ]
    prog = program_from_plan(plan, "p", 0)
    assert [c.name for c in prog.skills] == ["pick", "place_on", "flip", "rotate"]
    assert program_skills(prog) == ["pnp", "flip:left_flip", "rotate:cw"]


def _chain_engine(
    slots: list[int], states_by_slot: dict[int, list[int]], one_chain: bool
) -> PlayEngine:
    eng = PlayEngine.__new__(PlayEngine)
    eng.play = PlayConfig(one_chain_per_reset_state=one_chain)
    eng._env_slot = slots
    eng._rs_by_slot = states_by_slot
    eng._unplayed = {s: deque(v) for s, v in states_by_slot.items()}
    eng._played = set()
    eng._chain_rid = [None] * len(slots)
    eng.mode = ["reset"] * len(slots)
    import random

    eng.rng = random.Random(0)
    return eng


def test_one_chain_per_reset_state_plays_each_state_of_the_env_profile_once() -> None:
    eng = _chain_engine([0, 1, 0], {0: [0, 2, 4], 1: [1, 3]}, one_chain=True)
    got = [eng._chain_reset_state(e) for e in (0, 2, 1, 0, 1, 2, 1)]
    assert got == [0, 2, 1, 4, 3, None, None]


def test_without_one_chain_each_env_plays_one_chain_of_its_profile() -> None:
    eng = _chain_engine([0, 1], {0: [0, 2], 1: [1, 3]}, one_chain=False)
    first = eng._chain_reset_state(1)
    assert first in (1, 3)
    eng._chain_rid[1] = first
    assert eng._chain_reset_state(1) is None


def test_program_skills_ignore_bookkeeping_calls() -> None:
    prog = Program([SkillCall("pick", ("a",)), SkillCall("place_side", ("a", ""))], "p", 0)
    assert program_skills(prog) == []


def test_a_reset_state_matching_several_profiles_is_played_once() -> None:
    # equal profile values: every reset state is compatible with both slots
    eng = _chain_engine([0, 1], {0: [0, 1, 2], 1: [0, 1, 2]}, one_chain=True)
    got = [eng._chain_reset_state(e) for e in (0, 1, 1, 0, 1)]
    assert got == [0, 1, 2, None, None]
