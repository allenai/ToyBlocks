"""The oracle's plan tracking: step checks, the last step, and re-planning on drift.

The real ``RecoveringOracleHLPolicy`` (the ported oracle + its recovery overrides) plans
with the real (vendored) FF binary against a stand-in world: the success gate's
``batched_goal_conjunct`` / ``resolve_root_pose_w`` are patched to read a set of
``(top, bottom)`` stacks and per-block orientations, so no Isaac is needed. The pure
``live_state`` helpers and the sim adapter's sizes are tested directly.
"""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import torch

from blocksuite.eval.oracle import live_state as ls

ROOT = Path(__file__).resolve().parents[2]
DOMAIN = ROOT / "src/blocksuite/tamp/blocks/domain.pddl"
TABLE = "workspace_table"

ON_TABLE_ABC = (
    "(on_table block_a workspace_table) (clear block_a) (on_table block_b workspace_table) "
    "(clear block_b) (on_table block_c workspace_table) (clear block_c)"
)
A_ON_B = (
    "(on block_a block_b) (clear block_a) (on_table block_b workspace_table) "
    "(on_table block_c workspace_table) (clear block_c)"
)


def _problem(tmp_path: Path, init: str, goal: str) -> str:
    path = tmp_path / "problem.pddl"
    path.write_text(
        "(define (problem t) (:domain blocks-tamp)\n"
        "  (:objects block_a block_b block_c - block workspace_table - table robot - gripper)\n"
        f"  (:init (gripper_at_init) (handempty robot) {init}\n"
        "         (ori_of block_a o1) (ori_of block_b o1) (ori_of block_c o1))\n"
        f"  (:goal (and {goal})))\n"
    )
    return str(path)


# --------------------------------------------------------------------------
# live_state (pure)
# --------------------------------------------------------------------------


def test_norm_fact_repairs_the_oracle_parsers_zero_arity_facts() -> None:
    assert ls.norm_fact("(initialized )") == "(initialized)"
    assert ls.norm_fact("(ON_TABLE  block_a workspace_table)") == "(on_table block_a workspace_table)"
    assert ls.parse_fact("(on block_a block_b)") == ("on", ("block_a", "block_b"))


def test_flat_goal_literals_only_for_plain_conjunctions(tmp_path: Path) -> None:
    flat = Path(_problem(tmp_path, ON_TABLE_ABC, "(on block_a block_b) (not (on block_c block_a))")).read_text()
    assert ls.flat_goal_literals(flat) == (("on", ("block_a", "block_b")),)
    quantified = flat.replace(
        "(:goal (and (on block_a block_b) (not (on block_c block_a))))",
        "(:goal (exists (?x - block) (on ?x block_b)))",
    )
    assert ls.flat_goal_literals(quantified) == ()


def test_replace_init_swaps_only_the_init_body(tmp_path: Path) -> None:
    text = Path(_problem(tmp_path, ON_TABLE_ABC, "(on block_a block_b)")).read_text()
    out = ls.replace_init(text, ["(handempty robot)", "(on block_a block_b)"])
    assert "(:init\n        (handempty robot)\n        (on block_a block_b)\n    )" in out
    assert "(on_table block_c" not in out and "(:goal (and (on block_a block_b)))" in out


def test_step_violations_name_covered_and_stacked_blocks() -> None:
    live = ls.LiveState(support={"block_a": "block_b", "block_b": None, "block_c": None})
    assert ls.step_violations("rotate", "block_a", None, live) == ["block_a is stacked on block_b"]
    assert ls.step_violations("pnp_table_block", "block_b", None, live) == [
        "block_b is not clear (block_a on it)"
    ]
    assert ls.step_violations("pnp_table_block", "block_c", "block_b", live) == [
        "place target block_b is not clear (block_a on it)"
    ]
    assert ls.step_violations("pnp_table_block", "block_a", "block_c", live) == []


def test_corrected_init_replaces_observed_facts_and_keeps_the_rest() -> None:
    expected = {
        "(initialized )", "(yaw_next cw o1 o2)", "(at robot block_c)", "(handempty robot)",
        "(on_table block_a workspace_table)", "(clear block_a)", "(clear block_b)",
        "(on_table block_b workspace_table)", "(ori_of block_a o1)", "(ori_of block_b o3)",
        "(pick_obstructs block_a block_b)",
    }
    live = ls.LiveState(
        support={"block_a": "block_b", "block_b": None},
        on_table={"block_b"},
        ori={"block_a": "o2"},  # block_b tilted past tolerance: keeps the expected o3
    )
    assert set(ls.corrected_init(expected, live, TABLE, "robot")) == {
        "(initialized)", "(yaw_next cw o1 o2)", "(pick_obstructs block_a block_b)",
        "(handempty robot)", "(on block_a block_b)", "(clear block_a)",
        "(on_table block_b workspace_table)", "(ori_of block_a o2)", "(ori_of block_b o3)",
    }
    assert ls.blocks_and_table(expected) == (["block_a", "block_b"], TABLE)


# --------------------------------------------------------------------------
# sim adapter sizes
# --------------------------------------------------------------------------


def test_adapter_sizes_come_from_the_bound_envs_spawn(monkeypatch: pytest.MonkeyPatch) -> None:
    # sim_adapter imports tmp3's quaternion ops, whose package pulls in curobo; sizes need
    # neither, so stand in for that package.
    shared = types.ModuleType("tmp3.Backends._shared")
    quat_ops = types.ModuleType("tmp3.Backends._shared.quat_ops")
    quat_ops.quat_inv = quat_ops.quat_mul = quat_ops.quat_rotate_vec = None
    for name, mod in (
        ("tmp3", types.ModuleType("tmp3")),
        ("tmp3.Backends", types.ModuleType("tmp3.Backends")),
        ("tmp3.Backends._shared", shared),
        ("tmp3.Backends._shared.quat_ops", quat_ops),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.delitem(sys.modules, "blocksuite.eval.oracle.sim_adapter", raising=False)
    from blocksuite.eval.oracle.sim_adapter import BaseEnvOracleSimAdapter

    cuboid = types.SimpleNamespace(size=(1.0, 0.8, 0.05))  # a CuboidCfg's fields
    multi = types.SimpleNamespace(assets_cfg=[types.SimpleNamespace(size=(0.1,) * 3), cuboid], asset_indices=[0, 1])
    table = types.SimpleNamespace(cfg=types.SimpleNamespace(spawn=cuboid))
    tray = types.SimpleNamespace(cfg=types.SimpleNamespace(spawn=multi))
    block = types.SimpleNamespace(cfg=types.SimpleNamespace(spawn=None))
    base_env = types.SimpleNamespace(
        scene=types.SimpleNamespace(rigid_objects={TABLE: table, "tray": tray, "block_a": block}, articulations={}),
        randomizations={"block_sizes": {"block_a": [[0.04] * 3, [0.05] * 3]}},
    )
    adapter = object.__new__(BaseEnvOracleSimAdapter)
    adapter._base_env, adapter._i = base_env, 1
    assert adapter.get_rigid_object(TABLE).get_size(None) == (1.0, 0.8, 0.05)
    assert adapter.get_rigid_object("tray").get_size(None) == (1.0, 0.8, 0.05)  # env 1's slot
    # env 1 spawned the second scale slot, whatever env index the single-env oracle passes
    assert adapter.get_rigid_object("block_a").get_size(0) == (0.05, 0.05, 0.05)
    with pytest.raises(ValueError, match="not found"):
        adapter.get_rigid_object("block_zz")
    sys.modules.pop("blocksuite.eval.oracle.sim_adapter")  # built on the stand-ins


# --------------------------------------------------------------------------
# the oracle (OracleHighLevelPolicy + RecoveringOracleHLPolicy) on a stand-in world
# --------------------------------------------------------------------------


class _World:
    """Stacks ``{(top, bottom)}`` and orientations; answers the success gate's checks."""

    def __init__(self, on: set[tuple[str, str]] | None = None, raise_for: tuple[str, ...] = ()) -> None:
        self.on = set(on or ())
        self.ori: dict[str, str] = {}
        self.raise_for = raise_for

    def holds(self, pred: str, args: tuple[str, ...]) -> bool:
        if pred in self.raise_for:
            raise RuntimeError(f"cannot evaluate {pred}")
        if pred == "on":
            return tuple(args) in self.on
        if pred == "on_table":
            return all(top != args[0] for top, _ in self.on)
        if pred == "clear":
            return all(bottom != args[0] for _, bottom in self.on)
        if pred == "ori_of":
            return self.ori.get(args[0], "o1") == args[1]
        raise ValueError(pred)

    def z(self, name: str) -> float:
        below = [b for t, b in self.on if t == name]
        return 0.045 + (self.z(below[0]) if below else 0.0)


class _Sim:
    """The sim adapter surface the oracle and the overrides use (no ``get_rigid_object``,
    like the adapter before the fix)."""

    def __init__(self, world: _World) -> None:
        self.base_env, self.env_index = world, 0
        self.robots: dict[str, Any] = {}
        self.scene = types.SimpleNamespace(sensors={}, rigid_objects={}, env_origins=torch.zeros(1, 3))

    def get_pose(self, name: str, env_idx: int | None = 0, local: bool = True) -> torch.Tensor:
        return torch.tensor([0.5, 0.0, self.base_env.z(name), 1.0, 0.0, 0.0, 0.0])

    def get_velocity(self, name: str, env_idx: int | None = 0, local: bool = False) -> torch.Tensor:
        return torch.zeros(6)


@pytest.fixture
def make_oracle(monkeypatch: pytest.MonkeyPatch) -> Any:
    from blocksuite.eval.oracle import recovering
    from blocksuite.tamp.blocks import pddl_goal_tensor_conversions as g
    from blocksuite.vecgen.symbolic.ff import default_ff_binary

    try:
        default_ff_binary()
    except FileNotFoundError:
        pytest.skip("needs the FF binary (make -C src/blocksuite/vecgen/symbolic/ff_planner)")
    monkeypatch.setattr(g, "batched_goal_conjunct", lambda env, p, a: torch.tensor([env.holds(p, a)]))
    monkeypatch.setattr(
        g, "resolve_root_pose_w", lambda env, n: torch.tensor([[0.5, 0.0, env.z(n), 1.0, 0.0, 0.0, 0.0]])
    )

    def make(problem_file: str, world: _World, **kwargs: Any) -> Any:
        sim = _Sim(world)
        return recovering.make_oracle(
            domain_file=str(DOMAIN), problem_file=problem_file, simulator=sim, scene=sim.scene, **kwargs
        )

    return make


def _plan(oracle: Any) -> list[tuple[str, list[str]]]:
    return [(s["action"], s["effect_predicates"]) for s in oracle.get_plan_summary()]


def test_unstack_waits_for_the_block_to_reach_the_table(make_oracle: Any, tmp_path: Path) -> None:
    world = _World(on={("block_a", "block_b")})
    oracle = make_oracle(_problem(tmp_path, A_ON_B, "(on block_b block_a)"), world)
    assert _plan(oracle) == [
        ("pnp_block_table", ["(on_table block_a workspace_table)"]),
        ("pnp_table_block", ["(on block_b block_a)"]),  # the last step has effects now
    ]
    # Upstream (TAMP >= 7a0eff62) no longer passes an on_table check that cannot run.
    upstream = type(oracle).__mro__[1]._evaluate_effect_predicates
    assert upstream(oracle, 0) is False
    assert oracle._evaluate_effect_predicates(0) is False
    world.on.clear()
    assert oracle._evaluate_effect_predicates(0) is True


def test_a_check_that_cannot_run_does_not_pass(make_oracle: Any, tmp_path: Path, caplog: Any) -> None:
    world = _World(on={("block_a", "block_b")}, raise_for=("on_table",))
    oracle = make_oracle(_problem(tmp_path, A_ON_B, "(on block_b block_a)"), world)
    world.on.clear()
    with caplog.at_level(logging.ERROR):
        assert oracle._evaluate_effect_predicates(0) is False
    assert "treating it as NOT holding" in caplog.text


def test_boundary_replan_finishes_when_the_drifted_world_already_meets_the_goal(
    make_oracle: Any, tmp_path: Path
) -> None:
    world = _World()
    oracle = make_oracle(
        _problem(tmp_path, ON_TABLE_ABC, "(on block_a block_b) (on block_c block_a)"), world, replan=True
    )
    assert [a for a, _ in _plan(oracle)] == ["pnp_table_block", "pnp_table_block"]
    world.on = {("block_a", "block_b"), ("block_c", "block_a")}  # the LL did both at once
    oracle._advance(0)  # step 0 done; step 1's place target block_a is now covered
    assert oracle.replans == 1 and oracle.plan_length == 0
    assert oracle.get_current_high_level_action(0) == "done"


def test_boundary_replan_unstacks_a_tower_left_on_the_next_pick(make_oracle: Any, tmp_path: Path) -> None:
    world = _World()
    oracle = make_oracle(
        _problem(tmp_path, ON_TABLE_ABC, "(on block_a block_b) (on block_c block_a)"), world, replan=True
    )
    # Step 0 put block_a on block_b as planned, but that tower now stands on block_c, the
    # block step 1 picks.
    world.on = {("block_a", "block_b"), ("block_b", "block_c")}
    oracle._advance(0)
    assert oracle.replans == 1
    first = oracle.get_plan_summary()[0]
    assert first["action"].startswith("pnp_block") and first["args"][1] in ("block_a", "block_b")


def test_mid_step_replan_when_the_block_to_rotate_got_stacked(make_oracle: Any, tmp_path: Path) -> None:
    world = _World()
    oracle = make_oracle(_problem(tmp_path, ON_TABLE_ABC, "(ori_of block_a o2)"), world, replan=True)
    assert [a for a, _ in _plan(oracle)] == ["rotate"]
    world.on = {("block_a", "block_b")}  # the LL stacked block_a instead of rotating it
    every = type(oracle)._DRIFT_CHECK_EVERY
    for _ in range(every):  # first failing check: noted, not acted on
        oracle.get_current_high_level_action(0)
    assert oracle.replans == 0
    for _ in range(every):  # second consecutive failing check: re-plan
        oracle.get_current_high_level_action(0)
    assert oracle.replans == 1
    first = oracle.get_plan_summary()[0]
    assert first["action"] == "pnp_block_table" and first["args"][1] == "block_a"


def test_replans_are_capped(make_oracle: Any, tmp_path: Path) -> None:
    world = _World()
    oracle = make_oracle(_problem(tmp_path, ON_TABLE_ABC, "(ori_of block_a o2)"), world, replan=True, max_replans=0)
    world.on = {("block_a", "block_b")}
    for _ in range(4 * type(oracle)._DRIFT_CHECK_EVERY):
        oracle.get_current_high_level_action(0)
    assert oracle.replans == 0 and [a for a, _ in _plan(oracle)] == ["rotate"]


def test_no_replanning_unless_asked(make_oracle: Any, tmp_path: Path) -> None:
    world = _World()
    oracle = make_oracle(_problem(tmp_path, ON_TABLE_ABC, "(ori_of block_a o2)"), world)
    world.on = {("block_a", "block_b")}
    for _ in range(4 * type(oracle)._DRIFT_CHECK_EVERY):
        oracle.get_current_high_level_action(0)
    assert oracle.replans == 0
