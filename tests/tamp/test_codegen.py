"""CPU tests for the blocks codegen: PDDL parsing, randomization, env packaging.

``generate_task_from_pddl.py`` is otherwise exercised only end-to-end by GPU CI
on a single hand-written PDDL, so a malformed randomization annotation or a
broken goal compile is found after a full datagen run instead of in seconds.

Three things here back specific downstream guarantees:

- The randomization tests are the CPU guard on the mechanism ``tests/tamp/
  datagen.sh`` and ``tests/eval/pick_profile_episodes.py`` both depend on: the
  PDDL ``;*_range`` comments must reach codegen and must produce *distinct*
  profiles, or the eval profile selection has nothing to spread across.
- The scene/robot override tests cover the substitution that
  ``generate_task_from_dataset.py`` uses to bake a dataset's recorded
  provenance, which is what makes the eval scene test non-vacuous.
- The version test guards ``AUTOGEN_PACKAGE_VERSION``: autogen dirs are
  gitignored, so a stale package is indistinguishable from a fresh one.

Runs without Isaac Lab: codegen needs only stdlib, numpy and gymnasium (pulled in
by ``blocksuite/tamp/blocks/template/__init__.py``, which registers the template
env), and the templates themselves are read as text rather than imported.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

from blocksuite.tamp.blocks.template.codegen import (
    AUTOGEN_PACKAGE_VERSION,
    PREDICATE_ORDER,
    _override_env_cfg_fields,
    assert_goal_text_supported,
    compile_goal_text,
    extract_block_names,
    extract_clear_blocks,
    extract_goal_block,
    generate_environment_code,
    goal_conjunct_labels,
    parse_pddl_goal_from_text,
    parse_pddl_init_from_text,
    parse_pddl_randomization_ranges_from_text,
    reorder_predicates_by_dependencies,
    task_slug_from_pddl_path,
    write_env_package,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DOMAIN_PATH = REPO_ROOT / "src" / "blocksuite" / "tamp" / "blocks"
PROBLEMS_DIR = DOMAIN_PATH / "problems"

RANDOMIZATION_KEYS = {"block_size_range", "table_height_range", "friction_range"}

# The triple the production problems use (see problems/flip.pddl). Tests use
# these values rather than a narrowed range because generate_plans.py feeds the
# exact per-slot size to CuRobo, so wide sizes are what large-scale datagen runs.
PRODUCTION_RANGES = {
    "block_size_range": [0.03, 0.05],
    "table_height_range": [0, 0.04],
    "friction_range": [0.1, 0.4],
}

# What every checked-in problem is expected to annotate, spelled out rather than
# asserted as a subset: half of these files carry no annotations at all, so a
# subset check passes on them having verified nothing -- including if a typo made
# a range stop parsing. Listing the empties makes an unannotated problem a
# recorded decision, and the completeness assertion below forces a new problem to
# be added here rather than silently joining them.
EXPECTED_RANGES_BY_PROBLEM = {
    "flip.pddl": PRODUCTION_RANGES,
    "flip_a.pddl": PRODUCTION_RANGES,
    # Block size and table height are randomized; friction is left at its
    # default (the flip-then-place goal does not depend on it).
    "flip_then_pnp.pddl": {
        "block_size_range": [0.03, 0.05],
        "table_height_range": [0, 0.04],
    },
    "isolate.pddl": PRODUCTION_RANGES,
    "isolate_hard.pddl": PRODUCTION_RANGES,
    "push_align_hard.pddl": PRODUCTION_RANGES,
    # Block size and table height are fixed here: the push alignment goal is
    # specified against a particular geometry, so only friction is randomized.
    "push_align.pddl": {"friction_range": [0.1, 0.4]},
    # The production triple, as a letter / colour-glyph datagen problem.
    "problem_letter_color_10_20260730_002811_580532_4009.pddl": PRODUCTION_RANGES,
    # Unannotated: codegen falls back to one un-randomized profile repeated
    # across all 8 slots. Fine for these, but it means a dataset recorded from
    # them has a single (cube_scale, table_height_delta) profile and cannot
    # drive tests/eval/pick_profile_episodes.py with --num-profiles > 1.
    "ai2.pddl": {},
    "ai2_colored.pddl": {},
    "pnp_stack.pddl": {},
    "problem_rotate_block_3dist.pddl": {},
    "r_on_red.pddl": {},
    "r_on_red_5block.pddl": {},
    "red_on_green_2block.pddl": {},
    "red_stack3_15dist.pddl": {},
    "red_stack3_3dist.pddl": {},
    "red_tower_3.pddl": {},
    "rotate.pddl": {},
    "rotate_block.pddl": {},
}

SIMPLE_PDDL = """(define (problem simple)
    (:domain blocks-tamp)
    (:objects
        block_r - block_r_e
        block_red1 block_blue - block
        workspace_table - table
        robot - gripper
    )
    (:init
        (on block_red1 block_blue)
        (on_table block_blue workspace_table)
        (handempty robot)
        (clear block_r)
        (ori_of block_r o17)
    )
    (:goal (and
        (on block_red1 block_blue)
    ))
)
"""


def generate(**overrides):
    """``generate_environment_code`` for a two-block task, with overrides."""
    kwargs = dict(
        domain_path=str(DOMAIN_PATH),
        task_name="unit_test_task",
        block_names=["block_red1", "block_blue"],
        goal_texts=["(and (on block_red1 block_blue))"],
        goal_display_name="unit_test.pddl",
    )
    kwargs.update(overrides)
    return generate_environment_code(**kwargs)


# --------------------------------------------------------------------------
# Randomization annotations
# --------------------------------------------------------------------------


def test_every_checked_in_problem_has_an_expected_randomization_entry():
    """A new problem must declare what it randomizes, not inherit a silent pass."""
    on_disk = {p.name for p in PROBLEMS_DIR.glob("*.pddl")}
    assert on_disk == set(EXPECTED_RANGES_BY_PROBLEM), (
        "EXPECTED_RANGES_BY_PROBLEM is out of sync with problems/: "
        f"missing {sorted(on_disk - set(EXPECTED_RANGES_BY_PROBLEM))}, "
        f"stale {sorted(set(EXPECTED_RANGES_BY_PROBLEM) - on_disk)}"
    )


@pytest.mark.parametrize(
    "pddl_path", sorted(PROBLEMS_DIR.glob("*.pddl")), ids=lambda p: p.name
)
def test_checked_in_problems_parse_randomization(pddl_path):
    """Every checked-in problem parses into exactly the ranges it declares.

    A typo in a ``;*_range`` line is otherwise invisible -- the parser just
    doesn't match it, and datagen quietly runs one un-randomized profile.
    """
    ranges = parse_pddl_randomization_ranges_from_text(pddl_path.read_text())
    assert ranges == EXPECTED_RANGES_BY_PROBLEM[pddl_path.name]
    assert set(ranges) <= RANDOMIZATION_KEYS
    for key, value in ranges.items():
        assert len(value) == 2, f"{key} must be [lo, hi], got {value!r}"
        lo, hi = value
        assert lo < hi, f"{key} has lo >= hi: {value!r}"


def test_ranges_produce_distinct_profiles():
    """The property tests/eval/pick_profile_episodes.py assumes.

    Without distinct profiles there is nothing for the eval side to spread its
    cases across, and the randomization coverage would be silently vacuous.
    """
    profiles = generate(randomization_ranges=PRODUCTION_RANGES)["profiles"]["profiles"]
    assert len(profiles) == 8
    sizes = {p["block_sizes"]["block_red1"][0] for p in profiles}
    heights = {p["table_height"] for p in profiles}
    frictions = {p["static_friction"] for p in profiles}
    assert len(sizes) == 8
    assert len(heights) == 8
    assert len(frictions) == 8
    lo, hi = PRODUCTION_RANGES["block_size_range"]
    assert all(lo <= s <= hi for s in sizes)


def test_no_ranges_gives_identical_default_profiles():
    profiles = generate()["profiles"]["profiles"]
    assert len(profiles) == 8
    assert {p["block_sizes"]["block_red1"][0] for p in profiles} == {0.04}
    assert {p["table_height"] for p in profiles} == {0.0}
    assert {p["static_friction"] for p in profiles} == {0.3}


def test_cube_scale_slots_give_one_profile_per_distinct_scale():
    """Eval packages spawn one slot per recorded scale, in sorted order."""
    env_info = generate(cube_scales=[0.05, 0.03, 0.05, 0.04])
    profiles = env_info["profiles"]
    assert profiles["num_profiles"] == 3
    assert profiles["derive_block_scale_from_size"] is True
    assert env_info["non_sim_state_randomizations"]["cube_scale_slots"] == [
        0.03,
        0.04,
        0.05,
    ]


# --------------------------------------------------------------------------
# Env package round-trip
# --------------------------------------------------------------------------


def test_write_env_package_round_trip(tmp_path):
    env_info = generate(randomization_ranges=PRODUCTION_RANGES)
    write_env_package(str(tmp_path), env_info=env_info, env_id="unit_test_task-v0")

    for name in (
        "env.py",
        "env_cfg.py",
        "__init__.py",
        "non_sim_state_randomizations.json",
        "profiles.json",
    ):
        assert (tmp_path / name).is_file(), f"{name} missing from generated package"

    # Parse rather than import: the generated modules import isaaclab.
    ast.parse((tmp_path / "env.py").read_text())
    ast.parse((tmp_path / "env_cfg.py").read_text())

    profiles = json.loads((tmp_path / "profiles.json").read_text())
    assert profiles["codegen_version"] == AUTOGEN_PACKAGE_VERSION

    init_code = (tmp_path / "__init__.py").read_text()
    assert 'id="unit_test_task-v0"' in init_code
    assert "unit_test_task_Env" in init_code
    assert "unit_test_task_EnvCfg" in init_code


# --------------------------------------------------------------------------
# Scene / robot override
# --------------------------------------------------------------------------


def test_scene_and_robot_override_land_in_generated_cfg():
    """Both fields substitute, and the template annotates only ``robot_id``.

    ``color_only_blocks_table`` rather than ``droid_with_box_table``: the latter
    is verbatim the commented-out alternative on ``template_env_cfg.py:12``, so
    the assertion would be satisfied by that comment whether or not the
    substitution ran.
    """
    code = generate(
        scene_preset_id="color_only_blocks_table", robot_id="fr3_robotiq_gripper_soft"
    )["env_cfg_code"]
    assert 'scene_preset_id = "color_only_blocks_table"' in code
    assert 'robot_id: str = "fr3_robotiq_gripper_soft"' in code
    assert "sim2real" not in code
    assert "fr3_robotiq_gripper_sysid" not in code
    ast.parse(code)


def test_override_keeps_the_class_body_indent():
    """A substitution that dropped the indent would emit a module-level name.

    The generated file still parses -- the class simply loses the field and
    picks up ``BaseEnvCfg``'s default -- so nothing downstream would notice
    until the eval ran on the wrong scene. The assertions above match on
    substrings and cannot see this.
    """
    source = 'class C:\n    scene_preset_id = "sim2real"\n    x = 1\n'
    out = _override_env_cfg_fields(
        source, "<test>", scene_preset_id="color_only_blocks_table"
    )
    assert out == 'class C:\n    scene_preset_id = "color_only_blocks_table"\n    x = 1\n'


def test_two_live_assignments_to_one_field_raise():
    """Rewriting only the first would be silently undone by the second.

    Python evaluates a class body top to bottom, so the *last* assignment is the
    one that survives -- an override applied to the first is discarded while
    codegen reports success. ``template_env_cfg.py`` already carries a
    commented-out second ``scene_preset_id`` one line below the real one, so
    this is one uncommented line away rather than hypothetical.
    """
    source = 'class C:\n    scene_preset_id = "sim2real"\n    scene_preset_id = "table"\n'
    with pytest.raises(ValueError, match="could not override scene_preset_id"):
        _override_env_cfg_fields(source, "<test>", scene_preset_id="color_only_blocks_table")


def test_commented_out_assignment_is_not_a_second_declaration():
    """The template's own layout must keep passing: the comment is not code."""
    source = (
        'class C:\n'
        '    scene_preset_id = "sim2real"\n'
        '    # scene_preset_id = "droid_with_box_table"\n'
    )
    out = _override_env_cfg_fields(source, "<test>", scene_preset_id="table")
    assert '    scene_preset_id = "table"\n' in out
    assert '    # scene_preset_id = "droid_with_box_table"\n' in out


def test_no_override_keeps_the_template_defaults():
    code = generate()["env_cfg_code"]
    assert 'scene_preset_id = "sim2real"' in code
    assert 'robot_id: str = "fr3_robotiq_gripper_sysid"' in code


@pytest.mark.parametrize(
    "field,template_line,override",
    [
        (
            "robot_id",
            'robot_id: str = "fr3_robotiq_gripper_sysid"',
            "fr3_robotiq_gripper_soft",
        ),
        (
            "scene_preset_id",
            'scene_preset_id = "sim2real"',
            "droid_with_box_table",
        ),
    ],
)
def test_unapplied_override_raises(tmp_path, field, template_line, override):
    """A silently-unapplied override runs the wrong scene while everything passes.

    The generated package would keep the template's table baseline, invalidating
    every table_height_delta comparison downstream, so codegen must refuse to
    emit it. Both fields are covered because they are substituted separately --
    only ``robot_id`` carries a type annotation.
    """
    fake_domain = tmp_path / "blocks"
    (fake_domain / "template").mkdir(parents=True)
    for name in ("template_env.py", "template_env_cfg.py"):
        (fake_domain / "template" / name).write_text(
            (DOMAIN_PATH / "template" / name).read_text()
        )
    cfg_path = fake_domain / "template" / "template_env_cfg.py"
    original = cfg_path.read_text()
    assert template_line in original, f"template no longer declares {field}"
    cfg_path.write_text(original.replace(template_line, ""))

    with pytest.raises(ValueError, match=f"could not override {field}"):
        generate(domain_path=str(fake_domain), **{field: override})


# --------------------------------------------------------------------------
# PDDL parsing and errors
# --------------------------------------------------------------------------


def test_extract_block_names_handles_multi_type_objects():
    """Only ``block*``-typed objects; the table and gripper are not blocks."""
    assert extract_block_names(SIMPLE_PDDL) == ["block_r", "block_red1", "block_blue"]


def test_extract_block_names_without_objects_section_raises():
    with pytest.raises(ValueError, match=r"\(:objects"):
        extract_block_names("(define (problem p) (:domain blocks-tamp))")


def test_unbalanced_goal_parens_raises():
    truncated = SIMPLE_PDDL[: SIMPLE_PDDL.index("(:goal")] + "(:goal (and (on a b)"
    with pytest.raises(ValueError, match="unbalanced parentheses"):
        extract_goal_block(truncated)


def test_init_predicates_follow_predicate_order():
    """Sampling order matters: a block must be supported before ``on`` runs."""
    names = [p["name"] for p in parse_pddl_init_from_text(SIMPLE_PDDL)]
    ordered = [n for n in names if n in PREDICATE_ORDER]
    ranks = [PREDICATE_ORDER.index(n) for n in ordered]
    assert ranks == sorted(ranks)


def test_goal_negation_is_parsed():
    goal = parse_pddl_goal_from_text("(:goal (and (not (on block_a block_b))))")
    assert goal == [{"name": "on", "args": ["block_a", "block_b"], "negated": True}]


def test_unsupported_goal_predicate_names_itself():
    """Predicates needing IK / contact are not expressible as world-state tensors."""
    with pytest.raises(ValueError, match="grasped"):
        assert_goal_text_supported("(and (grasped block_r))", ["block_r"])


def test_compile_goal_grounds_exists():
    """``exists`` expands to an OR over every object of the quantified type.

    Types are inferred by stripping trailing digits from block names, so
    ``block_red1``/``block_red2`` are both ``block_red``.
    """
    conjuncts = compile_goal_text(
        "(exists (?b - block_red) (on block_blue ?b))",
        ["block_red1", "block_red2", "block_blue"],
    )
    assert len(conjuncts) == 1
    assert "block_red1" in conjuncts[0]
    assert "block_red2" in conjuncts[0]
    assert " | " in conjuncts[0]


def test_compile_goal_grounds_forall():
    conjuncts = compile_goal_text(
        "(forall (?b - block_red) (on_table ?b workspace_table))",
        ["block_red1", "block_red2"],
    )
    assert len(conjuncts) == 1
    assert "block_red1" in conjuncts[0]
    assert "block_red2" in conjuncts[0]
    assert " & " in conjuncts[0]


def test_quantifier_over_an_unknown_type_raises():
    with pytest.raises(ValueError, match="matches no scene blocks"):
        compile_goal_text("(exists (?b - block_green) (on block_blue ?b))", ["block_blue"])


STACK_GOAL = """(and
        (on_table block_w workspace_table)
        (on block_x block_w)
        (clear block_x)
        (ori_of block_w o2)
    )"""


def test_goal_conjunct_labels_name_each_compiled_conjunct_in_order():
    """Label c is the literal behind goal_conj_<c>, so the two lists must line up."""
    blocks = ["block_w", "block_x"]
    labels = goal_conjunct_labels(STACK_GOAL, blocks)
    conjuncts = compile_goal_text(STACK_GOAL, blocks)
    assert labels == [
        "(on_table block_w workspace_table)",
        "(on block_x block_w)",
        "(clear block_x)",
        "(ori_of block_w o2)",
    ]
    assert len(conjuncts) == len(labels)
    for label, expr in zip(labels, conjuncts):
        assert f"'{label[1:].split()[0]}'" in expr


def test_goal_conjunct_labels_keep_quantified_form_and_drop_folded_constants():
    """A quantified conjunct is one goal_conj; a conjunct folding to True has no stat."""
    goal = (
        "(and (exists (?b - block_red) (on block_blue ?b)) "
        "(not (= block_blue block_red1)))"
    )
    blocks = ["block_red1", "block_red2", "block_blue"]
    assert goal_conjunct_labels(goal, blocks) == [
        "(exists (?b - block_red) (on block_blue ?b))"
    ]
    assert len(compile_goal_text(goal, blocks)) == 1


def test_generated_env_carries_conjunct_labels_per_goal():
    """The env names goal_conj_<c> per goal, so eval can label each episode's stats."""
    second_goal = "(and (on block_blue block_red1) (clear block_blue))"
    env_code = generate(goal_texts=["(and (on block_red1 block_blue))", second_goal])["env_code"]
    labels = None
    for node in ast.walk(ast.parse(env_code)):
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "_GOAL_CONJUNCT_LABELS"
        ):
            labels = ast.literal_eval(node.value)
    assert labels == (
        ("(on block_red1 block_blue)",),
        ("(on block_blue block_red1)", "(clear block_blue)"),
    )
    assert "def goal_conjunct_labels(self, pddl_goal: str) -> list[str]:" in env_code


def test_generated_env_logs_settle_gate_terms_per_goal_object_slot():
    """Eval logs what the settle gate saw: one speed slot per block of the longest goal."""
    env_code = generate(
        block_names=["block_red1", "block_blue", "block_green"],
        goal_texts=[
            "(and (on block_red1 block_blue))",
            "(and (on block_red1 block_blue) (on block_blue block_green))",
        ],
    )["env_code"]
    tree = ast.parse(env_code)
    class_consts: dict[str, object] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in ("_GOAL_OBJECTS", "_MAX_GOAL_OBJECTS"):
                class_consts[node.target.id] = ast.literal_eval(node.value)
    assert class_consts["_GOAL_OBJECTS"] == (
        ("block_blue", "block_red1"),
        ("block_blue", "block_green", "block_red1"),
    )
    assert class_consts["_MAX_GOAL_OBJECTS"] == 3

    assert "import goal_settle_terms" in env_code
    assert "scene_settled_and_gripper_open" not in env_code
    for k in range(3):
        assert f'stats["goal_obj_{k}_lin_vel"] = self._goal_obj_lin_vel[:, {k}]' in env_code
        assert f'stats["goal_obj_{k}_ang_vel"] = self._goal_obj_ang_vel[:, {k}]' in env_code
    assert 'stats["goal_obj_3_lin_vel"]' not in env_code
    assert 'stats["goal_finger_pos"] = self._goal_finger_pos' in env_code
    assert 'stats["goal_blocks_settled"] = self._goal_blocks_settled.float()' in env_code
    assert 'stats["goal_gripper_open"] = self._goal_gripper_open.float()' in env_code
    # Success itself still gates on settled blocks and an open gripper together.
    assert "goal_settled = torch.where(on_goal, terms.settled & terms.gripper_open, goal_settled)" in env_code
    assert "self._goal_success = goal_conjuncts.all(dim=-1) & goal_settled" in env_code


def test_task_slug_from_pddl_path_normalizes_hyphens():
    assert task_slug_from_pddl_path("flip-pnp.pddl") == "flip_pnp"
    assert task_slug_from_pddl_path("/a/b/r_on_red.pddl") == "r_on_red"


# --------------------------------------------------------------------------
# Init-predicate helpers moved out of generate_task_from_pddl.py
# --------------------------------------------------------------------------


def test_extract_clear_blocks():
    assert extract_clear_blocks(parse_pddl_init_from_text(SIMPLE_PDDL)) == {"block_r"}


def test_reorder_puts_on_after_its_support():
    """``(on a b)`` may only run once ``b`` itself is supported."""
    predicates = [
        {"name": "on", "args": ["block_top", "block_mid"]},
        {"name": "on", "args": ["block_mid", "block_base"]},
        {"name": "on_table", "args": ["block_base", "workspace_table"]},
    ]
    ordered = reorder_predicates_by_dependencies(predicates)
    on_order = [p["args"][0] for p in ordered if p["name"] == "on"]
    assert on_order == ["block_mid", "block_top"]
    assert ordered[0]["name"] == "on_table"


def test_reorder_drops_on_targeting_a_clear_block():
    predicates = [
        {"name": "on_table", "args": ["block_base", "workspace_table"]},
        {"name": "on", "args": ["block_top", "block_base"]},
    ]
    ordered = reorder_predicates_by_dependencies(predicates, clear_blocks={"block_base"})
    assert [p["name"] for p in ordered] == ["on_table"]


# --------------------------------------------------------------------------
# Version discipline
# --------------------------------------------------------------------------


def test_stale_codegen_version_raises(tmp_path, monkeypatch):
    """Guards the AUTOGEN_PACKAGE_VERSION bump.

    Autogen dirs are gitignored, so a package from an older codegen keeps its own
    scene, goal selection and cube-scale slots while looking exactly like a fresh
    one -- differences that move success rates rather than failing.
    """
    import gymnasium as gym

    from blocksuite.utils.eval_helpers import _load_task_profiles

    package = tmp_path / "stale_autogen_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "profiles.json").write_text(
        json.dumps({"codegen_version": 1, "num_profiles": 8, "profiles": []})
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    task_id = "stale-codegen-test-v0"
    gym.register(
        id=task_id,
        entry_point="stale_autogen_pkg.env:StaleEnv",
        disable_env_checker=True,
    )
    try:
        with pytest.raises(RuntimeError, match="codegen version 1"):
            _load_task_profiles(task_id)
    finally:
        gym.registry.pop(task_id, None)
        sys.modules.pop("stale_autogen_pkg", None)
