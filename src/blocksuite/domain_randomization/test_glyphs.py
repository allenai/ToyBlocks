"""    python -m pytest src/blocksuite/domain_randomization/test_glyphs.py

Contract for the block-glyph augmentation (`randomize_task_letters`).

The expensive failure here is silent: a dataset whose pixels say one glyph and whose
`pddl_goal` / `task` string says another trains a language-conditioned policy on a lie, and
nothing downstream can detect it. So the atlas math is pinned against the shipped assets, and
the rewriting is pinned against the token shapes that actually occur in the data.
"""
import json
import string
from pathlib import Path

import numpy as np
import pytest

from blocksuite.domain_randomization import glyphs as G
# Imported at MODULE scope on purpose. test_build_camera_fov.py replaces sys.modules["blocksuite"]
# with a stub package inside its loader, so importing this from a test BODY picks up the stub and
# fails depending on file order. Module-level imports run at collection, before any test body.
from blocksuite.tamp.blocks import glyph_blocks as GB

ASSETS = (Path(__file__).resolve().parents[3]
          / "assets" / "objects" / "train_blocks_large" / "usd" / "dataset_description.json")


def test_atlas_covers_all_36_shipped_blocks():
    assert len(G.ATLAS_GLYPHS) == 36
    assert set(G.ATLAS_GLYPHS) == set(string.ascii_uppercase) | {str(n) for n in range(1, 11)}


def test_all_36_glyphs_are_assignable_including_10():
    """"10" has an atlas tile and a USD, so it renders; the only thing that ever excluded it was
    name parsing, and glyph_blocks now matches block_10 explicitly.

    It is the one glyph whose token is TWO characters. Nothing here may assume len(g) == 1.
    """
    assert len(G.GLYPHS) == 36 and "10" in G.GLYPHS
    assert set(G.GLYPHS) == set(G.ATLAS_GLYPHS)
    assert "0" not in G.GLYPHS, "there is no 0 tile"
    for g in G.GLYPHS:
        assert GB._GLYPH_BLOCK_PATTERN.match(G.token_for(g)), f"{g} makes an unparseable name"


def test_block_10_round_trips_through_name_parsing():
    """block_10 must read as glyph 10, NOT as glyph 1 with a stray 0."""
    assert G.authored_glyph("block_10") == "10"
    assert G.authored_glyph("block_10_2") == "10"      # glyph 10, copy 2
    assert G.authored_glyph("block_1") == "1"
    assert G.token_for("10") == "block_10"
    assert GB.block_type("10") == GB.DIGIT_TYPE


def test_task_generators_still_stop_at_9():
    """Widening the PARSER must not make the generators start authoring 10-blocks."""
    assert GB.DIGITS == "123456789"


@pytest.mark.skipif(not ASSETS.is_file(), reason="train_blocks_large assets not present")
def test_anchors_match_the_shipped_dataset_description():
    """The rule must reproduce the on-disk assets exactly -- one wrong anchor is one wrong glyph."""
    shipped = {e["name"]: tuple(e["anchor"]) for e in json.loads(ASSETS.read_text())}
    assert set(shipped) == set(G.ATLAS_GLYPHS), "asset set and atlas vocabulary disagree"
    for name, anchor in shipped.items():
        assert G.anchor_for(name) == anchor, f"{name}: rule {G.anchor_for(name)} vs asset {anchor}"


def test_uv_offset_matches_the_generator_formula():
    """Mirrors proc_blocks/simplified/alphabet_large.build_block: du=ax/16, dv=-ay/16."""
    for g in G.ATLAS_GLYPHS:
        ax, ay = G.anchor_for(g)
        assert G.uv_offset(g) == pytest.approx((ax / 16, -ay / 16))


def test_v_offset_is_negative_or_zero():
    """Atlas rows run top-down, UV v runs bottom-up. A positive v would mirror the atlas."""
    assert all(G.uv_offset(g)[1] <= 0 for g in G.ATLAS_GLYPHS)
    assert G.uv_offset("A") == (0.0, 0.0)          # A is the template's own quadrant


def test_first_glyph_of_each_row_wraps_to_column_zero():
    for g, expect_col in (("A", 0), ("F", 0), ("K", 0), ("P", 0), ("U", 0), ("Z", 0),
                          ("E", 12), ("J", 12), ("10", 0)):
        assert G.anchor_for(g)[0] == expect_col, g


def test_assignment_is_without_replacement():
    rng = np.random.default_rng(0)
    names = [f"block_{c}" for c in "abcdefghij"]
    m = G.assign_glyphs(names, rng)
    assert len(m) == len(names)
    assert len(set(m.values())) == len(names), "two blocks drew the same glyph -> ambiguous goal"


def test_assignment_is_deterministic_for_a_given_rng():
    names = ["block_a", "block_i", "block_2"]
    a = G.assign_glyphs(names, np.random.default_rng(7))
    b = G.assign_glyphs(names, np.random.default_rng(7))
    assert a == b, "same seed must reproduce -- shards would otherwise disagree"
    c = G.assign_glyphs(names, np.random.default_rng(8))
    assert a != c or len(names) == 0


def test_more_blocks_than_glyphs_degrades_instead_of_crashing():
    rng = np.random.default_rng(1)
    names = [f"block_{i}" for i in range(50)]
    m = G.assign_glyphs(names, rng)
    assert len(m) == 50, "a fleet run must not die because a scene has too many blocks"


def test_rewrite_is_simultaneous_not_chained():
    """a->b and b->c applied sequentially would leave the first block as c."""
    goal = "(on block_a block_b)"
    out = G.rewrite_tokens(goal, {"block_a": "B", "block_b": "C"})
    assert out == "(on block_b block_c)"


def test_rewrite_does_not_confuse_block_1_with_block_10():
    out = G.rewrite_tokens("(on block_1 block_10)", {"block_1": "X", "block_10": "Y"})
    assert out == "(on block_x block_y)"


def test_rewrite_leaves_non_blocks_alone():
    txt = "(on_table block_2 workspace_table)"
    assert G.rewrite_tokens(txt, {"block_2": "Q"}) == "(on_table block_q workspace_table)"


def test_rewrite_handles_a_real_pddl_goal():
    goal = ("(and\n        (on block_a block_i)\n        (on block_i block_2)\n"
            "        (on_table block_2 workspace_table)\n        (ori_of block_a o2)\n"
            "        (ori_of block_i o2)\n        (ori_of block_2 o2)\n    )")
    out = G.rewrite_tokens(goal, {"block_a": "X", "block_i": "7", "block_2": "M"})
    assert "(on block_x block_7)" in out and "(on block_7 block_m)" in out
    assert "(on_table block_m workspace_table)" in out
    assert "o2)" in out, "orientation tokens must survive untouched"
    assert "block_a" not in out and "block_i" not in out


def test_rewrite_handles_the_lerobot_task_string_shape():
    """Real shape from meta/tasks.parquet on sep3_ai2_stack_PARTIAL5k."""
    t = "isolate ['robot', 'block_2', 'workspace_table']"
    assert G.rewrite_tokens(t, {"block_2": "9"}) == "isolate ['robot', 'block_9', 'workspace_table']"


def test_glyph_map_json_is_stable():
    """FORWARD {original: glyph}, unchanged -- shipped datasets carry exactly this shape."""
    m = {"block_i": "Q", "block_a": "X"}
    assert G.glyph_map_json(m) == '{"block_a":"X","block_i":"Q"}'


def test_unknown_glyph_is_rejected_loudly():
    with pytest.raises(ValueError):
        G.anchor_for("!")


# ---------------------------------------------------------------------------------------------
# Sampling / wave behaviour. The USD half needs Isaac, so only the pure parts are covered here;
# the shader write is validated by an actual render in P6.
# ---------------------------------------------------------------------------------------------

BLOCKS = ["block_a", "block_i", "block_2"]


def _wave(episode_indices):
    return [{"episode_index": e} for e in episode_indices]


def test_same_episode_gets_same_glyphs_regardless_of_slot_or_wave():
    """A shard rendering one episode alone must reproduce the full run exactly."""
    full = G.sample_wave_glyphs(_wave([10, 11, 12]), [0, 1, 2], BLOCKS, base_seed=42)
    shard = G.sample_wave_glyphs(_wave([11]), [0], BLOCKS, base_seed=42)
    assert shard[0] == full[1], "episode 11 drew different glyphs when rendered in a shard"


def test_different_episodes_get_different_assignments():
    a = G.sample_wave_glyphs(_wave([0, 1]), [0, 1], BLOCKS, base_seed=42)
    assert a[0] != a[1] or len(BLOCKS) == 0


def test_base_seed_changes_the_draw():
    a = G.sample_wave_glyphs(_wave([5]), [0], BLOCKS, base_seed=1)
    b = G.sample_wave_glyphs(_wave([5]), [0], BLOCKS, base_seed=2)
    assert a[0] != b[0]


def test_every_env_assignment_is_collision_free():
    w = G.sample_wave_glyphs(_wave(range(32)), list(range(32)), BLOCKS, base_seed=3)
    for slot, m in w.items():
        assert len(set(m.values())) == len(BLOCKS), f"env {slot} drew a duplicate glyph"


def test_glyph_salt_decouples_from_other_per_episode_draws():
    """Must not reuse the raw episode_rng stream that cameras/materials already consume."""
    from blocksuite.domain_randomization.seeds import episode_rng
    import numpy as np
    plain = G.assign_glyphs(BLOCKS, episode_rng(42, 7))
    salted = G.sample_wave_glyphs(_wave([7]), [0], BLOCKS, base_seed=42)[0]
    assert plain != salted, "glyph draw is not salted apart from the unsalted episode stream"


def test_rewrite_episode_text_keeps_goal_and_task_consistent():
    goal = "(on block_a block_i)"
    task = "isolate ['robot', 'block_a', 'workspace_table']"
    m = {"block_a": "X", "block_i": "Q"}
    g2, t2 = G.rewrite_episode_text(goal, task, m)
    assert g2 == "(on block_x block_q)"
    assert t2 == "isolate ['robot', 'block_x', 'workspace_table']"


def test_rewrite_episode_text_tolerates_missing_strings():
    assert G.rewrite_episode_text("", None, {"block_a": "X"}) == ("", "")


# ---------------------------------------------------------------------------------------------
# Interpreter wiring: sampling must be pure (no USD) and must not re-draw between the episode
# record and the shader write.
# ---------------------------------------------------------------------------------------------

from types import SimpleNamespace  # noqa: E402

from blocksuite.domain_randomization import interpreter as I  # noqa: E402


class _AllSupported(dict):
    """Every block resolves to the calibrated asset library.

    The sampler now refuses any block whose asset is not train_blocks_large, and treats an
    unknown path as unsupported (fail safe: a wrong letter renders cleanly and is undetectable,
    a skipped swap is loud). Tests about sampling supply this so they are not implicitly also
    testing the library guard -- that has its own tests.
    """

    def get(self, key, default=None):          # noqa: D102 - dict override
        return "train_blocks_large/usd/X.usda"


def _ir(task_letters=True, seed=42):
    return SimpleNamespace(task_letters=task_letters, rseed_base=seed, rseed=seed,
                           glyphs_sampled={}, glyphs_applied={},
                           glyph_assets=_AllSupported())


def _unit(ep, blocks=("block_a", "block_i", "block_2")):
    return {"episode_index": ep, "objects": {b: None for b in blocks} | {"workspace_table": None}}


def test_sampling_is_pure_and_needs_no_usd():
    """Runs on a CPU box: if this ever imports pxr the record loop cannot call it."""
    out = I.sample_wave_task_letters(_ir(), [_unit(0), _unit(1)])
    assert set(out) == {0, 1}
    assert all(len(m) == 3 for m in out.values())


def test_sampling_excludes_non_block_objects():
    out = I.sample_wave_task_letters(_ir(), [_unit(0)])
    assert "workspace_table" not in out[0]


def test_flag_off_samples_nothing():
    assert I.sample_wave_task_letters(_ir(task_letters=False), [_unit(0)]) == {}


def test_sample_is_cached_on_ir_so_the_applier_cannot_redraw():
    """The record loop and the shader write must see the SAME draw."""
    ir = _ir()
    first = I.sample_wave_task_letters(ir, [_unit(5)])
    assert ir.glyphs_sampled == first and first[0]


def test_each_env_gets_only_its_own_blocks():
    """Envs with different block sets each get a collision-free assignment over THEIR blocks.

    Was test_blocks_are_unioned_across_the_wave, which pinned a bug: the caller unioned
    ``objects`` across the wave, so an env was handed glyphs for blocks it does not have.
    """
    wave = [_unit(0, ("block_a", "block_b")), _unit(1, ("block_c",))]
    out = I.sample_wave_task_letters(_ir(), wave)
    for slot, m in out.items():
        assert len(set(m.values())) == len(m)
    assert set(out[0]) == {"block_a", "block_b"}
    assert set(out[1]) == {"block_c"}


def test_an_episodes_glyphs_do_not_depend_on_who_shares_its_wave():
    """SHARD INVARIANCE. The whole module is keyed on episode_index so that a shard rendering
    episode 300 alone reproduces the full run. Unioning the wave's blocks broke that: the draw
    size became a function of the neighbours, and rng.choice(35, size=n, replace=False) does not
    agree with itself across n. Reproduced 2026-10-06 -- alone {block_a: '4'} vs beside a
    4-block episode {block_a: 'R'}.
    """
    mine = ("block_a", "block_i", "block_2")
    alone = I.sample_wave_task_letters(_ir(), [_unit(300, mine)])[0]
    with_same = I.sample_wave_task_letters(_ir(), [_unit(300, mine), _unit(7, mine)])[0]
    with_bigger = I.sample_wave_task_letters(
        _ir(), [_unit(300, mine), _unit(7, mine + ("block_q",))])[0]
    assert alone == with_same == with_bigger
    assert set(alone) == set(mine), "an episode must never be assigned a block it does not have"


def test_no_blocks_yields_empty_without_error():
    assert I.sample_wave_task_letters(_ir(), [{"episode_index": 0, "objects": {}}]) == {}


def test_texture_translate_sign_matches_the_mdl_convention():
    """OmniPBR adds the translation to the UV, so the shader offset equals the baked mesh offset.

    Derived from isaacsim/kit/mdl/core/Base/OmniPBR_ClearCoat.mdl:544, where the UV transform is
    rotation_translation_scale(..., translation: texture_translate) fed to
    base::transform_coordinate -- translation applied AFTER scale/rotation, i.e. uv + translate.
    A -1 here would mirror the atlas and pick a real-but-wrong glyph, which renders cleanly and
    is therefore invisible without reading the letters.
    """
    assert G.TEXTURE_TRANSLATE_SIGN == 1.0
    for g in ("A", "M", "Z", "1", "10"):
        assert G.texture_translate(g, "A") == G.uv_offset(g)   # "A" is (0,0)


def test_block_prim_path_matches_the_generated_env_layout():
    """Generated env_*/env.py spawns objects at prim_path=f"/World/envs/env_.*/{name}".

    Verified against the checked-in env for
    sep3_ai2led_random_pnp_o2/problem_letter_color_10_20260903_174710_638357_1654.
    A mismatch here makes the shader lookup find nothing, which is SILENT: the glyph_map column
    and the rewritten goal/task are all still written, and only the rendered letters are stale.
    """
    assert G.block_prim_path(0, "block_a") == "/World/envs/env_0/block_a"
    assert G.block_prim_path(31, "block_10") == "/World/envs/env_31/block_10"


def test_reconcile_reports_blocks_that_were_never_applied():
    """Text is rewritten from the SAMPLED map; anything not applied means pixels disagree."""
    sampled = {0: {"block_a": "X", "block_i": "Q"}, 1: {"block_a": "M"}}
    applied = {0: {"block_a": "X"}, 1: {"block_a": "M"}}
    assert G.reconcile(sampled, applied) == {0: ["block_i"]}


def test_reconcile_is_empty_when_everything_applied():
    m = {0: {"block_a": "X"}}
    assert G.reconcile(m, m) == {}


def test_reconcile_tolerates_missing_env_entries():
    assert G.reconcile({0: {"block_a": "X"}}, {}) == {0: ["block_a"]}


# --------------------------------------------------------------------------------------------
# texture_translate is a DELTA from the block's authored glyph, not an absolute offset.
#
# The generator bakes the authored anchor into the mesh UVs and OmniPBR ADDS texture_translate,
# so the rendered tile is `authored + written`. Writing the absolute offset produced a
# real-but-wrong letter on every block whose mesh is not authored at the atlas origin -- and it
# renders cleanly, so nothing downstream could tell. These cases are the ones actually observed
# on a render before the fix.
# --------------------------------------------------------------------------------------------
GRID = 16


def _tile(authored: str, target: str):
    """Atlas tile the renderer lands on, given the mesh's baked UVs plus our written translate."""
    ax, ay = G.anchor_for(authored)
    du, dv = G.texture_translate(target, authored)
    return (ax + round(du * GRID), ay - round(dv * GRID))


def test_authored_glyph_parses_block_names():
    assert G.authored_glyph("block_a") == "A"
    assert G.authored_glyph("block_2") == "2"
    assert G.authored_glyph("block_a_2") == "A"      # second copy of the same glyph
    assert G.authored_glyph("block_red1") is None    # colour cuboid: not a glyph block
    assert G.authored_glyph("workspace_table") is None


def test_delta_lands_exactly_on_the_target_tile():
    for authored in ("A", "D", "I", "2", "Z", "9"):
        for target in ("A", "B", "6", "S", "1", "9"):
            assert _tile(authored, target) == G.anchor_for(target), (
                f"{authored}->{target} landed on {_tile(authored, target)}, "
                f"want {G.anchor_for(target)}"
            )


def test_asking_for_the_authored_glyph_is_a_no_op():
    for g in ("A", "D", "I", "2", "9"):
        assert G.texture_translate(g, g) == (0.0, 0.0)


def test_absolute_offset_reproduces_the_three_observed_defects():
    """Pins WHY this was wrong, using the exact cases seen in the render."""
    def legacy_tile(authored, target):
        ax, ay = G.anchor_for(authored)
        du, dv = G.uv_offset(target)                  # what the old code wrote: absolute
        return (ax + round(du * GRID), ay - round(dv * GRID))

    # block_d ("D") asked for "6" and rendered "9"
    assert legacy_tile("D", "6") == G.anchor_for("9")
    # block_i ("I") asked for "A"; "A" is the origin so the shift was zero and it stayed "I"
    assert legacy_tile("I", "A") == G.anchor_for("I")
    # block_2 ("2") asked for "I" and landed on column 15 -- the atlas's unused 16th column
    assert legacy_tile("2", "I")[0] == 15
    # block_a ("A") asked for "B" and looked CORRECT: "A" is at the origin, so delta == absolute
    assert legacy_tile("A", "B") == G.anchor_for("B") == _tile("A", "B")


def test_every_authored_target_pair_stays_inside_the_atlas():
    for authored in G.ATLAS_GLYPHS:
        for target in G.GLYPHS:
            col, row = _tile(authored, target)
            # DERIVED, not fitted: a quadrant is QUAD_W x QUAD_H, so its far edge must land
            # inside the GRID. The old literal `row <= 13` happened to hold only because the
            # 36th glyph ("10", anchor row 14, rows 14-15) was excluded from GLYPHS.
            assert 0 <= col and col + G.QUAD_W - 1 <= G.GRID_W - 1, f"{authored}->{target} col"
            assert 0 <= row and row + G.QUAD_H - 1 <= G.GRID_H - 1, f"{authored}->{target} row"


def test_sampler_ignores_colour_cuboids_in_mixed_scenes():
    """Mixed scenes pair glyph blocks with colour cuboids that have no decal to swap.

    Selecting them would rename them in pddl_goal and leave the pixels untouched: a text/pixel
    mismatch by construction, on real datasets (aug14_2i_pnp, sep11_ai2_kp200_10blocks).
    """
    wave = [{"episode_index": 0,
             "objects": {"block_b": {}, "block_h": {}, "block_r": {},
                         "block_cyan2": {}, "block_yellow2": {}, "block_red1": {},
                         "block_green": {}, "block_o1": {}, "workspace_table": {}}}]
    out = I.sample_wave_task_letters(_ir(), wave)
    assigned = set(out[0])
    assert assigned == {"block_b", "block_h", "block_r"}, (
        f"only glyph blocks are assignable; got {sorted(assigned)}"
    )


def test_rewrite_handles_copy_indexed_block_names():
    """`block_a_2` is the SECOND "a" block (glyph_blocks.block_name(glyph, 2)).

    The token regex used to stop at the underscore, so a second-copy block was assigned a glyph,
    rendered with it, and left under its OLD name in pddl_goal -- a text/pixel mismatch.
    """
    m = {"block_a": "X", "block_a_2": "Y"}
    out = G.rewrite_tokens("(on block_a_2 block_a)", m)
    assert out == "(on block_y block_x)", out


def test_rewrite_does_not_match_a_prefix_of_a_longer_name():
    """Mixed scenes put block_r next to block_red1 and block_b next to block_blue."""
    m = {"block_b": "X", "block_r": "Z", "block_1": "Q"}
    out = G.rewrite_tokens(
        "(and (on block_r block_red1) (on block_b block_blue) (near block_cyan2 block_10))", m)
    assert "block_red1" in out and "block_blue" in out and "block_cyan2" in out, out
    assert "block_10" in out, "block_1 must not match inside block_10"
    assert "block_z" in out and "block_x" in out, out


def test_every_assignable_block_name_is_a_rewritable_token():
    """The two filters must agree: anything the sampler can assign, the rewriter must rename.

    A block the sampler assigns but the rewriter cannot see gets swapped in the image while the
    text keeps its old name -- exactly the failure this whole module guards against.
    """
    for name in ("block_a", "block_2", "block_a_2", "block_z_10"):
        assert G.authored_glyph(name) is not None, f"{name} should be assignable"
        assert G.rewrite_tokens(f"(on {name} t)", {name: "Q"}) == "(on block_q t)", name


def test_sampling_differs_across_problems_but_repeats_within_one():
    """prob_salt must reach the glyph draw.

    It seeded on rseed_base (which EXCLUDES prob_salt), so episode 0 of every problem drew the
    same letters: a 167-problem job carried ~30 distinct assignments instead of ~5000, and the
    augmentation was mostly duplicate with nothing downstream looking wrong.
    """
    from blocksuite.domain_randomization.seeds import _problem_seed_salt
    wave = [{"episode_index": 0, "objects": {b: {} for b in BLOCKS}}]

    def draw(problem):
        ir = SimpleNamespace(task_letters=True, rseed_base=0,
                             rseed=_problem_seed_salt({"out_dir": f"/x/{problem}"}),
                             glyphs_sampled={}, glyph_assets=_AllSupported())
        return tuple(sorted(I.sample_wave_task_letters(ir, wave)[0].items()))

    names = [f"problem_letter_color_10_2026090{i}_00000{i}_111{i}" for i in range(5)]
    draws = [draw(n) for n in names]
    assert len(set(draws)) == len(draws), "same episode_index must differ across problems"
    assert draw(names[0]) == draws[0], "and must be reproducible for a given problem"


# --------------------------------------------------------------------------------------------
# The atlas maths describe ONE asset library. On any other, the same UV shift selects a
# different, real-looking glyph -- observed on aug14_2i_pnp, whose blocks come from
# `train_blocks`: authored B/H/R asked for U/6/A and rendered M and friends. Metadata agreed
# throughout; only the pixels disagreed.
# --------------------------------------------------------------------------------------------
def test_library_supported_accepts_only_the_calibrated_library():
    assert G.library_supported("train_blocks_large/usd/A.usda")
    assert not G.library_supported("train_blocks/usd/B.usd")
    assert not G.library_supported("train_blocks_legacy_match/usd/B.usd")


def test_unknown_asset_path_is_treated_as_unsupported():
    """Fail safe: a confidently wrong letter is worse than no swap, because it looks correct."""
    assert not G.library_supported(None)
    assert not G.library_supported("")


def _env_with(tmp_path, specs):
    env = tmp_path / "env.py"
    env.write_text("def _setup_scene_objects(self):\n"
                   f"    block_names = {sorted(specs)!r}\n"
                   f"    block_specs = {specs!r}\n")
    return str(env)


def test_sampler_skips_blocks_from_an_unsupported_library(tmp_path):
    specs = {"block_b": {"type": "usd", "usd_path": "train_blocks/usd/B.usd"},
             "block_a": {"type": "usd", "usd_path": "train_blocks_large/usd/A.usda"}}
    envf = _env_with(tmp_path, specs)
    ir = SimpleNamespace(task_letters=True, rseed_base=0, rseed=5, glyphs_sampled={},
                         job={"env_file": envf})
    wave = [{"episode_index": 0, "objects": {"block_b": {}, "block_a": {}}}]
    out = I.sample_wave_task_letters(ir, wave)
    assert sorted(out[0]) == ["block_a"], f"only the supported-library block is assignable: {out}"


def test_sampler_assigns_nothing_when_no_block_is_supported(tmp_path):
    specs = {"block_b": {"type": "usd", "usd_path": "train_blocks/usd/B.usd"},
             "block_h": {"type": "usd", "usd_path": "train_blocks/usd/H.usd"}}
    ir = SimpleNamespace(task_letters=True, rseed_base=0, rseed=5, glyphs_sampled={},
                         job={"env_file": _env_with(tmp_path, specs)})
    wave = [{"episode_index": 0, "objects": {b: {} for b in specs}}]
    assert I.sample_wave_task_letters(ir, wave) == {}, \
        "an unsupported problem must swap nothing -- not text, not pixels"


# --------------------------------------------------------------- copy-index renumbering
def test_two_blocks_drawing_one_glyph_stay_distinct_objects():
    """``token_for`` drops the copy index, so without renumbering two blocks that draw the same
    glyph both become ``block_x`` and ``(on block_z block_a)`` rewrites to ``(on block_x block_x)``
    -- one name for two objects, an unsatisfiable goal. That collapse is the ONLY reason the draw
    had to be without-replacement, and it is what blocked per-character glyph sets.
    """
    goal = "(and (on block_z block_a) (on block_z_2 block_z))"
    out = G.rewrite_tokens(goal, {"block_z": "X", "block_z_2": "X", "block_a": "M"})
    assert out == "(and (on block_x block_m) (on block_x_2 block_x))"
    assert "block_x block_x)" not in out


def test_renumbering_matches_block_name():
    """The numbering convention is glyph_blocks.block_name's: first is bare, rest get _2.._k."""
    toks = G.output_tokens({"block_a": "X", "block_b": "X", "block_c": "X"})
    assert toks == {"block_a": "block_x", "block_b": "block_x_2", "block_c": "block_x_3"}
    for i, name in enumerate(sorted(toks, key=G._copy_sort_key), start=1):
        assert toks[name] == GB.block_name("x", i)


def test_renumbering_is_stable_and_order_independent():
    m = {"block_c": "X", "block_a": "X", "block_b": "X"}
    assert G.output_tokens(m) == G.output_tokens(dict(reversed(list(m.items()))))


def test_copy_indices_sort_naturally_not_lexically():
    """block_a_10 must follow block_a_9, not block_a_1 -- a plain string sort gets this wrong."""
    m = {"block_a": "X"} | {f"block_a_{i}": "X" for i in range(2, 12)}
    toks = G.output_tokens(m)
    assert toks["block_a_9"] == "block_x_9"
    assert toks["block_a_10"] == "block_x_10"


def test_distinct_glyphs_are_unnumbered():
    """Renumbering must not disturb the common case: every block drew a different glyph."""
    assert G.rewrite_tokens("(on block_a block_b)", {"block_a": "Q", "block_b": "M"}) \
        == "(on block_q block_m)"


# ------------------------------------------------------------------ glyph set specs
import yaml as _yaml


def _spec(y):
    return G.resolve_glyph_sets(_yaml.safe_load(y))


def test_wildcard_and_bare_forms_cover_everything():
    for form in ("'*'", "[A-Z, 1-9]"):
        assert _spec(form)["A"] == frozenset(G.GLYPHS) or len(_spec(form)["A"]) == 35
    assert G.resolve_glyph_sets(True)["A"] == frozenset(G.GLYPHS)


def test_a_list_or_string_applies_to_every_character():
    s = _spec("[A-C]")
    assert s["A"] == s["7"] == s["Q"] == frozenset("ABC")


def test_explicit_list_of_characters():
    assert _spec("[A, B, C, D]")["Q"] == frozenset("ABCD")


def test_ranges_and_singles_mix_in_one_list():
    assert _spec("[A-C, X-Z, 7]")["A"] == frozenset("ABCXYZ7")


def test_digit_keys_and_entries_survive_yaml_int_coercion():
    """YAML hands back ints for bare digits -- `7:` is 7 and `[1, 2]` is [1, 2]."""
    s = _spec("{7: [1, 2, 3]}")
    assert s["7"] == frozenset("123")


def test_a_mapping_is_an_allowlist_unlisted_characters_are_left_alone():
    s = _spec("{A: [X-Z], 7: [1-3]}")
    assert s["A"] == frozenset("XYZ")
    assert s["7"] == frozenset("123")
    assert s["Q"] == frozenset("Q"), "an unlisted character must keep its own glyph"


def test_the_star_key_is_the_catch_all_and_entries_override_it():
    s = _spec('{"*": [A-D], F: [F], 7: [1-9]}')
    assert s["Q"] == frozenset("ABCD")
    assert s["F"] == frozenset("F")
    assert s["7"] == frozenset("123456789")


def test_omitting_a_character_equals_pinning_it():
    """`F: [F]` and leaving F out are the same thing when there is no catch-all."""
    assert _spec("{A: [A-Z], F: [F]}")["F"] == _spec("{A: [A-Z]}")["F"] == frozenset("F")


def test_digit_ranges_are_numeric_so_1_10_spans_ten_glyphs():
    """[1-10] is what anyone writing it means -- and "10" being a glyph makes it expressible."""
    assert _spec("[1-10]")["A"] == frozenset(str(n) for n in range(1, 11))
    assert _spec("[8-10]")["A"] == frozenset(("8", "9", "10"))
    assert _spec("[10]")["A"] == frozenset(("10",))


@pytest.mark.parametrize("bad,needle", [
    ("[0]", "there is no 0 glyph"),
    ("[0-5]", "there is no 0 glyph"),
    ("[10-1]", "runs backwards"),
    ("[A-10]", "mixes letters and digits"),
    ("[A-9]", "mixes letters and digits"),
    ("[Z-A]", "runs backwards"),
    ("[]", "resolves to no glyphs"),
    ("{0: [A-Z]}", "not an assignable glyph to key on"),
])
def test_bad_specs_are_rejected_with_a_specific_message(bad, needle):
    with pytest.raises(ValueError) as e:
        _spec(bad)
    assert needle in str(e.value)


def test_constrained_draws_are_independent_per_block():
    """Three A-blocks each draw separately, so they MAY collide -- and must stay distinct objects.

    This is the whole point of the feature: restrict a character to a few targets and duplicates
    become routine. output_tokens renumbers them.
    """
    sets = G.resolve_glyph_sets(_yaml.safe_load("[A-C]"))
    rng = np.random.default_rng(0)
    m = G.assign_glyphs(["block_a", "block_a_2", "block_a_3"], rng, sets)
    assert set(m.values()) <= set("ABC")
    toks = G.output_tokens(m)
    assert len(set(toks.values())) == 3, "three blocks must remain three distinct objects"


def test_the_unconstrained_draw_is_byte_identical_to_before():
    """charsets=None must keep the historical without-replacement draw, or every existing
    glyph dataset stops reproducing."""
    names = ["block_a", "block_i", "block_2"]
    a = G.assign_glyphs(names, np.random.default_rng(7))
    b = G.assign_glyphs(names, np.random.default_rng(7), None)
    assert a == b
    assert len(set(a.values())) == 3
