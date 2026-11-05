"""Cuboid colour randomization: the invariants that would otherwise fail silently.

This feature family has shipped broken twice with correct-looking output, so every test here pins
something that renders fine when wrong: the exact RGB a name maps to, which blocks are ours, that
two cuboids drawing one colour stay two PDDL objects, and that the goal text follows the pixels.
"""
import re

import numpy as np
import pytest

from blocksuite.domain_randomization import block_colors as bc
from blocksuite.domain_randomization import glyphs

# The real goal of problem_declarative_10_20261006_175602_749976_2647 (oct06_stack3_1000x32).
REAL_GOAL = ("(and (on_table block_red1 workspace_table) (on block_l block_red1) "
             "(on block_orange block_l) (clear block_orange) (ori_of block_l o2))")
# Its real block_specs types.
REAL_SPECS = {
    "block_orange": {"type": "cuboid", "color": (1.0, 0.5, 0.0)},
    "block_purple": {"type": "cuboid", "color": (0.5, 0.0, 1.0)},
    "block_red1":   {"type": "cuboid", "color": (1.0, 0.0, 0.0)},
    "block_yellow": {"type": "cuboid", "color": (1.0, 1.0, 0.0)},
    "block_1": {"type": "usd", "usd_path": "train_blocks_large/usd/1.usda"},
    "block_8": {"type": "usd", "usd_path": "train_blocks_large/usd/8.usda"},
    "block_d": {"type": "usd", "usd_path": "train_blocks_large/usd/D.usda"},
    "block_l": {"type": "usd", "usd_path": "train_blocks_large/usd/L.usda"},
    "block_v": {"type": "usd", "usd_path": "train_blocks_large/usd/V.usda"},
    "block_w": {"type": "usd", "usd_path": "train_blocks_large/usd/W.usda"},
}
# What the dataset's own env packages author. A recoloured red MUST equal an authored red.
AUTHORED_RGB = {"red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0),
                "yellow": (1.0, 1.0, 0.0), "purple": (0.5, 0.0, 1.0), "orange": (1.0, 0.5, 0.0)}


def test_palette_reproduces_the_authored_rgb_exactly():
    """A near-miss red renders cleanly and is wrong. Pin the exact triple."""
    got = {n: bc.hsv_to_rgb(h) for n, h in bc.DATASET_COLORS}
    assert got == AUTHORED_RGB


def test_no_srgb_transform_sneaks_in():
    """Mid-grey is where a gamma would show: linear 0.5 must stay 0.5, not become ~0.73."""
    assert bc.hsv_to_rgb((0.0, 0.0, 0.5)) == (0.5, 0.5, 0.5)


# --------------------------------------------------------------------------- palette parsing
def test_true_means_the_dataset_palette():
    assert bc.parse_palette(True) == bc.DATASET_COLORS


def test_degrees_are_refused_loudly():
    """H 0-360 read as turns lands on a real colour, so it must not be accepted."""
    with pytest.raises(ValueError, match="0-1 FLOATS"):
        bc.parse_palette({"colors": [{"name": "teal", "hsv": [180, 100, 100]}]})


@pytest.mark.parametrize("bad,msg", [
    ([{"name": "red"}], "needs 'hsv'"),
    ([{"name": "", "hsv": [0, 1, 1]}], "missing 'name'"),
    ([{"name": "red2", "hsv": [0, 1, 1]}], "letters only"),
    ([{"name": "red", "hsv": [0, 1, 1], "rgb": [1, 0, 0]}], "unknown key"),
    ([], "non-empty list"),
])
def test_bad_palettes_are_errors(bad, msg):
    with pytest.raises(ValueError, match=msg):
        bc.parse_palette({"colors": bad})


def test_same_name_twice_with_different_hsv_is_an_error():
    with pytest.raises(ValueError, match="one name is one colour"):
        bc.parse_palette({"colors": [{"name": "red", "hsv": [0, 1, 1]},
                                     {"name": "red", "hsv": [0.5, 1, 1]}]})


# --------------------------------------------------------------------------- ownership
def test_colour_owns_exactly_the_cuboids_and_glyphs_own_the_rest():
    """The partition is the spec 'type', never a name test: block_blue is a cuboid, block_b is not."""
    mine = set(bc.cuboid_names(REAL_SPECS))
    assert mine == {"block_orange", "block_purple", "block_red1", "block_yellow"}
    theirs = {n for n in REAL_SPECS if glyphs.authored_glyph(n) is not None}
    assert theirs == {"block_1", "block_8", "block_d", "block_l", "block_v", "block_w"}
    assert not (mine & theirs), "a block may not have two owners"


def test_name_lookalikes_are_not_misowned():
    """block_b is a letter block; block_blue/block_brown are cuboids."""
    assert glyphs.authored_glyph("block_b") == "B"
    for cuboid in ("block_blue", "block_brown", "block_red1", "block_green1"):
        assert glyphs.authored_glyph(cuboid) is None


def test_legacy_block_o1_belongs_to_neither():
    """type=='usd' with no authored glyph: glyphs skip it and so must we."""
    assert glyphs.authored_glyph("block_o1") is None
    assert not bc.is_cuboid({"type": "usd", "usd_path": "x/o1.usda"})


# --------------------------------------------------------------------------- assignment
def test_assignment_is_deterministic_and_order_independent():
    names = ["block_yellow", "block_red1", "block_orange", "block_purple"]
    a = bc.assign_colors(names, np.random.default_rng(7), bc.DATASET_COLORS)
    b = bc.assign_colors(list(reversed(names)), np.random.default_rng(7), bc.DATASET_COLORS)
    assert a == b, "draw must not depend on caller order"


def test_draw_is_with_replacement():
    """Two cuboids may share a colour -- intended, and output_tokens keeps them distinct."""
    one = bc.parse_palette({"colors": [{"name": "teal", "hsv": [0.5, 1, 1]}]})
    m = bc.assign_colors(["block_red1", "block_blue"], np.random.default_rng(0), one)
    assert m == {"block_red1": "teal", "block_blue": "teal"}
    assert len(set(bc.output_tokens(m).values())) == 2, "same colour must stay two objects"


# --------------------------------------------------------------------------- tokens
def test_tokens_follow_the_datagen_namer():
    """distractors.cube_name is the authority: red is ALWAYS numbered, others leave copy 1 bare."""
    m = {"block_red1": "blue", "block_orange": "blue", "block_purple": "green"}
    assert bc.output_tokens(m) == {"block_orange": "block_blue", "block_red1": "block_blue2",
                                   "block_purple": "block_green"}
    red = {"block_blue": "red", "block_green": "red"}
    assert bc.output_tokens(red) == {"block_blue": "block_red1", "block_green": "block_red2"}


def test_token_for_color_is_the_datagen_namer():
    """Delegation, not restatement -- these two must never drift."""
    from blocksuite.tamp.blocks.distractors import cube_name, COLOR_POOL
    for colour in COLOR_POOL:
        for i in range(1, 8):
            assert bc.token_for_color(colour, i) == cube_name(colour, i)


def test_rewrite_follows_the_pixels_on_the_real_goal():
    """The whole point: recolour red->cyan and orange->green and the goal must say so."""
    m = {"block_red1": "cyan", "block_orange": "green"}
    out = bc.rewrite_tokens(REAL_GOAL, m)
    assert "block_red1" not in out and "block_orange" not in out
    assert "(on block_l block_cyan)" in out
    assert "(on block_green block_l)" in out
    assert "block_l" in out and "workspace_table" in out and "o2" in out, "non-cuboids untouched"


def test_rewrite_is_simultaneous_not_chained():
    """red->blue then blue->green must not leave the first block green."""
    m = {"block_red1": "blue", "block_blue": "green"}
    out = bc.rewrite_tokens("(on block_red1 block_blue)", m)
    assert out == "(on block_blue block_green)"


def test_glyph_and_colour_rewrites_do_not_corrupt_each_other():
    """Both flags on: each pass must leave the other's tokens alone."""
    text = "(on block_l block_red1)"
    after_colour = bc.rewrite_tokens(text, {"block_red1": "cyan"})
    after_both = glyphs.rewrite_tokens(after_colour, {"block_l": "Q"})
    assert after_both == "(on block_q block_cyan)"


# --------------------------------------------------------------------------- record
def test_record_is_keyed_by_goal_token_and_points_back():
    """INVERSE: the key is what pddl_goal says; the value says which object and what colour."""
    rec = bc.block_randomization_record({"block_red1": "orange"}, bc.DATASET_COLORS)
    assert rec == {"block_orange": {
        "from": "block_red1",
        "pose_column": "observation.state.object_pose.block_red1",
        "color": "orange", "hsv": [1 / 12, 1.0, 1.0], "rgb": [1.0, 0.5, 0.0]}}


def test_record_disambiguates_a_token_that_collides_with_an_authored_name():
    """block_orange becomes red (token block_red1) while block_red1 becomes blue.

    Forward keying made the goal's block_red1 and object_pose.block_red1 silently different
    objects. Keyed by token, one lookup gives the right one.
    """
    rec = bc.block_randomization_record({"block_orange": "red", "block_red1": "blue"},
                                        bc.DATASET_COLORS)
    assert rec["block_red1"]["from"] == "block_orange"
    assert rec["block_blue"]["from"] == "block_red1"


def test_record_json_is_stable():
    m = {"block_red1": "blue", "block_orange": "green"}
    a = bc.cuboid_color_map_json(m, bc.DATASET_COLORS)
    b = bc.cuboid_color_map_json(dict(reversed(list(m.items()))), bc.DATASET_COLORS)
    assert a == b


def test_prim_path_matches_the_glyph_modules():
    """Duplicated on purpose (no coupling); this is the guard against the two drifting apart."""
    for i in (0, 7, 31):
        assert bc.block_prim_path(i, "block_red1") == glyphs.block_prim_path(i, "block_red1")
