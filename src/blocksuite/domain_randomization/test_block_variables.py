"""The unified goal-name -> object map, over both block families."""
import json

import pytest

from blocksuite.domain_randomization import block_colors as bc
from blocksuite.domain_randomization.block_variables import (
    POSE_PREFIX, BlockIdentity, block_variable_map, block_variable_map_json, identity_assignment,
    parse_block_variable_map, parse_glyph_map, rendered_text, source_goal)

# Episode 0 of oct30_stack3_train_rg_readable_1000x32rs_glyphcolor_symm_N1: the recorded forward
# draws, and its recorded (rendered) goal next to the source goal it was rewritten from.
EP0_GLYPHS = {"block_1": "F", "block_3": "Y", "block_5": "L", "block_c": "P", "block_d": "1",
              "block_e": "G", "block_f": "J", "block_g": "3", "block_j": "K", "block_k": "Y",
              "block_l": "E", "block_m": "D", "block_p": "P", "block_q": "M", "block_u": "P",
              "block_v": "C", "block_x": "S", "block_y": "D"}
EP0_COLORS = {"block_blue": "green", "block_green": "blue", "block_red1": "white",
              "block_white": "blue"}
EP0_PALETTE = bc.DATASET_COLORS + (("white", (0.0, 0.0, 1.0)),)
EP0_RENDERED = ("(and (on_table block_green workspace_table) (on block_l block_green) "
                "(on block_f block_l) (clear block_f) (ori_of block_l o2) (ori_of block_f o2))")
EP0_SOURCE = ("(and (on_table block_blue workspace_table) (on block_5 block_blue) "
              "(on block_1 block_5) (clear block_1) (ori_of block_5 o2) (ori_of block_1 o2))")


def _ep0() -> BlockIdentity:
    raw = block_variable_map_json(EP0_GLYPHS, EP0_COLORS, EP0_PALETTE)
    return identity_assignment(parse_block_variable_map(raw), EP0_GLYPHS)


def test_covers_both_families_keyed_by_the_rewritten_name():
    m = block_variable_map({"block_a": "L", "block_v": "Q"},
                           {"block_orange": "red"}, bc.DATASET_COLORS)
    assert m["block_l"]["from"] == "block_a" and m["block_l"]["kind"] == "glyph"
    assert m["block_q"]["from"] == "block_v"
    assert m["block_red1"]["from"] == "block_orange" and m["block_red1"]["kind"] == "cuboid"


def test_resolves_the_collision_that_name_matching_gets_wrong():
    """block_orange -> red takes token block_red1 while block_red1 itself became blue.

    Name-matching the goal's block_red1 to object_pose.block_red1 yields the WRONG object; this
    map yields the right one. Observed in a real render, not hypothetical.
    """
    m = block_variable_map(None, {"block_orange": "red", "block_red1": "blue"},
                           bc.DATASET_COLORS)
    assert m["block_red1"]["from"] == "block_orange"
    assert m["block_red1"]["pose_column"] == f"{POSE_PREFIX}block_orange"
    assert m["block_blue"]["from"] == "block_red1"


def test_duplicate_glyphs_point_back_to_distinct_blocks():
    m = block_variable_map({"block_a": "L", "block_d": "L"})
    assert m["block_l"]["from"] == "block_a"
    assert m["block_l_2"]["from"] == "block_d"
    assert len({v["from"] for v in m.values()}) == 2


def test_carries_rgb_so_the_palette_need_not_travel():
    m = block_variable_map(None, {"block_red1": "orange"}, bc.DATASET_COLORS)
    assert m["block_orange"]["rgb"] == [1.0, 0.5, 0.0]
    assert m["block_orange"]["hsv"] == [1 / 12, 1.0, 1.0]


@pytest.mark.parametrize("g,c", [({"block_a": "L"}, None), (None, {"block_red1": "blue"})])
def test_either_randomizer_alone_still_produces_a_map(g, c):
    assert block_variable_map(g, c, bc.DATASET_COLORS)


def test_both_off_is_empty_not_an_error():
    assert block_variable_map() == {}
    assert block_variable_map_json() == "{}"


def test_json_is_stable_and_sorted():
    a = block_variable_map_json({"block_a": "L"}, {"block_red1": "blue"}, bc.DATASET_COLORS)
    b = block_variable_map_json({"block_a": "L"}, {"block_red1": "blue"}, bc.DATASET_COLORS)
    assert a == b and list(json.loads(a)) == sorted(json.loads(a))


def test_recorded_goal_maps_back_to_the_source_goal():
    ident = _ep0()
    assert source_goal(EP0_RENDERED, ident) == EP0_SOURCE
    assert rendered_text(EP0_SOURCE, ident) == EP0_RENDERED


def test_rendered_token_that_is_another_blocks_source_name():
    """block_1 in the rendered text is block_d; block block_1 itself renders as F (block_f)."""
    ident = _ep0()
    assert ident.to_source["block_1"] == "block_d"
    assert ident.to_source["block_f"] == "block_1"
    assert source_goal("(on block_1 block_f)", ident) == "(on block_d block_1)"


def test_same_colour_cuboids_keep_distinct_tokens():
    ident = _ep0()
    assert ident.to_source["block_blue"] == "block_green"
    assert ident.to_source["block_blue2"] == "block_white"
    assert source_goal("(on block_blue2 block_blue)", ident) == "(on block_white block_green)"


def test_unmapped_tokens_pass_through():
    ident = _ep0()
    assert source_goal("(on_table block_zz workspace_table)", ident) == (
        "(on_table block_zz workspace_table)")


def test_identity_carries_paint_and_palette():
    ident = _ep0()
    assert ident.glyphs == EP0_GLYPHS
    assert ident.colors == EP0_COLORS
    assert dict(ident.palette)["white"] == (0.0, 0.0, 1.0)


def test_glyph_map_only_row_paints_but_keeps_the_goal():
    ident = identity_assignment({}, parse_glyph_map('{"block_a":"L"}'))
    assert ident and ident.glyphs == {"block_a": "L"} and not ident.to_source
    assert source_goal("(on block_a block_b)", ident) == "(on block_a block_b)"


def test_identity_survives_the_queue_wire():
    ident = _ep0()
    assert BlockIdentity.from_dict(json.loads(json.dumps(ident.to_dict()))) == ident


@pytest.mark.parametrize("raw", [None, "", "  ", b"", {}])
def test_empty_columns_parse_to_no_identity(raw):
    assert not identity_assignment(parse_block_variable_map(raw), parse_glyph_map(raw))


def test_entry_without_source_block_is_rejected():
    with pytest.raises(ValueError, match="no 'from'"):
        parse_block_variable_map('{"block_l": {"kind": "glyph"}}')


def test_every_entry_reconstructs_its_pose_column():
    m = block_variable_map({"block_a": "L", "block_d": "L"},
                           {"block_orange": "red", "block_red1": "blue"}, bc.DATASET_COLORS)
    for tok, v in m.items():
        assert v["pose_column"] == f"{POSE_PREFIX}{v['from']}"
