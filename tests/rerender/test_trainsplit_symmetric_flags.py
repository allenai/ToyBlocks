"""rerender_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml must constrain glyphs the way the
interpreter actually reads them.

The first version put the per-character mapping directly under ``randomize_task_letters``; the
interpreter reads ``randomize_task_letters.glyphs`` (interpreter.setup: ``_rtl.get("glyphs") if
isinstance(_rtl, dict)``), got None, and drew from all 36 glyphs -- a demo re-render came out with
A / I / 4 / 7 / 9. These tests resolve the file through that same lookup.
"""
from pathlib import Path

import yaml

from blocksuite.domain_randomization.block_colors import hsv_to_rgb, parse_palette
from blocksuite.domain_randomization.glyphs import resolve_glyph_sets

FLAGS = (Path(__file__).resolve().parents[2] / "src" / "blocksuite" / "domain_randomization" / "flags"
         / "rerender_flags_allrand_texcache_glyphs_trainsplit_symmetric.yaml")
UPSIDE_DOWN = set("IHNOSZX")   # scripts/task_gen/stack_three.py UPSIDE_DOWN_SYMMETRIC_LETTERS
CASE = set("COPSUVWXZ")        # scripts/task_gen/stack_three.py CASE_SYMMETRIC_LETTERS
TRAIN = set("CDEFGHJKLMNPQSUVWXYZ") | {"1", "3", "5"}


def _interpreter_glyph_sets():
    rtl = yaml.safe_load(FLAGS.read_text())["randomize_task_letters"]
    assert rtl and rtl is not True
    return resolve_glyph_sets(rtl.get("glyphs") if isinstance(rtl, dict) else rtl)


def test_glyphs_are_read_from_the_glyphs_key():
    rtl = yaml.safe_load(FLAGS.read_text())["randomize_task_letters"]
    assert isinstance(rtl, dict) and isinstance(rtl.get("glyphs"), dict)


def test_every_train_glyph_stays_in_its_symmetry_class_and_split():
    sets = _interpreter_glyph_sets()
    for g in TRAIN:
        cls = (g in UPSIDE_DOWN, g in CASE)
        assert sets[g] <= TRAIN, (g, sets[g] - TRAIN)
        assert all((t in UPSIDE_DOWN, t in CASE) == cls for t in sets[g]), (g, sets[g])
        assert len(sets[g]) > 1, g


def test_glyphs_outside_the_split_are_left_alone():
    sets = _interpreter_glyph_sets()
    for g in set(sets) - TRAIN:
        assert sets[g] == frozenset((g,)), (g, sets[g])


def test_palette_is_the_train_colours_at_the_authored_rgb():
    pal = parse_palette(yaml.safe_load(FLAGS.read_text())["randomize_block_colors"])
    assert {n: hsv_to_rgb(h) for n, h in pal} == {
        "red": (1.0, 0.0, 0.0), "green": (0.0, 1.0, 0.0), "blue": (0.0, 0.0, 1.0), "white": (1.0, 1.0, 1.0)}
