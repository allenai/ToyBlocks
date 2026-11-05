"""Unit tests for Molmo2 HL action / coordinate parsing in maniflow_serving_utils."""

from __future__ import annotations

import numpy as np

from blocksuite.serving.maniflow_serving_utils import (
    _parse_hl_action,
    _parse_target_xy,
    _strip_coords,
)


def _assert_xy(xy: np.ndarray | None, expected: list[int] | None) -> None:
    if expected is None:
        assert xy is None
        return
    assert xy is not None
    np.testing.assert_allclose(xy, np.asarray(expected, dtype=np.float32))


def test_place_side_float_bracket_coords() -> None:
    s = "place side <193, 14.0>"
    lang, xy = _parse_hl_action(s)
    assert lang == "place side"
    _assert_xy(xy, [193, 14])


def test_pick_int_bracket_coords() -> None:
    s = "pick <320, 180>"
    lang, xy = _parse_hl_action(s)
    assert lang == "pick"
    _assert_xy(xy, [320, 180])


def test_place_on_table_no_coords() -> None:
    s = "place on table"
    lang, xy = _parse_hl_action(s)
    assert lang == "place on table"
    _assert_xy(xy, None)


def test_pick_trailing_int_coords() -> None:
    s = "pick 320,180"
    lang, xy = _parse_hl_action(s)
    assert lang == "pick"
    _assert_xy(xy, [320, 180])


def test_pick_trailing_float_coords() -> None:
    s = "pick 320.0, 180.0"
    lang, xy = _parse_hl_action(s)
    assert lang == "pick"
    _assert_xy(xy, [320, 180])


def test_strip_coords_and_parse_target_xy_float_brackets() -> None:
    s = "place side <193, 14.0>"
    assert _strip_coords(s) == "place side"
    _assert_xy(_parse_target_xy(s), [193, 14])
