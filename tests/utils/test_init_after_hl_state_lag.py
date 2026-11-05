"""Init-after-HL cut row aligns with recorded high_level_state (action can lead by 1)."""

from __future__ import annotations

import pyarrow as pa

from toyblocks.utils.episode_dataset import (
    _COL_HIGH_LEVEL_ACTION,
    _COL_HIGH_LEVEL_STATE,
    _init_row_after_hl_action,
)


def _hl_table(
    actions: list[str],
    states: list[str],
) -> pa.Table:
    assert len(actions) == len(states)
    return pa.table(
        {
            _COL_HIGH_LEVEL_ACTION: actions,
            _COL_HIGH_LEVEL_STATE: states,
        }
    )


def test_init_after_hl_advances_when_state_lags_action_by_one() -> None:
    """Mimics flip→pnp: next action label appears before state updates."""
    # flip on [0:1], pnp from 2; state catches up on frame 3 (not 2).
    table = _hl_table(
        actions=["flip_block", "flip_block", "pnp_table_block", "pnp_table_block"],
        states=[
            "(on block_a table)",
            "(on block_a table) (ori_of block_r o17)",
            "(on block_a table) (ori_of block_r o17)",
            "(on block_a table) (ori_of block_r o18)",
        ],
    )
    start = _init_row_after_hl_action(
        table,
        first_idx=0,
        next_idx=4,
        actions=frozenset({"flip_block"}),
        label="flip_block",
        source="test",
    )
    assert start == 3


def test_init_after_hl_no_extra_advance_when_state_updates_on_boundary() -> None:
    table = _hl_table(
        actions=["flip_block", "flip_block", "pnp_table_block"],
        states=[
            "(pre)",
            "(post_flip)",
            "(post_flip) (on block_a table)",
        ],
    )
    start = _init_row_after_hl_action(
        table,
        first_idx=0,
        next_idx=3,
        actions=frozenset({"flip_block"}),
        label="flip_block",
        source="test",
    )
    assert start == 2


def test_init_after_hl_action_only_cut_without_state_column() -> None:
    table = pa.table({_COL_HIGH_LEVEL_ACTION: ["flip_block", "pnp_table_block"]})
    start = _init_row_after_hl_action(
        table,
        first_idx=0,
        next_idx=2,
        actions=frozenset({"flip_block"}),
        label="flip_block",
        source="test",
    )
    assert start == 1
