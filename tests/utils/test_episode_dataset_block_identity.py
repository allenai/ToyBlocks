"""Glyph/colour rerenders: queue setups grade on source block names, prompt on the recorded goal."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from blocksuite.domain_randomization.block_variables import BlockIdentity
from blocksuite.utils.episode_dataset import load_setups

RENDERED = "(and (on_table block_green workspace_table) (on block_l block_green))"
SOURCE = "(and (on_table block_blue workspace_table) (on block_5 block_blue))"
BVM = json.dumps({
    "block_l": {"from": "block_5", "kind": "glyph", "glyph": "L"},
    "block_green": {"from": "block_blue", "kind": "cuboid", "color": "green",
                    "hsv": [1 / 3, 1.0, 1.0], "rgb": [0.0, 1.0, 0.0]},
})
GLYPH_MAP = json.dumps({"block_5": "L"})


def _write_dataset(root: Path, extra: dict[str, str]) -> None:
    rows = [
        {
            "episode_index": 0,
            "frame_index": f,
            "is_first": f == 0,
            "observation.state.joint_position": [0.0] * 7,
            "action.high_level_action": "pnp_block_block",
            "pddl_goal": extra.get("pddl_goal", RENDERED),
            **{k: v for k, v in extra.items() if k != "pddl_goal"},
        }
        for f in range(3)
    ]
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), data / "file-000.parquet")
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 25, "chunks_size": 1000}))


def test_rerender_setup_splits_grading_and_policy_goal(tmp_path: Path) -> None:
    _write_dataset(tmp_path, {"block_variable_map": BVM, "glyph_map": GLYPH_MAP})
    (setup,) = load_setups(tmp_path)
    assert setup["pddl_goal"] == SOURCE
    assert setup["policy_pddl_goal"] == RENDERED
    ident = BlockIdentity.from_dict(setup["block_identity"])
    assert ident.glyphs == {"block_5": "L"} and ident.colors == {"block_blue": "green"}


def test_dataset_without_maps_is_unchanged(tmp_path: Path) -> None:
    _write_dataset(tmp_path, {"pddl_goal": SOURCE})
    (setup,) = load_setups(tmp_path)
    assert setup["pddl_goal"] == SOURCE
    assert "policy_pddl_goal" not in setup and "block_identity" not in setup


def test_empty_map_cells_are_unchanged(tmp_path: Path) -> None:
    _write_dataset(tmp_path, {"pddl_goal": SOURCE, "block_variable_map": "", "glyph_map": ""})
    (setup,) = load_setups(tmp_path)
    assert setup["pddl_goal"] == SOURCE
    assert "policy_pddl_goal" not in setup and "block_identity" not in setup
