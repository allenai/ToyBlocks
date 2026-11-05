"""generate_task_from_dataset compiles rerendered goals in source block names."""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "generate_task_from_dataset.py"

RENDERED = "(and (on_table block_green workspace_table) (on block_l block_green))"
SOURCE = "(and (on_table block_blue workspace_table) (on block_5 block_blue))"
BVM = json.dumps({
    "block_l": {"from": "block_5", "kind": "glyph", "glyph": "L"},
    "block_green": {"from": "block_blue", "kind": "cuboid", "color": "green",
                    "hsv": [1 / 3, 1.0, 1.0], "rgb": [0.0, 1.0, 0.0]},
})


@pytest.fixture
def codegen(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    # The script launches Isaac and imports Isaac Lab arg types at import time.
    isaac_app = types.ModuleType("toyblocks.cli.isaac_app")
    isaac_app.launch_app = lambda: None  # type: ignore[attr-defined]
    base_args = types.ModuleType("toyblocks.cli.base_args")
    base_args.BaseArgs = type("BaseArgs", (), {})  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "toyblocks.cli.isaac_app", isaac_app)
    monkeypatch.setitem(sys.modules, "toyblocks.cli.base_args", base_args)
    spec = importlib.util.spec_from_file_location("generate_task_from_dataset_under_test", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, mod)
    spec.loader.exec_module(mod)
    return mod


def _write_dataset(root: Path, goal: str, bvm: str | None) -> None:
    rows = []
    for ep in range(2):
        for f in range(3):
            row = {
                "episode_index": ep,
                "is_first": f == 0,
                "pddl_goal": goal,
                "observation.state.object_pose.block_5": [0.0] * 7,
                "observation.state.object_pose.block_blue": [0.0] * 7,
                "observation.state.object_pose.block_l": [0.0] * 7,
                "observation.state.object_pose.block_green": [0.0] * 7,
            }
            if bvm is not None:
                row["block_variable_map"] = bvm
            rows.append(row)
    data = root / "data" / "chunk-000"
    data.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(rows), data / "file-000.parquet")
    (root / "meta").mkdir()
    (root / "meta" / "info.json").write_text(json.dumps({"fps": 25, "chunks_size": 1000}))


def test_rerendered_goals_compile_in_source_names(codegen: types.ModuleType, tmp_path: Path) -> None:
    _write_dataset(tmp_path, RENDERED, BVM)
    _, goals, counts, _ = codegen.read_task_spec_from_dataset(tmp_path)
    assert goals == [SOURCE]
    assert counts == {SOURCE: 2}


def test_goals_without_map_compile_as_recorded(codegen: types.ModuleType, tmp_path: Path) -> None:
    _write_dataset(tmp_path, SOURCE, None)
    _, goals, _, _ = codegen.read_task_spec_from_dataset(tmp_path)
    assert goals == [SOURCE]
