"""_src_episode_index: passthrough matches output episodes by the rows load_episodes rendered."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from toyblocks.domain_randomization.episodes import FAILED_ATTEMPT_COL

RENDER = Path(__file__).resolve().parents[2] / "scripts" / "render" / "render.py"


def _load() -> Any:
    tree = ast.parse(RENDER.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_src_episode_index"]
    ns: dict[str, Any] = {"np": np, "Path": Path, "sys": sys}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "render.py", "exec"), ns)
    return ns["_src_episode_index"]


_src_episode_index = _load()


def _write(root: Path, flags: list[bool] | None) -> None:
    (root / "data" / "chunk-000").mkdir(parents=True)
    cols: dict[str, Any] = {"episode_index": pa.array([0, 0, 0, 0, 1, 1, 1], pa.int64())}
    if flags is not None:
        cols[FAILED_ATTEMPT_COL] = flags
    pq.write_table(pa.table(cols), root / "data" / "chunk-000" / "file-000.parquet")


def test_rendered_count_excludes_failed_attempt_rows(tmp_path: Path) -> None:
    _write(tmp_path, [False, True, True, False, False, False, False])
    ((_pf, spans, _rg),) = _src_episode_index(tmp_path)
    assert spans == {0: (0, 4, 2), 1: (4, 3, 3)}


def test_source_without_the_column_renders_every_row(tmp_path: Path) -> None:
    _write(tmp_path, None)
    ((_pf, spans, _rg),) = _src_episode_index(tmp_path)
    assert spans == {0: (0, 4, 4), 1: (4, 3, 3)}
