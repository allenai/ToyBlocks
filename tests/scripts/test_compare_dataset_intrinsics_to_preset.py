"""Tests for scripts/compare_dataset_intrinsics_to_preset.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMPARE_SCRIPT = _REPO_ROOT / "scripts" / "compare_dataset_intrinsics_to_preset.py"


def _load_compare_module():
    spec = importlib.util.spec_from_file_location("compare_intrinsics", _COMPARE_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def compare_mod():
    return _load_compare_module()


def test_discover_episode_ids_sorted_non_contiguous(tmp_path: Path, compare_mod) -> None:
    table = pa.table({"episode_index": [7, 2, 2, 5]})
    pq.write_table(table, tmp_path / "data.parquet")
    assert compare_mod.discover_episode_ids(tmp_path) == [2, 5, 7]


def test_discover_episode_ids_empty_dataset(tmp_path: Path, compare_mod) -> None:
    table = pa.table({"other": [0]})
    pq.write_table(table, tmp_path / "data.parquet")
    assert compare_mod.discover_episode_ids(tmp_path) == []
