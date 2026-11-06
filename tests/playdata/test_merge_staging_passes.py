"""The play pipeline's merge staging, with more than one render pass.

Every pass writes the SAME ``problem_*`` names into its own ``job_<name>`` directory. Staging
them all into one flat directory made the second pass abort on a name collision, so a play run
could only ever have one ``--render-pass``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PLAY = ROOT / "scripts/datagen/run_playdata_pipeline.py"


@pytest.fixture(scope="module")
def mod():
    sys.path.insert(0, str(ROOT / "scripts/datagen"))
    spec = importlib.util.spec_from_file_location("_rpp", PLAY)
    m = importlib.util.module_from_spec(spec)
    sys.modules["_rpp"] = m
    spec.loader.exec_module(m)
    return m


def _problem(root: Path, name: str, *, success: bool = True) -> None:
    split = "blocks_success" if success else "blocks_failure"
    (root / name / split / "meta").mkdir(parents=True, exist_ok=True)
    (root / name / split / "meta" / "info.json").write_text("{}")


def test_two_passes_holding_the_same_problems_do_not_collide(mod, tmp_path):
    out = tmp_path / "render" / "out"
    for job in ("job_plain", "job_allrand_texcache"):
        for prob in ("problem_a", "problem_b"):
            _problem(out / job, prob)

    roots = sorted(d for d in out.glob("job_*") if d.is_dir())
    staged, linked = mod.stage_merge_source(roots, tmp_path / "merge_src")

    assert [d.name for d in staged] == ["job_allrand_texcache", "job_plain"]
    assert len(linked) == 4, "every pass contributes both problems"
    for d in staged:
        assert sorted(p.name for p in d.iterdir()) == ["problem_a", "problem_b"]
        assert all(p.is_symlink() for p in d.iterdir())


def test_a_pass_with_no_success_split_contributes_no_staging_dir(mod, tmp_path):
    """merge_datasets picks its layout from the first problem dir, so a dir holding only
    blocks_failure must never be staged."""
    out = tmp_path / "render" / "out"
    _problem(out / "job_plain", "problem_a")
    _problem(out / "job_empty", "problem_a", success=False)

    roots = sorted(d for d in out.glob("job_*") if d.is_dir())
    staged, linked = mod.stage_merge_source(roots, tmp_path / "merge_src")

    assert [d.name for d in staged] == ["job_plain"]
    assert linked == ["problem_a"]
    assert not (tmp_path / "merge_src" / "job_empty").exists()


def test_staging_is_rebuilt_from_scratch(mod, tmp_path):
    """A stale link from an earlier run must not survive into this merge."""
    out = tmp_path / "render" / "out"
    _problem(out / "job_plain", "problem_a")
    stale = tmp_path / "merge_src" / "job_gone"
    stale.mkdir(parents=True)
    (stale / "problem_zzz").symlink_to(tmp_path)

    staged, _ = mod.stage_merge_source(
        sorted(d for d in out.glob("job_*") if d.is_dir()), tmp_path / "merge_src")

    assert [d.name for d in staged] == ["job_plain"]
    assert not stale.exists(), "the previous run's staging must be cleared"
