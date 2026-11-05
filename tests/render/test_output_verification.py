"""_verify_problem_outputs must catch a problem that recorded episodes but wrote no files.

render_params is built from episode_records, which are appended BEFORE the frames are encoded, so
it reports what the render intended. On 2026-08-24 a scene mismatch produced problems with
render_params saying completed=true / 3 episodes / 0 defects and NO data/ and NO videos at all.
The filesystem is the only honest witness.
"""
import importlib.util
import re
import sys
from pathlib import Path

RR = Path(__file__).resolve().parents[2] / "scripts/render/render.py"
FEATS = ["observation.images.camera:front_camera", "observation.depth.camera:front_camera"]


def _load_verifier():
    """Import just the function; render.py needs Isaac at import time, so exec the source of
    the one function against a namespace with the handful of names it uses."""
    src = RR.read_text()
    start = src.index("def _verify_problem_outputs(")
    end = src.index("def _read_video_defects(")
    ns = {"Path": Path, "re": re, "List": list, "Dict": dict, "Any": object}
    exec(compile(src[start:end], "<verify>", "exec"), ns)
    return ns["_verify_problem_outputs"]


verify = _load_verifier()


def _mk(tmp, split, n_eps, n_videos, with_data=True):
    root = tmp / split
    if with_data:
        d = root / "data" / "chunk-000"
        d.mkdir(parents=True, exist_ok=True)
        (d / "file-000.parquet").write_bytes(b"x")
    for f in FEATS:
        vd = root / "videos" / f / "chunk-000"
        vd.mkdir(parents=True, exist_ok=True)
        for i in range(n_videos):
            (vd / f"file-{i:03d}.mp4").write_bytes(b"x")
    return [{"split": split} for _ in range(n_eps)]


def test_complete_output_passes(tmp_path):
    recs = _mk(tmp_path, "blocks_success", n_eps=3, n_videos=3)
    assert verify(str(tmp_path), {}, recs) == ([], [])


def test_no_files_at_all_is_caught(tmp_path):
    """The exact failure: episodes recorded, nothing on disk."""
    (tmp_path / "blocks_success").mkdir(parents=True)
    recs = [{"split": "blocks_success"} for _ in range(3)]
    bad, holes = verify(str(tmp_path), {}, recs)
    assert bad, "an empty problem must not verify clean"
    assert any("no data" in b for b in bad), bad
    assert holes == []


def test_missing_videos_are_holes_not_failures(tmp_path):
    """A short feature is a patchable hole, NOT a reason to render the whole problem."""
    recs = _mk(tmp_path, "blocks_success", n_eps=4, n_videos=3)      # one short in every feature
    bad, holes = verify(str(tmp_path), {}, recs)
    assert bad == [], bad
    assert len(holes) == len(FEATS), holes
    assert {h["episode"] for h in holes} == {3}, holes
    assert {h["split"] for h in holes} == {"blocks_success"}
    assert {h["feature"] for h in holes} == set(FEATS)


def test_hole_names_the_gap_not_the_tail(tmp_path):
    """file-000 and file-002 present, file-001 gone -> the hole is episode 1."""
    recs = _mk(tmp_path, "blocks_success", n_eps=3, n_videos=3)
    for f in FEATS:
        (tmp_path / "blocks_success" / "videos" / f / "chunk-000" / "file-001.mp4").unlink()
    bad, holes = verify(str(tmp_path), {}, recs)
    assert bad == [], bad
    assert {h["episode"] for h in holes} == {1}, holes


def test_camera_with_nothing_is_still_only_holes(tmp_path):
    """Even a camera with zero files is patched episode-by-episode, never rendered whole:
    the patch pass renders exactly the listed episodes, so a full render is never the cheaper
    option."""
    recs = _mk(tmp_path, "blocks_success", n_eps=3, n_videos=0)
    bad, holes = verify(str(tmp_path), {}, recs)
    assert bad == [], bad
    assert len(holes) == 3 * len(FEATS), holes
    assert {h["episode"] for h in holes} == {0, 1, 2}


def test_majority_missing_is_still_only_holes(tmp_path):
    """No fraction threshold: 6 of 10 missing is 6 episodes to patch, not a failed problem."""
    recs = _mk(tmp_path, "blocks_success", n_eps=10, n_videos=4)
    bad, holes = verify(str(tmp_path), {}, recs)
    assert bad == [], bad
    assert {h["episode"] for h in holes} == {4, 5, 6, 7, 8, 9}, holes


def test_missing_data_parquet_is_caught(tmp_path):
    recs = _mk(tmp_path, "blocks_success", n_eps=2, n_videos=2, with_data=False)
    bad, _holes = verify(str(tmp_path), {}, recs)
    assert any("no data/*.parquet" in b for b in bad), bad


def test_zero_recorded_episodes_is_not_an_error(tmp_path):
    """A split with nothing to render (e.g. blocks_failure in a patch) must not be flagged."""
    assert verify(str(tmp_path), {}, []) == ([], [])


def test_both_splits_checked_independently(tmp_path):
    recs = _mk(tmp_path, "blocks_success", n_eps=4, n_videos=4)
    recs += _mk(tmp_path, "blocks_failure", n_eps=4, n_videos=3)     # failure split is short
    bad, holes = verify(str(tmp_path), {}, recs)
    assert bad == [], bad
    assert holes and all(h["split"] == "blocks_failure" for h in holes), holes
