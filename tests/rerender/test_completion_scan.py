"""scan_missing_videos is the gate that decides whether a run may be merged.

plan_status only asks whether each problem wrote a completed marker, which is why 35 missing
videos shipped in a dataset whose every marker said complete. This check looks at the files.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/rerender"))
import completion as bo

FEATS = ["observation.images.camera:front_camera", "observation.depth.camera:front_camera"]


def _problem(root: Path, name: str, split: str, n_eps: int, present: dict):
    """present: {feature: [episode indices that HAVE a video]}"""
    meta = root / name / split / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": n_eps}))
    for feat in FEATS:
        d = root / name / split / "videos" / feat / "chunk-000"
        d.mkdir(parents=True, exist_ok=True)
        for e in present.get(feat, range(n_eps)):
            (d / f"file-{e:03d}.mp4").write_bytes(b"x")


def _items(*names):
    return [{"fragment_id": "frag_000", "problems": list(names)}]


def test_complete_dataset_has_nothing_missing(tmp_path):
    _problem(tmp_path, "problem_a", "blocks_success", 4, {})
    assert bo.scan_missing_videos(str(tmp_path), _items("problem_a")) == []


def test_one_missing_video_is_found(tmp_path):
    _problem(tmp_path, "problem_a", "blocks_success", 4,
             {FEATS[0]: [0, 1, 3]})                      # episode 2 missing in one feature only
    out = bo.scan_missing_videos(str(tmp_path), _items("problem_a"))
    assert out == [{"problem": "problem_a", "split": "blocks_success", "episode": 2}], out


def test_episode_missing_in_two_features_collapses_to_one_render(tmp_path):
    _problem(tmp_path, "problem_a", "blocks_success", 3,
             {FEATS[0]: [0, 1], FEATS[1]: [0, 1]})       # episode 2 missing in BOTH
    out = bo.scan_missing_videos(str(tmp_path), _items("problem_a"))
    assert len(out) == 1 and out[0]["episode"] == 2, out


def test_missing_videos_dir_means_every_episode(tmp_path):
    meta = tmp_path / "problem_a" / "blocks_success" / "meta"
    meta.mkdir(parents=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": 3}))
    out = bo.scan_missing_videos(str(tmp_path), _items("problem_a"))
    assert sorted(o["episode"] for o in out) == [0, 1, 2], out


def test_both_splits_scanned(tmp_path):
    _problem(tmp_path, "problem_a", "blocks_success", 2, {})
    _problem(tmp_path, "problem_a", "blocks_failure", 2, {FEATS[0]: [0]})
    out = bo.scan_missing_videos(str(tmp_path), _items("problem_a"))
    assert out == [{"problem": "problem_a", "split": "blocks_failure", "episode": 1}], out


def test_unreadable_problem_is_skipped_not_reported_complete(tmp_path):
    """No info.json => unknown, contributes nothing; it must not be counted as 0 missing."""
    (tmp_path / "problem_ghost").mkdir()
    assert bo.scan_missing_videos(str(tmp_path), _items("problem_ghost")) == []


def test_patch_tsv_round_trips(tmp_path):
    _problem(tmp_path, "problem_a", "blocks_success", 4, {FEATS[0]: [0, 1, 3]})
    out = bo.scan_missing_videos(str(tmp_path), _items("problem_a"))
    tsv = tmp_path / "_patch" / "c.tsv"
    assert bo.write_patch_tsv(str(tsv), out) == 1
    lines = tsv.read_text().splitlines()
    assert lines[0] == "problem\tsplit\tepisode"
    assert lines[1].split("\t") == ["problem_a", "blocks_success", "2"]


# ---------------------------------------------------------------------------------------------
# A feature directory that was never created, and a problem that never rendered: neither can be
# found by listing what IS on disk, and both shipped in allrand while every other check passed.
# ---------------------------------------------------------------------------------------------
def _declare(root: Path, name: str, split: str, n_eps: int, features):
    """Rewrite a split's info.json so it DECLARES its video features, as a real render does."""
    meta = root / name / split / "meta" / "info.json"
    meta.write_text(json.dumps({
        "total_episodes": n_eps,
        "features": {f: {"dtype": "video"} for f in features},
    }))


def test_declared_feature_with_no_directory_is_missing_for_every_episode(tmp_path):
    """3 of 9 features on disk is a real allrand split; the other 6 must not read as complete."""
    _problem(tmp_path, "problem_a", "blocks_success", 3, {})
    _declare(tmp_path, "problem_a", "blocks_success", 3, FEATS + ["observation.depth.camera:left"])
    out = bo.scan_missing_videos(str(tmp_path), _items("problem_a"))
    assert sorted(o["episode"] for o in out) == [0, 1, 2], out


def test_declared_features_all_present_is_still_complete(tmp_path):
    _problem(tmp_path, "problem_a", "blocks_success", 3, {})
    _declare(tmp_path, "problem_a", "blocks_success", 3, FEATS)
    assert bo.scan_missing_videos(str(tmp_path), _items("problem_a")) == []


def test_feature_set_falls_back_to_disk_when_info_declares_none(tmp_path):
    """Pre-features datasets (info.json with only total_episodes) must keep scanning."""
    _problem(tmp_path, "problem_a", "blocks_success", 4, {FEATS[0]: [0, 1, 3]})
    out = bo.scan_missing_videos(str(tmp_path), _items("problem_a"))
    assert [o["episode"] for o in out] == [2], out


def test_problem_that_produced_nothing_is_named(tmp_path):
    _problem(tmp_path, "problem_a", "blocks_success", 2, {})
    out = bo.scan_missing_problems(str(tmp_path), _items("problem_a", "problem_ghost"))
    assert out == ["problem_ghost"], out


def test_rendered_problem_is_not_reported_absent(tmp_path):
    _problem(tmp_path, "problem_a", "blocks_success", 2, {})
    assert bo.scan_missing_problems(str(tmp_path), _items("problem_a")) == []


def test_one_split_is_enough_to_count_as_rendered(tmp_path):
    """Not every problem has failures; a success-only problem is complete, not absent."""
    _problem(tmp_path, "problem_a", "blocks_success", 2, {})
    (tmp_path / "problem_a" / "blocks_failure" / "meta").mkdir(parents=True)
    (tmp_path / "problem_a" / "blocks_failure" / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 0}))
    assert bo.scan_missing_problems(str(tmp_path), _items("problem_a")) == []


def test_absent_problem_has_no_holes_to_patch(tmp_path):
    """The two scans divide the work: holes are patchable, an absent problem must be re-rendered."""
    items = _items("problem_ghost")
    assert bo.scan_missing_videos(str(tmp_path), items) == []
    assert bo.scan_missing_problems(str(tmp_path), items) == ["problem_ghost"]
