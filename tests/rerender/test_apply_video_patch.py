"""The completion phase's patch-apply step.

A patch CANNOT be rendered into the dataset it repairs: the recorder numbers episodes by save
order into a fresh dataset, so a patch aimed at the real output writes episode 250's pixels to
file-000 and destroys a good episode. It renders into its own directory and this step maps the
intermediate index back to the source index via render_params[].src_episode_index.

These tests exist because getting that remap backwards is silent and unrecoverable.
"""
import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

ORCH = Path(__file__).resolve().parents[2] / "scripts/rerender/completion.py"
FEATS = ["observation.images.camera:front_camera", "observation.depth.camera:front_camera"]


def _load():
    """completion.py needs no Isaac, so loading it here is safe."""
    sys.path.insert(0, str(ORCH.parent))
    spec = importlib.util.spec_from_file_location("completion_under_test", ORCH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(f"completion not importable here: {type(e).__name__}: {e}")
    return mod


orch = _load()


def _mp4(path: Path, payload: bytes = b"", moov: bool = True) -> None:
    """Smallest thing that satisfies has_moov: an ftyp box, then optionally a moov box."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = struct.pack(">I", 8 + len(payload)) + b"ftyp" + payload
    if moov:
        body += struct.pack(">I", 8) + b"moov"
    path.write_bytes(body)


def _dataset(root: Path, problem: str, split: str, episodes: list[dict], videos: dict) -> None:
    """videos: {episode_index: bool present} per feature."""
    meta = root / problem / split / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "render_params.json").write_text(json.dumps(
        {"completed": True, "status": "complete", "episodes": episodes, "video_defects": []}))
    (meta / "info.json").write_text(json.dumps({"total_episodes": len(videos)}))
    for feat in FEATS:
        for ep, present in videos.items():
            if present:
                _mp4(Path(orch.video_path(str(root), problem, split, feat, ep)),
                     payload=f"{problem}/{feat}/{ep}".encode())


# ---------------------------------------------------------------- has_moov

def test_moov_present(tmp_path):
    f = tmp_path / "a.mp4"
    _mp4(f)
    assert orch.has_moov(str(f))


def test_moov_absent_is_rejected(tmp_path):
    f = tmp_path / "b.mp4"
    _mp4(f, payload=b"x" * 64, moov=False)
    assert not orch.has_moov(str(f))


def test_missing_file_is_not_moov(tmp_path):
    assert not orch.has_moov(str(tmp_path / "nope.mp4"))


def test_zero_length_box_does_not_loop_forever(tmp_path):
    f = tmp_path / "c.mp4"
    f.write_bytes(struct.pack(">I", 0) + b"free" + b"x" * 16)
    assert not orch.has_moov(str(f))


# ---------------------------------------------------------------- video_path

def test_video_path_layout():
    p = orch.video_path("/out", "problem_x", "blocks_success", "obs.cam", 1234)
    assert p.endswith("problem_x/blocks_success/videos/obs.cam/chunk-001/file-234.mp4")


# ---------------------------------------------------------------- apply_video_patch

def test_patch_lands_at_the_source_index_not_the_render_index(tmp_path):
    """THE test. The patch rendered 2 episodes as file-000/file-001; they belong at 7 and 19."""
    out, patch = tmp_path / "out", tmp_path / "patch"
    _dataset(out, "problem_a", "blocks_success",
             [{"episode_index": i, "split": "blocks_success"} for i in range(20)],
             {i: (i not in (7, 19)) for i in range(20)})
    _dataset(patch, "problem_a", "blocks_success",
             [{"episode_index": 0, "split": "blocks_success", "src_episode_index": 7},
              {"episode_index": 1, "split": "blocks_success", "src_episode_index": 19}],
             {0: True, 1: True})

    res = orch.apply_video_patch(str(out), str(patch))
    assert res["episodes"] == 2, res
    assert res["videos"] == 2 * len(FEATS), res
    assert res["skipped"] == [], res["skipped"]

    for feat in FEATS:
        # the holes are filled...
        for src_ep, render_ep in ((7, 0), (19, 1)):
            dst = Path(orch.video_path(str(out), "problem_a", "blocks_success", feat, src_ep))
            assert dst.exists()
            assert dst.read_bytes() == Path(
                orch.video_path(str(patch), "problem_a", "blocks_success", feat, render_ep)
            ).read_bytes()
        # ...and episode 0, which was fine, is untouched by the file-000 of the patch render
        ep0 = Path(orch.video_path(str(out), "problem_a", "blocks_success", feat, 0))
        assert f"problem_a/{feat}/0".encode() in ep0.read_bytes()


def test_truncated_patch_video_is_refused(tmp_path):
    """A patch video with no moov must leave the hole rather than replace it with garbage."""
    out, patch = tmp_path / "out", tmp_path / "patch"
    _dataset(out, "problem_a", "blocks_success",
             [{"episode_index": i, "split": "blocks_success"} for i in range(3)],
             {0: True, 1: False, 2: True})
    _dataset(patch, "problem_a", "blocks_success",
             [{"episode_index": 0, "split": "blocks_success", "src_episode_index": 1}], {0: True})
    for feat in FEATS:                                        # corrupt what the patch produced
        _mp4(Path(orch.video_path(str(patch), "problem_a", "blocks_success", feat, 0)),
             payload=b"y" * 32, moov=False)

    res = orch.apply_video_patch(str(out), str(patch))
    assert res["episodes"] == 0 and res["videos"] == 0, res
    assert len(res["skipped"]) == len(FEATS)
    assert all("moov" in why for why in res["skipped"]), res["skipped"]
    for feat in FEATS:
        assert not Path(orch.video_path(str(out), "problem_a", "blocks_success", feat, 1)).exists()


def test_render_params_records_the_patch_and_clears_its_defect(tmp_path):
    out, patch = tmp_path / "out", tmp_path / "patch"
    _dataset(out, "problem_a", "blocks_success",
             [{"episode_index": i, "split": "blocks_success", "dome": "old"} for i in range(3)],
             {0: True, 1: False, 2: True})
    rp = out / "problem_a" / "blocks_success" / "meta" / "render_params.json"
    d = json.loads(rp.read_text())
    d["video_defects"] = [{"episode": 1, "feature": FEATS[0]}, {"episode": 2, "feature": FEATS[0]}]
    d["status"] = "incomplete_videos"
    rp.write_text(json.dumps(d))
    _dataset(patch, "problem_a", "blocks_success",
             [{"episode_index": 0, "split": "blocks_success", "src_episode_index": 1,
               "dome": "new"}], {0: True})

    orch.apply_video_patch(str(out), str(patch))
    d = json.loads(rp.read_text())
    ep1 = next(e for e in d["episodes"] if e["episode_index"] == 1)
    assert ep1["dome"] == "new", "the record must describe the pixels now on disk"
    assert ep1["patched"] is True
    assert [x["episode"] for x in d["video_defects"]] == [2], d["video_defects"]
    assert d["status"] == "incomplete_videos"                 # episode 2 still open


def test_status_returns_to_complete_when_the_last_hole_closes(tmp_path):
    out, patch = tmp_path / "out", tmp_path / "patch"
    _dataset(out, "problem_a", "blocks_success",
             [{"episode_index": i, "split": "blocks_success"} for i in range(2)],
             {0: True, 1: False})
    rp = out / "problem_a" / "blocks_success" / "meta" / "render_params.json"
    d = json.loads(rp.read_text())
    d["video_defects"] = [{"episode": 1, "feature": FEATS[0]}]
    d["status"] = "incomplete_videos"
    rp.write_text(json.dumps(d))
    _dataset(patch, "problem_a", "blocks_success",
             [{"episode_index": 0, "split": "blocks_success", "src_episode_index": 1}], {0: True})

    orch.apply_video_patch(str(out), str(patch))
    assert json.loads(rp.read_text())["status"] == "complete"


def test_episode_without_provenance_is_skipped_not_guessed(tmp_path):
    """No src_episode_index means the target index is unknown -- guessing would overwrite a good
    episode."""
    out, patch = tmp_path / "out", tmp_path / "patch"
    _dataset(out, "problem_a", "blocks_success",
             [{"episode_index": i, "split": "blocks_success"} for i in range(2)],
             {0: True, 1: False})
    _dataset(patch, "problem_a", "blocks_success",
             [{"episode_index": 0, "split": "blocks_success"}], {0: True})

    res = orch.apply_video_patch(str(out), str(patch))
    assert res["episodes"] == 0
    assert any("src_episode_index" in why for why in res["skipped"]), res["skipped"]


def test_several_problems_in_one_patch_render(tmp_path):
    out, patch = tmp_path / "out", tmp_path / "patch"
    for prob, hole in (("problem_a", 3), ("problem_b", 5)):
        _dataset(out, prob, "blocks_success",
                 [{"episode_index": i, "split": "blocks_success"} for i in range(6)],
                 {i: (i != hole) for i in range(6)})
        _dataset(patch, prob, "blocks_success",
                 [{"episode_index": 0, "split": "blocks_success", "src_episode_index": hole}],
                 {0: True})
    res = orch.apply_video_patch(str(out), str(patch))
    assert res["problems"] == 2 and res["episodes"] == 2, res
    for prob, hole in (("problem_a", 3), ("problem_b", 5)):
        for feat in FEATS:
            assert Path(orch.video_path(str(out), prob, "blocks_success", feat, hole)).exists()


def test_scan_sees_the_holes_close(tmp_path):
    """End to end through the orchestrator's own completeness oracle."""
    out, patch = tmp_path / "out", tmp_path / "patch"
    _dataset(out, "problem_a", "blocks_success",
             [{"episode_index": i, "split": "blocks_success"} for i in range(4)],
             {0: True, 1: False, 2: True, 3: False})
    items = [{"problems": ["problem_a"]}]
    before = orch.scan_missing_videos(str(out), items)
    assert {r["episode"] for r in before} == {1, 3}, before

    _dataset(patch, "problem_a", "blocks_success",
             [{"episode_index": 0, "split": "blocks_success", "src_episode_index": 1},
              {"episode_index": 1, "split": "blocks_success", "src_episode_index": 3}],
             {0: True, 1: True})
    orch.apply_video_patch(str(out), str(patch))
    assert orch.scan_missing_videos(str(out), items) == []
