"""--cameras selection: named cameras vs the 'all' sentinel.

A merged dataset shipped without its 224x224 depth streams because --cameras named only the two
RGB cameras and everything else was silently dropped. 'all'/'*' keeps every video feature any
source declares, which is what a re-render that adds a stream partway through needs.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

MS = Path(__file__).resolve().parents[2] / "TAMP/factory/merge_success.py"


def _load():
    """merge_success pulls in heavy deps at import; skip rather than fail if they are absent."""
    spec = importlib.util.spec_from_file_location("merge_success_under_test", MS)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as e:                                    # noqa: BLE001
        pytest.skip(
            f"merge_success not importable here: {type(e).__name__}: {e}", allow_module_level=True
        )
    return mod


ms = _load()

FEATURES = {
    "observation.images.camera:front_camera": {"dtype": "video"},
    "observation.images.camera:gripper_camera": {"dtype": "video"},
    "observation.depth.camera:front_camera": {"dtype": "video"},
    "observation.depth.camera:front_camera_rg8_224": {"dtype": "video"},
    "observation.state.joint_position": {"dtype": "float32"},
}


def _kept(keep_cameras):
    """Mirror the filter fast_merge_with_filter applies to the (unioned) template features."""
    if ms.wants_all_cameras(keep_cameras):
        return [k for k, v in FEATURES.items() if v.get("dtype") == "video"]
    want = ms.normalize_camera_keys(keep_cameras)
    return [k for k, v in FEATURES.items() if v.get("dtype") == "video" and k in want]


@pytest.mark.parametrize("spec", ["all", "ALL", "*", " all "])
def test_sentinel_forms_are_recognised(spec):
    assert ms.wants_all_cameras([spec])


def test_named_cameras_are_not_the_sentinel():
    assert not ms.wants_all_cameras(["front_camera", "gripper_camera"])
    assert not ms.wants_all_cameras([])
    assert not ms.wants_all_cameras(None)


def test_bare_names_become_rgb_keys():
    assert ms.normalize_camera_keys(["front_camera"]) == ["observation.images.front_camera"]


def test_full_keys_pass_through():
    key = "observation.depth.camera:front_camera_rg8_224"
    assert ms.normalize_camera_keys([key]) == [key]


def test_named_cameras_drop_depth():
    """The regression: naming the RGB cameras loses every depth stream."""
    kept = _kept(["observation.images.camera:front_camera",
                  "observation.images.camera:gripper_camera"])
    assert kept == ["observation.images.camera:front_camera",
                    "observation.images.camera:gripper_camera"]
    assert not any("depth" in k for k in kept)


def test_bare_name_does_not_match_a_camera_prefixed_key():
    """Rerender writes observation.images.camera:<name>, so the bare-name default matches nothing
    in these datasets -- the merge would keep zero videos. Full keys or 'all' are the usable
    spellings."""
    assert _kept(["front_camera", "gripper_camera"]) == []


def test_all_keeps_every_video_feature_including_depth():
    kept = _kept(["all"])
    assert len(kept) == 4
    assert "observation.depth.camera:front_camera_rg8_224" in kept
    assert "observation.state.joint_position" not in kept


def _cli_default_for(flag):
    """The argparse default for `flag`, read from the source so no heavy import is needed."""
    import ast

    tree = ast.parse(MS.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and first.value == flag):
            continue
        for kw in node.keywords:
            if kw.arg == "default":
                return ast.literal_eval(kw.value)
    raise AssertionError(f"no add_argument({flag!r}) with a default found in {MS}")


def test_cli_default_is_the_all_sentinel():
    """A merge launched without --cameras must keep everything.

    The default used to be ['front_camera', 'gripper_camera'], which combined with
    test_bare_name_does_not_match_a_camera_prefixed_key means a rerender dataset merged with no
    --cameras kept NO videos at all, and with the full keys spelled out would still have dropped
    the depth streams and the --extra_rgb/--extra_depth sidecars. Any caller that forgets the flag
    -- run_tamper_pipeline.py did -- has to land on the safe side.
    """
    assert _cli_default_for("--cameras") == ["all"]


def test_cli_default_survives_the_sentinel_check():
    """Not just the literal: the default must actually satisfy wants_all_cameras and keep depth."""
    default = _cli_default_for("--cameras")
    assert ms.wants_all_cameras(default)
    kept = _kept(default)
    assert "observation.depth.camera:front_camera_rg8_224" in kept
    assert len(kept) == 4
