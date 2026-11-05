"""    python src/blocksuite/domain_randomization/test_build_camera_fov.py

apply_camera_flag_overrides must not crash on a flags yaml that pins a camera FOV.

build.py used math.tan/math.radians at the fov_deg branch without importing math, so any flags
file carrying a TOP-LEVEL `cameras: {<name>: {fov_deg: ...}}` raised

    NameError: name 'math' is not defined

the first time that branch was reached. Two checked-in yamls sit on it --
TAMP/scripts/rerender/rerender_flags.example.yaml and rerender_flags_camtest.yaml -- while the
production yaml survives only because its `cameras:` block is nested under `randomize:` and is
consumed by a different path. So the crash was latent, not theoretical: it waited for whoever
first pinned an FOV.

The module is imported by file path with torch stubbed, because build.py imports torch at module
scope and this test must run without a GPU stack. blocksuite.scenes is stubbed too -- the function
mutates the shared SCENE_PRESETS cfg, and the point here is the arithmetic, not Isaac.
"""
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

BUILD = Path(__file__).resolve().parent / "build.py"


def _load_build():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    spec = importlib.util.spec_from_file_location("dr_build_under_test", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_preset(horizontal_aperture=20.955):
    cam = SimpleNamespace(
        offset=SimpleNamespace(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=SimpleNamespace(horizontal_aperture=horizontal_aperture, focal_length=24.0),
    )
    scenes = types.ModuleType("blocksuite.scenes")
    scenes.SCENE_PRESETS = {"sim2real": SimpleNamespace(camera_cfgs={"front_camera": cam})}
    pkg = types.ModuleType("blocksuite")
    pkg.__path__ = []
    sys.modules["blocksuite"] = pkg
    sys.modules["blocksuite.scenes"] = scenes
    return cam


def test_fov_override_sets_focal_length():
    print("A. fov_deg override computes a focal length instead of raising NameError")
    mod = _load_build()
    cam = _stub_preset()
    mod.apply_camera_flag_overrides({"cameras": {"front_camera": {"fov_deg": 60.0}}})
    want = 20.955 / (2.0 * math.tan(math.radians(60.0) / 2.0))
    assert abs(cam.spawn.focal_length - want) < 1e-9, (cam.spawn.focal_length, want)
    print(f"   fov 60 deg -> focal_length {cam.spawn.focal_length:.6f} (expected {want:.6f})")

    # a wider FOV must give a SHORTER focal length; catches an inverted formula
    mod.apply_camera_flag_overrides({"cameras": {"front_camera": {"fov_deg": 90.0}}})
    wide = cam.spawn.focal_length
    assert wide < want, f"90 deg gave {wide}, not shorter than 60 deg's {want}"
    print(f"   fov 90 deg -> focal_length {wide:.6f} (shorter, as it must be)")


def test_position_and_quaternion_still_apply():
    print("B. the other override branches are unaffected")
    mod = _load_build()
    cam = _stub_preset()
    mod.apply_camera_flag_overrides(
        {"cameras": {"front_camera": {"position": [0.1, 0.2, 0.3],
                                      "quaternion": [0.0, 1.0, 0.0, 0.0]}}}
    )
    assert cam.offset.pos == (0.1, 0.2, 0.3), cam.offset.pos
    assert cam.offset.rot == (0.0, 1.0, 0.0, 0.0), cam.offset.rot
    print(f"   pos {cam.offset.pos}, rot {cam.offset.rot}")


def test_unknown_camera_and_empty_flags_are_noops():
    print("C. unknown camera / empty flags do not raise")
    mod = _load_build()
    cam = _stub_preset()
    before = cam.spawn.focal_length
    mod.apply_camera_flag_overrides({})
    mod.apply_camera_flag_overrides({"cameras": {}})
    mod.apply_camera_flag_overrides({"cameras": {"no_such_camera": {"fov_deg": 60.0}}})
    assert cam.spawn.focal_length == before
    print("   all three no-oped")


def test_robot_camera_override_is_deferred_not_dropped():
    """The gripper camera is NOT in the scene preset -- it is merged in later from the robot cfg.

    The old code printed "not in sim2real preset; skipping" and dropped the request. Measured
    2026-11-03 in pixels: camtest asks for gripper z=0.062 / fov 105 and every dataset it ever
    produced rendered at the robot default z=0.022 / ~88, while render_params recorded the ask.
    """
    import json
    mod = _load_build()
    cam = _stub_preset()
    env_cfg = SimpleNamespace(scene_preset_id="sim2real", camera_overrides_json=None)
    mod.apply_camera_flag_overrides(
        {"cameras": {"gripper_camera": {"position": [0.031, 0.074, 0.062], "fov_deg": 105}}},
        env_cfg,
    )
    body = json.loads(env_cfg.camera_overrides_json)["gripper_camera"]
    assert body["pos"] == [0.031, 0.074, 0.062], body
    # fov_deg is passed through, NOT converted here: the conversion needs the gripper's own
    # horizontal_aperture (6.916), and that cfg does not exist until BaseEnv merges it in.
    assert body["fov_deg"] == 105 and "focal_length" not in body, body
    assert cam.spawn.focal_length == 24.0, "a robot camera must not touch the preset cfg"


def test_an_explicit_camera_overrides_json_wins_per_field():
    """Deferral must merge, never clobber: the caller's explicit override is more specific."""
    import json
    mod = _load_build()
    _stub_preset()
    env_cfg = SimpleNamespace(
        scene_preset_id="sim2real",
        camera_overrides_json=json.dumps({"gripper_camera": {"pos": [9.0, 9.0, 9.0]}}),
    )
    mod.apply_camera_flag_overrides(
        {"cameras": {"gripper_camera": {"position": [0.0, 0.0, 0.0], "fov_deg": 60}}}, env_cfg
    )
    body = json.loads(env_cfg.camera_overrides_json)["gripper_camera"]
    assert body["pos"] == [9.0, 9.0, 9.0], "explicit override must win"
    assert body["fov_deg"] == 60, "a field it did not set must still come from the yaml"


def test_scene_cameras_still_applied_in_place():
    """front_camera is preset-owned, so it keeps the early path and must not be deferred."""
    mod = _load_build()
    cam = _stub_preset()
    env_cfg = SimpleNamespace(scene_preset_id="sim2real", camera_overrides_json=None)
    mod.apply_camera_flag_overrides({"cameras": {"front_camera": {"fov_deg": 50.0}}}, env_cfg)
    want = 20.955 / (2.0 * math.tan(math.radians(50.0) / 2.0))
    assert abs(cam.spawn.focal_length - want) < 1e-9, cam.spawn.focal_length
    assert not env_cfg.camera_overrides_json, "a preset camera must not be deferred"


def test_math_is_imported():
    print("D. the import itself, so the fix cannot be reverted silently")
    src = BUILD.read_text()
    assert "\nimport math\n" in src, "build.py must import math at module scope"
    assert "math." in src, "sanity: build.py should still use math"
    print("   'import math' present at module scope")


if __name__ == "__main__":
    test_fov_override_sets_focal_length()
    test_position_and_quaternion_still_apply()
    test_unknown_camera_and_empty_flags_are_noops()
    test_math_is_imported()
    print("\nPASS")
