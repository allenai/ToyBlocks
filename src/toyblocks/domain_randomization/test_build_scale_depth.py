"""    python src/toyblocks/domain_randomization/test_build_scale_depth.py

--render_scale and --skip_depth act on the sim2real preset cameras before scene creation:
scale_cameras keeps the aspect, refuses odd sizes (yuv420p needs even), and is a no-op when unset;
drop_depth removes only the depth data types; apply_build_flags leaves depth alone by default.

Same harness as test_build_camera_fov.py: build.py is loaded by path with torch and
toyblocks.scenes stubbed, so this runs without Isaac or a GPU.
"""
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

BUILD = Path(__file__).resolve().parent / "build.py"
TYPES = ["rgb", "distance_to_image_plane", "instance_segmentation_fast"]


def _load_build():
    sys.modules.setdefault("torch", types.ModuleType("torch"))
    spec = importlib.util.spec_from_file_location("dr_build_scale_depth", BUILD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stub_preset():
    cams = {
        "front_camera": SimpleNamespace(width=640, height=360, data_types=list(TYPES)),
        "left_camera": SimpleNamespace(width=640, height=480, data_types=list(TYPES)),
    }
    scenes = types.ModuleType("toyblocks.scenes")
    scenes.SCENE_PRESETS = {"sim2real": SimpleNamespace(camera_cfgs=cams)}
    pkg = types.ModuleType("toyblocks")
    pkg.__path__ = []
    sys.modules["toyblocks"] = pkg
    sys.modules["toyblocks.scenes"] = scenes
    return cams


def _env_cfg():
    return SimpleNamespace(scene=SimpleNamespace(env_spacing=None),
                           sim=SimpleNamespace(render=SimpleNamespace()),
                           scene_preset_id="sim2real")


def test_scale_keeps_aspect_and_is_off_by_default():
    mod = _load_build()
    cams = _stub_preset()
    mod.scale_cameras(None)
    mod.scale_cameras(1.0)
    assert (cams["front_camera"].width, cams["front_camera"].height) == (640, 360)
    mod.scale_cameras(2.0)
    assert (cams["front_camera"].width, cams["front_camera"].height) == (1280, 720)
    assert (cams["left_camera"].width, cams["left_camera"].height) == (1280, 960)
    print("A. 2.0 -> front 1280x720, left 1280x960; None/1.0 leave 640x360")


def test_scale_refuses_odd_sizes():
    mod = _load_build()
    _stub_preset()
    try:
        mod.scale_cameras(1.33)
    except ValueError as e:
        assert "even" in str(e)
        print(f"B. odd size refused: {e}")
        return
    raise AssertionError("1.33 should give 851x479 and be refused")


def test_skip_depth_drops_only_depth():
    mod = _load_build()
    cams = _stub_preset()
    mod.apply_build_flags(_env_cfg(), {})
    assert cams["front_camera"].data_types == TYPES, "depth must stay by default"
    mod.apply_build_flags(_env_cfg(), {}, skip_depth=True)
    for cam in cams.values():
        assert cam.data_types == ["rgb", "instance_segmentation_fast"], cam.data_types
    print("C. default keeps depth; skip_depth removes distance_to_image_plane only")


if __name__ == "__main__":
    test_scale_keeps_aspect_and_is_off_by_default()
    test_scale_refuses_odd_sizes()
    test_skip_depth_drops_only_depth()
    print("ALL PASS")
