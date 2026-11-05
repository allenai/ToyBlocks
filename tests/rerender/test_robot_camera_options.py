"""    python tests/rerender/test_robot_camera_options.py

--skip_cameras / --render_scale / --skip_depth must also reach ROBOT-mounted cameras. build.py only
edits the sim2real preset, and BaseEnv adds the robot's cameras (gripper_camera) afterwards, so
`--skip_cameras gripper_camera` used to print a warning and render the camera anyway.
rerender._robot_camera_options wraps common_utils.apply_camera_overrides_json (which BaseEnv calls
right after adding robot cameras) for the duration of gym.make.

The class is loaded from rerender.py's source by AST, with blocksuite stubbed: no Isaac, no GPU.
"""
import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[2] / "scripts" / "rerender"
_failures: list[str] = []


def _check(name, got, want):
    ok = got == want
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f": got {got!r}, want {want!r}"))
    if not ok:
        _failures.append(name)


def _load_class():
    src = (HERE / "rerender.py").read_text()
    node = next(n for n in ast.parse(src).body
                if isinstance(n, ast.ClassDef) and n.name == "_robot_camera_options")
    ns: dict = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), "rerender.py", "exec"), ns)
    return ns["_robot_camera_options"]


def _stub_blocksuite(calls):
    preset = {"front_camera": SimpleNamespace(width=1280, height=720, data_types=["rgb"])}
    scenes = types.ModuleType("blocksuite.scenes")
    scenes.SCENE_PRESETS = {"sim2real": SimpleNamespace(camera_cfgs=preset)}
    cu = types.ModuleType("blocksuite.utils.common_utils")
    cu.apply_camera_overrides_json = lambda cfg: calls.append("orig")
    utils = types.ModuleType("blocksuite.utils")
    utils.common_utils = cu
    pkg = types.ModuleType("blocksuite")
    pkg.__path__ = []
    utils.__path__ = []
    sys.modules.update({"blocksuite": pkg, "blocksuite.scenes": scenes, "blocksuite.utils": utils,
                        "blocksuite.utils.common_utils": cu})
    return cu


def _env_cfg():
    """camera_cfgs as BaseEnv has them when it calls the hook: preset (already scaled) + robot."""
    return SimpleNamespace(camera_cfgs={
        "front_camera": SimpleNamespace(width=1280, height=720, data_types=["rgb"]),
        "gripper_camera": SimpleNamespace(width=640, height=360,
                                          data_types=["rgb", "distance_to_image_plane"]),
    })


def main():
    opts_cls = _load_class()

    print("A: no options -> the hook is untouched")
    calls: list = []
    cu = _stub_blocksuite(calls)
    orig = cu.apply_camera_overrides_json
    with opts_cls(None, None, False):
        _check("hook not wrapped", cu.apply_camera_overrides_json is orig, True)

    print("B: --skip_cameras gripper_camera drops the robot camera")
    cfg = _env_cfg()
    with opts_cls(["left_camera", "camera:gripper_camera"], None, False):
        cu.apply_camera_overrides_json(cfg)
    _check("original hook still called", calls, ["orig"])
    _check("gripper removed, front kept", sorted(cfg.camera_cfgs), ["front_camera"])
    _check("hook restored after gym.make", cu.apply_camera_overrides_json is orig, True)

    print("C: --render_scale / --skip_depth reach robot cameras, preset cameras are not re-scaled")
    cfg = _env_cfg()
    with opts_cls([], 2.0, True):
        cu.apply_camera_overrides_json(cfg)
    g, f = cfg.camera_cfgs["gripper_camera"], cfg.camera_cfgs["front_camera"]
    _check("gripper scaled", (g.width, g.height), (1280, 720))
    _check("gripper depth dropped", g.data_types, ["rgb"])
    _check("front (preset) untouched here", (f.width, f.height), (1280, 720))

    print()
    if _failures:
        print(f"{len(_failures)} FAILURE(S): {_failures}")
        return 1
    print("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())


def test_main():
    # main() swaps stub blocksuite modules into sys.modules; put the real ones back for the
    # tests that run after it in the same session.
    def ours(k: str) -> bool:
        return k == "blocksuite" or k.startswith("blocksuite.")

    saved = {k: v for k, v in sys.modules.items() if ours(k)}
    try:
        assert main() == 0
    finally:
        for k in [k for k in sys.modules if ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
