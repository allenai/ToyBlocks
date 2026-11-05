"""    python -m pytest src/toyblocks/domain_randomization/test_scene_camera_base.py

`--camera_base scene` must record the FOV base the renderer composes onto -- for EVERY problem a
sim renders, not just the first.

render packs several problems into one process and builds the interpreter per problem, so
`_scene_camera_params` runs once per problem. It read the base FOV from the LIVE camera, which
after the first problem holds the previous wave's jittered FOV; the renderer meanwhile composes
onto the pristine `cam_base_K` captured once at sim build. So every later problem recorded
`episodes[].cameras.*.vertical_fov_deg` = (previous wave's jittered FOV) + (its own offset), while
the parquet's `observation.camera_intrinsics.*` carried the true FOV. Measured on oct21 vecgen
member 7409, rendered second in a process: front vfov recorded 45.592 deg against a true 47.247.

Runs on CPU: camera.py is loaded by file path, isaaclab's convention converter is stubbed (the
rotation is not under test), and a fake camera stands in for the TiledCamera the way
test_wave_intrinsics does.
"""
from __future__ import annotations

import ast
import importlib.util
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KEY = "camera:front"
NE = 4
DT = 0.05


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("dr_camera_scene_base", HERE / "camera.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


C = _load()


@pytest.fixture(autouse=True)
def _stub_isaaclab_math(monkeypatch: pytest.MonkeyPatch) -> None:
    """`_scene_camera_params` imports isaaclab's converter function-locally; importing the real
    isaaclab bootstraps Kit, so hand it an identity converter."""
    mod = types.ModuleType("isaaclab.utils.math")
    mod.convert_camera_frame_orientation_convention = lambda q, origin, target: q
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils", types.ModuleType("isaaclab.utils"))
    monkeypatch.setitem(sys.modules, "isaaclab.utils.math", mod)


class FakeCam:
    """TiledCamera stand-in: cfg.offset for the scene pose, live intrinsics the appliers write and
    the recorder reads back."""

    def __init__(self, n: int, fy: float = 196.0, cy: float = 180.0, cx: float = 320.0) -> None:
        k = torch.tensor([[fy, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        self.cfg = SimpleNamespace(prim_path="/World/envs/env_.*/front_cam",
                                   offset=SimpleNamespace(pos=(1.0, 0.0, 0.5),
                                                          rot=(1.0, 0.0, 0.0, 0.0),
                                                          convention="ros"))
        self.data = SimpleNamespace(intrinsic_matrices=k.repeat(n, 1, 1),
                                    pos_w=torch.zeros(n, 3),
                                    quat_w_world=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(n, 1))

    def set_intrinsic_matrices(self, mats: torch.Tensor, env_ids: list[int]) -> None:
        for j, i in enumerate(env_ids):
            self.data.intrinsic_matrices[int(i)] = mats[j]


def _fake_sim(cam: FakeCam) -> SimpleNamespace:
    return SimpleNamespace(sim=SimpleNamespace(render=lambda: None),
                           scene=SimpleNamespace(sensors={KEY: cam}, update=lambda dt: None))


def _vfov(K: torch.Tensor) -> float:
    return math.degrees(2 * math.atan(float(K[1, 2]) / float(K[1, 1])))


def _render_problem(sim: SimpleNamespace, cam: FakeCam, base_K: dict[str, torch.Tensor],
                    dfov: float, pass_base: bool) -> tuple[float, float]:
    """One single-wave problem the way replay_job runs it: build the `initial` record, render the
    wave from the pristine base, record one episode. Returns (recorded vfov, parquet vfov)."""
    kw: dict[str, Any] = {"base_K": base_K} if pass_base else {}
    initial = C._scene_camera_params(sim, [KEY], **kw)
    wave = [{"episode_index": i} for i in range(NE)]
    ep_off = {i: {KEY: {"position_offset_cm": [0.0, 0.0, 0.0], "pitch_deg": 0.0, "yaw_deg": 0.0,
                        "roll_deg": 0.0, "vertical_fov_deg": dfov}} for i in range(NE)}
    active = list(range(NE))
    C._apply_episode_camera_fov({KEY: cam}, wave, active, ep_off, "cpu",
                                base_K=base_K, camera_base="scene")
    rec = C._absolute_cam_params(initial[KEY], ep_off[0][KEY])
    return rec["vertical_fov_deg"], _vfov(cam.data.intrinsic_matrices[0])


def _two_problems(pass_base: bool) -> list[tuple[float, float]]:
    cam = FakeCam(NE)
    sim = _fake_sim(cam)
    _, base_K = C._capture_camera_bases(sim, {KEY: cam}, {KEY: cam}, DT)
    return [_render_problem(sim, cam, base_K, dfov, pass_base) for dfov in (-1.655, +2.1)]


def test_live_read_skews_the_second_problem():
    """The reproduction: reading the base from the live camera is right for the first problem and
    off by exactly the first problem's FOV jitter for the second."""
    (rec1, true1), (rec2, true2) = _two_problems(pass_base=False)
    assert rec1 == pytest.approx(true1, abs=1e-4)
    assert rec2 - true2 == pytest.approx(-1.655, abs=1e-3)


def test_pristine_base_records_the_rendered_fov_for_every_problem():
    for rec, true in _two_problems(pass_base=True):
        assert rec == pytest.approx(true, abs=1e-4)


def test_initial_block_is_the_pristine_base_on_a_reused_sim():
    cam = FakeCam(NE)
    sim = _fake_sim(cam)
    _, base_K = C._capture_camera_bases(sim, {KEY: cam}, {KEY: cam}, DT)
    pristine = _vfov(base_K[KEY][0])
    _render_problem(sim, cam, base_K, +2.9, pass_base=True)
    assert _vfov(cam.data.intrinsic_matrices[0]) != pytest.approx(pristine, abs=1e-3)
    assert C._scene_camera_params(sim, [KEY], base_K=base_K)[KEY]["vertical_fov_deg"] == \
        pytest.approx(pristine, abs=1e-6)


def test_camera_outside_base_K_falls_back_to_live():
    """A camera with no pristine snapshot (not FOV-randomized, so nothing rewrites its K) still
    gets its FOV from the live camera."""
    cam = FakeCam(NE, fy=250.0)
    out = C._scene_camera_params(_fake_sim(cam), [KEY], base_K={})
    assert out[KEY]["vertical_fov_deg"] == pytest.approx(_vfov(cam.data.intrinsic_matrices[0]))


def _call_kwargs(path: Path, func_name: str, attr_owner: str = "") -> list[set]:
    """Keyword names of every call to `func_name` (or `<attr_owner>.<func_name>`) in `path`."""
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        hit = (isinstance(f, ast.Name) and f.id == func_name and not attr_owner) or (
            isinstance(f, ast.Attribute) and f.attr == func_name
            and isinstance(f.value, ast.Name) and f.value.id == attr_owner)
        if hit:
            out.append({k.arg for k in node.keywords})
    return out


def test_pristine_base_reaches_the_scene_record():
    """Wiring: replay_job hands its sim-wide cam_base_K to the interpreter, and the interpreter
    hands it to _scene_camera_params. Either link missing silently restores the live read."""
    setup_calls = _call_kwargs(REPO / "scripts" / "render" / "render.py", "setup", "_dr_interp")
    assert setup_calls and all("cam_base_K" in kw for kw in setup_calls)
    scene_calls = _call_kwargs(HERE / "interpreter.py", "_scene_camera_params")
    assert scene_calls and all("base_K" in kw for kw in scene_calls)
