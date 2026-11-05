"""    python -m pytest src/blocksuite/domain_randomization/test_wave_intrinsics.py

Recorded camera intrinsics must equal the intrinsics the renderer actually uses.

Tiled rendering builds one render product from the whole camera list and does not follow
per-prim focal length, so writing a different matrix per env changes nothing in the image -- it
only desynchronises the recording, which reads intrinsics back per env. On shipped sep12 data
every episode in a wave of 32 rendered with env slot 0's focal length while the parquet recorded
its own planned FOV; predicting each frame's radial error as

    (fx[episode] - fx[wave leader]) / fx[episode]

with no free parameters explained 98.3% of the measured variance (r = +0.996). The resulting
annotation error is ~10 px at 200 px from the principal point.

The contract these tests pin: whatever is applied for a wave is ONE matrix, and it is the wave's
first active env -- the value that actually renders.
"""
import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

SRC = Path(__file__).resolve().parent / "camera.py"


def _load():
    spec = importlib.util.spec_from_file_location("dr_camera", SRC)
    m = importlib.util.module_from_spec(spec)
    sys.modules["dr_camera"] = m
    spec.loader.exec_module(m)
    return m


C = _load()


class FakeCam:
    """Stands in for a TiledCamera: records what was applied, and reports it back the way the
    LeRobot recorder reads it (``cam.data.intrinsic_matrices``)."""

    def __init__(self, n, fx=393.07, cx=320.0, cy=180.0):
        k = torch.tensor([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]])
        self.data = type("D", (), {})()
        self.data.intrinsic_matrices = k.repeat(n, 1, 1)
        self.applied = None

    def set_intrinsic_matrices(self, mats, env_ids=None):
        self.applied = (mats.clone(), list(env_ids) if env_ids is not None else None)
        for j, i in enumerate(self.applied[1]):
            self.data.intrinsic_matrices[int(i)] = mats[j]


def _wave(fovs):
    """One wave, one episode per env, each with its own vertical_fov_deg offset."""
    wave = [{"episode_index": i} for i in range(len(fovs))]
    ep_off = {i: {"cam": {"vertical_fov_deg": f}} for i, f in enumerate(fovs)}
    return wave, ep_off, list(range(len(fovs)))


def _fx_for(dfov, base_fx=393.07, cy=180.0):
    vfov = 2 * math.atan(cy / base_fx) + math.radians(dfov)
    return cy / math.tan(vfov / 2)


FOVS = [-2.9, +1.4, 0.0, +2.8, -1.7]


def test_one_matrix_for_the_whole_wave():
    cam = FakeCam(len(FOVS))
    wave, ep_off, active = _wave(FOVS)
    C._apply_episode_camera_fov({"cam": cam}, wave, active, ep_off, "cpu", camera_base="scene")
    mats, ids = cam.applied
    assert ids == active
    for j in range(1, len(mats)):
        assert torch.allclose(mats[j], mats[0]), (
            "per-env intrinsics do not reach the tiled render; applying them only "
            "desynchronises the recording")


def test_the_applied_value_is_the_wave_leader_not_the_base():
    """It must be env slot 0's FOV -- the one the tiled renderer uses -- not the unjittered base."""
    cam = FakeCam(len(FOVS))
    wave, ep_off, active = _wave(FOVS)
    C._apply_episode_camera_fov({"cam": cam}, wave, active, ep_off, "cpu", camera_base="scene")
    mats, _ = cam.applied
    assert mats[0][0, 0].item() == pytest.approx(_fx_for(FOVS[0]), rel=1e-4)
    assert abs(mats[0][0, 0].item() - 393.07) > 5, "jitter must still be applied, just per wave"


def test_recorded_readback_matches_for_every_episode():
    """The recorder reads cam.data.intrinsic_matrices per env; all of them must now agree."""
    cam = FakeCam(len(FOVS))
    wave, ep_off, active = _wave(FOVS)
    C._apply_episode_camera_fov({"cam": cam}, wave, active, ep_off, "cpu", camera_base="scene")
    read = cam.data.intrinsic_matrices
    for i in range(1, len(FOVS)):
        assert torch.allclose(read[i], read[0]), f"env {i} would record a different fx"


def test_it_warns_once_about_the_discarded_jitter(capsys):
    cam = FakeCam(len(FOVS))
    wave, ep_off, active = _wave(FOVS)
    C._apply_episode_camera_fov._warned = set()
    C._apply_episode_camera_fov({"cam": cam}, wave, active, ep_off, "cpu", camera_base="scene")
    out = capsys.readouterr().out
    assert "per WAVE" in out and "cannot vary intrinsics per env" in out
    C._apply_episode_camera_fov({"cam": cam}, wave, active, ep_off, "cpu", camera_base="scene")
    assert "cannot vary intrinsics per env" not in capsys.readouterr().out, "should warn once"


def test_no_warning_when_the_wave_agrees_anyway():
    cam = FakeCam(3)
    wave, ep_off, active = _wave([1.5, 1.5, 1.5])
    C._apply_episode_camera_fov._warned = set()
    C._apply_episode_camera_fov({"cam": cam}, wave, active, ep_off, "cpu", camera_base="scene")
    assert "cam" not in C._apply_episode_camera_fov._warned


def test_escape_hatch_restores_per_env_when_tiling_is_fixed():
    """If someone later shows tiled rendering honours per-prim intrinsics, one flag re-enables it."""
    cam = FakeCam(len(FOVS))
    wave, ep_off, active = _wave(FOVS)
    C.ALLOW_PER_ENV_INTRINSICS = True
    try:
        C._apply_episode_camera_fov({"cam": cam}, wave, active, ep_off, "cpu",
                                    camera_base="scene")
        mats, _ = cam.applied
        assert not torch.allclose(mats[1], mats[0])
    finally:
        C.ALLOW_PER_ENV_INTRINSICS = False


def test_the_old_behaviour_is_what_produced_the_shipped_defect():
    """Reconstruct the bug: per-env matrices recorded, wave leader rendered, ~10 px at r=200."""
    base = 393.07
    fx_rec = [_fx_for(f, base) for f in FOVS]
    rendered = fx_rec[0]
    worst = max(abs(fx / rendered - 1) for fx in fx_rec)
    assert worst * 200 > 8.0, (
        f"a {worst*100:.1f}% focal mismatch moves a marker {worst*200:.1f} px at r=200")


# --------------------------------------------------------------------------------------
# render_params.json must describe the camera that rendered, not the one that was sampled.
# recover_camera_columns rebuilds intrinsics from this file for camera-less sources, so a
# sampled-but-never-applied FOV propagates straight back into a "repaired" dataset.
# --------------------------------------------------------------------------------------


def _offs(fovs, cam="cam"):
    return {i: {cam: {"vertical_fov_deg": f, "position_offset_cm": [i, 0, 0],
                      "pitch_deg": float(i)}} for i, f in enumerate(fovs)}


def test_recorded_fov_is_the_leaders_for_every_episode():
    wave = [{"episode_index": i, "length": 10} for i in range(len(FOVS))]
    ep_off = _offs(FOVS)
    active = list(range(len(FOVS)))
    for i in range(len(FOVS)):
        got = C.applied_wave_offsets(ep_off[i], wave, active, ep_off)
        assert got["cam"]["vertical_fov_deg"] == pytest.approx(FOVS[0]), (
            f"episode {i} would record its sampled FOV, not the one it rendered with")


def test_position_and_rotation_are_left_per_episode():
    """Pose DOES apply per env -- collapsing it would throw away real randomization."""
    wave = [{"episode_index": i, "length": 10} for i in range(len(FOVS))]
    ep_off = _offs(FOVS)
    got = C.applied_wave_offsets(ep_off[3], wave, list(range(len(FOVS))), ep_off)
    assert got["cam"]["position_offset_cm"] == [3, 0, 0]
    assert got["cam"]["pitch_deg"] == 3.0


def test_leader_skips_envs_that_are_not_rendered():
    """`active` excludes _render=False slots, so the leader is the first RENDERED env."""
    wave = [{"episode_index": i, "length": 10} for i in range(len(FOVS))]
    ep_off = _offs(FOVS)
    active = [2, 3, 4]                      # slots 0 and 1 dropped
    got = C.applied_wave_offsets(ep_off[4], wave, active, ep_off)
    assert got["cam"]["vertical_fov_deg"] == pytest.approx(FOVS[2])


def test_apply_and_record_agree_on_who_the_leader_is():
    """The two halves of the fix must not drift apart -- same leader, same fx."""
    active = [2, 3, 4]
    wave = [{"episode_index": i, "length": 10} for i in range(len(FOVS))]
    ep_off = _offs(FOVS)
    cam = FakeCam(len(FOVS))
    C._apply_episode_camera_fov({"cam": cam}, wave, active, ep_off, "cpu", camera_base="scene")
    applied_fx = cam.applied[0][0][0, 0].item()
    recorded_fov = C.applied_wave_offsets(ep_off[4], wave, active, ep_off)["cam"]["vertical_fov_deg"]
    assert applied_fx == pytest.approx(_fx_for(recorded_fov), rel=1e-4)


def test_escape_hatch_also_restores_per_episode_recording():
    wave = [{"episode_index": i, "length": 10} for i in range(len(FOVS))]
    ep_off = _offs(FOVS)
    C.ALLOW_PER_ENV_INTRINSICS = True
    try:
        got = C.applied_wave_offsets(ep_off[3], wave, list(range(len(FOVS))), ep_off)
        assert got["cam"]["vertical_fov_deg"] == pytest.approx(FOVS[3])
    finally:
        C.ALLOW_PER_ENV_INTRINSICS = False
