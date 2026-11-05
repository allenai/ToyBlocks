"""Row semantics of the LeRobot recorder, driven through its public API without Isaac.

The scene and the encoder subprocesses are faked; frame building is replaced by synthetic
samples whose observation and action block both carry the step number, so the tests pin down
which action each row holds: row t = (s_t, a_{t+1}), terminal row = (s_N, no action).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest


def _install_sim_stubs() -> None:
    """Stub the Isaac Lab imports the recorder module makes, so it loads headless."""
    specs = {
        "isaaclab": (),
        "isaaclab.scene": (),
        "isaaclab.scene.interactive_scene": ("InteractiveScene",),
        "isaaclab.assets": ("Articulation", "RigidObject"),
        "isaaclab.sensors": ("ContactSensor",),
        "isaaclab.sensors.camera": ("Camera",),
    }
    for name, attrs in specs.items():
        if name in sys.modules:
            continue
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        sys.modules[name] = module
    # The CPU CI lane installs neither PyAV nor LeRobot; nothing here encodes or writes, so
    # stand-ins do when they are missing. Installed ones are left alone (the stitch tests
    # need the real LeRobot in the same session).
    optional = {
        "av": (),
        "lerobot": (),
        "lerobot.datasets": (),
        "lerobot.datasets.lerobot_dataset": ("LeRobotDataset", "embed_images"),
    }
    installed = {top: importlib.util.find_spec(top) is not None for top in ("av", "lerobot")}
    for name, attrs in optional.items():
        if name in sys.modules or installed[name.split(".")[0]]:
            continue
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, type(attr, (), {}))
        sys.modules[name] = module


_install_sim_stubs()

from toyblocks.vecgen.recording import lerobot_recorder as lr  # noqa: E402

Rec = lr.IsaacDataRecorderLeRobot


class _FakeWriter:
    def __init__(self, index: int, root: str) -> None:
        self.index = index
        self.success_root = f"{root}/w{index}/blocks_success"
        self.failure_root = f"{root}/w{index}/blocks_failure"
        self.queued_episode_counts = {"success": 0, "failure": 0}
        self.backpressure_s = 0.0


@pytest.fixture
def make_recorder(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[Any]:
    """``make(num_envs, **kwargs) -> (recorder, sent)``; ``sent`` collects ``(writer, item)``."""
    made: list[Any] = []

    def fake_create_dataset(self: Any, dataset_name: Any = None) -> None:
        self.dataset_name = self._resolved_dataset_name
        self._success_root = f"{self.data_dir}/blocks_success"
        self._failure_root = f"{self.data_dir}/blocks_failure"
        self._writers = [_FakeWriter(k, self.data_dir) for k in range(self._num_writers)]
        self._cam_configs = {}
        self._camera_names = []
        self._episode_started = np.zeros(self.num_envs, dtype=np.bool_)

    monkeypatch.setattr(Rec, "_create_dataset", fake_create_dataset)

    def make(num_envs: int = 1, **kwargs: Any) -> tuple[Any, list[tuple[int, tuple]]]:
        scene = types.SimpleNamespace(rigid_objects={}, articulations={}, sensors={})
        rec = Rec(
            scene,
            data_dir=str(tmp_path),
            num_envs=num_envs,
            record_every_n_steps=1,
            fps=25,
            enable_sharding=False,
            **kwargs,
        )
        sent: list[tuple[int, tuple]] = []
        rec._writer_put = lambda writer, item: sent.append((writer.index, item))
        rec._prefetch_scene_data = lambda: {"robot_name": None, "step": rec._step}
        rec._build_frame_cached = lambda env_idx, cache, **kw: _sample(
            rec, cache["step"], env_idx, kw["is_first"]
        )
        rec._step = 0
        made.append(rec)
        return rec, sent

    yield make
    for rec in made:
        sys.stdout = rec._stdout_capture._original


def _sample(rec: Any, step: int, env_idx: int, is_first: bool) -> dict[str, Any]:
    """A frame as built post-step: observation s_<step>, action block a_<step>."""
    frame: dict[str, Any] = {
        "observation.state": np.full(8, step, dtype=np.float32),
        "is_first": np.array([is_first], dtype=np.bool_),
        "is_last": np.array([False], dtype=np.bool_),
        "action.is_valid": np.array([True], dtype=np.bool_),
        "is_episode_successful": np.array([bool(rec.success_flags[env_idx])], dtype=np.bool_),
    }
    for key, shape in Rec._ACTION_NUMERIC_FIELDS.items():
        frame[key] = np.full(shape, step, dtype=np.float32)
    for key in Rec._ACTION_STRING_FIELDS:
        frame[key] = f"{key}@{step}"
    for key in Rec._ACTION_ALIGNED_OBSERVATION_FIELDS:
        frame[key] = np.full(13, step, dtype=np.float32)
    return frame


def _record(rec: Any, step: int, envs: list[int], reset_env_ids: tuple[int, ...] = ()) -> None:
    rec._step = step
    rec.record_step_batched(envs, reset_env_ids=reset_env_ids)
    rec._pending_build_future.result()


def _rows(sent: list[tuple[int, tuple]], env: int) -> list[dict[str, Any]]:
    return [item[3] for _, item in sent if item[0] == "frame" and item[1] == env]


def _obs(row: dict[str, Any]) -> int:
    return int(row["observation.state"][0])


def _act(row: dict[str, Any]) -> int | None:
    return int(row["action.joint_target"][0]) if row["action.is_valid"][0] else None


def test_rows_pair_each_state_with_the_action_taken_in_it(make_recorder: Any) -> None:
    rec, sent = make_recorder()
    for step in range(4):
        _record(rec, step, [0])
    rec.save_episode(0)
    rows = _rows(sent, 0)
    assert [(_obs(r), _act(r)) for r in rows] == [(0, 1), (1, 2), (2, 3), (3, None)]
    assert rows[0]["is_first"][0] and not rows[1]["is_first"][0]
    assert [item[0] for _, item in sent][-1] == "end_episode"


def test_auto_reset_sample_starts_the_next_episode(make_recorder: Any) -> None:
    rec, sent = make_recorder()
    for step in range(3):
        _record(rec, step, [0])
    rec.set_success_flags_batched(np.array([True]))
    _record(rec, 3, [0], reset_env_ids=(0,))  # the env reset inside step 3: sample is s_0'
    rec.save_episode(0)
    first = _rows(sent, 0)
    assert [(_obs(r), _act(r)) for r in first] == [(0, 1), (1, 2), (2, None)]
    assert sent[-1][1][:3] == ("end_episode", 0, True)

    sent.clear()
    for step in (4, 5):
        _record(rec, step, [0])
    rec.save_episode(0)
    second = _rows(sent, 0)
    assert [(_obs(r), _act(r)) for r in second] == [(3, 4), (4, 5), (5, None)]
    assert second[0]["is_first"][0] and not second[0]["is_episode_successful"][0]
    assert sent[-1][1][:3] == ("end_episode", 0, False)


def test_captured_reset_frame_is_row_zero(make_recorder: Any) -> None:
    rec, sent = make_recorder()
    rec._step = 7
    rec.capture_reset_frames([0])
    for step in (8, 9):
        _record(rec, step, [0])
    rec.save_episode(0)
    assert [(_obs(r), _act(r)) for r in _rows(sent, 0)] == [(7, 8), (8, 9), (9, None)]


def test_staged_extras_ride_with_the_action_they_describe(make_recorder: Any) -> None:
    rec, sent = make_recorder(action_dim=3, record_hl_model=True)
    for step in range(3):
        rec.set_pending_actions_batched(np.full((1, 2), step, dtype=np.float32))  # padded to 3
        rec.set_pending_pd_modes_batched(np.array([np.nan if step == 1 else 0.5]))
        rec.set_hl_model_outputs_batched([f"hl{step}"], [f"prev{step}"], [f"q{step}"], [step == 2])
        _record(rec, step, [0])
    rec.save_episode(0)
    rows = _rows(sent, 0)
    assert [r["action.policy"].tolist() for r in rows] == [[1, 1, 0], [2, 2, 0], [0, 0, 0]]
    assert [r["action.hl_model_output"] for r in rows] == ["hl1", "hl2", ""]
    assert [bool(r["action.hl_done"][0]) for r in rows] == [False, True, False]
    # NaN keeps the stiffness-derived mode (the synthetic sample's step); 0.5 overrides it
    assert [float(r["action.pd_mode"][0]) for r in rows] == [1.0, 0.5, 0.0]
    assert float(rows[1]["action"][-1]) == 0.5


def test_hl_overlay_is_drawn_on_the_row_of_the_action_it_describes(make_recorder: Any) -> None:
    rec, sent = make_recorder(hl_overlay_camera="front_camera")
    build = rec._build_frame_cached
    rec._build_frame_cached = lambda env_idx, cache, **kw: {
        **build(env_idx, cache, **kw),
        "observation.images.front_camera": np.zeros((4, 4, 3), dtype=np.uint8),
    }
    rec._shm_put = lambda writer, images: None
    drawn: list[str | None] = []

    def draw(img: np.ndarray, overlay: tuple | None) -> np.ndarray:
        drawn.append(None if overlay is None else overlay[0])
        return img

    rec._draw_hl_overlay = draw
    for step in range(3):
        rec.set_hl_overlay_batched([f"hl{step}"], np.zeros((1, 2, 2)), [f"prev{step}"], [f"q{step}"])
        _record(rec, step, [0])
    rec.save_episode(0)
    # row t carries a_{t+1}'s HL; the terminal row keeps the last one staged
    assert drawn == ["hl1", "hl2", "hl2"]
    images = [item[2] for _, item in sent if item[0] == "frame" and item[1] == 0]
    assert all("observation.images.front_camera_hl_overlay" in im for im in images)
    assert all("_hl_overlay" not in row for row in _rows(sent, 0))


def test_hl_overlay_side_by_side_doubles_the_width(make_recorder: Any) -> None:
    pytest.importorskip("PIL")
    rec, _ = make_recorder(hl_overlay_camera="front_camera", hl_overlay_side_by_side=True)
    img = np.zeros((32, 48, 3), dtype=np.uint8)
    out = rec._draw_hl_overlay(img, ("model", None, "prev", "q", "oracle", None))
    assert out.shape == (32, 96, 3) and out.dtype == np.uint8
    assert rec._draw_hl_overlay(img, None).shape == (32, 96, 3)


def test_discard_drops_the_episode_and_counts_its_ordinal(make_recorder: Any) -> None:
    rec, sent = make_recorder()
    for step in range(3):
        _record(rec, step, [0])
    rec.discard_episode(0)
    assert sent[-1] == (0, ("discard_episode", 0))
    for step in range(3, 5):
        _record(rec, step, [0])
    rec.save_episode(0)
    assert [(_obs(r), _act(r)) for r in _rows(sent, 0)][-2:] == [(3, 4), (4, None)]
    assert rec.merged_episode_index_map() == {(0, 1): (rec._failure_root, 0)}


def test_writers_route_envs_in_blocks_and_index_map_follows_stitch_order(
    make_recorder: Any,
) -> None:
    rec, sent = make_recorder(num_envs=4, num_writers=2)
    for step in range(2):
        _record(rec, step, [0, 1, 2, 3])
    assert {w for w, item in sent if item[1] in (0, 1)} == {0}
    assert {w for w, item in sent if item[1] in (2, 3)} == {1}

    rec.set_success_flags_batched(np.array([False, False, True, True]))
    rec.save_episode(3)  # writer 1, success #0
    rec.save_episode(0)  # writer 0, failure #0
    rec.discard_episode(1)
    rec.save_episode(2)  # writer 1, success #1
    rec.set_success_flags_batched(np.array([False, True, False, False]))
    for step in range(2, 4):
        _record(rec, step, [1])
    rec.save_episode(1)  # writer 0, success #0 (env 1's second episode)

    s, f = rec._success_root, rec._failure_root
    assert rec.merged_episode_index_map() == {
        (1, 1): (s, 0),
        (3, 0): (s, 1),
        (2, 0): (s, 2),
        (0, 0): (f, 0),
    }


def test_over_budget_items_spill_to_disk_instead_of_blocking(
    make_recorder: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    import pickle
    import queue

    monkeypatch.setenv("TOYBLOCKS_RECORDER_SPILL_DIR", str(tmp_path / "spill"))
    monkeypatch.setenv("TOYBLOCKS_RECORDER_SPILL_MIN_FREE_GB", "0")
    rec, _ = make_recorder()
    writer = types.SimpleNamespace(
        index=0,
        process=types.SimpleNamespace(is_alive=lambda: True, exitcode=None),
        dead=False,
        outbox_sent=0,
        consumed=types.SimpleNamespace(value=0),
        backpressure_s=0.0,
        queue=queue.Queue(),
        spill_dir=None,
        spilled_items=0,
        spilled_bytes=0,
        cap=1 << 20,
    )
    small = ("start_episode", 0, {})
    frame = ("frame", 0, {"observation.images.cam": np.ones((512, 1024), np.uint8)}, {"x": 1})
    lr.IsaacDataRecorderLeRobot._writer_put(rec, writer, frame)  # fits: straight to the queue
    lr.IsaacDataRecorderLeRobot._writer_put(rec, writer, frame)  # over budget: spilled
    lr.IsaacDataRecorderLeRobot._writer_put(rec, writer, small)  # control messages never spill
    items = [writer.queue.get_nowait() for _ in range(3)]
    assert items[0][0] == "frame" and items[2] == small
    assert items[1][0] == "spilled" and writer.spilled_items == 1 and writer.backpressure_s == 0.0
    with open(items[1][1], "rb") as f:
        loaded = pickle.load(f)
    assert loaded[:2] == frame[:2] and np.array_equal(
        loaded[2]["observation.images.cam"], frame[2]["observation.images.cam"]
    )
    # spilled bytes stay in flight until the encoder retires them
    assert writer.outbox_sent == 2 * lr.payload_nbytes(frame) + lr.payload_nbytes(small)


def test_batched_depth_and_rgb_conversion_match_the_per_frame_path() -> None:
    import torch

    rng = np.random.default_rng(0)
    depth = rng.uniform(-0.5, 5.0, (3, 8, 9, 1)).astype(np.float32)
    depth[0, 0, 0, 0] = np.nan
    batched = Rec._depth_mm16(torch.from_numpy(depth)).numpy()
    assert batched.dtype == np.uint16
    for e in range(3):
        assert np.array_equal(batched[e], Rec._prepare_depth_image(depth[e]))
    rgba = torch.from_numpy(rng.uniform(0, 1, (2, 4, 4, 4)).astype(np.float32))
    assert np.array_equal(
        Rec._rgb_uint8(rgba).numpy(), (rgba[..., :3] * 255).to(torch.uint8).numpy()
    )
    raw = torch.randint(0, 255, (2, 4, 4, 3), dtype=torch.uint8)
    snap = Rec._rgb_uint8(raw)
    raw.zero_()  # the sensor reusing its buffer must not change the snapshot
    assert int(snap.sum()) > 0


def test_shared_memory_slots_round_trip_and_free() -> None:
    from toyblocks.vecgen.recording._shm_slots import SlotArena, SlotReader, frame_image_bytes

    rgb = np.random.default_rng(0).integers(0, 255, (36, 64, 3), dtype=np.uint8)
    depth = np.arange(36 * 64, dtype=np.uint16).reshape(36, 64)
    images = {"observation.images.cam": rgb, "observation.depth.cam": depth}
    arena = SlotArena(frame_image_bytes(images), 2)
    try:
        reader = SlotReader(arena.name, 2)
        d0, d1 = arena.put(images), arena.put(images)
        assert arena.put(images) is None  # both slots taken: caller falls back to the queue
        slot, views = reader.views(d0)
        assert np.array_equal(views["observation.images.cam"], rgb)
        assert np.array_equal(views["observation.depth.cam"], depth)
        del views
        reader.free(slot)
        assert arena.put(images) is not None
        assert arena.put({"big": np.zeros(10**6, np.uint8)}) is None  # does not fit a slot
        del d1
        reader.flags = None
        reader.shm.close()
    finally:
        arena.close()


def test_hl_state_is_its_own_column_and_stays_out_of_ll_args(make_recorder: Any) -> None:
    class _State:  # TAMP's PDDLState interface, as vecgen.recorder passes it
        def getTrueProps(self) -> set[str]:  # noqa: N802
            return {"(on_table a t)", "(clear a)"}

    rec, _ = make_recorder()
    rec.update_symbolic_actions(
        "pnp_table_block", ["robot", "a", "b"], "pick",
        {"ll_action_name": "pick", "current_hl_state": _State()}, env_idx=0,
    )
    assert rec.hl_state_strs[0] == "(clear a)\n(on_table a t)"
    assert rec.ll_args_dict[0] == {"ll_action_name": "pick"}
    rec.update_symbolic_actions("pnp_table_block", ["robot", "a", "b"], "place", {"ll_action_name": "place"}, env_idx=0)
    assert rec.hl_state_strs[0] == "(clear a)\n(on_table a t)"  # no state given: the last one stands


def test_single_sample_episode_is_one_row_with_no_action(make_recorder: Any) -> None:
    rec, sent = make_recorder()
    _record(rec, 0, [0])
    rec.save_episode(0)
    rows = _rows(sent, 0)
    assert [(_obs(r), _act(r)) for r in rows] == [(0, None)]
    assert rows[0]["is_first"][0] and not rows[0]["action.is_valid"][0]


def test_gains_travel_with_the_action_and_the_terminal_row_keeps_its_own(
    make_recorder: Any,
) -> None:
    rec, sent = make_recorder()
    for step in range(3):
        _record(rec, step, [0])
    rec.save_episode(0)
    rows = _rows(sent, 0)
    for key in Rec._ACTION_ALIGNED_OBSERVATION_FIELDS:
        assert [int(r[key][0]) for r in rows] == [1, 2, 2], key


# --------------------------------------------------------------------------
# RecordEpisode.skip_recording_next_step through this recorder
# --------------------------------------------------------------------------
#
# A deferred PD-switch settle holds the switching env for a few ordinary ``env.step`` calls;
# those steps must leave no row. These drive the real ``RecordEpisode.step`` into the recorder:
# a row's state is ``100 * env + sim step`` and its action is the one ``RecordEpisode`` staged.

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

_RECORD_EPISODE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "toyblocks" / "wrappers" / "record_episode.py"
)
HOLD = -1.0


@pytest.fixture
def record_episode_module(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """``record_episode`` loaded privately, with its Isaac-only imports stubbed for this test."""
    stubs = {
        "toyblocks.tasks.base_env": {"BaseEnv": type("BaseEnv", (), {})},
        "toyblocks.utils.dataset_recording_utils": {
            "IsaacDataRecorderBase": type("IsaacDataRecorderBase", (), {}),
            "_extract_frame": None,
        },
        "toyblocks.utils.visualization_utils": {"images_to_video": None, "tile_images": None},
    }
    for name, attrs in stubs.items():
        module = types.ModuleType(name)
        for attr, value in attrs.items():
            setattr(module, attr, value)
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("_record_episode_under_test", _RECORD_EPISODE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeVecEnv(gym.Env):
    """Auto-resetting vector env whose only state is the step count."""

    def __init__(self, num_envs: int, rec: Any) -> None:
        self.num_envs = num_envs
        self._rec = rec
        self.done_at: dict[int, list[int]] = {}

    def step(self, action: Any) -> tuple[Any, ...]:
        self._rec._step += 1
        done = torch.zeros(self.num_envs, dtype=torch.bool)
        for i in self.done_at.get(self._rec._step, []):
            done[i] = True
        zeros = torch.zeros(self.num_envs)
        obs = {"state": torch.full((self.num_envs, 1), float(self._rec._step))}
        return obs, zeros, done, torch.zeros_like(done), {"_final_info": done}


class _EpisodeHarness:
    """The recorder behind a real ``RecordEpisode``, recording through ``record_step_batched``."""

    def __init__(self, module: types.ModuleType, make_recorder: Any, num_envs: int) -> None:
        rec, self.sent = make_recorder(num_envs=num_envs, action_dim=1)
        base_build = rec._build_frame_cached

        def build(env_idx: int, cache: dict, **kw: Any) -> dict[str, Any]:
            frame = base_build(env_idx, cache, **kw)
            frame["observation.state"] = np.full(8, 100 * env_idx + rec._step, dtype=np.float32)
            return frame

        rec._build_frame_cached = build
        self.rec = rec
        self.env = _FakeVecEnv(num_envs, rec)
        wrapper = object.__new__(module.RecordEpisode)
        gym.Wrapper.__init__(wrapper, self.env)
        wrapper._save_sensor_obs_video = False
        wrapper._save_rendering_video = False
        wrapper._last_obs = None
        wrapper._elapsed_steps = 0
        wrapper._data_recorder = rec
        wrapper._skip_recording_env_ids = set()
        wrapper.save_trajectory = True
        wrapper.should_save_episode = None
        wrapper.should_save_in_flight_episode = None
        self.wrapper = wrapper
        rec.capture_reset_frames(range(num_envs))

    def step(self, actions: list[float], *, skip: list[int] | None = None) -> None:
        if skip is not None:
            self.wrapper.skip_recording_next_step(skip)
        self.wrapper.step(torch.tensor([[a] for a in actions], dtype=torch.float32))
        if self.rec._pending_build_future is not None:
            self.rec._pending_build_future.result()

    def finish(self) -> None:
        for i in range(self.rec.num_envs):
            if self.rec.active_episodes[i]:
                self.rec.save_episode(i)

    def rows(self, env_idx: int) -> list[tuple[float, float | None]]:
        """``(state, action)`` per row, ``None`` for a row with no action."""
        return [
            (float(r["observation.state"][0]),
             float(r["action.policy"][0]) if r["action.is_valid"][0] else None)
            for r in _rows(self.sent, env_idx)
        ]


def test_skipped_steps_record_no_rows(record_episode_module: Any, make_recorder: Any) -> None:
    """Env 0 switches at step 2 and holds for steps 2-3; step 4 runs the deferred 2.0."""
    h = _EpisodeHarness(record_episode_module, make_recorder, num_envs=2)
    h.step([1.0, 1.0])
    h.step([HOLD, 2.0], skip=[0])
    h.step([HOLD, 3.0], skip=[0])
    h.step([2.0, 4.0])
    h.finish()
    # The deferred action lands on the last recorded state, s_1.
    assert h.rows(0) == [(0.0, 1.0), (1.0, 2.0), (4.0, None)]
    assert h.rows(1) == [(100.0, 1.0), (101.0, 2.0), (102.0, 3.0), (103.0, 4.0), (104.0, None)]


def test_skip_is_one_shot(record_episode_module: Any, make_recorder: Any) -> None:
    h = _EpisodeHarness(record_episode_module, make_recorder, num_envs=1)
    h.step([1.0], skip=[0])
    h.step([2.0])
    h.step([3.0])
    h.finish()
    assert h.rows(0) == [(0.0, 2.0), (2.0, 3.0), (3.0, None)]


def test_a_done_step_during_a_skip_still_opens_the_next_episode(
    record_episode_module: Any, make_recorder: Any,
) -> None:
    """The reset frame is captured, and the held frame closes the finished episode."""
    h = _EpisodeHarness(record_episode_module, make_recorder, num_envs=1)
    h.env.done_at[3] = [0]
    h.step([1.0])
    h.step([HOLD], skip=[0])
    h.step([HOLD], skip=[0])  # done: s_3 is the next episode's s_0
    h.step([4.0])
    h.finish()
    assert h.rows(0) == [(0.0, 1.0), (1.0, None), (3.0, 4.0), (4.0, None)]
    assert [bool(r["is_first"][0]) for r in _rows(h.sent, 0)] == [True, False, True, False]


def test_an_env_with_no_episode_to_run_is_not_recorded(
    record_episode_module: Any, make_recorder: Any,
) -> None:
    h = _EpisodeHarness(record_episode_module, make_recorder, num_envs=2)
    h.wrapper.should_save_in_flight_episode = lambda env_idx: env_idx == 0
    h.step([1.0, 1.0])
    h.step([2.0, 2.0])
    h.finish()
    assert h.rows(0) == [(0.0, 1.0), (1.0, 2.0), (2.0, None)]
    assert h.rows(1) == []


def test_dropped_frames_ride_with_end_episode_and_are_left_out(make_recorder: Any) -> None:
    """A failed attempt undone by a snapshot restore: its rows are named in ``end_episode`` and
    the writer saves the episode without them (LeRobot renumbers the rest)."""
    from toyblocks.vecgen.recording._video_encoder import keep_frames

    rec, sent = make_recorder()
    _record(rec, 0, [0])
    _record(rec, 1, [0])
    mark = rec.frames_recorded(0)  # snapshot before the attempt: rows 0, 1 recorded
    assert mark == 2
    for step in (2, 3, 4):  # the attempt
        _record(rec, step, [0])
    rec.drop_frames_since(0, mark)  # restore
    rec.drop_frames_since(0, rec.frames_recorded(0))  # nothing recorded since: no span
    for step in (5, 6):  # the retry
        _record(rec, step, [0])
    rec.save_episode(0)
    end = sent[-1][1]
    assert end[0] == "end_episode" and end[6] == [(2, 5)]
    kept = keep_frames(_rows(sent, 0), end[6])
    assert [_obs(r) for r in kept] == [0, 1, 5, 6]

    sent.clear()  # the next episode starts with no dropped spans
    _record(rec, 7, [0])
    rec.save_episode(0)
    assert sent[-1][1][6] == []


def test_trimmed_videos_leave_out_the_dropped_frames(tmp_path: Any) -> None:
    """With videos, the dropped frames are re-encoded out of every stream: depth (lossless)
    keeps the other frames bit-exact, RGB keeps them in order."""
    av = pytest.importorskip("av")
    if not hasattr(av, "open"):
        pytest.skip("PyAV not installed")
    from toyblocks.vecgen.recording._video_encoder import _VideoEncoderProcess

    enc = _VideoEncoderProcess(None, str(tmp_path), "t", 25, {}, {}, depth_codec="hevc")
    n, dropped = 40, [(2, 5), (9, 10), (20, 31)]
    kept = [i for i in range(n) if not any(a <= i < b for a, b in dropped)]
    rng = np.random.default_rng(0)
    depth = [rng.integers(0, 65536, (32, 48), dtype=np.uint16) for _ in range(n)]
    rgb = [np.full((32, 48, 3), 6 * i, dtype=np.uint8) for i in range(n)]
    keys = {"observation.depth.front_camera": depth, "observation.images.front_camera": rgb}
    vids = enc._open_episode_streams(str(tmp_path / "ep"), {k: (48, 32) for k in keys})
    for i in range(n):
        for k, frames in keys.items():
            enc._encode_stream_frame(vids, k, frames[i])
    valid = enc._close_videos(vids)
    assert valid == set(keys)

    from concurrent.futures import Future

    fut: Future = Future()
    enc._trim_videos(vids, valid, dropped, fut)
    _, valid = fut.result()
    assert valid == set(keys)

    def frames(k: str, fmt: str) -> list[np.ndarray]:
        with av.open(vids[k][2]) as c:
            decoded = list(c.decode(video=0))
        # back-to-back timestamps at 25 fps: the kept frames play as one continuous clip
        assert [float(f.pts * f.time_base) for f in decoded] == pytest.approx(
            [i / 25 for i in range(len(decoded))])
        return [f.to_ndarray(format=fmt) for f in decoded]

    got = frames("observation.depth.front_camera", "gray12le")
    assert len(got) == len(kept)
    assert all(np.array_equal(g, depth[i] >> 4) for g, i in zip(got, kept))
    means = [float(f.mean()) for f in frames("observation.images.front_camera", "rgb24")]
    assert len(means) == len(kept)
    assert np.allclose(means, [6 * i for i in kept], atol=2)
