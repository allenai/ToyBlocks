"""CPU tests for replay gain settling, the replay action source, and row-aligned diagnostics."""

import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.plot_diagnostics_from_eval_run import (
    _has_row_layout_marker,
    _is_replay_run,
    _load_episode_specs,
    _rollout_from_frames,
)
from toyblocks.eval.pd_mode_controller import DATASET_SWITCH_SETTLE_TICKS, PDModeController
from toyblocks.eval.replay import ReplayPolicy, _detect_unrecorded_ticks, _phase_slices
from toyblocks.utils.episode_queue import apply_setup, park_placed_blocks, restore_placed_blocks


class _StubBaseEnv:
    def __init__(self) -> None:
        data = SimpleNamespace(
            joint_pos=torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.4]]),
            joint_vel=torch.zeros(1, 8),
            joint_stiffness=torch.ones(1, 8),
            joint_damping=torch.ones(1, 8),
            joint_effort_limits=torch.ones(1, 8),
            joint_vel_limits=torch.ones(1, 8),
            root_pose_w=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
        )
        self.robot = SimpleNamespace(articulation=SimpleNamespace(data=data))
        self.scene = SimpleNamespace(rigid_objects={})
        self.cfg = SimpleNamespace(decimation=4)
        self.num_envs = 1
        self.gain_calls: list[dict[str, object]] = []

    def set_actuator_gains(self, **kwargs: object) -> None:
        self.gain_calls.append(kwargs)


_GAINS = {"stiffness": [100.0] * 7, "damping": [20.0] * 7, "effort": [87.0] * 7}
_SOFT_GAINS = {"stiffness": [40.0] * 7, "damping": [13.0] * 7, "effort": [87.0] * 7}


def _controller(base_env: _StubBaseEnv, ticks: tuple[int, int] | None) -> PDModeController:
    return PDModeController(
        base_env,
        torch.device("cpu"),
        num_envs=1,
        stiff_gains=_GAINS,
        compliant_gains=_SOFT_GAINS,
        switch_settle_ticks=ticks,
    )


def test_switch_settles_tamp_ticks_times_decimation_with_arm_frozen() -> None:
    base_env = _StubBaseEnv()
    controller = _controller(base_env, DATASET_SWITCH_SETTLE_TICKS)
    controller.initialize_after_reset(target_mode=1)
    base_env.gain_calls.clear()

    controller.apply(torch.tensor([0.0]))  # stiff -> compliant
    controller.apply(torch.tensor([1.0]))  # compliant -> stiff

    assert [c["settle_steps"] for c in base_env.gain_calls] == [
        DATASET_SWITCH_SETTLE_TICKS[0] * 4,
        DATASET_SWITCH_SETTLE_TICKS[1] * 4,
    ]
    for call in base_env.gain_calls:
        assert call["settle_joint_ids"] == list(range(7))
        assert bool(call["settle_env_mask"][0])


def test_initialize_after_reset_never_settles() -> None:
    base_env = _StubBaseEnv()
    controller = _controller(base_env, DATASET_SWITCH_SETTLE_TICKS)

    controller.initialize_after_reset(target_mode=0)

    assert len(base_env.gain_calls) == 1
    assert base_env.gain_calls[0]["settle_steps"] == 0


def test_no_settle_ticks_keeps_switch_settle_free() -> None:
    base_env = _StubBaseEnv()
    controller = _controller(base_env, None)
    controller.initialize_after_reset(target_mode=1)
    base_env.gain_calls.clear()

    controller.apply(torch.tensor([0.0]))

    assert len(base_env.gain_calls) == 1
    assert base_env.gain_calls[0]["settle_steps"] == 0


def _episode(n_frames: int, pd_modes: list[float] | None) -> dict:
    data = {
        "observation.state.joint_position": np.stack(
            [np.full(7, step, dtype=np.float32) for step in range(n_frames)]
        ),
        "action.joint_target": np.stack(
            [np.full(7, 10.0 + step, dtype=np.float32) for step in range(n_frames)]
        ),
        "action.gripper_position": np.zeros((n_frames, 1), dtype=np.float32),
    }
    if pd_modes is not None:
        data["action.pd_mode"] = np.asarray(pd_modes, dtype=np.float32).reshape(n_frames, 1)
    return data


def _policy(
    base_env: _StubBaseEnv, episode: dict, *, use_delta_actions: bool = False
) -> ReplayPolicy:
    policy = ReplayPolicy(base_env, torch.device("cpu"), 1, use_delta_actions=use_delta_actions)
    policy._load_episode = lambda setup: episode  # type: ignore[method-assign]
    policy.on_reset([0], [{"source_dataset_path": "/x", "episode_index": 3, "episode_id": "ep_3"}])
    return policy


def test_get_action_carries_recorded_pd_mode_as_ninth_channel() -> None:
    base_env = _StubBaseEnv()
    policy = _policy(base_env, _episode(3, [1.0, 0.0, 0.0]))

    a0 = policy.get_action({})
    a1 = policy.get_action({})

    assert a0.shape == (1, 9)
    torch.testing.assert_close(a0[0, :7], torch.full((7,), 10.0))
    assert a0[0, 7].item() == 1.0  # gripper open -> +1
    assert a0[0, 8].item() == 1.0
    assert a1[0, 8].item() == 0.0


def test_done_indices_fire_after_last_recorded_frame() -> None:
    base_env = _StubBaseEnv()
    policy = _policy(base_env, _episode(2, [1.0, 1.0]))

    policy.get_action({})
    assert policy.done_indices() == []
    policy.get_action({})
    assert policy.done_indices() == [0]


def test_on_reset_rejects_missing_pd_mode() -> None:
    base_env = _StubBaseEnv()
    with pytest.raises(ValueError, match="action.pd_mode"):
        _policy(base_env, _episode(2, None))
    nan_modes = _episode(2, [float("nan"), float("nan")])
    with pytest.raises(ValueError, match="action.pd_mode"):
        _policy(base_env, nan_modes)


def test_divergence_compares_the_row_the_action_led_to() -> None:
    base_env = _StubBaseEnv()
    episode = _episode(3, [1.0, 1.0, 1.0])
    episode["observation.state.joint_position"][1] = base_env.robot.articulation.data.joint_pos[0, :7]
    policy = _policy(base_env, episode)

    policy.get_action({})  # action row 0 -> state row 1
    policy.record_divergence()
    assert policy._joint_pos_errors[0] == [0.0]

    policy.get_action({})  # action row 1 -> state row 2
    policy.record_divergence()
    policy.get_action({})  # terminal row: no recorded state follows it
    policy.record_divergence()

    assert len(policy._joint_pos_errors[0]) == 2
    metrics = policy.episode_metrics(0)
    assert metrics["num_frames"] == 3.0
    assert np.isnan(metrics["profile_switch_frame"])
    assert np.isnan(metrics["pre_switch_joint_pos_error_mean"])


def test_replay_stops_before_the_blanked_terminal_row() -> None:
    """The terminal row (action.is_valid False, zeroed actions) is never replayed."""
    base_env = _StubBaseEnv()
    episode = _episode(3, [1.0, 1.0, 0.0])
    episode["action.joint_target"][2] = 0.0
    episode["action.is_valid"] = np.array([[1.0], [1.0], [0.0]], dtype=np.float32)
    policy = _policy(base_env, episode)

    policy.get_action({})
    policy.record_divergence()
    assert policy.done_indices() == []
    last = policy.get_action({})
    policy.record_divergence()

    assert policy.done_indices() == [0]
    torch.testing.assert_close(last[0, :7], torch.full((7,), 11.0))
    assert last[0, 8].item() == 1.0
    assert len(policy._joint_pos_errors[0]) == 2
    assert np.isnan(policy.episode_metrics(0)["profile_switch_frame"])


def test_record_divergence_skips_envs_that_reset_this_step() -> None:
    base_env = _StubBaseEnv()
    policy = _policy(base_env, _episode(2, [1.0, 1.0]))

    policy.get_action({})
    policy.record_divergence(skip_env_ids=[0])

    assert policy._joint_pos_errors[0] == []


def test_a_settle_hold_freezes_replay_until_the_deferred_row_runs() -> None:
    """Hold steps skip get_action and scoring, as run_eval drives a deferred settle.

    Row 1 switches the PD mode, so the env holds before row 1 runs: the counter
    stays put, the held steps are not scored, and the step that runs row 1 is
    scored against state row 2.
    """
    base_env = _StubBaseEnv()
    episode = _episode(3, [1.0, 0.0, 0.0])
    episode["observation.state.joint_position"][2] = base_env.robot.articulation.data.joint_pos[0, :7]
    policy = _policy(base_env, episode)

    policy.get_action({})  # action row 0 -> state row 1
    policy.record_divergence()
    assert len(policy._joint_pos_errors[0]) == 1 and policy._joint_pos_errors[0][0] > 0.0

    policy.get_action({})  # action row 1 switches the PD mode -> the env holds
    for _ in range(3):
        policy.record_divergence(skip_env_ids=[0])
        assert policy._step[0] == 2
        assert policy.done_indices() == []
    assert len(policy._joint_pos_errors[0]) == 1

    policy.record_divergence()  # the deferred row 1 ran -> state row 2
    assert policy._joint_pos_errors[0][1:] == [0.0]

    policy.get_action({})  # terminal row
    assert policy.done_indices() == [0]
    for _ in range(2):  # a switch on the terminal row: still exhausted after the hold
        policy.record_divergence(skip_env_ids=[0])
    assert policy._step[0] == 3
    assert policy.done_indices() == [0]
    assert len(policy._joint_pos_errors[0]) == 2


def test_phase_slices_use_row_aligned_switch_frame() -> None:
    pre, post = _phase_slices(4, 10)

    assert pre == slice(0, 4)
    assert post == slice(4, 10)


def _gap_episode(
    n_frames: int, gap_row: int, missing_ticks: int, tick_motion: float = 0.01
) -> dict:
    """Arm moving *tick_motion* rad per joint each 0.04 s tick under a held target.

    The *missing_ticks* ticks right before *gap_row* are not recorded.
    """
    ticks = np.arange(n_frames) + np.where(np.arange(n_frames) >= gap_row, missing_ticks, 0)
    episode = _episode(n_frames, [1.0] * n_frames)
    episode["timestamp"] = np.arange(n_frames) * 0.04
    episode["observation.state.joint_position"] = np.stack(
        [np.full(7, tick * tick_motion, dtype=np.float32) for tick in ticks]
    )
    episode["observation.state.joint_velocity"] = np.full(
        (n_frames, 7), tick_motion / 0.04, dtype=np.float32
    )
    episode["action.joint_target"] = np.full((n_frames, 7), 2.0, dtype=np.float32)
    return episode


def test_unrecorded_ticks_are_detected_from_the_recorded_motion() -> None:
    assert _detect_unrecorded_ticks(_gap_episode(8, gap_row=5, missing_ticks=2)) == {5: 2}
    assert _detect_unrecorded_ticks(_gap_episode(8, gap_row=5, missing_ticks=1)) == {5: 1}
    assert _detect_unrecorded_ticks(_gap_episode(8, gap_row=5, missing_ticks=0)) == {}
    assert _detect_unrecorded_ticks(_gap_episode(8, gap_row=5, missing_ticks=4)) == {5: 2}

    stepped = _gap_episode(8, gap_row=5, missing_ticks=2)
    stepped["action.joint_target"][4:] += 0.1
    assert _detect_unrecorded_ticks(stepped) == {}

    new_target = _gap_episode(8, gap_row=5, missing_ticks=2)
    new_target["action.joint_target"][5:] += 0.1
    assert _detect_unrecorded_ticks(new_target) == {}

    pd_switch = _gap_episode(8, gap_row=5, missing_ticks=2)
    pd_switch["action.pd_mode"][5:] = 0.0
    assert _detect_unrecorded_ticks(pd_switch) == {}

    barely_moving = _gap_episode(8, gap_row=5, missing_ticks=2, tick_motion=1e-4)
    assert _detect_unrecorded_ticks(barely_moving) == {}

    assert _detect_unrecorded_ticks(_episode(8, [1.0] * 8)) == {}


def test_unrecorded_ticks_replay_as_a_hold_of_the_previous_row() -> None:
    """Rows 0 and 1 are one tick apart; row 2 was recorded two ticks after that tick.

    The step applying row 1 is scored only after the hold re-applies row 1 for
    the two skipped ticks; row 2 then runs as the deferred action and is scored
    against state row 3, so every action row keeps exactly one score.
    """
    base_env = _StubBaseEnv()
    episode = _gap_episode(5, gap_row=2, missing_ticks=2)
    policy = _policy(base_env, episode)
    sim_joint_pos = base_env.robot.articulation.data.joint_pos
    available = torch.tensor([True])

    def sim_reaches(row: int) -> None:
        sim_joint_pos[0, :7] = torch.from_numpy(episode["observation.state.joint_position"][row])

    policy.get_action({})  # action row 0 -> state row 1
    assert policy.pop_gap_holds(available) is None
    sim_reaches(1)
    policy.record_divergence()
    assert policy._joint_pos_errors[0] == [0.0]

    row1 = policy.get_action({})  # action row 1 -> state row 2, three ticks later
    assert policy.pop_gap_holds(available) is None
    policy.record_divergence()  # one tick in: the score waits for the hold
    assert policy._joint_pos_errors[0] == [0.0]

    policy.get_action({})  # action row 2 follows the skipped ticks
    holds = policy.pop_gap_holds(available)
    assert holds is not None
    env_mask, ticks, hold_action = holds
    assert env_mask.tolist() == [True]
    assert ticks.tolist() == [2]
    torch.testing.assert_close(hold_action[0], row1[0, :8])
    assert policy.pop_gap_holds(available) is None

    policy.record_divergence(holding_env_ids=[0])
    assert policy._joint_pos_errors[0] == [0.0]
    sim_reaches(2)
    policy.record_divergence(holding_env_ids=[0])  # last hold step: row 1 -> state row 2
    assert policy._joint_pos_errors[0] == [0.0, 0.0]

    sim_reaches(3)
    policy.record_divergence()  # the deferred row 2 ran -> state row 3
    assert policy._joint_pos_errors[0] == [0.0, 0.0, 0.0]
    assert policy.episode_metrics(0)["unrecorded_ticks"] == 2.0


def test_a_gap_on_an_env_already_holding_is_dropped() -> None:
    base_env = _StubBaseEnv()
    policy = _policy(base_env, _gap_episode(5, gap_row=2, missing_ticks=2))

    for _ in range(2):
        policy.get_action({})
        policy.record_divergence()
    policy.get_action({})  # row 2 lands while a PD-switch settle holds the env

    assert policy.pop_gap_holds(torch.tensor([False])) is None
    assert policy._gap_score_row[0] is None
    assert policy.episode_metrics(0)["unrecorded_ticks"] == 0.0


def test_delta_replays_leave_unrecorded_ticks_unfilled() -> None:
    base_env = _StubBaseEnv()
    episode = _gap_episode(5, gap_row=2, missing_ticks=2)
    episode["action"] = np.zeros((5, 8), dtype=np.float32)
    policy = _policy(base_env, episode, use_delta_actions=True)

    for _ in range(3):
        policy.get_action({})
        assert policy.pop_gap_holds(torch.tensor([True])) is None
        policy.record_divergence()

    assert len(policy._joint_pos_errors[0]) == 3
    assert "unrecorded_ticks" not in policy.episode_metrics(0)


class _FakeRigidData:
    def __init__(self, num_envs: int) -> None:
        self.root_pose_w = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self.root_pose_w[:, :3]

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self.root_pose_w[:, 3:]


class _FakeRigidObject:
    def __init__(self, num_envs: int) -> None:
        self.data = _FakeRigidData(num_envs)
        self.velocity_writes: list[torch.Tensor] = []

    def write_root_pose_to_sim(self, root_pose: torch.Tensor, env_ids: torch.Tensor) -> None:
        self.data.root_pose_w[env_ids] = root_pose

    def write_root_velocity_to_sim(
        self, root_velocity: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.velocity_writes.append(root_velocity.clone())


class _FakeArticulation:
    def __init__(self, num_envs: int) -> None:
        self.data = SimpleNamespace(joint_pos=torch.zeros(num_envs, 8))

    def write_joint_state_to_sim(
        self, joint_pos: torch.Tensor, joint_vel: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.data.joint_pos[env_ids] = joint_pos

    def set_joint_position_target(self, target: torch.Tensor, env_ids: torch.Tensor) -> None:
        pass

    def set_joint_velocity_target(self, target: torch.Tensor, env_ids: torch.Tensor) -> None:
        pass


class _FakeQueueEnv:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.num_envs = 2
        self.robot = SimpleNamespace(articulation=_FakeArticulation(self.num_envs))
        self.scene = SimpleNamespace(
            env_origins=torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]),
            rigid_objects={
                name: _FakeRigidObject(self.num_envs) for name in ("block_red", "block_blue")
            },
        )


@pytest.fixture
def queue_env(monkeypatch: pytest.MonkeyPatch) -> _FakeQueueEnv:
    # apply_setup imports the Isaac-backed friction helper even for setups that record no friction.
    monkeypatch.setitem(
        sys.modules,
        "toyblocks.utils.randomization_utils",
        SimpleNamespace(set_rigid_body_friction=None),
    )
    return _FakeQueueEnv()


def _block_setup(*names: str) -> dict:
    return {
        "objects": {
            name: {"position": [0.5, 0.1 * k, 0.02], "quaternion": [0.0, 1.0, 0.0, 0.0]}
            for k, name in enumerate(names)
        }
    }


def test_placed_blocks_sit_out_the_next_substep_in_the_graveyard(queue_env: _FakeQueueEnv) -> None:
    base_env = queue_env
    red = base_env.scene.rigid_objects["block_red"]
    apply_setup(base_env, 1, _block_setup("block_red"))
    placed = red.data.root_pose_w.clone()

    park_placed_blocks(base_env)
    park_placed_blocks(base_env)  # nothing new to park

    assert torch.equal(red.data.root_pose_w[0], placed[0])
    assert red.data.root_pose_w[1, :3].tolist() == [11.0, 0.0, -5.0]
    blue = base_env.scene.rigid_objects["block_blue"]
    assert blue.data.root_pose_w[1, :3].tolist() == [10.0, 0.0, -5.0]  # unused-block row

    restore_placed_blocks(base_env)
    red.velocity_writes.clear()
    restore_placed_blocks(base_env)  # already restored

    assert torch.equal(red.data.root_pose_w, placed)
    assert red.velocity_writes == []


def test_a_second_reset_before_a_step_replaces_the_blocks_to_park(queue_env: _FakeQueueEnv) -> None:
    base_env = queue_env
    apply_setup(base_env, 0, _block_setup("block_red", "block_blue"))
    apply_setup(base_env, 0, _block_setup("block_red"))
    blue = base_env.scene.rigid_objects["block_blue"]
    unused = blue.data.root_pose_w.clone()

    park_placed_blocks(base_env)

    red = base_env.scene.rigid_objects["block_red"]
    assert torch.equal(blue.data.root_pose_w, unused)
    assert red.data.root_pose_w[0, :3].tolist() == [1.0, 0.0, -5.0]


_NAN = float("nan")


def _diagnostic_frames(n_rows: int = 4) -> pd.DataFrame:
    """Rows whose every column holds the row index; the last row is terminal."""
    frames: dict[str, list[object]] = {
        "action.is_valid": [np.array([row < n_rows - 1]) for row in range(n_rows)],
        "action.policy": [np.full(8, row, dtype=np.float32) for row in range(n_rows)],
        "observation.state.joint_position": [
            np.full(7, row, dtype=np.float32) for row in range(n_rows)
        ],
        "observation.state.gripper_position": [
            np.array([row], dtype=np.float32) for row in range(n_rows)
        ],
        "observation.state.ee_position": [
            np.full(6, row, dtype=np.float32) for row in range(n_rows)
        ],
        "observation.state.current_stiffness": [
            np.full(8, row, dtype=np.float32) for row in range(n_rows)
        ],
        "action.joint_target": [np.full(7, row, dtype=np.float32) for row in range(n_rows)],
    }
    return pd.DataFrame(frames)


def _series(rollout: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "state": rollout["joint_positions"][:, 0],
        "command": rollout["action_targets"][:, 0],
        "gain": rollout["joint_stiffness"][:, 0],
    }


@pytest.mark.parametrize("replay_recording", [True, False])
def test_offline_diagnostics_keep_marked_rows_at_their_own_step(replay_recording: bool) -> None:
    rollout = _rollout_from_frames(
        _diagnostic_frames(),
        env_spacing=None,
        label=0,
        row_layout_marked=True,
        replay_recording=replay_recording,
    )

    series = _series(rollout)
    np.testing.assert_array_equal(series["state"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(series["command"], [0.0, 1.0, 2.0, _NAN])
    np.testing.assert_array_equal(series["gain"], [0.0, 1.0, 2.0, 3.0])


def test_offline_diagnostics_realign_legacy_replay_rows() -> None:
    rollout = _rollout_from_frames(
        _diagnostic_frames(), env_spacing=None, label=0, replay_recording=True
    )

    series = _series(rollout)
    np.testing.assert_array_equal(series["state"], [_NAN, 0.0, 1.0])
    np.testing.assert_array_equal(series["command"], [0.0, 1.0, _NAN])
    np.testing.assert_array_equal(series["gain"], [0.0, 1.0, _NAN])


def test_offline_diagnostics_realign_legacy_policy_eval_rows() -> None:
    """Policy evals record ``action.joint_target`` too; only the launch flag tells them apart."""
    rollout = _rollout_from_frames(
        _diagnostic_frames(), env_spacing=None, label=0, replay_recording=False
    )

    series = _series(rollout)
    np.testing.assert_array_equal(series["state"], [_NAN, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(series["command"], [_NAN, 0.0, 1.0, 2.0])
    np.testing.assert_array_equal(series["gain"], [0.0, 1.0, 2.0, _NAN])


def test_replay_runs_are_recognised_from_launch_params(tmp_path) -> None:
    assert _is_replay_run(tmp_path, {"replay": True, "policy": "replay"})
    assert not _is_replay_run(tmp_path, {"replay": False, "policy": "maniflow"})
    assert _is_replay_run(tmp_path, {"policy": "replay"})
    assert not _is_replay_run(tmp_path, {})


def test_replay_runs_without_launch_params_are_recognised_by_their_metrics(tmp_path) -> None:
    (tmp_path / "eval_episodes.json").write_text('[{"sim_env_idx": 0}]')
    assert not _is_replay_run(tmp_path, {})
    (tmp_path / "eval_episodes.json").write_text('[{"sim_env_idx": 0, "replay_num_frames": 145}]')
    assert _is_replay_run(tmp_path, {})


def test_row_layout_marker_is_detected(tmp_path) -> None:
    assert not _has_row_layout_marker(tmp_path)
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"fps": 15}')
    assert not _has_row_layout_marker(tmp_path)
    (tmp_path / "meta" / "info.json").write_text(
        '{"fps": 15, "toyblocks_row_layout": {"layout": "state_then_action", "version": 2}}'
    )
    assert _has_row_layout_marker(tmp_path)


def test_row_layout_marker_file_from_older_recordings_is_detected(tmp_path) -> None:
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "info.json").write_text('{"fps": 15}')
    (tmp_path / "meta" / "toyblocks_row_layout.json").write_text("{}")
    assert _has_row_layout_marker(tmp_path)


def test_offline_diagnostics_use_replay_ground_truth_episode_indices(tmp_path) -> None:
    (tmp_path / "replay_results.json").write_text(
        '[{"gt_dataset_episode_idx": 4}, {"gt_dataset_episode_idx": 7}]'
    )

    specs = _load_episode_specs(tmp_path, tmp_path / "lerobot_dataset")

    assert [(spec.label, spec.output_lerobot_episode_index) for spec in specs] == [(4, 0), (7, 1)]
