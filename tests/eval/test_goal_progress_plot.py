"""CPU tests for the goal-predicate trace and its goal_progress.png plot.

Covers the ``RecordMetrics`` trace (episode clock, OR within hold steps, reset
between episodes), the ``eval_episodes.json`` fields it feeds, the offline plot
script's goal-progress pass, and the W&B episodes-table column.
"""

from __future__ import annotations

import dataclasses
import json
import math
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

try:
    import toyblocks.tasks.base_env  # noqa: F401
except ImportError:
    # RecordMetrics only uses BaseEnv as an annotation; without Isaac Lab, stand it in.
    _base_env_stub = types.ModuleType("toyblocks.tasks.base_env")
    _base_env_stub.BaseEnv = gym.Env  # type: ignore[attr-defined]
    sys.modules["toyblocks.tasks.base_env"] = _base_env_stub

from toyblocks.utils.wandb_logging import _goal_progress_fields
from toyblocks.wrappers.record_metrics import RecordMetrics
from scripts import wandb_consolidate
from scripts import plot_diagnostics_from_eval_run as plot_script
from scripts.plot_diagnostics_from_eval_run import (
    GOAL_PROGRESS_FILENAME,
    Args,
    GoalPlotJob,
    _goal_plot_jobs,
    _render_goal_progress,
    expand_trace,
    goal_final_count,
    goal_plot_label,
    goal_slot_names,
    goal_step_dt,
    load_settle_terms,
    main,
    trace_true_spans,
)

STACK_LABELS: list[str] = [
    "(on_table block_3 workspace_table)",
    "(on block_5 block_3)",
    "(ori_of block_q o2)",
]


class _ClockEnv(gym.Env):
    """Vectorized stand-in for BaseEnv: an episode clock, scripted stats and done masks.

    ``hold`` envs keep their clock for the step, as eval's hold steps do (the
    eval decrements ``episode_length_buf`` before the env increments it).
    """

    def __init__(self, num_envs: int, max_episode_length: int) -> None:
        self.num_envs = num_envs
        self.device = torch.device("cpu")
        self.max_episode_length = max_episode_length
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.observation_space = gym.spaces.Box(-1.0, 1.0, (1,))
        self.action_space = gym.spaces.Box(-1.0, 1.0, (1,))

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.episode_length_buf.zero_()
        return torch.zeros(self.num_envs, 1), {}

    def scripted_step(
        self, values: list[int], hold: list[bool], done: list[bool]
    ) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, dict]:
        self.episode_length_buf += 1
        self.episode_length_buf[torch.tensor(hold)] -= 1
        stats = {
            "goal_conj_0": torch.tensor(values, dtype=torch.float32),
            "goal_success": torch.zeros(self.num_envs),
            "other_metric": torch.ones(self.num_envs),
        }
        done_mask = torch.tensor(done)
        if done_mask.any():
            info: dict = {
                "_final_info": done_mask.clone(),
                "final_info": {
                    "stats": stats,
                    "episode": {"episode_len": self.episode_length_buf.clone()},
                },
                "stats": {},
            }
            self.episode_length_buf[done_mask] = 0
        else:
            info = {"stats": stats}
        no_term = torch.zeros(self.num_envs, dtype=torch.bool)
        return torch.zeros(self.num_envs, 1), 0.0, done_mask, no_term, info

    def step(self, action):
        values, hold, done = action
        return self.scripted_step(values, hold, done)


def test_record_metrics_trace_follows_episode_clock() -> None:
    env = _ClockEnv(num_envs=2, max_episode_length=10)
    wrapper = RecordMetrics(env)
    wrapper.reset()
    no, yes = False, True

    # env 1 is parked (clock held at 0) on the first step; its true value is not part of any episode.
    _, _, _, _, info = wrapper.step(([0, 1], [no, yes], [no, no]))
    # env 0 step 2 is true, then a hold step (still step 2) reads false: the OR keeps true.
    wrapper.step(([1, 0], [no, no], [no, no]))
    wrapper.step(([0, 0], [yes, no], [no, no]))
    wrapper.step(([0, 1], [no, no], [no, no]))
    _, _, _, _, info = wrapper.step(([1, 1], [no, no], [yes, no]))

    trace = info["final_info"]["goal_trace"]
    assert set(trace) == {0}
    assert trace[0]["goal_conj_0"] == [[1, 0], [2, 1], [3, 0], [4, 1]]
    assert trace[0]["goal_success"] == [[1, 0]]
    assert "other_metric" not in trace[0]
    assert info["final_info"]["episode"]["goal_conj_0/first_true_step"][0].item() == 2.0

    # env 0's next episode must start from a cleared trace (its old step 2 was true).
    wrapper.step(([1, 0], [no, no], [no, no]))
    _, _, _, _, info = wrapper.step(([0, 0], [no, no], [yes, yes]))
    trace = info["final_info"]["goal_trace"]
    assert trace[0]["goal_conj_0"] == [[1, 1], [2, 0]]
    # env 1 ran steps 1..6 reading 0, 0, 1, 1, 0, 0 after its parked step.
    assert trace[1]["goal_conj_0"] == [[1, 0], [3, 1], [5, 0]]


def test_record_metrics_csv_rows_carry_episode_clock(tmp_path: Path) -> None:
    env = _ClockEnv(num_envs=1, max_episode_length=10)
    wrapper = RecordMetrics(env, output_dir=str(tmp_path))
    wrapper.reset()
    no, yes = False, True
    wrapper.step(([0], [no], [no]))
    wrapper.step(([1], [yes], [no]))
    wrapper.step(([1], [no], [no]))
    wrapper.step(([1], [no], [yes]))

    rows = pd.read_csv(tmp_path / "step_metrics" / "env0_episode_0000.csv")
    assert rows["step"].tolist() == [0, 1, 2, 3]
    # The hold step repeats clock 1; the done row takes the pre-reset episode_len.
    assert rows["episode_step"].tolist() == [1, 1, 2, 3]


def test_record_metrics_skips_trace_when_buffer_too_large(monkeypatch) -> None:
    import toyblocks.wrappers.record_metrics as record_metrics

    monkeypatch.setattr(record_metrics, "_TRACE_MAX_BYTES", 1)
    env = _ClockEnv(num_envs=1, max_episode_length=10)
    wrapper = RecordMetrics(env)
    wrapper.reset()
    _, _, _, _, info = wrapper.step(([1], [False], [True]))
    assert "goal_trace" not in info["final_info"]
    assert info["final_info"]["episode"]["goal_conj_0/final"][0].item() == 1.0


def test_goal_progress_fields_attach_traces_and_drop_padding() -> None:
    metrics: dict = {}
    trace: dict[str, list[list[int]]] = {}
    for c in range(4):
        for stat in ("initial", "first_true_step", "final", "max", "mean"):
            metrics[f"goal_conj_{c}/{stat}"] = 1.0
        trace[f"goal_conj_{c}"] = [[1, 1]]
    trace["goal_conj_1"] = [[1, 0], [7, 1]]
    fields = _goal_progress_fields(STACK_LABELS, metrics, trace)
    progress = fields["goal_progress"]
    assert [entry["metric_key"] for entry in progress] == ["goal_conj_0", "goal_conj_1", "goal_conj_2"]
    assert progress[1]["trace"] == [[1, 0], [7, 1]]
    assert fields["num_goal_conjuncts"] == 3


def test_expand_trace_and_spans_handle_switch_off_and_blips() -> None:
    trace = [[1, 0], [3, 1], [5, 0], [7, 1], [8, 0]]
    assert expand_trace(trace, 9).tolist() == [0, 0, 1, 1, 0, 0, 1, 0, 0]
    assert trace_true_spans(trace, 9) == [(3, 2), (7, 1)]
    # A trace running to the end fills through the last step.
    assert expand_trace([[1, 0], [4, 1]], 5).tolist() == [0, 0, 0, 1, 1]
    assert trace_true_spans([[1, 0], [4, 1]], 5) == [(4, 2)]


def test_final_count_follows_stored_final_over_trace() -> None:
    progress = [
        {"final": 0.0, "trace": [[1, 0], [5, 1]]},
        {"final": 1.0, "trace": [[1, 1]]},
    ]
    dense = np.stack([expand_trace(entry["trace"], 5) for entry in progress])
    assert int(dense[:, -1].sum()) == 2
    assert goal_final_count(progress, dense) == 1
    progress[0]["final"] = None
    assert goal_final_count(progress, dense) == 2


def test_step_dt_prefers_record_then_fps() -> None:
    assert goal_step_dt({"step_dt_s": 0.04}, fallback_fps=10.0) == 0.04
    assert goal_step_dt({}, fallback_fps=25.0) == 0.04
    assert goal_step_dt({"step_dt_s": None}, fallback_fps=None) is None


def _record(
    traces: list[list[list[int]]],
    *,
    episode_len: int,
    success_trace: list[list[int]],
    labels: list[str] = STACK_LABELS,
    **extra: object,
) -> dict:
    progress = []
    for c, (label, trace) in enumerate(zip(labels, traces)):
        first = next((step for step, value in trace if value), math.nan)
        progress.append(
            {
                "index": c,
                "metric_key": f"goal_conj_{c}",
                "predicate": label,
                "initial": trace[0][1],
                "first_true_step": first,
                "final": trace[-1][1],
                "trace": trace,
            }
        )
    return {
        "episode_len": episode_len,
        "goal_progress": progress,
        "goal_success_trace": success_trace,
        "step_dt_s": 0.04,
        **extra,
    }


def test_render_replay_and_policy_style_records(tmp_path: Path) -> None:
    replay = _record(
        [[[1, 1]], [[1, 0], [2792, 1]], [[1, 0], [1921, 1], [2400, 0], [2401, 1], [2402, 0]]],
        episode_len=3400,
        success_trace=[[1, 0]],
    )
    # Policy eval with terminate_on_success: ends on the success step.
    policy = _record(
        [[[1, 1]], [[1, 0], [300, 1]], [[1, 0], [420, 1]]],
        episode_len=431,
        success_trace=[[1, 0], [430, 1]],
    )
    for label, rec in ((2, replay), (44, policy)):
        out_dir = tmp_path / f"episode_{label}"
        message = _render_goal_progress(GoalPlotJob(label=label, record=rec, output_dir=str(out_dir)))
        assert (out_dir / GOAL_PROGRESS_FILENAME).is_file()
        assert "/3" in message


def test_shorter_goal_plots_only_its_own_predicates(tmp_path: Path) -> None:
    rec = _record([[[1, 1]], [[1, 1]]], episode_len=10, success_trace=[[1, 0], [10, 1]])
    message = _render_goal_progress(GoalPlotJob(label=0, record=rec, output_dir=str(tmp_path)))
    assert "2/2" in message


def _write_settle_csv(
    path: Path, lin: list[list[float]], *, episode_steps: list[int] | None = None
) -> None:
    """A step-metrics CSV with ``goal_obj_<k>`` speeds (ang = 10 x lin) and the finger."""
    data: dict[str, list[float] | list[int]] = {"step": list(range(len(lin)))}
    if episode_steps is not None:
        data["episode_step"] = episode_steps
    for k in range(len(lin[0])):
        data[f"goal_obj_{k}_lin_vel"] = [row[k] for row in lin]
        data[f"goal_obj_{k}_ang_vel"] = [10.0 * row[k] for row in lin]
    data["goal_finger_pos"] = [0.1] * len(lin)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False)


def test_load_settle_terms_places_rows_on_episode_clock(tmp_path: Path) -> None:
    csv_path = tmp_path / "ep.csv"
    # A parked row at clock 0, then steps 1..3 with step 2 held for two rows.
    _write_settle_csv(
        csv_path, [[9.0, 0.0], [0.1, 0.0], [0.2, 0.0], [0.3, 0.0], [0.4, 0.0]],
        episode_steps=[0, 1, 2, 2, 3],
    )
    series = load_settle_terms(csv_path, n_steps=3, n_objects=2)
    assert series is not None
    assert series.steps.tolist() == [1, 2, 2, 3]
    assert series.lin.shape == (4, 2)
    assert series.lin[:, 0].tolist() == [0.1, 0.2, 0.3, 0.4]
    # Without goal_objects the all-zero padding slot is dropped.
    series = load_settle_terms(csv_path, n_steps=3)
    assert series is not None and series.lin.shape == (4, 1)
    # The CSV must end on the episode's last step.
    assert load_settle_terms(csv_path, n_steps=4, n_objects=2) is None
    assert load_settle_terms(csv_path, n_steps=3, n_objects=3) is None


def test_load_settle_terms_falls_back_to_row_count(tmp_path: Path) -> None:
    csv_path = tmp_path / "old.csv"
    _write_settle_csv(csv_path, [[0.1], [0.2], [0.3]])
    series = load_settle_terms(csv_path, n_steps=3)
    assert series is not None
    assert series.steps.tolist() == [1, 2, 3]
    assert np.allclose(series.ang[:, 0], [1.0, 2.0, 3.0])
    # More rows than steps means hold steps, which step + 1 can't place.
    assert load_settle_terms(csv_path, n_steps=2) is None


def test_goal_slot_names_prefer_record_then_predicate_tokens() -> None:
    rec = _record([[[1, 1]], [[1, 1]], [[1, 1]]], episode_len=2, success_trace=[[1, 0]])
    assert goal_slot_names({**rec, "goal_objects": ["block_a", "block_b"]}, 2) == ["block_a", "block_b"]
    assert goal_slot_names(rec, 3) == ["block_3", "block_5", "block_q"]
    assert goal_slot_names(rec, 2) == ["slot 0", "slot 1"]


def test_settle_thresholds_mirror_the_gate() -> None:
    from toyblocks.tamp.blocks import pddl_goal_tensor_conversions as gate

    assert plot_script._SETTLED_LIN_VEL == gate._SETTLED_LIN_VEL
    assert plot_script._SETTLED_ANG_VEL == gate._SETTLED_ANG_VEL
    assert plot_script._GRIPPER_OPEN_BELOW == gate._GRIPPER_OPEN_BELOW


def test_render_gate_rows_with_and_without_step_metrics(tmp_path: Path) -> None:
    rec = _record(
        [[[1, 1]], [[1, 0], [3, 1]], [[1, 1]]],
        episode_len=4,
        success_trace=[[1, 0], [4, 1]],
        goal_blocks_settled_trace=[[1, 1], [2, 0], [4, 1]],
        goal_gripper_open_trace=[[1, 0], [3, 1]],
        goal_objects=["block_3", "block_5", "block_q"],
        sim_env_idx=0,
        episode_idx_for_sim_env=0,
    )
    _write_run(tmp_path, [rec])
    jobs, _ = _goal_plot_jobs(tmp_path, tmp_path / "out", None)
    assert jobs[0].step_metrics_csv is None
    message = _render_goal_progress(jobs[0])
    assert "3/3" in message

    _write_settle_csv(
        tmp_path / "step_metrics" / "env0_episode_0000.csv",
        [[0.0, 0.01, 0.2], [0.05, 0.01, 0.02], [0.01, 0.0, 0.0], [0.0, 0.0, 0.0]],
        episode_steps=[1, 2, 3, 4],
    )
    jobs, _ = _goal_plot_jobs(tmp_path, tmp_path / "out", None)
    assert jobs[0].step_metrics_csv is not None
    _render_goal_progress(jobs[0])
    assert (tmp_path / "out" / "episode_0" / GOAL_PROGRESS_FILENAME).is_file()
    # A CSV that doesn't match the episode only drops the speed panels.
    mismatched = dataclasses.replace(jobs[0], record={**rec, "episode_len": 9})
    _render_goal_progress(mismatched)


def _write_run(run_dir: Path, records: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "eval_episodes.json").write_text(json.dumps(records))


def test_run_without_lerobot_still_gets_goal_plots(tmp_path: Path) -> None:
    traced = _record(
        [[[1, 1]], [[1, 0], [5, 1]], [[1, 0]]],
        episode_len=8,
        success_trace=[[1, 0]],
        source_dataset_episode_idx=7,
    )
    untraced = {"source_dataset_episode_idx": 9, "episode_len": 8}
    _write_run(tmp_path, [traced, untraced])
    main(Args(eval_dir=str(tmp_path)))
    out = tmp_path / "control_diagnostics_offline"
    assert (out / "episode_7" / GOAL_PROGRESS_FILENAME).is_file()
    assert not (out / "episode_9").exists()


def test_goal_progress_only_ignores_lerobot_data(tmp_path: Path) -> None:
    # An unusable lerobot_dataset/ would make the control pass raise.
    rec = _record([[[1, 1]], [[1, 1]], [[1, 0]]], episode_len=4, success_trace=[[1, 0]])
    _write_run(tmp_path, [rec])
    (tmp_path / "lerobot_dataset").mkdir()
    main(Args(eval_dir=str(tmp_path), goal_progress_only=True))
    assert (tmp_path / "control_diagnostics_offline" / "episode_0" / GOAL_PROGRESS_FILENAME).is_file()


def test_consolidate_label_matches_plot_script() -> None:
    records = [
        {"source_dataset_episode_idx": 3, "output_lerobot_episode_index": 0},
        {"source_dataset_episode_idx": None, "output_lerobot_episode_index": 5},
        {},
    ]
    for position, rec in enumerate(records):
        assert wandb_consolidate._goal_plot_label(rec, position) == goal_plot_label(rec, position)


def test_episodes_table_goal_progress_column(tmp_path: Path) -> None:
    rec = _record([[[1, 1]], [[1, 1]], [[1, 0]]], episode_len=4, success_trace=[[1, 0]])
    _render_goal_progress(GoalPlotJob(label=0, record=rec, output_dir=str(tmp_path / "episode_0")))
    with_plot = {"prompt": "stack", "goal_progress_plot_path": str(tmp_path / "episode_0" / GOAL_PROGRESS_FILENAME)}
    missing = {"prompt": "stack", "goal_progress_plot_path": str(tmp_path / "episode_1" / GOAL_PROGRESS_FILENAME)}

    table = wandb_consolidate._build_episodes_table([with_plot, missing], video_fps=25)
    assert "goal_progress" in table.columns
    column = table.columns.index("goal_progress")
    assert table.data[0][column] is not None
    assert table.data[1][column] is None

    table = wandb_consolidate._build_episodes_table([missing], video_fps=25)
    assert "goal_progress" not in table.columns
