"""RecordMetrics wrapper for vectorized Isaac Lab environments.

Accumulates per-step stats returned by ``evaluate()`` (via the ``stats`` key
in the info dict) into episode-level summaries.  When an episode ends the
wrapper injects ``{key}/mean``, ``{key}/min``, ``{key}/max``, and
``{key}/final`` into ``info["final_info"]["episode"]`` so that downstream
consumers (evaluate scripts, loggers, …) see them automatically.

Goal-progress stats (``goal_conj_<c>``, ``goal_success`` and the settle-gate
limbs ``goal_blocks_settled`` / ``goal_gripper_open``, see
:data:`_PROGRESS_STAT_PREFIXES`) also get ``{key}/initial`` and
``{key}/first_true_step``, timed with the env's episode clock
(``episode_length_buf``) rather than a count of wrapper steps: eval hides hold
steps (PD-switch settles, chunk-boundary waits, wave parking) from that clock,
so these line up with ``first_success_step`` and ``episode_len``.

The same progress stats are also traced step by step on the device, indexed by
that clock. Hold steps share one clock step and are OR-ed together, so a value
that was true at any point during a hold shows as true, matching
``first_true_step``. When an episode ends the trace is encoded as change points
and attached as ``info["final_info"]["goal_trace"] = {env_idx: {key:
[[step, value], ...]}}``; the first pair is always ``[1, initial]``.

When *output_dir* is provided, per-step stats are also written to individual
CSV files under ``<output_dir>/step_metrics/env{idx}_episode_NNNN.csv``, where
``{idx}`` is the sub-environment index and ``NNNN`` is a per-env episode
counter.  The ``should_save_episode`` / ``should_save_in_flight_episode``
callbacks let queue-driven eval drop the CSVs of "phantom" episodes the same
way :class:`~blocksuite.wrappers.record_episode.RecordEpisode` drops their
trajectories and videos.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Callable
from pathlib import Path
from typing import Optional

import torch
import gymnasium as gym

from blocksuite.tasks.base_env import BaseEnv

_PROGRESS_STAT_PREFIXES: tuple[str, ...] = (
    "goal_conj_",
    "goal_success",
    "goal_blocks_settled",
    "goal_gripper_open",
)
"""Boolean (0/1) stats that also get ``/initial`` and ``/first_true_step`` summaries."""

_TRACE_MAX_BYTES: int = 256 * 1024**2
"""Largest goal-trace buffer allocated; above it the trace is skipped with a warning."""


class RecordMetrics(gym.Wrapper):
    """Accumulate per-step ``evaluate()`` stats into episode-level summaries.

    Sits in the wrapper stack between the environment (or observation
    wrappers) and ``RecordEpisode``.  It does **not** modify observations,
    rewards, or done signals – only the ``info`` dict.

    The wrapper handles the two shapes of the info dict produced by
    :class:`BaseEnv`:

    * **No envs done** – ``info`` is the raw ``evaluate()`` dict; per-step
      stats live at ``info["stats"]``.
    * **Envs done (auto-reset)** – ``info`` wraps the pre-reset evaluate in
      ``info["final_info"]``.  Pre-reset stats are at
      ``info["final_info"]["stats"]``; post-reset stats at ``info["stats"]``
      are ignored (they belong to the new episode).

    Args:
        env: The environment to wrap.
        output_dir: If provided, per-step stats are saved as CSV files under
            ``<output_dir>/step_metrics/``.
        should_save_episode: Optional ``f(env_idx) -> bool`` checked when a
            finished episode's CSV is flushed. ``False`` drops the rows without
            writing and without bumping the per-env counter (written files stay
            sequential). Used by queue eval to discard "phantom" episodes
            (queue exhausted before this env got a setup).
        should_save_in_flight_episode: Same, for the ``_partial`` CSV of an
            episode still running at :meth:`close`.
    """

    def __init__(
        self,
        env: gym.Env,
        output_dir: Optional[str] = None,
        should_save_episode: Optional[Callable[[int], bool]] = None,
        should_save_in_flight_episode: Optional[Callable[[int], bool]] = None,
    ) -> None:
        super().__init__(env)

        """Running sum per env, used to compute the episode mean."""
        self._stats_sum: dict[str, torch.Tensor] = {}
        
        """Running element-wise minimum per env across the episode."""
        self._stats_min: dict[str, torch.Tensor] = {}

        """Running element-wise maximum per env across the episode."""
        self._stats_max: dict[str, torch.Tensor] = {}

        """Most recent value per env (last step before episode end)."""
        self._stats_final: dict[str, torch.Tensor] = {}

        """Progress stats only: value on the episode's first counted step (clock == 1), NaN before."""
        self._stats_initial: dict[str, torch.Tensor] = {}

        """Progress stats only: episode clock on the first counted step the stat was true, NaN until then."""
        self._stats_first_true: dict[str, torch.Tensor] = {}

        """Number of steps accumulated per env since the last reset."""
        self._stats_count = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        """Progress stats traced per step, in the column order of ``_trace``."""
        self._trace_keys: list[str] = []

        """``uint8[num_envs, max_episode_length + 2, len(_trace_keys)]``: the
        progress stats at each episode-clock step (index 0 unused), allocated
        the first time progress stats appear."""
        self._trace: Optional[torch.Tensor] = None

        """Set when the trace buffer would be too large; no trace is recorded."""
        self._trace_disabled = False

        # --- CSV per-step recording ---
        self._output_dir = output_dir
        self._step_metrics_dir: Optional[str] = None
        if output_dir is not None:
            self._step_metrics_dir = os.path.join(output_dir, "step_metrics")
            Path(self._step_metrics_dir).mkdir(parents=True, exist_ok=True)

        self._episode_counters: list[int] = [0 for _ in range(self.num_envs)]
        self._csv_buffers: list[list[dict[str, float | int | bool]]] = [
            [] for _ in range(self.num_envs)
        ]

        self.should_save_episode: Optional[Callable[[int], bool]] = should_save_episode
        self.should_save_in_flight_episode: Optional[Callable[[int], bool]] = (
            should_save_in_flight_episode
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped  # type: ignore[return-value]

    @property
    def num_envs(self) -> int:
        return self.base_env.num_envs

    @property
    def device(self) -> torch.device:
        return self.base_env.device

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------

    def reset(self, *args, **kwargs):
        # Replay-style callers reset the env directly between episodes rather
        # than letting `step()` observe a `terminated`/`truncated` env and
        # take the done-path flush below, so an in-flight episode's buffered
        # rows must be flushed here too or they're silently dropped when the
        # buffer is cleared for the next episode.
        if self._step_metrics_dir is not None:
            for env_idx in range(self.num_envs):
                if self._csv_buffers[env_idx]:
                    self._flush_csv(
                        env_idx,
                        suffix="_partial",
                        save=self._should_save_in_flight_episode(env_idx),
                    )

        obs, info = super().reset(*args, **kwargs)
        self._reset_accumulators(
            torch.arange(self.num_envs, device=self.device)
        )
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if "final_info" in info:
            pre_reset_stats = info["final_info"].get("stats", {})
            clock = self._episode_clock(info)
            self._accumulate(pre_reset_stats, clock)
            self._buffer_csv_row(pre_reset_stats, clock)

            done_mask = info["_final_info"]
            done_ids = done_mask.nonzero(as_tuple=False).squeeze(-1)

            if "episode" in info["final_info"]:
                summaries = self._compute_summaries()
                info["final_info"]["episode"].update(summaries)
                goal_trace = self._encode_trace(done_ids, clock)
                if goal_trace:
                    info["final_info"]["goal_trace"] = goal_trace

            if self._step_metrics_dir is not None:
                for env_idx in done_ids.tolist():
                    self._flush_csv(
                        env_idx, save=self._should_save_env_episode(env_idx)
                    )

            self._reset_accumulators(done_ids)
        else:
            stats = info.get("stats", {})
            clock = self._episode_clock(info)
            self._accumulate(stats, clock)
            self._buffer_csv_row(stats, clock)

        return obs, reward, terminated, truncated, info

    def close(self):
        if self._step_metrics_dir is not None:
            for env_idx in range(self.num_envs):
                if self._csv_buffers[env_idx]:
                    self._flush_csv(
                        env_idx,
                        suffix="_partial",
                        save=self._should_save_in_flight_episode(env_idx),
                    )
        return super().close()

    # ------------------------------------------------------------------
    # Internals – phantom-episode filtering
    # ------------------------------------------------------------------

    def _should_save_env_episode(self, env_idx: int) -> bool:
        """Whether the next finished-episode flush for ``env_idx`` hits disk."""
        if self.should_save_episode is None:
            return True
        return bool(self.should_save_episode(env_idx))

    def _should_save_in_flight_episode(self, env_idx: int) -> bool:
        """Whether ``env_idx``'s still-running episode is written at close."""
        if self.should_save_in_flight_episode is None:
            return True
        return bool(self.should_save_in_flight_episode(env_idx))

    # ------------------------------------------------------------------
    # Internals – accumulation (unchanged)
    # ------------------------------------------------------------------

    def _episode_clock(self, info: dict) -> torch.Tensor:
        """Per-env episode step this step's stats belong to (``episode_length_buf`` convention).

        Envs that finished this step were already auto-reset (clock zeroed), so
        theirs comes from the pre-reset ``episode_len`` in ``final_info``.
        """
        clock = self.base_env.episode_length_buf.to(self.device)
        if "final_info" in info and "episode" in info["final_info"]:
            done = info["_final_info"].to(self.device)
            final_len = info["final_info"]["episode"]["episode_len"].to(clock)
            clock = torch.where(done, final_len, clock)
        return clock

    def _accumulate(self, stats: dict, clock: torch.Tensor) -> None:
        """Update running aggregates with this step's stat values."""
        self._stats_count += 1
        # Steps the eval hid from the clock before the first counted step
        # (parked at 0) don't count as reaching a goal.
        counted = clock >= 1
        for key, val in stats.items():
            if not isinstance(val, torch.Tensor):
                continue
            if val.ndim != 1 or val.shape[0] != self.num_envs:
                continue

            if key not in self._stats_sum:
                self._stats_sum[key] = torch.zeros(
                    self.num_envs, device=self.device
                )
                self._stats_min[key] = torch.full(
                    (self.num_envs,), float("inf"), device=self.device
                )
                self._stats_max[key] = torch.full(
                    (self.num_envs,), float("-inf"), device=self.device
                )

            self._stats_sum[key] += val
            self._stats_min[key] = torch.minimum(self._stats_min[key], val)
            self._stats_max[key] = torch.maximum(self._stats_max[key], val)
            self._stats_final[key] = val

            if key.startswith(_PROGRESS_STAT_PREFIXES):
                if key not in self._stats_initial:
                    self._stats_initial[key] = torch.full(
                        (self.num_envs,), float("nan"), device=self.device
                    )
                    self._stats_first_true[key] = torch.full(
                        (self.num_envs,), float("nan"), device=self.device
                    )
                initial = self._stats_initial[key]
                first_true = self._stats_first_true[key]
                self._stats_initial[key] = torch.where(
                    (clock == 1) & torch.isnan(initial), val.float(), initial
                )
                self._stats_first_true[key] = torch.where(
                    counted & (val > 0.5) & torch.isnan(first_true), clock.float(), first_true
                )
        self._record_trace(stats, clock)

    def _record_trace(self, stats: dict, clock: torch.Tensor) -> None:
        """OR this step's progress stats into the trace at each env's clock step."""
        if self._trace_disabled:
            return
        if self._trace is None:
            keys = [
                key
                for key, val in stats.items()
                if key.startswith(_PROGRESS_STAT_PREFIXES)
                and isinstance(val, torch.Tensor)
                and val.ndim == 1
                and val.shape[0] == self.num_envs
            ]
            if not keys:
                return
            length = int(self.base_env.max_episode_length) + 2
            nbytes = self.num_envs * length * len(keys)
            if nbytes > _TRACE_MAX_BYTES:
                print(
                    f"[WARN]: RecordMetrics: goal trace would need {nbytes / 1024**2:.0f} MB "
                    f"({self.num_envs} envs x {length} steps x {len(keys)} stats); "
                    "not recording it, so no goal-progress plots for this run."
                )
                self._trace_disabled = True
                return
            self._trace_keys = keys
            self._trace = torch.zeros(
                (self.num_envs, length, len(keys)), dtype=torch.uint8, device=self.device
            )
        if any(key not in stats for key in self._trace_keys):
            return
        values = torch.stack(
            [stats[key].to(self.device) > 0.5 for key in self._trace_keys], dim=-1
        ).to(torch.uint8)
        # Steps the eval hid from the clock before the first counted step
        # (parked at 0) are not part of the episode.
        values &= (clock >= 1).to(torch.uint8).unsqueeze(-1)
        rows = torch.arange(self.num_envs, device=self.device)
        idx = clock.long().clamp(0, self._trace.shape[1] - 1)
        self._trace[rows, idx] |= values

    def _encode_trace(
        self, done_ids: torch.Tensor, clock: torch.Tensor
    ) -> dict[int, dict[str, list[list[int]]]]:
        """Change points ``[[step, value], ...]`` per progress stat for each finished env.

        ``clock`` holds each finished env's pre-reset ``episode_len``. One host
        copy covers every env that finished this step.
        """
        if self._trace is None or len(done_ids) == 0:
            return {}
        lengths = clock[done_ids].long().clamp(0, self._trace.shape[1] - 1).tolist()
        max_len = max(lengths)
        if max_len < 1:
            return {}
        rows = self._trace[done_ids, : max_len + 1].cpu()
        out: dict[int, dict[str, list[list[int]]]] = {}
        for j, env_idx in enumerate(done_ids.tolist()):
            n = int(lengths[j])
            if n < 1:
                continue
            seq = rows[j, 1 : n + 1]
            per_key: dict[str, list[list[int]]] = {}
            for k, key in enumerate(self._trace_keys):
                col = seq[:, k]
                changes = (col[1:] != col[:-1]).nonzero(as_tuple=False).squeeze(-1).tolist()
                per_key[key] = [[1, int(col[0])]] + [[i + 2, int(col[i + 1])] for i in changes]
            out[env_idx] = per_key
        return out

    def _compute_summaries(self) -> dict[str, torch.Tensor]:
        """Return ``{key}/{agg}`` tensors of shape ``[num_envs]``."""
        result: dict[str, torch.Tensor] = {}
        count = self._stats_count.clamp(min=1).float()
        for key in self._stats_sum:
            result[f"{key}/mean"] = self._stats_sum[key] / count
            result[f"{key}/min"] = self._stats_min[key].clone()
            result[f"{key}/max"] = self._stats_max[key].clone()
            result[f"{key}/final"] = self._stats_final[key].clone()
        for key in self._stats_initial:
            result[f"{key}/initial"] = self._stats_initial[key].clone()
            result[f"{key}/first_true_step"] = self._stats_first_true[key].clone()
        return result

    def _reset_accumulators(self, env_ids: torch.Tensor) -> None:
        """Zero out accumulators for the given environment indices."""
        if len(env_ids) == 0:
            return
        self._stats_count[env_ids] = 0
        for key in self._stats_sum:
            self._stats_sum[key][env_ids] = 0.0
            self._stats_min[key][env_ids] = float("inf")
            self._stats_max[key][env_ids] = float("-inf")
        for key in self._stats_initial:
            self._stats_initial[key][env_ids] = float("nan")
            self._stats_first_true[key][env_ids] = float("nan")
        if self._trace is not None:
            self._trace[env_ids] = 0

    # ------------------------------------------------------------------
    # Internals – CSV writing
    # ------------------------------------------------------------------

    def _buffer_csv_row(self, stats: dict, clock: torch.Tensor) -> None:
        """Buffer one row of per-step stats for each env.

        ``step`` counts wrapper steps; ``episode_step`` is the env's episode
        clock for the row (see :meth:`_episode_clock`), so hold steps repeat it
        and it lines up with the goal trace and ``episode_len``.

        The per-env stats are copied to the host once per dtype (a handful of
        syncs per step) rather than with one ``.item()`` per env per key, which
        at tens of envs and ~70 keys was thousands of syncs per step. Grouping
        by dtype keeps each value's Python type (bool / int / float) exactly as
        ``.item()`` returned it.
        """
        if self._step_metrics_dir is None:
            return
        keys = [
            key
            for key, val in stats.items()
            if isinstance(val, torch.Tensor) and val.ndim == 1 and val.shape[0] == self.num_envs
        ]
        if not keys:
            return
        by_dtype: dict[torch.dtype, list[str]] = {}
        for key in keys:
            by_dtype.setdefault(stats[key].dtype, []).append(key)
        columns: dict[str, list[float | int | bool]] = {}
        for group in by_dtype.values():
            host = torch.stack([stats[key] for key in group]).cpu().tolist()
            columns.update(zip(group, host))
        episode_steps: list[int] = clock.long().cpu().tolist()
        for env_idx in range(self.num_envs):
            row: dict[str, float | int | bool] = {
                "step": len(self._csv_buffers[env_idx]),
                "episode_step": episode_steps[env_idx],
            }
            for key in keys:
                row[key] = columns[key][env_idx]
            self._csv_buffers[env_idx].append(row)

    def _flush_csv(self, env_idx: int, suffix: str = "", save: bool = True) -> None:
        """Write buffered rows for one env to a CSV file and clear the buffer.

        ``save=False`` clears the buffer without writing and without bumping the
        per-env counter, so saved files remain sequential.
        """
        rows = self._csv_buffers[env_idx]
        if not rows or not save:
            self._csv_buffers[env_idx] = []
            return

        filename = (
            f"env{env_idx}_episode_{self._episode_counters[env_idx]:04d}{suffix}.csv"
        )
        path = os.path.join(self._step_metrics_dir, filename)  # type: ignore[arg-type]
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        self._episode_counters[env_idx] += 1
        self._csv_buffers[env_idx] = []
