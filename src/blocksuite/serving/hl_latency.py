"""Shared high-level (HL) latency buffer for the eval stack.

Decouples the high-level conditioning (oracle ``compute()`` or Molmo2 forward)
from per-step low-level (LL) inference via two independent knobs. Simulation
time is measured in **milliseconds** (``sim_time_ms = step * base_env.step_dt *
1000``) so the knobs read as wall-clock-style latencies:

* ``update_interval_ms`` (K): the HL is *invoked* at most once per K
  milliseconds, which is also the only time an entry (payload + one cached
  frame) is appended. ``K == 0`` commits every step (today's behavior).
* ``estimated_inference_time_ms`` (M): the wall-clock time the HL forward pass
  is assumed to take. Even though sim time can be frozen while the HL runs, this
  fakes the LL model and physics continuing during that inference: the LL is fed
  the freshest buffered HL output whose input observation is ``>= M`` ms old, so
  the LL always acts on conditioning that is at least one HL inference stale.
  Age never triggers an HL call; it only selects an older entry. ``M == 0`` ==
  latest available (instantaneous HL).

Lives in ``serving/`` (no Isaac deps) so both the client eval loop
(:mod:`blocksuite.eval.run_eval`) and the policy server
(:mod:`blocksuite.serving.serve_maniflow_policy`) can import it.
"""

from __future__ import annotations

import collections
import random
from typing import Any, Optional


def _env_rng(seed: int | None, env_idx: int) -> random.Random:
    """Per-env jitter RNG so interval draws do not depend on first-touch order."""
    if seed is None:
        return random.Random()
    return random.Random(seed + env_idx * 9973)


class HLLatencyBuffer:
    """Per-env ring buffer of ``(sim_time_ms, payload, cond_frame)`` HL outputs.

    ``payload`` is an opaque dict (e.g. ``{task_name, heatmap, done}``); the
    oracle ``done`` flag rides along so the latency-gated episode end reads the
    aged ``done`` rather than the fresh one. ``cond_frame`` is the HL input
    observation frame (uint8 HWC) cached for the side-by-side viz, or ``None``
    when the caller does not need it. All times are in milliseconds.

    Slots are keyed by env index and pre-registered at construction for
    ``0 .. num_envs - 1``. An unknown index raises rather than silently
    allocating, which keeps jitter reproducible (each slot draws its intervals
    from a dedicated ``random.Random(seed + env_idx * 9973)`` at init).
    """

    def __init__(
        self,
        update_interval_ms: float,
        estimated_inference_time_ms: float,
        num_envs: int,
        update_interval_jitter_ms: float = 0.0,
        seed: int | None = None,
        ll_infer_period_ms: float = 0.0,
    ) -> None:
        self._update_interval_ms = max(0.0, float(update_interval_ms))
        self._estimated_inference_time_ms = max(0.0, float(estimated_inference_time_ms))
        self._ll_infer_period_ms = max(0.0, float(ll_infer_period_ms))
        self._num_envs = int(num_envs)
        if self._num_envs < 1:
            raise ValueError(f"HLLatencyBuffer num_envs must be >= 1, got {self._num_envs}")
        self._jitter_ms = max(0.0, float(update_interval_jitter_ms))
        self._history: dict[int, collections.deque] = {}
        self._last_update: dict[int, Optional[float]] = {}
        self._next_interval: dict[int, float] = {}
        self._rngs: dict[int, random.Random] = {}
        for env_idx in range(self._num_envs):
            self._register_env(env_idx, seed)

    def _register_env(self, env_idx: int, seed: int | None) -> None:
        self._history[env_idx] = collections.deque()
        self._last_update[env_idx] = None
        self._rngs[env_idx] = _env_rng(seed, env_idx)
        self._next_interval[env_idx] = self._sample_interval(env_idx)

    def _require_env(self, env_idx: int) -> None:
        if env_idx not in self._history:
            raise IndexError(
                f"unknown HL latency env index {env_idx}; valid indices are "
                f"0..{self._num_envs - 1}"
            )

    def _sample_interval(self, env_idx: int) -> float:
        """Next update interval (ms), with an optional resampled uniform jitter."""
        if self._jitter_ms <= 0.0:
            return self._update_interval_ms
        rng = self._rngs[env_idx]
        return max(
            0.0,
            self._update_interval_ms + rng.uniform(-self._jitter_ms, self._jitter_ms),
        )

    def should_update(self, env_idx: int, sim_time_ms: float) -> bool:
        """True when ``env_idx`` is due for a fresh HL invocation at ``sim_time_ms``."""
        self._require_env(env_idx)
        last = self._last_update[env_idx]
        if last is None:
            return True
        return (float(sim_time_ms) - last) >= self._next_interval[env_idx]

    def commit(
        self,
        env_idx: int,
        sim_time_ms: float,
        payload: dict[str, Any],
        cond_frame: Any = None,
    ) -> None:
        """Append a fresh HL output. Call only when :meth:`should_update` fired.

        Prunes entries older than ``estimated_inference_time_ms +
        update_interval_ms (+ jitter) + ll_infer_period_ms`` so the ring stays
        bounded while always retaining the oldest entry that :meth:`select`
        could still return. ``ll_infer_period_ms`` is the minimum sim-time gap
        between HL commits on the learned-HL server path (``action_horizon *
        step_dt``); without it, ``K <`` that period can prune the only entry
        ``select`` needs when ``M > 0``.
        """
        self._require_env(env_idx)
        sim_time_ms = float(sim_time_ms)
        hist = self._history[env_idx]
        hist.append((sim_time_ms, payload, cond_frame))
        self._last_update[env_idx] = sim_time_ms
        self._next_interval[env_idx] = self._sample_interval(env_idx)

        # Keep a margin of one (jittered) interval beyond the age window so the
        # freshest entry with ts <= sim_time_ms - M is never pruned away.
        horizon = (
            self._estimated_inference_time_ms
            + self._update_interval_ms
            + self._jitter_ms
            + self._ll_infer_period_ms
        )
        while len(hist) > 1 and (sim_time_ms - hist[0][0]) > horizon:
            hist.popleft()

    def _select_entry(self, env_idx: int, sim_time_ms: float):
        """The ``(ts, payload, cond_frame)`` entry :meth:`select` resolves to."""
        self._require_env(env_idx)
        hist = self._history[env_idx]
        if not hist:
            return None
        cutoff = float(sim_time_ms) - self._estimated_inference_time_ms
        chosen = hist[0]  # oldest available (episode-start / pre-M fallback)
        for entry in hist:
            if entry[0] <= cutoff:
                chosen = entry
            else:
                break
        return chosen

    def select(self, env_idx: int, sim_time_ms: float) -> tuple[Any, Any]:
        """Freshest entry with ``ts <= sim_time_ms - M``; else the oldest entry.

        Returns ``(payload, cond_frame)``, or ``(None, None)`` if empty.
        """
        chosen = self._select_entry(env_idx, sim_time_ms)
        if chosen is None:
            return None, None
        return chosen[1], chosen[2]

    def selected_age_ms(self, env_idx: int, sim_time_ms: float) -> float | None:
        """How stale the conditioning :meth:`select` returns is, in ms.

        The diagnostic counterpart to :meth:`select`: the age sawtooths within
        ``[M, M + K)`` once the buffer is warm, so plotting it against sim time
        reads K off as the period and M as the floor. ``None`` when the buffer
        is empty.
        """
        chosen = self._select_entry(env_idx, sim_time_ms)
        if chosen is None:
            return None
        return float(sim_time_ms) - chosen[0]

    def latest(self, env_idx: int) -> tuple[Any, Any]:
        """Most recent (age-0) ``(payload, cond_frame)`` for the real-time viz."""
        self._require_env(env_idx)
        hist = self._history[env_idx]
        if not hist:
            return None, None
        return hist[-1][1], hist[-1][2]

    def reset(self, env_idx: int) -> None:
        """Clear one env's history on an episode boundary."""
        self._require_env(env_idx)
        self._history[env_idx].clear()
        self._last_update[env_idx] = None
        self._next_interval[env_idx] = self._sample_interval(env_idx)

    def reset_all(self) -> None:
        """Clear every registered env slot."""
        for env_idx in self._history:
            self.reset(env_idx)
