"""One-rollout diagnostic for HL conditioning staleness.

Answers a question the HL-latency video cannot: are the conditioning frame the
LL is actually conditioned on and the frame it is currently looking at really
different, and do they diverge and re-sync at the configured rate?

Two series per control step, written to ``hl_conditioning_staleness.png``
(plus the event marks described below):

* the mean absolute pixel difference between ``obs["hl_conditioning_image"]``
  and the live camera RGB, both put through the transform's ``process_image``
  so the numbers describe the 224x224 grid the policy consumes rather than the
  native render;
* the age of the entry :class:`~blocksuite.serving.hl_latency.HLLatencyBuffer`
  selected, read straight off the buffer.

The age is the reference signal: it sawtooths within ``[M, M + K)``, so K reads
off as the period and M as the floor, and a distance curve that stays flat at
zero while those knobs are nonzero means the staleness plumbing is a no-op.

Sampled every control step rather than at chunk boundaries -- the obs transform
only runs once per ``action_horizon`` steps under ``ActionChunkBroker``, so
sampling there would alias any K near or below the chunk period. The chunk
boundaries are also marked as purple dots on the age curve, because they are the
only steps the LL is actually conditioned on: a K that phase-locks with the chunk
period reads as a flat line of markers over a healthy-looking sawtooth.

Two event series are drawn as full-height vertical lines under the curves:

* ``LL call`` (purple) -- the steps whose observation reaches the policy server,
  read off ``ActionChunkBroker.needs_inference`` rather than
  ``step % action_horizon`` (an ``adapter.reset()`` from *any* env's episode end
  re-phases the shared broker mid-episode);
* ``HL commit (fresh conditioning)`` (brown) -- the steps on which a fresh HL
  output was latched into the buffer and became eligible for the LL. Not "HL
  call": with the oracle, ``hl_controller.compute()`` runs on every control step
  to keep its state machine advancing, so the commit is the observable event.

Where the two coincide, the HL is synchronous with the LL -- which together with
age dots at zero is the readout for a deliberate ``M = 0``,
``K = c * chunk_period`` configuration. Restricted to one env's first episode:
these are per-step buffers, and this is a diagnostic, not a metric.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from blocksuite.utils.common_utils import per_env_hwc_uint8

logger = logging.getLogger(__name__)

_PLOT_NAME = "hl_conditioning_staleness.png"


class HLStalenessRecorder:
    """Per-step conditioning-vs-live distance for one env's first episode.

    Args:
        output_dir: Eval output dir; the PNG and its JSON sidecar land here.
        env_idx: The env to follow (the first video env, by convention).
        camera: Blocksuite camera whose RGB the conditioning frame is compared
            against -- the same one the latency buffer caches, so a difference
            is staleness and not a change of viewpoint.
        transform: The live obs transform, used only for ``process_image`` and
            ``image_size`` so both frames land on the model's grid.
        update_interval_ms: The configured K, annotated on the plot.
        estimated_inference_time_ms: The configured M, annotated on the plot.
        chunk_period_ms: How long one action chunk covers. Annotated because a
            K below it cannot be resolved by the LL by construction.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        env_idx: int,
        camera: str,
        transform: Any,
        update_interval_ms: float,
        estimated_inference_time_ms: float,
        chunk_period_ms: float,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._env_idx = int(env_idx)
        self._camera = camera
        self._transform = transform
        self._k_ms = float(update_interval_ms)
        self._m_ms = float(estimated_inference_time_ms)
        self._chunk_period_ms = float(chunk_period_ms)
        self._sim_time_ms: list[float] = []
        self._mean_abs_diff: list[float] = []
        self._age_ms: list[float] = []
        self._inference_time_ms: list[float] = []
        self._inference_age_ms: list[float] = []
        self._hl_commit_time_ms: list[float] = []
        self._closed = False

    @property
    def closed(self) -> bool:
        return self._closed

    def close(self) -> None:
        """Stop recording (the followed env finished its first episode)."""
        self._closed = True

    def record(
        self,
        obs: dict[str, Any],
        sim_time_ms: float,
        latency_buffer: Any,
        *,
        at_inference: bool = False,
        at_hl_update: bool = False,
    ) -> None:
        """Append one control step's sample. No-op once :meth:`close` was called.

        ``at_inference`` marks a chunk boundary -- the step whose obs the LL is
        conditioned on -- and adds the age to the chunk-boundary series.
        ``at_hl_update`` marks a step on which a *fresh* HL output was committed
        to the buffer, which is the event the LL can observe; the oracle's own
        forward runs every step and is not it.
        """
        if self._closed:
            return
        cond = obs.get("hl_conditioning_image")
        if cond is None:
            return
        live = obs["images"][self._camera]["rgb"]

        size = self._transform.image_size
        cond_grid = self._transform.process_image(per_env_hwc_uint8(cond, self._env_idx), size)
        live_grid = self._transform.process_image(per_env_hwc_uint8(live, self._env_idx), size)

        self._sim_time_ms.append(float(sim_time_ms))
        self._mean_abs_diff.append(
            float(np.abs(cond_grid.astype(np.int16) - live_grid.astype(np.int16)).mean())
        )
        age = latency_buffer.selected_age_ms(self._env_idx, sim_time_ms)
        self._age_ms.append(float("nan") if age is None else age)
        if at_inference and age is not None:
            self._inference_time_ms.append(float(sim_time_ms))
            self._inference_age_ms.append(float(age))
        if at_hl_update:
            self._hl_commit_time_ms.append(float(sim_time_ms))

    def save(self) -> Path | None:
        """Write the PNG + JSON sidecar; ``None`` when nothing was recorded."""
        if not self._sim_time_ms:
            logger.warning(
                "HL staleness recorder collected no samples for env %d; no "
                "conditioning frame was ever injected.",
                self._env_idx,
            )
            return None

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        t = np.asarray(self._sim_time_ms)
        diff = np.asarray(self._mean_abs_diff)
        age = np.asarray(self._age_ms)

        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.plot(t, diff, color="tab:blue", lw=1.2, label="|conditioning − live| (mean, 0–255)")
        ax.set_xlabel("sim time (ms)")
        ax.set_ylabel("mean absolute pixel difference", color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")

        # Event series as full-height vertical lines: one LineCollection each
        # rather than hundreds of axvline artists, drawn under both curves. When
        # the two colors coincide the HL is synchronous with the LL -- every
        # commit lands on the step the LL is conditioned on -- and at K = 0 the
        # brown series is every step, which reads as "refreshed every step".
        if self._inference_time_ms:
            ax.vlines(
                self._inference_time_ms, 0, 1,
                transform=ax.get_xaxis_transform(),
                color="tab:purple", lw=0.9, alpha=0.35, zorder=0.5,
                label=f"LL call (n={len(self._inference_time_ms)})",
            )
        if self._hl_commit_time_ms:
            ax.vlines(
                self._hl_commit_time_ms, 0, 1,
                transform=ax.get_xaxis_transform(),
                color="tab:brown", lw=0.9, alpha=0.35, zorder=0.4,
                label=f"HL commit (fresh conditioning) (n={len(self._hl_commit_time_ms)})",
            )

        ax_age = ax.twinx()
        ax_age.plot(t, age, color="tab:orange", lw=1.0, alpha=0.8, label="selected entry age")
        ax_age.set_ylabel("selected conditioning age (ms)", color="tab:orange")
        ax_age.tick_params(axis="y", labelcolor="tab:orange")
        if self._inference_time_ms:
            ax_age.plot(
                self._inference_time_ms,
                self._inference_age_ms,
                color="tab:purple",
                ls="none",
                marker="o",
                ms=3.5,
                label="age at chunk boundary (what the LL sees)",
            )
        if self._m_ms > 0:
            ax_age.axhline(self._m_ms, color="tab:red", ls="--", lw=1,
                           label=f"M = {self._m_ms:g} ms (age floor)")
        if self._k_ms > 0:
            ax_age.axhline(self._m_ms + self._k_ms, color="tab:green", ls=":", lw=1,
                           label=f"M + K = {self._m_ms + self._k_ms:g} ms (age ceiling)")

        finite_age = age[np.isfinite(age)]
        title = (
            f"HL conditioning staleness · env {self._env_idx} · {self._camera}   "
            f"K={self._k_ms:g} ms  M={self._m_ms:g} ms  "
            f"chunk={self._chunk_period_ms:g} ms"
        )
        if finite_age.size:
            title += f"   observed age {finite_age.min():.0f}–{finite_age.max():.0f} ms"
        if self._inference_age_ms:
            inf_age = np.asarray(self._inference_age_ms)
            title += f"   at chunk boundary {inf_age.min():.0f}–{inf_age.max():.0f} ms"
        ax.set_title(title, fontsize=10)
        handles = ax.get_legend_handles_labels()
        age_handles = ax_age.get_legend_handles_labels()
        ax.legend(handles[0] + age_handles[0], handles[1] + age_handles[1],
                  fontsize=7, loc="upper right")
        ax.grid(alpha=0.2)
        fig.tight_layout()

        png = self._output_dir / _PLOT_NAME
        png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png, dpi=120)
        plt.close(fig)

        with open(self._output_dir / "hl_conditioning_staleness.json", "w") as f:
            json.dump(
                {
                    "env_idx": self._env_idx,
                    "camera": self._camera,
                    "update_interval_ms": self._k_ms,
                    "estimated_inference_time_ms": self._m_ms,
                    "chunk_period_ms": self._chunk_period_ms,
                    "sim_time_ms": self._sim_time_ms,
                    "mean_abs_diff": self._mean_abs_diff,
                    "selected_age_ms": self._age_ms,
                    "inference_sim_time_ms": self._inference_time_ms,
                    "inference_age_ms": self._inference_age_ms,
                    "hl_commit_sim_time_ms": self._hl_commit_time_ms,
                },
                f,
            )
        return png


