"""Wandb integration for ToyBlocks evaluation runs.

Multi-process layout (one policy server + N evaluators) can share a wandb ``group``.
Dashboard sections:
``eval/`` (running episode metrics + final ``eval/aggregates_table``),
``throughput/`` (env-step / policy-call rate, latency p50/p99), ``videos/``
(per-episode mp4s + final ``videos/episodes_table``), ``system/`` (auto).
Headline aggregates are mirrored into ``wandb.run.summary`` for the sidebar.

Callers own ``wandb.init`` / ``wandb.finish`` / ``run.summary``; this module
owns per-eval bookkeeping classes and helpers.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import deque
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def resolve_group(explicit: str | None) -> str | None:
    """The wandb group: the explicit one, else None."""
    return explicit or None


def sanitize_wandb_env() -> None:
    """Strip leading/trailing whitespace from wandb-related env vars.

    Secrets and pasted keys often include a trailing newline; wandb
    rejects API keys with surrounding whitespace during settings validation.
    """
    for key in ("WANDB_API_KEY",):
        raw = os.environ.get(key)
        if raw is None:
            continue
        stripped = raw.strip()
        if stripped != raw:
            logger.warning("Stripped surrounding whitespace from %s.", key)
            os.environ[key] = stripped


def setup_wandb(
    args: Any,
    *,
    run_name: str,
    output_dir: str,
) -> tuple[Any | None, Callable[[int, int, Path], None] | None, dict]:
    """Init the wandb run for one ``scripts/evaluate.py`` invocation.

    Returns ``(wandb_run, on_video_saved, wandb_state)``:

    - ``wandb_run`` is ``None`` when ``--wandb`` is unset (callers gate
      ``wandb.run.summary`` writes on it).
    - ``on_video_saved`` is always ``None`` (per-env streaming video uploads
      removed; mp4s are still written to disk for the episodes table).
    - ``wandb_state`` carries ``video_fps`` so the episodes table can render.

    Decoupled from the eval loop so the init boilerplate (config
    serialization, group resolution) doesn't
    bloat ``evaluate()``.
    """
    wandb_state: dict = {"env_steps": 0, "video_fps": 25}
    if not args.wandb:
        return None, None, wandb_state

    import wandb

    sanitize_wandb_env()

    wandb_config = {
        k: v
        for k, v in vars(args).items()
        if isinstance(v, (bool, int, float, str, type(None), tuple, list))
    }
    wandb_config.update(
        {
            "run_name": run_name,
            "output_dir": output_dir,
        }
    )
    wandb_run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=resolve_group(args.wandb_group),
        name=args.wandb_name
        or f"{args.task}__seed{args.seed if args.seed is not None else 'none'}",
        tags=list(args.wandb_tags),
        job_type=args.wandb_job_type,
        config=wandb_config,
        reinit=False,
    )

    return wandb_run, None, wandb_state


# Per-episode metrics surfaced as columns in the final episodes table.
# Other keys (stats/*, contact_pair/*, ...) are kept in eval_results.json only.
_RESULT_TABLE_METRIC_KEYS: tuple[str, ...] = (
    "success_once",
    "success_at_end",
    "first_success_step",
    "fail_once",
    "fail_at_end",
    "episode_len",
    "return",
    "reward",
    # HL-model runs only (--hl_from_model): the HL emitted its "done" token.
    "hl_done",
    # --ll_oracle: commands the oracle could not run, and Molmo2 calls made.
    "ll_oracle_failures",
    "hl_calls",
    # --collision: a non-target block was knocked / how many distinct ones.
    "collision",
    "collision_count",
    # --collision, oracle / Molmo HL runs: per sub-goal names and successes (lists).
    "sub_goals",
    "sub_goal_success",
    # --collision penalize: the same successes with collisions ignored.
    "success_once_ignore_collision",
    "success_at_end_ignore_collision",
    "sub_goal_success_ignore_collision",
    # --use_render_params: applied / count_mismatch / field_mismatch / no_record.
    "render_replay_status",
    # --replay_block_identity: applied / none.
    "block_identity_status",
    # Autogen PDDL tasks: goal predicates in total / true at the first counted
    # step / true at the end / true at some point (see ``goal_progress``).
    "num_goal_conjuncts",
    "num_goal_conjuncts_initial",
    "num_goal_conjuncts_final",
    "num_goal_conjuncts_ever",
)
# Per-episode divergence scalars a ``--replay`` run folds into its metrics
# (``replay_joint_pos_error_mean``, ``replay_object_<name>_pos_error_final``, ...).
_REPLAY_METRIC_PREFIX = "replay_"


class EvalThroughputTracker:
    """Step / episode rates and policy-call latency, flushed to wandb.

    Rates are over a sliding window so they reflect *recent* throughput.
    Use :meth:`record_step` / :meth:`record_episodes` per step, then
    :meth:`maybe_log` for periodic flushes and :meth:`final_log` at the end.
    """

    def __init__(
        self,
        *,
        num_envs: int,
        log_every_n_seconds: float = 5.0,
        latency_window: int = 200,
        log_to_wandb: bool = True,
    ) -> None:
        self._num_envs = int(num_envs)
        self._log_every = float(log_every_n_seconds)
        self._log_to_wandb = bool(log_to_wandb)
        self._latencies_ms: deque[float] = deque(maxlen=int(latency_window))
        self._obs_to_cpu_ms: deque[float] = deque(maxlen=int(latency_window))
        self._action_to_gpu_ms: deque[float] = deque(maxlen=int(latency_window))
        self._history: list[dict[str, Any]] = []
        self._total_policy_calls = 0
        self._total_episodes = 0
        self._obs_to_cpu_total_ms: float = 0.0
        self._obs_to_cpu_count: int = 0
        self._action_to_gpu_total_ms: float = 0.0
        self._action_to_gpu_count: int = 0
        now = time.perf_counter()
        self._wall_start = now
        self._last_log_time = now
        self._window_start = now
        self._window_calls = 0

    @property
    def total_env_steps(self) -> int:
        return self._total_policy_calls * self._num_envs

    def record_step(
        self,
        *,
        policy_call_seconds: float | None = None,
        transfer_obs_to_cpu_ms: float | None = None,
        transfer_action_to_gpu_ms: float | None = None,
    ) -> None:
        self._total_policy_calls += 1
        self._window_calls += 1
        if policy_call_seconds is not None:
            self._latencies_ms.append(policy_call_seconds * 1000.0)
        if transfer_obs_to_cpu_ms is not None:
            self._obs_to_cpu_ms.append(transfer_obs_to_cpu_ms)
            self._obs_to_cpu_total_ms += transfer_obs_to_cpu_ms
            self._obs_to_cpu_count += 1
        if transfer_action_to_gpu_ms is not None:
            self._action_to_gpu_ms.append(transfer_action_to_gpu_ms)
            self._action_to_gpu_total_ms += transfer_action_to_gpu_ms
            self._action_to_gpu_count += 1

    def record_episodes(self, n: int) -> None:
        self._total_episodes += int(n)

    def maybe_log(self, extra: dict | None = None) -> dict | None:
        """Flush throughput stats if at least ``log_every_n_seconds`` elapsed."""
        now = time.perf_counter()
        if now - self._last_log_time < self._log_every:
            return None
        return self._flush(now, extra or {})

    def final_log(self, extra: dict | None = None) -> dict:
        """Flush a final throughput record (call after the eval loop ends)."""
        return self._flush(time.perf_counter(), extra or {})

    def _flush(self, now: float, extra: dict) -> dict:
        window_dt = max(now - self._window_start, 1e-6)
        elapsed = max(now - self._wall_start, 1e-6)
        env_steps_window = self._window_calls * self._num_envs

        payload: dict = {
            "throughput/env_steps_per_sec": env_steps_window / window_dt,
            "throughput/policy_calls_per_sec": self._window_calls / window_dt,
            "throughput/episodes_per_min": (self._total_episodes / elapsed) * 60.0,
            "throughput/total_env_steps": self.total_env_steps,
            "throughput/total_policy_calls": self._total_policy_calls,
            "throughput/total_episodes": self._total_episodes,
            "throughput/elapsed_seconds": elapsed,
        }
        if self._latencies_ms:
            arr = np.asarray(self._latencies_ms, dtype=np.float64)
            payload["throughput/policy_call_ms_p50"] = float(np.percentile(arr, 50))
            payload["throughput/policy_call_ms_p99"] = float(np.percentile(arr, 99))
            payload["throughput/policy_call_ms_mean"] = float(arr.mean())
        if self._obs_to_cpu_ms:
            arr = np.asarray(self._obs_to_cpu_ms, dtype=np.float64)
            payload["throughput/obs_to_cpu_ms_p50"] = float(np.percentile(arr, 50))
            payload["throughput/obs_to_cpu_ms_p99"] = float(np.percentile(arr, 99))
            payload["throughput/obs_to_cpu_ms_mean"] = float(arr.mean())
        if self._obs_to_cpu_count > 0:
            payload["throughput/obs_to_cpu_total_ms"] = self._obs_to_cpu_total_ms
            payload["throughput/obs_to_cpu_count"] = self._obs_to_cpu_count
            payload["throughput/obs_to_cpu_pct_wall"] = (
                self._obs_to_cpu_total_ms / (elapsed * 1000.0)
            ) * 100.0
        if self._action_to_gpu_ms:
            arr = np.asarray(self._action_to_gpu_ms, dtype=np.float64)
            payload["throughput/action_to_gpu_ms_p50"] = float(np.percentile(arr, 50))
            payload["throughput/action_to_gpu_ms_p99"] = float(np.percentile(arr, 99))
            payload["throughput/action_to_gpu_ms_mean"] = float(arr.mean())
        if self._action_to_gpu_count > 0:
            payload["throughput/action_to_gpu_total_ms"] = self._action_to_gpu_total_ms
            payload["throughput/action_to_gpu_count"] = self._action_to_gpu_count
            payload["throughput/action_to_gpu_pct_wall"] = (
                self._action_to_gpu_total_ms / (elapsed * 1000.0)
            ) * 100.0
        payload.update(extra)

        step = self.total_env_steps
        self._history.append({"step": step, "payload": payload})
        if self._log_to_wandb:
            import wandb

            wandb.log(payload, step=step)
        self._last_log_time = now
        self._window_start = now
        self._window_calls = 0
        return payload

    def save_json_sidecar(self, output_dir: str | Path) -> Path:
        """Write throughput history for a coordinator-only wandb run."""
        path = Path(output_dir) / "eval_throughput.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._history, f, indent=2)
        return path


def log_aggregates_table(
    aggregated: dict[str, dict[str, float]],
    *,
    step: int | None = None,
    panel_key: str = "eval/aggregates_table",
) -> None:
    """Log per-metric mean/std/min/max/count aggregates as one sortable wandb.Table.

    ``aggregated`` comes from
    :func:`toyblocks.utils.dataset_recording_utils._aggregate_metrics`. Rows
    are sorted alphabetically so related keys cluster together.
    """
    if not aggregated:
        return

    import wandb

    columns = ["metric", "mean", "std", "min", "max", "count"]
    table = wandb.Table(columns=columns)
    for key in sorted(aggregated):
        stats = aggregated[key]
        table.add_data(
            key,
            float(stats["mean"]) if "mean" in stats else None,
            float(stats["std"]) if "std" in stats else None,
            float(stats["min"]) if "min" in stats else None,
            float(stats["max"]) if "max" in stats else None,
            int(stats["count"]) if "count" in stats else None,
        )

    payload = {panel_key: table}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


# Episode-record key prefix for the ``--save_diagnostic_plots`` panel PNGs
# (``policy_input_front``, ``policy_input_state``, ...). One prefix rather than a
# fixed column list because which cameras appear depends on the checkpoint's
# declared depth inputs.
POLICY_INPUT_DEBUG_PREFIX = "policy_input_"

# Identity columns the policy-input table leads with, so a row can be matched
# back to a row of ``videos/episodes_table``.
_POLICY_INPUT_ID_COLUMNS = ("episode_index", "episode_id", "env_idx", "episode_idx")


def log_policy_input_debug_table(
    records: list[dict],
    *,
    step: int | None = None,
    panel_key: str = "debug/policy_inputs_table",
) -> None:
    """Log the ``--save_diagnostic_plots`` policy-input panels as their own table.

    Deliberately separate from ``videos/episodes_table``: these are wide
    diagnostic images that would force the video table to be grouped or scrolled
    to read. Rows whose PNGs are all missing are skipped; a recorded path that
    does not exist raises, since silently dropping it would read as "the eval
    had no depth input".
    """
    columns = sorted(
        {
            k
            for rec in records
            for k, v in rec.items()
            if k.startswith(POLICY_INPUT_DEBUG_PREFIX) and v
        }
    )
    if not columns:
        return

    import wandb

    id_columns = [
        c for c in _POLICY_INPUT_ID_COLUMNS if any(r.get(c) is not None for r in records)
    ]
    table = wandb.Table(columns=[*id_columns, *columns])
    n_rows = 0
    for rec in records:
        if not any(rec.get(c) for c in columns):
            continue
        row: list[Any] = [rec.get(c) for c in id_columns]
        for col in columns:
            path = rec.get(col)
            if not path:
                row.append(None)
                continue
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"policy-input debug panel {col!r} for episode "
                    f"{rec.get('episode_id') or rec.get('episode_index')} is "
                    f"recorded as {path} but that file does not exist."
                )
            ep = rec.get("episode_index", rec.get("episode_idx", "?"))
            row.append(wandb.Image(str(path), caption=f"{col} · ep{ep}"))
        table.add_data(*row)
        n_rows += 1

    if n_rows == 0:
        return
    payload = {panel_key: table}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


# ``ll_input_ep{episode:04d}_t{sim_time_ms:07.0f}ms.png``, written by
# toyblocks.serving.policy_input_trace. The filename carries the row keys
# because the renderer runs in the policy-server process, which has no way to
# hand a sidecar back to the eval worker.
_LL_INPUT_PANEL_RE = re.compile(r"^ll_input_ep(\d+)_t(\d+)ms\.png$")
_HL_INPUT_PANEL_RE = re.compile(r"^hl_input_ep(\d+)_t(\d+)ms\.png$")


def log_ll_policy_input_table(
    trace_dir: str | Path,
    *,
    step: int | None = None,
    panel_key: str = "debug/ll_policy_inputs_table",
) -> None:
    """Log one row per ``predict_action`` handover from a policy-input trace dir.

    The panels are the exact tensors the LL model received (see
    ``toyblocks.serving.policy_input_trace``), one PNG per handover for one env's
    first episode. A missing directory is a no-op -- the trace is opt-in -- but a
    file that matches the naming and then cannot be read raises, matching
    :func:`log_policy_input_debug_table`: silently dropping it would read as
    "the eval made no handovers".
    """
    d = Path(trace_dir)
    if not d.is_dir():
        return
    panels = sorted(p for p in d.glob("ll_input_*.png"))
    if not panels:
        return

    import wandb

    table = wandb.Table(columns=["episode_id", "sim_time_ms", "handover_idx", "panel"])
    for idx, path in enumerate(panels):
        m = _LL_INPUT_PANEL_RE.match(path.name)
        if m is None:
            raise ValueError(
                f"policy-input trace panel {path} does not match the "
                f"'ll_input_ep<episode>_t<sim_time_ms>ms.png' naming its row keys "
                "are read from."
            )
        table.add_data(
            int(m.group(1)),
            float(m.group(2)),
            idx,
            wandb.Image(str(path), caption=f"ep{m.group(1)} · {m.group(2)}ms"),
        )

    payload = {panel_key: table}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


def log_hl_policy_input_table(
    trace_dir: str | Path,
    *,
    step: int | None = None,
    panel_key: str = "debug/hl_policy_inputs_table",
) -> None:
    """Log one row per Molmo2 HL invocation from a policy-input trace dir.

    Same contract as :func:`log_ll_policy_input_table`: a missing directory is
    a no-op (the HL trace is opt-in and only written on ``--hl_from_model``),
    but a file that matches the naming and then cannot be read raises.
    """
    d = Path(trace_dir)
    if not d.is_dir():
        return
    panels = sorted(p for p in d.glob("hl_input_*.png"))
    if not panels:
        return

    import wandb

    table = wandb.Table(columns=["episode_id", "sim_time_ms", "handover_idx", "panel"])
    for idx, path in enumerate(panels):
        m = _HL_INPUT_PANEL_RE.match(path.name)
        if m is None:
            raise ValueError(
                f"HL policy-input trace panel {path} does not match the "
                f"'hl_input_ep<episode>_t<sim_time_ms>ms.png' naming its row keys "
                "are read from."
            )
        table.add_data(
            int(m.group(1)),
            float(m.group(2)),
            idx,
            wandb.Image(str(path), caption=f"ep{m.group(1)} · {m.group(2)}ms"),
        )

    payload = {panel_key: table}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)


def _extract_expert_first_frame(
    video_path: str,
    from_timestamp: float | None = None,
) -> np.ndarray | None:
    """Read the first frame (or frame at *from_timestamp*) from an mp4 as RGB.

    Uses PyAV for decoding (handles AV1 and other codecs that cv2 struggles
    with). When *from_timestamp* is given (an episode's offset into a shared,
    multi-episode mp4), ``seek`` lands on the nearest keyframe at or *before* the
    target, so we then decode FORWARD until the presentation timestamp reaches
    the target and return that frame — otherwise we'd return the seek keyframe,
    which can belong to an earlier episode in the packed file. Returns ``None``
    when the file doesn't exist; raises on decode failures so the caller sees the
    real error.
    """
    import av

    if not Path(video_path).exists():
        logger.warning("Expert video not found: %s", video_path)
        return None

    container = av.open(video_path)
    try:
        stream = container.streams.video[0]
        if from_timestamp is not None and from_timestamp > 0:
            target_pts = int(from_timestamp / stream.time_base)
            container.seek(target_pts, stream=stream)
            for frame in container.decode(video=0):
                if frame.pts is not None and frame.pts >= target_pts:
                    return frame.to_ndarray(format="rgb24")
            raise ValueError(
                f"Expert video {video_path} has no frame at/after "
                f"from_timestamp={from_timestamp:.4f}s (target pts {target_pts})"
            )
        for frame in container.decode(video=0):
            return frame.to_ndarray(format="rgb24")
        raise ValueError(f"Expert video has no decodable frames: {video_path}")
    finally:
        container.close()


# Cameras that get their own expert first-frame image column in the wandb table,
# as ``(column_suffix, video_key_substring)``. Matched against the RGB
# ``observation.images.*`` keys in a setup's ``expert_videos``.
_EXPERT_FIRST_FRAME_CAMERAS: tuple[tuple[str, str], ...] = (
    ("front", "front"),
    ("gripper", "gripper"),
)

_IMAGE_VIDEO_KEY_PREFIX = "observation.images."


def short_camera_name(key: str) -> str:
    """``"observation.images.camera:front_camera"`` → ``"front"``."""
    name = key.rsplit(":", 1)[-1] if ":" in key else key
    return name.removesuffix("_camera")


def render_first_frame_comparison(
    eval_frame: np.ndarray,
    expert_frame: np.ndarray,
) -> np.ndarray:
    """Compose one camera's ``[eval | dataset | diff]`` strip as an RGB canvas.

    Trailing the diff are a colorbar mapping brightness back to raw pixel error
    and a histogram of the per-pixel mean RGB error. ``eval_frame`` is resized to
    the expert frame when the two resolutions differ.

    Module-level so ``scripts/plot_control_diagnostics_from_eval_run.py`` can
    render the same figure offline instead of reimplementing it.
    """
    from PIL import Image, ImageDraw, ImageFont

    h, w = expert_frame.shape[:2]
    eh, ew = eval_frame.shape[:2]
    if (eh, ew) != (h, w):
        eval_frame = np.array(Image.fromarray(eval_frame).resize((w, h), Image.LANCZOS))

    diff_raw = np.abs(eval_frame.astype(np.int16) - expert_frame.astype(np.int16))
    max_err = int(diff_raw.max())
    if max_err == 0:
        max_err = 1
    diff_vis = np.clip(
        (diff_raw.astype(np.float32) * 255.0 / max_err),
        0,
        255,
    ).astype(np.uint8)

    bar_w = max(30, w // 12)
    colorbar = np.zeros((h, bar_w, 3), dtype=np.uint8)
    for row in range(h):
        val = int(255 * (1.0 - row / max(h - 1, 1)))
        colorbar[row, :, :] = val

    # Histogram of per-pixel mean RGB error (avg of |R|,|G|,|B| diffs)
    mean_per_pixel = diff_raw.astype(np.float64).mean(axis=2)
    hist_w = w
    n_bins = 64
    counts, bin_edges = np.histogram(mean_per_pixel.ravel(), bins=n_bins, range=(0, max_err))
    max_count = int(counts.max()) or 1
    hist_img = np.zeros((h, hist_w, 3), dtype=np.uint8)
    bin_h = h / n_bins
    for bi in range(n_bins):
        bar_len = int(counts[bi] / max_count * (hist_w - 1))
        y0 = h - int((bi + 1) * bin_h)
        y1 = h - int(bi * bin_h)
        if bar_len > 0 and y0 < y1:
            hist_img[y0:y1, :bar_len] = (100, 180, 255)

    title_h = max(28, h // 10)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            title_h - 8,
        )
    except OSError:
        font = ImageFont.load_default()
    try:
        tick_font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            max(12, title_h // 2),
        )
    except OSError:
        tick_font = ImageFont.load_default()

    total_w = w * 3 + bar_w + hist_w
    canvas = Image.new("RGB", (total_w, title_h + h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for i, label_text in enumerate(["eval frame0", "dataset frame0", "diff"]):
        x = i * w + w // 2
        bbox = draw.textbbox((0, 0), label_text, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x - tw // 2, 2), label_text, fill=(255, 255, 255), font=font)

    hist_title = "mean RGB error"
    hist_cx = w * 3 + bar_w + hist_w // 2
    bbox = draw.textbbox((0, 0), hist_title, font=font)
    tw = bbox[2] - bbox[0]
    draw.text((hist_cx - tw // 2, 2), hist_title, fill=(255, 255, 255), font=font)

    concat = np.concatenate([eval_frame, expert_frame, diff_vis], axis=1)
    canvas.paste(Image.fromarray(concat), (0, title_h))
    canvas.paste(Image.fromarray(colorbar), (w * 3, title_h))
    canvas.paste(Image.fromarray(hist_img), (w * 3 + bar_w, title_h))

    # Draw evenly-spaced tick labels on the colorbar mapping
    # brightness back to raw pixel-error values (0 at bottom,
    # max_err at top).  Labels are centred on the bar and clamped
    # vertically so they stay inside the image.
    draw = ImageDraw.Draw(canvas)
    n_ticks = 6
    for t in range(n_ticks):
        frac = t / (n_ticks - 1)
        val = int(max_err * (1.0 - frac))
        y = title_h + int(frac * (h - 1))
        label_text = str(val)
        bbox = draw.textbbox((0, 0), label_text, font=tick_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = w * 3 + bar_w // 2 - tw // 2
        ty = max(title_h, min(y - th // 2, title_h + h - th))
        draw.text((tx, ty), label_text, fill=(0, 0, 0), font=tick_font)

    return np.array(canvas)


def _pick_camera_video_key(expert_videos: dict[str, dict], substring: str) -> str | None:
    """Return the RGB image key whose name contains ``substring`` (else ``None``).

    Restricted to ``observation.images.*`` so depth/segmentation feeds are never
    selected for a first-frame image (an ``observation.depth.front_camera`` key
    would otherwise match ``front``).
    """
    for k in sorted(expert_videos):
        if k.startswith(_IMAGE_VIDEO_KEY_PREFIX) and substring in k.lower():
            return k
    return None


class EpisodeResultsTable:
    """One row per finished queue-eval episode, logged as ``videos/episodes_table``.

    Column order: video, source episode (full expert demo clip when
    ``save_expert_video`` is set), per-camera expert first frames
    (``expert_first_frame_front``/``_gripper``; see
    :data:`_EXPERT_FIRST_FRAME_CAMERAS`), prompt (english), pddl_goal (raw),
    outcome (``success_once``/``success_at_end``),
    length (policy ``episode_len``/``_seconds`` next to expert, plus
    ``init_frame_index`` / ``init_after_hl_action`` when a mid-episode cut was
    used), remaining env metrics, provenance (``source_dataset_episode_idx``/``_id``,
    ``source_dataset_path``, ``expert_video_path``/``_start_s``), then debug
    ``sim_env_idx``/``episode_idx_for_sim_env``.
    Only columns with at least one populated record are emitted.

    Logged once at end of eval. Also writes ``output_dir/eval_episodes.json``
    with the full multi-camera ``expert_videos`` dict preserved (provenance
    ``expert_video_path``/``_start_s`` track the front-preferred camera via
    :func:`_pick_primary_video_key`). The mp4 path is reconstructed from
    ``output_dir/sensor_obs_videos/env<i>_episode_<j:04d>.mp4``.
    """

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_sensor_obs_video: bool,
        save_expert_video: bool = False,
        save_hl_latency_video: bool = False,
        hl_overlay_video_key: str | None = None,
    ) -> None:
        self._output_dir = Path(output_dir)
        self._save_sensor_obs_video = bool(save_sensor_obs_video)
        self._save_expert_video = bool(save_expert_video)
        self._save_hl_latency_video = bool(save_hl_latency_video)
        self._hl_overlay_video_key = hl_overlay_video_key
        self._records: list[dict] = []

    @property
    def num_records(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[dict]:
        """The buffered episode rows (same shape as ``eval_episodes.json``)."""
        return self._records

    def add_episode(
        self,
        *,
        env_idx: int,
        episode_idx: int,
        prompt: str | None,
        queue_setup: dict | None,
        metrics: dict[str, Any],
        eval_first_frames: dict[str, np.ndarray] | None = None,
        policy_input_debug: dict[str, str] | None = None,
        goal_conjunct_labels: list[str] | None = None,
        goal_objects: list[str] | None = None,
        goal_trace: dict[str, list[list[int]]] | None = None,
        step_dt_s: float | None = None,
    ) -> None:
        """Append one finished episode's metadata to the buffer.

        ``episode_idx`` must match the integer in the saved mp4 filename
        (``env<i>_episode_<j:04d>.mp4``) — pass the value that aligns with
        the mp4 ``RecordEpisode`` just flushed, not the next-episode counter.
        Stored as ``episode_idx_for_sim_env`` in the output JSON.
        Provenance is pulled from ``queue_setup`` when present.

        ``eval_first_frames`` maps camera name to the episode's first rendered
        eval frame; when given they are written as
        ``first_frames/<episode>_<cam>.png`` side-by-side with the expert frame.

        ``policy_input_debug`` maps ``policy_input_*`` column name to an
        already-written PNG (see :mod:`toyblocks.eval.policy_input_debug`); the
        paths ride along on the record so the wandb consolidator can render the
        same table from ``eval_episodes.json`` without re-deriving anything.

        ``goal_conjunct_labels`` is the PDDL text of each ``goal_conj_<c>`` of this
        episode's goal; when given, the record gets a ``goal_progress`` list (one
        entry per goal predicate) and the ``num_goal_conjuncts*`` counts.

        ``goal_objects`` names the blocks behind the ``goal_obj_<k>_*`` settle-gate
        stats, in slot order; stored as ``goal_objects`` so offline plots can
        label the per-block speeds in the step-metrics CSVs.

        ``goal_trace`` maps each progress stat (``goal_conj_<c>``, ``goal_success``)
        to its change points ``[[step, value], ...]`` over the episode clock (see
        :class:`~toyblocks.wrappers.record_metrics.RecordMetrics`); each
        ``goal_progress`` entry gets its ``trace`` and the record gets
        ``goal_success_trace`` (plus ``goal_blocks_settled_trace`` /
        ``goal_gripper_open_trace`` for the settle-gate limbs, when traced).
        ``step_dt_s`` is the env control period, so steps
        can be turned into seconds offline.
        """
        rec: dict = {
            "sim_env_idx": int(env_idx),
            "episode_idx_for_sim_env": int(episode_idx),
            "prompt": None if prompt is None else str(prompt),
            "pddl_goal": None,
            "source_dataset_episode_idx": None,
            "source_dataset_episode_id": None,
            "source_dataset_path": None,
            "expert_episode_length": None,
            "expert_video_path": None,
            "expert_video_start_s": None,
            "expert_episode_seconds": None,
            "expert_videos": None,
            "source_episode_video_path": None,
            "init_frame_index": None,
            "init_after_hl_action": None,
        }
        if queue_setup is not None:
            ep_index = queue_setup.get("episode_index")
            rec["source_dataset_episode_idx"] = int(ep_index) if ep_index is not None else None
            rec["source_dataset_episode_id"] = queue_setup.get("episode_id")
            pddl_goal = queue_setup.get("pddl_goal")
            rec["pddl_goal"] = None if pddl_goal is None else str(pddl_goal)
            rec["source_dataset_path"] = queue_setup.get("source_dataset_path")
            expert_len = queue_setup.get("expert_episode_length")
            rec["expert_episode_length"] = int(expert_len) if expert_len is not None else None
            init_frame = queue_setup.get("init_frame_index")
            rec["init_frame_index"] = int(init_frame) if init_frame is not None else None
            rec["init_after_hl_action"] = queue_setup.get("init_after_hl_action")
            expert_videos = queue_setup.get("expert_videos") or {}
            if expert_videos:
                rec["expert_videos"] = expert_videos
                # Provenance columns track the primary (front-preferred) camera.
                primary_key = _pick_primary_video_key(expert_videos)
                v = expert_videos[primary_key]
                rec["expert_video_path"] = v.get("path")
                rec["expert_video_start_s"] = v.get("from_timestamp")
                if v.get("from_timestamp") is not None and v.get("to_timestamp") is not None:
                    rec["expert_episode_seconds"] = float(v["to_timestamp"] - v["from_timestamp"])
                # Extract a separate first frame per camera (front + gripper) so
                # the wandb table renders one image column for each.
                self._extract_camera_first_frames(rec, expert_videos)
                if self._save_expert_video:
                    self._extract_source_episode_video(
                        rec,
                        expert_videos,
                        env_idx=env_idx,
                        episode_idx=episode_idx,
                    )
            else:
                logger.warning(
                    "No expert_videos metadata for queue episode %s; "
                    "queue_setup keys=%s; expert_first_frame will be absent "
                    "from the wandb table.",
                    rec.get("source_dataset_episode_id") or rec.get("source_dataset_episode_idx"),
                    sorted(queue_setup),
                )
        if policy_input_debug:
            rec.update(policy_input_debug)
        for key in _RESULT_TABLE_METRIC_KEYS:
            if key in metrics:
                rec[key] = _coerce_scalar(metrics[key])
        # Dataset-replay divergence scalars (--replay); the key set depends on
        # the objects in the recorded scene, so match by prefix.
        for key in sorted(k for k in metrics if k.startswith(_REPLAY_METRIC_PREFIX)):
            rec[key] = _coerce_scalar(metrics[key])
        if goal_conjunct_labels is not None:
            rec.update(_goal_progress_fields(goal_conjunct_labels, metrics, goal_trace))
        if goal_objects is not None:
            rec["goal_objects"] = [str(name) for name in goal_objects]
        if goal_trace is not None:
            for trace_key in ("goal_success", "goal_blocks_settled", "goal_gripper_open"):
                if trace_key in goal_trace:
                    rec[f"{trace_key}_trace"] = goal_trace[trace_key]
        if step_dt_s is not None:
            rec["step_dt_s"] = float(step_dt_s)
        self._records.append(rec)

    def log(self, *, fps: int, step: int | None = None) -> None:
        """Log buffered rows as ``videos/episodes_table`` and write the JSON sidecar.

        ``episode_seconds`` is derived from ``episode_len / fps``.
        """
        if fps > 0:
            for rec in self._records:
                ep_len = rec.get("episode_len")
                if ep_len is not None:
                    rec["episode_seconds"] = float(ep_len) / float(fps)
        self.save_json_sidecar()
        if not self._records:
            return

        import wandb

        # Group columns by usefulness; emit only those at least one record carries.
        leading = ["video"]
        if self._save_expert_video and any(
            self._source_episode_video_path(r) is not None
            and self._source_episode_video_path(r).exists()
            for r in self._records
        ):
            leading.append("source_episode")
        if any(
            self._hl_overlay_prev_action_video_path(r) is not None
            and self._hl_overlay_prev_action_video_path(r).exists()
            for r in self._records
        ):
            leading.append("hl_overlay_prev_action")
        if self._save_hl_latency_video and any(
            self._hl_latency_video_path(r) is not None
            and self._hl_latency_video_path(r).exists()
            for r in self._records
        ):
            leading.append("hl_latency")
        for cam, _ in _EXPERT_FIRST_FRAME_CAMERAS:
            col = f"expert_first_frame_{cam}"
            if any(r.get(col) is not None for r in self._records):
                leading.append(col)
        leading.append("prompt")
        if any(r.get("pddl_goal") is not None for r in self._records):
            leading.append("pddl_goal")
        outcome = [
            k for k in ("success_once", "success_at_end") if any(k in r for r in self._records)
        ]
        length = [
            k
            for k in (
                "episode_len",
                "episode_seconds",
                "first_success_step",
                "expert_episode_length",
                "expert_episode_seconds",
                "init_frame_index",
                "init_after_hl_action",
            )
            if any(r.get(k) is not None for r in self._records)
        ]
        rest_metrics = [
            k
            for k in _RESULT_TABLE_METRIC_KEYS
            if k not in outcome
            and k not in ("episode_len", "first_success_step")
            and any(k in r for r in self._records)
        ]
        rest_metrics += sorted(
            {k for r in self._records for k in r if k.startswith(_REPLAY_METRIC_PREFIX)}
        )
        provenance = [
            k
            for k in (
                "source_dataset_episode_idx",
                "source_dataset_episode_id",
                "source_dataset_path",
                "expert_video_path",
                "expert_video_start_s",
            )
            if any(r.get(k) is not None for r in self._records)
        ]
        trailing = ["sim_env_idx", "episode_idx_for_sim_env"]

        columns = leading + outcome + length + rest_metrics + provenance + trailing

        table = wandb.Table(columns=columns)
        for rec in self._records:
            row: list = []
            for col in columns:
                if col == "video":
                    row.append(self._video_cell(rec, fps=fps))
                elif col == "source_episode":
                    row.append(self._source_episode_cell(rec, fps=fps))
                elif col == "hl_overlay_prev_action":
                    row.append(self._hl_overlay_prev_action_cell(rec, fps=fps))
                elif col == "hl_latency":
                    row.append(self._hl_latency_cell(rec, fps=fps))
                elif col.startswith("expert_first_frame_"):
                    row.append(self._expert_frame_cell(rec, col))
                else:
                    v = rec.get(col)
                    # Lists (sub_goal_success) render as text: wandb types a list
                    # column from its first row, and episodes differ in length.
                    row.append(str(v) if isinstance(v, list) else v)
            table.add_data(*row)

        payload = {"videos/episodes_table": table}
        if step is None:
            wandb.log(payload)
        else:
            wandb.log(payload, step=step)

        # Its own table (no-op when the panels were off or skipped): the
        # panels are wide images and would make the video table unreadable.
        log_policy_input_debug_table(self._records, step=step)

    # Keys that carry non-serializable data (ndarrays for wandb rendering).
    _JSON_SKIP_KEYS = frozenset(
        f"expert_first_frame_{cam}" for cam, _ in _EXPERT_FIRST_FRAME_CAMERAS
    )

    def save_json_sidecar(self) -> Path:
        """Write the buffered rows to ``output_dir/eval_episodes.json``."""
        out: list[dict] = []
        for rec in self._records:
            mp4 = self._video_path(rec)
            overlay = self._hl_overlay_prev_action_video_path(rec)
            latency = self._hl_latency_video_path(rec)
            entry = {k: v for k, v in rec.items() if k not in self._JSON_SKIP_KEYS}
            entry["video_path"] = str(mp4) if mp4 is not None and mp4.exists() else None
            source_mp4 = self._source_episode_video_path(rec)
            if source_mp4 is not None and source_mp4.exists():
                entry["source_episode_video_path"] = str(source_mp4)
            else:
                entry["source_episode_video_path"] = rec.get("source_episode_video_path")
            entry["hl_overlay_prev_action_video_path"] = (
                str(overlay) if overlay is not None and overlay.exists() else None
            )
            entry["hl_latency_video_path"] = (
                str(latency) if latency is not None and latency.exists() else None
            )
            out.append(entry)
        path = self._output_dir / "eval_episodes.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        return path

    def attach_lerobot_episode_indices(
        self, episode_index_map: dict[tuple[int, int], tuple[str, int]]
    ) -> None:
        """Set each row's ``output_lerobot_dataset_dir`` / ``output_lerobot_episode_index``
        (``None`` when not recorded) from the closed recorder's ``merged_episode_index_map``."""
        if not episode_index_map:
            return
        unmatched = 0
        for rec in self._records:
            key = (int(rec["sim_env_idx"]), int(rec["episode_idx_for_sim_env"]))
            root, index = episode_index_map.get(key, (None, None))
            rec["output_lerobot_episode_index"] = index
            rec["output_lerobot_dataset_dir"] = root
            unmatched += index is None
        if unmatched:
            logger.warning(
                "%d of %d episodes have no LeRobot episode index (recorded episodes: %d)",
                unmatched, len(self._records), len(episode_index_map),
            )

    def _video_path(self, rec: dict) -> Path | None:
        if not self._save_sensor_obs_video:
            return None
        return (
            self._output_dir
            / "sensor_obs_videos"
            / f"env{rec['sim_env_idx']}_episode_{rec['episode_idx_for_sim_env']:04d}.mp4"
        )

    def _source_episode_video_path(self, rec: dict) -> Path | None:
        path = rec.get("source_episode_video_path")
        if path:
            return Path(path)
        if not self._save_expert_video:
            return None
        return (
            self._output_dir
            / "source_episode_videos"
            / f"env{rec['sim_env_idx']}_episode_{rec['episode_idx_for_sim_env']:04d}.mp4"
        )

    def _extract_source_episode_video(
        self,
        rec: dict,
        expert_videos: dict[str, dict],
        *,
        env_idx: int,
        episode_idx: int,
    ) -> None:
        dst = (
            self._output_dir
            / "source_episode_videos"
            / f"env{env_idx}_episode_{episode_idx:04d}.mp4"
        )
        extract_source_episode_video(expert_videos, dst)
        rec["source_episode_video_path"] = str(dst)

    def _source_episode_cell(self, rec: dict, *, fps: int) -> Any:
        import wandb

        mp4 = self._source_episode_video_path(rec)
        if mp4 is None or not mp4.exists():
            return None
        ep_idx = rec.get("source_dataset_episode_idx", "?")
        return wandb.Video(
            str(mp4),
            fps=fps,
            format="mp4",
            caption=(
                f"source ep{ep_idx} · env{rec['sim_env_idx']} · "
                f"episode {rec['episode_idx_for_sim_env']}"
            ),
        )

    def _hl_overlay_prev_action_video_path(self, rec: dict) -> Path | None:
        """The episode's HL overlay MP4 in the output LeRobot dataset (one file per episode)."""
        root = rec.get("output_lerobot_dataset_dir")
        index = rec.get("output_lerobot_episode_index")
        if self._hl_overlay_video_key is None or root is None or index is None:
            return None
        return (
            Path(root)
            / "videos"
            / self._hl_overlay_video_key
            / f"chunk-{index // 1000:03d}"
            / f"file-{index % 1000:03d}.mp4"
        )

    def _hl_overlay_prev_action_cell(self, rec: dict, *, fps: int) -> Any:
        import wandb

        mp4 = self._hl_overlay_prev_action_video_path(rec)
        if mp4 is None or not mp4.exists():
            return None
        return wandb.Video(
            str(mp4),
            fps=fps,
            format="mp4",
            caption=f"HL overlay · env{rec['sim_env_idx']} · episode {rec['episode_idx_for_sim_env']}",
        )

    def _hl_latency_video_path(self, rec: dict) -> Path | None:
        if not self._save_hl_latency_video:
            return None
        return (
            self._output_dir
            / "hl_latency_videos"
            / f"env{rec['sim_env_idx']}_episode_{rec['episode_idx_for_sim_env']:04d}.mp4"
        )

    def _hl_latency_cell(self, rec: dict, *, fps: int) -> Any:
        import wandb

        mp4 = self._hl_latency_video_path(rec)
        if mp4 is None or not mp4.exists():
            return None
        return wandb.Video(
            str(mp4),
            fps=fps,
            format="mp4",
            caption=f"HL latency (real-time | LL-view) · env{rec['sim_env_idx']} · episode {rec['episode_idx_for_sim_env']}",
        )

    _short_camera_name = staticmethod(short_camera_name)

    def _extract_camera_first_frames(self, rec: dict, expert_videos: dict[str, dict]) -> None:
        """Populate ``expert_first_frame_<cam>`` (ndarray) for each camera.

        Iterates :data:`_EXPERT_FIRST_FRAME_CAMERAS`, resolving the matching RGB
        video key per camera; cameras absent from ``expert_videos`` are skipped
        (their column stays empty). Missing files / undecodable frames raise.

        The ndarrays are kept in memory for the wandb table image column; they
        are not saved to disk separately because the ``first_frames/``
        comparison PNGs already contain the expert frame side-by-side with the
        eval frame.
        """
        for cam, substring in _EXPERT_FIRST_FRAME_CAMERAS:
            key = _pick_camera_video_key(expert_videos, substring)
            if key is None:
                continue
            v = expert_videos[key]
            vpath = v.get("path")
            if not vpath:
                logger.warning(
                    "Episode %s expert_videos[%s] has no path: %r",
                    rec.get("source_dataset_episode_id") or rec.get("source_dataset_episode_idx"),
                    key,
                    v,
                )
                continue
            if not Path(vpath).is_file():
                raise FileNotFoundError(f"Expert video not found: {vpath}")
            frame = _extract_expert_first_frame(vpath, v.get("from_timestamp"))
            if frame is None:
                raise RuntimeError(f"Failed to extract expert first frame from {vpath}")
            rec[f"expert_first_frame_{cam}"] = frame

    def _video_cell(self, rec: dict, *, fps: int) -> Any:
        import wandb

        mp4 = self._video_path(rec)
        if mp4 is None or not mp4.exists():
            return None
        return wandb.Video(
            str(mp4),
            fps=fps,
            format="mp4",
            caption=f"env{rec['sim_env_idx']} · episode {rec['episode_idx_for_sim_env']}",
        )

    @staticmethod
    def _expert_frame_cell(rec: dict, col: str) -> Any:
        import wandb

        frame = rec.get(col)
        if frame is None:
            return None
        cam = col[len("expert_first_frame_") :]
        ep_idx = rec.get("source_dataset_episode_idx", "?")
        return wandb.Image(frame, caption=f"expert {cam} ep{ep_idx}")


def _coerce_scalar(v: Any) -> Any:
    """Convert tensor / numpy scalars (and lists of them) into json-friendly Python types."""
    if hasattr(v, "item"):
        v = v.item()
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, list):
        return [_coerce_scalar(x) for x in v]
    raise TypeError(
        f"_coerce_scalar: unexpected type {type(v).__name__}: {v!r}. "
        f"Expected a numeric scalar, bool, or str."
    )


_GOAL_PROGRESS_STATS: tuple[str, ...] = ("initial", "first_true_step", "final", "max", "mean")


def _goal_progress_fields(
    labels: list[str],
    metrics: dict[str, Any],
    goal_trace: dict[str, list[list[int]]] | None = None,
) -> dict[str, Any]:
    """``goal_progress`` entries and ``num_goal_conjuncts*`` counts for one episode.

    Entry ``c`` pairs ``labels[c]`` with the ``goal_conj_<c>/<stat>`` episode
    summaries, plus its ``trace`` change points when ``goal_trace`` has them.
    Slots past ``len(labels)`` are padding for goals shorter than the longest
    compiled goal and are left out. A count is ``None`` when any of its
    summaries is missing.
    """
    progress: list[dict[str, Any]] = []
    for c, label in enumerate(labels):
        metric_key = f"goal_conj_{c}"
        entry: dict[str, Any] = {"index": c, "metric_key": metric_key, "predicate": label}
        for stat in _GOAL_PROGRESS_STATS:
            key = f"{metric_key}/{stat}"
            entry[stat] = _coerce_scalar(metrics[key]) if key in metrics else None
        if goal_trace is not None and metric_key in goal_trace:
            entry["trace"] = goal_trace[metric_key]
        progress.append(entry)

    def _count_true(stat: str) -> int | None:
        values = [e[stat] for e in progress]
        if any(v is None for v in values):
            return None
        return sum(1 for v in values if v > 0.5)

    return {
        "goal_progress": progress,
        "num_goal_conjuncts": len(labels),
        "num_goal_conjuncts_initial": _count_true("initial"),
        "num_goal_conjuncts_final": _count_true("final"),
        "num_goal_conjuncts_ever": _count_true("max"),
    }


def _pick_primary_video_key(expert_videos: dict[str, dict]) -> str:
    """Pick the camera for the wandb table: prefer one containing ``front``, else first sorted."""
    keys = sorted(expert_videos)
    for k in keys:
        if "front" in k.lower():
            return k
    return keys[0]


def _is_expert_video_entry(entry: dict) -> bool:
    path = str(entry.get("path") or "")
    return "from_timestamp" in entry or path.lower().endswith(".mp4")


def extract_source_episode_video(
    expert_videos: dict[str, dict],
    dst: Path,
) -> Path:
    """Stream-copy the primary RGB expert clip for one queue episode into *dst*.

    Uses the same front-preferred camera as ``expert_video_path`` provenance.
    Raises when no decodable video entry exists or ffmpeg fails.
    """
    from toyblocks.utils.tamp_dataset import VideoClip, extract_video_clip

    primary_key = _pick_primary_video_key(expert_videos)
    entry = expert_videos[primary_key]
    vpath = entry.get("path")
    if not vpath:
        raise ValueError(
            f"expert_videos[{primary_key!r}] has no path; cannot extract source episode"
        )
    if not _is_expert_video_entry(entry):
        raise ValueError(
            f"expert_videos[{primary_key!r}] at {vpath!r} is not a video clip "
            "(no from_timestamp and not an .mp4); cannot extract source episode"
        )
    clip = VideoClip(
        video_key=primary_key,
        video_path=Path(vpath),
        from_timestamp=float(entry.get("from_timestamp", 0.0)),
        to_timestamp=entry.get("to_timestamp"),
    )
    extract_video_clip(clip, dst)
    return dst


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _column(records: list[dict], key: str) -> list[float]:
    """Numeric values of ``key`` across records, skipping rows that lack it."""
    return [float(r[key]) for r in records if r.get(key) is not None]


def read_episode_rows(
    dataset_path: str, episode_indices: set[int], columns: list[str]
) -> Iterator[tuple]:
    """``(episode_index, *columns)`` rows of a LeRobot dataset's episodes, in frame order."""
    import pyarrow.dataset as pa_ds

    ds = pa_ds.dataset(str(Path(dataset_path) / "data"), format="parquet")
    table = ds.to_table(
        columns=["episode_index", "frame_index", *columns],
        filter=pa_ds.field("episode_index").isin(sorted(episode_indices)),
    ).sort_by([("episode_index", "ascending"), ("frame_index", "ascending")])
    return zip(*(table.column(c).to_pylist() for c in ["episode_index", *columns]))


def _demo_command_counts(dataset_path: str, episode_indices: set[int]) -> dict[int, int]:
    """``episode_index -> commands`` the demonstration took: runs of ``goal.action``
    (``pick`` / ``place`` / ...) within one high-level action, ``done`` dropped."""
    counts: dict[int, int] = {ep: 0 for ep in episode_indices}
    last: dict[int, tuple[str, str, str]] = {}
    for ep, goal, action, args in read_episode_rows(
        dataset_path,
        episode_indices,
        ["goal.action", "action.high_level_action", "action.high_level_args"],
    ):
        goal = (goal or "").strip().lower()
        if goal in ("", "done"):
            continue
        key = (goal, action, args)
        if last.get(ep) != key:
            counts[ep] += 1
            last[ep] = key
    return counts


def _success_efficiency(records: list[dict]) -> float | None:
    """Mean, over successful episodes with HL sub-goals, of the HL commands the eval
    took (``hl_calls``, ``done`` included, else its sub-goals) over the commands the
    source demonstration took (plus its ``done``). None for
    ``--init-after-hl-action`` runs, which start mid-episode."""
    if any(r.get("init_after_hl_action") for r in records):
        return None
    succeeded = [
        r
        for r in records
        if r.get("success_once")
        and r.get("sub_goals")
        and r.get("source_dataset_path")
        and r.get("source_dataset_episode_idx") is not None
    ]
    by_dataset: dict[str, set[int]] = {}
    for r in succeeded:
        by_dataset.setdefault(r["source_dataset_path"], set()).add(
            int(r["source_dataset_episode_idx"])
        )
    try:
        demo = {
            (path, ep): n
            for path, indices in by_dataset.items()
            for ep, n in _demo_command_counts(path, indices).items()
        }
    except Exception as exc:  # source dataset unreadable from this machine
        logger.warning("success_efficiency skipped: %s", exc)
        return None
    ratios = [
        int(r["hl_calls"]) / (n + 1) if r.get("hl_calls") else len(r["sub_goals"]) / n
        for r in succeeded
        if (n := demo.get((r["source_dataset_path"], int(r["source_dataset_episode_idx"]))))
    ]
    return _mean(ratios)


def log_run_stats_table(
    records: list[dict],
    *,
    run_scalars: dict[str, Any] | None = None,
    step: int | None = None,
    panel_key: str = "summary/run_stats_table",
) -> None:
    """Log one headline row summarizing the whole eval run.

    The per-metric detail lives in ``eval/aggregates_table``; this exists so the
    numbers you actually compare between runs (success rates, episode length
    against the expert's, throughput) are readable without grouping the video
    table. ``records`` are the episode rows the eval workers and the consolidator
    already hold, and ``run_scalars`` carries the run-level context that is not
    derivable from them (policy/task/paths, wall time, worker count).
    """
    if not records:
        return

    import wandb

    scalars = dict(run_scalars or {})
    lengths = _column(records, "episode_len")
    expert_lengths = _column(records, "expert_episode_length")
    mean_len = _mean(lengths)
    mean_expert_len = _mean(expert_lengths)
    first_success = [
        float(r["first_success_step"])
        for r in records
        if r.get("first_success_step") is not None and bool(r.get("success_once"))
    ]
    success_once = [1.0 if r.get("success_once") else 0.0 for r in records if "success_once" in r]
    success_end = [
        1.0 if r.get("success_at_end") else 0.0 for r in records if "success_at_end" in r
    ]
    hl_done = [1.0 if r.get("hl_done") else 0.0 for r in records if "hl_done" in r]

    row: dict[str, Any] = {
        "policy_name": scalars.get("policy_name"),
        "task": scalars.get("task"),
        "dataset_path": scalars.get("dataset_path"),
        "model_path": scalars.get("model_path"),
        "episodes_completed": len(records),
        "success_once_rate": _mean(success_once),
        "success_at_end_rate": _mean(success_end),
        "mean_partial_progress": _mean(_column(records, "partial_progress")),
        "mean_episode_len": mean_len,
        "median_episode_len": float(np.median(lengths)) if lengths else None,
        "mean_episode_seconds": _mean(_column(records, "episode_seconds")),
        "mean_first_success_step": _mean(first_success),
        # An episode that never succeeded ran to the episode_length_s cap, which
        # is the only "timeout" the loop has.
        "timeout_rate": (1.0 - _mean(success_once)) if success_once else None,
        "mean_expert_episode_length": mean_expert_len,
        "length_ratio": (
            mean_len / mean_expert_len if mean_len and mean_expert_len else None
        ),
        "wall_seconds": scalars.get("wall_seconds"),
        "env_steps_per_sec": scalars.get("env_steps_per_sec"),
        "num_workers": scalars.get("num_workers"),
    }
    if hl_done:
        row["hl_done_rate"] = _mean(hl_done)
        # Molmo2 HL runs only: HL commands taken over the demonstration's.
        row["success_efficiency"] = _success_efficiency(records)

    table = wandb.Table(columns=list(row))
    table.add_data(*row.values())
    payload = {panel_key: table}
    if step is None:
        wandb.log(payload)
    else:
        wandb.log(payload, step=step)
