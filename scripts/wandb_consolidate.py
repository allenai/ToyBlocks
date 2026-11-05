"""Single-wandb-run coordinator for multi-GPU eval jobs.

Launches ``wandb.init`` early so the background system monitor captures GPU
utilisation, memory, temperature, CPU, and disk I/O for **all** GPUs on the
node for the entire eval duration (nvidia-smi polling sees all devices
regardless of CUDA_VISIBLE_DEVICES).

Waits for a sentinel file, then reads every eval worker's output directory,
merges ``eval_episodes.json`` + ``eval_results.json``, renders one
representative episode's offline control-diagnostic plots per worker (via
``scripts/plot_diagnostics_from_eval_run.py``; see
``_render_offline_diagnostics``) and uploads them itself under
``diagnostic_plots/*`` (see ``_log_offline_diagnostic_plots`` — the rendering
subprocess never touches wandb, since a run this coordinator still holds open
can't be reliably joined by a second process), and logs a consolidated
``videos/episodes_with_expert_table``, ``eval/aggregates_table``, and headline
metrics before calling ``wandb.finish()``. Every episode's goal-progress plot
(same script, ``--goal-progress-only``) rides in the table's ``goal_progress``
column.

Typical orchestration (from scripts/eval/run_eval_pipeline.sh)::

    python scripts/wandb_consolidate.py \
      --wandb-project "$WANDB_PROJECT" \
      --wandb-name "${POLICY_NAME}__${TASK}__${OUT_SUBDIR}" \
      --wandb-tags "$POLICY_NAME" "$TASK" "$OUT_SUBDIR" \
      --output-dirs "runs/.../seed_*/${RUN_ID}" \
      --sentinel /tmp/eval_done &
    COORD_PID=$!
    # ... launch eval workers (no --wandb) ...
    touch /tmp/eval_done
    wait $COORD_PID
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# wandb rejects tags longer than this, both client-side and in the backend.
_MAX_WANDB_TAG_LEN = 64


def _clip_wandb_tags(tags: list[str]) -> list[str]:
    """Clip tags to wandb's length limit so long task slugs don't abort the run."""
    clipped: list[str] = []
    for tag in tags:
        if len(tag) > _MAX_WANDB_TAG_LEN:
            logger.warning(
                "Tag %r is %d characters; clipping to %d for wandb.",
                tag, len(tag), _MAX_WANDB_TAG_LEN,
            )
            tag = tag[:_MAX_WANDB_TAG_LEN]
        clipped.append(tag)
    return clipped


def _find_output_dirs(patterns: list[str]) -> list[Path]:
    """Expand shell globs and return existing directories."""
    dirs: list[Path] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern))
        for p in expanded:
            d = Path(p)
            if d.is_dir():
                dirs.append(d)
    return dirs


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _merge_aggregates(all_agg: list[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    """Re-aggregate per-metric stats across workers using count-weighted means.

    Each worker's ``eval_results.json["metrics"]`` carries
    ``{key: {mean, std, min, max, count}}``.  We reconstruct exact
    cross-worker mean/min/max and approximate the pooled std.
    """
    merged: dict[str, dict[str, float]] = {}
    for agg in all_agg:
        for key, stats in agg.items():
            if key not in merged:
                merged[key] = {
                    "sum": 0.0,
                    "sum_sq": 0.0,
                    "min": float("inf"),
                    "max": float("-inf"),
                    "count": 0,
                }
            n = stats.get("count", 0)
            if n == 0:
                continue
            m = merged[key]
            m["sum"] += stats["mean"] * n
            m["sum_sq"] += (stats["std"] ** 2 + stats["mean"] ** 2) * n
            m["min"] = min(m["min"], stats["min"])
            m["max"] = max(m["max"], stats["max"])
            m["count"] += n
    out: dict[str, dict[str, float]] = {}
    for key, m in merged.items():
        c = m["count"]
        if c == 0:
            continue
        mean = m["sum"] / c
        variance = max(m["sum_sq"] / c - mean ** 2, 0.0)
        out[key] = {
            "mean": mean,
            "std": variance ** 0.5,
            "min": m["min"],
            "max": m["max"],
            "count": int(c),
        }
    return out


_OFFLINE_DIAGNOSTICS_SCRIPT = Path(__file__).resolve().parent / "plot_diagnostics_from_eval_run.py"
# Must match plot_diagnostics_from_eval_run.py's Args.output_subdir default.
_OFFLINE_DIAGNOSTICS_SUBDIR = "control_diagnostics_offline"

# One rollout only, still rendered online and flat under the worker's
# output_dir root -- unaffected by the offline-diagnostics move.
# hl_conditioning_staleness.png comes from toyblocks.eval.hl_staleness under
# --save_diagnostic_plots.
_DIAGNOSTIC_PLOT_NAMES = [
    "hl_conditioning_staleness.png",
]

# save_eval_plots' fixed-name outputs, rendered by plot_diagnostics_from_eval_run.py
# into <output_dir>/control_diagnostics_offline/episode_<label>/ for the one
# representative episode _pick_representative_episode() chose. Block-name-dependent
# plots (<block_name>_pose_comparison.png, <block_name>_contact_heatmap.png, ...)
# are deliberately excluded: their key set varies per task/episode, which would
# defeat the fixed diagnostic_plots/<stem> namespace this table depends on for
# cross-run comparison. Use `plot_diagnostics_from_eval_run.py --eval-dir <dir>`
# by hand for those.
_OFFLINE_DIAGNOSTIC_PLOT_NAMES = [
    "joint_position_and_pd_gains_comparison.png",
    "joint_velocity.png",
    "ee_pose_comparison.png",
    "fingertip_comparison.png",
    "camera_params.png",
]
# Must match plot_diagnostics_from_eval_run.py's GOAL_PROGRESS_FILENAME. Rendered for
# every episode by its --goal-progress-only pass (no LeRobot data needed), uploaded
# as the episodes table's goal_progress column and, for the representative
# episode, under diagnostic_plots/goal_progress.
_GOAL_PROGRESS_PLOT_NAME = "goal_progress.png"
_GOAL_PROGRESS_RENDER_WORKERS = 8


def _goal_plot_label(ep: dict, position: int) -> int:
    """Folder label of a record; must match plot_diagnostics_from_eval_run.goal_plot_label."""
    for key in ("source_dataset_episode_idx", "output_lerobot_episode_index"):
        if ep.get(key) is not None:
            return int(ep[key])
    return position


def _goal_progress_plot_path(worker_dir: Path, ep: dict, position: int) -> str:
    """Where the offline script writes this record's goal_progress.png."""
    label = _goal_plot_label(ep, position)
    return str(worker_dir / _OFFLINE_DIAGNOSTICS_SUBDIR / f"episode_{label}" / _GOAL_PROGRESS_PLOT_NAME)


def _goal_progress_cell_path(ep: dict) -> str | None:
    """The record's rendered goal_progress.png, or None if it was not drawn."""
    path = ep.get("goal_progress_plot_path")
    return path if path and Path(path).is_file() else None


def _pick_representative_episode(
    episodes: list[dict], target_sim_env_idx: int = 0,
) -> int | None:
    """Pick the one episode to render and upload offline diagnostics for.

    Mirrors the retired online behaviour, which only ever traced
    ``video_env_ids[0]`` (default env 0). Takes that env's *first* completed
    episode (lowest ``episode_idx_for_sim_env``) rather than its last: the old
    flat PNG got overwritten every episode, so "whichever one was on disk when
    the sentinel fired" was an accident of timing, sometimes even a
    mid-episode partial write. Falls back to the globally-earliest-completed
    episode across whatever envs are present if the run used non-default
    ``--video_env_ids`` (not currently plumbed through by ``launch_eval.py``,
    so this fallback is rare in practice).

    Returns the ``source_dataset_episode_idx`` label
    ``plot_diagnostics_from_eval_run.py`` groups ``episode_<label>/`` folders
    by, or ``None`` if no episode has a recorded ``output_lerobot_episode_index`` yet
    (offline rendering needs that row).
    """
    candidates = [e for e in episodes if e.get("output_lerobot_episode_index") is not None]
    if not candidates:
        return None

    def _label(e: dict) -> int:
        idx = e.get("source_dataset_episode_idx")
        return int(idx) if idx is not None else int(e["output_lerobot_episode_index"])

    pool = [e for e in candidates if e.get("sim_env_idx") == target_sim_env_idx] or candidates
    best = min(pool, key=lambda e: (e.get("episode_idx_for_sim_env", 0), _label(e)))
    return _label(best)


def _render_offline_diagnostics(
    worker_dir: Path, episode_label: int, *, timeout: float,
) -> bool:
    """Render one worker's representative episode via the offline script.

    Subprocess, not an in-process import: the script's heavy deps
    (matplotlib/pyarrow/pandas) live in the ``maniflow`` uv extra, not this
    coordinator's base deps, and it's designed as a standalone CLI that raises
    freely on bad/missing data (``run_eval_pipeline.sh``'s all-episode
    render uses the same subprocess pattern). A bad
    episode, missing ground-truth dataset, or rendering crash should cost this
    one worker's diagnostic plots, not the whole consolidation run.

    Render-only — does NOT touch wandb. An earlier version had this subprocess
    upload directly via ``wandb.init(resume="allow", id=<this coordinator's own
    live run>)``; that hung and timed out in production, because a run another
    process (this coordinator) is still actively holding open cannot be
    reliably joined by a second, unrelated process — wandb's resume path is for
    reattaching to a finished/crashed run, not concurrent multi-process writes
    to a live one. So the coordinator uploads the rendered PNGs itself
    afterward, in the same process as its own already-open ``wandb.run`` (see
    ``_collect_offline_diagnostic_plots``/``_log_offline_diagnostic_plots``).

    Returns True if rendering succeeded, False otherwise (worker skipped or
    the subprocess failed/timed out).
    """
    if not (worker_dir / "lerobot_dataset").is_dir():
        logger.info(
            "%s: no lerobot_dataset/ (worker did not run with --save_lerobot); "
            "skipping offline diagnostic rendering.", worker_dir,
        )
        return False

    cmd = [
        sys.executable, str(_OFFLINE_DIAGNOSTICS_SCRIPT),
        "--eval-dir", str(worker_dir),
        "--episodes", str(episode_label),
        "--num-workers", "1",
    ]
    logger.info(
        "%s: rendering offline diagnostics for episode %d.", worker_dir, episode_label,
    )
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "%s: offline diagnostic rendering timed out after %.0fs; skipping.",
            worker_dir, timeout,
        )
        return False
    except OSError as exc:
        logger.warning("%s: failed to launch offline diagnostic rendering: %s", worker_dir, exc)
        return False

    if result.returncode != 0:
        logger.warning(
            "%s: offline diagnostic rendering failed (exit %d); skipping.\n"
            "--- stdout (tail) ---\n%s\n--- stderr (tail) ---\n%s",
            worker_dir, result.returncode, result.stdout[-4000:], result.stderr[-4000:],
        )
        return False
    return True


def _render_goal_progress_plots(worker_dir: Path, *, timeout: float) -> bool:
    """Render every episode's goal_progress.png for one worker via the offline script.

    Subprocess for the same reasons as ``_render_offline_diagnostics``; the
    ``--goal-progress-only`` pass reads only ``eval_episodes.json``, so it also
    covers workers that ran without ``--save_lerobot``. Returns True on success.
    """
    cmd = [
        sys.executable, str(_OFFLINE_DIAGNOSTICS_SCRIPT),
        "--eval-dir", str(worker_dir),
        "--goal-progress-only",
        "--num-workers", str(_GOAL_PROGRESS_RENDER_WORKERS),
    ]
    logger.info("%s: rendering goal-progress plots for every episode.", worker_dir)
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "%s: goal-progress rendering timed out after %.0fs; skipping.", worker_dir, timeout,
        )
        return False
    except OSError as exc:
        logger.warning("%s: failed to launch goal-progress rendering: %s", worker_dir, exc)
        return False
    if result.returncode != 0:
        logger.warning(
            "%s: goal-progress rendering failed (exit %d); skipping.\n"
            "--- stdout (tail) ---\n%s\n--- stderr (tail) ---\n%s",
            worker_dir, result.returncode, result.stdout[-4000:], result.stderr[-4000:],
        )
        return False
    return True


def _collect_offline_diagnostic_plots(
    episode_label_by_dir: dict[Path, int],
    names: list[str] = _OFFLINE_DIAGNOSTIC_PLOT_NAMES,
) -> dict[str, list[tuple[Path, str, int]]]:
    """Discover the rendered representative-episode PNGs across workers.

    Same ``{plot_name: [(path, worker_label, ...), ...]}`` shape
    ``_collect_diagnostic_plots`` uses, with the episode label tacked on so
    ``_log_offline_diagnostic_plots`` can surface it.
    """
    plots: dict[str, list[tuple[Path, str, int]]] = {}
    for d, label in episode_label_by_dir.items():
        episode_dir = d / _OFFLINE_DIAGNOSTICS_SUBDIR / f"episode_{label}"
        for name in names:
            p = episode_dir / name
            if p.is_file():
                plots.setdefault(name, []).append((p, d.name, label))
    return plots


def _log_offline_diagnostic_plots(plots: dict[str, list[tuple[Path, str, int]]]) -> None:
    """Upload the rendered representative-episode PNGs under ``diagnostic_plots/``.

    Mirrors ``_log_diagnostic_plots``' flat-vs-namespaced convention (flat key
    when only one worker produced that plot, namespaced by worker label
    otherwise), plus a ``diagnostic_plots[/<worker>]/episode_idx`` summary
    field — the image captions carry the same info, but a summary field is
    visible in the run overview/table without opening any image, which
    matters since the panel keys are deliberately flat/stable across runs and
    don't say which episode they came from.
    """
    import wandb

    if not plots:
        logger.info("No offline diagnostic plots found in output directories.")
        return

    total = sum(len(v) for v in plots.values())
    logger.info(
        "Uploading %d offline diagnostic plot(s) across %d file type(s) to wandb.",
        total, len(plots),
    )

    episode_idx_by_worker: dict[str | None, set[int]] = {}
    for name, entries in sorted(plots.items()):
        stem = name.rsplit(".", 1)[0]
        multi = len(entries) > 1
        for path, worker, label in entries:
            key = f"diagnostic_plots/{worker}/{stem}" if multi else f"diagnostic_plots/{stem}"
            wandb.log({key: wandb.Image(str(path), caption=f"episode {label}")})
            episode_idx_by_worker.setdefault(worker if multi else None, set()).add(label)

    for worker, labels in episode_idx_by_worker.items():
        key = f"diagnostic_plots/{worker}/episode_idx" if worker else "diagnostic_plots/episode_idx"
        sorted_labels = sorted(labels)
        wandb.run.summary[key] = sorted_labels[0] if len(sorted_labels) == 1 else sorted_labels


def _collect_diagnostic_plots(output_dirs: list[Path]) -> dict[str, list[tuple[Path, str]]]:
    """Discover diagnostic PNGs across worker output directories.

    Returns ``{plot_name: [(path, worker_label), ...]}``.  Only known
    diagnostic file names are collected (not arbitrary PNGs) to avoid
    accidentally uploading huge video frames or other noise.
    """
    plots: dict[str, list[tuple[Path, str]]] = {}
    for d in output_dirs:
        worker_label = d.name
        for name in _DIAGNOSTIC_PLOT_NAMES:
            p = d / name
            if p.is_file():
                plots.setdefault(name, []).append((p, worker_label))
    return plots


def _log_diagnostic_plots(plots: dict[str, list[tuple[Path, str]]]) -> None:
    """Upload discovered diagnostic PNGs to wandb under ``diagnostic_plots/``."""
    import wandb

    if not plots:
        logger.info("No diagnostic plots found in output directories.")
        return

    total = sum(len(v) for v in plots.values())
    logger.info(
        "Uploading %d diagnostic plot(s) across %d file type(s) to wandb.",
        total, len(plots),
    )

    for name, entries in sorted(plots.items()):
        stem = name.rsplit(".", 1)[0]
        if len(entries) == 1:
            path, worker = entries[0]
            wandb.log({
                f"diagnostic_plots/{stem}": wandb.Image(str(path), caption=worker),
            })
        else:
            for path, worker in entries:
                wandb.log({
                    f"diagnostic_plots/{stem}/{worker}": wandb.Image(
                        str(path), caption=worker,
                    ),
                })


def _build_episodes_table(
    all_episodes: list[dict],
    *,
    video_fps: int,
) -> Any:
    """Build a ``wandb.Table`` from merged ``eval_episodes.json`` rows."""
    import wandb

    from toyblocks.utils.wandb_logging import (
        _EXPERT_FIRST_FRAME_CAMERAS,
        _RESULT_TABLE_METRIC_KEYS,
        _extract_expert_first_frame,
        _pick_camera_video_key,
    )

    if not all_episodes:
        return None

    def _has_camera_input(ep: dict, cam: str, substring: str) -> bool:
        if ep.get(f"expert_first_frame_{cam}_path"):
            return True
        expert_videos = ep.get("expert_videos") or {}
        return bool(expert_videos) and _pick_camera_video_key(expert_videos, substring) is not None

    # Per-camera expert first-frame columns with at least one input across rows.
    expert_frame_cams = [
        (cam, substring)
        for cam, substring in _EXPERT_FIRST_FRAME_CAMERAS
        if any(_has_camera_input(ep, cam, substring) for ep in all_episodes)
    ]
    if expert_frame_cams:
        logger.info(
            "episodes_table: expert first-frame columns %s",
            [f"expert_first_frame_{cam}" for cam, _ in expert_frame_cams],
        )
    else:
        sample = [
            {
                "source_dataset_episode_id": ep.get("source_dataset_episode_id"),
                "source_dataset_episode_idx": ep.get("source_dataset_episode_idx"),
                "keys": sorted(ep),
            }
            for ep in all_episodes[:3]
        ]
        logger.warning(
            "episodes_table: no rows carry per-camera expert first-frame inputs "
            "(expert_first_frame_<cam>_path or expert_videos), so no "
            "expert_first_frame_* column will be emitted. Sample rows: %s",
            sample,
        )

    leading = ["video"]
    if any(ep.get("source_episode_video_path") for ep in all_episodes):
        leading.append("source_episode")
    if any(ep.get("hl_overlay_prev_action_video_path") for ep in all_episodes):
        leading.append("hl_overlay_prev_action")
    if any(ep.get("hl_latency_video_path") for ep in all_episodes):
        leading.append("hl_latency")
    leading += [f"expert_first_frame_{cam}" for cam, _ in expert_frame_cams]
    if any(_goal_progress_cell_path(ep) for ep in all_episodes):
        leading.append("goal_progress")
    leading.append("prompt")

    outcome = [k for k in ("success_once", "success_at_end")
               if any(k in ep for ep in all_episodes)]
    length = [k for k in ("episode_len", "episode_seconds",
                          "first_success_step",
                          "expert_episode_length", "expert_episode_seconds",
                          "init_frame_index", "init_after_hl_action")
              if any(ep.get(k) is not None for ep in all_episodes)]
    rest_metrics = [k for k in _RESULT_TABLE_METRIC_KEYS
                    if k not in outcome and k not in ("episode_len", "first_success_step")
                    and any(k in ep for ep in all_episodes)]
    provenance = [k for k in ("source_dataset_episode_idx", "source_dataset_episode_id",
                              "source_dataset_path",
                              "expert_video_path", "expert_video_start_s",
                              "expert_first_frame_front_path",
                              "expert_first_frame_gripper_path")
                  if any(ep.get(k) is not None for ep in all_episodes)]
    trailing = ["sim_env_idx", "episode_idx_for_sim_env", "worker_dir"]

    columns = leading + outcome + length + rest_metrics + provenance + trailing

    expert_cam_by_col = {
        f"expert_first_frame_{cam}": (cam, substring) for cam, substring in expert_frame_cams
    }

    table = wandb.Table(columns=columns)
    for ep in all_episodes:
        row: list = []
        for col in columns:
            if col == "video":
                vpath = ep.get("video_path")
                if vpath and Path(vpath).exists():
                    row.append(wandb.Video(vpath, fps=video_fps, format="mp4"))
                else:
                    row.append(None)
            elif col == "source_episode":
                spath = ep.get("source_episode_video_path")
                if spath and Path(spath).exists():
                    ep_idx = ep.get("source_dataset_episode_idx", "?")
                    row.append(
                        wandb.Video(
                            spath,
                            fps=video_fps,
                            format="mp4",
                            caption=(
                                f"source ep{ep_idx} · env{ep.get('sim_env_idx')} · "
                                f"episode {ep.get('episode_idx_for_sim_env')}"
                            ),
                        )
                    )
                else:
                    row.append(None)
            elif col == "hl_overlay_prev_action":
                opath = ep.get("hl_overlay_prev_action_video_path")
                if opath and Path(opath).exists():
                    row.append(wandb.Video(opath, fps=video_fps, format="mp4"))
                else:
                    row.append(None)
            elif col == "hl_latency":
                lpath = ep.get("hl_latency_video_path")
                if lpath and Path(lpath).exists():
                    row.append(wandb.Video(lpath, fps=video_fps, format="mp4"))
                else:
                    row.append(None)
            elif col == "goal_progress":
                gpath = _goal_progress_cell_path(ep)
                if gpath:
                    # The folder is episode_<label>.
                    caption = Path(gpath).parent.name.replace("_", " ")
                    row.append(wandb.Image(gpath, caption=caption))
                else:
                    row.append(None)
            elif col in expert_cam_by_col:
                cam, substring = expert_cam_by_col[col]
                row.append(
                    _expert_frame_cell(ep, cam, substring, _extract_expert_first_frame,
                                       _pick_camera_video_key)
                )
            else:
                row.append(ep.get(col))
        table.add_data(*row)
    logger.debug("episodes_table columns emitted to W&B: %s", columns)
    return table


def _total_env_steps_per_sec(throughput: list[dict], wall_seconds: float) -> float | None:
    """Aggregate env-step throughput across workers.

    Each worker's last throughput row carries its cumulative
    ``throughput/total_env_steps``; the workers run concurrently, so the run's
    rate is their summed steps over the (max, not summed) wall time.
    """
    if not throughput or wall_seconds <= 0:
        return None
    per_worker: dict[str, int] = {}
    for row in throughput:
        steps = (row.get("payload") or {}).get("throughput/total_env_steps")
        if steps is None:
            continue
        worker = str(row.get("worker_dir", ""))
        per_worker[worker] = max(per_worker.get(worker, 0), int(steps))
    if not per_worker:
        return None
    return sum(per_worker.values()) / wall_seconds


def _expert_frame_cell(ep, cam, substring, extract_first_frame, pick_camera_key):
    """Resolve one episode's expert first frame for camera ``cam`` into a wandb cell.

    Prefers the saved ``expert_first_frame_<cam>_path`` PNG; otherwise decodes the
    matching camera's clip from ``expert_videos``. Missing files / undecodable
    frames raise (no silent fallback).
    """
    import wandb

    ep_idx = ep.get("source_dataset_episode_idx", "?")
    frame_path = ep.get(f"expert_first_frame_{cam}_path")
    if frame_path:
        if not Path(frame_path).is_file():
            raise FileNotFoundError(f"Expert first frame not found: {frame_path}")
        return wandb.Image(frame_path, caption=f"expert {cam} ep{ep_idx}")
    expert_videos = ep.get("expert_videos") or {}
    key = pick_camera_key(expert_videos, substring) if expert_videos else None
    if key is None:
        return None
    video = expert_videos[key]
    vpath = video.get("path")
    if not vpath:
        return None
    if not Path(vpath).is_file():
        raise FileNotFoundError(f"Expert video not found: {vpath}")
    frame = extract_first_frame(vpath, video.get("from_timestamp"))
    if frame is None:
        raise RuntimeError(f"Failed to extract expert first frame from {vpath}")
    return wandb.Image(frame, caption=f"expert {cam} ep{ep_idx}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--wandb-project", required=True)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-name", required=True)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-tags", nargs="*", default=[])
    p.add_argument("--output-dirs", nargs="+", required=True,
                   help="Glob patterns for eval worker output directories.")
    p.add_argument("--sentinel", required=True,
                   help="Path to file whose existence signals eval is done.")
    p.add_argument("--eval-params", default=None,
                   help="JSON string of resolved launch_eval.py parameters "
                        "to store in the wandb run config for reproducibility.")
    p.add_argument("--video-fps", type=int, default=25)
    p.add_argument("--poll-interval", type=float, default=10.0,
                   help="Seconds between sentinel checks.")
    p.add_argument("--sentinel-timeout", type=float, default=24 * 3600.0,
                   help="Give up waiting for the sentinel after this many seconds. "
                        "The launcher touches the sentinel after the workers exit, so "
                        "reaching this means they died without it; without a timeout "
                        "the coordinator holds a GPU forever and never uploads.")
    p.add_argument("--diagnostic-env-idx", type=int, default=0,
                   help="sim_env_idx whose episode gets offline diagnostics "
                        "rendered+uploaded (matches --video_env_ids[0]; default 0).")
    p.add_argument("--render-diagnostic-plots", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Render + upload one representative episode's offline "
                        "control-diagnostic plots per worker before finishing. "
                        "Disable to skip the extra subprocess.")
    p.add_argument("--diagnostic-render-timeout", type=float, default=600.0,
                   help="Per-worker timeout (seconds) for the offline diagnostic subprocess.")
    args = p.parse_args()

    import wandb

    from toyblocks.utils.wandb_logging import (
        log_aggregates_table,
        log_hl_policy_input_table,
        log_ll_policy_input_table,
        log_policy_input_debug_table,
        log_run_stats_table,
        resolve_group,
    )

    eval_params: dict = {}
    if args.eval_params:
        eval_params = json.loads(args.eval_params)

    group = resolve_group(args.wandb_group)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=group,
        name=args.wandb_name,
        tags=_clip_wandb_tags(args.wandb_tags),
        job_type="eval",
        config={
            **eval_params,
            "coordinator": True,
            "tags_full": args.wandb_tags,
            "output_dir_patterns": args.output_dirs,
        },
    )
    logger.info(
        "wandb run started: %s (group=%s). Waiting for sentinel: %s",
        run.url, group, args.sentinel,
    )

    sentinel = Path(args.sentinel)
    deadline = time.monotonic() + args.sentinel_timeout
    sentinel_timed_out = False
    while not sentinel.exists():
        if time.monotonic() >= deadline:
            logger.error(
                "Sentinel %s did not appear within %.0fs; the eval workers exited "
                "without signalling. Uploading whatever results exist and finishing "
                "rather than holding the GPU indefinitely.",
                sentinel, args.sentinel_timeout,
            )
            sentinel_timed_out = True
            break
        time.sleep(args.poll_interval)
    else:
        logger.info("Sentinel found. Collecting results from output dirs...")

    output_dirs = _find_output_dirs(args.output_dirs)
    if not output_dirs:
        logger.warning("No output directories found for patterns: %s", args.output_dirs)
        wandb.finish(exit_code=1)
        sys.exit(1)
    logger.info("Found %d output dir(s): %s", len(output_dirs), [str(d) for d in output_dirs])

    all_episodes: list[dict] = []
    all_aggregates: list[dict[str, dict[str, float]]] = []
    all_throughput: list[dict] = []
    total_episodes = 0
    total_elapsed = 0.0
    episode_label_by_dir: dict[Path, int] = {}

    for d in output_dirs:
        episodes_path = d / "eval_episodes.json"
        results_path = d / "eval_results.json"
        throughput_path = d / "eval_throughput.json"

        if episodes_path.exists():
            episodes = _load_json(episodes_path)
            label = _pick_representative_episode(episodes, args.diagnostic_env_idx)
            if label is not None:
                episode_label_by_dir[d] = label
            for position, ep in enumerate(episodes):
                ep["worker_dir"] = str(d)
                if ep.get("goal_progress"):
                    ep["goal_progress_plot_path"] = _goal_progress_plot_path(d, ep, position)
            all_episodes.extend(episodes)
            logger.info("  %s: %d episodes", d, len(episodes))

        if results_path.exists():
            results = _load_json(results_path)
            metrics = results.get("metrics", {})
            if metrics:
                all_aggregates.append(metrics)
            total_episodes += results.get("num_episodes", 0)
            total_elapsed = max(total_elapsed, results.get("elapsed_seconds", 0.0))

        if throughput_path.exists():
            for row in _load_json(throughput_path):
                row["worker_dir"] = str(d)
                all_throughput.append(row)

    logger.info(
        "Totals: %d episodes, %d aggregate dicts, %d throughput records, %.1fs wall time",
        total_episodes, len(all_aggregates), len(all_throughput), total_elapsed,
    )

    merged_agg = _merge_aggregates(all_aggregates) if all_aggregates else {}

    if merged_agg:
        log_aggregates_table(merged_agg)
        for k, stats in merged_agg.items():
            if "/" in k:
                continue
            for stat in ("mean", "std", "min", "max"):
                wandb.run.summary[f"{k}_{stat}"] = stats[stat]

    # Workers number their steps independently, so logging N workers' rows at their
    # own `step` makes them collide and overwrite each other. Namespace the metric
    # keys per worker and use a single monotonic step so every worker's curve
    # survives.
    worker_slots = {d: i for i, d in enumerate(sorted(str(d) for d in output_dirs))}
    for i, row in enumerate(
        sorted(all_throughput, key=lambda r: (r.get("step", 0), r.get("worker_dir", "")))
    ):
        payload = row.get("payload", {})
        if not payload:
            continue
        slot = worker_slots.get(str(row.get("worker_dir", "")), 0)
        wandb.log(
            {
                **{f"worker_{slot}/{k}": v for k, v in payload.items()},
                "worker_step": int(row.get("step", 0)),
            },
            step=i,
        )

    # Before the table, which carries every episode's plot in its goal_progress column.
    goal_rendered_dirs: set[Path] = set()
    if args.render_diagnostic_plots:
        for d in output_dirs:
            if not any(ep.get("goal_progress_plot_path") for ep in all_episodes
                       if ep.get("worker_dir") == str(d)):
                continue
            try:
                if _render_goal_progress_plots(d, timeout=args.diagnostic_render_timeout):
                    goal_rendered_dirs.add(d)
            except Exception:
                logger.exception(
                    "%s: unexpected error rendering goal-progress plots; continuing.", d,
                )

    table = _build_episodes_table(all_episodes, video_fps=args.video_fps)
    if table is not None:
        wandb.log({"videos/episodes_with_expert_table": table})

    # No-op unless the workers ran with --save-diagnostic-plots; the PNG paths
    # ride along on the merged eval_episodes.json rows.
    log_policy_input_debug_table(all_episodes)

    log_run_stats_table(
        all_episodes,
        run_scalars={
            "policy_name": eval_params.get("policy_name"),
            "task": eval_params.get("task_slug"),
            "dataset_path": eval_params.get("dataset_path"),
            "model_path": eval_params.get("model_path"),
            "wall_seconds": total_elapsed,
            "env_steps_per_sec": _total_env_steps_per_sec(all_throughput, total_elapsed),
            "num_workers": len(output_dirs),
        },
    )

    if args.render_diagnostic_plots:
        rendered_episode_by_dir: dict[Path, int] = {}
        for d in output_dirs:
            episode_label = episode_label_by_dir.get(d)
            if episode_label is None:
                logger.info(
                    "%s: no episode with a recorded output_lerobot_episode_index yet; "
                    "skipping offline diagnostic rendering.", d,
                )
                continue
            try:
                if _render_offline_diagnostics(
                    d, episode_label, timeout=args.diagnostic_render_timeout,
                ):
                    rendered_episode_by_dir[d] = episode_label
            except Exception:
                logger.exception(
                    "%s: unexpected error rendering offline diagnostics; continuing.", d,
                )
        offline_plots = _collect_offline_diagnostic_plots(rendered_episode_by_dir)
        # From the goal pass, so a failed control render doesn't drop it.
        offline_plots.update(
            _collect_offline_diagnostic_plots(
                {d: label for d, label in episode_label_by_dir.items() if d in goal_rendered_dirs},
                names=[_GOAL_PROGRESS_PLOT_NAME],
            )
        )
        _log_offline_diagnostic_plots(offline_plots)

    diag_plots = _collect_diagnostic_plots(output_dirs)
    _log_diagnostic_plots(diag_plots)

    # No-op unless a worker ran with --save-diagnostic-plots against a maniflow
    # policy server, which renders one panel per predict_action handover into the
    # worker's ll_policy_input_trace/ (and, for learned HL, one panel per Molmo2
    # forward into hl_policy_input_trace/). Namespaced per worker for the same
    # reason the throughput rows are: one shared key would leave only the last
    # worker's.
    for d in output_dirs:
        log_ll_policy_input_table(
            d / "ll_policy_input_trace",
            panel_key=(
                "debug/ll_policy_inputs_table"
                if len(output_dirs) == 1
                else f"debug/ll_policy_inputs_table/{d.name}"
            ),
        )
        log_hl_policy_input_table(
            d / "hl_policy_input_trace",
            panel_key=(
                "debug/hl_policy_inputs_table"
                if len(output_dirs) == 1
                else f"debug/hl_policy_inputs_table/{d.name}"
            ),
        )

    wandb.run.summary["episodes_completed"] = total_episodes
    wandb.run.summary["wall_seconds"] = total_elapsed
    wandb.run.summary["num_workers"] = len(output_dirs)
    wandb.run.summary["sentinel_timed_out"] = sentinel_timed_out

    if sentinel_timed_out:
        # The upload above is best-effort salvage of a run whose workers died
        # without signalling; finishing green would hide that in wandb and in the
        # pipeline's exit status, matching the empty-output-dirs path above.
        wandb.finish(exit_code=1)
        logger.error("wandb run finished, marked failed (sentinel timeout).")
        sys.exit(1)

    wandb.finish()
    logger.info("wandb run finished.")


if __name__ == "__main__":
    main()
