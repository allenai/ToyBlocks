#!/usr/bin/env python3
"""Read-only audit of ``goal.position`` vs the recorded-camera gripper projection.

Samples a fraction of a LeRobot dataset, splits each episode into subskills
(contiguous runs of the same non-empty ``goal.action`` with a finite 2-D
``goal.position``), and scores the last tail of each run by pixel L2 between
``goal.position`` and the EE projected with that frame's recorded camera pose
and K. Large, consistent gaps mean the annotation was not projected with the
camera that produced the image (``annotate_target_xy.py`` still hardcodes
``FRONT_CAMERA_POS``).

Scoring is parquet-only. A small video sample is written under ``scratch/``.
Nothing under ``--root`` is written, moved, or rewritten.

    uv run python scripts/audit_goal_target_alignment.py --root <dataset>
"""

from __future__ import annotations

import csv
import dataclasses
import json
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import tyro

from toyblocks.utils.dataset_provenance import camera_short_name
from toyblocks.utils.episode_dataset import (
    _COL_DATA_CHUNK_INDEX,
    _COL_DATA_FILE_INDEX,
    _COL_EPISODE_INDEX,
    _format_video_path_template,
    _load_lerobot_info,
    _resolve_parquet_files,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GOAL_SPEC_SRC = _REPO_ROOT / "TAMP" / "goal_spec" / "src"

_COL_FRAME_INDEX = "frame_index"
_COL_TIMESTAMP = "timestamp"
_COL_GOAL_ACTION = "goal.action"
_COL_GOAL_SECONDARY = "goal.action_secondary"
_COL_GOAL_POSITION = "goal.position"
_COL_EE_POSITION = "observation.state.ee_position"
_COL_CAMERA_POSE_PREFIX = "observation.camera_pose."
_COL_CAMERA_INTRINSICS_PREFIX = "observation.camera_intrinsics."
_COL_IMAGE_PREFIX = "observation.images."

_DEFAULT_VIDEO_KEY = "observation.images.camera:front_camera"
_KNOWN_LOWRES_PX = 224.0
_LOWRES_VIDEO_MIN = 336
_CRF = 28
_GOP = 250

Bucket = Literal["fail", "boundary", "clean"]
_BUCKET_RANK: dict[Bucket, int] = {"clean": 0, "boundary": 1, "fail": 2}

SCALE = 0.46
THICK = 1
OUTLINE = 3
LINE_H = 18
PAD = 7
MARGIN = 8
GREY = (200, 200, 200)
CYAN = (120, 230, 255)
WHITE = (255, 255, 255)
DIM = (170, 170, 170)
GOLD = (40, 180, 220)
GOAL_MARKER_BGR = (0, 200, 255)
GRIP_MARKER_BGR = (255, 0, 255)
LINE_BGR = (220, 220, 220)
MARKER_RADIUS = 8
MARKER_THICKNESS = 2
MARKER_CROSS_LEN = 14
TAIL_BAR_PX = 6


@dataclasses.dataclass
class Args:
    root: Path
    """LeRobot dataset root. Read-only; never written."""
    out: Path | None = None
    """Scratch directory. Defaults to ``scratch/goal_target_audit_<timestamp>``."""
    sample_frac: float = 0.1
    """Fraction of episode recordings to consider (no replacement)."""
    max_episodes: int = 40
    """Hard cap on how many recordings to score. 10% of a merged set is too large."""
    seed: int = 0
    tail_frac: float = 0.2
    """Last this fraction of each subskill is scored."""
    fail_px: float = 80.0
    """Subskill median distance above this is ``fail``."""
    clean_px: float = 25.0
    """Subskill median distance below this is ``clean``."""
    video_key: str = _DEFAULT_VIDEO_KEY
    n_fail: int = 20
    n_boundary: int = 5
    n_clean: int = 5
    n_success: int = 2
    """Always write this many of the lowest-distance scored recordings to videos/success/."""


@dataclasses.dataclass(frozen=True)
class CameraColumns:
    pose: str
    intrinsics: str


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    """One recording. ``episode_index`` is not unique in merged datasets."""

    episode: int
    data_chunk: int
    data_file: int

    @property
    def label(self) -> str:
        return (
            f"{self.episode}@chunk-{self.data_chunk:03d}/file-{self.data_file:03d}"
        )


@dataclasses.dataclass(frozen=True)
class SubskillScore:
    episode: int
    data_chunk: int
    data_file: int
    action: str
    start_frame: int
    end_frame: int
    n_tail: int
    median_px: float
    mean_du: float
    mean_dv: float
    bucket: Bucket


@dataclasses.dataclass(frozen=True)
class EpisodeScore:
    ref: EpisodeRef
    bucket: Bucket
    worst_median: float
    subskills: tuple[SubskillScore, ...]

    @property
    def episode(self) -> int:
        return self.ref.episode


@dataclasses.dataclass(frozen=True)
class EpisodeFrames:
    ref: EpisodeRef
    episode: int
    frame_index: np.ndarray
    timestamp: np.ndarray | None
    action: list[str]
    secondary: list[str] | None
    goal_uv: np.ndarray
    grip_uv: np.ndarray
    dist: np.ndarray
    is_tail: np.ndarray


def _import_world_to_camera_points() -> Any:
    try:
        from goal_spec.geometry.transforms import world_to_camera_points
    except ImportError:
        world_to_camera_points = None
    if world_to_camera_points is not None:
        return world_to_camera_points
    if not _GOAL_SPEC_SRC.is_dir():
        raise ImportError(
            "goal_spec.geometry.transforms.world_to_camera_points is not "
            f"importable and {_GOAL_SPEC_SRC} does not exist. The audit "
            "projector must come from TAMP goal_spec; it will not use "
            "FRONT_CAMERA_POS."
        )
    src = str(_GOAL_SPEC_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        from goal_spec.geometry.transforms import world_to_camera_points
    except ImportError as exc:
        raise ImportError(
            "failed to import world_to_camera_points from "
            f"{_GOAL_SPEC_SRC}. The audit cannot drift onto a local projector."
        ) from exc
    return world_to_camera_points


def _cv2() -> Any:
    import cv2

    return cv2


def _require_lerobot_root(root: Path) -> Path:
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"--root is not a directory: {root}")
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"{info_path} is missing; --root must be a single LeRobot dataset "
            "(not a parent of several, and not a file)."
        )
    return root


def _list_episode_refs(root: Path) -> list[EpisodeRef]:
    """List recordings as ``(episode_index, data/chunk, data/file)``.

    Merged datasets reuse ``episode_index`` across source shards. Those are
    different trajectories; the data-shard coordinates distinguish them.
    """
    import pyarrow as pa

    meta_dir = root / "meta" / "episodes"
    files = sorted(meta_dir.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"{meta_dir} has no chunk-*/file-*.parquet; episode recordings "
            "cannot be listed."
        )
    columns = [_COL_EPISODE_INDEX, _COL_DATA_CHUNK_INDEX, _COL_DATA_FILE_INDEX]
    names = set(pq.read_schema(str(files[0])).names)
    missing = [c for c in columns if c not in names]
    if missing:
        raise ValueError(
            f"{files[0]}: missing {missing}; merged episode_index values "
            "cannot be disambiguated without data-shard coordinates."
        )
    table = pa.concat_tables([pq.read_table(str(path), columns=columns) for path in files])
    for col in columns:
        if table.column(col).null_count:
            raise ValueError(f"{meta_dir}: null {col}")
    uniq = table.group_by(columns).aggregate([]).to_pylist()
    if not uniq:
        raise ValueError(f"{meta_dir}: no episode recordings")
    refs = [
        EpisodeRef(
            episode=int(row[_COL_EPISODE_INDEX]),
            data_chunk=int(row[_COL_DATA_CHUNK_INDEX]),
            data_file=int(row[_COL_DATA_FILE_INDEX]),
        )
        for row in uniq
    ]
    refs.sort(key=lambda r: (r.episode, r.data_chunk, r.data_file))
    return refs


def _schema_names(root: Path) -> set[str]:
    shards = _resolve_parquet_files(root)
    return set(pq.read_schema(str(shards[0])).names)


def _resolve_camera_columns(names: set[str], video_key: str) -> CameraColumns:
    if not video_key.startswith(_COL_IMAGE_PREFIX):
        raise ValueError(
            f"--video-key {video_key!r} must start with {_COL_IMAGE_PREFIX!r}"
        )
    video_suffix = video_key[len(_COL_IMAGE_PREFIX) :]
    exact_pose = f"{_COL_CAMERA_POSE_PREFIX}{video_suffix}"
    exact_intr = f"{_COL_CAMERA_INTRINSICS_PREFIX}{video_suffix}"
    if exact_pose in names and exact_intr in names:
        return CameraColumns(pose=exact_pose, intrinsics=exact_intr)

    wanted = camera_short_name(video_key, _COL_IMAGE_PREFIX)
    pose_matches = [
        c
        for c in names
        if c.startswith(_COL_CAMERA_POSE_PREFIX)
        and camera_short_name(c, _COL_CAMERA_POSE_PREFIX) == wanted
    ]
    intr_matches = [
        c
        for c in names
        if c.startswith(_COL_CAMERA_INTRINSICS_PREFIX)
        and camera_short_name(c, _COL_CAMERA_INTRINSICS_PREFIX) == wanted
    ]
    if len(pose_matches) != 1 or len(intr_matches) != 1:
        raise ValueError(
            f"need exactly one recorded pose+K pair for camera {wanted!r} "
            f"(video-key {video_key!r}). pose={pose_matches} "
            f"intrinsics={intr_matches}. No FRONT_CAMERA_POS fallback."
        )
    return CameraColumns(pose=pose_matches[0], intrinsics=intr_matches[0])


def _require_columns(names: set[str], required: list[str], *, where: str) -> None:
    missing = [c for c in required if c not in names]
    if missing:
        raise ValueError(f"{where}: missing required columns {missing}")


def _video_hw(info: Mapping[str, Any], video_key: str) -> tuple[int, int]:
    spec = (info.get("features") or {}).get(video_key)
    if not spec:
        raise ValueError(
            f"meta/info.json has no feature {video_key!r}; video size is "
            "required for the goal.position resolution check."
        )
    shape = spec.get("shape")
    if not isinstance(shape, (list, tuple)) or len(shape) < 2:
        raise ValueError(
            f"meta/info.json feature {video_key!r} has no [H, W, ...] shape"
        )
    height, width = int(shape[0]), int(shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(
            f"meta/info.json feature {video_key!r} shape={shape} is not a "
            "positive HxW"
        )
    return height, width


def _action_str(raw: object) -> str:
    if raw is None:
        return ""
    if isinstance(raw, float) and not np.isfinite(raw):
        return ""
    if isinstance(raw, bytes):
        return raw.decode("utf-8").strip()
    return str(raw).strip()


def _finite_uv(raw: object) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < 2 or not np.isfinite(arr[0]) or not np.isfinite(arr[1]):
        return None
    return arr[:2].copy()


def _as_vec(raw: object, n: int, *, what: str, episode: int, frame: int) -> np.ndarray:
    if raw is None:
        raise ValueError(f"episode {episode} frame {frame}: {what} is null")
    arr = np.asarray(raw, dtype=np.float64).reshape(-1)
    if arr.size < n:
        raise ValueError(
            f"episode {episode} frame {frame}: {what} has {arr.size} values, "
            f"need {n}"
        )
    if not np.isfinite(arr[:n]).all():
        raise ValueError(
            f"episode {episode} frame {frame}: {what} is not finite: {arr[:n]}"
        )
    return arr[:n].copy()


def _project_ee(
    ee_xyz: np.ndarray,
    cam_pose: np.ndarray,
    cam_k: np.ndarray,
    world_to_camera_points: Any,
    *,
    episode: int,
    frame: int,
) -> np.ndarray:
    if not np.any(cam_pose):
        raise ValueError(
            f"episode {episode} frame {frame}: camera pose is all-zero; "
            "refusing FRONT_CAMERA_POS"
        )
    if not np.any(cam_k):
        raise ValueError(
            f"episode {episode} frame {frame}: camera K is all-zero; "
            "refusing a default K"
        )
    p_cam = world_to_camera_points(ee_xyz, cam_pose[:3], cam_pose[3:7])[0]
    z = float(p_cam[2])
    if z <= 0.0:
        raise ValueError(
            f"episode {episode} frame {frame}: EE is behind the recorded "
            f"camera (z={z})"
        )
    fx, fy, cx, cy = float(cam_k[0]), float(cam_k[4]), float(cam_k[2]), float(cam_k[5])
    u = fx * float(p_cam[0]) / z + cx
    v = fy * float(p_cam[1]) / z + cy
    if not np.isfinite(u) or not np.isfinite(v):
        raise ValueError(
            f"episode {episode} frame {frame}: non-finite gripper pixel ({u}, {v})"
        )
    return np.array([u, v], dtype=np.float64)


def _bucket(score: float, *, fail_px: float, clean_px: float) -> Bucket:
    if fail_px <= clean_px:
        raise ValueError(f"--fail-px ({fail_px}) must be > --clean-px ({clean_px})")
    if score < clean_px:
        return "clean"
    if score > fail_px:
        return "fail"
    return "boundary"


def _worse(a: Bucket, b: Bucket) -> Bucket:
    return a if _BUCKET_RANK[a] >= _BUCKET_RANK[b] else b


def _data_shard_for_ref(root: Path, ref: EpisodeRef) -> Path:
    parquet_files = _resolve_parquet_files(root)
    data_dir = parquet_files[0].parent.parent
    shard = data_dir / f"chunk-{ref.data_chunk:03d}" / f"file-{ref.data_file:03d}.parquet"
    if not shard.is_file():
        raise FileNotFoundError(f"{ref.label}: data shard missing at {shard}")
    return shard


def _read_episode_tables(
    root: Path,
    refs: list[EpisodeRef],
    columns: list[str],
) -> dict[EpisodeRef, dict[str, list[Any]]]:
    by_shard: dict[Path, list[EpisodeRef]] = defaultdict(list)
    for ref in refs:
        by_shard[_data_shard_for_ref(root, ref)].append(ref)
    out: dict[EpisodeRef, dict[str, list[Any]]] = {}
    for shard, shard_refs in by_shard.items():
        wanted_eps = sorted({r.episode for r in shard_refs})
        table = pq.read_table(
            str(shard),
            columns=columns,
            filters=[(_COL_EPISODE_INDEX, "in", wanted_eps)],
        )
        for ref in shard_refs:
            part = table.filter(pc.equal(table.column(_COL_EPISODE_INDEX), ref.episode))
            if part.num_rows == 0:
                raise ValueError(f"{ref.label}: no data rows in {shard}")
            part = part.sort_by([(_COL_FRAME_INDEX, "ascending")])
            frames = [int(v) for v in part.column(_COL_FRAME_INDEX).to_pylist()]
            if len(frames) != len(set(frames)):
                raise ValueError(f"{ref.label}: duplicate frame_index in {shard}")
            out[ref] = part.to_pydict()
    missing = [r.label for r in refs if r not in out]
    if missing:
        raise ValueError(f"{root}: no data rows for {missing}")
    return out


def _subskill_spans(
    actions: list[str], goal_uv: np.ndarray
) -> list[tuple[str, int, int]]:
    spans: list[tuple[str, int, int]] = []
    start: int | None = None
    current = ""
    for i, action in enumerate(actions):
        ok = bool(action) and bool(np.isfinite(goal_uv[i]).all())
        if not ok:
            if start is not None:
                spans.append((current, start, i - 1))
                start = None
            continue
        if start is None:
            start = i
            current = action
        elif action != current:
            spans.append((current, start, i - 1))
            start = i
            current = action
    if start is not None:
        spans.append((current, start, len(actions) - 1))
    return spans


def _score_episode(
    raw: Mapping[str, list[Any]],
    camera: CameraColumns,
    world_to_camera_points: Any,
    *,
    ref: EpisodeRef,
    tail_frac: float,
    fail_px: float,
    clean_px: float,
    has_secondary: bool,
    has_timestamp: bool,
) -> tuple[EpisodeScore, EpisodeFrames]:
    episode = ref.episode
    n = len(raw[_COL_EPISODE_INDEX])
    if n == 0:
        raise ValueError(f"{ref.label}: zero rows")
    frame_index = np.asarray(raw[_COL_FRAME_INDEX], dtype=np.int64)
    if frame_index.size != n:
        raise ValueError(f"episode {episode}: frame_index length mismatch")
    actions = [_action_str(a) for a in raw[_COL_GOAL_ACTION]]
    secondary = (
        [_action_str(a) for a in raw[_COL_GOAL_SECONDARY]] if has_secondary else None
    )
    timestamps: np.ndarray | None = None
    if has_timestamp:
        timestamps = np.empty(n, dtype=np.float64)
        for i, raw_t in enumerate(raw[_COL_TIMESTAMP]):
            if raw_t is None:
                raise ValueError(
                    f"episode {episode} frame {int(frame_index[i])}: "
                    "timestamp column is present but the cell is null"
                )
            t = float(raw_t)
            if not np.isfinite(t):
                raise ValueError(
                    f"episode {episode} frame {int(frame_index[i])}: "
                    f"non-finite timestamp {raw_t!r}"
                )
            timestamps[i] = t

    goal_uv = np.full((n, 2), np.nan, dtype=np.float64)
    grip_uv = np.empty((n, 2), dtype=np.float64)
    dist = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        frame = int(frame_index[i])
        uv = _finite_uv(raw[_COL_GOAL_POSITION][i])
        if uv is not None:
            goal_uv[i] = uv
        ee = _as_vec(
            raw[_COL_EE_POSITION][i],
            3,
            what=_COL_EE_POSITION,
            episode=episode,
            frame=frame,
        )
        pose = _as_vec(
            raw[camera.pose][i],
            7,
            what=camera.pose,
            episode=episode,
            frame=frame,
        )
        k = _as_vec(
            raw[camera.intrinsics][i],
            9,
            what=camera.intrinsics,
            episode=episode,
            frame=frame,
        )
        grip_uv[i] = _project_ee(
            ee,
            pose,
            k,
            world_to_camera_points,
            episode=episode,
            frame=frame,
        )
        if np.isfinite(goal_uv[i]).all():
            dist[i] = float(np.linalg.norm(grip_uv[i] - goal_uv[i]))

    is_tail = np.zeros(n, dtype=bool)
    spans = _subskill_spans(actions, goal_uv)
    if not spans:
        raise ValueError(
            f"episode {episode}: no scorable subskill (need a non-empty "
            f"{_COL_GOAL_ACTION} run with finite {_COL_GOAL_POSITION})"
        )
    scored: list[SubskillScore] = []
    for action, start, end in spans:
        length = end - start + 1
        n_tail = max(1, int(math.ceil(tail_frac * length)))
        tail = slice(end - n_tail + 1, end + 1)
        is_tail[tail] = True
        tail_dist = dist[tail]
        if not np.isfinite(tail_dist).all():
            raise ValueError(
                f"episode {episode} action {action!r} frames "
                f"{int(frame_index[start])}-{int(frame_index[end])}: tail "
                "distance is not finite"
            )
        delta = grip_uv[tail] - goal_uv[tail]
        median_px = float(np.median(tail_dist))
        scored.append(
            SubskillScore(
                episode=episode,
                data_chunk=ref.data_chunk,
                data_file=ref.data_file,
                action=action,
                start_frame=int(frame_index[start]),
                end_frame=int(frame_index[end]),
                n_tail=n_tail,
                median_px=median_px,
                mean_du=float(np.mean(delta[:, 0])),
                mean_dv=float(np.mean(delta[:, 1])),
                bucket=_bucket(median_px, fail_px=fail_px, clean_px=clean_px),
            )
        )
    worst = scored[0]
    for row in scored[1:]:
        if _BUCKET_RANK[row.bucket] > _BUCKET_RANK[worst.bucket] or (
            row.bucket == worst.bucket and row.median_px > worst.median_px
        ):
            worst = row
    return (
        EpisodeScore(
            ref=ref,
            bucket=worst.bucket,
            worst_median=worst.median_px,
            subskills=tuple(scored),
        ),
        EpisodeFrames(
            ref=ref,
            episode=episode,
            frame_index=frame_index,
            timestamp=timestamps,
            action=actions,
            secondary=secondary,
            goal_uv=goal_uv,
            grip_uv=grip_uv,
            dist=dist,
            is_tail=is_tail,
        ),
    )


def _assert_goal_resolution(
    positions: np.ndarray, *, width: int, height: int
) -> None:
    finite = positions[np.isfinite(positions).all(axis=1)]
    if finite.size == 0:
        raise ValueError("sampled episodes have no finite goal.position")
    u_hi = float(np.percentile(finite[:, 0], 99))
    v_hi = float(np.percentile(finite[:, 1], 99))
    if (
        width > _LOWRES_VIDEO_MIN
        and height > _LOWRES_VIDEO_MIN
        and u_hi <= _KNOWN_LOWRES_PX + 0.5
        and v_hi <= _KNOWN_LOWRES_PX + 0.5
    ):
        raise ValueError(
            f"goal.position is clustered in 0–{_KNOWN_LOWRES_PX:.0f} "
            f"(p99=({u_hi:.1f}, {v_hi:.1f})) but the video is {width}x{height}. "
            "Refusing to silently rescale."
        )


def _pick_videos(
    episodes: list[EpisodeScore],
    *,
    n_fail: int,
    n_boundary: int,
    n_clean: int,
    fail_px: float,
    clean_px: float,
) -> dict[Bucket, list[EpisodeScore]]:
    by_bucket: dict[Bucket, list[EpisodeScore]] = {
        "fail": [],
        "boundary": [],
        "clean": [],
    }
    for ep in episodes:
        by_bucket[ep.bucket].append(ep)
    by_bucket["fail"].sort(key=lambda e: e.worst_median, reverse=True)
    by_bucket["clean"].sort(key=lambda e: e.worst_median)
    mid = 0.5 * (fail_px + clean_px)
    by_bucket["boundary"].sort(key=lambda e: abs(e.worst_median - mid))
    return {
        "fail": by_bucket["fail"][:n_fail],
        "boundary": by_bucket["boundary"][:n_boundary],
        "clean": by_bucket["clean"][:n_clean],
    }


def _pick_success(episodes: list[EpisodeScore], n: int) -> list[EpisodeScore]:
    """Lowest worst-subskill distance in the sample (the most successful)."""
    ranked = sorted(episodes, key=lambda e: e.worst_median)
    return ranked[:n]


def _rates(counts: Mapping[str, int], total: int) -> dict[str, float] | None:
    if total <= 0:
        return None
    return {k: counts[k] / total for k in ("fail", "boundary", "clean")}


def _percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0:
        raise ValueError("no subskill scores")
    return float(np.percentile(values, q))


def _write_csv(path: Path, rows: list[SubskillScore]) -> None:
    fields = [
        "episode",
        "data_chunk",
        "data_file",
        "action",
        "start_frame",
        "end_frame",
        "n_tail",
        "median_px",
        "mean_du",
        "mean_dv",
        "bucket",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(dataclasses.asdict(row))


def _write_flagged_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fields = ["recording", "stage", "error"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_histogram(path: Path, medians: np.ndarray, *, fail_px: float, clean_px: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(medians, bins=40, color="#4c78a8", edgecolor="white")
    ax.axvline(clean_px, color="#54a24b", linewidth=1.5, label=f"clean < {clean_px:g}px")
    ax.axvline(fail_px, color="#e45756", linewidth=1.5, label=f"fail > {fail_px:g}px")
    ax.set_xlabel("subskill median gripper–goal distance (px)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _fmt_uv(uv: np.ndarray) -> str:
    if not np.isfinite(uv).all():
        return "n/a"
    return f"({uv[0]:.1f}, {uv[1]:.1f})"


def _draw_panel(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    cv2 = _cv2()
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = max(cv2.getTextSize(t, font, SCALE, OUTLINE)[0][0] for t, _ in lines) + 2 * PAD
    x1 = min(MARGIN + width, frame.shape[1] - 1)
    y1 = MARGIN + LINE_H * len(lines) + 2 * PAD
    box = frame[MARGIN:y1, MARGIN:x1]
    frame[MARGIN:y1, MARGIN:x1] = (box * 0.35).astype(np.uint8)
    cv2.rectangle(frame, (MARGIN, MARGIN), (x1, y1), (90, 90, 90), 1)
    y = MARGIN + PAD + 10
    for text, colour in lines:
        cv2.putText(frame, text, (MARGIN + PAD, y), font, SCALE, (0, 0, 0), OUTLINE, cv2.LINE_AA)
        cv2.putText(frame, text, (MARGIN + PAD, y), font, SCALE, colour, THICK, cv2.LINE_AA)
        y += LINE_H


def _draw_marker(frame: np.ndarray, uv: np.ndarray, color: tuple[int, int, int]) -> None:
    if not np.isfinite(uv).all():
        return
    cv2 = _cv2()
    u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
    h, w = frame.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return
    cv2.circle(frame, (u, v), MARKER_RADIUS, color, MARKER_THICKNESS)
    cv2.drawMarker(
        frame, (u, v), color, cv2.MARKER_CROSS, MARKER_CROSS_LEN, MARKER_THICKNESS
    )


def _overlay_frame(frame: np.ndarray, rec: EpisodeFrames, i: int, n_rows: int) -> None:
    cv2 = _cv2()
    if rec.is_tail[i]:
        frame[-TAIL_BAR_PX:, :] = GOLD
    goal = rec.goal_uv[i]
    grip = rec.grip_uv[i]
    if np.isfinite(goal).all() and np.isfinite(grip).all():
        p0 = (int(round(float(goal[0]))), int(round(float(goal[1]))))
        p1 = (int(round(float(grip[0]))), int(round(float(grip[1]))))
        cv2.line(frame, p0, p1, LINE_BGR, 1, cv2.LINE_AA)
    _draw_marker(frame, goal, GOAL_MARKER_BGR)
    _draw_marker(frame, grip, GRIP_MARKER_BGR)

    if rec.timestamp is None:
        t_str = "n/a"
    else:
        t_str = f"{rec.timestamp[i]:.2f}s"
    secondary = rec.secondary[i] if rec.secondary is not None else "-"
    dist = rec.dist[i]
    dist_str = f"{dist:.1f} px" if np.isfinite(dist) else "n/a"
    if rec.is_tail[i]:
        dist_str = f"{dist_str}  TAIL"
    _draw_panel(
        frame,
        [
            (
                f"ep {rec.ref.label}  frame {int(rec.frame_index[i])}/{n_rows - 1}"
                f"  t={t_str}",
                GREY,
            ),
            (f"goal.action: {rec.action[i] or '-'}", CYAN),
            (f"goal.position: {_fmt_uv(goal)}", WHITE),
            (f"secondary: {secondary or '-'}", DIM),
            (f"gripper: {_fmt_uv(grip)}", WHITE),
            (f"dist: {dist_str}", WHITE),
        ],
    )


def _probe(video: str) -> tuple[int, int, float]:
    out = subprocess.check_output(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-of",
            "json",
            "-show_entries",
            "stream=width,height,r_frame_rate",
            video,
        ]
    )
    stream = json.loads(out)["streams"][0]
    num, den = stream["r_frame_rate"].split("/")
    width, height = int(stream["width"]), int(stream["height"])
    fps = float(num) / float(den)
    if width <= 0 or height <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError(f"{video}: invalid probe width={width} height={height} fps={fps}")
    return width, height, fps


def _decoder_cmd(video: str, meta: Mapping[str, Any], n_rows: int) -> list[str]:
    cmd = ["ffmpeg", "-v", "error", "-i", video]
    from_ts = meta.get("from_timestamp")
    to_ts = meta.get("to_timestamp")
    if from_ts is not None:
        cmd.extend(["-ss", f"{float(from_ts):.6f}"])
    if to_ts is not None:
        cmd.extend(["-to", f"{float(to_ts):.6f}"])
    elif from_ts is not None:
        cmd.extend(["-frames:v", str(n_rows)])
    cmd.extend(["-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    return cmd


def _video_entries_for_refs(
    root: Path, refs: list[EpisodeRef], video_key: str
) -> dict[EpisodeRef, dict[str, Any]]:
    """Resolve the mp4 for each recording from its own meta/episodes row."""
    if not refs:
        return {}
    info = _load_lerobot_info(root)
    template = info.get("video_path") or (
        "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    chunk_col = f"videos/{video_key}/chunk_index"
    file_col = f"videos/{video_key}/file_index"
    from_col = f"videos/{video_key}/from_timestamp"
    to_col = f"videos/{video_key}/to_timestamp"
    meta_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not meta_files:
        raise FileNotFoundError(f"{root}/meta/episodes has no parquet shards")
    names = set(pq.read_schema(str(meta_files[0])).names)
    if chunk_col not in names or file_col not in names:
        raise ValueError(
            f"meta/episodes is missing {chunk_col!r}/{file_col!r}; cannot "
            "locate a video when episode_index is reused across shards."
        )
    columns = [
        _COL_EPISODE_INDEX,
        _COL_DATA_CHUNK_INDEX,
        _COL_DATA_FILE_INDEX,
        chunk_col,
        file_col,
    ]
    if from_col in names:
        columns.append(from_col)
    if to_col in names:
        columns.append(to_col)
    wanted = set(refs)
    filters = [(_COL_EPISODE_INDEX, "in", sorted({r.episode for r in refs}))]
    found: dict[EpisodeRef, dict[str, Any]] = {}
    for path in meta_files:
        table = pq.read_table(str(path), columns=columns, filters=filters)
        for row in table.to_pylist():
            ref = EpisodeRef(
                episode=int(row[_COL_EPISODE_INDEX]),
                data_chunk=int(row[_COL_DATA_CHUNK_INDEX]),
                data_file=int(row[_COL_DATA_FILE_INDEX]),
            )
            if ref not in wanted:
                continue
            v_chunk = row[chunk_col]
            v_file = row[file_col]
            if v_chunk is None or v_file is None:
                raise ValueError(f"{ref.label}: null video chunk/file in {path}")
            rel = _format_video_path_template(
                template,
                video_key=video_key,
                chunk_index=int(v_chunk),
                file_index=int(v_file),
            )
            video_path = Path(rel)
            if not video_path.is_absolute():
                video_path = root / video_path
            entry: dict[str, Any] = {"path": str(video_path.resolve())}
            if from_col in row and row[from_col] is not None:
                entry["from_timestamp"] = float(row[from_col])
            if to_col in row and row[to_col] is not None:
                entry["to_timestamp"] = float(row[to_col])
            prev = found.get(ref)
            if prev is not None and prev != entry:
                raise ValueError(
                    f"{ref.label}: conflicting video metadata {prev} vs {entry}"
                )
            found[ref] = entry
    missing = [r.label for r in refs if r not in found]
    if missing:
        raise FileNotFoundError(
            f"no meta/episodes video row for {missing} (no nearest-episode fallback)"
        )
    absent = [f"{r.label} -> {found[r]['path']}" for r in refs if not Path(found[r]["path"]).is_file()]
    if absent:
        raise FileNotFoundError(f"video file missing: {absent}")
    return found


def _annotate_video(
    rec: EpisodeFrames,
    meta: Mapping[str, Any],
    out_path: Path,
    *,
    expected_hw: tuple[int, int],
) -> int:
    video = str(meta["path"])
    if not Path(video).is_file():
        raise FileNotFoundError(f"episode {rec.episode}: video missing at {video}")
    n_rows = int(rec.frame_index.size)
    width, height, fps = _probe(video)
    exp_h, exp_w = expected_hw
    if (height, width) != (exp_h, exp_w):
        raise ValueError(
            f"episode {rec.episode}: probed video is {width}x{height} but "
            f"meta/info.json declares {exp_w}x{exp_h}"
        )
    nbytes = width * height * 3
    decoder = subprocess.Popen(
        _decoder_cmd(video, meta, n_rows),
        stdout=subprocess.PIPE,
        bufsize=nbytes * 4,
    )
    encoder = subprocess.Popen(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(_CRF),
            "-g",
            str(_GOP),
            "-keyint_min",
            str(_GOP),
            "-x264-params",
            f"keyint={_GOP}:min-keyint={_GOP}:scenecut=0",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
    )
    assert decoder.stdout is not None
    assert encoder.stdin is not None
    written = 0
    try:
        while True:
            buf = decoder.stdout.read(nbytes)
            if len(buf) < nbytes:
                break
            frame = np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()
            if written >= n_rows:
                raise ValueError(
                    f"episode {rec.episode}: video has more frames than "
                    f"parquet rows ({n_rows})"
                )
            _overlay_frame(frame, rec, written, n_rows)
            encoder.stdin.write(frame.tobytes())
            written += 1
    finally:
        decoder.stdout.close()
        decoder.wait()
        encoder.stdin.close()
        enc_rc = encoder.wait()
        dec_rc = decoder.returncode
    if dec_rc != 0:
        raise RuntimeError(
            f"ffmpeg decode failed for episode {rec.episode} ({video}), rc={dec_rc}"
        )
    if enc_rc != 0:
        raise RuntimeError(f"ffmpeg encode failed for episode {rec.episode}, rc={enc_rc}")
    if written != n_rows:
        raise ValueError(
            f"episode {rec.episode}: decoded {written} frames, parquet has {n_rows}"
        )
    return written


def _build_summary(
    *,
    root: Path,
    out: Path,
    args: Args,
    camera: CameraColumns,
    n_dataset: int,
    sampled: list[EpisodeRef],
    episode_scores: list[EpisodeScore],
    subskills: list[SubskillScore],
    chosen: dict[Bucket, list[EpisodeScore]],
    video_notes: list[str],
    flagged: list[dict[str, str]],
    success_videos: list[str],
) -> dict[str, Any]:
    medians = np.asarray([s.median_px for s in subskills], dtype=np.float64)
    dus = np.asarray([s.mean_du for s in subskills], dtype=np.float64)
    dvs = np.asarray([s.mean_dv for s in subskills], dtype=np.float64)
    fail_mask = np.asarray([s.bucket == "fail" for s in subskills])
    sub_counts = {
        "fail": int(sum(1 for s in subskills if s.bucket == "fail")),
        "boundary": int(sum(1 for s in subskills if s.bucket == "boundary")),
        "clean": int(sum(1 for s in subskills if s.bucket == "clean")),
    }
    ep_counts = {
        "fail": int(sum(1 for e in episode_scores if e.bucket == "fail")),
        "boundary": int(sum(1 for e in episode_scores if e.bucket == "boundary")),
        "clean": int(sum(1 for e in episode_scores if e.bucket == "clean")),
        "flagged": len(flagged),
    }
    by_action: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[SubskillScore]] = defaultdict(list)
    for row in subskills:
        grouped[row.action].append(row)
    for action, rows in sorted(grouped.items()):
        counts = {
            "fail": int(sum(1 for r in rows if r.bucket == "fail")),
            "boundary": int(sum(1 for r in rows if r.bucket == "boundary")),
            "clean": int(sum(1 for r in rows if r.bucket == "clean")),
        }
        vals = np.asarray([r.median_px for r in rows], dtype=np.float64)
        by_action[action] = {
            "n": len(rows),
            "counts": counts,
            "rates": _rates(counts, len(rows)),
            "median_px": float(np.median(vals)),
            "mean_du": float(np.mean([r.mean_du for r in rows])),
            "mean_dv": float(np.mean([r.mean_dv for r in rows])),
        }
    fail_du = float(np.mean(dus[fail_mask])) if fail_mask.any() else None
    fail_dv = float(np.mean(dvs[fail_mask])) if fail_mask.any() else None
    return {
        "root": str(root),
        "out": str(out),
        "sample_frac": args.sample_frac,
        "max_episodes": args.max_episodes,
        "seed": args.seed,
        "tail_frac": args.tail_frac,
        "fail_px": args.fail_px,
        "clean_px": args.clean_px,
        "video_key": args.video_key,
        "camera_pose_column": camera.pose,
        "camera_intrinsics_column": camera.intrinsics,
        "n_episodes_dataset": n_dataset,
        "n_episodes_sampled": len(sampled),
        "sampled_episodes": [r.label for r in sampled],
        "n_subskills": len(subskills),
        "n_flagged": len(flagged),
        "flagged": flagged,
        "subskill_counts": sub_counts,
        "subskill_rates": _rates(sub_counts, len(subskills)),
        "episode_counts": ep_counts,
        "episode_rates": _rates(ep_counts, len(episode_scores)),
        "per_goal_action": by_action,
        "median_px": float(np.median(medians)) if medians.size else None,
        "p90_px": _percentile(medians, 90) if medians.size else None,
        "p99_px": _percentile(medians, 99) if medians.size else None,
        "mean_du": float(np.mean(dus)) if dus.size else None,
        "mean_dv": float(np.mean(dvs)) if dvs.size else None,
        "fail_mean_du": fail_du,
        "fail_mean_dv": fail_dv,
        "videos_requested": {
            "fail": args.n_fail,
            "boundary": args.n_boundary,
            "clean": args.n_clean,
        },
        "videos_written": {k: [e.ref.label for e in v] for k, v in chosen.items()},
        "success_videos": success_videos,
        "video_notes": video_notes,
    }


def _write_summary_txt(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "goal-target alignment audit",
        f"root: {summary['root']}",
        f"out:  {summary['out']}",
        f"camera pose: {summary['camera_pose_column']}",
        f"camera K:    {summary['camera_intrinsics_column']}",
        (
            f"episodes: {summary['n_episodes_sampled']} / "
            f"{summary['n_episodes_dataset']} sampled "
            f"(frac={summary['sample_frac']}, seed={summary['seed']})"
        ),
        f"subskills: {summary['n_subskills']}",
        f"flagged: {summary['n_flagged']}",
        (
            "subskill rates: "
            f"{summary['subskill_rates']} (n={summary['subskill_counts']})"
        ),
        (
            "episode rates (worst subskill): "
            f"{summary['episode_rates']} (n={summary['episode_counts']})"
        ),
        (
            f"distance px: median={summary['median_px']} "
            f"p90={summary['p90_px']} p99={summary['p99_px']}"
        ),
        (
            f"mean (du, dv): ({summary['mean_du']}, {summary['mean_dv']})  "
            f"fails: ({summary['fail_mean_du']}, {summary['fail_mean_dv']})"
        ),
        "per goal.action:",
    ]
    for action, stats in summary["per_goal_action"].items():
        lines.append(
            f"  {action or repr('')}: n={stats['n']} rates={stats['rates']} "
            f"median={stats['median_px']:.1f}px "
            f"mean_d=({stats['mean_du']:.1f}, {stats['mean_dv']:.1f})"
        )
    lines.append(
        "videos: "
        + ", ".join(
            f"{bucket}={len(eps)}/{summary['videos_requested'][bucket]}"
            for bucket, eps in summary["videos_written"].items()
        )
    )
    lines.append(f"success videos: {summary['success_videos']}")
    for note in summary["video_notes"]:
        lines.append(f"note: {note}")
    if summary["flagged"]:
        lines.append("flagged recordings:")
        for row in summary["flagged"]:
            lines.append(f"  [{row['stage']}] {row['recording']}: {row['error']}")
    path.write_text("\n".join(lines) + "\n")


def main(args: Args) -> None:
    if not (0.0 < args.sample_frac <= 1.0):
        raise ValueError(f"--sample-frac must be in (0, 1], got {args.sample_frac}")
    if not (0.0 < args.tail_frac <= 1.0):
        raise ValueError(f"--tail-frac must be in (0, 1], got {args.tail_frac}")
    if args.fail_px <= args.clean_px:
        raise ValueError(f"--fail-px ({args.fail_px}) must be > --clean-px ({args.clean_px})")
    if args.max_episodes < 1:
        raise ValueError(f"--max-episodes must be >= 1, got {args.max_episodes}")
    if args.n_fail < 0 or args.n_boundary < 0 or args.n_clean < 0 or args.n_success < 0:
        raise ValueError("video counts must be >= 0")

    root = _require_lerobot_root(args.root)
    out = args.out
    if out is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = _REPO_ROOT / "scratch" / f"goal_target_audit_{stamp}"
    out = out.resolve()
    if out.is_relative_to(root):
        raise ValueError(f"--out {out} is inside --root {root}; refusing to write there")
    out.mkdir(parents=True, exist_ok=False)

    world_to_camera_points = _import_world_to_camera_points()
    info = _load_lerobot_info(root)
    if not info:
        raise FileNotFoundError(f"{root / 'meta' / 'info.json'} is empty")
    video_h, video_w = _video_hw(info, args.video_key)

    episodes = _list_episode_refs(root)
    names = _schema_names(root)
    camera = _resolve_camera_columns(names, args.video_key)
    required = [
        _COL_EPISODE_INDEX,
        _COL_FRAME_INDEX,
        _COL_GOAL_ACTION,
        _COL_GOAL_POSITION,
        _COL_EE_POSITION,
        camera.pose,
        camera.intrinsics,
    ]
    _require_columns(names, required, where=str(root))
    has_secondary = _COL_GOAL_SECONDARY in names
    has_timestamp = _COL_TIMESTAMP in names

    n_sample = min(
        args.max_episodes,
        int(math.ceil(args.sample_frac * len(episodes))),
    )
    rng = np.random.default_rng(args.seed)
    chosen_idx = rng.choice(len(episodes), size=n_sample, replace=False)
    sampled = [episodes[int(i)] for i in chosen_idx]
    sampled.sort(key=lambda r: (r.episode, r.data_chunk, r.data_file))
    print(
        f"scoring {len(sampled)} / {len(episodes)} recordings "
        f"(sample_frac={args.sample_frac}, max_episodes={args.max_episodes})",
        flush=True,
    )

    columns = list(required)
    if has_secondary:
        columns.append(_COL_GOAL_SECONDARY)
    if has_timestamp:
        columns.append(_COL_TIMESTAMP)
    flagged: list[dict[str, str]] = []
    tables: dict[EpisodeRef, dict[str, list[Any]]] = {}
    for ref in sampled:
        try:
            tables.update(_read_episode_tables(root, [ref], columns))
        except Exception as err:
            flagged.append({"recording": ref.label, "stage": "read", "error": str(err)})
            print(f"flagged {ref.label} [read]: {err}", flush=True)

    episode_scores: list[EpisodeScore] = []
    frames_by_ep: dict[EpisodeRef, EpisodeFrames] = {}
    subskills: list[SubskillScore] = []
    goal_positions: list[np.ndarray] = []
    for ref in sampled:
        if ref not in tables:
            continue
        try:
            score, frames = _score_episode(
                tables[ref],
                camera,
                world_to_camera_points,
                ref=ref,
                tail_frac=args.tail_frac,
                fail_px=args.fail_px,
                clean_px=args.clean_px,
                has_secondary=has_secondary,
                has_timestamp=has_timestamp,
            )
        except Exception as err:
            flagged.append({"recording": ref.label, "stage": "score", "error": str(err)})
            print(f"flagged {ref.label} [score]: {err}", flush=True)
            continue
        episode_scores.append(score)
        frames_by_ep[ref] = frames
        subskills.extend(score.subskills)
        goal_positions.append(frames.goal_uv)
    if goal_positions:
        try:
            _assert_goal_resolution(
                np.concatenate(goal_positions, axis=0), width=video_w, height=video_h
            )
        except Exception as err:
            flagged.append(
                {"recording": "*", "stage": "resolution", "error": str(err)}
            )
            print(f"flagged [resolution]: {err}", flush=True)

    chosen = _pick_videos(
        episode_scores,
        n_fail=args.n_fail,
        n_boundary=args.n_boundary,
        n_clean=args.n_clean,
        fail_px=args.fail_px,
        clean_px=args.clean_px,
    )
    video_notes: list[str] = []
    for bucket, want in (
        ("fail", args.n_fail),
        ("boundary", args.n_boundary),
        ("clean", args.n_clean),
    ):
        got = len(chosen[bucket])
        if got < want:
            video_notes.append(
                f"sample has only {got} {bucket} episode(s); requested {want}. "
                "Not pulling extra episodes from outside the sample."
            )

    _write_csv(out / "subskill_scores.csv", subskills)
    _write_flagged_csv(out / "flagged.csv", flagged)
    if subskills:
        medians = np.asarray([s.median_px for s in subskills], dtype=np.float64)
        _write_histogram(
            out / "distance_hist.png", medians, fail_px=args.fail_px, clean_px=args.clean_px
        )

    videos_written: dict[Bucket, list[EpisodeScore]] = {
        "fail": [],
        "boundary": [],
        "clean": [],
    }
    camera_name = camera_short_name(args.video_key, _COL_IMAGE_PREFIX)
    for bucket, rows in chosen.items():
        dest_dir = out / "videos" / bucket
        dest_dir.mkdir(parents=True, exist_ok=True)
        for ep_score in rows:
            dest = dest_dir / (
                f"episode_{ep_score.ref.episode}_chunk{ep_score.ref.data_chunk:03d}"
                f"_file{ep_score.ref.data_file:03d}_{camera_name}_audit.mp4"
            )
            try:
                video_meta = _video_entries_for_refs(
                    root, [ep_score.ref], args.video_key
                )
                _annotate_video(
                    frames_by_ep[ep_score.ref],
                    video_meta[ep_score.ref],
                    dest,
                    expected_hw=(video_h, video_w),
                )
            except Exception as err:
                video_notes.append(f"skipped viz {ep_score.ref.label}: {err}")
                print(f"skipped viz {ep_score.ref.label}: {err}", flush=True)
                continue
            videos_written[bucket].append(ep_score)

    success_written: list[str] = []
    success_rows = _pick_success(episode_scores, args.n_success)
    if len(success_rows) < args.n_success:
        video_notes.append(
            f"sample has only {len(success_rows)} scored episode(s) for "
            f"success videos; requested {args.n_success}."
        )
    if success_rows:
        dest_dir = out / "videos" / "success"
        dest_dir.mkdir(parents=True, exist_ok=True)
        for ep_score in success_rows:
            dest = dest_dir / (
                f"episode_{ep_score.ref.episode}_chunk{ep_score.ref.data_chunk:03d}"
                f"_file{ep_score.ref.data_file:03d}_{camera_name}"
                f"_{ep_score.bucket}_{ep_score.worst_median:.0f}px_audit.mp4"
            )
            try:
                video_meta = _video_entries_for_refs(
                    root, [ep_score.ref], args.video_key
                )
                _annotate_video(
                    frames_by_ep[ep_score.ref],
                    video_meta[ep_score.ref],
                    dest,
                    expected_hw=(video_h, video_w),
                )
            except Exception as err:
                video_notes.append(
                    f"skipped success viz {ep_score.ref.label}: {err}"
                )
                print(
                    f"skipped success viz {ep_score.ref.label}: {err}",
                    flush=True,
                )
                continue
            success_written.append(
                f"{ep_score.ref.label} {ep_score.bucket} "
                f"{ep_score.worst_median:.1f}px"
            )

    summary = _build_summary(
        root=root,
        out=out,
        args=args,
        camera=camera,
        n_dataset=len(episodes),
        sampled=sampled,
        episode_scores=episode_scores,
        subskills=subskills,
        chosen=videos_written,
        video_notes=video_notes,
        flagged=flagged,
        success_videos=success_written,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_summary_txt(out / "summary.txt", summary)
    print((out / "summary.txt").read_text(), end="")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
