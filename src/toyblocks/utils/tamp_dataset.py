"""TAMP dataset I/O and episode discovery utilities.

Handles loading trajectories recorded by the old TAMP repository (LeRobot v3.0
format: Parquet + MP4) in two on-disk layouts:

* **Per-episode (legacy/unmerged)** — one ``blocks/`` directory per episode,
  with a single ``data/chunk-000/file-000.parquet`` containing exactly that
  episode's frames.  Produced by the original TAMP recorder and by
  :mod:`toyblocks.scripts.extract_lerobot_episode`.

* **Merged LeRobot v3** — a single dataset root with many episodes packed into
  multi-shard parquet files (``data/chunk-000/file-NNN.parquet``).  Each row
  has an ``episode_index`` column.  ``meta/info.json`` records
  ``chunks_size`` (episodes per shard) and ``total_episodes``.

Both layouts use the same column schema, so downstream consumers (the replay
loop, the action adapter) can be agnostic to the source.  Episode selection
goes through :class:`EpisodeSpec` and :func:`load_episode_data`.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


# ---------------------------------------------------------------------------
# Episode selection
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class EpisodeSpec:
    """A pointer to one replayable episode.

    Exactly one of ``blocks_dir`` or ``(merged_root, episode_index)`` must be
    set.  ``name`` and ``source`` are human-readable identifiers used purely
    for logging and for the ``episode_name`` / ``episode_path`` fields written
    to ``replay_results.json``.
    """

    name: str
    source: str
    blocks_dir: Optional[Path] = None
    merged_root: Optional[Path] = None
    episode_index: Optional[int] = None

    def __post_init__(self) -> None:
        from_blocks = self.blocks_dir is not None
        from_merged = self.merged_root is not None and self.episode_index is not None
        if from_blocks == from_merged:
            raise ValueError(
                "EpisodeSpec must specify exactly one of blocks_dir or "
                "(merged_root, episode_index)."
            )


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def _table_to_dict(table: pa.Table) -> dict:
    """Convert a pyarrow table into the numpy dict shape used by the replay loop."""
    data: dict[str, np.ndarray] = {}
    for col_name in table.column_names:
        values = table.column(col_name).to_pylist()
        if not values:
            data[col_name] = np.array(values)
            continue
        if isinstance(values[0], list):
            data[col_name] = np.array(values, dtype=np.float32)
        elif isinstance(values[0], (int, float, bool)):
            data[col_name] = np.array(values)
        else:
            data[col_name] = np.array(values, dtype=object)
    return data


def robot_base_pose_np(robot, env_idx: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(position, quaternion_wxyz)`` for the robot articulation root."""
    root = robot.data.root_pose_w[env_idx].detach().cpu().numpy().astype(np.float32)
    return root[:3], root[3:7]


def load_tamp_episode(dataset_path: str | Path) -> dict:
    """Load a single per-episode TAMP LeRobot dataset.

    Args:
        dataset_path: Path to the ``blocks/`` directory of one episode
                      (containing ``data/chunk-000/file-000.parquet`` and
                      ``meta/info.json``).

    Returns:
        Dict of numpy arrays keyed by parquet column name, plus ``"_info"``
        if ``meta/info.json`` is present.
    """
    dataset_path = Path(dataset_path)
    parquet_path = dataset_path / "data" / "chunk-000" / "file-000.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    table = pq.read_table(str(parquet_path))
    episode_data = _table_to_dict(table)

    info_path = dataset_path / "meta" / "info.json"
    if info_path.exists():
        with open(info_path) as f:
            episode_data["_info"] = json.load(f)

    return episode_data


def _load_merged_info(root: Path) -> dict:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(
            f"Expected merged LeRobot dataset metadata at {info_path}"
        )
    with open(info_path) as f:
        return json.load(f)


def _shard_path_for(root: Path, info: dict, episode_index: int) -> Path:
    """Resolve the parquet shard that contains ``episode_index``."""
    chunks_size = int(info["chunks_size"])
    total_episodes = int(info["total_episodes"])
    if not 0 <= episode_index < total_episodes:
        raise ValueError(
            f"episode_index {episode_index} out of range [0, {total_episodes})"
        )
    shard_idx = episode_index // chunks_size
    # All TAMP merged datasets seen so far live under chunk-000/; if/when
    # multi-chunk layouts appear, the loader can be extended here without
    # changing the public API.
    shard_path = root / "data" / "chunk-000" / f"file-{shard_idx:03d}.parquet"
    if not shard_path.is_file():
        raise FileNotFoundError(
            f"Shard for episode {episode_index} not found: {shard_path}"
        )
    return shard_path


@functools.lru_cache(maxsize=8)
def _read_merged_data_shard(shard: Path) -> pa.Table:
    return pq.read_table(str(shard))


def load_lerobot_merged_episode(root: str | Path, episode_index: int) -> dict:
    """Load a single episode from a multi-shard LeRobot v3 dataset.

    Reads ``meta/info.json`` to find the right shard via
    ``episode_index // chunks_size``, then filters rows by ``episode_index``.
    Returns the same dict shape as :func:`load_tamp_episode` so the rest of
    the replay pipeline doesn't need to care which layout it came from. Shard
    reads are cached, since callers resolve one episode at a time and would
    otherwise re-read the same (potentially large) shard from scratch for
    every episode.
    """
    root = Path(root)
    info = _load_merged_info(root)
    shard_path = _shard_path_for(root, info, episode_index)

    table = _read_merged_data_shard(shard_path)
    ep_col = table.column("episode_index").to_numpy(zero_copy_only=False)
    mask = ep_col == episode_index
    n_rows = int(mask.sum())
    if n_rows == 0:
        chunks_size = int(info["chunks_size"])
        shard_idx = episode_index // chunks_size
        raise RuntimeError(
            f"No rows with episode_index={episode_index} in {shard_path.name} "
            f"(chunks_size={chunks_size}, expected episodes "
            f"{shard_idx * chunks_size}..{(shard_idx + 1) * chunks_size - 1})."
        )

    ep_table = table.filter(pa.array(mask))
    episode_data = _table_to_dict(ep_table)
    episode_data["_info"] = info
    return episode_data


def load_episode_data(spec: EpisodeSpec) -> dict:
    """Dispatch to the per-episode or merged loader based on *spec*."""
    if spec.blocks_dir is not None:
        return load_tamp_episode(spec.blocks_dir)
    assert spec.merged_root is not None and spec.episode_index is not None
    return load_lerobot_merged_episode(spec.merged_root, spec.episode_index)


# ---------------------------------------------------------------------------
# High-level action filtering
# ---------------------------------------------------------------------------

_HL_ACTION_COL = "action.high_level_action"


def _high_level_action_cell_to_str(raw: object) -> str:
    """Normalize a parquet cell to a plain string (LeRobot uses shape-[1] lists)."""
    if raw is None:
        return ""
    if isinstance(raw, (list, tuple)) and len(raw) > 0:
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return str(raw).strip()


def _row_matches_filter(text: str, needle: str, *, exact: bool) -> bool:
    return text == needle if exact else needle.lower() in text.lower()


def _episode_matches_high_level_filter(
    blocks_dir: Path, needle: str, *, exact: bool
) -> bool:
    """True if any row's ``action.high_level_action`` matches *needle*."""
    needle = needle.strip()
    if not needle:
        return True

    parquet_path = blocks_dir / "data" / "chunk-000" / "file-000.parquet"
    if not parquet_path.is_file():
        return False

    try:
        table = pq.read_table(str(parquet_path), columns=[_HL_ACTION_COL])
    except Exception:
        return False

    if table.num_rows == 0:
        return False

    col = table.column(0)
    for i in range(table.num_rows):
        text = _high_level_action_cell_to_str(col[i].as_py())
        if _row_matches_filter(text, needle, exact=exact):
            return True
    return False


def _iter_merged_matching_indices(
    root: Path,
    *,
    needle: str,
    exact: bool,
    indices: Optional[Iterable[int]] = None,
) -> Iterable[int]:
    """Yield episode indices from a merged dataset that match the high-level filter.

    If *indices* is given, episodes are checked in that order.  Otherwise we
    walk shard-by-shard from ``episode_index=0`` upward; this is much cheaper
    than re-reading the same shard once per episode index.
    """
    needle = (needle or "").strip()
    info = _load_merged_info(root)
    chunks_size = int(info["chunks_size"])
    total_episodes = int(info["total_episodes"])

    def _episode_in_shard_matches(shard_path: Path, ep_idx: int) -> bool:
        if not needle:
            return True
        # Read both columns once per shard via _shard_filter_cache below.
        col_ep, col_hla = _shard_filter_cache(shard_path)
        match_indices = np.flatnonzero(col_ep == ep_idx)
        if match_indices.size == 0:
            return False
        for row_i in match_indices:
            text = _high_level_action_cell_to_str(col_hla[int(row_i)])
            if _row_matches_filter(text, needle, exact=exact):
                return True
        return False

    cache: dict[Path, tuple[np.ndarray, list]] = {}

    def _shard_filter_cache(shard_path: Path) -> tuple[np.ndarray, list]:
        cached = cache.get(shard_path)
        if cached is not None:
            return cached
        cols = [_HL_ACTION_COL] if needle else []
        table = pq.read_table(
            str(shard_path), columns=["episode_index", *cols]
        )
        ep_arr = table.column("episode_index").to_numpy(zero_copy_only=False)
        hla = (
            table.column(_HL_ACTION_COL).to_pylist() if needle else []
        )
        result = (ep_arr, hla)
        cache[shard_path] = result
        return result

    if indices is None:
        indices_iter: Iterable[int] = range(total_episodes)
    else:
        indices_iter = indices

    for ep_idx in indices_iter:
        if not 0 <= ep_idx < total_episodes:
            raise ValueError(
                f"episode_index {ep_idx} out of range [0, {total_episodes})"
            )
        shard_idx = ep_idx // chunks_size
        shard_path = root / "data" / "chunk-000" / f"file-{shard_idx:03d}.parquet"
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        if _episode_in_shard_matches(shard_path, ep_idx):
            yield ep_idx


# ---------------------------------------------------------------------------
# Episode spec collection
# ---------------------------------------------------------------------------

def collect_episode_specs(
    *,
    dataset_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    merged_root: Optional[str] = None,
    merged_episode_indices: Optional[Iterable[int]] = None,
    max_episodes: int = 5,
    filter_high_level_action: Optional[str] = None,
    filter_high_level_exact: bool = False,
) -> list[EpisodeSpec]:
    """Resolve replay episodes from CLI-style arguments.

    Exactly one of *dataset_path*, *dataset_dir*, or *merged_root* must be
    provided.

    * ``dataset_path`` — path to a single episode's ``blocks/`` directory.
    * ``dataset_dir`` — parent directory containing per-episode children, each
      with a ``blocks/`` subfolder (legacy unmerged TAMP layout).
    * ``merged_root`` — root of a multi-shard LeRobot v3 dataset.  If
      ``merged_episode_indices`` is given, only those episodes are loaded; the
      high-level filter, if any, is then enforced strictly (mismatches raise).
      Otherwise we scan from episode 0 upward, taking the first
      ``max_episodes`` episodes that match the filter.

    The high-level filter does substring matching against
    ``action.high_level_action`` (case-insensitive) unless
    ``filter_high_level_exact`` is set.
    """
    sources_set = sum(
        x is not None for x in (dataset_path, dataset_dir, merged_root)
    )
    if sources_set != 1:
        raise ValueError(
            "Provide exactly one of --replay_dataset_path, "
            "--replay_dataset_dir, or --replay_lerobot_dataset."
        )

    needle = (filter_high_level_action or "").strip()
    use_filter = bool(needle)
    specs: list[EpisodeSpec] = []

    if dataset_path:
        blocks_dir = Path(dataset_path)
        if use_filter and not _episode_matches_high_level_filter(
            blocks_dir, needle, exact=filter_high_level_exact
        ):
            raise ValueError(
                f"Episode does not match high-level action filter {needle!r} "
                f"(exact={filter_high_level_exact}). Path: {blocks_dir}"
            )
        specs.append(
            EpisodeSpec(
                name=blocks_dir.parent.name or blocks_dir.name,
                source=str(blocks_dir),
                blocks_dir=blocks_dir,
            )
        )

    elif dataset_dir:
        dir_path = Path(dataset_dir)
        for child in sorted(dir_path.iterdir()):
            blocks_dir = child / "blocks"
            if not (blocks_dir.is_dir() and (blocks_dir / "data").is_dir()):
                continue
            if use_filter and not _episode_matches_high_level_filter(
                blocks_dir, needle, exact=filter_high_level_exact
            ):
                continue
            specs.append(
                EpisodeSpec(
                    name=child.name,
                    source=str(blocks_dir),
                    blocks_dir=blocks_dir,
                )
            )
            if len(specs) >= max_episodes:
                break

    else:
        assert merged_root is not None
        root = Path(merged_root)
        explicit = (
            list(merged_episode_indices)
            if merged_episode_indices is not None
            else None
        )
        if explicit is not None:
            # User gave an explicit index list; validate filter strictly.
            matched = list(
                _iter_merged_matching_indices(
                    root,
                    needle=needle,
                    exact=filter_high_level_exact,
                    indices=explicit,
                )
            )
            if use_filter and len(matched) != len(explicit):
                missed = sorted(set(explicit) - set(matched))
                raise ValueError(
                    f"Episodes {missed} do not match high-level action "
                    f"filter {needle!r} (exact={filter_high_level_exact})."
                )
            chosen = matched if use_filter else explicit
        else:
            iterator = _iter_merged_matching_indices(
                root,
                needle=needle,
                exact=filter_high_level_exact,
            )
            chosen = []
            for ep_idx in iterator:
                chosen.append(ep_idx)
                if len(chosen) >= max_episodes:
                    break

        dataset_label = root.name or str(root)
        for ep_idx in chosen:
            specs.append(
                EpisodeSpec(
                    name=f"{dataset_label}/ep{ep_idx:05d}",
                    source=f"{root}#episode={ep_idx}",
                    merged_root=root,
                    episode_index=int(ep_idx),
                )
            )

    if not specs:
        if use_filter:
            label = dataset_dir or merged_root or dataset_path
            raise ValueError(
                f"No episodes matched high-level action filter {needle!r} "
                f"(exact={filter_high_level_exact}) under {label}."
            )
        raise ValueError(
            "No episodes found. Provide --replay_dataset_path, "
            "--replay_dataset_dir, or --replay_lerobot_dataset."
        )
    return specs


def collect_episode_paths(
    dataset_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    max_episodes: int = 5,
    filter_high_level_action: Optional[str] = None,
    filter_high_level_exact: bool = False,
) -> list[str]:
    """Back-compat shim returning string paths to per-episode ``blocks/`` dirs.

    Prefer :func:`collect_episode_specs` for new code; that path also supports
    merged LeRobot v3 datasets.
    """
    specs = collect_episode_specs(
        dataset_path=dataset_path,
        dataset_dir=dataset_dir,
        max_episodes=max_episodes,
        filter_high_level_action=filter_high_level_action,
        filter_high_level_exact=filter_high_level_exact,
    )
    return [str(s.blocks_dir) for s in specs if s.blocks_dir is not None]


# ---------------------------------------------------------------------------
# Action format adapter
# ---------------------------------------------------------------------------

# Recorded ``action.gripper_position`` is a binarized open/close flag (0 =
# open, 1 = close); real recordings sit near the endpoints but rarely hit
# them exactly, hence the tolerance rather than a `> 0` comparison.
GRIPPER_CLOSE_THRESHOLD = 0.01


def _scalar_field(value: np.ndarray | float) -> float:
    """Extract a Python float from a per-step recorded field.

    Recorded fields come back as either a plain scalar or a size-1
    ``np.ndarray`` (shape ``()``, ``(1,)``, or ``(1, 1)`` depending on the
    loader), so this normalizes both to ``float``.
    """
    if isinstance(value, np.ndarray):
        return float(value.reshape(-1)[0])
    return float(value)


def gripper_is_closed(
    episode_data: dict, step: int, *, threshold: float = GRIPPER_CLOSE_THRESHOLD
) -> bool:
    """Whether the recorded ``action.gripper_position`` at ``step`` is closing."""
    return _scalar_field(episode_data["action.gripper_position"][step]) > threshold


def recorded_joint_target(
    episode_data: dict, step: int, device: torch.device
) -> torch.Tensor:
    """The recorded 7-D ``action.joint_target`` at ``step`` as a flat tensor."""
    return torch.as_tensor(
        episode_data["action.joint_target"][step], device=device, dtype=torch.float32
    ).reshape(-1)[:7]


def tamp_action_to_env_action(
    episode_data: dict,
    step: int,
    device: torch.device,
    *,
    use_delta_actions: bool = False,
    current_joint_pos: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convert a single recorded TAMP action into a tensor for ``env.step()``.

    For the ``arm_joint_gripper_binary`` control mode the output is an 8-D
    tensor: 7 absolute joint positions + 1 gripper binary.

    Gripper sign convention follows Isaac Lab's
    :class:`~isaaclab.envs.mdp.actions.BinaryJointPositionAction`:

    * negative values  → close (joint target = ``close_command_expr``)
    * non-negative     → open  (joint target = ``open_command_expr``)

    TAMP's ``action.gripper_position`` is a binarized open/close flag (0 = open,
    1 = close). ``action.gripper_raw`` holds the underlying ``finger_joint`` target
    in radians (~0.0 open, ~0.7 closed in TAMP datagen). We threshold the binary
    flag at 0.01 and map closing frames to -1.0 and opening frames to +1.0 for
    Isaac Lab's :class:`~isaaclab.envs.mdp.actions.BinaryJointPositionAction`.

    Args:
        episode_data: Episode dict from :func:`load_tamp_episode` or
            :func:`load_lerobot_merged_episode`.
        step: Frame index within the episode.
        device: Torch device for the returned tensor.
        use_delta_actions: If True, interpret
            ``episode_data["action"][:7]`` as deltas relative to
            *current_joint_pos*.
        current_joint_pos: Required when *use_delta_actions* is
            True.  The current sim joint positions (7-D).

    Returns:
        Action tensor of shape ``(num_envs, action_dim)`` (currently always
        ``(1, 8)`` since replay runs with ``num_envs=1``).
    """
    if use_delta_actions:
        if current_joint_pos is None:
            raise ValueError(
                "current_joint_pos required when use_delta_actions=True"
            )
        delta = torch.tensor(
            episode_data["action"][step][:7], device=device, dtype=torch.float32
        )
        joint_target = current_joint_pos + delta
    else:
        joint_target = recorded_joint_target(episode_data, step, device)

    # Negative = close, positive = open (Isaac Lab BinaryJointPositionAction).
    gripper_binary = -1.0 if gripper_is_closed(episode_data, step) else 1.0

    action = torch.cat([
        joint_target,
        torch.tensor([gripper_binary], device=device, dtype=torch.float32),
    ])
    return action.unsqueeze(0)


# ---------------------------------------------------------------------------
# Recorded ground-truth video extraction
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class VideoClip:
    """A trim region inside one of the dataset's recorded MP4 files."""

    video_key: str  # e.g. "observation.images.front_camera"
    video_path: Path
    from_timestamp: float
    to_timestamp: Optional[float]  # ``None`` means "to end of file" (legacy layout)


def _video_keys_from_info(info: dict) -> list[str]:
    """Return the ``observation.images.*`` feature keys present in *info*."""
    features = info.get("features", {}) or {}
    keys = sorted(
        k for k in features.keys() if k.startswith("observation.images.")
    )
    return keys


@functools.lru_cache(maxsize=8)
def _read_episode_meta_shard(shard: Path) -> pa.Table:
    return pq.read_table(str(shard))


def _find_merged_episode_meta_row(root: Path, episode_index: int) -> dict:
    """Locate the per-episode metadata row for *episode_index*.

    Walks ``meta/episodes/chunk-*/file-*.parquet`` shards in order (typically
    just one file for datasets we've seen). Shard reads are cached, since
    callers resolve one episode at a time and would otherwise re-read the
    same (potentially large) shard from scratch for every episode.
    """
    meta_root = root / "meta" / "episodes"
    if not meta_root.is_dir():
        raise FileNotFoundError(
            f"Expected per-episode metadata at {meta_root}; cannot resolve "
            "video clip ranges without it."
        )
    for shard in sorted(meta_root.glob("chunk-*/file-*.parquet")):
        table = _read_episode_meta_shard(shard)
        ep_col = table.column("episode_index").to_numpy(zero_copy_only=False)
        match = np.flatnonzero(ep_col == episode_index)
        if match.size == 0:
            continue
        row_idx = int(match[0])
        return {col: table.column(col)[row_idx].as_py() for col in table.column_names}
    raise RuntimeError(
        f"episode_index={episode_index} not found in any "
        f"{meta_root}/chunk-*/file-*.parquet"
    )


def _video_path_from_template(
    root: Path, template: str, video_key: str, chunk_index: int, file_index: int
) -> Path:
    """Format LeRobot's ``video_path`` template into an absolute path."""
    rel = template.format(
        video_key=video_key, chunk_index=chunk_index, file_index=file_index
    )
    return root / rel


def _iter_merged_video_clips_from_episode_files(
    root: Path,
    template: str,
    video_keys: list[str],
    episode_index: int,
    *,
    chunk_index: int = 0,
) -> list[VideoClip]:
    """Fallback when ``meta/episodes`` lacks ``videos/{key}/...`` columns.

    Some merged datasets store one MP4 per episode at
    ``videos/{key}/chunk-000/file-{episode_index:03d}.mp4`` without recording
    per-episode trim ranges in metadata.
    """
    clips: list[VideoClip] = []
    for key in video_keys:
        path = _video_path_from_template(
            root, template, key, chunk_index, episode_index
        )
        if not path.is_file():
            continue
        clips.append(
            VideoClip(
                video_key=key,
                video_path=path,
                from_timestamp=0.0,
                to_timestamp=None,
            )
        )
    return clips


def iter_episode_video_clips(spec: EpisodeSpec) -> list[VideoClip]:
    """Discover the recorded ground-truth video clips for one episode.

    For merged datasets, looks up the per-camera ``(file_index, from_timestamp,
    to_timestamp)`` triplet in ``meta/episodes/...parquet`` and resolves the
    MP4 path via ``info["video_path"]``.  For legacy per-episode ``blocks/``
    directories, returns one clip per MP4 found under ``<blocks_dir>/videos/``
    spanning the entire file.

    Returns an empty list if the dataset has no video features.
    """
    if spec.blocks_dir is not None:
        videos_root = spec.blocks_dir / "videos"
        if not videos_root.is_dir():
            return []
        clips: list[VideoClip] = []
        for mp4 in sorted(videos_root.rglob("*.mp4")):
            try:
                video_key = mp4.relative_to(videos_root).parts[0]
            except (IndexError, ValueError):
                video_key = mp4.stem
            clips.append(
                VideoClip(
                    video_key=video_key,
                    video_path=mp4,
                    from_timestamp=0.0,
                    to_timestamp=None,
                )
            )
        return clips

    assert spec.merged_root is not None and spec.episode_index is not None
    root = spec.merged_root
    info = _load_merged_info(root)
    video_keys = _video_keys_from_info(info)
    if not video_keys:
        return []
    template = info.get("video_path")
    if not template:
        raise RuntimeError(
            f"meta/info.json under {root} is missing the 'video_path' template."
        )
    meta_row = _find_merged_episode_meta_row(root, spec.episode_index)

    clips = []
    for key in video_keys:
        chunk_col = f"videos/{key}/chunk_index"
        file_col = f"videos/{key}/file_index"
        from_col = f"videos/{key}/from_timestamp"
        to_col = f"videos/{key}/to_timestamp"
        if chunk_col not in meta_row or file_col not in meta_row:
            # Some datasets only store a subset of cameras' ranges; skip those.
            continue
        path = _video_path_from_template(
            root, template, key, int(meta_row[chunk_col]), int(meta_row[file_col])
        )
        clips.append(
            VideoClip(
                video_key=key,
                video_path=path,
                from_timestamp=float(meta_row.get(from_col, 0.0)),
                to_timestamp=(
                    float(meta_row[to_col]) if meta_row.get(to_col) is not None else None
                ),
            )
        )
    if clips:
        return clips

    chunk_index = int(meta_row.get("data/chunk_index", 0))
    return _iter_merged_video_clips_from_episode_files(
        root,
        template,
        video_keys,
        spec.episode_index,
        chunk_index=chunk_index,
    )


def extract_video_clip(clip: VideoClip, dst: Path) -> None:
    """Write *clip* to *dst* using ffmpeg stream-copy (no re-encode)."""
    if not clip.video_path.is_file():
        raise FileNotFoundError(f"Source video not found: {clip.video_path}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    if clip.to_timestamp is None and clip.from_timestamp == 0.0:
        # Whole-file clip: a plain copy is fastest and avoids any keyframe
        # rounding from ffmpeg's seek heuristics.
        shutil.copyfile(clip.video_path, dst)
        return

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    # ``-ss`` before ``-i`` is fast (input seek). With ``-c copy`` it lands on
    # the nearest keyframe at or before the requested timestamp; for the 25 Hz
    # TAMP recordings this is accurate enough for visual side-by-side review.
    if clip.from_timestamp > 0.0:
        cmd += ["-ss", f"{clip.from_timestamp:.6f}"]
    cmd += ["-i", str(clip.video_path)]
    if clip.to_timestamp is not None:
        duration = clip.to_timestamp - clip.from_timestamp
        if duration <= 0:
            raise ValueError(
                f"Non-positive clip duration ({duration:.3f}s) for "
                f"{clip.video_path}"
            )
        cmd += ["-t", f"{duration:.6f}"]
    cmd += ["-c", "copy", str(dst)]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {clip.video_path}: {proc.stderr.strip()}"
        )


def save_gt_videos_for_episode(
    spec: EpisodeSpec, output_dir: Path, *, episode_label: str
) -> list[Path]:
    """Extract every recorded camera clip for *spec* into *output_dir*.

    Files are named ``<episode_label>__<camera_key>.mp4`` (one file per
    camera).  Returns the list of written paths.  Returns an empty list if
    the dataset has no video features.
    """
    clips = iter_episode_video_clips(spec)
    written: list[Path] = []
    for clip in clips:
        # Strip the LeRobot-standard ``observation.images.`` prefix from the
        # camera label so filenames stay readable.
        cam = clip.video_key.removeprefix("observation.images.")
        dst = output_dir / f"{episode_label}__{cam}.mp4"
        extract_video_clip(clip, dst)
        written.append(dst)
    return written


def camera_name_to_video_key(camera: Optional[str]) -> str:
    """Map a short camera name to a LeRobot ``observation.images.*`` key."""
    if camera is None:
        return "observation.images.front_camera"
    cam = camera.strip()
    if cam.startswith("observation.images."):
        return cam
    return f"observation.images.{cam}"


def camera_name_to_video_key_candidates(camera: Optional[str]) -> list[str]:
    """Return LeRobot video feature keys to try for a short camera name."""
    if camera is None:
        short = "front_camera"
    else:
        short = camera.strip().removeprefix("observation.images.")
    candidates = [
        f"observation.images.{short}",
        f"observation.images.camera:{short}",
    ]
    if short.startswith("camera:"):
        candidates.insert(0, f"observation.images.{short}")
    # de-dupe while preserving order
    out: list[str] = []
    for key in candidates:
        if key not in out:
            out.append(key)
    return out


def find_episode_video_clip(
    spec: EpisodeSpec,
    camera: Optional[str],
) -> Optional[VideoClip]:
    """Return the recorded MP4 clip for *camera*, with sensible fallbacks."""
    clips = iter_episode_video_clips(spec)
    by_key = {clip.video_key: clip for clip in clips}
    for key in camera_name_to_video_key_candidates(camera):
        if key in by_key:
            return by_key[key]
    for fallback in (
        "observation.images.front_camera",
        "observation.images.camera:front_camera",
        "observation.images.gripper_camera",
        "observation.images.camera:gripper_camera",
        "observation.images.left_camera",
        "observation.images.camera:left_camera",
    ):
        if fallback in by_key:
            return by_key[fallback]
    return clips[0] if clips else None


def _episode_index_from_blocks(blocks_dir: Path) -> Optional[int]:
    """Read ``episode_index`` from a per-episode ``blocks/`` parquet shard."""
    parquet_path = blocks_dir / "data" / "chunk-000" / "file-000.parquet"
    if not parquet_path.is_file():
        return None
    table = pq.read_table(str(parquet_path), columns=["episode_index"])
    if "episode_index" not in table.column_names or table.num_rows == 0:
        return None
    return int(table.column("episode_index")[0].as_py())


def enrich_spec_for_gt_video(
    spec: EpisodeSpec,
    gt_merged_root: Optional[str | os.PathLike],
) -> EpisodeSpec:
    """Use *gt_merged_root* to locate videos when *spec* has parquet only."""
    if iter_episode_video_clips(spec):
        return spec
    if spec.blocks_dir is None or gt_merged_root is None:
        return spec
    episode_index = _episode_index_from_blocks(spec.blocks_dir)
    if episode_index is None:
        return spec
    root = Path(gt_merged_root)
    return EpisodeSpec(
        name=spec.name,
        source=str(root),
        merged_root=root,
        episode_index=episode_index,
    )


def compose_side_by_side_videos(
    left_path: Path | os.PathLike,
    right_path: Path | os.PathLike,
    dst_path: Path | os.PathLike,
    *,
    left_label: str = "Dataset",
    right_label: str = "Replay",
) -> Path:
    """Write *left_path* | *right_path* (dataset | replay) via ffmpeg hstack."""
    left_path = Path(left_path)
    right_path = Path(right_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    label_style = (
        "fontsize=28:fontcolor=white:borderw=2:bordercolor=black@0.8:"
        "x=10:y=10"
    )
    filter_complex = (
        "[0:v]setpts=PTS-STARTPTS[dataset];"
        "[1:v]setpts=PTS-STARTPTS[replay];"
        "[replay][dataset]scale2ref[replay_scaled][dataset_scaled];"
        f"[dataset_scaled]drawtext=text={left_label!r}:{label_style}[dataset_labeled];"
        f"[replay_scaled]drawtext=text={right_label!r}:{label_style}[replay_labeled];"
        "[dataset_labeled][replay_labeled]hstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(left_path),
        "-i", str(right_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(dst_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg side-by-side compose failed: {proc.stderr.strip()}"
        )
    return dst_path


def save_replay_side_by_side_video(
    spec: EpisodeSpec,
    replay_mp4: Path | os.PathLike,
    output_dir: Path | os.PathLike,
    *,
    camera: Optional[str] = None,
    episode_label: Optional[str] = None,
) -> Optional[Path]:
    """Compose dataset (left) and replay (right) MP4s for one episode.

    Writes ``comparison_videos/data_vs_replay_<camera>_<label>.mp4`` under
    *output_dir*.  Returns ``None`` if the dataset has no matching video.
    """
    clip = find_episode_video_clip(spec, camera)
    if clip is None:
        return None

    replay_mp4 = Path(replay_mp4)
    output_dir = Path(output_dir)
    label = episode_label or spec.name
    cam_short = (
        clip.video_key.removeprefix("observation.images.")
        .removeprefix("camera:")
        .replace(":", "_")
    )
    comp_dir = output_dir / "comparison_videos"
    comp_dir.mkdir(parents=True, exist_ok=True)

    gt_path = comp_dir / f"_gt_{label}__{cam_short}.mp4"
    extract_video_clip(clip, gt_path)
    dst = comp_dir / f"data_vs_replay_{cam_short}_{label}.mp4"
    compose_side_by_side_videos(gt_path, replay_mp4, dst)
    gt_path.unlink(missing_ok=True)
    return dst
