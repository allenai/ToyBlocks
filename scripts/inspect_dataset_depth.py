"""Dump declared vs actual depth-video encoding for one LeRobot episode.

Read-only diagnostic for the gray12le_mm / swscale decode path. For each
``observation.depth.*`` feature it prints the contract in ``meta/info.json``
alongside the PyAV stream/frame format and pre-scale plane-0 percentiles from
three ``to_ndarray`` paths (native, gray16le, gray). Run against a real
dataset root to decide whether the millimetres survive recording or are lost
in the gray16le reformat.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import tyro

from toyblocks.utils.episode_dataset import (
    _COL_DEPTH_PREFIX,
    _load_episode_video_metadata,
    _load_lerobot_info,
)


@dataclasses.dataclass
class Args:
    dataset_root: str
    """LeRobot dataset root (directory containing ``meta/info.json``)."""
    episode_index: int = 0
    """Episode whose depth clips to open."""


_DECLARED_INFO_KEYS = (
    "video.channels",
    "video.pix_fmt",
    "video.depth_encoding",
    "video.depth_scale",
    "video.depth_max_mm",
)


def _depth_features(info: dict) -> dict[str, dict]:
    features = info.get("features", {}) or {}
    return {
        key: spec
        for key, spec in features.items()
        if key.startswith(_COL_DEPTH_PREFIX)
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "<missing>"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _plane0(arr: np.ndarray) -> np.ndarray:
    """Leading plane as a flat float64 vector for percentiles."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    elif a.ndim != 2:
        raise ValueError(f"expected HxW or HxWxC ndarray, got shape={a.shape}")
    return a.astype(np.float64).ravel()


def _percentile_row(samples: np.ndarray, scale: float | None) -> str:
    zeros = int(np.count_nonzero(samples == 0))
    p1, p50, p99 = np.percentile(samples, [1, 50, 99])
    amin = float(samples.min())
    amax = float(samples.max())
    line = (
        f"min={amin:.4g}  max={amax:.4g}  "
        f"p1={p1:.4g}  p50={p50:.4g}  p99={p99:.4g}  "
        f"zeros={zeros}/{samples.size}"
    )
    if scale is None:
        return line + "  (no video.depth_scale; metres N/A)"
    if scale == 0:
        raise ValueError("video.depth_scale is 0; cannot convert samples to metres")
    return (
        f"{line}  "
        f"→ m: min={amin / scale:.4g}  max={amax / scale:.4g}  "
        f"p1={p1 / scale:.4g}  p50={p50 / scale:.4g}  p99={p99 / scale:.4g}"
    )


def _seek_first_frame(container: Any, stream: Any, from_timestamp: float | None) -> Any:
    frame = None
    if from_timestamp:
        target_pts = int(from_timestamp / stream.time_base)
        container.seek(target_pts, stream=stream)
        for f in container.decode(video=0):
            if f.pts is not None and f.pts >= target_pts:
                frame = f
                break
        if frame is None:
            raise ValueError(
                f"no frame at/after from_timestamp={from_timestamp:.4f}s"
            )
        return frame
    for f in container.decode(video=0):
        frame = f
        break
    if frame is None:
        raise ValueError("video has no decodable frames")
    return frame


def _inspect_clip(
    key: str,
    spec: dict,
    entry: dict[str, Any] | None,
) -> None:
    print(f"\n=== {key} ===")
    shape = (spec or {}).get("shape")
    feat_info = (spec or {}).get("info") or {}
    print(f"  declared shape:            {_fmt(shape)}")
    for info_key in _DECLARED_INFO_KEYS:
        print(f"  declared {info_key:<22} {_fmt(feat_info.get(info_key))}")

    scale_raw = feat_info.get("video.depth_scale")
    scale = float(scale_raw) if scale_raw is not None else None

    if entry is None:
        raise FileNotFoundError(
            f"no resolved depth clip for {key!r} at the requested episode "
            "(meta/episodes missing the video columns, or the mp4 path does not exist)"
        )

    path = entry["path"]
    from_timestamp = entry.get("from_timestamp")
    print(f"  resolved path:             {path}")
    ts = f"{from_timestamp:.6f}" if from_timestamp is not None else "<none>"
    print(f"  from_timestamp:            {ts}")

    import av

    if not Path(path).is_file():
        raise FileNotFoundError(f"depth video not found: {path}")

    container = av.open(path)
    try:
        stream = container.streams.video[0]
        frame = _seek_first_frame(container, stream, from_timestamp)
        bits = frame.format.components[0].bits
        print(f"  stream.codec_context.pix_fmt: {stream.codec_context.pix_fmt}")
        print(f"  decoded frame.format.name:  {frame.format.name}")
        print(f"  decoded plane-0 bits:       {bits}")
        print(f"  decoded width x height:     {frame.width} x {frame.height}")

        reads = (
            ("to_ndarray()", frame.to_ndarray()),
            ("to_ndarray('gray16le')", frame.to_ndarray(format="gray16le")),
            ("to_ndarray('gray')", frame.to_ndarray(format="gray")),
        )
        print("  pre-scale plane-0 stats:")
        for label, arr in reads:
            samples = _plane0(arr)
            print(f"    {label}:")
            print(f"      dtype={arr.dtype} shape={arr.shape}")
            print(f"      {_percentile_row(samples, scale)}")
    finally:
        container.close()


def main(args: Args) -> None:
    root = Path(args.dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root is not a directory: {root}")

    info = _load_lerobot_info(root)
    if not info:
        raise FileNotFoundError(f"missing or empty {root / 'meta' / 'info.json'}")

    depth_specs = _depth_features(info)
    if not depth_specs:
        raise ValueError(
            f"{root}/meta/info.json has no features under {_COL_DEPTH_PREFIX!r}"
        )

    print(f"dataset_root:   {root.resolve()}")
    print(f"episode_index:  {args.episode_index}")
    print(f"depth features: {len(depth_specs)}")
    print(f"info.json keys under features: {sorted(depth_specs)}")

    _, depth_by_ep = _load_episode_video_metadata(
        root, episode_indices={args.episode_index}
    )
    episode_depth = depth_by_ep.get(args.episode_index, {})
    if not episode_depth:
        raise FileNotFoundError(
            f"episode_index={args.episode_index} resolved zero depth clips under {root}"
        )

    for key, spec in sorted(depth_specs.items()):
        _inspect_clip(key, spec, episode_depth.get(key))


if __name__ == "__main__":
    main(tyro.cli(Args))
