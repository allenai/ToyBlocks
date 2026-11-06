"""Depth tensor summaries for eval diagnostics (stdout on control step 0)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from toyblocks.serving.obs_transforms.depth_ptmap_bounds import (
    DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS,
    z_bounds_for_depth_key,
)

# Common depth values to count exactly in diagnostics (meters).
_DEFAULT_DEPTH_PROBE_M = (
    0.4,
    1.0,
    1.023,  # rg8_55_224 10-bit pack ceiling (mm)
)


def summarize_depth_meters(
    depth_m: np.ndarray,
    *,
    atol: float = 1e-5,
    clamp_probe_m: Sequence[float] = _DEFAULT_DEPTH_PROBE_M,
) -> dict[str, Any]:
    """Summarize a depth map in meters (invalid / holes are ``<= 0``)."""
    flat = np.asarray(depth_m, dtype=np.float32).reshape(-1)
    finite = np.isfinite(flat)
    flat = flat[finite]
    count = int(flat.size)
    if count == 0:
        return {
            "count": 0,
            "count_zero": 0,
            "fraction_zero": 1.0,
            "min_positive_m": None,
            "max_positive_m": None,
            "count_at_max_positive": 0,
            "fraction_at_max_positive": 0.0,
            "fraction_at_max_of_all": 0.0,
            "probe_exact_counts": {f"{v:g}": 0 for v in clamp_probe_m},
        }

    zero = flat <= 0.0
    count_zero = int(zero.sum())
    positive = flat[~zero]
    n_pos = int(positive.size)
    out: dict[str, Any] = {
        "count": count,
        "count_zero": count_zero,
        "fraction_zero": float(count_zero / count),
        "min_positive_m": None,
        "max_positive_m": None,
        "count_at_max_positive": 0,
        "fraction_at_max_positive": 0.0,
        "fraction_at_max_of_all": 0.0,
        "probe_exact_counts": {},
    }
    if n_pos == 0:
        for v in clamp_probe_m:
            out["probe_exact_counts"][f"{v:g}"] = 0
        return out

    d_min = float(positive.min())
    d_max = float(positive.max())
    at_max = np.isclose(positive, d_max, rtol=0.0, atol=atol)
    count_at_max = int(at_max.sum())
    out["min_positive_m"] = d_min
    out["max_positive_m"] = d_max
    out["count_at_max_positive"] = count_at_max
    out["fraction_at_max_positive"] = float(count_at_max / n_pos)
    out["fraction_at_max_of_all"] = float(count_at_max / count)
    for v in clamp_probe_m:
        out["probe_exact_counts"][f"{v:g}"] = int(
            np.sum(np.isclose(flat, float(v), rtol=0.0, atol=atol))
        )
    return out


def _depth_hw_from_wire(arr: np.ndarray, env_idx: int) -> np.ndarray:
    """``(B, 1, 1, H, W)`` or ``(B, H, W)`` model depth → ``(H, W)``."""
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 5:
        return a[env_idx, 0, 0]
    if a.ndim == 3:
        return a[env_idx]
    if a.ndim == 2:
        return a
    raise ValueError(f"unexpected depth wire shape {a.shape}")


def summarize_wire_depth_batch(
    wire: dict[str, Any],
    depth_keys: Sequence[str],
) -> dict[str, Any]:
    """Per depth key: pooled batch stats + per-env slices."""
    report: dict[str, Any] = {}
    for key in depth_keys:
        if key not in wire:
            continue
        arr = np.asarray(wire[key], dtype=np.float32)
        if arr.ndim == 5:
            pooled = summarize_depth_meters(arr[:, 0, 0])
            per_env = [
                summarize_depth_meters(_depth_hw_from_wire(arr, i)) for i in range(arr.shape[0])
            ]
        elif arr.ndim == 3:
            pooled = summarize_depth_meters(arr)
            per_env = [summarize_depth_meters(arr[i]) for i in range(arr.shape[0])]
        else:
            pooled = summarize_depth_meters(arr)
            per_env = [pooled]
        report[key] = {"pooled": pooled, "per_env": per_env}
    return report


def _ptmap_z_ranges_for_keys(
    depth_keys: Sequence[str],
    bounds: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for key in depth_keys:
        z_min, z_max = z_bounds_for_depth_key(key, bounds)
        out[key] = [z_min, z_max]
    return out


def build_depth_stats_payload(
    wire: dict[str, Any],
    *,
    depth_keys: Sequence[str],
    per_camera_ptmap_bounds: Mapping[str, Mapping[str, list[float]]] | None,
) -> dict[str, Any]:
    """Summarize model-input depth tensors before they are sent to the policy server."""
    bounds = (
        dict(per_camera_ptmap_bounds)
        if per_camera_ptmap_bounds is not None
        else dict(DEFAULT_FRANKA_PER_CAMERA_PTMAP_BOUNDS)
    )
    return {
        "per_camera_ptmap_bounds": bounds,
        "ptmap_z_ranges_m": _ptmap_z_ranges_for_keys(depth_keys, bounds),
        "depth_keys": list(depth_keys),
        "cameras": summarize_wire_depth_batch(wire, depth_keys),
    }


def log_depth_stats_summary(payload: dict[str, Any]) -> None:
    """Log depth stats to stdout."""
    z_ranges = payload.get("ptmap_z_ranges_m")
    print(
        f"[INFO]: step-0 depth stats (model input @224) "
        f"ptmap_z_ranges_m={z_ranges!r}",
        flush=True,
    )
    for key, block in payload.get("cameras", {}).items():
        p = block["pooled"]
        probes = p.get("probe_exact_counts") or {}
        probe_bits = " ".join(f"{k}m={v}" for k, v in sorted(probes.items()) if v)
        probe_suffix = f" probes {{{probe_bits}}}" if probe_bits else ""
        print(
            f"[INFO]: step-0 depth {key} (pooled batch): "
            f"zero={p['fraction_zero']:.1%} ({p['count_zero']}/{p['count']}) "
            f"pos_range=[{p['min_positive_m']}, {p['max_positive_m']}] "
            f"at_max={p['count_at_max_positive']} "
            f"({p['fraction_at_max_positive']:.1%} of positive, "
            f"{p['fraction_at_max_of_all']:.1%} of all pixels)"
            f"{probe_suffix}",
            flush=True,
        )
        for env_i, env_stats in enumerate(block.get("per_env") or []):
            ep = env_stats
            if ep["count"] == 0:
                continue
            probes_e = ep.get("probe_exact_counts") or {}
            probe_bits_e = " ".join(f"{k}m={v}" for k, v in sorted(probes_e.items()) if v)
            probe_suffix_e = f" probes {{{probe_bits_e}}}" if probe_bits_e else ""
            print(
                f"[INFO]: step-0 depth {key} env={env_i}: "
                f"zero={ep['fraction_zero']:.1%} "
                f"pos_range=[{ep['min_positive_m']}, {ep['max_positive_m']}] "
                f"at_max={ep['count_at_max_positive']}"
                f"{probe_suffix_e}",
                flush=True,
            )
