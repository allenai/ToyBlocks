#!/usr/bin/env python3
"""Compare LeRobot recorded camera K vs blocksuite sim2real + fr3_softer presets.

Reads ``observation.camera_intrinsics.*`` from the first frame of up to N episodes
per dataset and compares to analytic preset intrinsics at each camera's render
resolution. Front-camera recorded K uses a centered principal point (see
``blocksuite.tamp.blocks.camera_view``), not naive scaling of ``K_front``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from blocksuite.utils.calibrated_camera_intrinsics import preset_k_for_camera
from blocksuite.utils.dataset_provenance import read_dataset_provenance
from blocksuite.utils.episode_dataset import (
    _COL_EPISODE_INDEX,
    _COL_IS_FIRST,
    _resolve_parquet_files,
)

_INTR_PREFIX = "observation.camera_intrinsics.camera:"


def discover_episode_ids(root: Path) -> list[int]:
    """Sorted episode indices present in dataset parquet (may be non-contiguous)."""
    files = _resolve_parquet_files(root)
    seen: set[int] = set()
    for f in files:
        names = pq.read_schema(str(f)).names
        if _COL_EPISODE_INDEX not in names:
            continue
        table = pq.read_table(str(f), columns=[_COL_EPISODE_INDEX])
        for ep in table.column(_COL_EPISODE_INDEX).to_pylist():
            seen.add(int(ep))
    return sorted(seen)


def _first_frame_k_per_episode(
    root: Path,
    episode_ids: list[int],
    max_eps: int,
) -> dict[int, dict[str, np.ndarray]]:
    files = _resolve_parquet_files(root)
    intr_cols: list[str] | None = None
    for f in files:
        names = pq.read_schema(str(f)).names
        ic = [c for c in names if c.startswith(_INTR_PREFIX)]
        if ic:
            intr_cols = ic
            break
    if not intr_cols:
        raise FileNotFoundError(f"{root}: no observation.camera_intrinsics.* columns")

    targets = set(episode_ids) if episode_ids else None
    found: dict[int, dict[str, np.ndarray]] = {}

    for f in files:
        if targets is not None and targets.issubset(found.keys()):
            break
        names = pq.read_schema(str(f)).names
        read_cols = [c for c in [_COL_EPISODE_INDEX, _COL_IS_FIRST] + intr_cols if c in names]
        table = pq.read_table(str(f), columns=read_cols)
        ep_arr = [int(x) for x in table.column(_COL_EPISODE_INDEX).to_pylist()]
        is_first = (
            table.column(_COL_IS_FIRST).to_pylist()
            if _COL_IS_FIRST in read_cols
            else [True] * len(ep_arr)
        )
        for i, ep in enumerate(ep_arr):
            if targets is not None and ep not in targets:
                continue
            if ep in found:
                continue
            if not is_first[i]:
                continue
            found[ep] = {}
            for col in intr_cols:
                cam = col.split(":")[-1]
                k9 = list(table.column(col)[i].as_py())
                found[ep][cam] = np.asarray(k9, dtype=np.float64).reshape(3, 3)
        if targets is None and len(found) >= max_eps:
            break

    if targets is not None:
        missing = sorted(targets - found.keys())
        if missing:
            raise ValueError(f"{root}: missing episodes {missing}")
    return found


def _summarize_dataset(root: Path, max_eps: int, episode_ids: list[int]) -> None:
    prov = read_dataset_provenance(root)
    print(f"\n{'=' * 72}")
    print(f"Dataset: {root}")
    print(
        f"  provenance: robot_id={prov.get('robot_id')!r} "
        f"scene_preset_id={prov.get('scene_preset_id')!r}"
    )

    by_ep = _first_frame_k_per_episode(root, episode_ids, max_eps)
    eps = sorted(by_ep.keys())
    if len(eps) <= 20:
        print(f"  episodes sampled: {eps}")
    else:
        print(f"  episodes sampled: {eps[0]}..{eps[-1]} (n={len(eps)})")

    cameras = sorted({c for ep in by_ep for c in by_ep[ep]})
    for cam in cameras:
        mats = np.stack([by_ep[ep][cam] for ep in eps])
        mean_k = mats.mean(axis=0)
        same = np.allclose(mats, mats[0], rtol=0, atol=1e-4)
        preset = preset_k_for_camera(cam)
        diffs = [np.max(np.abs(mats[i] - preset)) for i in range(len(eps))]
        mean_diff = float(np.max(np.abs(mean_k - preset)))
        print(f"\n  {cam}:")
        print(f"    constant across eps: {same}")
        k0 = mats[0]
        print(
            f"    recorded (ep{eps[0]}) fx,fy,cx,cy = {k0[0,0]:.6f}, {k0[1,1]:.6f}, "
            f"{k0[0,2]:.1f}, {k0[1,2]:.1f}"
        )
        fx_vals = mats[:, 0, 0]
        print(
            f"    mean over eps   fx,fy,cx,cy = {mean_k[0,0]:.6f}, {mean_k[1,1]:.6f}, "
            f"{mean_k[0,2]:.1f}, {mean_k[1,2]:.1f}  (n={len(eps)})"
        )
        if not same:
            print(
                f"    fx over eps     min={fx_vals.min():.6f} max={fx_vals.max():.6f} "
                f"std={fx_vals.std():.6f}"
            )
        print(
            f"    preset          fx,fy,cx,cy = {preset[0,0]:.6f}, {preset[1,1]:.6f}, "
            f"{preset[0,2]:.1f}, {preset[1,2]:.1f}"
        )
        print(f"    max |K - preset| (per ep): min={min(diffs):.2e} max={max(diffs):.2e}")
        print(f"    max |mean(K) - preset|: {mean_diff:.2e}")
        ok = np.allclose(k0, preset, rtol=1e-5, atol=1e-2)
        ok_mean = np.allclose(mean_k, preset, rtol=1e-5, atol=1e-2)
        print(f"    matches preset (ep{eps[0]}): {ok}; matches preset (mean): {ok_mean}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        type=Path,
        help="LeRobot dataset roots",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=100,
        help="When --episode-indices is unset, sample up to this many existing "
        "episodes (lowest indices first; default: 100)",
    )
    parser.add_argument(
        "--episode-indices",
        type=str,
        default="",
        help="Comma/range spec like 0-9 (overrides discovery from parquet)",
    )
    args = parser.parse_args()

    explicit_episode_ids: list[int] | None = None
    if args.episode_indices.strip():
        explicit_episode_ids = []
        for part in args.episode_indices.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                explicit_episode_ids.extend(range(int(a), int(b) + 1))
            else:
                explicit_episode_ids.append(int(part))

    for path in args.dataset_paths:
        root = path.resolve()
        if explicit_episode_ids is not None:
            episode_ids = explicit_episode_ids
        else:
            discovered = discover_episode_ids(root)
            if not discovered:
                raise SystemExit(f"{root}: no episodes found in parquet")
            episode_ids = discovered[: args.max_episodes]
        _summarize_dataset(root, args.max_episodes, episode_ids)


if __name__ == "__main__":
    main()
