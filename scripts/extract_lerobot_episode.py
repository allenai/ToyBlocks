"""Extract a single episode from a multi-shard LeRobot dataset.

Reads a consolidated LeRobot v3 dataset (e.g.
``data/datasets/merged/train/apr15_pnp_10k``), filters one
episode by ``episode_index``, and writes it back out as a per-episode
``blocks/``-style directory in the format expected by
:func:`toyblocks.utils.tamp_dataset.load_tamp_episode`:

    <dst>/data/chunk-000/file-000.parquet
    <dst>/meta/info.json

Used to feed a known-successful TAMP-recorded trajectory into
``scripts/evaluate.py --replay_dataset_path ...`` for sanity-checking the eval
pipeline (does the success predicate fire when actions are guaranteed to solve
the task?).

Pure pyarrow + JSON; no Isaac Lab imports.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tyro


@dataclasses.dataclass
class Args:
    src: str
    """Path to a multi-shard LeRobot dataset root (containing ``data/chunk-000/file-NNN.parquet`` and ``meta/info.json``)."""
    dst: str
    """Output ``blocks/`` directory. Will be created (or overwritten if --overwrite)."""
    episode_index: int = 0
    """Which episode to extract."""
    require_pddl_goal: str | None = None
    """If set, abort unless the chosen episode's ``pddl_goal`` column contains this substring."""
    require_successful: bool = True
    """If True, abort unless ``is_episode_successful`` is True for the chosen episode."""
    overwrite: bool = False
    """If True, remove --dst before writing."""


def _cell_to_str(raw: object) -> str:
    if isinstance(raw, (list, tuple)) and raw:
        raw = raw[0]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    return "" if raw is None else str(raw)


def main(args: Args) -> None:
    src = Path(args.src)
    dst = Path(args.dst)

    info_path = src / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    chunks_size = int(info["chunks_size"])
    total_episodes = int(info["total_episodes"])
    if not (0 <= args.episode_index < total_episodes):
        raise ValueError(
            f"--episode-index {args.episode_index} out of range [0, {total_episodes})"
        )

    shard_idx = args.episode_index // chunks_size
    shard_path = src / "data" / "chunk-000" / f"file-{shard_idx:03d}.parquet"
    if not shard_path.exists():
        raise FileNotFoundError(f"Shard not found: {shard_path}")

    print(f"[extract] reading {shard_path}")
    table = pq.read_table(str(shard_path))

    ep_col = table.column("episode_index").to_numpy(zero_copy_only=False)
    mask = ep_col == args.episode_index
    n_rows = int(mask.sum())
    if n_rows == 0:
        raise RuntimeError(
            f"No rows with episode_index={args.episode_index} in {shard_path.name} "
            f"(chunks_size={chunks_size}, expected episodes "
            f"{shard_idx * chunks_size}..{(shard_idx + 1) * chunks_size - 1})."
        )

    ep_table = table.filter(pa.array(mask))

    pddl_goal = _cell_to_str(ep_table.column("pddl_goal")[0].as_py())
    success_col = ep_table.column("is_episode_successful").to_pylist()
    success_flat = [_cell_to_str(s) if isinstance(s, (list, tuple)) else s for s in success_col]
    is_successful = bool(np.any(np.asarray(success_flat, dtype=bool)))

    print(f"[extract] episode_index={args.episode_index}: {n_rows} frames")
    print(f"[extract]   pddl_goal = {pddl_goal!r}")
    print(f"[extract]   is_episode_successful = {is_successful}")

    if args.require_pddl_goal and args.require_pddl_goal not in pddl_goal:
        raise RuntimeError(
            f"--require-pddl-goal {args.require_pddl_goal!r} not in {pddl_goal!r}"
        )
    if args.require_successful and not is_successful:
        raise RuntimeError(
            f"--require-successful set but episode {args.episode_index} is not flagged successful"
        )

    if dst.exists():
        if not args.overwrite:
            raise FileExistsError(f"{dst} already exists; pass --overwrite to replace")
        shutil.rmtree(dst)
    (dst / "data" / "chunk-000").mkdir(parents=True)
    (dst / "meta").mkdir()

    out_parquet = dst / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(ep_table, str(out_parquet))
    print(f"[extract] wrote {out_parquet} ({n_rows} rows)")

    out_info = copy.deepcopy(info)
    out_info["total_episodes"] = 1
    out_info["total_frames"] = n_rows
    out_info["splits"] = {"train": "0:1"}
    with open(dst / "meta" / "info.json", "w") as f:
        json.dump(out_info, f, indent=2)
    print(f"[extract] wrote {dst / 'meta' / 'info.json'}")


if __name__ == "__main__":
    main(tyro.cli(Args))
