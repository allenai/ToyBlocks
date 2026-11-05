"""Merge the per-GPU seed runs of one eval into a single run directory.

Writes ``--out-dir/eval_episodes.json`` (every seed's episodes) and
``--out-dir/lerobot_dataset/<name>_success`` (the seeds' success datasets, merged in
``--seed-dirs`` order and then removed from the seeds; failure datasets stay in the seeds).

Usage:
    python scripts/merge_eval_runs.py --seed-dirs <seed_1_run> <seed_2_run> ... --out-dir <run_dir>
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
from pathlib import Path

import tyro

from toyblocks.vecgen.recording.merge import merge_datasets


@dataclasses.dataclass
class Args:
    seed_dirs: list[Path]
    """Seed run directories; those without ``eval_episodes.json`` are skipped."""
    out_dir: Path
    """Merged run directory."""


def _total_episodes(dataset_root: Path) -> int:
    return int(json.loads((dataset_root / "meta" / "info.json").read_text())["total_episodes"])


def main(args: Args) -> None:
    seed_dirs = [d for d in args.seed_dirs if (d / "eval_episodes.json").is_file()]
    sources = [
        p.parent.parent
        for d in seed_dirs
        for p in sorted((d / "lerobot_dataset").glob("*_success/meta/info.json"))
    ]
    # Episode offset of each source in the merged dataset (merge_datasets numbers them in order).
    remap: dict[str, tuple[str, int]] = {}
    if sources:
        # <task>__eval__<seed>__<time>_success without the seed.
        dest = args.out_dir / "lerobot_dataset" / re.sub(r"__eval__[^_]+__", "__eval__", sources[0].name)
        merge_datasets(dataset_paths=sources, output_dir=dest, keep_cameras=["all"])
        offset = 0
        for src in sources:
            remap[str(src)] = (str(dest), offset)
            offset += _total_episodes(src)
        # merge_datasets drops a source with an unreadable parquet, which would shift the offsets.
        if _total_episodes(dest) != offset:
            raise RuntimeError(f"{dest} holds {_total_episodes(dest)} episodes, expected {offset}")

    episodes: list[dict] = []
    for d in seed_dirs:
        for ep in json.loads((d / "eval_episodes.json").read_text()):
            if ep.get("output_lerobot_dataset_dir") in remap:
                root, offset = remap[ep["output_lerobot_dataset_dir"]]
                ep["output_lerobot_dataset_dir"] = root
                ep["output_lerobot_episode_index"] += offset
            episodes.append(ep)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "eval_episodes.json").write_text(json.dumps(episodes, indent=2))

    # The merged videos are hardlinks, so the seed copies can go.
    for src in remap:
        shutil.rmtree(src)


if __name__ == "__main__":
    main(tyro.cli(Args))
