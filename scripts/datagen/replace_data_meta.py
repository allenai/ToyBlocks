#!/usr/bin/env python3
"""Replace data/ and meta/ in a rerendered dataset with those from the original.

For each matching <problem>/{blocks_success,blocks_failure} folder, copies
data/ and meta/ from dataset_original into dataset_rerendered. Videos are
left untouched.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

BLOCK_FOLDERS = ("blocks_success", "blocks_failure")
COPY_FOLDERS = ("data", "meta")


def replace_folder(src: Path, dst: Path, dry_run: bool) -> None:
    if dry_run:
        print(f"  [dry-run] replace {dst} <- {src}")
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  replaced {dst}")


def process_block_folder(src_block: Path, dst_block: Path, dry_run: bool) -> int:
    if not src_block.is_dir():
        print(f"  skip (missing source): {src_block}")
        return 0
    if not dst_block.is_dir():
        print(f"  skip (missing dest): {dst_block}")
        return 0

    n = 0
    for name in COPY_FOLDERS:
        src = src_block / name
        dst = dst_block / name
        if not src.is_dir():
            print(f"  skip (missing source {name}): {src}")
            continue
        replace_folder(src, dst, dry_run)
        n += 1
    return n


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Copy data/ and meta/ from dataset_original into matching "
            "<problem>/{blocks_success,blocks_failure} folders of "
            "dataset_rerendered. Does not touch videos/."
        )
    )
    parser.add_argument(
        "dataset_original",
        type=Path,
        help="Path to the original dataset root (contains <problem>/ folders)",
    )
    parser.add_argument(
        "dataset_rerendered",
        type=Path,
        help="Path to the rerendered dataset root (data/meta will be replaced)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be replaced without copying",
    )
    args = parser.parse_args()

    original = args.dataset_original.resolve()
    rerendered = args.dataset_rerendered.resolve()

    if not original.is_dir():
        print(f"error: dataset_original is not a directory: {original}", file=sys.stderr)
        return 1
    if not rerendered.is_dir():
        print(f"error: dataset_rerendered is not a directory: {rerendered}", file=sys.stderr)
        return 1

    problems = sorted(p.name for p in original.iterdir() if p.is_dir())
    if not problems:
        print(f"error: no problem folders found in {original}", file=sys.stderr)
        return 1

    replaced = 0
    skipped = 0
    for problem in problems:
        src_problem = original / problem
        dst_problem = rerendered / problem
        if not dst_problem.is_dir():
            print(f"skip problem (not in rerendered): {problem}")
            skipped += 1
            continue

        print(f"problem: {problem}")
        for block in BLOCK_FOLDERS:
            replaced += process_block_folder(
                src_problem / block, dst_problem / block, args.dry_run
            )

    print(
        f"\nDone. Replaced {replaced} data/meta folders"
        f"{' (dry-run)' if args.dry_run else ''}"
        f"; skipped {skipped} problems missing from rerendered."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
