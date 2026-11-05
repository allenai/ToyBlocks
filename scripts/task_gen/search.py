#!/usr/bin/env python3
"""Search-scene PDDL generation: a hidden target letter among distractors.

Produces scenes for DFA-driven "find the letter" play data: one target letter
block (default R) plus a few distractor blocks, all on the table and clear, with
randomized orientations so the target's letter starts hidden and the robot must
flip blocks to reveal it.

The PDDL goal (``ori_of target READABLE_ORI``) states the search objective and
is what stage-1 reset sampling / env codegen consume; the play pipeline itself
gets its per-episode goals from the DFA strategy at run time, not from this file.

Usage:
    # Find R among 2-4 distractors:
    python scripts/task_gen/search.py --letter R --distractors 2 4 \
        --count 50 --out-dir ./datasets/basics/search_r/tasks --seed 42
"""

import argparse

from toyblocks.tamp.blocks.glyph_blocks import (
    LETTERS,
    READABLE_ORI,
    all_glyph_blocks,
    blocks_for_text,
)
from toyblocks.task_gen import (
    And,
    Block,
    Clear,
    DistractorPool,
    OnTable,
    Or,
    OriOf,
    ProblemSpec,
)
from toyblocks.task_gen.orientations import ORIENTATIONS

# Color cuboids mixed into the distractor pool alongside unused letters.
COLOR_DISTRACTORS = [
    ("block_red1", "block_red"),
    ("block_green", "block_green"),
    ("block_blue", "block_blue"),
    ("block_yellow", "block_yellow"),
]


def build_spec(
    letter: str = "R",
    distractor_range: tuple = (2, 4),
    block_size_range: tuple = (0.03, 0.05),
    table_height_range: tuple = (0, 0.04),
    friction_range: tuple = (0.1, 0.4),
) -> ProblemSpec:
    letter = letter.upper()
    if letter not in LETTERS:
        raise ValueError(f"--letter must be a single letter A-Z, got {letter!r}")

    target_name, target_type, _ = blocks_for_text(letter)[0]

    spec = ProblemSpec(domain="blocks-tamp")
    spec.add_block(Block(target_name, block_type=target_type))

    count_min, count_max = distractor_range
    pool_blocks = [
        Block(name, block_type=btype)
        for name, btype, _ in all_glyph_blocks(LETTERS)
        if name != target_name
    ]
    pool_blocks += [Block(name, block_type=btype) for name, btype in COLOR_DISTRACTORS]
    spec.add_distractor_pool(
        DistractorPool(
            blocks=pool_blocks,
            count_min=min(count_min, len(pool_blocks)),
            count_max=min(count_max, len(pool_blocks)),
        )
    )

    # Random orientations everywhere so the target's letter starts hidden. The
    # target is pinned on-table + clear + NON-readable; distractors get placement
    # (on-table / light stacking) and random ori from the init_spec, which is what
    # keeps the init complete (see problem_spec._build_init: without random_stacking
    # unplaced blocks get no on_table/clear facts at all).
    spec.init_spec.random_stacking = True
    spec.init_spec.random_ori = True
    spec.init = And(
        OnTable(target_name),
        Clear(target_name),
        Or(
            *[OriOf(target_name, o) for o in ORIENTATIONS if o != READABLE_ORI],
            mode="random_choice",
        ),
    )

    # The search objective: reveal the target letter to the camera.
    spec.goal = And(OriOf(target_name, READABLE_ORI))

    spec.block_size_range = block_size_range
    spec.table_height_range = table_height_range
    spec.friction_range = friction_range
    return spec


def main():
    parser = argparse.ArgumentParser(description="Generate letter-search problems")
    parser.add_argument("--letter", type=str, default="R", help="Target letter A-Z (default R)")
    parser.add_argument(
        "--distractors",
        type=int,
        nargs=2,
        default=[2, 4],
        metavar=("MIN", "MAX"),
        help="Distractor count range (default: 2 4)",
    )
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_distractors", type=int, default=0,
                        help="Ignored; accepted for the datagen pipeline (distractors come from --distractors MIN MAX)")
    parser.add_argument("--preview", action="store_true", help="Print one PDDL without writing")
    parser.add_argument(
        "--block-size-range",
        type=float,
        nargs=2,
        default=[0.03, 0.05],
        metavar=("MIN", "MAX"),
        help="Block size range (default: 0.03 0.05)",
    )
    parser.add_argument(
        "--table-height-range",
        type=float,
        nargs=2,
        default=[0, 0.04],
        metavar=("MIN", "MAX"),
        help="Table height range (default: 0 0.04)",
    )
    parser.add_argument(
        "--friction-range",
        type=float,
        nargs=2,
        default=[0.1, 0.4],
        metavar=("MIN", "MAX"),
        help="Friction range (default: 0.1 0.4)",
    )
    args = parser.parse_args()

    spec = build_spec(
        letter=args.letter,
        distractor_range=tuple(args.distractors),
        block_size_range=tuple(args.block_size_range),
        table_height_range=tuple(args.table_height_range),
        friction_range=tuple(args.friction_range),
    )

    if args.preview:
        import random

        if args.seed is not None:
            random.seed(args.seed)
        print(spec.preview())
        return

    spec.generate(count=args.count, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
