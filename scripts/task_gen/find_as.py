#!/usr/bin/env python3
"""Generate 'find the A blocks' search scenes: 10 glyph blocks, >=4 of them A.

Each problem places 4 A blocks (block_a, block_a_2, block_a_3, block_a_4) plus 6
distractor glyph blocks on the table, all clear, at random orientations. The
distractor letters differ per variant so the five problems are genuinely
different block sets. The PDDL goal (three A faces readable) is only what stage-1
codegen consumes; the play pipeline gets its per-episode goals from the DFA.

Usage:
    python scripts/task_gen/find_as.py --count 5 \
        --out-dir outputs/find_as/tasks --seed 42
"""

import argparse

from toyblocks.tamp.blocks.glyph_blocks import block_name, block_type
from toyblocks.task_gen import And, Block, Clear, OnTable, OriOf, ProblemSpec

# Five distinct distractor letter sets (6 each, no A) -> five different scenes.
DISTRACTOR_SETS = [
    ["B", "C", "D", "E", "F", "G"],
    ["H", "I", "J", "K", "L", "M"],
    ["N", "O", "P", "Q", "R", "S"],
    ["T", "U", "V", "W", "X", "Y"],
    ["Z", "B", "N", "R", "T", "W"],
]

NUM_A = 4
READABLE = "o2"


def build_spec(
    distractor_letters: list[str],
    block_size_range: tuple = (0.03, 0.05),
    table_height_range: tuple = (0, 0.04),
    friction_range: tuple = (0.1, 0.4),
) -> ProblemSpec:
    a_names = [block_name("A", i) for i in range(1, NUM_A + 1)]
    distractor_names = [block_name(letter) for letter in distractor_letters]

    spec = ProblemSpec(domain="blocks-tamp")
    for name in a_names:
        spec.add_block(Block(name, block_type=block_type("A")))
    for name, letter in zip(distractor_names, distractor_letters):
        spec.add_block(Block(name, block_type=block_type(letter)))

    # Flat scene, random orientation for every block: all 10 on the table and
    # clear (so each is flippable), each at a random face. random_stacking=False
    # keeps them unstacked; the explicit OnTable/Clear keep the init complete.
    spec.init_spec.random_stacking = False
    spec.init_spec.random_ori = True
    init_parts = []
    for name in a_names + distractor_names:
        init_parts += [OnTable(name), Clear(name)]
    spec.init = And(*init_parts)

    # Objective (for codegen only; the DFA supplies the real search goals): three
    # of the A faces revealed.
    spec.goal = And(*[OriOf(a_names[i], READABLE) for i in range(3)])

    spec.block_size_range = block_size_range
    spec.table_height_range = table_height_range
    spec.friction_range = friction_range
    return spec


def main():
    parser = argparse.ArgumentParser(description="Generate find-the-A-blocks scenes")
    parser.add_argument("--count", type=int, default=len(DISTRACTOR_SETS))
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preview", action="store_true", help="Print one PDDL without writing")
    parser.add_argument("--block-size-range", type=float, nargs=2, default=[0.03, 0.05], metavar=("MIN", "MAX"))
    parser.add_argument("--table-height-range", type=float, nargs=2, default=[0, 0.04], metavar=("MIN", "MAX"))
    parser.add_argument("--friction-range", type=float, nargs=2, default=[0.1, 0.4], metavar=("MIN", "MAX"))
    parser.add_argument("--num_distractors", type=int, default=0,
                        help="Ignored; accepted for the datagen pipeline (distractor sets are fixed)")
    args = parser.parse_args()
    ranges = dict(
        block_size_range=tuple(args.block_size_range),
        table_height_range=tuple(args.table_height_range),
        friction_range=tuple(args.friction_range),
    )

    if args.preview:
        import random

        random.seed(args.seed)
        print(build_spec(DISTRACTOR_SETS[0], **ranges).preview())
        return

    # One PDDL per variant (distractor set), each its own random-orientation draw.
    for i in range(args.count):
        letters = DISTRACTOR_SETS[i % len(DISTRACTOR_SETS)]
        spec = build_spec(letters, **ranges)
        spec.generate(count=1, out_dir=args.out_dir, seed=args.seed + i)


if __name__ == "__main__":
    main()
