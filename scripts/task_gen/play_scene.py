#!/usr/bin/env python3
"""Generate simple play-data scenes: a few glyph blocks on the table.

Every block starts on the table, clear, at a random orientation, so the first
goals can flip, rotate or stack any of them. The PDDL goal is only what stage-1
codegen consumes; play data samples its own goals per episode.

Usage:
    python scripts/task_gen/play_scene.py --count 8 --out-dir outputs/play/tasks --seed 0 \
        --blocks A I 2 E
"""
import argparse

from blocksuite.tamp.blocks.glyph_blocks import block_name, block_type
from blocksuite.task_gen import And, Block, Clear, On, OnTable, ProblemSpec


def build_spec(
    glyphs: list[str],
    block_size_range: tuple[float, float] | None = None,
    table_height_range: tuple[float, float] | None = None,
    friction_range: tuple[float, float] | None = None,
) -> ProblemSpec:
    """A scene with one block per glyph, all on the table and clear, at random orientations."""
    names: list[str] = [block_name(g) for g in glyphs]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate glyphs in {glyphs}")

    spec = ProblemSpec(domain="blocks-tamp")
    for name, glyph in zip(names, glyphs):
        spec.add_block(Block(name, block_type=block_type(glyph)))
    spec.init_spec.random_stacking = False
    spec.init_spec.random_ori = True
    init_parts = []
    for name in names:
        init_parts += [OnTable(name), Clear(name)]
    spec.init = And(*init_parts)
    # Codegen needs some goal; play data never uses it.
    spec.goal = And(On(names[0], names[1])) if len(names) > 1 else And(OnTable(names[0]))

    if block_size_range is not None:
        spec.block_size_range = block_size_range
    if table_height_range is not None:
        spec.table_height_range = table_height_range
    if friction_range is not None:
        spec.friction_range = friction_range
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple glyph-block play-data scenes")
    parser.add_argument("--count", type=int, default=1, help="Number of PDDL problems")
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_distractors", type=int, default=0,
                        help="Ignored; accepted for the datagen pipeline")
    parser.add_argument(
        "--blocks", nargs="+", default=["A", "I", "2", "E"],
        help="Glyphs to place, one block each (default: A I 2 E)",
    )
    parser.add_argument("--block-size-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--table-height-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--friction-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    args = parser.parse_args()

    for i in range(args.count):
        spec = build_spec(
            args.blocks,
            block_size_range=tuple(args.block_size_range) if args.block_size_range else None,
            table_height_range=tuple(args.table_height_range) if args.table_height_range else None,
            friction_range=tuple(args.friction_range) if args.friction_range else None,
        )
        spec.generate(count=1, out_dir=args.out_dir, seed=args.seed + i)


if __name__ == "__main__":
    main()
