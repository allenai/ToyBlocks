#!/usr/bin/env python3
"""Pick-and-place block_r once: onto a support block, or off it onto the table.

Supports one constraint:
  --mode   ``stack``: block_r starts on the table and goes onto the support block;
           ``unstack``: block_r starts on the support block and goes onto the table;
           ``all`` (default): each problem picks one of the two.

block_r keeps its orientation (the goal pins its initial ``ori_of``), so the task is one
pick-and-place, never a flip or rotate.

Fixed blocks: the pick target (default block_r; --target-block) and the support block
(default block_green; --support-block).
Distractors: 0-5 sampled from block_o1, block_b, block_t, block_green, block_blue, block_red1
(the target and the support are excluded from this pool).

Usage:
    # Either direction:
    python scripts/task_gen/pnp.py \
        --count 50 --out-dir ./datasets/basics/pnp/tasks --seed 42

    # Only stacking block_r onto block_blue:
    python scripts/task_gen/pnp.py \
        --count 50 --mode stack --support-block block_blue --out-dir ... --seed 42
"""

import argparse
import random

from toyblocks.task_gen import (
    And,
    Block,
    Clear,
    DistractorPool,
    On,
    OnTable,
    Or,
    OriOf,
    ProblemSpec,
)
from toyblocks.task_gen.resolved_init import ResolvedInit

MODES = ("stack", "unstack", "all")


def build_spec(
    mode: str = "all",
    num_distractors: int = 5,
    block_size_range: tuple[float, float] = (0.03, 0.05),
    table_height_range: tuple[float, float] = (0, 0.04),
    friction_range: tuple[float, float] = (0.1, 0.4),
    target_block: str = "block_r",
    support_block: str = "block_green",
    distractor_blocks: list[str] | None = None,
    distractor_count_min: int = 0,
) -> ProblemSpec:
    """Build a pick-and-place ProblemSpec.

    Args:
        mode: ``"stack"`` (table -> support block), ``"unstack"`` (support block -> table)
            or ``"all"`` (one of the two per problem).
        num_distractors: Maximum number of distractor blocks (0 to disable).
        block_size_range: (min, max) block size in meters.
        table_height_range: (min, max) table height offset in meters.
        friction_range: (min, max) friction coefficient.
        target_block: PDDL object name of the block that is picked (default "block_r").
        support_block: PDDL object name of the block it is placed on / taken off
            (default "block_green").
        distractor_blocks: Override the distractor pool with these block names (types
            inferred from the name). None keeps the default pool.
        distractor_count_min: Minimum distractors sampled per problem. With num_distractors
            as the max, set both equal for an exact count.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if target_block == support_block:
        raise ValueError("the target and the support block must differ")
    spec = ProblemSpec(domain="blocks-tamp")

    # -- blocks ---------------------------------------------------------------
    spec.add_block(Block(target_block))
    spec.add_block(Block(support_block))
    if num_distractors > 0:
        if distractor_blocks is not None:
            pool = [Block(name) for name in distractor_blocks]
        else:
            pool = [
                Block("block_o1", block_type="block_o_b"),
                Block("block_b", block_type="block_o_b"),
                Block("block_t", block_type="block_t_g"),
                Block("block_green"),
                Block("block_blue"),
                Block("block_red1"),
            ]
        pool = [b for b in pool if b.name not in (target_block, support_block)]
        if pool:
            spec.add_distractor_pool(
                DistractorPool(
                    blocks=pool,
                    count_min=min(distractor_count_min, len(pool)),
                    count_max=min(num_distractors, len(pool)),
                )
            )

    # -- init -----------------------------------------------------------------
    # Distractors get random stacking / orientations; so does block_r's orientation.
    spec.init_spec.random_stacking = True
    spec.init_spec.random_ori = True
    on_table = And(OnTable(target_block), Clear(target_block), OnTable(support_block), Clear(support_block))
    stacked = And(On(target_block, support_block), Clear(target_block), OnTable(support_block))
    if mode == "stack":
        spec.init = on_table
    elif mode == "unstack":
        spec.init = stacked
    else:
        spec.init = Or(on_table, stacked, mode="random_choice")

    # -- goal: the other placement, same orientation ----------------------------
    def pnp_goal(init: ResolvedInit) -> And:
        keep_ori = OriOf(target_block, init.arg("ori_of", target_block))
        if init.has("on", target_block, support_block):
            return And(OnTable(target_block), Clear(target_block), keep_ori)
        return And(On(target_block, support_block), Clear(target_block), keep_ori)

    spec.goal = pnp_goal

    # -- physical randomisation ranges ----------------------------------------
    spec.block_size_range = block_size_range
    spec.table_height_range = table_height_range
    spec.friction_range = friction_range
    return spec


def main() -> None:
    # allow_abbrev=False: `--block` style prefixes must not match `--block-size-range`
    parser = argparse.ArgumentParser(description="Generate pick-and-place problems", allow_abbrev=False)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=list(MODES),
        help="stack (table -> support block), unstack (support block -> table) or all (default)",
    )
    parser.add_argument(
        "--num_distractors",
        type=int,
        default=5,
        help="Max number of distractors to sample per problem (default: 5)",
    )
    parser.add_argument(
        "--target-block",
        type=str,
        default="block_r",
        help="Object name of the block that is picked (default: block_r)",
    )
    parser.add_argument(
        "--support-block",
        type=str,
        default="block_green",
        help="Object name of the block it is placed on / taken off (default: block_green)",
    )
    parser.add_argument(
        "--distractor-blocks",
        type=str,
        nargs="+",
        default=None,
        help="Override the distractor pool with these block names (default: built-in pool)",
    )
    parser.add_argument(
        "--distractor-count-min",
        type=int,
        default=0,
        help="Min distractors per problem; set equal to --num_distractors for an exact count",
    )
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
        mode=args.mode,
        num_distractors=args.num_distractors,
        block_size_range=tuple(args.block_size_range),
        table_height_range=tuple(args.table_height_range),
        friction_range=tuple(args.friction_range),
        target_block=args.target_block,
        support_block=args.support_block,
        distractor_blocks=args.distractor_blocks,
        distractor_count_min=args.distractor_count_min,
    )

    if args.preview:
        if args.seed is not None:
            random.seed(args.seed)
        print(spec.preview())
        return

    spec.generate(count=args.count, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
