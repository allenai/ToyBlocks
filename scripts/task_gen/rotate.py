#!/usr/bin/env python3
"""Rotate block_r by one yaw step (CW or CCW).

Supports two orthogonal constraints:
  --direction   Restrict the yaw direction (cw, ccw).  Omit for either.
  --goal-ori    Require a specific goal orientation (e.g. o2).  When set, the
                init orientation is constrained to orientations exactly one
                allowed yaw step away from the target.  Omit to let the yaw
                table determine the goal from a random init.

Fixed blocks: block_r.
Distractors: 0-5 sampled from block_o1, block_b, block_t, block_green, block_blue, block_red1.

Usage:
    # Any direction, any goal (fully dynamic):
    python scripts/task_gen/rotate.py \
        --count 50 --out-dir ./datasets/basics/rotate/tasks --seed 42

    # CW rotations only, goal determined by yaw table:
    python scripts/task_gen/rotate.py \
        --count 50 --direction cw --out-dir ... --seed 42

    # Any direction, but goal is always o2:
    python scripts/task_gen/rotate.py \
        --count 50 --goal-ori o2 --out-dir ... --seed 42

    # Only CCW rotations that land on o2:
    python scripts/task_gen/rotate.py \
        --count 50 --direction ccw --goal-ori o2 --out-dir ... --seed 42
"""

import argparse
import random
from typing import Dict, List, Optional

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
from toyblocks.task_gen.orientations import (
    ORIENTATIONS,
    YAW_CCW,
    YAW_CW,
    yaw,
)

YAW_TABLES: Dict[str, Dict[str, str]] = {
    "cw": YAW_CW,
    "ccw": YAW_CCW,
}


def _inverse_yaw(goal_ori: str, directions: List[str]) -> List[str]:
    """Return all init orientations that reach *goal_ori* via one yaw in *directions*.

    Each yaw table is a bijection on o1-o24, so for a single direction there is
    exactly one init orientation that maps to any given goal.  When both
    directions are allowed the union can be up to 2 orientations.
    """
    result = set()
    for d in directions:
        table = YAW_TABLES[d]
        for init_ori, result_ori in table.items():
            if result_ori == goal_ori:
                result.add(init_ori)
    return sorted(result, key=lambda o: int(o[1:]))


def build_spec(
    direction: str = "all",
    goal_ori: Optional[str] = None,
    num_distractors: int = 5,
    block_size_range: tuple = (0.03, 0.05),
    table_height_range: tuple = (0, 0.04),
    friction_range: tuple = (0.1, 0.4),
    target_block: str = "block_r",
    distractor_blocks: list[str] | None = None,
    distractor_count_min: int = 0,
) -> ProblemSpec:
    """Build a rotate-task ProblemSpec.

    Args:
        direction: Yaw direction to allow.  One of "cw", "ccw", or "all"
            (either direction).
        goal_ori: If set (e.g. "o2"), every generated problem targets this
            orientation and the init is back-solved from the yaw tables.
            If None, the goal is computed forward from a random init.
        num_distractors: Maximum number of distractor blocks (0 to disable).
        block_size_range: (min, max) block size in meters.
        table_height_range: (min, max) table height offset in meters.
        friction_range: (min, max) friction coefficient.
        target_block: PDDL object name of the block being rotated (default
            "block_r"). Its PDDL type is inferred from the name (e.g. a glyph
            block like "block_a" gets type "block").
        distractor_blocks: Override the distractor pool with these block names
            (types inferred from the name). None keeps the default pool.
        distractor_count_min: Minimum distractors sampled per problem. With
            num_distractors as the max, set both equal for an exact count.
    """
    spec = ProblemSpec(domain="blocks-tamp")

    # -- blocks ---------------------------------------------------------------
    spec.add_block(Block(target_block))

    # 0-N distractor blocks sampled per problem; they get random stacking and
    # orientations but are not involved in the rotate goal.
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
        spec.add_distractor_pool(
            DistractorPool(
                blocks=pool,
                count_min=distractor_count_min,
                count_max=num_distractors,
            )
        )

    # -- init -----------------------------------------------------------------
    # Distractors get random stacking/orientation; block_r's orientation is
    # either constrained (goal_ori mode) or fully random (dynamic mode).
    spec.init_spec.random_stacking = True
    spec.init_spec.random_ori = True

    # Resolve which yaw directions are allowed for this spec.
    directions = list(YAW_TABLES.keys()) if direction == "all" else [direction]

    # -- goal -----------------------------------------------------------------
    # Two modes depending on whether the caller pinned a goal orientation.
    if goal_ori is not None:
        # Fixed-goal mode: walk the yaw tables *backward* to find every init
        # orientation that lands on goal_ori in one yaw (within the allowed
        # directions), then constrain the init to one of those orientations.
        valid_inits = _inverse_yaw(goal_ori, directions)
        if not valid_inits:
            raise ValueError(
                f"No init orientation reaches {goal_ori} via {directions}"
            )
        spec.init = And(
            OnTable(target_block),
            Clear(target_block),
            # Each problem randomly picks one of the valid starting orientations.
            Or(*[OriOf(target_block, o) for o in valid_inits], mode="random_choice"),
        )
        spec.goal = And(Clear(target_block), OriOf(target_block, goal_ori))
    else:
        # Dynamic-goal mode: block_r starts at any of the 24 orientations
        # (assigned by random_ori), and the goal is computed *forward* through
        # the yaw table at instantiation time via a ResolvedInit callback.
        spec.init = And(
            OnTable(target_block),
            Clear(target_block),
        )

        def rotate_goal(init):
            init_ori = init.arg("ori_of", target_block)
            goal_o = yaw(init_ori, random.choice(directions))
            assert goal_o != init_ori, (
                f"yaw produced a no-op: {init_ori} -> {goal_o}"
            )
            return And(Clear(target_block), OriOf(target_block, goal_o))

        spec.goal = rotate_goal

    # -- physical randomisation ranges ----------------------------------------
    spec.block_size_range = block_size_range
    spec.table_height_range = table_height_range
    spec.friction_range = friction_range

    return spec


def main():
    parser = argparse.ArgumentParser(description="Generate rotate problems")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--direction", type=str, default="all",
        choices=["cw", "ccw", "all"],
        help="Constrain to a single yaw direction (default: any)",
    )
    parser.add_argument(
        "--goal-ori", type=str, default=None,
        choices=ORIENTATIONS,
        help="Require a specific goal orientation (default: determined by yaw table)",
    )
    parser.add_argument(
        "--num_distractors", type=int, default=5,
        help="Number of distractors to sample per problem (default: 5)",
    )
    parser.add_argument(
        "--target-block", type=str, default="block_r",
        help="Object name of the block being rotated (default: block_r)",
    )
    parser.add_argument(
        "--distractor-blocks", type=str, nargs="+", default=None,
        help="Override the distractor pool with these block names (default: built-in pool)",
    )
    parser.add_argument(
        "--distractor-count-min", type=int, default=0,
        help="Min distractors per problem; set equal to --num_distractors for an exact count",
    )
    parser.add_argument("--preview", action="store_true", help="Print one PDDL without writing")
    parser.add_argument(
        "--block-size-range", type=float, nargs=2, default=[0.03, 0.05],
        metavar=("MIN", "MAX"), help="Block size range (default: 0.03 0.05)",
    )
    parser.add_argument(
        "--table-height-range", type=float, nargs=2, default=[0, 0.04],
        metavar=("MIN", "MAX"), help="Table height range (default: 0 0.04)",
    )
    parser.add_argument(
        "--friction-range", type=float, nargs=2, default=[0.1, 0.4],
        metavar=("MIN", "MAX"), help="Friction range (default: 0.1 0.4)",
    )
    args = parser.parse_args()

    spec = build_spec(
        direction=args.direction,
        goal_ori=args.goal_ori,
        num_distractors=args.num_distractors,
        block_size_range=tuple(args.block_size_range),
        table_height_range=tuple(args.table_height_range),
        friction_range=tuple(args.friction_range),
        target_block=args.target_block,
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
