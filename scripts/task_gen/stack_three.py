#!/usr/bin/env python3
"""Generate 3-block stacking problems in letter + digit + color scenes.

Scene: --num-blocks MIN MAX blocks per problem (default 4 10), drawn with weight
proportional to the count so larger scenes are slightly more common (P(4) ~ 8%,
P(10) ~ 20%). Every scene holds exactly --num-letters letter blocks (default 4)
and at most --max-color color cubes (default 4); the rest are digit blocks. The
color count is uniform over what the digit pool leaves feasible. Glyph USDs come
from train_blocks_large (pass --usd-library train_blocks_large to the datagen
pipeline).

Splits (--train, the default, or --val) pick the pools:
    train: letters C D E F G H J K L M N P Q S U V W X Y Z, digits 1 3 5,
           colors red blue green white
    val:   letters A R O B T I, digits 2 4, colors yellow pink gray purple

Init: every block spawns randomly -- on the table or on another block, towers
capped at --max-init-stack blocks -- at a random orientation, or, with
--init-ori, at one drawn per block from the given orientation(s) (e.g.
--init-ori o2, or --init-ori o1 o2 o5). --init-readable instead starts every glyph
block readable: letters at one of their goal orientations (below), digits and
color cubes at --glyph-ori (o2 when --glyph-ori is none). --init-flip-unreadable
starts every glyph block (letter or digit) with its glyph face visible but not
readable: one or more left / right flips (which turn the block in the plane of that
face) from a readable orientation, and not readable itself (o2 -> o6 / o10 / o14);
color cubes at --glyph-ori.

Goal: three randomly chosen scene blocks stacked bottom -> middle -> top, the
bottom one on the table and the top one clear. Every glyph block in the tower
must end readable: digits at --glyph-ori (default o2), letters at --glyph-ori or
any orientation that shows the same glyph, as an ``(or ...)`` goal:
  * I H N O S Z X read the same upside down, so two right flips from --glyph-ori
    (o2 -> o10) is accepted too;
  * C O P S U V W X Z look the same in lower case, which sits on the opposite face
    (two cw rotates, o2 -> o4);
  * O S X Z are in both sets and also accept the lower-case face upside down (o12).
--glyph-ori none sets no orientation goal; color cubes never have one.
A tower that already holds in the init is resampled. stack_two.py and
stack_four.py are the same generator with a 2- / 4-block tower.

Isolate scenes (--isolate-prob, default 0; e.g. 0.125): a target block T and a
neighbour N both start on the table and clear with (next_to T N) -- stage 1 places
T touching N -- and (pick_obstructs N T), and T is above the bottom of the goal
tower, so the plan must isolate N before picking T. Only the isolate action clears
next_to, so these problems need it enabled in domain.pddl (currently commented
out); without it the planner reports the goal unreachable.

Usage:
    python scripts/task_gen/stack_three.py --count 10 --seed 0 --preview
    python scripts/task_gen/stack_three.py --count 300 --out-dir .../tasks --seed 42
    python scripts/task_gen/stack_three.py --val --count 50 --out-dir .../val_tasks --seed 7
"""

import argparse
import random
from dataclasses import dataclass

from toyblocks.tamp.blocks.glyph_blocks import LETTERS, block_name, block_type
from toyblocks.task_gen import And, Block, Clear, NextTo, On, OnTable, Or, OriOf, Pred, ProblemSpec
from toyblocks.task_gen.height_capped_spec import HeightCappedSpec
from toyblocks.task_gen.orientations import ORIENTATIONS, flip, yaw
from toyblocks.task_gen.predicates import Expr
from toyblocks.task_gen.resolved_init import ResolvedInit

#: One cube per color; names and PDDL types match ``template/codegen.py`` and ``domain.pddl``.
COLOR_CUBES: dict[str, tuple[str, str]] = {
    "red": ("block_red1", "block_red"),
    "blue": ("block_blue", "block_blue"),
    "green": ("block_green", "block_green"),
    "white": ("block_white", "block_white"),
    "yellow": ("block_yellow", "block_yellow"),
    "pink": ("block_pink", "block_pink"),
    "gray": ("block_gray", "block_gray"),
    "purple": ("block_purple", "block_purple"),
}


@dataclass(frozen=True)
class Split:
    """Letter / digit / color pools of one data split."""

    letters: tuple[str, ...]
    digits: tuple[str, ...]
    colors: tuple[str, ...]


VAL_LETTERS: tuple[str, ...] = tuple("AROBTI")
SPLITS: dict[str, Split] = {
    "train": Split(
        letters=tuple(c for c in LETTERS if c not in VAL_LETTERS),
        digits=("1", "3", "5"),
        colors=("red", "blue", "green", "white"),
    ),
    "val": Split(letters=VAL_LETTERS, digits=("2", "4"), colors=("yellow", "pink", "gray", "purple")),
}

#: Letters that read the same upside down (two right flips away from the readable ori).
UPSIDE_DOWN_SYMMETRIC_LETTERS: frozenset[str] = frozenset("IHNOSZX")
#: Letters whose lower-case glyph, on the opposite face (two cw rotates away), looks the same.
CASE_SYMMETRIC_LETTERS: frozenset[str] = frozenset("COPSUVWXZ")
STACK_SIZE = 3
DEFAULT_NUM_BLOCKS: tuple[int, int] = (4, 10)


def glyph_goal_oris(glyph: str, readable_ori: str) -> list[str]:
    """Orientations in which ``glyph`` shows the same readable face as ``readable_ori``."""
    oris = [readable_ori]
    upper = glyph.upper()
    if upper in CASE_SYMMETRIC_LETTERS:
        oris.append(yaw(yaw(readable_ori, "cw"), "cw"))
    if upper in UPSIDE_DOWN_SYMMETRIC_LETTERS:
        oris += [flip(flip(o, "right_flip"), "right_flip") for o in list(oris)]
    return oris


def _holds(init: ResolvedInit, expr: Expr) -> bool:
    """True when ``expr`` (a fact or an Or of facts) already holds in ``init``."""
    if isinstance(expr, Or):
        return any(_holds(init, c) for c in expr.children)
    assert isinstance(expr, Pred)
    return init.has(expr.name, *expr.args)


class SceneSamplingSpec(HeightCappedSpec):
    """Draws a fresh block count and letter / digit / color mix for every problem."""

    def __init__(
        self,
        split: Split,
        num_blocks: tuple[int, int],
        num_letters: int,
        max_color: int,
        max_init_stack: int,
        isolate_prob: float = 0.0,
        init_oris: list[str] | None = None,
        init_readable_ori: str | None = None,
        init_flip_unreadable: bool = False,
    ) -> None:
        super().__init__(domain="blocks-tamp", max_stack_height=max_init_stack)
        self.split = split
        self.num_blocks = num_blocks
        self.num_letters = num_letters
        self.max_color = max_color
        self.color_names: set[str] = {COLOR_CUBES[c][0] for c in split.colors}
        self.isolate_prob = isolate_prob
        #: Initial orientations to draw from per block; None keeps the uniform random ori.
        self.init_oris = init_oris
        #: Readable orientation every block starts at (letters: any equivalent one); None = off.
        self.init_readable_ori = init_readable_ori
        #: Start glyph blocks unreadable but left / right flips from readable (needs init_readable_ori).
        self.init_flip_unreadable = init_flip_unreadable
        #: (target, obstructor) of the current problem's isolate setup, or None.
        self.isolate_pair: tuple[str, str] | None = None
        #: Block name -> glyph of the current problem's glyph blocks.
        self.glyph_of: dict[str, str] = {}

    def _sample_active_blocks(self) -> list[Block]:
        lo, hi = self.num_blocks
        counts = list(range(lo, hi + 1))
        # Weight proportional to the count: a slight skew towards fuller scenes.
        n = random.choices(counts, weights=[float(c) for c in counts])[0]
        n_rest = n - self.num_letters
        min_color = max(0, n_rest - len(self.split.digits))
        max_color = min(self.max_color, len(self.split.colors), n_rest)
        n_color = random.randint(min_color, max_color)
        letters = random.sample(self.split.letters, self.num_letters)
        digits = random.sample(self.split.digits, n_rest - n_color)
        colors = random.sample(self.split.colors, n_color)
        self.glyph_of = {block_name(g): g for g in (*letters, *digits)}
        blocks = [Block(name, block_type=block_type(g)) for name, g in self.glyph_of.items()]
        blocks += [Block(*COLOR_CUBES[c]) for c in colors]
        random.shuffle(blocks)
        return blocks

    def _validate_declared_refs(self) -> None:
        # Blocks are drawn per problem, so there is no fixed declared set to check.
        return None

    def _build_init(self, active_blocks: list[Block]):
        """Pin an isolate setup (for an ``isolate_prob`` share) and the --init-ori orientations into the init."""
        self.isolate_pair = None
        parts: list[Pred] = []
        if random.random() < self.isolate_prob:
            target, obstructor = random.sample([b.name for b in active_blocks], 2)
            self.isolate_pair = (target, obstructor)
            parts += [
                OnTable(target), Clear(target), OnTable(obstructor), Clear(obstructor),
                NextTo(target, obstructor), Pred("pick_obstructs", obstructor, target),
            ]
        if self.init_oris:
            # An explicit ori_of keeps ProblemSpec from drawing a random one for that block.
            parts += [OriOf(b.name, random.choice(self.init_oris)) for b in active_blocks]
        elif self.init_flip_unreadable:
            parts += [OriOf(b.name, random.choice(self._flip_unreadable_oris(b.name))) for b in active_blocks]
        elif self.init_readable_ori:
            parts += [OriOf(b.name, random.choice(self._readable_oris(b.name))) for b in active_blocks]
        self.init_spec.predicates = And(*parts) if parts else None
        return super()._build_init(active_blocks)

    def _readable_oris(self, block: str) -> list[str]:
        """Orientations in which ``block`` starts readable (color cubes: the readable ori)."""
        assert self.init_readable_ori is not None
        glyph = self.glyph_of.get(block)
        if glyph is not None and block_type(glyph) == "block_letter":
            return glyph_goal_oris(glyph, self.init_readable_ori)
        return [self.init_readable_ori]

    def _flip_unreadable_oris(self, block: str) -> list[str]:
        """Orientations one or more left / right flips from one of ``block``'s readable ones
        that are not readable themselves (color cubes: the readable ori)."""
        readable = self._readable_oris(block)
        if block not in self.glyph_of:
            return readable
        # left and right flips are inverse quarter turns, so 1-3 left flips reach the whole cycle
        reach: set[str] = set()
        for ori in readable:
            for _ in range(3):
                ori = flip(ori, "left_flip")
                reach.add(ori)
        oris = sorted(reach - set(readable), key=ORIENTATIONS.index)
        assert oris, f"{block}: every flip of a readable orientation is readable"
        return oris


def build_spec(
    split: Split,
    num_blocks: tuple[int, int] = DEFAULT_NUM_BLOCKS,
    num_letters: int = 4,
    max_color: int = 4,
    max_init_stack: int = 2,
    glyph_ori: str = "o2",
    isolate_prob: float = 0.0,
    init_oris: list[str] | None = None,
    init_readable: bool = False,
    init_flip_unreadable: bool = False,
    block_size_range: tuple[float, float] | None = None,
    table_height_range: tuple[float, float] | None = None,
    friction_range: tuple[float, float] | None = None,
    stack_size: int = STACK_SIZE,
) -> ProblemSpec:
    lo, hi = num_blocks
    if stack_size < 2:
        raise ValueError("the goal tower needs at least 2 blocks")
    if lo > hi or lo < max(stack_size, num_letters):
        raise ValueError(f"--num-blocks must satisfy max({stack_size}, --num-letters) <= MIN <= MAX")
    if num_letters > len(split.letters):
        raise ValueError(f"--num-letters exceeds the split's {len(split.letters)} letters")
    if hi - num_letters > min(max_color, len(split.colors)) + len(split.digits):
        raise ValueError("not enough digits / colors in the split for --num-blocks MAX")
    if not 0.0 <= isolate_prob <= 1.0:
        raise ValueError("--isolate-prob must be in [0, 1]")
    if glyph_ori != "none" and glyph_ori not in ORIENTATIONS:
        raise ValueError(f"--glyph-ori: unknown orientation {glyph_ori!r}; expected o1..o{len(ORIENTATIONS)} or none")
    unknown = sorted(set(init_oris or ()) - set(ORIENTATIONS))
    if unknown:
        raise ValueError(f"--init-ori: unknown orientation(s) {unknown}; expected o1..o{len(ORIENTATIONS)}")
    if sum((init_readable, init_flip_unreadable, bool(init_oris))) > 1:
        raise ValueError("--init-readable, --init-flip-unreadable and --init-ori are mutually exclusive")
    readable_ori = (
        (glyph_ori if glyph_ori != "none" else "o2") if init_readable or init_flip_unreadable else None
    )
    spec = SceneSamplingSpec(
        split, num_blocks, num_letters, max_color, max_init_stack, isolate_prob, init_oris, readable_ori,
        init_flip_unreadable,
    )
    spec.init_spec.random_stacking = True
    spec.init_spec.random_ori = True

    def ori_goal(block: str) -> Expr:
        glyph = spec.glyph_of[block]
        oris = glyph_goal_oris(glyph, glyph_ori) if block_type(glyph) == "block_letter" else [glyph_ori]
        return OriOf(block, oris[0]) if len(oris) == 1 else Or(*[OriOf(block, o) for o in oris])

    def tower_goal(init: ResolvedInit) -> And:
        names = sorted({p.args[0] for p in init.predicates if p.name in ("on", "on_table")})
        for _ in range(100):
            if spec.isolate_pair is not None:
                # The obstructed target has to be picked: it goes anywhere above the bottom.
                target = spec.isolate_pair[0]
                tower = random.sample([n for n in names if n != target], stack_size - 1)
                tower.insert(1 + int(random.random() * (stack_size - 1)), target)
            else:
                tower = random.sample(names, stack_size)
            # Bottom -> top: bottom on the table, each block on the one below, top clear.
            preds: list[Expr] = [OnTable(tower[0])]
            preds += [On(above, below) for below, above in zip(tower, tower[1:])]
            preds += [Clear(tower[-1])]
            if glyph_ori != "none":
                preds += [ori_goal(b) for b in tower if b not in spec.color_names]
            if not all(_holds(init, p) for p in preds):
                return And(*preds)
        raise RuntimeError("could not sample a tower that is not already built")

    spec.goal = tower_goal
    if block_size_range is not None:
        spec.block_size_range = block_size_range
    if table_height_range is not None:
        spec.table_height_range = table_height_range
    if friction_range is not None:
        spec.friction_range = friction_range
    return spec


def main(stack_size: int = STACK_SIZE) -> None:
    parser = argparse.ArgumentParser(
        description=f"Generate {stack_size}-block stacking problems in letter + digit + color scenes")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument("--train", dest="split", action="store_const", const="train", default="train",
                             help="Training pools: letters minus A R O B T I, digits 1 3 5, colors red blue green white (default)")
    split_group.add_argument("--val", dest="split", action="store_const", const="val",
                             help="Validation pools: letters A R O B T I, digits 2 4, colors yellow pink gray purple")
    parser.add_argument("--num-blocks", type=int, nargs=2, default=list(DEFAULT_NUM_BLOCKS), metavar=("MIN", "MAX"),
                        help="Blocks per scene, drawn with weight proportional to the count (default: 4 10)")
    parser.add_argument("--num-letters", type=int, default=4, help="Letter blocks in every scene (default: 4)")
    parser.add_argument("--max-color", type=int, default=4, help="Max color cubes per scene (default: 4)")
    parser.add_argument("--max-init-stack", type=int, default=2,
                        help="Tallest init tower in blocks; 1 puts every block on the table (default: 2)")
    parser.add_argument("--glyph-ori", default="o2",
                        help="Readable orientation of glyph blocks in the tower; symmetric letters also accept the "
                             "orientations that show the same glyph; 'none' sets no orientation goal "
                             "(pick-and-place-only towers) (default: o2)")
    parser.add_argument("--init-ori", nargs="+", default=None, metavar="ORI",
                        help="Initial orientation(s): one (e.g. o2) for every block, or several (e.g. o1 o2 o5) "
                             "drawn uniformly per block (default: uniform over all 24)")
    parser.add_argument("--init-readable", action="store_true",
                        help="Start every block readable: letters at one of their goal orientations (--glyph-ori "
                             "or a symmetric equivalent), digits and color cubes at --glyph-ori; excludes --init-ori")
    parser.add_argument("--init-flip-unreadable", action="store_true",
                        help="Start every glyph block (letter or digit) visible but unreadable: one or more left / "
                             "right flips from a readable orientation and not readable itself; color cubes at "
                             "--glyph-ori; excludes --init-ori and --init-readable")
    parser.add_argument("--isolate-prob", type=float, default=0.0,
                        help="Share of problems whose goal block starts obstructed by a neighbour, so the plan "
                             "must isolate it; needs the isolate action enabled in domain.pddl (default: 0)")
    parser.add_argument("--block-size-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--table-height-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--friction-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--num_distractors", type=int, default=0, help="Ignored; accepted for the datagen pipeline")
    parser.add_argument("--preview", action="store_true", help="Print one PDDL without writing")
    args = parser.parse_args()

    spec = build_spec(
        SPLITS[args.split],
        num_blocks=(args.num_blocks[0], args.num_blocks[1]),
        num_letters=args.num_letters,
        max_color=args.max_color,
        max_init_stack=args.max_init_stack,
        glyph_ori=args.glyph_ori,
        isolate_prob=args.isolate_prob,
        init_oris=args.init_ori,
        init_readable=args.init_readable,
        init_flip_unreadable=args.init_flip_unreadable,
        block_size_range=tuple(args.block_size_range) if args.block_size_range else None,
        table_height_range=tuple(args.table_height_range) if args.table_height_range else None,
        friction_range=tuple(args.friction_range) if args.friction_range else None,
        stack_size=stack_size,
    )
    if args.preview:
        if args.seed is not None:
            random.seed(args.seed)
        print(spec.preview())
        return
    spec.generate(count=args.count, out_dir=args.out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
