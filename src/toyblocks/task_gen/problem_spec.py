"""ProblemSpec: declarative PDDL problem generation for the Blocks domain."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from toyblocks.task_gen.blocks import Block, Distractor, DistractorPool
from toyblocks.task_gen.materialize import collect_block_refs, materialize
from toyblocks.task_gen.orientations import (
    ORIENTATIONS,
    generate_static_physics,
    static_physics_predicates,
)
from toyblocks.task_gen.predicates import (
    And,
    Clear,
    Expr,
    GripperAtInit,
    HandEmpty,
    On,
    OnTable,
    Or,
    OriOf,
    Pred,
)
from toyblocks.task_gen.problem import SymbolicProblem, render_problem
from toyblocks.task_gen.resolved_init import ResolvedInit

GoalSpec = Union[Expr, Callable[[ResolvedInit], Expr]]


def _range_literal(spec: Sequence) -> str:
    """Render a spawn range as a PDDL comment literal.

    A single ``(lo, hi)`` pair renders as ``[lo, hi]``; a sequence of pairs
    renders as ``[[lo, hi], [lo, hi]]``. Both round-trip through the ToyBlocks
    parser, which accepts either shape.
    """
    items = list(spec)
    if items and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in items):
        return str([float(v) for v in items])
    return str([[float(lo), float(hi)] for lo, hi in items])


@dataclass
class InitSpec:
    """Specification for the initial state."""
    predicates: Optional[Expr] = None
    random_stacking: bool = True
    random_ori: bool = True
    default_ori: Optional[str] = None
    include_static_physics: bool = False
    include_handempty: bool = True
    include_gripper_at_init: bool = True
    forbid_stack: Optional[Callable[[str, str], bool]] = None
    """Optional predicate ``(top, bottom) -> bool``. When it returns True for a
    candidate pair, random stacking will not place ``top`` directly on
    ``bottom``. Defaults to allowing every stacking."""


class ProblemSpec:
    """Declarative specification for generating PDDL problems.

    Sampling builds a :class:`SymbolicProblem` (objects, init, goal as Expr
    trees). PDDL text is produced only by :func:`render_problem` at the end.
    """

    def __init__(self, domain: str = "blocks-tamp"):
        self.domain = domain
        self._blocks: List[Block] = []
        self._distractors: List[Distractor] = []
        self._distractor_pools: List[DistractorPool] = []
        self.init_spec = InitSpec()
        self.goal: Optional[GoalSpec] = None

        self.block_size_range: Optional[Tuple[float, float]] = None
        self.table_height_range: Optional[Tuple[float, float]] = None
        self.friction_range: Optional[Tuple[float, float]] = None
        # Table-frame spawn window for on-table blocks. Either a single
        # (lo, hi) interval or a sequence of disjoint intervals, which spawns
        # nothing in the gaps. None leaves the sampler's default window.
        self.block_x_range: Optional[Sequence] = None
        self.block_y_range: Optional[Sequence] = None

    def add_block(self, block: Block) -> "ProblemSpec":
        self._blocks.append(block)
        return self

    def add_distractor(self, distractor: Distractor) -> "ProblemSpec":
        self._distractors.append(distractor)
        return self

    def add_distractor_pool(self, pool: DistractorPool) -> "ProblemSpec":
        self._distractor_pools.append(pool)
        return self

    def _declared_block_names(self) -> Set[str]:
        names = {b.name for b in self._blocks}
        names.update(d.name for d in self._distractors)
        for pool in self._distractor_pools:
            names.update(b.name for b in pool.blocks)
        return names

    def _validate_block_refs(
        self,
        expr: Expr,
        allowed: Set[str],
        *,
        context: str,
    ) -> None:
        missing = collect_block_refs(expr) - allowed
        if missing:
            raise ValueError(
                f"{context} references unknown block(s): {sorted(missing)}. "
                f"Known blocks: {sorted(allowed)}"
            )

    def _validate_declared_refs(self) -> None:
        """Ensure init/goal only name blocks declared on this spec."""
        declared = self._declared_block_names()
        if not declared:
            raise ValueError(
                "No blocks declared. Use add_block(), add_distractor(), "
                "or add_distractor_pool() before setting init/goal."
            )
        if self.init_spec.predicates is not None:
            self._validate_block_refs(
                self.init_spec.predicates, declared, context="init"
            )
        if self.goal is not None and not callable(self.goal):
            self._validate_block_refs(self.goal, declared, context="goal")

    @property
    def init(self) -> Optional[Expr]:
        return self.init_spec.predicates

    @init.setter
    def init(self, expr: Expr):
        self.init_spec.predicates = expr

    def _sample_active_blocks(self) -> List[Block]:
        active = list(self._blocks)
        for d in self._distractors:
            if random.random() < d.prob:
                active.append(Block(d.name, d.block_type))
        for pool in self._distractor_pools:
            active.extend(pool.sample())
        return active

    def _force_random_choice(self, expr: Expr) -> None:
        if isinstance(expr, Or):
            expr.mode = "random_choice"
            for child in expr.children:
                self._force_random_choice(child)
        elif isinstance(expr, And):
            for child in expr.children:
                self._force_random_choice(child)

    def _instantiate(self, active_blocks: List[Block]) -> SymbolicProblem:
        """Sample one symbolic problem (no PDDL text yet)."""
        active_names = {b.name for b in active_blocks}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        random_suffix = random.randint(0, 9999)
        name = f"declarative_{len(active_blocks)}_{timestamp}_{random_suffix:04d}"

        objects_by_type: Dict[str, List[str]] = {}
        for b in active_blocks:
            objects_by_type.setdefault(b.block_type, []).append(b.name)

        init_expr, resolved_init = self._build_init(active_blocks)
        goal_expr = self._build_goal(resolved_init)

        self._validate_block_refs(init_expr, active_names, context="init")
        self._validate_block_refs(goal_expr, active_names, context="goal")

        return SymbolicProblem(
            name=name,
            domain=self.domain,
            objects_by_type=objects_by_type,
            init=init_expr,
            goal=goal_expr,
        )

    def _build_init(self, active_blocks: List[Block]) -> Tuple[Expr, ResolvedInit]:
        block_names = [b.name for b in active_blocks]
        init_parts: List[Expr] = []
        resolved = ResolvedInit()

        explicit_on_blocks: Set[str] = set()
        explicit_ori_blocks: Set[str] = set()

        if self.init_spec.predicates:
            self._force_random_choice(self.init_spec.predicates)
            explicit = materialize(self.init_spec.predicates)
            init_parts.append(explicit)
            resolved.absorb(explicit)
            explicit_on_blocks, explicit_ori_blocks = self._parse_explicit_placements(
                self.init_spec.predicates
            )

        if self.init_spec.random_stacking:
            unplaced = [b for b in block_names if b not in explicit_on_blocks]
            if unplaced:
                stacking = self._random_stacking(
                    unplaced,
                    explicit_clear=self._get_explicit_clear(self.init_spec.predicates),
                )
                for pred in stacking:
                    init_parts.append(pred)
                    resolved.record(pred)

        if self.init_spec.random_ori or self.init_spec.default_ori:
            for b in block_names:
                if b not in explicit_ori_blocks:
                    ori = (
                        self.init_spec.default_ori
                        if self.init_spec.default_ori
                        else random.choice(ORIENTATIONS)
                    )
                    pred = OriOf(b, ori)
                    init_parts.append(pred)
                    resolved.record(pred)

        if self.init_spec.include_handempty:
            pred = HandEmpty()
            init_parts.append(pred)
            resolved.record(pred)

        if self.init_spec.include_gripper_at_init:
            pred = GripperAtInit()
            init_parts.append(pred)
            resolved.record(pred)

        if self.init_spec.include_static_physics:
            for pred in static_physics_predicates():
                init_parts.append(pred)
                resolved.record(pred)

        return And(*init_parts), resolved

    def _build_goal(self, init: ResolvedInit) -> Expr:
        if self.goal is None:
            raise ValueError(
                "No goal specified. Set spec.goal to an Expr or "
                "Callable[[ResolvedInit], Expr]"
            )
        if callable(self.goal):
            goal_expr = self.goal(init)
        else:
            goal_expr = self.goal
        if not isinstance(goal_expr, Expr):
            raise TypeError(
                f"goal builder must return Expr, got {type(goal_expr).__name__}"
            )
        return goal_expr

    def _parse_explicit_placements(self, expr: Expr) -> Tuple[Set[str], Set[str]]:
        on_blocks: Set[str] = set()
        ori_blocks: Set[str] = set()
        self._walk_for_placements(expr, on_blocks, ori_blocks)
        return on_blocks, ori_blocks

    def _walk_for_placements(self, expr: Expr, on_blocks: Set[str], ori_blocks: Set[str]):
        if isinstance(expr, (And, Or)):
            for child in expr.children:
                self._walk_for_placements(child, on_blocks, ori_blocks)
        elif isinstance(expr, Pred):
            if expr.name in ("on_table", "on"):
                on_blocks.add(expr.args[0])
            elif expr.name == "ori_of":
                ori_blocks.add(expr.args[0])

    def _get_explicit_clear(self, expr: Optional[Expr]) -> Set[str]:
        clear_blocks: Set[str] = set()
        if expr is None:
            return clear_blocks
        self._walk_for_clear(expr, clear_blocks)
        return clear_blocks

    def _walk_for_clear(self, expr: Expr, clear_blocks: Set[str]):
        if isinstance(expr, (And, Or)):
            for child in expr.children:
                self._walk_for_clear(child, clear_blocks)
        elif isinstance(expr, Pred):
            if expr.name == "clear":
                clear_blocks.add(expr.args[0])

    def _random_stacking(
        self, blocks: List[str], explicit_clear: Set[str]
    ) -> List[Pred]:
        """Return symbolic placement predicates for unplaced blocks."""
        table = "workspace_table"
        on_rel: Dict[str, str] = {}
        above: Dict[str, List[str]] = {b: [] for b in blocks}
        preds: List[Pred] = []

        shuffled = blocks.copy()
        random.shuffle(shuffled)

        forbid_stack = self.init_spec.forbid_stack

        for b in shuffled:
            if random.random() < 0.4:
                possible_below = [
                    x
                    for x in blocks
                    if x != b
                    and x in on_rel
                    and len(above[x]) == 0
                    and x not in explicit_clear
                    and not (forbid_stack is not None and forbid_stack(b, x))
                ]
                if possible_below:
                    below = random.choice(possible_below)
                    on_rel[b] = below
                    above[below].append(b)
                    continue
            on_rel[b] = table

        for b in blocks:
            if on_rel.get(b) == table:
                preds.append(OnTable(b, table))
            elif b in on_rel:
                preds.append(On(b, on_rel[b]))

        for b in blocks:
            if not above[b] and b not in explicit_clear:
                preds.append(Clear(b))

        return preds

    def _build_pddl(self, active_blocks: List[Block]) -> str:
        problem = self._instantiate(active_blocks)
        pddl = render_problem(problem)
        comments: List[str] = []
        if self.block_size_range is not None:
            comments.append(f"    ;block_size_range = {list(self.block_size_range)}")
        if self.table_height_range is not None:
            comments.append(f"    ;table_height_range = {list(self.table_height_range)}")
        if self.friction_range is not None:
            comments.append(f"    ;friction_range = {list(self.friction_range)}")
        if self.block_x_range is not None:
            comments.append(f"    ;block_x_range = {_range_literal(self.block_x_range)}")
        if self.block_y_range is not None:
            comments.append(f"    ;block_y_range = {_range_literal(self.block_y_range)}")
        if comments:
            pddl = pddl[:-1] + "\n".join([""] + comments) + "\n)"
        return pddl

    def instantiate(self, active_blocks: List[Block] | None = None) -> SymbolicProblem:
        """Return the symbolic problem without rendering PDDL."""
        self._validate_declared_refs()
        if active_blocks is None:
            active_blocks = self._sample_active_blocks()
        return self._instantiate(active_blocks)

    def generate_one(self, out_dir: str | None = None) -> str:
        self._validate_declared_refs()
        active_blocks = self._sample_active_blocks()
        pddl_text = self._build_pddl(active_blocks)

        if out_dir is None:
            out_dir = os.path.join(os.path.dirname(__file__), "..", "Tasks")
        os.makedirs(out_dir, exist_ok=True)

        problem_name = pddl_text.split("(problem ")[1].split(")")[0]
        out_path = os.path.join(out_dir, f"problem_{problem_name}.pddl")

        with open(out_path, "w") as f:
            f.write(pddl_text)
        return out_path

    def generate(
        self,
        count: int = 1,
        out_dir: str | None = None,
        seed: int | None = None,
        verbose: bool = True,
    ) -> List[str]:
        if seed is not None:
            random.seed(seed)

        self._validate_declared_refs()

        paths: List[str] = []
        seen: Set[str] = set()
        max_retries = count * 100

        for i in range(count):
            for _attempt in range(max_retries):
                active_blocks = self._sample_active_blocks()
                pddl_text = self._build_pddl(active_blocks)
                block_names = tuple(sorted(b.name for b in active_blocks))
                fingerprint = (block_names, pddl_text)
                if fingerprint not in seen:
                    seen.add(fingerprint)
                    break
            else:
                if verbose:
                    print(
                        f"Warning: could not generate unique problem {i + 1}/{count} "
                        f"after {max_retries} attempts"
                    )

            if out_dir is None:
                out_dir = os.path.join(os.path.dirname(__file__), "..", "Tasks")
            os.makedirs(out_dir, exist_ok=True)

            problem_name = pddl_text.split("(problem ")[1].split(")")[0]
            out_path = os.path.join(out_dir, f"problem_{problem_name}.pddl")

            with open(out_path, "w") as f:
                f.write(pddl_text)
            paths.append(out_path)

            if verbose:
                print(f"[{i + 1}/{count}] {out_path}")

        if verbose:
            print(f"Generated {len(paths)} problems ({len(seen)} unique)")
        return paths

    def preview(self) -> str:
        self._validate_declared_refs()
        active_blocks = self._sample_active_blocks()
        return self._build_pddl(active_blocks)

    def preview_symbolic(self) -> SymbolicProblem:
        """Return the symbolic problem without rendering PDDL."""
        self._validate_declared_refs()
        return self.instantiate()
