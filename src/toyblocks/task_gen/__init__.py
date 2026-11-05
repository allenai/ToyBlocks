"""Declarative PDDL problem generation for the Blocks domain.

Example usage (equivalent to pnp_flip_back.yaml):

    from test_domains.Blocks.task_gen import (
        ProblemSpec, Block, Distractor, And, Or, Pred,
        OnTable, On, Clear, OriOf, HandEmpty,
    )

    spec = ProblemSpec(domain="blocks-tamp")

    # Fixed blocks always present
    spec.add_block(Block("block_r", block_type="block_r_e"))
    spec.add_block(Block("target_obj"))

    # Distractors: randomly sample 1-3 from this pool per problem
    spec.add_distractor(Distractor("distractor1", prob=0.7))
    spec.add_distractor(Distractor("distractor2", prob=0.7))
    spec.add_distractor(Distractor("distractor3", prob=0.7))

    # Init state
    spec.init = And(
        OnTable("block_r"),
        Clear("block_r"),
        Clear("target_obj"),
        OriOf("block_r", "o21"),
        HandEmpty(),
    )

    # Goal state
    spec.goal = And(
        On("block_r", "target_obj"),
        Clear("block_r"),
        Or(OriOf("block_r", "o2"), OriOf("block_r", "o4")),
    )

    # Generate 50 unique PDDL files
    spec.generate(count=50, out_dir="./tasks", seed=42)
"""

from toyblocks.task_gen.blocks import Block, Distractor, DistractorPool
from toyblocks.task_gen.materialize import materialize
from toyblocks.task_gen.predicates import (
    And,
    Clear,
    Exists,
    GripperAtInit,
    HandEmpty,
    NextTo,
    On,
    OnRight,
    OnTable,
    Or,
    OriOf,
    Pred,
    Raw,
)
from toyblocks.task_gen.problem import SymbolicProblem, render_problem
from toyblocks.task_gen.problem_spec import ProblemSpec
from toyblocks.task_gen.resolved_init import ResolvedInit

__all__ = [
    "ProblemSpec",
    "Block",
    "Distractor",
    "DistractorPool",
    "Pred",
    "And",
    "Or",
    "Exists",
    "OnTable",
    "On",
    "OnRight",
    "NextTo",
    "Clear",
    "OriOf",
    "ResolvedInit",
    "SymbolicProblem",
    "materialize",
    "render_problem",
    "HandEmpty",
    "GripperAtInit",
    "Raw",
]
