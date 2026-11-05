"""Block and Distractor declarations for declarative PDDL generation."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional

# Block type inference from name
BLOCK_TYPE_MAP = {
    "block_r": "block_r_e",
    "block_o1": "block_o_b",
    "block_o2": "block_o_b",
    "block_b": "block_o_b",
    "block_t": "block_t_g",
}


@dataclass
class Block:
    """A fixed block that is always present in the environment.

    Args:
        name: PDDL object name (e.g. "block_r", "target_obj").
        block_type: PDDL type. Defaults to "block" unless the name matches
            a known type (block_r_e, block_o_b, block_t_g).
    """
    name: str
    block_type: str = ""

    def __post_init__(self):
        if not self.block_type:
            self.block_type = BLOCK_TYPE_MAP.get(self.name, "block")


@dataclass
class Distractor:
    """An optional distractor block that may or may not appear.

    Args:
        name: PDDL object name.
        block_type: PDDL type (defaults same as Block).
        prob: Probability [0, 1] that this distractor is included in any
            given problem. Default 1.0 (always included).
        group: Optional group name. When set, at most `group_max` distractors
            from the same group will be sampled. Use this when you have a pool
            of interchangeable distractors but want at most N.
    """
    name: str
    block_type: str = ""
    prob: float = 1.0
    group: Optional[str] = None

    def __post_init__(self):
        if not self.block_type:
            self.block_type = BLOCK_TYPE_MAP.get(self.name, "block")
        if not 0.0 <= self.prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {self.prob}")


@dataclass
class DistractorPool:
    """A pool of interchangeable distractors with min/max sampling.

    Instead of individual probabilities, sample k distractors uniformly from
    the pool where k is in [count_min, count_max].

    Args:
        blocks: List of (name, block_type) pairs in the pool.
        count_min: Minimum number to sample per problem.
        count_max: Maximum number to sample per problem.
    """
    blocks: List[Block] = field(default_factory=list)
    count_min: int = 0
    count_max: int = 0

    def __post_init__(self):
        if self.count_max < self.count_min:
            raise ValueError("count_max must be >= count_min")
        if self.count_min < 0:
            raise ValueError("count_min must be non-negative")

    def add(self, name: str, block_type: str = "block") -> "DistractorPool":
        self.blocks.append(Block(name, block_type))
        return self

    def sample(self) -> List[Block]:
        """Sample a random subset of blocks from the pool."""
        max_k = min(self.count_max, len(self.blocks))
        min_k = min(self.count_min, max_k)
        k = random.randint(min_k, max_k)
        return random.sample(self.blocks, k)
