"""Skill programs and per-env program state.

A *program* is the per-env output of the symbolic planner: an ordered list of skill
calls such as ``pick(block_a)``, ``place(block_a, block_b)``, ``flip(block_c, o2)``.
The engine advances each env through its program one skill at a time; everything in
this module is plain data and is only touched at skill boundaries, never per tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


@dataclass(frozen=True)
class SkillCall:
    """One skill call: the skill name and its object arguments, plus the PDDL action it
    came from (recorded as ``action.high_level_action`` / ``high_level_args``) and the
    symbolic state that action starts from (``observation.state.high_level_state``)."""

    name: str
    args: tuple[str, ...]
    hl_name: str = ""
    hl_args: tuple[str, ...] = ()
    hl_state: tuple[str, ...] = field(default=(), compare=False, repr=False)

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


@dataclass
class Program:
    """The skill sequence an env must execute to reach its goal."""

    skills: list[SkillCall]
    problem_id: str
    reset_state_id: int

    def __len__(self) -> int:
        return len(self.skills)


class Phase(IntEnum):
    """Where an env is inside its program."""

    IDLE = 0  # no program assigned
    READY = 1  # at a skill boundary, needs keyframes/IK for skills[pc]
    EXECUTING = 2  # a segment table is installed and running
    VERIFY = 3  # segment finished, effect check pending
    SUCCESS = 4  # program finished and goal verified
    FAILURE = 5  # gave up on this episode
    WAITING = 6  # a worker thread is building this env's next segment


@dataclass
class EnvProgramState:
    """Mutable per-env bookkeeping for the engine."""

    env_id: int
    program: Program | None = None
    pc: int = 0
    attempts: int = 0
    phase: Phase = Phase.IDLE
    episode_id: int = -1
    failure_reason: str = ""
    sampled: dict[str, float] = field(default_factory=dict)
    replans: int = 0

    @property
    def current(self) -> SkillCall | None:
        if self.program is None or self.pc >= len(self.program):
            return None
        return self.program.skills[self.pc]

    def assign(self, program: Program, episode_id: int) -> None:
        self.program = program
        self.pc = 0
        self.attempts = 0
        self.phase = Phase.READY
        self.episode_id = episode_id
        self.failure_reason = ""
        self.sampled = {}
        self.replans = 0

    def advance(self) -> None:
        """Current skill verified: move to the next one (or finish)."""
        self.pc += 1
        self.attempts = 0
        self.sampled = {}
        self.phase = Phase.SUCCESS if self.pc >= len(self.program or []) else Phase.READY

    def retry(self, max_attempts: int, reason: str) -> bool:
        """Current skill failed: retry from the pre-skill snapshot or give up."""
        self.attempts += 1
        if self.attempts >= max_attempts:
            self.phase = Phase.FAILURE
            self.failure_reason = reason
            return False
        self.phase = Phase.READY
        return True

    def clear(self) -> None:
        self.program = None
        self.pc = 0
        self.attempts = 0
        self.phase = Phase.IDLE
        self.episode_id = -1
        self.failure_reason = ""
        self.sampled = {}
        self.replans = 0
