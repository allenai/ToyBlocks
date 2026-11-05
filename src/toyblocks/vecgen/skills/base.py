"""Skill interface: batched keyframe generators driven by the engine.

A skill is a short list of *phases*. For a batch of envs at the same phase the engine
asks the skill for a :class:`PhaseRequest` (goal poses, scripted rows, or a gripper
command), resolves it with batched IK / planning, and installs the resulting segments.
Skills keep per-env scratch in ``ctx.params[env]``; they never touch the sim per tick.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from toyblocks.vecgen.engine import EngineContext

Kind = Literal["plan", "ik_interp", "gripper", "rows", "hold", "policy"]


@dataclass
class PhaseRequest:
    """What a batch of envs should do in one phase.

    ``plan``: cuRobo transport plan to ``goals`` (``[B, 7]``).
    ``ik_interp``: IK to ``goals`` then ``ticks`` linear joint rows (+ ``hold``).
    ``gripper``: hold the current arm target for ``ticks`` ticks with the gripper
    commanded ``closed``.
    ``rows``: scripted joint rows per env (``rows[i]`` is ``[T_i, dof]``).
    ``hold``: hold the current arm target for ``ticks`` ticks.
    ``policy``: closed-loop control by ``extra["runner"]`` (an :class:`RLRunner`): the
    engine settles ``ticks`` ticks (gripper at ``gripper_target``, gains from
    ``extra["profile"]``), then commands the runner's targets every tick until it
    reports the env done. ``extra["env_data"]`` maps env -> the runner's goal data.

    ``exclude`` ``[B, K]`` drops blocks from the collision world for this phase (intended
    contact / held object); ``pad_fraction`` and ``no_pad`` ``[B, K]`` control the XY
    padding of the remaining blocks; ``ok`` ``[B]`` marks envs whose request could not be
    formed (they fail the skill without touching the sim).
    """

    kind: Kind
    label: str
    goals: Tensor | None = None
    rows: list[Tensor] | None = None
    closed: bool | None = None
    gripper_target: float | None = None  # explicit finger target (overrides ``closed``)
    ticks: int = 0
    hold: int = 0
    # ``rows`` / ``plan``: end ``hold`` early once every joint is within this many radians of
    # the last row (``hold`` is then a cap). None: hold exactly ``hold`` ticks.
    settle_tol: float | None = None
    exclude: Tensor | None = None
    pad_fraction: float = 0.0
    no_pad: Tensor | None = None
    ok: Tensor | None = None
    extra: dict[str, Any] = field(default_factory=dict)


class Skill:
    """Base class; subclasses set ``name`` and ``phases`` and implement the hooks."""

    name: str = ""
    phases: list[str] = []
    #: retry from the live state (no snapshot restore), e.g. a flip that moved the block
    retry_from_live: bool = False
    #: phases whose request() is expensive per env (scripted searches): built on a worker
    #: thread while the sim keeps stepping the other envs
    async_phases: frozenset[str] = frozenset()

    def begin(self, ctx: EngineContext, env_ids: Tensor, args: list[tuple[str, ...]]) -> Tensor:
        """Sample per-env parameters for a fresh attempt; return ``ok [B]``."""
        return torch.ones(int(env_ids.numel()), dtype=torch.bool, device=ctx.device)

    def request(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> PhaseRequest:
        raise NotImplementedError

    def skips(self, ctx: EngineContext, phase: str, env: int) -> bool:
        """True when ``phase`` is a no-op for env ``env`` this attempt (e.g. an optional
        sub-step it does not need): the engine moves past it without a segment, so no tick
        is spent or recorded and neither ``request`` nor ``after_phase`` runs for it."""
        return False

    def after_phase(self, ctx: EngineContext, phase: str, env_ids: Tensor) -> None:
        """Bookkeeping once the phase's segment finished (attach/detach, cached poses)."""

    def verify(self, ctx: EngineContext, env_ids: Tensor) -> Tensor:
        """Effect check on the live state after the last phase; ``ok [B]``."""
        return torch.ones(int(env_ids.numel()), dtype=torch.bool, device=ctx.device)

    def hl_name(self) -> str:
        return self.name
