"""The context object every DFA guard, effect, and move reads from.

A guard is a pure function of a :class:`DFACtx`; it never touches the simulator
directly. The orchestrator assembles one context per environment per snapshot
from pieces it already holds, and hands the same object to guards, effects, and
moves. Critically the context carries *asset identity* (:func:`glyph_of`), not
just geometry, so a guard can ask "which block is the letter R?" as well as
"where is it?".
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from toyblocks.playdata.symbolic_state import SymbolicState
from toyblocks.tamp.blocks.glyph_blocks import glyph_usd_subpath
from toyblocks.tamp.blocks.template.codegen import USD_OVERRIDES

HoldsFn = Callable[[str, list[str]], bool]


def glyph_of(block: str) -> str | None:
    """Resolve a block name to the glyph on its USD asset, or ``None``.

    Uses the same name -> asset mapping the rest of the codebase trusts:
    :func:`~toyblocks.tamp.blocks.glyph_blocks.glyph_usd_subpath` for glyph
    blocks (``block_r`` -> ``.../R.usd`` -> ``"R"``) plus the legacy
    ``USD_OVERRIDES`` aliases (``block_o1`` -> ``.../O.usd`` -> ``"O"``). Color
    cuboids and any block without a lettered/numbered asset return ``None``.
    """
    sub = glyph_usd_subpath(block)
    if sub is None:
        override = USD_OVERRIDES.get(block)
        sub = override[0] if override else None
    if sub is None:
        return None
    return Path(sub).stem


@dataclass
class DFACtx:
    """Everything a guard / effect / move is allowed to see for one env."""

    env_idx: int
    state: SymbolicState
    """Current symbolic snapshot: ``.ori[b]`` (snapped o1..o24), ``.on``, ``.blocks``."""
    memory: dict[str, Any]
    """The DFA runtime's per-env memory (bound variables, counters). Mutable —
    effects write here."""
    rng: random.Random
    """Per-env RNG, seeded deterministically from the run seed and env index."""
    holds: HoldsFn
    """Predicate oracle bound to this env (same thresholds that judge goals)."""
    table: str
    """PDDL table object name (for table-relative moves)."""
    glyph_of: Callable[[str], str | None]
    """Block name -> glyph on its USD asset (see :func:`glyph_of`)."""
    pose_of: Callable[[str], list[float] | None]
    """Block name -> raw 7-dof pose, if a guard/move needs geometry directly."""
