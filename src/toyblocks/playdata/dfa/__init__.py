"""DFA-driven goal policies for play-data collection.

A DFA strategy is a declarative finite-state machine (see :mod:`.engine` for the
spec grammar) that decides, per environment, what goal to pursue next — replacing
random goal sampling with a stateful, reactive strategy such as "flip blocks
until the letter R faces the camera".

Public surface:

* :class:`.engine.DFAGoalPolicy` — the policy the orchestrator drives.
* :func:`.spec_loader.load_dfa_spec` — YAML/JSON -> validated :class:`.engine.DFASpec`.
* :func:`.registries.catalog` — machine-readable guard/effect/move listing for
  authoring or auto-generating specs.
* :func:`.context.glyph_of` — block name -> glyph on its USD asset.
"""

from __future__ import annotations

from toyblocks.playdata.dfa.context import DFACtx, glyph_of
from toyblocks.playdata.dfa.engine import (
    ACCEPT,
    REJECT,
    DFAGoalPolicy,
    DFARuntime,
    DFASpec,
)
from toyblocks.playdata.dfa.registries import catalog
from toyblocks.playdata.dfa.spec_loader import build_dfa_spec, load_dfa_spec

__all__ = [
    "ACCEPT",
    "REJECT",
    "DFACtx",
    "DFAGoalPolicy",
    "DFASpec",
    "DFARuntime",
    "build_dfa_spec",
    "catalog",
    "glyph_of",
    "load_dfa_spec",
]
