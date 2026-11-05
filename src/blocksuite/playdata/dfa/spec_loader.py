"""Parse and validate a YAML/JSON DFA strategy into a :class:`DFASpec`.

The grammar is documented in :mod:`blocksuite.playdata.dfa.engine`. This module
turns a spec file (or an already-loaded dict) into typed dataclasses, resolves
``$param`` tokens, and validates it loudly: an unknown registry key, a dangling
``next``, or a missing ``initial`` is an error, not a silent misbehaviour at
run time. Unreachable states are warned about but not fatal.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

from blocksuite.playdata.dfa.engine import (
    TERMINAL,
    DFASpec,
    EffectSpec,
    GuardSpec,
    MoveSpec,
    Transition,
)
from blocksuite.playdata.dfa.registries import COMBINATORS, EFFECTS, GUARDS, MOVES


def load_dfa_spec(path: str | Path, *, overrides: dict[str, Any] | None = None) -> DFASpec:
    """Load and validate a DFA spec from a YAML or JSON file (JSON is valid YAML)."""
    source = Path(path)
    data = yaml.safe_load(source.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"DFA spec {source} must be a mapping at top level")
    return build_dfa_spec(data, overrides=overrides, source=str(source))


def build_dfa_spec(
    data: dict[str, Any], *, overrides: dict[str, Any] | None = None, source: str = "<dict>"
) -> DFASpec:
    """Build a :class:`DFASpec` from a raw mapping (already-loaded YAML/JSON)."""
    params = dict(data.get("params") or {})
    if overrides:
        params.update(overrides)
    data = _resolve_params(data, params)

    name = str(data.get("name", "dfa"))
    initial = data.get("initial")
    if not initial:
        raise ValueError(f"DFA spec {source} is missing required key 'initial'")
    memory0 = dict(data.get("memory") or {})

    raw_states = data.get("states")
    if not isinstance(raw_states, dict) or not raw_states:
        raise ValueError(f"DFA spec {source} is missing a non-empty 'states' mapping")

    states: dict[str, list[Transition]] = {}
    for sname, transitions in raw_states.items():
        if sname in TERMINAL:
            raise ValueError(
                f"DFA spec {source}: state name {sname!r} is reserved (terminal); "
                "remove it from 'states'"
            )
        if not isinstance(transitions, list):
            raise ValueError(
                f"DFA spec {source}: state {sname!r} must map to a list of transitions"
            )
        states[sname] = [_parse_transition(t, sname, source) for t in transitions]

    spec = DFASpec(name=name, initial=str(initial), states=states, memory0=memory0, params=params)
    _validate(spec, source)
    return spec


# -------------------------------- parsing ---------------------------------


def _resolve_params(node: Any, params: dict[str, Any]) -> Any:
    """Recursively replace exact ``$key`` strings with ``params[key]``.

    The ``params`` block itself is left untouched (its values are the source of
    substitution, not targets).
    """
    if isinstance(node, dict):
        return {k: (v if k == "params" else _resolve_params(v, params)) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_params(v, params) for v in node]
    if isinstance(node, str) and node.startswith("$"):
        key = node[1:]
        if key not in params:
            raise KeyError(f"unresolved DFA param ${key} (define it under 'params')")
        return params[key]
    return node


def _parse_named(node: Any, kind: str) -> tuple[str, Any]:
    """A named node is a bare string or a single-key mapping ``{name: params}``."""
    if isinstance(node, str):
        return node, {}
    if isinstance(node, dict):
        if len(node) != 1:
            raise ValueError(f"{kind} node must have exactly one key, got {list(node)}")
        ((name, params),) = node.items()
        return str(name), ({} if params is None else params)
    raise ValueError(f"{kind} node must be a string or single-key mapping, got {node!r}")


def _parse_guard(node: Any) -> GuardSpec:
    name, params = _parse_named(node, "guard")
    if name == "not":
        return GuardSpec("not", children=[_parse_guard(params)])
    if name in ("all", "any"):
        if not isinstance(params, list):
            raise ValueError(f"guard {name!r} takes a list of sub-guards, got {params!r}")
        return GuardSpec(name, children=[_parse_guard(p) for p in params])
    if not isinstance(params, dict):
        raise ValueError(f"guard {name!r} params must be a mapping, got {params!r}")
    return GuardSpec(name, params=params)


def _parse_move(node: Any) -> MoveSpec:
    name, params = _parse_named(node, "move")
    if not isinstance(params, dict):
        raise ValueError(f"move {name!r} params must be a mapping, got {params!r}")
    return MoveSpec(name, params=params)


def _parse_effect(node: Any) -> EffectSpec:
    name, params = _parse_named(node, "effect")
    if not isinstance(params, dict):
        raise ValueError(f"effect {name!r} params must be a mapping, got {params!r}")
    return EffectSpec(name, params=params)


def _parse_transition(t: Any, sname: str, source: str) -> Transition:
    if not isinstance(t, dict):
        raise ValueError(f"DFA spec {source}: transition in state {sname!r} must be a mapping")
    nxt = t.get("next")
    if not nxt:
        raise ValueError(f"DFA spec {source}: transition in state {sname!r} is missing 'next'")
    guard = _parse_guard(t.get("guard", "always"))
    emit = _parse_move(t["emit"]) if t.get("emit") is not None else None
    effects = [_parse_effect(e) for e in (t.get("effects") or [])]
    return Transition(guard=guard, next=str(nxt), emit=emit, effects=effects)


# ------------------------------ validation --------------------------------


def _check_guard_names(g: GuardSpec, source: str) -> None:
    if g.name in COMBINATORS:
        for child in g.children:
            _check_guard_names(child, source)
        return
    if g.name not in GUARDS:
        raise KeyError(f"DFA spec {source}: unknown guard {g.name!r} (available: {sorted(GUARDS)})")


def _validate(spec: DFASpec, source: str) -> None:
    if spec.initial not in spec.states:
        raise ValueError(
            f"DFA spec {source}: initial state {spec.initial!r} is not defined in 'states'"
        )

    valid_targets = set(spec.states) | TERMINAL
    for sname, transitions in spec.states.items():
        for t in transitions:
            if t.next not in valid_targets:
                raise ValueError(
                    f"DFA spec {source}: state {sname!r} transitions to unknown state "
                    f"{t.next!r} (must be a defined state, 'ACCEPT', or 'REJECT')"
                )
            _check_guard_names(t.guard, source)
            if t.emit is not None and t.emit.name not in MOVES:
                raise KeyError(
                    f"DFA spec {source}: unknown move {t.emit.name!r} (available: {sorted(MOVES)})"
                )
            for eff in t.effects:
                if eff.name not in EFFECTS:
                    raise KeyError(
                        f"DFA spec {source}: unknown effect {eff.name!r} "
                        f"(available: {sorted(EFFECTS)})"
                    )

    # Reachability (warn only): BFS from initial over transition targets.
    reachable = {spec.initial}
    frontier = [spec.initial]
    while frontier:
        cur = frontier.pop()
        for t in spec.states.get(cur, []):
            if t.next not in TERMINAL and t.next not in reachable:
                reachable.add(t.next)
                frontier.append(t.next)
    unreachable = sorted(set(spec.states) - reachable)
    if unreachable:
        logger.warning(f"[DFA] {source}: unreachable states never entered: {unreachable}")
