"""Named guards, effects, and moves — the DFA's extension points.

Every capability a YAML spec can name lives here as a registered function. A
strategy file references only these names and their parameters; it never touches
Python. Adding a capability is one decorated function; adding a *strategy* is a
new YAML file.

Three registries, each keyed by the name used in YAML:

* **guards**  ``(ctx, **params) -> bool`` — whether a transition may fire.
* **effects** ``(ctx, **params) -> None`` — mutate the DFA's per-env memory
  (bind a variable, bump a counter) when a transition fires.
* **moves**   ``(ctx, **params) -> ProposedGoal | None`` — turn intent into a
  concrete PDDL goal; ``None`` means "no goal available right now".

:func:`catalog` returns a machine-readable description of all three registries
(names, parameters, defaults, one-line docs) so specs can be validated or
auto-generated.

Argument conventions used below:

* ``var`` names a *memory key* whose value is a block name bound by an earlier
  effect (e.g. ``bind_random_block``).
* block arguments to ``holds`` / ``stack`` / ``next_to`` may be a literal block
  name, or ``@key`` / a memory key, resolved from memory at fire time.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from blocksuite.playdata.dfa.context import DFACtx
from blocksuite.playdata.dfa.ori_flip import FLIP_DIRECTIONS, ori_after_flip, ori_after_rotate
from blocksuite.playdata.goal_proposer import ProposedGoal

GuardFn = Callable[..., bool]
EffectFn = Callable[..., None]
MoveFn = Callable[..., "ProposedGoal | None"]

GUARDS: dict[str, GuardFn] = {}
EFFECTS: dict[str, EffectFn] = {}
MOVES: dict[str, MoveFn] = {}

#: Guard names handled by the engine itself (combinators), not this registry.
COMBINATORS: frozenset[str] = frozenset({"not", "all", "any"})


def guard(name: str) -> Callable[[GuardFn], GuardFn]:
    def deco(fn: GuardFn) -> GuardFn:
        GUARDS[name] = fn
        return fn

    return deco


def effect(name: str) -> Callable[[EffectFn], EffectFn]:
    def deco(fn: EffectFn) -> EffectFn:
        EFFECTS[name] = fn
        return fn

    return deco


def move(name: str) -> Callable[[MoveFn], MoveFn]:
    def deco(fn: MoveFn) -> MoveFn:
        MOVES[name] = fn
        return fn

    return deco


def _norm_glyph(value: Any) -> str:
    return str(value).upper()


def _resolve_block(ctx: DFACtx, key: str) -> str | None:
    """Resolve a block argument: a memory binding if present, else a literal.

    ``@name`` forces a memory lookup (and yields ``None`` if unbound); a bare
    token is returned as-is when it is not a memory key, so both literal block
    names and bound variables work.
    """
    if isinstance(key, str) and key.startswith("@"):
        return ctx.memory.get(key[1:])
    val = ctx.memory.get(key)
    return val if val is not None else key


def _block_with_glyph(ctx: DFACtx, letter: str) -> str | None:
    target = _norm_glyph(letter)
    return next((b for b in ctx.state.blocks if ctx.glyph_of(b) == target), None)


# --------------------------------- guards ---------------------------------


@guard("always")
def _always(ctx: DFACtx) -> bool:  # noqa: ARG001
    """Always fires (the default guard / fall-through edge)."""
    return True


#: Orientations from which the front camera can read a glyph block's letter face.
#: Verified by rendering block_r at all 24 poses (scripts/render_orientations.py):
#: at each of these the letter face points at the camera and is legible.
FOUND_ORIS: tuple[str, ...] = ("o2", "o6", "o10", "o14", "o17", "o18", "o19", "o20")

#: The single strictly-upright, cleanly-readable orientation.
UPRIGHT_ORIS: tuple[str, ...] = ("o2",)


def _letter_at(ctx: DFACtx, letter: str, oris: tuple[str, ...]) -> bool:
    """True when the block whose ASSET is ``letter`` sits at one of ``oris``."""
    target = _block_with_glyph(ctx, letter)
    if target is None:
        return False
    return any(ctx.holds("ori_of", [target, o]) for o in oris)


@guard("found_letter")
def _found_letter(ctx: DFACtx, *, letter: str, oris: Any = FOUND_ORIS) -> bool:
    """The block bearing ``letter`` is at a camera-visible orientation.

    "Found" = the letter face points at the front camera and is legible. The
    default :data:`FOUND_ORIS` is the eight orientations verified by rendering
    the block at all 24 poses; pass ``oris`` to override. Ground-truth
    orientation is used here only as a stand-in for "the camera sees the letter"
    — a termination signal, never an action.
    """
    if isinstance(oris, str):
        oris = (oris,)
    return _letter_at(ctx, letter, tuple(oris))


@guard("upright_letter")
def _upright_letter(ctx: DFACtx, *, letter: str, oris: Any = UPRIGHT_ORIS) -> bool:
    """The block bearing ``letter`` is upright and cleanly readable (default: o2)."""
    if isinstance(oris, str):
        oris = (oris,)
    return _letter_at(ctx, letter, tuple(oris))


@guard("ori_is")
def _ori_is(ctx: DFACtx, *, var: str, o: str) -> bool:
    """The block bound to memory ``var`` is at orientation ``o``."""
    block = ctx.memory.get(var)
    if block is None:
        return False
    return ctx.holds("ori_of", [block, o])


@guard("counter_ge")
def _counter_ge(ctx: DFACtx, *, name: str, n: int) -> bool:
    """Memory counter ``name`` is >= ``n`` (0 if unset)."""
    return int(ctx.memory.get(name, 0)) >= int(n)


@guard("counter_lt")
def _counter_lt(ctx: DFACtx, *, name: str, n: int) -> bool:
    """Memory counter ``name`` is < ``n`` (0 if unset)."""
    return int(ctx.memory.get(name, 0)) < int(n)


@guard("collected_ge")
def _collected_ge(ctx: DFACtx, *, name: str, n: int) -> bool:
    """Memory list ``name`` has at least ``n`` items (for counting distinct finds)."""
    got = ctx.memory.get(name)
    return len(got) >= int(n) if isinstance(got, (list, tuple, set)) else False


@guard("is_bound")
def _is_bound(ctx: DFACtx, *, var: str) -> bool:
    """The memory variable ``var`` holds a block (not ``None``).

    Use after ``bind_random_block`` to detect that no block qualified (e.g. every
    block has already been found or set aside) so the strategy can give up.
    """
    return ctx.memory.get(var) is not None


@guard("block_is_found_letter")
def _block_is_found_letter(ctx: DFACtx, *, var: str, letter: str, oris: Any = FOUND_ORIS) -> bool:
    """The specific block bound to ``var`` is a ``letter`` block now camera-visible.

    Unlike ``found_letter`` (which scans all blocks), this checks the one block the
    strategy is currently manipulating — so it can branch on "did *this* flip reveal
    the target?".
    """
    block = ctx.memory.get(var)
    if block is None or ctx.glyph_of(block) != _norm_glyph(letter):
        return False
    if isinstance(oris, str):
        oris = (oris,)
    result = any(ctx.holds("ori_of", [block, o]) for o in oris)
    _log_found_diag(ctx, block, tuple(oris), result)
    return result


def _log_found_diag(ctx: DFACtx, block: str, oris: tuple[str, ...], result: bool) -> None:
    """DIAG: dump exactly what the found-guard saw for ``block`` so we stop guessing.

    Logs the raw settled quaternion, the strict ``ori_of`` verdict per candidate
    orientation with its (axis-tilt, total-rotation) decomposition, the lenient
    nearest-neighbour label, and whether the state read marked the block tilted.
    """
    import logging

    logger = logging.getLogger(__name__)
    try:
        import math

        import torch

        from blocksuite.playdata.symbolic_state import nearest_orientation
        from blocksuite.vecgen.math import quat_inverse, quat_mul
        from blocksuite.vecgen.symbolic.predicates import ORI_NAMES, quat_angle, quat_z_axis
        from blocksuite.vecgen.symbolic.predicates import ORI_QUATERNIONS as ori_table

        ORI_QUATERNIONS = dict(zip(ORI_NAMES, ori_table.tolist()))

        def quat_error_deg(q_cur: torch.Tensor, q_tgt: torch.Tensor) -> tuple[float, float]:
            """(axis tilt, total rotation) of ``q_cur`` vs ``q_tgt`` in degrees, as check_quat tests."""
            q_err = quat_mul(q_cur[None].float(), quat_inverse(q_tgt[None].float()))
            tilt = torch.acos(quat_z_axis(q_err)[..., 2].clamp(-1.0, 1.0))
            return math.degrees(float(tilt[0])), math.degrees(float(quat_angle(q_err)[0]))

        pose = ctx.pose_of(block)
        if pose is None:
            logger.info(f"[FOUND-DIAG] env {ctx.env_idx} {block}: NO POSE; result={result}")
            return
        quat = torch.tensor(pose[3:7], dtype=torch.float64)
        near_label, near_exact = nearest_orientation(quat)
        snap_label = ctx.state.ori.get(block, "?")
        is_tilted = block in getattr(ctx.state, "tilted", [])

        per_ori = []
        for o in oris:
            ref = torch.tensor(ORI_QUATERNIONS[o], dtype=torch.float64)
            tilt, ang = quat_error_deg(quat, ref)
            hold = ctx.holds("ori_of", [block, o])
            per_ori.append(f"{o}:hold={int(hold)} tilt={tilt:.1f} rot={ang:.1f}")

        logger.info(
            f"[FOUND-DIAG] env {ctx.env_idx} {block} glyph={ctx.glyph_of(block)} "
            f"RESULT={result} | snap={snap_label} nearest={near_label}(exact={near_exact}) "
            f"tilted={is_tilted} | quat(wxyz)=[{quat[0]:.3f},{quat[1]:.3f},"
            f"{quat[2]:.3f},{quat[3]:.3f}] | " + " ".join(per_ori)
        )
    except Exception as exc:  # never let diagnostics break the guard
        logger.info(f"[FOUND-DIAG] env {ctx.env_idx} {block}: diag error {exc!r}; result={result}")


@guard("holds")
def _holds(ctx: DFACtx, *, pred: str, args: list[str]) -> bool:
    """Raw predicate check; block args may be literals or memory keys."""
    resolved = [_resolve_block(ctx, a) for a in args]
    if any(a is None for a in resolved):
        return False
    return ctx.holds(pred, resolved)


# --------------------------------- effects --------------------------------


@effect("bind_random_block")
def _bind_random_block(
    ctx: DFACtx,
    *,
    var: str,
    clear_only: bool = False,
    glyph: Any = None,
    exclude_glyph: Any = None,
    avoid: Any = None,
) -> None:
    """Bind ``var`` to a random block, or ``None`` if none qualify.

    ``avoid`` is one or more memory-list names (e.g. ``[found, sided]``); every
    block already in those lists is excluded, so the search never re-picks a block
    it has already resolved.
    """
    blocks = ctx.state.clear_blocks() if clear_only else list(ctx.state.blocks)
    if glyph is not None:
        want = _norm_glyph(glyph)
        blocks = [b for b in blocks if ctx.glyph_of(b) == want]
    if exclude_glyph is not None:
        skip = _norm_glyph(exclude_glyph)
        blocks = [b for b in blocks if ctx.glyph_of(b) != skip]
    if avoid:
        avoided: set[str] = set()
        for key in ((avoid,) if isinstance(avoid, str) else avoid):
            val = ctx.memory.get(key)
            if isinstance(val, (list, tuple, set)):
                avoided.update(val)
            elif val is not None:
                avoided.add(val)
        blocks = [b for b in blocks if b not in avoided]
    ctx.memory[var] = ctx.rng.choice(blocks) if blocks else None


@effect("bind_letter_block")
def _bind_letter_block(ctx: DFACtx, *, var: str, letter: str) -> None:
    """Bind ``var`` to the block whose asset is ``letter`` (``None`` if absent)."""
    ctx.memory[var] = _block_with_glyph(ctx, letter)


@effect("set")
def _set(ctx: DFACtx, *, name: str, value: Any) -> None:
    """Set memory ``name`` to ``value``."""
    ctx.memory[name] = value


@effect("incr")
def _incr(ctx: DFACtx, *, name: str, by: int = 1) -> None:
    """Increment memory counter ``name`` by ``by`` (default 1)."""
    ctx.memory[name] = int(ctx.memory.get(name, 0)) + int(by)


@effect("reset")
def _reset(ctx: DFACtx, *, name: str, value: Any = 0) -> None:
    """Reset memory ``name`` to ``value`` (default 0)."""
    ctx.memory[name] = value


@effect("list_add")
def _list_add(ctx: DFACtx, *, name: str, var: str) -> None:
    """Append the block bound to ``var`` to memory list ``name`` (deduped)."""
    block = ctx.memory.get(var)
    if block is None:
        return
    lst = ctx.memory.setdefault(name, [])
    if block not in lst:
        lst.append(block)


@effect("record_found_letter")
def _record_found_letter(ctx: DFACtx, *, var: str, letter: str, into: str) -> None:
    """If the block bound to ``var`` is a ``letter`` block now visible to the camera,
    add it to the memory list ``into`` (deduped).

    Cumulative "found" tracking for multi-target searches: once a specific letter
    block has been revealed at any point, it stays counted even if later flipped
    away. Uses ground truth only as a stand-in for perception, never as an action.
    """
    block = ctx.memory.get(var)
    if block is None or ctx.glyph_of(block) != _norm_glyph(letter):
        return
    if not any(ctx.holds("ori_of", [block, o]) for o in FOUND_ORIS):
        return
    found = ctx.memory.setdefault(into, [])
    if block not in found:
        found.append(block)


# ---------------------------------- moves ---------------------------------


@move("flip")
def _flip(ctx: DFACtx, *, var: str, direction: str = "forward", turns: int = 1) -> ProposedGoal | None:
    """Flip the block bound to ``var`` a quarter turn in ``direction``.

    ``direction`` is ``forward``/``back``/``left``/``right``, or ``random`` to
    pick one uniformly per flip (a random-walk over faces, matching "flip the
    block at random to reveal it").
    """
    block = ctx.memory.get(var)
    if block is None:
        return None
    current = ctx.state.ori.get(block)
    if current is None:
        return None
    if direction == "random":
        direction = ctx.rng.choice(FLIP_DIRECTIONS)
    target = ori_after_flip(current, direction, int(turns))
    return ProposedGoal(predicates=[f"(ori_of {block} {target})"], families=["flip"])


@move("rotate")
def _rotate(ctx: DFACtx, *, var: str, turns: int = 1) -> ProposedGoal | None:
    """Spin the block bound to ``var`` ``turns`` quarter turns about Z."""
    block = ctx.memory.get(var)
    if block is None:
        return None
    current = ctx.state.ori.get(block)
    if current is None:
        return None
    target = ori_after_rotate(current, int(turns))
    return ProposedGoal(predicates=[f"(ori_of {block} {target})"], families=["rotate"])


@move("flip_or_rotate")
def _flip_or_rotate(ctx: DFACtx, *, var: str, rotate_prob: float = 0.5) -> ProposedGoal | None:
    """Random-walk the block bound to ``var``: a random flip OR a random rotate.

    With probability ``rotate_prob`` spin it a random number of quarter turns
    about Z; otherwise flip it in a random direction. This is the honest,
    perception-free search primitive — the current ground-truth orientation is
    read only to express the single move as a concrete PDDL goal, never to steer
    toward a target face.
    """
    block = ctx.memory.get(var)
    if block is None:
        return None
    current = ctx.state.ori.get(block)
    if current is None:
        return None
    if ctx.rng.random() < float(rotate_prob):
        target = ori_after_rotate(current, ctx.rng.choice([1, 2, 3]))
        family = "rotate"
    else:
        target = ori_after_flip(current, ctx.rng.choice(FLIP_DIRECTIONS))
        family = "flip"
    return ProposedGoal(predicates=[f"(ori_of {block} {target})"], families=[family])


@move("set_ori")
def _set_ori(ctx: DFACtx, *, var: str, o: str) -> ProposedGoal | None:
    """Reorient the block bound to ``var`` to absolute orientation ``o``."""
    block = ctx.memory.get(var)
    if block is None:
        return None
    return ProposedGoal(predicates=[f"(ori_of {block} {o})"], families=["set_ori"])


@move("on_table")
def _on_table(ctx: DFACtx, *, var: str) -> ProposedGoal | None:
    """Put the block bound to ``var`` back on the table."""
    block = ctx.memory.get(var)
    if block is None:
        return None
    return ProposedGoal(predicates=[f"(on_table {block} {ctx.table})"], families=["on_table"])


@move("move_to_side")
def _move_to_side(ctx: DFACtx, *, var: str) -> ProposedGoal | None:
    """Move the block bound to ``var`` to the side of the table (mark it checked).

    Emits ``(on_table_side B table)`` — the planner uses the analytic ``move_to_side``
    action (no RL policy, so it is safe in cluttered scenes). Pair it with
    ``list_add`` into a "sided" list so the block is not re-picked.
    """
    block = ctx.memory.get(var)
    if block is None:
        return None
    return ProposedGoal(predicates=[f"(on_table_side {block} {ctx.table})"], families=["move_to_side"])


@move("stack")
def _stack(ctx: DFACtx, *, a: str, b: str) -> ProposedGoal | None:
    """Stack block ``a`` on block ``b`` (each a literal or memory key)."""
    ba, bb = _resolve_block(ctx, a), _resolve_block(ctx, b)
    if ba is None or bb is None or ba == bb:
        return None
    return ProposedGoal(predicates=[f"(on {ba} {bb})"], families=["stack"])


@move("next_to")
def _next_to(ctx: DFACtx, *, a: str, b: str) -> ProposedGoal | None:
    """Push block ``a`` next to block ``b`` (each a literal or memory key)."""
    ba, bb = _resolve_block(ctx, a), _resolve_block(ctx, b)
    if ba is None or bb is None or ba == bb:
        return None
    return ProposedGoal(predicates=[f"(next_to {ba} {bb})"], families=["next_to"])


# --------------------------------- catalog --------------------------------


def _describe(fn: Callable[..., Any]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for pname, p in inspect.signature(fn).parameters.items():
        if pname == "ctx":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        params[pname] = None if p.default is inspect.Parameter.empty else p.default
    doc = (fn.__doc__ or "").strip().splitlines()
    return {"params": params, "doc": doc[0] if doc else ""}


def catalog() -> dict[str, Any]:
    """Machine-readable description of every guard/effect/move for spec authoring.

    Returns ``{"guards": {name: {params, doc}}, "effects": {...}, "moves": {...},
    "combinators": [...]}``. Feed this to a spec generator (or an LLM) so it emits
    only registry keys that exist, with the right parameter names.
    """
    return {
        "guards": {name: _describe(fn) for name, fn in sorted(GUARDS.items())},
        "effects": {name: _describe(fn) for name, fn in sorted(EFFECTS.items())},
        "moves": {name: _describe(fn) for name, fn in sorted(MOVES.items())},
        "combinators": sorted(COMBINATORS),
    }
