"""Build an FF-plannable PDDL problem from the episode's recorded symbolic state.

At eval time the episode queue carries ``pddl_goal`` (a goal body string) plus
the block list and the recorded ``observation.state.high_level_state`` -- the
datagen planner's true-prop set at the first frame.  The client-side TAMP oracle
(:class:`OracleHighLevelPolicy`) needs a complete problem (``:objects`` +
``:init`` + ``:goal``) to run FF.  The first-frame ``high_level_state`` *is* the
exact ``:init`` the datagen planned from (correct ``on`` / ``on_table`` /
``clear`` / ``ori_of`` with the exact orientation labels, plus the static
``up_face`` / ``yaw_next`` / ``flip_next`` tables and ``(initialized)`` the
domain's ``init`` action establishes), so we emit it verbatim rather than
re-deriving ``:init`` geometrically from live poses.

Contract:

- ``:objects`` : every episode block under its domain subtype (color cubes as
  ``block_red`` etc., letter cubes as ``block_r_e``/``block_o_b``/``block_t_g``;
  see :func:`_derive_block_type`), the table ``- table``, the gripper
  ``- gripper``.  Subtyping matters because goals may quantify over a subtype
  (``(exists (?b - block_red) ...)``); a block declared only as the ``block``
  supertype leaves ``block_red`` empty and FF simplifies the goal to FALSE.
  Grounded goals are unaffected (predicates are typed on the ``block`` supertype).
  Orientation labels (``o1..o24``) are domain ``:constants``, so no extra objects
  are added.
- ``:init`` : the literals parsed out of the recorded ``high_level_state``,
  emitted verbatim and indented.  Because the datagen state is post-``init``, it
  already carries ``(initialized)`` + the static tables; FF sees ``(initialized)``
  and skips re-running ``init`` (its ``(not (initialized))`` precondition fails).
- ``:goal`` : the recorded ``pddl_goal`` body, passed through verbatim (wrapped
  in ``(and ...)`` only if it isn't already) to avoid parser round-trip bugs.

Per the eval "fail loud" policy, a missing/empty ``high_level_state`` raises
rather than degrading to a geometrically-guessed problem (see
:func:`~blocksuite.eval.oracle.controller.OracleHLController._build_reconstructed_oracle`).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from blocksuite.utils.episode_dataset import SCENE_FURNITURE_PREFIXES

logger = logging.getLogger(__name__)

# Letter cubes carry a printed glyph and are declared under a composite domain
# subtype rather than a color type (see TAMP ``data_gen/task_gen/blocks.py``
# BLOCK_TYPE_MAP). Their names don't encode the subtype, so map them explicitly;
# every other block is a color cube whose subtype is its name minus the trailing
# instance digits (``block_red1`` -> ``block_red``).
_LETTER_BLOCK_TYPES = {
    "block_r": "block_r_e",
    "block_o1": "block_o_b",
    "block_o2": "block_o_b",
    "block_b": "block_o_b",
    "block_t": "block_t_g",
}


class ProblemReconstructionError(RuntimeError):
    """Raised when the reset state can't be turned into a valid PDDL problem."""


def parse_block_subtypes(domain_file: str) -> frozenset[str]:
    """Return every type declared as a subtype of ``block`` in ``domain_file``.

    The oracle's :objects must declare each cube under its true subtype so that
    typed goal quantifiers (e.g. ``(exists (?b - block_red) ...)``) ground; a
    block declared only as the ``block`` supertype leaves ``block_red`` empty and
    FF simplifies the goal to FALSE. We validate derived subtypes against this set
    so a name that doesn't match a real domain type falls back to ``block`` rather
    than emitting an undefined type. Raises on an unreadable/typeless domain.
    """
    text = Path(domain_file).read_text()
    m = re.search(r"\(:types\b(.*?)\)", text, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ProblemReconstructionError(
            f"domain {domain_file} has no (:types ...) section"
        )
    # PDDL typing lists names then ``- parent``; collect names whose parent is
    # ``block`` (the manipulable-cube supertype).
    tokens = m.group(1).split()
    subtypes: set[str] = {"block"}
    pending: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i] == "-":
            parent = tokens[i + 1] if i + 1 < len(tokens) else ""
            if parent == "block":
                subtypes.update(pending)
            pending = []
            i += 2
            continue
        pending.append(tokens[i])
        i += 1
    return frozenset(subtypes)


def _derive_block_type(name: str, valid_subtypes: frozenset[str]) -> str:
    """Best PDDL subtype for a cube name, validated against the domain's types.

    Letter cubes use the explicit :data:`_LETTER_BLOCK_TYPES` map; color cubes use
    the name with trailing instance digits stripped (``block_red1`` ->
    ``block_red``). Anything that doesn't resolve to a declared subtype falls back
    to ``block`` (still correct for grounded goals; only typed quantifiers need
    the subtype).
    """
    mapped = _LETTER_BLOCK_TYPES.get(name)
    if mapped is not None and mapped in valid_subtypes:
        return mapped
    stripped = name.rstrip("0123456789")
    if stripped != "block" and stripped in valid_subtypes:
        return stripped
    return "block"


def filter_block_names(names: Iterable[str], *, table_name: str = "workspace_table") -> list[str]:
    """Keep manipulable blocks; drop the table and any scene furniture.

    ``workspace_table`` is itself an entry in ``scene.rigid_objects`` (the scene
    preset spawns it), so a naive "iterate rigid_objects" would treat it as a
    block.  Blocks are always named ``block_*``; everything matched by
    :data:`SCENE_FURNITURE_PREFIXES` (or equal to ``table_name``) is furniture.
    """
    out: list[str] = []
    for n in names:
        if n == table_name or n.startswith(SCENE_FURNITURE_PREFIXES):
            continue
        out.append(n)
    if not out:
        raise ProblemReconstructionError(
            f"no manipulable blocks left after filtering furniture from {list(names)!r} "
            f"(table_name={table_name!r})"
        )
    return out


def _wrap_goal_body(pddl_goal: str) -> str:
    """Return a ``(and ...)``-wrapped goal body, tolerant of the queue's shapes."""
    body = (pddl_goal or "").strip()
    if not body:
        raise ProblemReconstructionError("pddl_goal is empty; nothing to plan toward")
    # Accept a bare literal ``(on a b)`` or an already-conjoined ``(and ...)``.
    head = body[1:].lstrip().lower() if body.startswith("(") else ""
    if head.startswith("and") or head.startswith("or") or head.startswith("not"):
        return body
    return f"(and {body})"


def _split_sexprs(text: str) -> list[str]:
    """Split a run of flat PDDL literals into individual ``(pred args...)`` strings.

    The datagen records ``high_level_state`` as ``"\\n".join(sorted(props))`` where
    each prop is a flat literal like ``(on_table block_r workspace_table)``.  We
    parse by tracking paren depth rather than splitting on newlines so the reader
    is robust to whitespace variation (space- or newline-separated, stray indent).
    Raises :class:`ProblemReconstructionError` on unbalanced parens so a corrupt
    recording fails loudly instead of yielding a truncated ``:init``.
    """
    exprs: list[str] = []
    depth = 0
    start = -1
    for i, ch in enumerate(text):
        if ch == "(":
            if depth == 0:
                start = i
            depth += 1
        elif ch == ")":
            if depth == 0:
                raise ProblemReconstructionError(
                    f"high_level_state has an unmatched ')' at index {i}: {text!r}"
                )
            depth -= 1
            if depth == 0:
                exprs.append(text[start : i + 1])
    if depth != 0:
        raise ProblemReconstructionError(
            f"high_level_state has {depth} unclosed '(': {text!r}"
        )
    if not exprs:
        raise ProblemReconstructionError(
            f"high_level_state has no PDDL literals: {text!r}"
        )
    return exprs


def _assert_init_objects_declared(
    init_exprs: Sequence[str],
    block_objs: Sequence[str],
    table_name: str,
    gripper_name: str,
) -> None:
    """Raise unless every block named in ``:init`` is also declared in ``:objects``.

    ``:init`` is the recorded ``high_level_state`` verbatim, while ``:objects``
    comes from the episode's ``setup["objects"]`` keys -- and ``episode_dataset``
    drops blocks with empty poses. The two can therefore disagree, and FF given a
    literal over an undeclared object either errors obscurely or drops it,
    yielding a plan for a different problem than the recording describes.

    Orientation labels (``o1..o24``) are domain ``:constants`` and predicate names
    are not objects, so only ``block_*`` tokens are checked; that is exactly the
    set ``:objects`` is responsible for.
    """
    declared = set(block_objs) | {table_name, gripper_name}
    referenced: set[str] = set()
    for expr in init_exprs:
        # Literals are flat ``(pred arg...)``, so tokens are all we need.
        for token in expr.strip("()").split():
            if token.startswith("block_"):
                referenced.add(token)
    undeclared = sorted(referenced - declared)
    if undeclared:
        raise ProblemReconstructionError(
            f"recorded high_level_state references block(s) {undeclared} that are not "
            f"in the episode's object list {sorted(block_objs)}. The reconstructed "
            ":init would describe blocks FF has no declaration for, so it would plan a "
            "different problem than the recording. This usually means the episode's "
            "recorded object poses are incomplete; regenerate the dataset so every "
            "block in high_level_state also has a recorded pose."
        )


def build_problem_from_recorded_state(
    *,
    blocks: Sequence[str],
    high_level_state: str,
    pddl_goal: str,
    valid_block_subtypes: Optional[frozenset[str]] = None,
    table_name: str = "workspace_table",
    gripper_name: str = "robot",
    domain_name: str = "blocks-tamp",
    problem_name: str = "reconstructed",
) -> str:
    """Build a full PDDL problem string from the recorded first-frame HL state.

    ``:init`` is the recorded ``high_level_state`` literals emitted verbatim (the
    exact set the datagen planned from); ``:objects`` declares each episode block
    under its domain subtype (see :func:`_derive_block_type`) so both grounded and
    typed-quantifier goals plan, plus the table and gripper; ``:goal`` is the
    recorded ``pddl_goal`` body.  ``valid_block_subtypes`` is the domain's declared
    ``block`` subtypes (from :func:`parse_block_subtypes`); when ``None`` every
    block is declared as the ``block`` supertype.  Raises
    :class:`ProblemReconstructionError` on an empty block list, empty/corrupt
    ``high_level_state``, or empty ``pddl_goal``.
    """
    block_objs = filter_block_names(blocks, table_name=table_name)

    state = (high_level_state or "").strip()
    if not state:
        raise ProblemReconstructionError(
            "high_level_state is empty; cannot build :init from the recording"
        )
    init_exprs = _split_sexprs(state)
    init_lines = [f"        {expr}" for expr in init_exprs]

    _assert_init_objects_declared(init_exprs, block_objs, table_name, gripper_name)

    subtypes = valid_block_subtypes if valid_block_subtypes is not None else frozenset({"block"})
    type_groups: dict[str, list[str]] = {}
    for b in block_objs:
        type_groups.setdefault(_derive_block_type(b, subtypes), []).append(b)
    object_lines = [
        f"    {' '.join(sorted(names))} - {btype}"
        for btype, names in sorted(type_groups.items())
    ]
    object_lines.append(f"    {table_name} - table")
    object_lines.append(f"    {gripper_name} - gripper")
    objects_line = "\n".join(object_lines)
    goal_body = _wrap_goal_body(pddl_goal)

    return (
        f"(define (problem {problem_name})\n"
        f"    (:domain {domain_name})\n"
        f"    (:objects\n{objects_line}\n    )\n"
        f"    (:init\n" + "\n".join(init_lines) + "\n    )\n"
        f"    (:goal {goal_body})\n"
        f")\n"
    )


# ``act`` is the oracle's generic stand-in for an HL action it can't map to a
# pick/place/flip/rotate (e.g. push).  A plan step or recording that reduces to it
# carries no identity to compare, so :func:`assert_oracle_consistent_with_recording`
# warns instead of failing.
_UNCHECKABLE_ORACLE_VERBS = frozenset({"act"})


def _plan_action_name(step: Any) -> str:
    """TAMP action name of a ``get_plan_summary()`` entry (e.g. ``pnp_table_block``)."""
    if not isinstance(step, dict):
        raise ProblemReconstructionError(
            f"oracle plan step is not a dict: {step!r}; cannot read its action name."
        )
    return str(step.get("action") or "").strip().lower()


def assert_oracle_consistent_with_recording(
    oracle: Any,
    recorded_high_level_action: Optional[str],
    *,
    problem_path: Optional[str] = None,
) -> None:
    """Gate a freshly built oracle against the recorded first HL instruction.

    The oracle blindly trusts FF's plan derived from ``:init``; a syntactically
    valid but semantically wrong ``:init`` silently makes FF plan a *different*
    task.  The recorded ``high_level_action`` is the ground-truth first step, so
    we confirm (a) FF produced an actionable plan and (b) its first step matches
    the recording.  Raises :class:`ProblemReconstructionError` on failure -- never
    degrades silently.

    This reads ``get_plan_summary()`` rather than calling
    ``get_current_high_level_action()``, which is not read-only: it ticks the
    per-env state machine (``_update_phase`` -> ``pick_phase_steps`` /
    ``_advance``).  Gating with it therefore consumed part of the oracle's first
    instruction before the LL ever saw it, and could advance the plan outright
    when step 0's effects already held at reset.

    Comparison is at plan level, not verb level: the recorded
    ``high_level_action`` is a coarse TAMP action name (``pnp_table_block``) while
    the emitted verbs are its fine-grained expansion (``pick``/``place``), so the
    two are only comparable as plan-step action names.
    """
    try:
        plan = oracle.get_plan_summary()
    except Exception as e:  # planner/runtime failure -> loud
        raise ProblemReconstructionError(
            f"oracle failed to produce a plan summary (problem={problem_path}): {e}"
        ) from e

    if not plan:
        raise ProblemReconstructionError(
            f"reconstructed problem yields an empty plan (problem={problem_path}). FF "
            "found the goal already satisfied or failed to plan, so there is no "
            "instruction to condition the LL on."
        )

    first_action = _plan_action_name(plan[0])
    if not first_action:
        raise ProblemReconstructionError(
            f"oracle plan step 0 has no action name (step={plan[0]!r}, "
            f"problem={problem_path})."
        )

    recorded = (recorded_high_level_action or "").strip()
    if not recorded:
        raise ProblemReconstructionError(
            f"episode has no recorded high_level_action to gate the reconstructed "
            f"problem against (plan step 0 = {first_action!r}, problem={problem_path}). "
            "Without it a semantically wrong :init cannot be detected and the oracle "
            "would drive the LL toward a different task. Regenerate the dataset "
            "recording observation.state.high_level_action."
        )

    # Both sides are TAMP action names, so compare them as-is. Running the
    # oracle *language* reduction (parse_oracle_action + first token) over an
    # action name is a category error: a no-op for today's names, but it would
    # silently truncate any name containing a space or a coordinate pair.
    recorded_action = recorded.lower()
    if first_action in _UNCHECKABLE_ORACLE_VERBS or recorded_action in _UNCHECKABLE_ORACLE_VERBS:
        logger.warning(
            "[OracleHL] first plan action not strictly checkable "
            "(oracle=%r vs recorded=%r); accepting.", first_action, recorded_action,
        )
        return

    # Datasets not planned by FF (vecgen) can solve the same task differently, e.g.
    # unstack to the table first where FF moves block-to-block. Accept a first step
    # from the same skill family (pnp_* variants differ only in source/target
    # support) or one that FF still plans later; fail on any other mismatch.
    if first_action != recorded_action and (
        first_action.split("_")[0] == recorded_action.split("_")[0]
        or any(_plan_action_name(step) == recorded_action for step in plan[1:])
    ):
        logger.warning(
            "[OracleHL] plan step 0 %r differs from the recorded %r but is consistent "
            "(same skill family or planned later); accepting (problem=%s).",
            first_action, recorded_action, problem_path,
        )
        return

    if first_action != recorded_action:
        raise ProblemReconstructionError(
            f"reconstructed problem plans {first_action!r} as its first step but the "
            f"episode recorded {recorded_action!r} (problem={problem_path}). The "
            "reconstructed :init describes a different world than the one datagen "
            "planned from, so the oracle would drive the LL toward a different task."
        )
    logger.info(
        "[OracleHL] reconstruction gate OK: plan step 0 %r matches the recording.",
        first_action,
    )
