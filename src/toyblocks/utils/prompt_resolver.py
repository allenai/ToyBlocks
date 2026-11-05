"""Per-episode prompt resolution for queue-driven eval.

:class:`PromptResolver` tracks the current per-env prompt. Maniflow
transforms call :meth:`PromptResolver.prompt_for_env` so each env's
``task_name`` matches that env's queue setup; :meth:`PromptResolver.current_prompt`
is the batched/droid path (one prompt for the whole batch).
``EpisodeQueueClient.fetch_reset_states`` calls :meth:`PromptResolver.update_setups`
after every reset.

:func:`pddl_goal_to_prompt` converts a PDDL goal expression into a
natural-language instruction. It handles numbered colour cuboids
(``block_red4`` / ``block_blue3`` → "the red block" / "the blue block" when
only one of that colour appears in the goal), letter blocks (``block_r`` →
"the R block"),
stacking chains, and spatial relations.  ``ori_of`` predicates are
currently skipped until a block face annotation catalog is available.
:func:`parse_prompt_for_setup` prefers ``setup["hl_prompt"]`` (parquet
``goal.hl_prompt``) when present; otherwise it converts ``setup["pddl_goal"]``
via :func:`pddl_goal_to_prompt`.

TODO: :func:`_collapse_typed_stack` is a deliberate hack that re-derives
stack structure from the rendered quantified goal so same-type stacks read as
"stack 2 red blocks". It is a stopgap until PDDL-goal-to-English is unified
(ideally one canonical implementation shared with TAMP's ``goal_spec``).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

from toyblocks.domain_randomization.block_colors import _COLOR_TOKEN_RE
from toyblocks.tamp.blocks.glyph_blocks import _GLYPH_BLOCK_PATTERN

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Block-name classification and English noun phrases
# ---------------------------------------------------------------------------

# PDDL block name → English noun phrase. codegen.py derives block colors/types
# from the name itself; add an entry here when a new name needs a nicer phrase.
_BLOCK_PHRASES: dict[str, str] = {
    "block_red1": "the red block",
    "block_red2": "the dark red block",
    "block_green": "the green block",
    "block_blue": "the blue block",
    "block_yellow": "the yellow block",
    "block_white": "the white block",
    "block_gray": "the gray block",
    "block_pink": "the pink block",
    "block_purple": "the purple block",
    "block_r": "the R block",
    "block_o1": "the O block",  # they're the same? slightly unclear
    "block_o2": "the O block",
    "block_t": "the T block",
    "block_b": "the B block",
}


_BLOCK_TOKEN_IN_GOAL_RE = re.compile(r"\bblock_[a-z][a-z0-9]*(?:_\d+)?\b", re.IGNORECASE)


def _glyph_phrase_for_name(pddl_name: str) -> str | None:
    """English for a letter/digit glyph block, or ``None`` if not a glyph name."""
    match = _GLYPH_BLOCK_PATTERN.match(pddl_name.strip())
    if match is None:
        return None
    return f"the {match.group(1).upper()} block"


def _parse_color_cuboid_token(pddl_name: str) -> tuple[str, int] | None:
    """``(color_word, index)`` for cuboid tokens like ``block_blue3``; ``None`` for glyphs."""
    if _GLYPH_BLOCK_PATTERN.match(pddl_name.strip()) is not None:
        return None
    m = _COLOR_TOKEN_RE.match(pddl_name.strip().lower())
    if m is None:
        return None
    color = m.group(1)
    if len(color) < 2:
        return None
    index = int(m.group(2) or "1")
    return color, index


def _color_cuboid_sort_key(name: str) -> tuple[int, str]:
    """Order ``block_blue10`` after ``block_blue2`` (numeric suffix, not lexicographic)."""
    parsed = _parse_color_cuboid_token(name)
    if parsed is not None:
        return (parsed[1], name.lower())
    return (0, name.lower())


def _color_cuboid_tokens_by_color(pddl_goal: str) -> dict[str, list[str]]:
    """Group color-cuboid object names in a goal by colour word (``blue``, ``red``, …)."""
    by_color: dict[str, list[str]] = {}
    seen: dict[str, None] = {}
    for match in _BLOCK_TOKEN_IN_GOAL_RE.finditer(pddl_goal):
        name = match.group(0).lower()
        if name in seen:
            continue
        seen[name] = None
        parsed = _parse_color_cuboid_token(name)
        if parsed is None:
            continue
        color, _index = parsed
        by_color.setdefault(color, []).append(name)
    for color in by_color:
        by_color[color] = sorted(by_color[color], key=_color_cuboid_sort_key)
    return by_color


def _numbered_color_phrase_map(color: str, names: list[str]) -> dict[str, str]:
    """Map numbered cuboids of one colour in a goal to contrasting English phrases.

    One cube of that colour in the goal → ``"the <color> block"`` regardless of
    index (``block_red4`` alone is not "RED4").  Two or more → shade labels by
    sorted index, same convention as the legacy red-only path.
    """
    ordered = sorted(set(names), key=_color_cuboid_sort_key)
    if not ordered:
        return {}
    out: dict[str, str] = {}
    for i, name in enumerate(ordered):
        key = name.lower()
        if i == 0:
            out[key] = f"the {color} block"
        elif i == 1:
            out[key] = f"the dark {color} block"
        elif i == 2:
            out[key] = f"the light {color} block"
        else:
            out[key] = (
                f"the lighter {color} block"
                if (i - 3) % 2 == 0
                else f"the darker {color} block"
            )
    return out


def _object_phrases_for_goal(pddl_goal: str) -> dict[str, str]:
    """Per-goal overrides for numbered colour cuboids (all colours, not only red)."""
    out: dict[str, str] = {}
    for color, names in _color_cuboid_tokens_by_color(pddl_goal).items():
        out.update(_numbered_color_phrase_map(color, names))
    return out


def _object_phrase(
    pddl_name: str,
    object_phrases: dict[str, str] | None = None,
) -> str:
    """Convert a PDDL block name to a natural English noun phrase."""
    key = pddl_name.strip().lower()
    if object_phrases and key in object_phrases:
        return object_phrases[key]
    if pddl_name in _BLOCK_PHRASES:
        return _BLOCK_PHRASES[pddl_name]
    if key in _BLOCK_PHRASES:
        return _BLOCK_PHRASES[key]
    glyph = _glyph_phrase_for_name(key)
    if glyph is not None:
        return glyph
    parsed = _parse_color_cuboid_token(key)
    if parsed is not None:
        color, _index = parsed
        return f"the {color} block"
    if pddl_name.startswith("block_") or key.startswith("block_"):
        suffix = pddl_name[6:] if pddl_name.startswith("block_") else key[6:]
        return f"the {suffix.upper()} block"
    return pddl_name.replace("_", " ")


# PDDL block *type* → bare noun (no article), used to describe quantified goals
# like ``(exists (?b1 ?b2 - block_red) ...)`` where the goal names a type rather
# than a concrete block.  Colors read naturally lowercased; letter/pattern types
# fall back to an uppercased suffix (``block_r`` → "R block").
_TYPE_NOUNS: dict[str, str] = {
    "block": "block",
    "block_red": "red block",
    "block_red1": "red block",
    "block_red2": "dark red block",
    "block_green": "green block",
    "block_blue": "blue block",
    "block_yellow": "yellow block",
    "block_white": "white block",
    "block_gray": "gray block",
    "block_pink": "pink block",
    "block_purple": "purple block",
}


def _type_noun(type_name: str) -> str:
    """Bare noun (no article) for a PDDL block type used in a quantifier."""
    t = type_name.lower()
    if t in _TYPE_NOUNS:
        return _TYPE_NOUNS[t]
    if t.startswith("block_"):
        return f"{t[6:].upper()} block"
    return t.replace("_", " ")


# ---------------------------------------------------------------------------
# Orientation map — parsed from domain.pddl
# ---------------------------------------------------------------------------
# TODO: Orientation-to-face resolution is disabled until a block face
# annotation catalog (block_catalog.json) exists that maps die-face constants
# (one–six) to the actual symbols painted on each block's faces.  Without
# that catalog, orientation prompts produce meaningless descriptions like
# "face 1 up" instead of grounded labels like "the R face up."
# See the block randomizer / catalog request for details.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Spatial relation phrases
# ---------------------------------------------------------------------------

_SPATIAL_PHRASES: dict[str, str] = {
    "on_right": "to the right of",
    "on_left": "to the left of",
    "on_front": "in front of",
    "on_back": "behind",
    "next_to": "next to",
}

# ---------------------------------------------------------------------------
# PDDL goal parsing (lightweight — no isaaclab dependency)
# ---------------------------------------------------------------------------

_PDDL_IDENT = r"(?:\?[\w-]+|[\w-]+)"
_LEAF_PRED_RE = re.compile(rf"\((\w+)\s+((?:{_PDDL_IDENT})(?:\s+{_PDDL_IDENT})*)\)")
# Bare verb for queue-exhausted / never-assigned env slots (masked in eval).
IDLE_PROMPT_PLACEHOLDER = "pick"
_NOT_PICK_OBSTRUCTS_RE = re.compile(
    r"\(\s*not\s+\(\s*pick_obstructs\s+([\w-]+)\s+([\w-]+)\s*\)\s*\)",
    re.IGNORECASE,
)
# A top-level ``(or ...)`` of grounded literals; no nested connectives inside.
_OR_GROUP_RE = re.compile(r"\(\s*or\s+((?:\(\w+\s+[^()]+\)\s*)+)\)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Quantified-goal handling (exists/forall over typed vars)
# ---------------------------------------------------------------------------
#
# The flat regex parser below only handles a conjunction of *grounded* literals.
# Datagen also emits quantified, typed goals, e.g.
# ``(exists (?b1 ?b2 - block_red) (and (not (= ?b1 ?b2)) (on ?b1 ?b2)))``.
# Feeding those to the flat parser yields raw ``?b1``/``?b2`` in the prompt, so
# quantified goals are routed through a small s-expression walker instead.

_QUANTIFIERS = frozenset({"exists", "forall"})


def _has_quantifier_or_var(pddl_goal: str) -> bool:
    """True when the goal uses a quantifier or carries lifted ``?`` variables."""
    return "?" in pddl_goal or bool(
        re.search(r"\b(?:exists|forall)\b", pddl_goal, re.IGNORECASE)
    )


def _tokenize_sexpr(text: str) -> list[str]:
    """Split a PDDL s-expression into ``(`` / ``)`` / atom tokens."""
    text = re.sub(r";.*", "", text)
    return re.findall(r"[()]|[^\s()]+", text)


def _read_sexpr(tokens: list[str], i: int):
    """Read one (possibly nested) parenthesized form; returns ``(node, next_i)``."""
    if i >= len(tokens):
        raise ValueError("unexpected end of goal expression")
    if tokens[i] != "(":
        return tokens[i], i + 1
    i += 1
    items: list = []
    while i < len(tokens) and tokens[i] != ")":
        sub, i = _read_sexpr(tokens, i)
        items.append(sub)
    if i >= len(tokens):
        raise ValueError("unbalanced parentheses in goal expression")
    return items, i + 1


def _parse_typed_vars(decl: list) -> list[tuple[str, str]]:
    """Parse ``(?b1 - block_red ?b2 - block_red)`` (or shared-type) declarations."""
    out: list[tuple[str, str]] = []
    pending: list[str] = []
    i = 0
    while i < len(decl):
        tok = decl[i]
        if not isinstance(tok, str):
            raise ValueError(f"nested form inside quantifier variable list: {decl}")
        if tok == "-":
            if i + 1 >= len(decl):
                raise ValueError(f"missing type after '-' in quantifier vars: {decl}")
            for v in pending:
                out.append((v, decl[i + 1]))
            pending = []
            i += 2
        else:
            pending.append(tok)
            i += 1
    if pending:
        raise ValueError(f"untyped variables {pending} in quantifier declaration {decl}")
    return out


def _parse_goal_sexpr(pddl_goal: str):
    """Parse a goal (bare body or ``(:goal ...)``) into a raw s-expression."""
    tokens = _tokenize_sexpr(pddl_goal)
    if not tokens or tokens[0] != "(":
        raise ValueError(f"goal expression must start with '(': {pddl_goal!r}")
    sexpr, _ = _read_sexpr(tokens, 0)
    if isinstance(sexpr, list) and sexpr and sexpr[0] == ":goal":
        if len(sexpr) < 2:
            raise ValueError(f"empty (:goal ...) form: {pddl_goal!r}")
        return sexpr[1]
    return sexpr


def _literal_to_english(
    name: str,
    args: list[str],
    var_phrases: dict[str, str],
    object_phrases: dict[str, str] | None = None,
) -> str:
    """Render one relational literal to English, substituting bound vars."""
    def term(a: str) -> str:
        if a.startswith("?"):
            if a not in var_phrases:
                raise ValueError(f"unbound variable {a!r} in literal ({name} {' '.join(args)})")
            return var_phrases[a]
        return _object_phrase(a, object_phrases)

    if name == "on" and len(args) >= 2:
        return f"put {term(args[0])} on {term(args[1])}"
    if name == "on_table" and len(args) >= 1:
        return f"put {term(args[0])} on the table"
    if name in _SPATIAL_PHRASES and len(args) >= 2:
        return f"put {term(args[0])} {_SPATIAL_PHRASES[name]} {term(args[1])}"
    if name == "ori_of" and len(args) >= 2:
        return f"rotate {term(args[0])} to orientation {args[1]}"
    if name == "clear" and len(args) >= 1:
        return f"ensure {term(args[0])} is clear"
    if name == "pick_obstructs" and len(args) >= 2:
        return f"isolate {term(args[1])} from {term(args[0])}"
    logger.warning("pddl_goal_to_english: unknown predicate %r in quantified goal", name)
    return ""


def _collapse_typed_stack(body, var_types: dict[str, str]) -> str | None:
    """Collapse a same-type quantified stack goal into ``"stack N <color> blocks"``.

    TODO: this is a hack. It re-derives stack structure from the already
    lifted goal body (``on``/``on_table`` over quantified vars) purely to make
    the rendered English read naturally, instead of the goal being described
    properly at its source. It should remain a stopgap until goal-to-language
    is unified — ideally sharing one canonical PDDL->English implementation
    with TAMP's ``goal_spec`` — at which point this can be deleted.

    Returns the collapsed phrase, or ``None`` (so callers fall back to the
    existing per-literal rendering) when the body is anything other than a
    single uniform-type stack: mixed types, branching, cycles, stray
    predicates, or on_table on a non-bottom block all disqualify it.
    """
    if not isinstance(body, list) or not body or body[0] != "and":
        return None

    all_vars = set(var_types)
    on_edges: list[tuple[str, str]] = []
    on_table_vars: set[str] = set()
    for child in body[1:]:
        if not isinstance(child, list) or not child or not isinstance(child[0], str):
            return None
        head = child[0].lower()
        if head in ("not", "="):
            # Distinctness / negation constraints don't affect stack structure.
            continue
        args = [a for a in child[1:] if isinstance(a, str)]
        if head == "on" and len(args) == 2:
            on_edges.append((args[0], args[1]))
        elif head == "on_table" and len(args) >= 1:
            # ``(on_table ?b)`` and ``(on_table ?b workspace_table)`` are both
            # emitted by datagen; the block is always the first arg and any
            # trailing table object is irrelevant to the stack structure.
            on_table_vars.add(args[0])
        else:
            return None

    if not on_edges:
        return None
    for top, bottom in on_edges:
        if top not in all_vars or bottom not in all_vars:
            return None
    if not on_table_vars <= all_vars:
        return None

    # Build the top->bottom chain; a clean stack has one root and no branches.
    top_to_bottom: dict[str, str] = {}
    bottoms: set[str] = set()
    for top, bottom in on_edges:
        if top in top_to_bottom:
            return None  # a block can't sit on two things
        top_to_bottom[top] = bottom
        bottoms.add(bottom)
    roots = [t for t in top_to_bottom if t not in bottoms]
    if len(roots) != 1:
        return None
    chain = [roots[0]]
    visited = {roots[0]}
    cur = roots[0]
    while cur in top_to_bottom:
        nxt = top_to_bottom[cur]
        if nxt in visited:
            return None  # cycle
        visited.add(nxt)
        chain.append(nxt)
        cur = nxt

    # The single chain must cover every quantified var (exactly one stack).
    if set(chain) != all_vars:
        return None
    if len(chain) < 2:
        return None
    # on_table, if present, may only close the bottom of the stack.
    if on_table_vars and on_table_vars != {chain[-1]}:
        return None

    types = {var_types[v] for v in chain}
    if len(types) != 1:
        return None
    plural = _type_noun(next(iter(types))) + "s"
    return f"stack {len(chain)} {plural}"


def _sexpr_to_english(
    node,
    var_phrases: dict[str, str],
    object_phrases: dict[str, str] | None = None,
) -> str:
    """Recursively render a goal s-expression to English."""
    if isinstance(node, str):
        return ""
    if not node or not isinstance(node[0], str):
        return ""
    head = node[0].lower()
    if head in _QUANTIFIERS:
        typed = _parse_typed_vars(node[1]) if len(node) > 1 and isinstance(node[1], list) else []
        if len(node) > 2:
            collapsed = _collapse_typed_stack(node[2], dict(typed))
            if collapsed is not None:
                return collapsed
        local = dict(var_phrases)
        seen: dict[str, int] = {}
        for var, typ in typed:
            noun = _type_noun(typ)
            n = seen.get(noun, 0)
            local[var] = f"a {noun}" if n == 0 else f"another {noun}"
            seen[noun] = n + 1
        return (
            _sexpr_to_english(node[2], local, object_phrases) if len(node) > 2 else ""
        )
    if head == "and":
        parts = [_sexpr_to_english(c, var_phrases, object_phrases) for c in node[1:]]
        return ", ".join(p for p in parts if p)
    if head == "or":
        parts = [_sexpr_to_english(c, var_phrases, object_phrases) for c in node[1:]]
        return " or ".join(p for p in parts if p)
    if head == "not" and len(node) > 1:
        child = node[1]
        if (
            isinstance(child, list)
            and child
            and isinstance(child[0], str)
            and child[0].lower() == "pick_obstructs"
        ):
            args = [a for a in child[1:] if isinstance(a, str)]
            if len(args) >= 2:
                return _literal_to_english(
                    "pick_obstructs", args, var_phrases, object_phrases
                )
        return ""
    if head == "=":
        # Distinctness constraints carry no user-facing phrasing.
        return ""
    args = [a for a in node[1:] if isinstance(a, str)]
    return _literal_to_english(head, args, var_phrases, object_phrases)


def _quantified_goal_to_english(pddl_goal: str) -> str:
    """English for a quantified/lifted goal; raises if any ``?`` var survives."""
    object_phrases = _object_phrases_for_goal(pddl_goal)
    english = _sexpr_to_english(_parse_goal_sexpr(pddl_goal), {}, object_phrases)
    if not english:
        raise ValueError(
            f"pddl_goal_to_english: produced no phrases for quantified goal {pddl_goal!r}"
        )
    if "?" in english:
        raise ValueError(
            f"pddl_goal_to_english: unresolved variable in {english!r} from {pddl_goal!r}"
        )
    return english


def _pop_ori_alternatives(
    pddl_goal: str,
    object_phrases: dict[str, str] | None = None,
) -> tuple[str, list[str]]:
    """Phrase each ``(or (ori_of b o2) (ori_of b o4) ...)`` conjunct and cut it from the goal.

    Task generation emits such a disjunction for glyph blocks whose symmetric faces
    all read the same (stack_three.py); flattening it would ask for every
    orientation at once. Any other ``(or ...)`` is left for the flat parser.
    """
    phrases: list[str] = []

    def replace(m: re.Match) -> str:
        leaves = _parse_goal_predicates(m.group(1))
        blocks = {p["args"][0] for p in leaves if p["args"]}
        if not leaves or len(blocks) != 1 or any(p["name"] != "ori_of" or len(p["args"]) < 2 for p in leaves):
            return m.group(0)
        oris = " or ".join(p["args"][1] for p in leaves)
        phrases.append(f"rotate {_object_phrase(blocks.pop(), object_phrases)} to orientation {oris}")
        return ""

    return _OR_GROUP_RE.sub(replace, pddl_goal), phrases


def _phrase_not_pick_obstructs(
    pddl_goal: str,
    object_phrases: dict[str, str] | None = None,
) -> str | None:
    """English for ``(not (pick_obstructs <obstructor> <target>))`` isolate goals."""
    m = _NOT_PICK_OBSTRUCTS_RE.search(pddl_goal)
    if not m:
        return None
    obstructor, target = m.group(1), m.group(2)
    return (
        f"isolate {_object_phrase(target, object_phrases)} "
        f"from {_object_phrase(obstructor, object_phrases)}"
    )


def _parse_goal_predicates(pddl_goal: str) -> list[dict]:
    """Extract leaf predicates as ``[{"name": ..., "args": [...]}, ...]``."""
    preds: list[dict] = []
    for m in _LEAF_PRED_RE.finditer(pddl_goal):
        name = m.group(1).lower()
        if name in ("and", "or", "not"):
            continue
        args = [a for a in m.group(2).split() if a]
        if not args:
            continue
        preds.append({"name": name, "args": args})
    return preds


# ---------------------------------------------------------------------------
# Stacking chain detection
# ---------------------------------------------------------------------------


def _extract_stack_chains(
    preds: list[dict],
) -> tuple[list[list[str]], set[int]]:
    """Merge consecutive ``on`` predicates into stacking chains.

    Returns ``(chains, consumed_indices)`` where each chain is
    ``[top, ..., bottom]`` (length >= 3) and ``consumed_indices`` are the
    predicate indices folded into chains so callers can skip them.
    """
    on_preds = [(i, p) for i, p in enumerate(preds) if p["name"] == "on"]
    if len(on_preds) < 2:
        return [], set()

    top_to_bottom: dict[str, tuple[str, int]] = {}
    bottom_set: set[str] = set()
    for idx, p in on_preds:
        top, bottom = p["args"][0], p["args"][1]
        top_to_bottom[top] = (bottom, idx)
        bottom_set.add(bottom)

    chains: list[list[str]] = []
    consumed: set[int] = set()
    for start in (t for t in top_to_bottom if t not in bottom_set):
        chain = [start]
        chain_indices: list[int] = []
        cur = start
        visited: set[str] = {start}
        while cur in top_to_bottom:
            nxt, idx = top_to_bottom[cur]
            if nxt in visited:
                break
            visited.add(nxt)
            chain_indices.append(idx)
            chain.append(nxt)
            cur = nxt
        if len(chain) >= 3:
            chains.append(chain)
            consumed.update(chain_indices)

    return chains, consumed


# ---------------------------------------------------------------------------
# Core converter
# ---------------------------------------------------------------------------


def pddl_goal_to_english(
    pddl_goal: str,
    *,
    domain_pddl: str | Path | None = None,
    orientation_map: dict[str, dict[str, str]] | None = None,
) -> str:
    """Convert a PDDL goal expression to natural English.

    Handles color vs. letter/pattern blocks, stacking chains, and spatial
    relations.  ``ori_of`` predicates emit a raw orientation-index phrase
    (``"rotate X to orientation N"``, logged as a warning) until a block face
    annotation catalog lets us describe the target in painted-symbol terms.
    Raises ``ValueError`` if no phrases can be produced.

    Parameters
    ----------
    pddl_goal:
        Raw PDDL goal string, e.g.
        ``"(and (on block_r block_o1) (ori_of block_r o2))"``.
    domain_pddl:
        Accepted for backward compatibility but currently unused.
    orientation_map:
        Accepted for backward compatibility but currently unused.
    """

    # Quantified / lifted goals (exists/forall, ``?vars``) can't be handled by
    # the flat grounded-literal parser (it would emit raw ``?b1``); route them
    # through the s-expression walker instead.
    if _has_quantifier_or_var(pddl_goal):
        return _quantified_goal_to_english(pddl_goal)

    object_phrases = _object_phrases_for_goal(pddl_goal)
    isolate_phrase = _phrase_not_pick_obstructs(pddl_goal, object_phrases)
    pddl_goal, or_phrases = _pop_ori_alternatives(pddl_goal, object_phrases)

    preds = _parse_goal_predicates(pddl_goal)
    if not preds and isolate_phrase is None and not or_phrases:
        raise ValueError(f"pddl_goal_to_english: no supported predicates found in {pddl_goal!r}")

    chains, consumed = _extract_stack_chains(preds)

    # Check for on_table at the bottom of stacking chains so we can merge.
    on_table_by_block: dict[str, int] = {}
    for i, p in enumerate(preds):
        if p["name"] == "on_table":
            on_table_by_block[p["args"][0]] = i

    phrases: list[str] = []
    if isolate_phrase is not None:
        phrases.append(isolate_phrase)

    for chain in chains:
        names = [_object_phrase(b, object_phrases) for b in chain]
        bottom = chain[-1]
        if bottom in on_table_by_block:
            consumed.add(on_table_by_block[bottom])
            phrases.append("stack " + " on ".join(names) + " on the table")
        else:
            phrases.append("stack " + " on ".join(names))

    for i, p in enumerate(preds):
        if i in consumed:
            continue
        name = p["name"]
        args = p["args"]

        # TODO: describe the target in painted-symbol terms once a block
        # face catalog exists; until then emit the raw orientation index.
        if name == "ori_of" and len(args) >= 2:
            logger.warning(
                "pddl_goal_to_english: ori_of predicate %r emitted with a raw "
                "orientation index — block face catalog not yet available",
                p,
            )
            phrases.append(
                f"rotate {_object_phrase(args[0], object_phrases)} to orientation {args[1]}"
            )
            continue
        elif name == "on" and len(args) >= 2:
            phrases.append(
                f"put {_object_phrase(args[0], object_phrases)} "
                f"on {_object_phrase(args[1], object_phrases)}"
            )
        elif name == "on_table":
            phrases.append(f"put {_object_phrase(args[0], object_phrases)} on the table")
        elif name == "clear" and len(args) >= 1:
            if isolate_phrase is None:
                phrases.append(f"ensure {_object_phrase(args[0], object_phrases)} is clear")
        elif name == "pick_obstructs":
            # Rendered via ``(not (pick_obstructs ...))`` as an isolate phrase.
            continue
        elif name in _SPATIAL_PHRASES and len(args) >= 2:
            rel = _SPATIAL_PHRASES[name]
            phrases.append(
                f"put {_object_phrase(args[0], object_phrases)} {rel} "
                f"{_object_phrase(args[1], object_phrases)}"
            )
        else:
            logger.warning(
                "pddl_goal_to_english: unknown predicate %r in %r",
                name,
                pddl_goal,
            )

    phrases += or_phrases
    if not phrases:
        raise ValueError(
            f"pddl_goal_to_english: could not produce any phrases from {pddl_goal!r}"
        )

    return ", ".join(phrases)


# ---------------------------------------------------------------------------
# Public API (backward-compatible)
# ---------------------------------------------------------------------------

_DEFAULT_DOMAIN_PDDL = Path(__file__).resolve().parents[1] / "tamp" / "blocks" / "domain.pddl"


def pddl_goal_to_prompt(
    pddl_goal: str,
    *,
    domain_pddl: str | Path | None = None,
    orientation_map: dict[str, dict[str, str]] | None = None,
) -> str:
    """Convert a PDDL goal expression to a natural-language prompt.

    Backward-compatible wrapper around :func:`pddl_goal_to_english`.
    Auto-discovers ``domain.pddl`` for orientation resolution when neither
    *domain_pddl* nor *orientation_map* is provided.

    >>> pddl_goal_to_prompt("(and (on block_r block_o1) (on_table block_o1))")  # doctest: +SKIP
    'put the R block on the O1 block, put the O1 block on the table'
    """
    if orientation_map is None and domain_pddl is None and _DEFAULT_DOMAIN_PDDL.exists():
        domain_pddl = _DEFAULT_DOMAIN_PDDL
    return pddl_goal_to_english(
        pddl_goal,
        domain_pddl=domain_pddl,
        orientation_map=orientation_map,
    )


def policy_goal(setup: dict, replay_block_identity: bool) -> str | None:
    """The PDDL goal the policies are prompted with for one queue setup.

    On glyph/colour renders ``pddl_goal`` is in source block names (what success
    is graded on) and ``policy_pddl_goal`` is the recorded, rendered goal. The
    rendered names only match the scene when the recorded identities are painted
    (``--replay_block_identity``), so they are used only then.
    """
    if replay_block_identity and setup.get("policy_pddl_goal"):
        return setup["policy_pddl_goal"]
    return setup.get("pddl_goal")


def parse_prompt_for_setup(
    setup: dict,
    *,
    orientation_map: dict[str, dict[str, str]] | None = None,
    replay_block_identity: bool = False,
) -> str:
    """Return the language prompt for one queue setup.

    Prefers ``setup["hl_prompt"]`` (parquet ``goal.hl_prompt``) when that key
    is present. Otherwise derives English from :func:`policy_goal`. Raises if
    ``hl_prompt`` is present but empty, or if ``pddl_goal`` is missing or
    empty on the fallback path. If the setup dict carries a ``"domain_pddl"``
    key, that domain file is used for orientation resolution instead of
    *orientation_map* or the auto-discovered default.
    """
    if "hl_prompt" in setup:
        raw_hl = setup["hl_prompt"]
        if not isinstance(raw_hl, str) or not raw_hl.strip():
            raise ValueError(
                f"queue setup hl_prompt must be a non-empty string, got {raw_hl!r}"
            )
        return raw_hl.strip()
    if "pddl_goal" not in setup:
        raise KeyError(
            "queue setup is missing 'pddl_goal' and 'hl_prompt'; the LeRobot "
            "dataset must have a `pddl_goal` column for queue-driven eval "
            "(or `goal.hl_prompt` for a recorded language instruction). "
            f"setup keys: {sorted(setup.keys())}"
        )
    raw = policy_goal(setup, replay_block_identity)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"queue setup pddl_goal must be a non-empty string, got {raw!r}")
    per_setup_domain = setup.get("domain_pddl")
    if per_setup_domain is not None:
        return pddl_goal_to_prompt(raw, domain_pddl=per_setup_domain)
    return pddl_goal_to_prompt(raw, orientation_map=orientation_map)


class PromptResolver:
    """Per-env prompt cache resolved from the live queue state.

    Under batched policy inference, :meth:`current_prompt` still emits one
    prompt across all envs (the first non-``None`` setup in env-id order) and
    warns once at update time if per-env setups disagree. Maniflow transforms
    should use :meth:`prompt_for_env` instead for per-env routing.

    The cache is intentionally unset at construction; the queue-driven
    flow guarantees :meth:`update_setups` runs during the first
    ``env.reset()`` before the obs transform asks for a prompt. Calling
    :meth:`current_prompt` before any setup landed raises.
    :meth:`prompt_for_env` returns :data:`IDLE_PROMPT_PLACEHOLDER` for env slots
    that never received a setup; drained-queue envs keep their last prompt.

    # TODO: orientation map loading is disabled until a block face
    # annotation catalog is available.  Re-enable when block_catalog.json
    # lands and orientation prompts can reference painted symbols.
    """

    def __init__(
        self,
        num_envs: int,
        *,
        domain_pddl: str | Path | None = None,
        replay_block_identity: bool = False,
    ) -> None:
        if num_envs < 1:
            raise ValueError(f"PromptResolver num_envs must be >= 1, got {num_envs}")
        self._num_envs = num_envs
        self._replay_block_identity = replay_block_identity
        self._setups: list[dict | None] = [None] * num_envs
        self._mismatch_warned = False
        self._cached: str | None = None
        self._per_env_prompts: dict[int, str] = {}

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def update_setups(self, setups: list[dict | None]) -> None:
        """Replace the resolver's view of which setup is in each env.

        Called by ``EpisodeQueueClient.fetch_reset_states`` after every reset.
        Treat the dicts as read-only — they're shared with the client.
        """
        if len(setups) != self._num_envs:
            raise ValueError(
                f"PromptResolver got {len(setups)} setups but tracks {self._num_envs} envs"
            )
        self._setups = list(setups)
        for env_idx, setup in enumerate(self._setups):
            if setup is not None:
                self._per_env_prompts[env_idx] = self.prompt_for_setup(setup)
        self._cached = self._resolve(self._setups)

    def prompt_for_setup(self, setup: dict) -> str:
        """:func:`parse_prompt_for_setup` with this resolver's ``replay_block_identity``."""
        return parse_prompt_for_setup(setup, replay_block_identity=self._replay_block_identity)

    def current_prompt(self) -> str:
        if self._cached is None:
            raise RuntimeError(
                "PromptResolver.current_prompt() called before any setup "
                "with a prompt has been registered. Queue-driven eval "
                "(--queue_url) is the only supported prompt source today; "
                "running a prompt-using obs transform without it requires "
                "wiring an explicit override."
            )
        return self._cached

    def prompt_for_env(self, env_index: int) -> str:
        """Return the language prompt for one env.

        An env whose setup is ``None`` (drained queue) keeps its last prompt.
        An env that never had a setup with a prompt returns
        :data:`IDLE_PROMPT_PLACEHOLDER` (queue-exhausted / spare env slot).
        """
        return self._per_env_prompts.get(env_index, IDLE_PROMPT_PLACEHOLDER)

    def _resolve(self, setups: list[dict | None]) -> str | None:
        prompts = [self.prompt_for_setup(s) for s in setups if s is not None]
        if not prompts:
            return None
        if len(set(prompts)) > 1 and not self._mismatch_warned:
            logger.warning(
                "PromptResolver: per-env prompts disagree under batched "
                "inference (using the first non-None setup's prompt). prompts=%s",
                prompts,
            )
            self._mismatch_warned = True
        return prompts[0]
